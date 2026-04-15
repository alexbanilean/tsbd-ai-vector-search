from __future__ import annotations

import array
import logging
import re
import textwrap
from typing import Any, Final

import oracledb

from papersearch.config import Settings, get_settings
from papersearch.models import PaperIn, PaperOut, combined_text, paper_out_from_row
from papersearch.vector_memory import _ora_codes

logger = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "PAPERS_EMBEDDING_HNSW_IDX"

# Lighter HNSW tiers reduce vector-pool use when the first build hits ORA-51962.
_HNSW_PLANS: Final[list[tuple[str, int, int, int | None]]] = [
    ("standard (16/200, acc 95)", 16, 200, 95),
    ("compact (8/64, acc 90)", 8, 64, 90),
    ("minimal (8/48, no acc target)", 8, 48, None),
    ("tiny (4/32, no acc target)", 4, 32, None),
]


def _hnsw_create_sql(*, neighbors: int, efconstruction: int, target_accuracy: int | None) -> str:
    acc = (
        f"\n    WITH TARGET ACCURACY {target_accuracy}"
        if target_accuracy is not None
        else ""
    )
    return textwrap.dedent(
        f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME}
        ON papers (embedding)
        ORGANIZATION INMEMORY NEIGHBOR GRAPH
        DISTANCE COSINE{acc}
        PARAMETERS (TYPE HNSW, NEIGHBORS {neighbors}, EFCONSTRUCTION {efconstruction})
        """
    ).strip()


def _sanitize_index_literal(k: int) -> int:
    if k < 1 or k > 50:
        raise ValueError("top_k out of bounds")
    return int(k)


def table_exists(cursor: oracledb.Cursor) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) FROM user_tables WHERE table_name = 'PAPERS'
        """
    )
    (n,) = cursor.fetchone()
    return int(n) > 0


def vector_index_exists(cursor: oracledb.Cursor) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) FROM user_indexes
        WHERE index_name = :name AND table_name = 'PAPERS'
        """,
        dict(name=VECTOR_INDEX_NAME),
    )
    (n,) = cursor.fetchone()
    return int(n) > 0


def drop_vector_index_if_any(cursor: oracledb.Cursor) -> None:
    if vector_index_exists(cursor):
        cursor.execute(f"DROP INDEX {VECTOR_INDEX_NAME}")


def _category_column_exists(cursor: oracledb.Cursor) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) FROM user_tab_columns
        WHERE table_name = 'PAPERS' AND column_name = 'CATEGORY'
        """
    )
    (n,) = cursor.fetchone()
    return int(n) > 0


def ensure_category_column(cursor: oracledb.Cursor) -> None:
    """Add arXiv categories column for hybrid filters (idempotent)."""
    if not table_exists(cursor):
        return
    if _category_column_exists(cursor):
        return
    logger.info("Adding papers.category column (migration)")
    cursor.execute("ALTER TABLE papers ADD (category VARCHAR2(500))")


def init_schema(
    conn: oracledb.Connection,
    *,
    force: bool = False,
    settings: Settings | None = None,
):
    cfg = settings or get_settings()
    dim = cfg.embedding_dimension
    with conn.cursor() as cursor:
        if force and table_exists(cursor):
            drop_vector_index_if_any(cursor)
            cursor.execute("DROP TABLE papers PURGE")
            conn.commit()

        if table_exists(cursor):
            ensure_category_column(cursor)
            conn.commit()
            logger.info("papers table already exists (schema migration checked)")
            return

        cursor.execute(
            f"""
            CREATE TABLE papers (
              paper_id VARCHAR2(64) PRIMARY KEY,
              title VARCHAR2(4000) NOT NULL,
              abstract_text CLOB NOT NULL,
              authors VARCHAR2(2000),
              year_published NUMBER(4),
              venue VARCHAR2(500),
              doi VARCHAR2(256),
              category VARCHAR2(500),
              combined_text CLOB NOT NULL,
              embedding VECTOR({dim}, FLOAT32) NOT NULL,
              created_at TIMESTAMP(9) DEFAULT SYSTIMESTAMP NOT NULL
            )
            """
        )
        conn.commit()
        logger.info("Created papers table with VECTOR(%s, FLOAT32)", dim)


def ensure_vector_index(
    conn: oracledb.Connection,
    settings: Settings | None = None,
    *,
    skip: bool = False,
) -> bool:
    """Create HNSW vector index when table has rows. Returns True if index exists at end."""
    cfg = settings or get_settings()
    logger.info("Ensuring vector index (embedding_dim=%s)", cfg.embedding_dimension)
    with conn.cursor() as cursor:
        if not table_exists(cursor):
            return False
        if vector_index_exists(cursor):
            return True
        cursor.execute("SELECT COUNT(*) FROM papers")
        (count,) = cursor.fetchone()
        nrows = int(count)
        if nrows == 0:
            logger.warning("Skipping vector index: papers table is empty")
            return False
        if skip:
            logger.info(
                "Skipping vector index creation (--skip-vector-index); exact search will be used."
            )
            return False

        # HNSW approximate index — Oracle AI Vector Search (tiered fallbacks on ORA-51962).
        last_err: oracledb.DatabaseError | None = None
        for label, neighbors, ef, acc in _HNSW_PLANS:
            ddl = _hnsw_create_sql(
                neighbors=neighbors, efconstruction=ef, target_accuracy=acc
            )
            try:
                cursor.execute(ddl)
                conn.commit()
                logger.info("Created vector index %s [%s]", VECTOR_INDEX_NAME, label)
                return True
            except oracledb.DatabaseError as e:
                conn.rollback()
                last_err = e
                if 51962 in _ora_codes(e):
                    logger.debug(
                        "ORA-51962 on HNSW tier %s; retrying lighter: %s",
                        label,
                        e,
                    )
                    continue
                logger.error("Could not create vector index (non-51962): %s", e)
                return False

        if last_err is not None:
            logger.info(
                "Vector HNSW index not created: vector memory pool could not fit any tier "
                "(%s rows). This is common on Oracle Database Free with a small pool — "
                "semantic search still uses exact VECTOR_DISTANCE (fast at this scale). "
                "Optional: `set-vector-memory` if ORA-51955 does not block you; or "
                "`seed --skip-vector-index` to skip index attempts.",
                nrows,
            )
            logger.debug("Last ORA from vector index build: %s", last_err)
        return False


def count_papers(conn: oracledb.Connection) -> int:
    with conn.cursor() as cursor:
        if not table_exists(cursor):
            return 0
        cursor.execute("SELECT COUNT(*) FROM papers")
        (n,) = cursor.fetchone()
        return int(n)


def ingest_papers(
    conn: oracledb.Connection,
    papers: list[PaperIn],
    vectors: Any,
    *,
    settings: Settings | None = None,
) -> int:
    """Upsert papers with precomputed embedding matrix (float32)."""
    cfg = settings or get_settings()
    if vectors.shape[0] != len(papers):
        raise ValueError("papers / vectors length mismatch")
    if vectors.shape[1] != cfg.embedding_dimension:
        raise ValueError("vector dimension mismatch")

    init_schema(conn, settings=cfg)

    sql = """
        MERGE INTO papers t
        USING (
          SELECT :paper_id paper_id,
                 :title title,
                 :abstract_text abstract_text,
                 :authors authors,
                 :year_published year_published,
                 :venue venue,
                 :doi doi,
                 :category category,
                 :combined_text combined_text,
                 :embedding embedding
          FROM dual
        ) s
        ON (t.paper_id = s.paper_id)
        WHEN MATCHED THEN UPDATE SET
          t.title = s.title,
          t.abstract_text = s.abstract_text,
          t.authors = s.authors,
          t.year_published = s.year_published,
          t.venue = s.venue,
          t.doi = s.doi,
          t.category = s.category,
          t.combined_text = s.combined_text,
          t.embedding = s.embedding
        WHEN NOT MATCHED THEN INSERT (
          paper_id,
          title,
          abstract_text,
          authors,
          year_published,
          venue,
          doi,
          category,
          combined_text,
          embedding
        ) VALUES (
          s.paper_id, s.title, s.abstract_text, s.authors, s.year_published, s.venue, s.doi,
          s.category, s.combined_text, s.embedding
        )
    """

    n = 0
    with conn.cursor() as cursor:
        for i, p in enumerate(papers):
            emb = array.array("f", vectors[i].tolist())
            cursor.execute(
                sql,
                paper_id=p.paper_id,
                title=p.title,
                abstract_text=p.abstract,
                authors=p.authors,
                year_published=p.year,
                venue=p.venue,
                doi=p.doi,
                category=p.category,
                combined_text=combined_text(p.title, p.abstract),
                embedding=emb,
            )
            n += 1
        conn.commit()
    return n


def _search_where_clause(
    min_year: int | None,
    max_year: int | None,
    category_contains: str | None,
) -> tuple[str, dict[str, Any]]:
    """Build SQL WHERE fragment and binds (validated scalars only)."""
    wheres: list[str] = []
    binds: dict[str, Any] = {}
    if min_year is not None:
        wheres.append("year_published >= :ymin")
        binds["ymin"] = int(min_year)
    if max_year is not None:
        wheres.append("year_published <= :ymax")
        binds["ymax"] = int(max_year)
    if category_contains is not None:
        wheres.append("category IS NOT NULL AND UPPER(category) LIKE '%' || UPPER(:catpat) || '%'")
        binds["catpat"] = category_contains
    if not wheres:
        return "", {}
    return "WHERE " + " AND ".join(wheres), binds


def search_semantic(
    conn: oracledb.Connection,
    query_embedding: array.array,
    top_k: int,
    *,
    settings: Settings | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    category_contains: str | None = None,
) -> tuple[list[PaperOut], bool]:
    """
    Run VECTOR_DISTANCE in Oracle. Uses FETCH APPROXIMATE when enabled and a vector index exists.
    Optional hybrid predicates on year_published and category (arXiv category string).
    """
    cfg = settings or get_settings()
    k = _sanitize_index_literal(top_k)
    approx = bool(cfg.use_approximate_fetch)
    where_sql, where_binds = _search_where_clause(min_year, max_year, category_contains)

    select_cols = """
                SELECT paper_id, title, abstract_text, authors, year_published,
                       venue, doi, category,
                       VECTOR_DISTANCE(embedding, :qvec, COSINE) AS dist
                FROM papers
    """

    with conn.cursor() as cursor:
        has_idx = vector_index_exists(cursor)
        use_approx = approx and has_idx

        if use_approx:
            sql = f"""
                {select_cols.strip()}
                {where_sql}
                ORDER BY dist
                FETCH APPROXIMATE FIRST {k} ROWS ONLY
            """
        else:
            sql = f"""
                {select_cols.strip()}
                {where_sql}
                ORDER BY dist
                FETCH FIRST :k ROWS ONLY
            """

        qvec = query_embedding
        params: dict[str, Any] = {**where_binds, "qvec": qvec}
        if not use_approx:
            params["k"] = k

        cursor.execute(sql, params)

        rows = cursor.fetchall()
        return [paper_out_from_row(tuple(r)) for r in rows], use_approx


def oracle_version_string(conn: oracledb.Connection) -> str:
    statements = (
        "SELECT banner FROM v$version WHERE ROWNUM = 1",
        "SELECT version_full FROM product_component_version WHERE ROWNUM = 1",
    )
    with conn.cursor() as cursor:
        for stmt in statements:
            try:
                cursor.execute(stmt)
                row = cursor.fetchone()
                if row and row[0]:
                    return str(row[0])
            except oracledb.DatabaseError:
                continue
        return "Oracle Database (connected)"


def safe_banner(banner: str) -> str:
    # avoid noisy control chars in logs / JSON
    return re.sub(r"[\x00-\x1f\x7f]", " ", banner).strip()
