from __future__ import annotations

import array
import logging
from collections.abc import Generator
from typing import Annotated

import oracledb
from fastapi import APIRouter, Depends, HTTPException

from papersearch.config import Settings, get_settings
from papersearch.db import connection
from papersearch.embeddings import embed_query, embed_texts
from papersearch.models import (
    HealthResponse,
    IngestResponse,
    PaperIn,
    SearchRequest,
    SearchResponse,
    combined_text,
)
from papersearch.repository import (
    count_papers,
    ensure_vector_index,
    ingest_papers,
    init_schema,
    oracle_version_string,
    safe_banner,
    search_semantic,
    vector_index_exists,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_settings_dep() -> Settings:
    return get_settings()


def db_conn_dep() -> Generator[oracledb.Connection, None, None]:
    with connection() as conn:
        yield conn


DbConn = Annotated[oracledb.Connection, Depends(db_conn_dep)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep):
    try:
        with connection(settings) as conn:
            banner = safe_banner(oracle_version_string(conn))
            with conn.cursor() as cursor:
                has_idx = vector_index_exists(cursor)
                idx = "yes" if has_idx else "no"
            path = (
                "approximate"
                if (has_idx and settings.use_approximate_fetch)
                else "exact"
            )
            return HealthResponse(
                status="ok",
                oracle=banner,
                embedding_model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension,
                papers_count=count_papers(conn),
                vector_index=idx,
                search_path=path,
            )
    except Exception as e:  # noqa: BLE001 — surface connectivity issues for demos
        logger.warning("Health degraded: %s", e)
        return HealthResponse(
            status="degraded",
            oracle=str(e),
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            papers_count=None,
            vector_index=None,
            search_path="exact",
        )


@router.post("/search", response_model=SearchResponse)
def semantic_search(
    settings: SettingsDep,
    conn: DbConn,
    body: SearchRequest,
):
    if count_papers(conn) == 0:
        raise HTTPException(
            status_code=409,
            detail="Corpus is empty. Run: python -m papersearch.cli seed",
        )
    vec = embed_query(body.query, settings=settings)
    qemb = array.array("f", vec.tolist())
    results, approx = search_semantic(
        conn,
        qemb,
        body.top_k,
        settings=settings,
        min_year=body.min_year,
        max_year=body.max_year,
        category_contains=body.category_contains,
    )
    return SearchResponse(
        query=body.query,
        top_k=body.top_k,
        approximate=approx,
        min_year=body.min_year,
        max_year=body.max_year,
        category_contains=body.category_contains,
        results=results,
    )


@router.post("/admin/init", response_model=dict)
def admin_init(
    settings: SettingsDep,
    conn: DbConn,
    force: bool = False,
):
    init_schema(conn, force=force, settings=settings)
    return {"ok": True, "force": force}


@router.post("/admin/ingest", response_model=IngestResponse)
def admin_ingest(
    settings: SettingsDep,
    conn: DbConn,
    papers: list[PaperIn],
    rebuild_index: bool = True,
):
    if not papers:
        raise HTTPException(status_code=400, detail="empty list")
    combined = [combined_text(p.title, p.abstract) for p in papers]
    matrix = embed_texts(combined, settings=settings)
    n = ingest_papers(conn, papers, matrix, settings=settings)
    idx_ok = False
    if rebuild_index:
        idx_ok = ensure_vector_index(conn, settings=settings)
    return IngestResponse(
        inserted_or_updated=n,
        message=f"Upserted {n} papers; vector_index={'ok' if idx_ok else 'skipped/failed'}",
    )
