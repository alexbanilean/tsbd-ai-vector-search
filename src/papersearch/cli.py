from __future__ import annotations

import argparse
import array
import json
import logging
import os
import sys
from pathlib import Path

import oracledb
import uvicorn
from tabulate import tabulate

from papersearch.arxiv_kaggle import build_corpus_json, ensure_snapshot
from papersearch.config import get_settings
from papersearch.db import close_pool, direct_connection, format_connect_help
from papersearch.embeddings import embed_query, embed_texts
from papersearch.models import PaperIn, combined_text
from papersearch.repository import (
    count_papers,
    drop_vector_index_if_any,
    ensure_vector_index,
    ingest_papers,
    init_schema,
    search_semantic,
)
from papersearch.vector_memory import main as vector_memory_main

logger = logging.getLogger(__name__)


def cmd_init(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        with direct_connection(settings) as conn:
            init_schema(conn, force=args.force, settings=settings)
    except oracledb.Error as e:
        print(format_connect_help(settings), file=sys.stderr)
        print(e, file=sys.stderr)
        return 1
    print("Schema ready.")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    settings = get_settings()
    path = Path(args.path)
    if not path.is_file():
        print(f"Seed file not found: {path}", file=sys.stderr)
        return 1
    raw = json.loads(path.read_text(encoding="utf-8"))
    papers = [PaperIn.model_validate(x) for x in raw]
    cap = (
        args.max_papers if args.max_papers is not None else settings.import_max_papers
    )
    if len(papers) > cap:
        logger.info(
            "Truncating seed corpus from %s to %s papers (env PAPERSEARCH_IMPORT_MAX_PAPERS "
            "or --max-papers)",
            len(papers),
            cap,
        )
        papers = papers[:cap]

    combined = [combined_text(p.title, p.abstract) for p in papers]
    matrix = embed_texts(combined, settings=settings)

    try:
        with direct_connection(settings) as conn:
            init_schema(conn, settings=settings)
            if args.replace:
                with conn.cursor() as cur:
                    drop_vector_index_if_any(cur)
                    cur.execute("TRUNCATE TABLE papers")
                conn.commit()
            n = ingest_papers(conn, papers, matrix, settings=settings)
            idx = ensure_vector_index(
                conn, settings=settings, skip=args.skip_vector_index
            )
    except oracledb.Error as e:
        print(format_connect_help(settings), file=sys.stderr)
        print(e, file=sys.stderr)
        return 1
    if idx:
        print(f"Seeded {n} papers. Vector index: ready (HNSW + approximate fetch when enabled).")
    elif args.skip_vector_index:
        print(
            f"Seeded {n} papers. Vector index: skipped by --skip-vector-index "
            f"(exact VECTOR_DISTANCE search)."
        )
    else:
        print(
            f"Seeded {n} papers. Vector index: not created (vector pool full — normal on "
            f"Oracle Free). Data and embeddings are in Oracle; exact search is active. "
            f"Try: python3 -m papersearch.cli search \"your topic\""
        )
    close_pool()
    return 0


def cmd_set_vector_memory(args: argparse.Namespace) -> int:
    return vector_memory_main(["--size", args.size])


def cmd_reindex_vector(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        with direct_connection(settings) as conn:
            with conn.cursor() as cur:
                drop_vector_index_if_any(cur)
            conn.commit()
            ok = ensure_vector_index(conn, settings=settings)
    except oracledb.Error as e:
        print(format_connect_help(settings), file=sys.stderr)
        print(e, file=sys.stderr)
        return 1
    if ok:
        print("Vector index ready.")
        return 0
    print(
        "Vector index was not created (pool full or unsupported). "
        "If your PDB allows it: python3 -m papersearch.cli set-vector-memory\n"
        "On capped Oracle Free this is expected — exact VECTOR_DISTANCE search still works.",
        file=sys.stderr,
    )
    return 2


def cmd_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        vec = embed_query(args.query, settings=settings)
        qemb = array.array("f", vec.tolist())
        with direct_connection(settings) as conn:
            if count_papers(conn) == 0:
                print("Corpus is empty. Run: python3 -m papersearch.cli seed", file=sys.stderr)
                return 1
            results, approx = search_semantic(
                conn,
                qemb,
                args.top_k,
                settings=settings,
                min_year=args.min_year,
                max_year=args.max_year,
                category_contains=args.category,
            )
    except oracledb.Error as e:
        print(format_connect_help(settings), file=sys.stderr)
        print(e, file=sys.stderr)
        return 1
    rows = []
    for r in results:
        abst = (r.abstract or "")[: args.abstract_chars]
        rows.append(
            [
                (r.title or "")[:72],
                r.year,
                (r.category or "")[:36],
                f"{r.similarity:.4f}",
                abst.replace("\n", " ")[: args.abstract_chars],
            ]
        )
    print(f"approximate_index={approx}")
    print(
        tabulate(
            rows,
            headers=["title", "year", "category", "sim", "abstract_snip"],
            tablefmt="github",
        )
    )
    return 0


def cmd_import_kaggle(args: argparse.Namespace) -> int:
    settings = get_settings()
    max_papers = (
        args.max_papers if args.max_papers is not None else settings.import_max_papers
    )
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        print(
            f"Output file already exists: {out}. Refusing to overwrite (protects large imports). "
            f"Pass --overwrite to replace it, or use a different --output path.",
            file=sys.stderr,
        )
        return 1
    dl = Path(args.download_dir)
    snap = Path(args.snapshot) if args.snapshot else None
    try:
        path = ensure_snapshot(download_dir=dl, snapshot_path=snap)
        n = build_corpus_json(
            path,
            out,
            max_papers=max_papers,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("import-kaggle failed")
        print(e, file=sys.stderr)
        return 1
    print(f"Wrote {n} papers to {args.output}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    settings = get_settings()
    uvicorn.run(
        "papersearch.main:create_app",
        factory=True,
        host=args.host or settings.api_host,
        port=int(args.port or settings.api_port),
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="papersearch",
        description="Academic semantic search using Oracle AI Vector Search",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create papers table (VECTOR column)")
    p_init.add_argument("--force", action="store_true", help="DROP TABLE papers (destructive)")
    p_init.set_defaults(func=cmd_init)

    p_seed = sub.add_parser("seed", help="Load JSON corpus and build HNSW index")
    p_seed.add_argument("--path", default="data/papers.json", help="Path to papers JSON array")
    p_seed.add_argument("--replace", action="store_true", help="TRUNCATE papers before load")
    p_seed.add_argument(
        "--skip-vector-index",
        action="store_true",
        help="Do not attempt CREATE VECTOR INDEX (quieter on capped Oracle Free)",
    )
    p_seed.add_argument(
        "--max-papers",
        type=int,
        default=None,
        dest="max_papers",
        metavar="N",
        help="Load at most N papers from JSON (overrides PAPERSEARCH_IMPORT_MAX_PAPERS)",
    )
    p_seed.set_defaults(func=cmd_seed)

    p_vm = sub.add_parser(
        "set-vector-memory",
        help="SYSDBA: raise VECTOR_MEMORY_SIZE in PDB (fixes ORA-51962 for HNSW)",
    )
    p_vm.add_argument(
        "--size",
        default=os.environ.get("ORACLE_VECTOR_MEMORY_SIZE", "512M"),
        help="Pool size, e.g. 512M or 1G (must fit within PDB SGA limits)",
    )
    p_vm.set_defaults(func=cmd_set_vector_memory)

    p_ridx = sub.add_parser(
        "reindex-vector",
        help="Drop and recreate HNSW index (after set-vector-memory)",
    )
    p_ridx.set_defaults(func=cmd_reindex_vector)

    p_search = sub.add_parser("search", help="CLI semantic search (Oracle VECTOR_DISTANCE)")
    p_search.add_argument("query", help="Natural language query")
    p_search.add_argument("--top-k", type=int, default=8, dest="top_k")
    p_search.add_argument("--min-year", type=int, default=None, dest="min_year")
    p_search.add_argument("--max-year", type=int, default=None, dest="max_year")
    p_search.add_argument(
        "--category",
        type=str,
        default=None,
        help="Hybrid filter: substring match on category",
    )
    p_search.add_argument(
        "--abstract-chars",
        type=int,
        default=160,
        dest="abstract_chars",
        help="Max abstract characters in table output",
    )
    p_search.set_defaults(func=cmd_search)

    p_ig = sub.add_parser(
        "import-kaggle",
        help="Download Cornell arXiv metadata from Kaggle and write JSON for seed",
    )
    p_ig.add_argument(
        "--download-dir",
        default="data/raw",
        help="Directory for the snapshot (download + unzip)",
    )
    p_ig.add_argument(
        "--snapshot",
        default=None,
        help="Optional path to existing arxiv-metadata-oai-snapshot.json",
    )
    p_ig.add_argument(
        "--output",
        default="data/papers.kaggle.json",
        help="Output JSON path (default avoids overwriting bundled data/papers.json)",
    )
    p_ig.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output file",
    )
    p_ig.add_argument(
        "--max-papers",
        type=int,
        default=None,
        dest="max_papers",
        metavar="N",
        help="Cap papers written (default: PAPERSEARCH_IMPORT_MAX_PAPERS from .env, else 1200)",
    )
    p_ig.set_defaults(func=cmd_import_kaggle)

    p_srv = sub.add_parser("serve", help="Run FastAPI + static UI")
    p_srv.add_argument("--host", default=None)
    p_srv.add_argument("--port", default=None, type=int)
    p_srv.add_argument("--reload", action="store_true")
    p_srv.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    code = int(args.func(args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
