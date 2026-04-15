"""
Build a PaperSearch JSON corpus from the Kaggle *Cornell-University/arxiv* metadata snapshot.

Requires Kaggle API credentials (~/.kaggle/kaggle.json or env KAGGLE_USERNAME + KAGGLE_KEY).
Dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "Cornell-University/arxiv"
KAGGLE_JSON = "arxiv-metadata-oai-snapshot.json"

# arXiv primary categories aligned with the external project brief
DEFAULT_CATEGORY_PREFIXES = ("cs.AI", "cs.LG", "cs.CL")


def _year_from_record(rec: dict[str, Any]) -> int | None:
    ud = rec.get("update_date")
    if isinstance(ud, str):
        y = _parse_year(ud)
        if y is not None:
            return y
    vers = rec.get("versions")
    if isinstance(vers, list) and vers:
        first = vers[0]
        if isinstance(first, dict):
            return _parse_year(first.get("created"))
        if isinstance(first, str):
            return _parse_year(first[:10])
    return None


def _parse_year(update_date: str | None) -> int | None:
    if not update_date or len(update_date) < 4:
        return None
    try:
        return int(update_date[:4])
    except ValueError:
        return None


def _arxiv_id_to_paper_id(raw_id: str) -> str:
    s = (raw_id or "").strip().replace("/", "-")
    if len(s) <= 64:
        return s or "unknown"
    return s[:64]


def _keep_record(categories: str | None, prefixes: tuple[str, ...]) -> bool:
    if not categories:
        return False
    return any(p in categories for p in prefixes)


def _clean_title(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t[:4000] if t else ""


def _clean_abstract(text: str) -> str:
    a = (text or "").strip()
    a = re.sub(r"\s+", " ", a)
    return a if a else ""


def iter_snapshot_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSON objects (one JSON object per line — arXiv OAI snapshot format)."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping bad JSON at line %s", line_no)


def record_to_paper_dict(
    rec: dict[str, Any],
    *,
    category_prefixes: tuple[str, ...] = DEFAULT_CATEGORY_PREFIXES,
) -> dict[str, Any] | None:
    cats = rec.get("categories") or ""
    if not _keep_record(cats, category_prefixes):
        return None
    title = _clean_title(str(rec.get("title", "")))
    abstract = _clean_abstract(str(rec.get("abstract", "")))
    if not title or not abstract:
        return None
    pid = _arxiv_id_to_paper_id(str(rec.get("id", "")))
    authors = str(rec.get("authors", ""))[:2000] or None
    year = _year_from_record(rec)
    doi = rec.get("doi")
    doi_s = str(doi).strip()[:256] if doi else None
    journal = rec.get("journal-ref")
    venue = str(journal).strip()[:500] if journal else None
    return {
        "paper_id": pid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi_s,
        "category": cats[:500],
    }


def locate_snapshot_file(search_dir: Path) -> Path | None:
    p = search_dir / KAGGLE_JSON
    return p if p.is_file() else None


def download_snapshot(dest_dir: Path) -> Path:
    """Download the metadata JSON from Kaggle into dest_dir (creates parents)."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Install the Kaggle CLI dependency: pip install kaggle "
            "(and configure ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)."
        ) from e

    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading %s / %s …", KAGGLE_SLUG, KAGGLE_JSON)
    api.dataset_download_file(
        KAGGLE_SLUG,
        file_name=KAGGLE_JSON,
        path=str(dest_dir),
        force=False,
        quiet=False,
    )
    zfile = dest_dir / f"{KAGGLE_JSON}.zip"
    if zfile.is_file():
        with zipfile.ZipFile(zfile) as zf:
            for member in zf.namelist():
                if member.endswith(KAGGLE_JSON):
                    (dest_dir / KAGGLE_JSON).write_bytes(zf.read(member))
                    break
        zfile.unlink(missing_ok=True)
    found = locate_snapshot_file(dest_dir)
    if found is None:
        raise FileNotFoundError(f"Expected {KAGGLE_JSON} under {dest_dir} after download")
    return found


def build_corpus_json(
    snapshot_path: Path,
    output_path: Path,
    *,
    max_papers: int = 1200,
    category_prefixes: tuple[str, ...] = DEFAULT_CATEGORY_PREFIXES,
) -> int:
    """Filter + cap records and write JSON array for `papersearch.cli seed`."""
    out: list[dict[str, Any]] = []
    for rec in iter_snapshot_records(snapshot_path):
        row = record_to_paper_dict(rec, category_prefixes=category_prefixes)
        if row is None:
            continue
        out.append(row)
        if len(out) >= max_papers:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s papers to %s", len(out), output_path)
    return len(out)


def ensure_snapshot(
    *,
    download_dir: Path,
    snapshot_path: Path | None,
) -> Path:
    if snapshot_path and snapshot_path.is_file():
        return snapshot_path.resolve()
    found = locate_snapshot_file(download_dir)
    if found:
        return found.resolve()
    return download_snapshot(download_dir).resolve()
