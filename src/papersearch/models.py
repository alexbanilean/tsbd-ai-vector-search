from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PaperIn(BaseModel):
    paper_id: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=1, max_length=4000)
    abstract: str = Field(..., min_length=1)
    authors: str | None = Field(default=None, max_length=2000)
    year: int | None = Field(default=None, ge=1500, le=2100)
    venue: str | None = Field(default=None, max_length=500)
    doi: str | None = Field(default=None, max_length=256)
    category: str | None = Field(default=None, max_length=500)


class PaperOut(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    category: str | None = None
    distance: float = Field(..., description="Oracle VECTOR_DISTANCE with COSINE (lower = closer)")
    similarity: float = Field(
        ...,
        description="Presentation-friendly score: 1/(1+distance), higher is closer",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    min_year: int | None = Field(default=None, ge=1800, le=2100)
    max_year: int | None = Field(default=None, ge=1800, le=2100)
    category_contains: str | None = Field(
        default=None,
        max_length=200,
        description="Hybrid filter: substring match on arXiv categories (case-insensitive)",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 2:
            raise ValueError("query too short after trim")
        return s

    @field_validator("category_contains")
    @classmethod
    def validate_category(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not s:
            return None
        if not re.fullmatch(r"[A-Za-z0-9.,_\s\-]+", s):
            raise ValueError("category_contains: only letters, digits, space, comma, dot, _, -")
        return s

    @model_validator(mode="after")
    def year_range(self) -> SearchRequest:
        if (
            self.min_year is not None
            and self.max_year is not None
            and self.min_year > self.max_year
        ):
            raise ValueError("min_year cannot be greater than max_year")
        return self


class SearchResponse(BaseModel):
    query: str
    top_k: int
    engine: str = Field(
        default="Oracle AI Vector Search (VECTOR + VECTOR_DISTANCE)",
    )
    approximate: bool
    min_year: int | None = None
    max_year: int | None = None
    category_contains: str | None = None
    results: list[PaperOut]


class HealthResponse(BaseModel):
    status: str
    oracle: str
    embedding_model: str
    embedding_dimension: int
    papers_count: int | None = None
    vector_index: str | None = None
    # "approximate" only when HNSW exists and PAPERSEARCH_USE_APPROXIMATE_FETCH is true
    search_path: Literal["exact", "approximate"] = "exact"


class IngestResponse(BaseModel):
    inserted_or_updated: int
    message: str


def combined_text(title: str, abstract: str) -> str:
    return f"{title.strip()}\n\n{abstract.strip()}".strip()


def distance_to_similarity(distance: float) -> float:
    return float(1.0 / (1.0 + max(distance, 0.0)))


def oracle_str(value: Any) -> str | None:
    """Normalize VARCHAR2/CLOB/LOB values from python-oracledb into str for Pydantic."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "read"):
        try:
            data = value.read()
        except Exception:  # noqa: BLE001
            return str(value)
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)
    return str(value)


def _maybe_int_year(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def paper_out_from_row(row: tuple) -> PaperOut:
    paper_id, title, abstract_text, authors, year_published, venue, doi, category, dist = row
    d = float(dist)
    return PaperOut(
        paper_id=oracle_str(paper_id) or "",
        title=oracle_str(title) or "",
        abstract=oracle_str(abstract_text) or "",
        authors=oracle_str(authors),
        year=_maybe_int_year(year_published),
        venue=oracle_str(venue),
        doi=oracle_str(doi),
        category=oracle_str(category),
        distance=d,
        similarity=distance_to_similarity(d),
    )
