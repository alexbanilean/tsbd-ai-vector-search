from __future__ import annotations

import pytest
from pydantic import ValidationError

from papersearch.models import (
    SearchRequest,
    combined_text,
    distance_to_similarity,
    oracle_str,
    paper_out_from_row,
)


def test_combined_text_normalizes_whitespace():
    t = combined_text("  Title  ", "  Abs\n\npara  ")
    assert t.startswith("Title")
    assert "Abs" in t


def test_distance_to_similarity_monotonic():
    assert distance_to_similarity(0.0) == pytest.approx(1.0)
    assert distance_to_similarity(1.0) == pytest.approx(0.5)
    assert distance_to_similarity(3.0) < distance_to_similarity(1.0)


def test_search_request_rejects_short_trimmed_query():
    with pytest.raises(ValidationError):
        SearchRequest(query=" x ", top_k=5)


def test_search_request_accepts_valid_query():
    r = SearchRequest(query="  neural information retrieval  ", top_k=3)
    assert r.query.startswith("neural")


def test_search_request_hybrid_filters():
    r = SearchRequest(
        query="optimization",
        top_k=5,
        min_year=2018,
        max_year=2024,
        category_contains="cs.AI",
    )
    assert r.min_year == 2018
    assert r.category_contains == "cs.AI"


def test_search_request_year_order():
    with pytest.raises(ValidationError):
        SearchRequest(query="ok", top_k=2, min_year=2020, max_year=2010)


def test_oracle_str_reads_lob_like():
    class _Lob:
        def read(self) -> bytes:
            return b"hello\xc3\xa9"

    assert oracle_str(_Lob()) == "helloé"


def test_paper_out_from_row_plain_tuple():
    row = ("p1", "T", "Abstract", "A", 2020, "v", "doi", "cs.AI", 0.25)
    o = paper_out_from_row(row)
    assert o.paper_id == "p1"
    assert o.abstract == "Abstract"
    assert o.category == "cs.AI"
    assert o.year == 2020
