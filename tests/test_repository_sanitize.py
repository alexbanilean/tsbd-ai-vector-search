from __future__ import annotations

import pytest

from papersearch.repository import (
    VECTOR_INDEX_NAME,
    _hnsw_create_sql,
    _sanitize_index_literal,
)


def test_sanitize_top_k():
    assert _sanitize_index_literal(10) == 10


def test_sanitize_rejects_out_of_range():
    with pytest.raises(ValueError):
        _sanitize_index_literal(0)
    with pytest.raises(ValueError):
        _sanitize_index_literal(99)


def test_hnsw_create_sql_standard():
    sql = _hnsw_create_sql(neighbors=16, efconstruction=200, target_accuracy=95)
    assert VECTOR_INDEX_NAME in sql
    assert "NEIGHBORS 16" in sql
    assert "EFCONSTRUCTION 200" in sql
    assert "TARGET ACCURACY 95" in sql


def test_hnsw_create_sql_without_target_accuracy():
    sql = _hnsw_create_sql(neighbors=8, efconstruction=48, target_accuracy=None)
    assert "NEIGHBORS 8" in sql
    assert "EFCONSTRUCTION 48" in sql
    assert "TARGET ACCURACY" not in sql
