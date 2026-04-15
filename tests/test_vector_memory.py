from __future__ import annotations

import pytest

from papersearch.vector_memory import (
    _is_vector_memory_cap_error,
    _ora_codes,
    parse_vector_memory_size,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("512m", "512M"),
        (" 1g ", "1G"),
        ("256M", "256M"),
    ],
)
def test_parse_vector_memory_size_ok(raw: str, expected: str) -> None:
    assert parse_vector_memory_size(raw) == expected


@pytest.mark.parametrize("bad", ["", "12", "512MB", "x", "512 M"])
def test_parse_vector_memory_size_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_vector_memory_size(bad)


def test_ora_codes_from_message() -> None:
    exc = Exception("ORA-02097: parameter cannot ... ORA-51955: ... Vector Memory ...")
    assert _ora_codes(exc) == {2097, 51955}


def test_is_vector_memory_cap_51955() -> None:
    assert _is_vector_memory_cap_error(Exception("ORA-51955: cannot be increased"))


def test_is_vector_memory_cap_2097_with_vector_in_message() -> None:
    assert _is_vector_memory_cap_error(
        Exception("ORA-02097: ... ORA-51955 ... VECTOR_MEMORY_SIZE ...")
    )


def test_is_vector_memory_cap_false_for_unrelated_2097() -> None:
    assert not _is_vector_memory_cap_error(Exception("ORA-02097: invalid value for parameter foo"))
