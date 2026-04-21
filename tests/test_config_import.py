from __future__ import annotations

import pytest
from pydantic import ValidationError

from papersearch.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_import_max_papers_default(tmp_path, monkeypatch):
    """Default 1200 when no .env and env var unset (isolate from repo .env)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PAPERSEARCH_IMPORT_MAX_PAPERS", raising=False)
    get_settings.cache_clear()
    assert get_settings().import_max_papers == 1200


def test_import_max_papers_from_env(monkeypatch):
    monkeypatch.setenv("PAPERSEARCH_IMPORT_MAX_PAPERS", "400")
    get_settings.cache_clear()
    assert get_settings().import_max_papers == 400


def test_import_max_papers_validation():
    with pytest.raises(ValidationError):
        Settings(import_max_papers=0)
