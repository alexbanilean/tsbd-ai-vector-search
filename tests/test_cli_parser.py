from __future__ import annotations

from argparse import Namespace

from papersearch.cli import build_parser, cmd_import_kaggle


def test_cli_parser_init():
    p = build_parser()
    args = p.parse_args(["init", "--force"])
    assert args.command == "init"
    assert args.force is True


def test_cli_parser_seed_defaults():
    p = build_parser()
    args = p.parse_args(["seed"])
    assert args.path == "data/papers.json"
    assert args.replace is False


def test_cli_set_vector_memory():
    p = build_parser()
    args = p.parse_args(["set-vector-memory", "--size", "1G"])
    assert args.size == "1G"


def test_cli_set_vector_memory_default_size():
    p = build_parser()
    args = p.parse_args(["set-vector-memory"])
    assert args.size == "512M"


def test_cli_reindex_vector():
    p = build_parser()
    args = p.parse_args(["reindex-vector"])
    assert args.command == "reindex-vector"


def test_cli_search_parser():
    p = build_parser()
    args = p.parse_args(["search", "deep learning", "--top-k", "3", "--min-year", "2019"])
    assert args.query == "deep learning"
    assert args.top_k == 3
    assert args.min_year == 2019


def test_cli_import_kaggle_defaults():
    p = build_parser()
    args = p.parse_args(["import-kaggle"])
    assert args.max_papers == 1200
    assert args.output == "data/papers.kaggle.json"
    assert args.overwrite is False


def test_cli_import_kaggle_overwrite():
    p = build_parser()
    args = p.parse_args(["import-kaggle", "--overwrite", "--output", "data/out.json"])
    assert args.overwrite is True
    assert args.output == "data/out.json"


def test_import_kaggle_refuses_existing_without_overwrite(tmp_path, monkeypatch):
    out = tmp_path / "papers.kaggle.json"
    out.write_text("[]", encoding="utf-8")

    def _boom(*_a, **_k):
        raise AssertionError("ensure_snapshot should not run when output exists")

    monkeypatch.setattr("papersearch.cli.ensure_snapshot", _boom)
    ns = Namespace(
        output=str(out),
        download_dir=str(tmp_path / "raw"),
        snapshot=None,
        max_papers=10,
        overwrite=False,
    )
    assert cmd_import_kaggle(ns) == 1
