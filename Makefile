.PHONY: install oracle-up oracle-wait oracle-vector-memory demo-vector-memory init seed seed-kaggle reindex-vector import-kaggle serve demo lint test

install:
	python3 -m pip install -e ".[dev]"

oracle-up:
	docker compose up -d oracle

oracle-wait:
	python3 scripts/wait_for_oracle.py

oracle-vector-memory:
	python3 -m papersearch.cli set-vector-memory

# Best-effort: Oracle Free often returns ORA-51955 (exit 3); demo still works without HNSW.
demo-vector-memory:
	-python3 -m papersearch.cli set-vector-memory

reindex-vector:
	python3 -m papersearch.cli reindex-vector

import-kaggle:
	python3 -m papersearch.cli import-kaggle

init:
	python3 -m papersearch.cli init

seed:
	python3 -m papersearch.cli seed --path data/papers.json --replace

seed-kaggle:
	python3 -m papersearch.cli seed --path data/papers.kaggle.json --replace

serve:
	python3 -m papersearch.cli serve

demo: oracle-up install oracle-wait demo-vector-memory init seed serve

lint:
	python3 -m ruff check src tests

test:
	python3 -m pytest -q
