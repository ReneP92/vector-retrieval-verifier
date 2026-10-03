# FiQA RAG Retrieval Workbench

A local FastAPI and HTMX application for inspecting BM25, dense, RRF, and
cross-encoder-reranked retrieval over the BEIR FiQA-2018 dataset.

Interactive search and offline evaluation replay the same retrieval path. See
[docs/fiqa.md](docs/fiqa.md) for how the corpus, queries, and qrels link
together.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for dense indexing and optional answer generation
- A Cohere API key for cross-encoder reranking
- Docker and Docker Compose, optionally

## Setup

```bash
cd demo
cp .env.example .env
uv sync
uv run python -m scripts.download_fiqa
uv run python -m scripts.build_indexes --all
uv run uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000). API documentation is at
[http://localhost:8000/docs](http://localhost:8000/docs).

Dense indexing sends the non-empty FiQA retrieval units to the configured
embedding provider. The index builder caches completed batches and can resume
an interrupted build. Use `--force` only when intentionally replacing an index.

FiQA contains 38 retrieval units with no title or text. Both indexes exclude
these unusable passages and record their IDs in the index manifest. One empty
passage has a positive test qrel, so the build reports that dataset-quality
issue without modifying the original corpus or judgments.

BM25 can be prepared independently without an API key:

```bash
uv run python -m scripts.build_indexes --bm25
```

## Evaluation

Score retrieval strategies against the FiQA test qrels. The JSON report is
printed to stdout and written to `var/eval/`; logs go to stderr.

```bash
uv run python -m scripts.evaluate run --strategy bm25 --mode qrels --sample 100
```

Omit `--strategy` to evaluate every available strategy and `--sample` to use
all 648 test queries.

## Docker

Dataset and index preparation stays explicit and runs on the host so paid API
operations are never triggered by container startup.

```bash
docker compose run --rm workbench uv run python -m scripts.download_fiqa
docker compose run --rm workbench uv run python -m scripts.build_indexes --all
docker compose up --build
```

The `demo/var` directory is mounted at `/app/var` and persists datasets,
indexes, and provider caches outside the image.

## Quality Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the production mapping and future
evaluation boundary.
