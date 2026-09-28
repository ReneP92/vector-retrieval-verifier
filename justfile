# Justfile for the FiQA RAG Retrieval Workbench demo.
# All recipes run inside the demo/ directory.
set working-directory := 'demo'

# List available recipes.
default:
    @just --list

# Create demo/.env from the example if it does not exist yet.
env:
    @test -f .env || (cp .env.example .env && echo "Created demo/.env - fill in your API keys")

# Install dependencies with uv.
sync:
    uv sync

# Download the BEIR FiQA-2018 dataset (skips if already present).
download:
    @test -f var/datasets/fiqa/corpus.jsonl \
        && echo "FiQA already downloaded - skipping (use 'just download-force' to replace)" \
        || uv run python -m scripts.download_fiqa

# Re-download the dataset, replacing any existing copy.
download-force:
    uv run python -m scripts.download_fiqa --force

# Build the BM25 index only (no API key required; skips if already built).
index-bm25:
    @test -f var/indexes/fiqa/bm25/manifest.json \
        && echo "BM25 index already built - skipping (use 'just index-bm25-force' to rebuild)" \
        || uv run python -m scripts.build_indexes --bm25

# Rebuild the BM25 index, replacing any existing copy.
index-bm25-force:
    uv run python -m scripts.build_indexes --bm25 --force

# Build the dense index only (requires an OpenAI API key). Resumes across OpenAI
# rate-limit failures, retrying until the build completes. Override the attempt
# cap with `just attempts=N index-dense`.
attempts := "50"
index-dense:
    #!/usr/bin/env bash
    set -uo pipefail
    if find var/indexes/fiqa/dense -name manifest.json 2>/dev/null | grep -q .; then
        echo "Dense index already built - skipping (use 'just index-dense-force' to rebuild)"
        exit 0
    fi
    for i in $(seq 1 {{attempts}}); do
        echo "Dense index build attempt $i/{{attempts}}"
        if uv run python -m scripts.build_indexes --dense; then
            exit 0
        fi
        if find var/indexes/fiqa/dense -name manifest.json 2>/dev/null | grep -q .; then
            exit 0
        fi
        echo "Attempt $i failed (likely OpenAI rate limit); resuming in 10s..."
        sleep 10
    done
    echo "Dense index still incomplete after {{attempts}} attempts." >&2
    exit 1

# Rebuild the dense index, replacing any existing copy.
index-dense-force:
    uv run python -m scripts.build_indexes --dense --force

# Build all indexes (BM25 + dense; requires an OpenAI API key). Builds only what is missing.
index: index-bm25 index-dense

# Run the app with autoreload at http://localhost:8000.
serve:
    uv run uvicorn app.main:app --reload

# One-shot setup for the API-key-free path: env, deps, dataset, BM25 index.
setup-bm25: env sync download index-bm25

# Full one-shot setup: env, deps, dataset, all indexes.
setup: env sync download index

# Run the whole quality suite.
check: format-check lint typecheck test

# Check formatting without modifying files.
format-check:
    uv run ruff format --check .

# Apply formatting.
format:
    uv run ruff format .

# Lint.
lint:
    uv run ruff check .

# Type-check.
typecheck:
    uv run ty check

# Run the test suite.
test:
    uv run pytest

# --- Docker ---

# Download the dataset inside a container.
docker-download:
    docker compose run --rm workbench uv run python -m scripts.download_fiqa

# Build all indexes inside a container.
docker-index:
    docker compose run --rm workbench uv run python -m scripts.build_indexes --all

# Build and start the app with Docker Compose.
docker-up:
    docker compose up --build

# Run the evaluation pipeline, e.g. `just eval bm25 deterministic`.
eval strategy mode *flags:
    uv run python -m scripts.evaluate run {{strategy}} {{mode}} {{flags}}
