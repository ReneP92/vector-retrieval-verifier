from pathlib import Path

from fastapi.testclient import TestClient

from app.adapters.datasets.beir import BeirCorpusRepository
from app.adapters.retrieval.bm25 import build_bm25_index
from app.config import Settings
from app.main import create_app

FIXTURE = Path(__file__).parents[1] / "fixtures" / "beir-mini"


def test_health_reports_setup_required_without_indexes(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=FIXTURE,
        index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "setup_required"
    assert payload["available_strategies"] == []
    assert payload["components"]["dataset"] == "ready (3 passages)"


def test_home_renders_setup_guidance(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=FIXTURE,
        index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "RAG Retrieval Workbench" in response.text
    assert "Prepare FiQA before running a trace" in response.text
    assert '<html lang="en" data-theme="light">' in response.text
    assert "data-theme-toggle" in response.text
    assert 'aria-label="Enable night mode"' in response.text
    assert "/static/styles.css?v=" in response.text
    assert "/static/app.js?v=" in response.text


def test_query_endpoint_rejects_unavailable_strategy(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=FIXTURE,
        index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/query",
            json={"query": "Where should savings go?", "strategy": "bm25"},
        )

    assert response.status_code == 422
    assert "unavailable" in response.json()["detail"]


def test_bm25_search_runs_end_to_end_through_html_form(tmp_path: Path) -> None:
    index_dir = tmp_path / "indexes"
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")
    build_bm25_index(repository, index_dir / "bm25")
    settings = Settings(
        data_dir=FIXTURE,
        index_dir=index_dir,
        cache_dir=tmp_path / "cache",
    )

    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health")
        response = client.post(
            "/search",
            data={
                "query": "emergency savings",
                "strategy": "bm25",
                "top_k": "2",
            },
        )

    assert health.json()["available_strategies"] == ["bm25"]
    assert response.status_code == 200
    assert "Trace complete / bm25" in response.text
    assert "doc-a" in response.text
