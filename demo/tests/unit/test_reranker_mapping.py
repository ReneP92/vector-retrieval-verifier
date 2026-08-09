from pathlib import Path
from types import SimpleNamespace

import cohere

from app.adapters.retrieval.cohere_reranker import CohereReranker
from app.domain.models import SearchHit


class FakeCohereClient:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, **kwargs):
        del kwargs
        self.calls += 1
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=1, relevance_score=0.9),
                SimpleNamespace(index=0, relevance_score=0.7),
                SimpleNamespace(index=2, relevance_score=0.5),
            ]
        )


def test_maps_provider_indices_back_to_candidates(monkeypatch, tmp_path: Path) -> None:
    client = FakeCohereClient()
    monkeypatch.setattr(cohere, "ClientV2", lambda **kwargs: client)
    reranker = CohereReranker(api_key="test", model_name="test-model", cache_dir=tmp_path)
    candidates = [
        SearchHit(document_id="a", text="A", rank=1, score=1.0),
        SearchHit(document_id="b", text="B", rank=2, score=0.5),
        SearchHit(document_id="c", text="C", rank=3, score=0.25),
    ]

    results = reranker.rerank("query", candidates, 2)
    cached_results = reranker.rerank("query", candidates, 3)

    assert [result.document_id for result in results] == ["b", "a"]
    assert [result.document_id for result in cached_results] == ["b", "a", "c"]
    assert results[0].stage_scores["rerank"] == 0.9
    assert client.calls == 1
    assert len(list((tmp_path / "rerank").glob("*.json"))) == 1
