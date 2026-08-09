from dataclasses import dataclass

import pytest

from app.domain.models import RetrievalStrategy, SearchHit
from app.services.retrieval import RetrievalPipeline, UnavailableStrategyError


@dataclass
class FakeRetriever:
    name: str
    document_ids: list[str]

    def search(self, query: str, k: int) -> list[SearchHit]:
        del query
        return [
            SearchHit(
                document_id=document_id,
                text=document_id,
                rank=rank,
                score=float(10 - rank),
                stage_scores={self.name: float(10 - rank)},
                stage_ranks={self.name: rank},
            )
            for rank, document_id in enumerate(self.document_ids[:k], start=1)
        ]


class FakeReranker:
    model_name = "fake-cross-encoder"

    def rerank(self, query: str, candidates: list[SearchHit], k: int) -> list[SearchHit]:
        del query
        return [
            hit.model_copy(
                update={
                    "rank": rank,
                    "score": float(100 - rank),
                    "stage_scores": {**hit.stage_scores, "rerank": float(100 - rank)},
                    "stage_ranks": {**hit.stage_ranks, "rerank": rank},
                }
            )
            for rank, hit in enumerate(reversed(candidates[:k]), start=1)
        ]


def pipeline(reranker: FakeReranker | None = None) -> RetrievalPipeline:
    return RetrievalPipeline(
        dataset_name="fixture",
        corpus_hash="abc123",
        bm25=FakeRetriever("bm25", ["a", "b"]),
        dense=FakeRetriever("dense", ["b", "c"]),
        reranker=reranker,
        candidate_k=10,
        rrf_k=60,
    )


def test_runs_rrf_through_the_shared_pipeline() -> None:
    result = pipeline().search("question", RetrievalStrategy.RRF, 3)

    assert [hit.document_id for hit in result.hits] == ["b", "a", "c"]
    assert set(result.trace.stage_durations_ms) == {"bm25", "dense", "rrf"}
    assert result.trace.corpus_hash == "abc123"


def test_runs_cross_encoder_after_rrf() -> None:
    result = pipeline(FakeReranker()).search("question", RetrievalStrategy.RRF_RERANK, 2)

    assert [hit.document_id for hit in result.hits] == ["a", "b"]
    assert result.hits[0].stage_ranks["rerank"] == 1
    assert result.trace.parameters["reranker_model"] == "fake-cross-encoder"


def test_rejects_unavailable_strategy() -> None:
    with pytest.raises(UnavailableStrategyError):
        pipeline().search("question", RetrievalStrategy.RRF_RERANK, 2)
