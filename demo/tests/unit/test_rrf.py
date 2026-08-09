import pytest

from app.adapters.retrieval.rrf import reciprocal_rank_fusion
from app.domain.models import SearchHit


def hit(document_id: str, rank: int, stage: str) -> SearchHit:
    return SearchHit(
        document_id=document_id,
        text=document_id,
        rank=rank,
        score=float(10 - rank),
        stage_scores={stage: float(10 - rank)},
        stage_ranks={stage: rank},
    )


def test_fuses_rankings_and_preserves_stage_lineage() -> None:
    fused = reciprocal_rank_fusion(
        [
            [hit("doc-a", 1, "bm25"), hit("doc-b", 2, "bm25")],
            [hit("doc-b", 1, "dense"), hit("doc-c", 2, "dense")],
        ],
        rrf_k=60,
    )

    assert [item.document_id for item in fused] == ["doc-b", "doc-a", "doc-c"]
    assert fused[0].stage_ranks == {"bm25": 2, "dense": 1, "rrf": 1}
    assert fused[0].score == pytest.approx((1 / 62) + (1 / 61))


def test_rejects_non_positive_rrf_constant() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([], rrf_k=0)
