from collections import defaultdict

from app.domain.models import SearchHit


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]],
    *,
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[SearchHit]:
    if rrf_k < 1:
        raise ValueError("RRF k must be positive")

    scores: dict[str, float] = defaultdict(float)
    hits_by_id: dict[str, SearchHit] = {}
    stage_scores: dict[str, dict[str, float]] = defaultdict(dict)
    stage_ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            scores[hit.document_id] += 1.0 / (rrf_k + position)
            hits_by_id.setdefault(hit.document_id, hit)
            stage_scores[hit.document_id].update(hit.stage_scores)
            stage_ranks[hit.document_id].update(hit.stage_ranks)

    ordered_ids = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    if limit is not None:
        ordered_ids = ordered_ids[:limit]

    fused: list[SearchHit] = []
    for rank, document_id in enumerate(ordered_ids, start=1):
        source = hits_by_id[document_id]
        score = scores[document_id]
        fused.append(
            SearchHit(
                document_id=document_id,
                title=source.title,
                text=source.text,
                rank=rank,
                score=score,
                stage_scores={**stage_scores[document_id], "rrf": score},
                stage_ranks={**stage_ranks[document_id], "rrf": rank},
            )
        )
    return fused
