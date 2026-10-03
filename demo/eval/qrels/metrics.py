"""
IR ranking metrics (nDCG, Recall, MRR) scored against human-labeled binary qrels.

"""

import math
from collections.abc import Sequence

NDCG_K = 10
MRR_K = 10
RECALL_KS = (10, 100)


def _dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in ranked_ids[:k]]
    idcg = _dcg([1.0] * min(len(relevant), k))
    return _dcg(gains) / idcg if idcg else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def score_query(ranked_ids: Sequence[str], relevant: set[str]) -> dict[str, float]:
    """All per-query metrics for a single ranked result list."""
    scores = {
        f"ndcg@{NDCG_K}": ndcg_at_k(ranked_ids, relevant, NDCG_K),
        f"mrr@{MRR_K}": mrr_at_k(ranked_ids, relevant, MRR_K),
    }
    for k in RECALL_KS:
        scores[f"recall@{k}"] = recall_at_k(ranked_ids, relevant, k)
    return scores


def mean_scores(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric across all evaluated queries."""
    if not per_query:
        return {}
    return {
        name: sum(scores[name] for scores in per_query) / len(per_query) for name in per_query[0]
    }
