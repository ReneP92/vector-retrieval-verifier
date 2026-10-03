"""
Replay FiQA queries through the retrieval pipeline and score each strategy.
"""

from collections.abc import Mapping, Sequence
from time import perf_counter

from app.domain.models import RetrievalStrategy
from app.services.retrieval import RetrievalPipeline
from eval.models import EvaluationMode, EvaluationReport, StrategyReport
from eval.qrels import metrics


def evaluate_strategies(
    pipeline: RetrievalPipeline,
    qrels: Mapping[str, set[str]],
    queries: Mapping[str, str],
    strategies: Sequence[RetrievalStrategy],
    *,
    dataset: str,
    corpus_hash: str,
    depth: int = 100,
) -> EvaluationReport:
    reports = [
        _score_strategy(
            pipeline=pipeline, qrels=qrels, queries=queries, strategy=strategy, depth=depth
        )
        for strategy in strategies
    ]
    return EvaluationReport(
        dataset=dataset,
        corpus_hash=corpus_hash,
        mode=EvaluationMode.QRELS,
        depth=depth,
        reports=reports,
    )


def _score_strategy(
    pipeline: RetrievalPipeline,
    qrels: Mapping[str, set[str]],
    queries: Mapping[str, str],
    strategy: RetrievalStrategy,
    depth: int,
) -> StrategyReport:
    started = perf_counter()
    per_query: list[dict[str, float]] = []
    for query_id, relevant in qrels.items():
        result = pipeline.search(queries[query_id], strategy, depth)
        ranked_ids = [hit.document_id for hit in result.hits]
        per_query.append(metrics.score_query(ranked_ids, relevant))
    return StrategyReport(
        strategy=strategy,
        num_queries=len(per_query),
        metrics=metrics.mean_scores(per_query),
        duration_seconds=perf_counter() - started,
    )
