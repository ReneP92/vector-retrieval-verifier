import logging

import pytest

from app.domain.models import RetrievalStrategy
from app.services.retrieval import RetrievalPipeline
from eval.eval import _log_summary
from eval.models import EvaluationReport
from eval.qrels import runner
from tests.fakes import FakeReranker, FakeRetriever

QRELS = {"q1": {"a"}, "q2": {"e"}}
QUERIES = {"q1": "first question", "q2": "second question"}


def evaluate(*strategies: RetrievalStrategy) -> EvaluationReport:
    pipeline = RetrievalPipeline(
        dataset_name="fixture",
        corpus_hash="abc123",
        bm25=FakeRetriever("bm25", ["a", "b", "c"]),
        dense=FakeRetriever("dense", ["a", "d", "e"]),
        reranker=FakeReranker(),
        candidate_k=2,
        rrf_k=60,
    )
    return runner.evaluate_strategies(
        pipeline, QRELS, QUERIES, strategies, dataset="fixture", corpus_hash="abc123", depth=5
    )


def test_single_retriever_strategies_return_full_depth() -> None:
    bm25, dense = evaluate(RetrievalStrategy.BM25, RetrievalStrategy.DENSE).reports

    assert bm25.mean_hits == 3
    assert bm25.metrics["recall@10"] == 0.5
    assert dense.mean_hits == 3
    assert dense.metrics["recall@10"] == 1.0


def test_fused_strategies_are_scored_with_production_candidate_k() -> None:
    rrf, rrf_rerank = evaluate(RetrievalStrategy.RRF, RetrievalStrategy.RRF_RERANK).reports

    # "e" is dense rank 3, beyond candidate_k=2, so fusion can never surface it.
    assert rrf.mean_hits == 2
    assert rrf.metrics["recall@10"] == 0.5
    assert rrf.parameters == {"candidate_k": 2, "result_k": 5, "rrf_k": 60}
    assert rrf_rerank.mean_hits == 2
    assert rrf_rerank.parameters["reranker_model"] == "fake-cross-encoder"


def test_summary_warns_when_hits_fall_short_of_depth(caplog: pytest.LogCaptureFixture) -> None:
    report = evaluate(RetrievalStrategy.RRF)

    with caplog.at_level(logging.INFO, logger="eval.eval"):
        _log_summary(report)

    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert warnings == [
        (
            "rrf returned 2.0 hits per query on average, fewer than depth 5; "
            "recall at deeper cutoffs is capped"
        )
    ]
