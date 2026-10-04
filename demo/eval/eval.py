"""Implementation of the eval pipeline"""

from collections.abc import Sequence
from pathlib import Path
from typing import assert_never

from app.adapters.datasets.qrels import load_qrels
from app.bootstrap import build_container
from app.config import Settings, get_settings
from app.domain.models import RetrievalStrategy
from common.telemetry import TelemetryLogger
from eval.models import EvaluationMode, EvaluationReport
from eval.qrels import runner as qrels_runner

logger = TelemetryLogger(__name__)


def run_evaluation(
    mode: EvaluationMode,
    strategies: Sequence[RetrievalStrategy] | None = None,
    *,
    depth: int = 100,
    sample: int | None = None,
    settings: Settings | None = None,
) -> EvaluationReport:
    """Load the FiQA test split, replay it through each strategy, and persist a report."""
    settings = settings or get_settings()
    container = build_container(settings)
    if container.pipeline is None or container.repository is None:
        raise RuntimeError(f"Dataset unavailable: {container.component_status.get('dataset')}")

    selected = _resolve_strategies(strategies, container.available_strategies)

    qrels = load_qrels(settings.qrels_path)
    if sample is not None:
        qrels = {qid: qrels[qid] for qid in list(qrels)[:sample]}
    queries = {q.query_id: q.text for q in container.repository.queries() if q.query_id in qrels}
    qrels = {qid: relevant for qid, relevant in qrels.items() if qid in queries}

    logger.info(
        f"Evaluating {[s.value for s in selected]} over {len(qrels)} queries at depth {depth}"
    )

    report: EvaluationReport
    if mode is EvaluationMode.QRELS:
        report = qrels_runner.evaluate_strategies(
            container.pipeline,
            qrels,
            queries,
            selected,
            dataset=settings.dataset_name,
            corpus_hash=container.repository.corpus_hash,
            depth=depth,
        )
    elif mode is EvaluationMode.LLM_JUDGE:
        raise NotImplementedError(f"Evaluation mode '{mode}' is not implemented yet.")
    else:
        assert_never(mode)

    _persist(report, settings.eval_dir)
    _log_summary(report)
    return report


def _resolve_strategies(
    requested: Sequence[RetrievalStrategy] | None,
    available: Sequence[RetrievalStrategy],
) -> list[RetrievalStrategy]:
    wanted = list(requested) if requested else list(available)
    selected = [strategy for strategy in wanted if strategy in available]
    if unavailable := [strategy for strategy in wanted if strategy not in available]:
        logger.warning(f"Skipping unavailable strategies: {[s.value for s in unavailable]}")
    if not selected:
        raise RuntimeError(
            "No requested strategy is available. Build indexes / configure providers."
        )
    return selected


def _persist(report: EvaluationReport, eval_dir: Path) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / f"{report.dataset}-{report.mode.value}.json"
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info(f"Wrote evaluation report to {path}")


def _log_summary(report: EvaluationReport) -> None:
    for item in report.reports:
        scores = "  ".join(f"{name}={value:.4f}" for name, value in item.metrics.items())
        logger.info(
            f"{item.strategy.value:<12} [{item.num_queries} q,{item.mean_hits:.1f} hits,"
            f"{item.duration_seconds:.1f}s] {scores}"
        )
        # Fused strategies return at most candidate_k hits, which caps recall above that depth.
        if item.mean_hits < report.depth:
            logger.warning(
                f"{item.strategy.value} returned {item.mean_hits:.1f} hits per query on average, "
                f"fewer than depth {report.depth}; recall at deeper cutoffs is capped"
            )
