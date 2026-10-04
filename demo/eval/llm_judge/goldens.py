"""
The golden data model
"""
from collections.abc import Mapping
import logging

from deepeval.dataset import Golden
from app.adapters.datasets.beir import BeirCorpusRepository
from eval.models import QrelSplit

from common.telemetry import TelemetryLogger

TelemetryLogger.configure(level=logging.INFO)
logger = TelemetryLogger(name=__name__)


def build_goldens(
    repository: BeirCorpusRepository,
    qrels: Mapping[str, set[str]], split: QrelSplit
) -> list[Golden]:
    queries = {q.query_id: q.text for q in repository.queries()}
    goldens = []
    for query_id, relevant in qrels.items():
        ids = sorted(relevant)
        texts = [t for t in (repository.document(d).text.strip() for d in ids ) if t]
        if not texts:
            logger.warning(f"Dropping query: {query_id}: no non-empty reference passage")
            continue
        goldens.append(
            Golden(
                input=queries[query_id],
                expected_output="\n\n".join(texts),
                context=texts,
                additional_metadata={"query_id": query_id, "releveant_ids": ids, "split": split.value},
                multimodal=False
            )
        )
    return goldens
