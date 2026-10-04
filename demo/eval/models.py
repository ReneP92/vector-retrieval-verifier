from enum import StrEnum

from pydantic import BaseModel

from app.config import QrelSplit
from app.domain.models import RetrievalStrategy


class EvaluationMode(StrEnum):
    QRELS = "qrels"
    LLM_JUDGE = "llm_judge"


class StrategyReport(BaseModel):
    strategy: RetrievalStrategy
    num_queries: int
    metrics: dict[str, float]
    mean_hits: float
    parameters: dict[str, str | int | float | bool]
    duration_seconds: float


class EvaluationReport(BaseModel):
    dataset: str
    corpus_hash: str
    mode: EvaluationMode
    depth: int
    reports: list[StrategyReport]
    split: QrelSplit
