from enum import StrEnum
from pydantic import BaseModel

from app.domain.models import RetrievalStrategy


class EvaluationMode(StrEnum):
    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"


class StrategyReport(BaseModel):
    strategy: RetrievalStrategy
    num_queries: int
    metrics: dict[str, float]
    duration_seconds: float


class EvaluationReport(BaseModel):
    dataset: str
    corpus_hash: str
    mode: EvaluationMode
    depth: int
    reports: list[StrategyReport]
