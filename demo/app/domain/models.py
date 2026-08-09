from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RetrievalStrategy(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    RRF = "rrf"
    RRF_RERANK = "rrf_rerank"


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str = ""
    text: str

    @property
    def index_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


class QueryExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str


class SearchHit(BaseModel):
    document_id: str
    title: str = ""
    text: str
    rank: int = Field(ge=1)
    score: float
    stage_scores: dict[str, float] = Field(default_factory=dict)
    stage_ranks: dict[str, int] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    strategy: RetrievalStrategy
    dataset: str
    corpus_hash: str
    stage_durations_ms: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    hits: list[SearchHit]
    trace: RetrievalTrace


class GeneratedAnswer(BaseModel):
    text: str
    cited_document_ids: list[str] = Field(default_factory=list)


class RagResult(BaseModel):
    retrieval: RetrievalResult
    answer: GeneratedAnswer | None = None
