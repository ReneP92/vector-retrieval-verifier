from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEMO_ROOT = Path(__file__).resolve().parents[1]


class QrelSplit(StrEnum):
    DEV = "dev"
    TEST = "test"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=DEMO_ROOT / ".env",
        env_prefix="RAG_DEMO_",
        extra="ignore",
    )

    app_name: str = "RAG Retrieval Workbench"
    dataset_name: str = "fiqa"
    data_dir: Path = DEMO_ROOT / "var" / "datasets" / "fiqa"
    index_dir: Path = DEMO_ROOT / "var" / "indexes" / "fiqa"
    cache_dir: Path = DEMO_ROOT / "var" / "cache"

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(default=128, ge=1, le=2048)
    chat_model: str = "gpt-4.1-mini"

    cohere_api_key: SecretStr | None = None
    cohere_rerank_model: str = "rerank-v4.0-fast"

    rrf_k: int = Field(default=60, ge=1)
    candidate_k: int = Field(default=50, ge=1, le=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    generation_k: int = Field(default=5, ge=1, le=20)

    judge_model: str = "gpt-4.1"
    judge_concurrency: int = 8

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus.jsonl"

    @property
    def queries_path(self) -> Path:
        return self.data_dir / "queries.jsonl"

    def qrels_path(self, split: QrelSplit) -> Path:
        return self.data_dir / "qrels" / f"{split}.tsv"

    @property
    def eval_dir(self) -> Path:
        return DEMO_ROOT / "var" / "eval"

    @property
    def bm25_dir(self) -> Path:
        return self.index_dir / "bm25"

    @property
    def dense_dir(self) -> Path:
        return self.index_dir / "dense" / self.embedding_model.replace("/", "--")


@lru_cache
def get_settings() -> Settings:
    return Settings()
