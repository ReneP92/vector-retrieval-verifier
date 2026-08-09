from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from app.domain.models import Document, GeneratedAnswer, QueryExample, SearchHit


class CorpusRepository(Protocol):
    @property
    def corpus_hash(self) -> str: ...

    def documents(self) -> Sequence[Document]: ...

    def document(self, document_id: str) -> Document: ...

    def sample_queries(self, limit: int) -> list[QueryExample]: ...


class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def search(self, query: str, k: int) -> list[SearchHit]: ...


class TextEmbedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, texts: Iterable[str]) -> np.ndarray: ...


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def rerank(self, query: str, candidates: list[SearchHit], k: int) -> list[SearchHit]: ...


class AnswerGenerator(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate(self, query: str, evidence: list[SearchHit]) -> GeneratedAnswer: ...
