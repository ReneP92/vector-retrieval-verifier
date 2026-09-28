import json
import random
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.adapters.retrieval.corpus_policy import (
    policy_manifest,
    prepare_documents,
    validate_index_policy,
)
from app.domain.models import SearchHit
from app.domain.ports import CorpusRepository, TextEmbedder

# Errors worth retrying: provider-side rate limits, transient 5xx, and network
# blips. Client errors (auth, bad request) are not retried - they never recover.
_RETRYABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
        max_retries: int = 8,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # Retries are handled here, so the SDK does not add a second, hidden layer.
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._model_name = model_name
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._sleep = sleep

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        materialized = list(texts)
        if not materialized:
            return np.empty((0, 0), dtype=np.float32)
        response = self._create_with_retry(materialized)
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(materialized):
            raise ValueError("Embedding provider returned an unexpected number of vectors")
        return np.asarray([item.embedding for item in ordered], dtype=np.float32)

    def _create_with_retry(self, inputs: list[str]):
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.embeddings.create(model=self._model_name, input=inputs)
            except _RETRYABLE_ERRORS as error:
                if attempt == self._max_retries:
                    raise
                self._sleep(self._backoff_delay(attempt, error))

    def _backoff_delay(self, attempt: int, error: Exception) -> float:
        delay = min(self._max_backoff, self._initial_backoff * (2**attempt))
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            delay = min(self._max_backoff, max(delay, retry_after))
        # Full jitter avoids synchronized retries hammering the provider in lockstep.
        return random.uniform(0.0, delay)


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        repository: CorpusRepository,
        index_dir: Path,
        embedder: TextEmbedder,
    ) -> None:
        embeddings_path = index_dir / "embeddings.npy"
        doc_ids_path = index_dir / "doc_ids.json"
        manifest_path = index_dir / "manifest.json"
        if not index_dir.exists():
            raise FileNotFoundError(
                "Dense index has not been built. Run "
                "`uv run python -m scripts.build_indexes --dense`."
            )
        if (
            not embeddings_path.is_file()
            or not doc_ids_path.is_file()
            or not manifest_path.is_file()
        ):
            raise FileNotFoundError(
                f"Dense index is incomplete: {index_dir}. Re-run the dense index command to resume."
            )

        self._repository = repository
        self._embedder = embedder
        self._embeddings = np.load(embeddings_path, mmap_mode="r")
        self._document_ids: list[str] = json.loads(doc_ids_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if self._embeddings.shape[0] != len(self._document_ids):
            raise ValueError("Dense embeddings and document mapping have different lengths")
        validate_index_policy(manifest, repository, self._document_ids)
        if manifest.get("model") != embedder.model_name:
            raise ValueError("Dense index belongs to a different embedding model")

    def search(self, query: str, k: int) -> list[SearchHit]:
        query_embedding = self._embedder.embed([query])[0]
        query_embedding = _normalize_vector(query_embedding)
        scores = self._embeddings @ query_embedding
        top_indices = np.argsort(-scores, kind="stable")[: min(k, len(self._document_ids))]

        hits: list[SearchHit] = []
        for rank, index in enumerate(top_indices, start=1):
            document = self._repository.document(self._document_ids[int(index)])
            score = float(scores[index])
            hits.append(
                SearchHit(
                    document_id=document.document_id,
                    title=document.title,
                    text=document.text,
                    rank=rank,
                    score=score,
                    stage_scores={self.name: score},
                    stage_ranks={self.name: rank},
                )
            )
        return hits


def build_dense_index(
    repository: CorpusRepository,
    index_dir: Path,
    embedder: TextEmbedder,
    batch_size: int,
) -> None:
    prepared = prepare_documents(list(repository.documents()))
    documents = prepared.documents
    if not documents:
        raise ValueError("Cannot build a dense index for an empty corpus")

    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = index_dir / "embeddings.npy"
    partial_path = index_dir / "embeddings.partial.npy"
    progress_path = index_dir / "progress.json"
    policy = policy_manifest(repository, prepared)

    start = 0
    matrix: np.memmap | None = None
    if progress_path.is_file() and partial_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("index_format_version") != policy["index_format_version"]:
            raise ValueError(
                "Partial dense index uses an outdated format. Rebuild it with --force."
            )
        if progress.get("text_policy") != policy["text_policy"]:
            raise ValueError(
                "Partial dense index uses an incompatible text policy. Rebuild it with --force."
            )
        if progress.get("corpus_hash") != repository.corpus_hash:
            raise ValueError("Partial dense index belongs to a different corpus")
        if progress.get("model") != embedder.model_name:
            raise ValueError("Partial dense index belongs to a different embedding model")
        if progress.get("indexed_document_count") != len(documents):
            raise ValueError(
                "Partial dense index has a different effective corpus size. Rebuild it with --force."
            )
        if progress.get("excluded_document_ids") != prepared.excluded_document_ids:
            raise ValueError(
                "Partial dense index has different document exclusions. Rebuild it with --force."
            )
        start = int(progress["completed"])
        shape = tuple(progress["shape"])
        if not shape or shape[0] != len(documents):
            raise ValueError("Partial dense index has an invalid shape. Rebuild it with --force.")
        matrix = np.lib.format.open_memmap(partial_path, mode="r+", dtype=np.float32, shape=shape)

    for offset in range(start, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        vectors = _normalize_rows(embedder.embed(document.index_text for document in batch))
        if matrix is None:
            matrix = np.lib.format.open_memmap(
                partial_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(documents), vectors.shape[1]),
            )
        matrix[offset : offset + len(batch)] = vectors
        matrix.flush()
        progress_path.write_text(
            json.dumps(
                {
                    **policy,
                    "completed": offset + len(batch),
                    "model": embedder.model_name,
                    "shape": list(matrix.shape),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if matrix is None:
        raise RuntimeError("Dense index builder did not produce embeddings")
    del matrix
    partial_path.replace(embeddings_path)
    progress_path.unlink(missing_ok=True)
    (index_dir / "doc_ids.json").write_text(
        json.dumps([document.document_id for document in documents]),
        encoding="utf-8",
    )
    (index_dir / "manifest.json").write_text(
        json.dumps(
            {
                **policy,
                "model": embedder.model_name,
                "normalized": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Embedding provider returned a zero vector")
    return vector / norm


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding provider returned at least one zero vector")
    return vectors / norms
