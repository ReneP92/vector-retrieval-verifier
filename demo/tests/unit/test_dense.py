import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import BadRequestError, RateLimitError

from app.adapters.datasets.beir import BeirCorpusRepository
from app.adapters.retrieval import dense_openai
from app.adapters.retrieval.corpus_policy import policy_manifest, prepare_documents
from app.adapters.retrieval.dense_openai import (
    DenseRetriever,
    OpenAIEmbedder,
    build_dense_index,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "beir-mini"


class FakeEmbedder:
    model_name = "fake-embedding"

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        materialized = list(texts)
        return np.asarray([[1.0, 0.0] for _ in materialized], dtype=np.float32)


def write_index(path: Path, repository: BeirCorpusRepository, model: str) -> None:
    path.mkdir(parents=True)
    np.save(path / "embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]))
    (path / "doc_ids.json").write_text(json.dumps(["doc-a", "doc-b", "doc-c"]))
    prepared = prepare_documents(list(repository.documents()))
    (path / "manifest.json").write_text(
        json.dumps({**policy_manifest(repository, prepared), "model": model})
    )


def test_searches_normalized_dense_index(tmp_path: Path) -> None:
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")
    index_dir = tmp_path / "dense"
    write_index(index_dir, repository, FakeEmbedder.model_name)

    results = DenseRetriever(repository, index_dir, FakeEmbedder()).search("query", 2)

    assert [result.document_id for result in results] == ["doc-a", "doc-c"]
    assert results[0].stage_ranks == {"dense": 1}


def test_rejects_index_from_another_model(tmp_path: Path) -> None:
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")
    index_dir = tmp_path / "dense"
    write_index(index_dir, repository, "another-model")

    with pytest.raises(ValueError, match="different embedding model"):
        DenseRetriever(repository, index_dir, FakeEmbedder())


def test_reports_when_dense_index_has_not_been_built(tmp_path: Path) -> None:
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")

    with pytest.raises(FileNotFoundError, match="has not been built"):
        DenseRetriever(repository, tmp_path / "dense", FakeEmbedder())


def test_excludes_empty_documents_before_calling_embedder(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "usable", "title": "", "text": "Evidence"}),
                json.dumps({"_id": "empty", "title": "  ", "text": ""}),
            ]
        )
    )
    (dataset / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "Evidence?"}))
    repository = BeirCorpusRepository(dataset / "corpus.jsonl", dataset / "queries.jsonl")
    embedder = CapturingEmbedder()
    index_dir = tmp_path / "dense"

    build_dense_index(repository, index_dir, embedder, batch_size=2)

    assert embedder.inputs == ["Evidence"]
    assert json.loads((index_dir / "doc_ids.json").read_text()) == ["usable"]
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["excluded_document_ids"] == ["empty"]
    assert manifest["indexed_document_count"] == 1


def test_rejects_partial_index_from_old_document_policy(tmp_path: Path) -> None:
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")
    index_dir = tmp_path / "dense"
    index_dir.mkdir()
    matrix = np.lib.format.open_memmap(
        index_dir / "embeddings.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(3, 2),
    )
    matrix.flush()
    del matrix
    (index_dir / "progress.json").write_text(
        json.dumps(
            {
                "completed": 1,
                "corpus_hash": repository.corpus_hash,
                "model": FakeEmbedder.model_name,
                "shape": [3, 2],
            }
        )
    )

    with pytest.raises(ValueError, match="outdated format"):
        build_dense_index(repository, index_dir, FakeEmbedder(), batch_size=2)


def test_resumes_partial_index_with_matching_document_policy(tmp_path: Path) -> None:
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")
    prepared = prepare_documents(list(repository.documents()))
    index_dir = tmp_path / "dense"
    index_dir.mkdir()
    matrix = np.lib.format.open_memmap(
        index_dir / "embeddings.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(3, 2),
    )
    matrix[0] = [1.0, 0.0]
    matrix.flush()
    del matrix
    (index_dir / "progress.json").write_text(
        json.dumps(
            {
                **policy_manifest(repository, prepared),
                "completed": 1,
                "model": CapturingEmbedder.model_name,
                "shape": [3, 2],
            }
        )
    )
    embedder = CapturingEmbedder()

    build_dense_index(repository, index_dir, embedder, batch_size=2)

    assert embedder.inputs == [document.index_text for document in prepared.documents[1:]]
    assert np.load(index_dir / "embeddings.npy").shape == (3, 2)
    assert not (index_dir / "progress.json").exists()


class CapturingEmbedder:
    model_name = "capturing-embedding"

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        materialized = list(texts)
        self.inputs.extend(materialized)
        return np.asarray([[1.0, 0.0] for _ in materialized], dtype=np.float32)


def _status_error(cls, status: int, headers: dict[str, str] | None = None):
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("provider error", response=response, body=None)


def _embedding_response(count: int):
    return SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=[1.0, 0.0]) for i in range(count)]
    )


class _FakeEmbeddings:
    def __init__(self, behaviors: list) -> None:
        self._behaviors = behaviors
        self.calls = 0

    def create(self, *, model: str, input: list[str]):
        behavior = self._behaviors[self.calls]
        self.calls += 1
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


def _embedder_with_client(behaviors: list) -> tuple[OpenAIEmbedder, list[float], _FakeEmbeddings]:
    slept: list[float] = []
    embedder = OpenAIEmbedder(
        api_key="test-key",
        model_name="text-embedding-3-small",
        max_retries=5,
        initial_backoff=1.0,
        max_backoff=60.0,
        sleep=slept.append,
    )
    fake = _FakeEmbeddings(behaviors)
    # A structural stand-in for the OpenAI client; only .embeddings.create is exercised.
    embedder._client = SimpleNamespace(embeddings=fake)  # ty: ignore[invalid-assignment]
    return embedder, slept, fake


def test_embed_retries_rate_limit_then_succeeds() -> None:
    behaviors = [
        _status_error(RateLimitError, 429),
        _status_error(RateLimitError, 429),
        _embedding_response(1),
    ]
    embedder, slept, fake = _embedder_with_client(behaviors)

    vectors = embedder.embed(["passage"])

    assert vectors.shape == (1, 2)
    assert fake.calls == 3
    assert len(slept) == 2


def test_embed_gives_up_after_max_retries() -> None:
    behaviors = [_status_error(RateLimitError, 429) for _ in range(4)]
    embedder, slept, fake = _embedder_with_client(behaviors)
    embedder._max_retries = 3

    with pytest.raises(RateLimitError):
        embedder.embed(["passage"])

    assert fake.calls == 4
    assert len(slept) == 3


def test_embed_does_not_retry_client_error() -> None:
    behaviors = [_status_error(BadRequestError, 400)]
    embedder, slept, fake = _embedder_with_client(behaviors)

    with pytest.raises(BadRequestError):
        embedder.embed(["passage"])

    assert fake.calls == 1
    assert slept == []


def test_backoff_grows_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Collapse full jitter to its upper bound so the base delay is observable.
    monkeypatch.setattr(dense_openai.random, "uniform", lambda _low, high: high)
    embedder, _slept, _fake = _embedder_with_client([])
    error = _status_error(RateLimitError, 429)

    assert embedder._backoff_delay(0, error) == 1.0
    assert embedder._backoff_delay(3, error) == 8.0
    assert embedder._backoff_delay(10, error) == 60.0


def test_backoff_honors_retry_after_within_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dense_openai.random, "uniform", lambda _low, high: high)
    embedder, _slept, _fake = _embedder_with_client([])

    with_hint = _status_error(RateLimitError, 429, {"retry-after": "5"})
    assert embedder._backoff_delay(0, with_hint) == 5.0

    huge_hint = _status_error(RateLimitError, 429, {"retry-after": "1000"})
    assert embedder._backoff_delay(0, huge_hint) == 60.0
