import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from app.adapters.datasets.beir import BeirCorpusRepository
from app.adapters.retrieval.corpus_policy import policy_manifest, prepare_documents
from app.adapters.retrieval.dense_openai import (
    DenseRetriever,
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
