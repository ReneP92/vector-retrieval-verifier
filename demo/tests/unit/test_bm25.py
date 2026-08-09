import json
from pathlib import Path

import pytest

from app.adapters.datasets.beir import BeirCorpusRepository
from app.adapters.retrieval.bm25 import BM25Retriever, build_bm25_index

FIXTURE = Path(__file__).parents[1] / "fixtures" / "beir-mini"


def test_builds_and_searches_bm25_index(tmp_path: Path) -> None:
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")
    index_dir = tmp_path / "bm25"
    build_bm25_index(repository, index_dir)

    retriever = BM25Retriever(repository, index_dir)
    results = retriever.search("emergency savings", 2)

    assert results[0].document_id == "doc-a"
    assert results[0].stage_ranks == {"bm25": 1}
    assert (index_dir / "manifest.json").is_file()


def test_rejects_index_from_another_corpus_snapshot(tmp_path: Path) -> None:
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")
    index_dir = tmp_path / "bm25"
    build_bm25_index(repository, index_dir)
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_hash"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="different corpus snapshot"):
        BM25Retriever(repository, index_dir)


def test_excludes_empty_documents_from_bm25_mapping(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "usable", "title": "", "text": "Evidence"}),
                json.dumps({"_id": "empty", "title": "", "text": ""}),
            ]
        )
    )
    (dataset / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "Evidence?"}))
    repository = BeirCorpusRepository(dataset / "corpus.jsonl", dataset / "queries.jsonl")
    index_dir = tmp_path / "bm25-empty"

    build_bm25_index(repository, index_dir)

    assert json.loads((index_dir / "doc_ids.json").read_text()) == ["usable"]
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["excluded_document_ids"] == ["empty"]
