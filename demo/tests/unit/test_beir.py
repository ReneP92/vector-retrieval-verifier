from pathlib import Path

import pytest

from app.adapters.datasets.beir import BeirCorpusRepository

FIXTURE = Path(__file__).parents[1] / "fixtures" / "beir-mini"


def test_loads_beir_corpus_and_queries() -> None:
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")

    assert len(repository.documents()) == 3
    assert repository.document("doc-a").title == "Emergency funds"
    assert repository.sample_queries(1)[0].query_id == "query-1"
    assert len(repository.corpus_hash) == 64


def test_rejects_unknown_document_id() -> None:
    repository = BeirCorpusRepository(FIXTURE / "corpus.jsonl", FIXTURE / "queries.jsonl")

    with pytest.raises(KeyError, match="Unknown BEIR document ID"):
        repository.document("missing")
