from dataclasses import dataclass

from app.domain.models import Document
from app.domain.ports import CorpusRepository

INDEX_FORMAT_VERSION = 2
TEXT_POLICY = "exclude_empty_title_and_text_v1"


@dataclass(frozen=True)
class PreparedDocuments:
    documents: list[Document]
    excluded_document_ids: list[str]


def prepare_documents(documents: list[Document]) -> PreparedDocuments:
    indexable: list[Document] = []
    excluded: list[str] = []
    for document in documents:
        if document.index_text.strip():
            indexable.append(document)
        else:
            excluded.append(document.document_id)
    return PreparedDocuments(documents=indexable, excluded_document_ids=excluded)


def policy_manifest(
    repository: CorpusRepository,
    prepared: PreparedDocuments,
) -> dict[str, object]:
    return {
        "index_format_version": INDEX_FORMAT_VERSION,
        "text_policy": TEXT_POLICY,
        "corpus_hash": repository.corpus_hash,
        "corpus_document_count": len(repository.documents()),
        "indexed_document_count": len(prepared.documents),
        "excluded_document_ids": prepared.excluded_document_ids,
    }


def validate_index_policy(
    manifest: dict[str, object],
    repository: CorpusRepository,
    document_ids: list[str],
) -> None:
    if manifest.get("index_format_version") != INDEX_FORMAT_VERSION:
        raise ValueError("Index uses an outdated document policy. Rebuild it with --force.")
    if manifest.get("text_policy") != TEXT_POLICY:
        raise ValueError(
            "Index uses an incompatible document text policy. Rebuild it with --force."
        )
    if manifest.get("corpus_hash") != repository.corpus_hash:
        raise ValueError("Index belongs to a different corpus snapshot")

    prepared = prepare_documents(list(repository.documents()))
    expected_ids = [document.document_id for document in prepared.documents]
    if document_ids != expected_ids:
        raise ValueError("Index document mapping does not match the effective corpus")
    if manifest.get("corpus_document_count") != len(repository.documents()):
        raise ValueError("Index manifest has an invalid corpus document count")
    if manifest.get("indexed_document_count") != len(expected_ids):
        raise ValueError("Index manifest has an invalid indexed document count")
    if manifest.get("excluded_document_ids") != prepared.excluded_document_ids:
        raise ValueError("Index manifest has an invalid empty-document exclusion list")
