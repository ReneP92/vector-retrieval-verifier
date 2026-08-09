from app.adapters.retrieval.corpus_policy import prepare_documents
from app.domain.models import Document


def test_preserves_order_while_excluding_only_whitespace_documents() -> None:
    prepared = prepare_documents(
        [
            Document(document_id="text", text="Evidence"),
            Document(document_id="empty", title="  ", text="\n"),
            Document(document_id="title", title="Useful title", text=""),
        ]
    )

    assert [document.document_id for document in prepared.documents] == ["text", "title"]
    assert prepared.excluded_document_ids == ["empty"]
