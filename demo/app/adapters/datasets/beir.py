import hashlib
import json
from pathlib import Path

from app.domain.models import Document, QueryExample


class BeirCorpusRepository:
    def __init__(self, corpus_path: Path, queries_path: Path) -> None:
        if not corpus_path.is_file():
            raise FileNotFoundError(f"BEIR corpus not found: {corpus_path}")
        if not queries_path.is_file():
            raise FileNotFoundError(f"BEIR queries not found: {queries_path}")

        self._corpus_path = corpus_path
        self._documents = self._load_documents(corpus_path)
        self._documents_by_id = {document.document_id: document for document in self._documents}
        self._queries = self._load_queries(queries_path)
        self._corpus_hash = self._sha256(corpus_path)

        if len(self._documents_by_id) != len(self._documents):
            raise ValueError("BEIR corpus contains duplicate document IDs")

    @property
    def corpus_hash(self) -> str:
        return self._corpus_hash

    def documents(self) -> list[Document]:
        return self._documents

    def document(self, document_id: str) -> Document:
        try:
            return self._documents_by_id[document_id]
        except KeyError as error:
            raise KeyError(f"Unknown BEIR document ID: {document_id}") from error

    def sample_queries(self, limit: int) -> list[QueryExample]:
        return self._queries[: max(0, limit)]

    def queries(self) -> list[QueryExample]:
        return self._queries

    @staticmethod
    def _load_documents(path: Path) -> list[Document]:
        documents: list[Document] = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    documents.append(
                        Document(
                            document_id=str(record["_id"]),
                            title=str(record.get("title", "")),
                            text=str(record["text"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid corpus record at {path}:{line_number}") from error
        return documents

    @staticmethod
    def _load_queries(path: Path) -> list[QueryExample]:
        queries: list[QueryExample] = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    queries.append(
                        QueryExample(query_id=str(record["_id"]), text=str(record["text"]))
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid query record at {path}:{line_number}") from error
        return queries

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
