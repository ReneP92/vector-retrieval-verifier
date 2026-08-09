import json
from pathlib import Path

import bm25s

from app.adapters.retrieval.corpus_policy import (
    policy_manifest,
    prepare_documents,
    validate_index_policy,
)
from app.domain.models import SearchHit
from app.domain.ports import CorpusRepository


class BM25Retriever:
    name = "bm25"

    def __init__(self, repository: CorpusRepository, index_dir: Path) -> None:
        doc_ids_path = index_dir / "doc_ids.json"
        manifest_path = index_dir / "manifest.json"
        if not doc_ids_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"BM25 index is incomplete: {index_dir}")

        self._repository = repository
        self._document_ids: list[str] = json.loads(doc_ids_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_index_policy(manifest, repository, self._document_ids)
        self._retriever = bm25s.BM25.load(str(index_dir / "model"), load_corpus=False)

    def search(self, query: str, k: int) -> list[SearchHit]:
        result_count = min(k, len(self._document_ids))
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        indices, scores = self._retriever.retrieve(tokens, k=result_count, show_progress=False)

        hits: list[SearchHit] = []
        for rank, (index, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1):
            document = self._repository.document(self._document_ids[int(index)])
            numeric_score = float(score)
            hits.append(
                SearchHit(
                    document_id=document.document_id,
                    title=document.title,
                    text=document.text,
                    rank=rank,
                    score=numeric_score,
                    stage_scores={self.name: numeric_score},
                    stage_ranks={self.name: rank},
                )
            )
        return hits


def build_bm25_index(repository: CorpusRepository, index_dir: Path) -> None:
    prepared = prepare_documents(list(repository.documents()))
    documents = prepared.documents
    if not documents:
        raise ValueError("Cannot build a BM25 index for an empty corpus")

    index_dir.mkdir(parents=True, exist_ok=True)
    corpus_tokens = bm25s.tokenize(
        [document.index_text for document in documents],
        stopwords="en",
        show_progress=True,
    )
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(str(index_dir / "model"))
    (index_dir / "doc_ids.json").write_text(
        json.dumps([document.document_id for document in documents]),
        encoding="utf-8",
    )
    (index_dir / "manifest.json").write_text(
        json.dumps(
            {
                **policy_manifest(repository, prepared),
                "tokenizer": {"stopwords": "en"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
