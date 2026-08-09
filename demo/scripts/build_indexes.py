import argparse
import shutil
from pathlib import Path

from app.adapters.datasets.beir import BeirCorpusRepository
from app.adapters.datasets.qrels import find_excluded_positive_qrels
from app.adapters.retrieval.bm25 import build_bm25_index
from app.adapters.retrieval.corpus_policy import prepare_documents
from app.adapters.retrieval.dense_openai import OpenAIEmbedder, build_dense_index
from app.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FiQA retrieval indexes")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--bm25", action="store_true", help="build only the BM25 index")
    selection.add_argument("--dense", action="store_true", help="build only the dense index")
    selection.add_argument("--all", action="store_true", help="build both indexes")
    parser.add_argument("--force", action="store_true", help="replace completed indexes")
    args = parser.parse_args()

    settings = Settings()
    repository = BeirCorpusRepository(settings.corpus_path, settings.queries_path)
    prepared = prepare_documents(list(repository.documents()))
    _report_exclusions(settings.qrels_path, prepared.excluded_document_ids)

    if args.bm25 or args.all:
        _prepare_target(settings.bm25_dir, args.force, "BM25")
        print(f"Building BM25 index for {len(prepared.documents):,} passages", flush=True)
        try:
            build_bm25_index(repository, settings.bm25_dir)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        print(f"BM25 index is ready at {settings.bm25_dir}", flush=True)

    if args.dense or args.all:
        if settings.openai_api_key is None:
            raise SystemExit("RAG_DEMO_OPENAI_API_KEY is required to build the dense index")
        _prepare_target(settings.dense_dir, args.force, "dense", allow_partial=True)
        embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key.get_secret_value(),
            model_name=settings.embedding_model,
            base_url=settings.openai_base_url or None,
        )
        print(
            f"Building {settings.embedding_model} index for {len(prepared.documents):,} passages",
            flush=True,
        )
        try:
            build_dense_index(
                repository,
                settings.dense_dir,
                embedder,
                settings.embedding_batch_size,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from None
        print(f"Dense index is ready at {settings.dense_dir}", flush=True)


def _prepare_target(
    path: Path,
    force: bool,
    label: str,
    *,
    allow_partial: bool = False,
) -> None:
    manifest = path / "manifest.json"
    partial = path / "progress.json"
    if manifest.is_file() and not force:
        raise SystemExit(f"{label} index already exists at {path}. Use --force to replace it.")
    if path.exists() and force:
        shutil.rmtree(path)
    if path.exists() and not allow_partial and not manifest.is_file():
        raise SystemExit(f"Incomplete {label} index found at {path}. Use --force to replace it.")
    if path.exists() and allow_partial and not partial.is_file() and not manifest.is_file():
        raise SystemExit(
            f"Unknown files found in {label} index path {path}. Use --force to replace it."
        )


def _report_exclusions(qrels_path: Path, excluded_document_ids: list[str]) -> None:
    if not excluded_document_ids:
        return
    print(
        f"Excluding {len(excluded_document_ids)} passages with empty title and text "
        "from every retrieval index.",
        flush=True,
    )
    conflicts = find_excluded_positive_qrels(qrels_path, excluded_document_ids)
    for conflict in conflicts:
        print(
            "Warning: excluded empty passage "
            f"{conflict.document_id} has positive qrel {conflict.score} "
            f"for query {conflict.query_id}.",
            flush=True,
        )


if __name__ == "__main__":
    main()
