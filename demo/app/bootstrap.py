from dataclasses import dataclass, field

from app.adapters.datasets.beir import BeirCorpusRepository
from app.adapters.generation.openai_compatible import OpenAICompatibleGenerator
from app.adapters.retrieval.bm25 import BM25Retriever
from app.adapters.retrieval.cohere_reranker import CohereReranker
from app.adapters.retrieval.dense_openai import DenseRetriever, OpenAIEmbedder
from app.config import Settings
from app.domain.models import QueryExample, RetrievalStrategy
from app.services.rag import RagService
from app.services.retrieval import RetrievalPipeline


@dataclass
class DemoContainer:
    settings: Settings
    rag: RagService | None
    pipeline: RetrievalPipeline | None
    repository: BeirCorpusRepository | None
    component_status: dict[str, str] = field(default_factory=dict)

    @property
    def available_strategies(self) -> list[RetrievalStrategy]:
        return self.pipeline.available_strategies if self.pipeline else []

    @property
    def generation_available(self) -> bool:
        return bool(self.rag and self.rag.generation_available)

    def sample_queries(self, limit: int = 8) -> list[QueryExample]:
        return self.repository.sample_queries(limit) if self.repository else []


def build_container(settings: Settings) -> DemoContainer:
    status: dict[str, str] = {}
    try:
        repository = BeirCorpusRepository(settings.corpus_path, settings.queries_path)
        status["dataset"] = f"ready ({len(repository.documents()):,} passages)"
    except (FileNotFoundError, ValueError) as error:
        status["dataset"] = str(error)
        return DemoContainer(
            settings=settings,
            rag=None,
            pipeline=None,
            repository=None,
            component_status=status,
        )

    bm25 = None
    try:
        bm25 = BM25Retriever(repository, settings.bm25_dir)
        status["bm25"] = "ready"
    except (FileNotFoundError, ValueError) as error:
        status["bm25"] = str(error)

    embedder = None
    dense = None
    openai_key = (
        settings.openai_api_key.get_secret_value() if settings.openai_api_key is not None else None
    )
    if openai_key:
        embedder = OpenAIEmbedder(
            api_key=openai_key,
            model_name=settings.embedding_model,
            base_url=settings.openai_base_url or None,
        )
        try:
            dense = DenseRetriever(repository, settings.dense_dir, embedder)
            status["dense"] = "ready"
        except (FileNotFoundError, ValueError) as error:
            status["dense"] = str(error)
    else:
        status["dense"] = "RAG_DEMO_OPENAI_API_KEY is not configured"

    reranker = None
    cohere_key = (
        settings.cohere_api_key.get_secret_value() if settings.cohere_api_key is not None else None
    )
    if cohere_key:
        reranker = CohereReranker(
            api_key=cohere_key,
            model_name=settings.cohere_rerank_model,
            cache_dir=settings.cache_dir,
        )
        status["reranker"] = "ready"
    else:
        status["reranker"] = "RAG_DEMO_COHERE_API_KEY is not configured"

    generator = None
    if openai_key:
        generator = OpenAICompatibleGenerator(
            api_key=openai_key,
            model_name=settings.chat_model,
            base_url=settings.openai_base_url or None,
        )
        status["generator"] = "ready"
    else:
        status["generator"] = "RAG_DEMO_OPENAI_API_KEY is not configured"

    pipeline = RetrievalPipeline(
        dataset_name=settings.dataset_name,
        corpus_hash=repository.corpus_hash,
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        candidate_k=settings.candidate_k,
        rrf_k=settings.rrf_k,
    )
    rag = RagService(pipeline, generator, settings.generation_k)
    return DemoContainer(
        settings=settings,
        rag=rag,
        pipeline=pipeline,
        repository=repository,
        component_status=status,
    )
