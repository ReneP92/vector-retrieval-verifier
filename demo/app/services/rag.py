from app.domain.models import RagResult, RetrievalStrategy
from app.domain.ports import AnswerGenerator
from app.services.retrieval import RetrievalPipeline


class RagService:
    def __init__(
        self,
        pipeline: RetrievalPipeline,
        generator: AnswerGenerator | None,
        generation_k: int,
    ) -> None:
        self._pipeline = pipeline
        self._generator = generator
        self._generation_k = generation_k

    @property
    def generation_available(self) -> bool:
        return self._generator is not None

    def query(
        self,
        query: str,
        strategy: RetrievalStrategy,
        k: int,
        *,
        generate_answer: bool,
    ) -> RagResult:
        retrieval = self._pipeline.search(query, strategy, k)
        if not generate_answer:
            return RagResult(retrieval=retrieval)
        if self._generator is None:
            raise RuntimeError("Answer generation requires an OpenAI-compatible provider")
        answer = self._generator.generate(query, retrieval.hits[: self._generation_k])
        return RagResult(retrieval=retrieval, answer=answer)
