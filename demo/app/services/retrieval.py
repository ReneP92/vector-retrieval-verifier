from time import perf_counter

from app.adapters.retrieval.rrf import reciprocal_rank_fusion
from app.domain.models import (
    RetrievalResult,
    RetrievalStrategy,
    RetrievalTrace,
    SearchHit,
)
from app.domain.ports import Reranker, Retriever


class UnavailableStrategyError(RuntimeError):
    pass


class RetrievalPipeline:
    def __init__(
        self,
        *,
        dataset_name: str,
        corpus_hash: str,
        bm25: Retriever | None,
        dense: Retriever | None,
        reranker: Reranker | None,
        candidate_k: int,
        rrf_k: int,
    ) -> None:
        self._dataset_name = dataset_name
        self._corpus_hash = corpus_hash
        self._bm25 = bm25
        self._dense = dense
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    @property
    def available_strategies(self) -> list[RetrievalStrategy]:
        available: list[RetrievalStrategy] = []
        if self._bm25 is not None:
            available.append(RetrievalStrategy.BM25)
        if self._dense is not None:
            available.append(RetrievalStrategy.DENSE)
        if self._bm25 is not None and self._dense is not None:
            available.append(RetrievalStrategy.RRF)
            if self._reranker is not None:
                available.append(RetrievalStrategy.RRF_RERANK)
        return available

    def search(self, query: str, strategy: RetrievalStrategy, k: int) -> RetrievalResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty")
        if k < 1:
            raise ValueError("Result count must be positive")
        if strategy not in self.available_strategies:
            raise UnavailableStrategyError(
                f"Strategy '{strategy}' is unavailable. Build its index and configure its provider."
            )

        durations: dict[str, float] = {}
        parameters: dict[str, str | int | float | bool] = {
            "candidate_k": self._candidate_k,
            "result_k": k,
            "rrf_k": self._rrf_k,
        }

        if strategy == RetrievalStrategy.BM25:
            hits = self._timed_search(self._required_bm25(), normalized_query, k, durations)
        elif strategy == RetrievalStrategy.DENSE:
            hits = self._timed_search(self._required_dense(), normalized_query, k, durations)
        else:
            bm25_hits = self._timed_search(
                self._required_bm25(), normalized_query, self._candidate_k, durations
            )
            dense_hits = self._timed_search(
                self._required_dense(), normalized_query, self._candidate_k, durations
            )
            started = perf_counter()
            fused = reciprocal_rank_fusion(
                [bm25_hits, dense_hits],
                rrf_k=self._rrf_k,
                limit=self._candidate_k,
            )
            durations["rrf"] = _elapsed_ms(started)
            if strategy == RetrievalStrategy.RRF:
                hits = _rerank_positions(fused[:k])
            else:
                reranker = self._required_reranker()
                started = perf_counter()
                hits = reranker.rerank(normalized_query, fused, k)
                durations["rerank"] = _elapsed_ms(started)
                parameters["reranker_model"] = reranker.model_name

        return RetrievalResult(
            query=normalized_query,
            hits=hits,
            trace=RetrievalTrace(
                strategy=strategy,
                dataset=self._dataset_name,
                corpus_hash=self._corpus_hash,
                stage_durations_ms=durations,
                parameters=parameters,
            ),
        )

    @staticmethod
    def _timed_search(
        retriever: Retriever,
        query: str,
        k: int,
        durations: dict[str, float],
    ) -> list[SearchHit]:
        started = perf_counter()
        hits = retriever.search(query, k)
        durations[retriever.name] = _elapsed_ms(started)
        return hits

    def _required_bm25(self) -> Retriever:
        if self._bm25 is None:
            raise UnavailableStrategyError("BM25 index is unavailable")
        return self._bm25

    def _required_dense(self) -> Retriever:
        if self._dense is None:
            raise UnavailableStrategyError("Dense index or embedding provider is unavailable")
        return self._dense

    def _required_reranker(self) -> Reranker:
        if self._reranker is None:
            raise UnavailableStrategyError("Cross-encoder provider is unavailable")
        return self._reranker


def _rerank_positions(hits: list[SearchHit]) -> list[SearchHit]:
    return [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(hits, start=1)]


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)
