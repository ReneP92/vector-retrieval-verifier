import hashlib
import json
from pathlib import Path

import cohere

from app.domain.models import SearchHit


class CohereReranker:
    def __init__(self, *, api_key: str, model_name: str, cache_dir: Path) -> None:
        self._client = cohere.ClientV2(api_key=api_key)
        self._model_name = model_name
        self._cache_dir = cache_dir / "rerank"

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(self, query: str, candidates: list[SearchHit], k: int) -> list[SearchHit]:
        if not candidates:
            return []

        cached = self._read_cache(query, candidates)
        if cached is None:
            response = self._client.rerank(
                model=self._model_name,
                query=query,
                documents=[_document_text(candidate) for candidate in candidates],
                top_n=len(candidates),
            )
            cached = [
                {"index": int(result.index), "score": float(result.relevance_score)}
                for result in response.results
            ]
            self._write_cache(query, candidates, cached)

        seen: set[int] = set()
        reranked: list[SearchHit] = []
        for rank, result in enumerate(cached[:k], start=1):
            index = int(result["index"])
            if index < 0 or index >= len(candidates) or index in seen:
                raise ValueError("Cohere returned an invalid or duplicate candidate index")
            seen.add(index)
            source = candidates[index]
            score = float(result["score"])
            reranked.append(
                source.model_copy(
                    update={
                        "rank": rank,
                        "score": score,
                        "stage_scores": {**source.stage_scores, "rerank": score},
                        "stage_ranks": {**source.stage_ranks, "rerank": rank},
                    }
                )
            )
        return reranked

    def _cache_key(self, query: str, candidates: list[SearchHit]) -> str:
        payload = json.dumps(
            {
                "model": self._model_name,
                "query": query,
                "candidates": [candidate.document_id for candidate in candidates],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _read_cache(
        self, query: str, candidates: list[SearchHit]
    ) -> list[dict[str, int | float]] | None:
        path = self._cache_dir / f"{self._cache_key(query, candidates)}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data["results"])

    def _write_cache(
        self,
        query: str,
        candidates: list[SearchHit],
        results: list[dict[str, int | float]],
    ) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{self._cache_key(query, candidates)}.json"
        path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


def _document_text(hit: SearchHit) -> str:
    return f"{hit.title}\n{hit.text}" if hit.title else hit.text
