from dataclasses import dataclass

from app.domain.models import SearchHit


@dataclass
class FakeRetriever:
    name: str
    document_ids: list[str]

    def search(self, query: str, k: int) -> list[SearchHit]:
        del query
        return [
            SearchHit(
                document_id=document_id,
                text=document_id,
                rank=rank,
                score=float(10 - rank),
                stage_scores={self.name: float(10 - rank)},
                stage_ranks={self.name: rank},
            )
            for rank, document_id in enumerate(self.document_ids[:k], start=1)
        ]


class FakeReranker:
    model_name = "fake-cross-encoder"

    def rerank(self, query: str, candidates: list[SearchHit], k: int) -> list[SearchHit]:
        del query
        return [
            hit.model_copy(
                update={
                    "rank": rank,
                    "score": float(100 - rank),
                    "stage_scores": {**hit.stage_scores, "rerank": float(100 - rank)},
                    "stage_ranks": {**hit.stage_ranks, "rerank": rank},
                }
            )
            for rank, hit in enumerate(reversed(candidates[:k]), start=1)
        ]
