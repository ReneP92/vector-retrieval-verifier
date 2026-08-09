from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.bootstrap import DemoContainer
from app.domain.models import QueryExample, RagResult, RetrievalStrategy
from app.services.retrieval import UnavailableStrategyError

router = APIRouter(prefix="/api")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: RetrievalStrategy = RetrievalStrategy.RRF_RERANK
    top_k: int = Field(default=10, ge=1, le=100)
    generate_answer: bool = False


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]
    available_strategies: list[RetrievalStrategy]
    generation_available: bool


def container_from(request: Request) -> DemoContainer:
    return request.app.state.container


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    container = container_from(request)
    return HealthResponse(
        status="ready" if container.available_strategies else "setup_required",
        components=container.component_status,
        available_strategies=container.available_strategies,
        generation_available=container.generation_available,
    )


@router.get("/sample-queries", response_model=list[QueryExample])
def sample_queries(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=50)] = 8,
) -> list[QueryExample]:
    return container_from(request).sample_queries(limit)


@router.post("/query", response_model=RagResult)
async def query(payload: QueryRequest, request: Request) -> RagResult:
    container = container_from(request)
    if container.rag is None:
        raise HTTPException(status_code=503, detail="Dataset and indexes are not ready")
    try:
        return await run_in_threadpool(
            container.rag.query,
            payload.query,
            payload.strategy,
            payload.top_k,
            generate_answer=payload.generate_answer,
        )
    except (UnavailableStrategyError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
