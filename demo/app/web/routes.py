from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.bootstrap import DemoContainer
from app.domain.models import RetrievalStrategy


def create_web_router(
    templates: Jinja2Templates,
    asset_versions: dict[str, str],
) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        container: DemoContainer = request.app.state.container
        default_strategy = (
            container.available_strategies[-1]
            if container.available_strategies
            else RetrievalStrategy.RRF_RERANK
        )
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "container": container,
                "asset_versions": asset_versions,
                "strategies": list(RetrievalStrategy),
                "default_strategy": default_strategy,
                "sample_queries": container.sample_queries(),
            },
        )

    @router.post("/search", response_class=HTMLResponse)
    async def search(
        request: Request,
        query: Annotated[str, Form(min_length=1, max_length=2000)],
        strategy: Annotated[RetrievalStrategy, Form()],
        top_k: Annotated[int, Form(ge=1, le=100)] = 10,
        generate_answer: Annotated[bool, Form()] = False,
    ) -> HTMLResponse:
        container: DemoContainer = request.app.state.container
        if container.rag is None:
            return templates.TemplateResponse(
                request=request,
                name="partials/error.html",
                context={"message": "Download FiQA and build at least one index first."},
                status_code=503,
            )
        try:
            result = await run_in_threadpool(
                container.rag.query,
                query,
                strategy,
                top_k,
                generate_answer=generate_answer,
            )
        except (RuntimeError, ValueError) as error:
            return templates.TemplateResponse(
                request=request,
                name="partials/error.html",
                context={"message": str(error)},
                status_code=422,
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/results.html",
            context={"result": result},
        )

    return router
