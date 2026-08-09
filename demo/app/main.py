from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.bootstrap import build_container
from app.config import Settings, get_settings
from app.web.api import router as api_router
from app.web.routes import create_web_router

WEB_ROOT = Path(__file__).parent / "web"
STATIC_ROOT = WEB_ROOT / "static"


def asset_versions() -> dict[str, str]:
    return {
        filename: str((STATIC_ROOT / filename).stat().st_mtime_ns)
        for filename in ("styles.css", "app.js")
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = build_container(resolved_settings)
        yield

    app = FastAPI(
        title=resolved_settings.app_name,
        description="Inspect BM25, dense, RRF, and reranked retrieval over BEIR FiQA.",
        lifespan=lifespan,
    )
    templates = Jinja2Templates(directory=WEB_ROOT / "templates")
    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    app.include_router(api_router)
    app.include_router(create_web_router(templates, asset_versions()))
    return app


app = create_app()
