from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from papersearch import __version__
from papersearch.api import router as api_router
from papersearch.config import get_settings
from papersearch.db import close_pool, init_pool

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_STATIC = PACKAGE_DIR.parent.parent / "static"


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        init_pool(settings)
        logger.info("PaperSearch ready — %s", settings.embedding_model)
        yield
        close_pool()

    app = FastAPI(
        title="PaperSearch",
        version=__version__,
        lifespan=lifespan,
        description="Semantic search over academic abstracts using Oracle AI Vector Search.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/v1")

    if PROJECT_STATIC.is_dir():
        app.mount("/ui-assets", StaticFiles(directory=str(PROJECT_STATIC)), name="ui-assets")

        @app.get("/")
        def spa_index():
            return FileResponse(PROJECT_STATIC / "index.html")

    return app
