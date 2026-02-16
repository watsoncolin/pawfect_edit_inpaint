import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.flux_inpaint import is_ready, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FLUX.1-Fill-dev model...")
    load_model()
    logger.info("FLUX.1-Fill-dev model loaded and ready")
    yield


app = FastAPI(title="Pawfect Edit FLUX Inpaint", lifespan=lifespan)


@app.get("/health")
async def health():
    """Liveness probe — always returns OK."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness probe — returns OK only when model is loaded."""
    if not is_ready():
        return {"status": "loading"}, 503
    return {"status": "ready"}


from app.routers.inpaint import router as inpaint_router  # noqa: E402

app.include_router(inpaint_router)
