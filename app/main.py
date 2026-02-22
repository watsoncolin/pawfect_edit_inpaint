import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.services.flux_inpaint import is_ready, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_model_background():
    try:
        load_model()
        logger.info("FLUX.1-Fill-dev model loaded and ready")
    except Exception:
        logger.exception("Failed to load FLUX.1-Fill-dev model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FLUX.1-Fill-dev model loading in background...")
    thread = threading.Thread(target=_load_model_background, daemon=True)
    thread.start()
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
        return JSONResponse(content={"status": "loading"}, status_code=503)
    return {"status": "ready"}


from app.routers.inpaint import router as inpaint_router  # noqa: E402

app.include_router(inpaint_router)

from app.routers.audit import router as audit_router  # noqa: E402

app.include_router(audit_router)
