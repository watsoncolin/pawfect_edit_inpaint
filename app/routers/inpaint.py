import base64
import json
import logging

from fastapi import APIRouter, Request, Response

from app.services.flux_inpaint import is_ready
from app.services.pipeline import run_inpaint

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/inpaint")
async def handle_inpaint(request: Request):
    """Pub/Sub push handler for inpainting jobs."""
    if not is_ready():
        logger.warning("Model not ready yet, nacking for retry")
        return Response(status_code=503)

    try:
        envelope = await request.json()
        message = envelope.get("message", {})
        data_b64 = message.get("data", "")
        payload = json.loads(base64.b64decode(data_b64))

        user_id = payload["userId"]
        session_id = payload["sessionId"]
        logger.info(f"Received inpaint job: userId={user_id}, sessionId={session_id}")

        run_inpaint(user_id, session_id)

        return Response(status_code=200)
    except KeyError as e:
        logger.error(f"Missing required field in message: {e}")
        # Ack to prevent retries on bad messages
        return Response(status_code=200)
    except Exception as e:
        logger.exception(f"Transient error processing inpaint: {e}")
        # Nack so Pub/Sub retries
        return Response(status_code=500)
