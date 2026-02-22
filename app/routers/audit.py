import logging
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.audit_runner import run_audit

logger = logging.getLogger(__name__)

router = APIRouter()


class AuditRequest(BaseModel):
    user_id: str
    session_id: str
    run_id: str | None = None  # auto-generated if not provided


@router.post("/audit")
def handle_audit(req: AuditRequest):
    """
    Trigger a full baseline audit run.
    Synchronous — runs the full matrix (3 quants × 5 guidance = 15 inferences + compile check).
    Estimated duration: 25-60 minutes on L4.
    Set client timeout accordingly (e.g., curl --max-time 7200).
    """
    run_id = req.run_id or f"audit-{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting audit run: run_id={run_id}, session={req.session_id}")

    try:
        result = run_audit(req.user_id, req.session_id, run_id)
        logger.info(f"Audit complete: run_id={run_id}, report={result['report_path']}")
        return {
            "status": "complete",
            "run_id": run_id,
            "report_url": result["report_url"],
            "report_path": result["report_path"],
            "gpu": result["gpu"],
            "sm": result["sm"],
            "compile_viable": result["compile_viable"],
            "result_count": len(result["results"]),
        }
    except Exception as e:
        logger.exception(f"Audit failed: run_id={run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
