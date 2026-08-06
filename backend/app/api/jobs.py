import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.dependencies import get_broadcaster, get_job_store, get_task_runner
from app.jobs.events import SSEBroadcaster
from app.jobs.runner import TaskRunner
from app.jobs.store import JobStore
from app.models.job import Job
from app.ai.token_logger import current_user_email
import os
ADMIN_EMAILS = {e.strip() for e in os.getenv("ADMIN_EMAILS", "mehul.20331814@ltimindtree.com").split(",") if e.strip()}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> Job:
    """Get job status and details."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.get("/{job_id}/stream")
async def stream_events(
    job_id: str,
    broadcaster: SSEBroadcaster = Depends(get_broadcaster),
):
    """SSE event stream for a job or pipeline run.

    Accepts any run_id — does not require the job to exist in the DB.
    Interactive pipeline routes use pre-generated run_ids that may not
    have a DB record yet when the SSE connection opens.
    """
    return StreamingResponse(
        broadcaster.event_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("")
async def list_jobs(
    job_store: JobStore = Depends(get_job_store),
) -> list[Job]:
    """List jobs: the caller's own, or all jobs for admins."""
    user_email = current_user_email.get()
    if user_email in ADMIN_EMAILS:
        return job_store.list_jobs()
    return job_store.list_jobs(user_email=user_email)


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
    task_runner: TaskRunner = Depends(get_task_runner),
) -> dict:
    """Cancel a running job."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    cancelled = task_runner.cancel(job_id)
    if cancelled:
        job_store.cancel_job(job_id)
        return {"status": "cancelled", "job_id": job_id}
    else:
        job_store.cancel_job(job_id)
        return {"status": "cancelled", "job_id": job_id, "note": "No running task found"}
