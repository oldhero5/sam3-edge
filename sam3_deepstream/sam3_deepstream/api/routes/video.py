"""Video processing endpoints."""

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ..models.requests import OutputFormat, VideoProcessRequest
from ..models.responses import (
    JobStatus,
    JobStatusResponse,
    VideoProcessingResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/video", tags=["video"])


@router.post("/process", response_model=VideoProcessingResponse)
async def process_video(
    request: Request,
    file: UploadFile = File(..., description="Video file to process"),
    text_prompt: Optional[str] = Form(None, description="Text prompt for segmentation"),
    keyframe_interval: int = Form(5, ge=1, le=30, description="Keyframe interval"),
    output_format: OutputFormat = Form(OutputFormat.VIDEO, description="Output format"),
    max_objects: int = Form(50, ge=1, le=100, description="Maximum objects to track"),
) -> VideoProcessingResponse:
    """
    Upload and process a video for segmentation.

    Returns a job ID that can be used to check status and retrieve results.
    """
    job_manager = request.app.state.job_manager

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    config = request.app.state.config
    upload_path = config.api.upload_dir / f"{job_id}_{file.filename}"

    try:
        content = await file.read()

        # Check file size
        max_size = config.api.max_upload_size_mb * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {config.api.max_upload_size_mb}MB",
            )

        with open(upload_path, "wb") as f:
            f.write(content)

    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Create processing job
    job_request = VideoProcessRequest(
        text_prompt=text_prompt,
        keyframe_interval=keyframe_interval,
        output_format=output_format,
        max_objects=max_objects,
    )

    await job_manager.create_job(
        job_id=job_id,
        video_path=upload_path,
        request=job_request,
    )

    return VideoProcessingResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Video queued for processing",
    )


@router.get("/job/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    request: Request,
    job_id: str,
) -> JobStatusResponse:
    """Get the status of a processing job."""
    job_manager = request.app.state.job_manager

    job = await job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        progress=job.progress,
        frames_processed=job.frames_processed,
        total_frames=job.total_frames,
        created_at=job.created_at,
        completed_at=job.completed_at,
        error=job.error,
    )


@router.get("/job/{job_id}/result")
async def get_job_result(
    request: Request,
    job_id: str,
):
    """
    Download the result of a completed job.

    Returns either a video file (MP4) or a JSON archive depending on output_format.
    """
    job_manager = request.app.state.job_manager

    job = await job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status}",
        )

    if job.output_path is None or not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    # Determine media type based on output format
    if job.output_format == OutputFormat.VIDEO:
        media_type = "video/mp4"
        filename = f"segmented_{job_id}.mp4"
    else:
        media_type = "application/json"
        filename = f"masks_{job_id}.json"

    return FileResponse(
        path=job.output_path,
        media_type=media_type,
        filename=filename,
    )


@router.delete("/job/{job_id}")
async def cancel_job(
    request: Request,
    job_id: str,
) -> dict:
    """Cancel a pending or processing job."""
    job_manager = request.app.state.job_manager

    success = await job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

    return {"message": "Job cancelled", "job_id": job_id}
