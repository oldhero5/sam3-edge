"""Async job management for video processing."""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ...config import SAM3DeepStreamConfig, get_config
from ..models.requests import OutputFormat, VideoProcessRequest
from ..models.responses import JobStatus
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents a video processing job."""
    job_id: str
    video_path: Path
    request: VideoProcessRequest
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    frames_processed: int = 0
    total_frames: Optional[int] = None
    output_path: Optional[Path] = None
    output_format: OutputFormat = OutputFormat.VIDEO
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class JobManager:
    """
    Manages async video processing jobs.

    Uses an in-memory queue with a background worker thread.
    For production, consider using Celery or Redis Queue.
    """

    def __init__(
        self,
        config: Optional[SAM3DeepStreamConfig] = None,
        max_concurrent: int = 2,
    ):
        """
        Initialize job manager.

        Args:
            config: Configuration object
            max_concurrent: Maximum concurrent jobs
        """
        self.config = config or get_config()
        self.max_concurrent = max_concurrent

        self._jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list = []
        self._shutdown_event = threading.Event()
        self._processor: Optional[VideoProcessor] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        """Start job processing workers."""
        self._shutdown_event.clear()
        self._processor = VideoProcessor(self.config)

        # Start worker thread
        self._loop = asyncio.new_event_loop()
        worker_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
        )
        worker_thread.start()
        self._workers.append(worker_thread)

        logger.info(f"Job manager started with {self.max_concurrent} workers")

    def _run_event_loop(self) -> None:
        """Run async event loop in thread."""
        asyncio.set_event_loop(self._loop)

        for _ in range(self.max_concurrent):
            self._loop.create_task(self._worker())

        self._loop.run_forever()

    async def _worker(self) -> None:
        """Worker coroutine that processes jobs from queue."""
        while not self._shutdown_event.is_set():
            try:
                job_id = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            if job_id is None:
                break

            await self._process_job(job_id)

    async def _process_job(self, job_id: str) -> None:
        """Process a single job."""
        job = self._jobs.get(job_id)
        if job is None:
            return

        logger.info(f"Processing job: {job_id}")
        job.status = JobStatus.PROCESSING

        try:
            # Determine output path
            if job.request.output_format == OutputFormat.VIDEO:
                ext = ".mp4"
            else:
                ext = ".json"

            output_path = self.config.api.output_dir / f"{job_id}{ext}"

            # Progress callback
            def update_progress(progress: float):
                job.progress = min(progress, 1.0)
                job.frames_processed = int(progress * (job.total_frames or 100))

            # Process video
            result = await self._processor.process_video(
                job.video_path,
                job.request,
                output_path,
                progress_callback=update_progress,
            )

            job.output_path = Path(result["output_path"])
            job.frames_processed = result["frames_processed"]
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.completed_at = datetime.utcnow()

            logger.info(f"Job completed: {job_id}")

        except Exception as e:
            logger.error(f"Job failed: {job_id} - {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()

    def stop(self) -> None:
        """Stop job processing."""
        self._shutdown_event.set()

        # Put None to signal workers to exit
        for _ in range(self.max_concurrent):
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(None),
                    self._loop,
                )

        # Stop event loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for workers
        for worker in self._workers:
            worker.join(timeout=5.0)

        logger.info("Job manager stopped")

    async def create_job(
        self,
        job_id: str,
        video_path: Path,
        request: VideoProcessRequest,
    ) -> Job:
        """
        Create a new processing job.

        Args:
            job_id: Unique job identifier
            video_path: Path to video file
            request: Processing request

        Returns:
            Created job object
        """
        job = Job(
            job_id=job_id,
            video_path=video_path,
            request=request,
            output_format=request.output_format,
        )

        self._jobs[job_id] = job

        # Queue for processing
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._queue.put(job_id),
                self._loop,
            )
        else:
            await self._queue.put(job_id)

        logger.info(f"Job created: {job_id}")
        return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        job = self._jobs.get(job_id)
        if job is None:
            return False

        if job.status in (JobStatus.PENDING, JobStatus.PROCESSING):
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            return True

        return False

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job and its files."""
        job = self._jobs.get(job_id)
        if job is None:
            return False

        # Delete output file
        if job.output_path and job.output_path.exists():
            job.output_path.unlink()

        # Delete input file
        if job.video_path and job.video_path.exists():
            job.video_path.unlink()

        del self._jobs[job_id]
        return True

    def get_all_jobs(self) -> Dict[str, Job]:
        """Get all jobs."""
        return self._jobs.copy()
