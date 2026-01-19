"""Natural Language Query API routes for SAM3-Edge."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..services.detection_store import (
    Detection,
    DetectionStore,
    SearchResult,
    VideoMetadata,
    get_detection_store,
)
from ..services.embedding_service import EmbeddingService, get_embedding_service
from ...config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["nlq"])


# Request/Response Models

class DetectRequest(BaseModel):
    """Request for text-based detection."""
    text_prompt: str = Field(..., description="Text description of object to detect")
    video_id: Optional[str] = Field(None, description="Existing video ID to process")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    keyframe_interval: int = Field(5, ge=1, le=30)
    max_objects: int = Field(50, ge=1, le=200)
    store_masks: bool = Field(True, description="Store RLE masks in database")
    store_embeddings: bool = Field(True, description="Store embeddings for semantic search")


class DetectResponse(BaseModel):
    """Response for detection job submission."""
    job_id: str
    status: str
    message: str


class SearchRequest(BaseModel):
    """Request for semantic search."""
    query: str = Field(..., description="Natural language query")
    limit: int = Field(100, ge=1, le=1000)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)
    video_id: Optional[str] = None


class SearchResultItem(BaseModel):
    """Single search result."""
    detection_id: int
    video_id: str
    frame_idx: int
    text_prompt: str
    label: Optional[str]
    confidence: float
    similarity: float
    bbox: List[float]


class SearchResponse(BaseModel):
    """Response for semantic search."""
    query: str
    total_results: int
    results: List[SearchResultItem]


class ObjectAggregation(BaseModel):
    """Aggregated object statistics."""
    label: Optional[str] = None
    text_prompt: Optional[str] = None
    video_id: Optional[str] = None
    count: int
    avg_confidence: float
    first_seen: Optional[str]
    last_seen: Optional[str]


class ObjectsResponse(BaseModel):
    """Response for object aggregation."""
    total: int
    objects: List[ObjectAggregation]


class DetectionDetail(BaseModel):
    """Detailed detection information."""
    detection_id: int
    video_id: str
    device_id: Optional[str]
    frame_idx: int
    timestamp_ms: Optional[float]
    object_id: Optional[int]
    text_prompt: str
    label: Optional[str]
    confidence: float
    bbox: Dict[str, float]
    mask_rle: Optional[str]
    created_at: Optional[str]


class StoreStatsResponse(BaseModel):
    """Storage statistics."""
    videos: int
    detections: int
    devices: int
    embeddings: int
    db_path: str
    index_path: str


# Endpoints

@router.post("/detect", response_model=DetectResponse)
async def detect(
    request: Request,
    text_prompt: str = Form(...),
    confidence_threshold: float = Form(0.5),
    keyframe_interval: int = Form(5),
    max_objects: int = Form(50),
    store_masks: bool = Form(True),
    store_embeddings: bool = Form(True),
    file: Optional[UploadFile] = File(None),
    video_id: Optional[str] = Form(None),
):
    """
    Submit a text prompt for object detection.

    Either upload a video file or provide an existing video_id.
    The detection job runs asynchronously - use /job/{job_id}/status to check progress.
    """
    # Get job manager from app state
    job_manager = getattr(request.app.state, "job_manager", None)
    if job_manager is None:
        raise HTTPException(status_code=503, detail="Job manager not available")

    config = get_config()

    # Handle video file upload
    if file is not None:
        # Save uploaded file
        job_id = str(uuid.uuid4())
        filename = f"{job_id}_{file.filename}"
        upload_path = config.api.upload_dir / filename

        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        video_path = str(upload_path)
    elif video_id:
        # Use existing video
        video_path = video_id
        job_id = str(uuid.uuid4())
    else:
        raise HTTPException(
            status_code=400,
            detail="Either upload a video file or provide video_id"
        )

    # Create processing request
    from ..models.requests import VideoProcessRequest
    process_request = VideoProcessRequest(
        text_prompt=text_prompt,
        keyframe_interval=keyframe_interval,
        max_objects=max_objects,
        segmentation_threshold=confidence_threshold,
    )

    # Queue job with text prompt metadata
    job = await job_manager.create_job(
        video_path=video_path,
        request=process_request,
        job_id=job_id,
        metadata={
            "text_prompt": text_prompt,
            "store_masks": store_masks,
            "store_embeddings": store_embeddings,
        }
    )

    return DetectResponse(
        job_id=job.job_id,
        status=job.status.value,
        message=f"Detection job queued for: {text_prompt}"
    )


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Natural language search query"),
    limit: int = Query(100, ge=1, le=1000),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    video_id: Optional[str] = Query(None),
):
    """
    Search detections using natural language.

    Uses semantic similarity to find detections matching the query,
    even if the exact text prompt differs.
    """
    store = await get_detection_store()
    embedding_service = get_embedding_service()

    # Generate query embedding
    query_embedding = embedding_service.encode(q)

    # Search FAISS
    results = await store.search(
        query_embedding=query_embedding,
        k=limit,
        min_confidence=min_confidence,
        video_id=video_id,
    )

    # Format response
    items = [
        SearchResultItem(
            detection_id=r.detection.detection_id,
            video_id=r.detection.video_id,
            frame_idx=r.detection.frame_idx,
            text_prompt=r.detection.text_prompt,
            label=r.detection.label,
            confidence=r.detection.confidence,
            similarity=r.similarity,
            bbox=list(r.detection.bbox),
        )
        for r in results
    ]

    return SearchResponse(
        query=q,
        total_results=len(items),
        results=items,
    )


@router.get("/objects", response_model=ObjectsResponse)
async def list_objects(
    group_by: str = Query("label", regex="^(label|text_prompt|video_id)$"),
    video_id: Optional[str] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
):
    """
    List detected objects with aggregation.

    Group by label, text_prompt, or video_id to see detection statistics.
    """
    store = await get_detection_store()

    aggregations = await store.aggregate_objects(
        group_by=group_by,
        video_id=video_id,
        min_confidence=min_confidence,
    )

    objects = []
    for agg in aggregations:
        obj = ObjectAggregation(
            count=agg["count"],
            avg_confidence=agg["avg_confidence"],
            first_seen=agg.get("first_seen"),
            last_seen=agg.get("last_seen"),
        )
        # Set the grouped field
        if group_by == "label":
            obj.label = agg.get("label")
        elif group_by == "text_prompt":
            obj.text_prompt = agg.get("text_prompt")
        elif group_by == "video_id":
            obj.video_id = agg.get("video_id")
        objects.append(obj)

    return ObjectsResponse(
        total=len(objects),
        objects=objects,
    )


@router.get("/detections/{detection_id}", response_model=DetectionDetail)
async def get_detection(detection_id: int):
    """Get detailed information about a specific detection."""
    store = await get_detection_store()

    detection = await store.get_detection(detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="Detection not found")

    return DetectionDetail(
        detection_id=detection.detection_id,
        video_id=detection.video_id,
        device_id=detection.device_id,
        frame_idx=detection.frame_idx,
        timestamp_ms=detection.timestamp_ms,
        object_id=detection.object_id,
        text_prompt=detection.text_prompt,
        label=detection.label,
        confidence=detection.confidence,
        bbox={
            "x1": detection.bbox[0],
            "y1": detection.bbox[1],
            "x2": detection.bbox[2],
            "y2": detection.bbox[3],
        },
        mask_rle=detection.mask_rle,
        created_at=str(detection.created_at) if detection.created_at else None,
    )


@router.get("/detections", response_model=List[DetectionDetail])
async def list_detections(
    video_id: Optional[str] = Query(None),
    label: Optional[str] = Query(None),
    text_prompt: Optional[str] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List detections with filters."""
    store = await get_detection_store()

    detections = await store.list_detections(
        video_id=video_id,
        label=label,
        text_prompt=text_prompt,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset,
    )

    return [
        DetectionDetail(
            detection_id=d.detection_id,
            video_id=d.video_id,
            device_id=d.device_id,
            frame_idx=d.frame_idx,
            timestamp_ms=d.timestamp_ms,
            object_id=d.object_id,
            text_prompt=d.text_prompt,
            label=d.label,
            confidence=d.confidence,
            bbox={
                "x1": d.bbox[0],
                "y1": d.bbox[1],
                "x2": d.bbox[2],
                "y2": d.bbox[3],
            },
            mask_rle=d.mask_rle,
            created_at=str(d.created_at) if d.created_at else None,
        )
        for d in detections
    ]


@router.delete("/detections")
async def delete_detections(
    video_id: str = Query(..., description="Video ID to delete detections for"),
):
    """Delete all detections for a video."""
    store = await get_detection_store()

    deleted_count = await store.delete_video_detections(video_id)

    return {"deleted_count": deleted_count, "video_id": video_id}


@router.get("/store/stats", response_model=StoreStatsResponse)
async def get_store_stats():
    """Get storage statistics."""
    store = await get_detection_store()
    stats = await store.get_stats()
    return StoreStatsResponse(**stats)


@router.get("/export")
async def export_detections(
    video_id: Optional[str] = Query(None),
    since: Optional[str] = Query(None, description="ISO timestamp to export from"),
    format: str = Query("json", regex="^(json|jsonl)$"),
):
    """
    Export detections for federation or backup.

    Returns detection data in JSON or JSONL format.
    """
    store = await get_detection_store()

    # Get detections
    detections = await store.list_detections(
        video_id=video_id,
        limit=10000,  # Export limit
    )

    if format == "jsonl":
        # Stream JSONL
        import json

        async def generate():
            for d in detections:
                yield json.dumps({
                    "detection_id": d.detection_id,
                    "video_id": d.video_id,
                    "device_id": d.device_id,
                    "frame_idx": d.frame_idx,
                    "text_prompt": d.text_prompt,
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": list(d.bbox),
                    "created_at": str(d.created_at) if d.created_at else None,
                }) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=detections.jsonl"},
        )
    else:
        # JSON response
        return {
            "total": len(detections),
            "detections": [
                {
                    "detection_id": d.detection_id,
                    "video_id": d.video_id,
                    "device_id": d.device_id,
                    "frame_idx": d.frame_idx,
                    "text_prompt": d.text_prompt,
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": list(d.bbox),
                    "created_at": str(d.created_at) if d.created_at else None,
                }
                for d in detections
            ],
        }
