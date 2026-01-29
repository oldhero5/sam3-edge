"""Detection storage service using SQLite."""

import asyncio
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...config import get_config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Detection result from a video frame."""
    video_id: str
    device_id: str
    frame_idx: int
    object_id: int
    text_prompt: str
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    mask_rle: Optional[str] = None  # JSON-serialized RLE dict
    timestamp_ms: Optional[float] = None
    detection_id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class VideoMetadata:
    """Video source metadata."""
    video_id: str
    device_id: str
    filename: str
    file_hash: Optional[str] = None
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    total_frames: Optional[int] = None
    created_at: Optional[datetime] = None


# SQL schema
SCHEMA = """
-- Device tracking for federation
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    hostname TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Video source metadata
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    device_id TEXT,
    filename TEXT NOT NULL,
    file_hash TEXT,
    duration_seconds REAL,
    width INTEGER,
    height INTEGER,
    total_frames INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detection results
CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    device_id TEXT,
    frame_idx INTEGER NOT NULL,
    timestamp_ms REAL,
    object_id INTEGER,
    text_prompt TEXT,
    label TEXT,
    confidence REAL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    mask_rle TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_detections_video ON detections(video_id);
CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label);
CREATE INDEX IF NOT EXISTS idx_detections_prompt ON detections(text_prompt);
CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(video_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence DESC);
"""


class DetectionStore:
    """
    Manages detection storage with SQLite.

    Thread-safe for concurrent access from async handlers.
    """

    def __init__(self, db_path: Optional[Path] = None):
        config = get_config()
        self.db_path = db_path or config.database.db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database."""
        await asyncio.get_event_loop().run_in_executor(None, self._init_sync)

    def _init_sync(self) -> None:
        """Synchronous initialization."""
        with self._lock:
            if self._initialized:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
            self._conn.commit()
            logger.info(f"SQLite database initialized: {self.db_path}")
            self._initialized = True

    async def store_video(self, video: VideoMetadata) -> None:
        """Store video metadata."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._store_video_sync, video
        )

    def _store_video_sync(self, video: VideoMetadata) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO videos
                   (video_id, device_id, filename, file_hash, duration_seconds,
                    width, height, total_frames)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    video.video_id,
                    video.device_id,
                    video.filename,
                    video.file_hash,
                    video.duration_seconds,
                    video.width,
                    video.height,
                    video.total_frames,
                ),
            )
            self._conn.commit()

    async def store_detections_batch(
        self,
        detections: List[Detection],
        embeddings=None,  # Ignored, kept for API compatibility
    ) -> List[int]:
        """Store multiple detections in a batch."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._store_detections_batch_sync, detections
        )

    def _store_detections_batch_sync(self, detections: List[Detection]) -> List[int]:
        with self._lock:
            detection_ids = []
            for detection in detections:
                cursor = self._conn.execute(
                    """INSERT INTO detections
                       (video_id, device_id, frame_idx, timestamp_ms, object_id,
                        text_prompt, label, confidence,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, mask_rle)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        detection.video_id,
                        detection.device_id,
                        detection.frame_idx,
                        detection.timestamp_ms,
                        detection.object_id,
                        detection.text_prompt,
                        detection.label,
                        detection.confidence,
                        detection.bbox[0],
                        detection.bbox[1],
                        detection.bbox[2],
                        detection.bbox[3],
                        detection.mask_rle,
                    ),
                )
                detection_ids.append(cursor.lastrowid)
            self._conn.commit()
            return detection_ids

    async def list_detections(
        self,
        video_id: Optional[str] = None,
        label: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Detection]:
        """List detections with filters."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._list_detections_sync,
            video_id,
            label,
            min_confidence,
            limit,
            offset,
        )

    def _list_detections_sync(
        self,
        video_id: Optional[str],
        label: Optional[str],
        min_confidence: float,
        limit: int,
        offset: int,
    ) -> List[Detection]:
        with self._lock:
            query = "SELECT * FROM detections WHERE 1=1"
            params = []

            if video_id:
                query += " AND video_id = ?"
                params.append(video_id)

            if label:
                query += " AND label = ?"
                params.append(label)

            if min_confidence > 0:
                query += " AND confidence >= ?"
                params.append(min_confidence)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = self._conn.execute(query, params).fetchall()
            return [self._row_to_detection(row) for row in rows]

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return await asyncio.get_event_loop().run_in_executor(None, self._get_stats_sync)

    def _get_stats_sync(self) -> Dict[str, Any]:
        with self._lock:
            video_count = self._conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            detection_count = self._conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

            return {
                "videos": video_count,
                "detections": detection_count,
                "db_path": str(self.db_path),
            }

    def close(self) -> None:
        """Close database."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._initialized = False

    def _row_to_detection(self, row: sqlite3.Row) -> Detection:
        """Convert SQLite row to Detection object."""
        return Detection(
            detection_id=row["detection_id"],
            video_id=row["video_id"],
            device_id=row["device_id"],
            frame_idx=row["frame_idx"],
            timestamp_ms=row["timestamp_ms"],
            object_id=row["object_id"],
            text_prompt=row["text_prompt"],
            label=row["label"],
            confidence=row["confidence"],
            bbox=(row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]),
            mask_rle=row["mask_rle"],
            created_at=row["created_at"],
        )


# Global store instance
_detection_store: Optional[DetectionStore] = None


async def get_detection_store() -> DetectionStore:
    """Get or create global detection store."""
    global _detection_store
    if _detection_store is None:
        _detection_store = DetectionStore()
        await _detection_store.initialize()
    return _detection_store
