"""Detection storage service using SQLite for metadata and FAISS for vector search."""

import asyncio
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ...config import get_config, DatabaseConfig

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
    mask_rle: Optional[str] = None
    timestamp_ms: Optional[float] = None
    detection_id: Optional[int] = None
    embedding_id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class SearchResult:
    """Search result with similarity score."""
    detection: Detection
    similarity: float


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
    embedding_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_detections_video ON detections(video_id);
CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label);
CREATE INDEX IF NOT EXISTS idx_detections_prompt ON detections(text_prompt);
CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(video_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_detections_embedding ON detections(embedding_id);
"""


class DetectionStore:
    """
    Manages detection storage with SQLite for metadata and FAISS for vector search.

    Thread-safe for concurrent access from async handlers.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        embedding_dim: int = 256,
        use_gpu: bool = True,
    ):
        config = get_config()
        self.db_path = db_path or config.database.db_path
        self.index_path = index_path or config.database.faiss_index_path
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and FAISS_AVAILABLE

        self._conn: Optional[sqlite3.Connection] = None
        self._index: Optional[Any] = None  # faiss.Index
        self._gpu_res: Optional[Any] = None  # faiss.StandardGpuResources
        self._lock = threading.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database and FAISS index."""
        await asyncio.get_event_loop().run_in_executor(None, self._init_sync)

    def _init_sync(self) -> None:
        """Synchronous initialization."""
        with self._lock:
            if self._initialized:
                return

            # Initialize SQLite
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
            self._conn.commit()
            logger.info(f"SQLite database initialized: {self.db_path}")

            # Initialize FAISS
            if FAISS_AVAILABLE:
                self._init_faiss()
            else:
                logger.warning("FAISS not available, semantic search disabled")

            self._initialized = True

    def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        if self.index_path.exists():
            # Load existing index
            self._index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
        else:
            # Create new index
            self._index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
            logger.info(f"Created new FAISS index (dim={self.embedding_dim})")

        # Move to GPU if available
        if self.use_gpu:
            try:
                self._gpu_res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move FAISS to GPU: {e}")

    async def register_device(self, device_id: str, hostname: str) -> None:
        """Register a device for federation tracking."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._register_device_sync, device_id, hostname
        )

    def _register_device_sync(self, device_id: str, hostname: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO devices (device_id, hostname) VALUES (?, ?)",
                (device_id, hostname),
            )
            self._conn.commit()

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

    async def store_detection(
        self,
        detection: Detection,
        embedding: Optional[np.ndarray] = None,
    ) -> int:
        """
        Store a detection result.

        Args:
            detection: Detection metadata
            embedding: Optional 256-dim embedding for semantic search

        Returns:
            detection_id of stored detection
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._store_detection_sync, detection, embedding
        )

    def _store_detection_sync(
        self,
        detection: Detection,
        embedding: Optional[np.ndarray] = None,
    ) -> int:
        with self._lock:
            embedding_id = None

            # Add embedding to FAISS if provided
            if embedding is not None and self._index is not None:
                embedding_id = self._index.ntotal
                # Normalize for inner product similarity
                embedding = embedding.astype(np.float32)
                faiss.normalize_L2(embedding.reshape(1, -1))
                self._index.add(embedding.reshape(1, -1))

            # Store in SQLite
            cursor = self._conn.execute(
                """INSERT INTO detections
                   (video_id, device_id, frame_idx, timestamp_ms, object_id,
                    text_prompt, label, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, mask_rle, embedding_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    embedding_id,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid

    async def store_detections_batch(
        self,
        detections: List[Detection],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Store multiple detections in a batch."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._store_detections_batch_sync, detections, embeddings
        )

    def _store_detections_batch_sync(
        self,
        detections: List[Detection],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[int]:
        with self._lock:
            detection_ids = []

            # Add embeddings to FAISS in batch
            embedding_ids = [None] * len(detections)
            if embeddings is not None and self._index is not None:
                start_id = self._index.ntotal
                embeddings = embeddings.astype(np.float32)
                faiss.normalize_L2(embeddings)
                self._index.add(embeddings)
                embedding_ids = list(range(start_id, start_id + len(detections)))

            # Store in SQLite
            for i, detection in enumerate(detections):
                cursor = self._conn.execute(
                    """INSERT INTO detections
                       (video_id, device_id, frame_idx, timestamp_ms, object_id,
                        text_prompt, label, confidence,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, mask_rle, embedding_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        embedding_ids[i],
                    ),
                )
                detection_ids.append(cursor.lastrowid)

            self._conn.commit()
            return detection_ids

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
        min_confidence: float = 0.0,
        video_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Semantic search using FAISS.

        Args:
            query_embedding: 256-dim query embedding
            k: Maximum results to return
            min_confidence: Minimum detection confidence
            video_id: Optional filter by video

        Returns:
            List of SearchResult with similarity scores
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._search_sync, query_embedding, k, min_confidence, video_id
        )

    def _search_sync(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
        min_confidence: float = 0.0,
        video_id: Optional[str] = None,
    ) -> List[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        with self._lock:
            # Normalize query
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            # Search FAISS (get more results to filter later)
            search_k = min(k * 5, self._index.ntotal)
            distances, indices = self._index.search(query_embedding, search_k)

            # Fetch matching detections from SQLite
            results = []
            for similarity, embedding_id in zip(distances[0], indices[0]):
                if embedding_id < 0:  # Invalid index
                    continue

                # Build query with filters
                query = "SELECT * FROM detections WHERE embedding_id = ?"
                params = [int(embedding_id)]

                if min_confidence > 0:
                    query += " AND confidence >= ?"
                    params.append(min_confidence)

                if video_id:
                    query += " AND video_id = ?"
                    params.append(video_id)

                row = self._conn.execute(query, params).fetchone()
                if row:
                    detection = self._row_to_detection(row)
                    results.append(SearchResult(detection=detection, similarity=float(similarity)))

                if len(results) >= k:
                    break

            return results

    async def get_detection(self, detection_id: int) -> Optional[Detection]:
        """Get a single detection by ID."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_detection_sync, detection_id
        )

    def _get_detection_sync(self, detection_id: int) -> Optional[Detection]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM detections WHERE detection_id = ?",
                (detection_id,),
            ).fetchone()
            return self._row_to_detection(row) if row else None

    async def list_detections(
        self,
        video_id: Optional[str] = None,
        label: Optional[str] = None,
        text_prompt: Optional[str] = None,
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
            text_prompt,
            min_confidence,
            limit,
            offset,
        )

    def _list_detections_sync(
        self,
        video_id: Optional[str],
        label: Optional[str],
        text_prompt: Optional[str],
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

            if text_prompt:
                query += " AND text_prompt = ?"
                params.append(text_prompt)

            if min_confidence > 0:
                query += " AND confidence >= ?"
                params.append(min_confidence)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = self._conn.execute(query, params).fetchall()
            return [self._row_to_detection(row) for row in rows]

    async def aggregate_objects(
        self,
        group_by: str = "label",
        video_id: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Aggregate detections by label or other field."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._aggregate_objects_sync, group_by, video_id, min_confidence
        )

    def _aggregate_objects_sync(
        self,
        group_by: str,
        video_id: Optional[str],
        min_confidence: float,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            # Validate group_by to prevent SQL injection
            if group_by not in ("label", "text_prompt", "video_id"):
                group_by = "label"

            query = f"""
                SELECT {group_by},
                       COUNT(*) as count,
                       AVG(confidence) as avg_confidence,
                       MIN(created_at) as first_seen,
                       MAX(created_at) as last_seen
                FROM detections
                WHERE 1=1
            """
            params = []

            if video_id:
                query += " AND video_id = ?"
                params.append(video_id)

            if min_confidence > 0:
                query += " AND confidence >= ?"
                params.append(min_confidence)

            query += f" GROUP BY {group_by} ORDER BY count DESC"

            rows = self._conn.execute(query, params).fetchall()
            return [
                {
                    group_by: row[0],
                    "count": row[1],
                    "avg_confidence": row[2],
                    "first_seen": row[3],
                    "last_seen": row[4],
                }
                for row in rows
            ]

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return await asyncio.get_event_loop().run_in_executor(None, self._get_stats_sync)

    def _get_stats_sync(self) -> Dict[str, Any]:
        with self._lock:
            video_count = self._conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            detection_count = self._conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            device_count = self._conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
            index_size = self._index.ntotal if self._index else 0

            return {
                "videos": video_count,
                "detections": detection_count,
                "devices": device_count,
                "embeddings": index_size,
                "db_path": str(self.db_path),
                "index_path": str(self.index_path),
            }

    async def delete_video_detections(self, video_id: str) -> int:
        """Delete all detections for a video."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._delete_video_detections_sync, video_id
        )

    def _delete_video_detections_sync(self, video_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM detections WHERE video_id = ?",
                (video_id,),
            )
            self._conn.commit()
            # Note: FAISS doesn't support deletion, embeddings remain but are orphaned
            return cursor.rowcount

    def save_index(self) -> None:
        """Persist FAISS index to disk."""
        if self._index is not None:
            with self._lock:
                # Move back to CPU for saving if on GPU
                if self._gpu_res is not None:
                    cpu_index = faiss.index_gpu_to_cpu(self._index)
                    faiss.write_index(cpu_index, str(self.index_path))
                else:
                    faiss.write_index(self._index, str(self.index_path))
                logger.info(f"Saved FAISS index to {self.index_path}")

    def close(self) -> None:
        """Close database and save index."""
        with self._lock:
            if self._index is not None:
                self.save_index()
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
            embedding_id=row["embedding_id"],
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


def set_detection_store(store: DetectionStore) -> None:
    """Set global detection store."""
    global _detection_store
    _detection_store = store
