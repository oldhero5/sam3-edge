"""Keyframe-based inference processor for video segmentation."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""
    object_id: int
    mask: np.ndarray
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    last_keyframe: int = 0


@dataclass
class FrameResult:
    """Results for a single frame."""
    frame_idx: int
    is_keyframe: bool
    masks: List[np.ndarray]
    boxes: List[Tuple[float, float, float, float]]
    scores: List[float]
    object_ids: List[int]


class KeyframeProcessor:
    """
    Keyframe-based segmentation processor.

    Runs full SAM3 inference on keyframes and propagates masks
    to intermediate frames using tracking.

    Architecture:
    - Keyframe (every N frames): Full encoder + decoder inference
    - Intermediate frames: Tracker-based mask propagation
    """

    def __init__(
        self,
        encoder_fn,
        decoder_fn,
        keyframe_interval: int = 5,
        max_objects: int = 50,
    ):
        """
        Initialize keyframe processor.

        Args:
            encoder_fn: Function to encode image -> embeddings
            decoder_fn: Function to decode embeddings + prompts -> masks
            keyframe_interval: Run full inference every N frames
            max_objects: Maximum tracked objects
        """
        self.encoder_fn = encoder_fn
        self.decoder_fn = decoder_fn
        self.keyframe_interval = keyframe_interval
        self.max_objects = max_objects

        # Tracking state
        self._tracked_objects: Dict[int, TrackedObject] = {}
        self._next_object_id = 1
        self._frame_idx = 0

        # Cached embeddings for current keyframe
        self._keyframe_embeddings: Optional[Tensor] = None

        # Mask propagator (set externally)
        self._propagator = None

    def set_propagator(self, propagator) -> None:
        """Set mask propagation module."""
        self._propagator = propagator

    def process_frame(
        self,
        frame: np.ndarray,
        prompts: Optional[List[dict]] = None,
    ) -> FrameResult:
        """
        Process a single frame.

        Args:
            frame: Input frame (H, W, 3) RGB
            prompts: Optional prompts for keyframe inference

        Returns:
            FrameResult with masks and tracking info
        """
        is_keyframe = (self._frame_idx % self.keyframe_interval == 0)

        if is_keyframe:
            result = self._process_keyframe(frame, prompts)
        else:
            result = self._process_intermediate_frame(frame)

        self._frame_idx += 1
        return result

    def _process_keyframe(
        self,
        frame: np.ndarray,
        prompts: Optional[List[dict]] = None,
    ) -> FrameResult:
        """Process a keyframe with full inference."""
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()

        # Encode image
        embeddings = self.encoder_fn(frame_tensor)
        self._keyframe_embeddings = embeddings

        # Run decoder with prompts
        if prompts and self.decoder_fn:
            masks, scores = self._run_decoder(embeddings, prompts)
        else:
            # Use existing tracked objects as prompts
            masks, scores = self._recompute_tracked_masks(embeddings)

        # Update tracking state
        self._update_tracking(masks, scores)

        return FrameResult(
            frame_idx=self._frame_idx,
            is_keyframe=True,
            masks=[obj.mask for obj in self._tracked_objects.values()],
            boxes=[obj.box for obj in self._tracked_objects.values()],
            scores=[obj.score for obj in self._tracked_objects.values()],
            object_ids=list(self._tracked_objects.keys()),
        )

    def _process_intermediate_frame(
        self,
        frame: np.ndarray,
    ) -> FrameResult:
        """Process intermediate frame with mask propagation."""
        if self._propagator is None:
            # No propagation - return last known masks
            return FrameResult(
                frame_idx=self._frame_idx,
                is_keyframe=False,
                masks=[obj.mask for obj in self._tracked_objects.values()],
                boxes=[obj.box for obj in self._tracked_objects.values()],
                scores=[obj.score for obj in self._tracked_objects.values()],
                object_ids=list(self._tracked_objects.keys()),
            )

        # Propagate masks from previous frame
        propagated = self._propagator.propagate(frame, self._tracked_objects)

        # Update tracked objects with propagated masks
        for obj_id, (mask, box) in propagated.items():
            if obj_id in self._tracked_objects:
                self._tracked_objects[obj_id].mask = mask
                self._tracked_objects[obj_id].box = box

        return FrameResult(
            frame_idx=self._frame_idx,
            is_keyframe=False,
            masks=[obj.mask for obj in self._tracked_objects.values()],
            boxes=[obj.box for obj in self._tracked_objects.values()],
            scores=[obj.score for obj in self._tracked_objects.values()],
            object_ids=list(self._tracked_objects.keys()),
        )

    def _run_decoder(
        self,
        embeddings: Tensor,
        prompts: List[dict],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Run decoder with prompts."""
        if self.decoder_fn is None:
            return [], []

        # Convert prompts to tensors
        # This is a simplified version - full implementation would
        # handle point, box, and text prompts
        masks = []
        scores = []

        for prompt in prompts:
            mask, score = self.decoder_fn(embeddings, prompt)
            masks.append(mask.cpu().numpy())
            scores.append(score.item())

        return masks, scores

    def _recompute_tracked_masks(
        self,
        embeddings: Tensor,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Recompute masks for tracked objects using their boxes."""
        masks = []
        scores = []

        for obj in self._tracked_objects.values():
            # Use object's box as prompt
            prompt = {"box": obj.box}
            if self.decoder_fn:
                mask, score = self.decoder_fn(embeddings, prompt)
                masks.append(mask.cpu().numpy())
                scores.append(score.item())

        return masks, scores

    def _update_tracking(
        self,
        masks: List[np.ndarray],
        scores: List[float],
    ) -> None:
        """Update tracking state with new masks."""
        # Simple update - match by IoU
        # Full implementation would use Hungarian matching

        if not masks:
            return

        for i, (mask, score) in enumerate(zip(masks, scores)):
            if i < len(self._tracked_objects):
                # Update existing object
                obj_id = list(self._tracked_objects.keys())[i]
                self._tracked_objects[obj_id].mask = mask
                self._tracked_objects[obj_id].score = score
                self._tracked_objects[obj_id].box = self._mask_to_box(mask)
                self._tracked_objects[obj_id].last_keyframe = self._frame_idx
            else:
                # Add new object
                if len(self._tracked_objects) < self.max_objects:
                    obj = TrackedObject(
                        object_id=self._next_object_id,
                        mask=mask,
                        box=self._mask_to_box(mask),
                        score=score,
                        last_keyframe=self._frame_idx,
                    )
                    self._tracked_objects[self._next_object_id] = obj
                    self._next_object_id += 1

    def _mask_to_box(self, mask: np.ndarray) -> Tuple[float, float, float, float]:
        """Extract bounding box from mask."""
        if mask.sum() == 0:
            return (0.0, 0.0, 0.0, 0.0)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        h, w = mask.shape
        return (x1 / w, y1 / h, x2 / w, y2 / h)

    def add_object(
        self,
        mask: np.ndarray,
        score: float = 1.0,
    ) -> int:
        """Add a new tracked object."""
        obj = TrackedObject(
            object_id=self._next_object_id,
            mask=mask,
            box=self._mask_to_box(mask),
            score=score,
            last_keyframe=self._frame_idx,
        )
        self._tracked_objects[self._next_object_id] = obj
        self._next_object_id += 1
        return obj.object_id

    def remove_object(self, object_id: int) -> bool:
        """Remove a tracked object."""
        if object_id in self._tracked_objects:
            del self._tracked_objects[object_id]
            return True
        return False

    def reset(self) -> None:
        """Reset processor state."""
        self._tracked_objects.clear()
        self._next_object_id = 1
        self._frame_idx = 0
        self._keyframe_embeddings = None

    @property
    def tracked_objects(self) -> Dict[int, TrackedObject]:
        """Get current tracked objects."""
        return self._tracked_objects

    @property
    def frame_count(self) -> int:
        """Get processed frame count."""
        return self._frame_idx
