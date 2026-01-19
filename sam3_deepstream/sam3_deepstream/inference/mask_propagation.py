"""Mask propagation for intermediate frames using optical flow."""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MaskPropagator:
    """
    Propagates masks between keyframes using optical flow.

    Uses OpenCV's Farneback optical flow algorithm for CPU inference,
    with optional VPI acceleration on Jetson platforms.
    """

    def __init__(
        self,
        use_vpi: bool = False,
        flow_params: Optional[dict] = None,
    ):
        """
        Initialize mask propagator.

        Args:
            use_vpi: Use NVIDIA VPI for hardware-accelerated optical flow
            flow_params: Parameters for Farneback optical flow
        """
        self.use_vpi = use_vpi
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None

        # Default Farneback parameters
        self.flow_params = flow_params or {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0,
        }

        # Try to load VPI if requested
        self._vpi = None
        if use_vpi:
            self._init_vpi()

    def _init_vpi(self) -> None:
        """Initialize NVIDIA VPI for Jetson."""
        try:
            import vpi
            self._vpi = vpi
            logger.info("VPI initialized for hardware-accelerated optical flow")
        except ImportError:
            logger.warning("VPI not available, falling back to CPU optical flow")
            self.use_vpi = False

    def propagate(
        self,
        current_frame: np.ndarray,
        tracked_objects: Dict[int, "TrackedObject"],
    ) -> Dict[int, Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        """
        Propagate masks from previous frame to current frame.

        Args:
            current_frame: Current frame (H, W, 3) RGB
            tracked_objects: Dictionary of tracked objects with masks

        Returns:
            Dictionary mapping object_id to (propagated_mask, new_box)
        """
        # Convert to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

        if self._prev_gray is None:
            self._prev_frame = current_frame.copy()
            self._prev_gray = current_gray.copy()
            # Return original masks on first frame
            return {
                obj_id: (obj.mask, obj.box)
                for obj_id, obj in tracked_objects.items()
            }

        # Compute optical flow
        if self.use_vpi and self._vpi is not None:
            flow = self._compute_flow_vpi(self._prev_gray, current_gray)
        else:
            flow = self._compute_flow_cpu(self._prev_gray, current_gray)

        # Propagate each mask using flow
        results = {}
        for obj_id, obj in tracked_objects.items():
            propagated_mask = self._warp_mask(obj.mask, flow)
            new_box = self._mask_to_box(propagated_mask)
            results[obj_id] = (propagated_mask, new_box)

        # Update previous frame
        self._prev_frame = current_frame.copy()
        self._prev_gray = current_gray.copy()

        return results

    def _compute_flow_cpu(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> np.ndarray:
        """Compute optical flow using CPU (Farneback)."""
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            **self.flow_params,
        )
        return flow

    def _compute_flow_vpi(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> np.ndarray:
        """Compute optical flow using VPI (Jetson PVA)."""
        if self._vpi is None:
            return self._compute_flow_cpu(prev_gray, curr_gray)

        vpi = self._vpi

        # Create VPI images
        prev_vpi = vpi.asimage(prev_gray, format=vpi.Format.U8)
        curr_vpi = vpi.asimage(curr_gray, format=vpi.Format.U8)

        # Create output flow image
        flow_vpi = vpi.Image(prev_gray.shape, vpi.Format.F32)

        # Run optical flow on PVA
        with vpi.Backend.PVA:
            flow_vpi = vpi.optflow_dense(prev_vpi, curr_vpi)

        # Convert back to numpy
        flow = flow_vpi.cpu()
        return flow

    def _warp_mask(
        self,
        mask: np.ndarray,
        flow: np.ndarray,
    ) -> np.ndarray:
        """Warp mask using optical flow field."""
        h, w = mask.shape[:2]

        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack([x, y], axis=-1).astype(np.float32)

        # Add flow to get new coordinates
        new_coords = coords + flow

        # Remap mask
        warped = cv2.remap(
            mask.astype(np.float32),
            new_coords[..., 0],
            new_coords[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Threshold to binary
        return (warped > 0.5).astype(np.uint8)

    def _mask_to_box(
        self,
        mask: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Extract normalized bounding box from mask."""
        if mask.sum() == 0:
            return (0.0, 0.0, 0.0, 0.0)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        if len(row_indices) == 0 or len(col_indices) == 0:
            return (0.0, 0.0, 0.0, 0.0)

        y1, y2 = row_indices[[0, -1]]
        x1, x2 = col_indices[[0, -1]]

        h, w = mask.shape
        return (x1 / w, y1 / h, x2 / w, y2 / h)

    def reset(self) -> None:
        """Reset propagator state."""
        self._prev_frame = None
        self._prev_gray = None


class NvDCFMaskPropagator:
    """
    Mask propagation using NvDCF tracker output.

    Uses NvDCF bounding box predictions combined with
    mask warping for accurate propagation.
    """

    def __init__(
        self,
        mask_propagator: Optional[MaskPropagator] = None,
    ):
        """
        Initialize NvDCF-based propagator.

        Args:
            mask_propagator: Optical flow propagator for mask warping
        """
        self.mask_propagator = mask_propagator or MaskPropagator()

        # NvDCF tracker state (populated from DeepStream)
        self._tracker_boxes: Dict[int, Tuple[float, float, float, float]] = {}

    def update_tracker_boxes(
        self,
        boxes: Dict[int, Tuple[float, float, float, float]],
    ) -> None:
        """
        Update tracked bounding boxes from NvDCF.

        Called by DeepStream pipeline probe.

        Args:
            boxes: Dictionary mapping object_id to normalized box (x1, y1, x2, y2)
        """
        self._tracker_boxes = boxes.copy()

    def propagate(
        self,
        current_frame: np.ndarray,
        tracked_objects: Dict[int, "TrackedObject"],
    ) -> Dict[int, Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        """
        Propagate masks using NvDCF boxes and optical flow.

        Args:
            current_frame: Current frame
            tracked_objects: Tracked objects with masks

        Returns:
            Dictionary mapping object_id to (mask, box)
        """
        # First propagate with optical flow
        flow_results = self.mask_propagator.propagate(
            current_frame, tracked_objects
        )

        # Refine with NvDCF boxes
        results = {}
        for obj_id, (mask, flow_box) in flow_results.items():
            # Use NvDCF box if available
            if obj_id in self._tracker_boxes:
                tracker_box = self._tracker_boxes[obj_id]
                # Crop mask to tracker box
                refined_mask = self._crop_mask_to_box(mask, tracker_box)
                results[obj_id] = (refined_mask, tracker_box)
            else:
                results[obj_id] = (mask, flow_box)

        return results

    def _crop_mask_to_box(
        self,
        mask: np.ndarray,
        box: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Crop mask to bounding box region."""
        h, w = mask.shape
        x1, y1, x2, y2 = box

        # Convert normalized coords to pixel coords
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)

        # Create cropped mask
        cropped = np.zeros_like(mask)
        cropped[py1:py2, px1:px2] = mask[py1:py2, px1:px2]

        return cropped

    def reset(self) -> None:
        """Reset propagator state."""
        self.mask_propagator.reset()
        self._tracker_boxes.clear()
