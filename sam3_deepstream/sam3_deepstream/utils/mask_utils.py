"""Mask manipulation utilities including RLE encoding."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def encode_rle(mask: np.ndarray) -> dict:
    """
    Encode binary mask as Run-Length Encoding (RLE).

    Args:
        mask: Binary mask array (H, W) with values 0 or 1

    Returns:
        Dictionary with 'counts' (string) and 'size' (H, W)
    """
    mask = np.asfortranarray(mask.astype(np.uint8))
    h, w = mask.shape

    # Flatten in column-major order
    flat = mask.flatten(order='F')

    # Find run lengths
    diff = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Build RLE counts: [zeros_before_first, length_first, zeros_after_first, ...]
    counts = []
    prev_end = 0

    for start, end in zip(starts, ends):
        counts.append(start - prev_end)  # zeros before this run
        counts.append(end - start)  # length of run
        prev_end = end

    # Add trailing zeros if needed
    if prev_end < len(flat):
        counts.append(len(flat) - prev_end)

    # If starts with 1s, prepend 0
    if len(counts) > 0 and flat[0] == 1:
        counts = [0] + counts

    # Convert to string
    counts_str = " ".join(map(str, counts))

    return {
        "counts": counts_str,
        "size": (h, w),
    }


def decode_rle(rle: dict) -> np.ndarray:
    """
    Decode Run-Length Encoding to binary mask.

    Args:
        rle: Dictionary with 'counts' and 'size'

    Returns:
        Binary mask array (H, W)
    """
    h, w = rle["size"]

    # Parse counts
    if isinstance(rle["counts"], str):
        counts = list(map(int, rle["counts"].split()))
    else:
        counts = rle["counts"]

    # Decode
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0

    for i, count in enumerate(counts):
        if i % 2 == 1:  # Odd indices are foreground runs
            mask[pos:pos + count] = 1
        pos += count

    return mask.reshape((h, w), order='F')


def mask_to_polygon(
    mask: np.ndarray,
    simplify_tolerance: float = 1.0,
) -> List[np.ndarray]:
    """
    Convert binary mask to polygon contours.

    Args:
        mask: Binary mask (H, W)
        simplify_tolerance: Polygon simplification tolerance

    Returns:
        List of polygon contours
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            # Simplify polygon
            epsilon = simplify_tolerance * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 3:
                polygons.append(approx.reshape(-1, 2))

    return polygons


def polygon_to_mask(
    polygon: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert polygon to binary mask.

    Args:
        polygon: Polygon vertices (N, 2)
        size: Mask size (height, width)

    Returns:
        Binary mask
    """
    mask = np.zeros(size, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two masks.

    Args:
        mask1: First binary mask
        mask2: Second binary mask

    Returns:
        IoU score (0-1)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def resize_mask(
    mask: np.ndarray,
    size: Tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """
    Resize mask to target size.

    Args:
        mask: Binary mask
        size: Target size (width, height)
        interpolation: OpenCV interpolation method

    Returns:
        Resized mask
    """
    resized = cv2.resize(
        mask.astype(np.uint8),
        size,
        interpolation=interpolation,
    )
    return (resized > 0.5).astype(np.uint8)


def visualize_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay masks on image with colors.

    Args:
        image: Input image (H, W, 3) BGR
        masks: List of binary masks
        colors: Optional list of RGB colors
        alpha: Transparency (0-1)

    Returns:
        Image with mask overlays
    """
    if not masks:
        return image.copy()

    # Generate colors if not provided
    if colors is None:
        colors = generate_colors(len(masks))

    result = image.copy().astype(np.float32)

    for mask, color in zip(masks, colors):
        if mask.shape[:2] != image.shape[:2]:
            mask = resize_mask(mask, (image.shape[1], image.shape[0]))

        # Create colored overlay
        overlay = np.zeros_like(image, dtype=np.float32)
        overlay[mask > 0] = color

        # Blend
        mask_3d = np.stack([mask] * 3, axis=-1).astype(np.float32)
        result = result * (1 - alpha * mask_3d) + overlay * alpha * mask_3d

    return result.astype(np.uint8)


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for visualization."""
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 255, 128),  # Spring green
        (255, 0, 128),  # Rose
    ]

    # Extend if needed
    while len(colors) < n:
        colors.append((
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        ))

    return colors[:n]


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Combine multiple masks into a single labeled mask.

    Args:
        masks: List of binary masks

    Returns:
        Labeled mask where each object has a unique ID
    """
    if not masks:
        return np.zeros((1, 1), dtype=np.int32)

    h, w = masks[0].shape
    combined = np.zeros((h, w), dtype=np.int32)

    for i, mask in enumerate(masks, start=1):
        combined[mask > 0] = i

    return combined
