"""Integration tests for KeyframeProcessor with TRT engines."""

import pytest
import numpy as np
from pathlib import Path
from typing import Optional

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import vpi
    VPI_AVAILABLE = True
except ImportError:
    VPI_AVAILABLE = False


@pytest.fixture
def engine_dir():
    """Get engine directory path."""
    return Path(__file__).parent.parent / "engines"


@pytest.fixture
def checkpoint_path():
    """Get SAM3 checkpoint path."""
    return Path(__file__).parent.parent.parent / "sam3" / "checkpoints" / "sam3.pt"


class TestKeyframeProcessor:
    """Test KeyframeProcessor logic without TRT engines."""

    def test_keyframe_detection_logic(self):
        """Test that keyframes are correctly identified by frame index."""
        from sam3_deepstream.inference.keyframe_processor import KeyframeProcessor

        # Create processor with dummy encoder/decoder
        processor = KeyframeProcessor(
            encoder_fn=lambda x: torch.zeros(1, 256, 64, 64) if TORCH_AVAILABLE else None,
            decoder_fn=None,
            keyframe_interval=5,
            max_objects=50,
        )

        # Test keyframe detection for first 15 frames
        expected_keyframes = {0, 5, 10}
        for i in range(15):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = processor.process_frame(frame)

            expected = i in expected_keyframes
            assert result.is_keyframe == expected, f"Frame {i}: expected keyframe={expected}, got {result.is_keyframe}"
            assert result.frame_idx == i

    def test_tracked_object_management(self):
        """Test adding and removing tracked objects."""
        from sam3_deepstream.inference.keyframe_processor import KeyframeProcessor

        processor = KeyframeProcessor(
            encoder_fn=lambda x: torch.zeros(1, 256, 64, 64) if TORCH_AVAILABLE else None,
            decoder_fn=None,
            keyframe_interval=5,
        )

        # Add object with dummy mask
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 100:200] = 1

        obj_id = processor.add_object(mask, score=0.95)
        assert obj_id == 1
        assert len(processor.tracked_objects) == 1

        # Add another object
        obj_id2 = processor.add_object(mask, score=0.85)
        assert obj_id2 == 2
        assert len(processor.tracked_objects) == 2

        # Remove first object
        assert processor.remove_object(1) == True
        assert len(processor.tracked_objects) == 1
        assert 2 in processor.tracked_objects

        # Try to remove non-existent object
        assert processor.remove_object(99) == False

    def test_processor_reset(self):
        """Test that reset clears all state."""
        from sam3_deepstream.inference.keyframe_processor import KeyframeProcessor

        processor = KeyframeProcessor(
            encoder_fn=lambda x: None,
            decoder_fn=None,
            keyframe_interval=5,
        )

        # Add objects and process some frames
        mask = np.zeros((480, 640), dtype=np.uint8)
        processor.add_object(mask, score=0.9)
        processor.add_object(mask, score=0.8)

        # Process a few frames
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            processor.process_frame(frame)

        assert processor.frame_count == 5
        assert len(processor.tracked_objects) == 2

        # Reset
        processor.reset()

        assert processor.frame_count == 0
        assert len(processor.tracked_objects) == 0


class TestMaskPropagation:
    """Test MaskPropagator functionality."""

    def test_mask_propagator_initialization(self):
        """Test MaskPropagator initializes correctly."""
        from sam3_deepstream.inference.mask_propagation import MaskPropagator

        # Without VPI
        propagator = MaskPropagator(use_vpi=False)
        assert propagator.use_vpi == False

        # With VPI (may fall back to CPU)
        propagator_vpi = MaskPropagator(use_vpi=True)
        # VPI availability depends on system

    def test_mask_warp_basic(self):
        """Test basic mask warping with identity flow."""
        from sam3_deepstream.inference.mask_propagation import MaskPropagator

        propagator = MaskPropagator(use_vpi=False)

        # Create a simple mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1

        # Zero flow should preserve mask
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        warped = propagator._warp_mask(mask, flow)

        # Mask should be approximately preserved
        assert warped.shape == mask.shape
        assert warped.sum() > 0

    def test_mask_to_box_extraction(self):
        """Test bounding box extraction from mask."""
        from sam3_deepstream.inference.mask_propagation import MaskPropagator

        propagator = MaskPropagator(use_vpi=False)

        # Create mask with known bounding box
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[20:40, 50:100] = 1  # y: 20-40, x: 50-100

        box = propagator._mask_to_box(mask)
        x1, y1, x2, y2 = box

        # Check normalized coordinates
        assert abs(x1 - 50/200) < 0.01
        assert abs(y1 - 20/100) < 0.01
        assert abs(x2 - 99/200) < 0.02  # -1 because of 0-indexing
        assert abs(y2 - 39/100) < 0.02

    def test_empty_mask_box(self):
        """Test box extraction from empty mask."""
        from sam3_deepstream.inference.mask_propagation import MaskPropagator

        propagator = MaskPropagator(use_vpi=False)

        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        box = propagator._mask_to_box(empty_mask)

        assert box == (0.0, 0.0, 0.0, 0.0)


class TestNvDCFMaskPropagator:
    """Test NvDCF-integrated mask propagation."""

    def test_nvdcf_propagator_initialization(self):
        """Test NvDCFMaskPropagator initializes correctly."""
        from sam3_deepstream.inference.mask_propagation import (
            MaskPropagator,
            NvDCFMaskPropagator,
        )

        base_propagator = MaskPropagator(use_vpi=False)
        nvdcf_propagator = NvDCFMaskPropagator(mask_propagator=base_propagator)

        assert nvdcf_propagator.mask_propagator is not None

    def test_tracker_box_update(self):
        """Test updating tracker boxes from NvDCF."""
        from sam3_deepstream.inference.mask_propagation import NvDCFMaskPropagator

        propagator = NvDCFMaskPropagator()

        # Update with tracked boxes
        boxes = {
            1: (0.1, 0.1, 0.3, 0.3),
            2: (0.5, 0.5, 0.8, 0.8),
        }
        propagator.update_tracker_boxes(boxes)

        assert 1 in propagator._tracker_boxes
        assert 2 in propagator._tracker_boxes
        assert propagator._tracker_boxes[1] == (0.1, 0.1, 0.3, 0.3)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTRTIntegration:
    """Test integration with TensorRT engines (requires engines to be built)."""

    def test_trt_runtime_initialization(self, engine_dir):
        """Test TRT runtime can be initialized if engines exist."""
        encoder_path = engine_dir / "sam3_encoder.engine"

        if not encoder_path.exists():
            pytest.skip("Encoder engine not built yet")

        from sam3_deepstream.inference.trt_runtime import SAM3TRTRuntime

        runtime = SAM3TRTRuntime(encoder_engine_path=encoder_path)
        assert runtime is not None

    def test_keyframe_processor_with_trt(self, engine_dir):
        """Test KeyframeProcessor with TRT encoder."""
        encoder_path = engine_dir / "sam3_encoder.engine"

        if not encoder_path.exists():
            pytest.skip("Encoder engine not built yet")

        from sam3_deepstream.inference.trt_runtime import SAM3TRTRuntime
        from sam3_deepstream.inference.keyframe_processor import KeyframeProcessor
        from sam3_deepstream.inference.mask_propagation import MaskPropagator

        # Initialize TRT runtime
        runtime = SAM3TRTRuntime(encoder_engine_path=encoder_path)

        # Create processor
        processor = KeyframeProcessor(
            encoder_fn=runtime.encode_image,
            decoder_fn=None,
            keyframe_interval=5,
        )

        # Set up propagation (use CPU for testing)
        propagator = MaskPropagator(use_vpi=False)
        processor.set_propagator(propagator)

        # Process test frames
        for i in range(10):
            frame = np.random.randint(0, 255, (1008, 1008, 3), dtype=np.uint8)
            result = processor.process_frame(frame)

            expected_keyframe = (i % 5 == 0)
            assert result.is_keyframe == expected_keyframe
