"""
Async TensorRT runtime with CUDA streams for pipelined inference.

This module implements triple-buffered async inference to overlap:
- H2D transfer (frame N+1)
- Inference (frame N)
- D2H transfer (frame N-1)

Expected performance improvement: 1.2 FPS -> 2.5-3.5 FPS
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from async inference."""
    frame_id: int
    embeddings: Tensor
    inference_time_ms: float
    stream_idx: int


class AsyncTRTInferenceEngine:
    """
    Async TensorRT inference engine with multi-stream pipelining.

    Uses multiple CUDA streams and execution contexts to overlap
    data transfers with computation for higher throughput.
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        device: int = 0,
        num_streams: int = 3,
        use_pinned_memory: bool = True,
    ):
        """
        Initialize async TRT inference engine.

        Args:
            engine_path: Path to TensorRT engine file
            device: CUDA device index
            num_streams: Number of CUDA streams for pipelining
            use_pinned_memory: Use pinned memory for faster H2D transfers
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT not found")

        self.device = device
        self.num_streams = num_streams
        self.use_pinned_memory = use_pinned_memory

        torch.cuda.set_device(device)

        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(str(engine_path), "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

        # Create multiple execution contexts (one per stream)
        self.contexts = [
            self.engine.create_execution_context()
            for _ in range(num_streams)
        ]

        # Create CUDA streams
        self.streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

        # Get IO tensor info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # Determine dtypes
        input_dtype = self.engine.get_tensor_dtype(self.input_name)
        output_dtype = self.engine.get_tensor_dtype(self.output_name)

        self.input_torch_dtype = self._trt_to_torch_dtype(input_dtype)
        self.output_torch_dtype = self._trt_to_torch_dtype(output_dtype)

        # Allocate buffers for each stream
        self._input_buffers: List[Tensor] = []
        self._output_buffers: List[Tensor] = []
        self._pinned_inputs: List[Optional[Tensor]] = []

        self._allocate_buffers()

        # Pipeline state
        self._current_stream_idx = 0
        self._pending_results: Dict[int, int] = {}  # frame_id -> stream_idx
        self._frame_counter = 0

        logger.info(f"Async TRT engine loaded: {engine_path}")
        logger.info(f"  Streams: {num_streams}, Pinned memory: {use_pinned_memory}")
        logger.info(f"  Input: {self.input_name} {self.input_shape}")
        logger.info(f"  Output: {self.output_name} {self.output_shape}")

    def _trt_to_torch_dtype(self, trt_dtype) -> torch.dtype:
        """Convert TensorRT dtype to PyTorch dtype."""
        import tensorrt as trt
        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
        }
        return dtype_map.get(trt_dtype, torch.float32)

    def _allocate_buffers(self) -> None:
        """Allocate GPU and pinned memory buffers for each stream."""
        device_str = f"cuda:{self.device}"

        for i in range(self.num_streams):
            # GPU buffers
            input_buf = torch.empty(
                self.input_shape,
                dtype=self.input_torch_dtype,
                device=device_str
            )
            output_buf = torch.empty(
                self.output_shape,
                dtype=self.output_torch_dtype,
                device=device_str
            )

            self._input_buffers.append(input_buf)
            self._output_buffers.append(output_buf)

            # Set tensor addresses for this context
            self.contexts[i].set_tensor_address(
                self.input_name, input_buf.data_ptr()
            )
            self.contexts[i].set_tensor_address(
                self.output_name, output_buf.data_ptr()
            )

            # Pinned memory for fast H2D transfer
            if self.use_pinned_memory:
                pinned = torch.empty(
                    self.input_shape,
                    dtype=self.input_torch_dtype,
                    pin_memory=True
                )
                self._pinned_inputs.append(pinned)
            else:
                self._pinned_inputs.append(None)

    def infer_async(self, image: Tensor) -> int:
        """
        Submit image for async inference.

        Args:
            image: Input tensor (can be on CPU or GPU)

        Returns:
            Frame ID for retrieving result later
        """
        stream_idx = self._current_stream_idx
        self._current_stream_idx = (self._current_stream_idx + 1) % self.num_streams

        stream = self.streams[stream_idx]
        context = self.contexts[stream_idx]
        input_buf = self._input_buffers[stream_idx]

        frame_id = self._frame_counter
        self._frame_counter += 1

        with torch.cuda.stream(stream):
            # H2D transfer
            if image.device.type == 'cpu':
                if self.use_pinned_memory:
                    # Fast path: CPU -> pinned -> GPU
                    self._pinned_inputs[stream_idx].copy_(image)
                    input_buf.copy_(self._pinned_inputs[stream_idx], non_blocking=True)
                else:
                    input_buf.copy_(image, non_blocking=True)
            else:
                # Already on GPU
                input_buf.copy_(image, non_blocking=True)

            # Execute inference
            context.execute_async_v3(stream.cuda_stream)

        # Track pending result
        self._pending_results[frame_id] = stream_idx

        return frame_id

    def get_result(self, frame_id: int, wait: bool = True) -> Optional[InferenceResult]:
        """
        Get inference result for a frame.

        Args:
            frame_id: Frame ID returned from infer_async
            wait: If True, block until result is ready

        Returns:
            InferenceResult or None if not ready (when wait=False)
        """
        if frame_id not in self._pending_results:
            raise ValueError(f"Unknown frame_id: {frame_id}")

        stream_idx = self._pending_results[frame_id]
        stream = self.streams[stream_idx]

        if wait:
            stream.synchronize()
        elif not stream.query():
            return None

        # Get result
        output = self._output_buffers[stream_idx].clone()

        # Cleanup
        del self._pending_results[frame_id]

        return InferenceResult(
            frame_id=frame_id,
            embeddings=output,
            inference_time_ms=0.0,  # TODO: Add timing
            stream_idx=stream_idx,
        )

    def __call__(self, image: Tensor) -> Tensor:
        """
        Synchronous inference (for compatibility).

        Args:
            image: Input tensor

        Returns:
            Output tensor
        """
        frame_id = self.infer_async(image)
        result = self.get_result(frame_id, wait=True)
        return result.embeddings


class PipelinedInference:
    """
    Triple-buffered pipelined inference for maximum throughput.

    Overlaps:
    - Frame N-2: D2H transfer complete, result ready
    - Frame N-1: Inference running
    - Frame N: H2D transfer in progress

    This achieves near-100% GPU utilization for steady-state processing.
    """

    def __init__(
        self,
        engine: AsyncTRTInferenceEngine,
        buffer_size: int = 3,
    ):
        """
        Initialize pipelined inference.

        Args:
            engine: Async TRT inference engine
            buffer_size: Number of frames to buffer (default 3 for triple buffering)
        """
        self.engine = engine
        self.buffer_size = buffer_size

        # Circular buffer of pending frames
        self._pending_frames: List[Optional[int]] = [None] * buffer_size
        self._buffer_idx = 0
        self._frames_submitted = 0
        self._results_ready = 0

    def submit(self, image: Tensor) -> Optional[InferenceResult]:
        """
        Submit a frame and return the oldest ready result (if any).

        This implements the sliding window pipeline:
        - Submits the new frame for inference
        - Returns the result from (buffer_size) frames ago

        Args:
            image: Input tensor

        Returns:
            InferenceResult from oldest frame, or None if pipeline not yet full
        """
        result = None

        # Check if we have a result ready from the oldest slot
        oldest_idx = self._buffer_idx
        if self._pending_frames[oldest_idx] is not None:
            frame_id = self._pending_frames[oldest_idx]
            result = self.engine.get_result(frame_id, wait=True)
            self._results_ready += 1

        # Submit new frame
        frame_id = self.engine.infer_async(image)
        self._pending_frames[self._buffer_idx] = frame_id
        self._frames_submitted += 1

        # Advance buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self.buffer_size

        return result

    def flush(self) -> List[InferenceResult]:
        """
        Flush remaining results from the pipeline.

        Call this after all frames have been submitted to get
        the remaining buffered results.

        Returns:
            List of remaining InferenceResults
        """
        results = []

        for i in range(self.buffer_size):
            idx = (self._buffer_idx + i) % self.buffer_size
            if self._pending_frames[idx] is not None:
                frame_id = self._pending_frames[idx]
                result = self.engine.get_result(frame_id, wait=True)
                results.append(result)
                self._pending_frames[idx] = None
                self._results_ready += 1

        return results

    @property
    def stats(self) -> Dict[str, int]:
        """Get pipeline statistics."""
        return {
            'frames_submitted': self._frames_submitted,
            'results_ready': self._results_ready,
            'pending': self._frames_submitted - self._results_ready,
        }


class CUDAGraphInference:
    """
    CUDA Graph-captured inference for minimal kernel launch overhead.

    CUDA Graphs capture a sequence of GPU operations and replay them
    with minimal CPU overhead. This is most effective when:
    - Input shape is fixed
    - Same operations are repeated many times

    Typical improvement: 5-15% additional throughput over async inference.
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        device: int = 0,
        warmup_iterations: int = 3,
    ):
        """
        Initialize CUDA Graph inference.

        Args:
            engine_path: Path to TensorRT engine
            device: CUDA device index
            warmup_iterations: Number of warmup runs before capturing graph
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT not found")

        self.device = device
        torch.cuda.set_device(device)

        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(str(engine_path), "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)

        # Get IO info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # Allocate buffers
        device_str = f"cuda:{device}"
        self._input_buffer = torch.empty(self.input_shape, dtype=torch.float32, device=device_str)
        self._output_buffer = torch.empty(self.output_shape, dtype=torch.float32, device=device_str)

        self.context.set_tensor_address(self.input_name, self._input_buffer.data_ptr())
        self.context.set_tensor_address(self.output_name, self._output_buffer.data_ptr())

        # Capture CUDA graph
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._capture_graph(warmup_iterations)

        logger.info(f"CUDA Graph inference initialized: {engine_path}")

    def _capture_graph(self, warmup_iterations: int) -> None:
        """Capture CUDA graph after warmup."""
        # Warmup runs
        for _ in range(warmup_iterations):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # Capture graph
        self._graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._graph, stream=self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)

        logger.info("CUDA graph captured successfully")

    def __call__(self, image: Tensor) -> Tensor:
        """
        Run inference using captured CUDA graph.

        Args:
            image: Input tensor (must match captured input shape)

        Returns:
            Output tensor
        """
        if image.shape != self.input_shape:
            raise ValueError(
                f"Input shape {image.shape} doesn't match captured shape {self.input_shape}"
            )

        # Copy input (graph uses the same buffer addresses)
        self._input_buffer.copy_(image)

        # Replay graph
        self._graph.replay()
        self.stream.synchronize()

        return self._output_buffer.clone()


class AsyncSAM3Runtime:
    """
    High-level async runtime for SAM3 encoder inference.

    Combines AsyncTRTInferenceEngine with PipelinedInference
    for maximum throughput.
    """

    def __init__(
        self,
        encoder_engine_path: Union[str, Path],
        decoder_engine_path: Optional[Union[str, Path]] = None,
        device: int = 0,
        num_streams: int = 3,
        use_cuda_graphs: bool = False,
    ):
        """
        Initialize async SAM3 runtime.

        Args:
            encoder_engine_path: Path to encoder TRT engine
            decoder_engine_path: Path to decoder TRT engine (optional)
            device: CUDA device index
            num_streams: Number of CUDA streams
            use_cuda_graphs: Use CUDA graphs for additional speedup
        """
        self.device = device
        torch.cuda.set_device(device)

        # Load encoder
        if use_cuda_graphs:
            self.encoder = CUDAGraphInference(encoder_engine_path, device)
            self._pipeline = None
        else:
            async_engine = AsyncTRTInferenceEngine(
                encoder_engine_path,
                device=device,
                num_streams=num_streams,
            )
            self._pipeline = PipelinedInference(async_engine, buffer_size=num_streams)
            self.encoder = async_engine

        # Load decoder (sync for now - typically much faster than encoder)
        self.decoder: Optional[AsyncTRTInferenceEngine] = None
        if decoder_engine_path:
            self.decoder = AsyncTRTInferenceEngine(
                decoder_engine_path,
                device=device,
                num_streams=1,  # Decoder is fast, single stream sufficient
            )

        self._use_cuda_graphs = use_cuda_graphs

    def encode_pipelined(self, image: Tensor) -> Optional[Tensor]:
        """
        Encode image using pipelined inference.

        Returns result from 2-3 frames ago (pipeline latency).

        Args:
            image: Input image tensor

        Returns:
            Embeddings from oldest frame, or None if pipeline not full
        """
        if self._use_cuda_graphs:
            # CUDA graphs don't support pipelining
            return self.encoder(image)

        result = self._pipeline.submit(image)
        return result.embeddings if result else None

    def flush_pipeline(self) -> List[Tensor]:
        """Flush remaining results from pipeline."""
        if self._use_cuda_graphs or self._pipeline is None:
            return []

        results = self._pipeline.flush()
        return [r.embeddings for r in results]

    def encode_sync(self, image: Tensor) -> Tensor:
        """Synchronous encode (bypasses pipeline)."""
        return self.encoder(image)
