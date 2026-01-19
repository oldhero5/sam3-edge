#!/usr/bin/env python3
"""Quick test to validate TRT encoder engine."""

import time
import torch
import tensorrt as trt
import numpy as np

def test_encoder_engine(engine_path: str, image_size: int = 336):
    """Test encoder engine with dummy input."""
    print(f"Loading engine: {engine_path}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("FAILED: Could not load engine")
        return False

    context = engine.create_execution_context()

    # Get binding info
    print(f"\nEngine bindings ({engine.num_io_tensors}):")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = "INPUT" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT"
        print(f"  [{i}] {name}: {shape} ({dtype}) - {mode}")

    # Allocate buffers
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    # Create dummy input
    print(f"\nCreating dummy input: {input_shape}")
    input_tensor = torch.randn(tuple(input_shape), dtype=torch.float32, device='cuda')
    output_tensor = torch.empty(tuple(output_shape), dtype=torch.float32, device='cuda')

    # Set tensor addresses
    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    # Warm up
    print("Running warm-up inference...")
    stream = torch.cuda.current_stream()
    context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    # Benchmark
    print("Running benchmark (10 iterations)...")
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    print(f"\nResults:")
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Throughput: {1000/avg_time:.1f} FPS")
    print(f"  Output shape: {output_tensor.shape}")
    print(f"  Output range: [{output_tensor.min().item():.4f}, {output_tensor.max().item():.4f}]")

    # Basic sanity check
    if output_tensor.isnan().any():
        print("WARNING: Output contains NaN values!")
        return False

    if output_tensor.isinf().any():
        print("WARNING: Output contains Inf values!")
        return False

    print("\nSUCCESS: Engine validation passed!")
    return True


if __name__ == '__main__':
    import sys
    engine_path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/sam3_deepstream/engines/sam3_encoder.engine'
    test_encoder_engine(engine_path)
