"""TensorRT adapter that drop-in replaces the SAM3 ViT trunk."""

from pathlib import Path
from typing import List, Union

import torch
from torch import Tensor, nn

from .trt_runtime import TRTInferenceEngine


class TRTTrunkAdapter(nn.Module):
    """
    Replaces ``Sam3DualViTDetNeck.trunk`` with a TensorRT engine.

    The neck only consumes ``xs[-1]``, so we return a single-element list whose
    sole tensor matches the original trunk's final feature: ``(B, C, H/14, W/14)``.
    The TRT engine was traced with ``TRTViTWrapper`` at a fixed input size, so
    callers must feed images at that exact resolution (typically 1008x1008).
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        device: int = 0,
    ):
        super().__init__()
        self.engine = TRTInferenceEngine(str(engine_path), device=device)
        in_name = next(iter(self.engine.inputs))
        out_name = next(iter(self.engine.outputs))
        self.input_shape = self.engine.inputs[in_name]["shape"]
        self.output_shape = self.engine.outputs[out_name]["shape"]
        self.input_dtype = self.engine.inputs[in_name]["dtype"]

    def forward(self, x: Tensor) -> List[Tensor]:
        orig_dtype = x.dtype
        x = x.to(
            dtype=self.input_dtype,
            device=f"cuda:{self.engine.device}",
            non_blocking=True,
        ).contiguous()
        out = self.engine(x)
        return [out.to(orig_dtype)]
