# SAM3 Fork Patches

Custom modifications to the [SAM3](https://github.com/facebookresearch/sam3) codebase for Jetson Orin TensorRT optimization and Perception Encoder integration.

These files are vendored here so you don't need the separate SAM3 fork.

## Files

### New files (copy into `sam3/sam3/model/`)

- `sam3/model/trt_export.py` — TensorRT export utilities: ONNX-compatible attention, RoPE, ViT wrapper, engine builder
- `sam3/model/pe_encoder.py` — Perception Encoder (PE) vision backbone with alignment tuning
- `sam3/model/pe_text_encoder.py` — PE text encoder integration

### Patches (apply to existing SAM3 checkout)

- `trt_export.patch` — Diff for `trt_export.py` (if you already have an older version)
- `model_builder.patch` — Adds PE backbone integration to `sam3/model_builder.py`

## Usage

### Option A: Copy files directly

```bash
# From the repo root
cp patches/sam3/model/trt_export.py    sam3/sam3/model/trt_export.py
cp patches/sam3/model/pe_encoder.py    sam3/sam3/model/pe_encoder.py
cp patches/sam3/model/pe_text_encoder.py sam3/sam3/model/pe_text_encoder.py

# Apply model_builder patch
cd sam3 && git apply ../patches/model_builder.patch
```

### Option B: Apply all patches to a fresh SAM3 checkout

```bash
cd sam3
git apply ../patches/trt_export.patch
git apply ../patches/model_builder.patch
cp ../patches/sam3/model/pe_encoder.py    sam3/model/pe_encoder.py
cp ../patches/sam3/model/pe_text_encoder.py sam3/model/pe_text_encoder.py
```
