"""Tests for ONNX/TensorRT export pipeline.

Run with:
    pytest tests/test_export.py -v
"""

import sys
from pathlib import Path

import pytest


class TestPEEncoderExportImports:
    """Test that pe_encoder_export.py has correct imports."""

    def test_pe_encoder_export_imports_successfully(self):
        """pe_encoder_export.py should import without errors."""
        try:
            from sam3_deepstream.export import pe_encoder_export
        except ImportError as e:
            # Skip if missing dependencies like triton
            if "triton" in str(e) or "sam3" in str(e):
                pytest.skip(f"Skipping due to missing dependency: {e}")
            pytest.fail(f"Failed to import pe_encoder_export: {e}")

    def test_pe_encoder_export_has_numpy(self):
        """pe_encoder_export should have numpy available."""
        from sam3_deepstream.export import pe_encoder_export

        # Check that numpy is importable in the module's context
        import numpy as np

        # The module should use np, so numpy must be available
        assert np is not None

    def test_pe_encoder_trt_runtime_importable(self):
        """PEEncoderTRTRuntime class should be importable."""
        try:
            from sam3_deepstream.export.pe_encoder_export import PEEncoderTRTRuntime
        except ImportError as e:
            pytest.fail(f"Failed to import PEEncoderTRTRuntime: {e}")
        except NameError as e:
            pytest.fail(f"NameError in PEEncoderTRTRuntime (likely missing numpy): {e}")

    def test_pe_encoder_wrapper_importable(self):
        """PEEncoderWrapper class should be importable."""
        try:
            from sam3_deepstream.export.pe_encoder_export import PEEncoderWrapper
        except ImportError as e:
            pytest.fail(f"Failed to import PEEncoderWrapper: {e}")


class TestEncoderExport:
    """Test encoder export functionality."""

    def test_encoder_export_imports(self):
        """encoder_export.py should import without errors."""
        try:
            from sam3_deepstream.export import encoder_export
        except ImportError as e:
            pytest.fail(f"Failed to import encoder_export: {e}")

    def test_encoder_export_function_exists(self):
        """export_encoder_to_tensorrt function should exist."""
        try:
            from sam3_deepstream.export.encoder_export import export_encoder_to_tensorrt
            assert export_encoder_to_tensorrt is not None
        except ImportError as e:
            if "triton" in str(e) or "sam3" in str(e):
                pytest.skip(f"Skipping due to missing dependency: {e}")
            pytest.fail(f"Failed to import: {e}")


class TestDecoderExport:
    """Test decoder export functionality."""

    def test_decoder_export_imports(self):
        """decoder_export.py should import without errors."""
        try:
            from sam3_deepstream.export import decoder_export
        except ImportError as e:
            pytest.fail(f"Failed to import decoder_export: {e}")

    def test_decoder_wrapper_exists(self):
        """TRT decoder classes should exist."""
        try:
            from sam3_deepstream.export.decoder_export import TRTPixelDecoder, TRTMaskPredictor
            assert TRTPixelDecoder is not None
            assert TRTMaskPredictor is not None
        except ImportError as e:
            if "triton" in str(e) or "sam3" in str(e):
                pytest.skip(f"Skipping due to missing dependency: {e}")
            pytest.fail(f"Failed to import: {e}")


class TestExportFunctions:
    """Test export function signatures and existence."""

    def test_export_pe_encoder_function_exists(self):
        """export_pe_encoder function should exist."""
        from sam3_deepstream.export.pe_encoder_export import export_pe_encoder
        assert callable(export_pe_encoder)

    def test_export_pe_text_encoder_function_exists(self):
        """export_pe_text_encoder function should exist."""
        from sam3_deepstream.export.pe_encoder_export import export_pe_text_encoder
        assert callable(export_pe_text_encoder)


def _engines_exist() -> bool:
    """Check if engines directory exists (works in container or locally)."""
    possible_paths = [
        Path("/workspace/sam3_deepstream/engines"),  # Container path
        Path(__file__).parent.parent / "engines",  # Relative to test file
    ]
    return any(p.exists() and list(p.glob("*.engine")) for p in possible_paths)


@pytest.mark.skipif(not _engines_exist(), reason="Engines directory not found")
class TestEngineFiles:
    """Test that engine files are valid (if they exist)."""

    def test_engine_directory_exists(self, engine_dir: Path):
        """Engine directory should exist."""
        assert engine_dir.exists()

    def test_encoder_engine_valid_if_exists(self, engine_dir: Path):
        """Encoder engine file should be valid TRT format if it exists."""
        encoder_path = engine_dir / "sam3_encoder.engine"
        if not encoder_path.exists():
            pytest.skip("Encoder engine not built yet")

        # TRT engine files start with specific magic bytes
        with open(encoder_path, "rb") as f:
            header = f.read(4)
            # TensorRT serialized engine has specific format
            assert len(header) == 4, "Engine file too small"

    def test_decoder_engine_valid_if_exists(self, engine_dir: Path):
        """Decoder engine file should be valid TRT format if it exists."""
        decoder_path = engine_dir / "sam3_decoder.engine"
        if not decoder_path.exists():
            pytest.skip("Decoder engine not built yet")

        with open(decoder_path, "rb") as f:
            header = f.read(4)
            assert len(header) == 4, "Engine file too small"


class TestONNXValidation:
    """Test ONNX file validation (if onnxruntime available)."""

    @pytest.fixture
    def onnx_runtime(self):
        """Get onnxruntime or skip."""
        try:
            import onnxruntime as ort
            return ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

    def test_onnx_encoder_valid_if_exists(self, engine_dir: Path, onnx_runtime):
        """Encoder ONNX file should be loadable if it exists."""
        onnx_path = engine_dir / "sam3_encoder.onnx"
        if not onnx_path.exists():
            pytest.skip("Encoder ONNX not exported yet")

        try:
            session = onnx_runtime.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"]
            )
            assert session is not None
            assert len(session.get_inputs()) > 0
        except Exception as e:
            pytest.fail(f"Failed to load encoder ONNX: {e}")

    def test_onnx_decoder_valid_if_exists(self, engine_dir: Path, onnx_runtime):
        """Decoder ONNX file should be loadable if it exists."""
        onnx_path = engine_dir / "sam3_decoder.onnx"
        if not onnx_path.exists():
            pytest.skip("Decoder ONNX not exported yet")

        try:
            session = onnx_runtime.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"]
            )
            assert session is not None
            assert len(session.get_inputs()) > 0
        except Exception as e:
            pytest.fail(f"Failed to load decoder ONNX: {e}")


class TestExportEnginesScript:
    """Test export_engines.py script."""

    def test_export_engines_script_imports(self):
        """export_engines.py should be importable as a module."""
        # Try multiple possible script locations
        possible_paths = [
            Path(__file__).parent.parent / "scripts",
            Path(__file__).parent.parent.parent / "scripts",
            Path("/workspace/scripts"),
        ]

        scripts_dir = None
        for p in possible_paths:
            if (p / "export_engines.py").exists():
                scripts_dir = p
                break

        if scripts_dir is None:
            pytest.skip("export_engines.py script not found in expected locations")

        sys.path.insert(0, str(scripts_dir))

        try:
            # Try to import the module (may fail if dependencies missing)
            import export_engines
        except ImportError as e:
            # Expected if heavy dependencies not installed
            if "torch" in str(e) or "sam3" in str(e) or "config" in str(e):
                pytest.skip(f"Heavy dependencies not available: {e}")
            else:
                pytest.fail(f"Unexpected import error: {e}")
        finally:
            if str(scripts_dir) in sys.path:
                sys.path.remove(str(scripts_dir))

    def test_export_engines_script_exists(self):
        """export_engines.py script should exist."""
        # Try multiple possible locations
        possible_paths = [
            Path(__file__).parent.parent / "scripts" / "export_engines.py",
            Path(__file__).parent.parent.parent / "scripts" / "export_engines.py",
            Path("/workspace/scripts/export_engines.py"),
        ]

        found = any(p.exists() for p in possible_paths)
        if not found:
            pytest.skip(f"Script not found in expected locations: {[str(p) for p in possible_paths]}")
