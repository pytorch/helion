from __future__ import annotations

import unittest

import torch

from helion._compiler.backend import CuteDSLBackend
from helion._compiler.backend import _TORCH_TO_CUTLASS_DTYPE
from helion._testing import TestCase


def _has_cutlass() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


skipIfNoCutlass = unittest.skipUnless(_has_cutlass(), "cutlass not installed")


class TestCuteDSLBackendRegistration(TestCase):
    """Tests for CuteDSL backend registration and basic properties."""

    def test_backend_literal_includes_cutedsl(self):
        """Settings accepts 'cutedsl' as a valid backend literal."""
        from helion.runtime.settings import Settings

        settings = Settings(backend="cutedsl")
        self.assertEqual(settings.backend, "cutedsl")

    def test_backend_selection_via_settings(self):
        """CompileEnvironment creates CuteDSLBackend when settings.backend == 'cutedsl'."""
        from helion._compiler.compile_environment import CompileEnvironment
        from helion.runtime.settings import Settings

        settings = Settings(backend="cutedsl")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = CompileEnvironment(device, settings)
        self.assertIsInstance(env.backend, CuteDSLBackend)
        self.assertEqual(env.backend_name, "cutedsl")

    def test_backend_name(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.name, "cutedsl")

    def test_backend_dtype_str_float16(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.dtype_str(torch.float16), "cutlass.Float16")

    def test_backend_dtype_str_float32(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.dtype_str(torch.float32), "cutlass.Float32")

    def test_backend_dtype_str_bfloat16(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.dtype_str(torch.bfloat16), "cutlass.BFloat16")

    def test_backend_dtype_str_int32(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.dtype_str(torch.int32), "cutlass.Int32")

    def test_backend_dtype_str_bool(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.dtype_str(torch.bool), "cutlass.Boolean")

    def test_backend_dtype_str_unsupported(self):
        backend = CuteDSLBackend()
        with self.assertRaises(ValueError):
            backend.dtype_str(torch.complex128)

    def test_backend_acc_type_float16_promotes(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.acc_type(torch.float16), "cutlass.Float32")

    def test_backend_acc_type_bfloat16_promotes(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.acc_type(torch.bfloat16), "cutlass.Float32")

    def test_backend_acc_type_float32_stays(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.acc_type(torch.float32), "cutlass.Float32")

    def test_backend_function_decorator(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.function_decorator, "cute.kernel")

    def test_backend_constexpr_type(self):
        backend = CuteDSLBackend()
        self.assertEqual(backend.constexpr_type, "cutlass.Constexpr[int]")

    def test_backend_imports_include_cutlass(self):
        backend = CuteDSLBackend()
        imports = backend.library_imports
        self.assertIn("cutlass", imports)
        self.assertEqual(imports["cutlass"], "import cutlass")

    def test_backend_imports_include_cute(self):
        backend = CuteDSLBackend()
        imports = backend.library_imports
        self.assertIn("cute", imports)
        self.assertEqual(imports["cute"], "import cutlass.cute as cute")

    def test_backend_imports_include_launcher(self):
        backend = CuteDSLBackend()
        imports = backend.library_imports
        self.assertIn("_default_cutedsl_launcher", imports)

    def test_backend_imports_no_triton(self):
        backend = CuteDSLBackend()
        imports = backend.library_imports
        self.assertNotIn("triton", imports)
        self.assertNotIn("tl", imports)

    def test_dtype_mapping_completeness(self):
        """Verify all expected dtypes are covered."""
        expected_dtypes = [
            "torch.float16",
            "torch.float32",
            "torch.float64",
            "torch.bfloat16",
            "torch.int8",
            "torch.int16",
            "torch.int32",
            "torch.int64",
            "torch.uint8",
            "torch.bool",
        ]
        for dtype_str in expected_dtypes:
            self.assertIn(dtype_str, _TORCH_TO_CUTLASS_DTYPE, f"Missing {dtype_str}")


class TestCuteDSLLauncher(TestCase):
    """Tests for cutedsl_launcher module."""

    def test_launcher_importable(self):
        from helion.runtime.cutedsl_launcher import default_cutedsl_launcher

        self.assertTrue(callable(default_cutedsl_launcher))


if __name__ == "__main__":
    unittest.main()
