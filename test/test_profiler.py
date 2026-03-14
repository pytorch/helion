"""Tests for helion.tools.profiler."""

from __future__ import annotations

import gc
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch


class TestGpuSpecs(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_specs_keys(self) -> None:
        from helion.tools.profiler import _gpu_specs

        specs = _gpu_specs(0)
        for key in ("name", "sm_count", "peak_bw_GBps", "peak_fp32_tflops", "total_mem_GB"):
            self.assertIn(key, specs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_peak_bw_positive(self) -> None:
        from helion.tools.profiler import _gpu_specs

        specs = _gpu_specs(0)
        self.assertGreater(specs["peak_bw_GBps"], 0)


class TestClassify(unittest.TestCase):
    def test_memory_bound(self) -> None:
        from helion.tools.profiler import _classify

        result = _classify(75.0, None)
        self.assertIn("Memory-bound", result)
        self.assertIn("75.0%", result)

    def test_compute_bound(self) -> None:
        from helion.tools.profiler import _classify

        result = _classify(20.0, 80.0)
        self.assertIn("Compute-bound", result)

    def test_latency_bound(self) -> None:
        from helion.tools.profiler import _classify

        result = _classify(10.0, 5.0)
        self.assertIn("Latency-bound", result)

    def test_memory_bound_wins_over_compute(self) -> None:
        from helion.tools.profiler import _classify

        # Both over 60 % — memory-bound takes precedence
        result = _classify(70.0, 70.0)
        self.assertIn("Memory-bound", result)


class TestTimeKernelGCBehavior(unittest.TestCase):
    """Verify that _time_kernel re-enables GC even when the kernel raises."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gc_restored_on_success(self) -> None:
        from helion.tools.profiler import _time_kernel

        was_enabled = gc.isenabled()

        def noop(*_: object) -> None:
            x = torch.zeros(1, device="cuda")
            return x

        gc.enable()
        _time_kernel(noop, (torch.zeros(1, device="cuda"),), warmup=1, iterations=2)
        self.assertTrue(gc.isenabled(), "GC should be re-enabled after timing")
        if not was_enabled:
            gc.disable()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gc_restored_on_exception(self) -> None:
        from helion.tools.profiler import _time_kernel

        def bad_fn(*_: object) -> None:
            raise RuntimeError("boom")

        gc.enable()
        with self.assertRaises(RuntimeError):
            _time_kernel(bad_fn, (), warmup=0, iterations=1)
        self.assertTrue(gc.isenabled(), "GC should be re-enabled after exception")


class TestKernelProfilerCpuOnly(unittest.TestCase):
    """Unit-test KernelProfiler internals that don't need a GPU."""

    def _make_profiler(self) -> object:
        from helion.tools.profiler import KernelProfiler

        return KernelProfiler(
            helion_fn=lambda x: x,
            reference_fn=lambda x: x,
            input_generator=lambda: (torch.zeros(4),),
            kernel_name="identity",
        )

    def test_default_kernel_name_from_fn(self) -> None:
        from helion.tools.profiler import KernelProfiler

        def my_fn(x: object) -> object:
            return x

        p = KernelProfiler(
            helion_fn=my_fn,
            reference_fn=my_fn,
            input_generator=lambda: (torch.zeros(1),),
        )
        self.assertEqual(p.kernel_name, "my_fn")

    def test_explicit_kernel_name(self) -> None:
        p = self._make_profiler()
        self.assertEqual(p.kernel_name, "identity")  # type: ignore[union-attr]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_run_returns_expected_keys(self) -> None:
        from helion.tools.profiler import KernelProfiler

        x = torch.zeros(256, device="cuda")

        def fn(t: torch.Tensor) -> torch.Tensor:
            return t + 1

        p = KernelProfiler(
            helion_fn=fn,
            reference_fn=fn,
            input_generator=lambda: (x,),
            kernel_name="add_one",
            bytes_read_fn=lambda a: a[0].numel() * a[0].element_size(),
            bytes_written_fn=lambda a: a[0].numel() * a[0].element_size(),
            warmup=1,
            iterations=3,
        )
        results = p.run(capture_triton=False)
        for key in ("gpu", "ref_timing", "helion_timing", "speedup",
                    "bw_util_pct", "classification"):
            self.assertIn(key, results)
        self.assertGreater(results["speedup"], 0)  # type: ignore[operator]
        self.assertGreater(results["bw_util_pct"], 0)  # type: ignore[operator]


class TestCaptureTritonCode(unittest.TestCase):
    def test_fallback_when_no_output(self) -> None:
        from helion.tools.profiler import _capture_triton_code

        # A plain Python function (not a Helion kernel) will not write to stderr
        # — we should get the fallback message.
        def plain(x: object) -> object:
            return x

        result = _capture_triton_code(plain, (torch.zeros(1),))
        # Should not be empty and should contain a helpful message
        self.assertTrue(len(result) > 0)


class TestPublicAPI(unittest.TestCase):
    def test_profile_function_is_callable(self) -> None:
        from helion.tools import profile

        self.assertTrue(callable(profile))

    def test_kernel_profiler_importable(self) -> None:
        from helion.tools import KernelProfiler

        self.assertTrue(KernelProfiler is not None)

    def test_tools_in_helion_namespace(self) -> None:
        import helion

        self.assertTrue(hasattr(helion, "tools"))


if __name__ == "__main__":
    unittest.main()
