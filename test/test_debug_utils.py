from __future__ import annotations

import contextlib
import linecache
import os
import unittest
from unittest import mock

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfCpu
import helion.language as hl


@pytest.fixture(autouse=True)
def _store_capfd_on_class(request, capfd):
    """
    Expose pytest's capfd fixture as `self._capfd` inside the TestDebugUtils class
    (works for unittest.TestCase-style tests).
    """
    if request.cls is not None:
        request.cls._capfd = capfd


@pytest.fixture(autouse=True)
def _store_caplog_on_class(request, caplog):
    """
    Expose pytest's caplog fixture as `self._caplog` inside the TestDebugUtils class
    (works for unittest.TestCase-style tests).
    """
    if request.cls is not None:
        request.cls._caplog = caplog


class TestDebugUtils(RefEagerTestDisabled, TestCase):
    @contextlib.contextmanager
    def _with_print_repro_enabled(self):
        """Context manager to temporarily set HELION_PRINT_REPRO=1."""
        original = os.environ.get("HELION_PRINT_REPRO")
        os.environ["HELION_PRINT_REPRO"] = "1"
        try:
            yield
        finally:
            if original is None:
                os.environ.pop("HELION_PRINT_REPRO", None)
            else:
                os.environ["HELION_PRINT_REPRO"] = original

    def _create_kernel(self, **kwargs):
        """Create a simple 1D kernel for testing.

        Args:
            **kwargs: Arguments to pass to @helion.kernel decorator.
        """

        @helion.kernel(**kwargs)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.shape[0]
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] + 1
            return out

        return kernel

    def test_print_repro_env_var(self):
        """Ensure HELION_PRINT_REPRO=1 emits an executable repro script."""
        with self._with_print_repro_enabled():
            kernel = self._create_kernel(
                config=helion.Config(block_sizes=[32], num_warps=4),
                static_shapes=True,
            )

            torch.manual_seed(0)
            x = torch.randn([128], dtype=torch.float32, device=DEVICE)

            with self.capture_logs() as log_capture:
                result = kernel(x)
                torch.testing.assert_close(result, x + 1)

                # Extract repro script from logs (use records to get the raw message without formatting)
                repro_script = None
                for record in log_capture.records:
                    if "# === HELION KERNEL REPRO ===" in record.message:
                        repro_script = record.message
                        break

                if repro_script is None:
                    self.fail("No repro script found in logs")

            # Normalize range_warp_specializes=[None] to [] for comparison
            normalized_script = repro_script.replace(
                "range_warp_specializes=[None]", "range_warp_specializes=[]"
            )

            # Verify repro script matches expected script
            self.assertExpectedJournal(normalized_script)

            # Setup linecache so inspect.getsource() works on exec'd code
            filename = "<helion_repro_test>"
            linecache.cache[filename] = (
                len(repro_script),
                None,
                [f"{line}\n" for line in repro_script.splitlines()],
                filename,
            )

            # Execute the repro script
            namespace = {}
            exec(compile(repro_script, filename, "exec"), namespace)

            # Call the generated helper and verify it runs successfully
            helper = namespace["helion_repro_caller"]
            repro_result = helper()

            # Verify the output
            torch.testing.assert_close(repro_result, x + 1)

            linecache.cache.pop(filename, None)

    @skipIfCpu("debug")
    def test_print_repro_on_autotune_error(self):
        """Ensure HELION_PRINT_REPRO=1 prints repro when configs fail during autotuning.

        This test mocks do_bench to fail on the second config, guaranteeing the repro
        printing code path is exercised for "warn" level errors.
        """
        with self._with_print_repro_enabled():
            kernel = self._create_kernel(
                configs=[
                    helion.Config(block_sizes=[32], num_warps=4),
                    helion.Config(block_sizes=[64], num_warps=8),
                ],
                autotune_precompile=False,
            )

            torch.manual_seed(0)
            x = torch.randn([128], dtype=torch.float32, device=DEVICE)

            # Mock do_bench to fail on the second config with PTXASError (warn level)
            from torch._inductor.runtime.triton_compat import PTXASError
            from triton.testing import do_bench as original_do_bench

            call_count = [0]

            def mock_do_bench(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:  # Fail on second config
                    raise PTXASError("Mocked PTXAS error")
                return original_do_bench(*args, **kwargs)

            with self.capture_output() as output_capture:
                with mock.patch("helion.autotuner.base_search.do_bench", mock_do_bench):
                    # Autotune will try both configs, second one will fail and print repro
                    kernel.autotune([x], force=False)

                # Extract repro script from stderr
                captured = "".join(output_capture.readouterr())

            # Verify that a repro script was printed for the failing config
            self.assertIn("# === HELION KERNEL REPRO ===", captured)
            self.assertIn("# === END HELION KERNEL REPRO ===", captured)
            self.assertIn("kernel", captured)
            self.assertIn("helion_repro_caller()", captured)

    def test_print_repro_on_device_ir_lowering_error(self):
        """Ensure HELION_PRINT_REPRO=1 prints repro when compilation fails during device IR lowering."""
        with self._with_print_repro_enabled():

            @helion.kernel(config=helion.Config(block_sizes=[32], num_warps=4))
            def kernel_with_compile_error(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                n = x.shape[0]
                for tile_n in hl.tile([n]):
                    # Using torch.nonzero inside device loop causes compilation error
                    # because it produces data-dependent output shape
                    torch.nonzero(x[tile_n])
                    out[tile_n] = x[tile_n]
                return out

            torch.manual_seed(0)
            x = torch.randn([128], dtype=torch.float32, device=DEVICE)

            with self.capture_logs() as log_capture:
                # This should trigger a compilation error during device IR lowering
                with self.assertRaises(RuntimeError):
                    kernel_with_compile_error(x)

                # Extract repro script from logs
                repro_script = None
                for record in log_capture.records:
                    if "# === HELION KERNEL REPRO ===" in record.message:
                        repro_script = record.message
                        break

                # Verify that a repro script was printed when compilation failed
                self.assertIsNotNone(
                    repro_script,
                    "Expected repro script to be printed when device IR lowering fails",
                )
                self.assertIn("# === HELION KERNEL REPRO ===", repro_script)
                self.assertIn("# === END HELION KERNEL REPRO ===", repro_script)
                self.assertIn("kernel_with_compile_error", repro_script)
                self.assertIn("helion_repro_caller()", repro_script)

    def test_print_repro_on_triton_codegen_error(self):
        """Ensure HELION_PRINT_REPRO=1 prints repro when Triton codegen fails."""
        with self._with_print_repro_enabled():

            @helion.kernel(config=helion.Config(block_sizes=[32], num_warps=4))
            def kernel_with_triton_error(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                n = x.shape[0]
                for tile_n in hl.tile([n]):
                    out[tile_n] = x[tile_n] + 1
                return out

            torch.manual_seed(0)
            x = torch.randn([128], dtype=torch.float32, device=DEVICE)

            # Mock PyCodeCache.load to simulate a Triton codegen error
            from torch._inductor.codecache import PyCodeCache

            original_load = PyCodeCache.load

            def mock_load(code, *args, **kwargs):
                if "kernel_with_triton_error" in code:
                    raise RuntimeError("Simulated Triton codegen error")
                return original_load(code, *args, **kwargs)

            with (
                self.capture_logs() as log_capture,
                mock.patch.object(PyCodeCache, "load", mock_load),
            ):
                # This should trigger a Triton codegen error
                with self.assertRaises(RuntimeError):
                    kernel_with_triton_error(x)

                # Extract repro script from logs
                repro_script = None
                for record in log_capture.records:
                    if "# === HELION KERNEL REPRO ===" in record.message:
                        repro_script = record.message
                        break

                # Verify that a repro script was printed when Triton codegen failed
                self.assertIsNotNone(
                    repro_script,
                    "Expected repro script to be printed when Triton codegen fails",
                )
                self.assertIn("# === HELION KERNEL REPRO ===", repro_script)
                self.assertIn("# === END HELION KERNEL REPRO ===", repro_script)
                self.assertIn("kernel_with_triton_error", repro_script)
                self.assertIn("helion_repro_caller()", repro_script)


if __name__ == "__main__":
    unittest.main()
