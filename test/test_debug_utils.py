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

    def _clear_captures(self):
        """Clear pytest capture fixtures if available."""
        if hasattr(self, "_capfd"):
            self._capfd.readouterr()
        if hasattr(self, "_caplog"):
            self._caplog.clear()

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

            self._clear_captures()

            result = kernel(x)
            torch.testing.assert_close(result, x + 1)

            # Extract repro script from logs (use records to get the raw message without formatting)
            assert hasattr(self, "_caplog"), "caplog fixture not available"
            repro_script = None
            for record in self._caplog.records:
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

            self._clear_captures()

            # Mock do_bench to fail on the second config with PTXASError (warn level)
            from torch._inductor.runtime.triton_compat import PTXASError
            from triton.testing import do_bench as original_do_bench

            call_count = [0]

            def mock_do_bench(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:  # Fail on second config
                    raise PTXASError("Mocked PTXAS error")
                return original_do_bench(*args, **kwargs)

            with mock.patch("helion.autotuner.base_search.do_bench", mock_do_bench):
                # Autotune will try both configs, second one will fail and print repro
                kernel.autotune([x], force=False)

            # Extract repro script from stderr
            assert hasattr(self, "_capfd"), "capfd fixture not available"
            captured = "".join(self._capfd.readouterr())

            # Verify that a repro script was printed for the failing config
            self.assertIn("# === HELION KERNEL REPRO ===", captured)
            self.assertIn("# === END HELION KERNEL REPRO ===", captured)
            self.assertIn("kernel", captured)
            self.assertIn("helion_repro_caller()", captured)


if __name__ == "__main__":
    unittest.main()
