from __future__ import annotations

import linecache
import os
import unittest

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


class TestDebugUtils(RefEagerTestDisabled, TestCase):
    def test_print_repro_env_var(self):
        """Ensure HELION_PRINT_REPRO=1 emits an executable repro script."""
        original = os.environ.get("HELION_PRINT_REPRO")
        os.environ["HELION_PRINT_REPRO"] = "1"
        try:

            @helion.kernel(
                config=helion.Config(
                    block_sizes=[2, 2],
                    flatten_loops=[False],
                    indexing=["pointer", "pointer"],
                    l2_groupings=[1],
                    load_eviction_policies=[""],
                    loop_orders=[[0, 1]],
                    num_stages=1,
                    num_warps=4,
                    pid_type="flat",
                    range_flattens=[None],
                    range_multi_buffers=[None],
                    range_num_stages=[0],
                    range_unroll_factors=[0],
                ),
                static_shapes=True,
            )
            def kernel1(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + 1
                return out

            torch.manual_seed(0)
            x = torch.randn([2, 2], dtype=torch.float32, device=DEVICE)

            if hasattr(self, "_capfd"):
                self._capfd.readouterr()

            result = kernel1(x)
            torch.testing.assert_close(result, x + 1)

            if not hasattr(self, "_capfd"):
                return  # Cannot test without capture

            captured = "".join(self._capfd.readouterr())

            # Extract repro script
            lines = captured.splitlines()
            start = next(
                i
                for i, line in enumerate(lines)
                if "# === HELION KERNEL REPRO ===" in line
            )
            end = next(
                i
                for i, line in enumerate(lines[start:], start)
                if "# === END HELION KERNEL REPRO ===" in line
            )
            repro_script = "\n".join(lines[start : end + 1])

            # Normalize range_warp_specializes=[None] to [] for comparison
            normalized_script = repro_script.replace(
                "range_warp_specializes=[None]", "range_warp_specializes=[]"
            )

            # Verify repro script matches expected script
            self.assertExpectedJournal(normalized_script)

            # Extract the actual code (without the comment markers) for execution
            repro_lines = repro_script.splitlines()
            code_start = 1 if repro_lines[0].startswith("# === HELION") else 0
            code_end = len(repro_lines) - (
                1 if repro_lines[-1].startswith("# === END") else 0
            )
            repro_code = "\n".join(repro_lines[code_start:code_end])

            # Setup linecache so inspect.getsource() works on exec'd code
            filename = "<helion_repro_test>"
            linecache.cache[filename] = (
                len(repro_code),
                None,
                [f"{line}\n" for line in repro_code.splitlines()],
                filename,
            )

            # Execute the repro script
            namespace = {}
            exec(compile(repro_code, filename, "exec"), namespace)

            # Call the generated helper and verify it runs successfully
            helper = namespace["helion_repro_caller"]
            repro_result = helper()

            # Verify the output
            torch.testing.assert_close(repro_result, x + 1)

            linecache.cache.pop(filename, None)
        finally:
            if original is None:
                os.environ.pop("HELION_PRINT_REPRO", None)
            else:
                os.environ["HELION_PRINT_REPRO"] = original


if __name__ == "__main__":
    unittest.main()
