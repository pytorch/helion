from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(static_shapes=True, autotune_effort="none")
def checked_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(tile.begin < x.size(0), "tile past the end")
        hl.device_assert(tile.index < x.size(0), "lane past the end")
        hl.device_assert(tile.index + 1, "zero")
        v = x[tile, :]
        hl.device_assert(v != 0, "zero element")
        s = v.sum(-1)
        hl.device_assert(s > 0, "row sum")
        out[tile] = s
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def checked_columns(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        cols = hl.arange(100)
        v = hl.load(x, [tile, cols])
        hl.device_assert(v != 0, "zero element")
        hl.store(out, [tile, cols], v * 2)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def checked_range(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(hl.arange(100) < 100, "range past its length")
        out[tile] = x[tile]
    return out


@helion.kernel(
    static_shapes=True, config=helion.Config(block_sizes=[1], reduction_loops=[256])
)
def checked_shift(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile, :]
        hl.device_assert(v > 0, "nonpositive")
        out[tile, :] = v - v.amax(-1, keepdim=True)
    return out


@onlyBackends(["triton", "cute"])
class TestDeviceAssert(RefEagerTestDisabled, TestCase):
    def test_true_conditions(self):
        # Every condition holds on the elements (tile.begin < 1000 is one per
        # tile; tile.index + 1 is an int), and fails on the lanes past them:
        # the rows past 1000 of the last tile (tile.index, a masked load's 0,
        # the sum of no element) and the 28 lanes past the 100 elements of
        # hl.arange(100), which runs over 128, in the dim of a block of 128
        # (checked_columns; test_padded_range the dim of none).
        x = torch.rand(1000, 100, device=DEVICE) + 0.5
        torch.testing.assert_close(checked_rows(x), x.sum(-1))
        torch.testing.assert_close(checked_columns(x), x * 2)
        # A reduction loop over 4000 columns in chunks of 256, the last partial.
        y = torch.rand(64, 4000, device=DEVICE) + 0.5
        torch.testing.assert_close(checked_shift(y), y - y.amax(-1, keepdim=True))
        torch.accelerator.synchronize()

    @skipUnlessBackends(["triton"])  # CuTe's hl.arange needs a tile.
    def test_padded_range(self):
        # hl.arange(100) in the dim of no block: a 1-D x (a 2-D one gives the
        # range the block of its 100 columns).
        x = torch.rand(1000, device=DEVICE) + 0.5
        torch.testing.assert_close(checked_range(x), x)
        torch.accelerator.synchronize()

    @pytest.mark.slow
    @skipIfNotCUDA()
    def test_false_condition(self):
        # A false condition stops the kernel, which leaves the CUDA context
        # unusable, so it runs in its own process: in a gather, and in the
        # first chunk of a reduction loop.  HELION_DEVICE_ASSERTS=0 compiles
        # the check out.
        script = textwrap.dedent(
            """
            import sys
            import torch
            import helion
            import helion.language as hl

            @helion.kernel(static_shapes=True, autotune_effort="none")
            def gather(x, idx):
                out = torch.empty([idx.size(0)], dtype=x.dtype, device=x.device)
                for tile in hl.tile(idx.size(0)):
                    i = idx[tile]
                    hl.device_assert(i < x.size(0), "index out of bounds")
                    out[tile] = hl.load(x, [i % x.size(0)])
                return out

            @helion.kernel(
                static_shapes=True,
                config=helion.Config(block_sizes=[1], reduction_loops=[256]),
            )
            def row_sums(x):
                out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
                for tile in hl.tile(x.size(0)):
                    v = x[tile, :]
                    hl.device_assert(v > 0, "nonpositive element")
                    out[tile] = v.sum(-1)
                return out

            if sys.argv[1] == "gather":
                idx = torch.arange(777, device="cuda")
                idx[5] = 1000
                gather(torch.randn(1000, device="cuda"), idx)
            else:
                x = torch.rand(64, 4000, device="cuda") + 0.5
                x[3, 5] = -1.0
                row_sums(x)
            torch.cuda.synchronize()
            print("no error")
            """
        )
        failure = {
            "triton": "device-side assert triggered",
            "cute": "unspecified launch failure",
        }[helion.Settings().backend]
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "gather.py"
        # Helion reads a kernel's source from its file.
        path.write_text(script)
        for case, message, enabled in (
            ("gather", "index out of bounds", "1"),
            ("gather", "index out of bounds", "0"),
            ("rolled", "nonpositive element", "1"),
        ):
            result = subprocess.run(
                [sys.executable, str(path), case],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HELION_DEVICE_ASSERTS": enabled,
                    "PYTHONPATH": os.pathsep.join(
                        filter(
                            None,
                            [
                                str(Path(helion.__file__).parents[1]),
                                os.environ.get("PYTHONPATH"),
                            ],
                        )
                    ),
                },
                timeout=600,
            )
            output = result.stdout + result.stderr
            if enabled == "1":
                self.assertNotEqual(result.returncode, 0, output)
                self.assertIn(message, output)
                self.assertIn(failure, output)
            else:
                self.assertEqual(result.returncode, 0, output)
                self.assertIn("no error", output)


class TestDeviceAssertRefMode(TestCase):
    def test_ref_mode(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def positive(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                hl.device_assert(x[tile] > 0, "nonpositive element")
                out[tile] = x[tile]
            return out

        x = torch.rand(100, device=DEVICE) + 0.5
        torch.testing.assert_close(positive(x), x)
        x[7] = -1.0
        with self.assertRaisesRegex(AssertionError, "nonpositive element"):
            positive(x)

    def test_ref_mode_compiled_out(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER, device_asserts=False)
        def unchecked(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                hl.device_assert(x[tile] > 0, "nonpositive element")
                out[tile] = x[tile]
            return out

        x = -torch.rand(100, device=DEVICE)
        torch.testing.assert_close(unchecked(x), x)


if __name__ == "__main__":
    unittest.main()
