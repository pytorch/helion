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
from helion import exc
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfPallasTpu
from helion._testing import skipIfTileIR
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl
from helion.runtime.settings import _get_backend


@helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[64]))
def checked_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(tile.index < x.size(0), "lane past the end")
        v = x[tile, :]
        hl.device_assert(v != 0, "zero element")
        s = v.sum(-1)
        hl.device_assert(s > 0, "row sum")
        out[tile] = s
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def launch_option_names(
    debug: torch.Tensor, sanitize_overflow: torch.Tensor
) -> torch.Tensor:
    hl.register_tunable("_triton_config_debug", EnumFragment(choices=(False,)))
    out = torch.empty_like(debug)
    for tile in hl.tile(debug.size(0)):
        hl.device_assert(debug[tile] >= 0, "negative")
        out[tile] = debug[tile] + sanitize_overflow[tile]
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def int_condition(x: torch.Tensor, scalar: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        if scalar:
            hl.device_assert(tile.begin + 1, "zero")
        else:
            hl.device_assert(tile.index + 1, "zero")
        out[tile] = x[tile]
    return out


@onlyBackends(["triton", "cute"])
@skipIfTileIR("TileIR raises BackendImplementationMissing (TestDeviceAssertTileIR)")
class TestDeviceAssert(RefEagerTestDisabled, TestCase):
    def test_true_conditions(self):
        # Each condition holds on the 1000 rows and fails on the lanes past
        # them in the last tile of 64 (the index, the 0 a masked load gives,
        # the sum of no element) and past the 100 columns in 128 lanes.
        x = torch.rand(1000, 100, device=DEVICE) + 0.5
        code, out = code_and_output(checked_rows, (x,))
        torch.testing.assert_close(out, x.sum(-1))
        torch.accelerator.synchronize()
        if _get_backend() == "triton":
            call = r"tl\.device_assert"
            options = "debug=True, sanitize_overflow=False"
        else:
            call = r"cute\.testing\.assert_"
            options = "--enable-assertions"
        for message in ("lane past the end", "zero element", "row sum"):
            self.assertRegex(code, rf"{call}\(_mask_to\w*, '{message}'\)")
        self.assertIn(options, code)

    def test_launch_option_names(self):
        # Arguments named like Triton's debug=True, sanitize_overflow=False are
        # renamed in the device code, which the launch binds by name, and a
        # _triton_config_debug tunable gives way to them.
        x, y = torch.rand(100, device=DEVICE), torch.rand(100, device=DEVICE)
        torch.testing.assert_close(launch_option_names(x, y), x + y)

    def test_bool_condition(self):
        for scalar in (False, True):
            with self.assertRaises(exc.TypeInferenceError):
                int_condition.bind((torch.rand(100, device=DEVICE), scalar))

    @pytest.mark.slow
    @skipIfNotCUDA()
    def test_false_condition(self):
        # A false condition leaves the CUDA context unusable, so each case
        # runs in its own process: an index out of bounds, and an element in
        # the first chunk of a reduction loop (a loop would check only the
        # last chunk; the effect keeps the reduction persistent).
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
            """
        )
        failure = {
            "triton": "device-side assert triggered",
            "cute": "unspecified launch failure",
        }[_get_backend()]
        # Helion reads a kernel's source from its file.
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "checks.py"
        path.write_text(script)
        pythonpath = os.pathsep.join(
            filter(
                None,
                [str(Path(helion.__file__).parents[1]), os.environ.get("PYTHONPATH")],
            )
        )
        for case, message in (
            ("gather", "index out of bounds"),
            ("rows", "nonpositive element"),
        ):
            result = subprocess.run(
                [sys.executable, str(path), case],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": pythonpath},
                timeout=600,
            )
            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn(message, output)
            self.assertIn(failure, output)


@onlyBackends(["tileir"])
class TestDeviceAssertTileIR(TestCase):
    def test_unsupported(self):
        with self.assertRaises(exc.BackendImplementationMissing):
            checked_rows(torch.rand(1000, 100, device=DEVICE))


@skipIfPallasTpu("ref mode reads TorchTPU's internal get_rng_state")
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


if __name__ == "__main__":
    unittest.main()
