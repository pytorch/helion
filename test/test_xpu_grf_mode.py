from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import is_xpu
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner import EnumFragment
import helion.language as hl
from helion.runtime.triton.launcher import get_xpu_grf_modes


@helion.kernel(autotune_effort="none")
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] + y[tile]
    return out


@onlyBackends(["triton"])
class TestXPUGRFMode(TestCase):
    @unittest.skipUnless(is_xpu(), "grf_mode is XPU only")
    def test_tunable_fragment(self) -> None:
        x = torch.randn(64, 64, device=DEVICE)
        fragment = add.bind((x, x)).config_spec.backend_tunable_fragments["grf_mode"]
        self.assertEqual(fragment, EnumFragment((0, *get_xpu_grf_modes(DEVICE))))
        self.assertEqual(fragment.default(), 0)

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "grf_mode is XPU only")
    def test_launch_argument(self) -> None:
        x = torch.randn(64, 64, device=DEVICE)
        bound = add.bind((x, x))
        code = bound.to_triton_code(helion.Config(block_sizes=[32, 32]))
        self.assertNotIn("grf_mode=", code)
        for grf_mode in get_xpu_grf_modes(DEVICE):
            code = bound.to_triton_code(
                helion.Config(block_sizes=[32, 32], grf_mode=grf_mode)
            )
            self.assertIn(f"grf_mode='{grf_mode}'", code)

    @unittest.skipUnless(is_xpu(), "grf_mode is XPU only")
    def test_results(self) -> None:
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        for grf_mode in (0, *get_xpu_grf_modes(DEVICE)):
            kernel = helion.kernel(
                add.fn, config=helion.Config(block_sizes=[32, 32], grf_mode=grf_mode)
            )
            _, result = code_and_output(kernel, (x, y))
            torch.testing.assert_close(result, x + y)

    def test_unsupported_without_xpu(self) -> None:
        with (
            patch("helion._compat.is_xpu", return_value=False),
            patch("helion._compat.is_hip", return_value=False),
        ):
            env = CompileEnvironment(DEVICE, helion.Settings(backend="triton"))
            self.assertNotIn("grf_mode", env.config_spec.backend_tunable_fragments)
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                rf"Unsupported config keys for backend '{env.backend_name}': \['grf_mode'\]",
            ):
                env.config_spec.normalize(helion.Config(grf_mode=256))


if __name__ == "__main__":
    unittest.main()
