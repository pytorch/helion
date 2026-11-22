"""The c10 preamble (``c10_preamble``) on the GPU: kernels give the same
bits with the c10 declarations they use as with the whole headers, and a
pruned preamble that doesn't compile gives way to the headers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest import mock

import pytest
import torch

from helion._compiler.metal import c10_preamble
from helion._testing import skipUnlessBackends
from helion._testing.metal import run_with_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_preamble import bfloat_mul_add
from helion._testing.metal_preamble import erf_of
from helion._testing.metal_preamble import row_sums

if TYPE_CHECKING:
    import helion

DEVICE = "mps"

# Metal kernels on macOS only: other backends' suites skip the module.
pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests require macOS"),
    skipUnlessBackends(["metal"]),
]
_requires_mps = unittest.skipUnless(
    sys.platform == "darwin" and torch.backends.mps.is_available(),
    "requires a Metal GPU",
)


def _run(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    config: helion.Config | None,
    key: str,
    whole_c10: bool = False,
) -> tuple[torch.Tensor, str]:
    config = config or kernel.bind(args).config_spec.default_config()
    out, _, msl = run_with_msl(
        kernel, args, config, module_key=f"c10:{key}", whole_c10=whole_c10
    )
    return cast("torch.Tensor", out), msl


@_requires_mps
class TestC10PreambleMps(unittest.TestCase):
    def test_same_bits(self) -> None:
        """A reduction, a special function, bfloat math and an affine
        matmul on fragments give bit-identical results with the pruned
        preamble and with the whole c10 headers."""
        x = torch.randn(64, 64, device=DEVICE)
        for kernel, args, config in (
            (row_sums, (torch.randn(32, 128, device=DEVICE),), None),
            (erf_of, (x,), None),
            (bfloat_mul_add, (x.bfloat16(), (x * 0.5).bfloat16()), None),
            (
                affine_mm,
                affine_args(4, 32, 8, 64, 512, DEVICE),
                fragment_config([16, 32, 64]),
            ),
        ):
            with self.subTest(kernel=kernel.name):
                pruned, msl = _run(kernel, args, config, "pruned")
                full, full_msl = _run(kernel, args, config, "full", whole_c10=True)
                self.assertNotIn("#include <c10/", msl)
                self.assertIn("#include <c10/", full_msl)
                self.assertTrue(torch.equal(pruned, full))

    def test_fallback(self) -> None:
        """A pruned preamble that doesn't compile (here, none at all) gives
        way to the whole headers."""
        x = torch.randn(64, 64, device=DEVICE)
        with (
            mock.patch.object(
                c10_preamble,
                "prune",
                lambda msl: c10_preamble._C10_INCLUDE.sub("", msl),
            ),
            self.assertLogs("helion._compiler.metal.metal_jit", "WARNING") as logs,
        ):
            out, msl = _run(erf_of, (x,), None, "fallback")
        self.assertIn("#include <c10/", msl)
        # The warning names the compiler's error, not a warning before it.
        self.assertIn("error:", logs.output[0])
        torch.testing.assert_close(out, torch.erf(x))
