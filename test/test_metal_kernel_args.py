"""How a Metal kernel's host values reach its MSL: a 0-dim tensor binds as
a buffer its loads index, and a baked float is a literal MSL compiles.  Binds
CPU tensors only; ``test_metal.py`` runs a 0-dim tensor's kernel."""

from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._compiler.metal.metal_jit import _constexpr_decl
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal import scale_by_0dim

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


class TestKernelArgs(unittest.TestCase):
    def test_0dim_tensor_binds_as_a_buffer(self) -> None:
        args = (torch.randn(64), torch.tensor(0.5))
        msl = render_msl(scale_by_0dim, args, helion.Config(block_sizes=[64]))
        self.assertIn("device float* s [[buffer(1)]]", msl)
        self.assertIn("s[0]", msl)

    def test_baked_float_is_a_literal(self) -> None:
        """A baked float compiles whatever its value: MSL spells infinities
        and NaN with macros, not as ``repr`` writes them."""
        for value, literal in (
            (0.5, "0.5"),
            (float("inf"), "INFINITY"),
            (float("-inf"), "-INFINITY"),
            (float("nan"), "NAN"),
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    _constexpr_decl("f", value), f"constexpr float f = {literal};"
                )


if __name__ == "__main__":
    unittest.main()
