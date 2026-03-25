from __future__ import annotations

import unittest

import sympy
import torch

from helion._compiler.compile_environment import BlockSizeInfo
from helion._compiler.compile_environment import _to_sympy
from helion.autotuner.config_spec import BlockSizeSpec


def _make_info(block_id: int, size: int, sym: torch.SymInt) -> BlockSizeInfo:
    return BlockSizeInfo(
        block_id=block_id,
        size=size,
        var=sym,
        reduction=False,
        block_size_source=BlockSizeSpec(block_id=block_id, size_hint=size),
    )


class TestSizeMatches(unittest.TestCase):
    """Tests for BlockSizeInfo.size_matches (concrete numel comparison)."""

    def test_same_concrete(self) -> None:
        sym = torch.SymInt(torch._C._get_tracing_state is not None or 42)
        info = _make_info(0, 128, sym)
        self.assertTrue(info.size_matches(sympy.Integer(128)))

    def test_different_concrete(self) -> None:
        sym = torch.SymInt(torch._C._get_tracing_state is not None or 42)
        info = _make_info(0, 128, sym)
        self.assertFalse(info.size_matches(sympy.Integer(256)))

    def test_symbol_returns_false(self) -> None:
        sym = torch.SymInt(torch._C._get_tracing_state is not None or 42)
        info = _make_info(0, 128, sym)
        self.assertFalse(info.size_matches(sympy.Symbol("u0")))

    def test_none(self) -> None:
        sym = torch.SymInt(torch._C._get_tracing_state is not None or 42)
        info = _make_info(0, 128, sym)
        self.assertFalse(info.size_matches(None))


class TestDimMatches(unittest.TestCase):
    """Tests for BlockSizeInfo.dim_matches (symbolic dimension comparison)."""

    def test_same_symbol(self) -> None:
        shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
        sym = shape_env.create_unbacked_symint()
        info = _make_info(0, 128, sym)
        self.assertTrue(info.dim_matches(_to_sympy(sym)))

    def test_different_symbol(self) -> None:
        shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
        sym1 = shape_env.create_unbacked_symint()
        sym2 = shape_env.create_unbacked_symint()
        info = _make_info(0, 128, sym1)
        self.assertFalse(info.dim_matches(_to_sympy(sym2)))

    def test_concrete_returns_false(self) -> None:
        shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
        sym = shape_env.create_unbacked_symint()
        info = _make_info(0, 128, sym)
        self.assertFalse(info.dim_matches(sympy.Integer(128)))

    def test_none(self) -> None:
        shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
        sym = shape_env.create_unbacked_symint()
        info = _make_info(0, 128, sym)
        self.assertFalse(info.dim_matches(None))


if __name__ == "__main__":
    unittest.main()
