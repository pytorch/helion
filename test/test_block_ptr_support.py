from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._compiler.indexing_strategy import BlockPtrIndexingStrategy
from helion._compiler.indexing_strategy import IndexingStrategy
from helion._compiler.indexing_strategy import PointerIndexingStrategy
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.language as hl


@helion.kernel(static_shapes=True)
def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _removed_make_block_ptr(
    base: object,
    shape: object,
    strides: object,
    offsets: object,
    block_shape: object,
    order: object,
    _semantic: object = None,
) -> object:
    # The Triton >= 3.9 stub (triton-lang/triton#10833).
    raise NotImplementedError(
        "Block pointers have been removed in favor of the tensor descriptor API"
    )


def _present_make_block_ptr(
    base: object,
    shape: object,
    strides: object,
    offsets: object,
    block_shape: object,
    order: object,
    _semantic: object = None,
) -> object:
    return _semantic.make_block_ptr(  # type: ignore[attr-defined]
        base, shape, strides, offsets, block_shape, order
    )


def _present_advance(base: object, offsets: object, _semantic: object = None) -> object:
    return _semantic.advance(base, offsets)  # type: ignore[attr-defined]


@unittest.skipUnless(_compat.triton_is_available(), "requires triton")
class TestBlockPtrProbe(unittest.TestCase):
    def _probe(self, make_block_ptr: object, advance: object) -> bool:
        import triton.language as tl

        with (
            patch.object(tl, "make_block_ptr", make_block_ptr, create=True),
            patch.object(tl, "advance", advance, create=True),
        ):
            _compat._supports_block_ptr.cache_clear()
            try:
                return _compat._supports_block_ptr()
            finally:
                _compat._supports_block_ptr.cache_clear()

    def test_probe_reads_the_builtin_body(self) -> None:
        self.assertTrue(self._probe(_present_make_block_ptr, _present_advance))
        self.assertFalse(self._probe(_removed_make_block_ptr, _present_advance))

    def test_probe_without_the_builtins(self) -> None:
        # Triton 3.9 nightlies drop tl.advance outright; a build without
        # tl.make_block_ptr at all is unsupported as well.
        self.assertFalse(self._probe(_present_make_block_ptr, None))
        self.assertFalse(self._probe(None, _present_advance))


@onlyBackends(["triton"])
@skipIfTileIR("TileIR never offers block_ptr indexing")
@skipIfRefEager("checks generated Triton code")
class TestBlockPtrFallback(TestCase):
    def test_select_follows_the_probe(self) -> None:
        with patch.object(_compat, "_supports_block_ptr", lambda: True):
            self.assertIsInstance(
                IndexingStrategy.select("block_ptr"), BlockPtrIndexingStrategy
            )
        with patch.object(_compat, "_supports_block_ptr", lambda: False):
            self.assertIsInstance(
                IndexingStrategy.select("block_ptr"), PointerIndexingStrategy
            )

    def test_block_ptr_config_lowers_as_pointer_without_support(self) -> None:
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        with patch.object(_compat, "_supports_block_ptr", lambda: False):
            code, out = code_and_output(
                _add, (x, y), block_sizes=[32, 32], indexing="block_ptr"
            )
        self.assertNotIn("make_block_ptr", code)
        self.assertIn("tl.load(", code)
        torch.testing.assert_close(out, x + y)

    def test_search_space_omits_block_ptr_without_support(self) -> None:
        # The indexing fragment is built at bind time, so bind a fresh shape
        # under each patch and look at the choices the autotuner samples.
        with patch.object(_compat, "_supports_tensor_descriptor", lambda: False):
            with patch.object(_compat, "_supports_block_ptr", lambda: True):
                x = torch.randn(48, 48, device=DEVICE)
                spec = _add.bind((x, x)).config_spec
                self.assertEqual(spec.valid_indexing_types(), ("pointer", "block_ptr"))
                self.assertEqual(spec.indexing.inner.choices, ("pointer", "block_ptr"))
            with patch.object(_compat, "_supports_block_ptr", lambda: False):
                x = torch.randn(80, 80, device=DEVICE)
                spec = _add.bind((x, x)).config_spec
                self.assertEqual(spec.valid_indexing_types(), ("pointer",))
                self.assertEqual(spec.indexing.inner.choices, ("pointer",))


if __name__ == "__main__":
    unittest.main()
