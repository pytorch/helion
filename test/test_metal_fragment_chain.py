"""Fragment operands built from another fragment group's result
(``metal_matmul="fragment"``, ``fragment_mma._chain_refusal``): attention's
P from its scores' fragments, in place, with the consumer taking K in its
natural order, and element values two consumers read computed once into
registers (``fragment_lines.plan_holds``).  Renders on CPU tensors
(``helion._testing.metal.render_msl``), with no Apple GPU;
``test_metal_fragment_mma.py`` runs the kernels."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._compiler.metal.search_surface import matmul_choices
from helion._testing import TestCase
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import chained_rhs
from helion._testing.metal_fragment import fragment_config
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

_attention = import_example("attention")


def _metal(kernel: helion.Kernel) -> helion.Kernel:
    return helion.kernel(kernel.fn, backend="metal", static_shapes=True, fast_math=True)


attention = _metal(_attention.attention)


@helion.kernel(backend="metal", static_shapes=True)
def int_epilogue(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The accumulator converted to int32, read by two element ops (``q &
    7`` and ``q >> 2``): an int element value computed once into
    registers."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        q = acc.to(torch.int32)
        out[tile_m, tile_n] = ((q & 7) + (q >> 2)).to(torch.float32)
    return out


def _qkv(s: int = 64) -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn(2, 2, s, 128, dtype=torch.bfloat16) for _ in range(3))


class TestFragmentChain(TestCase):
    def test_attention(self) -> None:
        """``examples/attention.py`` lowers on fragments: P's A fragments of
        each K step are its scores' fragments of that step's columns, in
        natural K order, held in bfloat; the exponential, read by the row
        sum and by P, runs once per element; the row max, sum, alpha and
        running values stay in registers; nothing reaches threadgroup
        memory.  The search offers fragments for it.  (``causal_attention``
        reads ``tile.index`` through a load, which no Metal lowering
        takes yet.)"""
        args = _qkv()
        msl = render_msl(attention, args, fragment_config([1, 16, 16]))
        self.assertNotIn("threadgroup_barrier", msl)
        self.assertNotRegex(msl, r"\bthreadgroup \w+ ")
        self.assertIn("int _e_k = 8 * _fs + _fn + (_fi & 1);", msl)
        self.assertIn("simdgroup_matrix<bfloat, 8, 8> _frag_a[2];", msl)
        self.assertRegex(
            msl, r"_held_\d+\[2 \* \(\(_fi >> 1\) \* 2 \+ _fs\) \+ \(_fi & 1\)\]"
        )
        # One exponential per score element (2 x 2 fragments of a 16 x 16
        # tile, 8 per lane) and one per row value (2 rows).
        self.assertEqual(msl.count("metal::fast::exp("), 8 + 2)
        self.assertIn("simd_shuffle_xor(_lr, 8)", msl)
        self.assertIn("fragment", matmul_choices(attention.bind(args).config_spec))

    def test_chained_rhs(self) -> None:
        """A chained rhs, the consumer's K the scores' M: B fragment ``j`` of
        step ``s`` is the scores' fragment at row ``s``, column ``j``; float
        operands keep float fragments."""
        args = (torch.randn(32, 16), torch.randn(16, 24), torch.randn(16, 32))
        msl = render_msl(chained_rhs, args, fragment_config([16, 16]))
        self.assertIn("int _e_k = 8 * _fs + _fm;", msl)
        self.assertRegex(
            msl, r"_mpp_c_0\[\(_fs \* 2 \+ \(_fi >> 1\)\)\]\.thread_elements\(\)"
        )
        self.assertIn("simdgroup_float8x8 _frag_b[2];", msl)
        self.assertNotIn("threadgroup_barrier", msl)
        self.assertIsNone(re.search(r"_held_", msl))

    def test_held_dtype(self) -> None:
        """A held element value's registers take its dtype: an int32
        conversion of the accumulator, which the two element ops shift and
        mask, is held in ``int`` registers."""
        args = (torch.randn(16, 32), torch.randn(32, 16))
        msl = render_msl(int_epilogue, args, fragment_config([16, 16, 16]))
        (held,) = re.findall(r"\b(\w+) (_held_\d+)\[8\];", msl)
        self.assertEqual(held[0], "int")
        self.assertRegex(msl, rf"\({held[1]}\[_fi\] >> v_\d+\)")
