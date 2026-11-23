"""The few-row affine matmul's code on fragments (``fragment_mma``): the
widened activation read as a view, which transposes the accumulator, and
the MMAs fragment by fragment.  Numerics are
``test/test_metal_fragment_mma.py``'s."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._testing import TestCase
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_fragment import fragment_transposed
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


def _msl(
    bits: int,
    group: int,
    rows: int,
    k: int,
    block_sizes: list[int],
    x_dtype: torch.dtype = torch.float16,
    **knobs: object,
) -> str:
    x, w, scales, biases = affine_args(bits, group, rows, 64, k)
    knobs.setdefault("num_warps", 8)
    return render_msl(
        affine_mm,
        (x.to(x_dtype), w, scales, biases),
        fragment_config(block_sizes, [1, 1, 32], **knobs),
    )


@helion.kernel(backend="metal", static_shapes=True)
def dense_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ y with both operands widened from 16 bits."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                x[tile_m, tile_k].to(torch.float32),
                y[tile_k, tile_n].to(torch.float32),
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def g_idx_mm(
    x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor, g_idx: torch.Tensor
) -> torch.Tensor:
    """x @ W.T for 4-bit codes in int16 words whose group each K position
    reads from ``g_idx`` (GPTQ's act-order): the weight reads its scale at
    an index loaded from memory."""
    m, k = x.size()
    n, _ = w.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            bit = tile_k.index * 4
            q = (w[tile_n, bit // 16].to(torch.int32) >> (bit % 16)[None, :]) & 15
            scale = scales[tile_n, g_idx[tile_k]].to(torch.float32)
            acc = torch.addmm(
                acc,
                x[tile_m, tile_k].to(torch.float32),
                (q.to(torch.float32) * scale).T,
            )
        out[tile_m, tile_n] = acc
    return out


def _mma_order(msl: str) -> list[tuple[int, str]]:
    """Each MMA's accumulator fragment and B fragment, in emission order."""
    return [
        (int(acc), b)
        for acc, b in re.findall(
            r"simdgroup_multiply_accumulate\(acc_copy_\d+\[(\d+)\], _frag_a\[\d+\], "
            r"(_frag_b\d*\[\d+\])",
            msl,
        )
    ]


class TestFewRowAffine(TestCase):
    def test_widened_view(self) -> None:
        """x.to(float32) at 16 rows or fewer is read as x's half or bfloat
        view when the weight reads one word per code, so only the weight is
        computed and the accumulator is transposed: the weight is the MMA's
        A, each lane's elements one run of a row's K positions.  At 32 rows,
        or beside 5- or 6-bit codes, which read two words, x stays computed
        and the accumulator isn't transposed."""
        for rows in (1, 8, 16):
            with self.subTest(rows=rows):
                msl = _msl(4, 64, rows, 1024, [16, 32, 128])
                self.assertTrue(fragment_transposed(msl))
                self.assertNotIn("_mpp_fv_1", msl)
        msl = _msl(4, 64, 8, 1024, [16, 32, 128], x_dtype=torch.bfloat16)
        self.assertTrue(fragment_transposed(msl))
        self.assertIn("device bfloat* x", msl)
        self.assertTrue(fragment_transposed(_msl(8, 64, 8, 1024, [16, 32, 128])))
        self.assertFalse(fragment_transposed(_msl(4, 64, 32, 1024, [32, 32, 64])))
        for bits in (5, 6):
            with self.subTest(bits=bits):
                msl = _msl(bits, 128, 8, 1024, [16, 32, 128])
                self.assertFalse(fragment_transposed(msl))

    def test_fragment_order(self) -> None:
        """The few-row run builds every step's B fragment (x's) first, then
        per A fragment of the weight its elements and MMA at each step in
        turn; each accumulator still adds its steps in order."""
        msl = _msl(4, 64, 8, 1024, [16, 32, 128])
        self.assertEqual(
            _mma_order(msl),
            [(a, f"_frag_b{s}[0]") for a in range(4) for s in range(16)],
        )
        b_fragments = msl.index("simdgroup_float8x8 _frag_b15[1];")
        self.assertLess(b_fragments, msl.index("_frag_a[0], _frag_b0[0]"))

    def test_dense_weight(self) -> None:
        """A dense weight beside the few-row activation, widened from 16 bits
        or from int8 codes it doesn't unpack, or one read at an index loaded
        from memory, keeps the group untransposed: the few-row form was
        measured on codes unpacked from words."""
        x = torch.randn(8, 1024).to(torch.float16)
        config = fragment_config([16, 32, 128], [1, 1, 32], num_warps=8)
        for y in (
            torch.randn(1024, 64).to(torch.float16),
            torch.randint(-128, 128, (1024, 64), dtype=torch.int8),
        ):
            with self.subTest(dtype=y.dtype):
                msl = render_msl(dense_mm, (x, y), config)
                self.assertFalse(fragment_transposed(msl))
                self.assertNotIn("_frag_b15[", msl)
        # Nor does a weight read at an index loaded from memory: codes
        # whose groups come from a g_idx table.
        w = torch.randint(-(2**15), 2**15, (64, 256), dtype=torch.int16)
        scales = torch.randn(64, 16).to(torch.float16)
        g_idx = torch.randint(0, 16, (1024,), dtype=torch.int32)
        msl = render_msl(g_idx_mm, (x, w, scales, g_idx), config)
        self.assertFalse(fragment_transposed(msl))
        self.assertNotIn("_frag_b15[", msl)

    def test_step_order_elsewhere(self) -> None:
        """The planner keeps x computed, and the run its step-by-step order,
        where the K loop ends in a partial tile or loads a tile ahead
        (range_num_stages=2).  A transposed group whose activation is
        already a float32 view keeps the step order too."""
        cases = [
            ((4, 64, 8, 1024 + 64, [16, 32, 128]), {}),
            ((4, 64, 8, 1024, [16, 32, 128]), {"range_num_stages": [2]}),
        ]
        for args, knobs in cases:
            with self.subTest(args=args, knobs=knobs):
                msl = _msl(*args, **knobs)
                self.assertFalse(fragment_transposed(msl))
                self.assertIn("simdgroup_float8x8 _frag_b[", msl)
        msl = _msl(4, 64, 8, 1024, [16, 32, 128], x_dtype=torch.float32)
        self.assertTrue(fragment_transposed(msl))
        self.assertEqual(
            _mma_order(msl), [(a, "_frag_b[0]") for s in range(16) for a in range(4)]
        )
