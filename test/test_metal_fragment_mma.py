"""K-loop matmuls lowered on 8x8 simdgroup-matrix fragments built in
registers (``metal_matmul="fragment"``, ``metal/fragment_mma.py``).

Each kernel runs once per configuration; its values are checked against
float64 references on the CPU and its MSL for the fragment form: the op is
``simdgroup_multiply_accumulate`` on operand elements each lane computes,
with no ``matmul2d`` and no threadgroup tile.
"""

from __future__ import annotations

import functools
import math
import re
import sys
from typing import TYPE_CHECKING
from typing import cast
import unittest

import pytest
import torch

import helion
from helion._compiler.metal.fragment_matmul_heuristic import (
    MetalFragmentMatmulHeuristic,
)
from helion._testing import pairwise
from helion._testing import skipUnlessBackends
from helion._testing.metal import heuristic_seeds
from helion._testing.metal import run_with_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_fragment import affine_reference
from helion._testing.metal_fragment import e4m3
from helion._testing.metal_fragment import filled
from helion._testing.metal_fragment import fp8_rows
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_fragment import fragment_transposed
from helion._testing.metal_fragment import k_from_16
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import seeded_rhs
from helion._testing.metal_fragment import two_groups
from helion._testing.metal_fragment import views_epilogue
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.finite_search import FiniteSearch
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.benchmark_provider import BenchmarkResult
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel

DEVICE = "mps"
#: The few-row affine cases.  The few-row form takes 4- or 8-bit codes over
#: whole K tiles, both at once, so its cases pair each of those bit widths
#: with every group (one, two or four per K tile of 128) and every row count
#: (1 and 4 pad x's 8-column fragment, 16 takes two); the cases that keep x
#: computed, 5- and 6-bit codes (which straddle words) and a K tail (masked
#: loads), pair their own groups with the row counts.
_FEW_ROWS = [
    *(
        (bits, group, rows, False)
        for bits, group, rows in pairwise(
            bits=(4, 8), group=(32, 64, 128), rows=(1, 4, 8, 16)
        )
    ),
    *((bits, 128, rows, False) for bits, rows in ((5, 1), (5, 16), (6, 4), (6, 8))),
    *((4, group, rows, True) for group, rows in ((32, 8), (64, 4))),
]
F32 = torch.float32
S = {"backend": "metal", "static_shapes": True}

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
    kernel: helion.Kernel, args: tuple[object, ...], config: helion.Config
) -> tuple[torch.Tensor, str]:
    """Compile and run ``kernel`` at ``config``; its output and its MSL."""
    out, _, msl = run_with_msl(kernel, args, config)
    return cast("torch.Tensor", out), msl


@helion.kernel(backend="metal", static_shapes=False)
def dequant_rows(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """x @ W.T for any number of rows of x, W 4-bit (element k is nibble
    k % 8 of word k // 8, scaled per 32 elements)."""
    m = x.size(0)
    k = hl.specialize(x.size(1))
    n = hl.specialize(wq.size(0))
    hl.specialize(x.stride())
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            words = wq[tile_n, tile_k.index // 8]
            nib = (words >> ((tile_k.index % 8) * 4)[None, :]) & 0xF
            w = (nib.to(F32) - 8.0) * s[tile_n, tile_k.index // 32]
            acc = torch.addmm(acc, x[tile_m, tile_k], w.T.to(x.dtype))
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def transposed_views(x: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    """hl.dot of views of x and of yt's transpose (no unit stride along N)."""
    m, k = x.size()
    n = yt.size(0)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], yt[tile_n, tile_k].T, acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def computed_both(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Both operands computed; the accumulator seeded from c, and scaled
    before each K step's op (an in-place update)."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = c[tile_m, tile_n].to(F32)
        for tile_k in hl.tile(k):
            acc = acc * 0.5
            a = (x[tile_m, tile_k].to(F32) * 2.0).to(x.dtype)
            b = (y[tile_k, tile_n].to(F32) + 1.0).to(x.dtype)
            acc = torch.addmm(acc, a, b)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def computed_lhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A computed lhs beside a view rhs: the accumulator is not transposed."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, torch.sigmoid(x[tile_m, tile_k]), y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def broadcast_lhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x's K dim of size 1, broadcast over y's K tiles."""
    m = x.size(0)
    k, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def dot_computed_lhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """hl.dot of a computed lhs and a view rhs."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = hl.dot(torch.sigmoid(x[tile_m, tile_k]), y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def whole_k_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """hl.dot over a ':' K, outside any K loop."""
    m, _ = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = hl.dot(x[tile_m, :], y[:, tile_n], out_dtype=F32)
    return out


@helion.kernel(**S)
def running_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """out[b] = x[0] @ y[0] + ... + x[b] @ y[b]: an accumulator an outer
    loop carries from one K loop to the next, stored after each."""
    nb, m, k = x.size()
    n = y.size(2)
    out = torch.empty([nb, m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for b in hl.grid(nb):
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[b, tile_m, tile_k], y[b, tile_k, tile_n])
            out[b, tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def two_k_loops(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x @ y + 1 and z @ w: two K loops one after the other, the first's
    accumulator starting from 1, each result stored."""
    m, k = x.size()
    n = y.size(1)
    k2 = z.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    out2 = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        a = hl.full([tile_m, tile_n], 1.0, dtype=F32)
        for tile_k in hl.tile(k):
            a = torch.addmm(a, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = a
        b = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k2):
            b = torch.addmm(b, z[tile_m, tile_k], w[tile_k, tile_n])
        out2[tile_m, tile_n] = b
    return out, out2


def _nibbles(wq: torch.Tensor, k: int, bits: int) -> torch.Tensor:
    """The codes of words ``wq`` as integers, code ``i`` of word ``i * bits //
    32`` at bit ``(i * bits) % 32``."""
    per = 32 // bits
    shifts = torch.arange(per, device=wq.device) * bits
    codes = (wq.long()[..., None] >> shifts) & ((1 << bits) - 1)
    return codes.reshape(wq.size(0), k)


def _mm_t(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w.T`` in float64 on the CPU."""
    return x.cpu().double() @ w.cpu().double().T


@functools.cache
def _quantized(
    fmt: str, m: int, n: int, k: int
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Arguments of the ``fmt`` kernel and its float64 reference, once per
    shape."""
    torch.manual_seed(0)
    x = torch.randn(m, k, device=DEVICE).to(torch.bfloat16)
    if fmt == "int4":
        wq = torch.randint(
            -(2**31), 2**31 - 1, (n, k // 8), device=DEVICE, dtype=torch.int32
        )
        s = torch.rand(n, k // 32, device=DEVICE) + 0.5
        w = (_nibbles(wq, k, 4) - 8.0) * s.repeat_interleave(32, 1)
        return (x, wq, s), _mm_t(x, w.to(torch.bfloat16))
    if fmt == "nvfp4":
        wq = torch.randint(
            -(2**31), 2**31 - 1, (n, k // 8), device=DEVICE, dtype=torch.int32
        )
        s = torch.randint(0, 0x60, (n, k // 16), device=DEVICE, dtype=torch.uint8)
        codes = _nibbles(wq, k, 4)
        mag = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=DEVICE)
        q = mag[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
        w = q * e4m3(s.to(torch.int32)).repeat_interleave(16, 1)
        return (x, wq, s), _mm_t(x, w.to(torch.bfloat16)).to(torch.bfloat16)
    assert fmt == "fp8"
    wq = torch.randint(
        -(2**31), 2**31 - 1, (n, k // 4), device=DEVICE, dtype=torch.int32
    )
    codes = _nibbles(wq, k, 8)
    # Finite E4M3 codes only: 0x7F and 0xFF are NaN.
    codes = torch.where((codes & 0x7F) == 0x7F, 0, codes)
    wq = (codes.reshape(n, k // 4, 4) << (torch.arange(4, device=DEVICE) * 8)).sum(-1)
    wq = wq.to(torch.int64).where(wq < 2**31, wq - 2**32).to(torch.int32)
    s = torch.randint(120, 130, (n, k // 32), device=DEVICE, dtype=torch.uint8)
    sc = torch.pow(2.0, s.float() - 127).repeat_interleave(32, 1)
    w = (e4m3(codes.to(torch.int32)) * sc).to(torch.bfloat16)
    return (x, wq, s), _mm_t(x, w)


@_requires_mps
class TestFragmentMma(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _assert_fragments(self, msl: str, transposed: bool | None = None) -> None:
        """The op is MMAs on register fragments: no matmul2d, no threadgroup
        memory, no barrier."""
        self.assertIn("simdgroup_multiply_accumulate(", msl)
        self.assertIn("[[thread_index_in_simdgroup]]", msl)
        self.assertNotIn("matmul2d", msl)
        self.assertNotIn("metal_tensor", msl)
        self.assertNotIn("threadgroup_barrier", msl)
        self.assertNotRegex(msl, r"\bthreadgroup \w+ ")
        if transposed is not None:
            self.assertEqual(fragment_transposed(msl), transposed)

    def _close(self, out: torch.Tensor, ref: torch.Tensor, tol: float = 1e-2) -> None:
        torch.testing.assert_close(out.cpu().double(), ref.double(), atol=tol, rtol=tol)

    def test_dequant_rows(self) -> None:
        """int4 weights decoded into the rhs, the computed operand, with M
        dynamic (rows past m masked by the stores) and K on lanes or on
        threads; the accumulator is transposed so the weights are the
        MMA's A."""
        for m, config in (
            (5, fragment_config([16, 16, 32], [1, 1, 32])),
            (11, fragment_config([16, 16, 64])),
            (8, fragment_config([16, 16, 128], [1, 1, 32])),
        ):
            with self.subTest(m=m, config=config):
                args, ref = _quantized("int4", m, 64, 256)
                out, msl = _run(dequant_rows, args, config)
                self._close(out, ref)
                self._assert_fragments(msl, transposed=True)
        # Bound to static shapes, its unmasked words and scales load a K tile
        # early under 2 stages; of its decode, ``(q - 8) * scale``, the code's
        # conversion is a 16-entry lookup and the scale per group stays
        # computed.
        static = helion.kernel(dequant_rows.fn, backend="metal", static_shapes=True)
        args, ref = _quantized("int4", 8, 64, 256)
        config = fragment_config(
            [16, 16, 64],
            [1, 1, 512],
            range_num_stages=[2],
            metal_tabulate_max_entries=16,
        )
        out, msl = _run(static, args, config)
        self._close(out, ref)
        self.assertRegex(msl, r"auto _early_\d+_next = wq\[")
        self.assertRegex(msl, r"auto _early_\d+_next = s\[")
        self.assertIn("constant float _helion_table_0[16] = {", msl)
        self.assertNotIn("_helion_table_1", msl)

    def test_rows_past_end(self) -> None:
        """Rows past M's end and columns past N's read at their tile's first
        and are never stored (``test_metal_fragment_rows``): computed
        operands with tails along M, N and K, also split over two SIMD
        groups, and 4 rows of a dequantizing matmul's 8-row tile."""
        x = torch.randn(13, 40, device=DEVICE)
        y = torch.randn(40, 20, device=DEVICE)
        ref = x.half().cpu().double()[:, 16:] @ y.half().cpu().double()[16:]
        for num_warps in (1, 2):
            with self.subTest(num_warps=num_warps):
                config = fragment_config([16, 16, 16], num_warps=num_warps)
                out, msl = _run(k_from_16, (x, y), config)
                self._close(out, ref)
                self.assertIn("_clamped_indices_0", msl)
        (xq, wq, s), ref = _quantized("nvfp4", 8, 48, 256)
        out, msl = _run(
            nvfp4_rows, (xq[:4], wq, s), fragment_config([8, 16, 64], [1, 1, 512])
        )
        self._close(out, ref[:4])
        self.assertRegex(msl, r"\? \(offset_\d+ \+ _e_m\) : offset_\d+\)")

    def test_nvfp4(self) -> None:
        """nvfp4 codes through half bits and a bitcast (view(dtype)), E4M3
        scales, at 8 rows and at one (decode)."""
        (x, wq, s), ref = _quantized("nvfp4", 8, 48, 256)
        out, msl = _run(
            nvfp4_rows, (x[:1], wq, s), fragment_config([1, 16, 64], [1, 1, 512])
        )
        self._close(out, ref[:1])
        self._assert_fragments(msl, transposed=True)
        # The row's block of 1 grows to the B fragment's 8 columns, which the
        # element loops skip past the block.
        self.assertIn("if (_e_m < 1 && _e_n < 16) {", msl)
        knobs = {"metal_tabulate_max_entries": 16, "range_num_stages": [2]}
        config = fragment_config([16, 16, 64], [1, 1, 512], **knobs)
        out, msl = _run(nvfp4_rows, (x, wq, s), config)
        self._close(out, ref)
        self._assert_fragments(msl, transposed=True)
        # The steps are unrolled and each element's word and scale indices
        # folded over the K tile's offset, a multiple of 64.  A lane's K
        # positions are one run of 16 (column j of step s is K 8 * j + s), so
        # over all 8 steps it reads two words, offset / 8 + fn and the next,
        # and no division is left in the operand loops (the two left compute
        # the grid).  The indices stay ints, as the signed tile index is.
        words = set(re.findall(r"= static_cast<int>\((\(.*offset_\d+ / 8.*\))\);", msl))
        self.assertEqual(
            {w.replace(re.findall(r"offset_\d+", w)[0], "o") for w in words},
            {"(_fn + (o / 8))", "((_fn + (o / 8)) + 1)"},
        )
        self.assertEqual(msl.count("c10::metal::floor_divide"), 2)
        # The code's decode, from its nibble, is a lookup in a 16-entry table
        # (metal_tabulate_max_entries=16; none without it).
        self.assertIn("constant float _helion_table_0[16] = {", msl)
        self.assertRegex(msl, r"= _helion_table_0\[v_\d+\];")
        # With 2 stages the K loop loads each word, scale and x element a
        # tile early: for the first tile before the loop, for the next at
        # its top.
        self.assertRegex(msl, r"auto _early_0_next = wq\[")
        self.assertRegex(msl, r"if \(offset_\d+ \+ 64 < 256\) \{")
        self.assertRegex(msl, r"auto words = _early_\d+;")
        # x is loaded unmasked, so a tile early too: every tile of M and K
        # lies inside its extent.
        self.assertRegex(msl, r"_early_\d+_next = x\[_fm \* 8 \+ _fn \* 256 \+")
        # Without the keys the loads stay in the operand loop, re-run per
        # element at the element's K, and there is no table.
        _, msl = _run(
            nvfp4_rows, (x, wq, s), fragment_config([16, 16, 64], [1, 1, 512])
        )
        self.assertNotIn("_early", msl)
        self.assertNotIn("_helion_table", msl)
        self.assertRegex(
            msl,
            r"\bx\[\(\(\(offset_\d+ \+ _e_m\) \* 256\) \+ \(offset_\d+ \+ _e_k\)\)\]",
        )

    def test_fp8(self) -> None:
        """mxfp8 codes and E8M0 scales through hl.dot; a K tail (K = 224 in
        tiles of 64) masks the computed operand past the live length.  In
        tiles of 16, a tile is half a scale group: a lane's scale index
        folds to one floor term of its K run and the tile's offset, a
        multiple of 16 only."""
        args, ref = _quantized("fp8", 8, 40, 224)
        out, msl = _run(fp8_rows, args, fragment_config([16, 16, 64]))
        self._close(out, ref)
        self._assert_fragments(msl, transposed=True)
        self.assertRegex(msl, r"_e_k < \(end_\d+ - offset_\d+\)\) \? ")
        self.assertNotIn("_helion_table", msl)
        # A byte's decode takes a 256-entry table, and a 16-entry cap keeps it.
        for cap, table in ((16, False), (256, True)):
            with self.subTest(cap=cap):
                config = fragment_config([16, 16, 64], metal_tabulate_max_entries=cap)
                out, msl = _run(fp8_rows, args, config)
                self._close(out, ref)
                self.assertEqual("constant float _helion_table_0[256]" in msl, table)
        # Under 2 stages the K tail's loads, all masked, stay in their tile.
        out, msl = _run(
            fp8_rows, args, fragment_config([16, 16, 64], range_num_stages=[2])
        )
        self._close(out, ref)
        self.assertNotIn("_early", msl)
        # Without a tail the words, scales and x load a tile early, beside a
        # 256-entry table: neither mechanism keys on one format.
        args, ref = _quantized("fp8", 8, 48, 256)
        config = fragment_config(
            [16, 16, 64],
            [1, 1, 512],
            range_num_stages=[2],
            metal_tabulate_max_entries=256,
        )
        out, msl = _run(fp8_rows, args, config)
        self._close(out, ref)
        self.assertRegex(msl, r"auto _early_\d+_next = wq\[")
        self.assertIn("constant float _helion_table_0[256]", msl)
        out, msl = _run(fp8_rows, args, fragment_config([16, 16, 16]))
        self._close(out, ref)
        self.assertRegex(
            msl, r"= static_cast<int>\(\(\(\(_fn \* 2\) \+ offset_\d+\) / 32\)\);"
        )

    def test_views(self) -> None:
        """Both operands views with tails along M, N and K (M = 13, N = 20,
        K = 40 in tiles of 16), clamped past M's and N's extents and masked
        past K's, with a stored epilogue reading a bias, also over one row;
        and a view of a transposed tensor."""
        x = torch.randn(13, 40, device=DEVICE, dtype=torch.float16)
        y = torch.randn(40, 20, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(20, device=DEVICE)
        ref = torch.relu(x.cpu().double() @ y.cpu().double() + bias.cpu().double())
        out, msl = _run(views_epilogue, (x, y, bias), fragment_config([16, 16, 16]))
        self._close(out, ref)
        self._assert_fragments(msl, transposed=False)
        # One row: its block of 1 grows to an 8-row fragment, whose rows past
        # the block the element loops skip.
        out, msl = _run(views_epilogue, (x[:1], y, bias), fragment_config([1, 16, 16]))
        self._close(out, ref[:1])
        self._assert_fragments(msl, transposed=False)
        self.assertIn("if (_e_m < 1 && _e_n < 16) {", msl)
        yt = torch.randn(24, 40, device=DEVICE, dtype=torch.float16)
        out, msl = _run(transposed_views, (x, yt), fragment_config([16, 16, 16]))
        self._close(out, x.cpu().double() @ yt.cpu().double().T)
        self._assert_fragments(msl, transposed=False)
        # A dim of size 1 broadcasts: K, and K and M alike.
        for rows in (13, 1):
            with self.subTest(broadcast_rows=rows):
                x1 = torch.randn(rows, 1, device=DEVICE, dtype=torch.float16)
                out, msl = _run(broadcast_lhs, (x1, y), fragment_config([16, 16, 16]))
                ref = x1.cpu().double().expand(rows, 40) @ y.cpu().double()
                self._close(out, ref)
                self._assert_fragments(msl, transposed=False)

    def test_computed_both(self) -> None:
        """Both operands computed, the accumulator seeded from a tensor
        (staged, then read into the fragments between barriers) and updated
        in place before each op."""
        x = torch.randn(16, 48, device=DEVICE, dtype=torch.float16)
        y = torch.randn(48, 24, device=DEVICE, dtype=torch.float16)
        c = torch.randn(16, 24, device=DEVICE)
        out, msl = _run(computed_both, (x, y, c), fragment_config([16, 16, 16]))
        acc = c.cpu().double()
        a, b = x.cpu().double() * 2, y.cpu().double() + 1
        for k0 in range(0, 48, 16):
            acc = acc * 0.5 + a[:, k0 : k0 + 16] @ b[k0 : k0 + 16]
        self._close(out, acc)
        self.assertIn("simdgroup_multiply_accumulate(", msl)
        self.assertNotIn("matmul2d", msl)
        # A seed of a transposed accumulator, also over one row (the row's
        # block of 1 grows to 8 columns of the fragments).
        y = torch.randn(48, 16, device=DEVICE, dtype=torch.float16)
        for rows, config in (
            (16, fragment_config([16, 16, 16])),
            (1, fragment_config([1, 16, 16])),
        ):
            with self.subTest(rows=rows):
                args = (x[:rows], y, c[:rows, :16])
                out, msl = _run(seeded_rhs, args, config)
                ref = c[:rows, :16].cpu().double() + x[:rows].cpu().double() @ (
                    y.cpu().double() * 2
                )
                self._close(out, ref)
                self.assertRegex(msl, r"int _e_n = 8 \* \(_fi >> 1\) \+ _fm;")

    def test_computed_lhs(self) -> None:
        """A computed lhs beside a view rhs keeps the accumulator's
        orientation, also over one row, which a fragment group needn't
        stage (MPP refuses a staged operand over a block of 1)."""
        x = torch.randn(24, 32, device=DEVICE, dtype=torch.float16)
        y = torch.randn(32, 16, device=DEVICE, dtype=torch.float16)
        ref = torch.sigmoid(x.cpu().double()).half().double() @ y.cpu().double()
        for rows, config in (
            (24, fragment_config([16, 16, 16])),
            (1, fragment_config([1, 16, 16])),
        ):
            with self.subTest(rows=rows):
                out, msl = _run(computed_lhs, (x[:rows], y), config)
                self._close(out, ref[:rows])
                self._assert_fragments(msl, transposed=False)

    def test_dots(self) -> None:
        """hl.dot is admitted by the fragment rules: one row of a computed lhs
        lowers on fragments, and so does a ':' K, one run from zeroed
        fragments."""
        x = torch.randn(16, 32, device=DEVICE, dtype=torch.float16)
        y = torch.randn(32, 16, device=DEVICE, dtype=torch.float16)
        xd, yd = x.cpu().double(), y.cpu().double()
        out, msl = _run(dot_computed_lhs, (x[:1], y), fragment_config([1, 16, 16]))
        self._close(out, torch.sigmoid(xd[:1]).half().double() @ yd)
        self._assert_fragments(msl, transposed=False)
        out, msl = _run(whole_k_dot, (x, y), fragment_config([16, 16]))
        self._close(out, xd @ yd)
        self.assertIn("simdgroup_multiply_accumulate", msl)

    def test_two_groups(self) -> None:
        """Two fragment groups in one K loop share the lane preamble, each
        with its own step loop, also on tiles that keep 13 fragments live in
        each lane, which run slower but right."""
        x = torch.randn(8, 48, device=DEVICE, dtype=torch.float16)
        y = torch.randn(48, 32, device=DEVICE, dtype=torch.float16)
        z = torch.randn(48, 32, device=DEVICE, dtype=torch.float16)
        out, _, msl = run_with_msl(two_groups, (x, y, z), fragment_config([16, 16, 16]))
        a, b = cast("tuple[torch.Tensor, torch.Tensor]", out)
        xd = x.cpu().double()
        self._close(a, xd @ y.cpu().double())
        self._close(b, xd @ z.cpu().double())
        self._assert_fragments(msl, transposed=False)
        self.assertEqual(msl.count("constexpr short _fs = 0;"), 2)
        # Under 2 stages both runs load ahead, under one test for a next tile.
        config = fragment_config([16, 16, 16], range_num_stages=[2])
        out, _, msl = run_with_msl(two_groups, (x, y, z), config)
        a, b = cast("tuple[torch.Tensor, torch.Tensor]", out)
        self._close(a, xd @ y.cpu().double())
        self._close(b, xd @ z.cpu().double())
        self.assertIn("_early_0_next = ", msl)
        self.assertEqual(len(re.findall(r"if \(offset_\d+ \+ 16 < 48\) \{", msl)), 1)
        out, _, msl = run_with_msl(two_groups, (x, y, z), fragment_config([16, 32, 16]))
        a, b = cast("tuple[torch.Tensor, torch.Tensor]", out)
        self._close(a, xd @ y.cpu().double())
        self._close(b, xd @ z.cpu().double())
        self._assert_fragments(msl, transposed=False)

    def test_split(self) -> None:
        """SIMD groups splitting the K loop (``num_warps`` > 1) give one SIMD
        group's results up to the order of the fp32 additions, the same bits
        at every launch: with a table and loads a tile early, under a
        persistent grid, where a SIMD group has no K tile (K 40 in tiles of
        16 over four) and M and N leave tails, for two accumulators in one
        loop, an accumulator that starts from 0.5, over a K tail (K 224 in
        tiles of 64), for a dynamic M, for an accumulator an outer loop
        carries from one split K loop to the next, for two split K loops in
        a row, and under a persistent grid whose threadgroups each run
        several tiles, which combine again into the same threadgroup
        memory."""
        (x, wq, s), nvfp4_ref = _quantized("nvfp4", 8, 48, 256)
        # More tiles of 16 columns than threadgroups at one per GPU core.
        wide = 16 * (2 * helion.runtime.get_num_sm(torch.device(DEVICE)) + 1)
        (xw, wqw, sw), wide_ref = _quantized("nvfp4", 8, wide, 256)
        xs = torch.randn(3, 8, 48, device=DEVICE, dtype=torch.float16)
        ys = torch.randn(3, 48, 32, device=DEVICE, dtype=torch.float16)
        sums_ref = torch.cumsum(xs.cpu().double() @ ys.cpu().double(), 0)
        z3 = torch.randn(8, 32, device=DEVICE, dtype=torch.float16)
        w3 = torch.randn(32, 32, device=DEVICE, dtype=torch.float16)
        xv = torch.randn(13, 40, device=DEVICE, dtype=torch.float16)
        yv = torch.randn(40, 20, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(20, device=DEVICE)
        views_ref = torch.relu(xv.cpu().double() @ yv.cpu().double() + bias.cpu())
        x2 = torch.randn(8, 48, device=DEVICE, dtype=torch.float16)
        y2 = torch.randn(48, 32, device=DEVICE, dtype=torch.float16)
        z2 = torch.randn(48, 32, device=DEVICE, dtype=torch.float16)
        fp8_args, fp8_ref = _quantized("fp8", 8, 40, 224)
        int4_args, int4_ref = _quantized("int4", 11, 64, 256)
        threads = [1, 2, 16]
        for kernel, args, ref, config in (
            (
                nvfp4_rows,
                (x, wq, s),
                nvfp4_ref,
                fragment_config([8, 16, 16], threads, num_warps=2),
            ),
            (
                nvfp4_rows,
                (x, wq, s),
                nvfp4_ref,
                fragment_config(
                    [8, 16, 16],
                    threads,
                    num_warps=8,
                    range_num_stages=[2],
                    metal_tabulate_max_entries=16,
                ),
            ),
            (
                nvfp4_rows,
                (x, wq, s),
                nvfp4_ref,
                fragment_config(
                    [8, 16, 16],
                    threads,
                    num_warps=4,
                    pid_type="persistent_blocked",
                    range_num_stages=[2],
                ),
            ),
            (
                views_epilogue,
                (xv, yv, bias),
                views_ref,
                fragment_config([16, 16, 16], num_warps=4),
            ),
            (
                two_groups,
                (x2, y2, z2),
                (
                    x2.cpu().double() @ y2.cpu().double(),
                    x2.cpu().double() @ z2.cpu().double(),
                ),
                fragment_config([16, 16, 16], num_warps=2),
            ),
            (
                filled,
                (x2[:, :32], y2[:32, :16]),
                x2[:, :32].cpu().double() @ y2[:32, :16].cpu().double() + 0.5,
                fragment_config([16, 16, 16], num_warps=4),
            ),
            (fp8_rows, fp8_args, fp8_ref, fragment_config([16, 16, 64], num_warps=4)),
            (
                dequant_rows,
                int4_args,
                int4_ref,
                fragment_config([16, 16, 64], num_warps=2),
            ),
            (
                running_sums,
                (xs, ys),
                sums_ref,
                fragment_config([8, 16, 16], num_warps=2),
            ),
            (
                two_k_loops,
                (x2, y2, z3, w3),
                (
                    x2.cpu().double() @ y2.cpu().double() + 1,
                    z3.cpu().double() @ w3.cpu().double(),
                ),
                fragment_config([8, 16, 16, 16], num_warps=4),
            ),
            (
                nvfp4_rows,
                (xw, wqw, sw),
                wide_ref,
                fragment_config(
                    [8, 16, 16],
                    threads,
                    num_warps=2,
                    pid_type="persistent_blocked",
                    num_sm_multiplier=1,
                ),
            ),
        ):
            with self.subTest(kernel=kernel.name, config=config):
                out, _, msl = run_with_msl(kernel, args, config)
                rerun, _, _ = run_with_msl(kernel, args, config)
                outs = out if isinstance(out, tuple) else (out,)
                refs = ref if isinstance(ref, tuple) else (ref,)
                reruns = rerun if isinstance(rerun, tuple) else (rerun,)
                for o, r, a in zip(outs, refs, reruns, strict=True):
                    self._close(o, r)
                    torch.testing.assert_close(a, o, atol=0, rtol=0)
                simdgroups = config["num_warps"]
                self.assertIn("[[simdgroup_index_in_threadgroup]]", msl)
                self.assertRegex(msl, rf"\+= {simdgroups} \* _BLOCK_SIZE_\d+\)")
                self.assertIn("for (uint _s = 0; _s < ", msl)
                self.assertNotIn("matmul2d", msl)

    def test_straddling_codes(self) -> None:
        """5-bit codes that straddle 16-bit words (``affine_mm``) at each
        of the fragment matmul heuristic's seeds (a 128-wide K block: a
        lane's run of 16 codes is five whole words, its word offsets and
        shifts constants) and at a 32-wide K block loading a tile ahead
        (each element shifts by a computed amount and reads two words)."""
        args = affine_args(5, 128, 8, 256, 1024, DEVICE)
        ref = affine_reference(args)
        seeds = heuristic_seeds(MetalFragmentMatmulHeuristic, affine_mm.bind(args))
        self.assertEqual(len(seeds), 2)
        k32 = fragment_config([16, 32, 32], num_warps=8, range_num_stages=[2])
        for config in [*seeds, k32]:
            with self.subTest(config=config):
                out, msl = _run(affine_mm, args, config)
                self._close(out, ref)
                # On fragments, its K loop split over SIMD groups (a
                # threadgroup combine, which _assert_fragments refuses).
                self.assertIn("simdgroup_multiply_accumulate(", msl)
                self.assertNotIn("matmul2d", msl)

    def test_few_row_affine(self) -> None:
        """Affine weights of 4 to 8 bits at groups of 32 to 128 and 1 to 16
        rows, against float64.  At 4 and 8 bits over whole K tiles x is read
        as its half view, so the accumulator is transposed and the MMAs run
        fragment by fragment, a bfloat x as a half one.  A K tail, and 5-
        and 6-bit codes, which straddle words, keep x computed and the
        accumulator untransposed."""
        for bits, group, rows, tail in _FEW_ROWS:
            with self.subTest(bits=bits, group=group, rows=rows, tail=tail):
                k = 1024 + (64 if tail else 0)
                args = affine_args(bits, group, rows, 96, k, DEVICE)
                out, msl = _run(
                    affine_mm,
                    args,
                    fragment_config([16, 32, 128], [1, 1, 32], num_warps=4),
                )
                self._close(out, affine_reference(args), tol=2e-3)
                few_rows = bits in (4, 8) and k % 128 == 0
                self._assert_fragments_split(msl, transposed=few_rows)
                self.assertEqual("simdgroup_float8x8 _frag_b15[" in msl, few_rows)
        # A bfloat activation reads its view into the float fragments too.
        x, *weight = affine_args(4, 64, 8, 96, 1024, DEVICE)
        args = (x.to(torch.bfloat16), *weight)
        config = fragment_config([16, 32, 128], [1, 1, 32], num_warps=4)
        out, msl = _run(affine_mm, args, config)
        self._close(out, affine_reference(args), tol=8e-3)
        self._assert_fragments_split(msl, transposed=True)

    def _assert_fragments_split(self, msl: str, transposed: bool) -> None:
        """The op is MMAs on register fragments whose K loop SIMD groups
        split, and its accumulator transposed or not."""
        self.assertIn("simdgroup_multiply_accumulate(", msl)
        self.assertIn("_frag_split", msl)
        self.assertNotIn("matmul2d", msl)
        self.assertEqual(fragment_transposed(msl), transposed)

    def test_k_from_16(self) -> None:
        """A K loop from 16 under 2 stages loads its first tile, K 16 to 48,
        before the loop, and the next at the top of the iteration before:
        the result is x[:, 16:] @ y[16:]."""
        x = torch.randn(16, 80, device=DEVICE)
        y = torch.randn(80, 16, device=DEVICE)
        a, b = x.half().cpu().double(), y.half().cpu().double()
        config = fragment_config([16, 16, 32], [1, 1, 32], range_num_stages=[2])
        out, msl = _run(k_from_16, (x, y), config)
        self._close(out, a[:, 16:] @ b[16:])
        # The first tile's loads, before the loop, read from K = 16: column
        # 16 of x's row, row 16 of y (16 * 16 elements in).
        self.assertRegex(
            msl,
            r"auto _early_0_next = x\[_fm \* 80 \+ _fn \* 4 \+ offset_\d+ \* 80 \+ 16\];",
        )
        self.assertRegex(
            msl, r"auto _early_\d+_next = y\[_fm \* 64 \+ _fn \+ offset_\d+ \+ 256\];"
        )
        self.assertRegex(msl, r"for \(int offset_\d+ = 16;")


def _seeds(spec: ConfigSpec) -> list[helion.Config]:
    """The spec's compiler seeds completed from the defaults, as the
    search benchmarks them."""
    return [config for _, config in ConfigGeneration(spec).seed_flat_config_pairs()]


class _TimedSearch(FiniteSearch):
    """A finite search that keeps its own measurements."""

    def benchmark_batch(
        self,
        configs: list[helion.Config],
        *,
        desc: str = "Benchmarking",
        raise_if_no_viable_config: bool = True,
    ) -> list[BenchmarkResult]:
        results = super().benchmark_batch(
            configs, desc=desc, raise_if_no_viable_config=raise_if_no_viable_config
        )
        self.timings = {r.config: r.perf for r in results}
        return results


@_requires_mps
class TestFragmentAutotune(unittest.TestCase):
    def _search(
        self,
        bound: BoundKernel,
        args: tuple[torch.Tensor, ...],
        configs: list[helion.Config],
    ) -> None:
        """The autotuner over ``configs`` compiles and times every one, both
        lowerings among them; which one wins is the GPU's call, not the
        test's."""
        search = _TimedSearch(bound, args, configs)
        search.autotune()
        timings = search.timings
        self.assertEqual(len(timings), len(configs))
        self.assertTrue(all(math.isfinite(t) for t in timings.values()), timings)
        lowerings = {c.config.get("metal_matmul", "mpp") for c in timings}
        self.assertEqual(lowerings, {"mpp", "fragment"})

    @pytest.mark.slow
    def test_autotune_times_every_seed(self) -> None:
        """A few-row nvfp4 matmul over the MPP default and the spec's seeds,
        fragments with and without tables."""
        (x, wq, s), _ = _quantized("nvfp4", 8, 4096, 4096)
        bound = nvfp4_rows.bind((x, wq, s))
        spec = bound.config_spec
        configs = [spec.default_config(), *_seeds(spec)]
        self.assertEqual(
            {bool(c.config.get("metal_tabulate_max_entries")) for c in configs},
            {False, True},
        )
        self._search(bound, (x, wq, s), configs)


if __name__ == "__main__":
    unittest.main()
