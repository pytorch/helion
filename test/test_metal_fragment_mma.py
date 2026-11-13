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
from helion import exc
from helion._testing import skipUnlessBackends
from helion._testing.metal import run_with_msl
from helion._testing.metal_fragment import e4m3
from helion._testing.metal_fragment import fp8_rows
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import views_epilogue
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.finite_search import FiniteSearch
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.benchmark_provider import BenchmarkResult
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel

DEVICE = "mps"
F32 = torch.float32
S = {"backend": "metal", "static_shapes": True}
FRAG = {"metal_matmul": "fragment", "num_warps": 1}

# Metal kernels on macOS only: other backends' suites skip the module.
pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests require macOS"),
    skipUnlessBackends(["metal"]),
]
_requires_mps = unittest.skipUnless(
    sys.platform == "darwin" and torch.backends.mps.is_available(),
    "requires a Metal GPU",
)


def _frag(
    block_sizes: list[int], num_threads: list[int] | None = None, **knobs: object
) -> helion.Config:
    extra = {} if num_threads is None else {"num_threads": num_threads}
    return helion.Config(block_sizes=block_sizes, **extra, **knobs, **FRAG)


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
def seeded_rhs(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Only the rhs computed, so the accumulator is transposed, and seeded
    from c."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = c[tile_m, tile_n].to(F32)
        for tile_k in hl.tile(k):
            b = (y[tile_k, tile_n].to(F32) * 2.0).to(x.dtype)
            acc = torch.addmm(acc, x[tile_m, tile_k], b)
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
def two_groups(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two matmuls of x in one K loop, each result stored."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    out2 = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        a = hl.zeros([tile_m, tile_n], dtype=F32)
        b = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            xt = x[tile_m, tile_k]
            a = torch.addmm(a, xt, y[tile_k, tile_n])
            b = torch.addmm(b, xt, z[tile_k, tile_n])
        out[tile_m, tile_n] = a
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
            # A transposed accumulator's A fragments run over N.
            a_rows = re.search(r"int (_e_[mn]) = 8 \* \(_fi >> 1\) \+ _fm;", msl)
            assert a_rows is not None
            self.assertEqual(a_rows.group(1), "_e_n" if transposed else "_e_m")

    def _close(self, out: torch.Tensor, ref: torch.Tensor, tol: float = 1e-2) -> None:
        torch.testing.assert_close(out.cpu().double(), ref.double(), atol=tol, rtol=tol)

    def test_dequant_rows(self) -> None:
        """int4 weights decoded into the rhs, the computed operand, with M
        dynamic (rows past m masked by the stores) and K on lanes or on
        threads; the accumulator is transposed so the weights are the
        MMA's A."""
        for m, config in (
            (5, _frag([16, 16, 32], [1, 1, 32])),
            (11, _frag([16, 16, 64])),
            (8, _frag([16, 16, 128], [1, 1, 32])),
        ):
            with self.subTest(m=m, config=config):
                args, ref = _quantized("int4", m, 64, 256)
                out, msl = _run(dequant_rows, args, config)
                self._close(out, ref)
                self._assert_fragments(msl, transposed=True)
        # Bound to static shapes, of its decode, ``(q - 8) * scale``, the
        # code's conversion is a 16-entry lookup and the scale per group stays
        # computed.
        static = helion.kernel(dequant_rows.fn, backend="metal", static_shapes=True)
        args, ref = _quantized("int4", 8, 64, 256)
        config = _frag([16, 16, 64], [1, 1, 512], metal_tabulate_max_entries=16)
        out, msl = _run(static, args, config)
        self._close(out, ref)
        self.assertIn("constant float _helion_table_0[16] = {", msl)
        self.assertNotIn("_helion_table_1", msl)

    def test_nvfp4(self) -> None:
        """nvfp4 codes through half bits and a bitcast (view(dtype)), E4M3
        scales, at 8 rows and at one (decode)."""
        (x, wq, s), ref = _quantized("nvfp4", 8, 48, 256)
        out, msl = _run(nvfp4_rows, (x[:1], wq, s), _frag([1, 16, 64], [1, 1, 512]))
        self._close(out, ref[:1])
        self._assert_fragments(msl, transposed=True)
        # The row's block of 1 grows to the B fragment's 8 columns, which the
        # element loops skip past the block.
        self.assertIn("if (_e_m < 1 && _e_n < 16) {", msl)
        config = _frag([16, 16, 64], [1, 1, 512], metal_tabulate_max_entries=16)
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
        # The weights' word and scale loads are in the operand loop, re-run
        # per element at the element's K, and x is loaded unmasked: every
        # tile of M and K lies inside its extent.
        self.assertRegex(
            msl,
            r"\bx\[\(\(\(offset_\d+ \+ _e_m\) \* 256\) \+ \(offset_\d+ \+ _e_k\)\)\]",
        )
        # The code's decode, from its nibble, is a lookup in a 16-entry table
        # (metal_tabulate_max_entries=16; none without it).
        self.assertIn("constant float _helion_table_0[16] = {", msl)
        self.assertRegex(msl, r"= _helion_table_0\[v_\d+\];")
        _, msl = _run(nvfp4_rows, (x, wq, s), _frag([16, 16, 64], [1, 1, 512]))
        self.assertNotIn("_helion_table", msl)

    def test_fp8(self) -> None:
        """mxfp8 codes and E8M0 scales through hl.dot; a K tail (K = 224 in
        tiles of 64) masks the computed operand past the live length.  In
        tiles of 16, a tile is half a scale group: a lane's scale index
        folds to one floor term of its K run and the tile's offset, a
        multiple of 16 only."""
        args, ref = _quantized("fp8", 8, 40, 224)
        out, msl = _run(fp8_rows, args, _frag([16, 16, 64]))
        self._close(out, ref)
        self._assert_fragments(msl, transposed=True)
        self.assertRegex(msl, r"_e_k < \(end_\d+ - offset_\d+\)\) \? ")
        self.assertNotIn("_helion_table", msl)
        # A byte's decode takes a 256-entry table, and a 16-entry cap keeps it.
        for cap, table in ((16, False), (256, True)):
            with self.subTest(cap=cap):
                config = _frag([16, 16, 64], metal_tabulate_max_entries=cap)
                out, msl = _run(fp8_rows, args, config)
                self._close(out, ref)
                self.assertEqual("constant float _helion_table_0[256]" in msl, table)
        out, msl = _run(fp8_rows, args, _frag([16, 16, 16]))
        self._close(out, ref)
        self.assertRegex(
            msl, r"= static_cast<int>\(\(\(\(_fn \* 2\) \+ offset_\d+\) / 32\)\);"
        )

    def test_views(self) -> None:
        """Both operands views, masked at the tensors' extents (M = 13, N =
        20, K = 40 in tiles of 16), with a stored epilogue reading a bias,
        also over one row; and a view of a transposed tensor."""
        x = torch.randn(13, 40, device=DEVICE, dtype=torch.float16)
        y = torch.randn(40, 20, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(20, device=DEVICE)
        ref = torch.relu(x.cpu().double() @ y.cpu().double() + bias.cpu().double())
        out, msl = _run(views_epilogue, (x, y, bias), _frag([16, 16, 16]))
        self._close(out, ref)
        self._assert_fragments(msl, transposed=False)
        # One row: its block of 1 grows to an 8-row fragment, whose rows past
        # the block the element loops skip.
        out, msl = _run(views_epilogue, (x[:1], y, bias), _frag([1, 16, 16]))
        self._close(out, ref[:1])
        self._assert_fragments(msl, transposed=False)
        self.assertIn("if (_e_m < 1 && _e_n < 16) {", msl)
        yt = torch.randn(24, 40, device=DEVICE, dtype=torch.float16)
        out, msl = _run(transposed_views, (x, yt), _frag([16, 16, 16]))
        self._close(out, x.cpu().double() @ yt.cpu().double().T)
        self._assert_fragments(msl, transposed=False)
        # A dim of size 1 broadcasts: K, and K and M alike.
        for rows in (13, 1):
            with self.subTest(broadcast_rows=rows):
                x1 = torch.randn(rows, 1, device=DEVICE, dtype=torch.float16)
                out, msl = _run(broadcast_lhs, (x1, y), _frag([16, 16, 16]))
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
        out, msl = _run(computed_both, (x, y, c), _frag([16, 16, 16]))
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
        for rows, config in ((16, _frag([16, 16, 16])), (1, _frag([1, 16, 16]))):
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
        for rows, config in ((24, _frag([16, 16, 16])), (1, _frag([1, 16, 16]))):
            with self.subTest(rows=rows):
                out, msl = _run(computed_lhs, (x[:rows], y), config)
                self._close(out, ref[:rows])
                self._assert_fragments(msl, transposed=False)

    def test_dots(self) -> None:
        """hl.dot is admitted by the fragment rules: one row of a computed lhs
        lowers on fragments, and a ':' K, which fragments refuse, keeps the
        scalar contraction."""
        x = torch.randn(16, 32, device=DEVICE, dtype=torch.float16)
        y = torch.randn(32, 16, device=DEVICE, dtype=torch.float16)
        xd, yd = x.cpu().double(), y.cpu().double()
        out, msl = _run(dot_computed_lhs, (x[:1], y), _frag([1, 16, 16]))
        self._close(out, torch.sigmoid(xd[:1]).half().double() @ yd)
        self._assert_fragments(msl, transposed=False)
        out, msl = _run(whole_k_dot, (x, y), _frag([16, 16]))
        self._close(out, xd @ yd)
        self.assertNotIn("simdgroup_multiply_accumulate", msl)

    def test_two_groups(self) -> None:
        """Two fragment groups in one K loop share the lane preamble, each
        with its own step loop; the cap counts both accumulators."""
        x = torch.randn(8, 48, device=DEVICE, dtype=torch.float16)
        y = torch.randn(48, 32, device=DEVICE, dtype=torch.float16)
        z = torch.randn(48, 32, device=DEVICE, dtype=torch.float16)
        out, _, msl = run_with_msl(two_groups, (x, y, z), _frag([16, 16, 16]))
        a, b = cast("tuple[torch.Tensor, torch.Tensor]", out)
        xd = x.cpu().double()
        self._close(a, xd @ y.cpu().double())
        self._close(b, xd @ z.cpu().double())
        self._assert_fragments(msl, transposed=False)
        self.assertEqual(msl.count("constexpr short _fs = 0;"), 2)
        message = "13 8x8 fragments live (max 9, 4 of them the K loop's other"
        with self.assertRaisesRegex(exc.BackendUnsupported, re.escape(message)):
            _run(two_groups, (x, y, z), _frag([16, 32, 16]))


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
        """The autotuner over ``configs`` times every one, both lowerings
        among them, and picks the fastest by its own measurements; which one
        wins is the GPU's call, not the test's."""
        search = _TimedSearch(bound, args, configs)
        picked = search.autotune()
        timings = search.timings
        self.assertEqual(len(timings), len(configs))
        self.assertTrue(all(math.isfinite(t) for t in timings.values()), timings)
        lowerings = {c.config.get("metal_matmul", "mpp") for c in timings}
        self.assertEqual(lowerings, {"mpp", "fragment"})
        self.assertEqual(picked, min(timings, key=timings.__getitem__))

    @pytest.mark.slow
    def test_autotune_picks(self) -> None:
        """A few-row nvfp4 matmul over the MPP default and the spec's seeds
        (fragments with and without tables), and a large square one over MPP
        tiles and the fragment seeds."""
        (x, wq, s), _ = _quantized("nvfp4", 8, 4096, 4096)
        bound = nvfp4_rows.bind((x, wq, s))
        spec = bound.config_spec
        configs = [spec.default_config(), *_seeds(spec)]
        self.assertEqual(
            {bool(c.config.get("metal_tabulate_max_entries")) for c in configs},
            {False, True},
        )
        self._search(bound, (x, wq, s), configs)

        a = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
        b = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(4096, device=DEVICE)
        bound = views_epilogue.bind((a, b, bias))
        base = bound.config_spec.default_config()
        # The search reaches MPP's larger tiles by mutating block sizes; give
        # them here, beside the default and the fragment seeds.
        tiles = [base.config["block_sizes"], [32, 32, 32], [64, 64, 32]]
        configs = [
            *(helion.Config(**{**base.config, "block_sizes": t}) for t in tiles),
            *_seeds(bound.config_spec),
        ]
        self._search(bound, (a, b, bias), configs)


if __name__ == "__main__":
    unittest.main()
