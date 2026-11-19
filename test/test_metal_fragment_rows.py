"""Fragment operand elements past the accumulator's live rows and columns
(``metal_matmul="fragment"``, ``fragment_mma``): their values reach only
accumulator elements that no store writes, so their loads read a clamped
in-bounds element instead of a masked one.  Renders on CPU tensors
(``helion._testing.metal.render_msl``), with no Apple GPU."""

from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._testing import TestCase
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_fragment import k_from_16
from helion._testing.metal_fragment import nvfp4_cpu_args
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import views_epilogue
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32
S = {"backend": "metal", "static_shapes": True}


def _xy(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """M = 13, K = 40 and N = 20: tails in tiles of 16 along all three."""
    return torch.randn(13, 40, dtype=dtype), torch.randn(40, 20, dtype=dtype)


@helion.kernel(**S)
def rows_of_c(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """x @ y over c's rows, which may outnumber x's."""
    m = c.size(0)
    k = x.size(1)
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def kept_rows(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """x @ y over c's rows, reading x's rows where it has them and from row
    40 on: a load mask term of the row index besides the tile's mask."""
    m = c.size(0)
    k = x.size(1)
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            keep = (tile_m.index < x.size(0)) | (tile_m.index >= 40)
            a = hl.load(x, [tile_m, tile_k], extra_mask=keep[:, None])
            acc = torch.addmm(acc, a.to(torch.float16), y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=False)
def view_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ y over any number of rows of x, with K, N and the strides fixed:
    x a view whose rows' extent is known only at run time."""
    m = x.size(0)
    k = hl.specialize(x.size(1))
    n = hl.specialize(y.size(1))
    hl.specialize(x.stride())
    hl.specialize(y.stride())
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def packed_n(x: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """x @ W for int4 codes packed two per byte along N (wq [K, N / 2])."""
    m, k = x.size()
    n = wq.size(1) * 2
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            b = wq[tile_k, tile_n.index // 2].to(torch.int32)
            q = (b >> ((tile_n.index % 2) * 4)[None, :]) & 0xF
            acc = torch.addmm(acc, x[tile_m, tile_k], (q.to(F32) - 8.0).to(x.dtype))
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def even_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x[::2] @ y: the lhs rows read through tile_m.index * 2."""
    m = x.size(0) // 2
    k = x.size(1)
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m.index * 2, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def col_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The column sums of x @ y: the accumulator reduced over M."""
    m, k = x.size()
    n = y.size(1)
    out = torch.zeros([n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        hl.atomic_add(out, [tile_n], acc.sum(0))
    return out


class TestFragmentRows(TestCase):
    def test_whole_tiles(self) -> None:
        """Where codegen masks no M or N tile, nothing is clamped: a
        dequantizing kernel at 8 rows split over 4 SIMD groups, and at 16
        rows loading a K tile early, as ExecuTorch's verify matmuls run at
        T=8 and T=16 (a view lhs, a computed rhs, tiles of 8 and 16 rows)."""
        for args, config in (
            (
                nvfp4_cpu_args(8),
                fragment_config([8, 16, 64], num_threads=[1, 1, 512], num_warps=4),
            ),
            (
                nvfp4_cpu_args(16),
                fragment_config(
                    [16, 16, 64],
                    num_threads=[1, 1, 512],
                    num_warps=4,
                    range_num_stages=[2],
                ),
            ),
        ):
            with self.subTest(rows=args[0].size(0)):
                msl = render_msl(nvfp4_rows, args, config)
                self.assertNotIn("_clamped_", msl)
                self.assertNotRegex(msl, r"\? \(offset_\d+ \+ _e_[mn]\) : offset_\d+\)")

    def test_computed(self) -> None:
        """Computed operands (fp32 cast to half) read a row past M's end, or
        a column past N's, at the tile's first, unmasked; the K tail keeps
        its mask, and so its zeros."""
        msl = render_msl(k_from_16, _xy(F32), fragment_config([16, 16, 16]))
        self.assertIn("auto _clamped_indices_0 = (mask_0 ? indices_0 : offset_0);", msl)
        self.assertIn("auto _clamped_indices_1 = (mask_1 ? indices_1 : offset_1);", msl)
        self.assertIn(
            "_helion_masked(x[(mask_2 ? ((_clamped_indices_0 * 40) + indices_2) : 0)], "
            "mask_2, 0)",
            msl,
        )
        self.assertIn(
            "_helion_masked(y[(mask_2 ? ((indices_2 * 20) + _clamped_indices_1) : 0)], "
            "mask_2, 0)",
            msl,
        )
        # Before, x's loads were masked by (mask_0 & mask_2), y's by
        # (mask_2 & mask_1).
        self.assertNotIn("(mask_0 & mask_2)", msl)
        self.assertNotIn("(mask_2 & mask_1)", msl)

    def test_views(self) -> None:
        """Views read a row past M's end, or a column past N's, at the
        tile's first: the value mask keeps only K's terms."""
        x, y = _xy(torch.float16)
        msl = render_msl(
            views_epilogue, (x, y, torch.randn(20)), fragment_config([16, 16, 16])
        )
        self.assertIn(
            "_helion_masked(x[(((offset_2 + _e_k) < 40) ? (((((offset_0 + _e_m) < 13) ? "
            "(offset_0 + _e_m) : offset_0) * 40) + (offset_2 + _e_k)) : 0)], "
            "((offset_2 + _e_k) < 40), 0)",
            msl,
        )
        self.assertIn(
            "_helion_masked(y[(((offset_2 + _e_k) < 40) ? (((offset_2 + _e_k) * 20) + "
            "(((offset_1 + _e_n) < 20) ? (offset_1 + _e_n) : offset_1)) : 0)], "
            "((offset_2 + _e_k) < 40), 0)",
            msl,
        )

    def test_dynamic(self) -> None:
        """Dynamic shapes clamp alike: every tile the loop runs starts
        inside M's and N's runtime ends, so its first row and column are
        in bounds without any bound known at codegen.  Computed operands
        (the kernels bound to dynamic shapes), then a view of runtime rows."""
        x, y = _xy(torch.float16)
        for kernel, args in (
            (k_from_16, (x, y)),
            (views_epilogue, (x, y, torch.randn(20))),
        ):
            with self.subTest(kernel=kernel.name):
                dynamic = helion.kernel(kernel.fn, backend="metal", static_shapes=False)
                msl = render_msl(dynamic, args, fragment_config([16, 16, 16]))
                self.assertIn(
                    "auto _clamped_indices_0 = (mask_0 ? indices_0 : offset_0);", msl
                )
                self.assertRegex(
                    msl, r"auto mask_0 = \(\(tid\[0\] < 4\) && \(indices_0 < m\)\);"
                )
                self.assertIn(
                    "_helion_masked(x[(mask_2 ? ((_clamped_indices_0 * x_stride_0) + "
                    "(indices_2 * x_stride_1)) : 0)], mask_2, 0)",
                    msl,
                )
        msl = render_msl(
            view_rows,
            (
                torch.randn(5, 32, dtype=torch.float16),
                torch.randn(32, 16, dtype=torch.float16),
            ),
            fragment_config([16, 16, 16]),
        )
        self.assertIn(
            "x[((static_cast<long>((((offset_0 + _e_m) < m) ? (offset_0 + _e_m) : "
            "offset_0)) * 32) + static_cast<long>((offset_2 + _e_k)))]",
            msl,
        )

    def test_derived_index(self) -> None:
        """A load whose address reaches the M or N position through a name
        computed from it, not through the tile's index itself, keeps its
        mask: its address past the tile's end would be neither masked nor
        clamped.  Codes packed two per byte along N read
        ``wq[k, tile_n.index // 2]``; every other row reads
        ``x[tile_m.index * 2, k]``."""
        x, _ = _xy(torch.float16)
        wq = torch.randint(0, 255, (40, 10), dtype=torch.uint8)
        msl = render_msl(packed_n, (x, wq), fragment_config([16, 16, 16]))
        self.assertRegex(msl, r"_helion_masked\(wq\[\(\(mask_2 & mask_1\) \? ")
        self.assertIn("? (offset_0 + _e_m) : offset_0)", msl)  # x's rows still clamp
        rows = torch.randn(26, 40, dtype=torch.float16)
        msl = render_msl(
            even_rows,
            (rows, torch.randn(40, 20, dtype=torch.float16)),
            fragment_config([16, 16, 16]),
        )
        self.assertRegex(msl, r"_helion_masked\(x\[\(\(mask_0 & mask_2\) \? ")
        self.assertNotIn("_clamped_", msl)

    def test_position_term(self) -> None:
        """A load whose other mask terms read the M or N position keeps its
        mask: past the loop's end such a term can admit a row while it
        excludes the tile's first, whose address the kernel then never
        reads.  x has 20 rows and the loop 36, so the tile at row 32 holds
        none of x's, and its rows 40 to 47, past the loop's end with
        ``keep`` true, would read x's row 32 if clamped.  y's columns still
        clamp."""
        msl = render_msl(
            kept_rows,
            (
                torch.randn(20, 40),
                torch.randn(40, 20, dtype=torch.float16),
                torch.randn(36, 1),
            ),
            fragment_config([16, 16, 16]),
        )
        self.assertIn(
            "_helion_masked(x[(((mask_0 & mask_2) & subscript) ? "
            "((indices_0 * 40) + indices_2) : 0)], ((mask_0 & mask_2) & subscript), 0)",
            msl,
        )
        self.assertNotIn("_clamped_", msl)
        self.assertIn("(((offset_1 + _e_n) < 20) ? (offset_1 + _e_n) : offset_1)", msl)

    def test_rows_past_tensor(self) -> None:
        """A view whose M loop runs past the tensor's rows keeps its row
        mask: those rows are stored, as zeros, and a tile past the rows has
        no first row inside the tensor to clamp to."""
        x, y = _xy(torch.float16)
        msl = render_msl(
            rows_of_c, (x, y, torch.randn(16, 1)), fragment_config([16, 16, 16])
        )
        self.assertIn(
            "_helion_masked(x[((((offset_0 + _e_m) < 13) && ((offset_2 + _e_k) < 40)) ? "
            "(((offset_0 + _e_m) * 40) + (offset_2 + _e_k)) : 0)], "
            "(((offset_0 + _e_m) < 13) && ((offset_2 + _e_k) < 40)), 0)",
            msl,
        )

    def test_rows_reduced(self) -> None:
        """An accumulator reduced over M would sum the rows past M's end, so
        the clamp rests on the fragment lowering's refusal to lower one: its
        result is read only by element loops, which re-run the store's
        mask."""
        x, y = _xy(torch.float16)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "a matmul whose result is read in the Scalar layout"
        ):
            render_msl(col_sums, (x, y), fragment_config([16, 16, 16]))
