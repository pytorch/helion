"""Tests for GRID-level lane-loop vectorization on the CuTe backend.

Pointwise kernels (``for tile in hl.tile(...)`` at grid level) get the same
outer x constexpr-V lane partition + hoisted ``cute.arch.load(..., V)`` /
``_cute_store_u*_vec`` flush protocol as device loops when the config sets
``num_threads[block] < block_size`` and ``cute_vector_widths[block] > 1``.

Covers both strategies (``PerThreadNDTileStrategy`` for N-D tiles,
``PerThreadFlattenedTileStrategy`` for 1D), fp32 Uint32-carrier stores, the
``fast_math`` setting on cute, and two regressions found in review:

- an index-independent store value must not drop the vec wrapper (the store
  flush lives inside it),
- a lane block accepted on a NON-stride-1 tensor dim must not be vectorized
  (the hoist reads V contiguous elements).
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _add2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _mul1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * y[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _const_store(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = s[0] * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _col0_gather(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, 0] * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _tanh1d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.tanh(x[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _bias_add2d(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(out.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + b[tile_n]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _tile_index_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = (
            x[tile_m, tile_n]
            + b[tile_n]
            + hl.tile_index(tile_m).to(torch.float32)[:, None]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def _tanh1d_fastmath(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.tanh(x[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def _div1d_fastmath(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] / y[tile]
    return out


@onlyBackends(["cute"])
class TestCutePointwiseVec(TestCase):
    def test_nd_grid_vec_bf16(self) -> None:
        """2D grid tile with nt<bs and V=8 emits one LDG.128 hoist per input
        and a single u16 vec-store flush; results match eager."""
        x = torch.randn(256, 2048, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(256, 2048, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _add2d,
            (x, y),
            block_sizes=[1, 2048],
            num_threads=[1, 256],
            cute_vector_widths=[1, 8],
        )
        self.assertIn("ir.VectorType.get([8], cutlass.Uint16.mlir_type)", code)
        self.assertIn("_cute_store_u16_vec", code)
        torch.testing.assert_close(out, x + y)

    def test_flattened_grid_vec_fp32_u32_store(self) -> None:
        """1D (flattened) grid tile with V=4 fp32: Uint32 vec loads and the
        Uint32-carrier vec-store flush."""
        x = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _mul1d,
            (x, y),
            block_sizes=[1024],
            num_threads=[256],
            cute_vector_widths=[4],
        )
        self.assertIn("ir.VectorType.get([4], cutlass.Uint32.mlir_type)", code)
        self.assertIn("_cute_store_u32_vec", code)
        torch.testing.assert_close(out, x * y)

    def test_strided_lane_layout_vec(self) -> None:
        """Strided lane layout with vec: thread vec chunks interleave by NT."""
        x = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _mul1d,
            (x, y),
            block_sizes=[2048],
            num_threads=[256],
            cute_vector_widths=[4],
            cute_lane_layouts=["strided"],
        )
        self.assertIn("ir.VectorType.get([4], cutlass.Uint32.mlir_type)", code)
        torch.testing.assert_close(out, x * y)

    def test_masked_tail_tile_vec(self) -> None:
        """Extent not divisible by the block (but divisible by V): the tail
        tile is masked per element and the flush is mask-gated."""
        x = torch.randn(8, 1664, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(8, 1664, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _add2d,
            (x, y),
            block_sizes=[1, 1024],
            num_threads=[1, 128],
            cute_vector_widths=[1, 8],
        )
        self.assertIn("ir.VectorType.get([8], cutlass.Uint16.mlir_type)", code)
        torch.testing.assert_close(out, x + y)

    def test_index_independent_store_keeps_wrapper(self) -> None:
        """Regression: a store whose VALUE reads no per-lane vars must keep
        the vec wrapper (its flush IS the store) — previously the wrapper
        was dropped as dead, losing the store entirely."""
        for dtype, vec in ((torch.float32, 4), (torch.bfloat16, 8)):
            x = torch.randn(2**14, device=DEVICE, dtype=dtype)
            s = torch.full((1,), 3.0, device=DEVICE, dtype=dtype)
            _, out = code_and_output(
                _const_store,
                (x, s),
                block_sizes=[1024],
                num_threads=[128],
                cute_vector_widths=[vec],
            )
            torch.testing.assert_close(out, torch.full_like(x, 6.0))

    def test_non_stride1_lane_axis_not_vectorized(self) -> None:
        """Regression: the lane block sits on dim 0 of ``x`` while dim 1 is
        contiguous — the hoist (V contiguous elements) must NOT fire."""
        x = torch.randn(4096, 8, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _col0_gather,
            (x,),
            block_sizes=[2048],
            num_threads=[256],
            cute_vector_widths=[8],
        )
        self.assertNotIn("ir.VectorType.get([8]", code)
        torch.testing.assert_close(out, x[:, 0] * 2.0)

    def test_flat_multi_vec_full_cover(self) -> None:
        """flatten_loops + vec on a 2D kernel: full-cover contiguous tensors
        get FLAT base-pointer hoists; results match eager."""
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _add2d,
            (x, y),
            block_sizes=[1, 2048],
            num_threads=[1, 256],
            cute_vector_widths=[1, 8],
            flatten_loops=[True],
        )
        self.assertIn("ir.VectorType.get([8], cutlass.Uint16.mlir_type)", code)
        torch.testing.assert_close(out, x + y)

    def test_flat_multi_vec_odd_row_broadcast(self) -> None:
        """ODD row length (V does not divide N, but divides the total):
        x/out vectorize via flat chunks that straddle rows; the broadcast
        bias fails the full-cover gate and stays scalar per element."""
        m, n = 64, 4093
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        b = torch.randn(n, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(
            _bias_add2d,
            (x, b),
            block_sizes=[1, 4096],
            num_threads=[1, 512],
            cute_vector_widths=[1, 8],
            flatten_loops=[True],
        )
        self.assertIn("ir.VectorType.get([8], cutlass.Uint16.mlir_type)", code)
        torch.testing.assert_close(out, x + b)

    def test_flat_multi_vec_reordered_loops_stays_scalar(self) -> None:
        """A non-identity loop_order breaks the row-major flat equivalence;
        the ctx gate must fall back to scalar and stay correct."""
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _add2d,
            (x, y),
            block_sizes=[1, 2048],
            num_threads=[1, 256],
            cute_vector_widths=[1, 8],
            flatten_loops=[True],
            loop_orders=[[1, 0]],
        )
        self.assertNotIn("ir.VectorType.get([8]", code)
        torch.testing.assert_close(out, x + y)

    def test_tile_index_disables_flatten_reregistration(self) -> None:
        """Regression: hl.tile_index only disabled flatten at CODEGEN time,
        so the pointwise flatten re-registration resurrected it -> wrong
        results under flatten_loops=[True]."""
        x = torch.randn(64, 512, device=DEVICE, dtype=torch.float32)
        b = torch.randn(512, device=DEVICE, dtype=torch.float32)
        spec = _tile_index_bias.bind((x, b)).config_spec
        self.assertEqual(len(spec.flatten_loops), 0)
        _, out = code_and_output(
            _tile_index_bias, (x, b), block_sizes=[16, 128], num_threads=[1, 64]
        )
        ref = (
            x + b + torch.arange(x.size(0), device=DEVICE, dtype=torch.float32)[:, None]
        )
        torch.testing.assert_close(out, ref)

    def test_scalar_strided_lane_keeps_launch_width(self) -> None:
        """Regression: a SCALAR lane loop with cute_lane_layouts=strided
        must not change the launch width.  The launch-dim recovery regex
        derives the thread extent from the ``thread_idx()[a] * epT``
        multiplier of ``indices_*`` lines; an ``offset + tid + lane*NT``
        form parsed as epT=1 and inflated the launch to block_size,
        sending surplus threads out of bounds (cudaErrorIllegalAddress
        during autotuning)."""
        x = torch.randn(1024, 512, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1024, 512, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _add2d,
            (x, y),
            block_sizes=[128, 512],
            num_threads=[64, 1],
            cute_vector_widths=[1, 1],
            cute_lane_layouts=["strided", "strided"],
        )
        self.assertIn("block=(64, 1, 1)", code)
        torch.testing.assert_close(out, x + y)

    def test_fast_math_setting_routes_cute_fastmath(self) -> None:
        """The ``fast_math`` SETTING (user opt-in) routes fastmath=True into
        cute.math calls; without it the accurate form is emitted.  Numerics
        changes are never a tunable config knob — only this setting."""
        x = torch.randn(2**14, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _tanh1d_fastmath,
            (x,),
            block_sizes=[1024],
            num_threads=[256],
            cute_vector_widths=[4],
        )
        self.assertIn("fastmath=True", code)
        torch.testing.assert_close(out, torch.tanh(x), rtol=1e-4, atol=1e-4)
        code, out = code_and_output(
            _tanh1d,
            (x,),
            block_sizes=[1024],
            num_threads=[256],
            cute_vector_widths=[4],
        )
        self.assertNotIn("fastmath=True", code)
        torch.testing.assert_close(out, torch.tanh(x))

    def test_fastmath_div_non_fp32_keeps_accurate_path(self) -> None:
        """Regression: ``cute.math.div`` lowers to fp32-only NVVM intrinsics,
        so under ``fast_math`` a 16-bit division must keep the accurate IEEE
        path instead of raising at DSL trace time; fp32 still gets the
        approx+ftz form."""
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(2**14, device=DEVICE, dtype=dtype)
            y = torch.rand(2**14, device=DEVICE, dtype=dtype) + 0.5
            code, out = code_and_output(_div1d_fastmath, (x, y), block_sizes=[1024])
            self.assertNotIn("cute.math.div", code)
            torch.testing.assert_close(out, x / y)
        x = torch.randn(2**14, device=DEVICE, dtype=torch.float32)
        y = torch.rand(2**14, device=DEVICE, dtype=torch.float32) + 0.5
        code, out = code_and_output(_div1d_fastmath, (x, y), block_sizes=[1024])
        self.assertIn("cute.math.div", code)
        torch.testing.assert_close(out, x / y, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    import unittest

    unittest.main()
