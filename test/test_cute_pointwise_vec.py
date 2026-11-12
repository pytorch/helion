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

import unittest

import pytest
import torch
from torch._inductor.utils import fresh_cache
from torch._inductor.utils import run_and_get_code

import helion
from helion._compat import supports_torch_compile_template_lowering
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
def _repeated_scale(
    x: torch.Tensor, scale: torch.Tensor, repeat: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale[tile.index // repeat]
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
def _gather_rows(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The embedding gather whose seeds test_cute_pointwise_resident_seeds.py
    pins, run here; keep the two copies the same kernel."""
    out = torch.empty(
        [x.size(0), weight.size(1)], dtype=weight.dtype, device=weight.device
    )
    for tile_b, tile_e in hl.tile([x.size(0), weight.size(1)]):
        out[tile_b, tile_e] = weight[x[tile_b], tile_e]
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


@helion.kernel(backend="cute", static_shapes=True)
def _sigmoid1d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(x[tile])
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


@helion.kernel(backend="cute", static_shapes=True)
def _div1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] / y[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _add_explicit_evict(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        a = hl.load(x, [tile], eviction_policy="evict_last")
        out[tile] = a + y[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _cast_i16_to_bf16(w: torch.Tensor) -> torch.Tensor:
    k, n = w.shape
    out = torch.empty((k, n), dtype=torch.bfloat16, device=w.device)
    for tk, tn in hl.tile((k, n)):
        out[tk, tn] = w[tk, tn].to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _cast_i32_to_f32(w: torch.Tensor) -> torch.Tensor:
    k, n = w.shape
    out = torch.empty((k, n), dtype=torch.float32, device=w.device)
    for tk, tn in hl.tile((k, n)):
        out[tk, tn] = w[tk, tn].to(torch.float32)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _double_bf16(w: torch.Tensor) -> torch.Tensor:
    k, n = w.shape
    out = torch.empty((k, n), dtype=torch.bfloat16, device=w.device)
    for tk, tn in hl.tile((k, n)):
        out[tk, tn] = w[tk, tn] * 2
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _unsigned_to_f32_and_flag(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(w.shape, dtype=torch.float32, device=w.device)
    big = torch.empty(w.shape, dtype=torch.int32, device=w.device)
    for tk, tn in hl.tile(w.shape):
        v = w[tk, tn]
        out[tk, tn] = v.to(torch.float32)
        big[tk, tn] = (v > 100).to(torch.int32)
    return out, big


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

    def test_streaming_store_policy(self) -> None:
        """``cute_store_policy="streaming"`` flushes the full 16-byte packets
        with ``st.global.cs`` (evict-first) for bf16 and fp32; results match."""
        x = torch.randn(256, 2048, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(256, 2048, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _add2d,
            (x, y),
            block_sizes=[1, 2048],
            num_threads=[1, 256],
            cute_vector_widths=[1, 8],
            cute_store_policy="streaming",
        )
        self.assertIn("_cute_store_u16x8_streaming(", code)
        self.assertNotIn("_cute_store_u16_vec(", code)
        torch.testing.assert_close(out, x + y)
        a = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _mul1d,
            (a, b),
            block_sizes=[1024],
            num_threads=[256],
            cute_vector_widths=[4],
            cute_store_policy="streaming",
        )
        self.assertIn("_cute_store_u32x4_streaming(", code)
        torch.testing.assert_close(out, a * b)

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

    def test_wide_row_packet_seed_masks_a_ragged_row(self) -> None:
        """The wide row-packet seed of a 384-wide bf16 row: eight strided
        lanes of 16-element packets over a 256-wide tile, so each row's
        second tile is half masked.  It gathers the rows exactly, with and
        without proven bounds."""
        idx = torch.randint(4096, (256,), device=DEVICE, dtype=torch.int32)
        weight = torch.randn(4096, 384, device=DEVICE, dtype=torch.bfloat16)
        spec = _gather_rows.bind((idx, weight)).config_spec
        # The cute_pdl twins only change the launch.
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.config.get("cute_min_blocks_per_mp") == 1
            and not seed.config.get("cute_pdl")
        ]
        self.assertEqual(
            sorted(bool(seed.config.get("cute_proven_bounds")) for seed in seeds),
            [False, True],
        )
        for seed in seeds:
            with self.subTest(proven_bounds=seed.config.get("cute_proven_bounds")):
                # Only a guard that this run masks a second tile; the CPU test
                # test_wide_row_packets_round_a_row_down_to_a_power_of_two_tile
                # pins the seed.
                self.assertEqual(seed.config["block_sizes"][1], 256)
                code, out = code_and_output(_gather_rows, (idx, weight), **seed.config)
                self.assertIn("ir.VectorType.get([16], cutlass.Uint16.mlir_type)", code)
                self.assertTrue(torch.equal(out, weight[idx]))

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

    def test_gather_index_extent_does_not_imply_contiguous_loads(self) -> None:
        """Divided lane indices must retain repeated addresses when vectorizing."""
        for dtype, vector_width in ((torch.float32, 4), (torch.bfloat16, 8)):
            for repeat in (1, 64):
                with self.subTest(dtype=dtype, repeat=repeat):
                    x = torch.linspace(0.5, 1.5, 1024, device=DEVICE, dtype=dtype)
                    # Keep all possible wrong lane-base reads in bounds, with
                    # distinct values so a contiguous-load rewrite is visible.
                    scale = torch.arange(1024, device=DEVICE).to(dtype)
                    _, out = code_and_output(
                        _repeated_scale,
                        (x, scale, repeat),
                        block_sizes=[1024],
                        num_threads=[128],
                        cute_vector_widths=[vector_width],
                    )
                    expected = x * scale[torch.arange(1024, device=DEVICE) // repeat]
                    torch.testing.assert_close(out, expected)

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

    def test_l2_last_eviction_policy(self) -> None:
        """``load_eviction_policies=["l2_last", ...]`` routes 16-byte vec
        hoists through the inline-PTX L2::evict_last cache-hint load."""
        x = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _mul1d,
            (x, y),
            block_sizes=[1024],
            num_threads=[256],
            cute_vector_widths=[4],
            load_eviction_policies=["l2_last", "l2_last"],
        )
        self.assertIn("_cute_load_l2_evict_last(", code)
        torch.testing.assert_close(out, x * y)

    def test_l2_last_eviction_policy_narrow_packets(self) -> None:
        """``l2_last`` reaches 8- and 4-byte packets through the helper of
        their width (it used to be dropped below 16 bytes)."""
        for dtype, width, suffix in (
            (torch.float32, 2, "_8b"),
            (torch.float16, 4, "_8b"),
            (torch.float16, 2, "_4b"),
            (torch.bfloat16, 2, "_4b"),
        ):
            with self.subTest(dtype=dtype, width=width):
                x = torch.randn(2**14, device=DEVICE, dtype=dtype)
                y = torch.randn_like(x)
                code, out = code_and_output(
                    _mul1d,
                    (x, y),
                    block_sizes=[1024],
                    num_threads=[128],
                    cute_vector_widths=[width],
                    load_eviction_policies=["l2_last", "l2_last"],
                )
                self.assertIn(f"_cute_load_l2_evict_last{suffix}(", code)
                torch.testing.assert_close(out, x * y, atol=0, rtol=0)

    def test_two_level_eviction_policy(self) -> None:
        """8/16-byte cache hints and scalar fallbacks preserve every element."""
        for dtype, width in (
            (torch.float32, 4),
            (torch.float16, 8),
            (torch.bfloat16, 8),
        ):
            for policy in ("l1_l2_first", "l1_l2_last"):
                for selected_width in sorted({width, 4, 2, 1}, reverse=True):
                    with self.subTest(dtype=dtype, policy=policy, width=selected_width):
                        x = torch.randn(2**14, device=DEVICE, dtype=dtype)
                        y = torch.randn_like(x)
                        code, out = code_and_output(
                            _mul1d,
                            (x, y),
                            block_sizes=[1024],
                            num_threads=[128],
                            cute_vector_widths=[selected_width],
                            load_eviction_policies=[policy, policy],
                        )
                        helper = "_cute_load_l1_l2_evict_" + policy.rsplit("_", 1)[1]
                        packet_bytes = selected_width * x.element_size()
                        self.assertEqual(f"{helper}(" in code, packet_bytes == 16)
                        self.assertEqual(f"{helper}_8b(" in code, packet_bytes == 8)
                        # A width-1 site is a scalar load, not a packet.
                        self.assertEqual(
                            f"{helper}_4b(" in code,
                            packet_bytes == 4 and selected_width > 1,
                        )
                        torch.testing.assert_close(out, x * y, atol=0, rtol=0)

    def test_epilogue_subtile_disables_vec(self) -> None:
        """Regression: epilogue_subtile stages stores through smem with a
        sync inside the per-element pipeline; combined with the vec
        hoist/flush protocol it silently corrupted ~50% of the output.
        Subtiled configs must stay on the scalar form."""
        x = torch.randn(64, 2048, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 2048, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _add2d,
            (x, y),
            block_sizes=[1, 2048],
            num_threads=[1, 256],
            cute_vector_widths=[1, 4],
            epilogue_subtile=2,
        )
        self.assertNotIn("ir.VectorType.get(", code)
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

    def test_default_div_does_not_depend_on_torch_fastmath_context(self) -> None:
        """Default division reads Helion's setting, not a private Torch symbol."""
        x = torch.randn(2**14, device=DEVICE, dtype=torch.float32)
        y = torch.rand(2**14, device=DEVICE, dtype=torch.float32) + 0.5
        code, out = code_and_output(_div1d, (x, y), block_sizes=[1024])
        self.assertNotIn("cute.math.div", code)
        torch.testing.assert_close(out, x / y)

    def test_seed_eviction_list_sized_from_spec_slots(self) -> None:
        """Regression: a load carrying an explicit ``hl.load(...,
        eviction_policy=...)`` gets no config slot, so the l2_last seed must
        size ``load_eviction_policies`` from the spec (not the load count) or
        the flat search surface rejects it (`Expected list of length 1`)."""
        x = torch.randn(2**14, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2**14, device=DEVICE, dtype=torch.float32)
        bound = _add_explicit_evict.bind((x, y))
        spec = bound.env.config_spec
        n_slots = spec.load_eviction_policies.length
        self.assertEqual(n_slots, 1)
        seeds = spec.compiler_seed_configs
        self.assertTrue(seeds)
        seeded_policies = [
            s.config["load_eviction_policies"]
            for s in seeds
            if s.config.get("load_eviction_policies")
        ]
        self.assertTrue(seeded_policies)
        for policies in seeded_policies:
            self.assertEqual(list(policies), ["l2_last"] * n_slots)
        config_gen = spec.create_config_generation()
        for seed in seeds:
            config_gen.canonicalize_flat(config_gen.flatten(seed))

    def test_default_sigmoid_rcp_approx(self) -> None:
        """Default (no fast_math) sigmoid lowers through the triton-parity
        RCP.APPROX + EX2.APPROX sequence: same accuracy class as the branchy
        IEEE-div form it replaces (both are dominated by the x*log2e
        argument rounding; max 64 ulp / mean 3.5 ulp vs fp64 on [-87, 87])
        and ~20% faster on fp16 streams.  Specials must be preserved."""
        x = torch.randn(2**14, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(_sigmoid1d, (x,), block_sizes=[1024])
        self.assertIn("cute.math.rcp", code)
        self.assertNotIn("1.0 / ", code)
        torch.testing.assert_close(
            out, torch.sigmoid(x.to(torch.float32)).to(torch.float16)
        )
        xs = torch.tensor(
            [0.0, -0.0, float("inf"), float("-inf"), float("nan"), -200.0, 200.0, 30.0],
            device=DEVICE,
            dtype=torch.float32,
        )
        _, out32 = code_and_output(_sigmoid1d, (xs,), block_sizes=[8])
        torch.testing.assert_close(out32, torch.sigmoid(xs), equal_nan=True)

    def test_unsigned_loads_keep_their_unsigned_type(self) -> None:
        """A plain ``(ptr).load()`` of a uint8 / uint32 element yields the
        signed integer of its width in CuTe DSL; values with the top bit set
        must still convert and compare as unsigned, scalar or vectorized."""
        torch.manual_seed(0)
        for dtype, high in ((torch.uint8, 2**8), (torch.uint32, 2**32)):
            w = torch.randint(0, high, (64, 1024), device=DEVICE).to(dtype)
            wide = w.to(torch.int64)
            for config in (
                {"block_sizes": [8, 64]},
                {
                    "block_sizes": [8, 1024],
                    "num_threads": [8, 128],
                    "cute_vector_widths": [1, 4],
                },
            ):
                with self.subTest(dtype=dtype, config=config):
                    _, (out, big) = code_and_output(
                        _unsigned_to_f32_and_flag, (w,), **config
                    )
                    torch.testing.assert_close(out, wide.to(torch.float32))
                    torch.testing.assert_close(big, (wide > 100).to(torch.int32))

    def test_int16_packet_cast_to_bf16_is_bitwise_exact(self) -> None:
        """Every int16 value rides a V=8 Uint16 packet, is bitcast back to
        Int16 per lane and takes the program's own bf16 cast: sign and
        round-to-nearest-even match torch bit for bit."""
        w = (
            torch.arange(-32768, 32768, dtype=torch.int32, device=DEVICE)
            .to(torch.int16)
            .reshape(64, 1024)
        )
        code, out = code_and_output(
            _cast_i16_to_bf16,
            (w,),
            block_sizes=[8, 1024],
            num_threads=[8, 128],
            cute_vector_widths=[1, 8],
        )
        self.assertIn("ir.VectorType.get([8], cutlass.Uint16.mlir_type)", code)
        self.assertIn(".bitcast(cutlass.Int16)", code)
        self.assertIn("_cute_store_u16_vec", code)
        self.assertTrue(
            torch.equal(out.view(torch.int16), w.to(torch.bfloat16).view(torch.int16))
        )

    def test_int32_packet_cast_to_fp32_is_bitwise_exact(self) -> None:
        """int32 words ride a V=4 Uint32 packet; the fp32 cast rounds the
        values above 2**24 (ties included) exactly as torch does."""
        generator = torch.Generator().manual_seed(0)
        w = torch.randint(
            -(2**31), 2**31 - 1, (128, 512), generator=generator, dtype=torch.int64
        ).to(torch.int32)
        w[0, :16] = torch.tensor(
            [
                0,
                1,
                -1,
                2**24 + 1,
                -(2**24) - 1,
                2**24 + 3,
                2**31 - 1,
                -(2**31),
                2**25 + 2,
                -(2**25) - 2,
                33554433,
                -33554433,
                12345679,
                -12345679,
                100000001,
                -100000001,
            ],
            dtype=torch.int32,
        )
        w = w.to(DEVICE)
        code, out = code_and_output(
            _cast_i32_to_f32,
            (w,),
            block_sizes=[8, 512],
            num_threads=[8, 128],
            cute_vector_widths=[1, 4],
        )
        self.assertIn("ir.VectorType.get([4], cutlass.Uint32.mlir_type)", code)
        self.assertIn("_cute_store_u32_vec", code)
        self.assertTrue(
            torch.equal(out.view(torch.int32), w.to(torch.float32).view(torch.int32))
        )

    def test_misaligned_views_take_scalar_loads(self) -> None:
        """A packet needs an aligned base and row stride.  A 132-wide row
        (264 bytes) or a view four elements into its storage stays on scalar
        loads and still computes the exact result; the contiguous copy of the
        same data keeps its packet."""
        packet = "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"
        for kernel, big in (
            (
                _cast_i16_to_bf16,
                torch.randint(
                    -32768, 32767, (64, 136), dtype=torch.int16, device=DEVICE
                ),
            ),
            (_double_bf16, torch.randn(64, 136, dtype=torch.bfloat16, device=DEVICE)),
        ):
            row_stride = torch.empty((64, 132), dtype=big.dtype, device=DEVICE)
            row_stride = row_stride[:, :128]
            row_stride.copy_(big[:, :128])
            views = {
                "contiguous": big[:, :128].contiguous(),
                "row_stride_132": row_stride,
                "offset_4": big[:, 4:132],
            }
            for name, w in views.items():
                with self.subTest(kernel=kernel.name, view=name):
                    code, out = code_and_output(
                        kernel,
                        (w,),
                        block_sizes=[8, 128],
                        num_threads=[8, 16],
                        cute_vector_widths=[1, 8],
                    )
                    self.assertEqual(packet in code, name == "contiguous")
                    reference = (
                        w.to(torch.bfloat16) if w.dtype is torch.int16 else w * 2
                    )
                    self.assertTrue(torch.equal(out, reference))

    @unittest.skipUnless(
        supports_torch_compile_template_lowering(),
        "this PyTorch build lacks Helion's Inductor template lowering",
    )
    def test_misaligned_view_under_torch_compile_takes_scalar_loads(self) -> None:
        """Inductor lowers the kernel against example inputs, which must have
        the runtime address residue: a view four bf16 elements into an aligned
        base, a misaligned graph input, and an aligned-looking slice of one are
        all eight bytes off a packet boundary, and a packet emitted for any of
        them faults with a misaligned address."""

        @helion.kernel(
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                block_sizes=[8, 128], num_threads=[8, 16], cute_vector_widths=[1, 8]
            ),
        )
        def double_bf16(w: torch.Tensor) -> torch.Tensor:
            k, n = w.shape
            out = torch.empty((k, n), dtype=torch.bfloat16, device=w.device)
            for tk, tn in hl.tile((k, n)):
                out[tk, tn] = w[tk, tn] * 2
            return out

        packet = "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"
        big = torch.randn(64, 144, dtype=torch.bfloat16, device=DEVICE)
        # (graph input, slice taken inside the graph, packet expected)
        cases = {
            "sliced_offset_4": (big, 4, False),
            "sliced_offset_8": (big, 8, True),
            "misaligned_input": (big[:, 4:132], 0, False),
            "aligned_slice_of_misaligned_input": (big[:, 4:], 8, False),
        }
        for name, (graph_input, start, expect_packet) in cases.items():
            with self.subTest(case=name):

                def f(x: torch.Tensor, start: int = start) -> torch.Tensor:
                    return double_bf16(x[:, start : start + 128]) + 1

                torch._dynamo.reset()
                with fresh_cache():
                    out, (code,) = run_and_get_code(
                        torch.compile(f, fullgraph=True, backend="inductor"),
                        graph_input,
                    )
                self.assertEqual(packet in code, expect_packet)
                expected = graph_input[:, start : start + 128] * 2 + 1
                self.assertTrue(torch.equal(out, expected))

    @unittest.skipUnless(
        supports_torch_compile_template_lowering(),
        "this PyTorch build lacks Helion's Inductor template lowering",
    )
    def test_dynamic_lane_length_under_torch_compile_takes_scalar_loads(
        self,
    ) -> None:
        """Under ``torch.compile(dynamic=True)`` Inductor binds the kernel to
        placeholder inputs (a 64-long weight) and runs its code at every
        weight length.  A packet proven from the placeholder's length would
        load the last partial packet of a 255-long weight from offset 0, so
        lanes 248-254 would read weights 0-6: the weight stays on scalar loads,
        exact at each length."""

        @helion.kernel(
            backend="cute",
            static_shapes=False,
            config=helion.Config(
                block_sizes=[4, 2], cute_vector_widths=[8, 8], num_threads=[4, 2, 32]
            ),
        )
        def head_rms_norm(q: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            tokens, heads, head_dim = q.shape
            out = torch.empty_like(q)
            for tile_t, tile_h in hl.tile([tokens, heads]):
                x = q[tile_t, tile_h, :].float()
                rstd = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
                # Lanes past the end of a shorter weight read zeros.
                cols = hl.arange(head_dim)
                out[tile_t, tile_h, :] = (x * rstd * weight[cols].float()).to(q.dtype)
            return out

        def f(q: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return head_rms_norm(q, weight) + 1

        compiled = torch.compile(f, fullgraph=True, backend="inductor", dynamic=True)
        torch._dynamo.reset()
        with fresh_cache():
            for tokens, heads, length in ((64, 8, 255), (32, 4, 250)):
                with self.subTest(length=length):
                    q = torch.randn(
                        tokens, heads, 256, dtype=torch.bfloat16, device=DEVICE
                    )
                    # A static head dim gives hl.arange its extent.
                    torch._dynamo.mark_static(q, 2)
                    weight = torch.randn(length, dtype=torch.bfloat16, device=DEVICE)
                    out, codes = run_and_get_code(compiled, q, weight)
                    if length == 255:
                        self.assertEqual(len(codes), 1)
                    for code in codes:
                        self.assertIn("weight.iterator", code)
                        self.assertNotIn("cute.arch.load(weight.iterator", code)
                    x = q.float()
                    rstd = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
                    padded = torch.zeros(256, dtype=torch.float32, device=DEVICE)
                    padded[:length] = weight.float()
                    expected = (x * rstd * padded).to(q.dtype) + 1
                    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)

    @unittest.skipUnless(
        supports_torch_compile_template_lowering(),
        "this PyTorch build lacks Helion's Inductor template lowering",
    )
    def test_dynamic_reduction_length_under_torch_compile_masks_each_lane(
        self,
    ) -> None:
        """Under ``torch.compile(dynamic=True)`` Inductor binds the kernel to
        a 64-column placeholder and runs its code at every row length.  Eight
        lanes masked as one packet, proven from the placeholder's length,
        would add columns 61-63 of a 64-wide buffer to each 61-column row."""

        @helion.kernel(
            backend="cute",
            static_shapes=False,
            config=helion.Config(
                block_sizes=[1],
                reduction_loops=[32],
                num_threads=[1, 4],
                cute_vector_widths=[8, 1],
            ),
        )
        def row_logsumexp(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
            for row in hl.tile(x.shape[0]):
                values = x[row, :].float()
                maximum = values.amax(-1)
                denominator = torch.exp(values - maximum[:, None]).sum(-1)
                out[row] = denominator.log() + maximum
            return out

        def f(x: torch.Tensor) -> torch.Tensor:
            return row_logsumexp(x) + 1

        compiled = torch.compile(f, fullgraph=True, backend="inductor", dynamic=True)
        torch._dynamo.reset()
        buffer = torch.randn(48, 64, dtype=torch.bfloat16, device=DEVICE)
        with fresh_cache():
            for length in (64, 61):
                with self.subTest(length=length):
                    x = buffer[:, :length]
                    out, codes = run_and_get_code(compiled, x)
                    for code in codes:
                        self.assertNotIn("cute.arch.load(x.iterator", code)
                    expected = torch.logsumexp(x.float(), dim=-1) + 1
                    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    @unittest.skipUnless(
        supports_torch_compile_template_lowering(),
        "this PyTorch build lacks Helion's Inductor template lowering",
    )
    def test_symbolic_inner_stride_under_torch_compile_takes_scalar_loads(
        self,
    ) -> None:
        """A graph viewing its input at a dynamic column step has a symbolic
        inner stride, and Inductor binds the kernel to a placeholder whose
        inner stride is 1 (its sizes stay static).  Eight-lane packets
        proven from that stride would sum eight adjacent columns of the
        step-2 view instead of every other one."""

        @helion.kernel(
            backend="cute",
            static_shapes=False,
            config=helion.Config(
                block_sizes=[1], num_threads=[1, 8], cute_vector_widths=[8, 1]
            ),
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
            for row in hl.tile(x.shape[0]):
                out[row] = x[row, :].float().sum(-1)
            return out

        def f(buffer: torch.Tensor, step: int) -> torch.Tensor:
            return row_sum(buffer.as_strided((48, 64), (128, step))) + 1

        compiled = torch.compile(f, fullgraph=True, backend="inductor", dynamic=True)
        torch._dynamo.reset()
        buffer = torch.randn(48, 128, dtype=torch.bfloat16, device=DEVICE)
        torch._dynamo.mark_static(buffer)
        with fresh_cache():
            for step in (1, 2):
                with self.subTest(step=step):
                    out, codes = run_and_get_code(compiled, buffer, step)
                    for code in codes:
                        self.assertNotIn("cute.arch.load(x.iterator", code)
                    x = buffer.as_strided((48, 64), (128, step))
                    expected = x.float().sum(-1) + 1
                    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    @unittest.skipUnless(
        supports_torch_compile_template_lowering(),
        "this PyTorch build lacks Helion's Inductor template lowering",
    )
    def test_static_graph_under_torch_compile_takes_the_bound_layout_proofs(
        self,
    ) -> None:
        """Inductor's render of a static graph proves what the binding's
        cache key fixes (the bound signature, ``keyed_hint``), short of base
        alignment past 16 bytes: a row logsumexp's register-row copies, a
        matmul's asynchronous operand copies and an in-place state update's
        proven bounds.  While those proofs read live example inputs, which
        no Inductor render has, the copies were scalar and the state
        update's config was refused.  Each graph matches eager."""

        @helion.kernel(
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                block_sizes=[1],
                num_threads=[1, 256],
                reduction_loops=[16384],
                cute_vector_widths=[4, 1],
                cute_lane_layouts=["strided", "blocked"],
                cute_reduction_reloads=["register"],
                cute_cluster_n=2,
                cute_cluster_warp_online=True,
                cute_scalar_epilogue=True,
                cute_register_row=True,
                cute_warp_fp32_redux=True,
                cute_register_row_tensor_copy=True,
            ),
        )
        def row_logsumexp(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
            for row in hl.tile(x.shape[0]):
                values = x[row, :].float()
                maximum = values.amax(-1)
                denominator = torch.exp(values - maximum[:, None]).sum(-1)
                out[row] = denominator.log() + maximum
            return out

        @helion.kernel(
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                block_sizes=[64, 32, 64],
                num_threads=[4, 32, 1],
                cute_collective_copy="async",
                cute_collective_mma=True,
            ),
        )
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            _, n = y.shape
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        @helion.kernel(
            backend="cute",
            static_shapes=False,
            config=helion.Config(
                block_sizes=[128],
                num_threads=[8, 16],
                cute_vector_widths=[8, 1, 1, 1],
                pid_type="xyz",
                cute_proven_bounds=True,
            ),
        )
        def state_update(
            state_indices: torch.Tensor, state: torch.Tensor, packed: torch.Tensor
        ) -> torch.Tensor:
            batches = state_indices.size(0)
            heads = hl.specialize(state.size(1))
            rows = hl.specialize(state.size(2))
            width = hl.specialize(state.size(3))
            out = torch.empty(
                [batches, heads, rows], dtype=state.dtype, device=state.device
            )
            for tile_batch, tile_head, tile_row in hl.tile(
                [batches, heads, rows], block_size=[1, 1, None]
            ):
                batch_index = tile_batch.id
                head_index = tile_head.id
                state_index = state_indices[batch_index]
                cols = hl.arange(width)
                state_values = state[state_index, head_index, tile_row, cols].float()
                packed_values = packed[batch_index, head_index, cols].float()
                updated = state_values + packed_values[None, :]
                out[batch_index, head_index, tile_row] = updated.sum(-1).to(state.dtype)
                state[state_index, head_index, tile_row, cols] = updated
            return out

        def compile_and_run(f: object, *args: torch.Tensor) -> tuple[object, list[str]]:
            compiled = torch.compile(
                f, fullgraph=True, backend="inductor", dynamic=False
            )
            torch._dynamo.reset()
            with fresh_cache():
                return run_and_get_code(compiled, *args)

        with self.subTest(kernel="row_logsumexp"):
            x = torch.randn(3, 32000, device=DEVICE)
            out, codes = compile_and_run(lambda x: row_logsumexp(x) + 1, x)
            self.assertIn("_cute_complete_row_tensor_copy(", "\n".join(codes))
            expected = torch.logsumexp(x, dim=-1) + 1
            torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

        with self.subTest(kernel="matmul"):
            x = torch.randn(256, 256, dtype=torch.bfloat16, device=DEVICE)
            y = torch.randn(256, 256, dtype=torch.bfloat16, device=DEVICE)
            out, codes = compile_and_run(lambda x, y: matmul(x, y) + 1, x, y)
            self.assertEqual("\n".join(codes).count("cp_async_shared_global("), 2)
            expected = (x.float() @ y.float()).to(x.dtype) + 1
            torch.testing.assert_close(out, expected, atol=2e-1, rtol=2e-2)

        with self.subTest(kernel="state_update"):
            state_indices = torch.tensor([2, 0], dtype=torch.int32, device=DEVICE)
            state = torch.randn(3, 3, 128, 128, dtype=torch.bfloat16, device=DEVICE)
            packed = torch.randn(2, 3, 128, dtype=torch.bfloat16, device=DEVICE)
            expected_state = state.clone()
            indices = state_indices.long()
            updated = expected_state[indices].float() + packed.float()[:, :, None, :]
            expected_state[indices] = updated.to(state.dtype)
            expected = updated.sum(-1).to(state.dtype) + 1
            out, codes = compile_and_run(
                lambda i, s, p: state_update(i, s, p) + 1, state_indices, state, packed
            )
            self.assertIn("_cute_proven_bounds_", "\n".join(codes))
            torch.testing.assert_close(out, expected, atol=5e-2, rtol=2e-2)
            torch.testing.assert_close(state, expected_state, atol=0, rtol=0)

    @unittest.skipUnless(
        supports_torch_compile_template_lowering(),
        "this PyTorch build lacks Helion's Inductor template lowering",
    )
    def test_eager_after_torch_compile_runs_the_eager_code(self) -> None:
        """Inductor tunes its configs on a binding of its own.  Its render of
        a row logsumexp proves the input only 16-byte aligned, so the
        eight-lane fp32 packets (32 bytes) stay scalar there; a later eager
        call tunes the kernel's binding and runs the packets ``to_code``
        shows, not the render's scalar loads."""
        from helion._testing.cute_flash import compile_and_source

        @helion.kernel(
            backend="cute",
            static_shapes=True,
            # Two configs, both of eight-lane packets, so that the lowering
            # tunes rather than taking a config the user gave.
            configs=[
                helion.Config(
                    block_sizes=[1],
                    num_threads=[1, num_threads],
                    reduction_loops=[16384],
                    cute_vector_widths=[8, 1],
                )
                for num_threads in (256, 128)
            ],
            autotune_benchmark_subprocess=False,
        )
        def row_logsumexp(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
            for row in hl.tile(x.shape[0]):
                values = x[row, :].float()
                maximum = values.amax(-1)
                denominator = torch.exp(values - maximum[:, None]).sum(-1)
                out[row] = denominator.log() + maximum
            return out

        packet = "cute.arch.load(x.iterator"
        x = torch.randn(3, 32000, device=DEVICE)
        expected = torch.logsumexp(x, dim=-1)
        compiled = torch.compile(
            lambda x: row_logsumexp(x) + 1,
            fullgraph=True,
            backend="inductor",
            dynamic=False,
        )
        torch._dynamo.reset()
        with fresh_cache():
            out, codes = run_and_get_code(compiled, x)
            self.assertNotIn(packet, "\n".join(codes))
            torch.testing.assert_close(out, expected + 1, atol=1e-4, rtol=1e-4)
            bound = row_logsumexp.bind((x,))
            self.assertIsNone(bound._config)
            eager = row_logsumexp(x)
            assert bound._config is not None
            _, source = compile_and_source(bound, bound._config)
            self.assertIn(packet, source)
            torch.testing.assert_close(eager, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
