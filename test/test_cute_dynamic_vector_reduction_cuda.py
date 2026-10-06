"""GPU numerics for dynamic-shape rolled row reductions on the vector path.

Companion to ``test_cute_dynamic_vector_reduction.py``: runs the dynamic-shape
``rms_norm_batched`` example under configs whose vector width divides the row
extent (chunk-uniform mask) and padded-row configs where it does not (vector
interior plus a per-element tail, pipelined or not), checks that inputs with
different size, stride, or base residues bind and run separately through one
kernel object, covers the V-misaligned strided views whose packets used to
fault, the row-mask guard of a partially filled row tile, static extents that
do not divide the block, rows shorter than a chunk, the trace-time fragment
budget's fallback for long rows (also when a static group straddles the guard),
and a max reduction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples.aot_example import rms_norm_batched
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytest.importorskip("cutlass")


def _rms_norm_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Like the example, but writes into a caller-provided (padded) output."""
    m, n = x.size()
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + 1e-5)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _rms_norm_specialized_n(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Dynamic row count over a specialized (static) row length."""
    m, n = x.size()
    hl.specialize(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + eps)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _rms_norm_into_specialized_n(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Dynamic row count into a caller-provided output over a specialized row length."""
    m, n = x.size()
    hl.specialize(n)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + 1e-5)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _dynamic_rows_between_static_column_sweeps(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rolled sweeps over dynamic ``n`` interleaved with static sweeps over ``k``."""
    m, n = x.size()
    _, k = y.size()
    hl.specialize(k)
    block_k = hl.register_block_size(k)
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        sum_sq = torch.sum(x_tile * x_tile, dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            acc = acc + torch.sum(y[tile_m, tile_k].to(torch.float32), dim=-1)
        out_x[tile_m, :] = (x_tile / sum_sq[:, None]).to(out_x.dtype)
        for tile_k in hl.tile(k, block_size=block_k):
            y_tile = y[tile_m, tile_k].to(torch.float32)
            out_y[tile_m, tile_k] = (y_tile / acc[:, None]).to(out_y.dtype)
    return out_x, out_y


def _row_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], device=x.device, dtype=torch.float32)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amax(x[tile_m, :].to(torch.float32), dim=-1)
    return out


def _rolled_config(
    bound: BoundKernel,
    *,
    rows: int,
    threads: int,
    vec: int,
    chunk: int,
) -> helion.Config:
    """Assign the reduction knobs by block id (the positional lists differ per kernel)."""
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    return spec.normalized_config(
        helion.Config(
            block_sizes=[rows],
            num_threads=[
                threads if block_id == reduction else rows
                for block_id in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[chunk],
            cute_vector_widths=[
                vec if block_id == reduction else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            cute_reduction_reloads=["register"],
        )
    )


def _straddle_config(bound: BoundKernel) -> helion.Config:
    """64-row tiles on 4x32 threads, V=8 rolled ``n`` sweeps, ``k`` tiles of one."""
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    return spec.normalized_config(
        helion.Config(
            block_sizes=[64, 1],
            num_threads=[
                {reduction: 32, 0: 4}.get(block_id, 1)
                for block_id in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[1024],
            cute_vector_widths=[
                8 if block_id == reduction else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            cute_reduction_reloads=["register"],
        )
    )


def _kernel(fn=rms_norm_batched.fn, *, static_shapes: bool = False) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _reference(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x32 = x.float()
    return (x32 / torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)).to(
        x.dtype
    )


@onlyBackends(["cute"])
class TestCuteDynamicVectorReduction(TestCase):
    @skipIfNotCUDA()
    def test_divisible_extent_matches_reference(self) -> None:
        # Study config: 256 threads, V=2, 2048-element chunks; 4096 % 2 == 0.
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[1],
            num_threads=[1, 256],
            reduction_loops=[2048],
            cute_vector_widths=[2, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-2, rtol=1e-2)
        self.assertIn("mask_1 = reduction_lane_base_1 < n", code)
        self.assertIn("_REDUCTION_TRIPS_1: cutlass.Constexpr", code)
        self.assertIn("_cute_store_u16_vec(out.iterator", code)

    @skipIfNotCUDA()
    def test_divisible_v8_row_is_pipelined(self) -> None:
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-2, rtol=1e-2)
        self.assertIn("_pipe_load_0 = cute.arch.load(", code)
        self.assertIn("ir.VectorType.get([8], cutlass.Uint16.mlir_type)", code)

    @skipIfNotCUDA()
    def test_padded_rows_use_vector_interior_and_scalar_tail(self) -> None:
        # 4100 % 8 == 4 with 16-byte aligned (4104-wide) rows: the last V=8
        # chunk of every row straddles the extent.
        x = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)[:, :4100]
        out = torch.zeros(64, 4104, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            _kernel(_rms_norm_into),
            (x, out[:, :4100]),
            block_sizes=[1],
            num_threads=[1, 64],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(result, _reference(x), atol=1e-2, rtol=1e-2)
        # The padding columns are never written.
        self.assertEqual(out[:, 4100:].abs().sum().item(), 0.0)
        self.assertIn("reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n", code)
        self.assertIn("if not reduction_chunk_full_2 and (mask_0 and mask_1):", code)
        self.assertIn("_cute_store_u16_vec(out.iterator", code)

    @skipIfNotCUDA()
    def test_rows_narrower_than_a_chunk_and_partial_grid(self) -> None:
        # Fewer fp32 elements than one 1024-wide chunk, a row count that leaves
        # the last thread block partially masked, and rows that are only
        # 8-byte aligned (1030 * 4 bytes): V=2 packets apply, V=4 must not.
        x = torch.randn(13, 1030, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[4],
            num_threads=[4, 64],
            reduction_loops=[1024],
            cute_vector_widths=[2, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-4, rtol=1e-4)
        self.assertIn("ir.VectorType.get([2], cutlass.Uint32.mlir_type)", code)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[4],
            num_threads=[4, 64],
            reduction_loops=[1024],
            cute_vector_widths=[4, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-4, rtol=1e-4)
        self.assertNotIn("cute.arch.load(", code)

    @skipIfNotCUDA()
    def test_fp32_rows_vectorize_with_v4(self) -> None:
        x = torch.randn(32, 2048, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[1],
            num_threads=[1, 256],
            reduction_loops=[1024],
            cute_vector_widths=[4, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-4, rtol=1e-4)
        self.assertIn("ir.VectorType.get([4], cutlass.Uint32.mlir_type)", code)
        self.assertIn("_cute_store_u32_vec(out.iterator", code)

    @skipIfNotCUDA()
    def test_misaligned_views_fall_back_to_scalar_and_run(self) -> None:
        # Regression: these strided views used to emit LDG.128 packets at
        # 8-byte or 2-byte aligned addresses and fault, for static shapes too.
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        for static_shapes in (True, False):
            stride_view = torch.randn(64, 4100, device=DEVICE, dtype=torch.bfloat16)
            base_view = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)
            for x in (stride_view[:, :4096], base_view[:, 1:4097]):
                bound = _kernel(static_shapes=static_shapes).bind((x, 1e-5))
                bound.set_config(config)
                self.assertNotIn("cute.arch.load(", bound.to_code(config))
                torch.testing.assert_close(
                    bound(x, 1e-5), _reference(x), atol=1e-2, rtol=1e-2
                )

    @skipIfNotCUDA()
    def test_size_residue_rebinds_and_both_shapes_run(self) -> None:
        # ``n % 8`` is part of the bound cache key: 4096 (chunk-uniform V=8)
        # and 4100 (8-byte rows, scalar) bind separately and both stay
        # correct in either call order; 2056 shares the residue-0 bound.
        kernel = _kernel()
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        bounds = []
        for width in (4096, 4100, 4096, 2056):
            x = torch.randn(8, width, device=DEVICE, dtype=torch.bfloat16)
            bound = kernel.bind((x, 1e-5))
            bound.set_config(config)
            torch.testing.assert_close(
                bound(x, 1e-5), _reference(x), atol=1e-2, rtol=1e-2
            )
            bounds.append(bound)
        self.assertIs(bounds[0], bounds[2])
        self.assertIs(bounds[0], bounds[3])
        self.assertIsNot(bounds[0], bounds[1])

    @skipIfNotCUDA()
    def test_partial_row_tile_never_reads_rows_past_the_tensor(self) -> None:
        # 13 rows in tiles of 4 over a 4096-wide row that divides the block:
        # the roll needs no bounds mask, so only the row mask keeps the packets
        # of rows 13-15 off memory.  Input and output are views of larger
        # buffers whose rows 13-15 are NaN: the values a row-masked lane loads
        # are gated and its stores are predicated, so no NaN may reach rows
        # 0-12 of the output and rows 13-15 of the output buffer must stay
        # untouched.  A packet pointer that ignored the row mask would read
        # past the tensor instead, which is observable only as a fault, so the
        # pointer selection is asserted on the generated code.
        for static_shapes, fn in (
            (True, _rms_norm_into),
            (False, _rms_norm_into_specialized_n),
        ):
            big = torch.full(
                (16, 4096), float("nan"), device=DEVICE, dtype=torch.bfloat16
            )
            big[:13] = torch.randn(13, 4096, device=DEVICE, dtype=torch.bfloat16)
            out_buffer = torch.full(
                (16, 4096), float("nan"), device=DEVICE, dtype=torch.bfloat16
            )
            x = big[:13]
            code, out = code_and_output(
                _kernel(fn, static_shapes=static_shapes),
                (x, out_buffer[:13]),
                block_sizes=[4],
                num_threads=[4, 64],
                reduction_loops=[1024],
                cute_vector_widths=[8, 1],
                cute_reduction_reloads=["register"],
            )
            packet_loads = [
                line for line in code.splitlines() if "cute.arch.load(" in line
            ]
            self.assertTrue(packet_loads)
            for line in packet_loads:
                self.assertIn(" if mask_0 else x.iterator + cutlass.Int32(0)", line)
            self.assertNotIn("mask_1", code)
            self.assertTrue(torch.isfinite(out).all().item())
            torch.testing.assert_close(out, _reference(x), atol=1e-2, rtol=1e-2)
            self.assertTrue(torch.isnan(out_buffer[13:]).all().item())

    @skipIfNotCUDA()
    def test_static_extents_not_a_multiple_of_the_block_take_the_chunk_forms(
        self,
    ) -> None:
        # Static row lengths that do not divide the block used to keep every
        # access scalar; they now take the chunk-level forms.
        x = torch.randn(64, 4100, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _kernel(static_shapes=True),
            (x, 1e-5),
            block_sizes=[1],
            num_threads=[1, 256],
            reduction_loops=[2048],
            cute_vector_widths=[4, 1],
            cute_reduction_reloads=["register"],
        )
        self.assertIn("mask_1 = reduction_lane_base_1 < 4100", code)
        self.assertIn("ir.VectorType.get([4], cutlass.Uint16.mlir_type)", code)
        torch.testing.assert_close(out, _reference(x), atol=1e-2, rtol=1e-2)
        x = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)[:, :4100]
        padded = torch.zeros(64, 4104, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            _kernel(_rms_norm_into, static_shapes=True),
            (x, padded[:, :4100]),
            block_sizes=[1],
            num_threads=[1, 64],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        self.assertIn("reduction_chunk_full_1 = reduction_lane_base_1 + 7 < 4100", code)
        self.assertIn("if not reduction_chunk_full_2 and", code)
        torch.testing.assert_close(result, _reference(x), atol=1e-2, rtol=1e-2)
        self.assertEqual(padded[:, 4100:].abs().sum().item(), 0.0)

    @skipIfNotCUDA()
    def test_fragment_cap_falls_back_to_reloads_for_long_rows(self) -> None:
        # Bound at n=4096 (two trips) the register cache holds TRIPS * 32
        # elements under a trace-time budget of the 64-element size-hint
        # fragment.  The same bound kernel run at n=73728 (36 trips) takes the
        # pre-fusion sweeps instead of growing the dynamically indexed local
        # array past the footprint the binding was admitted with.
        kernel = _kernel()
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 64],
            reduction_loops=[2048],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        x = torch.randn(4, 4096, device=DEVICE, dtype=torch.bfloat16)
        bound = kernel.bind((x, 1e-5))
        bound.set_config(config)
        code = bound.to_code(config)
        self.assertIn("if cutlass.const_expr(_REDUCTION_TRIPS_1 * 32 <= 64):", code)
        self.assertIn(
            "_fuse_cache_0 = cute.make_rmem_tensor(_REDUCTION_TRIPS_1 * 32, cutlass.Uint16)",
            code,
        )
        torch.testing.assert_close(bound(x, 1e-5), _reference(x), atol=1e-2, rtol=1e-2)
        long_x = torch.randn(4, 73728, device=DEVICE, dtype=torch.bfloat16)
        self.assertIs(kernel.bind((long_x, 1e-5)), bound)
        torch.testing.assert_close(
            bound(long_x, 1e-5), _reference(long_x), atol=1e-2, rtol=1e-2
        )

    @skipIfNotCUDA()
    def test_static_group_straddling_the_guard_runs_whole_in_both_branches(
        self,
    ) -> None:
        # The static ``k`` group populates its cache between the rolled ``n``
        # sweeps and consumes it after them.  The guard covers the whole ``k``
        # group, so the fallback (8 or 72 trips over the four-trip budget of
        # the 4096 binding) reloads both groups instead of consuming a cache
        # that only the fused branch fills.
        kernel = _kernel(_dynamic_rows_between_static_column_sweeps)
        y = torch.rand(64, 256, device=DEVICE, dtype=torch.bfloat16) + 0.5
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        bound = kernel.bind((x, y))
        config = _straddle_config(bound)
        bound.set_config(config)
        code = bound.to_code(config)
        self.assertIn("if cutlass.const_expr(_REDUCTION_TRIPS_2 * 32 <= 128):", code)
        self.assertIn(
            "_fuse_cache_1 = cute.make_rmem_tensor(64, cutlass.BFloat16)", code
        )
        y32 = y.float()
        expected_y = (y32 / y32.sum(dim=-1, keepdim=True)).to(y.dtype)

        def check(x: torch.Tensor) -> None:
            out_x, out_y = bound(x, y)
            x32 = x.float()
            expected_x = (x32 / (x32 * x32).sum(dim=-1, keepdim=True)).to(x.dtype)
            torch.testing.assert_close(
                out_x.float(), expected_x.float(), atol=1e-7, rtol=2e-2
            )
            torch.testing.assert_close(
                out_y.float(), expected_y.float(), atol=1e-6, rtol=2e-2
            )

        # The binding shape takes the fused branch.  A compiler seed of this
        # nested tile nest keys exact input metadata, so ``kernel.bind`` would
        # rebind longer rows; the bound kernel itself is compiled for dynamic
        # shapes and every fact behind its code (size residues, alignment,
        # disjoint storage) still holds, so it is launched directly and takes
        # the fallback.
        check(x)
        bound_key = kernel.specialization_key((x, y))
        for width in (8192, 73728):
            x = torch.randn(64, width, device=DEVICE, dtype=torch.bfloat16)
            (changed,) = [
                entry
                for entry, other in zip(
                    bound_key, kernel.specialization_key((x, y)), strict=True
                )
                if entry != other
            ]
            self.assertEqual(changed[0], "compiler_seed_results")
            check(x)

    @skipIfNotCUDA()
    def test_rows_shorter_than_a_chunk_reuse_the_bound_kernel(self) -> None:
        # A 520-wide row is shorter than the 1024 block a 4096-wide binding
        # chose: one partial trip whose chunk predicates carry the whole roll.
        # 520 shares the ``n % 8 == 0`` residue (chunk-level mask); 516 of a
        # 520-wide buffer shares the residue-4, aligned-stride class of 4100 of
        # 4104 (vector interior with a per-element tail).
        kernel = _kernel(_rms_norm_into)
        config = helion.Config(
            block_sizes=[4],
            num_threads=[4, 64],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        for width, stride, short_width, short_stride, predicate in (
            (4096, 4096, 520, 520, "mask_1 = reduction_lane_base_1 < n"),
            (
                4100,
                4104,
                516,
                520,
                "reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n",
            ),
        ):
            x = torch.randn(13, stride, device=DEVICE, dtype=torch.bfloat16)[:, :width]
            out = torch.zeros(13, stride, device=DEVICE, dtype=torch.bfloat16)
            bound = kernel.bind((x, out[:, :width]))
            bound.set_config(config)
            self.assertIn(predicate, bound.to_code(config))
            torch.testing.assert_close(
                bound(x, out[:, :width]), _reference(x), atol=1e-2, rtol=1e-2
            )
            short = torch.randn(13, short_stride, device=DEVICE, dtype=torch.bfloat16)[
                :, :short_width
            ]
            short_out = torch.zeros(
                13, short_stride, device=DEVICE, dtype=torch.bfloat16
            )
            self.assertIs(kernel.bind((short, short_out[:, :short_width])), bound)
            torch.testing.assert_close(
                bound(short, short_out[:, :short_width]),
                _reference(short),
                atol=1e-2,
                rtol=1e-2,
            )
            self.assertEqual(short_out[:, short_width:].abs().sum().item(), 0.0)

    @skipIfNotCUDA()
    def test_pipelined_tail_form_matches_reference(self) -> None:
        # 128 threads x V=8 cover the 1024 chunk in one lane iteration, so the
        # tail form is software pipelined: the prefetch is guarded on its last
        # lane and the whole-chunk predicate follows the snapshot.
        x = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)[:, :4100]
        padded = torch.zeros(64, 4104, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            _kernel(_rms_norm_into),
            (x, padded[:, :4100]),
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        self.assertIn("_pipe_load_0 = cute.arch.load(", code)
        self.assertIn("if mask_0 and _pipe_lane_base_0 + 7 < n else", code)
        self.assertIn("reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n", code)
        torch.testing.assert_close(result, _reference(x), atol=1e-2, rtol=1e-2)
        self.assertEqual(padded[:, 4100:].abs().sum().item(), 0.0)

    @skipIfNotCUDA()
    def test_alignment_facts_rebind_the_same_kernel(self) -> None:
        # One kernel object sees an aligned input (vector path), a view whose
        # row stride is not a multiple of V, and a view whose base is 2 bytes
        # off; each alignment class binds separately, the misaligned ones stay
        # scalar, and the aligned input returns to its original binding.
        kernel = _kernel()
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        aligned = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        stride_view = torch.randn(64, 4100, device=DEVICE, dtype=torch.bfloat16)[
            :, :4096
        ]
        base_view = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)[
            :, 1:4097
        ]
        bounds = []
        vectorized = []
        for x in (aligned, stride_view, base_view, aligned):
            bound = kernel.bind((x, 1e-5))
            bound.set_config(config)
            bounds.append(bound)
            vectorized.append("cute.arch.load(" in bound.to_code(config))
            torch.testing.assert_close(
                bound(x, 1e-5), _reference(x), atol=1e-2, rtol=1e-2
            )
        self.assertEqual(vectorized, [True, False, False, True])
        self.assertIs(bounds[0], bounds[3])
        self.assertEqual(len({id(bound) for bound in bounds[:3]}), 3)

    @skipIfNotCUDA()
    def test_row_max_reduction_matches_reference(self) -> None:
        # A max reduction over both chunk forms; every element is negative so
        # a masked lane leaking its zero placeholder would win the max.
        kernel = _kernel(_row_max)
        for x, predicate in (
            (
                torch.randn(13, 4096, device=DEVICE, dtype=torch.bfloat16) - 5.0,
                "mask_1 = reduction_lane_base_1 < n",
            ),
            (
                (torch.randn(13, 4104, device=DEVICE, dtype=torch.bfloat16) - 5.0)[
                    :, :4100
                ],
                "reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n",
            ),
        ):
            bound = kernel.bind((x,))
            config = _rolled_config(bound, rows=4, threads=64, vec=8, chunk=1024)
            bound.set_config(config)
            code = bound.to_code(config)
            self.assertIn("ir.VectorType.get([8], cutlass.Uint16.mlir_type)", code)
            self.assertIn(predicate, code)
            torch.testing.assert_close(bound(x), x.float().amax(dim=-1))
