from __future__ import annotations

import types
import unittest

import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import assert_ref_eager_mode
from helion._testing import code_and_output
from helion._testing import skipIfPallasInterpret
from helion._testing import skipUnlessPallas
import helion.language as hl


class TestTileMaxExtent(TestCase):
    def test_ref_eager_ignores_max_extent(self) -> None:
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def copy_inner(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(1, x.size(0) - 1, block_size=2, max_extent=8):
                out[tile] = x[tile] + 1
            out[0] = x[0]
            out[x.size(0) - 1] = x[x.size(0) - 1]
            return out

        with assert_ref_eager_mode():
            x = torch.arange(8, device="cpu", dtype=torch.float32)
            result = copy_inner(x)
            expected = x.clone()
            expected[1:-1] += 1
            torch.testing.assert_close(result, expected)

    def test_max_extent_rejects_tensor_value(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def bad_max_extent(x: torch.Tensor, max_extent: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(0, x.size(0), max_extent=max_extent):
                out[tile] = x[tile]
            return out

        args = (
            torch.randn(8, device=DEVICE),
            torch.tensor(8, device=DEVICE),
        )
        with self.assertRaisesRegex(
            exc.IncorrectTileUsage,
            r"hl\.tile\(max_extent=\.\.\.\) must be a statically known integer",
        ):
            bad_max_extent.bind(args)

    def test_max_extent_rejects_multidim_tile(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def bad_multidim(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([x.size(0), x.size(1)], max_extent=8):
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        with self.assertRaisesRegex(
            exc.IncorrectTileUsage,
            "max_extent=.*one-dimensional tile loops",
        ):
            bad_multidim.bind((torch.randn(8, 8, device=DEVICE),))

    def test_reused_block_size_conflicting_max_extent_rejects(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def conflicting_max_extent(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            block = hl.register_block_size(x.size(1))
            for row in hl.tile(x.size(0)):
                for tile in hl.tile(0, x.size(1), block_size=block, max_extent=8):
                    out[row, tile] = x[row, tile]
                for tile in hl.tile(0, x.size(1), block_size=block, max_extent=16):
                    out[row, tile] = x[row, tile]
            return out

        with self.assertRaisesRegex(
            exc.IncorrectTileUsage,
            "Conflicting hl.tile\\(max_extent=\\.\\.\\.\\) values",
        ):
            conflicting_max_extent.bind((torch.randn(4, 8, device=DEVICE),))

    def test_backed_symbolic_max_extent_compiles(self) -> None:
        @helion.kernel(
            backend="pallas",
            static_shapes=False,
            autotune_effort="none",
        )
        def symbolic_max_extent(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                for tile in hl.tile(0, x.size(1), max_extent=x.size(1)):
                    out[row, tile] = x[row, tile]
            return out

        bound = symbolic_max_extent.bind((torch.randn(4, 8, device=DEVICE),))
        code = bound.to_triton_code(
            helion.Config(block_sizes=[2, 4], pallas_loop_type="fori_loop")
        )
        self.assertIn("jax.lax.fori_loop", code)

    def test_outer_pipeline_symbolic_max_extent_static_shapes_false_rejects(
        self,
    ) -> None:
        @helion.kernel(
            backend="pallas",
            static_shapes=False,
            autotune_effort="none",
        )
        def symbolic_max_extent(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                for tile in hl.tile(0, x.size(1), max_extent=x.size(1)):
                    out[row, tile] = x[row, tile]
            return out

        bound = symbolic_max_extent.bind((torch.randn(4, 8, device=DEVICE),))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "requires a static folded",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[2, 4], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_static_shapes_false_int_max_extent_compiles(self) -> None:
        @helion.kernel(
            backend="pallas",
            static_shapes=False,
            autotune_effort="none",
        )
        def integer_max_extent(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                for tile in hl.tile(0, x.size(1), max_extent=8):
                    out[row, tile, :] = x[row, tile, :]
            return out

        bound = integer_max_extent.bind((torch.randn(4, 8, 128, device=DEVICE),))
        code = bound.to_triton_code(
            helion.Config(block_sizes=[2, 8], pallas_loop_type="outer_pipeline")
        )
        self.assertIn("pltpu.emit_pipeline", code)


@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasOuterPipelineScaffold(TestCase):
    def test_outer_pipeline_is_explicit_but_not_autotuned(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def inner_loop(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                for tile in hl.tile(0, x.size(1), max_extent=x.size(1)):
                    out[row, tile] = x[row, tile]
            return out

        x = torch.randn(8, 128, device=DEVICE)
        bound = inner_loop.bind((x,))
        choices = bound.config_spec._flat_fields()["pallas_loop_type"].choices
        self.assertNotIn("outer_pipeline", choices)

        code, result = code_and_output(
            inner_loop,
            (x,),
            block_sizes=[8, 128],
            pallas_loop_type="outer_pipeline",
        )
        torch.testing.assert_close(result, x)
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn("_launcher(_helion_inner_loop, (),", code)
        self.assertIn("grid=((8 + _BLOCK_SIZE_0 - 1) // _BLOCK_SIZE_0, 1)", code)
        self.assertIn("lambda _o0, _j", code)
        self.assertNotIn("pl.program_id", code)

    @skipIfPallasInterpret("Pallas interpret requires static BoundedSlice extents")
    def test_outer_pipeline_dynamic_end_uses_max_extent_guard(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def prefix_copy(x: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for batch in hl.tile(x.size(0)):
                end = hl.load(ends, [0])
                for col in hl.tile(0, end, max_extent=x.size(1)):
                    out[batch, col, :, :] = x[batch, col, :, :]
            return out

        x = torch.randn(8, 128, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        end = 37
        ends = torch.tensor([end], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            prefix_copy,
            (x, ends),
            block_sizes=[8, 128],
            pallas_loop_type="outer_pipeline",
        )

        expected = torch.zeros_like(x)
        expected[:, :end] = x[:, :end]
        torch.testing.assert_close(result, expected)
        self.assertIn("jnp.maximum(0, jnp.minimum", code)
        self.assertIn("lax.cond", code)

    @skipIfPallasInterpret("Pallas interpret requires static BoundedSlice extents")
    def test_outer_pipeline_dynamic_end_clamps_masked_store_lanes(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def prefix_copy_into(
            x: torch.Tensor, ends: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            for batch in hl.tile(x.size(0)):
                end = hl.load(ends, [0])
                for col in hl.tile(0, end, max_extent=x.size(1)):
                    out[batch, col, :, :] = x[batch, col, :, :]
            return out

        x = torch.randn(8, 128, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        end = 37
        ends = torch.tensor([end], device=DEVICE, dtype=torch.int32)
        out = torch.full_like(x, -7.0)
        code, result = code_and_output(
            prefix_copy_into,
            (x, ends, out),
            block_sizes=[8, 128],
            pallas_loop_type="outer_pipeline",
        )

        expected = torch.full_like(x, -7.0)
        expected[:, :end] = x[:, :end]
        torch.testing.assert_close(result, expected)
        self.assertNotIn("out_preserve_hbm", code)
        self.assertNotIn("_outer_pipeline_preserve_arg_indices", code)
        self.assertIn("jnp.maximum(0, jnp.minimum", code)

    def test_outer_pipeline_dynamic_innermost_dim_rejects(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def prefix_copy(x: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for row in hl.tile(x.size(0)):
                end = hl.load(ends, [0])
                for col in hl.tile(0, end, max_extent=x.size(1)):
                    out[row, col] = x[row, col]
            return out

        x = torch.randn(8, 128, device=DEVICE)
        ends = torch.tensor([37], device=DEVICE, dtype=torch.int32)
        bound = prefix_copy.bind((x, ends))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "innermost tensor dimension is not supported",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[8, 128], pallas_loop_type="outer_pipeline")
            )

    @skipIfPallasInterpret("Pallas interpret requires static BoundedSlice extents")
    def test_outer_pipeline_zero_begin_multitile_clamps_load_and_store_extents(
        self,
    ) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def prefix_copy_into(
            x: torch.Tensor, ends: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            for batch in hl.tile(x.size(0)):
                end = hl.load(ends, [0])
                for col in hl.tile(0, end, max_extent=x.size(1)):
                    out[batch, col, :, :] = x[batch, col, :, :]
            return out

        x = torch.randn(1, 3000, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        end = 2500
        ends = torch.tensor([end], device=DEVICE, dtype=torch.int32)
        out = torch.full_like(x, -7.0)
        code, result = code_and_output(
            prefix_copy_into,
            (x, ends, out),
            block_sizes=[1, 1024],
            pallas_loop_type="outer_pipeline",
        )

        expected = torch.full_like(x, -7.0)
        expected[:, :end] = x[:, :end]
        torch.testing.assert_close(result, expected)
        emit_line = next(
            line for line in code.splitlines() if "pltpu.emit_pipeline" in line
        )
        self.assertIn("grid=(1, 3)", emit_line)
        self.assertIn("jnp.maximum(0, jnp.minimum", emit_line)
        self.assertNotIn("_ds_pad_dims", code)
        self.assertNotIn("out_preserve_hbm", code)

    @skipIfPallasInterpret("Pallas interpret requires static BoundedSlice extents")
    def test_outer_pipeline_full_dim_nondivisible_uses_dynamic_ds(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def copy_full_dim(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for batch in hl.tile(x.size(0)):
                for col in hl.tile(0, x.size(1), max_extent=x.size(1)):
                    out[batch, col, :, :] = x[batch, col, :, :]
            return out

        x = torch.randn(1, 10, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            copy_full_dim,
            (x,),
            block_sizes=[1, 8],
            pallas_loop_type="outer_pipeline",
        )

        torch.testing.assert_close(result, x)
        emit_line = next(
            line for line in code.splitlines() if "pltpu.emit_pipeline" in line
        )
        self.assertIn("grid=(1, 2)", emit_line)
        self.assertIn("pl.BoundedSlice(_BLOCK_SIZE_1)", emit_line)
        self.assertIn("jnp.maximum(0, jnp.minimum", emit_line)
        self.assertNotIn("_ds_pad_dims", code)

    @skipIfPallasInterpret("Pallas interpret requires static BoundedSlice extents")
    def test_outer_pipeline_data_dependent_nonzero_begin_uses_lambda_scope(
        self,
    ) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def packed_copy(
            x: torch.Tensor, offsets: torch.Tensor, max_len: int
        ) -> torch.Tensor:
            max_len = hl.specialize(max_len)
            out = torch.zeros_like(x)
            for seq in hl.grid(offsets.size(0) - 1):
                start = offsets[seq]
                end = offsets[seq + 1]
                for tile in hl.tile(start, end, max_extent=max_len):
                    out[tile, :, :] = x[tile, :, :]
            return out

        x = torch.randn(64, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        offsets = torch.tensor([3, 11, 19, 27], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            packed_copy,
            (x, offsets, 8),
            block_sizes=[8],
            pallas_loop_type="outer_pipeline",
        )

        expected = torch.zeros_like(x)
        expected[3:27] = x[3:27]
        torch.testing.assert_close(result, expected)
        emit_line = next(
            line for line in code.splitlines() if "pltpu.emit_pipeline" in line
        )
        self.assertIn("offsets", emit_line)
        self.assertIn("_o0", emit_line)
        self.assertNotIn("start", emit_line)
        self.assertNotIn("end", emit_line)
        self.assertNotIn("_ds_pad_dims", code)
        self.assertIn("load = x_vmem[:, :, :] *", code)

    @skipIfPallasInterpret("Pallas interpret requires static BoundedSlice extents")
    def test_outer_pipeline_masks_partial_folded_load(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def partial_load(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for _seq in hl.grid(1):
                for tile in hl.tile(0, 2, block_size=4, max_extent=4):
                    out[tile, :, :] = x[tile, :, :] + 1
            return out

        x = torch.randn(4, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            partial_load,
            (x,),
            pallas_loop_type="outer_pipeline",
        )

        expected = torch.zeros_like(x)
        expected[:2] = x[:2] + 1
        torch.testing.assert_close(result, expected)
        self.assertIn("grid=(1, 1)", code)
        self.assertIn("mask_", code)
        self.assertIn("load = x_vmem[:, :, :] *", code)

    @skipIfPallasInterpret("Pallas interpret requires static BoundedSlice extents")
    def test_outer_pipeline_unit_block_overlaunch_uses_zero_extent_ds(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def packed_copy_unit_block(
            x: torch.Tensor, offsets: torch.Tensor, max_len: int
        ) -> torch.Tensor:
            max_len = hl.specialize(max_len)
            out = torch.zeros_like(x)
            for seq in hl.grid(offsets.size(0) - 1):
                start = offsets[seq]
                end = offsets[seq + 1]
                for tile in hl.tile(start, end, block_size=1, max_extent=max_len):
                    out[tile, :, :] = x[tile, :, :]
            return out

        x = torch.randn(2, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        offsets = torch.tensor([0, 2], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            packed_copy_unit_block,
            (x, offsets, 4),
            pallas_loop_type="outer_pipeline",
        )

        torch.testing.assert_close(result, x)
        emit_line = next(
            line for line in code.splitlines() if "pltpu.emit_pipeline" in line
        )
        self.assertIn("grid=(1, 4)", emit_line)
        self.assertIn("pl.BoundedSlice", emit_line)
        self.assertIn("pl.ds(", emit_line)
        self.assertIn("jnp.maximum(0, jnp.minimum", emit_line)

    @skipIfPallasInterpret("Pallas interpret requires static BoundedSlice extents")
    def test_outer_pipeline_multitile_overlaunch_clamps_extents_without_ds_pad(
        self,
    ) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def packed_copy_into(
            x: torch.Tensor,
            offsets: torch.Tensor,
            out: torch.Tensor,
            max_len: int,
        ) -> torch.Tensor:
            max_len = hl.specialize(max_len)
            for seq in hl.grid(offsets.size(0) - 1):
                start = offsets[seq]
                end = offsets[seq + 1]
                for tile in hl.tile(start, end, max_extent=max_len):
                    out[tile, :, :] = x[tile, :, :]
            return out

        x = torch.randn(512, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        offsets = torch.tensor([3, 259, 388, 390], device=DEVICE, dtype=torch.int32)
        out = torch.full_like(x, -7.0)
        code, result = code_and_output(
            packed_copy_into,
            (x, offsets, out, 256),
            block_sizes=[128],
            pallas_loop_type="outer_pipeline",
        )

        expected = torch.full_like(x, -7.0)
        expected[3:390] = x[3:390]
        torch.testing.assert_close(result, expected)
        emit_line = next(
            line for line in code.splitlines() if "pltpu.emit_pipeline" in line
        )
        self.assertIn("grid=(3, 2)", emit_line)
        self.assertIn("jnp.maximum(0, jnp.minimum", emit_line)
        self.assertNotIn("_ds_pad_dims", code)
        self.assertNotIn("out_preserve_hbm", code)
        self.assertNotIn("_outer_pipeline_preserve_arg_indices", code)

    @skipIfPallasInterpret("Pallas interpret requires static BoundedSlice extents")
    def test_outer_pipeline_outer_prologue_uses_folded_offset(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def add_sequence_bias(
            x: torch.Tensor,
            bias: torch.Tensor,
            offsets: torch.Tensor,
            max_len: int,
        ) -> torch.Tensor:
            max_len = hl.specialize(max_len)
            out = torch.zeros_like(x)
            for seq in hl.grid(offsets.size(0) - 1):
                start = offsets[seq]
                end = offsets[seq + 1]
                bias_blk = bias[seq, :, :]
                for tile in hl.tile(start, end, max_extent=max_len):
                    out[tile, :, :] = x[tile, :, :] + bias_blk[None, :, :]
            return out

        x = torch.randn(24, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(3, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        offsets = torch.tensor([0, 8, 16, 24], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            add_sequence_bias,
            (x, bias, offsets, 8),
            block_sizes=[8],
            pallas_loop_type="outer_pipeline",
        )

        expected = torch.empty_like(x)
        expected[0:8] = x[0:8] + bias[0]
        expected[8:16] = x[8:16] + bias[1]
        expected[16:24] = x[16:24] + bias[2]
        torch.testing.assert_close(result, expected)
        uncommented_code = "\n".join(
            line for line in code.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertIn("bias_vmem[0", uncommented_code)
        self.assertIn("pl.BlockSpec((1, 8, 128), lambda _o0, _j", uncommented_code)
        self.assertNotIn("bias[0, :, :]", uncommented_code)

    def test_outer_pipeline_stepped_root_grid_uses_axis_step(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def copy_even_rows(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for row in hl.grid(0, x.size(0), 2):
                for col in hl.tile(0, x.size(1), max_extent=x.size(1)):
                    out[row, col, :] = x[row, col, :]
            return out

        x = torch.randn(8, 8, 128, device=DEVICE)
        code, result = code_and_output(
            copy_even_rows,
            (x,),
            block_size=8,
            pallas_loop_type="outer_pipeline",
        )

        expected = torch.zeros_like(x)
        expected[0::2] = x[0::2]
        torch.testing.assert_close(result, expected)
        uncommented_code = "\n".join(
            line for line in code.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertIn("0 + _o0 * 2", uncommented_code)
        self.assertIn("pl.BoundedSlice(1)", uncommented_code)

    def test_outer_pipeline_dynamic_end_without_max_extent_rejects(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def prefix_copy(x: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for row in hl.tile(x.size(0)):
                end = hl.load(ends, [0])
                for col in hl.tile(0, end):
                    out[row, col] = x[row, col]
            return out

        x = torch.randn(8, 128, device=DEVICE)
        ends = torch.tensor([37], device=DEVICE, dtype=torch.int32)
        bound = prefix_copy.bind((x, ends))
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            r"outer_pipeline over a data-dependent hl\.tile\(begin, end\) requires",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[8, 128], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_captures_outer_dependent_prologue(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def scale_by_row(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for row in hl.tile(x.size(0)):
                scale = (row.index + 1).to(torch.float32)
                for col in hl.tile(0, x.size(1), max_extent=x.size(1)):
                    out[row, col] = x[row, col] * scale[:, None]
            return out

        x = torch.randn(8, 128, device=DEVICE)
        code, result = code_and_output(
            scale_by_row,
            (x,),
            block_sizes=[8, 128],
            pallas_loop_type="outer_pipeline",
        )

        expected = x * (torch.arange(8, device=DEVICE, dtype=x.dtype) + 1)[:, None]
        torch.testing.assert_close(result, expected)
        uncommented_code = "\n".join(
            line for line in code.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertIn("def _pipeline_body", uncommented_code)
        self.assertIn("indices_0 = offset_0 + jnp.arange", uncommented_code)
        self.assertLess(
            uncommented_code.index("def _pipeline_body"),
            uncommented_code.index("indices_0 = offset_0 + jnp.arange"),
        )
        self.assertLess(
            uncommented_code.index("indices_0 = offset_0 + jnp.arange"),
            uncommented_code.index("def _valid_pipeline_body"),
        )

    def test_outer_pipeline_rejects_tile_vector_prologue_tensor_access(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def add_row_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                bias_blk = bias[row, :, :]
                for col in hl.tile(0, x.size(1), max_extent=x.size(1)):
                    out[row, col, :, :] = x[row, col, :, :] + bias_blk[:, None, :, :]
            return out

        x = torch.randn(4, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(4, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = add_row_bias.bind((x, bias))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "Tile-vector outer indices are not supported",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[4, 16], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_rejects_multiple_top_level_grids(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def two_grid_loops(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out_x = torch.empty_like(x)
            out_y = torch.empty_like(y)
            for tile in hl.tile(x.size(0)):
                out_x[tile] = x[tile]
            for tile in hl.tile(y.size(0)):
                out_y[tile] = y[tile]
            return out_x, out_y

        args = (torch.randn(8, device=DEVICE), torch.randn(8, device=DEVICE))
        bound = two_grid_loops.bind(args)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "outer_pipeline currently supports exactly one top-level grid",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[4, 4], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_rejects_unsupported_prologue_tensor_slice(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def partial_prologue_slice(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for row in hl.grid(x.size(0)):
                bias = x[row, 0, :, :]
                for col in hl.tile(16):
                    out[row, col, :, :] = x[row, col, :, :] + bias
            return out

        x = torch.randn(4, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = partial_prologue_slice.bind((x,))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "unsupported outer_pipeline prologue tensor access",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[16], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_rejects_post_fold_grid_body_statement(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def post_fold_store(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.grid(x.size(0)):
                for col in hl.tile(x.size(1)):
                    out[row, col, :, :] = x[row, col, :, :]
                out[row, 0, :, :] = x[row, 0, :, :]
            return out

        x = torch.randn(4, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = post_fold_store.bind((x,))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "folded hl.tile loop to be the final statement",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[16], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_rejects_pre_fold_store(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def pre_fold_store(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.grid(x.size(0)):
                out[row, 0, :, :] = x[row, 0, :, :]
                for col in hl.tile(x.size(1)):
                    out[row, col, :, :] = x[row, col, :, :]
            return out

        x = torch.randn(4, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = pre_fold_store.bind((x,))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "prologue capture only supports simple single-name assignments",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[16], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_rejects_folded_store_without_folded_index(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def folded_store_alias(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.grid(x.size(0)):
                for col in hl.tile(x.size(1)):
                    out[row, 0, :, :] = x[row, 0, :, :]
            return out

        x = torch.randn(4, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = folded_store_alias.bind((x,))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "folded-body stores must reference the folded hl.tile index",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[16], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_rejects_folded_read_modify_write(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def folded_read_modify_write(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.grid(x.size(0)):
                for col in hl.tile(x.size(1)):
                    out[row, col, :, :] = out[row, col, :, :] + x[row, col, :, :]
            return out

        x = torch.randn(4, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = folded_read_modify_write.bind((x,))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "folded-body read-modify-write stores",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[16], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_rejects_folded_atomic(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def folded_atomic(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for row in hl.grid(x.size(0)):
                for col in hl.tile(x.size(1)):
                    hl.atomic_add(
                        out,
                        [row, col, slice(None), slice(None)],
                        x[row, col, :, :],
                    )
            return out

        x = torch.randn(4, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = folded_atomic.bind((x,))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "does not support atomics in the folded hl.tile body",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[16], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_rejects_folded_reduction(self) -> None:
        from helion.language._tracing_ops import _validate_outer_pipeline_folded_body

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.call_function(torch.ops.aten.sum.dim_IntList, args=(x, [-1]))
        graph_info = types.SimpleNamespace(graph=graph)

        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "does not support reductions in the folded hl.tile body",
        ):
            _validate_outer_pipeline_folded_body(
                graph_info,
                [0],
                None,  # type: ignore[arg-type]
                {},
                {},
            )

    def test_outer_pipeline_rejects_loop_carried_state(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def folded_accumulator(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            block = hl.register_block_size(x.size(1))
            for row in hl.grid(x.size(0)):
                acc = hl.zeros([block, 8, 128], dtype=x.dtype)
                for col in hl.tile(x.size(1), block_size=block):
                    acc = acc + x[row, col, :, :]
                    out[row, col, :, :] = acc
            return out

        x = torch.randn(4, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = folded_accumulator.bind((x,))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "loop-carried state in the folded hl.tile loop",
        ):
            bound.to_triton_code(
                helion.Config(block_sizes=[16], pallas_loop_type="outer_pipeline")
            )

    def test_outer_pipeline_rejects_multiple_folded_tile_loops(self) -> None:
        @helion.kernel(backend="pallas", autotune_effort="none")
        def two_pass_copy(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out_a = torch.empty_like(x)
            out_b = torch.empty_like(x)
            for row in hl.grid(x.size(0)):
                for col in hl.tile(x.size(1)):
                    out_a[row, col, :, :] = x[row, col, :, :]
                for col in hl.tile(x.size(1)):
                    out_b[row, col, :, :] = x[row, col, :, :] * 2.0
            return out_a, out_b

        x = torch.randn(4, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = two_pass_copy.bind((x,))
        with self.assertRaisesRegex(exc.BackendUnsupported, "one folded hl.tile loop"):
            bound.to_triton_code(
                helion.Config(
                    block_sizes=[16, 16],
                    pallas_loop_type="outer_pipeline",
                )
            )


if __name__ == "__main__":
    unittest.main()
