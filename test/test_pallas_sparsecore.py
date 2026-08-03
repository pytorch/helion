from __future__ import annotations

from unittest.mock import patch

import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfPallasInterpret
from helion._testing import skipUnlessPallas
import helion.language as hl


@helion.kernel(backend="pallas", static_shapes=True)
def _embedding(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (x.size(0), weight.size(1)), dtype=weight.dtype, device=weight.device
    )
    for tile in hl.tile(x.size(0)):
        out[tile, :] = weight[x[tile], :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _independent_gathers(
    index: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    out = torch.empty(
        (index.size(0), left.size(1)), dtype=left.dtype, device=left.device
    )
    for tile in hl.tile(index.size(0)):
        out[tile, :] = left[index[tile], :] + right[index[tile], :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _dependent_embedding_bag(
    pointers: torch.Tensor, table: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    out = torch.empty(
        (index.size(0), table.size(1)), dtype=table.dtype, device=table.device
    )
    for tile in hl.tile(index.size(0)):
        gathered_index = pointers[index[tile], :]
        out[tile, :] = table[gathered_index, :].sum(dim=1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _embedding_bag_sum(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    rows, _ = index.size()
    out = torch.empty((rows, table.size(1)), dtype=table.dtype, device=table.device)
    for tile in hl.tile(rows):
        out[tile, :] = table[index[tile, :], :].sum(dim=1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _scatter_rows(
    source: torch.Tensor, index: torch.Tensor, output_rows: hl.constexpr
) -> torch.Tensor:
    out = torch.empty(
        (output_rows, source.size(1)), dtype=source.dtype, device=source.device
    )
    for tile in hl.tile(source.size(0)):
        out[index[tile], :] = source[tile, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _embedding_bag_mean(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (index.size(0), table.size(1)), dtype=table.dtype, device=table.device
    )
    for tile in hl.tile(index.size(0)):
        out[tile, :] = table[index[tile, :], :].mean(dim=1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _embedding_bag_epilogue(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (index.size(0), table.size(1)), dtype=table.dtype, device=table.device
    )
    for tile in hl.tile(index.size(0)):
        out[tile, :] = torch.relu(table[index[tile, :], :].sum(dim=1)) * 2.0
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _round_rows(source: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(source)
    for tile in hl.tile(source.size(0)):
        out[tile, :] = torch.round(source[tile, :])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _row_max(source: torch.Tensor) -> torch.Tensor:
    out = torch.empty((source.size(0),), dtype=source.dtype, device=source.device)
    for tile in hl.tile(source.size(0)):
        out[tile] = source[tile, :].amax(dim=1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _scatter_add_rows(
    source: torch.Tensor, index: torch.Tensor, output_rows: hl.constexpr
) -> torch.Tensor:
    out = torch.zeros(
        (output_rows, source.size(1)), dtype=source.dtype, device=source.device
    )
    for tile in hl.tile(source.size(0)):
        hl.atomic_add(out, [index[tile], slice(None)], source[tile, :])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _scatter_add_scaled_rows(
    source: torch.Tensor, index: torch.Tensor, output_rows: hl.constexpr
) -> torch.Tensor:
    out = torch.zeros(
        (output_rows, source.size(1)), dtype=source.dtype, device=source.device
    )
    for tile in hl.tile(source.size(0)):
        hl.atomic_add(out, [index[tile], slice(None)], source[tile, :] * 1.25)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _scatter_add_onto_ones(
    source: torch.Tensor, index: torch.Tensor, output_rows: hl.constexpr
) -> torch.Tensor:
    out = torch.ones(
        (output_rows, source.size(1)), dtype=source.dtype, device=source.device
    )
    for tile in hl.tile(source.size(0)):
        hl.atomic_add(out, [index[tile], slice(None)], source[tile, :])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _grouped_layer_norm(
    tables: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    groups = weight.size(0)
    width = tables.size(1)
    table_rows = tables.size(0) // groups
    rows = index.size(1)
    out = torch.empty((rows, groups, width), dtype=tables.dtype, device=tables.device)
    for tile in hl.tile(rows):
        gathered = torch.stack(
            [
                tables[index[group, tile] + group * table_rows, :]
                for group in hl.static_range(groups)
            ],
            dim=1,
        )
        flat = gathered.view(gathered.size(0), groups * width).to(torch.float32)
        mean = flat.mean(dim=1, keepdim=True)
        centered = flat - mean
        variance = (centered * centered).mean(dim=1, keepdim=True)
        normalized = (centered * torch.rsqrt(variance + 1e-5)).view(
            gathered.size(0), groups, width
        )
        out[tile, :, :] = (normalized * weight[None, :, :] + bias[None, :, :]).to(
            tables.dtype
        )
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _grouped_scale(
    tables: torch.Tensor, index: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    groups = index.size(0)
    width = tables.size(1)
    table_rows = tables.size(0) // groups
    out = torch.empty(
        (index.size(1), groups * width), dtype=tables.dtype, device=tables.device
    )
    for tile in hl.tile(index.size(1)):
        gathered = torch.stack(
            [
                tables[index[group, tile] + group * table_rows, :]
                for group in hl.static_range(groups)
            ],
            dim=1,
        )
        out[tile, :] = gathered.view(gathered.size(0), groups * width) * scale[None, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _bf16_gather_with_side_output(
    table: torch.Tensor,
    index: torch.Tensor,
    scale: torch.Tensor,
    side: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        (index.size(0), table.size(1)), dtype=table.dtype, device=table.device
    )
    side_out = torch.empty(
        (index.size(0), side.size(0)), dtype=side.dtype, device=side.device
    )
    for tile in hl.tile(index.size(0)):
        out[tile, :] = table[index[tile], :] * scale[None, :]
        side_out[tile, :] = side[None, :] + 1
    return out, side_out


def _code(kernel: object, args: tuple[object, ...], block: int = 256) -> str:
    with patch.dict("os.environ", {"HELION_SC_ASSUME_MESH": "2x16"}):
        bound = kernel.bind(args)  # type: ignore[attr-defined]
        return bound.to_triton_code(
            helion.Config(block_sizes=[block], core_type="sparsecore")
        )


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas not available")
class TestPallasSparseCoreCodegen(TestCase):
    def test_embedding_uses_native_sparsecore_pipeline(self) -> None:
        index = torch.randint(0, 1024, (1000,), dtype=torch.int32)
        table = torch.randn(1024, 64)

        code = _code(_embedding, (index, table))

        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn("pltpu.make_async_copy", code)
        self.assertIn("_sc_launcher_spec=", code)
        self.assertNotIn("one_hot", code)
        self.assertNotIn("codegen_grid_override", code)

    def test_independent_gathers_need_no_stack_pattern(self) -> None:
        index = torch.randint(0, 1024, (1000,), dtype=torch.int32)
        left = torch.randn(1024, 64)
        right = torch.randn(1024, 64)

        code = _code(_independent_gathers, (index, left, right))

        self.assertEqual(code.count("make_async_copy"), 4)
        self.assertIn(" + ", code)

    def test_grouped_layer_norm_composes_access_and_compute_lowerings(self) -> None:
        index = torch.randint(0, 512, (2, 513), dtype=torch.int32)
        weight = torch.randn(2, 48)
        bias = torch.randn(2, 48)

        f32_code = _code(
            _grouped_layer_norm,
            (torch.randn(1024, 48), index, weight, bias),
            block=512,
        )
        self.assertIn("lax.rsqrt", f32_code)
        self.assertIn("sc_cached", f32_code)

        with self.assertRaisesRegex(exc.InvalidConfig, "code: layout"):
            _code(
                _grouped_layer_norm,
                (torch.randn(1024, 48), index, weight, bias),
                block=256,
            )

    def test_packed_bf16_rearranges_only_its_cached_inputs(self) -> None:
        code = _code(
            _bf16_gather_with_side_output,
            (
                torch.randn(512, 64, dtype=torch.bfloat16),
                torch.randint(0, 512, (513,), dtype=torch.int32),
                torch.randn(64),
                torch.randn(16),
            ),
            block=512,
        )

        self.assertIn("'packed_inputs': [[0, 64]]", code)
        self.assertNotIn("[1, 16]", code.split("'packed_inputs':", 1)[1])

    def test_atomic_add_uses_shared_accumulator_phases(self) -> None:
        source = torch.randn(1024, 64)
        index = torch.randint(0, 128, (1024,), dtype=torch.int32)

        code = _code(_scatter_add_rows, (source, index, 128), block=512)

        self.assertIn("'vmem_shared'", code)
        self.assertIn("add=True", code)
        self.assertEqual(code.count("plsc.subcore_barrier()"), 2)
        self.assertIn("'combine_outputs': [[", code)
        self.assertIn("_inplace_indices=[]", code)
        self.assertIn("out = _launcher(", code)

    def test_atomic_add_requires_zero_initialized_output(self) -> None:
        source = torch.randn(1024, 64)
        index = torch.randint(0, 128, (1024,), dtype=torch.int32)

        with self.assertRaisesRegex(exc.InvalidConfig, "code: atomic_init"):
            _code(_scatter_add_onto_ones, (source, index, 128), block=512)

    def test_bf16_chunks_compose_across_stack_boundaries(self) -> None:
        index = torch.randint(0, 512, (2, 513), dtype=torch.int32)
        code = _code(
            _grouped_layer_norm,
            (
                torch.randn(1024, 48, dtype=torch.bfloat16),
                index,
                torch.randn(2, 48),
                torch.randn(2, 48),
            ),
            block=512,
        )

        self.assertIn("plsc.unpack", code)
        self.assertIn("plsc.pack", code)
        self.assertIn("sc_output[_sc_item, pl.ds(48, 32)]", code)
        self.assertNotIn("jnp.concatenate", code)
        self.assertNotIn("one_hot", code)

    def test_indirect_store_uses_layout_derived_tail_row(self) -> None:
        source = torch.randn(513, 64)
        index = torch.randperm(513, dtype=torch.int32)

        code = _code(_scatter_rows, (source, index, 513))

        self.assertIn("768, True, 513", code)
        self.assertIn("'pad_outputs': [[", code)
        self.assertIn("pltpu.sync_copy", code)

    def test_dependent_gather_depth_comes_from_dependencies(self) -> None:
        pointers = torch.randint(0, 4096, (1024, 8), dtype=torch.int32)
        table = torch.randn(4096, 64)
        index = torch.randint(0, 1024, (513,), dtype=torch.int32)

        code = _code(_dependent_embedding_bag, (pointers, table, index))

        self.assertEqual(code.count("def _sc_stage"), 3)
        self.assertIn("for _sc_h in range(8)", code)
        self.assertNotIn("one_hot", code)

    def test_reductions_and_epilogues_use_shared_compute_lowerings(self) -> None:
        table = torch.randn(4096, 64)
        index = torch.randint(0, 4096, (513, 8), dtype=torch.int32)

        mean_code = _code(_embedding_bag_mean, (table, index))
        epilogue_code = _code(_embedding_bag_epilogue, (table, index))
        max_code = _code(_row_max, (torch.randn(513, 64),))
        round_code = _code(_round_rows, (torch.randn(513, 64),))

        self.assertIn(" / ", mean_code)
        self.assertIn("jnp.maximum", epilogue_code)
        self.assertIn("jnp.max", max_code)
        self.assertIn("8388608.0", round_code)

    def test_direct_streams_and_multiple_outputs_compose(self) -> None:
        from examples.sparsecore_ops import act_quant
        from examples.sparsecore_ops import block_mask
        from examples.sparsecore_ops import gelu_multiply

        left = torch.randn(513, 128)
        right = torch.randn(513, 128)
        gelu_code = _code(gelu_multiply, (left, right))
        quant_code = _code(act_quant, (left,))
        mask_code = _code(block_mask, (left, 0.5))

        self.assertIn("jnp.exp", gelu_code)
        self.assertEqual(gelu_code.count("make_async_copy"), 4)
        self.assertEqual(quant_code.count("make_async_copy"), 2)
        self.assertEqual(mask_code.count("make_async_copy"), 2)
        self.assertGreaterEqual(quant_code.count("sc_output"), 2)
        self.assertIn("'scalar_outputs': [[", quant_code)
        self.assertIn("'kernel_output_dtypes': [[", quant_code)
        self.assertIn("jnp.max", mask_code)

    def test_stock_embedding_uses_covering_minor_tile(self) -> None:
        from examples.embedding import embedding

        index = torch.randint(0, 2048, (4, 129), dtype=torch.int32)
        table = torch.randn(2048, 64)
        with patch.dict("os.environ", {"HELION_SC_ASSUME_MESH": "2x16"}):
            code = embedding.bind((index, table)).to_triton_code(
                helion.Config(block_sizes=[256, 64], core_type="sparsecore")
            )

        self.assertIn("make_async_copy", code)
        self.assertNotIn("one_hot", code)


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
@skipIfPallasInterpret("SparseCore kernels have no interpret mode")
class TestPallasSparseCoreNumerics(TestCase):
    def test_embedding_bag_sum(self) -> None:
        rows, entries, width = 257, 8, 64
        table = torch.randn(4096, width, device=DEVICE)
        index = torch.randint(
            0, table.size(0), (rows, entries), dtype=torch.int32, device=DEVICE
        )

        _, result = code_and_output(
            _embedding_bag_sum,
            (table, index),
            block_sizes=[256],
            core_type="sparsecore",
        )

        expected = table.cpu()[index.cpu().long()].sum(dim=1)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_embedding_bag_mean(self) -> None:
        rows, entries, width = 513, 8, 64
        table = torch.randn(4096, width, device=DEVICE)
        index = torch.randint(
            0, table.size(0), (rows, entries), dtype=torch.int32, device=DEVICE
        )

        _, result = code_and_output(
            _embedding_bag_mean,
            (table, index),
            block_sizes=[256],
            core_type="sparsecore",
        )

        expected = table.cpu()[index.cpu().long()].mean(dim=1)
        torch.testing.assert_close(result.cpu(), expected)

    def test_embedding_bag_epilogue(self) -> None:
        rows, entries, width = 513, 8, 64
        table = torch.randn(4096, width, device=DEVICE)
        index = torch.randint(
            0, table.size(0), (rows, entries), dtype=torch.int32, device=DEVICE
        )

        _, result = code_and_output(
            _embedding_bag_epilogue,
            (table, index),
            block_sizes=[256],
            core_type="sparsecore",
        )

        expected = torch.relu(table.cpu()[index.cpu().long()].sum(dim=1)) * 2.0
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-4)

    def test_round_and_max_reductions(self) -> None:
        source = torch.randn(513, 64, device=DEVICE)
        source[7] = float("-inf")
        ties = torch.arange(-16.0, 16.0, 0.5, device=DEVICE).repeat(256, 1)

        _, maximum = code_and_output(
            _row_max,
            (source,),
            block_sizes=[256],
            core_type="sparsecore",
        )
        _, rounded = code_and_output(
            _round_rows,
            (ties,),
            block_sizes=[256],
            core_type="sparsecore",
        )

        torch.testing.assert_close(maximum.cpu(), source.cpu().amax(dim=1))
        torch.testing.assert_close(rounded.cpu(), torch.round(ties.cpu()))

    def test_dense_gelu_multiply(self) -> None:
        from examples.sparsecore_ops import gelu_multiply

        left = torch.randn(513, 128, device=DEVICE)
        right = torch.randn(513, 128, device=DEVICE)

        _, result = code_and_output(
            gelu_multiply,
            (left, right),
            block_sizes=[256],
            core_type="sparsecore",
        )

        expected = (
            torch.nn.functional.gelu(left.cpu(), approximate="tanh") * right.cpu()
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_activation_quantization_multiple_outputs(self) -> None:
        from examples.sparsecore_ops import act_quant

        source = torch.randn(513, 128, device=DEVICE)

        _, (quantized, scales) = code_and_output(
            act_quant,
            (source,),
            block_sizes=[256],
            core_type="sparsecore",
        )

        host = source.cpu()
        expected_scales = host.abs().amax(dim=1, keepdim=True) / 127.0
        expected_quantized = torch.clamp(
            torch.round(host / expected_scales), -128, 127
        ).to(torch.int8)
        torch.testing.assert_close(scales.cpu(), expected_scales)
        quantization_error = (
            quantized.cpu().to(torch.int16) - expected_quantized.to(torch.int16)
        ).abs()
        self.assertLessEqual(quantization_error.max().item(), 1)

    def test_boolean_scalar_output(self) -> None:
        from examples.sparsecore_ops import block_mask

        source = torch.randn(513, 128, device=DEVICE)

        _, result = code_and_output(
            block_mask,
            (source, 0.5),
            block_sizes=[256],
            core_type="sparsecore",
        )

        torch.testing.assert_close(result.cpu(), source.cpu().amax(dim=1) > 0.5)

    def test_stock_embedding_and_bf16_gather(self) -> None:
        from examples.embedding import embedding

        index = torch.randint(0, 2048, (4, 129), dtype=torch.int32, device=DEVICE)
        table = torch.randn(2048, 64, device=DEVICE)
        bf16_table = table.to(torch.bfloat16)

        _, stock = code_and_output(
            embedding,
            (index, table),
            block_sizes=[256, 64],
            core_type="sparsecore",
        )
        _, bf16 = code_and_output(
            _embedding,
            (index.reshape(-1), bf16_table),
            block_sizes=[256],
            core_type="sparsecore",
        )

        torch.testing.assert_close(stock.cpu(), table.cpu()[index.cpu().long()])
        torch.testing.assert_close(
            bf16.cpu(), bf16_table.cpu()[index.cpu().reshape(-1).long()]
        )

    def test_weighted_and_masked_gather_reductions(self) -> None:
        from examples.sparsecore_ops import masked_gather_sum
        from examples.sparsecore_ops import moe_combine

        rows, entries, width = 513, 8, 64
        table = torch.randn(4096, width, device=DEVICE)
        index = torch.randint(
            0, table.size(0), (rows, entries), dtype=torch.int32, device=DEVICE
        )
        weights = torch.rand(rows, entries, device=DEVICE)

        _, combined = code_and_output(
            moe_combine,
            (table, index, weights),
            block_sizes=[256],
            core_type="sparsecore",
        )
        _, masked = code_and_output(
            masked_gather_sum,
            (table, index, weights),
            block_sizes=[256],
            core_type="sparsecore",
        )

        expected = (table.cpu()[index.cpu().long()] * weights.cpu()[..., None]).sum(
            dim=1
        )
        torch.testing.assert_close(combined.cpu(), expected, rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(masked.cpu(), expected, rtol=1e-5, atol=1e-4)

    def test_flat_grouped_parameter(self) -> None:
        groups, table_rows, width, rows = 2, 512, 64, 513
        tables = torch.randn(groups * table_rows, width, device=DEVICE)
        index = torch.randint(
            0, table_rows, (groups, rows), dtype=torch.int32, device=DEVICE
        )
        scale = torch.randn(groups * width, device=DEVICE)

        _, result = code_and_output(
            _grouped_scale,
            (tables, index, scale),
            block_sizes=[512],
            core_type="sparsecore",
        )

        gathered = torch.stack(
            [
                tables.cpu()[index.cpu()[group].long() + group * table_rows]
                for group in range(groups)
            ],
            dim=1,
        )
        expected = gathered.reshape(rows, groups * width) * scale.cpu()[None, :]
        torch.testing.assert_close(result.cpu(), expected)

    def test_independent_gathers(self) -> None:
        rows, width = 513, 64
        index = torch.randint(0, 4096, (rows,), dtype=torch.int32, device=DEVICE)
        left = torch.randn(4096, width, device=DEVICE)
        right = torch.randn(4096, width, device=DEVICE)

        _, result = code_and_output(
            _independent_gathers,
            (index, left, right),
            block_sizes=[256],
            core_type="sparsecore",
        )

        expected = left.cpu()[index.cpu().long()] + right.cpu()[index.cpu().long()]
        torch.testing.assert_close(result.cpu(), expected)

    def test_indirect_store(self) -> None:
        rows, width = 513, 64
        source = torch.randn(rows, width, device=DEVICE)
        index = torch.randperm(rows, dtype=torch.int32, device=DEVICE)

        _, result = code_and_output(
            _scatter_rows,
            (source, index, rows),
            block_sizes=[256],
            core_type="sparsecore",
        )

        expected = torch.empty_like(source.cpu())
        expected[index.cpu().long()] = source.cpu()
        torch.testing.assert_close(result.cpu(), expected)

    def test_dependent_gather(self) -> None:
        rows, pointer_rows, entries, table_rows, width = 513, 1024, 8, 4096, 64
        pointers = torch.randint(
            0,
            table_rows,
            (pointer_rows, entries),
            dtype=torch.int32,
            device=DEVICE,
        )
        table = torch.randn(table_rows, width, device=DEVICE)
        index = torch.randint(
            0, pointer_rows, (rows,), dtype=torch.int32, device=DEVICE
        )

        _, result = code_and_output(
            _dependent_embedding_bag,
            (pointers, table, index),
            block_sizes=[256],
            core_type="sparsecore",
        )

        expected = table.cpu()[pointers.cpu()[index.cpu().long()].long()].sum(dim=1)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_grouped_layer_norm(self) -> None:
        groups, table_rows, width, rows = 2, 512, 48, 513
        tables = torch.randn(groups * table_rows, width, device=DEVICE)
        index = torch.randint(
            0, table_rows, (groups, rows), dtype=torch.int32, device=DEVICE
        )
        weight = torch.randn(groups, width, device=DEVICE)
        bias = torch.randn(groups, width, device=DEVICE)

        _, result = code_and_output(
            _grouped_layer_norm,
            (tables, index, weight, bias),
            block_sizes=[512],
            core_type="sparsecore",
        )

        gathered = torch.stack(
            [
                tables.cpu()[index.cpu()[group].long() + group * table_rows]
                for group in range(groups)
            ],
            dim=1,
        )
        flat = gathered.reshape(rows, groups * width)
        mean = flat.mean(dim=1, keepdim=True)
        variance = ((flat - mean) ** 2).mean(dim=1, keepdim=True)
        expected = ((flat - mean) * torch.rsqrt(variance + 1e-5)).reshape(
            rows, groups, width
        )
        expected = expected * weight.cpu()[None] + bias.cpu()[None]
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_scatter_add(self) -> None:
        rows, output_rows, width = 1024, 128, 64
        source = torch.randn(rows, width, device=DEVICE)
        index = torch.randint(0, output_rows, (rows,), dtype=torch.int32, device=DEVICE)

        _, result = code_and_output(
            _scatter_add_rows,
            (source, index, output_rows),
            block_sizes=[512],
            core_type="sparsecore",
        )

        expected = torch.zeros(output_rows, width)
        expected.index_add_(0, index.cpu().long(), source.cpu())
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-4)

    def test_scatter_add_computed_values(self) -> None:
        rows, output_rows, width = 1024, 128, 64
        source = torch.randn(rows, width, device=DEVICE)
        index = torch.randint(0, output_rows, (rows,), dtype=torch.int32, device=DEVICE)

        _, result = code_and_output(
            _scatter_add_scaled_rows,
            (source, index, output_rows),
            block_sizes=[512],
            core_type="sparsecore",
        )

        expected = torch.zeros(output_rows, width)
        expected.index_add_(0, index.cpu().long(), source.cpu() * 1.25)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-4)

    def test_grouped_layer_norm_bf16(self) -> None:
        groups, table_rows, width, rows = 2, 512, 48, 513
        tables = torch.randn(
            groups * table_rows, width, dtype=torch.bfloat16, device=DEVICE
        )
        index = torch.randint(
            0, table_rows, (groups, rows), dtype=torch.int32, device=DEVICE
        )
        weight = torch.randn(groups, width, device=DEVICE)
        bias = torch.randn(groups, width, device=DEVICE)

        _, result = code_and_output(
            _grouped_layer_norm,
            (tables, index, weight, bias),
            block_sizes=[512],
            core_type="sparsecore",
        )

        gathered = torch.stack(
            [
                tables.cpu().float()[index.cpu()[group].long() + group * table_rows]
                for group in range(groups)
            ],
            dim=1,
        )
        flat = gathered.reshape(rows, groups * width)
        mean = flat.mean(dim=1, keepdim=True)
        variance = ((flat - mean) ** 2).mean(dim=1, keepdim=True)
        expected = ((flat - mean) * torch.rsqrt(variance + 1e-5)).reshape(
            rows, groups, width
        )
        expected = expected * weight.cpu()[None] + bias.cpu()[None]
        self.assertLess((result.cpu().float() - expected).abs().max().item(), 0.11)
