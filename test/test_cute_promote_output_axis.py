from __future__ import annotations

import ast
from contextlib import ExitStack
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
from examples.jagged_dense_bmm import jagged_dense_bmm
from examples.moe_matmul_ogs import moe_matmul_ogs
import pytest
import torch

import helion
from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.cute.promote_output_axis import _fresh_tensors
from helion._compiler.cute.promote_output_axis import promote_partitioned_output_axis
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.type_info import GridIndexType
from helion._compiler.type_info import IterType
from helion._compiler.type_info import SequenceType
from helion._compiler.type_info import TileIndexType
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CPU_DEVICE = "cpu"

if TYPE_CHECKING:
    from helion._compiler.host_function import HostFunction
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_columns(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, column] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_tiled_root(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    for batch in hl.tile(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, column] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _self_read_columns(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.zeros_like(x)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            previous = out[batch, row, column]
            value = x[batch, row, column]
            out[batch, row, column] = torch.where(value > 0, value, previous)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shifted_self_read(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.zeros_like(x)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            previous = out[batch, row, (column.index + 1) % columns]
            out[batch, row, column] = x[batch, row, column] + previous
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _full_row_self_read(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.ones_like(x)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            previous = out[batch, row, :].sum(dim=-1)
            out[batch, row, column] = x[batch, row, column] + previous[:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _write_argument(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, column] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rebound_output(x: torch.Tensor, storage: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    out.set_(storage)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, (column.index + 1) % columns]
    return out


def _rebind_storage(out: torch.Tensor, storage: torch.Tensor) -> torch.Tensor:
    return out.set_(storage)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _opaque_rebound_output(x: torch.Tensor, storage: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    out = _rebind_storage(out, storage)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, (column.index + 1) % columns]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _host_loop_rebound_output(x: torch.Tensor, storage: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    for _index in range(2):
        out.set_(storage)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, (column.index + 1) % columns]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _after_grids_rebound_output(x: torch.Tensor, storage: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    scratch = torch.empty_like(x)
    for first_batch, first_row, first_column in hl.tile(x.shape):
        scratch[first_batch, first_row, first_column] = (
            x[first_batch, first_row, first_column] + 1
        )
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, (column.index + 1) % columns]
    out.set_(storage)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fresh_two_grid_output(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    scratch = torch.empty_like(x)
    for first_tile in hl.tile(x.shape):
        scratch[first_tile] = x[first_tile] + 1
    for second_tile in hl.tile(x.shape):
        out[second_tile] = x[second_tile] + 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fresh_view_output(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    storage = torch.empty_like(x)
    out = storage.view(batches, rows, columns)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, column] + 1
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _registered_output(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    height = hl.register_block_size(1, rows)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns], block_size=[height, None]):
            out[batch, row, column] = x[batch, row, column] + 1
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _specialized_output(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    batches, rows, columns = hl.specialize(x.shape)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, column] + 1
    return out


def _rebind_and_size(out: torch.Tensor, x: torch.Tensor) -> int:
    out.set_(x)
    return x.size(-1)


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _specialized_mutated_output(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    columns = hl.specialize(_rebind_and_size(out, x))
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, (column.index + 1) % columns]
    return out


def specialize(out: torch.Tensor, x: torch.Tensor) -> int:
    return _rebind_and_size(out, x)


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _shadowed_specialized_output(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    columns = specialize(out, x)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, (column.index + 1) % columns]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _write_view(x: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    storage = torch.empty_like(x)
    out = torch.as_strided(storage, x.shape, [0, 0, 1])
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, column] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _atomic_side_effect(x: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.empty_like(x)
    for batch in hl.grid(batches):
        for row, column in hl.tile([rows, columns]):
            out[batch, row, column] = x[batch, row, column] + 1
            hl.atomic_add(counter, [0], 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _dynamic_columns(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batches, rows, columns = x.shape
    out = torch.zeros_like(x)
    for batch in hl.grid(batches):
        width = lengths[batch]
        for row, column in hl.tile([rows, width]):
            out[batch, row, column] = x[batch, row, column] + 1
    return out


def _bind(kernel: Kernel[torch.Tensor], *args: object) -> BoundKernel[torch.Tensor]:
    """Trace and generate source on CPU; no CUDA allocation or launch occurs."""
    with ExitStack() as stack:
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
            ("helion.runtime.get_num_sm", 148),
            ("helion._compat._is_hip", False),
        ):
            stack.enter_context(patch(name, return_value=value))
        return kernel._bind_isolated(tuple(args))


def _input() -> torch.Tensor:
    return torch.empty((3, 13, 65), device=CPU_DEVICE)


def _host(bound: BoundKernel[torch.Tensor]) -> HostFunction:
    host = bound.host_function
    assert host is not None
    return host


def test_promoted_grid_retains_serial_rows_and_configuration() -> None:
    bound = _bind(_copy_columns, _input())
    ir = _host(bound).device_ir
    assert ir.grid_block_ids == [[0, 2]]
    assert [
        graph.block_ids for graph in ir.graphs if type(graph) is ForLoopGraphInfo
    ] == [[1]]
    assert [axis.block_id for axis in ir.task_families[0].axes] == [0, 2]
    assert bound.config_spec.grid_block_ids == [0, 2]
    assert [item.block_ids for item in bound.config_spec.loop_orders] == [[0, 2]]
    assert len(bound.config_spec.flatten_loops) == 0
    root = next(stmt for stmt in _host(bound).body if isinstance(stmt, ast.For))
    assert isinstance(root.iter, ExtendedAST)
    assert isinstance(root.iter._type_info, IterType)
    assert isinstance(root.iter._type_info.inner, SequenceType)
    grid, tile = root.iter._type_info.inner.unpack()
    assert isinstance(grid, GridIndexType)
    assert isinstance(tile, TileIndexType)
    for graph in ir.graphs:
        graph.graph.lint()
    # Tiled and full-width configurations both generate valid Python source.
    # The final-axis loop becomes a PID axis, while row traversal remains serial.
    for block_columns in (16, 128):
        config = helion.Config(block_sizes=[4, block_columns])
        code = bound.to_code(config)
        ast.parse(code)
        assert "for tile_offset_1 in range" in code
        assert "for tile_offset_2 in range" not in code
        assert "tile_offset_2 = pid_1 * _BLOCK_SIZE_2" in code


def test_self_read_stays_with_its_column_partition() -> None:
    bound = _bind(_self_read_columns, _input())
    assert _host(bound).device_ir.grid_block_ids == [[0, 2]]
    ast.parse(bound.to_code(bound.config_spec.default_config()))


def test_tiled_root_preserves_explicit_device_loop_configuration() -> None:
    bound = _bind(_copy_tiled_root, _input())
    ir = _host(bound).device_ir
    assert ir.grid_block_ids == [[0]]
    assert [
        graph.block_ids for graph in ir.graphs if type(graph) is ForLoopGraphInfo
    ] == [[1, 2]]
    config = helion.Config(
        block_sizes=[1, 4, 16], flatten_loops=[True], loop_orders=[[1, 0]]
    )
    ast.parse(bound.to_code(config))


def test_flat_partition_promotes_one_axis_without_losing_nested_thread_uses() -> None:
    kernel = helion.kernel(
        jagged_dense_bmm.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = _bind(
        kernel,
        torch.tensor([0, 3, 3, 137]),
        torch.empty((137, 78)),
        torch.empty((3, 78, 128)),
        torch.empty((3, 128)),
    )
    ir = _host(bound).device_ir
    assert ir.grid_block_ids == [[0, 2]]
    assert [info.block_ids for info in ir.graphs if type(info) is ForLoopGraphInfo] == [
        [3],
        [1],
    ]
    config = helion.Config(
        block_sizes=[1, 64, 32, 64],
        num_threads=[0, 4, 32, 1],
        cute_vector_widths=[1] * 4,
        cute_lane_layouts=["blocked"] * 4,
    )
    source = bound.to_code(config)
    ast.parse(source)
    assert "for tile_offset_1 in range" in source
    assert "for tile_offset_2 in range" not in source
    assert "block=(32, 4, 1)" in source
    assert (
        "indices_2 = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[0])" in source
    )
    assert "mask_1" in source and "mask_3" in source


def test_non_power_of_two_flat_columns_keep_original_loop() -> None:
    kernel = helion.kernel(
        jagged_dense_bmm.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = _bind(
        kernel,
        torch.tensor([0, 3, 3, 137]),
        torch.empty((137, 78)),
        torch.empty((3, 78, 70)),
        torch.empty((3, 70)),
    )
    assert _host(bound).device_ir.grid_block_ids == [[0]]


@pytest.mark.parametrize(
    "kernel", [_shifted_self_read, _full_row_self_read, _write_view]
)
def test_rejects_cross_partition_aliases(kernel: Kernel[torch.Tensor]) -> None:
    bound = _bind(kernel, _input())
    assert _host(bound).device_ir.grid_block_ids == [[0]]


def test_rejects_mutated_input_without_alias_specialization() -> None:
    x = _input()
    bound = _bind(_write_argument, x, x)
    assert _host(bound).device_ir.grid_block_ids == [[0]]


@pytest.mark.parametrize(
    "kernel", [_rebound_output, _opaque_rebound_output, _host_loop_rebound_output]
)
@pytest.mark.parametrize("alias", [False, True])
def test_rejects_factory_storage_rebound_to_input(
    kernel: Kernel[torch.Tensor], alias: bool
) -> None:
    x = _input()
    storage = x if alias else torch.empty_like(x)
    bound = _bind(kernel, x, storage)
    host = _host(bound)
    assert host.device_ir.grid_block_ids == [[0]]
    with bound.env, host:
        # Signed-byte packet admission also consumes this shared proof.
        assert not _fresh_tensors(host)


@pytest.mark.parametrize("alias", [False, True])
def test_rejects_storage_rebinding_after_grids(alias: bool) -> None:
    x = _input()
    bound = _bind(_after_grids_rebound_output, x, x if alias else torch.empty_like(x))
    host = _host(bound)
    assert len(host.device_ir.grid_block_ids) == 2
    with bound.env, host:
        assert not _fresh_tensors(host)


def test_keeps_fresh_storage_across_two_grids() -> None:
    bound = _bind(_fresh_two_grid_output, _input())
    host = _host(bound)
    assert len(host.device_ir.grid_block_ids) == 2
    with bound.env, host:
        assert len(_fresh_tensors(host)) == 2


def test_keeps_fresh_view_output_promotion() -> None:
    bound = _bind(_fresh_view_output, _input())
    host = _host(bound)
    assert host.device_ir.grid_block_ids == [[0, 2]]
    with bound.env, host:
        fresh = _fresh_tensors(host)
        assert fresh
        inputs = {t.untyped_storage()._cdata for t in bound.env.input_sources}
        assert all(t.untyped_storage()._cdata not in inputs for t in fresh)


@pytest.mark.parametrize("kernel", [_registered_output, _specialized_output])
def test_host_metadata_dsl_keeps_fresh_output(
    kernel: Kernel[torch.Tensor],
) -> None:
    bound = _bind(kernel, _input())
    host = _host(bound)
    with bound.env, host:
        fresh = _fresh_tensors(host)
        assert fresh
        inputs = {t.untyped_storage()._cdata for t in bound.env.input_sources}
        assert all(t.untyped_storage()._cdata not in inputs for t in fresh)


def test_specialization_keeps_output_promotion() -> None:
    bound = _bind(_specialized_output, _input())
    assert len(_host(bound).device_ir.grid_block_ids[0]) == 2


@pytest.mark.parametrize(
    "kernel", [_specialized_mutated_output, _shadowed_specialized_output]
)
def test_specialize_does_not_hide_opaque_storage_mutation(
    kernel: Kernel[torch.Tensor],
) -> None:
    bound = _bind(kernel, _input())
    host = _host(bound)
    assert len(host.device_ir.grid_block_ids[0]) == 1
    with bound.env, host:
        assert not _fresh_tensors(host)


def test_rejects_noncontiguous_output() -> None:
    x = torch.empty((3, 65, 13), device=CPU_DEVICE).transpose(1, 2)
    bound = _bind(_copy_columns, x)
    assert _host(bound).device_ir.grid_block_ids == [[0]]


def test_rejects_side_effects() -> None:
    bound = _bind(_atomic_side_effect, _input(), torch.zeros(1, device=CPU_DEVICE))
    assert _host(bound).device_ir.grid_block_ids == [[0]]


def test_rejects_device_dependent_grid_extent() -> None:
    bound = _bind(_dynamic_columns, _input(), torch.ones(3, dtype=torch.int32))
    assert _host(bound).device_ir.grid_block_ids == [[0]]


def test_rejects_carried_register_values_without_partial_mutation() -> None:
    with patch(
        "helion._compiler.cute.promote_output_axis.promote_partitioned_output_axis",
        return_value=False,
    ):
        bound = _bind(_copy_columns, _input())
    ir = _host(bound).device_ir
    graph = next(graph for graph in ir.graphs if type(graph) is ForLoopGraphInfo)
    output = next(iter(graph.graph.find_nodes(op="output")))
    value = next(
        node for node in graph.graph.nodes if node.target is torch.ops.aten.add.Tensor
    )
    output.args = ([value],)
    before = ir.semantic_debug_str()
    config_before = bound.config_spec.default_config()
    roots = [stmt for stmt in _host(bound).body if isinstance(stmt, ast.For)]
    with bound.env, _host(bound):
        assert not promote_partitioned_output_axis(_host(bound), ir, roots)
    assert ir.semantic_debug_str() == before
    assert bound.config_spec.default_config() == config_before
    assert ir.grid_block_ids == [[0]]


def test_triton_keeps_original_loop_nest() -> None:
    kernel = helion.kernel(
        _copy_columns.fn, backend="triton", static_shapes=True, autotune_effort="none"
    )
    bound = _bind(kernel, _input())
    assert _host(bound).device_ir.grid_block_ids == [[0]]
    assert [
        graph.block_ids
        for graph in _host(bound).device_ir.graphs
        if type(graph) is ForLoopGraphInfo
    ] == [[1, 2]]


@pytest.mark.parametrize("static_shapes", [False, True])
def test_moe_keeps_gathered_row_order(static_shapes: bool) -> None:
    kernel = helion.kernel(
        moe_matmul_ogs.fn, backend="cute", static_shapes=static_shapes
    )
    bound = _bind(
        kernel,
        torch.empty((128, 256), dtype=torch.bfloat16),
        torch.empty((4, 256, 128), dtype=torch.bfloat16),
        torch.empty(4, dtype=torch.int32),
        torch.empty(5, dtype=torch.int32),
        torch.empty(128, dtype=torch.int32),
        32,
    )
    assert _host(bound).device_ir.grid_block_ids == [[0, 2]]
    assert [
        graph.block_ids
        for graph in _host(bound).device_ir.graphs
        if type(graph) is ForLoopGraphInfo
    ] == [[3], [1]]
    code = bound.to_code(
        helion.Config(
            block_sizes=[64, 32, 64],
            num_threads=[4, 32, 1],
            cute_vector_widths=[1, 1, 1, 1],
        )
    )
    ast.parse(code)
    assert "for tile_offset_1 in range" in code
    assert "for tile_offset_2 in range" not in code
    # Untuned grid block 0 is absent from config.block_sizes. In particular,
    # block ID 2 has size 32, although config.block_sizes[2] is 64.
    assert "block=(32, 4, 1)" in code


@pytest.mark.parametrize("static_shapes", [False, True])
def test_grouped_gemm_keeps_dynamic_row_loop(static_shapes: bool) -> None:
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=static_shapes,
        cute_segmented_matmul_tiling=False,
    )
    bound = _bind(
        kernel,
        torch.empty((128, 256), dtype=torch.bfloat16),
        torch.empty((256, 128), dtype=torch.bfloat16),
        torch.empty(5, dtype=torch.int32),
    )
    assert _host(bound).device_ir.grid_block_ids == [[0, 2]]
    code = bound.to_code(helion.Config(block_sizes=[16, 32, 16]))
    ast.parse(code)
    assert "for tile_offset_1 in range" in code
    assert "for tile_offset_2 in range" not in code


@pytest.mark.parametrize("static_shapes", [False, True])
def test_grouped_gemm_segmented_rows_use_masked_grid(static_shapes: bool) -> None:
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=static_shapes,
        cute_segmented_matmul_tiling=True,
    )
    bound = _bind(
        kernel,
        torch.empty((128, 256), dtype=torch.bfloat16),
        torch.empty((256, 128), dtype=torch.bfloat16),
        torch.empty(5, dtype=torch.int32),
    )
    host = _host(bound)
    assert host.device_ir.grid_block_ids == [[0, 1, 2]]
    assert [
        graph.block_ids
        for graph in host.device_ir.graphs
        if type(graph) is ForLoopGraphInfo
    ] == [[3]]
    masks = [
        ast.unparse(keyword.value)
        for node in ast.walk(ast.Module(body=host.body, type_ignores=[]))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "extra_mask"
    ]
    assert masks == ["_helion_segment_valid[:, None]"] * 2
    code = bound.to_code(helion.Config(block_sizes=[16, 32, 16]))
    ast.parse(code)
    assert "for tile_offset_1 in range" not in code
    assert "for tile_offset_2 in range" not in code
