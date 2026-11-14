from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_mma_dag_grouped import grouped_matmul
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _offset_gemm(
    values: torch.Tensor, weight: torch.Tensor, begin: int
) -> torch.Tensor:
    begin = hl.specialize(begin)
    output = torch.empty((2, 32, 32), dtype=values.dtype, device=values.device)
    for item in hl.tile(begin, begin + 2, block_size=1):
        for rows in hl.tile(32):
            index = rows.index + item.begin * 32
            output[item.id - begin, rows, :] = hl.dot(values[index, :], weight[:, :])
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _canonical_offset_gemm(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    output = torch.empty((2, 32, 32), dtype=values.dtype, device=values.device)
    for item in hl.tile(2, block_size=1):
        for rows in hl.tile(32):
            index = rows.index + item.begin * 32
            output[item.id, rows, :] = hl.dot(values[index, :], weight[:, :])
    return output


@pytest.mark.parametrize("shifted", [False, True])
def test_chunk_offset_proof_requires_canonical_lead_origin(shifted: bool) -> None:
    # At begin=1 the second task addresses rows 64..95 of a 64-row tensor.
    # The inner row tile has a canonical origin, so this reaches the separate
    # offset proof for the shifted lead. Unmasked MMA transfers must decline.
    args = (
        torch.empty((64, 32), dtype=torch.bfloat16),
        torch.empty((32, 32), dtype=torch.bfloat16),
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = (
            cpu_bind(_offset_gemm, (*args, 1))
            if shifted
            else cpu_bind(_canonical_offset_gemm, args)
        )
        facts = bound.config_spec.cute_mma_dag_facts
        if shifted:
            assert bound.host_function.device_ir.noncanonical_task_origin_block_ids
            assert facts is None
        else:
            assert facts is not None
            source = bound.to_code(
                helion.Config(
                    block_sizes=[32],
                    cute_mma_dag="warp",
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
            assert "_helion_mma_dag.cp_async_16" in source


@helion.kernel(backend="cute", static_shapes=True)
def _shifted_loop_extent(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    block_m = hl.register_block_size(values.size(0))
    block_n = hl.register_block_size(weight.size(1))
    block_k = hl.register_block_size(32, 32)
    output = torch.empty_like(values)
    for rows, columns in hl.tile(
        [32, 0],
        [values.size(0) + 32, weight.size(1)],
        block_size=[block_m, block_n],
    ):
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile((rows.id + 1) * block_m, block_size=block_k):
            acc = hl.dot(values[rows, inner], weight[inner, columns], acc=acc)
        output[rows, columns] = acc.to(output.dtype)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _canonical_loop_extent(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    block_m = hl.register_block_size(values.size(0))
    block_n = hl.register_block_size(weight.size(1))
    block_k = hl.register_block_size(32, 32)
    output = torch.empty_like(values)
    for rows, columns in hl.tile(
        [values.size(0), weight.size(1)], block_size=[block_m, block_n]
    ):
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile((rows.id + 1) * block_m, block_size=block_k):
            acc = hl.dot(values[rows, inner], weight[inner, columns], acc=acc)
        output[rows, columns] = acc.to(output.dtype)
    return output


@pytest.mark.parametrize("shifted", [False, True])
def test_varying_tile_extent_requires_canonical_coordinate(shifted: bool) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((64, 64), dtype=torch.bfloat16),
    )
    # Shifted root coordinates reach a loop bound beyond the 64-column input.
    # Their trip count is not enough to prove an unmasked operand transfer.
    kernel = _shifted_loop_extent if shifted else _canonical_loop_extent
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        if shifted:
            assert bound.config_spec.cute_mma_dag_facts is None
        else:
            assert bound.config_spec.cute_mma_dag_facts is not None
            source = bound.to_code(
                helion.Config(
                    block_sizes=[32, 32, 32],
                    cute_mma_dag="warp",
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                    cute_mma_dag_loads="lane",
                )
            )
            assert "_helion_mma_dag.mma_bf16" in source


@helion.kernel(backend="cute", static_shapes=True)
def _shifted_static_extent(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    block_m = hl.register_block_size(values.size(0))
    block_n = hl.register_block_size(weight.size(1))
    block_k = hl.register_block_size(32, 32)
    output = torch.empty_like(values)
    for rows, columns in hl.tile(
        [32, 0],
        [values.size(0) + 32, weight.size(1)],
        block_size=[block_m, block_n],
    ):
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile(values.size(1), block_size=block_k):
            # A rescaled carry exercises the MMA-DAG recurrence owner.
            acc = acc * 0.5
            acc = hl.dot(values[rows, inner], weight[inner, columns], acc=acc)
        output[rows, columns] = acc.to(output.dtype)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _canonical_static_extent(
    values: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    block_m = hl.register_block_size(values.size(0))
    block_n = hl.register_block_size(weight.size(1))
    block_k = hl.register_block_size(32, 32)
    output = torch.empty_like(values)
    for rows, columns in hl.tile(
        [values.size(0), weight.size(1)], block_size=[block_m, block_n]
    ):
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile(values.size(1), block_size=block_k):
            # A rescaled carry exercises the MMA-DAG recurrence owner.
            acc = acc * 0.5
            acc = hl.dot(values[rows, inner], weight[inner, columns], acc=acc)
        output[rows, columns] = acc.to(output.dtype)
    return output


@pytest.mark.parametrize("shifted", [False, True])
def test_static_tile_access_requires_canonical_coordinate(shifted: bool) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((64, 64), dtype=torch.bfloat16),
    )
    # The static inner bound does not make shifted root rows safe: the second
    # task still transfers rows 64..95 from a 64-row input without a predicate.
    kernel = _shifted_static_extent if shifted else _canonical_static_extent
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        if shifted:
            assert bound.config_spec.cute_mma_dag_facts is None
        else:
            assert bound.config_spec.cute_mma_dag_facts is not None
            source = bound.to_code(
                helion.Config(
                    block_sizes=[32, 32, 32],
                    cute_mma_dag="warp",
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                    cute_mma_dag_loads="lane",
                )
            )
            assert "_helion_mma_dag.mma_bf16" in source


@helion.kernel(backend="cute", static_shapes=True)
def _grouped_origin_matmul(
    values: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, begin: int
) -> torch.Tensor:
    begin = hl.specialize(begin)
    columns = hl.specialize(weight.size(1))
    output = torch.empty(
        (values.size(0), columns), dtype=torch.float32, device=values.device
    )
    for rows in hl.tile(begin, values.size(0) + begin):
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for _repeat in hl.grid(2):
            scale = scales[rows.index // 3, :].float()
            value = (torch.exp2(scale - 127) * values[rows, :].float()).to(weight.dtype)
            acc = hl.dot(value, weight[:, :], acc=acc)
        output[rows, :] = acc
    return output


@pytest.mark.parametrize("shifted", [False, True])
def test_grouped_index_proof_requires_canonical_root_origin(shifted: bool) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.int8),
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((22, 64), dtype=torch.uint8),
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = (
            cpu_bind(_grouped_origin_matmul, (*args, 32))
            if shifted
            else cpu_bind(
                helion.kernel(grouped_matmul, backend="cute", static_shapes=True),
                (*args, 3, True, False),
            )
        )
        assert (bound.config_spec.cute_mma_dag_facts is None) == shifted
        with bound.env, bound.host_function:
            matcher = dag._Matcher(bound.host_function.device_ir.graphs)
            if shifted:
                with pytest.raises(dag._NotMine):
                    matcher._match()
            else:
                matcher._match()
            index = next(
                node
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if node.target
                in (torch.ops.aten.floor_divide.default, torch.ops.aten.div.Tensor_mode)
            )
            block_id = bound.env.resolve_block_id(index.meta["val"].size(0))
            assert block_id is not None
            assert bound.env.canonical_block_id(block_id) in matcher.root_block_ids
            # Other tiled-access guards also reject the whole shifted kernel.
            # Probe this proof independently: rows 32..95 require groups 10..31,
            # beyond the 22-row scale table despite the root's 64-row trip count.
            if shifted:
                with pytest.raises(dag._NotMine):
                    matcher.grouped_index(index, index.meta["val"].size(0), 22)
            else:
                grouped = matcher.grouped_index(index, index.meta["val"].size(0), 22)
                assert grouped is not None
                axis, divisor = grouped
                assert axis.kind == dag.AXIS_ROOT and divisor == 3
