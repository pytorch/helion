"""CPU source-generation coverage of uniform table-loaded MMA operands."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _paged_matmul(
    a: torch.Tensor,
    pages: torch.Tensor,
    order: torch.Tensor,
    length: torch.Tensor,
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(pages.size(2))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(0, length[0]):
            page = order[step]
            acc = hl.dot(a[tm, :], pages[page, :, :], acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_table_loaded_operand_index(mode: str, dtype: torch.dtype) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((5, 64, 64), dtype=torch.bfloat16),
        torch.empty((4,), dtype=dtype),
        torch.empty((1,), dtype=torch.int32),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_paged_matmul, backend="cute", static_shapes=True), args
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            assert len(program.index_loads) == 1
            (access,) = program.index_loads.values()
            assert access.name == "order"
            assert access.loops == frozenset({0})
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=3,
                    cute_mma_dag_loads="lane"
                    if mode == "warp" and dtype == torch.int32
                    else "tma",
                )
            )
        source = bound.to_code(config)
    assert "_cute_grouped_reduce_shared_two_stage" not in source
    assert "_helion_mma_dag" in source
    refill = source[source.index("if md_i + 2 < md_scalar") :]
    assert re.search(
        r"ldg_s(?:32|64)\(md_base_\d+ \+ cutlass.Int64\(md_i \+ 2\)", refill
    )
    if mode == "warp" and dtype == torch.int32:
        assert "cp_async_16" in refill


def _nested_table_matmul(
    a: torch.Tensor,
    pages: torch.Tensor,
    order: torch.Tensor,
    lookup: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(pages.size(2))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        stop = lengths[lookup[0]]
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(0, stop):
            page = order[lookup[step]]
            acc = hl.dot(a[tm, :], pages[page, :, :], acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_nested_table_addresses_and_table_indexed_bounds(
    mode: str, dtype: torch.dtype
) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((5, 64, 64), dtype=torch.bfloat16),
        torch.empty((5,), dtype=dtype),
        torch.empty((4,), dtype=dtype),
        torch.empty((5,), dtype=torch.int32),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_nested_table_matmul, backend="cute", static_shapes=True),
            args,
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            assert len(program.index_loads) == 3
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=3,
                    cute_mma_dag_loads="tma",
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    assert source.index("lookup.iterator.toint()") < source.index(
        "lengths.iterator.toint()"
    )


def _batch_paged_matmul(
    a: torch.Tensor, b: torch.Tensor, order: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(1), hl.specialize(b.size(2))
    hl.specialize(a.size(2))
    out = torch.empty([order.size(0), m, n], dtype=torch.float32, device=a.device)
    for batch, tm in hl.tile([order.size(0), m], block_size=[1, None]):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(2):
            page = order[batch.id, step]
            acc = hl.dot(a[page, tm, :], b[page, :, :], acc=acc)
        out[batch.id, tm, :] = acc
    return out


def _guarded_invariant_table_matmul(
    a: torch.Tensor,
    pages: torch.Tensor,
    order: torch.Tensor,
    lookup: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(pages.size(3))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(0, lengths[0]):
            page = order[lookup[0]]
            acc = hl.dot(a[tm, :], pages[page, step, :, :], acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_invariant_index_loads_stay_inside_the_loop_guard(mode: str) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((5, 4, 64, 64), dtype=torch.bfloat16),
        torch.empty((5,), dtype=torch.int32),
        torch.empty((1,), dtype=torch.int32),
        torch.empty((1,), dtype=torch.int32),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(
                _guarded_invariant_table_matmul, backend="cute", static_shapes=True
            ),
            args,
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            order = next(
                access
                for access in program.index_loads.values()
                if access.name == "order"
            )
            assert order.loops == frozenset({0})
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=3,
                    cute_mma_dag_loads="tma",
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    bases = re.findall(r"(md_base(?:_\d+)?) = (?:lookup|order)\.iterator", source)
    assert len(bases) == 2
    prologue, guard, body = source.partition("if 0 < md_scalar:")
    assert guard
    for base in bases:
        assert f"ldg_s32({base}" not in prologue
        assert f"ldg_s32({base}" in body


def test_single_loop_persistent_does_not_prefetch_another_tiles_page_table() -> None:
    args = (
        torch.empty((5, 64, 64), dtype=torch.bfloat16),
        torch.empty((5, 64, 64), dtype=torch.bfloat16),
        torch.empty((512, 2), dtype=torch.int32),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_batch_paged_matmul, backend="cute", static_shapes=True), args
        )
        with bound.env:
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag="tcgen05",
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=3,
                    cute_mma_dag_loads="tma",
                    cute_mma_dag_persistent=1,
                )
            )
        source = bound.to_code(config)
    assert "for virtual_pid in range(" in source
    assert "_helion_mma_dag" in source
    assert "md_npid" not in source
