from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute.chained_contraction_groups import contraction_groups
from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
from helion._compiler.cute.chained_tcgen_stage import stage_geometry
from helion._compiler.cute.contraction_region import collect_contraction_region
from helion._compiler.device_ir import RootGraphInfo
import helion.language as hl
from helion.language import scan_ops
from helion.language.matmul_ops import dot

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.autotuner.config_spec import ConfigSpec


def _input(graph: Graph, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> Node:
    node = graph.placeholder(name)
    node.meta["val"] = torch.empty(shape, dtype=dtype)
    return node


def _call(
    graph: Graph,
    target: Callable[..., object],
    args: tuple[Any, ...],
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> Node:
    node = graph.call_function(target, args)
    node.meta["val"] = torch.empty(shape, dtype=dtype)
    return node


def _convert(graph: Graph, source: Node, dtype: torch.dtype) -> Node:
    return _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (source, dtype),
        tuple(source.meta["val"].shape),
        dtype,
    )


def _dot(graph: Graph, lhs: Node, rhs: Node, acc: Node | None = None) -> Node:
    return _call(
        graph,
        dot,
        (lhs, rhs, acc, torch.float32),
        (lhs.meta["val"].shape[0], rhs.meta["val"].shape[1]),
        torch.float32,
    )


def _plan(graph: Graph, outputs: tuple[Node, ...]) -> ChainedMatmulPlan:
    output = graph.output(outputs)
    graph.lint()
    region = collect_contraction_region(RootGraphInfo(0, graph))
    assert region is not None
    return ChainedMatmulPlan(
        root_graph_id=0,
        dots=tuple(spec.node for spec in region.contractions),
        store=output,
        axes=((0, 1, 2),) * len(region.contractions),
        shapes=tuple(
            (
                spec.lhs.meta["val"].shape[0],
                spec.rhs.meta["val"].shape[1],
                spec.lhs.meta["val"].shape[1],
            )
            for spec in region.contractions
        ),
        dtype=region.contractions[0].operand_dtypes[0],
        threads=128,
        region=region,
    )


def _groups(plan: ChainedMatmulPlan) -> tuple[tuple[int, ...], ...]:
    geometries = tuple(stage_geometry(shape) for shape in plan.shapes)
    assert all(geometry is not None for geometry in geometries)
    return tuple(
        group.stages
        for group in contraction_groups(
            plan, tuple(geometry for geometry in geometries if geometry is not None)
        )
    )


@pytest.mark.parametrize("through_pointwise", [False, True])
def test_explicit_accumulator_dependency_prevents_grouping(
    through_pointwise: bool,
) -> None:
    graph = Graph()
    common = _input(graph, "common", (128, 16), torch.bfloat16)
    rhs = _input(graph, "rhs", (16, 64), torch.bfloat16)
    seed = _input(graph, "seed", (128, 64), torch.float32)
    first = _dot(graph, common, rhs)
    acc = (
        _call(
            graph,
            torch.ops.aten.add.Tensor,
            (first, seed),
            (128, 64),
            torch.float32,
        )
        if through_pointwise
        else first
    )
    second = _dot(graph, common, rhs, acc)
    plan = _plan(graph, (first, second))
    assert plan.region is not None
    assert plan.region.contractions[1].accumulator is acc
    assert _groups(plan) == ((0,), (1,))


@pytest.mark.parametrize("dependent_stage", [0, 1])
def test_accumulator_dependency_on_any_group_member_prevents_extension(
    dependent_stage: int,
) -> None:
    graph = Graph()
    common = _input(graph, "common", (128, 16), torch.bfloat16)
    rhs = _input(graph, "rhs", (16, 64), torch.bfloat16)
    independent = (_dot(graph, common, rhs), _dot(graph, common, rhs))
    third = _dot(graph, common, rhs, independent[dependent_stage])
    assert _groups(_plan(graph, (*independent, third))) == ((0, 1), (2,))


@pytest.mark.parametrize(
    "extra_casts",
    [(), (torch.float16,), (torch.bfloat16, torch.float32)],
)
def test_common_left_requires_identical_narrowing_history(
    extra_casts: tuple[torch.dtype, ...],
) -> None:
    graph = Graph()
    common = _input(graph, "common", (128, 16), torch.float32)
    first_rhs = _input(graph, "first_rhs", (16, 64), torch.bfloat16)
    second_rhs = _input(graph, "second_rhs", (16, 64), torch.bfloat16)
    first_left = _convert(graph, common, torch.bfloat16)
    second_left = common
    for dtype in extra_casts:
        second_left = _convert(graph, second_left, dtype)
    second_left = _convert(graph, second_left, torch.bfloat16)
    first = _dot(graph, first_left, first_rhs)
    second = _dot(graph, second_left, second_rhs)
    assert _groups(_plan(graph, (first, second))) == (
        ((0,), (1,)) if extra_casts else ((0, 1),)
    )


@pytest.mark.parametrize("collective", ["scan", "reduction"])
@pytest.mark.parametrize("between", [False, True])
def test_grouping_does_not_cross_a_collective(collective: str, between: bool) -> None:
    graph = Graph()
    common = _input(graph, "common", (128, 16), torch.bfloat16)
    rhs = _input(graph, "rhs", (16, 64), torch.bfloat16)
    first = _dot(graph, common, rhs) if between else None
    if collective == "scan":
        barrier = _call(
            graph,
            scan_ops._associative_scan,
            (0, common, 1, False, False),
            (128, 16),
            torch.bfloat16,
        )
    else:
        barrier = _call(
            graph, torch.ops.aten.sum.dim_IntList, (common, [1]), (128,), torch.bfloat16
        )
    if first is None:
        first = _dot(graph, common, rhs)
    second = _dot(graph, common, rhs)
    plan = _plan(graph, (first, second, barrier))
    assert plan.region is not None
    assert (*plan.region.scans, *plan.region.reductions) == (barrier,)
    assert _groups(plan) == (((0,), (1,)) if between else ((0, 1),))


def test_mixed_operand_dtypes_are_not_grouped() -> None:
    graph = Graph()
    common = _input(graph, "common", (128, 16), torch.float32)
    bf16_rhs = _input(graph, "bf16_rhs", (16, 64), torch.bfloat16)
    fp16_rhs = _input(graph, "fp16_rhs", (16, 64), torch.float16)
    bf16_left = _convert(graph, common, torch.bfloat16)
    fp16_left = _convert(graph, common, torch.float16)
    first = _dot(graph, bf16_left, bf16_rhs)
    second = _dot(graph, fp16_left, fp16_rhs)
    plan = _plan(graph, (first, second))
    assert (plan.operand_dtype(0), plan.operand_dtype(1)) == (
        torch.bfloat16,
        torch.float16,
    )
    assert _groups(plan) == ((0,), (1,))


@pytest.mark.parametrize(
    ("widths", "expected"),
    [
        ((128, 128), ((0, 1),)),
        ((128, 128, 16), ((0, 1), (2,))),
        ((256, 16), ((0,), (1,))),
        # Logical 120 + 136 fits 256, but physical 128 + 144 does not.
        ((120, 136), ((0,), (1,))),
    ],
)
def test_group_width_limit_counts_physical_padding(
    widths: tuple[int, ...], expected: tuple[tuple[int, ...], ...]
) -> None:
    graph = Graph()
    common = _input(graph, "common", (128, 16), torch.bfloat16)
    right = tuple(
        _input(graph, f"rhs_{index}", (16, width), torch.bfloat16)
        for index, width in enumerate(widths)
    )
    dots = tuple(_dot(graph, common, rhs) for rhs in right)
    assert _groups(_plan(graph, dots)) == expected


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _group_config_candidate(left, right, initial):
    steps, m, k = left.shape
    n = right.shape[-1]
    history = torch.empty((steps, m, n), device=left.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[m, n]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            common = left[step.id, rows, kk]
            rhs = right[step.id, kk, cols]
            first = hl.dot(common, rhs, out_dtype=torch.float32)
            state = hl.dot(common, rhs, acc=state, out_dtype=torch.float32)
            history[step.id, rows, cols] = first
        final[rows, cols] = state
    return history, final


def _config_spec() -> ConfigSpec:
    with _cpu_codegen():
        bound = _group_config_candidate._bind_isolated(
            (
                torch.empty((3, 64, 16), dtype=torch.bfloat16),
                torch.empty((3, 16, 32), dtype=torch.bfloat16),
                torch.empty((64, 32), dtype=torch.float32),
            )
        )
    assert bound.config_spec.cute_chained_group_search_enabled
    return bound.config_spec


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_grouping_config_requires_bool(value: object) -> None:
    spec = _config_spec()
    with pytest.raises(
        exc.InvalidConfig, match="cute_chained_group_contractions must be bool"
    ):
        spec.normalized_config(
            helion.Config(
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
                cute_chained_group_contractions=value,
            )
        )


@pytest.mark.parametrize("schedule", [None, "cp_async"])
def test_grouping_config_requires_tcgen05(schedule: str | None) -> None:
    spec = _config_spec()
    config = helion.Config(num_warps=4, cute_chained_group_contractions=True)
    if schedule is not None:
        config.config["cute_chained_mma_schedule"] = schedule
    with pytest.raises(
        exc.InvalidConfig, match="requires a resident TCgen05 contraction loop"
    ):
        spec.normalized_config(config)
