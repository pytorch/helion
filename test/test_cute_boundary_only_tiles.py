from __future__ import annotations

import ast
import copy
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.loop_nesting import boundary_only_grid_blocks
from helion._compiler.cute.thread_budget import tile_loop_thread_count
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import tile_ops

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


@helion.kernel(
    backend="cute",
    autotune_effort="none",
    config=helion.Config(block_sizes=[4, 4]),
)
def _coarse_copy(
    x: torch.Tensor, use_outer_elements: hl.constexpr = False
) -> torch.Tensor:
    width = hl.register_block_size(x.size(0))
    out = torch.empty_like(x)
    for outer in hl.tile(x.size(0), block_size=width):
        if use_outer_elements:
            outer_values = x[outer, :].sum(0)
        for inner in hl.tile(outer.begin, outer.end):
            value = x[inner, :]
            if use_outer_elements:
                value = value + outer_values[None, :]
            out[inner, :] = value
    return out


def _inactive(bound: BoundKernel) -> set[int]:
    with bound.env, bound.host_function:
        ir = bound.host_function.device_ir
        return boundary_only_grid_blocks(bound.env, ir, ir.graphs)


def test_boundary_only_grid_has_no_element_distribution() -> None:
    bound = _coarse_copy.bind((torch.empty(128, 64), False))
    assert _inactive(bound) == {0}
    config = helion.Config(block_sizes=[32, 32])
    ir = bound.host_function.device_ir
    with bound.env, bound.host_function:
        assert tile_loop_thread_count(bound.env, ir, ir.graphs, config) == 1024
        assert (
            tile_loop_thread_count(
                bound.env, ir, ir.graphs, config, inactive_block_ids={0}
            )
            == 32
        )


def test_outer_tensor_elements_keep_grid_active() -> None:
    bound = _coarse_copy.bind((torch.empty(128, 64), True))
    assert _inactive(bound) == set()


def test_other_scalar_use_keeps_grid_active() -> None:
    bound = _coarse_copy.bind((torch.empty(128, 64), False))
    ir = bound.host_function.device_ir
    graphs = copy.deepcopy(ir.graphs)
    with bound.env, bound.host_function:
        node = next(
            node
            for info in graphs
            for node in info.graph.nodes
            if isinstance(node.meta.get("val"), torch.SymInt)
            and bound.env.get_block_id(node.meta["val"]) == 0
        )
        with node.graph.inserting_after(node):
            node.graph.call_function(operator.add, (node, 1))
        assert boundary_only_grid_blocks(bound.env, ir, graphs) == set()


def test_bounds_used_outside_inner_loop_keep_grid_active() -> None:
    bound = _coarse_copy.bind((torch.empty(128, 64), False))
    ir = bound.host_function.device_ir
    graphs = copy.deepcopy(ir.graphs)
    begin = next(
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is tile_ops.tile_begin
    )
    # A free arange constructed from these bounds can borrow the coarse grid
    # coordinates. Unknown bound consumers must not deactivate its axis.
    with begin.graph.inserting_after(begin):
        begin.graph.call_function(operator.add, (begin, 1))
    with bound.env, bound.host_function:
        assert boundary_only_grid_blocks(bound.env, ir, graphs) == set()


def test_boundary_only_grid_preserves_partial_tile_bounds() -> None:
    bound = _coarse_copy.bind((torch.empty(130, 64), False))
    assert _inactive(bound) == {0}
    code = bound.to_code(helion.Config(block_sizes=[4, 4]))
    tile_end = next(
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "tile_end"
            for target in node.targets
        )
    )
    expression = compile(ast.Expression(tile_end), "<tile_end>", "eval")
    # Suppressing the element axis must retain the end clamp; otherwise the
    # last inner tile would read and write past row 129.
    for offset in range(0, 130, 4):
        end = eval(
            expression,
            {
                "cutlass": SimpleNamespace(Int32=int),
                "tile_offset_0": offset,
                "_BLOCK_SIZE_0": 4,
            },
        )
        assert end == min(offset + 4, 130)
    assert "indices_1 < tile_end" in code


@pytest.fixture(scope="module", params=("rms_norm", "layer_norm"))
def norm_bound(request: pytest.FixtureRequest) -> BoundKernel:
    from examples.layer_norm import layer_norm_bwd
    from examples.rms_norm import rms_norm_bwd

    x = torch.empty(256, 1024, dtype=torch.float16)
    weight = torch.empty(1024, dtype=torch.float16)
    if request.param == "rms_norm":
        example = rms_norm_bwd
        inputs = (torch.empty_like(x), x, weight, torch.empty(256, 1))
    else:
        example = layer_norm_bwd
        inputs = (
            torch.empty_like(x),
            x,
            torch.empty(256),
            torch.empty(256),
            weight,
            True,
        )
    kernel = helion.kernel(
        example.fn,
        backend="cute",
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    return kernel.bind(inputs)


def _synthetic_extents(code: str) -> set[int]:
    return {
        ast.literal_eval(node.iter.args[0])
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("synthetic_lane_")
        and isinstance(node.iter, ast.Call)
    }


def test_norm_backward_reuses_coarse_grid_threads(norm_bound: BoundKernel) -> None:
    assert _inactive(norm_bound) == {0}
    config = norm_bound.env.config_spec.default_config()
    with patch(
        "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
        return_value=128 * 1024,
    ):
        code = norm_bound.to_code(config)
        with patch(
            "helion._compiler.cute.loop_nesting.boundary_only_grid_blocks",
            return_value=set(),
        ):
            original = norm_bound.to_code(config)

    # The coarse grid's 32 redundant threads previously consumed the budget
    # needed by the feature dimension. Retain its tile width and row loop,
    # remove that duplicate distribution, and cover N with 32 lanes/thread.
    assert _synthetic_extents(original) == {1024}
    assert _synthetic_extents(code) == {32}
    # ND tile.begin now names the logical offset directly, so both versions
    # omit the unused element coordinate. The actual coarse/inner row domains
    # must remain identical despite the different feature-thread budgets.
    domains = []
    for generated in (original, code):
        tree = ast.parse(generated)
        inner = [
            ast.dump(node.iter)
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and any(
                isinstance(child, ast.Name) and child.id == "tile_offset_0"
                for child in ast.walk(node.iter)
            )
        ]
        ends = [
            ast.dump(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "tile_end"
                for target in node.targets
            )
        ]
        assert len(inner) == len(ends) == 2
        domains.append((inner, ends))
    assert domains[0] == domains[1]
    assert "indices_0" not in original
    assert "indices_0" not in code
    assert "_BLOCK_SIZE_0 = 32" in code
    assert "block=(32, 32, 1)" in code


def test_norm_backward_does_not_create_unused_coarse_lanes(
    norm_bound: BoundKernel,
) -> None:
    # A tuner-requested single thread used to create 64 scalar lanes for the
    # unused coarse tile. Inactivity must suppress both threads and lane loops.
    config = helion.Config(block_sizes=[64, 64], num_threads=[1, 64, 16])
    with patch(
        "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
        return_value=128 * 1024,
    ):
        code = norm_bound.to_code(config)
    assert "lane_0" not in code
    assert "indices_0" not in code
    assert "_BLOCK_SIZE_0 = 64" in code
