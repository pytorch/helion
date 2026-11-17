"""Reduction ownership is live even when all of its inputs are broadcast."""

from __future__ import annotations

import ast
import sys
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from helion._compiler.cute.analysis import broadcast_axis
from helion._compiler.cute.analysis.broadcast_axis import explicitly_replicated_axis
from helion._compiler.cute.lane_loops.markers import _lane_reduce_marker_expr
from helion._compiler.cute.lane_loops.markers import _LaneReduceMarker
from helion._compiler.cute.lane_loops.markers import required_lane_reduce_owners
from helion._compiler.cute.lane_loops.markers import set_marker_owner
from helion._compiler.cute.reduction_emitter import CuteReductionEmitter
from helion._compiler.tile_strategy import _live_lane_setup
from helion._testing import cpu_render_target
from helion._testing.cute_broadcast_reduction import BROADCAST_PHYSICAL_CONFIGS
from helion._testing.cute_broadcast_reduction import broadcast_count
from helion._testing.cute_codegen import cpu_bind

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("broadcast", "coordinate_read", "live"),
    [(True, False, True), (False, False, False), (False, True, True)],
)
def test_marker_owner_is_semantic_lane_dependency(
    broadcast: bool, coordinate_read: bool, live: bool
) -> None:
    expression = _lane_reduce_marker_expr(
        "value",
        "sum",
        "cutlass.Int64(0)",
        1,
        owner_lane="synthetic",
        broadcast_owner=broadcast,
    )
    body = list(ast.parse(f"result = {expression}").body)
    if coordinate_read:
        body.extend(ast.parse("value = synthetic + 1").body)
    needed, setup, splices = _live_lane_setup(
        body, [], {}, semantic_reads=required_lane_reduce_owners(body)
    )
    assert ("synthetic" in needed) is live
    assert not setup and not splices
    call = ast.parse(expression, mode="eval").body
    assert isinstance(call, ast.Call)
    marker = _LaneReduceMarker.from_call(call, result_var="", wrap_template="")
    assert marker.to_call() == expression


def test_rebinding_owner_retires_its_synthetic_replication_proof() -> None:
    body = ast.parse(
        "result = "
        + _lane_reduce_marker_expr(
            "value",
            "sum",
            "cutlass.Int64(0)",
            1,
            owner_lane="synthetic",
            broadcast_owner=True,
        )
    ).body
    statement = body[0]
    assert isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
    set_marker_owner(statement.value, "real_lane")
    assert not required_lane_reduce_owners(body)


def test_other_marker_fallback_does_not_prune_broadcast_owner() -> None:
    original = CuteReductionEmitter._lane_reduce_owner

    def owner(self, state, *, reshape_group=None):
        result = original(self, state, reshape_group=reshape_group)
        if reshape_group is None:
            # Model an earlier marker's recorded physical fallback on this
            # synthetic axis. This marker has its own sibling-group proof;
            # that older fallback must not make its live lane disposable.
            self.strategy.fn.cute_state.lanes.reshape_physical_fallbacks[result] = (
                1,
                4,
                "cutlass.Int32(cute.arch.thread_idx()[0])",
            )
        return result

    with (
        patch.object(CuteReductionEmitter, "_lane_reduce_owner", owner),
        cpu_render_target(),
    ):
        bound = cpu_bind(broadcast_count, (torch.ones(8), 8))
        config = bound.config_spec.default_config()
        config.config.update(num_threads=[2, 4], cute_vector_widths=[1, 1])
        source = bound.to_code(config)
    assert "_helion_lane_reduce" not in source
    assert any(
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("synthetic_lane_")
        for node in ast.walk(ast.parse(source))
    )


@pytest.mark.parametrize("copies", [2, 8])
def test_broadcast_reduction_keeps_its_default_lane_loop(copies: int) -> None:
    with cpu_render_target():
        bound = cpu_bind(broadcast_count, (torch.ones(8), copies))
        source = bound.to_code(bound.config_spec.default_config())
    assert "_helion_lane_reduce" not in source
    assert any(
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("synthetic_lane_")
        for node in ast.walk(ast.parse(source))
    )


@pytest.mark.parametrize(("copies", "threads", "width"), BROADCAST_PHYSICAL_CONFIGS)
def test_broadcast_reduction_keeps_its_physical_thread_extent(
    copies: int, threads: int, width: int
) -> None:
    with cpu_render_target():
        bound = cpu_bind(broadcast_count, (torch.ones(8), copies))
        config = bound.config_spec.default_config()
        config.config.update(num_threads=[2, threads], cute_vector_widths=[1, width])
        source = bound.to_code(config)
    launches = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert len(launches) == 1
    block = next(
        keyword.value for keyword in launches[0].keywords if keyword.arg == "block"
    )
    assert ast.literal_eval(block) == (threads, 2, 1)


@pytest.mark.parametrize("replicated", [True, False])
def test_reshape_does_not_manufacture_replication(replicated: bool) -> None:
    def expression(value):
        if replicated:
            return value.expand(4, 2).unsqueeze(-1)
        return value.reshape(4, 2, 1)

    graph = make_fx(expression)(torch.ones(4, 1) if replicated else torch.ones(2, 4))
    output = next(node for node in graph.graph.nodes if node.op == "output")
    assert explicitly_replicated_axis(output.args[0], 1) is replicated


def test_broadcast_axis_proof_visits_shared_dag_once() -> None:
    def expression(value):
        repeated = value.expand(4, 2)
        for _ in range(20):
            repeated = repeated + repeated.sin()
        return repeated

    graph = make_fx(expression)(torch.ones(4, 1))
    nodes = list(graph.graph.nodes)
    output = next(node for node in nodes if node.op == "output")
    with patch.object(
        broadcast_axis, "_replicated_axis", wraps=broadcast_axis._replicated_axis
    ) as visited:
        assert explicitly_replicated_axis(output.args[0], 1)
    # The shared diamond has linearly many FX nodes; proof work must not
    # recursively duplicate the predecessor proof down each incoming edge.
    assert visited.call_count <= len(nodes)


@pytest.mark.parametrize("replicated", [True, False])
def test_broadcast_axis_proof_handles_deep_pointwise_chains(replicated: bool) -> None:
    graph = torch.fx.Graph()
    value = torch.ones(4, 1 if replicated else 2)
    current = graph.placeholder("value")
    current.meta["val"] = value
    current = graph.call_function(torch.ops.aten.expand.default, (current, [4, 2]))
    current.meta["val"] = value.expand(4, 2)
    shape = torch.empty(4, 2)
    depth = max(2048, sys.getrecursionlimit() + 100)
    for _ in range(depth):
        current = graph.call_function(torch.ops.aten.sin.default, (current,))
        current.meta["val"] = shape
    with patch.object(
        broadcast_axis, "_replicated_axis", wraps=broadcast_axis._replicated_axis
    ) as visited:
        assert explicitly_replicated_axis(current, -1) is replicated
    assert visited.call_count <= len(graph.nodes)


@pytest.mark.parametrize("leading_axis", [False, True])
@pytest.mark.parametrize("replicated", [False, True])
def test_noop_and_reexpanded_axes_keep_the_source_proof(
    leading_axis: bool, replicated: bool
) -> None:
    def expression(value):
        repeated = value.expand(4, 2).sin()
        return repeated.expand(3, 4, 2) if leading_axis else repeated.expand(-1, -1)

    graph = make_fx(expression)(torch.ones(4, 1 if replicated else 2))
    output = next(node for node in graph.graph.nodes if node.op == "output")
    result = output.args[0]
    assert explicitly_replicated_axis(result, -1) is replicated
    assert not explicitly_replicated_axis(result, -2)
    if leading_axis:
        assert explicitly_replicated_axis(result, 0)


@pytest.mark.parametrize("reexpand", [1, 2])
def test_reexpanded_reduction_keeps_its_synthetic_owner(reexpand: int) -> None:
    with cpu_render_target():
        bound = cpu_bind(broadcast_count, (torch.ones(8), 8, reexpand))
        source = bound.to_code(bound.config_spec.default_config())
    assert "_helion_lane_reduce" not in source
    assert any(
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("synthetic_lane_")
        for node in ast.walk(ast.parse(source))
    )
