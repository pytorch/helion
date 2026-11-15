from __future__ import annotations

import ast
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import GraphInfo
from helion._compiler.device_ir import IfGraphInfo
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._compiler.device_ir import RolledReductionInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.device_ir import control_flow_child_graph_ids
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.rolled_control_flow import carried_reduction
from helion._testing.rolled_control_flow import else_only_carried_reduction
from helion._testing.rolled_control_flow import two_axis_carried_reductions
from helion.language import _tracing_ops

pytestmark = pytest.mark.cpu_only


def _reachable(
    graphs: list[GraphInfo], graph_id: int, active: frozenset[int] = frozenset()
) -> list[tuple[int, frozenset[int]]]:
    graph = graphs[graph_id]
    if isinstance(graph, ReductionLoopGraphInfo):
        # Each axis must have exactly one owner on a call path, including
        # through a branch or ordinary device loop.
        assert active.isdisjoint(graph.block_ids)
        active = active.union(graph.block_ids)
    result = [(graph_id, active)]
    for node in graph.graph.nodes:
        if node.target is _tracing_ops._if:
            branch = graphs[node.args[1]]
            assert isinstance(branch, IfGraphInfo)
            assert branch.else_branch is not None
            assert branch.else_branch == node.args[2]
        for child_id in control_flow_child_graph_ids(node):
            result.extend(_reachable(graphs, child_id, active))
    return result


@onlyBackends(["triton", "cute"])
def test_else_only_sweep_keeps_both_branch_edges_consistent(tmp_path, record_property):
    with cpu_render_target():
        bound = cpu_bind(else_only_carried_reduction, (torch.empty(2, 2, 128),))
        assert len(bound.config_spec.reduction_loops) == 1
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32]
        ir = bound.host_function.device_ir
        original = ir.semantic_debug_str()
        original_count = len(ir.graphs)
        rebindings = []
        bind_original = DeviceIR._bind_caller_rolled_graphs

        def inspect_rebinding(current, caller_bodies, inner_graph_ids):
            def standalone():
                return [
                    (
                        info.graph_id,
                        str(info.graph),
                        info.else_branch if isinstance(info, IfGraphInfo) else None,
                    )
                    for info in current.graphs[:original_count]
                ]

            before = standalone()
            bind_original(current, caller_bodies, inner_graph_ids)
            assert standalone() == before
            rebindings.append(sorted(caller_bodies))

        with (
            bound.env,
            bound.host_function,
            patch.object(DeviceIR, "_bind_caller_rolled_graphs", inspect_rebinding),
        ):
            graphs = ir.build_codegen_graphs(config)
        assert rebindings
        for root in ir.root_ids:
            _reachable(graphs, root)
        branches = [
            (info.graph_id, info.else_branch)
            for info in graphs
            if isinstance(info, IfGraphInfo)
        ]
        assert any(index >= original_count for index, _ in branches)
        proof = tmp_path / "branch-edges.json"
        proof.write_text(
            json.dumps({"branches": branches, "caller_bodies": rebindings})
        )
        record_property("branch_edges", str(proof))
        ast.parse(bound.to_code(config))
        assert ir.semantic_debug_str() == original


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("conditional", (False, True))
def test_rolled_child_uses_caller_sweep_and_keeps_standalone_entry(conditional):
    with cpu_render_target():
        bound = cpu_bind(carried_reduction, (torch.empty(2, 2, 128), conditional))
        ir = bound.host_function.device_ir
        original = ir.semantic_debug_str()
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32]
        assert bound._normalized_config_copy(config).config["reduction_loops"] == [32]
        with bound.env, bound.host_function:
            graphs = ir.build_codegen_graphs(config)
        assert ir.semantic_debug_str() == original
        paths = [item for root in ir.root_ids for item in _reachable(graphs, root)]
        child_ids = {
            info.original_graph_id
            for info in ir.rolled_reductions
            if info.can_be_rolled_by_caller
            and isinstance(ir.graphs[info.original_graph_id], ForLoopGraphInfo)
        }
        assert child_ids
        for child_id in child_ids:
            # This entry still works for a caller outside a reduction sweep.
            standalone = _reachable(graphs, child_id)
            assert any(active for _, active in standalone)
            # Inside the parent's sweep, that same child's body is reached
            # through a distinct entry, with no second reduction loop.
            child = ir.graphs[child_id]
            assert isinstance(child, ForLoopGraphInfo)
            owned = [
                entry_id
                for entry_id, active in paths
                if active
                and isinstance(entry := graphs[entry_id], ForLoopGraphInfo)
                and not isinstance(entry, ReductionLoopGraphInfo)
                and entry.block_ids == child.block_ids
            ]
            assert owned and child_id not in owned
        ast.parse(bound.to_code(config))
        assert ir.semantic_debug_str() == original


@onlyBackends(["triton", "cute"])
def test_independent_rolled_axes_keep_separate_child_contexts():
    with cpu_render_target():
        bound = cpu_bind(
            two_axis_carried_reductions, (torch.empty(2, 2, 128), torch.empty(2, 2, 96))
        )
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32, 32]
        assert bound._normalized_config_copy(config).config["reduction_loops"] == [
            32,
            32,
        ]
        ir = bound.host_function.device_ir
        original = ir.semantic_debug_str()
        with bound.env, bound.host_function:
            graphs = ir.build_codegen_graphs(config)
        paths = [item for root in ir.root_ids for item in _reachable(graphs, root)]
        assert len({axis for _, active in paths for axis in active}) == 2
        ast.parse(bound.to_code(config))
        assert ir.semantic_debug_str() == original


@onlyBackends(["triton", "cute"])
def test_apply_rolling_preserves_root_phase_metadata() -> None:
    class _FakeReductionLoopSpecs(list[object]):
        def config_get(
            self, values: list[int | None], block_id: int, default: object
        ) -> object:
            for idx, spec in enumerate(self):
                spec_obj = cast("SimpleNamespace", spec)
                if spec_obj.block_id == block_id:
                    return values[idx]
            return default

    class _FakeReductionRoller:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.graphs_added: list[int] = []

        def process(self, graph: torch.fx.Graph) -> torch.fx.Graph:
            return graph

    device_ir = DeviceIR()
    graph0 = torch.fx.Graph()
    graph0.output(())
    graph1 = torch.fx.Graph()
    graph1.output(())
    device_ir.add_root_graph(graph0)
    device_ir.add_root_graph(graph1)

    root0 = cast("RootGraphInfo", device_ir.graphs[device_ir.root_ids[0]])
    root1 = cast("RootGraphInfo", device_ir.graphs[device_ir.root_ids[1]])
    root0.phase_index = 0
    root1.phase_index = 1

    block_id = 7
    device_ir.rolled_reductions = [
        RolledReductionInfo(
            rolled_block_ids=[block_id],
            original_graph_id=1,
            used_rdim=True,
            can_be_rolled_by_caller=True,
        )
    ]

    reduction_specs = _FakeReductionLoopSpecs(
        [SimpleNamespace(block_id=block_id, persistent_reason=None)]
    )
    fake_env = SimpleNamespace(
        block_sizes=[SimpleNamespace(block_id=block_id, reduction=True)],
        config_spec=SimpleNamespace(reduction_loops=reduction_specs),
    )

    with (
        patch(
            "helion._compiler.device_ir.CompileEnvironment.current",
            return_value=fake_env,
        ),
        patch("helion._compiler.device_ir.ReductionRoller", _FakeReductionRoller),
    ):
        device_ir._apply_rolling(helion.Config(reduction_loops=[16]))

    rolled_root1 = cast("RootGraphInfo", device_ir.graphs[device_ir.root_ids[1]])
    assert rolled_root1 is root1
    assert rolled_root1.phase_index == 1
    assert device_ir.graphs[device_ir.root_ids[0]] is root0
    assert root0.phase_index == 0
