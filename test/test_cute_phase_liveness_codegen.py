"""Full static axes survive projection into an ordered consumer phase."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.materialized_fission_codegen import _region_graph_ids
from helion._compiler.cute.materialized_fission_codegen import _stage_host
from helion._compiler.cute.ordered_phases import _ordered_phase_bundle
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_phase_liveness import padded_factory_phases
from helion._testing.cute_phase_liveness import slice_sum_phases

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("tensor_indices", [False, True])
def test_projected_consumer_retains_static_memory_axes(tensor_indices):
    with cpu_target():
        bound = cpu_bind(
            slice_sum_phases,
            (
                torch.ones(2, 3, 7, 8, dtype=torch.bfloat16),
                torch.ones(2, 3, 8, 11, dtype=torch.bfloat16),
                tensor_indices,
            ),
        )
        config = dict(bound.config_spec.default_config()) | {
            "pid_type": "flat",
        }
        code = bound.to_code(helion.Config.from_dict(config))
    stages = stage_sources(code)
    assert len(stages) == 2
    assert "thread_idx()[0]" in stages[1] and "thread_idx()[1]" in stages[1]
    expected_block = "(7, 11, 1)" if tensor_indices else "(8, 16, 1)"
    assert f"block={expected_block}" in stages[1]


def test_factory_padding_does_not_resurrect_another_phases_axes():
    with cpu_target():
        bound = cpu_bind(padded_factory_phases, (torch.ones(8, 16),))
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            bundle, plan = _ordered_phase_bundle(bound.host_function, ir)
            graph_ids = _region_graph_ids(ir.graphs, ir.root_ids[1])
            stage = _stage_host(
                bundle, ir.phases[1].root_nodes[0], 1, ir.graphs, graph_ids, plan
            )
            unrelated = {bound.env.resolve_block_id(size) for size in (8, 16)}
            assert None not in unrelated
            assert unrelated.isdisjoint(stage.device_ir.codegen_active_block_ids)
            logical = {bound.env.resolve_block_id(size) for size in (7, 11)}
            assert None not in logical
            assert logical.issubset(stage.device_ir.codegen_active_block_ids)
            assert any(
                isinstance(value := node.meta.get("val"), torch.Tensor)
                and tuple(value.shape) == (8, 16)
                for graph in stage.device_ir.graphs
                for node in graph.graph.nodes
                if node.op == "call_function"
            )
