from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import computed_fragment
from helion._compiler.cute import materialized_fission_codegen
from helion._compiler.cute import target_facts
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.kernel_generators import KERNEL_GENERATORS
from helion._compiler.cute.state import cute_env_state
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir_analysis import DeviceIRAnalysis
from helion._compiler.emitter_protocols import KernelTransforms
from helion._compiler.host_function import HostFunction
from helion._compiler.tile_dependency import build_tile_dependency_graph
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_dependent_full_votes import dependent_dot_votes
from helion._testing.cute_dependent_full_votes import dependent_full_votes
from helion._testing.cute_dependent_full_votes import explicit_dependent_votes
from helion._testing.cute_dependent_full_votes import opaque_host_dependent_votes
from helion._testing.cute_dependent_full_votes import three_dependent_votes
from helion._testing.cute_dependent_full_votes import while_dependent_votes
from helion._testing.cute_full_logical_votes import barrier_phase_full_votes
from helion._testing.cute_full_logical_votes import shared_full_votes
from helion._testing.cute_full_logical_votes import two_phase_full_votes
from helion.autotuner.config_generation import ConfigGeneration
from helion.language import _tracing_ops

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def phased_source():
    original = CuteBackend.defer_unmerged_reduction_chain
    proofs = []

    def inspect(self, result, source, dims):
        ir = DeviceIR.current()
        env = CompileEnvironment.current()
        assert len(ir.root_ids) == 2
        assert len(ir.phases) == 1
        assert not env.backend.roots_fissioned(env)
        analysis = DeviceIRAnalysis.build(ir, env)
        accesses = analysis.tile_accesses(ir, env, ir.host_function)
        graph = build_tile_dependency_graph(
            accesses, device_ir=ir, root_phases=(0, 0), world_size=1
        )
        assert graph.edges
        assert {(edge.producer_root, edge.consumer_root) for edge in graph.edges} == {
            (0, 1)
        }
        assert not graph.crosses_ranks()
        assert not graph.inband_allocation_ids
        proofs.append(graph)
        # Projection may use this proof, but may never pretend fission has
        # already happened and thereby skip the normal dependency analysis.
        return original(self, result, source, dims)

    with (
        cpu_render_target(fake_device_sm_count=148),
        patch.object(CuteBackend, "defer_unmerged_reduction_chain", inspect),
    ):
        bound = cpu_bind(dependent_full_votes, (torch.empty(3, 32, 32),))
        plan = cute_env_state(bound.env).dependent_phase_plan
        assert plan is not None and plan.confirmed
        ir = bound.host_function.device_ir
        assert [phase.roots for phase in ir.phases] == [[0], [1]]
        assert ir.implicit_dependency_starts == frozenset({1})
        code = bound.to_code(helion.Config())
    assert proofs
    return bound, code


def test_dependency_proof_is_available_before_reduction_merge(phased_source):
    bound, code = phased_source
    ir = bound.host_function.device_ir
    plan = cute_env_state(bound.env).dependent_phase_plan
    assert plan is not None
    assert plan.root_ids == tuple(ir.root_ids)
    assert plan.root_ids[1] > 0
    assert not plan.graph_regions[0] & plan.graph_regions[1]
    # Both roots have their own if captures; the second also has a for carry.
    # IDs are the original global graph IDs, not reindexed speculative copies.
    for region in plan.graph_regions:
        assert any(
            node.target is _tracing_ops._if
            for index in region
            for node in ir.graphs[index].graph.nodes
        )
    assert any(
        _tracing_ops.is_for_loop_target(node.target)
        for index in plan.graph_regions[1]
        for node in ir.graphs[index].graph.nodes
    )
    modules = stage_sources(code)
    assert len(modules) == 2
    assert all("fragment_smem" in module for module in modules)
    assert all("_helion_lane_reduce" not in module for module in modules)


def test_generated_hosts_capture_allocations_once_and_order_launches(
    phased_source, tmp_path
):
    bound, code = phased_source
    path = tmp_path / "dependent_full_votes.py"
    path.write_text(code)
    calls = []

    def launch(kernel, grid, *args, **kwargs):
        calls.append((kernel, grid, args, kwargs))

    x = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32)
    original = x.clone()
    with cpu_render_target(fake_device_sm_count=148):
        module = load_generated_module(path)
        output = module.dependent_full_votes(x, _launcher=launch)
    assert len(calls) == 2
    assert calls[0][0].__name__.endswith("__region_0")
    assert calls[1][0].__name__.endswith("__region_1")
    assert output.shape == (3,) and output.dtype == torch.bool
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    tensors = [
        [arg for arg in args if isinstance(arg, torch.Tensor)]
        for _, _, args, _ in calls
    ]
    common = [arg for arg in tensors[0] if any(arg is item for item in tensors[1])]
    assert len(common) == 1
    assert common[0].shape == x.shape and common[0] is not x
    assert not torch._C._is_alias_of(common[0], x)
    assert any(arg is output for arg in tensors[1])
    plan = cute_env_state(bound.env).dependent_phase_plan
    assert plan is not None
    outer = next(
        stmt
        for stmt in ast.parse(code).body
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "dependent_full_votes"
    )
    for name in plan.captures:
        assert (
            sum(
                isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
                for stmt in outer.body
            )
            == 1
        )
        for source in stage_sources(code):
            host = next(
                stmt
                for stmt in ast.parse(source).body
                if isinstance(stmt, ast.FunctionDef)
                and stmt.name.startswith("dependent_full_votes__region_")
            )
            assert name in [arg.arg for arg in host.args.kwonlyargs]
            assert not any(
                isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
                for stmt in host.body
            )


@pytest.mark.parametrize(
    "config", [{"pid_type": "persistent_blocked"}, {"cute_pdl": True}]
)
def test_ordered_launch_policy_remains_mandatory(phased_source, config):
    bound, _ = phased_source
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.InvalidConfig, match="flat ordered launches"),
    ):
        bound.to_code(helion.Config.from_dict(config))


def test_search_keeps_dependent_launches_without_pdl(phased_source):
    bound, _ = phased_source
    spec = bound.config_spec
    assert spec.cute_dependent_phase_launches
    with cpu_render_target(fake_device_sm_count=148):
        generation = ConfigGeneration(spec)
        fields = generation._key_to_flat_indices
        # The flat default already omits the PID coordinate for these roots.
        assert "pid_type" not in fields
        for key in ("cute_pdl", "cute_pdl_spread"):
            indices, is_sequence = fields[key]
            assert not is_sequence
            (index,) = indices
            assert generation.flat_spec[index].search_values() == [False]
        config = generation.unflatten(generation.flatten(spec.default_config()))
        assert config.pid_type == "flat" and not config["cute_pdl"]
        assert len(stage_sources(bound.to_code(config))) == 2


def test_generator_requires_the_actual_bound_phase_proof(phased_source):
    bound, _ = phased_source
    generator = KERNEL_GENERATORS[0]
    assert generator.name == "dependent_phases"
    with bound.env:
        assert generator.applies(
            bound.host_function, helion.Config(), KernelTransforms()
        )
    with cpu_render_target(fake_device_sm_count=148):
        ordinary = cpu_bind(shared_full_votes, (torch.empty(3, 32, 32),))
    with ordinary.env:
        assert cute_env_state(ordinary.env).dependent_phase_plan is None
        assert not ordinary.config_spec.cute_dependent_phase_launches
        generation = ConfigGeneration(ordinary.config_spec)
        indices, is_sequence = generation._key_to_flat_indices["cute_pdl"]
        assert not is_sequence
        (index,) = indices
        assert generation.flat_spec[index].search_values() == [False, True]
        # PDL and fragment choices cannot manufacture a dependency proof.
        for config in (helion.Config(), helion.Config(cute_pdl=True)):
            assert not generator.applies(
                ordinary.host_function, config, KernelTransforms()
            )


def test_second_stage_owner_cannot_fall_through(phased_source):
    bound, _ = phased_source
    second_root = bound.host_function.device_ir.root_ids[1]
    original = computed_fragment.computed_fragment_supported
    declined = []

    def refuse_second(env, graphs, **kwargs):
        if HostFunction.current().device_ir.root_ids == [second_root]:
            declined.append(second_root)
            return False
        return original(env, graphs, **kwargs)

    with (
        cpu_render_target(fake_device_sm_count=148),
        patch.object(computed_fragment, "computed_fragment_supported", refuse_second),
        pytest.raises(
            helion.exc.InvalidConfig,
            match="require a supported complete fragment root",
        ),
    ):
        bound.to_code(helion.Config())
    assert declined


def test_each_projected_stage_retains_shared_capacity_admission(phased_source):
    bound, _ = phased_source
    with (
        cpu_render_target(fake_device_sm_count=148),
        patch.object(target_facts, "per_cta_smem_capacity_bytes", return_value=16),
        pytest.raises(
            helion.exc.InvalidConfig,
            match="computed fragments need .* shared bytes, exceeding 16",
        ),
    ):
        bound.to_code(helion.Config())


@pytest.mark.parametrize("kernel", [two_phase_full_votes, barrier_phase_full_votes])
def test_independent_or_explicit_roots_do_not_gain_implicit_projection(kernel):
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="merge|reduction"),
    ):
        cpu_bind(kernel, (torch.empty(3, 32, 32), torch.empty(3, 32, 32)))


@pytest.mark.parametrize(
    "kernel",
    [
        while_dependent_votes,
        opaque_host_dependent_votes,
        explicit_dependent_votes,
        three_dependent_votes,
    ],
)
def test_unsupported_dependent_roots_fail_before_config_search(kernel):
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(kernel, (torch.empty(3, 32, 32),))


@pytest.mark.parametrize("fast_math", [False, True])
def test_computed_dot_values_use_normal_per_root_lowering(fast_math):
    original = materialized_fission_codegen._stage_host
    stages = []

    def inspect(host, root, stage_index, graphs, graph_ids, plan):
        stage = original(host, root, stage_index, graphs, graph_ids, plan)
        assert stage.compiler_state is host.compiler_state
        assert stage.local_types is host.local_types
        assert stage.device_ir.root_ids == [host.device_ir.root_ids[stage_index]]
        assert stage.device_ir.grid_block_ids == [
            host.device_ir.grid_block_ids[stage_index]
        ]
        assert stage.device_ir.task_families == [
            host.device_ir.task_families[stage_index]
        ]
        assert CompileEnvironment.current().settings.fast_math is fast_math
        assert not CompileEnvironment.current().backend.roots_fissioned(
            CompileEnvironment.current()
        )
        stages.append(stage)
        return stage

    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            helion.kernel(backend="cute", static_shapes=True, fast_math=fast_math)(
                dependent_dot_votes.fn
            ),
            (torch.empty(3, 32, 16), torch.empty(16, 32)),
        )
        with patch.object(materialized_fission_codegen, "_stage_host", inspect):
            code = bound.to_code(helion.Config())
    assert len(stages) == 2
    sources = stage_sources(code)
    assert len(sources) == 2
    assert all("fragment_smem" in source for source in sources)
    assert all("_helion_lane_reduce" not in source for source in sources)
