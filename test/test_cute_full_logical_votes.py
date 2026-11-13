from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.backend import Backend
from helion._compiler.cute import computed_fragment
from helion._compiler.cute import target_facts
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.emitters import ComputedFragmentRootEmitter
from helion._compiler.cute.full_reduction_owner import fragment_store_destinations
from helion._compiler.cute.full_reduction_owner import requires_full_reduction_owner
from helion._compiler.device_function import DeviceFunction
from helion._compiler.runtime_tensor_guards import (
    _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY,
)
from helion._compiler.runtime_tensor_guards import (
    tensor_pairs_with_proven_disjoint_storage,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_full_logical_votes import barrier_phase_full_votes
from helion._testing.cute_full_logical_votes import captured_full_votes
from helion._testing.cute_full_logical_votes import (
    captured_reshape_grid_axis_full_votes,
)
from helion._testing.cute_full_logical_votes import carried_complete_votes
from helion._testing.cute_full_logical_votes import carried_tile_domain_votes
from helion._testing.cute_full_logical_votes import conditional_result_tile_votes
from helion._testing.cute_full_logical_votes import dot_full_vote
from helion._testing.cute_full_logical_votes import grid_axis_full_votes
from helion._testing.cute_full_logical_votes import initial_tile_domain_votes
from helion._testing.cute_full_logical_votes import literal_reshape_grid_axis_full_votes
from helion._testing.cute_full_logical_votes import loop_axis_full_votes
from helion._testing.cute_full_logical_votes import output_argument_full_votes
from helion._testing.cute_full_logical_votes import reshaped_grid_axis_full_votes
from helion._testing.cute_full_logical_votes import shared_full_votes
from helion._testing.cute_full_logical_votes import two_phase_full_votes
from helion._testing.cute_full_logical_votes import while_carried_tile_votes
from helion._testing.cute_full_logical_votes import while_condition_tile_votes

pytestmark = pytest.mark.cpu_only


def _global_stores_have_coordinate_owners(code: str, threads: int = 128) -> None:
    """Inspect the emitted store schedule, not a successful-render sentinel."""
    module = ast.parse(code)
    device = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    parents = {
        id(child): parent
        for parent in ast.walk(device)
        for child in ast.iter_child_nodes(parent)
    }
    thread = next(
        node.targets[0].id
        for node in ast.walk(device)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and ast.unparse(node.value) == "cutlass.Int32(cute.arch.thread_idx()[0])"
    )
    stores = [
        node
        for node in ast.walk(device)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
    ]
    assert stores
    for store in stores:
        node = store
        owned = False
        while id(node) in parents:
            node = parents[id(node)]
            if not isinstance(node, ast.For):
                continue
            iterator = node.iter
            if not (
                isinstance(iterator, ast.Call) and ast.unparse(iterator.func) == "range"
            ):
                continue
            if len(iterator.args) != 3 or ast.unparse(iterator.args[0]) != thread:
                continue
            _, extent, stride = iterator.args
            if isinstance(extent, ast.Constant) and isinstance(stride, ast.Constant):
                assert 0 < extent.value <= stride.value == threads
                owned = True
        assert owned, ast.unparse(store)
    assert f"block=({threads}, 1, 1)" in code
    assert "_helion_lane_reduce" not in code


@pytest.mark.parametrize("kind", ("shared", "shared_square", "dot"))
def test_computed_full_votes_keep_complete_coordinates_and_unique_stores(kind):
    initialized = torch.cuda.is_initialized()
    if kind.startswith("shared"):
        columns = 32 if kind == "shared_square" else 128
        kernel, args = shared_full_votes, (torch.empty(3, 32, columns),)
    else:
        kernel, args = (
            dot_full_vote,
            (
                torch.empty(3, 32, 128, dtype=torch.bfloat16),
                torch.empty(128, 128, dtype=torch.bfloat16),
            ),
        )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert requires_full_reduction_owner(bound.host_function.device_ir.graphs)
        code = bound.to_code(helion.Config())
    _global_stores_have_coordinate_owners(code)
    assert "fragment_smem" in code and "cute.arch.sync_threads()" in code
    if kind == "dot":
        assert "fma.rn.f32" in code
    assert torch.cuda.is_initialized() == initialized


def test_required_vote_owner_cannot_decline_to_scalar_codegen():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(shared_full_votes, (torch.empty(3, 32, 128),))
        with (
            patch.object(
                computed_fragment, "codegen_computed_fragment_root", return_value=False
            ) as emit,
            pytest.raises(
                helion.exc.InvalidConfig,
                match="require a supported complete fragment root",
            ),
        ):
            bound.to_code(helion.Config())
        assert emit.call_count == 1


def test_required_vote_owner_retains_supported_operation_checks():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(shared_full_votes, (torch.empty(3, 32, 128),))
        with (
            patch.object(
                computed_fragment, "computed_fragment_supported", return_value=False
            ),
            pytest.raises(
                helion.exc.InvalidConfig,
                match="require a supported complete fragment root",
            ),
        ):
            bound.to_code(helion.Config())


def test_required_vote_owner_retains_shared_capacity_check():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(shared_full_votes, (torch.empty(3, 32, 128),))
        with (
            patch.object(target_facts, "per_cta_smem_capacity_bytes", return_value=16),
            patch.object(
                CuteBackend,
                "explain_root_failure",
                side_effect=AssertionError("capacity must reject directly"),
            ),
            pytest.raises(
                helion.exc.InvalidConfig,
                match="computed fragments need .* shared bytes, exceeding 16",
            ),
        ):
            bound.to_code(helion.Config())


def test_required_vote_owner_rejects_unknown_shared_capacity_at_bind():
    with (
        cpu_render_target(fake_device_sm_count=148),
        patch.object(target_facts, "per_cta_smem_capacity_bytes", return_value=0),
        pytest.raises(
            helion.exc.BackendUnsupported,
            match="require a known shared memory capacity",
        ),
    ):
        cpu_bind(shared_full_votes, (torch.empty(3, 32, 128),))


@pytest.mark.parametrize("kernel", (two_phase_full_votes, barrier_phase_full_votes))
def test_multiple_roots_fail_during_binding_before_config_search(kernel):
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(kernel, (torch.empty(3, 32, 128), torch.empty(5, 32, 128)))


@pytest.mark.parametrize(
    "kernel",
    (
        grid_axis_full_votes,
        reshaped_grid_axis_full_votes,
        literal_reshape_grid_axis_full_votes,
    ),
)
def test_reduced_grid_axis_with_tail_stays_outside_deferral(kernel):
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        # Three rows and a two-row tile include a masked grid tail. Neither
        # any nor all may erase that axis's mask without its own proof.
        cpu_bind(kernel, (torch.empty(3, 32, 32),))


def test_captured_literal_reshape_keeps_original_tile_domain():
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(
            captured_reshape_grid_axis_full_votes,
            (torch.empty(3, 32, 32), torch.empty(3, dtype=torch.bool)),
        )


def test_captured_complete_tensor_keeps_its_owner():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            captured_full_votes,
            (torch.empty(3, 32, 32), torch.empty(3, dtype=torch.bool)),
        )
        assert requires_full_reduction_owner(bound.host_function.device_ir.graphs)
        _global_stores_have_coordinate_owners(bound.to_code(helion.Config()))


def test_reduced_inner_loop_axis_with_tail_stays_outside_deferral():
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(loop_axis_full_votes, (torch.empty(2, 3, 32),))


def test_optional_emitter_patches_cannot_bypass_required_owner():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(shared_full_votes, (torch.empty(3, 32, 128),))
        with (
            patch.object(
                ComputedFragmentRootEmitter, "try_emit", return_value=False
            ) as tried,
            patch.object(
                ComputedFragmentRootEmitter, "emit", return_value=False
            ) as emitted,
        ):
            code = bound.to_code(helion.Config())
        assert tried.call_count == emitted.call_count == 0
        _global_stores_have_coordinate_owners(code)


@pytest.mark.parametrize("alias", (False, True))
def test_required_full_owner_proves_output_argument_disjointness(alias):
    left = torch.empty(3, 32)
    right = left.view_as(left) if alias else torch.empty_like(left)
    with cpu_render_target(fake_device_sm_count=148):
        args = (torch.empty(3, 32, 128), left, right)
        if alias:
            with pytest.raises(
                helion.exc.BackendUnsupported, match="require disjoint storage"
            ):
                cpu_bind(output_argument_full_votes, args)
        else:
            bound = cpu_bind(output_argument_full_votes, args)
            _global_stores_have_coordinate_owners(bound.to_code(helion.Config()))


def test_disjoint_discovery_respects_backend_policy_and_bound_default():
    args = (torch.empty(3, 32, 128), torch.empty(3, 32), torch.empty(3, 32))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(output_argument_full_votes, args)
        env = bound.env
        with (
            env,
            bound.host_function,
            env.use_runtime_arg_values(
                dict(zip(("x", "left", "right"), args, strict=True))
            ),
        ):
            destinations = fragment_store_destinations(
                bound.host_function.device_ir.graphs
            )
            assert len(destinations) == 2
            tensors = {tensor: tensor for tensor in destinations}
            expected = {frozenset(destinations)}
            assert tensor_pairs_with_proven_disjoint_storage(env, tensors) == expected
            key = _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
            saved = env.bound_runtime_input_specialization_results.pop(key)
            try:
                # Only explicit discovery may use the registered live classifier.
                assert tensor_pairs_with_proven_disjoint_storage(env, tensors) == set()
                assert (
                    tensor_pairs_with_proven_disjoint_storage(
                        env, tensors, allow_unbound=True
                    )
                    == expected
                )
                # A backend that does not own this runtime fact may veto it even
                # when registered live data proves the two inputs disjoint.
                with patch.object(
                    CuteBackend,
                    "tensors_proven_disjoint",
                    Backend.tensors_proven_disjoint,
                ):
                    assert (
                        tensor_pairs_with_proven_disjoint_storage(
                            env, tensors, allow_unbound=True
                        )
                        == set()
                    )
            finally:
                env.bound_runtime_input_specialization_results[key] = saved


@pytest.mark.parametrize(
    "slot",
    (
        "register_region_plan",
        "register_subblocks_plan",
        "gdn_recurrence_plan",
        "warp_mma_gemm_plan",
        "mma_dag_plan",
    ),
)
def test_required_full_owner_cannot_override_preselected_plan(slot):
    original = CuteBackend.pre_codegen

    def pre_codegen(self, *args, **kwargs):
        original(self, *args, **kwargs)
        setattr(
            DeviceFunction.current().cute_state.root_plans,
            slot,
            SimpleNamespace(threads=32),
        )

    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(shared_full_votes, (torch.empty(3, 32, 128),))
        with (
            patch.object(CuteBackend, "pre_codegen", pre_codegen),
            pytest.raises(
                helion.exc.InvalidConfig, match="cannot override a planned root"
            ),
        ):
            bound.to_code(helion.Config())


@pytest.mark.parametrize(
    "kind,threads,mode",
    (("shared", 64, "serial"), ("dot", 256, "warp"), ("dot", 128, "cta")),
)
def test_full_votes_admit_existing_fragment_knobs(kind, threads, mode):
    if kind == "shared":
        kernel, args = shared_full_votes, (torch.empty(3, 32, 32),)
    else:
        kernel, args = (
            dot_full_vote,
            (
                torch.empty(3, 32, 128, dtype=torch.bfloat16),
                torch.empty(128, 128, dtype=torch.bfloat16),
            ),
        )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        assert spec.cute_fragment_thread_root_ids
        # uint8 votes keep the serial combine. The dot's FP32 sibling max has
        # the existing warp/CTA choices; this does not widen uint8 eligibility.
        assert bool(spec.cute_fragment_reduction_root_ids) == (kind == "dot")
        requested = dict(spec.default_config()) | {"cute_fragment_threads": threads}
        if mode != "serial":
            requested["cute_fragment_reduction"] = mode
        _flat, config = spec.create_config_generation().strict_config_pair(
            helion.Config.from_dict(requested)
        )
        code = bound.to_code(config)
    _global_stores_have_coordinate_owners(code, threads)


def test_uint8_vote_refuses_unavailable_parallel_reduction_knob():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(shared_full_votes, (torch.empty(3, 32, 32),))
        spec = bound.config_spec
        assert not spec.cute_fragment_reduction_root_ids
        for mode in ("warp", "cta"):
            config = dict(spec.default_config()) | {"cute_fragment_reduction": mode}
            with pytest.raises(
                helion.exc.InvalidConfig, match="computed scalar reduction"
            ):
                spec.normalize(config)


@pytest.mark.parametrize("option", ("cute_collective_mma", "cute_register_chain"))
def test_conflicting_fragment_config_is_invalid_config(option):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            dot_full_vote,
            (
                torch.empty(3, 32, 128, dtype=torch.bfloat16),
                torch.empty(128, 128, dtype=torch.bfloat16),
            ),
        )
        with pytest.raises(
            helion.exc.InvalidConfig, match="unconflicted complete fragment root"
        ):
            bound.to_code(
                helion.Config.from_dict(
                    dict(bound.config_spec.default_config()) | {option: True}
                )
            )


def test_loop_carried_vote_includes_later_iteration_domain():
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(carried_tile_domain_votes, (torch.empty(2, 3, 32, 32),))


def test_complete_loop_carried_vote_keeps_its_owner():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(carried_complete_votes, (torch.empty(2, 3, 32, 32),))
        assert requires_full_reduction_owner(bound.host_function.device_ir.graphs)
        _global_stores_have_coordinate_owners(bound.to_code(helion.Config()))


@pytest.mark.parametrize("tile_in_then", (False, True))
def test_conditional_result_retains_both_branch_domains(tile_in_then):
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(
            conditional_result_tile_votes,
            (torch.empty(3, 32, 32), torch.empty(3, dtype=torch.bool), tile_in_then),
        )


@pytest.mark.parametrize("steps", (0, 2))
def test_loop_result_preserves_initial_tile_domain(steps):
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(initial_tile_domain_votes, (torch.empty(3, 32, 32), steps))


@pytest.mark.parametrize("site", ("body", "condition"))
def test_while_carried_vote_includes_body_domain(site):
    x = torch.empty(3, 32, 32)
    kernel, args = (
        (while_carried_tile_votes, (x, torch.empty(3, dtype=torch.int32)))
        if site == "body"
        else (while_condition_tile_votes, (x,))
    )
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(kernel, args)
