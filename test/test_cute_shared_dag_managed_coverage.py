"""Managed shared-DAG coverage reads registered live aliases, not partial facts."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test.test_cute_bool_aux_native import _args
from test.test_cute_bool_aux_native import _external_bool_epilogue
from test.test_cute_shared_epilogue_dag import configuration

import helion
from helion._compiler.autotuner_heuristics import cute_epilogue_fanout as coverage
from helion._compiler.cute import shared_epilogue_dag as shared
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.exc import BackendUnsupported

KEY = "tensor_storage_disjoint_matrix_v1"


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def kernel_and_args():
    policy = helion.CuteStructuralPolicy.from_dict(
        {
            "schema": "helion.cute.structural_policy",
            "version": 1,
            "settings": {
                "cute_region_fission": True,
                "cute_full_slice_matmul_tiling": True,
                "cute_segmented_matmul_tiling": True,
                "cute_flatten_nested_reductions": True,
                "cute_materialize_transformed_operands": True,
            },
        }
    )
    kernel = helion.kernel(
        _external_bool_epilogue.fn,
        settings=_external_bool_epilogue.settings.copy(autotune_effort="full"),
        cute_structural_policy=policy,
    )
    return kernel, (*_args(), "explicit")


def managed_bind(kernel, args):
    # cpu_bind supplies only target metadata mocks; dispatch/registration and
    # complete normal snapshot use the actual managed Kernel.bind implementation.
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        return cpu_bind(kernel, args)


def test_managed_coverage_before_snapshot_uses_live_classifier_without_publication():
    original = coverage.prove_epilogue_fanout
    observed = []

    def prove(env, host, graphs, stores, anchors, **kwargs):
        before = dict(env.bound_runtime_input_specialization_results)
        classifiers = dict(env.runtime_input_specializations)
        result = original(env, host, graphs, stores, anchors, **kwargs)
        assert env.bound_runtime_input_specialization_results == before
        assert env.runtime_input_specializations == classifiers
        if isinstance(result, shared.SharedDagFanoutPlan):
            observed.append(
                (before, classifiers, env.early_mma_metadata_scope_active, kwargs)
            )
        return result

    kernel, args = kernel_and_args()
    with patch.object(coverage, "prove_epilogue_fanout", prove):
        bound = managed_bind(kernel, args)
    assert observed
    assert all(
        KEY not in facts
        and KEY in registered
        and not scope
        and kw == {"allow_unbound": True}
        for facts, registered, scope, kw in observed
    )
    assert KEY in bound.env.bound_runtime_input_specialization_results
    assert set(bound.env.bound_runtime_input_specialization_results) == set(
        bound.env.runtime_input_specializations
    )
    assert len(bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans) == 1
    assert bound.config_spec._cute_tcgen05_config.epilogue_fanout_search_enabled
    source = bound.to_code(configuration())
    assert source.count("cutlass.Uint8(0)") == 1
    assert source.count("PipelineTmaStore.create(") == 4


def test_managed_and_isolated_policy_keep_same_source_and_complete_shared_plan():
    kernel0, args0 = kernel_and_args()
    kernel1, args1 = kernel_and_args()
    managed = managed_bind(kernel0, args0)
    isolated = cpu_bind(kernel1, args1)
    assert managed.to_code(configuration()) == isolated.to_code(configuration())
    assert managed.to_code(configuration(shared=False)) == isolated.to_code(
        configuration(shared=False)
    )
    assert (
        managed.config_spec.cache_fingerprint_hash()
        == isolated.config_spec.cache_fingerprint_hash()
    )


@pytest.mark.parametrize(
    "fault",
    (
        "no_registration",
        "missing_args",
        "fake_args",
        "contradictory_args",
        "bad_classifier_identity",
        "bad_reusable_property",
        "contradictory_bound_fact",
    ),
)
def test_explicit_live_coverage_is_not_an_alias_proof_bypass(fault):
    kernel, args = kernel_and_args()
    bound = managed_bind(kernel, args)
    state = bound.config_spec._cute_tcgen05_config
    (plan,) = state.epilogue_fanout_plans
    graphs = bound.host_function.device_ir.graphs
    anchors = {
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    }
    values = dict(zip(kernel.signature.parameters, args, strict=True))
    env = bound.env
    saved_facts = dict(env.bound_runtime_input_specialization_results)
    classifiers = dict(env.runtime_input_specializations)
    env.bound_runtime_input_specialization_results = {}
    if fault == "no_registration":
        env.runtime_input_specializations.pop(KEY)
    elif fault == "missing_args":
        values = {}
    elif fault == "fake_args":
        with FakeTensorMode() as mode:
            values = {
                name: mode.from_tensor(value)
                if isinstance(value, torch.Tensor)
                else value
                for name, value in values.items()
            }
    elif fault == "contradictory_args":
        values["out1"] = values["out0"]
    elif fault == "bad_classifier_identity":
        env.runtime_input_specializations[KEY] = replace(
            classifiers[KEY], classifier_identity=("unproved",)
        )
    elif fault == "bad_reusable_property":
        env.runtime_input_specializations[KEY] = replace(
            classifiers[KEY], reusable_tensor_properties=frozenset()
        )
    elif fault == "contradictory_bound_fact":
        env.bound_runtime_input_specialization_results[KEY] = tuple(
            False for _ in saved_facts[KEY]
        )
    try:
        with env, bound.host_function, env.use_runtime_arg_values(values):
            before = dict(env.bound_runtime_input_specialization_results)
            registered = dict(env.runtime_input_specializations)
            assert (
                shared.prove_shared_dag_fanout(
                    env,
                    bound.host_function,
                    graphs,
                    plan.stores,
                    anchors,
                    allow_unbound=True,
                )
                is None
            )
            assert env.bound_runtime_input_specialization_results == before
            assert env.runtime_input_specializations == registered
    finally:
        env.bound_runtime_input_specialization_results = saved_facts
        env.runtime_input_specializations = classifiers


def test_final_codegen_still_requires_complete_bound_alias_snapshot():
    kernel, args = kernel_and_args()
    bound = managed_bind(kernel, args)
    env = bound.env
    saved = env.bound_runtime_input_specialization_results.pop(KEY)
    try:
        with pytest.raises(BackendUnsupported):
            bound.to_code(configuration())
        assert KEY not in env.bound_runtime_input_specialization_results
    finally:
        env.bound_runtime_input_specialization_results[KEY] = saved
    assert bound.to_code(configuration())


def test_managed_cache_alias_rebind_does_not_reuse_disjoint_fanout():
    kernel, args = kernel_and_args()
    original = managed_bind(kernel, args)
    changed = list(args)
    changed[8] = changed[7]
    rebound = managed_bind(kernel, tuple(changed))
    assert rebound is not original
    assert original.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    assert not rebound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    assert (
        rebound.env.bound_runtime_input_specialization_results[KEY]
        != original.env.bound_runtime_input_specialization_results[KEY]
    )
