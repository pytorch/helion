"""Earlier-group coverage chronology, without changing any proof or group order."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from test.test_cute_shared_seed_schema import bind_inductor
from test.test_cute_shared_seed_schema import cpu_only as cpu_only
from test.test_cute_shared_seed_schema import fields

from helion._compiler import autotuner_heuristics as heuristics
from helion._compiler.cute import shared_gemm_seeds as seeds
from helion.autotuner.compiler_coverage import CompilerCoverageGroup
from helion.autotuner.compiler_coverage import CoverageWitness
from helion.autotuner.config_generation import ConfigGeneration


@pytest.mark.parametrize("early_failure", (False, True))
def test_real_prior_coverage_group_preserves_schema_and_early_failure_decision(
    tmp_path, early_failure
):
    phases = []
    original_cluster = heuristics.register_cluster_coverage
    original_fanout = heuristics.register_epilogue_fanout_coverage
    original_proof = seeds.prove_shared_gemm_seed_facts
    proof_calls = []

    def traced_fanout(env, ir, **kwargs):
        preparation = kwargs.get("prepare_only", False)
        phases.append("prepare" if preparation else "fanout")
        original_fanout(env, ir, **kwargs)
        if preparation:
            assert env.config_spec.compiler_coverage_groups == ()

    def earlier_group(env, ir):
        phases.append("cluster")
        spec = env.config_spec
        # A real scalar coverage group whose owned field is neither PID nor
        # persistence. Its snapshot exercises the exact old-field assertion
        # that naturally applies to earlier cluster coverage registrations.
        carrier = spec.default_config()
        spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism="test.prior_scalar_group",
                version=1,
                key="tcgen05_aux_load_placement",
                domain=("post_acc_wait", "pre_acc_wait"),
                legacy="post_acc_wait",
                witnesses=(CoverageWitness(carrier, "pre_acc_wait"),),
            )
        )
        original_cluster(env, ir)
        assert fields(spec) == spec._compiler_coverage_fields

    def transient_failure(env, ir):
        proof_calls.append(True)
        if len(proof_calls) == 1:
            raise RuntimeError("early optional probe unavailable")
        return original_proof(env, ir)

    with (
        patch.object(heuristics, "register_epilogue_fanout_coverage", traced_fanout),
        patch.object(heuristics, "register_cluster_coverage", earlier_group),
        patch.object(
            seeds,
            "prove_shared_gemm_seed_facts",
            transient_failure if early_failure else original_proof,
        ),
    ):
        _, bound, _ = bind_inductor(tmp_path)
    assert phases == ["prepare", "cluster", "fanout"]
    with bound.env, bound.host_function:
        spec = bound.config_spec
        spec.validate_compiler_coverage_groups()
        ConfigGeneration(spec).initial_population_view()
        assert fields(spec) == spec._compiler_coverage_fields
        assert [g.mechanism for g in spec.compiler_coverage_groups][:2] == [
            "test.prior_scalar_group",
            "cute.epilogue_fanout",
        ]
        if early_failure:
            assert not spec._cute_tcgen05_config._shared_gemm_seed_encoding_available
            assert spec.compiler_seed_configs == []
            assert len(proof_calls) == 1
            assert original_proof(bound.env, bound.host_function.device_ir) is not None
            seeds.append_shared_gemm_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            assert spec.compiler_seed_configs == []
        else:
            assert len(ConfigGeneration(spec).seed_flat_config_pairs()) == 48


def test_preparation_does_not_change_registered_carriers_or_group_order(tmp_path):
    _, bound, _ = bind_inductor(tmp_path)
    with bound.env, bound.host_function:
        spec = bound.config_spec
        before_groups = spec.compiler_coverage_groups
        before_seeds = spec.compiler_seed_configs.copy()
        before_fields = fields(spec)
        # Preparation is never run after registration in production. Its
        # carrier choices themselves are independent of group registration.
        spec.validate_compiler_coverage_groups()
        assert before_groups and before_seeds
        assert spec.compiler_coverage_groups == before_groups
        assert spec.compiler_seed_configs == before_seeds
        assert fields(spec) == before_fields
