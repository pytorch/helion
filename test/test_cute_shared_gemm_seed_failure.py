"""Optional registration is transactional; explicit config/proof errors stay loud."""

from __future__ import annotations

import logging
import random
from unittest.mock import patch

import pytest
import torch

from test.test_cute_shared_gemm_seeds import bind
from test.test_cute_shared_gemm_seeds import cpu_only as cpu_only
from test.test_cute_shared_gemm_seeds import generated
from test.test_cute_shared_gemm_seeds import kernel
from test.test_cute_shared_gemm_seeds import values

import helion
from helion._compiler.cute import shared_gemm_seeds as seeds
from helion.exc import InvalidConfig


def state_record(env):
    spec = env.config_spec
    return {
        "seed_list": spec.compiler_seed_configs.copy(),
        "default": spec.default_config(),
        "cache": spec.cache_fingerprint_hash(),
        "structural": spec.structural_fingerprint_hash(),
        "overflow": spec.compiler_seed_overflow_requested,
        "retry": spec.compiler_seed_timeout_retry_repetitions,
        "heuristics": spec.autotuner_heuristics.copy(),
        "facts": env.compiler_fact_specialization_facts,
    }


@pytest.mark.parametrize(
    "phase", ("before", "after_yield", "mutation_then_failure", "dedup_failure")
)
def test_real_managed_bind_survives_optional_failure_with_exact_registry_cache_rollback(
    phase, caplog
):
    snapshots = []
    registry_refs = []
    heuristic_refs = []
    original = seeds.append_shared_gemm_seed_configs

    def traced(env, ir):
        before = state_record(env)
        assert not before["seed_list"]
        registry_refs.append(env.config_spec.compiler_seed_configs)
        heuristic_refs.append(env.config_spec.autotuner_heuristics)
        original(env, ir)
        assert state_record(env) == before
        assert env.config_spec.compiler_seed_configs is registry_refs[-1]
        assert env.config_spec.autotuner_heuristics is heuristic_refs[-1]
        snapshots.append(before)

    class FailedComparison(helion.Config):
        def __eq__(self, other):
            raise RuntimeError("injected optional dedup failure")

    def injected(env, ir):
        if phase == "mutation_then_failure":
            spec = env.config_spec
            spec.compiler_seed_configs.append(helion.Config(block_sizes=[128, 32, 64]))
            spec.compiler_seed_configs = [helion.Config(block_sizes=[64, 32, 64])]
            spec.compiler_default_config = helion.Config(block_sizes=[64, 64, 64])
            spec.compiler_seed_overflow_requested = True
            spec.compiler_seed_timeout_retry_repetitions = 999
            spec.autotuner_heuristics.append("injected-seed-facts")
            spec.autotuner_heuristics = ["injected-rebound-facts"]
            env.compiler_fact_specialization_facts = frozenset({"device_num_sm"})
        elif phase != "before":
            yield (
                FailedComparison(block_sizes=[128, 32, 64])
                if phase == "dedup_failure"
                else helion.Config(block_sizes=[128, 32, 64])
            )
            if phase == "dedup_failure":
                yield helion.Config(block_sizes=[64, 32, 64])
        raise RuntimeError("injected optional discovery failure")

    # The importer keeps its own hook reference; patch the registration call,
    # not the whole bound constructor or its explicit config normalizer.
    with (
        caplog.at_level(logging.DEBUG, logger=seeds.__name__),
        patch(
            "helion._compiler.autotuner_heuristics.append_shared_gemm_seed_configs",
            traced,
        ),
        patch.object(seeds, "shared_gemm_seed_configs", injected),
    ):
        bound = bind(kernel(), (*values(), 3))
    assert len(snapshots) == 1
    assert "optional shared-GEMM seed discovery failed" in caplog.text
    assert bound.config_spec.compiler_seed_configs == []
    assert bound.config_spec.default_config() == snapshots[0]["default"]
    assert bound.config_spec.cache_fingerprint_hash() == snapshots[0]["cache"]
    # The unchanged default remains source-generatable after the failed optional
    # addition; no output store or original proof was bypassed.
    assert "out0" in bound.to_code(bound.config_spec.default_config())


@pytest.mark.parametrize(
    "error",
    (
        RuntimeError,
        ValueError,
        KeyError,
        InvalidConfig,
        OSError,
        TypeError,
        AssertionError,
    ),
)
def test_only_optional_hook_catches_discovery_exceptions(error):
    bound = bind(kernel(), (*values(), 3))
    with bound.env, bound.host_function:
        registry = bound.config_spec.compiler_seed_configs
        before = registry.copy()
        registry.clear()
        try:
            with patch.object(
                seeds,
                "prove_shared_gemm_seed_facts",
                side_effect=error("optional failure"),
            ):
                with pytest.raises(error):
                    seeds.shared_gemm_seed_configs(
                        bound.env, bound.host_function.device_ir
                    )
                seeds.append_shared_gemm_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
            assert registry == []
        finally:
            registry[:] = before


@pytest.mark.parametrize("error", (KeyboardInterrupt, SystemExit))
def test_process_control_exceptions_are_not_suppressed(error):
    bound = bind(kernel(), (*values(), 3))
    with bound.env, bound.host_function:
        registry = bound.config_spec.compiler_seed_configs
        before = registry.copy()
        registry.clear()
        try:
            with (
                patch.object(seeds, "shared_gemm_seed_configs", side_effect=error()),
                pytest.raises(error),
            ):
                seeds.append_shared_gemm_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
        finally:
            registry[:] = before


def test_nonempty_seed_registry_remains_noop_and_does_not_invoke_discovery():
    bound = bind(kernel(), (*values(), 3))
    before = list(bound.config_spec.compiler_seed_configs)
    assert before
    with bound.env, bound.host_function:
        with patch.object(
            seeds,
            "shared_gemm_seed_configs",
            side_effect=RuntimeError("must not execute"),
        ) as discovery:
            seeds.append_shared_gemm_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        discovery.assert_not_called()
    assert bound.config_spec.compiler_seed_configs == before


def test_successful_discovery_commits_once_after_completion_without_RNG_change():
    bound = bind(kernel(), (*values(), 3))
    configs = generated(bound)
    rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    with bound.env, bound.host_function:
        registry = bound.config_spec.compiler_seed_configs
        before = registry.copy()
        registry.clear()

        def staged(env, ir):
            assert registry == []
            yield configs[0]
            assert registry == []
            yield configs[0]  # duplicate is removed without early publication
            assert registry == []
            yield configs[-1]

        try:
            with patch.object(seeds, "shared_gemm_seed_configs", staged):
                seeds.append_shared_gemm_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
            assert registry == [configs[0], configs[-1]]
        finally:
            registry[:] = before
    assert random.getstate() == rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_explicit_selected_bad_config_and_lost_shared_proof_still_raise():
    bound = bind(kernel(), (*values(), 3))
    selected = generated(bound)[0]
    with pytest.raises(InvalidConfig):
        bound._normalized_config_copy(
            helion.Config.from_dict({**selected.config, "tcgen05_num_epi_warps": 999})
        )
    state = bound.config_spec._cute_tcgen05_config
    with (
        patch.object(state, "epilogue_fanout_plans", ()),
        pytest.raises(InvalidConfig),
    ):
        bound.to_code(selected)
