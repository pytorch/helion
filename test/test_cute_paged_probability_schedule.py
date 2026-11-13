from __future__ import annotations

import pytest

import helion
from helion._compiler.cute.autotune.heuristics.cute_mma_dag import CuteMmaDagHeuristic
from helion._compiler.cute.cute_mma_dag import MMA_DAG_V_LAG_CHOICES
from helion._testing import cpu_render_target
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_reference import reference_attention_bound
from helion._testing.cute_paged_v_lag import paired_v_lag_bound
from helion._testing.cute_paged_v_lag import without_retirement_seeds

pytestmark = pytest.mark.cpu_only
KEY = "cute_mma_dag_probability_schedule"


def _config(value):
    config = optional_lse_config()
    config.config.update(
        block_sizes=[64],
        cute_mma_dag_pass_warps="attention_paired",
        cute_mma_dag_probability_schedule=value,
    )
    return config


@pytest.mark.parametrize("schedule", ["serial", "early", "early_tmem"])
def test_probability_schedule_reaches_selected_paired_render(schedule):
    bound, _ = reference_attention_bound("per_key", 5)
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        field = bound.config_spec._flat_fields()[KEY]
        assert field.choices == ("serial", "early", "early_tmem")
        config = bound.config_spec.normalized_config(_config(schedule))
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "paged_attention_paired_body(" in source
    assert "reference_scores=True" in source
    assert (f"probability_schedule={schedule!r}" in source) == (schedule != "serial")


@pytest.mark.parametrize("heads,per_key", [(1, True), (2, True), (2, False)])
def test_probability_schedule_is_absent_without_reference_proof(heads, per_key):
    bound = paired_v_lag_bound(heads=heads, per_key=per_key)
    assert bound.host_function is not None
    assert bound.config_spec.cute_mma_dag_probability_schedule is None
    config = _config("early_tmem")
    config.config["cute_mma_dag_pass_warps"] = "attention"
    with bound.env, bound.host_function:
        assert KEY not in bound.config_spec._flat_fields()
        with pytest.raises(helion.exc.InvalidConfig, match=KEY):
            bound.config_spec.normalized_config(config)


def test_nonpaired_reference_schedule_pins_probability_choice():
    bound, _ = reference_attention_bound("per_key", 5)
    assert bound.host_function is not None
    config = _config("early_tmem")
    config.config["cute_mma_dag_pass_warps"] = "attention"
    with bound.env, bound.host_function:
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.get(KEY) == "serial"
        assert bound.env.backend.autotune_config_is_viable(
            bound.config_spec, normalized
        )


@pytest.mark.parametrize("value", [True, 3, None, "unknown"])
def test_probability_schedule_invalid_values_and_repair(value):
    bound, _ = reference_attention_bound("per_key", 5)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        assert not bound.env.backend.autotune_config_is_viable(
            bound.config_spec, _config(value)
        )
        with pytest.raises(helion.exc.InvalidConfig, match=KEY):
            bound.config_spec.normalized_config(_config(value))
        repaired = _config(value)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired.get(KEY) == "serial"
        again = helion.Config.from_dict(repaired.config)
        bound.config_spec.normalize(again, _fix_invalid=True)
        assert again == repaired


def test_probability_seeds_share_admission_and_keep_serial_default():
    bound, _ = reference_attention_bound("per_key", 5)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        seeds = CuteMmaDagHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert seeds is not None
        assert all(not seed.get("cute_pdl_capture_overlap", False) for seed in seeds)
        assert all(
            seed.get("cute_pdl_release", "auto") != "loader_tail" for seed in seeds
        )
        seeds = without_retirement_seeds(seeds)
        assert seeds[0].get(KEY, "serial") == "serial"
        probed = [seed for seed in seeds if seed.get(KEY, "serial") != "serial"]
        assert sorted(
            (seed.get(KEY), seed.get("cute_mma_dag_v_lag", 0)) for seed in probed
        ) == [
            (schedule, lag)
            for schedule in ("early", "early_tmem")
            for lag in MMA_DAG_V_LAG_CHOICES
        ]
        assert seeds[-len(probed) :] == probed
        for seed in probed:
            normalized = bound.config_spec.normalized_config(seed)
            assert normalized.get(KEY) == seed.get(KEY)
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, normalized
            )
