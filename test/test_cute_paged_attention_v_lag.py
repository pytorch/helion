from __future__ import annotations

import pytest

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_fixture
from helion._testing.cute_paged_v_lag import paired_v_lag_bound
from helion._testing.cute_paged_v_lag import paired_v_lag_config

pytestmark = pytest.mark.cpu_only
LAG = "cute_mma_dag_v_lag"
SLEEP = "cute_mma_dag_sleep_waits"
QUERY = "cute_mma_dag_query_tma"


def _paired_config(key, value):
    config = paired_v_lag_config(0)
    config.config[key] = value
    return config


@pytest.mark.parametrize(
    "key,value",
    [(LAG, lag) for lag in range(4)]
    + [(key, value) for key in (SLEEP, QUERY) for value in (False, True)],
)
def test_paired_knob_roundtrip_and_codegen(key, value):
    bound = paired_v_lag_bound()
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        fields = bound.config_spec._flat_fields()
        fragment = {
            LAG: bound.config_spec.cute_mma_dag_v_lag,
            SLEEP: bound.config_spec.cute_mma_dag_sleep_waits,
            QUERY: bound.config_spec.cute_mma_dag_query_tma,
        }[key]
        assert fragment is not None and fields[key] is fragment
        assert fragment.choices == ((0, 1, 2, 3) if key == LAG else (False, True))
        default = 0 if key == LAG else False
        assert fragment.default() == default
        assert type(fragment.default()) is type(default)
        config = bound.config_spec.normalized_config(_paired_config(key, value))
        assert config.get(key) == value
        assert type(config.get(key)) is type(value)
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
        again = bound.config_spec.normalized_config(
            helion.Config.from_dict(config.config)
        )
        assert again == config
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "paged_attention_paired_body(" in source
    marker = (
        f"v_lag={value}"
        if key == LAG
        else ("sleep_waits=True" if key == SLEEP else "query_map=pa_query_map")
    )
    assert (marker in source) == bool(value)


@pytest.mark.parametrize(
    "key,value",
    # Pin both lag endpoints and all rejected scalar types. In particular,
    # Python bool/int equality must not make either knob accept the other type.
    [(LAG, value) for value in (-1, 4, True, 0.0, "1", None)]
    + [(key, value) for key in (SLEEP, QUERY) for value in (1, 0, "True", None)],
)
def test_paired_knob_rejects_invalid_values(key, value):
    bound = paired_v_lag_bound()
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        with pytest.raises(helion.exc.InvalidConfig, match=key):
            bound.config_spec.normalized_config(_paired_config(key, value))
        assert not bound.env.backend.autotune_config_is_viable(
            bound.config_spec, _paired_config(key, value)
        )


@pytest.mark.parametrize("heads,per_key", [(3, True), (2, False)])
@pytest.mark.parametrize("key,value", [(LAG, 3), (SLEEP, True), (QUERY, True)])
def test_paired_knob_requires_paired_transport(heads, per_key, key, value):
    bound = paired_v_lag_bound(heads=heads, per_key=per_key)
    assert {
        LAG: bound.config_spec.cute_mma_dag_v_lag,
        SLEEP: bound.config_spec.cute_mma_dag_sleep_waits,
        QUERY: bound.config_spec.cute_mma_dag_query_tma,
    }[key] is None
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        assert key not in bound.config_spec._flat_fields()
        config = _paired_config(key, value)
        config.config["cute_mma_dag_pass_warps"] = "attention"
        with pytest.raises(helion.exc.InvalidConfig, match=key):
            bound.config_spec.normalized_config(config)


@pytest.mark.parametrize(
    "key,value,default", [(LAG, 3, 0), (SLEEP, True, False), (QUERY, True, False)]
)
def test_nonpaired_schedule_pins_inert_knobs(key, value, default):
    bound = paired_v_lag_bound()
    assert bound.host_function is not None
    config = _paired_config(key, value)
    config.config["cute_mma_dag_pass_warps"] = "attention"
    with bound.env, bound.host_function:
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.get(key) == default
        assert type(normalized.get(key)) is type(default)
        assert bound.env.backend.autotune_config_is_viable(
            bound.config_spec, normalized
        )


@pytest.mark.parametrize(
    "key,value,default", [(LAG, 3, 0), (SLEEP, True, False), (QUERY, True, False)]
)
def test_failed_mapped_repair_pins_inert_knobs(key, value, default):
    kernel, args = optional_lse_fixture("mapped", True, query_len=5)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
    assert bound.host_function is not None
    config = _paired_config(key, value)
    # Paired attention permits one CTA per SM. The mapped source has no
    # ordinary MMA fallback when repairing this incompatible occupancy.
    config.config["cute_min_blocks_per_mp"] = 2
    with bound.env, bound.host_function:
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert config.get("cute_mma_dag") == "off"
        assert config.get("cute_mma_dag_pass_warps") == "off"
        assert config.get(key) == default
        assert type(config.get(key)) is type(default)
        again = helion.Config.from_dict(config.config)
        bound.config_spec.normalize(again, _fix_invalid=True)
        assert again == config
