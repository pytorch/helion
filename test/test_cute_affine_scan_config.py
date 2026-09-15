from __future__ import annotations

import json

import pytest

import helion
from helion import exc
from helion._compiler.backend import CuteBackend
from helion._compiler.backend import TritonBackend
from helion._compiler.cute.direct_affine_plan import DIRECT_AFFINE_SCHEDULES
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import CUTE_AFFINE_SCAN_SCHEDULE_KEY
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.search_space_logger import canonical_config_id


def _enabled_spec(*, step_count: int | None = 4) -> ConfigSpec:
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0))
    spec.enable_cute_affine_scan_search(step_count=step_count)
    return spec


def test_affine_scan_config_defaults_and_search_metrics() -> None:
    config = helion.Config()
    assert config.cute_affine_scan_schedule == "ordinary"

    spec = _enabled_spec()
    assert spec.default_config()[CUTE_AFFINE_SCAN_SCHEDULE_KEY] == "ordinary"
    dimensions = {
        dimension.name: dimension for dimension in spec.iter_search_dimensions()
    }
    assert dimensions[CUTE_AFFINE_SCAN_SCHEDULE_KEY].cardinality == 3
    assert dimensions[CUTE_AFFINE_SCAN_SCHEDULE_KEY].values == list(
        DIRECT_AFFINE_SCHEDULES
    )


def test_affine_scan_config_round_trip_and_cache_identity() -> None:
    spec = _enabled_spec()
    config = helion.Config(cute_affine_scan_schedule="direct_m16n16_v1")
    spec.normalize(config)

    assert dict(helion.Config.from_json(config.to_json())) == dict(config)
    assert json.loads(config.to_json())[CUTE_AFFINE_SCAN_SCHEDULE_KEY] == (
        "direct_m16n16_v1"
    )

    generation = ConfigGeneration(spec)
    round_trip = generation.unflatten(generation.flatten(config))
    assert round_trip[CUTE_AFFINE_SCAN_SCHEDULE_KEY] == "direct_m16n16_v1"

    configs = []
    for schedule in DIRECT_AFFINE_SCHEDULES:
        variant = helion.Config(cute_affine_scan_schedule=schedule)
        spec.normalize(variant)
        configs.append(variant)
    assert len(set(configs)) == len(DIRECT_AFFINE_SCHEDULES)
    assert len(set(map(canonical_config_id, configs))) == len(DIRECT_AFFINE_SCHEDULES)


@pytest.mark.parametrize("value", ["m16n8", "direct_m8n8_v1", 1])
def test_affine_scan_config_rejects_invalid_values(value: object) -> None:
    spec = _enabled_spec()
    config = helion.Config.from_dict({CUTE_AFFINE_SCAN_SCHEDULE_KEY: value})
    with pytest.raises(exc.InvalidConfig, match="must be one of"):
        spec.normalize(config)

    spec.normalize(config, _fix_invalid=True)
    assert config[CUTE_AFFINE_SCAN_SCHEDULE_KEY] == "ordinary"


def test_affine_scan_config_is_scoped_to_enabled_cute_scan() -> None:
    config = helion.Config(cute_affine_scan_schedule="direct_m16n8_v1")
    cute_spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 0),
    )
    with pytest.raises(exc.InvalidConfig, match="compatible affine scan"):
        cute_spec.normalize(config)

    triton_spec = ConfigSpec(backend=TritonBackend())
    with pytest.raises(exc.InvalidConfig, match="Unsupported config keys"):
        triton_spec.normalize(config)


def test_affine_scan_config_prunes_step_incompatible_schedule() -> None:
    spec = _enabled_spec(step_count=5)
    dimensions = {
        dimension.name: dimension for dimension in spec.iter_search_dimensions()
    }
    assert dimensions[CUTE_AFFINE_SCAN_SCHEDULE_KEY].values == [
        "ordinary",
        "direct_m16n16_v1",
    ]

    with pytest.raises(exc.InvalidConfig, match="must be one of"):
        spec.normalize(helion.Config(cute_affine_scan_schedule="direct_m16n8_v1"))


def test_affine_scan_config_keeps_both_direct_schedules_when_step_count_is_ambiguous() -> (
    None
):
    spec = _enabled_spec(step_count=None)
    dimensions = {
        dimension.name: dimension for dimension in spec.iter_search_dimensions()
    }
    assert dimensions[CUTE_AFFINE_SCAN_SCHEDULE_KEY].values == list(
        DIRECT_AFFINE_SCHEDULES
    )


@pytest.mark.parametrize(
    "old_key",
    [
        "cute_affine_scan_mma",
        "cute_affine_scan_coefficient_layout",
        "cute_affine_scan_state_ingress",
        "cute_affine_scan_phase_order",
    ],
)
def test_affine_scan_config_rejects_removed_keys(old_key: str) -> None:
    spec = _enabled_spec()
    with pytest.raises(exc.InvalidConfig, match="Invalid config keys"):
        spec.normalize(helion.Config.from_dict({old_key: "removed"}))
