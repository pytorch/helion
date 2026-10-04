from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _inputs
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_residency_search import _root_residency
from .test_cute_chained_scan_producer import _scan_sequence
from .test_cute_chained_scan_producer import _sequence_args
from .test_cute_chained_scan_producer import _sequence_config
import helion
from helion import exc
from helion._compiler.cute import chained_matmul as matmul
from helion._compiler.cute import chained_pipeline_storage as storage
from helion._compiler.cute import chained_preparation_pipeline as pipeline
from helion._compiler.cute.chained_preparation_cohorts import (
    supports_twenty_warp_preparation,
)
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import CUTE_CHAINED_COMPACT_PREPARATION_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import ConfigSpec


def _config(enabled=True, **overrides):
    return helion.Config.from_dict(
        {
            "num_warps": 16,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_warp_mma_rows": 32,
            "cute_chained_preparation_pipeline": True,
            "cute_chained_preparation_cohorts": 3,
            KEY: enabled,
            **overrides,
        }
    )


def _twenty(**overrides):
    return _config(
        num_warps=20,
        cute_chained_preparation_cohorts=4,
        cute_min_blocks_per_mp=1,
        **overrides,
    )


def test_public_default_choices_roundtrip_and_unchanged_seed_prefix(retention_bound):
    spec = retention_bound.config_spec
    assert KEY in VALID_KEYS and KEY in BACKEND_SPECIFIC_KEYS
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    assert not ConfigSpec._requests_cute_chained_loop({KEY: False})
    assert ConfigSpec._requests_cute_chained_loop({KEY: True})
    fields = spec._flat_fields()
    assert tuple(fields)[-11:] == (
        CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY,
        "cute_min_blocks_per_mp",
        KEY,
        "cute_chained_leaf_issue_batching",
        "cute_chained_broadcast_retention",
        "cute_chained_completed_member_store",
        "cute_native_matmul_metadata",
        "cute_chained_drain_tile_columns",
        "cute_chained_island_consumers",
        "cute_chained_async_vector_store",
        "cute_chained_fragment_epilogues",
    )
    assert fields["cute_min_blocks_per_mp"].search_values() == [0, 1]
    assert fields[KEY].search_values() == [False, True]
    assert fields["num_warps"].search_values() == [4, 8, 2, 1, 16, 32, 20]
    seeds = tuple(spec.compiler_seed_configs)
    before = copy.deepcopy([seed.config for seed in seeds])
    generation = spec.create_config_generation()
    pairs = generation.seed_flat_config_pairs()
    old_fields = {
        key: value
        for key, value in fields.items()
        if key
        not in (
            "cute_min_blocks_per_mp",
            KEY,
            "cute_chained_leaf_issue_batching",
            "cute_chained_broadcast_retention",
            "cute_chained_completed_member_store",
            "cute_native_matmul_metadata",
            "cute_chained_drain_tile_columns",
            "cute_chained_island_consumers",
        )
    }
    old_fields["num_warps"] = EnumFragment((4, 8, 2, 1, 16, 32))
    with patch.object(spec, "_flat_fields", return_value=old_fields):
        old = spec.create_config_generation()
        for (flat, config), (old_flat, old_config) in zip(
            pairs, old.seed_flat_config_pairs(), strict=True
        ):
            assert flat[:-10] + flat[-2:] == old_flat
            assert flat[-10:-2] == [0, False, False, False, False, False, 0, False]
            assert config == old_config
    assert before == [seed.config for seed in seeds]
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))
    for requested in (
        _config(False),
        _config(),
        _twenty(),
        _twenty(cute_chained_group_contractions=True),
    ):
        normalized = spec.normalized_config(requested)
        assert generation.unflatten(generation.flatten(normalized)) == normalized
        assert (
            generation.flatten(normalized)[_flat_scalar_index(spec, KEY)]
            is (requested[KEY])
        )
    default = helion.Config.from_dict({KEY: False})
    assert spec.normalized_config(default) == spec.normalized_config(helion.Config())
    assert KEY not in spec.normalized_config(default).config
    zero = helion.Config.from_dict({KEY: False, "cute_min_blocks_per_mp": 0})
    assert spec.normalized_config(zero) == spec.normalized_config(helion.Config())


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [None, 0, 1, -1, 0.0, 1.0, "false", [], {}])
def test_strict_boolean_before_repair(retention_bound, repair, value):
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "key,value",
    [
        ("cute_chained_mma_schedule", None),
        ("cute_chained_mma_schedule", "coalesced"),
        ("cute_chained_preparation_pipeline", None),
        ("cute_chained_preparation_pipeline", False),
        ("cute_chained_preparation_cohorts", None),
        ("cute_chained_preparation_cohorts", 1),
    ],
)
def test_never_inject_pipeline_or_cohorts(retention_bound, repair, key, value):
    config = _config()
    if value is None:
        config.config.pop(key)
    else:
        config.config[key] = value
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize(
    "key,value",
    [
        (KEY, False),
        ("cute_chained_mma_schedule", "k_major"),
        ("cute_chained_preparation_pipeline", False),
        ("cute_chained_preparation_cohorts", 1),
        ("cute_chained_preparation_cohorts", 2),
        ("cute_min_blocks_per_mp", None),
        ("cute_min_blocks_per_mp", True),
        ("cute_min_blocks_per_mp", 1.0),
        ("cute_min_blocks_per_mp", 2),
    ],
)
@pytest.mark.parametrize("repair", [False, True])
def test_twenty_warp_capability_is_strict_and_never_repaired(
    retention_bound, key, value, repair
):
    config = _twenty()
    if value is None:
        config.config.pop(key)
    else:
        config.config[key] = value
    assert not supports_twenty_warp_preparation(config.config)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize(
    "flag",
    [
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ],
)
def test_no_discovery_no_new_field_or_warp_choice(retention_bound, flag):
    spec = retention_bound.config_spec
    with patch.object(spec, flag, False):
        assert KEY not in spec._flat_fields()
        assert "cute_min_blocks_per_mp" not in spec._flat_fields()
        assert 20 not in spec._flat_fields()["num_warps"].search_values()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_twenty())


@pytest.mark.parametrize("backend", ["triton", "pallas"])
def test_other_backends_reject_positive_keep_default(retention_bound, backend):
    spec = retention_bound.config_spec
    with patch.object(spec, "backend_name", backend):
        default = {KEY: False}
        spec.normalize(default)
        assert KEY not in default
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config())


def test_root_cannot_acquire_compact_or_twenty_warp_path():
    with _cpu_codegen():
        bound = _root_residency._bind_isolated(
            (
                torch.empty((128, 16), dtype=torch.bfloat16),
                torch.empty((16, 32), dtype=torch.bfloat16),
            )
        )
        assert KEY not in bound.config_spec._flat_fields()
        assert "cute_min_blocks_per_mp" not in bound.config_spec._flat_fields()
        for requested in (_config(), _twenty(), _twenty(**{KEY: False})):
            with pytest.raises(exc.InvalidConfig):
                bound.config_spec.normalize(requested)


def test_absent_and_false_do_not_call_compact_builder():
    original = pipeline.emit_preparation_pipeline
    seen = []

    def observe(*args, **kwargs):
        assert "compact_preparation" not in kwargs
        seen.append(True)
        return original(*args, **kwargs)

    config = _config(False)
    with patch.object(pipeline, "emit_preparation_pipeline", observe):
        explicit = _source(_typed_sequence, _inputs(False, typed=True), config)
        config.config.pop(KEY)
        default = _source(_typed_sequence, _inputs(False, typed=True), config)
        config.config["cute_min_blocks_per_mp"] = 0
        zero = _source(_typed_sequence, _inputs(False, typed=True), config)
    assert seen == [True, True, True] and default == explicit == zero


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_public_twenty_warp_route_consumes_exact_final_table(dtype):
    config = _sequence_config()
    config.config.update(_twenty().config)
    original = storage.finalize_pipeline_storage
    results = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.preparation is kwargs["preparation"]
        results.append(result)
        return result

    with patch.object(storage, "finalize_pipeline_storage", observe):
        source = _source(_scan_sequence, _sequence_args(dtype), config)
    assert len(results) == 1
    result = results[0]
    bound = result.preparation
    assert bound is not None and bound._state.consumed
    assert bound.physical.accepted.execution.threads == 128
    assert dict(result.allocations)["frames"] == 4 * bound.stride
    assert result.charged_bytes <= 232448
    assert "block=(640, 1, 1)" in source
    assert "_helion_cute_min_blocks_per_mp = 1" in source
    assert "chain_cohort = chain_thread // 128" in source
    assert "chain_generation = chain_iteration // 4" in source
    assert "chain_recurrence_thread = chain_thread - 512" in source


def test_failed_late_table_cannot_install_body_or_fall_back():
    with (
        patch.object(storage, "finalize_pipeline_storage", return_value=None),
        patch.object(
            matmul, "_install_chained_body", side_effect=AssertionError("installed")
        ),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _source(_typed_sequence, _inputs(False, typed=True), _twenty())


def test_failed_graph_cannot_fall_through_to_generic_layouts():
    with (
        patch.object(matmul, "plan_chained_matmul", return_value=None),
        patch(
            "helion._compiler.cute.layout_propagation.plan_layouts",
            side_effect=AssertionError("generic layout"),
        ),
        pytest.raises(exc.BackendUnsupported),
    ):
        _source(_typed_sequence, _inputs(False, typed=True), _twenty())
