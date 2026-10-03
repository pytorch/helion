from __future__ import annotations

from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _inputs
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_preparation_search import _config
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import CUTE_CHAINED_ASYNC_VECTOR_STORE_KEY as KEY


@pytest.fixture(scope="module")
def bound():
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        result = _typed_sequence._bind_isolated(args)
        with (
            result.env,
            result.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)),
        ):
            yield result


def _selected(**overrides):
    return _config(**{"cute_chained_pointwise_vectorize": True, KEY: True, **overrides})


def test_public_field_roundtrip_default_and_exact_existing_prefix(bound):
    spec = bound.config_spec
    fields = spec._flat_fields()
    assert tuple(fields)[-2:] == (KEY, "cute_chained_fragment_epilogues")
    field = fields[KEY]
    assert isinstance(field, EnumFragment)
    assert field.search_values() == [False, True] and field.default() is False
    # The old field provider is unchanged; the outer wrapper only appends its
    # preexisting native/drain/island suffix followed by the new coordinate.
    original = spec._flat_fields_without_native_metadata()
    assert tuple(fields) == (
        *original,
        "cute_native_matmul_metadata",
        "cute_chained_drain_tile_columns",
        "cute_chained_island_consumers",
        KEY,
        "cute_chained_fragment_epilogues",
    )
    default = spec.default_config()
    assert KEY not in default.config
    assert spec.normalized_config(default.config | {KEY: False}) == default
    generation = spec.create_config_generation()
    for enabled in (False, True):
        config = spec.normalized_config(_selected(**{KEY: enabled}))
        assert config.config.get(KEY, False) is enabled
        assert generation.unflatten(generation.flatten(config)) == config


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [None, 0, 1, "true", 0.0, [], {}])
def test_public_bool_is_strict_before_repair(bound, value, repair):
    with pytest.raises(exc.InvalidConfig, match=f"{KEY} must be bool"):
        bound.config_spec.normalize(_selected(**{KEY: value}), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "override",
    [
        {"cute_chained_preparation_pipeline": False},
        {"cute_chained_pointwise_vectorize": False},
        {"cute_chained_mma_schedule": "coalesced"},
    ],
)
def test_public_choice_cannot_enable_prerequisites(bound, override, repair):
    with pytest.raises(exc.InvalidConfig):
        bound.config_spec.normalize(_selected(**override), _fix_invalid=repair)


@pytest.mark.parametrize(
    "missing",
    [
        "cute_chained_mma_schedule",
        "cute_chained_preparation_pipeline",
        "cute_chained_pointwise_vectorize",
    ],
)
def test_actual_field_membership_is_required(bound, missing):
    spec = bound.config_spec
    original = spec._flat_fields_without_native_metadata()
    assert missing in original
    original.pop(missing)
    with patch.object(
        spec, "_flat_fields_without_native_metadata", return_value=original
    ):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig, match="explicit vectorized common"):
            spec.normalize(_selected())


@pytest.mark.parametrize(
    "capability",
    [
        "cute_chained_loop_search_enabled",
        "cute_chained_tcgen05_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ],
)
def test_capabilities_do_not_expose_an_inactive_coordinate(bound, capability):
    spec = bound.config_spec
    with patch.object(spec, capability, False):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_selected())


@pytest.mark.parametrize("target", [None, (7, 5)])
def test_target_without_cp_async_cannot_expose_the_choice(bound, target):
    spec = bound.config_spec
    with patch.object(spec, "target_device_capability", target):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig, match="explicit vectorized common"):
            spec.normalize(_selected())


def test_generic_structural_seed_suffix_preserves_old_parents(bound):
    env = bound.env
    device_ir = bound.host_function.device_ir
    heuristic = CuteChainedMatmulHeuristic
    with patch.object(
        heuristic, "_with_async_vector_store_seeds", side_effect=lambda e, d, s: s
    ):
        old = heuristic.get_seed_configs(env, device_ir)
    assert old
    new = heuristic.get_seed_configs(env, device_ir)
    assert new is not None and new[: len(old)] == old
    additions = new[len(old) :]
    assert 1 <= len(additions) <= 2
    for sibling in additions:
        assert sibling[KEY] is True
        values = dict(sibling.config)
        values.pop(KEY)
        assert helion.Config.from_dict(values) in old
        assert bound.config_spec.normalized_config(sibling)[KEY] is True
    with patch.object(env.config_spec, "cute_chained_tcgen05_search_enabled", False):
        assert heuristic._with_async_vector_store_seeds(env, device_ir, old) is old
