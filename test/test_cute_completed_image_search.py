from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from test._cute_aux import _cpu_codegen
from test.test_cute_chained_coordinates import _split_view_sequence
from test.test_cute_chained_coordinates import _view_args

from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig

KEY = "cute_chained_completed_image"


@pytest.fixture
def spec():
    with _cpu_codegen():
        yield _split_view_sequence._bind_isolated(
            _view_args(torch.bfloat16)
        ).config_spec


def test_identity_roundtrip_and_polish(spec):
    gen = ConfigGeneration(spec)
    default = spec.default_config()
    flat = gen.flatten(default)
    indices, is_sequence = gen._key_to_flat_indices[KEY]
    assert len(indices) == 1 and not is_sequence
    index = indices[0]
    requested = list(flat)
    requested[index] = True
    selected = gen.unflatten(requested)
    restored = gen.flatten(selected)
    assert restored[:index] == flat[:index]
    assert restored[index + 1 :] == flat[index + 1 :]
    assert selected.config[KEY] is True
    assert restored[index] is True
    assert spec.normalized_config(selected.config) == selected
    assert spec.normalized_config(default.config | {KEY: False}) == default
    assert set(gen.flat_spec[index].search_values(10)) == {False, True}
    assert gen.canonicalize_flat(gen.flatten(selected))[1] == selected


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        {KEY: 1},
        {KEY: None},
        {"cute_chained_warp_register_a": True},
        {"cute_chained_mma_schedule": "tcgen05"},
        {"cute_chained_preparation_pipeline": True},
        {"cute_chained_warp_result_transport": "packed"},
        {"pid_type": "persistent"},
        {"cute_cluster_n": 2},
    ],
)
def test_strict_conflicts_cannot_be_repaired_away(spec, repair, change):
    with pytest.raises(InvalidConfig):
        spec.normalize(
            (copy.deepcopy(spec.default_config().config) | {KEY: True} | change),
            _fix_invalid=repair,
        )


@pytest.mark.parametrize(
    "backend,chain,loop",
    [("triton", True, False), ("cute", False, False), ("cute", True, True)],
)
def test_no_field_for_other_root_families(spec, backend, chain, loop):
    with (
        patch.object(spec, "backend_name", backend),
        patch.object(spec, "cute_chained_matmul_search_enabled", chain),
        patch.object(spec, "cute_chained_loop_search_enabled", loop),
    ):
        assert KEY not in spec._flat_fields()


def test_false_override_does_not_add_true_seed(spec):
    gen = ConfigGeneration(spec, overrides={KEY: False})
    assert all(
        (not cfg.config.get(KEY, False) for _, cfg in gen.seed_flat_config_pairs())
    )
