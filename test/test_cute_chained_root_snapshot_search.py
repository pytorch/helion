from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_root_pair import _three
from .test_cute_chained_tcgen05 import _tcgen_chain
from .test_cute_chained_tcgen05 import _tcgen_inputs
import helion
from helion import exc
from helion._compiler.cute import chained_tcgen05 as root
from helion.autotuner.config_spec import CUTE_CHAINED_MMA_SCHEDULE_KEY as SCHEDULE
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY as KEY


@pytest.fixture(scope="module")
def root_bound():
    with _cpu_codegen():
        yield _tcgen_chain._bind_isolated((*_tcgen_inputs("cpu"), "scan"))


def _config(columns=0, **overrides):
    return helion.Config.from_dict(
        {
            "block_sizes": [128, 64],
            "num_warps": 4,
            SCHEDULE: "tcgen05_tmem",
            KEY: columns,
            **overrides,
        }
    )


def test_root_snapshot_appends_coordinate_and_preserves_all_seed_values(root_bound):
    spec = root_bound.config_spec
    assert spec.cute_chained_matmul_search_enabled
    assert not spec.cute_chained_loop_search_enabled
    fields = spec._flat_fields()
    assert tuple(fields)[-4:] == (
        KEY,
        "cute_native_matmul_metadata",
        "cute_chained_drain_tile_columns",
        "cute_chained_fragment_epilogues",
    )
    assert fields[KEY].search_values() == [0, 32]
    assert spec.flatten_missing_field_default(KEY, {}) == (True, 0)
    seeds = tuple(spec.compiler_seed_configs)
    before = copy.deepcopy([seed.config for seed in seeds])
    generation = spec.create_config_generation()
    pairs = generation.seed_flat_config_pairs()
    index = _flat_scalar_index(spec, KEY)
    original_seeds = [seed for seed in seeds if not seed.config.get(KEY)]
    original_pairs = [
        (flat, config) for flat, config in pairs if not config.config.get(KEY)
    ]
    with (
        patch.object(
            spec,
            "_flat_fields",
            return_value={k: v for k, v in fields.items() if k != KEY},
        ),
        patch.object(spec, "compiler_seed_configs", original_seeds),
    ):
        old = spec.create_config_generation()
        for (flat, config), (previous_flat, previous) in zip(
            original_pairs, old.seed_flat_config_pairs(), strict=True
        ):
            assert flat[:index] + flat[index + 1 :] == previous_flat
            assert flat[index] == 0 and flat[-3:] == [False, 0, False]
            assert config == previous
    assert [seed.config for seed in seeds] == before
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))
    for columns in (0, 32):
        config = spec.normalized_config(_config(columns))
        flat = generation.flatten(config)
        assert flat[_flat_scalar_index(spec, KEY)] == columns
        assert generation.unflatten(flat) == config
        assert (KEY in config.config) is bool(columns)
    assert spec.normalized_config(
        helion.Config.from_dict({"block_sizes": [128, 64], KEY: 0})
    ) == (spec.normalized_config(helion.Config(block_sizes=[128, 64])))


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [None, False, True, 1, 16, 64, 0.0, 32.0, "32"])
def test_root_snapshot_strict_integer_before_repair(root_bound, repair, value):
    with pytest.raises(exc.InvalidConfig, match=KEY):
        root_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("schedule", [None, "coalesced", "k_major"])
def test_root_snapshot_does_not_inject_tcgen(root_bound, repair, schedule):
    config = _config(32, **{SCHEDULE: schedule})
    if schedule is None:
        config.config.pop(SCHEDULE)
    with pytest.raises(exc.InvalidConfig):
        root_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize(
    "flag",
    ["cute_chained_matmul_search_enabled", "cute_chained_tcgen05_search_enabled"],
)
def test_root_snapshot_requires_discovery(root_bound, flag):
    with (
        patch.object(root_bound.config_spec, flag, False),
        pytest.raises(exc.InvalidConfig),
    ):
        root_bound.config_spec.normalize(_config(32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["plain", "scan"])
def test_public_root_snapshot_routes_exactly_private_source(dtype, mode):
    arguments = (*_tcgen_inputs("cpu", dtype), mode)
    original = root.codegen_shared_root_sequence

    def private(cg, plan):
        return original(cg, plan, snapshot_tile_columns=32)

    with _cpu_codegen():
        with patch.object(root, "codegen_shared_root_sequence", private):
            expected = _tcgen_chain._bind_isolated(arguments).to_code(_config())
        actual = _tcgen_chain._bind_isolated(arguments).to_code(_config(32))
    assert actual == expected
    assert "for chain_1_bridge_panel in cutlass.range(4, unroll=1)" in actual


def test_public_root_snapshot_unsupported_positive_never_falls_back():
    arguments = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((128, 128), (128, 128), (128, 64))
    )
    with (
        _cpu_codegen(),
        patch.object(
            root, "codegen_chained_tcgen05", side_effect=AssertionError("fallback")
        ),
        pytest.raises(exc.BackendUnsupported, match="snapshot.*root pair"),
    ):
        _three._bind_isolated(arguments).to_code(_config(32))
