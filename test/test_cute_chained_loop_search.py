from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import CUTE_CHAINED_SEED_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loop_search(left, right, other, initial):
    steps, m, k = left.shape
    n = right.shape[-1]
    history = torch.empty((steps, m, n), device=left.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[128, None]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            a = left[step.id, rows, kk]
            first = hl.dot(a, right[step.id, kk, cols], out_dtype=torch.float32)
            second = hl.dot(a, other[step.id, kk, cols], out_dtype=torch.float32)
            state = state * 0.5 + first + second
            history[step.id, rows, cols] = state
        final[rows, cols] = state
    return history, final


@pytest.fixture(scope="module")
def loop_bound() -> Iterator[BoundKernel]:
    args = (
        torch.empty((3, 128, 32), dtype=torch.bfloat16),
        torch.empty((3, 32, 128), dtype=torch.bfloat16),
        torch.empty((3, 32, 128), dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.float32),
    )
    with _cpu_codegen():
        yield _loop_search._bind_isolated(args)


_ROOT_DEFAULTS: dict[str, object] = {
    "cute_chained_pointwise_read_cache": False,
    "cute_chained_pointwise_inplace_async": False,
    "cute_chained_auxiliary_cache": False,
    "cute_chained_tmem_free": "legacy",
    "cute_chained_tmem_early_release": False,
    "cute_chained_startup_transfer": "legacy",
    "cute_chained_initialized_accumulator": False,
    "cute_chained_late_rhs_reuse": False,
    "cute_chained_k_schedule": "full",
    "cute_chained_leaf_pipeline": "legacy",
    "cute_chained_direct_output": False,
    "cute_loop_vectorize": False,
    "cute_loop_load_schedule": "current",
}
_LOOP_SCHEDULES = (
    "coalesced",
    "cp_async",
    "coalesced_unrolled",
    "k_major",
    "k_major_padded",
    "tcgen05_tmem",
)


def _config(**overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [128],
            "num_warps": 4,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            **overrides,
        }
    )


def test_loop_flat_fields_and_seeds_have_only_effective_choices(
    loop_bound: BoundKernel,
) -> None:
    spec = loop_bound.config_spec
    assert spec.cute_chained_loop_search_enabled
    assert spec.cute_chained_group_search_enabled
    assert list(spec._flat_fields()) == [
        "block_sizes",
        "num_warps",
        "cute_chained_mma_schedule",
        "cute_chained_scratch_layout",
        "cute_chained_warp_mma_rows",
        "cute_chained_group_contractions",
        "cute_chained_pointwise_vectorize",
        "cute_chained_pointwise_unroll",
        CUTE_CHAINED_SEED_TILE_COLUMNS_KEY,
        CUTE_CHAINED_VECTOR_GROUP_KEY,
        "cute_native_matmul_metadata",
        "cute_chained_fragment_epilogues",
    ]
    seed_tiles = spec._flat_fields()[CUTE_CHAINED_SEED_TILE_COLUMNS_KEY]
    assert isinstance(seed_tiles, EnumFragment)
    assert seed_tiles.default() == 0
    assert seed_tiles.search_values() == [0, 32, 64]
    assert spec._cute_chained_mma_schedules() == _LOOP_SCHEDULES
    seeds = CuteChainedMatmulHeuristic._loop_seed_configs(loop_bound.env)
    assert 0 < len(seeds) <= 102
    assert len({repr(seed) for seed in seeds}) == len(seeds)
    generation = spec.create_config_generation()
    for seed in seeds:
        assert not seed.config.keys() & _ROOT_DEFAULTS.keys()
        assert seed["cute_chained_mma_schedule"] in _LOOP_SCHEDULES
        normalized = spec.normalized_config(seed)
        restored = generation.unflatten(generation.flatten(normalized))
        assert normalized == restored
        if restored.config.get("cute_chained_group_contractions"):
            assert restored["cute_chained_mma_schedule"] == "tcgen05_tmem"
            assert restored["num_warps"] in (4, 8, 16, 32)
    assert {
        seed["num_warps"]
        for seed in seeds
        if seed["cute_chained_mma_schedule"] == "tcgen05_tmem"
    } == {4, 8, 16, 32}
    assert all(
        seed.config.get("cute_chained_scratch_layout") == "xor"
        for seed in seeds
        if seed["cute_chained_mma_schedule"] == "tcgen05_tmem" and seed.num_warps > 4
    )
    assert any(
        seed["block_sizes"] == [128]
        and seed.config.get("cute_chained_group_contractions")
        for seed in seeds
    )


@pytest.mark.parametrize("schedule", _LOOP_SCHEDULES)
def test_loop_advertised_schedules_generate_common_lowering(
    loop_bound: BoundKernel, schedule: str
) -> None:
    source = loop_bound.to_code(_config(cute_chained_mma_schedule=schedule))
    assert "chain_loop_index" in source
    assert "chain_0_mma" in source
    assert "chain_1_mma" in source


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("key", list(_ROOT_DEFAULTS))
def test_loop_rejects_root_only_knobs_before_repair(
    loop_bound: BoundKernel, key: str, repair: bool
) -> None:
    default = _ROOT_DEFAULTS[key]
    value = {
        "cute_chained_tmem_free": "last_read",
        "cute_chained_startup_transfer": "tma",
        "cute_chained_k_schedule": "serial64",
        "cute_chained_leaf_pipeline": "paired_tma",
        "cute_loop_load_schedule": "group2",
    }.get(key, True if type(default) is bool else 2)
    config = _config(**{key: value})
    with pytest.raises(exc.InvalidConfig, match=f"{key}.*contraction loops"):
        loop_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize(
    "key,value",
    [
        ("cute_chained_auxiliary_cache", None),
    ],
)
def test_loop_inactive_defaults_keep_strict_types(
    loop_bound: BoundKernel, key: str, value: object
) -> None:
    with pytest.raises(exc.InvalidConfig, match=f"{key}.*contraction loops"):
        loop_bound.config_spec.normalize(_config(**{key: value}), _fix_invalid=True)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "schedule",
    ["cp_async_register", "cp_async_register_reuse", "cp_async_register_reuse_scan"],
)
def test_loop_rejects_inactive_register_bridge_schedules(
    loop_bound: BoundKernel, schedule: str, repair: bool
) -> None:
    with pytest.raises(
        exc.InvalidConfig, match="not implemented for contraction loops"
    ):
        loop_bound.config_spec.normalize(
            _config(cute_chained_mma_schedule=schedule), _fix_invalid=repair
        )


def test_loop_explicit_root_defaults_are_canonical_and_preserve_source(
    loop_bound: BoundKernel,
) -> None:
    spec = loop_bound.config_spec
    plain = spec.normalized_config(_config())
    defaults = spec.normalized_config(_config(**_ROOT_DEFAULTS))
    assert defaults == plain
    assert not plain.config.keys() & _ROOT_DEFAULTS.keys()
    assert loop_bound.to_code(defaults) == loop_bound.to_code(plain)


def test_loop_override_cannot_silently_lose_requested_mechanism(
    loop_bound: BoundKernel,
) -> None:
    spec = loop_bound.config_spec
    generation = spec.create_config_generation(
        overrides={"cute_chained_pointwise_vectorize": True}
    )
    flat = generation.flatten(spec.normalized_config(_config()))
    config = generation.unflatten(flat)
    assert config["cute_chained_pointwise_vectorize"] is True
    assert "_last_pointer =" in loop_bound.to_code(config)


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_loop_vector_staging_requires_strict_boolean(
    loop_bound: BoundKernel, value: object
) -> None:
    with pytest.raises(exc.InvalidConfig, match="pointwise_vectorize must be bool"):
        loop_bound.config_spec.normalize(
            _config(cute_chained_pointwise_vectorize=value), _fix_invalid=True
        )


def test_loop_vector_staging_rejects_inactive_warp_schedule(
    loop_bound: BoundKernel,
) -> None:
    with pytest.raises(exc.InvalidConfig, match="loop vector staging requires"):
        loop_bound.config_spec.normalize(
            _config(
                cute_chained_mma_schedule="coalesced",
                cute_chained_pointwise_vectorize=True,
            ),
            _fix_invalid=True,
        )


def test_non_loop_search_fields_and_seed_order_are_unchanged() -> None:
    from .test_cute_chained_caches import _aux_cache_args
    from .test_cute_chained_caches import _auxiliary_chain

    with _cpu_codegen():
        bound = _auxiliary_chain._bind_isolated(_aux_cache_args("cpu", "plain"))
    spec = bound.config_spec
    assert not spec.cute_chained_loop_search_enabled
    assert list(spec._flat_fields()) == [
        "block_sizes",
        "num_warps",
        "cute_chained_mma_schedule",
        "cute_chained_scratch_layout",
        "cute_chained_pointwise_vectorize",
        "cute_chained_auxiliary_cache",
        "cute_chained_tmem_free",
        "cute_chained_tmem_early_release",
        "cute_chained_startup_transfer",
        CUTE_CHAINED_SEED_TILE_COLUMNS_KEY,
        CUTE_CHAINED_VECTOR_GROUP_KEY,
        "cute_chained_snapshot_tile_columns",
        "cute_native_matmul_metadata",
        "cute_chained_drain_tile_columns",
        "cute_chained_fragment_epilogues",
    ]
    full = [config.config for config in spec.compiler_seed_configs]
    snapshot_key = "cute_chained_snapshot_tile_columns"
    siblings = [
        (index, seed) for index, seed in enumerate(full) if seed.get(snapshot_key)
    ]
    assert len(siblings) == 2
    for index, sibling in siblings:
        assert sibling == full[index - 1] | {snapshot_key: 32}
    seeds = [seed for seed in full if not seed.get(snapshot_key)]
    # Captured before splitting loop search: includes every serialized seed,
    # their order and duplicate multiplicity, not only a set of configurations.
    assert len(seeds) == 108
    assert seeds[-1]["cute_chained_scratch_layout"] == "xor"
    assert seeds[-1]["cute_chained_mma_schedule"] == "coalesced"
    assert hashlib.sha256(
        json.dumps(seeds[:107], sort_keys=True).encode()
    ).hexdigest() == (
        "9ee84d0e4cf2cb81098e9d8fd9b5438d29ec9f88127af6220d186270edf17b03"
    )
