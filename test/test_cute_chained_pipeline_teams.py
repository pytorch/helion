from __future__ import annotations

from collections import Counter
import math
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_preparation_search import _config as _pipeline_config
from .test_cute_chained_preparation_search import pipeline_bound as pipeline_bound
from .test_cute_chained_residency_search import _root_residency
from .test_cute_chunk_recurrence import _bt16_fp32_chain
from .test_cute_chunk_recurrence import _fake_inputs
from .test_cute_chunk_recurrence import _real_dispatch_cpu
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteChunkRecurrenceHeuristic
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import CUTE_CHAINED_COLLECTIVE_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_LEAF_COUNT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_LEAF_PIPELINE_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_NATIVE_VECTOR_READS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OPERAND_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PIPELINE_CONSUMER_WARPS_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_COHORTS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_PIPELINE_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_UNROLL_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SEED_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion.runtime.kernel import BoundKernel


def _config(consumer_warps: object = 8, **overrides: object) -> helion.Config:
    return _pipeline_config(**{"num_warps": 16, KEY: consumer_warps, **overrides})


def test_team_field_default_and_flat_coordinate_prefix(
    pipeline_bound: BoundKernel,
) -> None:
    spec = pipeline_bound.config_spec
    fields = spec._flat_fields()
    assert tuple(fields)[-27:] == (
        CUTE_CHAINED_PREPARATION_PIPELINE_KEY,
        KEY,
        CUTE_CHAINED_LEAF_PIPELINE_KEY,
        CUTE_CHAINED_SEED_TILE_COLUMNS_KEY,
        CUTE_CHAINED_VECTOR_GROUP_KEY,
        CUTE_CHAINED_PREPARATION_COHORTS_KEY,
        CUTE_CHAINED_PREPARATION_UNROLL_KEY,
        CUTE_CHAINED_REGISTER_ISLANDS_KEY,
        CUTE_CHAINED_LEAF_COUNT_KEY,
        CUTE_CHAINED_COLLECTIVE_RETENTION_KEY,
        CUTE_CHAINED_OPERAND_RETENTION_KEY,
        CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY,
        CUTE_CHAINED_NATIVE_VECTOR_READS_KEY,
        CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY,
        CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY,
        CUTE_CHAINED_FRONTIER_STMATRIX_KEY,
        "cute_chained_snapshot_tile_columns",
        "cute_min_blocks_per_mp",
        "cute_chained_compact_preparation",
        "cute_chained_leaf_issue_batching",
        "cute_chained_broadcast_retention",
        "cute_chained_completed_member_store",
        "cute_native_matmul_metadata",
        "cute_chained_drain_tile_columns",
        "cute_chained_island_consumers",
        "cute_chained_async_vector_store",
        "cute_chained_fragment_epilogues",
    )
    field = fields[KEY]
    assert isinstance(field, EnumFragment)
    assert field.search_values() == [4, 8, 16]
    assert field.default() == 4
    default = spec.default_config()
    assert KEY not in default.config
    explicit = helion.Config.from_dict(default.config | {KEY: 4})
    assert spec.normalized_config(explicit) == default
    generation = spec.create_config_generation()
    assert generation.unflatten(generation.flatten(default)) == default
    index = _flat_scalar_index(spec, KEY)
    with patch.object(
        spec, "_flat_fields", return_value={k: v for k, v in fields.items() if k != KEY}
    ):
        old = spec.create_config_generation()
        flat = generation.flatten(default)
        assert flat[:index] + flat[index + 1 :] == old.flatten(default)
        assert spec.default_config() == default


@pytest.mark.parametrize(
    "consumer,total", [(4, 8), (4, 16), (8, 16), (8, 32), (16, 32)]
)
def test_legal_team_roundtrips_and_override(
    pipeline_bound: BoundKernel, consumer: int, total: int
) -> None:
    spec = pipeline_bound.config_spec
    config = spec.normalized_config(_config(consumer, num_warps=total))
    assert config.config.get(KEY, 4) == consumer
    generation = spec.create_config_generation()
    assert generation.unflatten(generation.flatten(config)) == config
    base = spec.normalized_config(_config(4, num_warps=total))
    override = spec.create_config_generation(overrides={KEY: consumer})
    restored = override.unflatten(override.flatten(base))
    assert restored.config.get(KEY, 4) == consumer
    assert restored.num_warps == total


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [True, False, None, 0, 1, 12, 32, 4.0, "8", [], {}])
def test_consumer_warps_are_strict_integers_before_repair(
    pipeline_bound: BoundKernel, value: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match=f"{KEY} must be 4, 8 or 16"):
        pipeline_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "consumer,overrides",
    [
        (8, {CUTE_CHAINED_PREPARATION_PIPELINE_KEY: False}),
        (8, {CUTE_CHAINED_PREPARATION_PIPELINE_KEY: 1}),
        (8, {"cute_chained_mma_schedule": "coalesced"}),
        (8, {"num_warps": 8}),
        (16, {"num_warps": 16}),
        (8, {"num_warps": 16.0}),
    ],
)
def test_incomplete_requests_and_insufficient_prep_team_fail_closed(
    pipeline_bound: BoundKernel,
    consumer: int,
    overrides: dict[str, object],
    repair: bool,
) -> None:
    with pytest.raises(exc.InvalidConfig, match=r"consumer_warps \+ 4"):
        pipeline_bound.config_spec.normalize(
            _config(consumer, **overrides), _fix_invalid=repair
        )


@pytest.mark.parametrize(
    "missing", [CUTE_CHAINED_PREPARATION_PIPELINE_KEY, "cute_chained_mma_schedule"]
)
def test_nondefault_team_requires_explicit_family(
    pipeline_bound: BoundKernel, missing: str
) -> None:
    config = _config()
    config.config.pop(missing)
    with pytest.raises(exc.InvalidConfig, match="explicit common TCgen05"):
        pipeline_bound.config_spec.normalize(config, _fix_invalid=True)


def test_team_requires_discovered_cut_and_correct_backend(
    pipeline_bound: BoundKernel,
) -> None:
    spec = pipeline_bound.config_spec
    for attribute, value in (
        ("cute_chained_preparation_pipeline_search_enabled", False),
        ("backend_name", "triton"),
    ):
        with patch.object(spec, attribute, value):
            with pytest.raises(exc.InvalidConfig, match="explicit common TCgen05"):
                spec.normalize(_config(), _fix_invalid=True)
            inactive: dict[str, object] = {KEY: 4}
            spec.normalize(inactive)
            assert KEY not in inactive


def test_team_seed_extension_preserves_complete_prior_pool(
    pipeline_bound: BoundKernel,
) -> None:
    assert pipeline_bound.host_function is not None
    heuristic = CuteChainedMatmulHeuristic
    team_seeds = heuristic._with_pipeline_team_seeds
    stages: list[tuple[list[helion.Config], list[helion.Config]]] = []

    def observe(
        env: CompileEnvironment, seeds: list[helion.Config]
    ) -> list[helion.Config]:
        parents = list(seeds)
        result = team_seeds(env, seeds)
        stages.append((parents, result))
        return result

    with pipeline_bound.env, pipeline_bound.host_function:
        ir = pipeline_bound.host_function.device_ir
        with patch.object(
            heuristic, "_with_pipeline_team_seeds", side_effect=lambda env, seeds: seeds
        ):
            before = heuristic.get_seed_configs(pipeline_bound.env, ir)
        with patch.object(heuristic, "_with_pipeline_team_seeds", side_effect=observe):
            after = heuristic.get_seed_configs(pipeline_bound.env, ir)
    assert before is not None and after is not None
    ((parents, stage_seeds),) = stages
    assert stage_seeds[: len(parents)] == parents
    assert after[: len(stage_seeds)] == stage_seeds
    # Later seed passes append their siblings after the new team seeds.
    assert Counter(before) <= Counter(after)
    suffix = stage_seeds[len(parents) :]
    assert 0 < len(suffix) <= 2
    assert len(set(suffix)) == len(suffix)
    transferred = {
        config
        for _, config in pipeline_bound.config_spec.create_config_generation().seed_flat_config_pairs()
    }
    assert all(
        pipeline_bound.config_spec.normalized_config(seed) in transferred
        for seed in before
    )
    for seed in suffix:
        eligible_parents = [
            parent
            for parent in parents
            if parent.config.get(CUTE_CHAINED_PREPARATION_PIPELINE_KEY) is True
            and parent.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
            and parent.num_warps == seed.num_warps
        ]
        parent = max(
            reversed(eligible_parents), key=lambda item: math.prod(item.block_sizes)
        )
        assert seed.config == parent.config | {KEY: 8}
        assert seed.num_warps in (16, 32)
        assert pipeline_bound.config_spec.normalized_config(seed) in transferred
    assert KEY not in before[0].config


def test_team_suffix_bound_with_both_eligible_total_widths(
    pipeline_bound: BoundKernel,
) -> None:
    parents = [
        pipeline_bound.config_spec.normalized_config(
            _config(4, num_warps=warps, cute_chained_group_contractions=grouped)
        )
        for warps in (8, 16, 32)
        for grouped in (False, True)
    ]
    result = CuteChainedMatmulHeuristic._with_pipeline_team_seeds(
        pipeline_bound.env, parents
    )
    assert all(a is b for a, b in zip(result, parents, strict=False))
    assert len(result) == len(parents) + 2
    assert [seed.num_warps for seed in result[len(parents) :]] == [16, 32]
    assert all(seed[KEY] == 8 for seed in result[len(parents) :])


@pytest.mark.parametrize("consumer,total", [(4, 8), (8, 16), (16, 32)])
def test_public_config_dispatches_exact_role_widths_cpu(
    pipeline_bound: BoundKernel, consumer: int, total: int
) -> None:
    source = pipeline_bound.to_code(_config(consumer, num_warps=total))
    prep_threads, consumer_threads = (total - consumer) * 32, consumer * 32
    assert f"if chain_thread < {prep_threads}:" in source
    assert f"chain_recurrence_thread = chain_thread - {prep_threads}" in source
    assert f"NamedBarrier(barrier_id=2, num_threads={prep_threads})" in source
    assert f"NamedBarrier(barrier_id=3, num_threads={consumer_threads})" in source
    if consumer == 4:
        absent = _config(4, num_warps=total)
        absent.config.pop(KEY)
        assert source == pipeline_bound.to_code(absent)


def test_default_team_does_not_activate_root_pipeline() -> None:
    args = (
        torch.empty((128, 16), dtype=torch.bfloat16),
        torch.empty((16, 32), dtype=torch.bfloat16),
    )
    with _cpu_codegen():
        bound = _root_residency._bind_isolated(args)
        assert KEY not in bound.config_spec._flat_fields()
        config = helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
        assert bound.to_code(config) == bound.to_code(
            helion.Config.from_dict(config.config | {KEY: 4})
        )
        with pytest.raises(exc.InvalidConfig, match="explicit common TCgen05"):
            bound.config_spec.normalize(_config(), _fix_invalid=True)


@pytest.mark.parametrize("sequences", [1, 2])
def test_default_team_preserves_legacy_default_and_seed_family(sequences: int) -> None:
    with (
        _real_dispatch_cpu(),
        patch.object(CuteChunkRecurrenceHeuristic, "should_promote", return_value=True),
    ):
        args = (*_fake_inputs(fp32_state=True, sequences=sequences), 128**-0.5)
        bound = _bt16_fp32_chain._bind_isolated(args)
        spec = bound.config_spec
        config = spec.default_config()
        assert "cute_chained_mma_schedule" not in config.config
        explicit = helion.Config.from_dict(config.config | {KEY: 4})
        assert spec.normalized_config(explicit) == config
        assert bound.to_code(config) == bound.to_code(explicit)
        generation = spec.create_config_generation()
        legacy = [
            config
            for _, config in generation.seed_flat_config_pairs()
            if "cute_chained_mma_schedule" not in config.config
        ]
        assert len(legacy) == 6
        for seed in legacy:
            assert KEY not in seed.config
            assert generation.unflatten(generation.flatten(seed)) == seed
