from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_mma_selection_search import _selection_loop
from .test_cute_chained_prefill_search import kda_prefill_native_math
from .test_cute_chained_prefill_search import kda_prefill_native_math_bt32
from .test_cute_chained_preparation_cut import _inputs
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_residency_search import _root_residency
from .test_cute_chunk_prefill import _inputs as _prefill_inputs
from .test_cute_chunk_recurrence import _bt16_fp32_chain
from .test_cute_chunk_recurrence import _fake_inputs
from .test_cute_chunk_recurrence import _real_dispatch_cpu
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteChunkRecurrenceHeuristic
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import default_cute_mma_support
from helion._testing import patch_cute_mma_support
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import CUTE_CHAINED_COLLECTIVE_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_LEAF_COUNT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_LEAF_PIPELINE_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_NATIVE_VECTOR_READS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OPERAND_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PIPELINE_CONSUMER_WARPS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_COHORTS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_PIPELINE_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_UNROLL_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SEED_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module")
def pipeline_bound() -> Iterator[BoundKernel]:
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)):
            yield bound


def _config(value: object = True, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "num_warps": 8,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_warp_mma_rows": 32,
            KEY: value,
            **overrides,
        }
    )


def test_field_roundtrip_explicit_override_and_default(
    pipeline_bound: BoundKernel,
) -> None:
    spec = pipeline_bound.config_spec
    assert spec.cute_chained_preparation_pipeline_search_enabled
    field = spec._flat_fields()[KEY]
    assert isinstance(field, EnumFragment)
    assert field.search_values() == [False, True]
    generation = spec.create_config_generation()
    for value in (False, True):
        config = spec.normalized_config(_config(value))
        assert config.config.get(KEY, False) is value
        assert generation.unflatten(generation.flatten(config)) == config
    inactive = spec.normalized_config(_config(False))
    assert KEY not in inactive.config
    override = spec.create_config_generation(overrides={KEY: True})
    assert override.unflatten(override.flatten(inactive))[KEY] is True
    default = spec.default_config()
    assert KEY not in default.config
    fields = spec._flat_fields()
    with patch.object(spec, "cute_chained_preparation_pipeline_search_enabled", False):
        assert spec.default_config() == default
        assert KEY not in spec._flat_fields()
        old_fields = tuple(spec._flat_fields())
        assert old_fields[-4:] == (
            CUTE_CHAINED_SEED_TILE_COLUMNS_KEY,
            CUTE_CHAINED_VECTOR_GROUP_KEY,
            "cute_native_matmul_metadata",
            "cute_chained_fragment_epilogues",
        )
        assert tuple(fields) == (
            *old_fields[:-4],
            KEY,
            CUTE_CHAINED_PIPELINE_CONSUMER_WARPS_KEY,
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
            CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY,
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


def test_rectangular_leaf_mode_roundtrip_keeps_legacy_default(pipeline_bound):
    spec = pipeline_bound.config_spec
    key = CUTE_CHAINED_LEAF_PIPELINE_KEY
    assert spec._flat_fields()[key].search_values() == ["legacy", "rectangular_tma"]
    generation = spec.create_config_generation()
    for mode in ("legacy", "rectangular_tma"):
        config = spec.normalized_config(_config(**{key: mode}))
        assert config.config.get(key, "legacy") == mode
        assert generation.unflatten(generation.flatten(config)) == config
    assert key not in spec.default_config().config


@pytest.mark.parametrize("repair", [False, True])
def test_rectangular_leaf_mode_requires_preparation_pipeline(pipeline_bound, repair):
    with pytest.raises(exc.InvalidConfig, match="requires a TCgen05 preparation"):
        pipeline_bound.config_spec.normalize(
            _config(False, **{CUTE_CHAINED_LEAF_PIPELINE_KEY: "rectangular_tma"}),
            _fix_invalid=repair,
        )


@pytest.mark.parametrize("mode", ["paired_tma", "unknown", 1, True])
def test_root_only_or_unknown_leaf_modes_remain_rejected_in_loops(pipeline_bound, mode):
    with pytest.raises(
        exc.InvalidConfig, match="not implemented for contraction loops"
    ):
        pipeline_bound.config_spec.normalize(
            _config(**{CUTE_CHAINED_LEAF_PIPELINE_KEY: mode})
        )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [0, 1, None, "True", 0.0, [], {}])
def test_bool_validation_precedes_autotune_repair(
    pipeline_bound: BoundKernel, value: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match=f"{KEY} must be bool"):
        pipeline_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"cute_chained_mma_schedule": "coalesced"},
        {"cute_chained_mma_schedule": None},
        {"cute_chained_warp_mma_rows": 0},
        {"cute_chained_warp_mma_rows": True},
        {"num_warps": 4},
        {"num_warps": 8.0},
        {"cute_affine_scan_schedule": "warp"},
        {"cute_chained_direct_output": True},
        {"cute_chunk_recurrence_register_cap": 72},
        {"cute_chunk_recurrence_pipeline": "serial"},
    ],
)
def test_pipeline_rejects_conflicting_or_incomplete_requests(
    pipeline_bound: BoundKernel, overrides: dict[str, object], repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="eligible common preparation loop"):
        pipeline_bound.config_spec.normalize(_config(**overrides), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
def test_explicit_schedule_and_discovery_are_required(
    pipeline_bound: BoundKernel, repair: bool
) -> None:
    config = _config()
    config.config.pop("cute_chained_mma_schedule")
    with pytest.raises(exc.InvalidConfig, match="eligible common preparation loop"):
        pipeline_bound.config_spec.normalize(config, _fix_invalid=repair)
    with (
        patch.object(
            pipeline_bound.config_spec,
            "cute_chained_preparation_pipeline_search_enabled",
            False,
        ),
        pytest.raises(exc.InvalidConfig, match="eligible common preparation loop"),
    ):
        pipeline_bound.config_spec.normalize(_config(), _fix_invalid=repair)


@pytest.mark.parametrize("value", [False, True])
def test_wrong_backend_accepts_only_inactive_option(
    pipeline_bound: BoundKernel, value: bool
) -> None:
    spec = pipeline_bound.config_spec
    config: dict[str, object] = {KEY: value}
    with patch.object(spec, "backend_name", "triton"):
        if value:
            with pytest.raises(exc.InvalidConfig, match="common preparation loop"):
                spec.normalize(config, _fix_invalid=True)
        else:
            spec.normalize(config)
            assert KEY not in config


def test_complete_existing_seed_prefix_is_preserved(
    pipeline_bound: BoundKernel,
) -> None:
    assert pipeline_bound.host_function is not None
    spec = pipeline_bound.config_spec
    preparation_seeds = CuteChainedMatmulHeuristic._with_preparation_pipeline_seeds
    stages: list[tuple[list[helion.Config], list[helion.Config]]] = []

    def observe(
        env: CompileEnvironment, seeds: list[helion.Config]
    ) -> list[helion.Config]:
        parents = list(seeds)
        result = preparation_seeds(env, seeds)
        stages.append((parents, result))
        return result

    with (
        pipeline_bound.env,
        pipeline_bound.host_function,
        patch.object(
            CuteChainedMatmulHeuristic,
            "_with_pipeline_team_seeds",
            side_effect=lambda env, seeds: seeds,
        ),
    ):
        device_ir = pipeline_bound.host_function.device_ir
        with patch.object(
            spec, "cute_chained_preparation_pipeline_search_enabled", False
        ):
            old = CuteChainedMatmulHeuristic.get_seed_configs(
                pipeline_bound.env, device_ir
            )
        with patch.object(
            CuteChainedMatmulHeuristic,
            "_with_preparation_pipeline_seeds",
            side_effect=observe,
        ):
            seeds = CuteChainedMatmulHeuristic.get_seed_configs(
                pipeline_bound.env, device_ir
            )
    assert old is not None and seeds is not None
    assert seeds[: len(old)] == old
    ((parents, stage_seeds),) = stages
    assert stage_seeds[: len(parents)] == parents
    assert seeds[: len(stage_seeds)] == stage_seeds
    suffix = stage_seeds[len(parents) :]
    assert 0 < len(suffix) <= 4
    assert len(set(suffix)) == len(suffix)
    assert not set(suffix) & set(old)
    transferred = {
        config for _, config in spec.create_config_generation().seed_flat_config_pairs()
    }
    assert all(spec.normalized_config(seed) in transferred for seed in old)
    for seed in suffix:
        eligible_parents = [
            parent
            for parent in parents
            if parent.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
            and parent.num_warps == seed.num_warps
            and parent.config.get("cute_chained_group_contractions", False)
            == seed.config.get("cute_chained_group_contractions", False)
        ]
        parent = max(
            reversed(eligible_parents), key=lambda item: math.prod(item.block_sizes)
        )
        assert seed.config == parent.config | {
            "cute_chained_warp_mma_rows": 32,
            KEY: True,
        }
        assert seed.num_warps in (8, 16)
        assert spec.normalized_config(seed) in transferred


@pytest.mark.parametrize("seeded", [False, True])
def test_one_sided_contractions_do_not_advertise_pipeline(seeded: bool) -> None:
    with _cpu_codegen():
        bound = _selection_loop._bind_isolated(
            (
                torch.empty((3, 16, 16), dtype=torch.bfloat16),
                torch.empty((3, 16, 32), dtype=torch.bfloat16),
                torch.empty((16, 32)),
                16,
                seeded,
            )
        )
        assert bound.config_spec.cute_chained_loop_search_enabled
        assert not bound.config_spec.cute_chained_preparation_pipeline_search_enabled
        assert KEY not in bound.config_spec._flat_fields()
        with pytest.raises(exc.InvalidConfig, match="eligible common preparation loop"):
            bound.config_spec.normalize(_config())


@pytest.mark.parametrize("missing", ["warp_f16bf16", "tcgen05_f16bf16"])
def test_discovery_requires_both_instruction_engines(missing: str) -> None:
    support = default_cute_mma_support()
    setattr(support, missing, False)
    with _cpu_codegen(), patch_cute_mma_support(support):
        bound = _typed_sequence._bind_isolated(_inputs(False, typed=True))
        assert not bound.config_spec.cute_chained_preparation_pipeline_search_enabled


def test_source_inactive_identity_and_explicit_dispatch(
    pipeline_bound: BoundKernel,
) -> None:
    inactive = _config(False)
    absent = helion.Config.from_dict(
        {k: v for k, v in inactive.config.items() if k != KEY}
    )
    assert pipeline_bound.to_code(inactive) == pipeline_bound.to_code(absent)
    source = pipeline_bound.to_code(_config())
    assert "chain_frames" in source and "chain_recurrence_thread" in source
    assert "chain_generation" in source and "chain_final_store_0" in source


def test_explicit_pipeline_fails_closed_at_physical_capacity(
    pipeline_bound: BoundKernel,
) -> None:
    with (
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=128
        ),
        pytest.raises(exc.BackendUnsupported, match="shared-memory plan"),
    ):
        pipeline_bound.to_code(_config())


def test_root_inactive_identity_without_pipeline_discovery() -> None:
    with _cpu_codegen():
        bound = _root_residency._bind_isolated(
            (
                torch.empty((128, 16), dtype=torch.bfloat16),
                torch.empty((16, 32), dtype=torch.bfloat16),
            )
        )
        assert not bound.config_spec.cute_chained_preparation_pipeline_search_enabled
        config = helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
        assert bound.to_code(config) == bound.to_code(
            helion.Config.from_dict(config.config | {KEY: False})
        )
        with pytest.raises(exc.InvalidConfig, match="eligible common preparation loop"):
            bound.config_spec.normalize(_config(), _fix_invalid=True)


@pytest.mark.parametrize("sequences", [1, 2])
def test_legacy_defaults_and_all_six_seeds_keep_absent_common_family(
    sequences: int,
) -> None:
    with (
        _real_dispatch_cpu(),
        patch.object(CuteChunkRecurrenceHeuristic, "should_promote", return_value=True),
    ):
        bound = _bt16_fp32_chain._bind_isolated(
            (*_fake_inputs(fp32_state=True, sequences=sequences), 128**-0.5)
        )
        spec = bound.config_spec
        generation = spec.create_config_generation()
        default = spec.default_config()
        assert "cute_chained_mma_schedule" not in default.config
        inactive = helion.Config.from_dict(default.config | {KEY: False})
        assert spec.normalized_config(inactive) == default
        assert generation.unflatten(generation.flatten(default)) == default
        assert bound.to_code(default) == bound.to_code(inactive)
        seeds = [
            config
            for _, config in generation.seed_flat_config_pairs()
            if "cute_chained_mma_schedule" not in config.config
        ]
        assert len(seeds) == 6
        for seed in seeds:
            assert KEY not in seed.config
            assert generation.unflatten(generation.flatten(seed)) == seed


def test_actual_kda_discovery_and_valid_pipeline_source() -> None:
    kernel, args = _kda_fixture()
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        assert bound.config_spec.cute_chained_preparation_pipeline_search_enabled
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            source = bound.to_code(
                _config(
                    num_warps=16,
                    block_sizes=[128],
                    cute_chained_group_contractions=True,
                    cute_chained_scratch_layout="xor",
                    cute_chained_scan_schedule="warp",
                    cute_chained_pointwise_cache_bytes=4096,
                    cute_chained_pointwise_vectorize=True,
                    cute_chained_pointwise_unroll=8,
                )
            )
        assert "chain_frames" in source and "chain_13_seed_14_segment" in source
        assert "chain_12_c =" not in source


def test_aliasing_runtime_does_not_advertise_preparation_pipeline() -> None:
    kernel, args = _kda_fixture()
    aliased = list(args)
    aliased[8] = args[2]
    with _cpu_codegen():
        bound = kernel._bind_isolated(tuple(aliased))
        assert not bound.config_spec.cute_chained_preparation_pipeline_search_enabled
        assert KEY not in bound.config_spec._flat_fields()


@pytest.mark.parametrize("chunk_size", [16, 32])
def test_explicit_ungrouped_pipeline_owns_true_prefill_tile_and_flat_schema(
    chunk_size: int,
) -> None:
    original = (
        kda_prefill_native_math if chunk_size == 16 else kda_prefill_native_math_bt32
    )
    overrides = _config(cute_chained_group_contractions=False).config
    kernel = helion.kernel(
        original.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides=overrides,
    )
    args = _prefill_inputs(heads=8, device=torch.device("cpu"))
    with _cpu_codegen():
        legacy = original._bind_isolated(args)
        bound = kernel._bind_isolated(args)
        spec = bound.config_spec
        assert spec.cute_chained_preparation_pipeline_search_enabled
        assert spec.cute_chunk_prefill_task_order is None
        assert spec.cute_chunk_prefill_schedule is None
        assert (spec.block_sizes[0].min_size, spec.block_sizes[0].max_size) == (64, 128)
        fields = spec._flat_fields()
        assert KEY in fields and "cute_chained_mma_schedule" in fields
        assert "cute_chunk_prefill_schedule" not in fields
        config = spec.normalized_config(
            _config(block_sizes=[128], cute_chained_group_contractions=False)
        )
        generation = spec.create_config_generation(overrides=overrides)
        restored = generation.unflatten(generation.flatten(config))
        assert restored[KEY] is True
        assert restored.config.get("cute_chained_group_contractions", False) is False
        assert restored.block_sizes == [128]
        assert legacy.config_spec.block_sizes[0].max_size == 64
        assert set(legacy.config_spec._flat_fields()) == {
            "block_sizes",
            "cute_chunk_prefill_task_order",
            "cute_chunk_prefill_schedule",
            "cute_state_transfer_max_bits",
        } | ({"cute_state_transfer_transport"} if chunk_size == 32 else set())
