from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _inputs
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_preparation_search import _config
import helion
from helion import exc
from helion._compiler.cute import chained_pipeline_storage as storage_module
from helion.autotuner.config_spec import CUTE_CHAINED_COLLECTIVE_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_LEAF_COUNT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_NATIVE_VECTOR_READS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OPERAND_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_COHORTS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_UNROLL_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY


def test_default_cohort_and_unroll_options_preserve_source() -> None:
    args = _inputs(False, typed=True)
    with (
        _cpu_codegen(),
        patch.object(
            storage_module,
            "finalize_pipeline_storage",
            side_effect=AssertionError("default storage path changed"),
        ),
    ):
        bound = _typed_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)):
            default = bound.to_code(_config())
            explicit = bound.to_code(
                _config(
                    cute_chained_preparation_cohorts=1,
                    cute_chained_preparation_unroll=0,
                )
            )
    assert default == explicit


def test_generic_cohorts_use_disjoint_slots_and_ordered_consumer() -> None:
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)):
            source = bound.to_code(
                _config(
                    num_warps=16,
                    cute_chained_preparation_cohorts=3,
                    cute_chained_preparation_unroll=1,
                )
            )
    assert "chain_cohort = chain_thread // 128" in source
    assert "chain_prep_thread = chain_thread % 128" in source
    assert "NamedBarrier(barrier_id=5 + chain_cohort, num_threads=128)" in source
    assert "chain_generation = chain_iteration // 3" in source
    assert (
        ast.unparse(
            ast.parse(
                "cute.arch.mbarrier_wait(chain_slot_bars + 3 + chain_slot, (chain_generation - 1) & 1)",
                mode="eval",
            )
        )
        in source
    )
    assert "chain_sync.arrive_mbarrier(chain_slot_bars + 3 + chain_slot)" in source
    assert "chain_loop_begin + chain_cohort *" in source
    assert (
        "for chain_loop_index in cutlass.range(chain_loop_begin, chain_loop_end,"
        in source
    )


def test_kda_cohorts_finalize_actual_late_transports() -> None:
    kernel, args = _kda_fixture()
    config = helion.Config(
        block_sizes=[128],
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_scan_schedule="warp",
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_unroll=8,
        cute_chained_vector_group=True,
        cute_chained_preparation_pipeline=True,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_seed_tile_columns=32,
        cute_chained_pointwise_cache_layout="xor",
        cute_chained_preparation_cohorts=3,
        cute_chained_preparation_unroll=1,
    )
    original = storage_module.finalize_pipeline_storage
    results = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    with (
        _cpu_codegen(),
        patch.object(storage_module, "finalize_pipeline_storage", side_effect=observe),
    ):
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            source = bound.to_code(config)
    assert len(results) == 1
    result = results[0]
    assert result is not None
    assert result.recurrence.a_bytes == result.recurrence.b_bytes == 0
    assert result.recurrence.layout.allocated_bytes == 16384
    assert result.charged_bytes == 166528
    assert len(result.carry_views) == 1 and result.carry_views[0].pool == "frames"
    assert "chain_a_workspace =" not in source
    assert "chain_b_workspace =" not in source
    assert "chain_c_workspace = cute.arch.alloc_smem(cutlass.Float32, 4096" in source
    assert "chain_slot_bars = cute.arch.alloc_smem(cutlass.Int64, 9" in source
    assert "chain_slot_bars + 6 + chain_slot" in source
    assert (
        "chain_loop_carry_2 = cute.make_tensor(cute.recast_ptr(chain_frames + 0"
        in source
    )


@pytest.mark.parametrize("value", [True, False, 0, 8, None, 3.0, "3"])
def test_cohort_option_requires_strict_count(value: object) -> None:
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        with pytest.raises(exc.InvalidConfig, match="preparation_cohorts must"):
            bound.config_spec.normalize(
                _config(num_warps=16, cute_chained_preparation_cohorts=value)
            )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [True, False, -1, 3, None, 1.0, "1"])
def test_preparation_unroll_requires_strict_supported_factor(
    value: object, repair: bool
) -> None:
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        with pytest.raises(exc.InvalidConfig, match="preparation_unroll must"):
            bound.config_spec.normalize(
                _config(cute_chained_preparation_unroll=value), _fix_invalid=repair
            )


@pytest.mark.parametrize(
    "count,warps,consumer", [(2, 16, 8), (3, 16, 4), (4, 32, 16), (7, 32, 4)]
)
def test_cohort_config_roundtrip_keeps_existing_flat_prefix(
    count: int, warps: int, consumer: int
) -> None:
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        spec = bound.config_spec
        fields = spec._flat_fields()
        keys = (
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
            "cute_chained_frontier_stmatrix",
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
        assert tuple(fields)[-len(keys) :] == keys
        config = spec.normalized_config(
            _config(
                num_warps=warps,
                cute_chained_pipeline_consumer_warps=consumer,
                cute_chained_preparation_cohorts=count,
                cute_chained_preparation_unroll=1,
            )
        )
        generation = spec.create_config_generation()
        assert generation.unflatten(generation.flatten(config)) == config
        default = spec.default_config()
        assert all(key not in default.config for key in keys)
        with patch.object(
            spec,
            "_flat_fields",
            return_value={
                key: value for key, value in fields.items() if key not in keys
            },
        ):
            old_generation = spec.create_config_generation()
            assert generation.flatten(default)[: -len(keys)] == old_generation.flatten(
                default
            )


@pytest.mark.parametrize("count,warps,consumer", [(2, 16, 4), (3, 16, 8), (7, 16, 4)])
def test_cohort_config_rejects_fractional_or_incomplete_teams(
    count: int, warps: int, consumer: int
) -> None:
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        with pytest.raises(exc.InvalidConfig, match="equal whole-128-thread"):
            bound.config_spec.normalize(
                _config(
                    num_warps=warps,
                    cute_chained_pipeline_consumer_warps=consumer,
                    cute_chained_preparation_cohorts=count,
                )
            )


def test_provisional_cohort_request_cannot_bypass_final_storage_admission() -> None:
    args = _inputs(False, typed=True)
    with (
        _cpu_codegen(),
        patch.object(storage_module, "finalize_pipeline_storage", return_value=None),
    ):
        bound = _typed_sequence._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)),
            pytest.raises(
                exc.BackendUnsupported,
                match="proved complete post-transport allocation",
            ),
        ):
            bound.to_code(_config(num_warps=16, cute_chained_preparation_cohorts=3))
