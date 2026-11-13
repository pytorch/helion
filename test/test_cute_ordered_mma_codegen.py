"""MMA phase admission, independent sibling configurations, and workspace strides."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import materialized_fission_codegen
from helion._compiler.cute.cute_mma_dag import MMA_DAG_KEYS
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_ordered_mma import duplicate_mma_loads
from helion._testing.cute_ordered_mma import partitioned_matmul_sum
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = pytest.mark.cpu_only


def test_off_mode_keeps_persistent_barrier_search():
    with cpu_target():
        bound = cpu_bind(
            duplicate_mma_loads,
            (
                torch.empty(128, 64, dtype=torch.bfloat16),
                torch.empty(64, 64, dtype=torch.bfloat16),
            ),
        )
        spec = bound.config_spec
        assert spec.cute_mma_dag_phase_index == 1
        assert spec.default_config().pid_type == "flat"
        generation = ConfigGeneration(spec)
        indices, is_sequence = generation._key_to_flat_indices["pid_type"]
        assert not is_sequence
        (index,) = indices
        choices = generation.flat_spec[index].search_values()
        assert choices is not None and "persistent_blocked" in choices
        off = helion.Config(
            block_sizes=[128, 64, 64, 64],
            cute_mma_dag="off",
            pid_type="persistent_blocked",
        )
        searched = generation.unflatten(generation.flatten(off))
        assert searched.pid_type == "persistent_blocked"
        assert searched["cute_mma_dag"] == "off"
        source = bound.to_code(searched)
        assert "_cute_grid_barrier" in source
        assert "_helion_mma_dag" not in source
        active = spec.normalized_config(
            helion.Config.from_dict(off.config | {"cute_mma_dag": "warp"})
        )
        assert active.pid_type == "flat"
        assert len(stage_sources(bound.to_code(active))) == 3


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("splits", [1, 8])
def test_mma_phase_preserves_sibling_owners_and_configured_strides(mode, splits):
    projected = {}
    original = materialized_fission_codegen._stage_config

    def record(config, plan, index, *args, **kwargs):
        result = original(config, plan, index, *args, **kwargs)
        projected[index] = result.config
        return result

    with cpu_target():
        bound = cpu_bind(
            partitioned_matmul_sum,
            (
                torch.empty(2, 8, 64, 16, dtype=torch.bfloat16),
                torch.empty(2, 8, 16, 16, dtype=torch.bfloat16),
            ),
        )
        spec = bound.config_spec
        assert spec.cute_mma_dag_phase_index == 1
        assert spec.default_config().pid_type == "flat"
        assert spec.cute_mma_dag_persistent is None
        assert spec.cute_mma_dag_cluster is None
        config = helion.Config(
            block_sizes=[splits, 128, 64],
            cute_mma_dag=mode,
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
            load_eviction_policies=["first", "l2_last", "", "last"],
        )
        with patch.object(materialized_fission_codegen, "_stage_config", record):
            source = bound.to_code(config)
    stages = stage_sources(source)
    assert len(stages) == 3
    assert not set(MMA_DAG_KEYS).intersection(projected[0])
    assert not set(MMA_DAG_KEYS).intersection(projected[2])
    assert projected[1]["cute_mma_dag"] == mode
    expected_policies = ["first", "l2_last" if mode == "warp" else "", "", "last"]
    assert projected[0]["load_eviction_policies"] == expected_policies
    assert projected[2]["load_eviction_policies"] == expected_policies
    assert len(projected[1]["load_eviction_policies"]) == 2
    assert "_helion_mma_dag" not in stages[0]
    assert "_helion_mma_dag" not in stages[2]
    assert "_helion_mma_dag" in stages[1]
    # The MMA stage bypasses ordinary load codegen but still owns global
    # slots 1 and 2. Inspect the emitted policy, not just the projected list.
    assert "_cute_scalar_policy_load(partial.iterator" in stages[2]
    assert ", cutlass.Float32, 'last')" in stages[2]
    # The leading group stride depends on the selected split count. The
    # tracing hint (64 splits) addresses outside the allocated workspace.
    pointer = next(
        line for line in stages[1].splitlines() if "= partial.iterator.toint()" in line
    )
    assert f"* {splits * 64 * 16 * 4} " in pointer


def test_two_matmul_phases_do_not_share_a_single_phase_owner():
    from helion._testing.cute_ordered_mma import two_matmul_phases

    with cpu_target():
        bound = cpu_bind(
            two_matmul_phases,
            (
                torch.empty(2, 8, 64, 16, dtype=torch.bfloat16),
                torch.empty(2, 8, 16, 16, dtype=torch.bfloat16),
            ),
        )
        assert bound.config_spec.cute_mma_dag_phase_index is None


def test_mma_phase_canonicalizes_only_its_own_config_slots():
    with cpu_target():
        bound = cpu_bind(
            partitioned_matmul_sum,
            (
                torch.empty(2, 8, 64, 16, dtype=torch.bfloat16),
                torch.empty(2, 8, 16, 16, dtype=torch.bfloat16),
            ),
        )
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[1, 128, 64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                num_threads=[0, 32, 16, 8],
                loop_orders=[[2, 1, 0], [1, 0]],
                l2_groupings=[8, 4],
                range_unroll_factors=[2, 2, 2, 2],
                load_eviction_policies=["first", "l2_last", "l2_last", "last"],
            )
        )
    # The pack and merge still consume their thread/loop/cache choices. The
    # MMA phase's row threads and loop schedule are replaced by its renderer.
    assert config["num_threads"] == [0, 32, 0, 8]
    assert config["loop_orders"] == [[0, 1, 2], [1, 0]]
    assert config["l2_groupings"] == [1, 4]
    # The roots are already static; the MMA-owned inner range is inert too.
    assert config["range_unroll_factors"] == [0, 0, 0, 0]
    assert config["load_eviction_policies"] == ["first", "", "", "last"]


def test_merged_mma_loads_keep_their_original_global_slots():
    with cpu_target():
        bound = cpu_bind(
            duplicate_mma_loads,
            (
                torch.empty(128, 64, dtype=torch.bfloat16),
                torch.empty(64, 64, dtype=torch.bfloat16),
            ),
        )
        spec = bound.config_spec
        assert spec.cute_mma_dag_phase_load_slots == frozenset({1, 2, 3, 4})
        assert set(spec.cute_mma_dag_phase_eviction_map.values()) == {1, 2}
        config = spec.normalized_config(
            helion.Config(
                block_sizes=[128, 64, 64, 64],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                load_eviction_policies=["", "l2_last", "", "l2_last", "last", "last"],
            )
        )
        assert config["load_eviction_policies"][3:5] == ["", ""]
        source = bound.to_code(config)
    trailing = stage_sources(source)[2]
    assert "_cute_scalar_policy_load(partial.iterator" in trailing
    assert ", cutlass.Float32, 'last')" in trailing
