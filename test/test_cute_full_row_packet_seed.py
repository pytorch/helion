from __future__ import annotations

import random
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_bounded_cache_codegen import _cpu_target
from test.test_cute_full_chunk_row_seeds import bind
from test.test_cute_rolled_cluster_packet_seeds import _fused_rows

import helion
from helion._compiler.autotuner_heuristics.cute_rolled_cluster_packet import (
    CuteRolledClusterPacketHeuristic,
)
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def cpu_only() -> Iterator[None]:
    initialized = torch.cuda.is_initialized()
    with _cpu_target(), _forbid_native_compile() as native:
        yield
        assert all(mock.call_count == 0 for mock in native)
    assert torch.cuda.is_initialized() == initialized


def full_rows(bound: BoundKernel) -> list[helion.Config]:
    with bound.env:
        return CuteRolledClusterPacketHeuristic.full_row_seed_configs(
            bound.env, bound.host_function.device_ir
        )


@pytest.mark.parametrize(
    "rows,size,dtype",
    [
        (16, 7168, torch.bfloat16),
        (128, 8192, torch.bfloat16),
        (4, 6144, torch.float16),
        (4, 8192, torch.float32),
    ],
)
def test_one_full_row_profile_strictly_roundtrips(
    rows: int, size: int, dtype: torch.dtype
) -> None:
    bound = bind(rows, size, dtype)
    seeds = full_rows(bound)
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.num_threads == [1, 512]
    assert seed.reduction_loops == [1 << (size - 1).bit_length()]
    assert seed["cute_vector_widths"] == [4, 1]
    assert seed["cute_cluster_n"] == 1
    assert seed["cute_independent_reduction"] is True
    assert seed["cute_replicated_reduction"] is True
    assert seed["cute_pdl"] is True
    assert seed["cute_min_blocks_per_mp"] == 0
    assert set(seed["load_eviction_policies"]) == {"l2_last"}
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        flat, effective = generation.strict_config_pair(seed)
        assert generation.strict_unflatten(flat) == effective == seed
    code = bound.to_code(seed)
    assert "block=(512, 1, 1)" in code
    assert "_cute_grouped_reduce_cluster(" not in code
    assert "_consume_prefetch_0 = cute.make_rmem_tensor" in code
    assert "l2_evict_last" in code
    assert "cute.arch.sync_threads()" in code


@pytest.mark.parametrize(
    "rows,size,dtype",
    [
        (16, 7168, torch.bfloat16),
        (128, 8192, torch.bfloat16),
        (128, 8192, torch.float32),
    ],
)
def test_actual_full_population_keeps_old_prefix_and_contains_full_row(
    rows: int, size: int, dtype: torch.dtype
) -> None:
    kernel = helion.kernel(
        _fused_rows,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        autotune_initial_population_strategy="from_random",
        autotune_seed_configs=None,
    )
    args = (
        torch.empty((rows, size), dtype=dtype),
        torch.empty((rows, size), dtype=dtype),
        torch.empty(size, dtype=dtype),
    )
    with patch.object(
        CuteRolledClusterPacketHeuristic, "full_row_seed_configs", return_value=[]
    ):
        previous = kernel._bind_isolated(args)
    current = kernel._bind_isolated(args)
    seeds = full_rows(current)
    assert len(seeds) == 1
    old = previous.config_spec.compiler_seed_configs
    assert current.config_spec.compiler_seed_configs[: len(old)] == old
    assert seeds[0] in current.config_spec.compiler_seed_configs
    assert (
        current.config_spec.compiler_default_config
        == previous.config_spec.compiler_default_config
    )
    with previous.env:
        default = previous.config_spec.default_config()
        reference = previous.config_spec.autotune_reference_config()
    with current.env:
        assert current.config_spec.default_config() == default
        assert current.config_spec.autotune_reference_config() == reference
        generation = ConfigGeneration(current.config_spec)
        profile = get_effort_profile("full").pattern_search
        assert (
            profile is not None and profile.initial_population_strategy == "from_random"
        )
        random.seed(152100)
        population = [
            generation.unflatten(flat)
            for flat in generation.random_population_flat(profile.initial_population)
        ]
        assert len(population) == profile.initial_population == 100
        assert seeds[0] in population
    assert current.to_code(default) == previous.to_code(default)


@pytest.mark.parametrize("layout", ["strided", "row_stride"])
def test_unproved_layout_declines(layout: str) -> None:
    assert not full_rows(bind(4, 7168, torch.bfloat16, layout=layout))


def test_dynamic_and_high_grid_decline() -> None:
    assert not full_rows(bind(4, 7168, torch.bfloat16, static=False))
    assert not full_rows(bind(149, 7168, torch.bfloat16))


@pytest.mark.parametrize("dtype", [torch.int8, torch.float64])
def test_unsupported_dtype_declines(dtype: torch.dtype) -> None:
    assert not full_rows(bind(4, 7168, dtype))


@pytest.mark.parametrize("reason", ["threads", "domain", "cluster", "policy", "budget"])
def test_unreachable_or_over_budget_profile_declines(reason: str) -> None:
    bound = bind(4, 7168, torch.bfloat16)
    spec = bound.config_spec
    if reason == "threads":
        with patch.object(spec, "max_reduction_threads", 256):
            assert not full_rows(bound)
    elif reason == "domain":
        item = spec.num_threads.block_id_lookup(spec.reduction_loops[0].block_id)
        with patch.object(item, "_fragment", return_value=EnumFragment((128, 256))):
            assert not full_rows(bound)
    elif reason == "cluster":
        fields = spec._flat_fields()
        fields["cute_cluster_n"] = EnumFragment((2, 4))
        with patch.object(spec, "_flat_fields", return_value=fields):
            assert not full_rows(bound)
    elif reason == "policy":
        with patch.object(
            spec.load_eviction_policies.inner,
            "search_values",
            return_value=["", "first"],
        ):
            assert not full_rows(bound)
    else:
        with patch.object(
            CuteRolledClusterPacketHeuristic,
            "_load_metadata",
            return_value=(torch.bfloat16, 100, ()),
        ):
            assert not full_rows(bound)
