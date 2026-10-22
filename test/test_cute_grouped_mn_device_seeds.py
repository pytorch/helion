from __future__ import annotations

import ast
import random
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.cute_population_contracts import checked_initial_population
from test.test_cute_grouped_ab_k import _unsupported_k_loop
from test.test_cute_grouped_large_device_seeds import _grouped_offsets_k64

import helion
from helion._compiler.autotuner_heuristics import cute as heuristics
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch


def _kernel():
    return helion.kernel(
        _grouped_offsets_k64.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )


def _args(m: int, groups: int, n: int, k: int):
    return (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((groups, n, k), dtype=torch.bfloat16),
        torch.arange(groups + 1, dtype=torch.int32) * (m // groups),
    )


@pytest.mark.parametrize(
    "m,groups,n,k", [(2048, 16, 1536, 2048), (8192, 16, 1536, 2048), (512, 4, 512, 256)]
)
@pytest.mark.parametrize("render_shard", range(8))
def test_mn_seeds_preserve_old_prefix_normalize_and_reach_native_source(
    m: int, groups: int, n: int, k: int, render_shard: int
) -> None:
    args = _args(m, groups, n, k)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        with patch.object(
            heuristics, "_tcgen05_grouped_mn_device_tile_seeds", return_value=[]
        ):
            previous = _cpu_bind(_kernel(), args)
            old_seeds = list(previous.config_spec.compiler_seed_configs)
            old_default = previous.config_spec.compiler_default_config
        bound = _cpu_bind(_kernel(), args)
        spec = bound.config_spec
        seeds = spec.compiler_seed_configs
        assert seeds[: len(old_seeds)] == old_seeds
        assert spec.compiler_default_config == old_default
        assert spec.block_sizes[2].declared_max_size == 64
        added = seeds[len(old_seeds) :]
        assert added
        for seed_index, seed in enumerate(added):
            assert seed.get("tcgen05_grouped_worklist_orientation") == "mn"
            assert seed.block_sizes == [256, 128, 64]
            assert seed["tcgen05_grouped_full_allocation_a"] is True
            assert seed["tcgen05_grouped_d_shift"] is True
            normalized = spec.normalized_config(seed)
            assert normalized.block_sizes[2] == 64
            # Keep all normalization checks in every case. Each added seed's
            # full source is rendered once across bounded CPU-only shards.
            if seed_index % 8 == render_shard:
                source = bound.to_code(normalized)
                ast.parse(source)
                assert "'bounded_output_shift': True" in source
                assert "'immutable_ab_tensormaps': True" in source
                assert "TensorMapManager" not in source
        mn = [
            seed
            for seed in added
            if seed.get("tcgen05_grouped_balanced_workers") is not True
        ]
        assert {seed["tcgen05_grouped_worklist_source_m_tile"] for seed in mn} == {
            128,
            256,
        }
        assert {seed["tcgen05_grouped_worklist_mma_n"] for seed in mn} == {
            128,
            192,
            224,
            256,
        }
        if n == 1536:
            assert any(
                seed["tcgen05_grouped_worklist_source_m_tile"] == 128
                and seed["tcgen05_grouped_worklist_mma_n"] == 192
                and seed["tcgen05_cluster_n"] == 2
                and seed["tcgen05_ab_stages"] == 10
                and seed["tcgen05_c_stages"] == 3
                for seed in mn
            )
            assert any(
                seed["tcgen05_grouped_worklist_source_m_tile"] == 256
                and seed["tcgen05_grouped_worklist_mma_n"] == 224
                and seed["tcgen05_cluster_n"] == 1
                and seed["tcgen05_ab_stages"] == 7
                for seed in mn
            )


def test_resource_and_complete_cluster_bounds_apply_to_each_proposal() -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args(512, 4, 256, 256))
        seeds = heuristics._tcgen05_grouped_mn_device_tile_seeds(
            bound.config_spec,
            groups=4,
            n=160,
            k=256,
            b_major="k",
            dtype_bytes=2,
            output_dtype_bytes=2,
            capacity_bytes=150_000,
        )
        assert seeds
        for seed in seeds:
            row_tile = seed["tcgen05_grouped_worklist_source_m_tile"]
            column_tile = seed["tcgen05_grouped_worklist_mma_n"]
            assert seed["tcgen05_cluster_n"] == 1
            assert (
                tcgen05_grouped_worklist_smem_bytes(
                    group_count=4,
                    device_split_sizes=True,
                    sched_stage_count=1,
                    bm=row_tile,
                    bn=column_tile,
                    bk=64,
                    dtype_bytes=2,
                    output_dtype_bytes=2,
                    ab_stages=seed["tcgen05_ab_stages"],
                    acc_stages=2,
                    c_stages=seed["tcgen05_c_stages"],
                    cluster_m=2,
                    orientation="mn",
                    immutable_ab_tensormaps=True,
                    immutable_d_tensormap=True,
                )
                <= 150_000
            )


@pytest.mark.parametrize(
    "updates",
    [
        {"b_major": "n"},
        {"dtype_bytes": 4},
        {"k": 96},
        {"groups": 1 << 30},
        {"capacity_bytes": 1},
    ],
)
def test_seed_helper_rejects_unproved_or_out_of_domain_inputs(updates) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args(512, 4, 256, 256))
        values = {
            "groups": 4,
            "n": 256,
            "k": 256,
            "b_major": "k",
            "dtype_bytes": 2,
            "output_dtype_bytes": 2,
            "capacity_bytes": 232448,
        }
        values.update(updates)
        assert not heuristics._tcgen05_grouped_mn_device_tile_seeds(
            bound.config_spec, **values
        )


def test_mn_seed_configs_are_in_a_fresh_full_search_population() -> None:
    args = _args(2048, 16, 1536, 2048)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), args)
        assert bound.config_spec.compiler_seed_overflow_requested
        rng_state = random.getstate()
        try:
            random.seed(9927)
            search = LFBOTreeSearch(
                bound,
                args,
                **LFBOTreeSearch.get_kwargs_from_profile(
                    get_effort_profile("full"), bound.settings
                ),
            )
            configs = [
                search.config_gen.unflatten(row)
                for row in checked_initial_population(search)
            ]
            mn = [
                config
                for _, config in search.config_gen.seed_flat_config_pairs()
                if config.get("tcgen05_grouped_worklist_orientation") == "mn"
            ]
            assert mn
            assert all(
                config in configs
                for _, config in search.config_gen.seed_flat_config_pairs()
            )
            missing = [config for config in mn if config not in configs]
            assert not missing, [
                (
                    config.block_sizes,
                    config["tcgen05_grouped_worklist_source_m_tile"],
                    config["tcgen05_grouped_worklist_mma_n"],
                    config["tcgen05_cluster_n"],
                    config["tcgen05_ab_stages"],
                    config["tcgen05_c_stages"],
                )
                for config in missing
            ]
            assert all(config.block_sizes[2] == 64 for config in mn)
            assert all(config.block_sizes == [256, 128, 64] for config in mn)
            for row, column, cluster_n, c_stages in (
                (128, 192, 2, 3),
                (256, 224, 1, 2),
            ):

                def matches(
                    config,
                    row=row,
                    column=column,
                    cluster_n=cluster_n,
                    c_stages=c_stages,
                ):
                    return (
                        config.get("tcgen05_grouped_worklist_orientation") == "mn"
                        and config.get("tcgen05_grouped_worklist_source_m_tile") == row
                        and config.get("tcgen05_grouped_worklist_mma_n") == column
                        and config.get("tcgen05_cluster_n") == cluster_n
                        and config.get("tcgen05_c_stages") == c_stages
                        and config.get("tcgen05_grouped_balanced_workers") is not True
                    )

                raw = next(
                    config
                    for config in bound.config_spec.compiler_seed_configs
                    if matches(config)
                )
                transferred = next(config for config in mn if matches(config))
                assert bound.to_code(raw) == bound.to_code(transferred)
        finally:
            random.setstate(rng_state)


def test_noncanonical_reduction_does_not_receive_immutable_mn_proposals() -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_unsupported_k_loop, (*_args(512, 4, 256, 256), 1))
        assert not bound.config_spec._cute_tcgen05_config.grouped_d_shift_supported
        assert not any(
            seed.get("tcgen05_grouped_worklist_orientation") == "mn"
            for seed in bound.config_spec.compiler_seed_configs
        )


def test_mn_descriptor_discount_requires_aligned_tma_inputs() -> None:
    rows, groups, columns, reduction = 512, 4, 512, 256
    storage = torch.empty(rows * reduction + 1, dtype=torch.bfloat16)
    other = _args(rows, groups, columns, reduction)[1:]
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        for offset in (0, 1):
            a = storage[offset : offset + rows * reduction].view(rows, reduction)
            bound = _cpu_bind(_kernel(), (a, *other))
            facts = bound.config_spec._cute_tcgen05_config
            assert facts.grouped_d_shift_supported
            assert facts.matmul_operands_tma_provable is (offset == 0)
            mn = [
                seed
                for seed in bound.config_spec.compiler_seed_configs
                if seed.get("tcgen05_grouped_worklist_orientation") == "mn"
            ]
            assert bool(mn) is (offset == 0)
