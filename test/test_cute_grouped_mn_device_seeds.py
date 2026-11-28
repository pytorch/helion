from __future__ import annotations

import ast
from contextlib import contextmanager
import random
from typing import TYPE_CHECKING
from typing import NamedTuple
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics import cute as heuristics
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_grouped_gemm import _grouped_offsets_k64
from helion._testing.cute_grouped_gemm import _unsupported_k_loop
from helion._testing.cute_population_contracts import _target
from helion._testing.cute_population_contracts import checked_initial_population
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


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


@contextmanager
def _mn_target() -> Iterator[None]:
    with (
        forbid_cuda_work(),
        _target(),
        mock_cuda_unavailable(),
        patch.object(
            heuristics, "_tcgen05_grouped_prefix_adaptive_n_seeds", return_value=[]
        ),
    ):
        yield


class _MNSeedCase(NamedTuple):
    columns: int
    old: ConfigSpec
    bound: BoundKernel
    added: list[helion.Config]
    representatives: tuple[helion.Config, ...]


@pytest.fixture(
    scope="module",
    params=[(2048, 16, 1536, 2048), (8192, 16, 1536, 2048), (512, 4, 512, 256)],
    ids=lambda shape: "-".join(map(str, shape)),
)
def mn_seed_case(request: pytest.FixtureRequest) -> Iterator[_MNSeedCase]:
    m, groups, n, k = request.param
    args = _args(m, groups, n, k)
    with _mn_target():
        with patch.object(
            heuristics, "_tcgen05_grouped_mn_device_tile_seeds", return_value=[]
        ):
            old = cpu_bind(_kernel(), args).config_spec
        bound = cpu_bind(_kernel(), args)
    added = bound.config_spec.compiler_seed_configs[len(old.compiler_seed_configs) :]
    representatives: dict[tuple[object, ...], helion.Config] = {}
    for seed in added:
        # Pipeline depths resize the same source. Keep the first seed for
        # every worklist geometry and L2 swizzle.
        representatives.setdefault(
            (
                seed["tcgen05_grouped_worklist_source_m_tile"],
                seed["tcgen05_grouped_worklist_mma_n"],
                seed["tcgen05_cluster_n"],
                seed.get("tcgen05_grouped_balanced_workers"),
                seed.get("tcgen05_l2_swizzle_size"),
                seed.get("tcgen05_grouped_l2_swizzle_alternate"),
                seed.get("tcgen05_grouped_l2_swizzle_mn_ratio"),
            ),
            seed,
        )
    # BoundKernel weakly retains runtime inputs: keep args alive while any
    # batch may render from this binding, with no target mocks active.
    yield _MNSeedCase(n, old, bound, added, tuple(representatives.values()))


def test_mn_seeds_preserve_old_prefix_normalize_and_reach_native_source(
    mn_seed_case: _MNSeedCase,
) -> None:
    old_seeds = mn_seed_case.old.compiler_seed_configs
    added = mn_seed_case.added
    with _mn_target():
        spec = mn_seed_case.bound.config_spec
        assert spec.compiler_seed_configs[: len(old_seeds)] == old_seeds
        assert spec.compiler_default_config == mn_seed_case.old.compiler_default_config
        assert spec.block_sizes[2].declared_max_size == 64
        assert added
        for seed in added:
            assert seed.get("tcgen05_grouped_worklist_orientation") == "mn"
            assert seed.block_sizes == [256, 128, 64]
            assert seed["tcgen05_grouped_full_allocation_a"] is True
            assert seed["tcgen05_grouped_d_shift"] is True
            assert spec.normalized_config(seed).block_sizes[2] == 64
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
        if mn_seed_case.columns == 1536:
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


# Cross the original three shapes with disjoint slices of their complete
# representative sets so one pytest item cannot spend minutes rendering them.
_RENDER_BATCHES = 8


@pytest.mark.parametrize("batch", range(_RENDER_BATCHES))
def test_mn_seed_native_sources(mn_seed_case: _MNSeedCase, batch: int) -> None:
    bound = mn_seed_case.bound
    with _mn_target():
        for seed in mn_seed_case.representatives[batch::_RENDER_BATCHES]:
            source = bound.to_code(bound.config_spec.normalized_config(seed))
            ast.parse(source)
            assert "'bounded_output_shift': True" in source
            assert "'immutable_ab_tensormaps': True" in source
            assert "TensorMapManager" not in source


def test_resource_and_complete_cluster_bounds_apply_to_each_proposal() -> None:
    with forbid_cuda_work(), _target(), mock_cuda_unavailable():
        bound = cpu_bind(_kernel(), _args(512, 4, 256, 256))
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
    with forbid_cuda_work(), _target(), mock_cuda_unavailable():
        bound = cpu_bind(_kernel(), _args(512, 4, 256, 256))
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
    with forbid_cuda_work(), _target(), mock_cuda_unavailable():
        bound = cpu_bind(_kernel(), args)
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
    with forbid_cuda_work(), _target(), mock_cuda_unavailable():
        bound = cpu_bind(_unsupported_k_loop, (*_args(512, 4, 256, 256), 1))
        assert not bound.config_spec._cute_tcgen05_config.grouped_d_shift_supported
        assert not any(
            seed.get("tcgen05_grouped_worklist_orientation") == "mn"
            for seed in bound.config_spec.compiler_seed_configs
        )


def test_mn_descriptor_discount_requires_aligned_tma_inputs() -> None:
    rows, groups, columns, reduction = 512, 4, 512, 256
    storage = torch.empty(rows * reduction + 1, dtype=torch.bfloat16)
    other = _args(rows, groups, columns, reduction)[1:]
    with forbid_cuda_work(), _target(), mock_cuda_unavailable():
        for offset in (0, 1):
            a = storage[offset : offset + rows * reduction].view(rows, reduction)
            bound = cpu_bind(_kernel(), (a, *other))
            facts = bound.config_spec._cute_tcgen05_config
            assert facts.grouped_d_shift_supported
            assert facts.matmul_operands_tma_provable is (offset == 0)
            mn = [
                seed
                for seed in bound.config_spec.compiler_seed_configs
                if seed.get("tcgen05_grouped_worklist_orientation") == "mn"
            ]
            assert bool(mn) is (offset == 0)
