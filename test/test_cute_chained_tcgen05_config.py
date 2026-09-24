from __future__ import annotations

import itertools
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import default_cute_mma_support
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _config_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    row_block: hl.constexpr,
    scan: hl.constexpr,
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), dtype=a.dtype, device=a.device)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, row_block, None]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        if scan:
            decay = hl.cumsum(v[bi, qq, 0].float(), dim=0)
            first *= torch.exp(decay[row][:, None] - decay[qq][None, :])
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, qq, col]).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _permuted_config_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), dtype=a.dtype, device=a.device)
    for col, batch, row in hl.tile([n, batches, m], block_size=[None, 1, None]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, qq, col]).to(a.dtype)
    return out


@pytest.fixture(autouse=True)
def _cpu_support():
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


def _inputs(
    *, m: int = 128, n: int = 256, k: int = 128, q: int = 128
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((2, m, k), (2, q, k), (2, q, n))
    )


def _bound(*, row_block: int | None = None, scan: bool = False, **shape: int):
    _config_chain.reset()
    return _config_chain.bind((*_inputs(**shape), row_block, scan))


def _tcgen_seeds(bound: Any) -> list[helion.Config]:
    return [
        seed
        for seed in _without_early_release_seed(
            bound.config_spec.compiler_seed_configs, bound.config_spec
        )
        if seed.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
        and not seed.config.get("cute_chained_pointwise_vectorize", False)
        and not seed.config.get("cute_chained_auxiliary_cache", False)
        and not seed.config.get("cute_chained_c_smem_padding", 0)
    ]


def _without_startup_seed(seeds: list[helion.Config]) -> list[helion.Config]:
    """Verify the sole startup child, then preserve every original seed object."""
    key = "cute_chained_startup_transfer"
    children = [seed for seed in seeds if seed.config.get(key) == "tma"]
    if not children:
        return seeds
    assert len(children) == 1
    child = children[0]
    index = next(i for i, seed in enumerate(seeds) if seed is child)
    assert index > 0
    parent = seeds[index - 1]
    assert child.config == parent.config | {key: "tma"}
    assert parent.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
    assert parent.config.get("cute_chained_k_schedule", "full") == "full"
    for excluded in (
        "cute_chained_initialized_accumulator",
        "cute_chained_late_rhs_reuse",
        "cute_chained_direct_output",
        "cute_host_selected_fastpath",
    ):
        assert not parent.config.get(excluded, False)
    legacy = [seed for seed in seeds if seed is not child]
    assert len(legacy) + 1 == len(seeds)
    assert all(
        actual is expected
        for actual, expected in zip(
            legacy, seeds[:index] + seeds[index + 1 :], strict=True
        )
    )
    return legacy


def _without_early_release_seed(
    seeds: list[helion.Config], spec: Any | None = None
) -> list[helion.Config]:
    """Check the sole new sibling before testing the entire legacy seed pool."""
    seeds = _without_startup_seed(seeds)
    fast_key = "cute_host_selected_fastpath"
    fast_children = [seed for seed in seeds if seed.config.get(fast_key)]
    if fast_children:
        assert spec is not None
        legacy = [seed for seed in seeds if not seed.config.get(fast_key)]
        expected = []
        geometries = set()
        for parent in legacy:
            if (
                parent.config.get("cute_chained_mma_schedule") != "tcgen05_tmem"
                or parent.config.get("cute_chained_pointwise_vectorize") is not True
                or parent.config.get("cute_chained_pointwise_inplace_async") is True
            ):
                continue
            geometry = tuple(parent.block_sizes)
            if geometry in geometries:
                continue
            geometries.add(geometry)
            normalized = helion.Config.from_dict(parent.config)
            spec.normalize(normalized)
            expected.append(normalized.config | {fast_key: True})
            if len(expected) == 2:
                break
        assert [child.config for child in fast_children] == expected
        assert seeds[1 : 1 + len(expected)] == fast_children
        assert len(seeds) == len(legacy) + len(expected)
        seeds = legacy
    free_key = "cute_chained_tmem_free"
    free_children = [seed for seed in seeds if seed.config.get(free_key) == "last_read"]
    free_legacy = [seed for seed in seeds if seed.config.get(free_key) != "last_read"]
    free_parents = [
        seed
        for seed in free_legacy
        if seed.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
    ]
    assert len(free_children) == int(bool(free_parents))
    if free_parents:
        assert free_children[0].config == free_parents[0].config | {
            free_key: "last_read"
        }
        assert seeds.index(free_children[0]) == seeds.index(free_parents[0]) + 1
    seeds = free_legacy
    coefficient_key = "cute_chained_coefficient_cache"
    coefficient_children = [seed for seed in seeds if seed.config.get(coefficient_key)]
    coefficient_legacy = [
        seed for seed in seeds if not seed.config.get(coefficient_key)
    ]
    coefficient_parents = [
        seed
        for seed in coefficient_legacy
        if seed.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
        and not seed.config.get("cute_chained_direct_output")
    ]
    coefficient_enabled = (
        spec is not None and spec.cute_chained_coefficient_cache_search_enabled
    )
    assert len(coefficient_children) == int(
        coefficient_enabled and bool(coefficient_parents)
    )
    assert len(seeds) == len(coefficient_legacy) + len(coefficient_children)
    if coefficient_children:
        child, parent = coefficient_children[0], coefficient_parents[0]
        assert child.config == parent.config | {coefficient_key: True}
        child_index = next(i for i, seed in enumerate(seeds) if seed is child)
        parent_index = next(i for i, seed in enumerate(seeds) if seed is parent)
        assert child_index == parent_index + 1
        assert all(
            actual is expected
            for actual, expected in zip(
                coefficient_legacy,
                seeds[:child_index] + seeds[child_index + 1 :],
                strict=True,
            )
        )
    seeds = coefficient_legacy
    key = "cute_chained_tmem_early_release"
    legacy = [seed for seed in seeds if not seed.config.get(key)]
    added = [seed for seed in seeds if seed.config.get(key)]
    parents = [
        seed
        for seed in legacy
        if seed.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
    ]
    assert len(added) == int(bool(parents))
    assert len(seeds) == len(legacy) + len(added)
    if parents:
        assert added[0].config == parents[0].config | {key: True}
        assert seeds.index(added[0]) == seeds.index(parents[0]) + 1
        assert seeds[0] is legacy[0]
    return legacy


def test_pointwise_seeds_and_config_roundtrip() -> None:
    spec = _bound().config_spec
    seeds = [
        seed
        for seed in _without_early_release_seed(spec.compiler_seed_configs, spec)
        if seed.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
    ]
    assert {seed.config["cute_chained_pointwise_vectorize"] for seed in seeds} == {
        False,
        True,
    }
    assert len(seeds) == 32
    assert {
        (
            tuple(seed.block_sizes),
            seed.config["cute_chained_pointwise_vectorize"],
            seed.config["cute_chained_auxiliary_cache"],
            seed.config.get("cute_chained_c_smem_padding", 0),
        )
        for seed in seeds
    } == {
        ((128, n), vector, cache, padding)
        for n, vector, cache, padding in itertools.product(
            (32, 64, 128, 256), (False, True), (False, True), (0, 4)
        )
    }
    assert {seed.config["cute_chained_auxiliary_cache"] for seed in seeds} == {
        False,
        True,
    }
    for seed in seeds:
        normalized = spec.normalized_config(seed)
        for key, default in (
            ("cute_chained_pointwise_vectorize", False),
            ("cute_chained_auxiliary_cache", False),
            ("cute_chained_c_smem_padding", 0),
        ):
            assert normalized.config[key] == seed.config.get(key, default)


def test_tcgen05_chain_seeds_are_additive_and_semantic() -> None:
    bound = _bound()
    spec = bound.config_spec
    seeds = _tcgen_seeds(bound)
    assert [seed.block_sizes for seed in seeds] == [
        [128, n] for n in (32, 64, 128, 256)
    ]
    assert all(seed.num_warps == 4 and seed.pid_type == "flat" for seed in seeds)
    assert (
        spec.compiler_seed_configs[0].config["cute_chained_mma_schedule"] == "coalesced"
    )
    assert spec.compiler_seed_configs[0].block_sizes == [16, 16]
    assert any(seed.num_warps == 8 for seed in spec.compiler_seed_configs)
    assert "tcgen05_tmem" in spec._cute_chained_mma_schedules()
    for seed in seeds:
        normalized = spec.normalized_config(seed)
        assert normalized.block_sizes == seed.block_sizes
        assert normalized.num_warps == 4 and normalized.pid_type == "flat"


def test_tcgen05_chain_seeds_follow_permuted_root_axes() -> None:
    _permuted_config_chain.reset()
    bound = _permuted_config_chain.bind(_inputs())
    assert [seed.block_sizes for seed in _tcgen_seeds(bound)] == [
        [n, 128] for n in (32, 64, 128, 256)
    ]


def test_tcgen05_chain_scan_persistent_reduction_axes() -> None:
    assert [seed.block_sizes for seed in _tcgen_seeds(_bound(scan=True))] == [
        [128, n] for n in (32, 64, 128, 256)
    ]


@pytest.mark.parametrize("shape", [{"m": 127}, {"m": 64}, {"n": 95}, {"q": 512}])
def test_tcgen05_chain_ineligible_shapes_keep_warp_schedules(
    shape: dict[str, Any],
) -> None:
    bound = _bound(**shape)
    assert not _tcgen_seeds(bound)
    assert "tcgen05_tmem" not in bound.config_spec._cute_chained_mma_schedules()
    assert (
        "cp_async_register_reuse_scan"
        in bound.config_spec._cute_chained_mma_schedules()
    )


def test_tcgen05_chain_honors_fixed_block_sizes() -> None:
    assert not _tcgen_seeds(_bound(row_block=16))
    assert [seed.block_sizes for seed in _tcgen_seeds(_bound(row_block=128))] == [
        [n] for n in (32, 64, 128, 256)
    ]


def test_tcgen05_chain_hardware_gate() -> None:
    support = default_cute_mma_support()
    support.tcgen05_f16bf16 = False
    with patch(
        "helion._compiler.cute.mma_support.get_cute_mma_support", return_value=support
    ):
        bound = _bound()
    assert not _tcgen_seeds(bound)
    with pytest.raises(exc.InvalidConfig, match="schedule"):
        bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[128, 64], cute_chained_mma_schedule="tcgen05_tmem"
            )
        )


@pytest.mark.parametrize("warps", [1, 2, 8])
def test_tcgen05_chain_requires_four_warps(warps: int) -> None:
    spec = _bound().config_spec
    config = helion.Config(
        block_sizes=[128, 64], num_warps=warps, cute_chained_mma_schedule="tcgen05_tmem"
    )
    with pytest.raises(exc.InvalidConfig, match="num_warps=4"):
        spec.normalized_config(config)
    spec.normalize(config, _fix_invalid=True)
    assert config.num_warps == 4
    assert config.config["cute_chained_mma_schedule"] == "tcgen05_tmem"


def test_tcgen05_chain_explicit_unsupported_plan_cannot_fall_back() -> None:
    bound = _bound()
    config = helion.Config(
        block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
    )
    with (
        patch(
            "helion._compiler.cute.chained_matmul.plan_chained_matmul",
            return_value=None,
        ),
        pytest.raises(exc.BackendUnsupported, match="tcgen05_tmem"),
    ):
        bound.to_code(config)
