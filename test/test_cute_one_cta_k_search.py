"""A proved native family may replace only its own K128 search ceiling.

Search-range growth also retains the registered K default in ordinary seeds.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import blackwell_capability
from test._cute_dense_tcgen05 import cpu_codegen_target
from test._cute_dense_tcgen05 import scaled_args
from test._cute_dense_tcgen05 import scaled_mm
from test._cute_dense_tcgen05 import user_bound_mm
import test.test_cute_backend as backend_tests

from helion._compiler.autotuner_heuristics.cute import CuteTcgen05ClusterM2FfiHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteTcgen05ClusterM2Heuristic
from helion._compiler.compile_environment import FixedBlockSizeSource
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    with cpu_codegen_target(), _forbid_cuda_work():
        yield


def _before_widening() -> BoundKernel:
    with patch.object(
        CuteTcgen05Config, "widen_one_cta_pipeline_k_search", return_value=False
    ):
        bound = scaled_mm._bind_isolated(scaled_args())
    state = bound.config_spec._cute_tcgen05_config
    cap = state._native_k_search_cap
    assert cap is not None
    block = bound.config_spec.block_sizes.block_id_lookup(cap[0])
    assert block.max_size == 128 and block.max_size_revision == cap[1]
    assert block.dim_max_size >= 256
    return bound


@pytest.mark.parametrize("mode", range(4))
def test_owned_cap_widens_and_bk256_seeds_round_trip(mode: int) -> None:
    bound = scaled_mm._bind_isolated(scaled_args(mode=mode))
    state = bound.config_spec._cute_tcgen05_config
    assert state._matmul_block_fragments()[2].high == 256
    seeds = state.one_cta_pipeline_seed_configs()
    seed = next(seed for seed in seeds if seed.block_sizes == [64, 32, 256])
    assert seed.config["tcgen05_ab_stages"] == 9
    with bound.env:
        gen = ConfigGeneration(bound.config_spec)
        flat, normalized = gen.canonicalize_flat(gen.flatten(seed))
        flat2, normalized2 = gen.canonicalize_flat(flat)
        assert normalized.block_sizes == seed.block_sizes
        assert normalized.config["tcgen05_ab_stages"] == 9
        assert flat == flat2 and normalized == normalized2
    source = bound.to_code(normalized)
    assert "CtaGroup.ONE" in source and "num_stages=9" in source


def test_owned_cap_preserves_default_and_is_idempotent() -> None:
    bound = _before_widening()
    state = bound.config_spec._cute_tcgen05_config
    before = bound.config_spec._base_default_config()
    assert state.widen_one_cta_pipeline_k_search(bound.env)
    assert not state.widen_one_cta_pipeline_k_search(bound.env)
    assert bound.config_spec._base_default_config() == before


def test_effective_clamped_default_stays_fixed() -> None:
    bound = _before_widening()
    state = bound.config_spec._cute_tcgen05_config
    cap = state._native_k_search_cap
    assert cap is not None
    block = bound.config_spec.block_sizes.block_id_lookup(cap[0])
    block.default_size = 256
    old_default = block._fragment(bound.config_spec).default()
    assert old_default == 128
    assert state.widen_one_cta_pipeline_k_search(bound.env)
    assert block._fragment(bound.config_spec).default() == old_default


def test_replacement_block_cannot_inherit_matching_revision() -> None:
    bound = _before_widening()
    state = bound.config_spec._cute_tcgen05_config
    cap = state._native_k_search_cap
    assert cap is not None
    index = bound.config_spec.block_sizes.block_id_to_index(cap[0])
    original = bound.config_spec.block_sizes[index]
    replacement = BlockSizeSpec(
        block_id=cap[0], size_hint=4096, min_size=original.min_size
    )
    replacement.update_max(128)
    assert replacement.max_size_revision == cap[1]
    bound.config_spec.block_sizes[index] = replacement
    assert not state.widen_one_cta_pipeline_k_search(bound.env)
    assert replacement.max_size == 128


def test_cap_aliasing_survives_bound_spec_deepcopy() -> None:
    bound = _before_widening()
    original = bound.config_spec
    replica = copy.deepcopy(original)
    state = replica._cute_tcgen05_config
    cap = state._native_k_search_cap
    assert cap is not None
    block = replica.block_sizes.block_id_lookup(cap[0])
    assert cap[2] is block
    assert block is not original.block_sizes.block_id_lookup(cap[0])
    assert replica.cache_fingerprint_hash() == original.cache_fingerprint_hash()
    assert state.widen_one_cta_pipeline_k_search(bound.env)
    assert block.max_size == 256
    assert original.block_sizes.block_id_lookup(cap[0]).max_size == 128


def test_internal_ownership_does_not_enter_persistent_cache_key() -> None:
    args = scaled_args()
    first = scaled_mm._bind_isolated(args)
    second = scaled_mm._bind_isolated(args)
    state = first.config_spec._cute_tcgen05_config
    assert (
        state._native_k_search_cap
        is not second.config_spec._cute_tcgen05_config._native_k_search_cap
    )
    expected = first.config_spec.cache_fingerprint_hash()
    assert second.config_spec.cache_fingerprint_hash() == expected
    state._native_k_search_cap = None
    assert first.config_spec.cache_fingerprint_hash() == expected
    assert first.extra_cache_key() == second.extra_cache_key()


@pytest.mark.parametrize("maximum", (64, 128, 256))
def test_explicit_user_maximum_survives(maximum: int) -> None:
    bound = user_bound_mm._bind_isolated((*scaled_args()[:2], maximum))
    state = bound.config_spec._cute_tcgen05_config
    fragments = state._matmul_block_fragments()
    assert fragments is not None
    assert fragments[2].high <= maximum
    if maximum <= 128:
        assert state._native_k_search_cap is None


@pytest.mark.parametrize("mutation", ("assign128", "update128", "update64", "hint"))
def test_intervening_constraint_invalidates_cap_ownership(mutation: str) -> None:
    bound = _before_widening()
    state = bound.config_spec._cute_tcgen05_config
    cap = state._native_k_search_cap
    assert cap is not None
    block = bound.config_spec.block_sizes.block_id_lookup(cap[0])
    if mutation == "assign128":
        block.max_size = 128
    elif mutation == "update128":
        block.update_max(128)
    elif mutation == "update64":
        block.update_max(64)
    else:
        block.update_hint(4096)
    assert block.max_size_revision != cap[1]
    assert not state.widen_one_cta_pipeline_k_search(bound.env)
    assert block.max_size <= 128


@pytest.mark.parametrize("reason", ("fixed", "bounded", "pid", "proof", "capacity"))
def test_independent_vetoes_survive(reason: str) -> None:
    bound = _before_widening()
    state = bound.config_spec._cute_tcgen05_config
    cap = state._native_k_search_cap
    assert cap is not None
    block = bound.config_spec.block_sizes.block_id_lookup(cap[0])
    if reason == "fixed":
        bound.env.block_sizes[cap[0]].block_size_source = FixedBlockSizeSource(128)
    elif reason == "bounded":
        block.bounded_by_block_id = 0
    elif reason == "pid":
        for pid in ("persistent_interleaved", "persistent_blocked"):
            bound.config_spec.disallow_pid_type(pid, reason="independent consumer")
    elif reason == "proof":
        state.one_cta_pipeline_facts = None
    else:
        facts = state.one_cta_pipeline_facts
        assert facts is not None
        state.one_cta_pipeline_facts = replace(
            facts, allocations=facts.allocations._replace(capacity_bytes=1024)
        )
    assert not state.widen_one_cta_pipeline_k_search(bound.env)
    assert block.max_size == 128


def test_k_tail_keeps_global_k128_cap() -> None:
    bound = scaled_mm._bind_isolated(scaled_args(k=2880))
    assert (
        bound.config_spec._cute_tcgen05_config._matmul_block_fragments()[2].high == 128
    )


def test_max_size_revision_preserves_existing_clamping() -> None:
    block = BlockSizeSpec(block_id=0, size_hint=4096)
    assert block.max_size == 4096
    token = block.max_size_revision
    block.update_max(128)
    assert block.max_size == 128 and block.max_size_revision > token
    token = block.max_size_revision
    block.update_max(256)
    assert block.max_size == 128 and block.max_size_revision > token
    token = block.max_size_revision
    block.max_size = 128
    assert block.max_size_revision > token


# Ordinary cluster_m=2 seeds keep the registered K default.


def _bind() -> BoundKernel:
    return scaled_mm._bind_isolated(scaled_args(m=2048, n=2048, mode=1))


def _k_block(bound: BoundKernel) -> BlockSizeSpec:
    indices = bound.config_spec._cute_tcgen05_config._matmul_block_indices()
    assert indices is not None
    return bound.config_spec.block_sizes[indices[2]]


@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
def test_existing_deep_seed_regression_under_cpu_target(
    capability: tuple[int, int],
) -> None:
    with (
        patch.object(backend_tests, "DEVICE", torch.device("cpu")),
        blackwell_capability(capability),
    ):
        backend_tests.TestCuteBackend(
            "test_fp8_matmul_tcgen05_deep_two_cta_seeds_fill_the_budget"
        ).debug()


@pytest.mark.parametrize("default", (32, 64, 128))
def test_registered_default_is_additive_and_does_not_mutate_bounds(
    default: int,
) -> None:
    bound = _bind()
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    block = _k_block(bound)
    block.default_size = default
    before = block.max_size, block.max_size_revision, state._native_k_search_cap
    expected = tuple(dict.fromkeys((256, 128, default)))
    assert state.cluster_m2_seed_bk_choices() == expected
    assert state.cluster_m2_seed_bk_choices() == expected
    assert (
        block.max_size,
        block.max_size_revision,
        state._native_k_search_cap,
    ) == before


@pytest.mark.parametrize("default", (None, 0, 48, 256, 512))
def test_unregistered_duplicate_or_unreachable_default_adds_no_seed(
    default: int | None,
) -> None:
    bound = _bind()
    state = bound.config_spec._cute_tcgen05_config
    state._native_k_widened_block = None
    block = _k_block(bound)
    block.default_size = default
    assert state.cluster_m2_seed_bk_choices() == (256,)


@pytest.mark.parametrize("constraint", ("lower", "upper", "bounded", "k_loop"))
def test_default_cannot_bypass_live_constraints(constraint: str) -> None:
    bound = _bind()
    state = bound.config_spec._cute_tcgen05_config
    state._native_k_widened_block = None
    block = _k_block(bound)
    block.default_size = 64
    if constraint == "lower":
        block.autotuner_min = 128
    elif constraint == "upper":
        block.max_size = 32
    elif constraint == "bounded":
        assert state.matmul_block_ids is not None
        block.bounded_by_block_id = state.matmul_block_ids[0]
    else:
        state.allow_cluster_m2_search(static_k=32768, max_k_tiles=256)
    choices = state.cluster_m2_seed_bk_choices()
    assert 64 not in choices
    assert choices == ((32,) if constraint == "upper" else (256,))


def test_ordinary_multi_seed_primary_default_and_search_roundtrip() -> None:
    bound = _bind()
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.cluster_m2_seed_bk_choices() == (256, 128, 32)
    assert CuteTcgen05ClusterM2Heuristic.is_eligible(
        bound.env, bound.host_function.device_ir
    )
    with bound.env:
        seeds = CuteTcgen05ClusterM2Heuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert seeds is not None
        assert seeds[0] == CuteTcgen05ClusterM2Heuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        )
        assert [seed.block_sizes[2] for seed in seeds] == [256, 128, 32]
        assert all(seed in spec.compiler_seed_configs for seed in seeds)
        generator = ConfigGeneration(spec)
        for seed in seeds:
            flat, canonical = generator.canonicalize_flat(generator.flatten(seed))
            again, recanonical = generator.canonicalize_flat(flat)
            assert flat == again and canonical == recanonical
            assert canonical.block_sizes == seed.block_sizes
            assert canonical.config["tcgen05_cluster_m"] == 2


def test_restored_row_scale_anchor_keeps_pre_wait_through_codegen() -> None:
    args = scaled_args(m=2048, n=2048, mode=1)
    bound = backend_tests.cute_matmul_mma_fp8_rowvec_scale._bind_isolated(
        (args[0], args[1], args[3])
    )
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state._native_k_widened_block is _k_block(bound)
    assert _k_block(bound).default_size == 32
    seed = next(
        seed
        for seed in state.autotune_seed_configs()
        if seed.block_sizes == [256, 256, 128]
        and seed.config.get("tcgen05_cluster_n") == 2
        and seed.config.get("tcgen05_aux_load_placement") == "pre_acc_wait"
    )
    with bound.env:
        generator = ConfigGeneration(spec)
        flat, canonical = generator.canonicalize_flat(generator.flatten(seed))
        assert generator.canonicalize_flat(flat) == (flat, canonical)
    assert canonical.block_sizes == seed.block_sizes
    assert canonical.config["tcgen05_ab_stages"] == 3
    assert canonical.config["tcgen05_aux_load_placement"] == "pre_acc_wait"
    assert canonical.config["tcgen05_cluster_n"] == 2
    source = bound.to_code(canonical)
    assert "PipelineTmaUmma.create" in source
    assert "PipelineTmaStore.create" in source


def test_clc_twins_keep_high_anchor_and_default_geometry() -> None:
    bound = backend_tests.cute_matmul_mma_fp8._bind_isolated(
        scaled_args(m=2048, n=2048)[:2]
    )
    state = bound.config_spec._cute_tcgen05_config
    assert state.cluster_m2_seed_bk_choices() == (256, 128, 32)
    seeds = [
        seed
        for seed in state.autotune_seed_configs()
        if seed.config.get("tcgen05_persistence_model") == "clc_persistent"
        and seed.block_sizes[2] in (256, 128, 32)
    ]
    assert {
        (seed.block_sizes[2], seed.config.get("tcgen05_cluster_n", 1)) for seed in seeds
    } == {(bk, cn) for bk in (256, 128, 32) for cn in (1, 2)}
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        for seed in seeds:
            _, canonical = generator.canonicalize_flat(generator.flatten(seed))
            assert canonical.block_sizes == seed.block_sizes
            cn = seed.config.get("tcgen05_cluster_n", 1)
            # Existing full-tile CLC validation keeps only cluster_n=1;
            # the multicast twin retains its static-persistent projection.
            assert canonical.config["tcgen05_persistence_model"] == (
                "clc_persistent" if cn == 1 else "static_persistent"
            )
            assert canonical.config.get("tcgen05_cluster_n", 1) == cn
    anchor = next(
        seed
        for seed in seeds
        if seed.block_sizes[2] == 128 and seed.config.get("tcgen05_cluster_n", 1) == 1
    )
    source = bound.to_code(anchor)
    assert "PipelineTmaUmma.create" in source
    assert "PipelineTmaStore.create" in source


def test_widened_k_block_is_cache_key_neutral_and_default_config_unchanged() -> None:
    bound = _bind()
    state = bound.config_spec._cute_tcgen05_config
    expected_key = bound.config_spec.cache_fingerprint_hash()
    expected_default = bound.config_spec.default_config()
    state._native_k_widened_block = None
    assert bound.config_spec.cache_fingerprint_hash() == expected_key
    assert bound.config_spec.default_config() == expected_default


@pytest.mark.parametrize("changed", ("equal_cap", "replacement"))
def test_failed_owned_widening_does_not_record_widened_k_block(changed: str) -> None:
    with patch.object(
        CuteTcgen05Config, "widen_one_cta_pipeline_k_search", return_value=False
    ):
        bound = _bind()
    state = bound.config_spec._cute_tcgen05_config
    indices = state._matmul_block_indices()
    assert indices is not None
    block = bound.config_spec.block_sizes[indices[2]]
    assert state._native_k_widened_block is None
    if changed == "equal_cap":
        block.max_size = block.max_size
    else:
        bound.config_spec.block_sizes[indices[2]] = copy.copy(block)
    assert not state.widen_one_cta_pipeline_k_search(bound.env)
    assert state._native_k_widened_block is None


@pytest.mark.parametrize("maximum", (32, 64, 128))
def test_user_bound_does_not_record_widened_k_block(maximum: int) -> None:
    bound = user_bound_mm._bind_isolated((*scaled_args(m=2048, n=2048)[:2], maximum))
    state = bound.config_spec._cute_tcgen05_config
    assert state._native_k_widened_block is None
    assert all(bk <= maximum for bk in state.cluster_m2_seed_bk_choices())


def test_widened_k_block_requires_same_block_and_survives_deepcopy() -> None:
    bound = _bind()
    spec = copy.deepcopy(bound.config_spec)
    state = spec._cute_tcgen05_config
    indices = state._matmul_block_indices()
    assert indices is not None
    block = spec.block_sizes[indices[2]]
    assert state._native_k_widened_block is block
    assert state.cluster_m2_seed_bk_choices() == (256, 128, 32)
    spec.block_sizes[indices[2]] = copy.copy(block)
    assert state.cluster_m2_seed_bk_choices() == (256, 32)


def test_ffi_keeps_its_independent_single_seed_hook() -> None:
    bound = _bind()
    assert (
        CuteTcgen05ClusterM2FfiHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        is None
    )
