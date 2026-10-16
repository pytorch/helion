from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from .test_cute_flash_stateful_choice import _resolve
from .test_cute_flash_stateful_choice import _spec
import helion
from helion._compiler.cute import cute_flash as flash
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        yield


def _old_seeds(spec):
    with patch.object(flash, "_flash_stateful_joint_seed_configs", return_value=()):
        return spec.autotune_seed_configs()


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("heads", (1, 8, 64))
def test_complete_seed_prefix_and_joint_surface(head_dim, dtype, heads) -> None:
    spec = _spec(head_dim, 32, dtype)
    spec._cute_flash_num_bh = heads
    old = _old_seeds(spec)
    seeds = spec.autotune_seed_configs()
    assert seeds[: len(old)] == old
    parents = [
        c
        for c in old
        if c.config.get(flash.FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful"
        and c.config.get(flash.FLASH_ROWMAX_KEY) == "tmem"
    ]
    added = seeds[len(old) :]
    assert len(added) == 2 * len(parents)
    generation = ConfigGeneration(spec)
    normalized = [generation.canonicalize_flat(generation.flatten(c))[1] for c in added]
    assert len(normalized) == len(set(normalized))
    expected_depth = 4 if head_dim == 64 else 3
    assert {c.config[flash.FLASH_KV_STAGE_KEY] for c in added} == {expected_depth}
    # The upstream register domain now extends down to 152. Joint seeds keep
    # the parent's allocation and the lowest admitted allocation.
    assert {c.config[flash.FLASH_SOFTMAX_REGS_KEY] for c in added} == {200, 152}
    for parent in parents:
        expected = dict(spec.normalized_config(parent).config)
        for registers in (200, 152):
            candidate = helion.Config.from_dict(
                {
                    **expected,
                    flash.FLASH_KV_STAGE_KEY: expected_depth,
                    flash.FLASH_SOFTMAX_REGS_KEY: registers,
                }
            )
            assert candidate in normalized
    # Both schedules and every geometry-derived mapping retain both profiles.
    for key in (flash.FLASH_ROW_SUM_SCHEDULE_KEY, flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY):
        assert {c.config.get(key, "post_acquire") for c in added} == {
            c.config.get(key, "post_acquire") for c in parents
        }


@pytest.mark.parametrize(
    "head_dim,depths,registers,expected_depth,expected_registers",
    (
        (64, (2, 3), (200, 192), 3, {200, 192}),
        (128, (2, 3, 4), (200, 184), 3, {200, 184}),
        (128, (2, 4), (200, 176), None, set()),
        (64, (2,), (200, 176), None, set()),
        (64, (2, 3), (200,), 3, {200}),
    ),
)
def test_active_domains_and_parent_storage_cap(
    head_dim, depths, registers, expected_depth, expected_registers
) -> None:
    spec = _spec(head_dim, 32, torch.float16)
    old = _old_seeds(spec)
    fragments = spec._cute_flash_autotune_fragments()
    fragments[flash.FLASH_KV_STAGE_KEY] = EnumFragment(depths, depths)
    fragments[flash.FLASH_SOFTMAX_REGS_KEY] = EnumFragment(registers, registers)
    added = flash._flash_stateful_joint_seed_configs(
        old,
        fragments,
        lambda values: _resolve(head_dim, 32, overrides=dict(values)),
    )
    assert {c.config[flash.FLASH_SOFTMAX_REGS_KEY] for c in added} == expected_registers
    assert {c.config[flash.FLASH_KV_STAGE_KEY] for c in added} == (
        set() if expected_depth is None else {expected_depth}
    )


def test_weighted_budget_rejection_and_canonical_duplicate_parents() -> None:
    spec = _spec(128, 32, torch.float16)
    old = _old_seeds(spec)
    parent = next(
        c
        for c in old
        if c.config.get(flash.FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful"
        and c.config.get(flash.FLASH_ROWMAX_KEY) == "tmem"
    )
    fragments = spec._cute_flash_autotune_fragments()

    def supplement(parents):
        return flash._flash_stateful_joint_seed_configs(
            parents, fragments, lambda values: _resolve(128, 32, overrides=dict(values))
        )

    assert supplement([parent, spec.normalized_config(parent)]) == supplement([parent])
    overbudget = helion.Config.from_dict(
        {
            **parent.config,
            flash.FLASH_CORR_REGS_KEY: 88,
            flash.FLASH_OTHER_REGS_KEY: 80,
        }
    )
    # The expanded domain can fit the 152-register endpoint under this budget.
    assert {
        c.config[flash.FLASH_SOFTMAX_REGS_KEY] for c in supplement([overbudget])
    } == {152}
    fragments[flash.FLASH_SOFTMAX_REGS_KEY] = EnumFragment((176, 200), (176, 200))
    assert supplement([overbudget]) == ()


def test_cache_identity_changes_without_layout_or_default_changes() -> None:
    spec = _spec(128, 32, torch.bfloat16)
    default = spec.default_config()
    layout = spec.structural_fingerprint_hash()
    old = _old_seeds(spec)
    spec.compiler_seed_configs = old
    old_identity = spec.cache_fingerprint_hash()
    spec.compiler_seed_configs = spec.autotune_seed_configs()
    assert spec.compiler_seed_configs[: len(old)] == old
    assert spec.cache_fingerprint_hash() != old_identity
    assert spec.structural_fingerprint_hash() == layout
    assert spec.default_config() == default


def test_detector_proof_remains_required() -> None:
    spec = _spec(128, 32, torch.float16, proof=False)
    assert spec.autotune_seed_configs() == _old_seeds(spec)
