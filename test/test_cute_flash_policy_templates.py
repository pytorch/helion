from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import TypedDict
from unittest.mock import patch

import pytest
import torch

from test.test_cute_flash_policy_compat import _config_spec

from helion._compiler.cute import cute_flash
from helion._compiler.cute.flash_policy import get_flash_target_policy
from helion.autotuner.config_fragment import EnumFragment

if TYPE_CHECKING:
    from collections.abc import Mapping

    from helion.runtime.config import Config


class _SeedOverrides(TypedDict, total=False):
    dtype: torch.dtype
    has_kv_tile_pruning: bool
    requires_ws_overlap: bool
    small_biased_candidate: bool
    standard_dense_output: bool
    standard_causal_output: bool
    target_device_capability: tuple[int, int]


def _target_seeds(
    num_kv: int,
    *,
    is_causal: bool,
    dtype: torch.dtype = torch.float16,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_dense_output: bool | None = None,
    standard_causal_output: bool | None = None,
    target_device_capability: tuple[int, int] = (10, 3),
) -> tuple[Config, ...]:
    return cute_flash._flash_target_seed_configs(
        64,
        num_kv,
        dtype=dtype,
        num_bh=64,
        tensor_4d_heads=None,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_dense_output=(
            not is_causal if standard_dense_output is None else standard_dense_output
        ),
        standard_causal_output=(
            is_causal if standard_causal_output is None else standard_causal_output
        ),
        target_device_capability=target_device_capability,
        supports_tensor_4d_tma=True,
        block_size_targets=(1, 128, 128),
    )


def _overrides(*, is_causal: bool) -> tuple[Mapping[str, object], ...]:
    tuning = get_flash_target_policy((10, 3)).tuning
    return (
        tuple(map(cute_flash._flash_causal_tuning_overrides, tuning.causal_policies))
        if is_causal
        else tuple(map(cute_flash._flash_dense_tuning_overrides, tuning.dense_policies))
    )


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize("num_kv", (4, 8, 256, 512, 1024, 2048, 4096))
def test_all_legal_target_templates_survive_strict_normalization(
    num_kv: int, is_causal: bool
) -> None:
    seeds = _target_seeds(num_kv, is_causal=is_causal)
    assert len(seeds) == len(_overrides(is_causal=is_causal)) == 4
    spec = _config_spec(num_kv, is_causal=is_causal)
    normalized = [spec.normalized_config(seed) for seed in seeds]
    for expected in _overrides(is_causal=is_causal):
        assert (
            sum(
                all(seed.config[key] == value for key, value in expected.items())
                for seed in normalized
            )
            == 1
        )
    assert normalized == [spec.normalized_config(seed) for seed in normalized]


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize("num_kv", (1, 2, 3, 4, 6, 8, 97, 386))
def test_template_projection_respects_pair_and_cluster_legality(
    num_kv: int, is_causal: bool
) -> None:
    seeds = _target_seeds(num_kv, is_causal=is_causal)
    alignment = 2 if is_causal else 4
    assert len(seeds) == (4 if num_kv >= alignment and num_kv % alignment == 0 else 0)


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize(
    "overrides",
    (
        {"has_kv_tile_pruning": True},
        {"requires_ws_overlap": True},
        {"small_biased_candidate": True},
        {"standard_dense_output": False, "standard_causal_output": False},
        {"target_device_capability": (10, 0)},
        {"target_device_capability": (999, 999)},
        {"dtype": torch.bfloat16},
    ),
)
def test_templates_do_not_cross_implementation_capabilities(
    is_causal: bool, overrides: _SeedOverrides
) -> None:
    assert _target_seeds(4096, is_causal=is_causal, **overrides) == ()


@pytest.mark.parametrize("is_causal", (False, True))
def test_template_identity_ignores_historical_lengths_and_declaration_order(
    is_causal: bool,
) -> None:
    original = get_flash_target_policy((10, 3))
    if is_causal:
        causal = tuple(
            dataclasses.replace(policy, num_kv=10000 + 4 * index)
            for index, policy in enumerate(reversed(original.tuning.causal_policies))
        )
        causal += (dataclasses.replace(causal[0], num_kv=11000),)
        tuning = dataclasses.replace(original.tuning, causal_policies=causal)
    else:
        dense = tuple(
            dataclasses.replace(policy, num_kv=10000 + 4 * index)
            for index, policy in enumerate(reversed(original.tuning.dense_policies))
        )
        dense += (dataclasses.replace(dense[0], num_kv=11000),)
        tuning = dataclasses.replace(original.tuning, dense_policies=dense)
    # A duplicate at another historical length must not duplicate a compiler seed.
    altered = dataclasses.replace(original, tuning=tuning)
    before = _target_seeds(4, is_causal=is_causal)
    with patch.object(cute_flash, "get_flash_target_policy", return_value=altered):
        assert _target_seeds(4, is_causal=is_causal) == before
        assert _target_seeds(4096, is_causal=is_causal) == before


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize("num_kv", (3, 4, 6, 4096))
def test_singular_and_plural_default_seed_routes_agree(
    num_kv: int, is_causal: bool
) -> None:
    common = {
        "dtype": torch.float16,
        "num_bh": 64,
        "is_causal": is_causal,
        "standard_dense_output": not is_causal,
        "standard_causal_output": is_causal,
        "target_device_capability": (10, 3),
    }
    singular = cute_flash.flash_attention_seed_config(64, num_kv, **common)
    plural = cute_flash.flash_attention_seed_configs(64, num_kv, **common)
    assert singular == plural[0]
    assert len(plural) == len(set(plural))


def test_over_budget_template_is_not_repaired_into_a_different_seed() -> None:
    original = get_flash_target_policy((10, 3))
    over_budget = dataclasses.replace(
        original.tuning.dense_policies[0], softmax_regs=200, corr_regs=80, other_regs=40
    )
    altered = dataclasses.replace(
        original,
        tuning=dataclasses.replace(original.tuning, dense_policies=(over_budget,)),
    )
    with patch.object(cute_flash, "get_flash_target_policy", return_value=altered):
        assert _target_seeds(4, is_causal=False) == ()


def test_pinned_family_receives_only_its_legal_template_coordinates() -> None:
    common = {
        "dtype": torch.float16,
        "is_causal": True,
        "standard_causal_output": True,
        "target_device_capability": (10, 3),
    }
    for num_kv in (4, 4096):
        ws = cute_flash.flash_autotune_fragments(
            64, num_kv, pipeline_family_override="ws_overlap", **common
        )
        fa4 = cute_flash.flash_autotune_fragments(
            64, num_kv, pipeline_family_override="fa4", **common
        )
        for key, value in (
            (cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY, "16/6"),
            (cute_flash.FLASH_EXP2_PACKET_KEY, "deg2_16x6"),
            (cute_flash.FLASH_WAIT_HINT_KEY, 0),
        ):
            ws_fragment, fa4_fragment = ws[key], fa4[key]
            assert isinstance(ws_fragment, EnumFragment)
            assert isinstance(fa4_fragment, EnumFragment)
            assert value not in ws_fragment.choices
            assert value in fa4_fragment.choices
