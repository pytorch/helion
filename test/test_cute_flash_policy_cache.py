from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from helion._compiler.cute import flash_policy
from helion._compiler.cute.flash_tuning import FlashCausalTuningPolicy
from helion._compiler.cute.flash_tuning import FlashDenseTuningPolicy
from helion._compiler.cute.flash_tuning import FlashTuningDType
from helion._compiler.cute.flash_tuning import FlashTuningPolicy
from helion._compiler.cute.flash_tuning import FlashTuningWorkload

if TYPE_CHECKING:
    from helion._compiler.cute.flash_policy import FlashTargetPolicy


def _with_templates(
    target: FlashTargetPolicy,
    templates: tuple[FlashCausalTuningPolicy | FlashDenseTuningPolicy, ...],
    *,
    is_causal: bool,
) -> FlashTargetPolicy:
    if is_causal:
        causal = []
        for template in templates:
            assert isinstance(template, FlashCausalTuningPolicy)
            causal.append(template)
        tuning = dataclasses.replace(target.tuning, causal_policies=tuple(causal))
    else:
        dense = []
        for template in templates:
            assert isinstance(template, FlashDenseTuningPolicy)
            dense.append(template)
        tuning = dataclasses.replace(target.tuning, dense_policies=tuple(dense))
    return dataclasses.replace(target, tuning=tuning)


def _identity(
    target: FlashTargetPolicy, *, is_causal: bool, num_kv: int = 768
) -> object:
    with patch.object(flash_policy, "get_flash_target_policy", return_value=target):
        return flash_policy.flash_target_policy_cache_identity(
            (10, 3),
            head_dim=64,
            torch_dtype="float16",
            num_kv=num_kv,
            is_causal=is_causal,
        )


@pytest.mark.parametrize("is_causal", (False, True))
def test_nonmatching_template_edits_invalidate_search_policy(
    is_causal: bool,
) -> None:
    target = flash_policy.get_flash_target_policy((10, 3))
    templates = (
        target.tuning.causal_policies if is_causal else target.tuning.dense_policies
    )
    changed_template = dataclasses.replace(
        templates[0], first_load_order=(templates[0].first_load_order or 0) + 1
    )
    changed = _with_templates(
        target, (changed_template, *templates[1:]), is_causal=is_causal
    )
    # 768 has no exact lowering policy. Its search still uses these templates.
    assert _identity(target, is_causal=is_causal) != _identity(
        changed, is_causal=is_causal
    )
    # An existing exact codegen policy does not hide other templates' changes.
    assert _identity(target, is_causal=is_causal, num_kv=templates[-1].num_kv) != (
        _identity(changed, is_causal=is_causal, num_kv=templates[-1].num_kv)
    )


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize("change", ("reorder", "rekey", "duplicate"))
def test_search_identity_ignores_historical_keys_order_and_duplicates(
    is_causal: bool, change: str
) -> None:
    target = flash_policy.get_flash_target_policy((10, 3))
    templates = (
        target.tuning.causal_policies if is_causal else target.tuning.dense_policies
    )
    if change == "reorder":
        changed_templates = tuple(reversed(templates))
    elif change == "rekey":
        changed_templates = tuple(
            dataclasses.replace(template, num_kv=template.num_kv + 10_000)
            for template in templates
        )
    else:
        changed_templates = (
            *templates,
            dataclasses.replace(templates[0], num_kv=12_345),
        )
    changed = _with_templates(target, changed_templates, is_causal=is_causal)
    assert _identity(target, is_causal=is_causal) == _identity(
        changed, is_causal=is_causal
    )


@pytest.mark.parametrize("is_causal", (False, True))
def test_exact_codegen_policy_selection_remains_in_cache_identity(
    is_causal: bool,
) -> None:
    target = flash_policy.get_flash_target_policy((10, 3))
    templates = (
        target.tuning.causal_policies if is_causal else target.tuning.dense_policies
    )
    changed_templates = (
        dataclasses.replace(templates[0], num_kv=12_345),
        *templates[1:],
    )
    changed = _with_templates(target, changed_templates, is_causal=is_causal)
    # The template union is identical, but the old key loses its exact emitter
    # policy. That remains a codegen-semantic change until lowerings are explicit.
    assert _identity(target, is_causal=is_causal, num_kv=templates[0].num_kv) != (
        _identity(changed, is_causal=is_causal, num_kv=templates[0].num_kv)
    )


@pytest.mark.parametrize("is_causal", (False, True))
def test_other_attention_mode_does_not_change_template_identity(
    is_causal: bool,
) -> None:
    target = flash_policy.get_flash_target_policy((10, 3))
    templates = (
        target.tuning.dense_policies if is_causal else target.tuning.causal_policies
    )
    changed = _with_templates(
        target,
        (
            dataclasses.replace(templates[0], first_load_order=3),
            *templates[1:],
        ),
        is_causal=not is_causal,
    )
    assert _identity(target, is_causal=is_causal) == _identity(
        changed, is_causal=is_causal
    )


@pytest.mark.parametrize("is_causal", (False, True))
def test_other_workload_does_not_change_template_identity(is_causal: bool) -> None:
    target = flash_policy.get_flash_target_policy((10, 3))
    additional = FlashTuningPolicy(
        workload=FlashTuningWorkload(head_dim=128, dtype=FlashTuningDType.BFLOAT16),
        dense_policies=(
            FlashDenseTuningPolicy(
                num_kv=768,
                exp2_packet="1x1",
                e2e_schedule="8/2",
                e2e_offset=0,
                e2e_offset0=0,
                stat_transport="ring2",
                kv_stage=3,
            ),
        ),
    )
    changed = dataclasses.replace(target, additional_tunings=(additional,))
    assert _identity(target, is_causal=is_causal) == _identity(
        changed, is_causal=is_causal
    )
