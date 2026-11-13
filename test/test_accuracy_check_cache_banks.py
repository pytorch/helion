from __future__ import annotations

import hashlib

import pytest

from helion.autotuner.accuracy import AccuracyCheck
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.effort_profile import FinalSelectionConfig
from helion.autotuner.local_cache import _search_policy_hash
from helion.autotuner.pattern_search import PatternSearch
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only


def _search(
    check: AccuracyCheck | None,
    *,
    banks: int,
    enabled: bool = True,
    override_final_selection: bool = False,
) -> PatternSearch:
    search = object.__new__(PatternSearch)
    search.settings = Settings(
        autotune_benchmark_input_banks=banks,
        autotune_accuracy_check=enabled,
        autotune_baseline_accuracy_check_fn=check,
    )
    search._benchmark_provider_cls = LocalBenchmarkProvider
    search.final_selection = search._final_selection_from_settings(search.settings)
    if override_final_selection:
        explicit = FinalSelectionConfig(rounds=6)
        assert explicit != search.final_selection
        search.final_selection = explicit
    return search


@pytest.mark.parametrize(
    "banks,search_mode,final_mode,override_final_selection",
    [
        (1, "replay", "replay", False),
        (3, "replay", "replay", False),
        (3, "off", "replay", False),
        (3, "replay", "off", False),
        (1, "replay", "replay", True),
        (3, "off", "off", True),
    ],
)
def test_accuracy_policy_composes_with_bank_and_finalist_keys(
    monkeypatch, banks, search_mode, final_mode, override_final_selection
):
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", search_mode)
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", final_mode)
    monkeypatch.delenv("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS", raising=False)

    def key(check, *, enabled=True, bank_count=banks):
        search = _search(
            check,
            banks=bank_count,
            enabled=enabled,
            override_final_selection=override_final_selection,
        )
        return _search_policy_hash(search, cute_flash_search_enabled=False)

    strict = AccuracyCheck(atol=0.01, rtol=0.0)
    loose = AccuracyCheck(atol=0.1, rtol=0.0)
    strict_key = key(strict)
    assert strict_key == key(AccuracyCheck(atol=0.01, rtol=0.0))
    assert strict_key != key(loose)
    assert strict_key != key(None)
    assert strict_key != key(strict, bank_count=3 if banks == 1 else 1)
    assert key(strict, enabled=False) == key(None)
    assert key(loose, enabled=False) == key(None)


def test_accuracy_check_preserves_clone_and_timing_partitions(monkeypatch):
    monkeypatch.delenv("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS", raising=False)
    check = AccuracyCheck(atol=0.01, rtol=0.0)
    search = _search(check, banks=3)
    keys = []
    for search_mode, final_mode in (
        ("replay", "replay"),
        ("off", "replay"),
        ("replay", "off"),
        ("off", "off"),
    ):
        monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", search_mode)
        monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", final_mode)
        keys.append(_search_policy_hash(search, cute_flash_search_enabled=False))
    assert len(set(keys)) == 4
    search.final_selection = FinalSelectionConfig(rounds=6)
    assert _search_policy_hash(search, cute_flash_search_enabled=False) not in keys

    # The clone version protects one-bank searches even without AccuracyCheck.
    # Disabling a configured check reaches that same current policy.
    baseline = _search(None, banks=1)
    expected = hashlib.sha256(b"helion.input_clone.v6\0").hexdigest()
    assert _search_policy_hash(baseline, cute_flash_search_enabled=False) == expected
    baseline.settings = baseline.settings.copy(
        autotune_accuracy_check=False,
        autotune_baseline_accuracy_check_fn=check,
    )
    assert _search_policy_hash(baseline, cute_flash_search_enabled=False) == expected

    baseline.settings = baseline.settings.copy(
        autotune_accuracy_check=True,
        autotune_baseline_accuracy_check_fn=lambda actual, reference: None,
    )
    assert _search_policy_hash(baseline, cute_flash_search_enabled=False) == expected
