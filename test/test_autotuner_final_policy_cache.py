from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from helion._testing import TestCase
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_cache import ShapeKey
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
from helion.autotuner.effort_profile import FinalSelectionConfig
from helion.autotuner.local_cache import _search_policy_hash
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only


class TestFinalPolicyCache(TestCase):
    def _search(
        self, banks: int = 1, effort: str = "full"
    ) -> DifferentialEvolutionSearch:
        search = object.__new__(DifferentialEvolutionSearch)
        search._benchmark_provider_cls = LocalBenchmarkProvider
        search.settings = Settings(
            autotune_benchmark_input_banks=banks, autotune_effort=effort
        )
        search.final_selection = search._final_selection_from_settings(search.settings)
        return search

    def _hash(self, search: DifferentialEvolutionSearch) -> str:
        return _search_policy_hash(search, cute_flash_search_enabled=False)

    def test_clone_version_invalidates_old_bank_and_regime_keys(self) -> None:
        keys = set()
        for banks, regime in (
            (1, "replay"),
            (2, "replay"),
            (3, "replay"),
            (2, "0"),
            (3, "0"),
        ):
            with (
                self.subTest(banks=banks, regime=regime),
                patch.dict(
                    os.environ,
                    {
                        "HELION_AUTOTUNE_L2_FLUSH": regime,
                        "HELION_AUTOTUNE_FINAL_L2_FLUSH": regime,
                    },
                ),
            ):
                search = self._search(banks)
                if banks == 1:
                    expected = ""
                elif regime == "replay":
                    expected = hashlib.sha256(
                        f"helion.benchmark_input_banks.v1:{banks}".encode()
                    ).hexdigest()
                else:
                    expected = hashlib.sha256(
                        json.dumps(
                            {
                                "input_banks": banks,
                                "search_l2_flush_mode": "off",
                                "final_l2_flush_mode": "off",
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                current = self._hash(search)
                self.assertNotEqual(current, expected)
                self.assertEqual(self._hash(self._search(banks, "quick")), current)
                keys.add(current)
        self.assertEqual(len(keys), 5)

    def test_aot_and_ordinary_search_share_clone_version(self) -> None:
        for banks in (1, 3):
            search = self._search(banks)
            cache = object.__new__(AOTAutotuneCache)
            cache.autotuner = search
            cache.args = ()
            cache.hardware_id = "hw"
            cache.kernel = SimpleNamespace(
                config_spec=SimpleNamespace(cute_flash_search_enabled=False),
                kernel=SimpleNamespace(
                    name="kernel", specialization_key=lambda args: ("shape",)
                ),
            )
            key = cache._create_shape_key()
            self.assertEqual(key.search_policy_hash, self._hash(search))
            legacy_policy = (
                ""
                if banks == 1
                else hashlib.sha256(b"helion.benchmark_input_banks.v1:3").hexdigest()
            )
            legacy = ShapeKey("kernel", ("shape",), "hw", legacy_policy)
            self.assertNotEqual(key.stable_hash(), legacy.stable_hash())
            restored = ShapeKey.from_dict(json.loads(json.dumps(key.to_dict())))
            self.assertEqual(restored.stable_hash(), key.stable_hash())

    def test_each_overridden_duration_changes_key_and_restoring_policy_restores_key(
        self,
    ) -> None:
        with patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS": "4"}):
            search = self._search(3)
            policy = search.final_selection
            assert policy is not None
            default = self._hash(search)
            keys = set()
            for name, value in (
                ("thermal_warmup_ms", 0),
                ("screen_warmup_ms", 0),
                ("screen_rep_ms", 1),
                ("warmup_ms", 0),
                ("rep_ms", 1),
            ):
                search.final_selection = dataclasses.replace(policy, **{name: value})
                key = self._hash(search)
                self.assertNotEqual(key, default)
                self.assertEqual(self._hash(search), key)
                keys.add(key)
            self.assertEqual(len(keys), 5)
            search.final_selection = policy
            self.assertEqual(self._hash(search), default)

    def test_explicit_enable_and_disable_are_distinct(self) -> None:
        with patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS": "4"}):
            full = self._search()
            default = self._hash(full)
            full.final_selection = None
            self.assertNotEqual(self._hash(full), default)
            quick = self._search(effort="quick")
            self.assertEqual(self._hash(quick), default)
            quick.final_selection = FinalSelectionConfig(rep_ms=1)
            self.assertNotEqual(self._hash(quick), default)
            self.assertNotEqual(self._hash(quick), self._hash(full))

    def test_override_keeps_bank_and_resident_regime_separation(self) -> None:
        keys = set()
        for banks, search_regime, final_regime in (
            (1, "replay", "replay"),
            (2, "replay", "replay"),
            (3, "replay", "replay"),
            (3, "0", "0"),
            (3, "0", "replay"),
        ):
            with patch.dict(
                os.environ,
                {
                    "HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS": "4",
                    "HELION_AUTOTUNE_L2_FLUSH": search_regime,
                    "HELION_AUTOTUNE_FINAL_L2_FLUSH": final_regime,
                },
            ):
                search = self._search(banks)
                assert search.final_selection is not None
                search.final_selection = dataclasses.replace(
                    search.final_selection, rep_ms=1
                )
                keys.add(self._hash(search))
        self.assertEqual(len(keys), 5)

    def test_flash_path_still_uses_complete_search_policy(self) -> None:
        search = self._search()
        serialized = {"complete_search_policy": 7}
        with patch.object(
            search, "cache_policy", return_value=serialized
        ) as cache_policy:
            first = _search_policy_hash(search, cute_flash_search_enabled=True)
            search.final_selection = None
            self.assertEqual(
                _search_policy_hash(search, cute_flash_search_enabled=True),
                first,
            )
            self.assertEqual(cache_policy.call_count, 2)
