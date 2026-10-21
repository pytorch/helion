from __future__ import annotations

import logging
import math
import os
from types import SimpleNamespace
from typing import Callable
import unittest
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch

import torch

from test._autotuner_test_utils import AutotunerTestCase
from test._autotuner_test_utils import _cute_flash_test_config_spec
from test._autotuner_test_utils import clean_final_rebenchmark_env
from test._autotuner_test_utils import without_env_var

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion.autotuner import LFBOPatternSearch
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkWorkerUnkillable
from helion.autotuner.benchmarking import MirroredBenchmarkTrace
from helion.autotuner.benchmarking import _mirrored_bench_call_layout
from helion.autotuner.effort_profile import AutotuneEffortProfile
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
import helion.language as hl
from helion.runtime.settings import Settings


@onlyBackends(["triton", "cute"])
class TestAutotunerRebenchmark(AutotunerTestCase):
    def test_autotune_benchmark_fn(self) -> None:
        """Test that custom benchmark function is used during rebenchmarking."""
        # Track benchmark function calls
        benchmark_calls: list[tuple[int, int]] = []  # (num_fns, repeat)

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            benchmark_calls.append((len(fns), repeat))
            # Return fake timings
            return [1.0] * len(fns)

        @helion.kernel(
            autotune_benchmark_fn=custom_benchmark_fn,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )

        bound_kernel = add.bind(args)
        # Use PatternSearch which has rebenchmark method
        search = PatternSearch(bound_kernel, args)

        # Compile two configs
        config1 = search.config_gen.random_config()
        config2 = search.config_gen.random_config()
        fn1 = bound_kernel.compile_config(config1)
        fn2 = bound_kernel.compile_config(config2)

        # Create population members (flat_values not used in rebenchmark)
        member1 = PopulationMember(fn1, [1.0], (), config1)
        member2 = PopulationMember(fn2, [1.1], (), config2)

        search._prepare()
        search.best_perf_so_far = 1.0

        # Call rebenchmark directly
        search.rebenchmark([member1, member2])

        # Verify custom benchmark function was called
        self.assertGreater(
            len(benchmark_calls), 0, "Custom benchmark function should be called"
        )
        # Should have been called with 2 functions
        self.assertEqual(benchmark_calls[0][0], 2)

    def test_rebenchmark_clears_jit_fast_path_caches(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.config_spec = SimpleNamespace(backend=None)
        events: list[str] = []

        class FakeJITFunction:
            def __init__(self, name: str) -> None:
                self.name = name

            def clear_fast_path_caches(self) -> None:
                events.append(f"clear {self.name}")

        def generated_kernel_a() -> None:
            events.append("run a")

        def generated_kernel_b() -> None:
            events.append("run b")

        globals_key_a = f"_helion_{generated_kernel_a.__name__}"
        globals_key_b = f"_helion_{generated_kernel_b.__name__}"
        generated_kernel_a.__globals__[globals_key_a] = FakeJITFunction("a")
        generated_kernel_b.__globals__[globals_key_b] = FakeJITFunction("b")

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            for _ in range(2):
                for fn in fns:
                    fn()
            return [100.0, 101.0]

        search.settings.autotune_benchmark_fn = custom_benchmark_fn
        member_a = PopulationMember(generated_kernel_a, [100.0], [], helion.Config())
        member_b = PopulationMember(generated_kernel_b, [101.0], [], helion.Config())
        try:
            search.rebenchmark([member_a, member_b])
        finally:
            del generated_kernel_a.__globals__[globals_key_a]
            del generated_kernel_b.__globals__[globals_key_b]

        self.assertEqual(
            events,
            [
                "run a",
                "clear a",
                "run b",
                "clear b",
                "run a",
                "clear a",
                "run b",
                "clear b",
                "clear a",
                "clear b",
            ],
        )

    def test_rebenchmark_bounds_interleaved_wall_clock(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        bench_calls: list[dict[str, object]] = []

        def fake_interleaved_bench(
            fns: list[Callable[[], object]],
            *,
            repeat: int,
            desc: str | None = None,
            max_total_ms: float | None = None,
        ) -> list[float]:
            bench_calls.append({"repeat": repeat, "max_total_ms": max_total_ms})
            return [100.0] * len(fns)

        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                get_interleaved_bench=lambda: fake_interleaved_bench
            )
        )
        member_a = PopulationMember(lambda: None, [100.0], [], helion.Config())
        member_b = PopulationMember(lambda: None, [101.0], [], helion.Config())
        search.rebenchmark(
            [member_a, member_b],
            target_ms=5000.0,
            use_isolated=False,
            confirm_suspicious=False,
            use_interleaved=True,
        )

        self.assertEqual(len(bench_calls), 1)
        # The wall-clock budget preserves the sequential-window budget the
        # interleaved shootout replaced: target_ms per candidate.
        self.assertEqual(bench_calls[0]["max_total_ms"], 10000.0)

    def test_final_rebenchmark_can_restore_earlier_stable_best(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0

        stable_config = helion.Config(num_warps=4)
        noisy_config = helion.Config(num_warps=8)
        stable = PopulationMember(lambda: None, [10.0], (), stable_config, status="ok")
        noisy = PopulationMember(lambda: None, [9.0], (), noisy_config, status="ok")
        search._benchmarked_members = {
            stable_config: stable,
            noisy_config: noisy,
        }
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [noisy]

        def fake_rebenchmark(
            members: list[PopulationMember],
            *,
            desc: str = "Rebenchmarking",
            target_ms: float = 200.0,
            use_isolated: bool = True,
            confirm_suspicious: bool = True,
            use_interleaved: bool = True,
            candidate_private_args: bool = False,
        ) -> None:
            self.assertIn("Final verification", desc)
            self.assertEqual(target_ms, 5000.0)
            self.assertFalse(use_isolated)
            self.assertFalse(confirm_suspicious)
            self.assertTrue(use_interleaved)
            self.assertFalse(candidate_private_args)
            for member in members:
                member.perfs.append(10.0 if member is stable else 12.0)

        with (
            clean_final_rebenchmark_env(),
            patch.object(search, "rebenchmark", side_effect=fake_rebenchmark),
        ):
            self.assertIs(search.final_rebenchmark_best(noisy), stable)

    def test_finalize_keeps_measured_best_when_worker_is_unkillable(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        best = PopulationMember(
            lambda: None, [9.0], (), helion.Config(num_warps=4), status="ok"
        )
        other = PopulationMember(
            lambda: None, [11.0], (), helion.Config(num_warps=8), status="ok"
        )
        search.population = [other, best]
        search._unkillable_configs = set()

        with patch.object(
            search,
            "final_rebenchmark_best",
            side_effect=BenchmarkWorkerUnkillable("worker remained alive"),
        ):
            self.assertEqual(search._finalize(), best.config)
        self.assertIs(search.best, best)

    def test_finalize_drops_the_candidate_whose_worker_could_not_be_reaped(
        self,
    ) -> None:
        """The config under verification when the worker hung may be the one
        that hung it: the fallback skips it, and raises when nothing is left."""
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        best = PopulationMember(
            lambda: None, [9.0], (), helion.Config(num_warps=4), status="ok"
        )
        other = PopulationMember(
            lambda: None, [11.0], (), helion.Config(num_warps=8), status="ok"
        )
        search.population = [other, best]
        search._unkillable_configs = set()
        error = BenchmarkWorkerUnkillable("worker remained alive")
        error.fn_index = 1
        search._record_unkillable([other, best], error)
        self.assertEqual(search._unkillable_configs, {best.config})

        with patch.object(search, "final_rebenchmark_best", side_effect=error):
            self.assertEqual(search._finalize(), other.config)
        self.assertIs(search.best, other)

        search._record_unkillable([other, best], BenchmarkWorkerUnkillable("x"))
        self.assertEqual(search._unkillable_configs, {best.config})
        search._unkillable_configs.add(other.config)
        with (
            patch.object(search, "final_rebenchmark_best", side_effect=error),
            self.assertRaises(BenchmarkWorkerUnkillable),
        ):
            search._finalize()

    def test_autotune_keeps_selected_config_when_worker_cleanup_is_unkillable(
        self,
    ) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.kernel = Mock()
        search.kernel.format_kernel_decorator.return_value = "@helion.kernel(...)"
        search.kernel.get_cached_path.return_value = None
        search.args = ()
        search._autotune_metrics = AutotuneMetrics()
        search._search_space_tracker = None
        search.benchmark_provider = Mock()
        search.benchmark_provider.cleanup.side_effect = BenchmarkWorkerUnkillable(
            "worker remained alive"
        )
        search._unkillable_configs = set()
        selected = helion.Config(num_warps=4)

        with (
            patch.object(search, "_prepare"),
            patch.object(search, "_autotune", return_value=selected),
            patch.object(search, "_finalize_autotune_metrics"),
        ):
            self.assertEqual(search.autotune(), selected)
        search.benchmark_provider.cleanup.assert_called_once()

        # ... unless the selected config is the one whose worker hung
        search._unkillable_configs = {selected}
        with (
            patch.object(search, "_prepare"),
            patch.object(search, "_autotune", return_value=selected),
            patch.object(search, "_finalize_autotune_metrics"),
            self.assertRaises(BenchmarkWorkerUnkillable),
        ):
            search.autotune()

    def test_final_rebenchmark_rejects_all_failed_finalists(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0
        search.config_spec = SimpleNamespace(backend_name="cute")

        config_a = helion.Config(num_warps=4)
        config_b = helion.Config(num_warps=8)
        member_a = PopulationMember(lambda: None, [10.0], (), config_a, status="ok")
        member_b = PopulationMember(lambda: None, [9.0], (), config_b, status="ok")
        search._benchmarked_members = {
            config_a: member_a,
            config_b: member_b,
        }
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [member_b]

        def fail_rebenchmark(members, **_kwargs):
            for member in members:
                member.perfs[:] = [float("inf")]
                member.status = "timeout"

        with (
            clean_final_rebenchmark_env(),
            patch.object(search, "rebenchmark", side_effect=fail_rebenchmark),
            self.assertRaises(exc.NoConfigFound),
        ):
            search.final_rebenchmark_best(member_b)

    def test_final_rebenchmark_replaces_invalid_best_with_only_live_candidate(
        self,
    ) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 10.0
        search.config_spec = SimpleNamespace(backend_name="cute")

        invalid_config = helion.Config(num_warps=8)
        live_config = helion.Config(num_warps=4)
        invalid = PopulationMember(
            lambda: None,
            [float("inf")],
            (),
            invalid_config,
            status="timeout",
        )
        live = PopulationMember(lambda: None, [10.0], (), live_config, status="ok")
        search._benchmarked_members = {live_config: live}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [invalid]

        with (
            clean_final_rebenchmark_env(),
            patch.object(
                search,
                "_autotune_budget_exceeded_across_ranks",
                return_value=False,
            ) as budget_exceeded,
            patch.object(search, "rebenchmark") as rebenchmark,
        ):
            self.assertIs(search.final_rebenchmark_best(invalid), live)

        budget_exceeded.assert_called_once_with()
        rebenchmark.assert_not_called()

    def test_final_rebenchmark_target_ms_env_is_bounded(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)

        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="50"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 200.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="1500"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 1500.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="1e308"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 60000.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="inf"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 5000.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="nan"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 5000.0)

    def test_final_rebenchmark_pinned_tolerance_env(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)

        with clean_final_rebenchmark_env():
            self.assertEqual(search._final_rebenchmark_pinned_tolerance(), 0.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE="0.02"
        ):
            self.assertEqual(search._final_rebenchmark_pinned_tolerance(), 0.02)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE="-1"
        ):
            self.assertEqual(search._final_rebenchmark_pinned_tolerance(), 0.0)

    def test_final_rebenchmark_top_k_default_scoped_to_cute(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)

        with clean_final_rebenchmark_env():
            # The larger finalist set is scoped to cute; other backends use the
            # cheaper default so autotune wall-time is not inflated.
            search.config_spec = SimpleNamespace(backend_name="cute")
            self.assertEqual(search._final_rebenchmark_top_k(), 32)
            search.config_spec = SimpleNamespace(backend_name="triton")
            self.assertEqual(search._final_rebenchmark_top_k(), 8)
            search.config_spec = SimpleNamespace(backend_name="pallas")
            self.assertEqual(search._final_rebenchmark_top_k(), 8)

        # An explicit env override wins regardless of backend.
        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search.config_spec = SimpleNamespace(backend_name="cute")
            self.assertEqual(search._final_rebenchmark_top_k(), 2)

    def test_final_rebenchmark_prefers_faster_generated_config_by_default(
        self,
    ) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0

        pinned_config = helion.Config(num_warps=4)
        noisy_config = helion.Config(num_warps=8)
        pinned = PopulationMember(lambda: None, [10.0], (), pinned_config, status="ok")
        noisy = PopulationMember(lambda: None, [9.0], (), noisy_config, status="ok")
        search._benchmarked_members = {
            pinned_config: pinned,
            noisy_config: noisy,
        }
        search._pinned_finalist_configs = {pinned_config}
        search._pinned_finalist_members = {pinned_config: pinned}
        search.population = [noisy]

        def fake_rebenchmark(
            members: list[PopulationMember],
            *,
            desc: str = "Rebenchmarking",
            target_ms: float = 200.0,
            use_isolated: bool = True,
            confirm_suspicious: bool = True,
            use_interleaved: bool = True,
            candidate_private_args: bool = False,
        ) -> None:
            self.assertFalse(candidate_private_args)
            for member in members:
                member.perfs.append(10.04 if member is pinned else 10.0)

        with (
            clean_final_rebenchmark_env(),
            patch.object(search, "rebenchmark", side_effect=fake_rebenchmark),
        ):
            self.assertIs(search.final_rebenchmark_best(noisy), noisy)

    def test_final_rebenchmark_can_prefer_near_tied_pinned_config(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0

        pinned_config = helion.Config(num_warps=4)
        noisy_config = helion.Config(num_warps=8)
        pinned = PopulationMember(lambda: None, [10.0], (), pinned_config, status="ok")
        noisy = PopulationMember(lambda: None, [9.0], (), noisy_config, status="ok")
        search._benchmarked_members = {
            pinned_config: pinned,
            noisy_config: noisy,
        }
        search._pinned_finalist_configs = {pinned_config}
        search._pinned_finalist_members = {pinned_config: pinned}
        search.population = [noisy]

        def fake_rebenchmark(
            members: list[PopulationMember],
            *,
            desc: str = "Rebenchmarking",
            target_ms: float = 200.0,
            use_isolated: bool = True,
            confirm_suspicious: bool = True,
            use_interleaved: bool = True,
            candidate_private_args: bool = False,
        ) -> None:
            self.assertFalse(candidate_private_args)
            for member in members:
                member.perfs.append(10.04 if member is pinned else 10.0)

        with (
            clean_final_rebenchmark_env(
                HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE="0.005"
            ),
            patch.object(search, "rebenchmark", side_effect=fake_rebenchmark),
        ):
            self.assertIs(search.final_rebenchmark_best(noisy), pinned)

    def test_final_rebenchmark_isolated_env(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)

        with clean_final_rebenchmark_env():
            self.assertFalse(search._final_rebenchmark_use_isolated())
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED="1"
        ):
            self.assertTrue(search._final_rebenchmark_use_isolated())
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED="maybe"
        ):
            self.assertFalse(search._final_rebenchmark_use_isolated())

    def test_final_rebenchmark_isolated_env_keeps_suspicious_confirmation(
        self,
    ) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0

        config_a = helion.Config(num_warps=4)
        config_b = helion.Config(num_warps=8)
        member_a = PopulationMember(lambda: None, [10.0], (), config_a, status="ok")
        member_b = PopulationMember(lambda: None, [9.0], (), config_b, status="ok")
        search._benchmarked_members = {
            config_a: member_a,
            config_b: member_b,
        }
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [member_b]

        def fake_rebenchmark(
            members: list[PopulationMember],
            *,
            desc: str = "Rebenchmarking",
            target_ms: float = 200.0,
            use_isolated: bool = True,
            confirm_suspicious: bool = True,
            use_interleaved: bool = True,
            candidate_private_args: bool = False,
        ) -> None:
            self.assertIn("Final verification", desc)
            self.assertEqual(target_ms, 5000.0)
            self.assertTrue(use_isolated)
            self.assertTrue(confirm_suspicious)
            self.assertFalse(use_interleaved)
            self.assertTrue(candidate_private_args)
            for member in members:
                member.perfs.append(9.5 if member is member_a else 10.0)

        with (
            clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED="1"),
            patch.object(search, "rebenchmark", side_effect=fake_rebenchmark),
        ):
            self.assertIs(search.final_rebenchmark_best(member_b), member_a)

    def test_rebenchmark_repeat_for_target_ms(self) -> None:
        self.assertEqual(PopulationBasedSearch._repeat_for_target_ms(200.0, 100.0), 3)
        self.assertEqual(
            PopulationBasedSearch._repeat_for_target_ms(1000.0, 0.05), 20000
        )
        self.assertEqual(
            PopulationBasedSearch._repeat_for_target_ms(1e308, 1e-308), 20_000
        )

    def test_mirrored_rebenchmark_attests_batched_call_sizing(self) -> None:
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_progress_bar=False,
        )
        search.args = ()
        search.log = AutotuningLogger(search.settings)
        search.best_perf_so_far = 20.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search._apply_rebenchmark_timings = Mock()
        members = [
            PopulationMember(lambda: None, [40.0], (), helion.Config(block_sizes=[1])),
            PopulationMember(lambda: None, [20.0], (), helion.Config(block_sizes=[2])),
        ]
        repeats: list[int] = []

        def mirrored_bench(_fns, *, repeat, desc=None, after_call=None):
            repeats.append(repeat)
            self.assertIsNone(desc)
            self.assertIsNotNone(after_call)
            sweep_count, calls_per_sample, total_calls = _mirrored_bench_call_layout(
                repeat
            )
            return MirroredBenchmarkTrace(
                orders=[[0, 1], [1, 0]] * (sweep_count // 2),
                elapsed_ms=[[40.0, 20.0], [20.0, 40.0]] * (sweep_count // 2),
                medians_ms=[40.0, 20.0],
                sweep_count=sweep_count,
                calls_per_sample=calls_per_sample,
                total_calls=total_calls,
            )

        with (
            clean_final_rebenchmark_env(),
            patch(
                "helion.autotuner.base_search.mirrored_bench_generic",
                side_effect=mirrored_bench,
            ),
            patch(
                "helion.autotuner.base_search.sync_object",
                side_effect=lambda value, **_kwargs: value,
            ),
            patch("helion.autotuner.base_search.clear_jit_fast_path_caches"),
        ):
            trace = search.mirrored_rebenchmark(
                members,
                desc="Terminal coordinate refinement",
                target_ms=200.0,
            )
            confirmation_trace = search.mirrored_rebenchmark(
                members,
                desc="Terminal coordinate refinement: confirming",
                target_ms=5000.0,
            )

        self.assertEqual(repeats, [6, 126])
        self.assertEqual(trace.target_ms, 200.0)
        self.assertEqual(trace.repeat_reference_perf_ms, 40.0)
        self.assertEqual(trace.sweep_count, 6)
        self.assertEqual(trace.calls_per_sample, 1)
        self.assertEqual(trace.total_calls, 6)
        self.assertEqual(confirmation_trace.target_ms, 5000.0)
        self.assertEqual(confirmation_trace.repeat_reference_perf_ms, 40.0)
        self.assertEqual(confirmation_trace.sweep_count, 64)
        self.assertEqual(confirmation_trace.calls_per_sample, 2)
        self.assertEqual(confirmation_trace.total_calls, 128)
        self.assertEqual(
            search._apply_rebenchmark_timings.call_args_list,
            [call(members, [40.0, 20.0]), call(members, [40.0, 20.0])],
        )
        metric = LFBOPatternSearch._flash_terminal_trace_metric(
            ["config-a", "config-b"],
            trace,
        )
        self.assertEqual(metric["target_ms"], 200.0)
        self.assertEqual(metric["repeat_reference_perf_ms"], 40.0)
        self.assertEqual(metric["sweep_count"], 6)
        self.assertEqual(metric["calls_per_sample"], 1)
        self.assertEqual(metric["total_calls"], 6)
        self.assertEqual(len(metric["elapsed_ms"]), 6)
        self.assertNotIn("sweeps", metric)

    def test_isolated_rebenchmark_rep_ms_is_timeout_bounded(self) -> None:
        self.assertEqual(PopulationBasedSearch._isolated_rep_ms(1000.0, 30), 1000)
        self.assertEqual(PopulationBasedSearch._isolated_rep_ms(60000.0, 30), 15000)
        with patch.dict(os.environ, {"HELION_CAP_REBENCHMARK_REPEAT": "50"}):
            self.assertEqual(PopulationBasedSearch._isolated_rep_ms(1000.0, 30), 50)

    def test_steady_rebenchmark_rep_ms_honors_cap(self) -> None:
        self.assertEqual(PopulationBasedSearch._steady_rebenchmark_rep_ms(1000.0), 1000)
        with patch.dict(os.environ, {"HELION_CAP_REBENCHMARK_REPEAT": "50"}):
            self.assertEqual(
                PopulationBasedSearch._steady_rebenchmark_rep_ms(1000.0), 50
            )

    def test_rebenchmark_repeat_ignores_global_optimistic_outlier(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        repeats: list[int] = []

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            repeats.append(repeat)
            return [50.0 for _ in fns]

        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.settings.autotune_benchmark_fn = custom_benchmark_fn
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 0.001
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.config_spec = SimpleNamespace(backend=None)
        member_a = PopulationMember(lambda: None, [50.0], (), helion.Config())
        member_b = PopulationMember(lambda: None, [51.0], (), helion.Config())

        search.rebenchmark([member_a, member_b], target_ms=1000.0)

        self.assertEqual(repeats, [19])

    def test_final_rebenchmark_can_skip_suspicious_confirmation(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0.9,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        search.benchmark_provider = SimpleNamespace(
            mutated_arg_indices=[],
            benchmark_isolated=Mock(side_effect=AssertionError("unexpected isolated")),
        )
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.config_spec = SimpleNamespace(backend=None)

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            return [50.0 for _ in fns]

        search.settings.autotune_benchmark_fn = custom_benchmark_fn
        member_a = PopulationMember(lambda: None, [100.0], (), helion.Config())
        member_b = PopulationMember(lambda: None, [101.0], (), helion.Config())

        search.rebenchmark(
            [member_a, member_b],
            target_ms=1000.0,
            use_isolated=False,
            confirm_suspicious=False,
        )

        self.assertEqual(member_a.perfs, [100.0, 50.0])
        self.assertEqual(member_b.perfs, [101.0, 50.0])

    def test_rebenchmark_can_use_steady_backend_timer(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        calls: list[tuple[int, int, str]] = []

        def steady_bench(
            fn: Callable[[], object],
            *,
            warmup: int,
            rep: int,
            return_mode: str,
            process_group_name: str | None = None,
        ) -> float:
            fn()
            calls.append((warmup, rep, return_mode))
            return 50.0 + len(calls)

        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(get_do_bench=lambda: steady_bench)
        )
        member_a = PopulationMember(lambda: None, [100.0], (), helion.Config())
        member_b = PopulationMember(lambda: None, [101.0], (), helion.Config())

        with patch("helion.autotuner.base_search.clear_jit_fast_path_caches") as clear:
            search.rebenchmark(
                [member_a, member_b],
                target_ms=1500.0,
                use_isolated=False,
                confirm_suspicious=False,
                use_interleaved=False,
            )

        self.assertEqual(calls, [(1000, 1500, "median"), (1000, 1500, "median")])
        self.assertEqual(member_a.perfs, [100.0, 51.0])
        self.assertEqual(member_b.perfs, [101.0, 52.0])
        self.assertEqual(clear.call_count, 2)

    def test_rebenchmark_isolates_mutated_args_when_worker_unavailable(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        original = torch.zeros(1)
        search.args = (original,)
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        benchmark_isolated = Mock(return_value=None)
        search.benchmark_provider = SimpleNamespace(
            mutated_arg_indices=[0],
            benchmark_isolated=benchmark_isolated,
        )
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        observed_pointers: dict[str, list[int]] = {"a": [], "b": []}

        def fn_a(value: torch.Tensor) -> None:
            observed_pointers["a"].append(value.data_ptr())
            value.add_(1)

        def fn_b(value: torch.Tensor) -> None:
            observed_pointers["b"].append(value.data_ptr())
            value.add_(1)

        def steady_bench(
            fn: Callable[[], object],
            *,
            warmup: int,
            rep: int,
            return_mode: str,
            process_group_name: str | None = None,
        ) -> float:
            fn()
            return 50.0

        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(get_do_bench=lambda: steady_bench)
        )
        member_a = PopulationMember(fn_a, [100.0], (), helion.Config(num_warps=4))
        member_b = PopulationMember(fn_b, [101.0], (), helion.Config(num_warps=8))

        with patch(
            "helion.autotuner.base_search._clone_args",
            side_effect=lambda args, process_group_name, idx_to_clone: tuple(
                value.clone()
                if idx_to_clone is None or index in idx_to_clone
                else value
                for index, value in enumerate(args)
            ),
        ) as clone_args:
            search.rebenchmark(
                [member_a, member_b],
                target_ms=1000.0,
                use_isolated=True,
                confirm_suspicious=False,
                use_interleaved=False,
                candidate_private_args=True,
            )

        benchmark_isolated.assert_called_once()
        self.assertTrue(benchmark_isolated.call_args.kwargs["fresh_process"])
        self.assertEqual(clone_args.call_count, 3)
        self.assertTrue(
            all(
                call.kwargs["idx_to_clone"] is None
                for call in clone_args.call_args_list
            )
        )
        self.assertNotEqual(observed_pointers["a"], observed_pointers["b"])
        self.assertNotEqual(observed_pointers["a"], [original.data_ptr()])
        self.assertNotEqual(observed_pointers["b"], [original.data_ptr()])
        self.assertTrue(torch.equal(original, torch.zeros_like(original)))

    def test_benchmark_isolated_names_the_candidate_whose_worker_hung(self) -> None:
        """A worker that cannot be reaped names the batch position of the
        candidate it ran, whether it refused to die under that candidate or at
        the fresh-process shutdown before the next one."""
        settings = Settings(autotune_log_level=logging.CRITICAL)
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = settings
        provider.log = Mock()
        provider.mutated_arg_indices = []
        provider._autotune_metrics = AutotuneMetrics()
        provider._subprocess_wrapper_unloadable = False
        provider._subprocess_benchmark_enabled = lambda: True
        provider._benchmark_worker = None
        runs: list[object] = []

        def hang_on_the_second(
            fn: object,
            *,
            warmup: int,
            rep: int,
            args_slot: int | None = None,
            timeout: float | None = None,
        ) -> float:
            runs.append(fn)
            if len(runs) == 2:
                raise BenchmarkWorkerUnkillable("worker remained alive")
            return 1.0

        provider._run_subprocess_benchmark_job = hang_on_the_second
        fns = [object(), object(), object()]
        with self.assertRaises(BenchmarkWorkerUnkillable) as caught:
            provider.benchmark_isolated(fns, warmup=1, rep=1)
        self.assertEqual(caught.exception.fn_index, 1)
        self.assertEqual(runs, fns[:2])

        # fresh processes: the worker that ran candidate 0 refuses to die when
        # it is shut down ahead of candidate 1
        runs.clear()
        worker = Mock()
        worker.shutdown.side_effect = BenchmarkWorkerUnkillable("worker remained alive")

        def run_and_leave_a_worker(
            fn: object,
            *,
            warmup: int,
            rep: int,
            args_slot: int | None = None,
            timeout: float | None = None,
        ) -> float:
            runs.append(fn)
            provider._benchmark_worker = worker
            return 1.0

        provider._run_subprocess_benchmark_job = run_and_leave_a_worker
        with self.assertRaises(BenchmarkWorkerUnkillable) as caught:
            provider.benchmark_isolated(fns, warmup=1, rep=1, fresh_process=True)
        self.assertEqual(caught.exception.fn_index, 0)
        self.assertEqual(runs, fns[:1])
        self.assertIsNone(provider._benchmark_worker)

    def test_rebenchmark_falls_back_when_isolated_wrapper_is_unloadable(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = settings
        provider.log = Mock()
        provider.mutated_arg_indices = []
        provider._autotune_metrics = AutotuneMetrics()
        provider._subprocess_wrapper_unloadable = False
        provider._subprocess_benchmark_enabled = lambda: True
        isolated_calls = 0

        def unloadable_wrapper(
            _fn: object,
            *,
            warmup: int,
            rep: int,
            args_slot: int | None = None,
            timeout: float | None = None,
        ) -> None:
            nonlocal isolated_calls
            isolated_calls += 1
            provider._subprocess_wrapper_unloadable = True
            return None

        provider._run_subprocess_benchmark_job = unloadable_wrapper

        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        search.benchmark_provider = provider
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        in_process_calls = 0

        def steady_bench(
            fn: Callable[[], object],
            *,
            warmup: int,
            rep: int,
            return_mode: str,
            process_group_name: str | None = None,
        ) -> float:
            nonlocal in_process_calls
            fn()
            in_process_calls += 1
            return 50.0 + in_process_calls

        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=False,
            backend=SimpleNamespace(
                get_do_bench=lambda: steady_bench,
                should_deduplicate_generated_sources=lambda _spec: False,
                probe_long_autotune_kernels=lambda _spec: False,
                default_autotune_cudagraph=lambda _spec: False,
            ),
        )
        provider.config_spec = search.config_spec
        member_a = PopulationMember(lambda: None, [100.0], (), helion.Config())
        member_b = PopulationMember(lambda: None, [101.0], (), helion.Config())

        search.rebenchmark(
            [member_a, member_b],
            target_ms=1000.0,
            use_isolated=True,
            confirm_suspicious=False,
            use_interleaved=False,
        )

        self.assertEqual(isolated_calls, 1)
        self.assertEqual(in_process_calls, 2)
        self.assertEqual(member_a.perfs, [100.0, 51.0])
        self.assertEqual(member_b.perfs, [101.0, 52.0])

    def test_final_rebenchmark_respects_custom_benchmark_fn(
        self,
    ) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        calls: list[str] = []

        def fn_a() -> None:
            calls.append("a")

        def fn_b() -> None:
            calls.append("b")

        def steady_bench(
            fn: Callable[[], object],
            *,
            warmup: int,
            rep: int,
            return_mode: str,
            process_group_name: str | None = None,
        ) -> float:
            raise AssertionError("unexpected steady benchmark")

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            self.assertIn("Final verification", desc or "")
            self.assertEqual(repeat, 500)
            timings = []
            for fn in fns:
                fn()
                timings.append(9.5 if calls[-1] == "a" else 10.0)
            return timings

        settings.autotune_benchmark_fn = custom_benchmark_fn

        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(get_do_bench=lambda: steady_bench)
        )
        config_a = helion.Config(num_warps=4)
        config_b = helion.Config(num_warps=8)
        member_a = PopulationMember(fn_a, [10.0], (), config_a, status="ok")
        member_b = PopulationMember(fn_b, [9.0], (), config_b, status="ok")
        search._benchmarked_members = {
            config_a: member_a,
            config_b: member_b,
        }
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [member_b]

        with (
            clean_final_rebenchmark_env(),
            without_env_var("HELION_CAP_REBENCHMARK_REPEAT"),
        ):
            self.assertIs(search.final_rebenchmark_best(member_b), member_a)

    def test_rebenchmark_uses_target_ms_for_isolated_benchmark(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 0.05
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        reps: list[int] = []

        def benchmark_isolated(
            fns: list[Callable[[], object]],
            *,
            warmup: int,
            rep: int,
            desc: str,
            fresh_process: bool = False,
        ) -> list[float]:
            reps.append(rep)
            return [0.05 for _ in fns]

        search.benchmark_provider = SimpleNamespace(
            benchmark_isolated=benchmark_isolated,
            mutated_arg_indices=[],
        )
        member_a = PopulationMember(lambda: None, [0.05], (), helion.Config())
        member_b = PopulationMember(lambda: None, [0.06], (), helion.Config())

        search.rebenchmark([member_a, member_b], target_ms=1000.0)

        self.assertEqual(reps, [1000])

    def test_benchmarked_member_history_is_bounded_and_deduped(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        config_a = helion.Config(num_warps=4)
        config_b = helion.Config(num_warps=8)
        config_c = helion.Config(num_warps=16)
        slow_a = PopulationMember(lambda: None, [10.0], (), config_a, status="ok")
        fast_a = PopulationMember(lambda: None, [8.0], (), config_a, status="ok")
        member_b = PopulationMember(lambda: None, [9.0], (), config_b, status="ok")
        member_c = PopulationMember(lambda: None, [7.0], (), config_c, status="ok")

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(slow_a)
            search._record_benchmarked_member(fast_a)
            search._record_benchmarked_member(member_b)
            search._record_benchmarked_member(member_c)

        # A snapshot of the faster member for config_a is retained (deduped).
        self.assertEqual(search._benchmarked_members[config_a].perfs, fast_a.perfs)
        self.assertIsNot(search._benchmarked_members[config_a].perfs, fast_a.perfs)
        self.assertNotIn(config_b, search._benchmarked_members)
        self.assertEqual(set(search._benchmarked_members), {config_a, config_c})

    def test_cute_finalist_history_uses_latest_verified_perf(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="cute", cute_flash_search_enabled=True
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        initial_fast_final_slow = PopulationMember(
            lambda: None,
            [1.0, 10.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        middle = PopulationMember(
            lambda: None,
            [2.0, 5.0],
            [],
            helion.Config(num_warps=8),
            status="ok",
        )
        initial_slow_final_fast = PopulationMember(
            lambda: None,
            [3.0, 0.5],
            [],
            helion.Config(num_warps=16),
            status="ok",
        )

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(initial_fast_final_slow)
            search._record_benchmarked_member(middle)
            search._record_benchmarked_member(initial_slow_final_fast)

        self.assertEqual(
            set(search._benchmarked_members),
            {middle.config, initial_slow_final_fast.config},
        )

    def test_non_cute_finalist_history_keeps_low_water_perf(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="triton", cute_flash_search_enabled=False
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        initial_fast_final_slow = PopulationMember(
            lambda: None,
            [1.0, 10.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        middle = PopulationMember(
            lambda: None,
            [2.0, 5.0],
            [],
            helion.Config(num_warps=8),
            status="ok",
        )
        initial_slow_final_fast = PopulationMember(
            lambda: None,
            [3.0, 0.5],
            [],
            helion.Config(num_warps=16),
            status="ok",
        )

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(initial_fast_final_slow)
            search._record_benchmarked_member(middle)
            search._record_benchmarked_member(initial_slow_final_fast)

        self.assertEqual(
            set(search._benchmarked_members),
            {initial_fast_final_slow.config, initial_slow_final_fast.config},
        )

    def test_non_cute_rebenchmark_refreshes_retained_snapshot(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="triton", cute_flash_search_enabled=False
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.best_perf_so_far = 5.5
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))

        improves = PopulationMember(
            lambda: None,
            [6.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        unchanged = PopulationMember(
            lambda: None,
            [5.5],
            [],
            helion.Config(num_warps=8),
            status="ok",
        )

        search.benchmark_provider = SimpleNamespace(
            benchmark_isolated=lambda fns, **kwargs: [5.0, 5.5],
            mutated_arg_indices=[],
        )

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(improves)
            search._record_benchmarked_member(unchanged)
            search.rebenchmark([improves, unchanged], target_ms=200.0)

        snapshot = search._benchmarked_members[improves.config]
        self.assertEqual(snapshot.perfs, [6.0, 5.0])
        self.assertIsNot(snapshot.perfs, improves.perfs)

    def test_non_cute_history_keeps_finite_low_water_after_failed_rebenchmark(
        self,
    ) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="triton", cute_flash_search_enabled=False
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        member = PopulationMember(
            lambda: None,
            [5.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(member)
            member.perfs.append(float("inf"))
            search._refresh_benchmarked_members_after_rebenchmark([member])

        self.assertEqual(
            search._benchmarked_members[member.config].perfs,
            [5.0, float("inf")],
        )

    def test_cute_rebenchmark_refresh_reinserts_pruned_finalist(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="cute", cute_flash_search_enabled=True
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.best_perf_so_far = 1.0
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))

        initial_fast_final_slow = PopulationMember(
            lambda: None,
            [1.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        middle = PopulationMember(
            lambda: None,
            [2.0],
            [],
            helion.Config(num_warps=8),
            status="ok",
        )
        initial_slow_final_fast = PopulationMember(
            lambda: None,
            [3.0],
            [],
            helion.Config(num_warps=16),
            status="ok",
        )
        members = [initial_fast_final_slow, middle, initial_slow_final_fast]

        def benchmark_isolated(
            fns: list[Callable[[], object]],
            *,
            warmup: int,
            rep: int,
            desc: str,
            fresh_process: bool = False,
        ) -> list[float]:
            self.assertEqual(len(fns), 3)
            return [10.0, 5.0, 0.5]

        search.benchmark_provider = SimpleNamespace(
            benchmark_isolated=benchmark_isolated,
            mutated_arg_indices=[],
        )

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            for member in members:
                search._record_benchmarked_member(member)
            self.assertNotIn(
                initial_slow_final_fast.config, search._benchmarked_members
            )
            initial_snapshot = search._benchmarked_members[
                initial_fast_final_slow.config
            ]
            self.assertIsNot(initial_snapshot.perfs, initial_fast_final_slow.perfs)

            search.rebenchmark(members, target_ms=200.0)

        self.assertEqual(
            set(search._benchmarked_members),
            {middle.config, initial_slow_final_fast.config},
        )
        refreshed = search._benchmarked_members[initial_slow_final_fast.config]
        self.assertEqual(refreshed.perfs, [3.0, 0.5])
        self.assertIsNot(refreshed.perfs, initial_slow_final_fast.perfs)

    def test_pinned_finalist_survives_benchmarked_member_pruning(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        pinned_config = helion.Config(num_warps=4)
        fast_config = helion.Config(num_warps=8)
        faster_config = helion.Config(num_warps=16)
        slow_config = helion.Config(num_warps=32)
        pinned = PopulationMember(lambda: None, [10.0], (), pinned_config, status="ok")
        fast = PopulationMember(lambda: None, [8.0], (), fast_config, status="ok")
        faster = PopulationMember(lambda: None, [7.0], (), faster_config, status="ok")
        slow = PopulationMember(lambda: None, [9.0], (), slow_config, status="ok")
        search.pin_finalist_config(pinned_config)

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(pinned)
            search._record_benchmarked_member(fast)
            search._record_benchmarked_member(faster)
            search._record_benchmarked_member(slow)

        # The pinned config survives pruning (kept as a snapshot in the map).
        self.assertIn(pinned_config, search._pinned_finalist_members)
        self.assertEqual(
            search._pinned_finalist_members[pinned_config].perfs, pinned.perfs
        )
        self.assertEqual(set(search._benchmarked_members), {fast_config, faster_config})

    def test_pinned_finalist_refresh_groups_equal_config_duplicates(self) -> None:
        for backend_name, cute_flash_search_enabled in (
            ("triton", False),
            ("cute", True),
        ):
            with self.subTest(backend=backend_name):
                settings = Settings(autotune_log_level=logging.CRITICAL)
                search = PopulationBasedSearch.__new__(PopulationBasedSearch)
                search.settings = settings
                search.log = AutotuningLogger(settings)
                search.config_spec = SimpleNamespace(
                    backend_name=backend_name,
                    cute_flash_search_enabled=cute_flash_search_enabled,
                )
                search._benchmarked_members = {}
                search._pinned_finalist_configs = set()
                search._pinned_finalist_members = {}

                config = helion.Config(num_warps=4)
                original = PopulationMember(
                    lambda: None, [10.0], (), config, status="ok"
                )
                better_duplicate = PopulationMember(
                    lambda: None,
                    [6.0],
                    (),
                    helion.Config(num_warps=4),
                    status="ok",
                )
                failed_duplicate = PopulationMember(
                    lambda: None,
                    [float("inf")],
                    (),
                    helion.Config(num_warps=4),
                    status="timeout",
                )
                worse_duplicate = PopulationMember(
                    lambda: None,
                    [9.0],
                    (),
                    helion.Config(num_warps=4),
                    status="ok",
                )
                search.pin_finalist_config(config)

                with clean_final_rebenchmark_env(
                    HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"
                ):
                    search._record_benchmarked_member(original)
                    search._refresh_benchmarked_members_after_rebenchmark(
                        [better_duplicate, failed_duplicate, worse_duplicate]
                    )

                self.assertIn(config, search._pinned_finalist_members)
                self.assertEqual(
                    search._pinned_finalist_members[config].perfs,
                    better_duplicate.perfs,
                )

    def test_failed_duplicate_does_not_evict_pinned_finalist(self) -> None:
        for backend_name, cute_flash_search_enabled in (
            ("triton", False),
            ("cute", True),
        ):
            with self.subTest(backend=backend_name):
                settings = Settings(autotune_log_level=logging.CRITICAL)
                search = PopulationBasedSearch.__new__(PopulationBasedSearch)
                search.settings = settings
                search.log = AutotuningLogger(settings)
                search.config_spec = SimpleNamespace(
                    backend_name=backend_name,
                    cute_flash_search_enabled=cute_flash_search_enabled,
                )
                search._benchmarked_members = {}
                search._pinned_finalist_configs = set()
                search._pinned_finalist_members = {}

                config = helion.Config(num_warps=4)
                original = PopulationMember(
                    lambda: None, [10.0], (), config, status="ok"
                )
                failed_duplicate = PopulationMember(
                    lambda: None,
                    [float("inf")],
                    (),
                    helion.Config(num_warps=4),
                    status="timeout",
                )
                search.pin_finalist_config(config)

                with clean_final_rebenchmark_env(
                    HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"
                ):
                    search._record_benchmarked_member(original)
                    search._refresh_benchmarked_members_after_rebenchmark(
                        [failed_duplicate]
                    )

                self.assertIn(config, search._pinned_finalist_members)
                self.assertEqual(
                    search._pinned_finalist_members[config].perfs, original.perfs
                )


class TestAutotuneBestOfKSettings(TestCase):
    """Settings-only coverage for ``autotune_best_of_k`` (no GPU/backend
    dependency)."""

    def test_default_is_one(self) -> None:
        with without_env_var("HELION_AUTOTUNE_BEST_OF_K"):
            self.assertEqual(helion.Settings().autotune_best_of_k, 1)

    def test_env_var_override(self) -> None:
        with patch.dict(os.environ, {"HELION_AUTOTUNE_BEST_OF_K": "5"}, clear=False):
            self.assertEqual(helion.Settings().autotune_best_of_k, 5)

    def test_setting_override(self) -> None:
        self.assertEqual(helion.Settings(autotune_best_of_k=3).autotune_best_of_k, 3)

    def test_k_zero_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"autotune_best_of_k must be >= 1"):
            helion.Settings(autotune_best_of_k=0)

    def test_k_negative_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"autotune_best_of_k must be >= 1"):
            helion.Settings(autotune_best_of_k=-3)

    def test_effort_profile_preserves_positional_fields(self) -> None:
        profile = AutotuneEffortProfile(
            None,
            None,
            None,
            None,
            None,
            7,
            1.25,
        )

        self.assertEqual(profile.finishing_rounds, 7)
        self.assertEqual(profile.rebenchmark_threshold, 1.25)
        self.assertIsNone(profile.flash_structural_search)


@onlyBackends(["triton", "cute"])
class TestAutotuneBestOfK(RefEagerTestDisabled, TestCase):
    """Best-of-K multi-seed autotune selection — cache key + K-loop coverage.

    Covers:
      - K = 1 leaves the cache hash byte-identical with the pre-feature
        repr (no field appended to the dataclass repr).
      - K > 1 differentiates the cache hash structurally; the K value
        appears as a field on the key dataclass, not concatenated into
        ``extra_cache_key``.
      - K > 1 with no ``_autotuner_factory`` wired raises a clear error.
      - The K-loop runs K trials with deterministic per-trial seeds,
        and the winner is picked by the **final rebench** (not the
        per-trial ``best_perf_so_far`` low-water mark).
      - The autotuner reference on the cache is restored to the
        original after the loop.
    """

    def test_cache_key_byte_identical_when_k_is_one(self) -> None:
        """K=1 cache hash must match the bytes produced by the original
        ``LooseAutotuneCacheKey`` repr (before ``best_of_k`` was added)."""
        from helion.autotuner.base_cache import LooseAutotuneCacheKey

        # Build a key with K=1 and one with K=5; the K=1 repr must equal
        # the repr that omits ``best_of_k`` entirely.
        common_kwargs = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "abc",
            "hardware": "B200",
            "runtime_name": "12.6",
            "backend": "triton",
            "config_spec_hash": "h1",
            "extra_cache_key": "",
        }
        k1 = LooseAutotuneCacheKey(**common_kwargs, best_of_k=1)
        k5 = LooseAutotuneCacheKey(**common_kwargs, best_of_k=5)
        # K=1 repr matches a manually-constructed repr without best_of_k.
        expected_k1_repr = (
            "LooseAutotuneCacheKey("
            "specialization_key=(), extra_results=(), "
            "kernel_source_hash='abc', hardware='B200', "
            "runtime_name='12.6', backend='triton', "
            "config_spec_hash='h1', extra_cache_key='')"
        )
        self.assertEqual(repr(k1), expected_k1_repr)
        # K=5 includes the field in the repr.
        self.assertIn("best_of_k=5", repr(k5))
        # Hashes differ structurally.
        self.assertNotEqual(k1.stable_hash(), k5.stable_hash())

    def test_cache_key_does_not_alias_extra_cache_key(self) -> None:
        """A K=1 key with ``extra_cache_key`` carrying a literal
        ``;best_of_k=5`` suffix must NOT collide with a K=5 key whose
        ``extra_cache_key`` is empty — i.e. the K must be a structural
        field, not folded into the string."""
        from helion.autotuner.base_cache import LooseAutotuneCacheKey

        k1_aliased = LooseAutotuneCacheKey(
            specialization_key=(),
            extra_results=(),
            kernel_source_hash="abc",
            hardware="B200",
            runtime_name="12.6",
            backend="triton",
            config_spec_hash="h1",
            extra_cache_key="foo;best_of_k=5",
            best_of_k=1,
        )
        k5 = LooseAutotuneCacheKey(
            specialization_key=(),
            extra_results=(),
            kernel_source_hash="abc",
            hardware="B200",
            runtime_name="12.6",
            backend="triton",
            config_spec_hash="h1",
            extra_cache_key="foo",
            best_of_k=5,
        )
        self.assertNotEqual(k1_aliased.stable_hash(), k5.stable_hash())

    def test_strict_cache_key_tracks_nondefault_best_of_k(self) -> None:
        from helion.autotuner.base_cache import StrictAutotuneCacheKey

        common_kwargs = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "abc",
            "hardware": "B200",
            "runtime_name": "12.6",
            "backend": "triton",
            "config_spec_hash": "h1",
            "extra_cache_key": "",
            "helion_key": "helion",
            "torch_key": "torch",
            "triton_key": "triton",
        }
        k1 = StrictAutotuneCacheKey(**common_kwargs, best_of_k=1)
        k5 = StrictAutotuneCacheKey(**common_kwargs, best_of_k=5)
        expected_k1_repr = (
            "StrictAutotuneCacheKey("
            "specialization_key=(), extra_results=(), "
            "kernel_source_hash='abc', hardware='B200', "
            "runtime_name='12.6', backend='triton', "
            "config_spec_hash='h1', extra_cache_key='', "
            "helion_key='helion', torch_key='torch', triton_key='triton')"
        )

        self.assertEqual(repr(k1), expected_k1_repr)
        self.assertIn("best_of_k=5", repr(k5))
        self.assertNotEqual(k1.stable_hash(), k5.stable_hash())

    def test_best_of_k_gt_1_without_factory_raises(self) -> None:
        """The bare ``Cache(autotuner)`` constructor must reject K>1 at
        run time rather than silently fall back to single-trial."""
        from helion.autotuner.local_cache import LocalAutotuneCache

        @helion.kernel(autotune_best_of_k=3, autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)

        # Build a cache with NO autotuner_factory wired; this models the
        # external ``Cache(autotuner)`` constructor path.
        class _MinimalSearch:
            def __init__(self):
                self.kernel = bound
                self.settings = bound.settings
                self.args = args
                self.best_perf_so_far = math.inf
                self._skip_cache = False

                class _Log:
                    def __call__(self, *a, **kw):
                        pass

                    def reset(self):
                        pass

                    def warning(self, *a, **kw):
                        pass

                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                return None  # never reached: K>1 must raise before this

        cache = LocalAutotuneCache(_MinimalSearch())  # no autotuner_factory
        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=True),
            self.assertRaisesRegex(
                RuntimeError,
                r"autotune_best_of_k > 1 requires a registered _autotuner_factory",
            ),
        ):
            cache.autotune()

    def test_single_trial_receives_effective_cache_bypass(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache

        config = helion.Config(block_sizes=[16])
        search = SimpleNamespace(
            settings=helion.Settings(autotune_best_of_k=1),
            autotune=Mock(return_value=config),
        )
        cache = object.__new__(LocalAutotuneCache)
        cache.autotuner = search

        self.assertEqual(cache._run_autotune_trials(skip_cache=True), config)
        search.autotune.assert_called_once_with(skip_cache=True)

    def test_k_loop_runs_k_trials_with_deterministic_seeds(self) -> None:
        """The K-loop runs K trials with seeds ``base + i``."""
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        seeds_seen: list[int] = []
        skip_cache_seen: list[bool] = []
        # ``block_sizes`` must be powers of two; pick four distinct values.
        trial_configs = [Config(block_sizes=[16, 1 << (3 + i)]) for i in range(4)]
        # Low-water perfs (per-trial best_perf_so_far) and rebench perfs
        # disagree on the winner: low-water best is index 2, rebench
        # best is index 3.
        low_water_perfs = [3.0, 5.0, 1.0, 2.0]
        rebench_perfs = [3.0, 5.0, 8.0, 2.0]
        trial_idx = {"n": 0}

        class MockTrialSearch:
            def __init__(self, bound_kernel, args, **kwargs):
                self.kernel = bound_kernel
                self.settings = bound_kernel.settings
                self.args = args
                self.best_perf_so_far = math.inf
                self._skip_cache = False

                class _Log:
                    def __call__(self, *a, **kw):
                        pass

                    def reset(self):
                        pass

                    def warning(self, *a, **kw):
                        pass

                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                i = trial_idx["n"]
                seeds_seen.append(self.settings.autotune_random_seed)
                skip_cache_seen.append(skip_cache)
                self.best_perf_so_far = low_water_perfs[i]
                cfg = trial_configs[i]
                trial_idx["n"] += 1
                return cfg

        def mock_autotuner_fn(bound_kernel, args, **kwargs):
            def factory():
                return MockTrialSearch(bound_kernel, args, **kwargs)

            return LocalAutotuneCache(factory(), autotuner_factory=factory)

        @helion.kernel(
            autotuner_fn=mock_autotuner_fn,
            autotune_best_of_k=4,
            autotune_random_seed=100,
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)

        # Patch the final-rebench step so we can return the desired perfs
        # without needing a real benchmark_provider.
        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=True),
            patch.object(
                LocalAutotuneCache,
                "_rebench_trial_configs",
                lambda self, configs: rebench_perfs,
            ),
        ):
            picked = bound.autotune(args)

        # K trials ran.
        self.assertEqual(trial_idx["n"], 4)
        # Deterministic seeds: base + i.
        self.assertEqual(seeds_seen, [100, 101, 102, 103])
        self.assertEqual(skip_cache_seen, [True, True, True, True])
        # Winner is picked by REBENCH (index 3), not by low-water (index 2).
        rebench_winner = rebench_perfs.index(min(rebench_perfs))
        low_water_winner = low_water_perfs.index(min(low_water_perfs))
        self.assertNotEqual(rebench_winner, low_water_winner)
        self.assertEqual(picked, trial_configs[rebench_winner])

    def test_k_loop_rejects_all_failed_final_rebenchmarks(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        settings = helion.Settings(
            autotune_best_of_k=2,
            autotune_random_seed=10,
            autotune_log_level=0,
        )
        search = SimpleNamespace(
            settings=settings,
            log=Mock(),
            config_spec=_cute_flash_test_config_spec(),
        )
        cache = LocalAutotuneCache.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache._autotuner_factory = Mock()
        configs = [Config(block_sizes=[16]), Config(block_sizes=[32])]

        with (
            patch.object(
                cache,
                "_run_one_trial",
                side_effect=[(configs[0], 1.0), (configs[1], 2.0)],
            ),
            patch.object(cache, "_release_trial_state"),
            patch.object(
                cache,
                "_rebench_trial_configs",
                return_value=[math.inf, math.inf],
            ) as rebench,
            self.assertRaises(exc.NoConfigFound),
        ):
            cache._run_autotune_trials()

        self.assertEqual(rebench.call_count, 2)

    def test_cute_k_loop_retries_partial_finalist_failure(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        settings = helion.Settings(
            autotune_best_of_k=2,
            autotune_random_seed=10,
            autotune_log_level=0,
        )
        search = SimpleNamespace(
            settings=settings,
            log=Mock(),
            config_spec=_cute_flash_test_config_spec(),
        )
        cache = LocalAutotuneCache.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache._autotuner_factory = Mock()
        configs = [Config(block_sizes=[16]), Config(block_sizes=[32])]

        with (
            patch.object(
                cache,
                "_run_one_trial",
                side_effect=[(configs[0], 1.0), (configs[1], 2.0)],
            ),
            patch.object(cache, "_release_trial_state"),
            patch.object(
                cache,
                "_rebench_trial_configs",
                side_effect=([math.inf, 2.0], [1.0, math.inf]),
            ) as rebench,
        ):
            selected = cache._run_autotune_trials()

        self.assertEqual(selected, configs[0])
        self.assertEqual(rebench.call_count, 2)

    def test_cute_k_loop_skips_one_failed_trial(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        settings = helion.Settings(
            autotune_best_of_k=2,
            autotune_random_seed=10,
            autotune_log_level=0,
        )
        search = SimpleNamespace(
            settings=settings,
            log=Mock(),
            config_spec=_cute_flash_test_config_spec(),
        )
        cache = LocalAutotuneCache.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache._autotuner_factory = Mock()
        winner = Config(block_sizes=[32])

        with (
            patch.object(
                cache,
                "_run_one_trial",
                side_effect=[exc.NoConfigFound(), (winner, 2.0)],
            ),
            patch.object(cache, "_release_trial_state"),
            patch.object(
                cache,
                "_rebench_trial_configs",
                return_value=[1.5],
            ) as rebench,
        ):
            selected = cache._run_autotune_trials()

        self.assertEqual(selected, winner)
        rebench.assert_called_once_with([winner])

    def test_non_cute_k_loop_preserves_transient_finalist_fallback(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        settings = helion.Settings(
            autotune_best_of_k=2,
            autotune_random_seed=10,
            autotune_log_level=0,
        )
        search = SimpleNamespace(
            settings=settings,
            log=Mock(),
            config_spec=SimpleNamespace(cute_flash_search_enabled=False),
        )
        cache = LocalAutotuneCache.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache._autotuner_factory = Mock()
        configs = [Config(block_sizes=[16]), Config(block_sizes=[32])]

        with (
            patch.object(
                cache,
                "_run_one_trial",
                side_effect=[(configs[0], 1.0), (configs[1], 2.0)],
            ),
            patch.object(cache, "_release_trial_state"),
            patch.object(
                cache,
                "_rebench_trial_configs",
                return_value=[math.inf, math.inf],
            ) as rebench,
        ):
            selected = cache._run_autotune_trials()

        self.assertEqual(selected, configs[0])
        rebench.assert_called_once()

    def test_k_loop_restores_autotuner_and_settings(self) -> None:
        """After the K-loop, ``cache.autotuner`` must equal the original
        instance, and the mutated settings must be restored to base."""
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        cfg = Config(block_sizes=[16, 32])

        class MockSearch:
            def __init__(self, bound_kernel, args, **kwargs):
                self.kernel = bound_kernel
                self.settings = bound_kernel.settings
                self.args = args
                self.best_perf_so_far = math.inf
                self._skip_cache = False

                class _Log:
                    def __call__(self, *a, **kw):
                        pass

                    def reset(self):
                        pass

                    def warning(self, *a, **kw):
                        pass

                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                # Simulate the adaptive-timeout mutation that the real
                # BaseSearch does inside _prepare()/set_adaptive_compile_timeout.
                self.settings.autotune_compile_timeout = 5
                self.best_perf_so_far = 1.0
                return cfg

        original_autotuner_ref = {"r": None}

        def mock_autotuner_fn(bound_kernel, args, **kwargs):
            def factory():
                return MockSearch(bound_kernel, args, **kwargs)

            inner = factory()
            original_autotuner_ref["r"] = inner
            return LocalAutotuneCache(inner, autotuner_factory=factory)

        @helion.kernel(
            autotuner_fn=mock_autotuner_fn,
            autotune_best_of_k=3,
            autotune_random_seed=100,
            autotune_compile_timeout=60,
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)

        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=True),
            patch.object(
                LocalAutotuneCache,
                "_rebench_trial_configs",
                lambda self, configs: [1.0] * len(configs),
            ),
        ):
            captured_cache = bound.settings.autotuner_fn(bound, args)
            self.assertIsNotNone(original_autotuner_ref["r"])
            self.assertIs(captured_cache.autotuner, original_autotuner_ref["r"])
            captured_cache.autotune()

        # After the K-loop, the cache's ``autotuner`` reference must be
        # restored to the original instance (no leaked trial swap).
        self.assertIs(captured_cache.autotuner, original_autotuner_ref["r"])
        # And the settings the autotuner mutated must be restored to base.
        self.assertEqual(bound.settings.autotune_compile_timeout, 60)
        self.assertEqual(bound.settings.autotune_random_seed, 100)

    def test_k_one_falls_through_to_single_trial(self) -> None:
        """With ``autotune_best_of_k=1`` the K-loop must not run; the
        cache calls the autotuner exactly once and returns its config.
        """
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        call_count = {"n": 0}
        only_config = Config(block_sizes=[16, 32])

        class _Log:
            def __call__(self, *a, **kw):
                pass

            def reset(self):
                pass

            def warning(self, *a, **kw):
                pass

        class SingleTrialSearch:
            def __init__(self, bound_kernel, args, **kwargs):
                self.kernel = bound_kernel
                self.settings = bound_kernel.settings
                self.args = args
                self.best_perf_so_far = 1.0
                self._skip_cache = False
                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                call_count["n"] += 1
                return only_config

        def mock_autotuner_fn(bound_kernel, args, **kwargs):
            def factory():
                return SingleTrialSearch(bound_kernel, args, **kwargs)

            return LocalAutotuneCache(factory(), autotuner_factory=factory)

        @helion.kernel(
            autotuner_fn=mock_autotuner_fn,
            autotune_best_of_k=1,
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)
        with patch("helion.autotuner.base_cache.should_skip_cache", return_value=True):
            picked = bound.autotune(args)
        self.assertEqual(call_count["n"], 1)
        self.assertEqual(picked, only_config)


if __name__ == "__main__":
    unittest.main()
