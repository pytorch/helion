from __future__ import annotations

import copy
import csv
import functools
import json
import logging
import math
import operator
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

from test._autotuner_test_utils import RecordingRandomSearch
from test._autotuner_test_utils import _cute_flash_test_config_spec
from test._autotuner_test_utils import without_env_var

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfXPU
from helion.autotuner import LFBOPatternSearch
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import _compile_config_failure_source_hash
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import StrictLocalAutotuneCache
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import KernelMetadata
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.search_space_logger import canonical_config_id
import helion.language as hl
from helion.runtime.settings import Settings
from helion.runtime.settings import _get_backend


@onlyBackends(["triton", "cute"])
class TestAutotuneBudget(TestCase):
    def _make_search(self, settings: Settings) -> BaseSearch:
        # NOTE: construct via __init__ (mock kernel) instead of hand-mirroring
        # its attributes, so new __init__ fields don't need to be added here.
        config_spec = SimpleNamespace(
            backend=SimpleNamespace(supports_precompile=lambda: True),
            default_config=lambda: helion.Config(block_sizes=[1]),
            compiler_seed_timeout_retry_repetitions=None,
            cute_flash_search_enabled=False,
        )
        kernel = SimpleNamespace(
            settings=settings,
            config_spec=config_spec,
            format_kernel_decorator=lambda config, s: "decorator",
            to_triton_code=lambda config: "code",
            maybe_log_repro=lambda log_func, args, config=None: None,
            supports_subprocess_benchmark=lambda: False,
            env=SimpleNamespace(process_group_name=None),
        )
        search = BaseSearch(kernel, ())
        with patch.object(
            LocalBenchmarkProvider,
            "_compute_baseline",
            return_value=(None, [], None),
        ):
            search._prepare()
        return search

    def test_setting_default_is_none(self) -> None:
        with without_env_var("HELION_AUTOTUNE_BUDGET_SECONDS"):
            settings = Settings()
        self.assertIsNone(settings.autotune_budget_seconds)

    def test_cute_backend_uses_default_autotune_budget_without_mutating(self) -> None:
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.backend import _CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS

        settings = Settings(
            autotune_budget_seconds=None,
            autotune_effort="quick",
            autotune_log_level=logging.CRITICAL,
        )
        bound_kernel = SimpleNamespace(settings=settings)
        observed_budgets: list[int | None] = []

        def fake_autotune(self_, bound_kernel_, args, **kwargs):
            observed_budgets.append(bound_kernel_.settings.autotune_budget_seconds)
            return helion.Config()

        with patch.object(
            Backend, "autotune", autospec=True, side_effect=fake_autotune
        ):
            CuteBackend().autotune(bound_kernel, ())

        self.assertEqual(observed_budgets, [_CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS])
        self.assertIsNone(bound_kernel.settings.autotune_budget_seconds)

    def test_cute_backend_leaves_full_autotune_unbudgeted_by_default(self) -> None:
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend

        settings = Settings(
            autotune_budget_seconds=None,
            autotune_effort="full",
            autotune_log_level=logging.CRITICAL,
        )
        bound_kernel = SimpleNamespace(settings=settings)
        observed_budgets: list[int | None] = []

        def fake_autotune(self_, bound_kernel_, args, **kwargs):
            observed_budgets.append(bound_kernel_.settings.autotune_budget_seconds)
            return helion.Config()

        with patch.object(
            Backend, "autotune", autospec=True, side_effect=fake_autotune
        ):
            CuteBackend().autotune(bound_kernel, ())

        self.assertEqual(observed_budgets, [None])
        self.assertIsNone(bound_kernel.settings.autotune_budget_seconds)

    def test_cute_backend_restores_default_autotune_budget_on_error(self) -> None:
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend

        settings = Settings(
            autotune_budget_seconds=None,
            autotune_effort="quick",
            autotune_log_level=logging.CRITICAL,
        )
        bound_kernel = SimpleNamespace(settings=settings)

        with (
            patch.object(
                Backend,
                "autotune",
                autospec=True,
                side_effect=RuntimeError("boom"),
            ),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            CuteBackend().autotune(bound_kernel, ())

        self.assertIsNone(bound_kernel.settings.autotune_budget_seconds)

    def test_cute_backend_preserves_explicit_autotune_budget(self) -> None:
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend

        settings = Settings(
            autotune_budget_seconds=42,
            autotune_log_level=logging.CRITICAL,
        )
        bound_kernel = SimpleNamespace(settings=settings)
        observed_budgets: list[int | None] = []

        def fake_autotune(self_, bound_kernel_, args, **kwargs):
            observed_budgets.append(bound_kernel_.settings.autotune_budget_seconds)
            return helion.Config()

        with patch.object(
            Backend, "autotune", autospec=True, side_effect=fake_autotune
        ):
            CuteBackend().autotune(bound_kernel, ())

        self.assertEqual(observed_budgets, [42])
        self.assertEqual(bound_kernel.settings.autotune_budget_seconds, 42)

    def test_setting_from_env_var(self) -> None:
        with patch.dict(
            os.environ,
            {"HELION_AUTOTUNE_BUDGET_SECONDS": "300"},
            clear=False,
        ):
            settings = Settings()
        self.assertEqual(settings.autotune_budget_seconds, 300)

    def test_setting_from_kwarg(self) -> None:
        settings = Settings(autotune_budget_seconds=42)
        self.assertEqual(settings.autotune_budget_seconds, 42)

    def test_no_budget_yields_full_range(self) -> None:
        settings = Settings(
            autotune_budget_seconds=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search._autotune_budget_start = time.perf_counter() - 1e9
        self.assertEqual(list(search._budgeted_range(1, 4)), [1, 2, 3])

    def test_budget_yields_while_time_remains(self) -> None:
        settings = Settings(
            autotune_budget_seconds=600,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        self.assertEqual(list(search._budgeted_range(1, 4)), [1, 2, 3])

    def test_budget_stops_range_when_elapsed_exceeds(self) -> None:
        settings = Settings(
            autotune_budget_seconds=1,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search._autotune_budget_start = time.perf_counter() - 2.0
        search.log = Mock()
        self.assertEqual(list(search._budgeted_range(10)), [])
        search.log.assert_called_once()

    def test_budget_exhaustion_is_agreed_across_distributed_ranks(self) -> None:
        settings = Settings(
            autotune_budget_seconds=600,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search.config_spec.cute_flash_search_enabled = True
        search.kernel.env.process_group_name = "autotune"

        with patch(
            "helion.autotuner.base_search.all_gather_object",
            return_value=[False, True],
        ) as gather:
            self.assertTrue(search._autotune_budget_exceeded_across_ranks())

        gather.assert_called_once_with(False, process_group_name="autotune")

    def test_non_flash_budget_exhaustion_does_not_add_collective(self) -> None:
        settings = Settings(
            autotune_budget_seconds=600,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search.kernel.env.process_group_name = "autotune"

        with patch(
            "helion.autotuner.base_search.all_gather_object",
            side_effect=AssertionError("non-flash search must not add a collective"),
        ):
            self.assertFalse(search._autotune_budget_exceeded_across_ranks())

    def test_budget_unset_when_prepare_not_called(self) -> None:
        settings = Settings(
            autotune_budget_seconds=1,
            autotune_log_level=logging.CRITICAL,
        )
        search = BaseSearch.__new__(BaseSearch)
        search.settings = settings
        search._autotune_budget_start = None
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        self.assertEqual(list(search._budgeted_range(3)), [0, 1, 2])

    def test_budget_resets_when_prepare_called_again(self) -> None:
        settings = Settings(
            autotune_budget_seconds=60,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search._autotune_budget_start = time.perf_counter() - 100.0
        self.assertEqual(list(search._budgeted_range(1)), [])

        search._prepared = False
        with patch.object(
            LocalBenchmarkProvider,
            "_compute_baseline",
            return_value=(None, [], None),
        ):
            search._prepare()
        self.assertEqual(list(search._budgeted_range(1)), [0])

    def test_budget_zero_immediately_exhausts(self) -> None:
        settings = Settings(
            autotune_budget_seconds=0,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        time.sleep(0.001)
        self.assertEqual(list(search._budgeted_range(10)), [])

    def test_setting_has_user_facing_description(self) -> None:
        self.assertIn("autotune_budget_seconds", Settings.__slots__)
        description = Settings.__slots__["autotune_budget_seconds"]
        self.assertIn("HELION_AUTOTUNE_BUDGET_SECONDS", description)
        self.assertIn("best", description.lower())

    def test_generation_loops_use_budgeted_range(self) -> None:
        import inspect

        from helion.autotuner.de_surrogate_hybrid import DESurrogateHybrid
        from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
        from helion.autotuner.llm_search import LLMGuidedSearch
        from helion.autotuner.pattern_search import PatternSearch
        from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch

        for cls in (
            DESurrogateHybrid,
            DifferentialEvolutionSearch,
            LLMGuidedSearch,
            PatternSearch,
            LFBOPatternSearch,
        ):
            source = inspect.getsource(cls)
            self.assertIn(
                "_budgeted_range",
                source,
                f"{cls.__name__} should use _budgeted_range for generation loops",
            )

    def test_finishing_phase_respects_budget(self) -> None:
        import inspect

        from helion.autotuner.base_search import PopulationBasedSearch

        source = inspect.getsource(PopulationBasedSearch.run_finishing_phase)
        self.assertIn(
            "_budgeted_range",
            source,
            "run_finishing_phase should stop when the autotune budget is exhausted",
        )

    def test_final_rebenchmark_respects_budget(self) -> None:
        settings = Settings(
            autotune_budget_seconds=1,
            autotune_log_level=logging.CRITICAL,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search._autotune_budget_start = time.perf_counter() - 2.0
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = []
        member = PopulationMember(lambda: None, [1.0], (), helion.Config())

        with patch.object(search, "rebenchmark") as rebenchmark:
            self.assertIs(search.final_rebenchmark_best(member), member)

        rebenchmark.assert_not_called()

    def test_final_rebenchmark_uses_distributed_flash_budget(self) -> None:
        settings = Settings(
            autotune_budget_seconds=600,
            autotune_log_level=logging.CRITICAL,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search._autotune_budget_start = time.perf_counter()
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = []
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
        search.kernel = SimpleNamespace(
            env=SimpleNamespace(process_group_name="autotune")
        )
        member = PopulationMember(lambda: None, [1.0], (), helion.Config())

        with (
            patch(
                "helion.autotuner.base_search.all_gather_object",
                return_value=[False, True],
            ) as gather,
            patch.object(search, "rebenchmark") as rebenchmark,
        ):
            self.assertIs(search.final_rebenchmark_best(member), member)

        gather.assert_called_once_with(False, process_group_name="autotune")
        rebenchmark.assert_not_called()

    def test_prepare_wires_budget_hook_into_provider(self) -> None:
        """``BaseSearch._prepare`` should install the budget-check hook on
        the benchmark provider so the initial-population
        compile/benchmark phase can short-circuit once the wall-clock
        budget is exhausted.
        """
        settings = Settings(
            autotune_budget_seconds=1,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        self.assertEqual(
            search.benchmark_provider.budget_exceeded_fn,
            search._autotune_budget_exceeded,
        )

    def test_prepare_registers_normalized_compiler_seeds(self) -> None:
        normalized = helion.Config(block_sizes=[64])

        class RecordingProvider(LocalBenchmarkProvider):
            def __init__(self, *, config_spec, **_kwargs) -> None:
                self.config_spec = config_spec

        settings = Settings(autotune_log_level=logging.CRITICAL)
        config_gen = Mock()
        config_gen.seed_flat_config_pairs.return_value = [([64], normalized)]
        config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=3,
            create_config_generation=Mock(return_value=config_gen),
            backend=SimpleNamespace(supports_precompile=lambda: True),
        )
        kernel = SimpleNamespace(
            settings=settings,
            config_spec=config_spec,
            env=SimpleNamespace(process_group_name=None),
        )
        search = BaseSearch(kernel, (), RecordingProvider)

        search._prepare()
        normalized.config["block_sizes"][0] = 128

        self.assertEqual(
            search.benchmark_provider._compiler_seed_configs,
            {helion.Config(block_sizes=[64])},
        )
        search.config_spec.create_config_generation.assert_called_once_with(
            overrides=None,
            advanced_controls_files=None,
            process_group_name=None,
        )

    def _make_stub_provider(self):
        """Construct a minimal ``LocalBenchmarkProvider`` for budget-loop
        tests without standing up a real kernel/config_spec.
        """
        from helion.autotuner.benchmark_provider import LocalBenchmarkProvider

        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.kernel = SimpleNamespace(
            compile_config=lambda config, allow_print: lambda *a, **kw: None,
            format_kernel_decorator=lambda config, s: "decorator",
            env=SimpleNamespace(process_group_name=None),
        )
        provider.settings = Settings(autotune_log_level=logging.CRITICAL)
        # Match the cute backend's path: CuteBackend.supports_precompile()
        # is False, which clears autotune_precompile in autotune setup.
        provider.settings.autotune_precompile = None
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=False,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                should_deduplicate_generated_sources=lambda config_spec: False
            ),
        )
        provider.args = ()
        provider.log = AutotuningLogger(provider.settings)
        provider._autotune_metrics = SimpleNamespace(
            num_configs_tested=0,
            num_compile_failures=0,
            num_worker_failures=0,
            num_accuracy_failures=0,
            num_successful_candidate_measurements=0,
            num_unique_sources=0,
            num_source_deduplications=0,
            num_generations=0,
            kernel_source="",
        )
        provider.mutated_arg_indices = ()
        provider._benchmark_worker = None
        provider._precompile_args_path = None
        provider._precompile_tmpdir = None
        provider._effective_source_hashes = set()
        provider._effective_source_results = {}
        provider._invalid_effective_source_hashes = set()
        provider._pending_effective_source_failures = {}
        provider._effective_source_repairs = {}
        provider._accuracy_failure_config_ids = []
        provider._compile_failure_config_ids = []
        provider._worker_failure_config_ids = []
        provider._compiler_seed_configs = set()
        provider._compiler_seed_source_hashes = set()
        return provider

    def test_cute_flash_benchmark_deduplicates_effective_sources(self) -> None:
        provider = self._make_stub_provider()

        def generated_source_hash(fn):
            return fn.source_hash

        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=generated_source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            def fn():
                return None

            block_size = config.get("block_sizes")[0]
            fn.source_hash = "shared" if block_size in (1, 2, 4) else "unique"
            return fn

        provider.kernel.compile_config = compile_config
        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
            helion.Config(block_sizes=[3]),
        ]
        repeated_config = helion.Config(block_sizes=[4])
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=(1.0, 2.0),
        ) as benchmark:
            results = provider.benchmark(configs)
            repeated_results = provider.benchmark([repeated_config])

        self.assertEqual(benchmark.call_count, 2)
        self.assertEqual([result.perf for result in results], [1.0, 1.0, 2.0])
        self.assertEqual(results[1].status, "deduplicated")
        self.assertIs(results[0].fn, results[1].fn)
        self.assertEqual(repeated_results[0].perf, 1.0)
        self.assertEqual(repeated_results[0].status, "deduplicated")
        self.assertIs(repeated_results[0].config, repeated_config)
        self.assertIs(repeated_results[0].fn, results[0].fn)
        self.assertIsNot(repeated_results[0], results[0])
        self.assertEqual(
            provider._autotune_metrics.num_successful_candidate_measurements, 2
        )
        self.assertEqual(provider._autotune_metrics.num_unique_sources, 2)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 2)

    def test_trial_timestamps_track_measurements_and_new_source_aliases(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec.backend = SimpleNamespace(
            generated_source_hash=lambda fn: fn.source_hash,
            should_deduplicate_generated_sources=lambda config_spec: True,
        )
        clock = [0.0]

        def compile_config(config, allow_print):
            def fn():
                return None

            index = config.block_sizes[0]
            fn.source_hash = "shared" if index in (1, 3) else str(index)
            return fn

        def benchmark(config, fn):
            clock[0] += 10.0
            return math.inf if config.block_sizes[0] == 2 else 1.0

        provider.kernel.compile_config = compile_config
        configs = [helion.Config(block_sizes=[index]) for index in range(1, 5)]
        with (
            patch("time.perf_counter", lambda: clock[0]),
            patch.object(provider, "_benchmark_function", side_effect=benchmark),
            patch.object(provider.log, "register_config", side_effect=repr),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            first = provider.benchmark(configs[:2])
            clock[0] = 100.0
            second = provider.benchmark(configs[2:])

        self.assertEqual([result.status for result in first], ["ok", "error"])
        self.assertEqual([result.completed_at for result in first], [10.0, 20.0])
        self.assertEqual([result.status for result in second], ["deduplicated", "ok"])
        self.assertEqual([result.completed_at for result in second], [100.0, 110.0])
        terminal_entries = {
            call.args[0].config: call.args[0]
            for call in record_entry.call_args_list
            if call.args[0].status != "started"
        }
        for result in [*first, *second]:
            self.assertEqual(
                terminal_entries[result.config].completed_at, result.completed_at
            )

    def test_compile_config_failure_has_unstarted_source_evidence(self) -> None:
        provider = self._make_stub_provider()
        provider.settings.autotune_progress_bar = False
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            block_size = config.get("block_sizes")[0]
            if block_size == 2:
                raise RuntimeError("simulated compile_config failure")

            def fn():
                return None

            fn.source_hash = str(block_size) * 64
            return fn

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            return 1.0

        provider.kernel.compile_config = compile_config
        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
            helion.Config(block_sizes=[3]),
        ]
        failed_config = configs[1]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune"
        metadata = KernelMetadata(kernel_source="source")

        with (
            provider.log.autotune_logging(
                str(base_path), metadata, collect_dataset=True
            ),
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
        ):
            results = provider.benchmark(configs)

        with base_path.with_suffix(".csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        with base_path.with_suffix(".sources.csv").open(newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        metadata_record = json.loads(base_path.with_suffix(".meta.jsonl").read_text())
        failed_id = canonical_config_id(failed_config)

        self.assertEqual([result.status for result in results], ["ok", "error", "ok"])
        self.assertEqual(
            set(metadata_record["configs"]),
            set(map(canonical_config_id, configs)),
        )
        self.assertEqual(len(rows), len(source_rows))
        self.assertEqual(
            [row["status"] for row in rows if row["config_id"] == failed_id],
            ["error"],
        )
        failed_source_hash = next(
            row["source_hash"] for row in source_rows if row["config_id"] == failed_id
        )
        self.assertEqual(
            failed_source_hash, _compile_config_failure_source_hash(failed_config)
        )
        failed_member = PopulationMember(
            results[1].fn,
            [results[1].perf],
            [],
            failed_config,
            status=results[1].status,
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = provider.config_spec
        self.assertEqual(
            search._flash_terminal_member_result(failed_member)["source_hash"],
            failed_source_hash,
        )
        self.assertEqual(
            sum(row["status"] == "started" for row in rows),
            provider._autotune_metrics.num_configs_tested,
        )
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 1)
        self.assertEqual(
            len({row["source_hash"] for row in source_rows}),
            provider._autotune_metrics.num_unique_sources,
        )

    def test_all_compile_failures_after_success_keep_prior_result(self) -> None:
        provider = self._make_stub_provider()
        provider.settings.autotune_progress_bar = False
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        fail_compilation = False

        def compile_config(config, allow_print):
            if fail_compilation:
                raise RuntimeError("simulated terminal compile failure")

            def fn():
                return None

            fn.source_hash = "a" * 64
            return fn

        provider.kernel.compile_config = compile_config
        initial_config = helion.Config(block_sizes=[1])
        failed_configs = [
            helion.Config(block_sizes=[2]),
            helion.Config(block_sizes=[3]),
        ]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune"
        metadata = KernelMetadata(kernel_source="source")

        with (
            provider.log.autotune_logging(
                str(base_path), metadata, collect_dataset=True
            ),
            patch.object(
                LocalBenchmarkProvider, "_benchmark_function", return_value=1.0
            ),
        ):
            initial_result = provider.benchmark([initial_config])
            fail_compilation = True
            failed_results = provider.benchmark(failed_configs)

        with base_path.with_suffix(".csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        with base_path.with_suffix(".sources.csv").open(newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        failed_ids = set(map(canonical_config_id, failed_configs))

        self.assertEqual([result.status for result in initial_result], ["ok"])
        self.assertEqual(
            [result.status for result in failed_results], ["error", "error"]
        )
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 2)
        self.assertEqual(
            {
                row["config_id"]
                for row in rows
                if row["status"] == "error" and row["config_id"] in failed_ids
            },
            failed_ids,
        )
        self.assertFalse(
            any(
                row["status"] == "started" and row["config_id"] in failed_ids
                for row in rows
            )
        )
        self.assertEqual(
            {
                row["source_hash"]
                for row in source_rows
                if row["config_id"] in failed_ids
            },
            {_compile_config_failure_source_hash(config) for config in failed_configs},
        )

    def test_initial_all_compile_failures_still_raise(self) -> None:
        provider = self._make_stub_provider()
        provider.settings.autotune_progress_bar = False
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            raise RuntimeError("simulated initial compile failure")

        provider.kernel.compile_config = compile_config
        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
        ]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune"
        metadata = KernelMetadata(kernel_source="source")

        with (
            provider.log.autotune_logging(
                str(base_path), metadata, collect_dataset=True
            ),
            self.assertRaisesRegex(RuntimeError, "simulated initial compile failure"),
        ):
            provider.benchmark(configs)

        with base_path.with_suffix(".csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        failed_ids = set(map(canonical_config_id, configs))
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 2)
        self.assertEqual(
            {row["config_id"] for row in rows if row["status"] == "error"},
            failed_ids,
        )

    def test_invalidated_effective_source_rejects_later_alias(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            def fn():
                return None

            fn.source_hash = "shared"
            return fn

        provider.kernel.compile_config = compile_config
        representative = helion.Config(block_sizes=[1])
        later_alias = helion.Config(block_sizes=[2])
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            return_value=1.0,
        ) as benchmark:
            first = provider.benchmark([representative])
            provider.invalidate_effective_source_hash("shared")
            alias = provider.benchmark([later_alias])

        benchmark.assert_called_once()
        self.assertEqual(first[0].status, "ok")
        self.assertEqual(alias[0].status, "source_rejected")
        self.assertEqual(alias[0].perf, math.inf)
        self.assertFalse(provider.has_measured_source_hash("shared"))

    def test_same_batch_source_alias_is_not_logged_as_benchmarked(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        representative = helion.Config(block_sizes=[1])
        alias = helion.Config(block_sizes=[2])

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            return 1.0

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=lambda config: (
                    "representative" if config is representative else "alias"
                ),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            results = provider.benchmark([representative, alias])

        entries = [call.args[0] for call in record_entry.call_args_list]
        self.assertEqual([result.status for result in results], ["ok", "deduplicated"])
        self.assertEqual(
            [entry.status for entry in entries if entry.config is representative],
            ["started", "ok"],
        )
        self.assertEqual(
            [entry.status for entry in entries if entry.config is alias],
            ["deduplicated"],
        )
        self.assertEqual(provider._autotune_metrics.num_configs_tested, 1)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 1)

    def test_compiler_seed_alias_marks_source_for_timeout_retry(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=3,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        representative = helion.Config(block_sizes=[1])
        compiler_seed_alias = helion.Config(block_sizes=[2])
        provider.set_compiler_seed_configs([compiler_seed_alias])

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            return_value=1.0,
        ) as benchmark:
            results = provider.benchmark([representative, compiler_seed_alias])

        self.assertEqual([result.perf for result in results], [1.0, 1.0])
        benchmark.assert_called_once_with(
            representative,
            results[0].fn,
            effective_source_hash="shared",
        )

    def test_compiler_seed_source_provenance_survives_batches(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=3,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        compiler_seed = helion.Config(block_sizes=[1])
        later_alias = helion.Config(block_sizes=[2])
        provider.set_compiler_seed_configs([compiler_seed])

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=(math.inf, 1.0),
        ) as benchmark:
            first = provider.benchmark([compiler_seed])
            second = provider.benchmark([later_alias])

        self.assertEqual(first[0].perf, math.inf)
        self.assertEqual(second[0].perf, 1.0)
        self.assertEqual(
            benchmark.call_args_list[1],
            unittest.mock.call(
                later_alias,
                second[0].fn,
                effective_source_hash="shared",
            ),
        )

    def test_disabled_compiler_seed_retry_does_not_register_seeds(self) -> None:
        provider = self._make_stub_provider()
        seed = helion.Config(block_sizes=[1])

        provider.set_compiler_seed_configs([seed])

        self.assertFalse(provider._is_compiler_seed_config(seed))
        self.assertEqual(provider._compiler_seed_configs, set())

    def test_effective_source_aliases_remain_positional_after_budget_expiry(
        self,
    ) -> None:
        provider = self._make_stub_provider()

        def generated_source_hash(fn):
            return fn.source_hash

        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=generated_source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            def fn():
                return None

            block_size = config.get("block_sizes")[0]
            fn.source_hash = "first" if block_size == 1 else "shared-tail"
            return fn

        provider.kernel.compile_config = compile_config
        benchmark_count = [0]
        provider.set_budget_exceeded_fn(lambda: benchmark_count[0] >= 1)

        def counting_benchmark(self_, config, fn):
            benchmark_count[0] += 1
            return 1.0

        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
            helion.Config(block_sizes=[3]),
        ]
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            new=counting_benchmark,
        ):
            results = provider.benchmark(configs)

        self.assertEqual([result.perf for result in results], [1.0, math.inf, math.inf])
        self.assertEqual(
            [result.status for result in results], ["ok", "error", "error"]
        )
        self.assertEqual([result.config for result in results], configs)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 0)

    def test_backend_policy_can_disable_effective_source_dedup(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=False,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: False,
            ),
        )
        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
        ]
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=(1.0, 2.0),
        ) as benchmark:
            results = provider.benchmark(configs)

        self.assertEqual(benchmark.call_count, 2)
        self.assertEqual([result.perf for result in results], [1.0, 2.0])
        self.assertEqual(provider._autotune_metrics.num_unique_sources, 0)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 0)

    def test_failed_effective_source_is_retried(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        configs = [helion.Config(block_sizes=[1])]
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=(math.inf, 1.0),
        ) as benchmark:
            first = provider.benchmark(configs)
            second = provider.benchmark(configs)

        self.assertEqual(benchmark.call_count, 2)
        self.assertEqual(first[0].status, "error")
        self.assertEqual(second[0].status, "ok")

    def test_failed_effective_source_retries_alias_independently(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        representative = helion.Config(block_sizes=[1])
        alias = helion.Config(block_sizes=[2])

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            if config is representative:
                provider._record_worker_failure(config, "timeout")
                return math.inf
            return 1.0

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ) as benchmark_mock,
            patch.object(
                provider.log,
                "register_config",
                side_effect=lambda config: (
                    "representative" if config is representative else "alias"
                ),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            results = provider.benchmark([representative, alias])

        entries = [call.args[0] for call in record_entry.call_args_list]
        self.assertEqual(benchmark_mock.call_count, 2)
        self.assertEqual([result.status for result in results], ["deduplicated", "ok"])
        self.assertEqual(
            [entry.status for entry in entries if entry.config == representative],
            ["started", "timeout", "deduplicated"],
        )
        self.assertEqual(
            [entry.status for entry in entries if entry.config is alias],
            ["started", "ok"],
        )
        self.assertEqual(
            [entry.perf_ms for entry in entries if entry.status == "deduplicated"],
            [1.0],
        )
        self.assertEqual(
            {entry.source_hash for entry in entries},
            {"shared"},
        )
        self.assertEqual(provider._autotune_metrics.num_configs_tested, 2)
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 1)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 1)

    def test_equal_failed_configs_each_receive_source_repair(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        configs = [helion.Config(block_sizes=[1]) for _ in range(3)]
        calls = 0

        def benchmark(config, fn):
            nonlocal calls
            calls += 1
            provider._autotune_metrics.num_configs_tested += 1
            if calls < 3:
                provider._record_worker_failure(config, "timeout")
                return math.inf
            return 1.0

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=("first", "second", "third"),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            results = provider.benchmark(configs)

        self.assertEqual(
            [result.status for result in results],
            ["deduplicated", "deduplicated", "ok"],
        )
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 2)
        self.assertEqual(
            sum(
                call.args[0].status == "deduplicated"
                for call in record_entry.call_args_list
            ),
            2,
        )

    def test_later_batch_repairs_failed_effective_source(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        representative = helion.Config(block_sizes=[1])
        alias = helion.Config(block_sizes=[2])

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            if config is representative:
                provider._record_worker_failure(config, "timeout")
                return math.inf
            return 1.0

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=("representative", "alias"),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            provider._autotune_metrics.num_generations = 3
            failed = provider.benchmark([representative])
            provider._autotune_metrics.num_generations = 4
            succeeded = provider.benchmark([alias])

        repairs = provider.take_effective_source_repairs()
        representative_entries = [
            call.args[0]
            for call in record_entry.call_args_list
            if call.args[0].config == representative
        ]
        self.assertEqual(failed[0].status, "timeout")
        self.assertEqual(succeeded[0].status, "ok")
        self.assertEqual(repairs[representative].status, "deduplicated")
        self.assertEqual(repairs[representative].perf, 1.0)
        self.assertEqual(
            [entry.status for entry in representative_entries],
            ["started", "timeout", "deduplicated"],
        )
        self.assertEqual(
            [entry.generation for entry in representative_entries],
            [3, 3, 4],
        )
        alias_entries = [
            call.args[0]
            for call in record_entry.call_args_list
            if call.args[0].config == alias
        ]
        self.assertEqual(
            [(entry.status, entry.generation) for entry in alias_entries],
            [("started", 4), ("ok", 4)],
        )

    def test_lfbo_source_repair_updates_prior_surrogate_target(self) -> None:
        config = helion.Config(block_sizes=[1])
        member = PopulationMember(
            fn=lambda: None,
            perfs=[math.inf],
            flat_values=[],
            config=config,
            status="timeout",
        )
        replacement_fn = Mock()
        replacement_fn.source_hash = "shared"
        repair = SimpleNamespace(
            config=config,
            fn=replacement_fn,
            perf=1.0,
            status="deduplicated",
            compile_time=None,
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.population = [member]
        search._compiler_seed_members = []
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: getattr(fn, "source_hash", None)
            )
        )
        search.train_configs = [copy.deepcopy(config)]
        search.train_source_hashes = [None]
        search.train_y = [math.inf]

        search._apply_effective_source_repairs({config: repair}, [])

        self.assertEqual(member.perfs, [1.0])
        self.assertIs(member.fn, replacement_fn)
        self.assertEqual(member.status, "deduplicated")
        self.assertEqual(search.train_y, [1.0])
        self.assertEqual(search.train_source_hashes, ["shared"])

    def test_completed_benchmark_is_logged_before_later_exception(self) -> None:
        provider = self._make_stub_provider()
        first = helion.Config(block_sizes=[1])
        second = helion.Config(block_sizes=[2])

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=(1.0, RuntimeError("boom")),
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=("first", "second"),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            provider.benchmark([first, second])

        first_entries = [
            call.args[0]
            for call in record_entry.call_args_list
            if call.args[0].config is first
        ]
        self.assertEqual([entry.status for entry in first_entries], ["started", "ok"])
        self.assertEqual(first_entries[-1].perf_ms, 1.0)

    def test_failed_benchmark_is_logged_before_later_exception(self) -> None:
        provider = self._make_stub_provider()
        first = helion.Config(block_sizes=[1])
        second = helion.Config(block_sizes=[2])

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            if config is first:
                provider._record_worker_failure(config, "timeout")
                return math.inf
            raise RuntimeError("boom")

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=("first", "second"),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            provider.benchmark([first, second])

        first_entries = [
            call.args[0]
            for call in record_entry.call_args_list
            if call.args[0].config is first
        ]
        self.assertEqual(
            [entry.status for entry in first_entries], ["started", "timeout"]
        )

    def test_accuracy_failure_invalidates_effective_source_aliases(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        invalid = helion.Config(block_sizes=[1])
        alias = helion.Config(block_sizes=[2])

        def fail_accuracy(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            provider._record_accuracy_failure(config)
            return math.inf

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=fail_accuracy,
            ) as benchmark,
            patch.object(
                provider.log,
                "register_config",
                side_effect=lambda config: "invalid" if config is invalid else "alias",
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            results = provider.benchmark([invalid, alias])

        entries = [call.args[0] for call in record_entry.call_args_list]
        benchmark.assert_called_once_with(invalid, results[0].fn)
        self.assertEqual(
            [result.status for result in results],
            ["accuracy_error", "source_rejected"],
        )
        self.assertEqual([result.perf for result in results], [math.inf, math.inf])
        self.assertEqual(
            [entry.status for entry in entries if entry.config is invalid],
            ["started", "accuracy_error"],
        )
        self.assertEqual(
            [entry.status for entry in entries if entry.config is alias],
            ["source_rejected"],
        )
        self.assertEqual(provider._autotune_metrics.num_configs_tested, 1)
        self.assertEqual(provider._autotune_metrics.num_accuracy_failures, 1)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 1)
        self.assertEqual(provider._invalid_effective_source_hashes, {"shared"})

    def test_accuracy_status_remains_error_when_source_dedup_is_disabled(self) -> None:
        provider = self._make_stub_provider()
        config = helion.Config(block_sizes=[1])

        def fail_accuracy(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            provider._record_accuracy_failure(config)
            return math.inf

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=fail_accuracy,
        ):
            result = provider.benchmark([config])[0]

        self.assertEqual(result.status, "error")
        self.assertEqual(provider._autotune_metrics.num_accuracy_failures, 1)
        self.assertEqual(provider._invalid_effective_source_hashes, set())

    def test_prior_accuracy_failure_does_not_poison_later_source(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        source_hashes = iter(("inaccurate", "timed-out"))

        def compile_config(config, allow_print):
            fn = Mock()
            fn.source_hash = next(source_hashes)
            return fn

        provider.kernel.compile_config = compile_config
        config = helion.Config(block_sizes=[1])
        calls = 0

        def fail_differently(config, fn):
            nonlocal calls
            calls += 1
            provider._autotune_metrics.num_configs_tested += 1
            if calls == 1:
                provider._record_accuracy_failure(config)
            else:
                provider._record_worker_failure(config, "timeout")
            return math.inf

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=fail_differently,
        ):
            first = provider.benchmark([config])[0]
            second = provider.benchmark([config])[0]

        self.assertEqual(first.status, "accuracy_error")
        self.assertEqual(second.status, "timeout")
        self.assertEqual(provider._invalid_effective_source_hashes, {"inaccurate"})

    def test_effective_source_dedup_is_disabled_for_distributed(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                should_deduplicate_generated_sources=lambda config_spec: True
            )
        )
        with patch(
            "helion.autotuner.benchmark_provider.dist.is_initialized",
            return_value=True,
        ):
            self.assertFalse(provider._effective_source_dedup_enabled())

    def test_cute_backend_reports_attached_generated_source_hash(self) -> None:
        from helion._compiler.backend import CuteBackend

        backend = CuteBackend()
        compiled = SimpleNamespace(
            __name__="kernel",
            __globals__={
                "_helion_kernel": SimpleNamespace(_helion_cute_source_hash="source-abc")
            },
        )
        self.assertEqual(backend.generated_source_hash(compiled), "source-abc")
        self.assertIsNone(backend.generated_source_hash(SimpleNamespace()))
        self.assertTrue(
            backend.should_deduplicate_generated_sources(_cute_flash_test_config_spec())
        )
        self.assertFalse(
            backend.should_deduplicate_generated_sources(
                SimpleNamespace(cute_flash_search_enabled=False)
            )
        )

    def test_benchmark_provider_short_circuits_compile_loop(self) -> None:
        """``LocalBenchmarkProvider.benchmark`` must stop compiling
        configs once ``budget_exceeded_fn`` returns ``True`` from inside
        the compile loop, and still return one ``BenchmarkResult`` per
        input config so callers receive a positionally-aligned list.
        """
        from helion.autotuner.benchmark_provider import BenchmarkResult
        from helion.autotuner.benchmark_provider import LocalBenchmarkProvider

        provider = self._make_stub_provider()

        compiled_count = [0]
        budget_calls = [0]

        original_compile = provider.kernel.compile_config

        def counting_compile(config, allow_print):
            compiled_count[0] += 1
            return original_compile(config, allow_print)

        provider.kernel.compile_config = counting_compile

        def budget_check():
            budget_calls[0] += 1
            return compiled_count[0] >= 2

        provider.set_budget_exceeded_fn(budget_check)

        from helion.runtime.config import Config

        configs = [Config() for _ in range(5)]

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            return_value=0.001,
        ):
            results = provider.benchmark(configs)

        self.assertEqual(len(results), len(configs))
        compiled_finite = sum(1 for r in results if math.isfinite(r.perf))
        self.assertLessEqual(compiled_count[0], 2)
        self.assertLessEqual(compiled_finite, compiled_count[0])
        for r in results[compiled_count[0] :]:
            self.assertEqual(r.status, "error")
            self.assertEqual(r.perf, float("inf"))
        for r in results:
            self.assertIsInstance(r, BenchmarkResult)
        # The hook must actually fire; without it the loop wouldn't break.
        self.assertGreater(budget_calls[0], 0)

    def test_benchmark_provider_short_circuits_benchmark_loop(self) -> None:
        """If compilation finishes cleanly and the budget only fires
        partway through the benchmark loop, remaining configs must be
        left at the default ``perf=inf, status="error"`` slots while the
        earlier benchmarked configs keep their measured perf.
        """
        from helion.autotuner.benchmark_provider import BenchmarkResult
        from helion.autotuner.benchmark_provider import LocalBenchmarkProvider

        provider = self._make_stub_provider()

        # All compiles succeed; budget stays clear during compilation.
        benchmark_count = [0]

        def budget_check():
            # Trip the budget after 2 benchmark calls.
            return benchmark_count[0] >= 2

        provider.set_budget_exceeded_fn(budget_check)

        def counting_benchmark(self_, config, fn):
            benchmark_count[0] += 1
            return 0.001

        from helion.runtime.config import Config

        configs = [Config() for _ in range(5)]

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            new=counting_benchmark,
        ):
            results = provider.benchmark(configs)

        self.assertEqual(len(results), len(configs))
        # At most 2 benchmarks ran (the loop checks before each call).
        self.assertLessEqual(benchmark_count[0], 2)
        # First few entries have finite measured perf.
        finite_count = sum(1 for r in results if math.isfinite(r.perf))
        self.assertEqual(finite_count, benchmark_count[0])
        # The tail entries must be the inf/error defaults.
        for r in results[benchmark_count[0] :]:
            self.assertEqual(r.status, "error")
            self.assertEqual(r.perf, float("inf"))
        for r in results:
            self.assertIsInstance(r, BenchmarkResult)

    def test_benchmark_provider_default_hook_is_no_op(self) -> None:
        """A provider that never had its hook installed must read the
        class-level no-op default rather than raising ``AttributeError``
        on the first ``budget_exceeded_fn()`` call.
        """
        from helion.autotuner.benchmark_provider import BenchmarkProvider
        from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
        from helion.autotuner.benchmark_provider import _never_exceeded

        # Class-level default lives on the abstract base so any subclass
        # picks it up even if it forgets to call super().__init__.
        self.assertIs(BenchmarkProvider.budget_exceeded_fn, _never_exceeded)
        # Subclass instance with no __init__ work still resolves the
        # default via class-level descriptor.
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        self.assertFalse(provider.budget_exceeded_fn())

    def test_cute_benchmark_uses_event_timed_subprocess_worker(self) -> None:
        # CuTe inherits the default (None) do_bench hooks: compiled launchers
        # run on the Torch current stream, so CUDA-event timing is accurate
        # and subprocess benchmark jobs must NOT switch to the wall-clock
        # path that charges per-launch host overhead to the kernel.
        from helion._compiler.backend import CuteBackend

        backend = CuteBackend()
        self.assertIsNone(backend.get_do_bench())
        self.assertIsNone(backend.get_interleaved_bench())

        provider = self._make_stub_provider()
        provider.kernel.supports_subprocess_benchmark = lambda: True
        provider.config_spec = SimpleNamespace(backend=backend)

        self.assertTrue(provider._subprocess_benchmark_enabled())
        self.assertFalse(provider._subprocess_benchmark_uses_wall_clock())

    def test_non_cute_custom_benchmark_stays_in_process(self) -> None:
        from helion.autotuner.benchmarking import do_bench_generic

        class OtherBackend:
            @property
            def name(self) -> str:
                return "other"

            def get_do_bench(self):
                return do_bench_generic

        provider = self._make_stub_provider()
        provider.kernel.supports_subprocess_benchmark = lambda: True
        provider.config_spec = SimpleNamespace(backend=OtherBackend())

        self.assertFalse(provider._subprocess_benchmark_enabled())
        self.assertFalse(provider._subprocess_benchmark_uses_wall_clock())

    def test_mtia_device_disables_subprocess_benchmark(self) -> None:
        # The MTIA runtime reserves the whole device for its first client,
        # so a spawned worker cannot open it while this process holds it.
        from helion._compiler.backend import CuteBackend

        backend = CuteBackend()
        self.assertIsNone(backend.get_do_bench())

        provider = self._make_stub_provider()
        provider.kernel.supports_subprocess_benchmark = lambda: True
        provider.kernel.env = SimpleNamespace(
            process_group_name=None, device=SimpleNamespace(type="mtia")
        )
        provider.config_spec = SimpleNamespace(backend=backend)

        self.assertFalse(provider._subprocess_benchmark_enabled())

    def test_missing_device_env_does_not_break_subprocess_check(self) -> None:
        # Stub kernels mock env without a device; the MTIA carve-out must
        # tolerate that instead of raising AttributeError.
        from helion._compiler.backend import CuteBackend

        provider = self._make_stub_provider()
        provider.kernel.supports_subprocess_benchmark = lambda: True
        provider.config_spec = SimpleNamespace(backend=CuteBackend())

        self.assertTrue(provider._subprocess_benchmark_enabled())


@onlyBackends(["triton", "cute"])
class TestAutotuneRandomSeed(RefEagerTestDisabled, TestCase):
    def _autotune_and_record(self, **settings: object) -> float:
        search_capture: dict[str, RecordingRandomSearch] = {}

        def autotuner_factory(bound_kernel, args, **kwargs):
            search = RecordingRandomSearch(bound_kernel, args, count=2, **kwargs)
            search_capture["search"] = search
            return search

        kernel_settings = {
            "autotuner_fn": autotuner_factory,
        }
        kernel_settings.update(settings)

        @helion.kernel(**kernel_settings)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound_kernel = add.bind(args)
        bound_kernel.autotune(args)
        torch.testing.assert_close(bound_kernel(*args), sum(args), rtol=1e-2, atol=1e-1)

        search = search_capture["search"]
        assert search.samples, (
            "expected RecordingRandomSearch to record a random sample"
        )
        return search.samples[0]

    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_autotune_random_seed_from_env_var(self) -> None:
        # same env var value -> same random sample
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "4242"}, clear=False
        ):
            first = self._autotune_and_record()
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "4242"}, clear=False
        ):
            second = self._autotune_and_record()
        self.assertEqual(first, second)

        # different env var values -> different random samples
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "101"}, clear=False
        ):
            first = self._autotune_and_record()
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "102"}, clear=False
        ):
            second = self._autotune_and_record()
        self.assertNotEqual(first, second)

    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_autotune_random_seed_from_settings(self) -> None:
        # same autotune_random_seed setting -> same random sample
        first = self._autotune_and_record(autotune_random_seed=4242)
        second = self._autotune_and_record(autotune_random_seed=4242)
        self.assertEqual(first, second)

        # different autotune_random_seed settings -> different random samples
        first = self._autotune_and_record(autotune_random_seed=101)
        second = self._autotune_and_record(autotune_random_seed=102)
        self.assertNotEqual(first, second)


@onlyBackends(["triton", "cute"])
class TestAutotuneSeedConfigs(TestCase):
    """Tests for seeding initial autotune populations with user configs."""

    def _seed_config(self) -> helion.Config:
        if _get_backend() == "cute":
            return helion.Config(num_threads=[32])
        return helion.Config(num_warps=8)

    def _has_seed_config(self, configs: list[helion.Config]) -> bool:
        if _get_backend() == "cute":
            return any(config.num_threads == [32] for config in configs)
        return any(config.num_warps == 8 for config in configs)

    def _make_kernel_and_args(self, **kernel_kwargs):
        @helion.kernel(autotune_log_level=0, **kernel_kwargs)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )
        return add, args

    def _population_configs(self, search: PatternSearch) -> list[helion.Config]:
        return [
            search.config_gen.unflatten(flat)
            for flat in search._generate_initial_population_flat()
        ]

    def test_decorator_accepts_single_seed_config(self) -> None:
        seed_config = self._seed_config()
        add, _args = self._make_kernel_and_args(autotune_seed_configs=seed_config)

        self.assertEqual(add.settings.autotune_seed_configs, seed_config)
        self.assertEqual(add.configs, [])

    def test_random_initial_population_includes_seed_configs(self) -> None:
        seed_config = self._seed_config()
        add, args = self._make_kernel_and_args(autotune_seed_configs=[seed_config])
        bound = add.bind(args)
        search = PatternSearch(bound, args, initial_population=3)

        configs = self._population_configs(search)

        self.assertGreaterEqual(len(configs), 3)
        self.assertTrue(self._has_seed_config(configs))

    def test_best_available_initial_population_includes_seed_configs(self) -> None:
        seed_config = self._seed_config()
        add, args = self._make_kernel_and_args(autotune_seed_configs=[seed_config])
        bound = add.bind(args)
        search = PatternSearch(
            bound,
            args,
            initial_population_strategy=InitialPopulationStrategy.FROM_BEST_AVAILABLE,
        )

        with patch.object(BaseSearch, "_find_similar_cached_configs", return_value=[]):
            configs = self._population_configs(search)

        self.assertGreaterEqual(len(configs), 2)
        self.assertTrue(self._has_seed_config(configs))

    def test_random_initial_population_logs_invalid_seed_configs(self) -> None:
        seed_config = helion.Config.from_dict({"block_sizes": ["bad"]})
        add, args = self._make_kernel_and_args(autotune_seed_configs=[seed_config])
        bound = add.bind(args)
        search = PatternSearch(bound, args, initial_population=3)
        search.log = Mock()

        configs = self._population_configs(search)

        self.assertGreaterEqual(len(configs), 3)
        search.log.assert_called_once()
        self.assertIn(
            "Failed to transfer autotune seed config 1", search.log.call_args[0][0]
        )


@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
@onlyBackends(["triton", "cute"])
class TestConfigFilter(TestCase):
    """Tests for the autotune_config_filter setting."""

    def _make_kernel_and_args(self, **kernel_kwargs):
        @helion.kernel(autotune_log_level=0, **kernel_kwargs)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )
        return add, args

    def test_finite_search_accepts_benchmark_provider_factory(self) -> None:
        class CustomBenchmarkProvider(LocalBenchmarkProvider):
            def __init__(
                self, *args: object, context: object, **kwargs: object
            ) -> None:
                self.context = context
                super().__init__(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]

        configs = [
            helion.Config(block_sizes=[16], num_warps=4),
            helion.Config(block_sizes=[32], num_warps=4),
        ]
        add, args = self._make_kernel_and_args()
        context = object()
        provider_factory = functools.partial(
            CustomBenchmarkProvider,
            context=context,
        )
        search = FiniteSearch(
            add.bind(args),
            args,
            configs=configs,
            benchmark_provider_cls=provider_factory,
        )
        search._prepare()

        self.assertIsInstance(search.benchmark_provider, CustomBenchmarkProvider)
        self.assertIs(search.benchmark_provider.context, context)

    def test_autotune_config_filter_skips_filtered_configs(self) -> None:
        """Filtered configs produce status='filtered' and perf=inf."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=4)
        cfg3 = helion.Config(block_sizes=[64], num_warps=4)

        filtered_out: list[helion.Config] = []

        def my_filter(config: helion.Config) -> helion.Config | None:
            if config.get("block_sizes") == [32]:
                filtered_out.append(config)
                return None
            return config

        add, args = self._make_kernel_and_args(
            autotune_config_filter=my_filter, autotune_precompile=None
        )
        bound = add.bind(args)
        search = FiniteSearch(bound, args, configs=[cfg1, cfg2, cfg3])
        search._prepare()
        results = search.benchmark_batch([cfg1, cfg2, cfg3])

        # cfg2 should be filtered
        self.assertEqual(len(filtered_out), 1)
        self.assertEqual(filtered_out[0].get("block_sizes"), [32])

        statuses = {tuple(r.config.get("block_sizes", [])): r.status for r in results}
        self.assertEqual(statuses[(16,)], "ok")
        self.assertEqual(statuses[(32,)], "filtered")
        self.assertEqual(statuses[(64,)], "ok")

        perfs = {tuple(r.config.get("block_sizes", [])): r.perf for r in results}
        self.assertEqual(perfs[(32,)], float("inf"))

    def test_autotune_config_filter_affects_autotune_winner(self) -> None:
        """The autotuner never picks a filtered config as the winner."""
        # cfg_fast would normally win (smallest block = least work per kernel launch
        # in this trivial test), but we filter it out.
        cfg_fast = helion.Config(block_sizes=[16], num_warps=4)
        cfg_slow = helion.Config(block_sizes=[128], num_warps=4)

        def reject_small_blocks(config: helion.Config) -> helion.Config | None:
            return config if (config.get("block_sizes") or [0])[0] >= 64 else None

        add, args = self._make_kernel_and_args(
            autotune_config_filter=reject_small_blocks,
            # Filtering happens before compile/benchmark; skip the benchmark
            # worker subprocess (seconds of startup overhead).
            autotune_benchmark_subprocess=False,
        )
        bound = add.bind(args)
        search = FiniteSearch(bound, args, configs=[cfg_fast, cfg_slow])
        winner = search.autotune()
        # cfg_fast is filtered out, so cfg_slow must win
        self.assertEqual(winner.get("block_sizes"), [128])

    def test_autotune_config_filter_none_is_noop(self) -> None:
        """When autotune_config_filter=None (default), all configs are benchmarked normally."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=4)

        add, args = self._make_kernel_and_args(
            autotune_precompile=None
        )  # no autotune_config_filter
        bound = add.bind(args)
        search = FiniteSearch(bound, args, configs=[cfg1, cfg2])
        search._prepare()
        results = search.benchmark_batch([cfg1, cfg2])

        for result in results:
            self.assertNotEqual(result.status, "filtered")
            self.assertFalse(math.isinf(result.perf))

    def test_autotune_config_filter_can_override_config(self) -> None:
        """autotune_config_filter can return a modified Config to override values before benchmarking."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=4)

        def override_num_warps(config: helion.Config) -> helion.Config | None:
            # Override num_warps to 2 for all configs
            return helion.Config.from_dict({**config.config, "num_warps": 2})

        add, args = self._make_kernel_and_args(
            autotune_config_filter=override_num_warps, autotune_precompile=None
        )
        bound = add.bind(args)
        search = FiniteSearch(bound, args, configs=[cfg1, cfg2])
        search._prepare()
        results = search.benchmark_batch([cfg1, cfg2])

        # All configs should run successfully (none filtered)
        for result in results:
            self.assertNotEqual(result.status, "filtered")
            self.assertFalse(math.isinf(result.perf))
        # The result configs should reflect the overridden values
        self.assertEqual(results[0].config.get("num_warps"), 2)
        self.assertEqual(results[1].config.get("num_warps"), 2)


class TestForcedAutotuneSeedCachePolicy(unittest.TestCase):
    @staticmethod
    def _search(*, cute_flash: bool) -> BaseSearch:
        search = BaseSearch.__new__(BaseSearch)
        search._skip_cache = True
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=cute_flash,
            cache_fingerprint_hash=Mock(return_value="spec"),
        )
        search.settings = SimpleNamespace(
            autotune_search_acf=None,
            autotune_best_available_max_cache_scan=16,
        )
        search.log = Mock()
        search._get_current_hardware_and_specialization = Mock(
            return_value=("gpu", "specialization")
        )
        return search

    def test_force_bypasses_warm_start_only_for_cute_flash(self) -> None:
        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=False),
            patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path("/cache"),
            ),
            patch(
                "helion.autotuner.local_cache.iter_cache_entries",
                return_value=iter(()),
            ) as scan,
            patch(
                "helion.autotuner.remote_cache._load_remote_backend_if_configured",
                return_value=None,
            ),
        ):
            self.assertEqual(
                self._search(cute_flash=False)._find_similar_cached_configs(4), []
            )
            scan.assert_called_once_with(Path("/cache"), max_scan=16)

            scan.reset_mock()
            self.assertEqual(
                self._search(cute_flash=True)._find_similar_cached_configs(4), []
            )
            scan.assert_not_called()


@onlyBackends(["triton", "cute"])
class TestAutotuneCacheSelection(TestCase):
    """Selection of the autotune cache via HELION_AUTOTUNE_CACHE."""

    def _make_bound(self):
        @helion.kernel(autotune_baseline_fn=operator.add, autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8], device=DEVICE),
            torch.randn([8], device=DEVICE),
        )
        return add.bind(args), args

    def test_autotune_cache_default_is_local(self):
        """Default (no env var set) -> LocalAutotuneCache."""
        with without_env_var("HELION_AUTOTUNE_CACHE"):
            bound, args = self._make_bound()
            with patch("torch.accelerator.synchronize", autospec=True) as sync:
                sync.return_value = None
                autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertIsInstance(autotuner, LocalAutotuneCache)
            self.assertNotIsInstance(autotuner, StrictLocalAutotuneCache)

    def test_autotune_cache_strict_selected_by_env(self):
        """HELION_AUTOTUNE_CACHE=StrictLocalAutotuneCache -> StrictLocalAutotuneCache."""
        with patch.dict(
            os.environ,
            {"HELION_AUTOTUNE_CACHE": "StrictLocalAutotuneCache"},
            clear=False,
        ):
            bound, args = self._make_bound()
            with patch("torch.accelerator.synchronize", autospec=True) as sync:
                sync.return_value = None
                autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertIsInstance(autotuner, StrictLocalAutotuneCache)

    def test_autotune_cache_invalid_raises(self):
        """Invalid HELION_AUTOTUNE_CACHE value should raise a ValueError."""
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_CACHE": "InvalidCacheName"}, clear=False
        ):
            bound, args = self._make_bound()
            with patch("torch.accelerator.synchronize", autospec=True) as sync:
                sync.return_value = None
                with self.assertRaisesRegex(
                    ValueError, "Unknown HELION_AUTOTUNE_CACHE"
                ):
                    bound.settings.autotuner_fn(bound, args)


if __name__ == "__main__":
    unittest.main()
