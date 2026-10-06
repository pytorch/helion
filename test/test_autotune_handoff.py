from __future__ import annotations

from collections import defaultdict
from collections import deque
import csv
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast
import unittest
from unittest.mock import patch

from helion import Config
from helion import Settings
from helion import exc
from helion.autotuner.base_cache import AutotuneCacheBase
from helion.autotuner.base_cache import CacheKeyBase
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import _AutotunableKernel
from helion.autotuner.benchmark_provider import BenchmarkProvider
from helion.autotuner.benchmark_provider import BenchmarkResult
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.handoff import HandoffPolicy
from helion.autotuner.handoff import HandoffProgress
from helion.autotuner.handoff import find_handoff
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.metrics import KernelMetadata
from helion.autotuner.search_space_logger import canonical_config_id

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from helion.autotuner.logger import AutotuningLogger

_Status = Literal["ok", "error", "timeout", "accuracy_error"]


def _config(index: int) -> Config:
    return Config(block_sizes=[index])


def _index(config: Config) -> int:
    return config.block_sizes[0]


class _Compiled:
    def __init__(self, index: int, source: str) -> None:
        self.index = index
        self.source_hash = source

    def __call__(self, *args: object) -> int:
        return self.index


class _Backend:
    def generated_source_hash(self, fn: object) -> str | None:
        return cast("_Compiled", fn).source_hash

    def autotune_config_is_viable(self, spec: object, config: Config) -> bool:
        return True


class _Spec:
    backend = _Backend()
    cute_flash_search_enabled = False

    def normalize(self, config: Config) -> None:
        pass


class _Script:
    def __init__(
        self,
        timings: dict[int, list[float | tuple[float, str]]],
        *,
        sources: dict[int, str] | None = None,
    ) -> None:
        self.timings = {index: deque(values) for index, values in timings.items()}
        self.sources = sources or {}
        self.providers: list[_Provider] = []
        self.calls: dict[int, int] = defaultdict(int)
        self.after_benchmark: Callable[[], None] | None = None

    def next(self, index: int) -> tuple[float, str]:
        self.calls[index] += 1
        values = self.timings[index]
        value = values.popleft() if len(values) > 1 else values[0]
        return value if isinstance(value, tuple) else (value, "ok")


class _Kernel:
    def __init__(self, script: _Script, settings: Settings | None = None) -> None:
        self.script = script
        self.settings = settings or Settings(
            backend="triton",
            autotune_log_level=0,
            autotune_log_details=False,
            autotune_log_search_space=False,
            autotune_log_search_space_verbose=False,
        )
        self.env = SimpleNamespace(process_group_name=None)
        self.config_spec = _Spec()
        self.kernel = SimpleNamespace(name="handoff_test")
        self.configs: tuple[Config, ...] = ()

    def compile_config(self, config: Config, *, allow_print: bool = True) -> _Compiled:
        index = _index(config)
        return _Compiled(index, self.script.sources.get(index, f"source-{index}"))

    bench_compile_config = compile_config

    def to_code(self, config: Config, **kwargs: object) -> str:
        return f"def kernel():\n    return {_index(config)}\n"

    to_triton_code = to_code

    def format_kernel_decorator(self, config: Config, settings: Settings) -> str:
        return f"@helion.kernel(config={config!r})"

    def get_cached_path(self, config: Config | None = None) -> None:
        return None

    def maybe_log_repro(self, *args: object) -> None:
        pass

    def is_cacheable(self) -> bool:
        return True


class _Provider(BenchmarkProvider):
    def __init__(
        self,
        kernel: _AutotunableKernel,
        settings: Settings,
        config_spec: object,
        args: Sequence[object],
        log: AutotuningLogger,
        autotune_metrics: AutotuneMetrics,
    ) -> None:
        self.kernel = cast("_Kernel", kernel)
        self.settings = settings
        self.config_spec = config_spec
        self.args = args
        self.log = log
        self.metrics = autotune_metrics
        self._autotune_metrics = autotune_metrics
        self._accuracy_failure_config_ids: list[int] = []
        self._compile_failure_config_ids: list[int] = []
        self._worker_failure_config_ids: list[int] = []
        self.setup_count = 0
        self.cleanup_count = 0
        self.mutated_arg_indices = ()
        self.kernel.script.providers.append(self)

    def setup(self) -> None:
        self.setup_count += 1

    def cleanup(self) -> None:
        self.cleanup_count += 1

    def benchmark(
        self,
        configs: list[Config],
        *,
        desc: str = "Benchmarking",
        raise_if_no_viable_config: bool = True,
    ) -> list[BenchmarkResult]:
        results = []
        for config in configs:
            perf, status = self.kernel.script.next(_index(config))
            results.append(
                BenchmarkResult(
                    config,
                    self.kernel.compile_config(config),
                    perf,
                    cast("_Status", status),
                    None,
                    completed_at=time.perf_counter(),
                )
            )
        self.metrics.num_configs_tested += len(configs)
        for result in results:
            self.log.record_autotune_entry(
                AutotuneLogEntry(
                    generation=0,
                    status=result.status,
                    perf_ms=result.perf if math.isfinite(result.perf) else None,
                    compile_time=None,
                    config=result.config,
                    config_id=self.log.register_config(result.config),
                    completed_at=result.completed_at,
                )
            )
        if self.kernel.script.after_benchmark is not None:
            self.kernel.script.after_benchmark()
        return results


class _Search(BaseSearch):
    def __init__(
        self,
        kernel: _Kernel,
        batches: list[list[int]],
        *,
        returned: int | None = None,
        args: Sequence[object] = (),
    ) -> None:
        super().__init__(cast("_AutotunableKernel", kernel), args, _Provider)
        self.batches = batches
        self.returned = returned
        self.started = False

    def _prepare(self) -> None:
        if self._prepared:
            return
        self._prepared = True
        self._autotune_metrics = AutotuneMetrics()
        self._kernel_metadata = KernelMetadata()
        provider_cls = (
            MultiShapeBenchmarkProvider
            if isinstance(self.args, _MultiShapeAutotuneArgs)
            else _Provider
        )
        self.benchmark_provider = provider_cls(
            self.kernel,
            self.settings,
            self.config_spec,
            self.args,
            self.log,
            self._autotune_metrics,
        )

    def _autotune(self) -> Config:
        self.started = True
        results = []
        for batch in self.batches:
            results.extend(self.benchmark_batch([_config(index) for index in batch]))
        if self.returned is not None:
            return _config(self.returned)
        valid = [result for result in results if math.isfinite(result.perf)]
        if not valid:
            raise exc.NoConfigFound
        return min(valid, key=lambda result: result.perf).config


class _Cache(AutotuneCacheBase):
    def __init__(self, search: _Search, cached: Config | None = None) -> None:
        super().__init__(search)
        self.cached = cached
        self.reads = 0
        self.writes: list[Config] = []

    def get(self) -> Config | None:
        self.reads += 1
        return self.cached

    def put(self, config: Config) -> None:
        self.writes.append(config)

    def _should_report_cache_hit(self) -> bool:
        return False

    def _get_cache_key(self) -> CacheKeyBase:
        return CacheKeyBase()

    def _list_cache_entries(self) -> Sequence[tuple[str, CacheKeyBase]]:
        return ()


class TestAutotuneHandoff(unittest.TestCase):
    def assert_cleaned_up(self, script: _Script) -> None:
        for provider in script.providers:
            self.assertEqual(provider.setup_count, 1)
            self.assertEqual(provider.cleanup_count, 1)

    def test_completion_remeasures_and_replaces_optimistic_winner(self) -> None:
        script = _Script({1: [0.5, 2.0], 2: [1.0]})
        search = _Search(_Kernel(script), [[1, 2]], returned=1)

        point = find_handoff(search)

        self.assertEqual(point.config, _config(2))
        self.assertEqual(point.fn(), 2)
        self.assertEqual(point.progress.trials, 2)
        self.assertEqual(point.progress.unique_sources, 2)
        self.assertEqual(point.objective_unit, "ms")
        self.assertEqual(len(point.finalists), 2)
        self.assertEqual(
            len([row for row in point.measurements if row.phase == "confirmation"]),
            6,
        )
        self.assertEqual(len(script.providers), 4)
        self.assert_cleaned_up(script)

    def test_trial_limit_stops_at_a_completed_batch(self) -> None:
        script = _Script({index: [float(index)] for index in range(1, 5)})
        search = _Search(_Kernel(script), [[1, 2, 3], [4]])

        point = find_handoff(search, HandoffPolicy(after_trials=2))

        self.assertEqual(point.progress.trials, 3)
        self.assertNotIn(4, script.calls)
        self.assertEqual(point.config, _config(1))
        self.assert_cleaned_up(script)

    def test_elapsed_time_limit(self) -> None:
        script = _Script({1: [1.0], 2: [2.0]})
        clock = [0.0]

        def advance() -> None:
            clock[0] += 10.0

        script.after_benchmark = advance
        search = _Search(_Kernel(script), [[1], [2]])
        with patch("helion.autotuner.handoff.time.perf_counter", lambda: clock[0]):
            point = find_handoff(search, HandoffPolicy(after_seconds=5.0))

        self.assertEqual(point.progress.trials, 1)
        self.assertNotIn(2, script.calls)
        self.assert_cleaned_up(script)

    def test_trial_completion_times_precede_batch_observation(self) -> None:
        script = _Script({1: [1.0], 2: [2.0]})
        clock = [100.0]
        next_result = script.next

        def measure(index: int) -> tuple[float, str]:
            clock[0] += float(index)
            return next_result(index)

        def finish_batch() -> None:
            clock[0] += 379.0

        script.after_benchmark = finish_batch
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(script, "next", side_effect=measure),
            patch("time.perf_counter", lambda: clock[0]),
            patch("time.time", return_value=1_700_000_000.0),
        ):
            base = Path(directory) / "run"
            settings = Settings(
                backend="triton",
                autotune_log=str(base),
                autotune_log_details=True,
                autotune_log_level=0,
            )
            point = find_handoff(
                _Search(_Kernel(script, settings), [[1, 2]]),
                HandoffPolicy(after_trials=2),
            )
            rows = [
                json.loads(line)
                for line in base.with_suffix(".trace.jsonl").read_text().splitlines()
            ]
            with base.with_suffix(".csv").open(newline="") as file:
                csv_rows = list(csv.DictReader(file))

        search_rows = [row for row in point.measurements if row.phase == "search"]
        self.assertEqual([row.elapsed_seconds for row in search_rows], [1.0, 3.0])
        trials = [row for row in rows if row["event"] == "trial"]
        self.assertEqual([row["elapsed_s"] for row in trials], [1.0, 3.0])
        self.assertEqual([float(row["timestamp_s"]) for row in csv_rows], [1.0, 3.0])
        self.assertEqual(
            [row["timestamp_s"] for row in trials],
            [1_700_000_001.0, 1_700_000_003.0],
        )
        self.assertGreaterEqual(point.progress.elapsed_seconds, 382.0)
        confirmations = [row for row in rows if row["event"] == "handoff_confirmation"]
        self.assertEqual(
            [row["elapsed_s"] for row in confirmations],
            [
                row.elapsed_seconds
                for row in point.measurements
                if row.phase == "confirmation"
            ],
        )
        self.assert_cleaned_up(script)

    def test_callback_receives_immutable_progress(self) -> None:
        script = _Script({1: [2.0], 2: [1.0], 3: [0.5]})
        seen: list[HandoffProgress] = []

        def callback(progress: HandoffProgress) -> bool:
            seen.append(progress)
            with self.assertRaises(FrozenInstanceError):
                progress.trials = 999
            return progress.trials >= 2

        point = find_handoff(
            _Search(_Kernel(script), [[1], [2], [3]]),
            HandoffPolicy(callback=callback),
        )

        self.assertEqual(point.progress.trials, 2)
        self.assertEqual(seen[0].trials, 1)
        self.assertEqual(seen[0].best_perf, 2.0)
        self.assertNotIn(3, script.calls)
        self.assert_cleaned_up(script)

    def test_source_aliases_do_not_inflate_exploration(self) -> None:
        script = _Script(
            {1: [1.0], 2: [1.0], 3: [2.0]},
            sources={1: "same", 2: "same", 3: "other"},
        )

        point = find_handoff(_Search(_Kernel(script), [[1, 2, 3]], returned=2))

        self.assertEqual(point.progress.trials, 3)
        self.assertEqual(point.progress.unique_sources, 2)
        # The explicit returned config is retained even when it aliases one
        # of the source-distinct finalists.
        self.assertEqual(len(point.finalists), 3)
        self.assertEqual(len({c.source_hash for c in point.finalists}), 2)
        self.assert_cleaned_up(script)

    def test_limit_without_valid_candidate_does_not_continue_search(self) -> None:
        script = _Script({1: [(math.inf, "error")], 2: [1.0], 3: [0.5]})

        with self.assertRaises(exc.NoConfigFound):
            find_handoff(
                _Search(_Kernel(script), [[1], [2], [3]]),
                HandoffPolicy(after_trials=1),
            )

        self.assertNotIn(2, script.calls)
        self.assertNotIn(3, script.calls)
        self.assert_cleaned_up(script)

    def test_failed_confirmation_disqualifies_candidate(self) -> None:
        script = _Script({1: [0.5, (math.inf, "accuracy_error")], 2: [1.0]})

        point = find_handoff(_Search(_Kernel(script), [[1, 2]]))

        self.assertEqual(point.config, _config(2))
        self.assertEqual(
            [candidate.config for candidate in point.finalists], [_config(2)]
        )
        self.assertTrue(
            any(row.status == "accuracy_error" for row in point.measurements)
        )
        self.assert_cleaned_up(script)

    def test_all_confirmation_candidates_invalid(self) -> None:
        script = _Script({1: [1.0, (math.inf, "timeout")]})

        with self.assertRaises(exc.NoConfigFound):
            find_handoff(_Search(_Kernel(script), [[1]]))

        self.assert_cleaned_up(script)

    def test_ordinary_autotune_has_no_confirmation(self) -> None:
        script = _Script({1: [0.5, 2.0], 2: [1.0]})

        result = _Search(_Kernel(script), [[1, 2]]).autotune()

        self.assertEqual(result, _config(1))
        self.assertEqual(dict(script.calls), {1: 1, 2: 1})
        self.assertEqual(len(script.providers), 1)
        self.assert_cleaned_up(script)

    def test_nested_search_stops_before_second_stage(self) -> None:
        script = _Script({1: [1.0], 2: [0.5]})
        kernel = _Kernel(script)
        stages: list[str] = []

        class NestedSearch(_Search):
            def _autotune(self) -> Config:
                stages.append("first")
                first = _Search(kernel, [[1]], args=self.args).autotune()
                stages.append("second")
                _Search(kernel, [[2]], args=self.args).autotune()
                return first

        point = find_handoff(NestedSearch(kernel, []), HandoffPolicy(after_trials=1))

        self.assertEqual(stages, ["first"])
        self.assertEqual(point.config, _config(1))
        self.assertEqual(point.progress.trials, 1)
        self.assert_cleaned_up(script)

    def test_cache_hit_is_confirmed_without_search_or_write(self) -> None:
        script = _Script({1: [1.0]})
        search = _Search(_Kernel(script), [[1]])
        cache = _Cache(search, _config(1))

        point = find_handoff(cache)

        self.assertEqual(point.config, _config(1))
        self.assertEqual(point.reason, "completed")
        self.assertFalse(search.started)
        self.assertEqual(point.progress.trials, 0)
        self.assertEqual(cache.reads, 1)
        self.assertEqual(cache.writes, [])
        self.assertEqual(script.calls[1], 3)
        self.assert_cleaned_up(script)

    def test_completed_cache_miss_preserves_search_result_in_cache(self) -> None:
        script = _Script({1: [0.5, 2.0], 2: [1.0]})
        cache = _Cache(_Search(_Kernel(script), [[1, 2]], returned=1))

        point = find_handoff(cache)

        self.assertEqual(point.reason, "completed")
        self.assertEqual(point.config, _config(2))
        self.assertEqual(cache.writes, [_config(1)])
        self.assert_cleaned_up(script)

    def test_early_handoff_unwinds_before_cache_write(self) -> None:
        script = _Script({1: [1.0], 2: [0.5]})
        cache = _Cache(_Search(_Kernel(script), [[1], [2]]))

        point = find_handoff(cache, HandoffPolicy(after_trials=1))

        self.assertEqual(point.reason, "trials")
        self.assertEqual(point.config, _config(1))
        self.assertEqual(cache.writes, [])
        self.assertNotIn(2, script.calls)
        self.assert_cleaned_up(script)

    def test_callback_failure_cleans_up_and_resets_context(self) -> None:
        script = _Script({1: [1.0]})
        kernel = _Kernel(script)

        def fail(progress: HandoffProgress) -> bool:
            raise RuntimeError("callback failed")

        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            find_handoff(_Search(kernel, [[1]]), HandoffPolicy(callback=fail))
        self.assert_cleaned_up(script)

        point = find_handoff(_Search(kernel, [[1]]))
        self.assertEqual(point.config, _config(1))
        self.assertEqual(point.progress.trials, 1)
        self.assert_cleaned_up(script)

    def test_skip_cache_bypasses_read_but_preserves_write(self) -> None:
        script = _Script({1: [1.0]})
        cache = _Cache(_Search(_Kernel(script), [[1]]), _config(2))

        point = find_handoff(cache, skip_cache=True)

        self.assertEqual(point.config, _config(1))
        self.assertEqual(cache.reads, 0)
        self.assertEqual(cache.writes, [_config(1)])
        self.assert_cleaned_up(script)

    def test_result_configs_are_detached_from_search_inputs(self) -> None:
        script = _Script({1: [1.0]})
        original = _config(1)

        class SharedConfigSearch(_Search):
            def _autotune(self) -> Config:
                self.benchmark_batch([original])
                return original

        point = find_handoff(SharedConfigSearch(_Kernel(script), []))
        original.config["block_sizes"] = [999]

        self.assertEqual(point.config, _config(1))
        self.assertEqual(point.finalists[0].config, _config(1))
        self.assertTrue(all(row.config == _config(1) for row in point.measurements))

    def test_invalid_policy_is_rejected(self) -> None:
        for kwargs in (
            {"after_trials": 0},
            {"after_seconds": math.inf},
            {"after_seconds": -1.0},
            {"finalists": 0},
            {"repetitions": 2},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HandoffPolicy(**kwargs)

    def test_best_of_k_stop_restores_settings_and_skips_later_trials(self) -> None:
        script = _Script({1: [1.0]})
        settings = Settings(
            backend="triton",
            autotune_best_of_k=3,
            autotune_random_seed=77,
            autotune_compile_timeout=30,
            autotune_log_level=0,
        )
        kernel = _Kernel(script, settings)
        original = _Search(kernel, [[1]])
        cache = _Cache(original)
        trials: list[_Search] = []

        class AdaptiveSearch(_Search):
            def _autotune(self) -> Config:
                self.settings.autotune_compile_timeout = 5
                return super()._autotune()

        def factory() -> _Search:
            search = AdaptiveSearch(kernel, [[1]], args=original.args)
            trials.append(search)
            return search

        cache._autotuner_factory = factory
        with patch.object(cache, "_release_trial_state") as release:
            point = find_handoff(cache, HandoffPolicy(after_trials=1))

        self.assertEqual(point.config, _config(1))
        self.assertEqual(len(trials), 1)
        self.assertIs(cache.autotuner, original)
        self.assertEqual(settings.autotune_random_seed, 77)
        self.assertEqual(settings.autotune_compile_timeout, 30)
        self.assertEqual(cache.writes, [])
        release.assert_called_once()
        self.assert_cleaned_up(script)

    def test_trace_records_handoff_and_confirmation_without_extra_trials(self) -> None:
        script = _Script({1: [1.0], 2: [2.0]})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "run.log"
            settings = Settings(
                backend="triton",
                autotune_log=str(base),
                autotune_log_details=True,
                autotune_log_level=0,
            )
            point = find_handoff(
                _Search(_Kernel(script, settings), [[1, 2]]),
                HandoffPolicy(after_trials=1),
            )
            rows = [
                json.loads(line)
                for line in base.with_suffix(".trace.jsonl").read_text().splitlines()
            ]
            with base.with_suffix(".csv").open(newline="") as file:
                csv_rows = list(csv.DictReader(file))

        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(len([row for row in rows if row["event"] == "trial"]), 2)
        self.assertEqual(
            len([row for row in rows if row["event"] == "handoff_confirmation"]), 6
        )
        self.assertEqual(len({row["run_id"] for row in rows}), 1)
        handoff = next(row for row in rows if row["event"] == "handoff")
        self.assertEqual(handoff["trials"], point.progress.trials)
        self.assertEqual(handoff["reason"], "trials")
        stage = next(
            row
            for row in rows
            if row["event"] == "stage_end" and row["algorithm"] == "_Search"
        )
        self.assertEqual(stage["status"], "handoff")
        self.assertEqual(rows[-1]["event"], "run_end")
        self.assertEqual(rows[-1]["status"], "ok")
        self.assert_cleaned_up(script)

    def test_multi_shape_source_identity_and_ratio_objective(self) -> None:
        anchor_script = _Script({1: [10.0], 2: [10.0]}, sources={1: "same", 2: "same"})
        second_script = _Script({1: [40.0], 2: [10.0]})
        anchor = _Kernel(anchor_script)
        second = _Kernel(second_script)
        args = _MultiShapeAutotuneArgs(
            cases=((anchor, ()), (second, ())),
            aggregation="geomean",
            relative_to="default",
            cache_tag=None,
            workload_key=("test",),
            reference_latencies=(10.0, 20.0),
        )
        search = _Search(anchor, [[1, 2]], args=args)
        with patch(
            "helion.autotuner.benchmark_provider.LocalBenchmarkProvider", _Provider
        ):
            point = find_handoff(search)

        self.assertEqual(point.progress.trials, 2)
        self.assertEqual(point.progress.unique_sources, 2)
        self.assertEqual(point.objective_unit, "ratio")
        self.assertEqual(point.reference_latencies, (10.0, 20.0))
        self.assertEqual(point.config, _config(2))
        self.assertEqual(point.fn(), 2)
        self.assertAlmostEqual(point.finalists[0].perf, math.sqrt(0.5))
        winner_rows = [row for row in point.measurements if row.config == _config(2)]
        self.assertTrue(all(row.per_shape == (10.0, 10.0) for row in winner_rows))
        self.assertEqual(
            len({candidate.source_hash for candidate in point.finalists}), 2
        )
        self.assert_cleaned_up(anchor_script)
        self.assert_cleaned_up(second_script)

    def test_multi_shape_completion_uses_last_shape_not_end_of_batch(self) -> None:
        script = _Script({1: [1.0], 2: [2.0]})
        anchor = _Kernel(script)
        second = _Kernel(script)
        args = _MultiShapeAutotuneArgs(
            cases=((anchor, ()), (second, ())),
            aggregation="max",
            relative_to=None,
            cache_tag=None,
            workload_key=("test",),
        )
        clock = [0.0]
        next_result = script.next

        def measure(index: int) -> tuple[float, str]:
            clock[0] += 1.0
            return next_result(index)

        with (
            patch.object(script, "next", side_effect=measure),
            patch("time.perf_counter", lambda: clock[0]),
            patch(
                "helion.autotuner.benchmark_provider.LocalBenchmarkProvider", _Provider
            ),
        ):
            point = find_handoff(_Search(anchor, [[1, 2]], args=args))

        search_rows = [row for row in point.measurements if row.phase == "search"]
        self.assertEqual([row.elapsed_seconds for row in search_rows], [3.0, 4.0])
        self.assert_cleaned_up(script)

    def test_failed_multi_shape_evidence_and_finite_trace(self) -> None:
        anchor_script = _Script({1: [1.0], 2: [2.0], 3: [3.0]})
        second_script = _Script(
            {
                1: [0.5, (math.inf, "accuracy_error")],
                2: [2.0],
                3: [(math.inf, "timeout")],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "run.log"
            settings = Settings(
                backend="triton",
                autotune_log=str(base),
                autotune_log_details=True,
                autotune_log_level=0,
            )
            anchor = _Kernel(anchor_script, settings)
            second = _Kernel(second_script)
            args = _MultiShapeAutotuneArgs(
                cases=((anchor, ()), (second, ())),
                aggregation="geomean",
                relative_to=None,
                cache_tag=None,
                workload_key=("test_failures",),
            )
            with patch(
                "helion.autotuner.benchmark_provider.LocalBenchmarkProvider", _Provider
            ):
                point = find_handoff(_Search(anchor, [[1, 2, 3]], args=args))
            rows = [
                json.loads(line, parse_constant=lambda token: self.fail(token))
                for line in base.with_suffix(".trace.jsonl").read_text().splitlines()
            ]

        self.assertEqual(point.config, _config(2))
        failed_search = next(
            row
            for row in point.measurements
            if row.phase == "search" and row.config == _config(3)
        )
        self.assertIsNone(failed_search.perf)
        self.assertEqual(failed_search.per_shape, (3.0, math.inf))
        self.assertEqual(failed_search.per_shape_statuses, ("ok", "timeout"))
        failed_confirmation = [
            row
            for row in point.measurements
            if row.phase == "confirmation" and row.config == _config(1)
        ]
        self.assertEqual(len(failed_confirmation), 3)
        for row in failed_confirmation:
            self.assertIsNone(row.perf)
            self.assertEqual(row.per_shape, (1.0, math.inf))
            self.assertEqual(row.per_shape_statuses, ("ok", "accuracy_error"))
        trace_failures = [
            row
            for row in rows
            if row["event"] == "handoff_confirmation" and row["status"] == "error"
        ]
        self.assertEqual(len(trace_failures), 3)
        for row in trace_failures:
            self.assertIsNone(row["objective"])
            self.assertEqual(row["per_shape_ms"], [1.0, None])
            self.assertEqual(row["per_shape_statuses"], ["ok", "accuracy_error"])
        self.assert_cleaned_up(anchor_script)
        self.assert_cleaned_up(second_script)

    def test_peer_finalist_failure_disqualifies_local_success(self) -> None:
        script = _Script({1: [0.5], 2: [1.0]})
        rejected = canonical_config_id(_config(1))

        def gather(value: object, group: str | None) -> list[object]:
            return [value, {rejected}] if isinstance(value, set) else [value, None]

        with patch("helion.autotuner.handoff.all_gather_object", side_effect=gather):
            point = find_handoff(_Search(_Kernel(script), [[1, 2]]))

        self.assertEqual(point.config, _config(2))
        self.assertEqual(
            [candidate.config for candidate in point.finalists], [_config(2)]
        )
        self.assertEqual(point.progress.best_perf, 1.0)
        self.assert_cleaned_up(script)

    def test_missing_backend_hash_uses_exported_source_identity(self) -> None:
        script = _Script({1: [1.0], 2: [2.0]})
        kernel = _Kernel(script)
        with (
            patch.object(_Backend, "generated_source_hash", return_value=None),
            patch.object(kernel, "to_code", wraps=kernel.to_code) as export,
        ):
            point = find_handoff(_Search(kernel, [[1, 2]]))

        self.assertGreater(export.call_count, 0)
        self.assertEqual(point.progress.unique_sources, 2)
        self.assertTrue(all(row.source_hash is not None for row in point.measurements))
        self.assertEqual(
            len({candidate.source_hash for candidate in point.finalists}), 2
        )
        self.assert_cleaned_up(script)


if __name__ == "__main__":
    unittest.main()
