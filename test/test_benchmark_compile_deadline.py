"""The benchmark worker's compile deadline: a candidate whose compile (and
first call) never finishes is killed, recorded as failed to compile and
named; the timing window starts once the candidate is compiled; a timed-out
default config gets the compiler seeds' fixed-repetition retry."""

from __future__ import annotations

import dataclasses
import logging
import math
from multiprocessing.connection import Connection
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

import helion
from helion._testing import TestCase
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_job import BenchmarkJob
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import PHASE_COMPILED
from helion.autotuner.benchmark_worker import BenchmarkCompileTimeout
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmark_worker import BenchmarkWorker
from helion.autotuner.benchmark_worker import BenchmarkWorkerDied
from helion.autotuner.benchmark_worker import PhasedJob
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
from helion.runtime.config import Config
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from helion.autotuner.precompile_future import SerializedCompiledFunction
    from helion.runtime.kernel import CompiledConfig


# Job callables live at module level so multiprocessing.spawn can re-import
# them in the worker.


@dataclasses.dataclass
class _SleepInPhases(PhasedJob):
    """A fake candidate: ``compile_seconds`` of compile, then the compiled
    report, then ``bench_seconds`` of timing."""

    compile_seconds: float
    bench_seconds: float

    def __call__(self, report_phase: Callable[[str], None] | None = None) -> float:
        time.sleep(self.compile_seconds)
        if report_phase is not None:
            report_phase(PHASE_COMPILED)
        time.sleep(self.bench_seconds)
        return self.compile_seconds + self.bench_seconds


@dataclasses.dataclass
class _Sleep:
    seconds: float

    def __call__(self) -> float:
        time.sleep(self.seconds)
        return self.seconds


@dataclasses.dataclass
class _ReturnValue:
    value: object

    def __call__(self) -> object:
        return self.value


class _CountCalls:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> torch.Tensor:
        self.calls += 1
        return torch.empty(())


class TestWorkerCompileDeadline(unittest.TestCase):
    def test_compile_deadline_kills_a_worker_stuck_in_compile(self) -> None:
        worker = BenchmarkWorker()
        try:
            t0 = time.time()
            with self.assertRaises(BenchmarkCompileTimeout) as caught:
                worker.run(_SleepInPhases(60, 0), timeout=30.0, compile_timeout=0.5)
            self.assertLess(time.time() - t0, 15.0)
            self.assertIn("compile and first call exceeded 0.5s", str(caught.exception))
            self.assertFalse(worker.alive())
            # Not a benchmark timeout: the candidate was never timed.
            self.assertNotIsInstance(caught.exception, BenchmarkTimeout)
            # The next job respawns the worker.
            self.assertEqual(
                worker.run(_SleepInPhases(0, 0), timeout=30.0, compile_timeout=30.0),
                0,
            )
        finally:
            worker.shutdown()

    def test_benchmark_timeout_counts_from_the_compiled_report(self) -> None:
        worker = BenchmarkWorker()
        try:
            # A compile longer than the benchmark timeout is fine when it fits
            # the compile deadline: the timing window opens afterwards.
            self.assertAlmostEqual(
                worker.run(_SleepInPhases(1.0, 0.2), timeout=0.6, compile_timeout=15.0),
                1.2,
            )
            # Timing past its own window is still a benchmark timeout.
            with self.assertRaises(BenchmarkTimeout) as caught:
                worker.run(_SleepInPhases(0.1, 60), timeout=0.5, compile_timeout=15.0)
            self.assertNotIsInstance(caught.exception, BenchmarkCompileTimeout)
            self.assertIn("benchmark timeout after 0.5s", str(caught.exception))
            self.assertFalse(worker.alive())
        finally:
            worker.shutdown()

    def test_a_job_without_phases_keeps_the_whole_job_timeout(self) -> None:
        worker = BenchmarkWorker()
        try:
            t0 = time.time()
            with self.assertRaises(BenchmarkTimeout):
                worker.run(_Sleep(60), timeout=0.5, compile_timeout=60.0)
            self.assertLess(time.time() - t0, 15.0)
            self.assertEqual(
                worker.run(_ReturnValue(7), timeout=30.0, compile_timeout=60.0), 7
            )
        finally:
            worker.shutdown()

    def test_phase_reports_are_ignored_without_a_compile_deadline(self) -> None:
        worker = BenchmarkWorker()
        try:
            self.assertAlmostEqual(
                worker.run(_SleepInPhases(0.1, 0.1), timeout=30.0), 0.2
            )
            # The single deadline still covers both phases.
            with self.assertRaises(BenchmarkTimeout):
                worker.run(_SleepInPhases(0.4, 60), timeout=0.6)
        finally:
            worker.shutdown()

    def test_compile_deadline_watchdog_when_poll_ignores_its_timeout(self) -> None:
        original_poll = Connection.poll

        def poll_until_ready(
            self: Connection,
            timeout: float | None = None,
        ) -> bool:
            return original_poll(self, None)

        worker = BenchmarkWorker()
        try:
            t0 = time.time()
            with (
                patch.object(Connection, "poll", poll_until_ready),
                self.assertRaises(BenchmarkCompileTimeout),
            ):
                worker.run(_SleepInPhases(60, 0), timeout=30.0, compile_timeout=0.5)
            self.assertLess(time.time() - t0, 15.0)
            self.assertFalse(worker.alive())
        finally:
            worker.shutdown()


class TestBenchmarkJobPhases(unittest.TestCase):
    def test_benchmark_job_reports_compiled_after_the_first_call(self) -> None:
        fn = _CountCalls()
        calls_at_report: list[int] = []

        def report_phase(phase: str) -> None:
            calls_at_report.append(fn.calls)
            self.assertEqual(phase, PHASE_COMPILED)

        with (
            patch(
                "helion.autotuner.benchmark_job._load_compiled_fn",
                return_value=fn,
            ),
            patch(
                "helion.autotuner.benchmark_job.load_trusted_kernel_args",
                return_value=(),
            ),
            patch(
                "helion.autotuner.benchmark_job.do_bench_generic",
                return_value=1.25,
            ) as wall_clock_bench,
        ):
            result = BenchmarkJob(
                fn_spec=cast("SerializedCompiledFunction", object()),
                args_path="/tmp/args.pt",
                use_wall_clock=True,
            )(report_phase)

        self.assertEqual(result, 1.25)
        # Exactly one call (the compile and first launch) precedes the report;
        # the timer then skips its own setup call.
        self.assertEqual(calls_at_report, [1])
        self.assertEqual(fn.calls, 1)
        self.assertIs(wall_clock_bench.call_args.kwargs["pre_warmed"], True)

    def test_benchmark_job_runs_without_a_reporter(self) -> None:
        fn = _CountCalls()
        with (
            patch(
                "helion.autotuner.benchmark_job._load_compiled_fn",
                return_value=fn,
            ),
            patch(
                "helion.autotuner.benchmark_job.load_trusted_kernel_args",
                return_value=(),
            ),
            patch("helion.autotuner.benchmark_job.do_bench", return_value=2.5),
        ):
            result = BenchmarkJob(
                fn_spec=cast("SerializedCompiledFunction", object()),
                args_path="/tmp/args.pt",
            )()
        self.assertEqual(result, 2.5)
        self.assertEqual(fn.calls, 1)


def _provider(
    *, retry_repetitions: int | None = None, **settings: object
) -> LocalBenchmarkProvider:
    provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
    provider.settings = Settings(
        autotune_log_level=logging.CRITICAL,
        autotune_accuracy_check=False,
        **settings,  # pyrefly: ignore[bad-argument-type]
    )
    provider.config_spec = SimpleNamespace(
        compiler_seed_timeout_retry_repetitions=retry_repetitions
    )
    provider.log = Mock()
    provider._autotune_metrics = AutotuneMetrics()
    provider.mutated_arg_indices = []
    provider._compile_failure_config_ids = []
    provider._worker_failure_config_ids = []
    provider._compiler_seed_configs = set()
    provider._compiler_seed_source_hashes = set()
    provider._compiler_seed_timeout_retry_claims = set()
    return provider


class TestProviderCompileDeadline(TestCase):
    def test_every_benchmark_job_carries_the_compile_deadline(self) -> None:
        for compile_timeout, expected in ((3, 17.0), (45, 45.0)):
            provider = _provider(
                autotune_benchmark_timeout=17,
                autotune_compile_timeout=compile_timeout,
            )
            provider.config_spec = SimpleNamespace(cute_flash_search_enabled=False)
            provider._precompile_args_path = "/tmp/args.pt"
            provider._subprocess_wrapper_unloadable = False
            provider._benchmark_worker = Mock()
            provider._benchmark_worker.run.return_value = 1.25
            provider._subprocess_benchmark_uses_wall_clock = lambda: True
            provider._probe_long_kernel = lambda: False
            with patch(
                "helion.autotuner.benchmark_provider._serialize_compiled_fn",
                return_value=cast("SerializedCompiledFunction", object()),
            ):
                provider._run_subprocess_benchmark_job(
                    cast("CompiledConfig", object()), warmup=1, rep=50
                )
            kwargs = provider._benchmark_worker.run.call_args.kwargs
            # The compile deadline is never below the benchmark timeout.
            self.assertEqual(kwargs["compile_timeout"], expected)
            self.assertEqual(kwargs["timeout"], 17.0)

    def test_compile_timeout_is_a_compile_failure_naming_the_config(self) -> None:
        provider = _provider(retry_repetitions=3)
        config = Config(block_sizes=[64], num_warps=4)
        provider.set_compiler_seed_configs([config])
        error = BenchmarkCompileTimeout(
            "compile and first call exceeded 60.0s in the benchmark worker"
        )
        with patch.object(
            provider, "_run_subprocess_benchmark_job", side_effect=error
        ) as run_job:
            result = provider._benchmark_function_subprocess(
                config, cast("CompiledConfig", object())
            )

        self.assertEqual(result, math.inf)
        # No fixed-repetition retry: re-running the compile cannot help.
        run_job.assert_called_once()
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 1)
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 0)
        self.assertEqual(provider._last_benchmark_failure_status, "timeout")
        self.assertEqual(provider._compile_failure_config_ids, [id(config)])
        # The INFO line names the config and the deadline's knob.
        provider.log.assert_called_once()  # pyrefly: ignore[missing-attribute]
        (message,) = provider.log.call_args.args  # pyrefly: ignore[missing-attribute]
        self.assertIn(repr(config), message)
        self.assertIn("compile and first call exceeded 60.0s", message)
        self.assertIn("HELION_AUTOTUNE_COMPILE_TIMEOUT", message)

    def test_compile_timeout_of_a_seed_names_its_role(self) -> None:
        provider = _provider()
        default = Config(block_sizes=[64], num_warps=4)
        provider.set_config_roles({default: "default config"})
        error = BenchmarkCompileTimeout(
            "compile and first call exceeded 60.0s in the benchmark worker"
        )
        with patch.object(provider, "_run_subprocess_benchmark_job", side_effect=error):
            provider._benchmark_function_subprocess(
                default, cast("CompiledConfig", object())
            )
        provider.log.warning.assert_called_once()  # pyrefly: ignore[missing-attribute]
        (message,) = provider.log.warning.call_args.args  # pyrefly: ignore[missing-attribute]
        self.assertIn("default config did not finish compiling", message)
        self.assertIn("HELION_AUTOTUNE_COMPILE_TIMEOUT", message)
        self.assertNotIn("HELION_AUTOTUNE_BENCHMARK_TIMEOUT", message)

        # An ordinary candidate's compile timeout is only the INFO line.
        provider.log.reset_mock()
        with patch.object(provider, "_run_subprocess_benchmark_job", side_effect=error):
            provider._benchmark_function_subprocess(
                Config(block_sizes=[32]), cast("CompiledConfig", object())
            )
        provider.log.warning.assert_not_called()  # pyrefly: ignore[missing-attribute]
        provider.log.assert_called_once()  # pyrefly: ignore[missing-attribute]


class TestDefaultConfigTimeoutRetry(TestCase):
    def test_default_config_timeout_retries_with_fixed_repetitions(self) -> None:
        provider = _provider()
        fn = cast("CompiledConfig", object())
        default = Config(block_sizes=[64], num_warps=4)
        provider.set_config_roles({default: "default config"})

        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=(BenchmarkTimeout("timed out"), 2.75),
        ) as run_job:
            result = provider._benchmark_function_subprocess(default, fn)

        self.assertEqual(result, 2.75)
        self.assertEqual(
            run_job.call_args_list,
            [
                unittest.mock.call(fn, warmup=1, rep=50),
                unittest.mock.call(fn, warmup=1, rep=50, fixed_repetitions=3),
            ],
        )
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 0)
        provider.log.warning.assert_not_called()  # pyrefly: ignore[missing-attribute]

    def test_default_config_retry_uses_the_specs_repetition_count(self) -> None:
        provider = _provider(retry_repetitions=5)
        fn = cast("CompiledConfig", object())
        default = Config(block_sizes=[64], num_warps=4)
        provider.set_config_roles({default: "default config"})
        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=(BenchmarkTimeout("timed out"), 1.5),
        ) as run_job:
            self.assertEqual(provider._benchmark_function_subprocess(default, fn), 1.5)
        self.assertEqual(run_job.call_args_list[1].kwargs["fixed_repetitions"], 5)

    def test_ordinary_candidates_are_not_retried(self) -> None:
        provider = _provider()
        fn = cast("CompiledConfig", object())
        provider.set_config_roles({Config(block_sizes=[64]): "default config"})
        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=BenchmarkTimeout("timed out"),
        ) as run_job:
            result = provider._benchmark_function_subprocess(
                Config(block_sizes=[32]), fn
            )
        self.assertEqual(result, math.inf)
        run_job.assert_called_once()
        self.assertEqual(provider._last_benchmark_failure_status, "timeout")
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 1)
        # A user seed is reported, not retried (only the default anchors the
        # search the way the compiler seeds do).
        user_seed = Config(block_sizes=[128])
        provider.set_config_roles({user_seed: "user seed config"})
        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=BenchmarkTimeout("timed out"),
        ) as run_job:
            self.assertEqual(
                provider._benchmark_function_subprocess(user_seed, fn), math.inf
            )
        run_job.assert_called_once()

    def test_default_config_retry_is_one_per_search_and_named(self) -> None:
        provider = _provider()
        fn = cast("CompiledConfig", object())
        default = Config(block_sizes=[64], num_warps=4)
        provider.set_config_roles({default: "default config"})
        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=(
                BenchmarkTimeout("first timeout"),
                BenchmarkTimeout("retry timeout"),
            ),
        ) as run_job:
            result = provider._benchmark_function_subprocess(default, fn)
        self.assertEqual(result, math.inf)
        self.assertEqual(run_job.call_count, 2)
        self.assertEqual(provider._last_benchmark_failure_status, "timeout")
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 1)
        (message,) = provider.log.warning.call_args.args  # pyrefly: ignore[missing-attribute]
        self.assertIn("default config timed out", message)
        self.assertIn("also with 3 fixed measured repetitions", message)

        # The retry was claimed: a later timeout of the default is final.
        provider.log.reset_mock()
        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=BenchmarkTimeout("later timeout"),
        ) as run_job:
            self.assertEqual(
                provider._benchmark_function_subprocess(default, fn), math.inf
            )
        run_job.assert_called_once()
        (message,) = provider.log.warning.call_args.args  # pyrefly: ignore[missing-attribute]
        self.assertNotIn("fixed measured repetitions", message)

    def test_default_config_retry_respects_the_budget(self) -> None:
        provider = _provider()
        provider.budget_exceeded_fn = Mock(return_value=True)
        default = Config(block_sizes=[64], num_warps=4)
        provider.set_config_roles({default: "default config"})
        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=BenchmarkTimeout("timed out"),
        ) as run_job:
            result = provider._benchmark_function_subprocess(
                default, cast("CompiledConfig", object())
            )
        self.assertEqual(result, math.inf)
        run_job.assert_called_once()
        provider.budget_exceeded_fn.assert_called_once_with()


class TestIsolatedCandidatesAreNamed(TestCase):
    def _isolated_provider(self) -> LocalBenchmarkProvider:
        provider = _provider()
        provider._subprocess_wrapper_unloadable = False
        provider._subprocess_benchmark_enabled = lambda: True  # pyrefly: ignore[bad-assignment]
        provider._benchmark_worker = None
        return provider

    def test_isolated_failure_names_the_registered_candidate(self) -> None:
        provider = self._isolated_provider()
        fns = [object(), object()]
        configs = [Config(block_sizes=[64]), Config(block_sizes=[128])]
        provider.register_isolated_candidates(fns, configs)
        errors = iter(
            [
                BenchmarkWorkerDied(
                    "worker pipe closed before sending result during compile "
                    "and first call"
                ),
                BenchmarkTimeout("benchmark timeout after 30.0s"),
            ]
        )

        def failing_job(_fn: object, **_kwargs: object) -> float:
            raise next(errors)

        provider._run_subprocess_benchmark_job = failing_job  # pyrefly: ignore[bad-assignment]
        results = provider.benchmark_isolated(
            fns, warmup=1, rep=1, desc="Generation 20: verifying top configs"
        )
        assert results is not None
        # A dead worker keeps the earlier timing; a timed-out re-timing is
        # conclusive.
        self.assertIsNone(results[0])
        self.assertEqual(results[1].status, "timeout")  # pyrefly: ignore[missing-attribute]
        messages = [
            call.args[0]
            for call in provider.log.warning.call_args_list  # pyrefly: ignore[missing-attribute]
        ]
        self.assertEqual(len(messages), 2)
        self.assertIn(
            "Generation 20: verifying top configs subprocess failed for "
            f"{configs[0]!r}: worker pipe closed before sending result during "
            "compile and first call",
            messages[0],
        )
        self.assertIn(f"subprocess failed for {configs[1]!r}", messages[1])

    def test_unregistered_candidates_keep_the_plain_line(self) -> None:
        provider = self._isolated_provider()

        def dying_job(_fn: object, **_kwargs: object) -> float:
            raise BenchmarkWorkerDied("worker pipe closed before sending result")

        provider._run_subprocess_benchmark_job = dying_job  # pyrefly: ignore[bad-assignment]
        self.assertEqual(
            provider.benchmark_isolated([object()], warmup=1, rep=1, desc="Final"),
            [None],
        )
        (message,) = provider.log.warning.call_args.args  # pyrefly: ignore[missing-attribute]
        self.assertEqual(
            message, "Final subprocess failed: worker pipe closed before sending result"
        )

    def test_rebenchmark_registers_its_members(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        provider = _provider()
        provider.mutated_arg_indices = []
        seen: list[list[object]] = []

        def benchmark_isolated(
            fns: list[Callable[[], object]], **_kwargs: object
        ) -> list[float]:
            seen.append(list(fns))
            return [0.04 for _ in fns]

        provider.benchmark_isolated = benchmark_isolated  # pyrefly: ignore[bad-assignment]
        search.benchmark_provider = provider
        members = [
            PopulationMember(lambda: None, [0.05], (), helion.Config(num_warps=4)),
            PopulationMember(lambda: None, [0.06], (), helion.Config(num_warps=8)),
        ]
        results = search._isolated_rebenchmark_results(
            members, rep=1, desc="verifying", candidate_private_args=False
        )
        self.assertEqual(results, [0.04, 0.04])
        self.assertEqual(seen, [[member.fn for member in members]])
        self.assertEqual(
            provider._isolated_candidate_configs,
            {id(member.fn): member.config for member in members},
        )


if __name__ == "__main__":
    unittest.main()
