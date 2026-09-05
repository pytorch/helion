"""Tests for the subprocess benchmark path used to hang-protect autotune."""

from __future__ import annotations

import dataclasses
import linecache
import math
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from pathlib import Path
import pickle
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfPallasTpu
from helion._testing import skipIfXPU
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_job import AccuracyCheckJob
from helion.autotuner.benchmark_job import AccuracyCheckResult
from helion.autotuner.benchmark_job import BenchmarkJob
from helion.autotuner.benchmark_provider import BenchmarkResult
from helion.autotuner.benchmark_provider import IsolatedBenchmarkFailure
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkSubprocessError
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmark_worker import BenchmarkWorker
from helion.autotuner.benchmark_worker import BenchmarkWorkerDied
from helion.autotuner.benchmark_worker import BenchmarkWorkerUnkillable
from helion.autotuner.benchmarking import _estimate_runtime_and_warmup
from helion.autotuner.benchmarking import do_bench
from helion.autotuner.benchmarking import do_bench_generic
from helion.autotuner.kernel_args import load_trusted_kernel_args
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.precompile_future import SerializedCompiledFunction
from helion.autotuner.precompile_future import _load_compiled_fn
from helion.autotuner.precompile_future import _run_kernel_in_subprocess_spawn
from helion.autotuner.precompile_future import _serialize_compiled_fn
from helion.autotuner.precompile_future import _unload_compiled_fn
from helion.autotuner.process_utils import signal_process_tree
from helion.autotuner.process_utils import start_isolated_process_group
from helion.autotuner.random_search import RandomSearch
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.runtime import _create_cute_wrapper
from helion.runtime.config import Config
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from helion.runtime.kernel import CompiledConfig


# Job callables: must be at module level so multiprocessing.spawn can
# re-import them in the child.


@dataclasses.dataclass
class _Sleep:
    seconds: float

    def __call__(self) -> float:
        time.sleep(self.seconds)
        return self.seconds


@dataclasses.dataclass
class _RaiseRuntimeError:
    message: str

    def __call__(self) -> object:
        raise RuntimeError(self.message)


@dataclasses.dataclass
class _RaiseUnpickleableLocalException:
    def __call__(self) -> object:
        class LocalError(Exception):
            pass

        raise LocalError("local exception")


@dataclasses.dataclass
class _Crash:
    def __call__(self) -> object:
        os.kill(os.getpid(), signal.SIGKILL)
        return None


@dataclasses.dataclass
class _SpawnChildAndSleep:
    pid_path: str

    def __call__(self) -> None:
        child = subprocess.Popen(["sleep", "60"])
        Path(self.pid_path).write_text(str(child.pid))
        time.sleep(60)


@dataclasses.dataclass
class _ReturnValue:
    value: object

    def __call__(self) -> object:
        return self.value


class _FakeTimingEvent:
    def __init__(self, elapsed_times: list[float]) -> None:
        self.elapsed_times = elapsed_times

    def record(self) -> None:
        pass

    def elapsed_time(self, end_event: _FakeTimingEvent) -> float:
        return self.elapsed_times.pop(0)


class TestAutotuneDataclassCompatibility(unittest.TestCase):
    def test_benchmark_job_preserves_fixed_repetitions_position(self) -> None:
        job = BenchmarkJob(
            cast("SerializedCompiledFunction", object()),
            "args.pt",
            2,
            3,
            True,
            7,
        )

        self.assertEqual(job.fixed_repetitions, 7)
        self.assertFalse(job.probe_long_kernel)

    def test_metrics_preserve_original_positional_fields(self) -> None:
        metrics = AutotuneMetrics(
            1.0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9.0,
            10.0,
            "kernel",
            "source",
            "shapes",
            "dtypes",
            "hardware",
            16,
            "search",
        )

        self.assertEqual(metrics.num_accuracy_failures, 5)
        self.assertEqual(metrics.num_unique_sources, 6)
        self.assertEqual(metrics.search_algorithm, "search")
        self.assertEqual(metrics.num_isolated_rebenchmark_timeouts, 0)
        self.assertEqual(metrics.num_successful_candidate_measurements, 0)


class TestBenchmarkWorkerFailureModes(unittest.TestCase):
    def test_precompile_worker_starts_isolated_process_group(self) -> None:
        with patch("helion.autotuner.process_utils.os.setsid") as setsid:
            start_isolated_process_group()

        setsid.assert_called_once_with()

    def test_precompile_timeout_signals_process_group(self) -> None:
        process = Mock(pid=123)
        with patch("helion.autotuner.process_utils.os.killpg") as killpg:
            signal_process_tree(process, signal.SIGKILL)

        killpg.assert_called_once_with(123, signal.SIGKILL)
        process.kill.assert_not_called()

    def test_precompile_timeout_falls_back_when_group_is_not_ready(self) -> None:
        process = Mock(pid=123)
        with patch(
            "helion.autotuner.process_utils.os.killpg",
            side_effect=ProcessLookupError,
        ):
            signal_process_tree(process, signal.SIGTERM)

        process.terminate.assert_called_once_with()

    @skipUnlessCuteAvailable("_create_cute_wrapper requires the CuTe DSL")
    def test_cute_wrapper_failure_does_not_leak_linecache(self) -> None:
        launcher_filenames_before = {
            filename
            for filename in linecache.cache
            if filename.startswith("<helion_cute_launcher:")
        }

        with (
            patch("builtins.exec", side_effect=RuntimeError("decoration failed")),
            self.assertRaisesRegex(RuntimeError, "decoration failed"),
        ):
            _create_cute_wrapper(object(), (), (32, 1, 1))

        launcher_filenames_after = {
            filename
            for filename in linecache.cache
            if filename.startswith("<helion_cute_launcher:")
        }
        self.assertEqual(launcher_filenames_after, launcher_filenames_before)

    def test_compiled_fn_restores_cute_source_hash_and_unloads(self) -> None:
        launcher_filename = "<helion_cute_launcher:test>"
        source = (
            "class Kernel:\n"
            "    pass\n"
            f"exec(compile('def launcher():\\n    pass\\n', {launcher_filename!r}, 'exec'))\n"
            "class Launcher:\n"
            "    pass\n"
            "compiled_launcher = Launcher()\n"
            "compiled_launcher._jit_func = launcher\n"
            "_helion_call = Kernel()\n"
            "_helion_call._helion_cute_compiled_launchers = "
            "{'compiled': compiled_launcher}\n"
            "_helion_call._helion_cute_launch_arg_cache = {'args': object()}\n"
            "def call():\n"
            "    return _helion_call._helion_cute_source_hash\n"
        )
        expected_hash = "parent-source-hash"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "compiled_module.py"
            path.write_text(source)
            source_module = ModuleType("_test_compiled_module")
            source_module.__file__ = str(path)
            sys.modules[source_module.__name__] = source_module
            try:
                exec(compile(source, str(path), "exec"), source_module.__dict__)
                source_module._helion_call._helion_cute_source_hash = expected_hash
                fn_spec = _serialize_compiled_fn(source_module.call)
            finally:
                sys.modules.pop(source_module.__name__, None)

        fn = _load_compiled_fn(fn_spec)
        module_name = fn.__module__
        loaded_module = sys.modules[module_name]
        cute_kernel = loaded_module._helion_call
        self.assertEqual(fn_spec.source_hash, expected_hash)
        self.assertEqual(fn(), expected_hash)
        self.assertIn(module_name, sys.modules)
        linecache.cache[launcher_filename] = (0, None, [], launcher_filename)

        _unload_compiled_fn(fn)
        self.assertNotIn(module_name, sys.modules)
        self.assertEqual(cute_kernel._helion_cute_compiled_launchers, {})
        self.assertEqual(cute_kernel._helion_cute_launch_arg_cache, {})
        self.assertNotIn(launcher_filename, linecache.cache)

    def test_compiled_fn_unload_clears_triton_fast_path_cache(self) -> None:
        source = (
            "class Kernel:\n"
            "    def __init__(self):\n"
            "        self.cleared = False\n"
            "    def clear_fast_path_caches(self):\n"
            "        self.cleared = True\n"
            "_helion_call = Kernel()\n"
            "def call():\n"
            "    pass\n"
        )
        fn_spec = SerializedCompiledFunction("call", source, None, None)
        fn = _load_compiled_fn(fn_spec)
        kernel = fn.__globals__["_helion_call"]

        _unload_compiled_fn(fn)

        self.assertTrue(kernel.cleared)

    def test_compiled_fn_load_failure_does_not_leak_module(self) -> None:
        module_prefix = "_helion_autotune_subprocess_"
        modules_before = {
            name for name in sys.modules if name.startswith(module_prefix)
        }
        fn_spec = SerializedCompiledFunction(
            function_name="call",
            source_code="raise RuntimeError('load failed')\n",
            filename=None,
            module_name=None,
        )

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            _load_compiled_fn(fn_spec)

        modules_after = {name for name in sys.modules if name.startswith(module_prefix)}
        self.assertEqual(modules_after, modules_before)

    def test_benchmark_job_can_use_wall_clock_bench(self) -> None:
        fn = _ReturnValue(torch.empty(()))

        with (
            patch(
                "helion.autotuner.benchmark_job._load_compiled_fn",
                return_value=fn,
            ) as load_fn,
            patch(
                "helion.autotuner.benchmark_job.load_trusted_kernel_args",
                return_value=(),
            ) as load_args,
            patch("helion.autotuner.benchmark_job.do_bench") as event_bench,
            patch(
                "helion.autotuner.benchmark_job.do_bench_generic",
                return_value=1.25,
            ) as wall_clock_bench,
        ):
            result = BenchmarkJob(
                fn_spec=cast("SerializedCompiledFunction", object()),
                args_path="/tmp/args.pt",
                use_wall_clock=True,
            )()

        self.assertEqual(result, 1.25)
        load_fn.assert_called_once()
        load_args.assert_called_once_with("/tmp/args.pt")
        event_bench.assert_not_called()
        wall_clock_bench.assert_called_once()
        self.assertIsNone(wall_clock_bench.call_args.kwargs["fixed_repetitions"])
        self.assertIs(wall_clock_bench.call_args.kwargs["probe_long_kernel"], False)

    def test_wall_clock_fixed_repetitions_run_setup_and_measurements(self) -> None:
        invocation_count = 0

        def fn() -> None:
            nonlocal invocation_count
            invocation_count += 1

        with (
            patch("helion.autotuner.benchmarking.synchronize_device"),
            patch(
                "helion.autotuner.benchmarking._make_l2_cache_clearer",
                return_value=lambda: None,
            ),
        ):
            result = do_bench_generic(
                fn,
                warmup=1,
                rep=50,
                return_mode="median",
                fixed_repetitions=3,
            )

        self.assertEqual(invocation_count, 4)
        self.assertTrue(math.isfinite(cast("float", result)))

    def test_benchmark_job_forwards_long_kernel_probe(self) -> None:
        fn = _ReturnValue(torch.empty(()))

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
            BenchmarkJob(
                fn_spec=cast("SerializedCompiledFunction", object()),
                args_path="/tmp/args.pt",
                use_wall_clock=True,
                probe_long_kernel=True,
            )()

        self.assertIs(wall_clock_bench.call_args.kwargs["probe_long_kernel"], True)

    def test_benchmark_job_forwards_long_kernel_probe_on_event_path(self) -> None:
        fn = _ReturnValue(torch.empty(()))

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
                "helion.autotuner.benchmark_job.do_bench",
                return_value=1.25,
            ) as event_bench,
            patch("helion.autotuner.benchmark_job.do_bench_generic") as wall_bench,
        ):
            BenchmarkJob(
                fn_spec=cast("SerializedCompiledFunction", object()),
                args_path="/tmp/args.pt",
                use_wall_clock=False,
                probe_long_kernel=True,
            )()

        wall_bench.assert_not_called()
        self.assertIs(event_bench.call_args.kwargs["probe_long_kernel"], True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_event_timed_long_kernel_skips_redundant_estimates(self) -> None:
        # A kernel longer than both timing windows must be measured with
        # setup + single-call estimate + one timed repeat (3 launches), not
        # the 5-call estimate loop.
        invocation_count = 0
        sleep_cycles = int(50e6)  # tens of ms at ~GHz clocks

        def fn() -> None:
            nonlocal invocation_count
            invocation_count += 1
            torch.cuda._sleep(sleep_cycles)

        result = do_bench(
            fn,
            warmup=1,
            rep=1,
            return_mode="median",
            probe_long_kernel=True,
        )

        self.assertEqual(invocation_count, 3)
        self.assertGreater(cast("float", result), 1.0)

    def test_wall_clock_long_kernel_skips_redundant_estimates(self) -> None:
        invocation_count = 0

        def fn() -> None:
            nonlocal invocation_count
            invocation_count += 1

        with (
            patch("helion.autotuner.benchmarking.synchronize_device"),
            patch(
                "helion.autotuner.benchmarking._make_l2_cache_clearer",
                return_value=lambda: None,
            ),
            patch(
                "helion.autotuner.benchmarking.time.perf_counter",
                side_effect=(0.0, 0.1, 0.1, 0.2),
            ),
        ):
            result = do_bench_generic(
                fn,
                warmup=1,
                rep=50,
                return_mode="median",
                probe_long_kernel=True,
            )

        self.assertEqual(invocation_count, 3)
        self.assertAlmostEqual(cast("float", result), 100.0)

    def test_wall_clock_default_does_not_enable_long_kernel_probe(self) -> None:
        invocation_count = 0

        def fn() -> None:
            nonlocal invocation_count
            invocation_count += 1

        with (
            patch("helion.autotuner.benchmarking.synchronize_device"),
            patch(
                "helion.autotuner.benchmarking._make_l2_cache_clearer",
                return_value=lambda: None,
            ),
            patch(
                "helion.autotuner.benchmarking.time.perf_counter",
                side_effect=(0.0, 0.1, 0.1, 0.2, 0.2, 0.3),
            ),
        ):
            result = do_bench_generic(
                fn,
                warmup=1,
                rep=50,
                return_mode="median",
            )

        self.assertEqual(invocation_count, 9)
        self.assertAlmostEqual(cast("float", result), 100.0)

    def test_wall_clock_short_kernel_keeps_five_call_estimate(self) -> None:
        invocation_count = 0

        def fn() -> None:
            nonlocal invocation_count
            invocation_count += 1

        clock = (
            0.0,
            0.001,
            0.001,
            0.005,
            0.005,
            0.006,
            0.006,
            0.007,
            0.007,
            0.008,
        )
        with (
            patch("helion.autotuner.benchmarking.synchronize_device"),
            patch(
                "helion.autotuner.benchmarking._make_l2_cache_clearer",
                return_value=lambda: None,
            ),
            patch(
                "helion.autotuner.benchmarking.time.perf_counter",
                side_effect=clock,
            ),
        ):
            result = do_bench_generic(
                fn,
                warmup=1,
                rep=3,
                return_mode="median",
                probe_long_kernel=True,
            )

        self.assertEqual(invocation_count, 10)
        self.assertAlmostEqual(cast("float", result), 1.0)

    def test_event_benchmark_keeps_five_call_estimate(self) -> None:
        invocation_count = 0
        elapsed_times = [5.0, 1.0, 1.0, 1.0]

        def fn() -> None:
            nonlocal invocation_count
            invocation_count += 1

        active = SimpleNamespace()
        active.get_device_interface = lambda: SimpleNamespace(
            Event=lambda **kwargs: _FakeTimingEvent(elapsed_times),
            synchronize=lambda: None,
        )
        active.get_empty_cache_for_benchmark = lambda: object()
        active.clear_cache = lambda cache: None
        runtime = SimpleNamespace(driver=SimpleNamespace(active=active))
        triton_module = ModuleType("triton")
        triton_module.runtime = runtime
        testing_module = ModuleType("triton.testing")
        testing_module._summarize_statistics = lambda times, quantiles, return_mode: (
            statistics.median(times)
        )

        with patch.dict(
            sys.modules,
            {"triton": triton_module, "triton.testing": testing_module},
        ):
            result = do_bench(fn, warmup=1, rep=3, return_mode="median")

        self.assertEqual(invocation_count, 10)
        self.assertAlmostEqual(cast("float", result), 1.0)

    def test_long_kernel_branch_uses_synchronized_probe(self) -> None:
        run_batch = Mock(return_value=0.1)

        with patch(
            "helion.autotuner.benchmarking.sync_object", return_value=100.0
        ) as sync:
            estimate_ms, n_warmup = _estimate_runtime_and_warmup(
                run_batch,
                warmup=1,
                rep=50,
                process_group_name="workers",
            )

        run_batch.assert_called_once_with(1)
        sync.assert_called_once_with(0.1, process_group_name="workers")
        self.assertEqual(estimate_ms, 100.0)
        self.assertEqual(n_warmup, 0)

    def test_benchmark_job_forwards_fixed_repetitions(self) -> None:
        fn = _ReturnValue(torch.empty(()))

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
                fixed_repetitions=1,
            )()

        self.assertEqual(result, 1.25)
        self.assertEqual(wall_clock_bench.call_args.kwargs["fixed_repetitions"], 1)

    def test_benchmark_job_unloads_module_after_failure(self) -> None:
        fn = _ReturnValue(torch.empty(()))

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
                "helion.autotuner.benchmark_job.do_bench",
                side_effect=RuntimeError("benchmark failed"),
            ),
            patch("helion.autotuner.benchmark_job._unload_compiled_fn") as unload_fn,
            self.assertRaisesRegex(RuntimeError, "benchmark failed"),
        ):
            BenchmarkJob(
                fn_spec=cast("SerializedCompiledFunction", object()),
                args_path="/tmp/args.pt",
            )()

        unload_fn.assert_called_once_with(fn)

    def test_accuracy_check_job_passes(self) -> None:
        fn = _ReturnValue(torch.tensor([1.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            args_path = Path(tmpdir) / "args.pt"
            baseline_path = Path(tmpdir) / "baseline.pt"
            torch.save((), args_path)
            torch.save(torch.tensor([1.0]), baseline_path)

            with patch(
                "helion.autotuner.benchmark_job._load_compiled_fn",
                return_value=fn,
            ):
                result = AccuracyCheckJob(
                    fn_spec=cast("SerializedCompiledFunction", object()),
                    args_path=str(args_path),
                    baseline_path=str(baseline_path),
                    atol=0.0,
                    rtol=0.0,
                )()

        self.assertTrue(result.ok)
        self.assertEqual(result.message, "")

    def test_accuracy_check_job_reports_mismatch(self) -> None:
        fn = _ReturnValue(torch.tensor([2.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            args_path = Path(tmpdir) / "args.pt"
            baseline_path = Path(tmpdir) / "baseline.pt"
            torch.save((), args_path)
            torch.save(torch.tensor([1.0]), baseline_path)

            with patch(
                "helion.autotuner.benchmark_job._load_compiled_fn",
                return_value=fn,
            ):
                result = AccuracyCheckJob(
                    fn_spec=cast("SerializedCompiledFunction", object()),
                    args_path=str(args_path),
                    baseline_path=str(baseline_path),
                    atol=0.0,
                    rtol=0.0,
                )()

        self.assertFalse(result.ok)
        self.assertIn("Tensor-likes are not equal", result.message)

    def test_accuracy_check_job_reports_shape_mismatch(self) -> None:
        fn = _ReturnValue(torch.zeros(2, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            args_path = Path(tmpdir) / "args.pt"
            baseline_path = Path(tmpdir) / "baseline.pt"
            torch.save((), args_path)
            torch.save(torch.zeros(3, 2), baseline_path)

            with patch(
                "helion.autotuner.benchmark_job._load_compiled_fn",
                return_value=fn,
            ):
                result = AccuracyCheckJob(
                    fn_spec=cast("SerializedCompiledFunction", object()),
                    args_path=str(args_path),
                    baseline_path=str(baseline_path),
                    atol=0.0,
                    rtol=0.0,
                )()

        self.assertFalse(result.ok)
        self.assertIn("Tensor shape mismatch", result.message)

    def test_accuracy_check_job_reports_tensor_leaf_type_mismatch(self) -> None:
        fn = _ReturnValue(torch.tensor([1.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            args_path = Path(tmpdir) / "args.pt"
            baseline_path = Path(tmpdir) / "baseline.pt"
            torch.save((), args_path)
            torch.save(1.0, baseline_path)

            with patch(
                "helion.autotuner.benchmark_job._load_compiled_fn",
                return_value=fn,
            ):
                result = AccuracyCheckJob(
                    fn_spec=cast("SerializedCompiledFunction", object()),
                    args_path=str(args_path),
                    baseline_path=str(baseline_path),
                    atol=0.0,
                    rtol=0.0,
                )()

        self.assertFalse(result.ok)
        self.assertIn("Output leaf type mismatch", result.message)

    def test_subprocess_accuracy_check_uses_benchmark_timeout(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings(
            autotune_compile_timeout=3,
            autotune_benchmark_timeout=17,
        )
        provider._precompile_args_path = "/tmp/args.pt"
        provider._precompile_baseline_path = "/tmp/baseline.pt"
        provider._effective_atol = 0.0
        provider._effective_rtol = 0.0
        provider._scale_atol = True
        provider._benchmark_worker = Mock()
        provider._benchmark_worker.run.return_value = object()
        provider._subprocess_accuracy_check_enabled = lambda: True

        with patch(
            "helion.autotuner.benchmark_provider._serialize_compiled_fn",
            return_value=cast("SerializedCompiledFunction", object()),
        ):
            provider._run_subprocess_accuracy_check_job(
                cast("CompiledConfig", object())
            )

        provider._benchmark_worker.run.assert_called_once()
        _, kwargs = provider._benchmark_worker.run.call_args
        self.assertEqual(kwargs["timeout"], 17.0)

    def test_benchmark_timeout_has_worker_metric_and_status(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        provider._worker_failure_config_ids = []

        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=BenchmarkTimeout("timed out"),
        ) as run_job:
            result = provider._benchmark_function_subprocess(
                Config(), cast("CompiledConfig", object())
            )

        self.assertEqual(result, math.inf)
        self.assertEqual(provider._last_benchmark_failure_status, "timeout")
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 1)
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 0)
        run_job.assert_called_once()

    def test_unkillable_worker_aborts_subprocess_benchmark(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        provider._worker_failure_config_ids = []

        with (
            patch.object(
                provider,
                "_run_subprocess_benchmark_job",
                side_effect=BenchmarkWorkerUnkillable("worker remained alive"),
            ),
            self.assertRaisesRegex(BenchmarkWorkerUnkillable, "remained alive"),
        ):
            provider._benchmark_function_subprocess(
                Config(), cast("CompiledConfig", object())
            )

        self.assertEqual(provider._autotune_metrics.num_worker_failures, 0)

    def test_unkillable_worker_aborts_subprocess_accuracy_check(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings(autotune_accuracy_check=True)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        provider._worker_failure_config_ids = []

        with (
            patch.object(provider, "_run_subprocess_benchmark_job", return_value=1.0),
            patch.object(
                provider,
                "_run_subprocess_accuracy_check_job",
                side_effect=BenchmarkWorkerUnkillable("worker remained alive"),
            ),
            self.assertRaisesRegex(BenchmarkWorkerUnkillable, "remained alive"),
        ):
            provider._benchmark_function_subprocess(
                Config(), cast("CompiledConfig", object())
            )

        self.assertEqual(provider._autotune_metrics.num_worker_failures, 0)

    def test_unkillable_worker_is_not_skippable_for_multi_shape(self) -> None:
        child = SimpleNamespace(
            config_spec=SimpleNamespace(backend=None),
            settings=SimpleNamespace(autotune_ignore_errors=True),
        )

        self.assertFalse(
            MultiShapeBenchmarkProvider._is_skippable_child_failure(
                cast("LocalBenchmarkProvider", child),
                BenchmarkWorkerUnkillable("worker remained alive"),
            )
        )

    def test_compiler_seed_timeout_retries_with_three_repetitions(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings(autotune_accuracy_check=False)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=3
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        fn = cast("CompiledConfig", object())
        config = Config()
        provider.set_compiler_seed_configs([config])

        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=(BenchmarkTimeout("timed out"), 2.75),
        ) as run_job:
            result = provider._benchmark_function_subprocess(config, fn)

        self.assertEqual(result, 2.75)
        self.assertEqual(
            run_job.call_args_list,
            [
                unittest.mock.call(fn, warmup=1, rep=50),
                unittest.mock.call(
                    fn,
                    warmup=1,
                    rep=50,
                    fixed_repetitions=3,
                ),
            ],
        )
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 0)

    def test_compiler_seed_timeout_retry_is_bounded_per_seed(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings(autotune_accuracy_check=False)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=3
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        provider._worker_failure_config_ids = []
        provider.budget_exceeded_fn = Mock(return_value=False)
        first = Config(num_warps=4)
        second = Config(num_warps=8)
        provider.set_compiler_seed_configs([first, second])
        fn = cast("CompiledConfig", object())

        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=(
                BenchmarkTimeout("first timeout"),
                2.75,
                BenchmarkTimeout("second timeout"),
                3.25,
            ),
        ) as run_job:
            first_result = provider._benchmark_function_subprocess(first, fn)
            second_result = provider._benchmark_function_subprocess(second, fn)

        self.assertEqual(first_result, 2.75)
        self.assertEqual(second_result, 3.25)
        self.assertEqual(run_job.call_count, 4)
        self.assertEqual(provider.budget_exceeded_fn.call_count, 2)
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 0)

    def test_compiler_seed_timeout_retry_is_shared_by_source_aliases(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings(autotune_accuracy_check=False)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=3
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        provider._worker_failure_config_ids = []
        provider.budget_exceeded_fn = Mock(return_value=False)
        first = Config(num_warps=4)
        second = Config(num_warps=8)
        provider.set_compiler_seed_configs([first, second])
        fn = cast("CompiledConfig", object())

        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=(
                BenchmarkTimeout("first timeout"),
                2.75,
                BenchmarkTimeout("alias timeout"),
            ),
        ) as run_job:
            first_result = provider._benchmark_function_subprocess(
                first, fn, effective_source_hash="shared"
            )
            second_result = provider._benchmark_function_subprocess(
                second, fn, effective_source_hash="shared"
            )

        self.assertEqual(first_result, 2.75)
        self.assertEqual(second_result, math.inf)
        self.assertEqual(run_job.call_count, 3)
        provider.budget_exceeded_fn.assert_called_once_with()
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 1)

    def test_compiler_seed_retry_timeout_is_one_worker_failure(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=3
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        provider._worker_failure_config_ids = []
        config = Config()
        provider.set_compiler_seed_configs([config])

        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=(
                BenchmarkTimeout("first timeout"),
                BenchmarkTimeout("retry timeout"),
            ),
        ) as run_job:
            result = provider._benchmark_function_subprocess(
                config,
                cast("CompiledConfig", object()),
            )

        self.assertEqual(result, math.inf)
        self.assertEqual(run_job.call_count, 2)
        self.assertEqual(provider._last_benchmark_failure_status, "timeout")
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 1)
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 0)

    def test_compiler_seed_timeout_does_not_retry_after_budget(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=3
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        provider._worker_failure_config_ids = []
        provider.budget_exceeded_fn = Mock(return_value=True)
        config = Config()
        provider.set_compiler_seed_configs([config])

        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=BenchmarkTimeout("timed out"),
        ) as run_job:
            result = provider._benchmark_function_subprocess(
                config,
                cast("CompiledConfig", object()),
            )

        self.assertEqual(result, math.inf)
        run_job.assert_called_once()
        provider.budget_exceeded_fn.assert_called_once_with()
        self.assertEqual(provider._last_benchmark_failure_status, "timeout")

    def test_disabled_compiler_seed_timeout_retry_does_not_run(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        provider._worker_failure_config_ids = []
        config = Config()
        provider.set_compiler_seed_configs([config])

        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=BenchmarkTimeout("timed out"),
        ) as run_job:
            result = provider._benchmark_function_subprocess(
                config,
                cast("CompiledConfig", object()),
            )

        self.assertEqual(result, math.inf)
        run_job.assert_called_once()
        self.assertEqual(provider._last_benchmark_failure_status, "timeout")

    def test_benchmark_worker_death_has_error_status(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None
        )
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()
        provider._worker_failure_config_ids = []

        with patch.object(
            provider,
            "_run_subprocess_benchmark_job",
            side_effect=BenchmarkWorkerDied("worker exited"),
        ) as run_job:
            result = provider._benchmark_function_subprocess(
                Config(), cast("CompiledConfig", object())
            )

        self.assertEqual(result, math.inf)
        self.assertEqual(provider._last_benchmark_failure_status, "error")
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 1)
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 0)
        run_job.assert_called_once()

    def test_fixed_repetition_job_uses_existing_benchmark_timeout(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings(autotune_benchmark_timeout=17)
        provider.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
        provider._precompile_args_path = "/tmp/args.pt"
        provider._benchmark_worker = Mock()
        provider._benchmark_worker.run.return_value = 1.25
        provider._subprocess_benchmark_uses_wall_clock = lambda: True

        with patch(
            "helion.autotuner.benchmark_provider._serialize_compiled_fn",
            return_value=cast("SerializedCompiledFunction", object()),
        ):
            result = provider._run_subprocess_benchmark_job(
                cast("CompiledConfig", object()),
                warmup=1,
                rep=50,
                fixed_repetitions=3,
            )

        self.assertEqual(result, 1.25)
        provider._benchmark_worker.run.assert_called_once()
        job = provider._benchmark_worker.run.call_args.args[0]
        self.assertEqual(job.fixed_repetitions, 3)
        self.assertTrue(job.probe_long_kernel)
        self.assertEqual(provider._benchmark_worker.run.call_args.kwargs["timeout"], 17)

    def test_long_kernel_probe_is_cute_flash_gated(self) -> None:
        # The probe applies to flash searches on both timer paths: the
        # event-timed do_bench now short-circuits its estimate loop the same
        # way do_bench_generic does for multi-second candidates.
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.config_spec = SimpleNamespace(cute_flash_search_enabled=False)
        self.assertFalse(provider._probe_long_cute_flash_kernel())

        provider.config_spec.cute_flash_search_enabled = True
        self.assertTrue(provider._probe_long_cute_flash_kernel())

    def test_subprocess_accuracy_check_skips_mutated_args(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings()
        provider.mutated_arg_indices = [0]
        provider._subprocess_benchmark_enabled = lambda: True

        self.assertFalse(provider._subprocess_accuracy_check_enabled())

    def test_load_trusted_kernel_args_accepts_python_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "args.pt"
            torch.save((_ReturnValue(3),), path)

            load_trusted_kernel_args.cache_clear()
            loaded = load_trusted_kernel_args(str(path))

        self.assertIsInstance(loaded[0], _ReturnValue)
        self.assertEqual(loaded[0].value, 3)

    @skipIfPallasTpu("spawned workers cannot acquire an initialized TPU device")
    def test_spawn_precompile_loads_trusted_python_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            args_path = tmp_path / "args.pt"
            result_path = tmp_path / "result.pkl"
            torch.save((_ReturnValue(3),), args_path)
            fn_spec = SerializedCompiledFunction(
                function_name="call_arg",
                source_code="def call_arg(fn):\n    return fn()\n",
                filename="<test_spawn_precompile_loads_trusted_python_args>",
                module_name=None,
            )

            process = mp.get_context("spawn").Process(
                target=_run_kernel_in_subprocess_spawn,
                args=(fn_spec, str(args_path), str(result_path), "@test"),
            )
            process.start()
            try:
                process.join(timeout=30)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)

                self.assertEqual(process.exitcode, 0)
                with result_path.open("rb") as f:
                    result = pickle.load(f)
                self.assertEqual(result, {"status": "ok"})
            finally:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)

    @skipIfPallasTpu("spawned workers cannot acquire an initialized TPU device")
    def test_spawn_loads_source_module_from_file(self) -> None:
        # Regression: a kernel loaded by path / notebook / exec lives under a
        # synthetic module name that only the parent's sys.modules knows about.
        # The generated code does `import <synthetic> as _source_module` for its
        # module-globals; a fresh spawn worker cannot import that name. The
        # serializer ships the origin module's real file in `source_modules` and
        # the worker re-registers it before exec, so the import resolves.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # The "origin" kernel module (only present here on disk, never on the
            # worker's import path under this name).
            origin_name = "helion_test_synthetic_origin_xyz"
            origin_file = tmp_path / "origin.py"
            origin_file.write_text("MAGIC = 4321\n", encoding="utf-8")
            args_path = tmp_path / "args.pt"
            result_path = tmp_path / "result.pkl"
            torch.save((), args_path)
            # Generated code references the origin via `_source_module`, exactly
            # like Helion codegen emits for a module-global.
            source_code = (
                f"import {origin_name} as _source_module\n"
                "def call_arg():\n"
                "    assert _source_module.MAGIC == 4321\n"
                "    return _source_module.MAGIC\n"
            )
            fn_spec = SerializedCompiledFunction(
                function_name="call_arg",
                source_code=source_code,
                filename="<test_spawn_loads_source_module_from_file>",
                module_name=None,
                source_modules=[(origin_name, str(origin_file))],
            )
            process = mp.get_context("spawn").Process(
                target=_run_kernel_in_subprocess_spawn,
                args=(fn_spec, str(args_path), str(result_path), "@test"),
            )
            process.start()
            try:
                process.join(timeout=30)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
                with result_path.open("rb") as f:
                    result = pickle.load(f)
                self.assertEqual(result, {"status": "ok"})
            finally:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)

    def test_timeout_kills_worker(self) -> None:
        worker = BenchmarkWorker()
        try:
            t0 = time.time()
            with self.assertRaises(BenchmarkTimeout):
                worker.run(_Sleep(60), timeout=0.5)
            self.assertLess(time.time() - t0, 15.0)
            self.assertFalse(worker.alive())
            # Next call respawns.
            self.assertEqual(worker.run(_ReturnValue(7), timeout=30.0), 7)
        finally:
            worker.shutdown()

    def test_timeout_with_unkillable_worker_prevents_respawn(self) -> None:
        worker = BenchmarkWorker()
        process = Mock(pid=123)
        process.is_alive.return_value = True
        connection = Mock()
        kill_started = threading.Event()

        def poll_until_watchdog_runs(timeout: float) -> bool:
            self.assertTrue(kill_started.wait(timeout=1.0))
            return False

        connection.poll.side_effect = poll_until_watchdog_runs
        worker._process = process
        worker._parent_connection = connection

        with (
            patch(
                "helion.autotuner.benchmark_worker.signal_process_tree",
                side_effect=lambda *_args: kill_started.set(),
            ) as signal_tree,
            self.assertRaisesRegex(
                BenchmarkWorkerUnkillable,
                "refusing to launch another worker",
            ),
        ):
            worker.run(_ReturnValue(7), timeout=0.01)

        signal_tree.assert_called_once_with(process, signal.SIGKILL)
        process.join.assert_called_once_with(timeout=5)
        connection.close.assert_called_once_with()
        self.assertIs(worker._process, process)
        self.assertIs(worker._parent_connection, connection)

        with (
            patch.object(worker, "_start") as start,
            self.assertRaises(BenchmarkWorkerUnkillable),
        ):
            worker.run(_ReturnValue(8), timeout=30.0)
        start.assert_not_called()

    def test_watchdog_claim_cannot_race_successful_completion(self) -> None:
        worker = BenchmarkWorker()
        process = Mock(pid=123)
        process.is_alive.side_effect = (True, False)
        connection = Mock()
        event_type = threading.Event
        watchdog_checked_done = event_type()
        release_watchdog = event_type()

        class PausingDoneEvent:
            def __init__(self) -> None:
                self._event = event_type()

            def is_set(self) -> bool:
                result = self._event.is_set()
                if not result:
                    watchdog_checked_done.set()
                    self.assert_released()
                return result

            def assert_released(self) -> None:
                if not release_watchdog.wait(timeout=2.0):
                    raise AssertionError("watchdog release timed out")

            def set(self) -> None:
                self._event.set()

        timeout_event = event_type()
        done_event = PausingDoneEvent()
        event_calls = 0

        def event_factory():
            nonlocal event_calls
            event_calls += 1
            if event_calls == 1:
                return timeout_event
            if event_calls == 2:
                return done_event
            return event_type()

        def receive_after_watchdog_check() -> int:
            if not watchdog_checked_done.wait(timeout=2.0):
                raise AssertionError("watchdog did not inspect completion state")
            return 7

        connection.poll.return_value = True
        connection.recv.side_effect = receive_after_watchdog_check
        worker._process = process
        worker._parent_connection = connection
        outcome: dict[str, object] = {}

        def run_worker() -> None:
            try:
                outcome["result"] = worker.run(_ReturnValue(7), timeout=0.01)
            except BaseException as error:
                outcome["error"] = error

        runner = threading.Thread(target=run_worker)
        with (
            patch(
                "helion.autotuner.benchmark_worker.threading.Event",
                side_effect=event_factory,
            ),
            patch("helion.autotuner.benchmark_worker.signal_process_tree"),
        ):
            runner.start()
            self.assertTrue(watchdog_checked_done.wait(timeout=2.0))
            runner.join(timeout=0.1)
            self.assertTrue(runner.is_alive())
            release_watchdog.set()
            runner.join(timeout=2.0)

        self.assertFalse(runner.is_alive())
        self.assertNotIn("result", outcome)
        self.assertIsInstance(outcome.get("error"), BenchmarkTimeout)

    def test_shutdown_surfaces_unkillable_worker(self) -> None:
        worker = BenchmarkWorker()
        process = Mock(pid=123)
        process.is_alive.side_effect = (True, True)
        connection = Mock()
        worker._process = process
        worker._parent_connection = connection

        with (
            patch("helion.autotuner.benchmark_worker.signal_process_tree"),
            self.assertRaisesRegex(
                BenchmarkWorkerUnkillable,
                "refusing to launch another worker",
            ),
        ):
            worker.shutdown()

        self.assertIs(worker._process, process)
        self.assertIs(worker._parent_connection, connection)

    def test_provider_cleanup_retains_unkillable_worker(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        worker = Mock()
        worker.shutdown.side_effect = BenchmarkWorkerUnkillable("worker remained alive")
        provider._benchmark_worker = worker

        with self.assertRaisesRegex(BenchmarkWorkerUnkillable, "remained alive"):
            provider.cleanup()

        self.assertIs(provider._benchmark_worker, worker)

    def test_multi_shape_cleanup_attempts_every_child(self) -> None:
        provider = MultiShapeBenchmarkProvider.__new__(MultiShapeBenchmarkProvider)
        first = Mock()
        second = Mock()
        first.cleanup.side_effect = BenchmarkWorkerUnkillable("worker remained alive")
        second.cleanup.side_effect = OSError("ordinary cleanup failure")
        provider.children = [first, second]
        provider._original_configs_by_materialized_key = {"a": []}
        provider._anchor_fns_by_materialized_key = {"a": Mock()}
        provider._effective_source_repairs = {Config(): Mock()}
        provider.budget_exceeded_fn = Mock()

        with self.assertRaisesRegex(BenchmarkWorkerUnkillable, "remained alive"):
            provider.cleanup()

        second.cleanup.assert_called_once_with()
        first.cleanup.assert_called_once_with()
        self.assertEqual(provider._original_configs_by_materialized_key, {})
        self.assertEqual(provider._anchor_fns_by_materialized_key, {})
        self.assertEqual(provider._effective_source_repairs, {})

    def test_timeout_kills_worker_process_group(self) -> None:
        worker = BenchmarkWorker()
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "child.pid"
            try:
                worker.run(_ReturnValue(None), timeout=30.0)
                with self.assertRaises(BenchmarkTimeout):
                    worker.run(_SpawnChildAndSleep(str(pid_path)), timeout=1.0)
                child_pid = int(pid_path.read_text())
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
                    except FileNotFoundError:
                        state = "gone"
                    if state in {"gone", "Z"}:
                        break
                    time.sleep(0.1)
                self.assertIn(state, {"gone", "Z"})
            finally:
                worker.shutdown()

    def test_watchdog_kills_worker_when_poll_timeout_is_ignored(self) -> None:
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
                self.assertRaises(BenchmarkTimeout),
            ):
                worker.run(_Sleep(60), timeout=0.5)
            self.assertLess(time.time() - t0, 15.0)
            self.assertFalse(worker.alive())
            self.assertEqual(worker.run(_ReturnValue(7), timeout=30.0), 7)
        finally:
            worker.shutdown()

    def test_sticky_error_kills_worker(self) -> None:
        # Errors matching _UNRECOVERABLE_RUNTIME_ERROR_RE force the worker
        # to be killed so the next call spawns a fresh CUDA context.
        worker = BenchmarkWorker()
        try:
            with self.assertRaises(RuntimeError) as ctx:
                worker.run(_RaiseRuntimeError("illegal memory access"), timeout=30.0)
            self.assertIn("illegal memory access", str(ctx.exception))
            self.assertFalse(worker.alive())
            self.assertEqual(worker.run(_ReturnValue(42), timeout=30.0), 42)
        finally:
            worker.shutdown()

    def test_worker_crash_raises_died(self) -> None:
        worker = BenchmarkWorker()
        try:
            with self.assertRaises(BenchmarkWorkerDied):
                worker.run(_Crash(), timeout=30.0)
            self.assertFalse(worker.alive())
        finally:
            worker.shutdown()

    def test_unpickleable_worker_exception_is_serialized(self) -> None:
        worker = BenchmarkWorker()
        try:
            with self.assertRaises(BenchmarkSubprocessError) as ctx:
                worker.run(_RaiseUnpickleableLocalException(), timeout=30.0)
            self.assertIn("unpickleable", str(ctx.exception))
            self.assertTrue(worker.alive())
            self.assertEqual(worker.run(_ReturnValue(7), timeout=30.0), 7)
        finally:
            worker.shutdown()


class TestSuspiciousRebenchmark(unittest.TestCase):
    def test_isolated_unkillable_worker_is_fatal(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings(autotune_benchmark_subprocess=True)
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()

        with (
            patch.object(provider, "_subprocess_benchmark_enabled", return_value=True),
            patch.object(
                provider,
                "_run_subprocess_benchmark_job",
                side_effect=BenchmarkWorkerUnkillable("worker remained alive"),
            ),
            self.assertRaisesRegex(BenchmarkWorkerUnkillable, "remained alive"),
        ):
            provider.benchmark_isolated(
                [cast("CompiledConfig", object())],
                warmup=1,
                rep=100,
            )

        self.assertEqual(
            provider._autotune_metrics.num_isolated_rebenchmark_timeouts, 0
        )

    def test_isolated_timeout_is_distinct_from_unavailable_timing(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings(autotune_benchmark_subprocess=True)
        provider.log = Mock()
        provider._autotune_metrics = AutotuneMetrics()

        with (
            patch.object(provider, "_subprocess_benchmark_enabled", return_value=True),
            patch.object(
                provider,
                "_run_subprocess_benchmark_job",
                side_effect=BenchmarkTimeout("timed out"),
            ),
        ):
            results = provider.benchmark_isolated(
                [cast("CompiledConfig", object())],
                warmup=1,
                rep=100,
            )

        self.assertEqual(results, [IsolatedBenchmarkFailure("timeout")])
        self.assertEqual(
            provider._autotune_metrics.num_isolated_rebenchmark_timeouts, 1
        )
        self.assertEqual(
            provider._autotune_metrics.to_dict()["num_isolated_rebenchmark_timeouts"],
            1,
        )

        with (
            patch.object(provider, "_subprocess_benchmark_enabled", return_value=True),
            patch.object(
                provider,
                "_run_subprocess_benchmark_job",
                side_effect=BenchmarkWorkerDied("worker exited"),
            ),
        ):
            results = provider.benchmark_isolated(
                [cast("CompiledConfig", object())],
                warmup=1,
                rep=100,
            )

        self.assertEqual(results, [None])
        self.assertEqual(
            provider._autotune_metrics.num_isolated_rebenchmark_timeouts, 1
        )

    def test_effective_source_quarantine_removes_cached_results_and_repairs(
        self,
    ) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        shared_fn = SimpleNamespace(source_hash="shared")
        other_fn = SimpleNamespace(source_hash="other")
        provider.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
            )
        )
        provider._invalid_effective_source_hashes = set()
        provider._effective_source_results = {
            "shared": BenchmarkResult(Config(), shared_fn, 1.0, "ok", None)
        }
        provider._pending_effective_source_failures = {"shared": [object()]}
        shared_config = Config(num_warps=4)
        other_config = Config(num_warps=8)
        provider._effective_source_repairs = {
            shared_config: BenchmarkResult(
                shared_config, shared_fn, 1.0, "deduplicated", None
            ),
            other_config: BenchmarkResult(
                other_config, other_fn, 2.0, "deduplicated", None
            ),
        }

        provider.invalidate_effective_source_hash("shared")

        self.assertEqual(provider._invalid_effective_source_hashes, {"shared"})
        self.assertNotIn("shared", provider._effective_source_results)
        self.assertNotIn("shared", provider._pending_effective_source_failures)
        self.assertNotIn(shared_config, provider._effective_source_repairs)
        self.assertIn(other_config, provider._effective_source_repairs)

    def test_subprocess_benchmark_defaults_suspicious_rebenchmark_ratio(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_AUTOTUNE_SUSPICIOUS_REBENCHMARK_RATIO", None)
            self.assertEqual(
                Settings(
                    autotune_benchmark_subprocess=True
                ).get_suspicious_rebenchmark_ratio(),
                0.9,
            )
            self.assertIsNone(
                Settings(
                    autotune_benchmark_subprocess=False
                ).get_suspicious_rebenchmark_ratio()
            )
        self.assertEqual(
            Settings(
                autotune_benchmark_subprocess=True,
                autotune_suspicious_rebenchmark_ratio=0.75,
            ).get_suspicious_rebenchmark_ratio(),
            0.75,
        )

    def test_confirm_suspicious_rebenchmark_timings(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.confirm_fns: list[object] | None = None
                self.confirm_warmup: int | None = None
                self.confirm_rep: int | None = None

            def benchmark_isolated(
                self,
                fns: list[object],
                *,
                warmup: int,
                rep: int,
                desc: str,
            ) -> list[float | None]:
                self.confirm_fns = fns
                self.confirm_warmup = warmup
                self.confirm_rep = rep
                return [0.92]

        def fn_a() -> None:
            pass

        def fn_b() -> None:
            pass

        provider = FakeProvider()
        search = SimpleNamespace(
            settings=Settings(autotune_benchmark_subprocess=True),
            benchmark_provider=provider,
        )
        members = [
            PopulationMember(fn=fn_a, perfs=[1.00], flat_values=[], config=Config()),
            PopulationMember(fn=fn_b, perfs=[1.00], flat_values=[], config=Config()),
        ]

        timings = PopulationBasedSearch._confirm_suspicious_rebenchmark_timings(
            cast("Any", search),
            members,
            [0.70, 0.95],
            desc="verify",
        )

        self.assertEqual(provider.confirm_fns, [fn_a])
        self.assertEqual(provider.confirm_warmup, 25)
        self.assertEqual(provider.confirm_rep, 100)
        self.assertEqual(timings, [0.92, 0.95])

    def test_confirm_suspicious_rebenchmark_keeps_unconfirmed_timings(self) -> None:
        class FakeProvider:
            def benchmark_isolated(
                self,
                fns: list[object],
                *,
                warmup: int,
                rep: int,
                desc: str,
            ) -> list[float | None]:
                return [0.92, None]

        def fn_a() -> None:
            pass

        def fn_b() -> None:
            pass

        search = SimpleNamespace(
            settings=Settings(autotune_benchmark_subprocess=True),
            benchmark_provider=FakeProvider(),
        )
        members = [
            PopulationMember(fn=fn_a, perfs=[1.00], flat_values=[], config=Config()),
            PopulationMember(fn=fn_b, perfs=[1.00], flat_values=[], config=Config()),
        ]

        timings = PopulationBasedSearch._confirm_suspicious_rebenchmark_timings(
            cast("Any", search),
            members,
            [0.70, 0.80],
            desc="verify",
        )

        self.assertEqual(timings, [0.92, 0.80])

    def test_cute_flash_suspicious_confirmation_timeout_invalidates(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.mutated_arg_indices: list[int] = []
                self.invalidated_sources: list[str] = []
                self.assert_one_fn = 0

            def benchmark_isolated(self, fns, **kwargs):
                self.assert_one_fn = len(fns)
                return [IsolatedBenchmarkFailure("timeout")]

            def invalidate_effective_source_hash(self, source_hash: str) -> None:
                self.invalidated_sources.append(source_hash)

        def custom_benchmark_fn(fns, *, repeat, desc=None):
            return [0.5, 0.95]

        def fn_a() -> None:
            pass

        def fn_b() -> None:
            pass

        fn_a.source_hash = "timed-out"  # type: ignore[attr-defined]
        fn_b.source_hash = "healthy"  # type: ignore[attr-defined]
        provider = FakeProvider()
        search = object.__new__(PatternSearch)
        search.settings = Settings(
            autotune_benchmark_fn=custom_benchmark_fn,
            autotune_benchmark_subprocess=True,
            autotune_suspicious_rebenchmark_ratio=0.9,
        )
        search.benchmark_provider = provider  # type: ignore[assignment]
        search.best_perf_so_far = 1.0
        search.args = ()
        search.log = Mock()
        search.kernel = SimpleNamespace(  # type: ignore[assignment]
            env=SimpleNamespace(process_group_name=None)
        )
        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
            ),
            backend_name="cute",
            cute_flash_search_enabled=True,
        )
        search._compiler_seed_members = []
        timed_out = PopulationMember(
            fn=fn_a,
            perfs=[1.0],
            flat_values=[],
            config=Config(num_warps=4),
            status="ok",
        )
        healthy = PopulationMember(
            fn=fn_b,
            perfs=[1.0],
            flat_values=[],
            config=Config(num_warps=8),
            status="ok",
        )
        search.population = [timed_out, healthy]
        search._benchmarked_members = {
            member.config: member for member in search.population
        }
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        search.rebenchmark([timed_out, healthy], use_isolated=False)

        self.assertEqual(provider.assert_one_fn, 1)
        self.assertEqual(provider.invalidated_sources, ["timed-out"])
        self.assertEqual(timed_out.perfs, [math.inf])
        self.assertEqual(timed_out.status, "timeout")
        self.assertEqual(healthy.perfs, [1.0, 0.95])

    def test_rebenchmark_uses_isolated_provider(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.mutated_arg_indices: list[int] = []
                self.fns: list[object] | None = None
                self.warmup: int | None = None
                self.rep: int | None = None
                self.desc: str | None = None

            def benchmark_isolated(
                self,
                fns: list[object],
                *,
                warmup: int,
                rep: int,
                desc: str,
            ) -> list[float | None]:
                self.fns = fns
                self.warmup = warmup
                self.rep = rep
                self.desc = desc
                return [0.50, None]

        def fn_a() -> None:
            pass

        def fn_b() -> None:
            pass

        provider = FakeProvider()
        search = object.__new__(PatternSearch)
        search.settings = Settings(autotune_benchmark_subprocess=True)
        search.benchmark_provider = provider  # type: ignore[assignment]
        search.best_perf_so_far = 1.0
        search.kernel = SimpleNamespace(  # type: ignore[assignment]
            env=SimpleNamespace(process_group_name=None)
        )
        members = [
            PopulationMember(fn=fn_a, perfs=[1.00], flat_values=[], config=Config()),
            PopulationMember(fn=fn_b, perfs=[0.80], flat_values=[], config=Config()),
        ]

        search.rebenchmark(members, desc="verify")

        self.assertEqual(provider.fns, [fn_a, fn_b])
        self.assertEqual(provider.warmup, 1)
        self.assertEqual(provider.rep, 200)
        self.assertEqual(provider.desc, "verify")
        self.assertEqual(members[0].perfs, [1.00, 0.50])
        self.assertEqual(members[1].perfs, [0.80, 0.80])
        self.assertEqual(search.best_perf_so_far, 0.50)

    def test_cute_flash_isolated_timeout_invalidates_source_aliases(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.mutated_arg_indices: list[int] = []
                self.invalidated_sources: list[str] = []

            def benchmark_isolated(self, fns, **kwargs):
                return [IsolatedBenchmarkFailure("timeout"), 2.5]

            def invalidate_effective_source_hash(self, source_hash: str) -> None:
                self.invalidated_sources.append(source_hash)

        backend = SimpleNamespace(
            generated_source_hash=lambda fn: fn.source_hash,
        )
        search = object.__new__(LFBOPatternSearch)
        search.settings = Settings(autotune_benchmark_subprocess=True)
        search.benchmark_provider = FakeProvider()  # type: ignore[assignment]
        search.best_perf_so_far = 1.0
        search.kernel = SimpleNamespace(  # type: ignore[assignment]
            env=SimpleNamespace(process_group_name=None)
        )
        search.config_spec = SimpleNamespace(
            backend=backend,
            backend_name="cute",
            cute_flash_search_enabled=True,
        )
        search._compiler_seed_members = []

        timed_out = PopulationMember(
            fn=SimpleNamespace(source_hash="shared"),
            perfs=[1.0],
            flat_values=[],
            config=Config(num_warps=4),
            status="ok",
        )
        alias = PopulationMember(
            fn=SimpleNamespace(source_hash="shared"),
            perfs=[1.0],
            flat_values=[],
            config=Config(num_warps=8),
            status="deduplicated",
        )
        healthy = PopulationMember(
            fn=SimpleNamespace(source_hash="healthy"),
            perfs=[2.0],
            flat_values=[],
            config=Config(num_warps=16),
            status="ok",
        )
        compiler_seed = PopulationMember(
            fn=SimpleNamespace(source_hash="compiler-healthy"),
            perfs=[1.75],
            flat_values=[],
            config=Config(num_warps=32),
            status="ok",
        )
        search.population = [timed_out, alias, healthy]
        search._compiler_seed_members = [compiler_seed]
        search._benchmarked_members = {
            member.config: member for member in search.population
        }
        search._pinned_finalist_configs = {timed_out.config}
        search._pinned_finalist_members = {timed_out.config: timed_out}
        pruned_alias_config = Config(num_warps=2)
        search.train_configs = [
            timed_out.config,
            alias.config,
            pruned_alias_config,
            healthy.config,
        ]
        search.train_source_hashes = ["shared", "shared", "shared", "healthy"]
        search.train_y = [1.0, 1.1, 1.2, 2.0]

        search.rebenchmark([timed_out, healthy], desc="verify")

        self.assertEqual(
            search.benchmark_provider.invalidated_sources,  # type: ignore[attr-defined]
            ["shared"],
        )
        for member in (timed_out, alias):
            self.assertEqual(member.perfs, [math.inf])
            self.assertEqual(member.status, "timeout")
            self.assertNotIn(member.config, search._benchmarked_members)
        self.assertNotIn(timed_out.config, search._pinned_finalist_members)
        self.assertEqual(healthy.perfs, [2.0, 2.5])
        self.assertEqual(search.train_y, [math.inf, math.inf, math.inf, 2.0])
        self.assertEqual(search.best_perf_so_far, 1.75)

    def test_cute_flash_transient_duplicate_failure_keeps_prior_snapshot(
        self,
    ) -> None:
        search = object.__new__(PopulationBasedSearch)
        search.config_spec = SimpleNamespace(
            backend_name="cute",
            cute_flash_search_enabled=True,
        )
        search.best_perf_so_far = 1.0
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        config = Config(num_warps=4)
        original = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[],
            config=config,
            status="ok",
        )
        failed_duplicate = PopulationMember(
            fn=lambda: None,
            perfs=[],
            flat_values=[],
            config=Config(num_warps=4),
            status="error",
        )
        search.pin_finalist_config(config)
        search._record_benchmarked_member(original)

        search._apply_rebenchmark_timings([failed_duplicate], [math.inf])

        self.assertEqual(failed_duplicate.perfs, [math.inf])
        self.assertEqual(
            search._pinned_finalist_members[config].perfs,
            original.perfs,
        )
        self.assertEqual(
            search._benchmarked_members[config].perfs,
            original.perfs,
        )

    def test_non_cute_isolated_timeout_keeps_prior_timing(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.mutated_arg_indices: list[int] = []

            def benchmark_isolated(self, fns, **kwargs):
                return [IsolatedBenchmarkFailure("timeout"), 0.75]

            def invalidate_effective_source_hash(self, source_hash: str) -> None:
                raise AssertionError("non-CuTe source must not be invalidated")

        search = object.__new__(PatternSearch)
        search.settings = Settings(autotune_benchmark_subprocess=True)
        search.benchmark_provider = FakeProvider()  # type: ignore[assignment]
        search.best_perf_so_far = 1.0
        search.kernel = SimpleNamespace(  # type: ignore[assignment]
            env=SimpleNamespace(process_group_name=None)
        )
        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
            ),
            backend_name="triton",
            cute_flash_search_enabled=False,
        )
        retained = PopulationMember(
            fn=SimpleNamespace(source_hash="first"),
            perfs=[1.0],
            flat_values=[],
            config=Config(num_warps=4),
            status="ok",
        )
        measured = PopulationMember(
            fn=SimpleNamespace(source_hash="second"),
            perfs=[0.8],
            flat_values=[],
            config=Config(num_warps=8),
            status="ok",
        )

        search.rebenchmark([retained, measured], desc="verify")

        self.assertEqual(retained.perfs, [1.0, 1.0])
        self.assertEqual(retained.status, "ok")
        self.assertEqual(measured.perfs, [0.8, 0.75])

    def test_non_cute_isolated_runtime_error_keeps_existing_inf_behavior(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.mutated_arg_indices: list[int] = []

            def benchmark_isolated(self, fns, **kwargs):
                return [IsolatedBenchmarkFailure("error"), 0.75]

            def invalidate_effective_source_hash(self, source_hash: str) -> None:
                raise AssertionError("non-CuTe source must not be quarantined")

        search = object.__new__(PatternSearch)
        search.settings = Settings(autotune_benchmark_subprocess=True)
        search.benchmark_provider = FakeProvider()  # type: ignore[assignment]
        search.best_perf_so_far = 1.0
        search.kernel = SimpleNamespace(  # type: ignore[assignment]
            env=SimpleNamespace(process_group_name=None)
        )
        search.config_spec = SimpleNamespace(
            backend_name="triton",
            cute_flash_search_enabled=False,
        )
        failed = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[],
            config=Config(num_warps=4),
            status="ok",
        )
        measured = PopulationMember(
            fn=lambda: None,
            perfs=[0.8],
            flat_values=[],
            config=Config(num_warps=8),
            status="ok",
        )

        search.rebenchmark([failed, measured], desc="verify")

        self.assertEqual(failed.perfs, [1.0, math.inf])
        self.assertEqual(failed.status, "ok")
        self.assertEqual(measured.perfs, [0.8, 0.75])


# Subprocess benchmarking depends on Backend.supports_precompile(); only the
# Triton backend supports it (Pallas/CuTe return False).
@onlyBackends(["triton"])
class TestSubprocessBenchmarkIntegration(RefEagerTestDisabled, unittest.TestCase):
    @skipIfXPU("matmul config space includes maxnreg, unsupported on XPU")
    def test_autotune_with_subprocess_bench(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

        examples_dir = Path(__file__).parent.parent / "examples"
        matmul = import_path(examples_dir / "matmul.py").matmul

        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = matmul.bind(args)
        bound_kernel.settings.autotune_benchmark_subprocess = True
        bound_kernel.settings.autotune_benchmark_timeout = 60
        bound_kernel.settings.autotune_precompile = None
        # The autotuner reseeds `random` from this setting, so pinning it (not
        # random.seed) is what makes the config sequence reproducible.
        bound_kernel.settings.autotune_random_seed = 123

        random.seed(123)
        RandomSearch(bound_kernel, args, 10).autotune()

    @skipIfXPU("matmul config space includes maxnreg, unsupported on XPU")
    def test_autotune_continues_when_subprocess_reports_inf(self) -> None:
        # Patches _benchmark_function_subprocess to return inf for a
        # fraction of configs, simulating BenchmarkTimeout / worker death;
        # autotune must still pick a best config from the rest.
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

        original = LocalBenchmarkProvider._benchmark_function_subprocess
        call_count = [0, 0]  # [total, simulated_failures]

        def maybe_fail(
            self: LocalBenchmarkProvider,
            config: Config,
            fn: CompiledConfig,
        ) -> float | None:
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                call_count[1] += 1
                self._last_benchmark_failure_status = "timeout"
                self._autotune_metrics.num_worker_failures += 1
                return math.inf
            return original(self, config, fn)

        examples_dir = Path(__file__).parent.parent / "examples"
        matmul = import_path(examples_dir / "matmul.py").matmul

        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = matmul.bind(args)
        bound_kernel.settings.autotune_benchmark_subprocess = True
        bound_kernel.settings.autotune_benchmark_timeout = 60
        bound_kernel.settings.autotune_precompile = None
        # The autotuner reseeds `random` from this setting, so pinning it (not
        # random.seed) is what makes the config sequence reproducible.
        bound_kernel.settings.autotune_random_seed = 123

        random.seed(123)
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function_subprocess",
            maybe_fail,
        ):
            search = RandomSearch(bound_kernel, args, 10)
            search.autotune()

        self.assertGreaterEqual(call_count[0], 6)
        self.assertGreaterEqual(call_count[1], 2)
        self.assertEqual(search._autotune_metrics.num_worker_failures, call_count[1])

    @skipIfXPU("matmul config space includes maxnreg, unsupported on XPU")
    def test_autotune_continues_when_accuracy_check_crashes(self) -> None:
        # A config can pass the timed run and then crash in the accuracy
        # check. Patches the accuracy job to raise a sticky CUDA error for one
        # config; the worker dies and respawns, and autotune must still pick a
        # best config from the rest instead of aborting.
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

        original = LocalBenchmarkProvider._run_subprocess_accuracy_check_job
        call_count = [0, 0]  # [total, simulated_crashes]

        def maybe_crash(
            self: LocalBenchmarkProvider,
            fn: CompiledConfig,
        ) -> AccuracyCheckResult | None:
            call_count[0] += 1
            # Crash exactly once, early: every induced crash costs a ~5s worker
            # kill/respawn cycle, and one cycle already proves autotune survives
            # an accuracy-check crash and keeps searching.
            if call_count[0] == 2:
                call_count[1] += 1
                if self._benchmark_worker is None:
                    self._benchmark_worker = BenchmarkWorker(device=None)
                # Run a job that raises a sticky error inside the worker, so the
                # worker is killed and a sticky error propagates from the
                # accuracy step, as a real accuracy-check crash would.
                self._benchmark_worker.run(
                    _RaiseRuntimeError("an illegal memory access was encountered"),
                    timeout=float(self.settings.autotune_benchmark_timeout),
                )
            return original(self, fn)

        examples_dir = Path(__file__).parent.parent / "examples"
        matmul = import_path(examples_dir / "matmul.py").matmul

        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = matmul.bind(args)
        bound_kernel.settings.autotune_benchmark_subprocess = True
        bound_kernel.settings.autotune_benchmark_timeout = 60
        bound_kernel.settings.autotune_precompile = None
        # The autotuner reseeds `random` from this setting, so pinning it (not
        # random.seed) is what makes the config sequence reproducible.
        bound_kernel.settings.autotune_random_seed = 123

        random.seed(123)
        with patch.object(
            LocalBenchmarkProvider,
            "_run_subprocess_accuracy_check_job",
            maybe_crash,
        ):
            best = RandomSearch(bound_kernel, args, 8).autotune()

        self.assertIsNotNone(best)
        # Random configs that fail to compile never reach the accuracy check
        # (hardware-dependent even with a pinned seed), so leave slack here.
        self.assertGreaterEqual(call_count[0], 4)
        self.assertEqual(call_count[1], 1)


if __name__ == "__main__":
    unittest.main()
