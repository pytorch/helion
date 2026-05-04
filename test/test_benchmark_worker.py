"""Tests for the subprocess benchmark path used to hang-protect autotune."""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import unittest
from unittest.mock import patch

import torch

from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfXPU
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_job import RebenchmarkJob
from helion.autotuner.benchmark_job import _load_args
from helion.autotuner.benchmark_pool import PoolBenchmarkManager
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmark_worker import BenchmarkWorker
from helion.autotuner.benchmark_worker import BenchmarkWorkerDied
from helion.autotuner.benchmark_worker import BenchmarkWorkerPool
from helion.autotuner.benchmark_worker import WorkerPoolResult
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.precompile_future import SerializedCompiledFunction
from helion.autotuner.random_search import RandomSearch
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
class _Crash:
    def __call__(self) -> object:
        os.kill(os.getpid(), signal.SIGKILL)
        return None


@dataclasses.dataclass
class _ReturnValue:
    value: object

    def __call__(self) -> object:
        return self.value


def _callable_kernel_arg(value: object) -> object:
    return value


class TestBenchmarkWorkerFailureModes(unittest.TestCase):
    def test_timeout_kills_worker(self) -> None:
        # A timed-out job should kill the worker and the next job should respawn it.
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

    def test_sticky_error_kills_worker(self) -> None:
        # Sticky CUDA-style errors should kill the worker before the next job.
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
        # A worker process crash should surface as BenchmarkWorkerDied.
        worker = BenchmarkWorker()
        try:
            with self.assertRaises(BenchmarkWorkerDied):
                worker.run(_Crash(), timeout=30.0)
            self.assertFalse(worker.alive())
        finally:
            worker.shutdown()

    def test_pool_map_reports_worker_and_elapsed(self) -> None:
        # Pool precompile mapping should preserve result order and report timing metadata.
        pool = BenchmarkWorkerPool(2)
        try:
            results = pool.map_jobs(
                [_ReturnValue("a"), _ReturnValue("b")],
                timeout=30.0,
            )
        finally:
            pool.shutdown()

        self.assertEqual([r.result for r in results], ["a", "b"])
        self.assertTrue(all(0 <= r.worker_index < 2 for r in results))
        self.assertTrue(all(r.elapsed >= 0 for r in results))


class TestWorkerPoolPrecompile(unittest.TestCase):
    def test_worker_arg_loading_allows_callable_kernel_args(self) -> None:
        # Worker arg loading must allow trusted callable args such as matmul epilogues.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "args.pt")
            torch.save((_callable_kernel_arg,), path)
            _load_args.cache_clear()
            try:
                loaded = _load_args(path)
            finally:
                _load_args.cache_clear()

        self.assertIs(loaded[0], _callable_kernel_arg)

    def test_subprocess_benchmark_keeps_effort_rebenchmark_threshold(self) -> None:
        # Subprocess benchmarking should not weaken full-effort rebenchmarking.
        settings = Settings(
            autotune_effort="full",
            autotune_benchmark_subprocess=True,
        )

        self.assertEqual(
            settings.get_rebenchmark_threshold(),
            get_effort_profile("full").rebenchmark_threshold,
        )

    def test_explicit_rebenchmark_threshold_overrides_subprocess_default(self) -> None:
        # Explicit rebenchmark thresholds should still override profile defaults.
        settings = Settings(
            autotune_effort="full",
            autotune_benchmark_subprocess=True,
            autotune_rebenchmark_threshold=1.25,
        )

        self.assertEqual(settings.get_rebenchmark_threshold(), 1.25)

    def test_pool_mode_env_value_is_supported(self) -> None:
        # HELION_AUTOTUNE_PRECOMPILE=pool should parse into the pool mode.
        with patch.dict(os.environ, {"HELION_AUTOTUNE_PRECOMPILE": "pool"}):
            self.assertEqual(Settings().autotune_precompile, "pool")

    def test_pool_auto_worker_cap_defaults_to_32(self) -> None:
        # Auto-sized pools should default to a 32-worker cap.
        with patch.dict(os.environ, {"HELION_AUTOTUNE_PRECOMPILE_WORKERS_CAP": ""}):
            self.assertEqual(Settings().autotune_precompile_workers_cap, 32)

    def test_pool_cleanup_shuts_down_worker_pool(self) -> None:
        # Pool workers should not retain CUDA memory after one autotune finishes.
        class FakePoolManager:
            def __init__(self) -> None:
                self.shutdown_called = False

            def shutdown(self) -> None:
                self.shutdown_called = True

        fake_pool_manager = FakePoolManager()
        provider = cast("Any", LocalBenchmarkProvider.__new__(LocalBenchmarkProvider))
        provider.settings = Settings(autotune_precompile="pool")
        provider._benchmark_worker = None
        provider._pool_manager = fake_pool_manager
        provider._serialized_fn_specs = {1: None}
        provider._precompile_tmpdir = None
        provider._precompile_args_path = "args.pt"

        provider.cleanup()

        self.assertTrue(fake_pool_manager.shutdown_called)
        self.assertIsNone(provider._pool_manager)
        self.assertEqual(provider._serialized_fn_specs, {})
        self.assertIsNone(provider._precompile_args_path)

    def test_pool_mode_implies_subprocess_benchmark(self) -> None:
        # Pool mode should benchmark in workers even without the subprocess env flag.
        provider = cast("Any", LocalBenchmarkProvider.__new__(LocalBenchmarkProvider))
        provider.settings = Settings(
            autotune_precompile="pool",
            autotune_benchmark_subprocess=False,
        )
        provider.config_spec = SimpleNamespace(backend=None)
        provider.mutated_arg_indices = []

        self.assertTrue(provider._subprocess_benchmark_enabled())

    def test_pool_mode_reports_disabled_worker_reason(self) -> None:
        # Explicitly disabled pool workers should fail with an actionable reason.
        provider = cast("Any", LocalBenchmarkProvider.__new__(LocalBenchmarkProvider))
        provider.settings = Settings(
            autotune_precompile="pool",
            autotune_precompile_workers=-1,
        )
        provider.config_spec = SimpleNamespace(backend=None)
        provider.mutated_arg_indices = []
        provider._precompile_args_path = "args.pt"
        provider._pool_size = lambda: 1

        reason = provider._pool_mode_unavailable_reason()
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("disabled", reason)

    def test_rebenchmark_uses_worker_pool(self) -> None:
        # Full-effort rebenchmarking should run on the worker that precompiled the config.
        class FakePoolManager:
            def __init__(self) -> None:
                self.worker_index: int | None = None
                self.job: object | None = None
                self.timeout: float | None = None

            def worker_index_for_fn(self, _fn: object) -> int:
                return 3

            def run_on(
                self,
                worker_index: int,
                job: object,
                timeout: float,
            ) -> list[float]:
                self.worker_index = worker_index
                self.job = job
                self.timeout = timeout
                return [1.0, 2.0]

        class FakeLog:
            def warning(self, *_args: object, **_kwargs: object) -> None:
                pass

            def debug(self, *_args: object, **_kwargs: object) -> None:
                pass

        def fake_fn() -> None:
            pass

        pool = FakePoolManager()
        provider = cast("Any", LocalBenchmarkProvider.__new__(LocalBenchmarkProvider))
        provider.settings = Settings(
            autotune_benchmark_subprocess=True,
            autotune_benchmark_timeout=10,
        )
        provider.config_spec = SimpleNamespace(backend=None)
        provider.mutated_arg_indices = []
        provider._precompile_args_path = "args.pt"
        provider._pool_manager = pool
        provider._benchmark_worker = None
        provider._autotune_metrics = SimpleNamespace(num_compile_failures=0)
        provider.log = FakeLog()
        provider._serialize_fn_for_worker = lambda _fn: SerializedCompiledFunction(
            function_name="fake_fn",
            source_code="def fake_fn(): pass",
            filename=None,
            module_name=None,
        )

        result = provider.rebenchmark([fake_fn, fake_fn], repeat=7, desc="verify")

        self.assertEqual(result, [1.0, 2.0])
        self.assertEqual(pool.worker_index, 3)
        self.assertIsInstance(pool.job, RebenchmarkJob)
        assert isinstance(pool.job, RebenchmarkJob)
        self.assertEqual(pool.job.repeat, 7)
        self.assertEqual(len(pool.job.fn_specs), 2)
        self.assertEqual(pool.timeout, 20.0)

    def test_population_rebenchmark_uses_provider_timings(self) -> None:
        # BaseSearch should use provider rebenchmark timings when available.
        class FakeProvider:
            def __init__(self) -> None:
                self.mutated_arg_indices: list[int] = []
                self.fns: list[object] | None = None
                self.repeat: int | None = None

            def rebenchmark(
                self,
                fns: list[object],
                *,
                repeat: int,
                desc: str,
            ) -> list[float]:
                self.fns = fns
                self.repeat = repeat
                return [0.70, 0.80]

        def fn_a() -> None:
            pass

        def fn_b() -> None:
            pass

        provider = FakeProvider()
        search = cast("Any", PopulationBasedSearch.__new__(PopulationBasedSearch))
        search.settings = Settings(autotune_precompile="pool")
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.best_perf_so_far = 1.0
        search.benchmark_provider = provider
        members = [
            PopulationMember(fn=fn_a, perfs=[1.00], flat_values=[], config=Config()),
            PopulationMember(fn=fn_b, perfs=[0.90], flat_values=[], config=Config()),
        ]

        search.rebenchmark(members, desc="verify")

        self.assertEqual(provider.fns, [fn_a, fn_b])
        self.assertEqual(provider.repeat, 200)
        self.assertEqual(members[0].perfs[-1], 0.70)
        self.assertEqual(members[1].perfs[-1], 0.80)

    def test_false_precompile_result_is_failure(self) -> None:
        # A worker precompile returning False should count as a real compile failure.
        class FakePool:
            def start_all(self, limit: int | None = None) -> None:
                self.limit = limit

            def map_jobs(
                self,
                jobs: list[object],
                timeout: float,
            ) -> list[WorkerPoolResult]:
                return [
                    WorkerPoolResult(worker_index=0, elapsed=0.25, result=False)
                    for _ in jobs
                ]

        class FakeLog:
            def debug(self, *_args: object, **_kwargs: object) -> None:
                pass

        def fake_fn() -> None:
            pass

        metrics = SimpleNamespace(num_compile_failures=0)
        manager = cast("Any", PoolBenchmarkManager.__new__(PoolBenchmarkManager))
        manager._pool = FakePool()
        manager._log = FakeLog()
        manager._autotune_metrics = metrics
        manager._precompile_worker_by_fn = {}

        def serialize_fn(_fn: object) -> SerializedCompiledFunction:
            return SerializedCompiledFunction(
                function_name="fake_fn",
                source_code="def fake_fn(): pass",
                filename=None,
                module_name=None,
            )

        result = manager.precompile(
            [Config()],
            [fake_fn],
            args_path="args.pt",
            timeout=1,
            desc=None,
            serialize_fn=serialize_fn,
        )

        self.assertEqual(result.is_workings, [False])
        self.assertEqual(result.statuses, ["error"])
        self.assertEqual(result.compile_times, [0.25])
        self.assertEqual(metrics.num_compile_failures, 1)
        self.assertEqual(manager._precompile_worker_by_fn, {})


# Subprocess benchmarking depends on Backend.supports_precompile(); only the
# Triton backend supports it (Pallas/CuTe return False).
@onlyBackends(["triton"])
class TestSubprocessBenchmarkIntegration(RefEagerTestDisabled, unittest.TestCase):
    @skipIfXPU("matmul config space includes maxnreg, unsupported on XPU")
    def test_autotune_with_subprocess_bench(self) -> None:
        # Subprocess benchmarking should support a small end-to-end autotune run.
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

        random.seed(123)
        RandomSearch(bound_kernel, args, 20).autotune()

    @skipIfXPU("matmul config space includes maxnreg, unsupported on XPU")
    def test_autotune_continues_when_subprocess_reports_inf(self) -> None:
        # A subset of failed subprocess benchmarks should not abort the search.
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
                self._autotune_metrics.num_compile_failures += 1
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

        random.seed(123)
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function_subprocess",
            maybe_fail,
        ):
            RandomSearch(bound_kernel, args, 20).autotune()

        self.assertGreaterEqual(call_count[0], 6)
        self.assertGreaterEqual(call_count[1], 2)


if __name__ == "__main__":
    unittest.main()
