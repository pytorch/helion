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
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

import torch

from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfXPU
from helion.autotuner.benchmark_job import _load_args
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmark_worker import BenchmarkWorker
from helion.autotuner.benchmark_worker import BenchmarkWorkerDied
from helion.autotuner.benchmark_worker import BenchmarkWorkerPool
from helion.autotuner.random_search import RandomSearch

if TYPE_CHECKING:
    from helion.runtime.config import Config
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

    def test_pool_run_jobs_reports_worker_and_elapsed(self) -> None:
        # Pool job execution should preserve result order and report timing metadata.
        pool = BenchmarkWorkerPool(2)
        try:
            results = pool.run_jobs(
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

        random.seed(123)
        RandomSearch(bound_kernel, args, 20).autotune()

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
