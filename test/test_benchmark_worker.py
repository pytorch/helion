"""Tests for the subprocess benchmark path used to hang-protect autotune."""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
import random
import signal
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
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmark_worker import BenchmarkWorker
from helion.autotuner.benchmark_worker import BenchmarkWorkerDied
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


class TestBenchmarkWorkerFailureModes(unittest.TestCase):
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


class TestSuspiciousRebenchmark(unittest.TestCase):
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
