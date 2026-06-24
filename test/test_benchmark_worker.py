"""Tests for the subprocess benchmark path used to hang-protect autotune."""

from __future__ import annotations

import dataclasses
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
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
from helion.autotuner.benchmark_job import BenchmarkJob
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkSubprocessError
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmark_worker import BenchmarkWorker
from helion.autotuner.benchmark_worker import BenchmarkWorkerDied
from helion.autotuner.kernel_args import load_trusted_kernel_args
from helion.autotuner.precompile_future import SerializedCompiledFunction
from helion.autotuner.precompile_future import _run_kernel_in_subprocess_spawn
from helion.autotuner.random_search import RandomSearch
from helion.runtime.config import Config
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from helion.autotuner.benchmarking import PerfStats
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
class _ReturnValue:
    value: object

    def __call__(self) -> object:
        return self.value


class TestBenchmarkWorkerFailureModes(unittest.TestCase):
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

    def test_load_trusted_kernel_args_accepts_python_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "args.pt"
            torch.save((_ReturnValue(3),), path)

            load_trusted_kernel_args.cache_clear()
            loaded = load_trusted_kernel_args(str(path))

        self.assertIsInstance(loaded[0], _ReturnValue)
        self.assertEqual(loaded[0].value, 3)

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
            *,
            perf_stats_out: list[PerfStats] | None = None,
        ) -> float | None:
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                call_count[1] += 1
                self._autotune_metrics.num_compile_failures += 1
                return math.inf
            return original(self, config, fn, perf_stats_out=perf_stats_out)

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

    @skipIfXPU("matmul config space includes maxnreg, unsupported on XPU")
    def test_autotune_continues_when_accuracy_check_crashes(self) -> None:
        # A config can pass the timed run and then crash in the accuracy
        # check. Patches the accuracy job to raise a sticky CUDA error for a
        # fraction of configs; the worker dies and respawns, and autotune must
        # still pick a best config from the rest instead of aborting.
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

        original = LocalBenchmarkProvider._run_subprocess_accuracy_job
        call_count = [0, 0]  # [total, simulated_crashes]

        def maybe_crash(
            self: LocalBenchmarkProvider,
            fn: CompiledConfig,
        ) -> bool | None:
            call_count[0] += 1
            if call_count[0] % 3 == 0:
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

        random.seed(123)
        with patch.object(
            LocalBenchmarkProvider,
            "_run_subprocess_accuracy_job",
            maybe_crash,
        ):
            best = RandomSearch(bound_kernel, args, 20).autotune()

        self.assertIsNotNone(best)
        self.assertGreaterEqual(call_count[0], 6)
        self.assertGreaterEqual(call_count[1], 2)


if __name__ == "__main__":
    unittest.main()
