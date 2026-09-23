from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkWorker
from helion.autotuner.kernel_args import load_trusted_kernel_args
from helion.autotuner.metrics import AutotuneMetrics
from helion.runtime.settings import Settings


@dataclass
class _UpdateCachedArgument:
    path: str
    fail: bool

    def __call__(self) -> float:
        (value,) = load_trusted_kernel_args(self.path)
        assert isinstance(value, torch.Tensor)
        value.add_(1)
        if self.fail:
            raise RuntimeError("candidate failed after updating its argument")
        return float(value.item())


class TestFreshFinalistProcess(unittest.TestCase):
    def run_candidates(self, *, fail_first: bool) -> None:
        # Exercise real spawn workers and the real cached-argument loader.
        # Only kernel serialization/device execution is replaced with a CPU job.
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args_path = Path(directory) / "args.pt"
            torch.save((torch.zeros(1),), args_path)
            provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
            provider.settings = Settings(autotune_benchmark_subprocess=True)
            provider.log = Mock()
            provider._autotune_metrics = AutotuneMetrics()
            provider._precompile_args_path = str(args_path)
            provider._subprocess_wrapper_unloadable = False
            provider._benchmark_worker = None
            workers: list[BenchmarkWorker] = []

            def new_worker(*, device: int | None) -> BenchmarkWorker:
                worker = BenchmarkWorker(device=device)
                workers.append(worker)
                return worker

            def cpu_job(
                fn_spec: bool, args_path: str, **kwargs: object
            ) -> _UpdateCachedArgument:
                return _UpdateCachedArgument(args_path, fn_spec)

            stack.enter_context(
                patch.object(
                    provider, "_subprocess_benchmark_enabled", return_value=True
                )
            )
            stack.enter_context(
                patch.object(
                    provider,
                    "_subprocess_benchmark_uses_wall_clock",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch.object(
                    provider, "_probe_long_cute_flash_kernel", return_value=False
                )
            )
            stack.enter_context(
                patch(
                    "helion.autotuner.benchmark_provider._serialize_compiled_fn",
                    side_effect=lambda fn: fn(),
                )
            )
            stack.enter_context(
                patch("helion.autotuner.benchmark_provider.BenchmarkJob", cpu_job)
            )
            stack.enter_context(
                patch("helion.autotuner.benchmark_provider.BenchmarkWorker", new_worker)
            )
            try:
                if not fail_first:
                    # Ordinary search still reuses its worker and loaded inputs.
                    self.assertEqual(
                        provider.benchmark_isolated(
                            [lambda: False, lambda: False], warmup=1, rep=1
                        ),
                        [1.0, 2.0],
                    )
                    self.assertTrue(workers[0].alive())
                results = provider.benchmark_isolated(
                    [lambda: fail_first, lambda: False],
                    warmup=1,
                    rep=1,
                    fresh_process=True,
                )
                self.assertEqual(results, [None if fail_first else 1.0, 1.0])
                self.assertIsNone(provider._benchmark_worker)
                self.assertTrue(all(not worker.alive() for worker in workers))
                self.assertEqual(len(workers), 2 if fail_first else 3)
                (unchanged,) = torch.load(args_path, weights_only=True)
                self.assertEqual(float(unchanged.item()), 0.0)
            finally:
                for worker in workers:
                    worker.shutdown()

    def test_finalists_do_not_inherit_cached_arguments(self) -> None:
        self.run_candidates(fail_first=False)

    def test_failed_finalist_does_not_leak_cached_arguments(self) -> None:
        self.run_candidates(fail_first=True)


def test_clear_fast_path_caches_handles_single_and_multi_kernel_hosts(monkeypatch):
    from helion.autotuner.benchmarking import clear_jit_fast_path_caches

    def host():
        pass

    legacy = Mock()
    monkeypatch.setitem(host.__globals__, "_helion_host", legacy)
    clear_jit_fast_path_caches(host)
    legacy.clear_fast_path_caches.assert_called_once_with()

    first, second = Mock(), Mock()
    host._helion_cute_kernels = (first, second)
    clear_jit_fast_path_caches(host)
    first.clear_fast_path_caches.assert_called_once_with()
    second.clear_fast_path_caches.assert_called_once_with()
    legacy.clear_fast_path_caches.assert_called_once_with()
