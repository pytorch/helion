from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Callable
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from helion._testing import skipIfRefEager
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkWorker
from helion.autotuner.kernel_args import load_trusted_kernel_args
from helion.autotuner.metrics import AutotuneMetrics
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only


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


@skipIfRefEager(
    "Spawns real fresh-process benchmark-worker subprocesses -- compile-mode "
    "machinery that HELION_INTERPRET=1 bypasses, and it flakes (crashing the "
    "xdist worker with no traceback) on the single-GPU A10G ref-eager runner."
)
class TestFreshFinalistProcess(unittest.TestCase):
    @pytest.mark.slow
    def test_finalists_do_not_inherit_cached_arguments(self) -> None:
        # Exercise real spawn workers and the real cached-argument loader.
        # Only kernel serialization/device execution is replaced with a CPU job.
        # One ordinary search first (its worker and loaded inputs are reused),
        # then two fresh-process finalists: the first fails after updating its
        # loaded argument, and the second must still see the saved input, which
        # also proves a failed finalist leaks nothing (each worker is a spawn).
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args_path = Path(directory) / "args.pt"
            torch.save((torch.zeros(1),), args_path)
            provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
            provider.settings = Settings(autotune_benchmark_subprocess=True)
            provider.log = Mock()
            provider._autotune_metrics = AutotuneMetrics()
            provider._precompile_args_path = str(args_path)
            provider._subprocess_wrapper_unloadable = False
            provider.mutated_arg_indices = []
            provider._benchmark_worker = None
            workers: list[BenchmarkWorker] = []

            def new_worker(
                *, device: int | None, log_warning: Callable[[str], object]
            ) -> BenchmarkWorker:
                worker = BenchmarkWorker(device=device, log_warning=log_warning)
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
                patch.object(provider, "_probe_long_kernel", return_value=False)
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
                # Ordinary search still reuses its worker and loaded inputs.
                self.assertEqual(
                    provider.benchmark_isolated(
                        [lambda: False, lambda: False], warmup=1, rep=1
                    ),
                    [1.0, 2.0],
                )
                self.assertTrue(workers[0].alive())
                results = provider.benchmark_isolated(
                    [lambda: True, lambda: False],
                    warmup=1,
                    rep=1,
                    fresh_process=True,
                )
                self.assertEqual(results, [None, 1.0])
                self.assertIsNone(provider._benchmark_worker)
                self.assertTrue(all(not worker.alive() for worker in workers))
                self.assertEqual(len(workers), 3)
                (unchanged,) = torch.load(args_path, weights_only=True)
                self.assertEqual(float(unchanged.item()), 0.0)
            finally:
                for worker in workers:
                    worker.shutdown()


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
