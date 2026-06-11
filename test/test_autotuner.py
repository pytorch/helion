from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import csv
import logging
import math
import multiprocessing as mp
import operator
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
from types import SimpleNamespace
from typing import Callable
from typing import ClassVar
from typing import Sequence
import unittest
from unittest import skip
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion import _compat
from helion import exc
from helion._compiler.tile_dispatch import BlockIDStrategyMapping
from helion._compiler.tile_dispatch import TileStrategyDispatch
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import assert_close_with_mismatch_tolerance
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner import DESurrogateHybrid
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner import LFBOPatternSearch
from helion.autotuner import LFBOTreeSearch
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import NumThreadsFragment
from helion.autotuner.config_fragment import PermutationFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import StrictLocalAutotuneCache
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.random_search import RandomSearch
import helion.language as hl
from helion.language import loops
from helion.runtime.settings import Settings
from helion.runtime.settings import _get_backend

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"


def _get_examples_matmul():
    """Lazy accessor to avoid CUDA init during pytest-xdist collection."""
    return import_path(examples_dir / "matmul.py").matmul


@contextmanager
def without_env_var(name: str):
    sentinel = object()
    previous = os.environ.pop(name, sentinel)
    try:
        yield
    finally:
        if previous is not sentinel:
            os.environ[name] = previous


class RecordingRandomSearch(RandomSearch):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.samples: list[float] = []

    def _autotune(self):
        self.samples.append(random.random())
        return super()._autotune()


@onlyBackends(["triton"])
class TestAutotuneIgnoreErrors(TestCase):
    def _make_search(
        self, settings: Settings, *, args: tuple[object, ...] = ()
    ) -> BaseSearch:
        search = BaseSearch.__new__(BaseSearch)
        search.settings = settings
        search.kernel = SimpleNamespace(
            format_kernel_decorator=lambda config, s: "decorator",
            to_triton_code=lambda config: "code",
            maybe_log_repro=lambda log_func, args, config=None: None,
            supports_subprocess_benchmark=lambda: False,
        )
        search.args = args
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            default_config=lambda: helion.Config(block_sizes=[1])
        )
        search._benchmark_provider_cls = LocalBenchmarkProvider
        search.best_perf_so_far = float("inf")
        search._prepared = False
        with patch.object(
            LocalBenchmarkProvider,
            "_compute_baseline",
            return_value=(None, [], None),
        ):
            search._prepare()
        return search

    def test_settings_flag_from_env(self):
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_IGNORE_ERRORS": "1"}, clear=False
        ):
            settings = Settings()
        self.assertTrue(settings.autotune_ignore_errors)

    def test_benchmark_raise_includes_hint(self):
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("boom")

        with patch("torch.accelerator.synchronize", autospec=True) as sync:
            sync.return_value = None
            with pytest.raises(exc.TritonError) as err:
                search.benchmark_provider._benchmark_function("cfg", bad_fn)

        assert "HELION_AUTOTUNE_IGNORE_ERRORS" in str(err.value)

    def test_llvm_translation_failure_skips_config(self):
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("failed to translate module to LLVM IR")

        with patch("torch.accelerator.synchronize", autospec=True) as sync:
            sync.return_value = None
            result = search.benchmark_provider._benchmark_function("cfg", bad_fn)

        self.assertEqual(result, float("inf"))
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)

    def test_cuda_oom_skips_config(self):
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        with patch("torch.accelerator.synchronize", autospec=True) as sync:
            sync.return_value = None
            result = search.benchmark_provider._benchmark_function("cfg", bad_fn)

        self.assertEqual(result, float("inf"))
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)

    def test_ignore_errors_skips_logging_and_raise(self):
        settings = Settings(
            autotune_ignore_errors=True,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("boom")

        with patch("torch.accelerator.synchronize", autospec=True) as sync:
            sync.return_value = None
            with patch.object(search.log, "warning") as warn:
                result = search.benchmark_provider._benchmark_function("cfg", bad_fn)

        self.assertEqual(result, float("inf"))
        warn.assert_not_called()

    def test_clear_jit_fast_path_caches(self):
        settings = Settings(
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        calls = []

        class FakeJITFunction:
            def clear_fast_path_caches(self) -> None:
                calls.append("cleared")

        def generated_kernel() -> None:
            return None

        globals_key = f"_helion_{generated_kernel.__name__}"
        generated_kernel.__globals__[globals_key] = FakeJITFunction()
        try:
            search.benchmark_provider._clear_jit_fast_path_caches(generated_kernel)
        finally:
            del generated_kernel.__globals__[globals_key]

        self.assertEqual(calls, ["cleared"])

    def test_clear_jit_fast_path_caches_does_not_clear_device_caches(self):
        settings = Settings(
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        class FakeJITFunction:
            def __init__(self) -> None:
                self.device_caches = {"compiled": object()}

            def clear_fast_path_caches(self) -> None:
                return None

        def generated_kernel() -> None:
            return None

        jit_fn = FakeJITFunction()
        device_caches = jit_fn.device_caches
        globals_key = f"_helion_{generated_kernel.__name__}"
        generated_kernel.__globals__[globals_key] = jit_fn
        try:
            search.benchmark_provider._clear_jit_fast_path_caches(generated_kernel)
        finally:
            del generated_kernel.__globals__[globals_key]

        self.assertIs(jit_fn.device_caches, device_caches)
        self.assertEqual(list(jit_fn.device_caches), ["compiled"])

    def test_clear_jit_fast_path_caches_ignores_cleanup_errors(self):
        settings = Settings(
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        class FakeJITFunction:
            def clear_fast_path_caches(self) -> None:
                raise RuntimeError("cleanup failed")

        def generated_kernel() -> None:
            return None

        globals_key = f"_helion_{generated_kernel.__name__}"
        generated_kernel.__globals__[globals_key] = FakeJITFunction()
        try:
            search.benchmark_provider._clear_jit_fast_path_caches(generated_kernel)
        finally:
            del generated_kernel.__globals__[globals_key]

    def test_benchmark_function_clears_jit_fast_path_caches_on_success(self):
        settings = Settings(
            autotune_accuracy_check=False,
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings, args=("arg0",))

        def compiled_fn(*_args):
            return None

        bench_fn = Mock(return_value=None)
        search.kernel.env = SimpleNamespace(process_group_name=None)
        search.kernel.bench_compile_config = Mock(return_value=bench_fn)

        with (
            patch("torch.accelerator.synchronize", autospec=True) as sync,
            patch(
                "helion.autotuner.benchmark_provider.do_bench",
                return_value=1.25,
            ),
            patch.object(
                search.benchmark_provider, "_clear_jit_fast_path_caches"
            ) as clear,
        ):
            sync.return_value = None
            result = search.benchmark_provider._benchmark_function("cfg", compiled_fn)

        self.assertEqual(result, 1.25)
        clear.assert_called_once_with(compiled_fn)

    def test_benchmark_function_clears_jit_fast_path_caches_on_error(self):
        settings = Settings(
            autotune_ignore_errors=True,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("boom")

        with (
            patch("torch.accelerator.synchronize", autospec=True) as sync,
            patch.object(
                search.benchmark_provider, "_clear_jit_fast_path_caches"
            ) as clear,
        ):
            sync.return_value = None
            result = search.benchmark_provider._benchmark_function("cfg", bad_fn)

        self.assertEqual(result, float("inf"))
        clear.assert_called_once_with(bad_fn)

    def test_traceback_cleared_str(self):
        """Test that str(e) still has meaningful content after e.__traceback__ = None."""
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("test error with meaningful message")

        with (
            patch("torch.accelerator.synchronize", autospec=True) as sync,
            patch(
                "helion.autotuner.benchmark_provider.classify_triton_exception",
                return_value="raise",
            ),
        ):
            sync.return_value = None
            with pytest.raises(exc.TritonError) as err:
                search.benchmark_provider._benchmark_function("cfg", bad_fn)

        # Verify the traceback was cleared
        assert err.value.__cause__.__traceback__ is None
        # Verify the error message is still accessible and meaningful
        assert "RuntimeError: test error with meaningful message" in str(err.value)

    def test_traceback_cleared_raise_from(self):
        """Test that 'raise ... from e' still has meaningful stack after e.__traceback__ = None."""
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        original_exception = RuntimeError("original error in except block")

        def bad_fn(*_args):
            raise original_exception

        with (
            patch("torch.accelerator.synchronize", autospec=True) as sync,
            patch(
                "helion.autotuner.benchmark_provider.classify_triton_exception",
                return_value="raise",
            ),
        ):
            sync.return_value = None
            with pytest.raises(exc.TritonError) as err:
                search.benchmark_provider._benchmark_function("cfg", bad_fn)

        # Verify the traceback was cleared
        assert err.value.__cause__.__traceback__ is None
        # Verify the exception chain is preserved even after __traceback__ = None
        assert err.value.__cause__ is original_exception
        assert str(original_exception) == "original error in except block"
        # Verify we can still get the error type and message
        assert type(err.value.__cause__).__name__ == "RuntimeError"

    def test_benchmark_results_aligned_when_compile_fails(self):
        """benchmark_batch must return one result per input config even when some
        fail to compile."""
        settings = Settings(
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        call_count = 0

        def fail_second(config, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated compile failure")
            return lambda *a, **kw: None

        search.kernel.compile_config = None
        search.kernel.env = SimpleNamespace(process_group_name=None)
        configs = ["cfg_a", "cfg_b", "cfg_c"]
        with (
            patch.object(search.kernel, "compile_config", side_effect=fail_second),
            patch.object(
                search.benchmark_provider,
                "_benchmark_function",
                return_value=1.0,
            ),
        ):
            results = search.benchmark_batch(configs, desc="test")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].perf, 1.0)
        self.assertEqual(results[1].perf, float("inf"))
        self.assertEqual(results[1].status, "error")
        self.assertEqual(results[2].perf, 1.0)

    def test_autotune_log_sink_writes_csv_and_log(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune_run"
        settings = Settings(
            autotune_log=str(base_path),
            autotune_log_level=logging.CRITICAL,
        )
        logger = AutotuningLogger(settings)
        with logger.autotune_logging():
            entry = AutotuneLogEntry(
                generation=5,
                status="ok",
                perf_ms=1.234,
                compile_time=0.5,
                config=helion.Config(foo=1, bar=[2, 3]),
            )
            logger.record_autotune_entry(entry)
            logger("finalized entry", level=logging.CRITICAL)

        csv_path = base_path.with_suffix(".csv")
        log_path = base_path.with_suffix(".log")
        self.assertTrue(csv_path.exists())
        self.assertTrue(log_path.exists())
        rows = list(csv.reader(csv_path.read_text().splitlines()))
        header = rows[0]
        self.assertEqual(
            header,
            [
                "run_id",
                "timestamp_s",
                "config_index",
                "generation",
                "status",
                "perf_ms",
                "compile_time_s",
                "config",
            ],
        )
        # Lean schema: no kernel_id/sample_id/decorator columns.
        for absent in ("kernel_id", "sample_id", "decorator"):
            self.assertNotIn(absent, header)
        # No metadata supplied here, so the run_id join key is empty.
        self.assertEqual(rows[1][header.index("run_id")], "")
        self.assertEqual(rows[1][header.index("config_index")], "1")
        self.assertEqual(rows[1][header.index("generation")], "5")
        self.assertEqual(rows[1][header.index("status")], "ok")
        self.assertEqual(rows[1][header.index("perf_ms")], "1.234000")
        log_text = log_path.read_text()
        self.assertIn("finalized entry", log_text)

    def test_differential_evolution_immediate_iter_uses_batch_helper(self):
        search = DifferentialEvolutionSearch.__new__(DifferentialEvolutionSearch)
        search.immediate_update = True
        search.population = [object(), object(), object()]

        calls: list[list[int]] = []

        def batch(indices: Sequence[int]) -> list[PopulationMember]:
            calls.append(list(indices))
            members: list[PopulationMember] = []
            for idx in indices:
                members.append(
                    PopulationMember(
                        lambda *args, **kwargs: None,
                        [float(idx)],
                        [],
                        SimpleNamespace(config={"idx": idx}),
                        status="ok",
                    )
                )
            return members

        search._benchmark_mutation_batch = batch  # type: ignore[assignment]
        candidates = list(search.iter_candidates())
        self.assertEqual(calls, [[0], [1], [2]])
        self.assertEqual([idx for idx, _ in candidates], [0, 1, 2])

    def test_differential_evolution_parallel_iter_uses_batch_helper(self):
        search = DifferentialEvolutionSearch.__new__(DifferentialEvolutionSearch)
        search.immediate_update = False
        search.population = [object(), object()]

        def batch(indices: Sequence[int]) -> list[PopulationMember]:
            members: list[PopulationMember] = []
            for idx in indices:
                members.append(
                    PopulationMember(
                        lambda *args, **kwargs: None,
                        [float(idx)],
                        [],
                        SimpleNamespace(config={"idx": idx}),
                        status="ok",
                    )
                )
            return members

        calls: list[list[int]] = []

        def recording_batch(indices: Sequence[int]) -> list[PopulationMember]:
            calls.append(list(indices))
            return batch(indices)

        search._benchmark_mutation_batch = recording_batch  # type: ignore[assignment]
        candidates = list(search.iter_candidates())
        self.assertEqual(calls, [[0, 1]])
        self.assertEqual([idx for idx, _ in candidates], [0, 1])

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork start method is unavailable on this platform",
    )
    def test_fork_precompile_avoids_cuda_reinit(self):
        settings = Settings(
            autotune_precompile="fork",
            autotune_log_level=logging.CRITICAL,
            autotune_compile_timeout=5,
        )
        search = self._make_search(settings, args=("arg0",))

        parent_pid = os.getpid()
        lazy_calls: list[int] = []

        def fake_lazy_init() -> None:
            lazy_calls.append(os.getpid())

        def fake_make_precompiler(_kernel_obj, _config, _bound_kernel):
            def binder(*_args: object, **_kwargs: object):
                def run() -> None:
                    return None

                return run

            return binder

        def fake_compiled_fn(
            *fn_args: object, _launcher: Callable[..., object]
        ) -> None:
            torch.cuda._lazy_init()
            _launcher("fake_kernel", (1,), *fn_args)

        with (
            patch(
                "helion.autotuner.precompile_future.make_precompiler",
                side_effect=fake_make_precompiler,
            ),
            patch("torch.cuda._lazy_init", side_effect=fake_lazy_init),
        ):
            future = search.benchmark_provider._create_precompile_future(
                "cfg", fake_compiled_fn
            )
            self.assertTrue(future())

        self.assertEqual(set(lazy_calls), {parent_pid})

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork start method is unavailable on this platform",
    )
    def test_fork_precompile_expected_errors_skip_config(self):
        from torch._inductor.runtime.triton_compat import OutOfResources

        expected_errors = [
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            OutOfResources(128, 64, "shared memory"),
            RuntimeError("out of resource: shared memory"),
            RuntimeError("too many resources requested for launch"),
            RuntimeError("CUDA error: out of memory"),
            RuntimeError("[CUDA]: out of memory"),
            RuntimeError("failed to translate module to LLVM IR"),
        ]
        for err in expected_errors:
            with self.subTest(error=type(err).__name__, msg=str(err)):
                settings = Settings(
                    autotune_precompile="fork",
                    autotune_ignore_errors=False,
                    autotune_log_level=logging.CRITICAL,
                )
                search = self._make_search(settings, args=("arg0",))

                def fake_compiled_fn(
                    *fn_args: object, _launcher: Callable[..., object]
                ) -> None:
                    _launcher("fake_kernel", (1,), *fn_args)

                with patch(
                    "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
                    side_effect=err,
                ):
                    future = search.benchmark_provider._create_precompile_future(
                        "cfg", fake_compiled_fn
                    )

                self.assertFalse(future.ok)

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork start method is unavailable on this platform",
    )
    def test_fork_precompile_illegal_memory_access_raises(self):
        settings = Settings(
            autotune_precompile="fork",
            autotune_ignore_errors=True,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings, args=("arg0",))

        def fake_compiled_fn(
            *fn_args: object, _launcher: Callable[..., object]
        ) -> None:
            _launcher("fake_kernel", (1,), *fn_args)

        with (
            patch(
                "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
                side_effect=RuntimeError("an illegal memory access was encountered"),
            ),
            pytest.raises(RuntimeError, match="illegal memory access"),
        ):
            search.benchmark_provider._create_precompile_future("cfg", fake_compiled_fn)

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork start method is unavailable on this platform",
    )
    def test_fork_precompile_unexpected_error_skipped_with_ignore_errors(self):
        settings = Settings(
            autotune_precompile="fork",
            autotune_ignore_errors=True,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings, args=("arg0",))

        def fake_compiled_fn(
            *fn_args: object, _launcher: Callable[..., object]
        ) -> None:
            _launcher("fake_kernel", (1,), *fn_args)

        with patch(
            "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
            side_effect=RuntimeError("something unexpected"),
        ):
            future = search.benchmark_provider._create_precompile_future(
                "cfg", fake_compiled_fn
            )

        self.assertFalse(future.ok)

    def _run_autotuner_and_check_logging(
        self, search_factory: Callable[[object, tuple[object, ...]], BaseSearch]
    ) -> None:
        """Helper to verify started/completion logging for any autotuner."""
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune_run"

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNE_LOG": str(base_path),
                "HELION_AUTOTUNE_LOG_LEVEL": "0",
            },
        ):

            @helion.kernel()
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            args = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_kernel = add.bind(args)
            random.seed(123)
            search = search_factory(bound_kernel, args)
            search.autotune()

        csv_path = base_path.with_suffix(".csv")
        self.assertTrue(csv_path.exists())
        rows = list(csv.reader(csv_path.read_text().splitlines()))
        status_idx = rows[0].index("status")  # look up by name; column order may change
        statuses = [row[status_idx] for row in rows[1:]]  # skip header
        started_count = sum(1 for s in statuses if s == "started")
        completed_count = sum(1 for s in statuses if s in ("ok", "error", "timeout"))
        self.assertGreater(started_count, 0, "Should log started entries")
        self.assertEqual(
            started_count, completed_count, "Each started should have completion"
        )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_autotune_log_started_completed(self):
        """Test started/completion logging with all autotuning algorithms."""
        configs = [
            helion.Config(block_sizes=[32], num_warps=4),
            helion.Config(block_sizes=[64], num_warps=8),
        ]
        search_factories = [
            (
                "FiniteSearch",
                lambda kernel, args: FiniteSearch(kernel, args, configs=configs),
            ),
            ("RandomSearch", lambda kernel, args: RandomSearch(kernel, args, count=3)),
            (
                "PatternSearch",
                lambda kernel, args: PatternSearch(
                    kernel, args, initial_population=3, max_generations=1, copies=1
                ),
            ),
            (
                "DifferentialEvolutionSearch",
                lambda kernel, args: DifferentialEvolutionSearch(
                    kernel, args, population_size=3, max_generations=1
                ),
            ),
        ]
        for name, factory in search_factories:
            with self.subTest(algorithm=name):
                self._run_autotuner_and_check_logging(factory)

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_autotune_skips_restricted_search(self):
        """A run restricted to user-pinned configs (``configs=[...]`` without
        ``force_autotune``) is excluded from data collection: neither the
        ``.csv`` nor the ``.meta.jsonl`` is written (PRD FR1)."""
        configs = [
            helion.Config(block_sizes=[32], num_warps=4),
            helion.Config(block_sizes=[64], num_warps=8),
        ]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune_run"

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNE_LOG": str(base_path),
                "HELION_AUTOTUNE_LOG_LEVEL": "0",
            },
        ):

            @helion.kernel(configs=configs)
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            args = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_kernel = add.bind(args)
            random.seed(123)
            search = FiniteSearch(bound_kernel, args, configs=configs)
            search.autotune()

        # Restricted search -> no telemetry files at all.
        self.assertFalse(base_path.with_suffix(".csv").exists())
        self.assertFalse(base_path.with_suffix(".meta.jsonl").exists())


@onlyBackends(["triton"])
class TestAutotuner(RefEagerTestDisabled, TestCase):
    def setUp(self):
        super().setUp()
        random.seed(112)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    def test_config_fragment0(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        spec = _get_examples_matmul().bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch(
        "helion.autotuner.config_generation.warps_to_threads",
        lambda num_warps: num_warps * 32,
    )
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @patch("torch.version.hip", None)
    @patch("torch.version.xpu", None)
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    def test_config_fragment1(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch(
        "helion.autotuner.config_generation.warps_to_threads",
        lambda num_warps: num_warps * 32,
    )
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @patch("torch.version.hip", None)
    @patch("torch.version.xpu", None)
    @skipIfTileIR("tileir backend will ignore `warp specialization` hint")
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    def test_config_warp_specialize_unroll(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        overrides = {"range_unroll_factors": [4], "range_warp_specializes": ([True])}
        # We expect all the unroll factors to be set to 0
        configs = ConfigGeneration(spec, overrides=overrides).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    def test_config_generation_overrides(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        overrides = {"indexing": "tensor_descriptor"}
        gen = ConfigGeneration(spec, overrides=overrides)

        flat = gen.default_flat()
        config = gen.unflatten([*flat])
        self.assertEqual(config["indexing"], "tensor_descriptor")
        configs = [gen.unflatten(gen.random_flat()) for _ in range(3)]
        self.assertEqual({cfg["indexing"] for cfg in configs}, {"tensor_descriptor"})
        indexing_choices = spec.valid_indexing_types()
        indexing_index = next(
            i
            for i, fragment in enumerate(gen.flat_spec)
            if isinstance(fragment, ListOf)
            and isinstance(fragment.inner, EnumFragment)
            and fragment.inner.choices == tuple(indexing_choices)
        )
        mutated = gen.random_flat()
        mutated[indexing_index] = "pointer"
        new_config = gen.unflatten(mutated)
        self.assertEqual(new_config["indexing"], "tensor_descriptor")
        self.assertEqual(mutated[indexing_index], "pointer")

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_save_load_config(self):
        config = helion.Config(
            block_sizes=[64, 64, 32],
            loop_orders=[[1, 0]],
            num_warps=2,
            num_stages=1,
            indexing="block_ptr",
            l2_grouping=32,
        )
        with tempfile.NamedTemporaryFile() as f:
            config.save(f.name)
            loaded_config = helion.Config.load(f.name)
            self.assertEqual(config, loaded_config)
        self.assertExpectedJournal(config.to_json())

    def test_config_pickle_roundtrip(self):
        config = helion.Config(
            block_sizes=[64, 64, 32],
            loop_orders=[[1, 0]],
            num_warps=4,
            num_stages=2,
            indexing="tensor_descriptor",
            extra_metadata={"nested": [1, 2, 3]},
        )
        restored = pickle.loads(pickle.dumps(config))
        self.assertIsInstance(restored, helion.Config)
        self.assertEqual(config, restored)
        self.assertIsNot(config, restored)
        self.assertIsNot(config.config, restored.config)

    def test_run_fixed_config(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[1024, 1, 1],
                flatten_loops=[True],
                loop_orders=[[0, 2, 1]],
                num_warps=8,
            )
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))

    def test_finite_search_all_configs_fail_raises(self):
        """Test that when all configs fail, the error is re-raised.

        Without this, compile failures would be silently swallowed and the
        autotuner would return no results. We must surface the error so
        users know their configs are incompatible with the input shape.
        """

        @helion.kernel(
            configs=[
                helion.Config(block_sizes=[64]),
                helion.Config(block_sizes=[128]),
            ],
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        with self.assertRaises(exc.InvalidConfig):
            add(*args)

    def test_run_finite_search(self):
        @helion.kernel(
            configs=[
                helion.Config(
                    block_sizes=[1024, 1, 1],
                    flatten_loops=[True],
                    loop_orders=[[0, 2, 1]],
                    num_warps=8,
                ),
                helion.Config(
                    block_sizes=[1024, 1, 1], flatten_loops=[True], num_warps=8
                ),
                helion.Config(block_sizes=[1, 64, 64], num_warps=8),
                helion.Config(block_sizes=[1, 1, 512], num_warps=8),
            ],
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))
        torch.testing.assert_close(add(*args), sum(args))

    def test_finite_search_skips_bad_configs(self):
        """Test that configs that fail to compile are skipped.

        Uses a config with wrong number of block_sizes (1 instead of 3)
        placed between two good configs, to verify the skip logic doesn't
        disrupt processing of subsequent valid configs.
        """

        @helion.kernel(
            configs=[
                # Good config
                helion.Config(block_sizes=[1, 64, 64], num_warps=8),
                # Bad config: insufficient block_sizes for a 3D kernel
                helion.Config(block_sizes=[64]),
                # Good config after bad one — must still work
                helion.Config(block_sizes=[1, 1, 512], num_warps=8),
            ],
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        # Bad config (block_sizes=[64]) has wrong number of block_sizes for
        # 3D input and should fail to compile. The surrounding good configs
        # should allow autotuning to succeed.
        torch.testing.assert_close(add(*args), sum(args))

    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_random_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = _get_examples_matmul().bind(args)
        bound_kernel.settings.autotune_precompile = None
        random.seed(123)
        best = RandomSearch(bound_kernel, args, 20).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    @skip("too slow")
    def test_differential_evolution_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = _get_examples_matmul().bind(args)
        random.seed(123)
        best = DifferentialEvolutionSearch(
            bound_kernel, args, 5, max_generations=3
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    @skip("too slow")
    def test_de_surrogate_hybrid(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = _get_examples_matmul().bind(args)
        random.seed(123)
        best = DESurrogateHybrid(
            bound_kernel, args, population_size=5, max_generations=3
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    def test_differential_evolution_early_stopping_parameters(self):
        """Test that early stopping is disabled by default and can be enabled."""
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)

        # Test 1: Default parameters (early stopping disabled)
        search = DifferentialEvolutionSearch(
            bound_kernel, args, population_size=5, max_generations=3
        )
        self.assertIsNone(search.min_improvement_delta)
        self.assertIsNone(search.patience)

        # Test 2: Enable early stopping with custom parameters
        search_custom = DifferentialEvolutionSearch(
            bound_kernel,
            args,
            population_size=5,
            max_generations=3,
            min_improvement_delta=0.01,
            patience=5,
        )
        self.assertEqual(search_custom.min_improvement_delta, 0.01)
        self.assertEqual(search_custom.patience, 5)

    def test_de_surrogate_early_stopping_parameters(self):
        """Test that DE-Surrogate early stopping parameters are optional with correct defaults."""
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)

        # Test 1: Default parameters (optional)
        search = DESurrogateHybrid(
            bound_kernel, args, population_size=5, max_generations=3
        )
        self.assertEqual(search.min_improvement_delta, 0.001)
        self.assertEqual(search.patience, 3)

        # Test 2: Custom parameters
        search_custom = DESurrogateHybrid(
            bound_kernel,
            args,
            population_size=5,
            max_generations=3,
            min_improvement_delta=0.01,
            patience=5,
        )
        self.assertEqual(search_custom.min_improvement_delta, 0.01)
        self.assertEqual(search_custom.patience, 5)

    @skip("too slow")
    def test_pattern_search(self):
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)
        random.seed(123)
        best = PatternSearch(
            bound_kernel, args, initial_population=10, max_generations=2, copies=1
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), sum(args), rtol=1e-2, atol=1e-1)

    def test_pattern_search_neighbor_values(self):
        self.assertEqual(
            PowerOfTwoFragment(1, 128, 32).pattern_neighbors(32),
            [16, 64],
        )
        self.assertEqual(
            sorted(IntegerFragment(1, 5, 3).pattern_neighbors(3)),
            [2, 4],
        )
        self.assertEqual(BooleanFragment().pattern_neighbors(True), [False])
        self.assertEqual(
            sorted(EnumFragment(("a", "b", "c")).pattern_neighbors("b")),
            ["a", "c"],
        )

    def test_pattern_search_neighbor_values_radius(self):
        # PowerOfTwoFragment: radius=2 should return 2 steps in exponent space
        self.assertEqual(
            PowerOfTwoFragment(1, 128, 32).pattern_neighbors(32, radius=2),
            [8, 16, 64, 128],
        )
        # PowerOfTwoFragment: radius=2 clamped at lower boundary
        self.assertEqual(
            PowerOfTwoFragment(16, 128, 16).pattern_neighbors(16, radius=2),
            [32, 64],
        )
        # PowerOfTwoFragment: radius=2 clamped at upper boundary
        self.assertEqual(
            PowerOfTwoFragment(1, 64, 64).pattern_neighbors(64, radius=2),
            [16, 32],
        )
        # IntegerFragment: radius=2 returns ±2 neighbors
        self.assertEqual(
            sorted(IntegerFragment(1, 10, 5).pattern_neighbors(5, radius=2)),
            [3, 4, 6, 7],
        )
        # IntegerFragment: radius=2 clamped at boundaries
        self.assertEqual(
            sorted(IntegerFragment(1, 5, 1).pattern_neighbors(1, radius=2)),
            [2, 3],
        )
        # BooleanFragment: radius is ignored, always returns [not current]
        self.assertEqual(BooleanFragment().pattern_neighbors(True, radius=3), [False])
        # EnumFragment: radius is ignored, always returns all other choices
        self.assertEqual(
            sorted(EnumFragment(("a", "b", "c")).pattern_neighbors("b", radius=5)),
            ["a", "c"],
        )
        # ListOf: radius is forwarded to inner fragment
        list_frag = ListOf(inner=IntegerFragment(1, 10, 5), length=2)
        neighbors = list_frag.pattern_neighbors([5, 5], radius=2)
        # Each position yields 4 neighbors (3,4,6,7), total 8
        self.assertEqual(len(neighbors), 8)
        # All neighbors differ from base in exactly one position
        for neighbor in neighbors:
            diffs = sum(1 for a, b in zip(neighbor, [5, 5], strict=True) if a != b)
            self.assertEqual(diffs, 1)

    def test_pattern_search_block_size_pair_neighbors(self):
        search = PatternSearch.__new__(PatternSearch)
        search._visited = set()
        search.config_gen = SimpleNamespace(
            flat_spec=[
                PowerOfTwoFragment(16, 128, 32),
                PowerOfTwoFragment(16, 128, 64),
                EnumFragment(("a", "b")),
            ],
            block_size_indices=[0, 1],
            overridden_flat_indices=set(),
            config_spec=SimpleNamespace(tensor_numel_constraints=[]),
        )
        search.num_neighbors_cap = -1

        base = [32, 64, "a"]
        neighbors = search._generate_neighbors(base)

        def diff_count(flat):
            return sum(
                1
                for current, original in zip(flat, base, strict=False)
                if current != original
            )

        pair_neighbors = [
            flat for flat in neighbors if diff_count(flat) == 2 and flat[2] == "a"
        ]
        expected = [
            [16, 32, "a"],
            [16, 128, "a"],
            [64, 32, "a"],
            [64, 128, "a"],
        ]
        self.assertEqual(sorted(pair_neighbors), sorted(expected))

    def test_pattern_search_skips_overridden_indices(self):
        """Neighbors are not generated along overridden (frozen) indices."""
        search = PatternSearch.__new__(PatternSearch)
        search._visited = set()
        search.config_gen = SimpleNamespace(
            flat_spec=[
                PowerOfTwoFragment(16, 128, 32),  # block_size[0] — index 0
                PowerOfTwoFragment(16, 128, 64),  # block_size[1] — index 1
                EnumFragment(("a", "b")),  # some enum — index 2
            ],
            block_size_indices=[0, 1],
            overridden_flat_indices={1},  # freeze block_size[1]
        )
        search.num_neighbors_cap = -1

        base = [32, 64, "a"]
        neighbors = search._generate_neighbors(base)

        # No neighbor should change index 1 (frozen)
        for flat in neighbors:
            self.assertEqual(flat[1], 64)

        # Neighbors should still vary indices 0 and 2
        changed_indices = set()
        for flat in neighbors:
            for i, (v, b) in enumerate(zip(flat, base, strict=False)):
                if v != b:
                    changed_indices.add(i)
        self.assertIn(0, changed_indices)
        self.assertIn(2, changed_indices)
        self.assertNotIn(1, changed_indices)

        # No block-size pair neighbors should be generated (only 1 non-frozen block index)
        pair_neighbors = [
            flat
            for flat in neighbors
            if sum(1 for v, b in zip(flat, base, strict=False) if v != b) == 2
        ]
        self.assertEqual(pair_neighbors, [])

    def test_differential_mutation_skips_overridden_indices(self):
        """Differential mutation does not mutate overridden indices."""
        random.seed(42)
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        overrides = {"num_warps": 8}
        gen = ConfigGeneration(spec, overrides=overrides)

        # Find the num_warps flat index
        warp_idx = gen.num_warps_index
        self.assertIn(warp_idx, gen.overridden_flat_indices)

        base = gen.default_flat()
        a = gen.random_flat()
        b = gen.random_flat()
        c = gen.random_flat()

        # Run many mutations — overridden index should never change
        for _ in range(50):
            result = gen.differential_mutation(base, a, b, c, crossover_rate=0.9)
            self.assertEqual(result[warp_idx], base[warp_idx])

    def test_lfbo_pattern_search_skips_overridden_indices(self):
        """LFBOPatternSearch._generate_neighbors skips overridden indices."""
        random.seed(123)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.num_neighbors = 50
        search.radius = 2
        search.config_gen = SimpleNamespace(
            flat_spec=[
                PowerOfTwoFragment(16, 128, 32),  # block_size[0]
                PowerOfTwoFragment(16, 128, 64),  # block_size[1]
                PowerOfTwoFragment(2, 16, 4),  # num_warps
                EnumFragment(("a", "b", "c")),  # some enum
                BooleanFragment(),  # some boolean
            ],
            block_size_indices=[0, 1],
            num_warps_index=2,
            overridden_flat_indices={1, 2},  # freeze block_size[1] and num_warps
        )
        search.num_neighbors_cap = -1

        base = [32, 64, 4, "b", True]
        neighbors = search._generate_neighbors(base)

        # No neighbor should change indices 1 or 2
        for flat in neighbors:
            self.assertEqual(flat[1], 64)
            self.assertEqual(flat[2], 4)

    def test_lfbo_pattern_search_generate_neighbors(self):
        """Test LFBOPatternSearch._generate_neighbors method."""
        random.seed(123)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.num_neighbors = 50
        search.radius = 2
        search.config_gen = SimpleNamespace(
            flat_spec=[
                PowerOfTwoFragment(16, 128, 32),  # block_size[0]
                PowerOfTwoFragment(16, 128, 64),  # block_size[1]
                PowerOfTwoFragment(2, 16, 4),  # num_warps
                EnumFragment(("a", "b", "c")),  # some enum
                BooleanFragment(),  # some boolean
            ],
            block_size_indices=[0, 1],
            num_warps_index=2,
            overridden_flat_indices=set(),
            config_spec=SimpleNamespace(tensor_numel_constraints=[]),
        )
        search.num_neighbors_cap = -1

        base = [32, 64, 4, "b", True]
        neighbors = search._generate_neighbors(base)

        # Check we generate the correct number of neighbors
        self.assertEqual(len(neighbors), search.num_neighbors)

        # Check all neighbors are different from base
        for neighbor in neighbors:
            self.assertNotEqual(neighbor, base)

        # Verify all block sizes are valid powers of two in range
        for neighbor in neighbors:
            # Check block_size[0]
            self.assertIn(neighbor[0], [16, 32, 64, 128])
            # Check block_size[1]
            self.assertIn(neighbor[1], [16, 32, 64, 128])
            # Check num_warps
            self.assertIn(neighbor[2], [2, 4, 8, 16])
            # Check enum
            self.assertIn(neighbor[3], ["a", "b", "c"])
            # Check boolean
            self.assertIn(neighbor[4], [True, False])

    def test_lfbo_pattern_search_surrogate_select_matches_legacy_prefix(self):
        """Top-k LFBO selection should match the legacy full-ranking implementation."""

        class MockSurrogate:
            def __init__(
                self, proba_by_id: dict[int, float], leaf_by_id: dict[int, list[int]]
            ) -> None:
                self.proba_by_id = proba_by_id
                self.leaf_by_id = leaf_by_id

            def predict_proba(self, X):
                ids = np.asarray(X)[:, 0].astype(int)
                return np.array(
                    [[1.0 - self.proba_by_id[i], self.proba_by_id[i]] for i in ids]
                )

            def apply(self, X):
                ids = np.asarray(X)[:, 0].astype(int)
                return np.array([self.leaf_by_id[i] for i in ids], dtype=int)

        def legacy_select(
            search: LFBOPatternSearch,
            candidates: list[SimpleNamespace],
            n_sorted: int,
        ) -> list[SimpleNamespace]:
            candidate_X = np.array(
                [
                    search.config_gen.encode_config(member.flat_values)
                    for member in candidates
                ]
            )
            proba = np.asarray(search.surrogate.predict_proba(candidate_X))[:, 1]
            similarity_matrix = search.compute_leaf_similarity(
                search.surrogate, candidate_X
            )
            selected_indices = []
            remaining_indices = list(range(len(candidate_X)))
            scores = np.zeros(len(candidate_X))

            for rank in range(len(candidate_X)):
                if selected_indices:
                    mean_similarities = np.zeros(len(remaining_indices))
                    for i, idx in enumerate(remaining_indices):
                        similarities_to_selected = similarity_matrix[
                            idx, selected_indices
                        ]
                        mean_similarities[i] = np.mean(similarities_to_selected)
                    ranked_scores = (
                        proba[remaining_indices]
                        - search.similarity_penalty * mean_similarities
                    )
                else:
                    ranked_scores = proba[remaining_indices]

                best_local_idx = int(np.argmax(ranked_scores))
                best_global_idx = remaining_indices[best_local_idx]
                scores[best_global_idx] = rank
                selected_indices.append(best_global_idx)
                remaining_indices.remove(best_global_idx)

            ranked = sorted(
                zip(candidates, scores, strict=True),
                key=operator.itemgetter(1),
            )[:n_sorted]
            return [member for member, _ in ranked]

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_gen = SimpleNamespace(encode_config=lambda flat: [flat[0]])
        search.similarity_penalty = 0.35
        search.log = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
        search.surrogate = MockSurrogate(
            proba_by_id={
                0: 0.95,
                1: 0.92,
                2: 0.90,
                3: 0.86,
                4: 0.84,
                5: 0.83,
            },
            leaf_by_id={
                0: [10, 20, 30, 40],
                1: [10, 20, 31, 41],
                2: [11, 21, 32, 42],
                3: [50, 60, 70, 80],
                4: [50, 61, 71, 81],
                5: [12, 22, 33, 43],
            },
        )
        candidates = [SimpleNamespace(name=f"c{i}", flat_values=[i]) for i in range(6)]

        expected = legacy_select(search, candidates, 3)

        with patch.object(
            search,
            "compute_leaf_similarity",
            side_effect=AssertionError("dense similarity matrix should not be built"),
        ):
            actual = search._surrogate_select(candidates, 3)

        self.assertEqual([c.name for c in actual], [c.name for c in expected])

    def test_tile_strategy_dispatch_compact_shape_uses_cached_block_lookup(self):
        """Fallback block-id lookups should reuse the precomputed strategy cache."""

        class DummyStrategy:
            block_ids: ClassVar[list[int]] = [3, 4]

            def block_size_var(self, block_idx: int) -> str:
                return f"_BLOCK_{block_idx}"

            def compact_shape(self, shapes):
                return shapes

        dispatch = TileStrategyDispatch.__new__(TileStrategyDispatch)
        dispatch.strategies = [DummyStrategy()]
        dispatch.block_id_to_strategy = BlockIDStrategyMapping()
        dispatch.block_id_to_strategy[(3, 4)] = dispatch.strategies[0]

        with patch(
            "helion._compiler.tile_dispatch.CompileEnvironment.current",
            return_value=SimpleNamespace(
                get_block_id=lambda _shape: 3,
                resolve_block_id=lambda _shape: 3,
            ),
        ):
            compacted = dispatch._compact_shape([object()])

        self.assertEqual(len(compacted), 1)
        self.assertEqual(compacted[0].size_str, "_BLOCK_3")
        self.assertEqual(compacted[0].block_ids, [3])

    @skip("too slow")
    def test_lfbo_pattern_search(self):
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)
        random.seed(123)
        best = LFBOPatternSearch(
            bound_kernel,
            args,
            initial_population=10,
            max_generations=2,
            copies=1,
            num_neighbors=10,
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), sum(args), rtol=1e-2, atol=1e-1)

    def test_accuracy_check_filters_bad_config_wrong_output(self) -> None:
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(configs=[bad_config, good_config], autotune_log_level=0)
        def add_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(b.size()):
                b[tile] = a[tile] + b[tile]
            return b

        def run_mode(mode: str, *, expect_error: bool) -> None:
            a = torch.randn([32], device=DEVICE)
            b = torch.randn([32], device=DEVICE)
            bound_kernel = add_inplace.bind((a, b))
            original_compile = bound_kernel.compile_config
            bound_kernel.settings.autotune_precompile = mode

            def make_bad_config_produce_wrong_output(
                config: helion.Config, *, allow_print: bool = True
            ):
                fn = original_compile(config, allow_print=allow_print)
                if config == bad_config:
                    return lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs) + 1
                return fn

            import helion.autotuner.base_search as base_search_module

            with patch.object(
                bound_kernel,
                "compile_config",
                side_effect=make_bad_config_produce_wrong_output,
            ):
                search = FiniteSearch(
                    bound_kernel, (a, b), configs=[bad_config, good_config]
                )
                search._prepare()
                if mode == "fork":
                    start_cm = patch.object(
                        search.benchmark_provider,
                        "_create_precompile_future",
                        side_effect=lambda config, fn: (
                            base_search_module.PrecompileFuture.skip(
                                search.benchmark_provider._precompile_context(),
                                config,
                                True,
                            )
                        ),
                    )
                else:
                    start_cm = nullcontext()

                with start_cm:
                    if expect_error:
                        with self.assertRaisesRegex(
                            helion.exc.AutotuneError,
                            'Set HELION_AUTOTUNE_PRECOMPILE="fork"',
                        ):
                            search.autotune()
                        return

                    bad_time = search.benchmark(bad_config).perf
                    assert math.isinf(bad_time)
                    self.assertEqual(search._autotune_metrics.num_accuracy_failures, 1)
                    search._autotune_metrics.num_accuracy_failures = 0

                    good_time = search.benchmark(good_config).perf
                    assert not math.isinf(good_time)
                    self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)
                    search._autotune_metrics.num_accuracy_failures = 0

                    best = search.autotune()
                    self.assertEqual(best, good_config)
                    self.assertEqual(search._autotune_metrics.num_accuracy_failures, 1)

        run_mode("fork", expect_error=False)
        run_mode("spawn", expect_error=True)

    def test_accuracy_check_filters_bad_config_wrong_arg_mutation(self) -> None:
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(configs=[bad_config, good_config], autotune_log_level=0)
        def add_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(b.size()):
                b[tile] = a[tile] + b[tile]
            return b

        def run_mode(mode: str, *, expect_error: bool) -> None:
            a = torch.randn([32], device=DEVICE)
            b = torch.randn([32], device=DEVICE)
            bound_kernel = add_inplace.bind((a, b))
            original_compile = bound_kernel.compile_config
            bound_kernel.settings.autotune_precompile = mode

            def make_bad_config_produce_wrong_input_arg_mutation(
                config: helion.Config, *, allow_print: bool = True
            ):
                fn = original_compile(config, allow_print=allow_print)
                if config == bad_config:

                    def wrong_fn(*fn_args, **fn_kwargs):
                        result = fn(*fn_args, **fn_kwargs)
                        # Introduce an extra mutation so inputs differ from baseline
                        fn_args[1].add_(1)
                        return result

                    return wrong_fn
                return fn

            import helion.autotuner.base_search as base_search_module

            with patch.object(
                bound_kernel,
                "compile_config",
                side_effect=make_bad_config_produce_wrong_input_arg_mutation,
            ):
                search = FiniteSearch(
                    bound_kernel, (a, b), configs=[bad_config, good_config]
                )
                search._prepare()
                if mode == "fork":
                    start_cm = patch.object(
                        search.benchmark_provider,
                        "_create_precompile_future",
                        side_effect=lambda config, fn: (
                            base_search_module.PrecompileFuture.skip(
                                search.benchmark_provider._precompile_context(),
                                config,
                                True,
                            )
                        ),
                    )
                else:
                    start_cm = nullcontext()

                with start_cm:
                    if expect_error:
                        with self.assertRaisesRegex(
                            helion.exc.AutotuneError,
                            'Set HELION_AUTOTUNE_PRECOMPILE="fork"',
                        ):
                            search.autotune()
                        return

                    bad_time = search.benchmark(bad_config).perf
                    assert math.isinf(bad_time)
                    self.assertEqual(search._autotune_metrics.num_accuracy_failures, 1)
                    search._autotune_metrics.num_accuracy_failures = 0

                    good_time = search.benchmark(good_config).perf
                    assert not math.isinf(good_time)
                    self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)
                    search._autotune_metrics.num_accuracy_failures = 0

                    best = search.autotune()
                    self.assertEqual(best, good_config)
                    self.assertGreaterEqual(
                        search._autotune_metrics.num_accuracy_failures, 1
                    )

        run_mode("fork", expect_error=False)
        run_mode("spawn", expect_error=True)

    def test_autotune_baseline_fn(self) -> None:
        """Test that custom baseline function is used for accuracy checking."""
        config1 = helion.Config(block_sizes=[32], num_warps=4)
        config2 = helion.Config(block_sizes=[64], num_warps=8)

        # Track whether the baseline function was called
        baseline_calls = []

        def custom_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            baseline_calls.append(True)
            # Return the expected result using PyTorch operations
            return a + b

        @helion.kernel(
            configs=[config1, config2],
            autotune_baseline_fn=custom_baseline,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )

        # Run autotuning
        result = add(*args)

        # Verify the custom baseline function was called during autotuning
        self.assertGreater(
            len(baseline_calls), 0, "Custom baseline function should be called"
        )

        # Verify the result is correct
        torch.testing.assert_close(result, args[0] + args[1])

    def test_autotune_baseline_fn_filters_bad_config(self) -> None:
        """Test that custom baseline function correctly filters incorrect configs."""
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        def custom_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
            # Return the correct expected result
            return a + b

        @helion.kernel(
            configs=[bad_config, good_config],
            autotune_baseline_fn=custom_baseline,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([32], device=DEVICE)
        b = torch.randn([32], device=DEVICE)
        bound_kernel = add.bind((a, b))
        original_compile = bound_kernel.compile_config
        bound_kernel.settings.autotune_precompile = "fork"

        # Make bad_config produce wrong output
        def make_bad_config_produce_wrong_output(
            config: helion.Config, *, allow_print: bool = True
        ):
            fn = original_compile(config, allow_print=allow_print)
            if config == bad_config:
                return lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs) + 1
            return fn

        import helion.autotuner.base_search as base_search_module

        with patch.object(
            bound_kernel,
            "compile_config",
            side_effect=make_bad_config_produce_wrong_output,
        ):
            search = FiniteSearch(
                bound_kernel, (a, b), configs=[bad_config, good_config]
            )
            search._prepare()
            with patch.object(
                search.benchmark_provider,
                "_create_precompile_future",
                side_effect=lambda config, fn: base_search_module.PrecompileFuture.skip(
                    search.benchmark_provider._precompile_context(), config, True
                ),
            ):
                # Bad config should be filtered out by accuracy check
                bad_time = search.benchmark(bad_config).perf
                self.assertTrue(math.isinf(bad_time))
                self.assertEqual(search._autotune_metrics.num_accuracy_failures, 1)

                # Good config should pass accuracy check
                search._autotune_metrics.num_accuracy_failures = 0
                good_time = search.benchmark(good_config).perf
                self.assertFalse(math.isinf(good_time))
                self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

                # Autotuning should select the good config
                best = search.autotune()
                self.assertEqual(best, good_config)

    def test_autotune_baseline_fn_raises_on_failure(self) -> None:
        """Test that AutotuneError is raised when custom baseline function fails."""
        config1 = helion.Config(block_sizes=[32], num_warps=4)
        config2 = helion.Config(block_sizes=[64], num_warps=8)

        def failing_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("Baseline computation failed!")

        @helion.kernel(
            configs=[config1, config2],
            autotune_baseline_fn=failing_baseline,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )

        # Attempting to run should raise AutotuneError
        with self.assertRaisesRegex(
            helion.exc.AutotuneError,
            "Custom baseline function failed while computing baseline",
        ):
            add(*args)

    def test_autotune_baseline_tolerance(self) -> None:
        cfg1 = helion.Config(block_sizes=[1], num_warps=4)
        cfg2 = helion.Config(block_sizes=[1], num_warps=8)
        a, b = torch.randn([32], device=DEVICE), torch.randn([32], device=DEVICE)

        # Baseline that returns slightly incorrect result (1e-4 error)
        def incorrect_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b + 1e-4

        # Test both strict (1e-5) and lenient (1e-3) tolerances
        for tol, expect_reject in [(1e-5, True), (1e-3, False)]:

            @helion.kernel(
                configs=[cfg1, cfg2],
                autotune_baseline_fn=incorrect_baseline,
                autotune_baseline_atol=tol,
                autotune_baseline_rtol=tol,
            )
            def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                o = torch.empty_like(a)
                for t in hl.tile(o.size()):
                    o[t] = a[t] + b[t]
                return o

            bound = add.bind((a, b))
            search = FiniteSearch(bound, (a, b), configs=[cfg1, cfg2])

            if expect_reject:
                # FiniteSearch currently raises AssertionError if every config fails validation
                with self.assertRaises(AssertionError):
                    search.autotune()
                # All configs should have tripped the accuracy mismatch counter
                self.assertEqual(
                    search._autotune_metrics.num_accuracy_failures, len(search.configs)
                )
            else:
                winner = search.autotune()
                self.assertIn(winner, (cfg1, cfg2))
                self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_autotune_fp8_automatic_tolerance(self) -> None:
        """Test that fp8 dtypes automatically get 0.0 tolerances."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=8)

        # Test with float8_e4m3fn as a representative fp8 dtype
        @helion.kernel(configs=[cfg1, cfg2])
        def cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.size(), dtype=torch.float8_e4m3fn, device=x.device)
            for t in hl.tile(x.size()):
                out[t] = x[t].to(torch.float8_e4m3fn)
            return out

        x = torch.randn([64], device=DEVICE)
        bound = cast_to_fp8.bind((x,))
        search = FiniteSearch(bound, (x,), configs=[cfg1, cfg2])
        search._prepare()

        # Verify that effective tolerances were set to 0.0 automatically
        self.assertEqual(
            search.benchmark_provider._effective_atol,
            0.0,
            f"Expected automatic atol=0.0 for fp8, got {search.benchmark_provider._effective_atol}",
        )
        self.assertEqual(
            search.benchmark_provider._effective_rtol,
            0.0,
            f"Expected automatic rtol=0.0 for fp8, got {search.benchmark_provider._effective_rtol}",
        )

        # Should successfully autotune without error
        winner = search.autotune()
        self.assertIn(winner, (cfg1, cfg2))
        self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_autotune_fp8_explicit_tolerance_override(self) -> None:
        """Test that explicit tolerances override automatic fp8 detection."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=8)

        # User explicitly sets non-zero tolerances despite fp8 output
        @helion.kernel(
            configs=[cfg1, cfg2],
            autotune_baseline_atol=1e-5,
            autotune_baseline_rtol=1e-5,
        )
        def cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.size(), dtype=torch.float8_e4m3fn, device=x.device)
            for t in hl.tile(x.size()):
                out[t] = x[t].to(torch.float8_e4m3fn)
            return out

        x = torch.randn([64], device=DEVICE)
        bound = cast_to_fp8.bind((x,))
        search = FiniteSearch(bound, (x,), configs=[cfg1, cfg2])
        search._prepare()

        # Should respect user's explicit tolerances, not override to 0.0
        self.assertEqual(search.benchmark_provider._effective_atol, 1e-5)
        self.assertEqual(search.benchmark_provider._effective_rtol, 1e-5)

    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_autotune_mixed_fp8_and_fp32_output(self) -> None:
        """Test that the accuracy check works with mixed fp8+fp32 outputs."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=8)

        @helion.kernel(configs=[cfg1, cfg2])
        def mixed_output(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            fp8_out = torch.empty(x.size(), dtype=torch.float8_e4m3fn, device=x.device)
            fp32_out = torch.empty(x.size(), dtype=torch.float32, device=x.device)
            for t in hl.tile(x.size()):
                fp8_out[t] = x[t].to(torch.float8_e4m3fn)
                fp32_out[t] = x[t] * 2.0
            return fp8_out, fp32_out

        x = torch.randn([64], device=DEVICE)
        bound = mixed_output.bind((x,))
        search = FiniteSearch(bound, (x,), configs=[cfg1, cfg2])

        # Should successfully autotune without error
        winner = search.autotune()
        self.assertIn(winner, (cfg1, cfg2))
        self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

    def test_max_generations(self):
        """Autotuner max generation respects explicit kwargs then setting override."""

        with patch.dict(os.environ, {"HELION_AUTOTUNER": "PatternSearch"}):

            @helion.kernel(autotune_max_generations=1)
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            args = (
                torch.randn([8], device=DEVICE),
                torch.randn([8], device=DEVICE),
            )

            bound = add.bind(args)
            autotuner_factory = bound.settings.autotuner_fn

            # Settings override defaults
            autotuner = autotuner_factory(bound, args)
            self.assertEqual(autotuner.autotuner.max_generations, 1)

            # Explicit constructor value wins
            autotuner_override = autotuner_factory(bound, args, max_generations=2)
            self.assertEqual(autotuner_override.autotuner.max_generations, 2)

    def test_autotune_effort_none(self):
        @helion.kernel(autotune_effort="none")
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        result = add(*args)
        torch.testing.assert_close(result, sum(args))

    def test_env_autotune_effort_none_ignores_force_autotune(self):
        autotuner_fn = Mock(
            side_effect=AssertionError("autotuner should not run with effort=none")
        )

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNE_EFFORT": "none",
                "HELION_FORCE_AUTOTUNE": "1",
            },
        ):

            @helion.kernel(autotuner_fn=autotuner_fn)
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

        args = (
            torch.randn([8], device=DEVICE),
            torch.randn([8], device=DEVICE),
        )
        result = add(*args)
        torch.testing.assert_close(result, sum(args))
        autotuner_fn.assert_not_called()

    def test_autotune_effort_quick(self):
        """Test that quick effort profile uses correct default values."""
        # Get the quick profile defaults
        quick_profile = get_effort_profile("quick")
        assert quick_profile.lfbo_pattern_search is not None
        expected_initial_pop = quick_profile.lfbo_pattern_search.initial_population
        expected_copies = quick_profile.lfbo_pattern_search.copies
        expected_max_gen = quick_profile.lfbo_pattern_search.max_generations

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )

        # Test 1: Default quick mode values from effort profile (LFBOTreeSearch is default)
        with patch.dict(os.environ, {"HELION_AUTOTUNER": "LFBOTreeSearch"}):

            @helion.kernel(autotune_effort="quick")
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            bound = add.bind(args)
            autotuner = bound.settings.autotuner_fn(bound, args)
            lfbo_tree = autotuner.autotuner
            self.assertIsInstance(lfbo_tree, LFBOTreeSearch)
            # Use exact values from quick profile
            self.assertEqual(lfbo_tree.initial_population, expected_initial_pop)
            self.assertEqual(lfbo_tree.copies, expected_copies)
            self.assertEqual(lfbo_tree.max_generations, expected_max_gen)

        # Test 2: HELION_AUTOTUNE_MAX_GENERATIONS overrides effort profile
        override_max_gen = 100
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "LFBOTreeSearch",
                "HELION_AUTOTUNE_MAX_GENERATIONS": str(override_max_gen),
            },
        ):

            @helion.kernel(autotune_effort="quick")
            def add_with_override(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            bound = add_with_override.bind(args)
            autotuner = bound.settings.autotuner_fn(bound, args)
            lfbo_tree = autotuner.autotuner
            self.assertIsInstance(lfbo_tree, LFBOTreeSearch)
            # initial_population and copies from profile, but max_generations from env var
            self.assertEqual(lfbo_tree.initial_population, expected_initial_pop)
            self.assertEqual(lfbo_tree.copies, expected_copies)
            self.assertEqual(lfbo_tree.max_generations, override_max_gen)

        # Test 3: Explicit constructor values take highest priority
        explicit_initial_pop = 500
        explicit_copies = 300
        explicit_max_gen = 150

        bound = add.bind(args)
        lfbo_tree = LFBOTreeSearch(
            bound,
            args,
            initial_population=explicit_initial_pop,
            copies=explicit_copies,
            max_generations=explicit_max_gen,
        )
        # All values from explicit constructor args
        self.assertEqual(lfbo_tree.initial_population, explicit_initial_pop)
        self.assertEqual(lfbo_tree.copies, explicit_copies)
        self.assertEqual(lfbo_tree.max_generations, explicit_max_gen)

    def test_finishing_rounds(self):
        """finishing_rounds comes from profile, env var overrides, explicit ctor arg wins."""
        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )

        @helion.kernel(autotune_effort="quick")
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        bound = add.bind(args)
        quick_profile = get_effort_profile("quick")

        # Default: comes from effort profile
        with patch.dict(os.environ, {"HELION_AUTOTUNER": "PatternSearch"}):
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertEqual(
                autotuner.autotuner.finishing_rounds, quick_profile.finishing_rounds
            )

        # Env var overrides effort profile
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "PatternSearch",
                "HELION_AUTOTUNE_FINISHING_ROUNDS": "7",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertEqual(autotuner.autotuner.finishing_rounds, 7)

        # Explicit constructor arg wins over env var
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "PatternSearch",
                "HELION_AUTOTUNE_FINISHING_ROUNDS": "7",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args, finishing_rounds=3)
            self.assertEqual(autotuner.autotuner.finishing_rounds, 3)

    def test_num_neighbors_cap(self):
        """num_neighbors_cap defaults to -1, env var overrides, explicit ctor arg wins."""
        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )

        @helion.kernel()
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        bound = add.bind(args)

        # Default: -1 (no cap)
        with patch.dict(os.environ, {"HELION_AUTOTUNER": "PatternSearch"}):
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertEqual(autotuner.autotuner.num_neighbors_cap, -1)

        # Env var overrides default
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "PatternSearch",
                "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "50",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertEqual(autotuner.autotuner.num_neighbors_cap, 50)

        # Explicit constructor arg wins over env var
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "PatternSearch",
                "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "50",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args, num_neighbors_cap=10)
            self.assertEqual(autotuner.autotuner.num_neighbors_cap, 10)

    def test_autotuner_disabled(self):
        @helion.kernel()
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        with (
            patch.dict(os.environ, {"HELION_DISALLOW_AUTOTUNING": "1"}),
            pytest.raises(
                expected_exception=helion.exc.AutotuningDisallowedInEnvironment,
                match="Autotuning is disabled by HELION_DISALLOW_AUTOTUNING=1, please provide a config to @helion.kernel via the config= argument.",
            ),
        ):
            add(*args)

    def test_fragment_encoding(self):
        """Test encoding functionality for all ConfigSpecFragment types."""
        # Test BooleanFragment
        bool_frag = BooleanFragment()
        self.assertEqual(bool_frag.dim(), 1)
        self.assertEqual(bool_frag.encode(True), [1.0])
        self.assertEqual(bool_frag.encode(False), [0.0])

        # Test IntegerFragment
        int_frag = IntegerFragment(low=1, high=10, default_val=5)
        self.assertEqual(int_frag.dim(), 1)
        self.assertEqual(int_frag.encode(5), [5.0])

        # Test PowerOfTwoFragment (log2 transformation)
        pow2_frag = PowerOfTwoFragment(low=2, high=128, default_val=8)
        self.assertEqual(pow2_frag.dim(), 1)
        self.assertEqual(pow2_frag.encode(8), [3.0])  # log2(8) = 3
        self.assertEqual(pow2_frag.encode(16), [4.0])  # log2(16) = 4

        # Test NumThreadsFragment (0 is the CuTe auto-thread sentinel)
        num_threads_frag = NumThreadsFragment(high=128)
        self.assertEqual(num_threads_frag.dim(), 1)
        self.assertEqual(num_threads_frag.encode(0), [0.0])
        self.assertEqual(num_threads_frag.encode(8), [4.0])

        # Test EnumFragment (one-hot encoding)
        enum_frag = EnumFragment(choices=("a", "b", "c"))
        self.assertEqual(enum_frag.dim(), 3)
        self.assertEqual(enum_frag.encode("a"), [1.0, 0.0, 0.0])
        self.assertEqual(enum_frag.encode("b"), [0.0, 1.0, 0.0])

        # Test PermutationFragment
        perm_frag = PermutationFragment(length=3)
        self.assertEqual(perm_frag.dim(), 3)
        encoded = perm_frag.encode([0, 1, 2])
        self.assertEqual(encoded, [0.0, 1.0, 2.0])

        # Test ListOf with BooleanFragment
        list_frag = ListOf(inner=BooleanFragment(), length=3)
        self.assertEqual(list_frag.dim(), 3)
        self.assertEqual(list_frag.encode([True, False, True]), [1.0, 0.0, 1.0])

        # Test encode_dim consistency
        for fragment, value in [
            (BooleanFragment(), True),
            (IntegerFragment(1, 10, 5), 5),
            (PowerOfTwoFragment(2, 128, 8), 16),
            (NumThreadsFragment(128), 0),
            (EnumFragment(choices=("a", "b")), "b"),
        ]:
            dim = fragment.dim()
            encoded = fragment.encode(value)
            self.assertEqual(len(encoded), dim)

    def test_block_size_fragment_autotuner_min_clamp(self):
        """random_config() must not crash when autotuner_min > max_size."""
        from examples.attention import attention

        q, k, v = [
            torch.randn(4, 48, 128, 128, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(3)
        ]
        bound = attention.bind((q, k, v))
        config_spec = bound.config_spec
        config_spec.raise_grid_block_minimums()
        gen = ConfigGeneration(config_spec)
        config = gen.random_config()
        self.assertEqual(config["block_sizes"][0], 1)

    def test_autotune_benchmark_fn(self) -> None:
        """Test that custom benchmark function is used during rebenchmarking."""
        # Track benchmark function calls
        benchmark_calls: list[tuple[int, int]] = []  # (num_fns, repeat)

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            benchmark_calls.append((len(fns), repeat))
            # Return fake timings
            return [1.0] * len(fns)

        @helion.kernel(
            autotune_benchmark_fn=custom_benchmark_fn,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )

        bound_kernel = add.bind(args)
        # Use PatternSearch which has rebenchmark method
        search = PatternSearch(bound_kernel, args)

        # Compile two configs
        config1 = search.config_gen.random_config()
        config2 = search.config_gen.random_config()
        fn1 = bound_kernel.compile_config(config1)
        fn2 = bound_kernel.compile_config(config2)

        # Create population members (flat_values not used in rebenchmark)
        member1 = PopulationMember(fn1, [1.0], (), config1)
        member2 = PopulationMember(fn2, [1.1], (), config2)

        search._prepare()
        search.best_perf_so_far = 1.0

        # Call rebenchmark directly
        search.rebenchmark([member1, member2])

        # Verify custom benchmark function was called
        self.assertGreater(
            len(benchmark_calls), 0, "Custom benchmark function should be called"
        )
        # Should have been called with 2 functions
        self.assertEqual(benchmark_calls[0][0], 2)

    def test_rebenchmark_clears_jit_fast_path_caches(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.config_spec = SimpleNamespace(backend=None)
        events: list[str] = []

        class FakeJITFunction:
            def __init__(self, name: str) -> None:
                self.name = name

            def clear_fast_path_caches(self) -> None:
                events.append(f"clear {self.name}")

        def generated_kernel_a() -> None:
            events.append("run a")

        def generated_kernel_b() -> None:
            events.append("run b")

        globals_key_a = f"_helion_{generated_kernel_a.__name__}"
        globals_key_b = f"_helion_{generated_kernel_b.__name__}"
        generated_kernel_a.__globals__[globals_key_a] = FakeJITFunction("a")
        generated_kernel_b.__globals__[globals_key_b] = FakeJITFunction("b")

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            for _ in range(2):
                for fn in fns:
                    fn()
            return [100.0, 101.0]

        search.settings.autotune_benchmark_fn = custom_benchmark_fn
        member_a = PopulationMember(generated_kernel_a, [100.0], [], helion.Config())
        member_b = PopulationMember(generated_kernel_b, [101.0], [], helion.Config())
        try:
            search.rebenchmark([member_a, member_b])
        finally:
            del generated_kernel_a.__globals__[globals_key_a]
            del generated_kernel_b.__globals__[globals_key_b]

        self.assertEqual(
            events,
            [
                "run a",
                "clear a",
                "run b",
                "clear b",
                "run a",
                "clear a",
                "run b",
                "clear b",
                "clear a",
                "clear b",
            ],
        )

    def test_autotune_configuration_cloning(self) -> None:
        """Tests base_search._clone_args function."""

        config1 = helion.Config(block_sizes=[32, 32], num_warps=4)
        config2 = helion.Config(block_sizes=[64, 64], num_warps=8)

        @helion.kernel(
            configs=[config1, config2],
            autotune_log_level=0,
        )
        def nested_in_place_add(
            a: Sequence[torch.Tensor],
            b: Sequence[torch.Tensor],
            epsilon: float,
            out: Sequence[torch.Tensor],
        ):
            for tile in hl.tile(out[0].size()):
                out[0][tile] += a[0][tile] + b[0][tile] + epsilon
            for tile in hl.tile(out[1].size()):
                out[1][tile] += a[1][tile] + b[1][tile] + epsilon

        epsilon = 1e-6
        args = (
            [torch.ones([128], device=DEVICE), torch.ones([128], device=DEVICE)],
            [torch.ones([128], device=DEVICE), torch.ones([128], device=DEVICE)],
            epsilon,
            [torch.zeros([128], device=DEVICE), torch.zeros([128], device=DEVICE)],
        )

        # Run autotuning
        nested_in_place_add(*args)

        # test that we overwrite c only once and the arguments are correctly
        #  cloned for each autotune run
        ref_out = [
            torch.full([128], 2.0, device=DEVICE) + epsilon,
            torch.full([128], 2.0, device=DEVICE) + epsilon,
        ]
        torch.testing.assert_close(args[3], ref_out)

    def test_only_mutated_tensors_cloned_during_benchmark(self) -> None:
        """
        During benchmarking, only mutated tensors should be cloned.
        Non-mutated tensors should only be cloned during initialization.
        """
        config1 = helion.Config(block_sizes=[32], num_warps=4)
        config2 = helion.Config(block_sizes=[64], num_warps=4)

        @helion.kernel(configs=[config1, config2], autotune_log_level=0)
        def inplace_add(
            a: torch.Tensor,
            b: torch.Tensor,
            epsilon: float,
            out: torch.Tensor,
        ):
            for tile in hl.tile(out.size()):
                out[tile] += a[tile] + b[tile] + epsilon

        a = torch.full([128], 1.0, device=DEVICE)
        b = torch.full([128], 2.0, device=DEVICE)
        epsilon = 1e-6
        out = torch.zeros([128], device=DEVICE)

        # Track clones separately for mutated vs non-mutated tensors
        mutated_ptrs = {out.data_ptr()}
        non_mutated_ptrs = {a.data_ptr(), b.data_ptr()}
        mutated_clones = [0]
        non_mutated_clones = [0]

        original_clone = torch.Tensor.clone

        def tracking_clone(self, *args, **kwargs):
            result = original_clone(self, *args, **kwargs)
            if self.data_ptr() in mutated_ptrs:
                mutated_ptrs.add(result.data_ptr())
                mutated_clones[0] += 1
            if self.data_ptr() in non_mutated_ptrs:
                non_mutated_ptrs.add(result.data_ptr())
                non_mutated_clones[0] += 1
            return result

        with patch.object(torch.Tensor, "clone", tracking_clone):
            inplace_add(a, b, epsilon, out)

        # Mutated tensor (out) should be cloned during baseline AND benchmarking:
        #   _compute_baseline: 1 + baseline_post_args: 1
        #   + 2 benchmark runs = 4 total
        self.assertEqual(
            mutated_clones[0],
            4,
            f"Mutated tensor cloned {mutated_clones[0]} times, expected 4.",
        )

        # Non-mutated tensors (a, b) should only be cloned during baseline:
        #   _compute_baseline: 2 = 2 total
        self.assertEqual(
            non_mutated_clones[0],
            2,
            f"Non-mutated tensors cloned {non_mutated_clones[0]} times, expected 2. "
            f"Only mutated tensors should be cloned during benchmarking.",
        )

        expected = torch.full([128], 3.0, device=DEVICE) + epsilon
        torch.testing.assert_close(out, expected)

    @skipIfXPU("CUDA specific API used to check memory usage")
    def test_chunked_allclose_memory(self):
        """Test that autotuning accuracy checks use chunked comparison for large tensors."""
        import helion.autotuner.benchmark_provider as _bs

        numel = 2**26  # 64M float32 elements (~256 MB each)

        config1 = helion.Config(block_sizes=[128], num_warps=4)
        config2 = helion.Config(block_sizes=[256], num_warps=4)

        # Pin the accuracy check to the parent so the patched _assert_close
        # below is observed; the default subprocess path runs it in the worker.
        @helion.kernel(
            configs=[config1, config2],
            autotune_log_level=0,
            autotune_benchmark_subprocess=False,
        )
        def vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(a.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn(numel, device=DEVICE)
        b = torch.randn(numel, device=DEVICE)

        # Measure naive baseline: peak memory of torch.testing.assert_close
        # on tensors of the same size
        ref_a = torch.randn(numel, device=DEVICE)
        ref_b = ref_a.clone()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        torch.testing.assert_close(ref_a, ref_b, atol=1e-2, rtol=1e-2)
        naive_peak = torch.cuda.max_memory_allocated() - base_mem
        del ref_a, ref_b

        # Patch _assert_close to record peak memory delta during each call
        real_assert_close = _bs._assert_close
        peaks: list[int] = []

        def measuring_assert_close(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            real_assert_close(*args, **kwargs)
            peak = torch.cuda.max_memory_allocated() - before
            peaks.append(peak)

        with patch.object(_bs, "_assert_close", measuring_assert_close):
            out = vec_add(a, b)

        # Accuracy check was called at least once
        self.assertGreater(len(peaks), 0, "Expected _assert_close to be called")

        # Every call's peak memory should be less than naive peak
        for i, p in enumerate(peaks):
            self.assertLess(
                p,
                naive_peak * 0.5,
                f"Call {i}: peak {p} should be < 50% of naive {naive_peak}",
            )

        # Kernel result is correct
        torch.testing.assert_close(out, a + b)

    def test_autotune_baseline_accuracy_check_fn(self) -> None:
        """Test the built-in assert_close_with_mismatch_tolerance utility.

        Simulates a scenario where most elements match exactly, but a
        tiny fraction (1/N) have large diffs.  The default
        torch.testing.assert_close would reject this, but the utility
        falls back to checking mismatch_pct, max_abs_diff, and
        max_rel_diff thresholds and accepts it.
        """
        import functools

        import helion.autotuner.base_search as base_search_module

        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(
            configs=[bad_config, good_config],
            autotune_log_level=0,
            autotune_baseline_accuracy_check_fn=functools.partial(
                assert_close_with_mismatch_tolerance,
                max_mismatch_pct=0.01,
                max_rel_diff=15.0,
            ),
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            o = torch.empty_like(a)
            for t in hl.tile(o.size()):
                o[t] = a[t] + b[t]
            return o

        # Use a large tensor so mismatch fraction is tiny (1/N)
        N = 4096
        a = torch.randn([N], device=DEVICE)
        b = torch.randn([N], device=DEVICE)
        bound = add.bind((a, b))
        original_compile = bound.compile_config

        def inject_large_diffs_to_some_elements(
            config: helion.Config, *, allow_print: bool = True
        ):
            fn = original_compile(config, allow_print=allow_print)
            if config == bad_config:
                # Simulate mismatch: 1 element out of N with rel diff ~12
                def patched(*fn_args, **fn_kwargs):
                    result = fn(*fn_args, **fn_kwargs)
                    result[0] = result[0] + 12.0 * result[0].abs().clamp(min=1e-6)
                    return result

                return patched
            return fn

        with patch.object(
            bound,
            "compile_config",
            side_effect=inject_large_diffs_to_some_elements,
        ):
            search = FiniteSearch(bound, (a, b), configs=[bad_config, good_config])
            search._prepare()

            with patch.object(
                search.benchmark_provider,
                "_create_precompile_future",
                side_effect=lambda config, fn: base_search_module.PrecompileFuture.skip(
                    search.benchmark_provider._precompile_context(), config, True
                ),
            ):
                # bad_config has a few large diffs — custom check should accept it
                bad_time = search.benchmark(bad_config).perf
                assert not math.isinf(bad_time), (
                    "custom check should allow config with 1/N large diffs"
                )
                self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

                # good_config produces exact match — should also pass
                good_time = search.benchmark(good_config).perf
                assert not math.isinf(good_time)
                self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

        # Direct checks: element 0 has abs_diff=9.0, rel_diff=9.0
        actual = torch.tensor([10.0, 1.0, 1.0, 1.0], device=DEVICE)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0], device=DEVICE)

        # Only max_rel_diff exceeded (abs_diff=9 < 20, rel_diff=9 > 5)
        with self.assertRaisesRegex(AssertionError, "Relative diff too large"):
            assert_close_with_mismatch_tolerance(
                actual,
                expected,
                max_mismatch_pct=0.5,
                max_abs_diff=20.0,
                max_rel_diff=5.0,
            )

        # Only max_abs_diff exceeded (abs_diff=9 > 5, rel_diff=9 < 20)
        with self.assertRaisesRegex(AssertionError, "Absolute diff too large"):
            assert_close_with_mismatch_tolerance(
                actual,
                expected,
                max_mismatch_pct=0.5,
                max_abs_diff=5.0,
                max_rel_diff=20.0,
            )

    def test_autotune_baseline_accuracy_check_fn_rejects(self) -> None:
        """Test that a strict custom check function properly rejects configs."""
        cfg1 = helion.Config(block_sizes=[1], num_warps=4)
        cfg2 = helion.Config(block_sizes=[1], num_warps=8)

        def strict_check(actual: object, expected: object) -> None:
            # Always reject
            raise AssertionError("strict check: always fails")

        @helion.kernel(
            configs=[cfg1, cfg2],
            autotune_log_level=0,
            autotune_baseline_accuracy_check_fn=strict_check,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            o = torch.empty_like(a)
            for t in hl.tile(o.size()):
                o[t] = a[t] + b[t]
            return o

        a = torch.randn([32], device=DEVICE)
        b = torch.randn([32], device=DEVICE)
        bound = add.bind((a, b))
        search = FiniteSearch(bound, (a, b), configs=[cfg1, cfg2])

        with self.assertRaises(AssertionError):
            search.autotune()
        self.assertEqual(
            search._autotune_metrics.num_accuracy_failures, len(search.configs)
        )


@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
@skipUnlessCuteAvailable("CUTLASS CuTe Python DSL is not available")
@onlyBackends(["cute"])
class TestCuteAutotuner(TestCase):
    def test_implicit_call_uses_autotuner_fn(self) -> None:
        calls: list[bool] = []

        def autotuner_fn(bound_kernel, args, **kwargs):
            class RecordingAutotuner:
                def autotune(self, *, skip_cache: bool = False):
                    calls.append(skip_cache)
                    return bound_kernel.config_spec.default_config()

            return RecordingAutotuner()

        @helion.kernel(
            backend="cute",
            autotuner_fn=autotuner_fn,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8], device=DEVICE),
            torch.randn([8], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))
        self.assertEqual(calls, [False])

    def test_cute_config_generation_repairs_num_threads(self) -> None:
        @helion.kernel(backend="cute", autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([16, 64], device=DEVICE),
            torch.randn([16, 64], device=DEVICE),
        )
        bound = add.bind(args)
        gen = ConfigGeneration(bound.config_spec)
        flat_keys = {
            key for key, _count, _is_sequence in gen.config_spec.flat_key_layout()
        }
        # ``loop_orders`` is exposed for the cute non-tcgen05 search
        # surface (audited fp32 1024^3 matmul finds ~3x bench-time wins
        # from ``[[1, 0]]`` over the default ``[[0, 1]]`` — see
        # ``cute_plan.md`` §7.0). ``cute_vector_widths`` is the per-axis
        # vec width slot registered for non-reduction tile blocks. The
        # set still excludes Triton-style knobs that the cute path does
        # not consume.
        self.assertEqual(
            flat_keys,
            {"block_sizes", "num_threads", "loop_orders", "cute_vector_widths"},
        )

        repaired = gen.unflatten(
            gen.flatten(helion.Config(block_sizes=[16, 64], num_threads=[128, 128]))
        )
        self.assertEqual(repaired.block_sizes, [16, 64])
        self.assertEqual(repaired.num_threads, [16, 64])

        configs = [gen.random_config() for _ in range(20)]
        self.assertTrue(any(config.num_threads for config in configs))
        for config in configs:
            self.assertLessEqual(
                set(config.config),
                {"block_sizes", "num_threads", "loop_orders", "cute_vector_widths"},
            )
            self.assertNotIn("persistent", config.pid_type)
            explicit_threads = [nt for nt in config.num_threads if nt > 0]
            if explicit_threads:
                self.assertLessEqual(math.prod(explicit_threads), 1024)
            for block_size, num_threads in zip(
                config.block_sizes,
                config.num_threads,
                strict=False,
            ):
                if num_threads > 0:
                    self.assertLessEqual(num_threads, block_size)
                    self.assertEqual(block_size % num_threads, 0)
        # Deterministic round-trip pins the widened surface: a config
        # explicitly built with ``loop_orders=[[1, 0]]`` must survive
        # flatten/unflatten unchanged (otherwise the autotuner cannot
        # actually explore the alternate order).
        round_tripped = gen.unflatten(
            gen.flatten(
                helion.Config(
                    block_sizes=[16, 64],
                    num_threads=[16, 64],
                    loop_orders=[[1, 0]],
                )
            )
        )
        self.assertEqual(round_tripped.loop_orders, [[1, 0]])

    def test_cute_tcgen05_search_surface_excludes_loop_orders(self) -> None:
        """tcgen05 persistent scheduler relies on a fixed
        ``pid_info[0]=M, pid_info[1]=N`` mapping (``cluster_m`` and
        virtual-PID logic). Sampling ``loop_orders=[[1, 0]]`` for a
        tcgen05 config would steer cluster logic onto the wrong axis.
        The cute non-tcgen05 widening must not leak into the tcgen05
        branch.
        """
        bound = _get_examples_matmul().bind(
            (
                torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
                torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            )
        )
        # Confirm we are on the tcgen05 search branch.
        self.assertTrue(bound.config_spec.cute_tcgen05_search_enabled)
        flat_keys = {
            key for key, _count, _is_sequence in bound.config_spec.flat_key_layout()
        }
        self.assertNotIn("loop_orders", flat_keys)


@onlyBackends(["triton"])
class TestAutotuneRandomSeed(RefEagerTestDisabled, TestCase):
    def _autotune_and_record(self, **settings: object) -> float:
        search_capture: dict[str, RecordingRandomSearch] = {}

        def autotuner_factory(bound_kernel, args, **kwargs):
            search = RecordingRandomSearch(bound_kernel, args, count=4, **kwargs)
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


class TestAutotuneBestOfKSettings(TestCase):
    """Settings-only coverage for ``autotune_best_of_k`` (no GPU/backend
    dependency)."""

    def test_default_is_one(self) -> None:
        with without_env_var("HELION_AUTOTUNE_BEST_OF_K"):
            self.assertEqual(helion.Settings().autotune_best_of_k, 1)

    def test_env_var_override(self) -> None:
        with patch.dict(os.environ, {"HELION_AUTOTUNE_BEST_OF_K": "5"}, clear=False):
            self.assertEqual(helion.Settings().autotune_best_of_k, 5)

    def test_setting_override(self) -> None:
        self.assertEqual(helion.Settings(autotune_best_of_k=3).autotune_best_of_k, 3)

    def test_k_zero_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"autotune_best_of_k must be >= 1"):
            helion.Settings(autotune_best_of_k=0)

    def test_k_negative_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"autotune_best_of_k must be >= 1"):
            helion.Settings(autotune_best_of_k=-3)


@onlyBackends(["triton"])
class TestAutotuneBestOfK(RefEagerTestDisabled, TestCase):
    """Best-of-K multi-seed autotune selection — cache key + K-loop coverage.

    Covers:
      - K = 1 leaves the cache hash byte-identical with the pre-feature
        repr (no field appended to the dataclass repr).
      - K > 1 differentiates the cache hash structurally; the K value
        appears as a field on the key dataclass, not concatenated into
        ``extra_cache_key``.
      - K > 1 with no ``_autotuner_factory`` wired raises a clear error.
      - The K-loop runs K trials with deterministic per-trial seeds,
        and the winner is picked by the **final rebench** (not the
        per-trial ``best_perf_so_far`` low-water mark).
      - The autotuner reference on the cache is restored to the
        original after the loop.
    """

    def test_cache_key_byte_identical_when_k_is_one(self) -> None:
        """K=1 cache hash must match the bytes produced by the original
        ``LooseAutotuneCacheKey`` repr (before ``best_of_k`` was added)."""
        from helion.autotuner.base_cache import LooseAutotuneCacheKey

        # Build a key with K=1 and one with K=5; the K=1 repr must equal
        # the repr that omits ``best_of_k`` entirely.
        common_kwargs = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "abc",
            "hardware": "B200",
            "runtime_name": "12.6",
            "backend": "triton",
            "config_spec_hash": "h1",
            "extra_cache_key": "",
        }
        k1 = LooseAutotuneCacheKey(**common_kwargs, best_of_k=1)
        k5 = LooseAutotuneCacheKey(**common_kwargs, best_of_k=5)
        # K=1 repr matches a manually-constructed repr without best_of_k.
        expected_k1_repr = (
            "LooseAutotuneCacheKey("
            "specialization_key=(), extra_results=(), "
            "kernel_source_hash='abc', hardware='B200', "
            "runtime_name='12.6', backend='triton', "
            "config_spec_hash='h1', extra_cache_key='')"
        )
        self.assertEqual(repr(k1), expected_k1_repr)
        # K=5 includes the field in the repr.
        self.assertIn("best_of_k=5", repr(k5))
        # Hashes differ structurally.
        self.assertNotEqual(k1.stable_hash(), k5.stable_hash())

    def test_cache_key_does_not_alias_extra_cache_key(self) -> None:
        """A K=1 key with ``extra_cache_key`` carrying a literal
        ``;best_of_k=5`` suffix must NOT collide with a K=5 key whose
        ``extra_cache_key`` is empty — i.e. the K must be a structural
        field, not folded into the string."""
        from helion.autotuner.base_cache import LooseAutotuneCacheKey

        k1_aliased = LooseAutotuneCacheKey(
            specialization_key=(),
            extra_results=(),
            kernel_source_hash="abc",
            hardware="B200",
            runtime_name="12.6",
            backend="triton",
            config_spec_hash="h1",
            extra_cache_key="foo;best_of_k=5",
            best_of_k=1,
        )
        k5 = LooseAutotuneCacheKey(
            specialization_key=(),
            extra_results=(),
            kernel_source_hash="abc",
            hardware="B200",
            runtime_name="12.6",
            backend="triton",
            config_spec_hash="h1",
            extra_cache_key="foo",
            best_of_k=5,
        )
        self.assertNotEqual(k1_aliased.stable_hash(), k5.stable_hash())

    def test_best_of_k_gt_1_without_factory_raises(self) -> None:
        """The bare ``Cache(autotuner)`` constructor must reject K>1 at
        run time rather than silently fall back to single-trial."""
        from helion.autotuner.local_cache import LocalAutotuneCache

        @helion.kernel(autotune_best_of_k=3, autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)

        # Build a cache with NO autotuner_factory wired; this models the
        # external ``Cache(autotuner)`` constructor path.
        class _MinimalSearch:
            def __init__(self):
                self.kernel = bound
                self.settings = bound.settings
                self.args = args
                self.best_perf_so_far = math.inf
                self._skip_cache = False

                class _Log:
                    def __call__(self, *a, **kw):
                        pass

                    def reset(self):
                        pass

                    def warning(self, *a, **kw):
                        pass

                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                return None  # never reached: K>1 must raise before this

        cache = LocalAutotuneCache(_MinimalSearch())  # no autotuner_factory
        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=True),
            self.assertRaisesRegex(
                RuntimeError,
                r"autotune_best_of_k > 1 requires a registered _autotuner_factory",
            ),
        ):
            cache.autotune()

    def test_k_loop_runs_k_trials_with_deterministic_seeds(self) -> None:
        """The K-loop runs K trials with seeds ``base + i``."""
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        seeds_seen: list[int] = []
        # ``block_sizes`` must be powers of two; pick four distinct values.
        trial_configs = [Config(block_sizes=[16, 1 << (3 + i)]) for i in range(4)]
        # Low-water perfs (per-trial best_perf_so_far) and rebench perfs
        # disagree on the winner: low-water best is index 2, rebench
        # best is index 3.
        low_water_perfs = [3.0, 5.0, 1.0, 2.0]
        rebench_perfs = [3.0, 5.0, 8.0, 2.0]
        trial_idx = {"n": 0}

        class MockTrialSearch:
            def __init__(self, bound_kernel, args, **kwargs):
                self.kernel = bound_kernel
                self.settings = bound_kernel.settings
                self.args = args
                self.best_perf_so_far = math.inf
                self._skip_cache = False

                class _Log:
                    def __call__(self, *a, **kw):
                        pass

                    def reset(self):
                        pass

                    def warning(self, *a, **kw):
                        pass

                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                i = trial_idx["n"]
                seeds_seen.append(self.settings.autotune_random_seed)
                self.best_perf_so_far = low_water_perfs[i]
                cfg = trial_configs[i]
                trial_idx["n"] += 1
                return cfg

        def mock_autotuner_fn(bound_kernel, args, **kwargs):
            def factory():
                return MockTrialSearch(bound_kernel, args, **kwargs)

            return LocalAutotuneCache(factory(), autotuner_factory=factory)

        @helion.kernel(
            autotuner_fn=mock_autotuner_fn,
            autotune_best_of_k=4,
            autotune_random_seed=100,
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)

        # Patch the final-rebench step so we can return the desired perfs
        # without needing a real benchmark_provider.
        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=True),
            patch.object(
                LocalAutotuneCache,
                "_rebench_trial_configs",
                lambda self, configs: rebench_perfs,
            ),
        ):
            picked = bound.autotune(args)

        # K trials ran.
        self.assertEqual(trial_idx["n"], 4)
        # Deterministic seeds: base + i.
        self.assertEqual(seeds_seen, [100, 101, 102, 103])
        # Winner is picked by REBENCH (index 3), not by low-water (index 2).
        rebench_winner = rebench_perfs.index(min(rebench_perfs))
        low_water_winner = low_water_perfs.index(min(low_water_perfs))
        self.assertNotEqual(rebench_winner, low_water_winner)
        self.assertEqual(picked, trial_configs[rebench_winner])

    def test_k_loop_restores_autotuner_and_settings(self) -> None:
        """After the K-loop, ``cache.autotuner`` must equal the original
        instance, and the mutated settings must be restored to base."""
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        cfg = Config(block_sizes=[16, 32])

        class MockSearch:
            def __init__(self, bound_kernel, args, **kwargs):
                self.kernel = bound_kernel
                self.settings = bound_kernel.settings
                self.args = args
                self.best_perf_so_far = math.inf
                self._skip_cache = False

                class _Log:
                    def __call__(self, *a, **kw):
                        pass

                    def reset(self):
                        pass

                    def warning(self, *a, **kw):
                        pass

                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                # Simulate the adaptive-timeout mutation that the real
                # BaseSearch does inside _prepare()/set_adaptive_compile_timeout.
                self.settings.autotune_compile_timeout = 5
                self.best_perf_so_far = 1.0
                return cfg

        original_autotuner_ref = {"r": None}

        def mock_autotuner_fn(bound_kernel, args, **kwargs):
            def factory():
                return MockSearch(bound_kernel, args, **kwargs)

            inner = factory()
            original_autotuner_ref["r"] = inner
            return LocalAutotuneCache(inner, autotuner_factory=factory)

        @helion.kernel(
            autotuner_fn=mock_autotuner_fn,
            autotune_best_of_k=3,
            autotune_random_seed=100,
            autotune_compile_timeout=60,
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)

        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=True),
            patch.object(
                LocalAutotuneCache,
                "_rebench_trial_configs",
                lambda self, configs: [1.0] * len(configs),
            ),
        ):
            captured_cache = bound.settings.autotuner_fn(bound, args)
            self.assertIsNotNone(original_autotuner_ref["r"])
            self.assertIs(captured_cache.autotuner, original_autotuner_ref["r"])
            captured_cache.autotune()

        # After the K-loop, the cache's ``autotuner`` reference must be
        # restored to the original instance (no leaked trial swap).
        self.assertIs(captured_cache.autotuner, original_autotuner_ref["r"])
        # And the settings the autotuner mutated must be restored to base.
        self.assertEqual(bound.settings.autotune_compile_timeout, 60)
        self.assertEqual(bound.settings.autotune_random_seed, 100)

    def test_k_one_falls_through_to_single_trial(self) -> None:
        """With ``autotune_best_of_k=1`` the K-loop must not run; the
        cache calls the autotuner exactly once and returns its config.
        """
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        call_count = {"n": 0}
        only_config = Config(block_sizes=[16, 32])

        class _Log:
            def __call__(self, *a, **kw):
                pass

            def reset(self):
                pass

            def warning(self, *a, **kw):
                pass

        class SingleTrialSearch:
            def __init__(self, bound_kernel, args, **kwargs):
                self.kernel = bound_kernel
                self.settings = bound_kernel.settings
                self.args = args
                self.best_perf_so_far = 1.0
                self._skip_cache = False
                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                call_count["n"] += 1
                return only_config

        def mock_autotuner_fn(bound_kernel, args, **kwargs):
            def factory():
                return SingleTrialSearch(bound_kernel, args, **kwargs)

            return LocalAutotuneCache(factory(), autotuner_factory=factory)

        @helion.kernel(
            autotuner_fn=mock_autotuner_fn,
            autotune_best_of_k=1,
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)
        with patch("helion.autotuner.base_cache.should_skip_cache", return_value=True):
            picked = bound.autotune(args)
        self.assertEqual(call_count["n"], 1)
        self.assertEqual(picked, only_config)


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
@onlyBackends(["triton"])
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
            autotune_config_filter=reject_small_blocks
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


class TestFiniteSearchWarmStart(TestCase):
    """Tests for helion.from_cache() — a CachedFiniteSearch autotuner_fn."""

    def _make_kernel_and_args(self):
        @helion.kernel(autotune_log_level=0)
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

    @staticmethod
    def _fake_cache_random(count):
        """Patch _find_similar_cached_configs to return `count` distinct backend-agnostic flats."""
        from helion.autotuner.local_cache import SavedBestConfig

        def fake_find(search_self, max_configs):
            distinct: list[tuple] = []
            for _ in range(20):
                if len(distinct) >= count:
                    break
                batch = search_self.config_gen.random_population_flat(
                    max(count * 3, 10)
                )
                for flat in batch:
                    t = tuple(flat)
                    if t not in distinct:
                        distinct.append(t)
                        if len(distinct) >= count:
                            break
            out = []
            for flat in distinct[:max_configs]:
                out.append(
                    SavedBestConfig(
                        hardware="x",
                        specialization_key="x",
                        config=search_self.config_gen.unflatten(list(flat)),
                        config_spec_hash="x",
                        flat_config=flat,
                    )
                )
            return out

        return fake_find

    def test_from_cache_factory(self):
        """helion.from_cache() returns a callable that creates a CachedFiniteSearch."""

        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        fn = helion.from_cache()
        self.assertTrue(callable(fn))
        with (
            patch.object(BaseSearch, "_find_similar_cached_configs", return_value=[]),
            self.assertRaises(exc.NotEnoughConfigs),
        ):
            fn(bound, args)

    def test_from_cache_empty_cache_raises(self):
        """CachedFiniteSearch with empty cache and no explicit configs raises NotEnoughConfigs."""
        from helion.autotuner.finite_search import CachedFiniteSearch

        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        with (
            patch.object(BaseSearch, "_find_similar_cached_configs", return_value=[]),
            self.assertRaises(exc.NotEnoughConfigs),
        ):
            CachedFiniteSearch(bound, args)

    def test_from_cache_prepends_cached(self):
        """Cached configs appear before explicit configs in CachedFiniteSearch.configs."""
        from helion.autotuner.finite_search import CachedFiniteSearch

        cfg1 = helion.Config(block_sizes=[16])
        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        fake_fn = self._fake_cache_random(2)
        fake_sizes: list[int] = []

        def spy(search_self, max_configs):
            result = fake_fn(search_self, max_configs)
            fake_sizes.append(len(result))
            return result

        with patch.object(BaseSearch, "_find_similar_cached_configs", spy):
            search = CachedFiniteSearch(bound, args, configs=[cfg1])
        self.assertEqual(len(search.configs), fake_sizes[0] + 1)
        self.assertEqual(search.configs[-1], cfg1)

    def test_from_cache_respects_max_parameter(self):
        """from_cache(max_configs=N) caps the number of cached configs."""
        from helion.autotuner.finite_search import CachedFiniteSearch

        cfg1 = helion.Config(block_sizes=[16])
        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        observed_caps: list[int] = []
        fake_sizes: list[int] = []
        fake_fn = self._fake_cache_random(5)

        def spy(search_self, max_configs):
            observed_caps.append(max_configs)
            result = fake_fn(search_self, max_configs)
            fake_sizes.append(len(result))
            return result

        with patch.object(BaseSearch, "_find_similar_cached_configs", spy):
            search = CachedFiniteSearch(bound, args, configs=[cfg1], max_configs=2)
        self.assertEqual(observed_caps, [2])
        self.assertLessEqual(fake_sizes[0], 2)
        self.assertEqual(len(search.configs), fake_sizes[0] + 1)

    def test_from_cache_uses_default_cap_from_settings(self):
        """Without max_configs, the cap falls back to autotune_best_available_max_configs."""
        from helion.autotuner.finite_search import CachedFiniteSearch

        cfg1 = helion.Config(block_sizes=[16])
        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        observed_caps: list[int] = []
        fake_sizes: list[int] = []
        fake_fn = self._fake_cache_random(5)

        def spy(search_self, max_configs):
            observed_caps.append(max_configs)
            result = fake_fn(search_self, max_configs)
            fake_sizes.append(len(result))
            return result

        with patch.object(BaseSearch, "_find_similar_cached_configs", spy):
            search = CachedFiniteSearch(bound, args, configs=[cfg1])
        cap_in_effect = search.settings.autotune_best_available_max_configs
        self.assertEqual(observed_caps, [cap_in_effect])
        self.assertLessEqual(fake_sizes[0], cap_in_effect)
        self.assertEqual(len(search.configs), fake_sizes[0] + 1)

    def test_kernel_autotuner_fn_accepts_from_cache(self):
        """@helion.kernel(autotuner_fn=helion.from_cache()) stores the callable in settings."""
        fn = helion.from_cache()

        @helion.kernel(autotuner_fn=fn, autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )
        bound = add.bind(args)
        self.assertIs(bound.settings.autotuner_fn, fn)


@onlyBackends(["triton", "cute"])
class TestAutotuneBudget(TestCase):
    def _make_search(self, settings: Settings) -> BaseSearch:
        search = BaseSearch.__new__(BaseSearch)
        search.settings = settings
        search.kernel = SimpleNamespace(
            format_kernel_decorator=lambda config, s: "decorator",
            to_triton_code=lambda config: "code",
            maybe_log_repro=lambda log_func, args, config=None: None,
            supports_subprocess_benchmark=lambda: False,
        )
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            default_config=lambda: helion.Config(block_sizes=[1])
        )
        search._benchmark_provider_cls = LocalBenchmarkProvider
        search.best_perf_so_far = float("inf")
        search._prepared = False
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
        from helion._compiler.backend import _CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend

        settings = Settings(
            autotune_budget_seconds=None,
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

    def test_cute_backend_restores_default_autotune_budget_on_error(self) -> None:
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend

        settings = Settings(
            autotune_budget_seconds=None,
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

    def test_budget_unset_when_prepare_not_called(self) -> None:
        settings = Settings(
            autotune_budget_seconds=1,
            autotune_log_level=logging.CRITICAL,
        )
        search = BaseSearch.__new__(BaseSearch)
        search.settings = settings
        search._autotune_budget_start = None
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
        provider.config_spec = SimpleNamespace()
        provider.args = ()
        provider.log = AutotuningLogger(provider.settings)
        provider._autotune_metrics = SimpleNamespace(
            num_configs_tested=0,
            num_compile_failures=0,
            num_accuracy_failures=0,
            num_generations=0,
            kernel_source="",
        )
        provider.mutated_arg_indices = ()
        provider._benchmark_worker = None
        provider._precompile_args_path = None
        provider._precompile_tmpdir = None
        return provider

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

    def test_cute_wall_clock_benchmark_uses_subprocess_worker(self) -> None:
        from helion._compiler.backend import CuteBackend

        provider = self._make_stub_provider()
        provider.kernel.supports_subprocess_benchmark = lambda: True
        provider.config_spec = SimpleNamespace(backend=CuteBackend())

        self.assertTrue(provider._subprocess_benchmark_enabled())
        self.assertTrue(provider._subprocess_benchmark_uses_wall_clock())

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


class TestConfigValuePriors(TestCase):
    """Backend-supplied per-key priors bias the random half of the initial
    population (config_generation), while the other half stays uniform."""

    def _add_config_gen(self) -> tuple[ConfigGeneration, object]:
        @helion.kernel(autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn(256, 256, device=DEVICE)
        bound = add.bind((a, a))
        return ConfigGeneration(bound.config_spec), bound

    @staticmethod
    def _forceable_slot(gen: ConfigGeneration) -> tuple[str, int, object]:
        """Pick a config key the live kernel exposes and a value to force it to.

        Backends expose different keys (e.g. the default backend has a scalar
        ``num_warps`` slot; the cute backend exposes only sequence keys such as
        ``block_sizes``), so the priors tests select a target dynamically rather
        than hard-coding ``num_warps``. Returns ``(key, flat_index, value)``
        where ``value`` is representable by the slot's fragment.
        """
        # Prefer a scalar ``num_warps`` slot when present: a power-of-two value
        # in its range that won't be altered by repair logic.
        for key, (indices, is_sequence) in gen._key_to_flat_indices.items():
            if key == "num_warps" and not is_sequence and indices:
                return key, indices[0], 4
        # Otherwise use the first block-size slot, which every backend exposes.
        block_key, (block_indices, _is_seq) = next(
            (k, v) for k, v in gen._key_to_flat_indices.items() if k == "block_sizes"
        )
        flat_idx = block_indices[0]
        fragment = gen.flat_spec[flat_idx]
        # Use the fragment's own default: guaranteed representable and stable.
        return block_key, flat_idx, fragment.default()

    def test_no_priors_falls_through_to_uniform(self) -> None:
        gen, _ = self._add_config_gen()
        # With no priors, biased sampling is exactly uniform sampling and the
        # population fill is unchanged. Control the priors explicitly rather than
        # relying on the ambient backend default (the cute backend, for example,
        # supplies non-empty priors).
        gen._config_value_priors = {}
        self.assertEqual(gen._config_value_priors, {})
        flat = gen.biased_random_flat()
        self.assertEqual(len(flat), len(gen.flat_spec))

    def test_prior_forces_value(self) -> None:
        from helion.autotuner.config_priors import weighted_choice

        gen, _ = self._add_config_gen()
        key, flat_idx, value = self._forceable_slot(gen)
        # Control the priors directly on the instance so the test is independent
        # of whatever priors the active backend supplies.
        gen._config_value_priors = {key: weighted_choice({value: 1.0})}
        # Disable size repair/shrink so the forced value is observed verbatim.
        with (
            patch.object(gen, "shrink_config", lambda *a, **k: None),
            patch.object(gen, "_repair_cute_num_threads", lambda *a, **k: None),
        ):
            for _ in range(25):
                self.assertEqual(gen.biased_random_flat()[flat_idx], value)

    def test_population_fill_is_half_biased(self) -> None:
        from helion.autotuner.config_priors import weighted_choice

        gen, _ = self._add_config_gen()
        key, _flat_idx, value = self._forceable_slot(gen)
        gen._config_value_priors = {key: weighted_choice({value: 1.0})}
        with (
            # Suppress backend-supplied compiler seeds so the random-fill count
            # is deterministic across backends (cute seeds a few configs).
            patch.object(gen, "seed_flat_config_pairs", return_value=[]),
            patch.object(
                gen, "biased_random_flat", wraps=gen.biased_random_flat
            ) as biased,
            patch.object(gen, "random_flat", wraps=gen.random_flat) as uniform,
        ):
            # 1 default + 10 random fill slots (seeds suppressed above).
            gen.random_population_flat(11)
        self.assertEqual(biased.call_count, 5)
        self.assertEqual(uniform.call_count, 5)


if __name__ == "__main__":
    unittest.main()
