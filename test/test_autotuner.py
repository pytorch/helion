from __future__ import annotations

import collections
from contextlib import contextmanager
from contextlib import nullcontext
import csv
from itertools import count
import logging
import math
import multiprocessing as mp
import operator
import os
from pathlib import Path
import pickle
import random
import tempfile
from types import SimpleNamespace
from typing import Callable
from typing import Sequence
import unittest
from unittest import skip
from unittest.mock import patch

import pytest
import torch

import helion
from helion import _compat
from helion import exc
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import skipIfCpu
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion.autotuner import DESurrogateHybrid
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner import LFBOPatternSearch
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import PermutationFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import StrictLocalAutotuneCache
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.random_search import RandomSearch
import helion.language as hl
from helion.language import loops
from helion.runtime.settings import Settings

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"
examples_matmul = import_path(examples_dir / "matmul.py").matmul


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
        )
        search.args = args
        search.counters = collections.Counter()
        search.log = AutotuningLogger(settings)
        search._kernel_mutates_args = False
        search.best_perf_so_far = float("inf")
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        search._precompile_tmpdir = tempdir
        search._precompile_args_path = None
        search._precompile_result_counter = count()
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
                search.benchmark_function("cfg", bad_fn)

        assert "HELION_AUTOTUNE_IGNORE_ERRORS" in str(err.value)

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
                result = search.benchmark_function("cfg", bad_fn)

        self.assertEqual(result, float("inf"))
        warn.assert_not_called()

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
                "helion.autotuner.base_search.classify_triton_exception",
                return_value="raise",
            ),
        ):
            sync.return_value = None
            with pytest.raises(exc.TritonError) as err:
                search.benchmark_function("cfg", bad_fn)

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
                "helion.autotuner.base_search.classify_triton_exception",
                return_value="raise",
            ),
        ):
            sync.return_value = None
            with pytest.raises(exc.TritonError) as err:
                search.benchmark_function("cfg", bad_fn)

        # Verify the traceback was cleared
        assert err.value.__cause__.__traceback__ is None
        # Verify the exception chain is preserved even after __traceback__ = None
        assert err.value.__cause__ is original_exception
        assert str(original_exception) == "original error in except block"
        # Verify we can still get the error type and message
        assert type(err.value.__cause__).__name__ == "RuntimeError"

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
        self.assertEqual(
            rows[0],
            [
                "timestamp_s",
                "config_index",
                "generation",
                "status",
                "perf_ms",
                "compile_time_s",
                "config",
            ],
        )
        self.assertEqual(rows[1][1], "1")
        self.assertEqual(rows[1][2], "5")
        self.assertEqual(rows[1][3], "ok")
        self.assertEqual(rows[1][4], "1.234000")
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
                "helion.autotuner.base_search.make_precompiler",
                side_effect=fake_make_precompiler,
            ),
            patch("torch.cuda._lazy_init", side_effect=fake_lazy_init),
        ):
            future = search.start_precompile_and_check_for_hangs(
                "cfg", fake_compiled_fn
            )
            self.assertTrue(future())

        self.assertEqual(set(lazy_calls), {parent_pid})

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
        statuses = [row[3] for row in rows[1:]]  # skip header
        started_count = sum(1 for s in statuses if s == "started")
        completed_count = sum(1 for s in statuses if s in ("ok", "error", "timeout"))
        self.assertGreater(started_count, 0, "Should log started entries")
        self.assertEqual(
            started_count, completed_count, "Each started should have completion"
        )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    @skipIfCpu("fails on Triton CPU backend")
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


class TestAutotuner(RefEagerTestDisabled, TestCase):
    def setUp(self):
        super().setUp()
        random.seed(112)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @skipIfRocm("failure on rocm")
    def test_config_fragment0(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        spec = examples_matmul.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch(
        "helion.autotuner.config_generation.warps_to_threads",
        lambda num_warps: num_warps * 32,
    )
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
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
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
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
        indexing_choices = spec._valid_indexing_types()
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

    @skipIfCpu("fails on Triton CPU backend")
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

    @skipIfRocm("too slow on rocm")
    @skipIfCpu("TritonError: Error from Triton code")
    def test_random_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = examples_matmul.bind(args)
        bound_kernel.settings.autotune_precompile = None
        random.seed(123)
        best = RandomSearch(bound_kernel, args, 20).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    @skipIfRocm("too slow on rocm")
    @skip("too slow")
    def test_differential_evolution_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = examples_matmul.bind(args)
        random.seed(123)
        best = DifferentialEvolutionSearch(
            bound_kernel, args, 5, max_generations=3
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    @skipIfRocm("too slow on rocm")
    @skip("too slow")
    def test_de_surrogate_hybrid(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = examples_matmul.bind(args)
        random.seed(123)
        best = DESurrogateHybrid(
            bound_kernel, args, population_size=5, max_generations=3
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    @skipIfRocm("too slow on rocm")
    @skipIfCpu("fails on Triton CPU backend")
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

    @skipIfRocm("too slow on rocm")
    @skipIfCpu("fails on Triton CPU backend")
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
        )

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
        )

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

    @skipIfRocm("too slow on rocm")
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

    @skipIfCpu("fails on Triton CPU backend")
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
                if mode == "fork":
                    start_cm = patch.object(
                        search,
                        "start_precompile_and_check_for_hangs",
                        side_effect=lambda config,
                        fn: base_search_module.PrecompileFuture.skip(
                            search, config, True
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

                    _, bad_time = search.benchmark(bad_config)
                    assert math.isinf(bad_time)
                    self.assertEqual(search.counters.get("accuracy_mismatch", 0), 1)
                    search.counters["accuracy_mismatch"] = 0

                    _, good_time = search.benchmark(good_config)
                    assert not math.isinf(good_time)
                    self.assertEqual(search.counters.get("accuracy_mismatch", 0), 0)
                    search.counters["accuracy_mismatch"] = 0

                    best = search.autotune()
                    self.assertEqual(best, good_config)
                    self.assertEqual(search.counters.get("accuracy_mismatch", 0), 1)

        run_mode("fork", expect_error=False)
        run_mode("spawn", expect_error=True)

    @skipIfCpu("fails on Triton CPU backend")
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
                if mode == "fork":
                    start_cm = patch.object(
                        search,
                        "start_precompile_and_check_for_hangs",
                        side_effect=lambda config,
                        fn: base_search_module.PrecompileFuture.skip(
                            search, config, True
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

                    _, bad_time = search.benchmark(bad_config)
                    assert math.isinf(bad_time)
                    self.assertEqual(search.counters.get("accuracy_mismatch", 0), 1)
                    search.counters["accuracy_mismatch"] = 0

                    _, good_time = search.benchmark(good_config)
                    assert not math.isinf(good_time)
                    self.assertEqual(search.counters.get("accuracy_mismatch", 0), 0)
                    search.counters["accuracy_mismatch"] = 0

                    best = search.autotune()
                    self.assertEqual(best, good_config)
                    self.assertGreaterEqual(
                        search.counters.get("accuracy_mismatch", 0), 1
                    )

        run_mode("fork", expect_error=False)
        run_mode("spawn", expect_error=True)

    @skipIfCpu("fails on Triton CPU backend")
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

    @skipIfCpu("fails on Triton CPU backend")
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
            with patch.object(
                search,
                "start_precompile_and_check_for_hangs",
                side_effect=lambda config, fn: base_search_module.PrecompileFuture.skip(
                    search, config, True
                ),
            ):
                # Bad config should be filtered out by accuracy check
                _, bad_time = search.benchmark(bad_config)
                self.assertTrue(math.isinf(bad_time))
                self.assertEqual(search.counters.get("accuracy_mismatch", 0), 1)

                # Good config should pass accuracy check
                search.counters["accuracy_mismatch"] = 0
                _, good_time = search.benchmark(good_config)
                self.assertFalse(math.isinf(good_time))
                self.assertEqual(search.counters.get("accuracy_mismatch", 0), 0)

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

    @skipIfCpu("fails on Triton CPU backend")
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
                    search.counters["accuracy_mismatch"], len(search.configs)
                )
            else:
                winner = search.autotune()
                self.assertIn(winner, (cfg1, cfg2))
                self.assertEqual(search.counters["accuracy_mismatch"], 0)

    @skipIfCpu("fails on Triton CPU backend")
    @skipIfRocm("fp8 dtypes not supported on ROCm")
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        "FP8 requires GPU with compute capability >= 9.0 (e.g., H100)",
    )
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

        # Verify that effective tolerances were set to 0.0 automatically
        self.assertEqual(
            search._effective_atol,
            0.0,
            f"Expected automatic atol=0.0 for fp8, got {search._effective_atol}",
        )
        self.assertEqual(
            search._effective_rtol,
            0.0,
            f"Expected automatic rtol=0.0 for fp8, got {search._effective_rtol}",
        )

        # Should successfully autotune without error
        winner = search.autotune()
        self.assertIn(winner, (cfg1, cfg2))
        self.assertEqual(search.counters["accuracy_mismatch"], 0)

    @skipIfCpu("fails on Triton CPU backend")
    @skipIfRocm("fp8 dtypes not supported on ROCm")
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        "FP8 requires GPU with compute capability >= 9.0 (e.g., H100)",
    )
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

        # Should respect user's explicit tolerances, not override to 0.0
        self.assertEqual(search._effective_atol, 1e-5)
        self.assertEqual(search._effective_rtol, 1e-5)

    @skipIfCpu("fails on Triton CPU backend")
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

    @skipIfCpu("fails on Triton CPU backend")
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

        # Test 1: Default quick mode values from effort profile
        with patch.dict(os.environ, {"HELION_AUTOTUNER": "LFBOPatternSearch"}):

            @helion.kernel(autotune_effort="quick")
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            bound = add.bind(args)
            autotuner = bound.settings.autotuner_fn(bound, args)
            lfbo_pattern = autotuner.autotuner
            self.assertIsInstance(lfbo_pattern, LFBOPatternSearch)
            # Use exact values from quick profile
            self.assertEqual(lfbo_pattern.initial_population, expected_initial_pop)
            self.assertEqual(lfbo_pattern.copies, expected_copies)
            self.assertEqual(lfbo_pattern.max_generations, expected_max_gen)

        # Test 2: HELION_AUTOTUNE_MAX_GENERATIONS overrides effort profile
        override_max_gen = 100
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "LFBOPatternSearch",
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
            lfbo_pattern = autotuner.autotuner
            self.assertIsInstance(lfbo_pattern, LFBOPatternSearch)
            # initial_population and copies from profile, but max_generations from env var
            self.assertEqual(lfbo_pattern.initial_population, expected_initial_pop)
            self.assertEqual(lfbo_pattern.copies, expected_copies)
            self.assertEqual(lfbo_pattern.max_generations, override_max_gen)

        # Test 3: Explicit constructor values take highest priority
        explicit_initial_pop = 500
        explicit_copies = 300
        explicit_max_gen = 150

        bound = add.bind(args)
        lfbo_pattern = LFBOPatternSearch(
            bound,
            args,
            initial_population=explicit_initial_pop,
            copies=explicit_copies,
            max_generations=explicit_max_gen,
        )
        # All values from explicit constructor args
        self.assertEqual(lfbo_pattern.initial_population, explicit_initial_pop)
        self.assertEqual(lfbo_pattern.copies, explicit_copies)
        self.assertEqual(lfbo_pattern.max_generations, explicit_max_gen)

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

        # Test EnumFragment (one-hot encoding)
        enum_frag = EnumFragment(choices=("a", "b", "c"))
        self.assertEqual(enum_frag.dim(), 3)
        self.assertEqual(enum_frag.encode("a"), [1.0, 0.0, 0.0])
        self.assertEqual(enum_frag.encode("b"), [0.0, 1.0, 0.0])

        # Test PermutationFragment
        perm_frag = PermutationFragment(length=3)
        self.assertEqual(perm_frag.dim(), 3)
        encoded = perm_frag.encode([0, 1, 2])
        self.assertEqual(encoded, [0, 1, 2])

        # Test ListOf with BooleanFragment
        list_frag = ListOf(inner=BooleanFragment(), length=3)
        self.assertEqual(list_frag.dim(), 3)
        self.assertEqual(list_frag.encode([True, False, True]), [1.0, 0.0, 1.0])

        # Test encode_dim consistency
        for fragment, value in [
            (BooleanFragment(), True),
            (IntegerFragment(1, 10, 5), 5),
            (PowerOfTwoFragment(2, 128, 8), 16),
            (EnumFragment(choices=("a", "b")), "b"),
        ]:
            dim = fragment.dim()
            encoded = fragment.encode(value)
            self.assertEqual(len(encoded), dim)

    @skipIfCpu("fails on Triton CPU backend")
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

        search.best_perf_so_far = 1.0

        # Call rebenchmark directly
        search.rebenchmark([member1, member2])

        # Verify custom benchmark function was called
        self.assertGreater(
            len(benchmark_calls), 0, "Custom benchmark function should be called"
        )
        # Should have been called with 2 functions
        self.assertEqual(benchmark_calls[0][0], 2)


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

    @skipIfRocm("accuracy difference")
    @skipIfCpu("fails on Triton CPU backend")
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

    @skipIfRocm("accuracy difference")
    @skipIfCpu("fails on Triton CPU backend")
    def test_autotune_random_seed_from_settings(self) -> None:
        # same autotune_random_seed setting -> same random sample
        first = self._autotune_and_record(autotune_random_seed=4242)
        second = self._autotune_and_record(autotune_random_seed=4242)
        self.assertEqual(first, second)

        # different autotune_random_seed settings -> different random samples
        first = self._autotune_and_record(autotune_random_seed=101)
        second = self._autotune_and_record(autotune_random_seed=102)
        self.assertNotEqual(first, second)


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


if __name__ == "__main__":
    unittest.main()
