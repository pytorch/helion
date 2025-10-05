from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import skip
from unittest.mock import patch

import pytest
import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import skipIfRocm
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner import PatternSearch
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.random_search import RandomSearch
import helion.language as hl
from helion.language import loops

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

    def test_config_generation_overrides(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        overrides = {"indexing": "block_ptr"}
        gen = ConfigGeneration(spec, overrides=overrides)

        flat = gen.default_flat()
        config = gen.unflatten([*flat])
        self.assertEqual(config["indexing"], "block_ptr")
        configs = [gen.unflatten(gen.random_flat()) for _ in range(3)]
        self.assertEqual({cfg["indexing"] for cfg in configs}, {"block_ptr"})
        indexing_choices = spec._valid_indexing_types()
        indexing_index = next(
            i
            for i, fragment in enumerate(gen.flat_spec)
            if isinstance(fragment, EnumFragment)
            and fragment.choices == indexing_choices
        )
        mutated = gen.random_flat()
        mutated[indexing_index] = "pointer"
        new_config = gen.unflatten(mutated)
        self.assertEqual(new_config["indexing"], "block_ptr")
        self.assertEqual(mutated[indexing_index], "pointer")

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
    def test_random_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = examples_matmul.bind(args)
        random.seed(123)
        best = RandomSearch(bound_kernel, args, 20).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    @skipIfRocm("too slow on rocm")
    def test_differential_evolution_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = examples_matmul.bind(args)
        random.seed(123)
        best = DifferentialEvolutionSearch(
            bound_kernel, args, 5, num_generations=3
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

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

    def test_accuracy_check_filters_bad_config_wrong_output(self) -> None:
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(configs=[bad_config, good_config], autotune_log_level=0)
        def add_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(b.size()):
                b[tile] = a[tile] + b[tile]
            return b

        a = torch.randn([32], device=DEVICE)
        b = torch.randn([32], device=DEVICE)
        bound_kernel = add_inplace.bind((a, b))

        original_compile = bound_kernel.compile_config

        def make_bad_config_produce_wrong_output(
            config: helion.Config, *, allow_print: bool = True
        ):
            fn = original_compile(config, allow_print=allow_print)
            if config == bad_config:
                return lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs) + 1
            return fn

        with patch.object(
            bound_kernel,
            "compile_config",
            side_effect=make_bad_config_produce_wrong_output,
        ):
            search = FiniteSearch(
                bound_kernel, (a, b), configs=[bad_config, good_config]
            )
            _, bad_time = search.benchmark(bad_config)
            assert math.isinf(bad_time)
            self.assertEqual(search.counters.get("accuracy_mismatch", 0), 1)
            search.counters["accuracy_mismatch"] = 0  # reset counter

            _, good_time = search.benchmark(good_config)
            assert not math.isinf(good_time)
            self.assertEqual(search.counters.get("accuracy_mismatch", 0), 0)
            search.counters["accuracy_mismatch"] = 0  # reset counter

            best = search._autotune()
            self.assertEqual(best, good_config)
            self.assertEqual(search.counters.get("accuracy_mismatch", 0), 1)

    def test_accuracy_check_filters_bad_config_wrong_arg_mutation(self) -> None:
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(configs=[bad_config, good_config], autotune_log_level=0)
        def add_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(b.size()):
                b[tile] = a[tile] + b[tile]
            return b

        a = torch.randn([32], device=DEVICE)
        b = torch.randn([32], device=DEVICE)
        bound_kernel = add_inplace.bind((a, b))

        original_compile = bound_kernel.compile_config

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

        with patch.object(
            bound_kernel,
            "compile_config",
            side_effect=make_bad_config_produce_wrong_input_arg_mutation,
        ):
            search = FiniteSearch(
                bound_kernel, (a, b), configs=[bad_config, good_config]
            )
            _, bad_time = search.benchmark(bad_config)
            assert math.isinf(bad_time)
            self.assertEqual(search.counters.get("accuracy_mismatch", 0), 1)
            search.counters["accuracy_mismatch"] = 0  # reset counter

            _, good_time = search.benchmark(good_config)
            assert not math.isinf(good_time)
            self.assertEqual(search.counters.get("accuracy_mismatch", 0), 0)
            search.counters["accuracy_mismatch"] = 0  # reset counter

            best = search._autotune()
            self.assertEqual(best, good_config)
            self.assertGreaterEqual(search.counters.get("accuracy_mismatch", 0), 1)

    def test_use_default_config(self):
        @helion.kernel(use_default_config=True)
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

    def test_autotuner_disabled(self):
        @helion.kernel(use_default_config=False)
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
    def test_autotune_random_seed_from_settings(self) -> None:
        # same autotune_random_seed setting -> same random sample
        first = self._autotune_and_record(autotune_random_seed=4242)
        second = self._autotune_and_record(autotune_random_seed=4242)
        self.assertEqual(first, second)

        # different autotune_random_seed settings -> different random samples
        first = self._autotune_and_record(autotune_random_seed=101)
        second = self._autotune_and_record(autotune_random_seed=102)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
