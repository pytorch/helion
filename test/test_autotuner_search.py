from __future__ import annotations

import contextlib
import functools
import logging
import math
import operator
import os
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast
import unittest
from unittest import skip
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test._autotuner_test_utils import AutotunerTestCase
from test._autotuner_test_utils import _get_examples_matmul
from test._autotuner_test_utils import basic_kernels
from test._autotuner_test_utils import without_env_var

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfXPU
from helion._testing.search_harness import mock_kernel
from helion.autotuner import DESurrogateHybrid
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner import LFBOPatternSearch
from helion.autotuner import LFBOTreeSearch
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import BenchmarkResult
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmarking import KERNEL_TRACE_TICK_US
from helion.autotuner.benchmarking import KernelTraceUnreadable
from helion.autotuner.benchmarking import do_bench
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.handoff import _active_handoff
from helion.autotuner.local_cache import _search_policy_hash
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import random_fallback_population_target
from helion.autotuner.pattern_search import rereading_spread
from helion.autotuner.pattern_search import trace_tie_tolerance
from helion.autotuner.random_search import RandomSearch
import helion.language as hl
from helion.runtime.config_cache import ConfigCache
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion.autotuner.base_search import _AutotunableKernel


@onlyBackends(["triton", "cute"])
class TestAutotunerSearch(AutotunerTestCase):
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
            # Config selection is the behavior under test; skip the benchmark
            # worker subprocess (seconds of startup overhead). The worker path
            # keeps coverage via test_autotune_noncontiguous_arg/broadcast_arg.
            autotune_benchmark_subprocess=False,
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

        # The compile-failure skip logic under test runs at the compile stage;
        # skip the benchmark worker subprocess (seconds of startup overhead).
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
            autotune_benchmark_subprocess=False,
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
        # The settings belong to the matmul example's kernel, which other tests
        # share: restore them afterwards.  Smoke test of the RandomSearch loop
        # itself; skip the benchmark worker subprocess (seconds of startup
        # overhead).
        with (
            patch.object(bound_kernel.settings, "autotune_precompile", None),
            patch.object(bound_kernel.settings, "autotune_benchmark_subprocess", False),
        ):
            random.seed(123)
            best = RandomSearch(bound_kernel, args, 5).autotune()
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

    @pytest.mark.slow
    @onlyBackends(["cute"])
    def test_short_kernel_search_traces_its_candidates_and_stops_on_ties(self):
        """A real pattern search of a kernel of a few microseconds on the
        backend that names a short-kernel bound: after its initial
        population the search settles on the trace timer, so its best
        reading leaves out the launch latency that CUDA events read, every
        path decision is taken with a tolerance of at least a tick and a
        half of the trace's clock, and the path stops well before its
        generation limit instead of walking on one-tick flips."""
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)
        random.seed(123)
        max_generations = 8
        search = PatternSearch(
            bound_kernel,
            args,
            initial_population=10,
            max_generations=max_generations,
            copies=1,
            num_neighbors_cap=16,
        )
        decisions: list[tuple[bool, float, float]] = []
        check = search._check_early_stopping

        def recorded(best: PopulationMember, current: PopulationMember) -> bool:
            stopped = check(best, current)
            decisions.append(
                (stopped, search._reading_tolerance(current.perf), current.perf)
            )
            return stopped

        search._check_early_stopping = recorded
        # The same search drops source aliases: the kernels of every
        # verification (the re-timings that the search traces).
        retimed: list[list[int]] = []
        passes: list[object] = []
        rebenchmark = search.rebenchmark

        def recorded_rebenchmark(
            members: list[PopulationMember], **kwargs: object
        ) -> None:
            retimed.append([id(member.fn) for member in members])
            passes.append(kwargs.get("desc"))
            rebenchmark(members, **kwargs)

        search.rebenchmark = recorded_rebenchmark
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS": "0"}, clear=False
        ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            best = search.autotune()
        self.assertTrue(search.traces_candidates)
        self.assertTrue(search._collapses_source_aliases())
        self.assertTrue(retimed)
        # the initial population, read with CUDA events, was read again on
        # the trace before the search verified or compared anything
        self.assertEqual(passes[0], "Reading the initial population on the trace")
        self.assertEqual(passes[1], "Verifying initial results")
        # no verification timed, or traced, one kernel under two configs
        for kernels in retimed:
            self.assertEqual(len(kernels), len(set(kernels)))
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), sum(args), rtol=1e-2, atol=1e-1)
        # CUDA events around one flushed launch, as a search without the
        # trace reads this config: the kernel plus 4-5 us of launch latency
        events_us = 1000.0 * do_bench(  # pyrefly: ignore[unsupported-operation]
            functools.partial(fn, *args), return_mode="median", warmup=1, rep=20
        )
        self.assertLess(search.best_perf_so_far * 1000.0 + 1.0, events_us)
        self.assertTrue(decisions)
        for _stopped, tolerance, perf in decisions:
            self.assertGreaterEqual(tolerance, 0.005)
            self.assertGreaterEqual(
                tolerance, 1.5 * KERNEL_TRACE_TICK_US / (perf * 1000.0) - 1e-12
            )
        # the last decision ended the path, before the generation limit
        self.assertTrue(decisions[-1][0])
        self.assertLess(len(decisions), max_generations)

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
        # Each position yields 4 neighbors (3,4,6,7) = 8 single-position
        # changes, plus uniform lists (all elements set to the same value
        # near the inner default): [3,3], [4,4], [6,6], [7,7].
        single = [n for n in neighbors if n.count(5) == 1]
        uniform = [n for n in neighbors if n.count(5) == 0]
        self.assertEqual(len(single), 8)
        self.assertEqual(sorted(uniform), [[3, 3], [4, 4], [6, 6], [7, 7]])
        self.assertEqual(len(neighbors), 12)

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
            conditional_neighbor=lambda base, requested, **kwargs: requested,
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
            conditional_neighbor=lambda base, requested, **kwargs: requested,
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
        key, value, _ = self._scalar_override()
        gen = ConfigGeneration(spec, overrides={key: value})

        # Find the overridden key's flat index
        (key_idx,), _ = gen._key_to_flat_indices[key]
        self.assertIn(key_idx, gen.overridden_flat_indices)

        base = gen.default_flat()
        a = gen.random_flat()
        b = gen.random_flat()
        c = gen.random_flat()

        # Run many mutations — overridden index should never change
        for _ in range(50):
            result = gen.differential_mutation(base, a, b, c, crossover_rate=0.9)
            self.assertEqual(result[key_idx], base[key_idx])

    def test_initial_population_refills_backend_rejections(self):
        configs = {value: helion.Config(block_sizes=[value]) for value in range(1, 7)}
        backend = SimpleNamespace(
            autotune_config_is_viable=lambda _spec, config: (
                config.config["block_sizes"][0] % 2 == 0
            )
        )
        config_gen = SimpleNamespace(
            config_spec=SimpleNamespace(backend=backend),
            invalid_config_count=0,
            canonicalize_flat=lambda flat: (flat, configs[flat[0]]),
            random_flat=Mock(side_effect=([3], [4], [5], [6])),
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.config_spec = config_gen.config_spec
        search.config_gen = config_gen
        search.log = Mock()

        population = search._pad_initial_population_with_unique_random(
            [[1], [2]], target=3
        )

        self.assertEqual(population, [[2], [4], [6]])
        self.assertEqual(config_gen.random_flat.call_count, 4)

    def test_population_adopts_config_filter_replacement(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)
        search = PatternSearch(bound_kernel, args, initial_population=1)
        search.benchmark_provider = SimpleNamespace(
            take_effective_source_repairs=dict, settles_source_aliases=lambda: False
        )
        member = search.make_unbenchmarked(search.config_gen.default_flat())
        assert member is not None
        key, value, _ = self._scalar_override()
        replacement = helion.Config.from_dict({**member.config.config, key: value})
        result = SimpleNamespace(
            config=replacement,
            perf=float("inf"),
            fn=lambda: None,
            status="error",
            compile_time=None,
        )

        with patch.object(search, "benchmark_batch", return_value=[result]):
            search.benchmark_population([member])

        self.assertIs(member.config, replacement)
        self.assertEqual(member.flat_values, search.config_gen.flatten(replacement))
        (key_idx,), _ = search.config_gen._key_to_flat_indices[key]
        self.assertEqual(member.flat_values[key_idx], value)
        search.config_gen.encode_config(member.flat_values)

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
            conditional_neighbor=lambda base, requested, **kwargs: requested,
        )
        search.num_neighbors_cap = -1

        base = [32, 64, 4, "b", True]
        neighbors = search._generate_neighbors(base)

        # No neighbor should change indices 1 or 2
        for flat in neighbors:
            self.assertEqual(flat[1], 64)
            self.assertEqual(flat[2], 4)

    def test_lfbo_disable_early_stopping_exhausts_generation_budget(self):
        family_key = "cute_flash_pipeline_family"

        def run_path(
            *, constraints: tuple[tuple[str, object], ...], disable: bool
        ) -> tuple[int, int]:
            def member(index: int) -> PopulationMember:
                return PopulationMember(
                    fn=lambda: None,
                    perfs=[1.0 if index == 0 else 2.0],
                    flat_values=[index, "fa4"],
                    config=helion.Config(
                        block_sizes=[1, 128, 128],
                        cute_flash_pipeline_family="fa4",
                        cute_flash_wait_hint=index,
                    ),
                    status="ok",
                )

            current = member(0)
            neighbors = iter(member(index) for index in range(1, 10))
            search = LFBOPatternSearch.__new__(LFBOPatternSearch)
            search.max_generations = 4
            search.patience = 0
            search.frac_selected = 1.0
            search.kernel = SimpleNamespace(
                env=SimpleNamespace(process_group_name=None)
            )
            search.log = lambda *_args, **_kwargs: None
            search.config_gen = SimpleNamespace(
                _key_to_flat_indices={family_key: ([1], False)}
            )
            search._generate_neighbors = lambda *_args, **_kwargs: [
                next(neighbors).flat_values
            ]
            by_index = {index: member(index) for index in range(1, 10)}
            search.make_unbenchmarked = lambda flat: by_index[flat[0]]
            search._surrogate_select = lambda candidates, _count: candidates
            search._check_early_stopping = Mock(return_value=True)

            generations = list(
                search._pruned_pattern_search_from(
                    0,
                    current,
                    set(),
                    constraints,
                    disable_early_stopping=disable,
                )
            )
            return len(generations), search._check_early_stopping.call_count

        self.assertEqual(run_path(constraints=(), disable=True), (4, 0))
        self.assertEqual(run_path(constraints=(), disable=False), (1, 1))
        self.assertEqual(
            run_path(constraints=((family_key, "fa4"),), disable=False),
            (1, 1),
        )

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
            conditional_neighbor=lambda base, requested, **kwargs: requested,
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

        random.seed(123)
        unchanged = search._generate_neighbors(base, fixed_flat_values={})
        random.seed(123)
        self.assertEqual(unchanged, search._generate_neighbors(base))

        random.seed(123)
        fixed = search._generate_neighbors(base, fixed_flat_values={3: "b"})
        self.assertTrue(fixed)
        self.assertTrue(all(neighbor[3] == "b" for neighbor in fixed))

    def test_lfbo_tree_search_generate_neighbors_cap(self):
        """LFBOTreeSearch applies num_neighbors_cap in the tree-guided path."""

        class MockEstimator:
            tree_ = SimpleNamespace(feature=np.array([0], dtype=int))

            def decision_path(self, _x):
                return SimpleNamespace(indices=np.array([0], dtype=int))

            def predict_proba(self, x):
                scores = np.asarray(x)[:, 0]
                probas = scores / np.max(scores)
                return np.column_stack((1.0 - probas, probas))

        spec = PowerOfTwoFragment(16, 128, 32)
        search = LFBOTreeSearch.__new__(LFBOTreeSearch)
        search.num_neighbors = 10
        search.num_neighbors_cap = 3
        search.radius = 2
        search.surrogate = SimpleNamespace(estimators_=[MockEstimator()])
        search._autotune_metrics = SimpleNamespace(num_generations=2)
        search._encoded_to_flat_mapping = None
        search.config_gen = SimpleNamespace(
            flat_spec=[spec],
            block_size_indices=[0],
            num_warps_index=-1,
            overridden_flat_indices=set(),
            conditional_neighbor=lambda base, requested, **kwargs: requested,
            encode_config=lambda flat: spec.encode(flat[0]),
        )

        capped_neighbors = search._generate_neighbors([32])

        search.num_neighbors_cap = -1
        uncapped_neighbors = search._generate_neighbors([32])

        self.assertEqual(len(capped_neighbors), 3)
        self.assertEqual(len(uncapped_neighbors), 10)
        self.assertTrue(all(neighbor != [32] for neighbor in capped_neighbors))

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
        self.assertIsNone(quick_profile.flash_structural_search)
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
            self.assertIsNone(lfbo_tree.flash_structural_search)

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
        self.assertIsNone(lfbo_tree.flash_structural_search)

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
        with (
            without_env_var("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS"),
            patch.dict(os.environ, {"HELION_AUTOTUNER": "PatternSearch"}),
        ):
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

        # Env var also applies to LFBO-based pattern search.
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "LFBOTreeSearch",
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

        # Explicit constructor arg wins over env var for LFBO-based search too.
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "LFBOTreeSearch",
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


class TestCandidateTimer(TestCase):
    """``PopulationBasedSearch._settle_candidate_timer``: a search chooses
    the timer of its candidates once, after its initial population."""

    @pytest.fixture(autouse=True)
    def _no_graph_replay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # (a search under HELION_BENCHMARK_CUDAGRAPH=1 keeps the events)
        monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)

    @staticmethod
    def _search(bound: float | None = 64.0) -> PopulationBasedSearch:
        """A search on a backend whose short-kernel bound is ``bound``,
        whose benchmark worker is running and can trace."""
        kernel = mock_kernel(Settings(autotune_log_level=logging.CRITICAL))
        kernel.config_spec.backend.autotune_short_kernel_below_us = lambda config_spec: (
            bound
        )
        search = PopulationBasedSearch(cast("_AutotunableKernel", kernel), ())
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider._benchmark_worker = Mock()
        provider.config_spec = kernel.config_spec
        provider._effective_source_results = {}
        search.benchmark_provider = provider

        def probe() -> bool:
            # the probe runs a profiling session in the worker
            provider.worker_profiled = True
            return True

        search._kernel_timer_usable = probe
        search.log = Mock()
        return search

    @staticmethod
    def _measured(
        search: PopulationBasedSearch, best_ms: float
    ) -> list[PopulationMember]:
        """An initial population of one kernel that reads ``best_ms`` by
        CUDA events, which is then the search's best reading, with a
        stand-in for the re-timing that reads a kernel 4.6 us lower on the
        trace."""

        def read_again(members: list[PopulationMember], **kwargs: object) -> None:
            for timed in members:
                timed.perfs.append(timed.perf - 0.0046)

        search.best_perf_so_far = best_ms
        search.rebenchmark = read_again
        return [PopulationMember(lambda: None, [best_ms], (0,), helion.Config())]

    @pytest.mark.cpu_only
    def test_short_kernel_search_settles_on_the_trace_timer(self) -> None:
        """The search reads its backend's bound when it is built. Once its
        initial population's best config reads below it, the provider's
        jobs are traced and the paths count ties; before any reading the
        choice waits."""
        search = self._search()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        self.assertEqual(search.short_kernel_below_us, 64.0)
        self.assertFalse(search.traces_candidates)
        self.assertFalse(provider.trace_search_candidates)
        search._settle_candidate_timer([])
        self.assertFalse(search.traces_candidates)
        population = self._measured(search, 0.0164)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            # graph replay switched off explicitly is the default
            os.environ["HELION_BENCHMARK_CUDAGRAPH"] = "0"
            search._settle_candidate_timer(population)
        self.assertTrue(search.traces_candidates)
        self.assertTrue(provider.trace_search_candidates)
        self.assertIn("CUDA activity trace", search.log.call_args.args[0])
        # the population's one kernel holds its trace reading alone
        self.assertEqual(len(population[0].perfs), 1)
        self.assertAlmostEqual(search.best_perf_so_far, 0.0164 - 0.0046)

    @pytest.mark.cpu_only
    def test_search_keeps_the_event_timer(self) -> None:
        """No bound (every backend but CuTe), a best reading at or above
        it, a ratio objective, a search regime that does not flush per call,
        candidates that were not timed in the benchmark worker, a worker
        that cannot trace, a provider without a worker,
        ``HELION_BENCHMARK_CUDAGRAPH=1`` (``do_bench`` then times a
        candidate's graph replay, which it never traces) and a used-up
        autotune budget each leave the event timer and the fixed step
        rule.  Whether the worker can trace is asked last: the first answer
        runs a profiling session in the worker, which a search that keeps
        the events for another reason must not leave behind there.  A
        worker whose probe ran one and said no gets a new process, so the
        candidates that follow are read as the initial population was; a no
        that needs no session keeps the worker's process."""

        def no_bound(search: PopulationBasedSearch) -> None:
            search.short_kernel_below_us = None

        def at_the_bound(search: PopulationBasedSearch) -> None:
            search.best_perf_so_far = 0.064

        def no_worker(search: PopulationBasedSearch) -> None:
            search.benchmark_provider._benchmark_worker = None

        def no_trace(search: PopulationBasedSearch) -> None:
            # the probe runs a session in the worker and says no
            def probe() -> bool:
                search.benchmark_provider.worker_profiled = True
                return False

            search._kernel_timer_usable = Mock(side_effect=probe)

        def no_trace_without_a_session(search: PopulationBasedSearch) -> None:
            # a no that needs no session (``autotune_benchmark_fn``, a
            # wall-clock worker, a process group, a build without CUDA
            # activity)
            search._kernel_timer_usable = Mock(return_value=False)

        def other_provider(search: PopulationBasedSearch) -> None:
            search.benchmark_provider = Mock()

        def multi_shape(search: PopulationBasedSearch) -> None:
            # joint readings of several shapes are no worker jobs to trace
            search.benchmark_provider = MultiShapeBenchmarkProvider.__new__(
                MultiShapeBenchmarkProvider
            )

        def graph_replay(search: PopulationBasedSearch) -> None:
            # do_bench times every candidate's graph replay: none is traced
            os.environ["HELION_BENCHMARK_CUDAGRAPH"] = "1"

        def over_budget(search: PopulationBasedSearch) -> None:
            # reading the population again is timing work under the budget
            search._autotune_budget_exceeded = lambda: True

        cases = [
            (no_bound, None, "ms"),
            (at_the_bound, None, "ms"),
            (None, None, "ratio"),
            (no_worker, None, "ms"),
            (no_trace, None, "ms"),
            (no_trace_without_a_session, None, "ms"),
            (other_provider, None, "ms"),
            (multi_shape, None, "ms"),
            (None, "0", "ms"),
            (None, "replay", "ms"),
            (graph_replay, None, "ms"),
            (over_budget, None, "ms"),
        ]
        for change, l2_flush, unit in cases:
            with (
                self.subTest(change=change, l2_flush=l2_flush, unit=unit),
                patch.dict(os.environ),
            ):
                os.environ.pop("HELION_BENCHMARK_CUDAGRAPH", None)
                search = self._search()
                provider = search.benchmark_provider
                worker = provider._benchmark_worker
                search.best_perf_so_far = 0.0164
                search._kernel_timer_usable = Mock(return_value=True)
                if change is not None:
                    change(search)
                population = self._measured(search, search.best_perf_so_far)
                objective = (
                    patch.object(
                        PopulationBasedSearch,
                        "performance_unit",
                        new_callable=lambda: "ratio",
                    )
                    if unit == "ratio"
                    else contextlib.nullcontext()
                )
                with patch.dict(os.environ, {}, clear=False), objective:
                    os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
                    if l2_flush is not None:
                        os.environ["HELION_AUTOTUNE_L2_FLUSH"] = l2_flush
                    search._settle_candidate_timer(population)
                self.assertFalse(search.traces_candidates)
                self.assertFalse(provider.trace_search_candidates)
                # the worker's trace was asked only where nothing else
                # keeps the events
                self.assertEqual(
                    search._kernel_timer_usable.call_count,
                    change in (no_trace, no_trace_without_a_session),
                )
                # ... and only a worker that ran a session for the answer
                # starts over
                self.assertEqual(worker.shutdown.call_count, change is no_trace)
                self.assertFalse(provider.worker_profiled)
                # nothing was read again
                self.assertEqual(len(population[0].perfs), 1)
                # (a backend without a bound has nothing to settle)
                self.assertEqual(
                    search._candidate_timer_settled, change is not no_bound
                )
                search.log.assert_not_called()

    @pytest.mark.cpu_only
    def test_the_users_retiming_function_keeps_the_event_timer(self) -> None:
        """Under ``autotune_benchmark_fn`` the re-timings run the user's
        function in this process, on its own scale.  The final selection's
        check (``_kernel_timer_usable``) already asks whether the benchmark
        worker times the re-timings, which it does not then, so the timer
        choice has no test of its own for the setting: through that check,
        on a provider whose worker can trace, the search with the function
        keeps the events and the same search without it moves to the
        trace."""
        for custom in (True, False):
            with self.subTest(custom=custom), patch.dict(os.environ):
                os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
                search = self._search()
                del search._kernel_timer_usable  # the real check from here on
                provider = search.benchmark_provider
                assert isinstance(provider, LocalBenchmarkProvider)
                provider.settings = search.settings
                provider._subprocess_benchmark_enabled = lambda: True
                provider._subprocess_benchmark_uses_wall_clock = lambda: False
                provider.worker_kernel_timer_available = lambda: True
                if custom:
                    search.settings.autotune_benchmark_fn = lambda fns, repeat: []
                population = self._measured(search, 0.0164)
                with patch(
                    "helion.autotuner.base_search.kernel_timer_available",
                    return_value=True,
                ):
                    self.assertEqual(search._kernel_timer_usable(), not custom)
                    search._settle_candidate_timer(population)
                self.assertEqual(search.traces_candidates, not custom)
                self.assertEqual(provider.trace_search_candidates, not custom)

    @pytest.mark.cpu_only
    def test_a_search_without_a_bound_reads_no_search_state(self) -> None:
        """A search of a backend that names no bound returns before it reads
        anything else, so a stand-in search that tests build without
        ``__init__`` (and every backend but CuTe) is unaffected."""
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search._settle_candidate_timer([])
        self.assertFalse(search.traces_candidates)
        self.assertFalse(search._candidate_timer_settled)

    @pytest.mark.cpu_only
    def test_the_timer_is_chosen_once_per_search(self) -> None:
        """The trace leaves the launch latency out, so a launch near the
        bound that chose its timer by its own reading read on two scales
        from call to call. The search keeps the timer its initial population
        chose: a later candidate under the bound does not move a search on
        the event timer to the trace, and a search on the trace stays there
        whatever its candidates read."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search = self._search()
            population = self._measured(search, 0.066)
            search._settle_candidate_timer(population)
            search.best_perf_so_far = 0.060
            search._settle_candidate_timer(population)
            self.assertFalse(search.traces_candidates)
            self.assertFalse(search.benchmark_provider.trace_search_candidates)

            search = self._search()
            population = self._measured(search, 0.060)
            search._settle_candidate_timer(population)
            search._kernel_timer_usable = lambda: False
            search._settle_candidate_timer(population)
            self.assertTrue(search.traces_candidates)
            self.assertTrue(search.benchmark_provider.trace_search_candidates)

    @pytest.mark.cpu_only
    def test_the_timer_choice_does_not_depend_on_the_last_candidate(self) -> None:
        """A candidate that runs into the benchmark timeout kills the
        worker's process, and the next timed candidate respawns it. Where
        the backend settles source aliases without a job, the candidates
        after it may all repeat earlier code: the choice asks whether the
        provider times in a worker, not whether its process is up, and
        settles on the trace as when a kernel was timed last."""
        for process_up in (True, False):
            with self.subTest(process_up=process_up), patch.dict(os.environ):
                os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
                search = self._search()
                provider = search.benchmark_provider
                assert isinstance(provider, LocalBenchmarkProvider)
                provider._benchmark_worker.alive.return_value = process_up
                search._settle_candidate_timer(self._measured(search, 0.0164))
                self.assertTrue(search.traces_candidates)
                self.assertTrue(provider.trace_search_candidates)

    @pytest.mark.cpu_only
    def test_the_initial_population_settles_the_timer(self) -> None:
        """``benchmark_initial_population`` settles the timer on its
        population's readings, also when only the random fallback ran."""
        member = PopulationMember(lambda: None, [], (0,), helion.Config())
        fallback = PopulationMember(lambda: None, [], (1,), helion.Config(num_warps=2))

        def batch(perfs: dict[int, float]) -> Callable[..., None]:
            def run(members: list[PopulationMember], **kwargs: object) -> None:
                for timed in members:
                    timed.perfs.append(perfs[id(timed)])
                    search.best_perf_so_far = min(search.best_perf_so_far, timed.perf)

            return run

        def read_again(members: list[PopulationMember], **kwargs: object) -> None:
            for timed in members:
                timed.perfs.append(timed.perf - 0.0046)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search = self._search()
            search._benchmark_initial_batch = batch({id(member): 0.0164})
            search.rebenchmark = read_again
            search.benchmark_initial_population([member])
            self.assertTrue(search.traces_candidates)
            # the trace reading replaced the event reading
            self.assertEqual(len(member.perfs), 1)

            del member.perfs[:]
            search = self._search()
            search.rebenchmark = read_again
            search._benchmark_initial_batch = batch(
                {id(member): math.inf, id(fallback): 0.0164}
            )
            search.benchmark_provider.take_no_viable_config_error = lambda: None
            with (
                patch.object(
                    search,
                    "_pad_initial_population_with_unique_random",
                    return_value=[fallback.flat_values],
                ),
                patch.object(search, "make_unbenchmarked", return_value=fallback),
            ):
                search.benchmark_initial_population([member], random_fallback_target=1)
            self.assertTrue(search.traces_candidates)

    @pytest.mark.cpu_only
    def test_settling_on_the_trace_reads_the_initial_population_again(self) -> None:
        """The initial population is read with CUDA events, 4.6 us above the
        trace here. A search that settles on the trace reads every member
        with a reading once more on it, also the ones a verification would
        not re-time (beyond the threshold of the best; all of them at quick
        effort, whose threshold is 0.9), and marks where each member's
        trace readings start. A search that keeps the events reads nothing
        again."""
        latency = 0.0046

        def rebenchmark(members: list[PopulationMember], **kwargs: object) -> None:
            timed.append(list(members))
            for member in members:
                member.perfs.append(member.perf - latency)

        for best_us, traces in ((3.0, True), (70.0, False)):
            with self.subTest(traces=traces), patch.dict(os.environ):
                os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
                timed: list[list[PopulationMember]] = []
                search = self._search()
                best = best_us / 1000.0 + latency
                members = [
                    PopulationMember(lambda: None, [perf], (index,), helion.Config())
                    for index, perf in enumerate(
                        [best, best * 1.2, best * 4.0, math.inf]
                    )
                ]
                search.best_perf_so_far = best
                with (
                    patch.object(search, "rebenchmark", side_effect=rebenchmark),
                    patch.object(search, "should_rebenchmark", return_value=False),
                ):
                    search._settle_candidate_timer(members)
                self.assertEqual(search.traces_candidates, traces)
                if not traces:
                    self.assertEqual(timed, [])
                    self.assertEqual({m.timer_readings_from for m in members}, {0})
                    continue
                self.assertEqual(timed, [members[:3]])
                # Each member that was read holds its trace reading alone:
                # no event reading is left to rank it or to pair with.
                self.assertEqual({m.timer_readings_from for m in members}, {0})
                self.assertEqual(
                    [len(member.perfs) for member in members], [1, 1, 1, 1]
                )
                self.assertAlmostEqual(members[0].perf, best_us / 1000.0)
                self.assertAlmostEqual(members[2].perf, best * 4.0 - latency)
                self.assertEqual(rereading_spread(members * 8), 0.0)
                self.assertAlmostEqual(search.best_perf_so_far, best_us / 1000.0)

    @pytest.mark.cpu_only
    def test_settling_on_the_trace_moves_every_event_reading_or_drops_it(
        self,
    ) -> None:
        """The switch is the one place a search changes timers, so whatever
        else holds an event reading moves there or goes.  The provider's
        record of a kernel's reading, which settles later configs with the
        same generated source, takes the trace reading; and the algorithm is
        told to drop the readings it holds outside its members.  A search
        whose trace read nothing stays on the events and keeps both."""
        latency = 0.0046
        for readable in (True, False):
            with self.subTest(readable=readable), patch.dict(os.environ):
                os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
                search = self._search()
                kernel = PopulationMember(
                    lambda: None, [0.003 + latency], (0,), helion.Config()
                )
                provider = search.benchmark_provider
                assert isinstance(provider, LocalBenchmarkProvider)
                source_of = {id(kernel.fn): "source"}
                provider.config_spec.backend.generated_source_hash = (
                    lambda fn, source_of=source_of: source_of.get(id(fn))
                )
                provider._effective_source_results = {
                    "source": BenchmarkResult(
                        config=kernel.config,
                        fn=kernel.fn,
                        perf=kernel.perf,
                        status="ok",
                        compile_time=1.0,
                    )
                }
                search.best_perf_so_far = kernel.perf

                def rebenchmark(
                    members: list[PopulationMember],
                    readable: bool = readable,
                    **kwargs: object,
                ) -> None:
                    for member in members:
                        if readable:
                            member.perfs.append(member.perf - latency)

                with (
                    patch.object(search, "rebenchmark", side_effect=rebenchmark),
                    patch.object(search, "_discard_event_readings") as discard,
                ):
                    search._settle_candidate_timer([kernel])

                record = provider._effective_source_results["source"]
                self.assertEqual(search.traces_candidates, readable)
                self.assertEqual(discard.call_count, int(readable))
                self.assertAlmostEqual(
                    record.perf, 0.003 if readable else 0.003 + latency
                )
                # a later config with this source is settled with the record
                self.assertEqual(record.perf, kernel.perf)

    @pytest.mark.cpu_only
    def test_a_traced_search_drops_training_rows_it_cannot_read_again(self) -> None:
        """The LLM-seeded hybrid hands the LFBO stage the LLM stage's
        readings as training rows (``seed_training_data``), which no member
        stands behind.  They are CUDA-event readings: 7.6 us for a kernel
        the trace reads at 3.0 us.  Once the LFBO stage settles on the trace
        its own labels follow its members, so the seeded readings go: left
        in, one kernel would carry a label of each timer, and the split
        would mark the region the LLM stage explored as slow.  A seeded row
        that records a failed config stays: it is a negative for the
        classifier whichever timer would have read it."""
        search = object.__new__(LFBOTreeSearch)
        search.log = Mock()
        search.config_gen = SimpleNamespace(
            flatten=lambda config: [config.num_warps],
            encode_config=lambda flat: [float(flat[0])],
        )
        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(generated_source_hash=lambda fn: None)
        )
        search.train_x, search.train_y, search._train_members = [], [], []
        search.train_configs, search.train_source_hashes = [], []
        seeded = [
            BenchmarkResult(
                config=helion.Config(num_warps=index),
                fn=lambda: None,
                perf=perf,
                status="ok",
                compile_time=1.0,
            )
            for index, perf in enumerate([0.0076, 0.0081, math.inf])
        ]
        search.seed_training_data(seeded)
        member = PopulationMember(
            lambda: None, [0.0076, 0.0030], (0,), helion.Config(num_warps=0)
        )
        search._append_training_sample(
            [0.0], 0.0076, member.config, member.fn, member=member
        )

        search._discard_event_readings()
        search._sync_training_labels()

        self.assertEqual(search.train_x, [[2.0], [0.0]])
        self.assertEqual(search.train_y, [math.inf, 0.0030])
        self.assertEqual(search._train_members, [None, member])
        self.assertEqual(search.train_configs, [seeded[2].config, member.config])
        self.assertEqual(search.train_source_hashes, [None, None])
        (message,), _kwargs = search.log.call_args
        self.assertIn("Dropped 2 seeded", message)

        # No seeded reading left: nothing dropped, nothing logged.
        search.log.reset_mock()
        search._discard_event_readings()
        self.assertEqual(search.train_y, [math.inf, 0.0030])
        search.log.assert_not_called()

    @staticmethod
    def _worker_readings(
        search: PopulationBasedSearch,
        readings: dict[int, float | list[float | None] | None],
    ) -> Mock:
        """Let the search's re-timings run through the real ``rebenchmark``
        against a provider whose worker reads ``readings`` (by the id of a
        member's function; ``None``: the window gave no reading, so the
        member keeps the one it had; a list: one entry per job of that
        function, the last one for every job after it). Returns the
        provider's ``benchmark_isolated``, whose calls also record whether
        the jobs were traced; a traced call marks the worker profiled, as
        the provider's job path does."""
        provider = search.benchmark_provider
        search.log.trace_enabled = False

        def reading(fn: Callable[..., object]) -> float | None:
            value = readings[id(fn)]
            if isinstance(value, list):
                return value.pop(0) if len(value) > 1 else value[0]
            return value

        def benchmark_isolated(
            fns: list[Callable[..., object]], **kwargs: object
        ) -> list[float | None]:
            traced.append(provider.trace_search_candidates)
            if provider.trace_search_candidates:
                # a traced window runs a profiling session in the worker
                provider.worker_profiled = True
            return [reading(fn) for fn in fns]

        traced: list[bool] = []
        provider.register_isolated_candidates = Mock()
        provider.benchmark_isolated = Mock(side_effect=benchmark_isolated)
        provider.benchmark_isolated.traced = traced
        return provider.benchmark_isolated

    @pytest.mark.cpu_only
    def test_settling_on_the_trace_reads_a_single_kernel(self) -> None:
        """A population with one kernel that has a reading (a seed beside
        configs that failed) moves to the trace like any other: a 3 us
        parent that reads 7.6 us by CUDA events is read again at 3.0 us, so
        a child that reads 4.5 us on the trace, 1.5x slower, does not beat
        it for reading below the parent's event reading. A search that
        keeps the events, here for a 70 us kernel, reads nothing again."""
        for kernel_us, traces in ((3.0, True), (70.0, False)):
            with self.subTest(traces=traces), patch.dict(os.environ):
                os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
                search = self._search()
                events_ms = (kernel_us + 4.6) / 1000.0
                parent = PopulationMember(
                    lambda: None, [events_ms], (0,), helion.Config(num_warps=1)
                )
                failed = PopulationMember(
                    lambda: None,
                    [math.inf],
                    (1,),
                    helion.Config(num_warps=2),
                    status="error",
                )
                child = PopulationMember(
                    lambda: None,
                    [1.5 * kernel_us / 1000.0],
                    (2,),
                    helion.Config(num_warps=3),
                )
                worker = self._worker_readings(
                    search,
                    {id(parent.fn): kernel_us / 1000.0, id(child.fn): child.perf},
                )
                search.best_perf_so_far = events_ms
                search._settle_candidate_timer([parent, failed])
                self.assertEqual(search.traces_candidates, traces)
                if traces:
                    # one traced job, for the one kernel
                    self.assertEqual(worker.traced, [True])
                    self.assertEqual(worker.call_args.args[0], [parent.fn])
                    self.assertEqual(parent.perfs, [kernel_us / 1000.0])
                    self.assertEqual(parent.timer_readings_from, 0)
                    self.assertEqual(search.compare(child, parent), 1)
                else:
                    worker.assert_not_called()
                    self.assertEqual(parent.perfs, [events_ms])
                    self.assertEqual(parent.timer_readings_from, 0)

    @pytest.mark.cpu_only
    def test_a_trace_that_reads_no_initial_config_keeps_the_events(self) -> None:
        """A traced window whose trace cannot be read gives no reading
        (``KernelTraceUnreadable`` in the worker; the member keeps the
        reading it had). Where that is every kernel of the initial
        population, in a first and in a second pass, the trace does not read
        this kernel's windows and every later candidate would fail the same
        way: the search keeps the events and the readings it has, and says
        so."""
        with patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search = self._search()
            members = [
                PopulationMember(
                    lambda: None,
                    [(7.6 + index) / 1000.0],
                    (index,),
                    helion.Config(num_warps=index + 1),
                )
                for index in range(3)
            ]
            visited = {member.config for member in members}
            worker = self._worker_readings(
                search, {id(member.fn): None for member in members}
            )
            search.best_perf_so_far = members[0].perf
            search._settle_candidate_timer(members, visited)
            # both passes ran traced, on every kernel with a reading
            self.assertEqual(worker.traced, [True, True])
            for call in worker.call_args_list:
                self.assertEqual(call.args[0], [m.fn for m in members])
            self.assertFalse(search.traces_candidates)
            self.assertFalse(search.benchmark_provider.trace_search_candidates)
            # nobody left the population, and every reading is as it was
            self.assertEqual(len(members), 3)
            self.assertEqual(len(visited), 3)
            self.assertEqual([len(member.perfs) for member in members], [1, 1, 1])
            self.assertEqual({member.timer_readings_from for member in members}, {0})
            self.assertIn("with CUDA events", search.log.call_args.args[0])
            # the windows ran profiling sessions in the worker, which read
            # the event readings the search keeps before them: its process
            # is replaced, the worker stays
            provider = search.benchmark_provider
            provider._benchmark_worker.shutdown.assert_called_once_with()
            self.assertTrue(provider.benchmark_worker_started())
            self.assertFalse(provider.worker_profiled)
            # the choice stands for the search
            self.assertTrue(search._candidate_timer_settled)
            search._settle_candidate_timer(members, visited)
            self.assertEqual(worker.call_count, 2)
            provider._benchmark_worker.shutdown.assert_called_once_with()

    @pytest.mark.cpu_only
    def test_a_population_without_a_reading_keeps_the_events(self) -> None:
        """The move reads the members that have a reading.  A population
        that holds none (it is empty, or its every member failed) has
        nothing to move: like a population none of whose kernels the trace
        reads, it leaves the search on the events with the best reading it
        has, and times nothing."""
        with patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search = self._search()
            failed = PopulationMember(
                lambda: None, [math.inf], (0,), helion.Config(), status="error"
            )
            worker = self._worker_readings(search, {})
            search.best_perf_so_far = 0.0164
            for members in ([], [failed]):
                self.assertFalse(search._read_on_settled_timer(list(members)))
                self.assertEqual(search.best_perf_so_far, 0.0164)
            search._settle_candidate_timer([failed])
            self.assertFalse(search.traces_candidates)
            self.assertFalse(search.benchmark_provider.trace_search_candidates)
            self.assertEqual(search.best_perf_so_far, 0.0164)
            self.assertEqual(failed.perfs, [math.inf])
            worker.assert_not_called()
            self.assertIn("with CUDA events", search.log.call_args.args[0])

    @pytest.mark.cpu_only
    def test_a_member_whose_trace_reading_fails_is_read_again(self) -> None:
        """The fastest kernel of the initial population, 3.0 us, reads
        7.6 us by CUDA events like the others, and its first traced window
        gives no reading.  Left with the event reading it would lose to
        every traced config that is slower than it but reads below 7.6 us,
        and nothing would read it again (it sits outside the re-timing
        threshold of a 3.5 us best).  A second pass reads it."""
        with patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search = self._search()
            fastest, other = (
                PopulationMember(
                    lambda: None, [0.0076], (index,), helion.Config(num_warps=index)
                )
                for index in (1, 2)
            )
            child = PopulationMember(
                lambda: None, [0.0045], (3,), helion.Config(num_warps=3)
            )
            worker = self._worker_readings(
                search,
                {
                    id(fastest.fn): [None, 0.0030],
                    id(other.fn): 0.0035,
                    id(child.fn): 0.0045,
                },
            )
            members = [fastest, other]
            search.best_perf_so_far = 0.0076
            search._settle_candidate_timer(members)

            self.assertTrue(search.traces_candidates)
            self.assertEqual(
                [call.args[0] for call in worker.call_args_list],
                [[fastest.fn, other.fn], [fastest.fn]],
            )
            self.assertEqual(members, [fastest, other])
            self.assertEqual(fastest.perfs, [0.0030])
            self.assertEqual(other.perfs, [0.0035])
            # a traced child that is 1.5x slower does not replace it
            self.assertEqual(search.compare(child, fastest), 1)
            search.log.warning.assert_not_called()

    @staticmethod
    def _history(search: PopulationBasedSearch) -> list[dict[int, list[float]]]:
        """The readings in the final selection's history and in its pins,
        by the configs' ``num_warps``."""
        return [
            {config.num_warps: [*entry.perfs] for config, entry in registry.items()}
            for registry in (
                search._benchmarked_members,
                search._pinned_finalist_members,
            )
        ]

    @pytest.mark.cpu_only
    def test_a_budget_used_up_before_the_second_pass_keeps_the_events(self) -> None:
        """A failed trace reading can cost a whole benchmark timeout, and so
        would its repeat: once ``autotune_budget_seconds`` is used up the
        second pass is not started.  The member whose first reading failed
        is the fastest kernel here, and a search over its budget times no
        candidate to make up for it, so the search does not move with the
        members it read: the move is undone, every member holds its event
        reading again, and the search stays on the events and says why.
        While the budget lasts, as in every search without one, the member
        is read again and the search moves."""

        def case(used_up: bool) -> None:
            search = self._search()
            failing, other = (
                PopulationMember(
                    lambda: None, [0.0076], (index,), helion.Config(num_warps=index)
                )
                for index in (1, 2)
            )
            worker = self._worker_readings(
                search, {id(failing.fn): [None, 0.0030], id(other.fn): 0.0035}
            )
            # The budget lasts until the search starts on the trace and is
            # used up when the first pass is over.
            search._autotune_budget_exceeded = lambda: used_up and worker.call_count > 0
            members = [failing, other]
            visited = {member.config for member in members}
            search.best_perf_so_far = 0.0076
            search._settle_candidate_timer(members, visited)

            passes = [call.args[0] for call in worker.call_args_list]
            search.log.warning.assert_not_called()
            shutdown = search.benchmark_provider._benchmark_worker.shutdown
            if not used_up:
                self.assertTrue(search.traces_candidates)
                self.assertEqual(passes, [[failing.fn, other.fn], [failing.fn]])
                self.assertEqual(members, [failing, other])
                self.assertEqual(failing.perfs, [0.0030])
                # a search that moved keeps its worker's process
                shutdown.assert_not_called()
                self.assertTrue(search.benchmark_provider.worker_profiled)
                return
            # the event readings are back, and so is a process without the
            # sessions of the pass that ran
            shutdown.assert_called_once_with()
            self.assertFalse(search.benchmark_provider.worker_profiled)
            self.assertEqual(passes, [[failing.fn, other.fn]])
            self.assertFalse(search.traces_candidates)
            self.assertFalse(search.benchmark_provider.trace_search_candidates)
            self.assertEqual(members, [failing, other])
            self.assertEqual(visited, {failing.config, other.config})
            self.assertEqual((failing.perfs, other.perfs), ([0.0076], [0.0076]))
            self.assertEqual(search.best_perf_so_far, 0.0076)
            (message,) = search.log.call_args.args
            self.assertIn("the autotune budget ran out", message)
            self.assertIn("with CUDA events", message)

        for used_up in (True, False):
            with self.subTest(used_up=used_up), patch.dict(os.environ):
                os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
                case(used_up)

    @pytest.mark.cpu_only
    def test_the_population_is_read_again_within_the_budget(self) -> None:
        """Reading the initial population again is timing work, which
        ``autotune_budget_seconds`` covers as it covers the candidates'.
        Through the real ``rebenchmark`` and the provider's real
        ``benchmark_isolated``, with the search's budget hook installed as
        ``_prepare`` installs it: the hook is asked before each kernel and
        the kernels after the budget ran out are not timed, in the first
        pass or in a second one.  A move that the budget cuts short is
        undone: the members that were read hold their CUDA-event readings
        alone again, the best reading and the final selection's history and
        pins are as they were, nobody leaves the population, and the search
        keeps the events.  A search without a budget (the default) times
        every kernel and moves, and a verification's re-timing is not cut
        by a used-up budget, as before."""

        def case(budget_jobs: int | None) -> None:
            search = self._search()
            provider = search.benchmark_provider
            assert isinstance(provider, LocalBenchmarkProvider)
            provider.settings = search.settings
            provider.log = search.log
            provider._autotune_metrics = AutotuneMetrics()
            search.log.trace_enabled = False
            members = [
                PopulationMember(
                    lambda: None,
                    [(7.6 + index) / 1000.0],
                    (index,),
                    helion.Config(num_warps=index + 1),
                )
                for index in range(4)
            ]
            events_of = {id(member.fn): member.perf for member in members}
            jobs: list[Callable[..., object]] = []

            def job(fn: Callable[..., object], **kwargs: object) -> float:
                jobs.append(fn)
                return events_of[id(fn)] - 0.0046

            if budget_jobs is not None:
                # the budget is used up once that many kernels were timed
                search._autotune_budget_exceeded = lambda: len(jobs) >= budget_jobs
            # (without one, the search's own check: no budget is set)
            provider.set_budget_exceeded_fn(search._autotune_budget_exceeded)
            visited = {member.config for member in members}
            search.best_perf_so_far = members[0].perf
            search.pin_finalist_config(members[1].config)
            with (
                patch.object(
                    provider, "_subprocess_benchmark_enabled", return_value=True
                ),
                patch.object(
                    provider, "_run_subprocess_benchmark_job", side_effect=job
                ),
                patch.object(search, "_final_rebenchmark_top_k", return_value=8),
            ):
                for member in members:
                    search._record_benchmarked_member(member)
                history = self._history(search)
                self.assertEqual([len(entries) for entries in history], [4, 1])
                search._settle_candidate_timer(members, visited)

                self.assertFalse(provider._isolated_timings_budgeted)
                self.assertEqual(len(members), 4)
                self.assertEqual(len(visited), 4)
                if budget_jobs is None:
                    self.assertTrue(search.traces_candidates)
                    self.assertEqual(jobs, [member.fn for member in members])
                    for member in members:
                        self.assertEqual(
                            member.perfs, [events_of[id(member.fn)] - 0.0046]
                        )
                    self.assertAlmostEqual(search.best_perf_so_far, 0.0030)
                    self.assertNotEqual(self._history(search), history)
                else:
                    # two kernels were read on the trace before the cut
                    self.assertEqual(jobs, [member.fn for member in members[:2]])
                    self.assertFalse(search.traces_candidates)
                    self.assertFalse(provider.trace_search_candidates)
                    for member in members:
                        self.assertEqual(member.perfs, [events_of[id(member.fn)]])
                    self.assertEqual(search.best_perf_so_far, 0.0076)
                    self.assertEqual(self._history(search), history)
                    (message,) = search.log.call_args.args
                    self.assertIn("the autotune budget ran out", message)
                search.log.warning.assert_not_called()

                # A verification re-times its members whatever the budget
                # says (it is used up here, where there is one): only the
                # population's move is cut.
                timed = len(jobs)
                search.rebenchmark(members)
                self.assertEqual(jobs[timed:], [member.fn for member in members])

        for budget_jobs in (2, None):
            with self.subTest(budget_jobs=budget_jobs), patch.dict(os.environ):
                os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
                case(budget_jobs)

    @pytest.mark.cpu_only
    def test_a_failed_trace_reading_is_no_reading_of_the_population(self) -> None:
        """A re-timing that leaves a sticky device error in the worker is a
        failure of the kernel, not a reading of the trace: its member reads
        infinite from there on, as after a verification that failed,
        whichever timer the search goes on with.  Through the real
        ``rebenchmark`` and the provider's real ``benchmark_isolated``, with
        the search's budget hook installed as ``_prepare`` installs it:

        - The fastest config by CUDA events fails and the budget runs out
          before the next kernel is timed.  The trace read no config, so the
          search keeps the events, and the failed config is not its best
          again.
        - A config is read, the next one fails and the budget runs out
          before the third.  The move is undone and the failure is not: the
          history and the pins hold the failed config as its re-timing left
          it.
        - A config fails and the trace reads none of the others in either
          pass.  It read no config, so the search keeps the events and
          nobody leaves the population.
        - A config fails and the trace reads the others.  The search moves;
          the failed config is not timed a second time and stays a member
          with its failure alone, without an entry in the history."""
        device_error = "CUDA error: an illegal memory access was encountered"

        def case(
            events_us: Sequence[float],
            outcomes: Sequence[str],
            budget_jobs: int | None,
        ) -> tuple[
            PopulationBasedSearch,
            list[PopulationMember],
            list[int],
            list[dict[int, list[float]]],
        ]:
            """Settle the timer of a search over one config per reading;
            returns it with its members, the timed configs' ``num_warps``
            and the history and pins."""
            search = self._search()
            provider = search.benchmark_provider
            assert isinstance(provider, LocalBenchmarkProvider)
            provider.settings = search.settings
            provider.log = search.log
            provider._autotune_metrics = AutotuneMetrics()
            search.log.trace_enabled = False
            members = [
                PopulationMember(
                    lambda: None,
                    [reading / 1000.0],
                    (index,),
                    helion.Config(num_warps=index + 1),
                )
                for index, reading in enumerate(events_us)
            ]
            of_fn = {
                id(member.fn): (member, outcome)
                for member, outcome in zip(members, outcomes, strict=True)
            }
            jobs: list[int] = []
            untimed: list[str] = []

            def job(fn: Callable[..., object], **kwargs: object) -> float:
                member, outcome = of_fn[id(fn)]
                jobs.append(member.config.num_warps)
                if outcome == "fails":
                    untimed.append(outcome)
                    raise RuntimeError(device_error)
                if outcome == "unreadable":
                    untimed.append(outcome)
                    raise KernelTraceUnreadable("kernel timer traced no settled call")
                return member.perfs[0] - 0.0046

            if budget_jobs is not None:
                # the budget is used up once that many kernels were timed
                search._autotune_budget_exceeded = lambda: len(jobs) >= budget_jobs
            provider.set_budget_exceeded_fn(search._autotune_budget_exceeded)
            visited = {member.config for member in members}
            search.best_perf_so_far = members[0].perf
            search.pin_finalist_config(members[1].config)
            with (
                patch.object(
                    provider, "_subprocess_benchmark_enabled", return_value=True
                ),
                patch.object(
                    provider, "_run_subprocess_benchmark_job", side_effect=job
                ),
                patch.object(search, "_final_rebenchmark_top_k", return_value=8),
            ):
                for member in members:
                    search._record_benchmarked_member(member)
                search._settle_candidate_timer(members, visited)
                history = self._history(search)
            self.assertEqual(len(visited), len(members))
            self.assertEqual(
                search.benchmark_provider.trace_search_candidates,
                search.traces_candidates,
            )
            # the provider warns once per job that gave no timing, and no
            # config leaves the population with a warning of the search's
            self.assertEqual(
                [
                    device_error in call.args[0]
                    for call in search.log.warning.call_args_list
                ],
                [outcome == "fails" for outcome in untimed],
            )
            return search, members, jobs, history

        with (
            self.subTest("the budget runs out after the failure"),
            patch.dict(os.environ),
        ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search, (failed, other), jobs, history = case(
                (7.6, 80.0), ("fails", "read"), budget_jobs=1
            )
            self.assertEqual(jobs, [1])
            self.assertFalse(search.traces_candidates)
            self.assertEqual(failed.perfs, [0.0076, math.inf])
            self.assertEqual(other.perfs, [0.080])
            self.assertIs(min((failed, other), key=operator.attrgetter("perf")), other)
            self.assertIn("the autotune budget ran out", search.log.call_args.args[0])

        with (
            self.subTest("a move that is undone keeps the failure"),
            patch.dict(os.environ),
        ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search, (read, failed, cut), jobs, history = case(
                (7.6, 8.6, 9.6), ("read", "fails", "read"), budget_jobs=2
            )
            self.assertEqual(jobs, [1, 2])
            self.assertFalse(search.traces_candidates)
            self.assertEqual(read.perfs, [0.0076])
            self.assertEqual(failed.perfs, [0.0086, math.inf])
            self.assertEqual(cut.perfs, [0.0096])
            self.assertEqual(search.best_perf_so_far, 0.0076)
            self.assertEqual(
                history,
                [
                    {1: [0.0076], 2: [0.0086, math.inf], 3: [0.0096]},
                    {2: [0.0086, math.inf]},
                ],
            )
            self.assertIn("the autotune budget ran out", search.log.call_args.args[0])

        with self.subTest("the trace reads none of the others"), patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search, members, jobs, history = case(
                (7.6, 8.6, 9.6), ("fails", "unreadable", "unreadable"), budget_jobs=None
            )
            # the failure is conclusive: the second pass asks for the others
            self.assertEqual(jobs, [1, 2, 3, 2, 3])
            self.assertFalse(search.traces_candidates)
            self.assertEqual(
                [member.perfs for member in members],
                [[0.0076, math.inf], [0.0086], [0.0096]],
            )
            self.assertEqual(
                history,
                [
                    {1: [0.0076, math.inf], 2: [0.0086], 3: [0.0096]},
                    {2: [0.0086]},
                ],
            )
            self.assertIn(
                "read none of the initial configs", search.log.call_args.args[0]
            )

        with self.subTest("the trace reads the others"), patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search, members, jobs, history = case(
                (7.6, 8.6, 9.6), ("fails", "read", "read"), budget_jobs=None
            )
            self.assertEqual(jobs, [1, 2, 3])
            self.assertTrue(search.traces_candidates)
            self.assertEqual(len(members), 3)
            self.assertEqual(members[0].perfs, [math.inf])
            self.assertEqual(
                [member.perfs for member in members[1:]],
                [[0.0086 - 0.0046], [0.0096 - 0.0046]],
            )
            self.assertEqual(search.best_perf_so_far, 0.0086 - 0.0046)
            self.assertEqual(
                history,
                [
                    {2: [0.0086 - 0.0046], 3: [0.0096 - 0.0046]},
                    {2: [0.0086 - 0.0046]},
                ],
            )

    @pytest.mark.cpu_only
    def test_a_member_the_trace_cannot_read_leaves_the_initial_population(
        self,
    ) -> None:
        """A member whose traced windows give no reading in two passes has
        no reading on the search's timer.  Its event reading is not kept
        for it, as a slow config or as a failure: it leaves the initial
        population and the visited set (the search may come back to the
        config), the final selection's history and pins, and the provider
        forgets the event reading of its generated source, so a later
        config with that code is timed on the trace."""
        with patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search = self._search()
            unread, pinned_unread, kept = (
                PopulationMember(
                    lambda: None, [0.0076], (index,), helion.Config(num_warps=index)
                )
                for index in (1, 2, 3)
            )
            failed = PopulationMember(
                lambda: None, [math.inf], (4,), helion.Config(num_warps=4)
            )
            members = [unread, pinned_unread, kept, failed]
            visited = {member.config for member in members}
            provider = search.benchmark_provider
            assert isinstance(provider, LocalBenchmarkProvider)
            source_of = {
                id(member.fn): f"source{index}" for index, member in enumerate(members)
            }
            provider.config_spec.backend.generated_source_hash = lambda fn: source_of[
                id(fn)
            ]
            provider._effective_source_results = {
                source_of[id(member.fn)]: BenchmarkResult(
                    config=member.config,
                    fn=member.fn,
                    perf=member.perf,
                    status="ok",
                    compile_time=1.0,
                )
                for member in (unread, pinned_unread, kept)
            }
            for member in (unread, pinned_unread, kept):
                search._benchmarked_members[member.config] = member
            search.pin_finalist_config(pinned_unread.config)
            search._pinned_finalist_members[pinned_unread.config] = pinned_unread
            worker = self._worker_readings(
                search,
                {
                    id(unread.fn): None,
                    id(pinned_unread.fn): None,
                    id(kept.fn): 0.0035,
                },
            )
            search.best_perf_so_far = 0.0076
            with patch.object(search, "_discard_event_readings") as discard:
                search._settle_candidate_timer(members, visited)

            self.assertTrue(search.traces_candidates)
            # the second pass asks for the two that gave no reading
            self.assertEqual(
                [call.args[0] for call in worker.call_args_list],
                [
                    [unread.fn, pinned_unread.fn, kept.fn],
                    [unread.fn, pinned_unread.fn],
                ],
            )
            self.assertEqual(members, [kept, failed])
            self.assertEqual(visited, {kept.config, failed.config})
            self.assertEqual(list(search._benchmarked_members.values()), [kept])
            self.assertEqual(list(search._pinned_finalist_members.values()), [])
            # the record of the kernel that was read moved to the trace; the
            # other two sources are timed when a config produces them again
            self.assertEqual(
                {
                    source: result.perf
                    for source, result in provider._effective_source_results.items()
                },
                {"source2": 0.0035},
            )
            discard.assert_called_once_with()
            (warning,) = search.log.warning.call_args_list
            self.assertIn("2 of the initial configs", warning.args[0])
            # no reading of the population is an event reading any more
            self.assertEqual(
                [member.perf for member in members if math.isfinite(member.perf)],
                [0.0035],
            )

    @pytest.mark.cpu_only
    def test_an_alias_is_read_on_the_trace_when_its_kernel_is(self) -> None:
        """Whether the trace read a member is asked of its kernel.  A source
        alias takes its kernel's readings and gains none where the trace
        reading equals the event reading it holds, so its own list does not
        say.  The alias of a kernel the trace read stays in the population
        with that reading, the kernel is not asked for a second time, and
        the provider keeps the kernel's record at the trace reading (the
        alias shares its function, so forgetting the alias's source would
        forget the kernel's).  The alias of a kernel the trace did not read
        leaves with it."""
        with patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search = self._search()
            search._collapses_source_aliases = lambda: True
            kernel, unread = (
                PopulationMember(
                    lambda: None, [0.0076], (index,), helion.Config(num_warps=index)
                )
                for index in (1, 2)
            )
            alias, unread_alias = (
                PopulationMember(
                    holder.fn,
                    [0.0076],
                    (index,),
                    helion.Config(num_warps=index),
                    status="deduplicated",
                )
                for index, holder in ((3, kernel), (4, unread))
            )
            for holder in (kernel, unread):
                search._kernels[id(holder.fn)] = holder
            members = [alias, kernel, unread_alias, unread]
            visited = {member.config for member in members}
            provider = search.benchmark_provider
            assert isinstance(provider, LocalBenchmarkProvider)
            source_of = {id(kernel.fn): "kernel", id(unread.fn): "unread"}
            provider.config_spec.backend.generated_source_hash = lambda fn: source_of[
                id(fn)
            ]
            provider._effective_source_results = {
                source_of[id(holder.fn)]: BenchmarkResult(
                    config=holder.config,
                    fn=holder.fn,
                    perf=holder.perf,
                    status="ok",
                    compile_time=1.0,
                )
                for holder in (kernel, unread)
            }
            # the trace reads the kernel at the very reading the events gave
            worker = self._worker_readings(
                search, {id(kernel.fn): 0.0076, id(unread.fn): None}
            )
            search.best_perf_so_far = 0.0076
            search._settle_candidate_timer(members, visited)

            self.assertTrue(search.traces_candidates)
            self.assertEqual(
                [call.args[0] for call in worker.call_args_list],
                [[kernel.fn, unread.fn], [unread.fn]],
            )
            self.assertEqual(members, [alias, kernel])
            self.assertEqual(visited, {alias.config, kernel.config})
            self.assertEqual((kernel.perfs, kernel.timer_readings_from), ([0.0076], 0))
            # a copy of the kernel's trace reading, none of its own
            self.assertEqual((alias.perfs, alias.timer_readings_from), ([0.0076], 1))
            self.assertEqual(list(search._kernels.values()), [kernel])
            self.assertEqual(
                {
                    source: result.perf
                    for source, result in provider._effective_source_results.items()
                },
                {"kernel": 0.0076},
            )

    @pytest.mark.cpu_only
    def test_a_search_inside_find_handoff_keeps_the_event_timer(self) -> None:
        """A handoff session ranks its shortlist by, and returns, the
        readings of every batch it observed (``HandoffPoint.measurements``),
        the initial population's among them, and none of them can be read
        again: its search stays on the events, without asking the worker
        for its trace (the answer takes a profiling session there, and the
        session's later batches are timed in that worker).  The same search
        outside a session moves to the trace."""
        for in_session in (True, False):
            with self.subTest(in_session=in_session), patch.dict(os.environ):
                os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
                search = self._search()
                search._kernel_timer_usable = Mock(return_value=True)
                population = self._measured(search, 0.0164)
                token = _active_handoff.set(Mock() if in_session else None)
                try:
                    search._settle_candidate_timer(population)
                finally:
                    _active_handoff.reset(token)
                self.assertEqual(search.traces_candidates, not in_session)
                self.assertEqual(
                    search.benchmark_provider.trace_search_candidates, not in_session
                )
                self.assertEqual(search._kernel_timer_usable.call_count, not in_session)
                self.assertTrue(search._candidate_timer_settled)

    @pytest.mark.cpu_only
    def test_the_switch_ranks_the_final_selections_history_by_the_trace(self) -> None:
        """The final selection's history keeps the best ``top_k`` configs,
        and the initial population fills it while it is timed with CUDA
        events, the timer that cannot tell these configs apart: three
        configs read 16.4, 16.4 and 16.5 us there and 13.0, 13.1 and
        11.8 us on the trace.  With room for two, the event readings keep
        the first two.  The switch records every member again by its trace
        reading, so the history holds the 11.8 us config, and so do the
        registry the compiler refinement reuses readings from and the best
        reading the re-timing threshold is taken from."""
        with patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search = self._search()
            search._terminal_refinement_members = ConfigCache()
            events = [0.0164, 0.0164, 0.0165]
            trace = [0.0130, 0.0131, 0.0118]
            members = [
                PopulationMember(
                    lambda: None,
                    [reading],
                    (index,),
                    helion.Config(num_warps=index + 1),
                    status="ok",
                )
                for index, reading in enumerate(events)
            ]
            worker = self._worker_readings(
                search,
                {id(m.fn): reading for m, reading in zip(members, trace, strict=True)},
            )
            search.best_perf_so_far = min(events)
            with patch.object(search, "_final_rebenchmark_top_k", return_value=2):
                for member in members:
                    search._record_benchmarked_member(member)
                self.assertEqual(
                    sorted(c.num_warps for c in search._benchmarked_members), [1, 2]
                )
                search._settle_candidate_timer(members)

                self.assertTrue(search.traces_candidates)
                self.assertEqual(worker.call_count, 1)
                self.assertEqual(
                    {
                        entry.config.num_warps: entry.perfs
                        for entry in search._benchmarked_members.values()
                    },
                    {3: [0.0118], 1: [0.0130]},
                )
            self.assertEqual(
                {
                    entry.config.num_warps: entry.perfs
                    for entry in search._terminal_refinement_members.values()
                },
                {1: [0.0130], 2: [0.0131], 3: [0.0118]},
            )
            self.assertEqual(search.best_perf_so_far, 0.0118)

    @pytest.mark.cpu_only
    def test_the_best_reading_does_not_keep_an_unread_members_events(self) -> None:
        """The re-timing threshold is a multiple of the best reading so far.
        The member that read best by CUDA events (7.6 us) gives no reading
        on the trace and leaves; the survivor reads 15.4 us there.  The
        best reading is the survivor's, so a verification still re-times
        it, and nothing else keeps the 7.6 us: not the registry the
        compiler refinement reuses readings from."""
        with patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search = self._search()
            search._terminal_refinement_members = ConfigCache()
            unread, survivor = (
                PopulationMember(
                    lambda: None,
                    [reading],
                    (index,),
                    helion.Config(num_warps=index),
                    status="ok",
                )
                for index, reading in ((1, 0.0076), (2, 0.0200))
            )
            self._worker_readings(
                search, {id(unread.fn): None, id(survivor.fn): 0.0154}
            )
            search.best_perf_so_far = 0.0076
            with patch.object(search, "_final_rebenchmark_top_k", return_value=8):
                for member in (unread, survivor):
                    search._record_benchmarked_member(member)
                search._settle_candidate_timer([unread, survivor])

            self.assertTrue(search.traces_candidates)
            self.assertEqual(search.best_perf_so_far, 0.0154)
            self.assertTrue(search.should_rebenchmark(survivor))
            self.assertEqual(
                [
                    entry.config.num_warps
                    for entry in search._terminal_refinement_members.values()
                ],
                [2],
            )

    @pytest.mark.cpu_only
    def test_a_search_without_a_verification_compares_on_one_timer(self) -> None:
        """A differential-evolution search has no pass that re-times its
        initial population. Its parents, 3.0-3.3 us kernels that read
        7.6-7.9 us by CUDA events, are read on the trace when the search
        settles on it, so children that are 1.5x slower by the trace do not
        replace them for reading below their parents' event readings."""
        latency = 0.0046
        search = object.__new__(DifferentialEvolutionSearch)
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=False,
            backend=SimpleNamespace(
                collapses_generated_source_aliases=lambda config_spec: True
            ),
        )
        search.settings = SimpleNamespace(
            get_rebenchmark_threshold=lambda: 1.5,
            autotune_budget_seconds=None,
            autotune_benchmark_fn=None,
        )
        search.log = Mock()
        search.short_kernel_below_us = 64.0
        search.population = []
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider._benchmark_worker = Mock()
        provider.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: None,
                should_deduplicate_generated_sources=lambda config_spec: True,
            )
        )
        provider._effective_source_results = {}
        search.benchmark_provider = provider
        search._kernel_timer_usable = lambda: True

        def member(us: float, index: int) -> PopulationMember:
            return PopulationMember(
                lambda: None, [us / 1000.0], (index,), helion.Config(num_warps=index)
            )

        parents = [member(3.0 + 0.1 * index + 4.6, index) for index in range(4)]
        children = [member(1.5 * (3.0 + 0.1 * index), 10 + index) for index in range(4)]
        search.population = list(parents)
        search.best_perf_so_far = parents[0].perf

        on_events = {id(parent) for parent in parents}

        def rebenchmark(members: list[PopulationMember], **kwargs: object) -> None:
            for timed in members:
                # the first trace reading of a parent leaves the latency out
                first = id(timed) in on_events
                on_events.discard(id(timed))
                again = timed.perf - (latency if first else 0.0)
                timed.perfs.append(again)
                search.best_perf_so_far = min(search.best_perf_so_far, again)

        search.rebenchmark = rebenchmark
        search._benchmarked_members = {}
        search._pinned_finalist_members = {}
        search._pinned_finalist_configs = set()
        search._final_rebenchmark_top_k = lambda: 8
        search.iter_candidates = lambda: iter(enumerate(children))
        with patch.dict(os.environ):
            os.environ.pop("HELION_AUTOTUNE_L2_FLUSH", None)
            search._settle_candidate_timer(parents)
        self.assertTrue(search.traces_candidates)
        self.assertAlmostEqual(search.best_perf_so_far, 0.003)
        self.assertEqual(search.evolve_population(), 0)
        self.assertEqual(search.population, parents)

    @pytest.mark.cpu_only
    def test_a_population_too_small_to_mutate_ends_the_evolution(self) -> None:
        """A differential-evolution mutation draws four members.  A
        population can be left with fewer (initial configs the trace could
        not read leave it; a budget cut does the same): the search then ends
        on what it has instead of failing in ``random.sample``."""
        for search_type in (DifferentialEvolutionSearch, DESurrogateHybrid):
            for size, evolves in ((3, False), (4, True)):
                with self.subTest(search_type=search_type.__name__, size=size):
                    search = object.__new__(search_type)
                    search.log = Mock()
                    search.population = [
                        PopulationMember(
                            lambda: None, [0.003 + index], (index,), helion.Config()
                        )
                        for index in range(size)
                    ]
                    search.population_size = 8
                    search.max_generations = 4
                    search.crossover_rate = 0.8
                    search.min_improvement_delta = None
                    search.patience = None
                    search.initial_population_strategy = (
                        InitialPopulationStrategy.FROM_RANDOM
                    )
                    search.compile_timeout_lower_bound = 0.0
                    search.compile_timeout_quantile = 0.0
                    search.surrogate_threshold = 100
                    search.candidate_ratio = 2
                    search.all_observations = []
                    stubs = {
                        "initial_two_generations": Mock(),
                        "set_adaptive_compile_timeout": Mock(),
                        "_budgeted_range": lambda *args: iter(range(*args)),
                        "set_generation": Mock(),
                        "evolve_population": Mock(return_value=0),
                        "_evolve_generation": Mock(),
                        "check_early_stopping": Mock(return_value=False),
                        "rebenchmark_population": Mock(),
                        "_finalize": Mock(return_value=helion.Config()),
                    }
                    for name, stub in stubs.items():
                        setattr(search, name, stub)

                    search._autotune()

                    evolved = (
                        search.evolve_population.called
                        or search._evolve_generation.called
                    )
                    self.assertEqual(evolved, evolves)
                    search._finalize.assert_called_once_with()
                    said = any(
                        "fewer than the four" in str(call.args[0])
                        for call in search.log.call_args_list
                        if call.args and isinstance(call.args[0], str)
                    )
                    self.assertEqual(said, not evolves)

    @pytest.mark.cpu_only
    def test_differential_evolution_stops_on_trace_ties(self) -> None:
        """The early stop of the two differential-evolution searches counts
        a move of the best reading within what a traced search's readings
        repeat to as no improvement. On the trace one of a generation's
        children usually reads a little below the best, which the fixed
        delta took for an improvement generation after generation (CUDA
        events read such children level, in steps of about 2 us). A best
        that drifts inside the tolerance ends a traced search after the
        patience and never one timed with CUDA events, a best that moves
        beyond it ends neither, and a search without a delta has no early
        stop on either timer. A delta of 0, which switches the stop off on
        the events, leaves the tolerance on the trace. Where the members
        were read twice (the comparisons of a plain differential evolution
        re-time them), three spreads of those re-readings count as well."""

        def stops_at(
            search_type: type[DifferentialEvolutionSearch],
            traces: bool,
            best: float,
            step: float,
            delta: float = 0.001,
            reread_step: float | None = None,
        ) -> tuple[int, Mock]:
            """The generation whose check ends a search whose best reading
            drops by ``step`` a generation (0: none of twelve does), and
            its log. With ``reread_step``, twenty members hold a second
            reading, -2 to 2 such steps from their first."""
            search = object.__new__(search_type)
            search.log = Mock()
            search.traces_candidates = traces
            search.min_improvement_delta = delta
            search.patience = 2
            search.generations_without_improvement = 0
            search._autotune_metrics = SimpleNamespace(num_generations=0)
            search.population = [
                PopulationMember(lambda: None, [perf], (index,), helion.Config())
                for index, perf in enumerate((best, 2 * best, 3 * best, 4 * best))
            ]
            if reread_step is not None:
                for index in range(20):
                    member = PopulationMember(
                        lambda: None, [2 * best], (20 + index,), helion.Config()
                    )
                    member.perfs.append(
                        2 * best * (1.0 + reread_step * (index % 5 - 2))
                    )
                    search.population.append(member)
            search.best_perf_history = [best]
            for generation in range(1, 13):
                # a child that reads lower takes the best member's place
                best *= 1.0 - step
                search.population[0] = PopulationMember(
                    lambda: None, [best], (10 + generation,), helion.Config()
                )
                search._autotune_metrics.num_generations = generation
                if search.check_early_stopping():
                    return generation, search.log
            return 0, search.log

        for search_type in (DifferentialEvolutionSearch, DESurrogateHybrid):
            with self.subTest(search_type=search_type.__name__):
                # 13 us: the 0.5% floor. 0.2% a generation is 0.4% over the
                # patience, a tie on the trace and four deltas on the events
                stopped, log = stops_at(search_type, True, 0.013, 0.002)
                self.assertEqual(stopped, 3)
                log.assert_called_once_with(
                    "Early stopping at generation 3: "
                    "no improvement >0.5% for 2 generations"
                )
                self.assertEqual(stops_at(search_type, False, 0.013, 0.002)[0], 0)
                # 2 us: a tick and a half of the trace's clock is 2.4%
                self.assertEqual(stops_at(search_type, True, 0.002, 0.008)[0], 3)
                self.assertEqual(stops_at(search_type, False, 0.002, 0.008)[0], 0)
                # beyond the tolerance: an improvement on either timer
                for traces in (True, False):
                    self.assertEqual(stops_at(search_type, traces, 0.013, 0.02)[0], 0)
                    # ... and inside the delta: none on either
                    self.assertEqual(stops_at(search_type, traces, 0.013, 0.0002)[0], 3)
                # a delta of 0: no early stop on the events, the readings'
                # repeatability on the trace
                self.assertEqual(
                    stops_at(search_type, True, 0.013, 0.002, delta=0.0)[0], 3
                )
                self.assertEqual(
                    stops_at(search_type, False, 0.013, 0.0, delta=0.0)[0], 0
                )
                # re-readings 0.8% apart: three spreads are 1.46%, so 0.5% a
                # generation (1% over the patience, twice the floor) is a tie
                # there and an improvement without them
                self.assertEqual(
                    stops_at(search_type, True, 0.013, 0.005, reread_step=0.004)[0],
                    3,
                )
                self.assertEqual(stops_at(search_type, True, 0.013, 0.005)[0], 0)
                self.assertEqual(
                    stops_at(search_type, False, 0.013, 0.005, reread_step=0.004)[0],
                    0,
                )
                # the tolerance is the traced searches' one definition, at
                # least the delta, and no early stop stays none
                search = object.__new__(search_type)
                search.population = []
                search.traces_candidates = True
                search.min_improvement_delta = 0.001
                self.assertEqual(
                    search._stop_tolerance(0.002), trace_tie_tolerance(search, 0.002)
                )
                self.assertAlmostEqual(search._stop_tolerance(0.002), 0.024)
                search.min_improvement_delta = 0.05
                self.assertEqual(search._stop_tolerance(0.002), 0.05)
                search.min_improvement_delta = None
                self.assertIsNone(search._stop_tolerance(0.002))
                search.traces_candidates = False
                self.assertIsNone(search._stop_tolerance(0.002))
                search.min_improvement_delta = 0.001
                self.assertEqual(search._stop_tolerance(0.002), 0.001)

    @pytest.mark.cpu_only
    def test_hybrid_generation_follows_the_populations_size(self) -> None:
        """The surrogate hybrid generates one candidate per member and
        compares them position by position.  A population that holds fewer
        than ``population_size`` members (initial configs the trace could
        not read left it; a budget cut does the same) gets as many
        candidates as it has members, with and without the surrogate, where
        ``population_size`` of them indexed past its end."""
        for use_surrogate in (False, True):
            with self.subTest(use_surrogate=use_surrogate):
                search = object.__new__(DESurrogateHybrid)
                search.log = Mock()
                search.format_performance = lambda perf: f"{perf:.4f}"
                search.population_size = 8
                search.candidate_ratio = 3
                search.refit_frequency = 1
                search.surrogate_threshold = 0 if use_surrogate else 100
                search.all_observations = []
                search.population = [
                    PopulationMember(
                        lambda: None, [0.004 + index], (index,), helion.Config()
                    )
                    for index in range(4)
                ]
                parents = list(search.population)
                generated: list[int] = []

                def generate(count: int, generated: list[int] = generated) -> list:
                    generated.append(count)
                    return [(100 + index,) for index in range(count)]

                search._generate_de_candidates = generate
                search._fit_surrogate = Mock()
                search._surrogate_select = lambda candidates, count: candidates[:count]
                search.benchmark_flat_batch = lambda flats: [
                    PopulationMember(lambda: None, [0.001], flat, helion.Config())
                    for flat in flats
                ]

                search._evolve_generation(2)

                self.assertEqual(generated, [12 if use_surrogate else 4])
                self.assertEqual(len(search.population), 4)
                # every member was replaced by its faster candidate
                self.assertTrue(
                    all(
                        new is not old
                        for new, old in zip(search.population, parents, strict=True)
                    )
                )
                self.assertEqual(len(search.all_observations), 4)

    @pytest.mark.cpu_only
    def test_the_bound_is_no_part_of_a_searchs_cache_policy(self) -> None:
        """A search reads its backend's bound when it is built, and no
        search's cache policy changes with it: the flash search, the one
        whose policy is hashed into its cache key and for which the backend
        names no bound (CuTe: its terminal refinement re-times members with
        mirrored wall-clock sweeps in this process), keeps its keys.  (The
        LLM-guided search reads no initial population, so it never reaches
        the timer choice whatever the bound:
        ``test_autotune_runs_full_llm_guided_loop_with_mocked_provider``.)"""

        def policy_and_flash_key(
            bound: float | None,
        ) -> tuple[dict[str, object], str]:
            kernel = mock_kernel(Settings(autotune_log_level=logging.CRITICAL))
            kernel.config_spec.backend.autotune_short_kernel_below_us = (
                lambda config_spec: bound
            )
            kernel.config_spec.compiler_seed_configs = ()
            search = PopulationBasedSearch(cast("_AutotunableKernel", kernel), ())
            self.assertEqual(search.short_kernel_below_us, bound)
            with (
                # (a registered algorithm with a keyable policy of its own)
                patch.dict(
                    helion.autotuner.search_algorithms,
                    {"stand-in": PopulationBasedSearch},
                ),
                patch.object(search, "_algorithm_cache_policy", return_value={}),
            ):
                policy = search.cache_policy()
                assert policy is not None
                return policy, _search_policy_hash(
                    search, cute_flash_search_enabled=True
                )

        self.assertEqual(policy_and_flash_key(64.0), policy_and_flash_key(None))


class TestPolishDescent(TestCase):
    def _make_search(
        self, polish_rounds: int, *, collapse: bool = False
    ) -> LFBOTreeSearch:
        search = object.__new__(LFBOTreeSearch)
        search.polish_rounds = polish_rounds
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=False,
            backend=SimpleNamespace(
                collapses_generated_source_aliases=lambda config_spec: collapse
            ),
        )
        search.benchmark_provider = SimpleNamespace(settles_source_aliases=lambda: True)
        search.min_improvement_delta = 0.001
        search.log = Mock()
        search._budgeted_range = lambda *args: iter(range(*args))
        search.set_generation = Mock()
        search._autotune_metrics = SimpleNamespace(num_generations=5)
        search.format_performance = lambda perf: f"{perf:.4f}"
        search.rebenchmark_population = Mock()
        return search

    @staticmethod
    def _member(perf: float | None, num_warps: int) -> PopulationMember:
        return PopulationMember(
            lambda: None,
            [] if perf is None else [perf],
            (num_warps,),
            helion.Config(num_warps=num_warps),
            status="ok",
        )

    def test_polish_descends_and_stops_when_no_round_improves(self) -> None:
        search = self._make_search(polish_rounds=10)
        start = self._member(10.0, 1)
        better = self._member(None, 2)
        visited_member = self._member(None, 3)
        worse = self._member(None, 4)
        search.population = [start]
        visited = {start.config, visited_member.config}

        # Round 1 proposes {better, visited_member}; visited_member must be
        # skipped. Round 2 (from better) proposes only worse -> stall -> stop.
        neighbors_by_round = [
            [better.flat_values, visited_member.flat_values],
            [worse.flat_values],
        ]
        members_by_flat = {
            better.flat_values: better,
            visited_member.flat_values: visited_member,
            worse.flat_values: worse,
        }
        perfs = {id(better): 8.0, id(worse): 11.0}

        def fake_benchmark(members, desc="") -> None:
            for member in members:
                member.perfs.append(perfs[id(member)])

        search.benchmark_population = fake_benchmark
        with (
            patch.object(
                PatternSearch,
                "_generate_neighbors",
                side_effect=lambda self_, base: neighbors_by_round[
                    0 if base == start.flat_values else 1
                ],
            ),
            patch.object(search, "make_unbenchmarked", side_effect=members_by_flat.get),
        ):
            search._polish_descent(visited)

        # Descended to the better neighbor and benchmarked only unvisited ones.
        self.assertIs(search.best, better)
        self.assertIn(better, search.population)
        self.assertIn(worse, search.population)
        self.assertNotIn(visited_member, search.population)
        self.assertEqual(len(better.perfs), 1)
        self.assertEqual(len(visited_member.perfs), 0)
        # Both proposed configs entered visited.
        self.assertIn(better.config, visited)
        self.assertIn(worse.config, visited)

    def _polish_rounds_on_ties(
        self,
        *,
        traces_candidates: bool,
        best_step: float,
        scale: float = 1.0,
        collapse: bool = False,
        patience: int | None = None,
    ) -> int:
        """The rounds a polish runs from an incumbent reading ``13 * scale``
        us when every round proposes sixteen neighbours whose two readings
        (the candidate's, then the round's re-timing) scatter within 0.4% of
        it and the best of them re-reads ``best_step`` below it. ``collapse``:
        the backend drops source aliases, so the descent crosses ties to new
        code within its ``patience``."""
        search = self._make_search(polish_rounds=4, collapse=collapse)
        if patience is not None:
            search.polish_patience = patience
        search.traces_candidates = traces_candidates
        start = self._member(0.013 * scale, 0)
        start.perfs.append(0.013 * scale)
        search.population = [start]
        rounds = 0
        created: dict[tuple[int, ...], PopulationMember] = {}

        def neighbors(self_: object, base: tuple[int, ...]) -> list[tuple[int, ...]]:
            nonlocal rounds
            rounds += 1
            return [(100 * rounds + index,) for index in range(16)]

        def make(flat: tuple[int, ...]) -> PopulationMember:
            created[flat] = self._member(None, flat[0])
            return created[flat]

        def first_reading(members: list[PopulationMember], desc: str = "") -> None:
            for member in members:
                jitter = (member.flat_values[0] % 9 - 4) * 0.001
                member.perfs.append(0.013 * scale * (1.0 + jitter))

        def second_reading(members: list[PopulationMember], desc: str = "") -> None:
            for member in members[1:]:
                jitter = (member.flat_values[0] % 7 - 3) * 0.001
                member.perfs.append(0.013 * scale * (1.0 + jitter))
            # the round's incumbent re-reads as it read; one neighbour steps
            members[0].perfs.append(members[0].perf)
            members[1].perfs[-1] = members[0].perf * (1.0 - best_step)

        search.benchmark_population = first_reading
        search.rebenchmark_population = second_reading
        with (
            patch.object(PatternSearch, "_generate_neighbors", neighbors),
            patch.object(search, "make_unbenchmarked", side_effect=make),
        ):
            search._polish_descent({start.config})
        return rounds

    @pytest.mark.cpu_only
    def test_polish_stops_on_a_step_within_the_readings_repeatability(self) -> None:
        """A search that reads its candidates from the trace
        (``traces_candidates``) ends its polish on the first round whose
        best neighbour is within the repeatability of the population's
        readings, where the fixed 0.1% rule walks every round; a step beyond
        it, and every step of a search timed with CUDA events, move as they
        always did."""
        # the neighbours re-read within 0.7%: three spreads are about 1.1%
        self.assertEqual(
            self._polish_rounds_on_ties(traces_candidates=True, best_step=0.006), 1
        )
        self.assertEqual(
            self._polish_rounds_on_ties(traces_candidates=False, best_step=0.006), 4
        )
        # a 130 us incumbent of a traced search: the same rule
        self.assertEqual(
            self._polish_rounds_on_ties(
                traces_candidates=True, best_step=0.006, scale=10.0
            ),
            1,
        )
        # a 5% step is an improvement under either rule
        self.assertEqual(
            self._polish_rounds_on_ties(traces_candidates=True, best_step=0.05), 4
        )

    @pytest.mark.cpu_only
    def test_polish_crosses_trace_ties_within_the_patience(self) -> None:
        """Where the backend drops source aliases, the same tie does not
        end a trace-timed polish: the descent moves to the new code that
        reads level and ends when ``polish_patience`` rounds brought no
        reading beyond the tolerance below its lowest. The tie rule says
        what an improvement is; the patience says when to stop."""
        rounds = self._polish_rounds_on_ties
        self.assertEqual(
            rounds(traces_candidates=True, best_step=0.006, collapse=True), 3
        )
        self.assertEqual(
            rounds(traces_candidates=True, best_step=0.006, collapse=True, patience=2),
            2,
        )
        # steps beyond the tolerance, and every 0.6% step of an event-timed
        # search (beyond its 0.1%), are new lowest readings: no patience used
        self.assertEqual(
            rounds(traces_candidates=True, best_step=0.05, collapse=True, patience=1),
            4,
        )
        self.assertEqual(
            rounds(traces_candidates=False, best_step=0.006, collapse=True, patience=1),
            4,
        )

    @pytest.mark.cpu_only
    def test_tie_tolerance_follows_the_rereading_spread(self) -> None:
        """The tolerance of a traced search is three spreads of the
        population's re-readings, at least the 0.5% floor (also when too few
        members were read twice to say); a search timed with CUDA events
        keeps the fixed delta whatever its population reads."""
        search = self._make_search(polish_rounds=1)
        search.traces_candidates = True
        incumbent = self._member(0.013, 0)

        def population(step: float, count: int = 20) -> list[PopulationMember]:
            members = []
            for index in range(count):
                member = self._member(0.013, index + 1)
                member.perfs.append(0.013 * (1.0 + step * (index % 5 - 2)))
                members.append(member)
            return members

        search.population = population(0.0001)
        self.assertEqual(search._reading_tolerance(incumbent.perf), 0.005)
        search.population = population(0.004)
        # steps of -0.8% .. 0.8%: nine in ten are within 0.8%, 1.645 deviations
        self.assertAlmostEqual(
            search._reading_tolerance(incumbent.perf), 3 * 0.008 / 1.6449, places=5
        )
        # a step inside the tolerance ends the path, one beyond it does not
        self.assertTrue(
            search._check_early_stopping(self._member(0.013 * 0.99, 99), incumbent)
        )
        self.assertFalse(
            search._check_early_stopping(self._member(0.013 * 0.97, 98), incumbent)
        )
        search.population = population(0.004, count=15)
        self.assertEqual(search._reading_tolerance(incumbent.perf), 0.005)
        search.population = population(0.004)
        search.traces_candidates = False
        self.assertEqual(search._reading_tolerance(incumbent.perf), 0.001)
        self.assertFalse(
            search._check_early_stopping(self._member(0.013 * 0.99, 97), incumbent)
        )

    @pytest.mark.cpu_only
    def test_tie_tolerance_covers_the_trace_clocks_tick(self) -> None:
        """Trace readings sit on 32 ns ticks, 1.2-1.6% of a 2-3 us kernel.
        Whether one re-reading in eight moves by a tick, half of them do, or
        none does, a one-tick step from such an incumbent is a tie and a
        two-tick step is an improvement: the tolerance is neither under one
        tick nor several ticks wide."""
        search = self._make_search(polish_rounds=1)
        search.traces_candidates = True
        tick = 0.000032

        def population(ticks: int, moved: int) -> list[PopulationMember]:
            """24 members first read at ``ticks`` ticks; the first ``moved``
            re-read one tick lower."""
            members = []
            for index in range(24):
                member = self._member(ticks * tick, index + 1)
                member.perfs.append((ticks - (index < moved)) * tick)
                members.append(member)
            return members

        # a 1.98 us kernel, a 2.78 us kernel, a 3.46 us kernel
        for ticks, moved in ((62, 3), (87, 12), (108, 0)):
            with self.subTest(ticks=ticks, moved=moved):
                search.population = population(ticks, moved)
                incumbent = self._member(ticks * tick, 0)
                tolerance = search._reading_tolerance(incumbent.perf)
                self.assertGreaterEqual(tolerance, 1.5 / ticks - 1e-12)
                self.assertLess(tolerance, 2.0 / ticks)
                self.assertTrue(
                    search._check_early_stopping(
                        self._member((ticks - 1) * tick, 98), incumbent
                    )
                )
                self.assertFalse(
                    search._check_early_stopping(
                        self._member((ticks - 2) * tick, 99), incumbent
                    )
                )
        # 13 us: a tick and a half is under the 0.5% floor
        search.population = population(406, 0)
        self.assertEqual(search._reading_tolerance(406 * tick), 0.005)

    @pytest.mark.cpu_only
    def test_rereading_spread_is_robust(self) -> None:
        """The spread is taken around the re-readings' median (a shift
        every re-reading shares is not noise), from the deviation nine in
        ten of them stay within (outliers and failed readings do not move
        it; readings that repeat to the bit do not zero it), and is zero
        below sixteen re-read members."""

        def members(steps: list[float]) -> list[PopulationMember]:
            return [
                PopulationMember(
                    lambda: None, [1.0, 1.0 + step], (index,), helion.Config()
                )
                for index, step in enumerate(steps)
            ]

        steps = [index * 0.001 for index in range(-8, 9)]
        # sorted deviations 0, 0.001 x2, ... 0.008 x2: position 14.4 of 16
        expected = 0.0074 / 1.6449
        self.assertAlmostEqual(rereading_spread(members(steps)), expected, places=6)
        shifted = [step + 0.05 for step in steps]
        self.assertAlmostEqual(rereading_spread(members(shifted)), expected, places=6)
        # one bimodal flip and one failed re-reading leave it where it was
        outliers = members([*steps, 1.0, 1.0])
        outliers[-1].perfs[-1] = math.inf
        self.assertAlmostEqual(rereading_spread(outliers), expected, delta=0.0005)
        # a quarter of the members move, the rest repeat to the bit
        quantized = [0.01] * 6 + [0.0] * 18
        self.assertAlmostEqual(
            rereading_spread(members(quantized)), 0.01 / 1.6449, places=6
        )
        self.assertEqual(rereading_spread(members(steps[:15])), 0.0)
        once = members(steps)
        for member in once:
            del member.perfs[0]
        self.assertEqual(rereading_spread(once), 0.0)

    @pytest.mark.cpu_only
    def test_rereading_spread_counts_readings_of_one_timer(self) -> None:
        """A member of the initial population holds a CUDA-event reading and
        then trace readings, 4.6 us apart for one kernel, and a config
        settled with a copy of an earlier candidate's reading holds that
        copy first. Neither pair says how a reading repeats: three of them
        among twenty kernels that repeat to 0.2% (more than the tenth the
        spread sets aside) leave the spread where it was, and each counts
        once the member has two readings of its own."""

        def member(
            index: int, perfs: list[float], readings_from: int = 0
        ) -> PopulationMember:
            return PopulationMember(
                lambda: None,
                perfs,
                (index,),
                helion.Config(),
                timer_readings_from=readings_from,
            )

        steady = [
            member(index, [0.003, 0.003 * (1.0 + 0.001 * (index % 5 - 2))])
            for index in range(20)
        ]
        expected = rereading_spread(steady)
        self.assertAlmostEqual(expected, 0.002 / 1.6449, places=6)
        switched = [
            member(100 + index, [0.0076 + 0.001 * index, 0.003 + 0.001 * index], 1)
            for index in range(3)
        ]
        self.assertEqual(rereading_spread([*steady, *switched]), expected)
        for late in switched:
            late.perfs.append(late.perf)
        self.assertEqual(rereading_spread([*steady, *switched]), expected)
        # counted as re-readings, the three pairs alone set the spread
        for late in switched:
            del late.perfs[-1]
            late.timer_readings_from = 0
        self.assertGreater(rereading_spread([*steady, *switched]), 0.1)

    def test_polish_disabled_or_cute_flash_is_a_no_op(self) -> None:
        for rounds, cute in ((0, False), (10, True)):
            search = self._make_search(polish_rounds=rounds)
            search.config_spec = SimpleNamespace(cute_flash_search_enabled=cute)
            start = self._member(10.0, 1)
            search.population = [start]
            with patch.object(PatternSearch, "_generate_neighbors") as generate:
                search._polish_descent({start.config})
            generate.assert_not_called()


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


@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
@onlyBackends(["triton", "cute"])
class TestSeedPopulationFallback(TestCase):
    """A seed-only initial population recovers when every seed fails to compile.

    Compiles and timings are stubbed: a no-op callable stands in for every
    compiled config and every surviving config ties at 1.0 ms, so each search
    runs end to end in a fraction of a second and only the fallback mechanics
    are under test.
    """

    # initial_population (PatternSearch, LFBOTreeSearch) and 2x population_size
    # (DifferentialEvolutionSearch) in _seed_only_searches.
    FALLBACK_TARGET = 8

    def _make_bound(self) -> tuple[object, tuple[torch.Tensor, torch.Tensor]]:
        # The baseline must not depend on the stubbed compile_config.
        @helion.kernel(
            autotune_log_level=0,
            autotune_precompile=None,
            autotune_benchmark_subprocess=False,
            autotune_baseline_fn=operator.add,
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
        return add.bind(args), args

    def _seed_configs(self, search: PopulationBasedSearch) -> set[helion.Config]:
        return {
            search.config_gen.canonicalize_flat(flat)[1]
            for flat in search._generate_initial_population_flat()
        }

    def _random_configs(
        self, search: PopulationBasedSearch, count: int
    ) -> list[helion.Config]:
        return [
            search.config_gen.unflatten(flat)
            for flat in search._pad_initial_population_with_unique_random([], count)
        ]

    @staticmethod
    def _stub_compile(
        should_fail: Callable[[helion.Config], bool],
    ) -> tuple[Callable[..., object], list[helion.Config]]:
        # Replaces bound.compile_config; nothing is really compiled.
        compiled: list[helion.Config] = []

        def compile_config(config: helion.Config, *, allow_print: bool = True):
            compiled.append(config)
            if should_fail(config):
                raise exc.BackendUnsupported("test", "seed config unsupported")
            return lambda *args, **kwargs: None

        return compile_config, compiled

    def _run(
        self,
        search: PopulationBasedSearch,
        bound: object,
        should_fail: Callable[[set[helion.Config], helion.Config], bool],
        batches: Mock | None = None,
        *,
        cached_seeds: int = 0,
    ) -> tuple[helion.Config, set[helion.Config], list[helion.Config], Mock, Mock]:
        # ``batches`` records benchmark_batch calls; callers expecting a raise
        # pass their own so it outlives the patch context.  ``cached_seeds``
        # stands in for best-config cache hits, which also join the seed set.
        if batches is None:
            batches = Mock(wraps=search.benchmark_batch)
        log = Mock(wraps=search.log)
        search._prepare()
        with patch.object(BaseSearch, "_find_similar_cached_configs", return_value=[]):
            if cached_seeds:
                search.set_best_available_seed_configs(
                    self._random_configs(search, cached_seeds)
                )
            seeds = self._seed_configs(search)
            compile_config, compiled = self._stub_compile(
                functools.partial(should_fail, seeds)
            )
            with (
                patch.object(bound, "compile_config", side_effect=compile_config),
                patch.object(
                    search.benchmark_provider, "_benchmark_function", return_value=1.0
                ),
                patch.object(search, "benchmark_batch", batches),
                patch.object(search, "log", log),
            ):
                best = search.autotune()
        return best, seeds, compiled, batches, log

    def _seed_only_searches(
        self, bound: object, args: tuple[torch.Tensor, torch.Tensor]
    ) -> list[PopulationBasedSearch]:
        # Mirrors the quick effort profile: FROM_BEST_AVAILABLE without random
        # padding, so the initial population is only seed/default configs.
        kwargs = {
            "initial_population_strategy": InitialPopulationStrategy.FROM_BEST_AVAILABLE,
            "best_available_pad_random": False,
        }
        return [
            PatternSearch(
                bound,
                args,
                initial_population=self.FALLBACK_TARGET,
                copies=1,
                max_generations=1,
                **kwargs,
            ),
            LFBOTreeSearch(
                bound,
                args,
                initial_population=self.FALLBACK_TARGET,
                copies=1,
                max_generations=1,
                **kwargs,
            ),
            DifferentialEvolutionSearch(
                bound,
                args,
                population_size=self.FALLBACK_TARGET // 2,
                max_generations=2,
                **kwargs,
            ),
        ]

    def _assert_fell_back_to_random(
        self,
        best: helion.Config,
        seeds: set[helion.Config],
        compiled: list[helion.Config],
        batches: Mock,
        log: Mock,
    ) -> None:
        target = self.FALLBACK_TARGET
        self.assertNotIn(best, seeds)
        self.assertTrue(set(compiled) >= seeds)
        self.assertGreaterEqual(batches.call_count, 2)
        initial_call, fallback_call = batches.call_args_list[:2]
        self.assertFalse(initial_call.kwargs["raise_if_no_viable_config"])
        self.assertEqual(fallback_call.kwargs["desc"], "Random fallback population")
        # The fallback holds exactly ``target`` new configs beyond the seeds,
        # however many seeds there were.
        fallback_configs = fallback_call.args[0]
        self.assertEqual(len(fallback_configs), target)
        self.assertEqual(len(set(fallback_configs)), target)
        self.assertTrue(seeds.isdisjoint(fallback_configs))
        warnings = [str(call.args[0]) for call in log.warning.call_args_list]
        self.assertTrue(
            any(
                message.startswith(
                    f"None of the {len(seeds)} seed/default/cache configs "
                    "compiled or ran"
                )
                and message.endswith(f"continuing with {target} random configs.")
                for message in warnings
            ),
            warnings,
        )

    def test_quick_profile_pads_seed_only_population(self) -> None:
        profile = get_effort_profile("quick")
        for search_config in (
            profile.pattern_search,
            profile.lfbo_pattern_search,
            profile.differential_evolution,
        ):
            assert search_config is not None
            self.assertEqual(
                search_config.initial_population_strategy, "from_best_available"
            )
            self.assertFalse(search_config.best_available_pad_random)
        spec = SimpleNamespace(cute_flash_search_enabled=False)
        flash_spec = SimpleNamespace(cute_flash_search_enabled=True)
        best_available = InitialPopulationStrategy.FROM_BEST_AVAILABLE
        self.assertEqual(
            random_fallback_population_target(best_available, False, spec, 30), 30
        )
        # Random padding, random populations, designed CuTe flash populations,
        # and empty targets keep the immediate compile-error re-raise.
        self.assertIsNone(
            random_fallback_population_target(best_available, True, spec, 30)
        )
        self.assertIsNone(
            random_fallback_population_target(
                InitialPopulationStrategy.FROM_RANDOM, False, spec, 30
            )
        )
        self.assertIsNone(
            random_fallback_population_target(best_available, False, flash_spec, 30)
        )
        self.assertIsNone(
            random_fallback_population_target(best_available, False, spec, 0)
        )

    def test_unsupported_seeds_fall_back_to_random_population(self) -> None:
        bound, args = self._make_bound()
        for search in self._seed_only_searches(bound, args):
            with self.subTest(search=type(search).__name__):
                best, seeds, compiled, batches, log = self._run(
                    search, bound, operator.contains
                )
                self.assertGreater(len(set(compiled)), len(seeds))
                self._assert_fell_back_to_random(best, seeds, compiled, batches, log)

    def test_fallback_pads_past_cached_seeds(self) -> None:
        # With at least FALLBACK_TARGET seeds (default plus cache hits) the
        # fallback must still add FALLBACK_TARGET new random configs instead of
        # padding only up to the target, which left nothing to benchmark.
        bound, args = self._make_bound()
        for search in self._seed_only_searches(bound, args):
            with self.subTest(search=type(search).__name__):
                best, seeds, compiled, batches, log = self._run(
                    search,
                    bound,
                    operator.contains,
                    cached_seeds=self.FALLBACK_TARGET + 2,
                )
                self.assertGreaterEqual(len(seeds), self.FALLBACK_TARGET)
                self._assert_fell_back_to_random(best, seeds, compiled, batches, log)

    def test_empty_fallback_reraises_compile_error(self) -> None:
        bound, args = self._make_bound()
        for search in self._seed_only_searches(bound, args):
            with self.subTest(search=type(search).__name__):
                batches = Mock(wraps=search.benchmark_batch)
                # Every random draw is the default config, already a seed, so
                # no fallback config survives deduplication.
                with (
                    patch.object(
                        search.config_gen,
                        "random_flat",
                        return_value=search.config_gen.default_flat(),
                    ),
                    self.assertRaisesRegex(
                        exc.BackendUnsupported, "seed config unsupported"
                    ),
                ):
                    self._run(search, bound, lambda seeds, config: True, batches)
                # Only the seed batch ran: its deferred compile error surfaced
                # instead of an empty fallback batch followed by NoConfigFound.
                self.assertEqual(batches.call_count, 1)
                self.assertFalse(
                    batches.call_args_list[0].kwargs["raise_if_no_viable_config"]
                )

    def test_fallback_population_failing_too_still_raises(self) -> None:
        bound, args = self._make_bound()
        for search in self._seed_only_searches(bound, args):
            with self.subTest(search=type(search).__name__):
                batches = Mock(wraps=search.benchmark_batch)
                with self.assertRaisesRegex(
                    exc.BackendUnsupported, "seed config unsupported"
                ):
                    self._run(search, bound, lambda seeds, config: True, batches)
                # The fallback batch was attempted before giving up.
                self.assertEqual(batches.call_count, 2)
                self.assertEqual(
                    batches.call_args_list[1].kwargs["desc"],
                    "Random fallback population",
                )

    def test_random_populations_keep_raising_without_fallback(self) -> None:
        bound, args = self._make_bound()
        # Full effort: FROM_RANDOM, plus FROM_BEST_AVAILABLE with random padding.
        searches = [
            PatternSearch(
                bound,
                args,
                initial_population=4,
                copies=1,
                max_generations=1,
                initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
            ),
            PatternSearch(
                bound,
                args,
                initial_population=4,
                copies=1,
                max_generations=1,
                initial_population_strategy=InitialPopulationStrategy.FROM_BEST_AVAILABLE,
                best_available_pad_random=True,
            ),
            DifferentialEvolutionSearch(
                bound,
                args,
                population_size=2,
                max_generations=2,
                initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
            ),
        ]
        for search in searches:
            with self.subTest(
                search=type(search).__name__,
                strategy=search.initial_population_strategy.name,  # pyrefly: ignore[missing-attribute]
            ):
                batches = Mock(wraps=search.benchmark_batch)
                with self.assertRaisesRegex(
                    exc.BackendUnsupported, "seed config unsupported"
                ):
                    self._run(search, bound, lambda seeds, config: True, batches)
                self.assertEqual(batches.call_count, 1)
                self.assertTrue(
                    batches.call_args_list[0].kwargs["raise_if_no_viable_config"]
                )


class TestSelectedSourceMetrics(TestCase):
    @staticmethod
    def _make_search(
        *,
        source_was_measured: bool,
        population_search: bool = True,
        source_tracking_enabled: bool = True,
    ) -> tuple[BaseSearch, helion.Config]:
        config = helion.Config(block_sizes=[64])
        fn = SimpleNamespace(source_hash="selected-source")
        search_type = PopulationBasedSearch if population_search else BaseSearch
        search = search_type.__new__(search_type)
        search.args = ()
        search.best_perf_so_far = 1.25
        search._autotune_metrics = AutotuneMetrics()
        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                generated_source_hash=lambda compiled: compiled.source_hash,
                should_deduplicate_generated_sources=(
                    lambda config_spec: source_tracking_enabled
                ),
            )
        )
        search.benchmark_provider = Mock()
        search.benchmark_provider.has_measured_source_hash.return_value = (
            source_was_measured
        )
        best = PopulationMember(
            fn=fn,
            perfs=[1.25],
            flat_values=[64],
            config=config,
            status="ok",
        )
        if population_search:
            search.population = [best]
        else:
            search.best = best
        return search, config

    def test_selected_source_records_verified_measurement(self) -> None:
        search, config = self._make_search(source_was_measured=True)

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        self.assertEqual(metrics.selected_source_hash, "selected-source")
        self.assertTrue(metrics.selected_source_was_measured)
        search.benchmark_provider.has_measured_source_hash.assert_called_once_with(
            "selected-source"
        )
        config.config["block_sizes"][0] = 128
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        serialized = metrics.to_dict()
        self.assertEqual(serialized["selected_config"], {"block_sizes": [64]})
        self.assertEqual(serialized["selected_source_hash"], "selected-source")
        self.assertTrue(serialized["selected_source_was_measured"])

    def test_selected_source_records_unmeasured_hash(self) -> None:
        search, config = self._make_search(source_was_measured=False)

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_source_hash, "selected-source")
        self.assertFalse(metrics.selected_source_was_measured)

    def test_selected_source_is_scoped_to_source_deduplication(self) -> None:
        search, config = self._make_search(
            source_was_measured=True,
            source_tracking_enabled=False,
        )

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        self.assertIsNone(metrics.selected_source_hash)
        self.assertFalse(metrics.selected_source_was_measured)
        search.benchmark_provider.has_measured_source_hash.assert_not_called()

    def test_selected_source_uses_finalist_after_noisy_population_best(self) -> None:
        search, config = self._make_search(source_was_measured=True)
        selected = search.population[0]
        stale_config = helion.Config(block_sizes=[128])
        stale = PopulationMember(
            fn=SimpleNamespace(source_hash="stale-source"),
            perfs=[0.5],
            flat_values=[128],
            config=stale_config,
            status="ok",
        )
        search.population = [stale]
        search._benchmarked_members = {config: selected}
        search._pinned_finalist_members = {}
        search.benchmark_provider.has_measured_source_hash.side_effect = (
            lambda source_hash: source_hash == "selected-source"
        )

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        self.assertEqual(metrics.selected_source_hash, "selected-source")
        self.assertTrue(metrics.selected_source_was_measured)

    def test_non_population_winner_leaves_source_unset(self) -> None:
        search, config = self._make_search(
            source_was_measured=True,
            population_search=False,
        )

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        self.assertIsNone(metrics.selected_source_hash)
        self.assertFalse(metrics.selected_source_was_measured)
        search.benchmark_provider.has_measured_source_hash.assert_not_called()

    def test_local_provider_requires_successful_finite_measurement(self) -> None:
        from helion.autotuner.benchmark_provider import BenchmarkResult

        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        config = helion.Config(block_sizes=[64])

        def fn() -> None:
            return None

        provider._effective_source_results = {
            "ok": BenchmarkResult(config, fn, 1.0, "ok", None),
            "deduplicated": BenchmarkResult(config, fn, 1.0, "deduplicated", None),
            "failed": BenchmarkResult(config, fn, math.inf, "error", None),
            "nonfinite": BenchmarkResult(config, fn, math.inf, "ok", None),
        }

        self.assertTrue(provider.has_measured_source_hash("ok"))
        self.assertTrue(provider.has_measured_source_hash("deduplicated"))
        self.assertFalse(provider.has_measured_source_hash("failed"))
        self.assertFalse(provider.has_measured_source_hash("nonfinite"))
        self.assertFalse(provider.has_measured_source_hash("missing"))


if __name__ == "__main__":
    unittest.main()
