from __future__ import annotations

import functools
import math
import operator
import os
import random
from types import SimpleNamespace
from typing import Callable
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
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import random_fallback_population_target
from helion.autotuner.random_search import RandomSearch
import helion.language as hl


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
        bound_kernel.settings.autotune_precompile = None
        # Smoke test of the RandomSearch loop itself; skip the benchmark
        # worker subprocess (seconds of startup overhead).
        bound_kernel.settings.autotune_benchmark_subprocess = False
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
        search.benchmark_provider = SimpleNamespace(take_effective_source_repairs=dict)
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


class TestPolishDescent(TestCase):
    def _make_search(self, polish_rounds: int) -> LFBOTreeSearch:
        search = object.__new__(LFBOTreeSearch)
        search.polish_rounds = polish_rounds
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=False)
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
