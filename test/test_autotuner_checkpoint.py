from __future__ import annotations

import functools
import math
import os
from pathlib import Path
import random
import tempfile
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.utils._ordered_set import OrderedSet

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import fork_rng
from helion._testing import seed_rng
from helion._testing import skipIfRefEager
from helion.autotuner import PatternSearch
from helion.autotuner import search_algorithms
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.base_search import _unset_fn
from helion.autotuner.utils import safe_globals
import helion.language as hl
from helion.runtime.config import Config

if TYPE_CHECKING:
    from typing import Any


class TestAutotunerCheckpoint(TestCase):
    # Algorithms excluded from checkpoint tests:
    # - FiniteSearch: Single-pass algorithm that benchmarks all configs in one call.
    #   No iterations/generations, so checkpointing doesn't apply.
    # - RandomSearch: Inherits from FiniteSearch, same single-pass behavior.
    #
    # If you are adding a new autotuner algorithm, please choose either of the following options:
    # Option 1 (Preferred): In your autotuner, call .set_generation() for each generation
    #                       (auto-checkpoint is already implemented in base class .set_generation()).
    #                       Then add checkpoint tests for it in _get_autotuner_configs() with a verify function.
    # Option 2: Add it to _checkpoint_excluded_algorithms with a comment explaining why.
    _checkpoint_excluded_algorithms: ClassVar[set[str]] = {
        "FiniteSearch",
        "RandomSearch",
    }

    def _get_autotuner_configs(self) -> dict[str, dict[str, Any]]:
        # Get algorithm classes from search_algorithms registry
        return {
            "PatternSearch": {
                "class": search_algorithms["PatternSearch"],
                "kwargs": {"initial_population": 3, "max_generations": 1, "copies": 1},
                "verify_fn": self._verify_PatternSearch,
            },
            "DifferentialEvolutionSearch": {
                "class": search_algorithms["DifferentialEvolutionSearch"],
                # Note: max_generations=3 needed so at least one evolution generation runs
                # (DE starts from generation 2 after initial population)
                "kwargs": {"population_size": 4, "max_generations": 3},
                "verify_fn": self._verify_DifferentialEvolutionSearch,
            },
            "LFBOPatternSearch": {
                "class": search_algorithms["LFBOPatternSearch"],
                "kwargs": {"initial_population": 3, "max_generations": 1, "copies": 1},
                "verify_fn": self._verify_LFBOPatternSearch,
            },
            "DESurrogateHybrid": {
                "class": search_algorithms["DESurrogateHybrid"],
                # Note: max_generations=3 needed so at least one evolution generation runs
                # (DE-based algorithms start from generation 2 after initial population)
                "kwargs": {"population_size": 4, "max_generations": 3},
                "verify_fn": self._verify_DESurrogateHybrid,
            },
            "LFBOTreeSearch": {
                "class": search_algorithms["LFBOTreeSearch"],
                "kwargs": {"initial_population": 3, "max_generations": 1, "copies": 1},
                "verify_fn": self._verify_LFBOPatternSearch,
            },
        }

    def test_all_algorithms_have_checkpoint_tests(self) -> None:
        """Verify all algorithms in search_algorithms are either tested or explicitly excluded."""
        tested_algos = set(self._get_autotuner_configs().keys())
        all_algos = set(search_algorithms.keys())
        expected_algos = all_algos - self._checkpoint_excluded_algorithms
        missing = expected_algos - tested_algos
        extra = tested_algos - all_algos
        self.assertFalse(missing, f"Missing checkpoint tests for algorithms: {missing}")
        self.assertFalse(extra, f"Unknown algorithms in checkpoint tests: {extra}")

    def setUp(self) -> None:
        super().setUp()
        random.seed(112)

    def _create_kernel_and_args(self) -> tuple[helion.Kernel, tuple[torch.Tensor, ...]]:
        @helion.kernel(autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([64], device=DEVICE),
            torch.randn([64], device=DEVICE),
        )
        return add, args

    # Keys in state_dict that are expected to differ between runs (not compared for values)
    _state_dict_skip_keys: ClassVar[set[str]] = {
        "rng_state",  # RNG state differs after running
        "best_perf_so_far",  # Timing measurement varies
        # Complex objects that are compared in algorithm-specific verify functions
        "visited",
        "search_copies",
        "population",
    }

    def _verify_state_dict_save_restore(
        self, search_a: BaseSearch, algorithm: str
    ) -> None:
        """Assert that state_dict() captures all state and load_state_dict() restores it.

        1. Verifies all instance attributes are either checkpointed or explicitly excluded
        2. Verifies state_dict() contains all expected keys for this algorithm
        3. Modifies all values in state_a, loads into a fresh search, verifies roundtrip
        """
        # Collect expected keys by walking MRO (uses class attributes on each autotuner class)
        expected_keys = type(search_a).get_all_checkpoint_state_dict_keys()

        # Collect excluded attrs by walking MRO
        excluded = type(search_a).get_all_checkpoint_excluded_attrs()

        # Check that all instance attributes are accounted for
        instance_attrs = set(vars(search_a).keys())
        uncovered = instance_attrs - expected_keys - excluded
        self.assertFalse(
            uncovered,
            f"{algorithm} has instance attributes not in state_dict or exclusion list: {uncovered}\n"
            f"Add to _checkpoint_state_dict_keys in the class if they should be checkpointed, "
            f"or to _checkpoint_excluded_attrs if they're derived/transient.",
        )

        state_a = search_a.state_dict()

        # Check that state_dict contains all expected keys
        actual_keys = set(state_a.keys())
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys

        self.assertFalse(
            missing_keys,
            f"{algorithm}.state_dict() is missing expected keys: {missing_keys}\n"
            f"If you are intentionally removing a key, please also remove it from {algorithm}.load_state_dict() "
            f"as well as update _checkpoint_state_dict_keys in the class",
        )
        self.assertFalse(
            extra_keys,
            f"{algorithm}.state_dict() has unexpected new keys: {extra_keys}\n"
            f"If you are intentionally adding a new key, please also add it to {algorithm}.load_state_dict() "
            f"as well as _checkpoint_state_dict_keys in the class",
        )

        # Modify all values in state_a to test that load_state_dict actually restores them
        modified_state = state_a.copy()
        modified_keys = []
        for key, val in list(modified_state.items()):
            if key in self._state_dict_skip_keys:
                continue
            if isinstance(val, bool):
                modified_state[key] = not val
                modified_keys.append(key)
            elif isinstance(val, int):
                modified_state[key] = val + 1000
                modified_keys.append(key)
            elif isinstance(val, float):
                modified_state[key] = 0.2 if val != 0.2 else 0.6
                modified_keys.append(key)

        # Create fresh search with default kwargs and load modified state
        fresh_search = type(search_a)(search_a.kernel, search_a.args)
        fresh_search.load_state_dict(modified_state)
        fresh_state = fresh_search.state_dict()

        # Verify all modified keys were restored correctly
        not_restored = []
        for key in modified_keys:
            expected = modified_state[key]
            actual = fresh_state.get(key)
            if expected != actual:
                not_restored.append(f"  {key}: expected {expected!r}, got {actual!r}")

        self.assertFalse(
            not_restored,
            f"{algorithm}.load_state_dict() did not restore fields:\n"
            + "\n".join(not_restored)
            + "\nMake sure load_state_dict() restores all fields saved by state_dict()",
        )

    def _assert_search_state_equivalent(
        self, search1: BaseSearch, search2: BaseSearch
    ) -> None:
        """Assert two searches have equivalent base state.

        Note: best_perf_so_far is not compared because it's a timing measurement
        that varies between runs due to rebenchmarking.
        """
        self.assertEqual(search1.counters, search2.counters)
        self.assertEqual(search1._current_generation, search2._current_generation)

    def _assert_population_equivalent(
        self,
        pop1: list[PopulationMember],
        pop2: list[PopulationMember],
    ) -> None:
        """Assert two populations have equivalent state (excluding fn and perfs).

        Note: perfs are not compared because they are timing measurements that
        vary between runs. The important state is configs and flat_values.
        """
        self.assertEqual(len(pop1), len(pop2))
        for m1, m2 in zip(pop1, pop2, strict=True):
            self.assertEqual(m1.flat_values, m2.flat_values)
            self.assertEqual(m1.config, m2.config)
            self.assertEqual(m1.status, m2.status)

    def _verify_PatternSearch(self, search_a: BaseSearch, search_b: BaseSearch) -> None:
        """Verify PatternSearch state and config generation determinism."""
        # Compare visited sets by config reprs (Config objects may differ after pickle)
        visited_a_reprs = {repr(c) for c in search_a.visited}  # type: ignore[attr-defined]
        visited_b_reprs = {repr(c) for c in search_b.visited}  # type: ignore[attr-defined]
        self.assertEqual(visited_a_reprs, visited_b_reprs)

        self.assertEqual(
            len(search_a.search_copies),
            len(search_b.search_copies),  # type: ignore[attr-defined]
        )
        for copy_a, copy_b in zip(
            search_a.search_copies,
            search_b.search_copies,
            strict=True,  # type: ignore[attr-defined]
        ):
            # Verify state
            self.assertEqual(copy_a.generation, copy_b.generation)
            self.assertEqual(copy_a.stopped, copy_b.stopped)
            self.assertEqual(copy_a.current.flat_values, copy_b.current.flat_values)
            self.assertEqual(copy_a.current.config, copy_b.current.config)

            # Verify determinism: temporarily override stopped to test candidate generation
            orig_stopped_a, orig_stopped_b = copy_a.stopped, copy_b.stopped
            copy_a.stopped, copy_b.stopped = False, False
            try:
                visited_a = set(search_a.visited)  # type: ignore[attr-defined]
                visited_b = set(search_b.visited)  # type: ignore[attr-defined]

                candidates_a = copy_a.generate_candidates(search_a, visited_a)
                candidates_b = copy_b.generate_candidates(search_b, visited_b)

                if candidates_a is None:
                    self.assertIsNone(candidates_b)
                else:
                    self.assertIsNotNone(candidates_b)
                    configs_a = [m.config for m in candidates_a]
                    configs_b = [m.config for m in candidates_b]
                    self.assertEqual(configs_a, configs_b)
            finally:
                copy_a.stopped, copy_b.stopped = orig_stopped_a, orig_stopped_b

    def _verify_DifferentialEvolutionSearch(
        self, search_a: BaseSearch, search_b: BaseSearch
    ) -> None:
        """Verify DifferentialEvolutionSearch state and mutate() determinism."""
        # Verify state
        self.assertEqual(
            search_a.best_perf_history,
            search_b.best_perf_history,  # type: ignore[attr-defined]
        )
        self.assertEqual(
            search_a.generations_without_improvement,  # type: ignore[attr-defined]
            search_b.generations_without_improvement,  # type: ignore[attr-defined]
        )

        # Verify population
        self._assert_population_equivalent(search_a.population, search_b.population)

        # Verify determinism
        num_rounds = 3
        for round_idx in range(num_rounds):
            seed = 42 + round_idx
            with fork_rng():
                seed_rng(seed)
                mutations_a = [
                    search_a.mutate(i)
                    for i in range(len(search_a.population))  # type: ignore[attr-defined]
                ]
            with fork_rng():
                seed_rng(seed)
                mutations_b = [
                    search_b.mutate(i)
                    for i in range(len(search_b.population))  # type: ignore[attr-defined]
                ]
            self.assertEqual(mutations_a, mutations_b, f"Mismatch at round {round_idx}")

    def _verify_LFBOPatternSearch(
        self, search_a: BaseSearch, search_b: BaseSearch
    ) -> None:
        """Verify LFBOPatternSearch state and config generation determinism."""
        # Verify base PatternSearch state (visited set)
        visited_a_reprs = {repr(c) for c in search_a.visited}  # type: ignore[attr-defined]
        visited_b_reprs = {repr(c) for c in search_b.visited}  # type: ignore[attr-defined]
        self.assertEqual(visited_a_reprs, visited_b_reprs)

        # Verify LFBO-specific state
        self.assertEqual(search_a.train_x, search_b.train_x)  # type: ignore[attr-defined]
        self.assertEqual(search_a.train_y, search_b.train_y)  # type: ignore[attr-defined]

        self.assertEqual(
            len(search_a.search_copies),
            len(search_b.search_copies),  # type: ignore[attr-defined]
        )

        # Verify state and determinism for each search copy
        num_rounds = 3
        for round_idx in range(num_rounds):
            seed = 42 + round_idx
            for copy_idx, (copy_a, copy_b) in enumerate(
                zip(
                    search_a.search_copies,
                    search_b.search_copies,
                    strict=True,  # type: ignore[attr-defined]
                )
            ):
                # Verify state
                if round_idx == 0:
                    self.assertEqual(copy_a.generation, copy_b.generation)
                    self.assertEqual(copy_a.stopped, copy_b.stopped)
                    self.assertEqual(
                        copy_a.current.flat_values, copy_b.current.flat_values
                    )
                    self.assertEqual(copy_a.current.config, copy_b.current.config)
                    self.assertEqual(
                        copy_a.patience_remaining, copy_b.patience_remaining
                    )

                # Verify determinism: temporarily override stopped to test candidate generation
                orig_stopped_a, orig_stopped_b = copy_a.stopped, copy_b.stopped
                copy_a.stopped, copy_b.stopped = False, False
                try:
                    visited_a = set(search_a.visited)  # type: ignore[attr-defined]
                    visited_b = set(search_b.visited)  # type: ignore[attr-defined]

                    with fork_rng():
                        seed_rng(seed)
                        candidates_a = copy_a.generate_candidates(search_a, visited_a)
                    with fork_rng():
                        seed_rng(seed)
                        candidates_b = copy_b.generate_candidates(search_b, visited_b)

                    if candidates_a is None:
                        self.assertIsNone(candidates_b)
                    else:
                        self.assertIsNotNone(candidates_b)
                        configs_a = [m.config for m in candidates_a]
                        configs_b = [m.config for m in candidates_b]
                        self.assertEqual(
                            configs_a,
                            configs_b,
                            f"Mismatch at round {round_idx}, copy {copy_idx}",
                        )
                finally:
                    copy_a.stopped, copy_b.stopped = orig_stopped_a, orig_stopped_b

    def _verify_DESurrogateHybrid(
        self, search_a: BaseSearch, search_b: BaseSearch
    ) -> None:
        """Verify DESurrogateHybrid state and _generate_de_candidates() determinism."""
        # Verify state
        self.assertEqual(
            search_a.all_observations,
            search_b.all_observations,  # type: ignore[attr-defined]
        )
        self.assertEqual(
            search_a.best_perf_history,
            search_b.best_perf_history,  # type: ignore[attr-defined]
        )
        self.assertEqual(
            search_a.generations_without_improvement,  # type: ignore[attr-defined]
            search_b.generations_without_improvement,  # type: ignore[attr-defined]
        )
        self._assert_population_equivalent(search_a.population, search_b.population)

        # Verify determinism
        num_rounds = 3
        for round_idx in range(num_rounds):
            seed = 42 + round_idx
            with fork_rng():
                seed_rng(seed)
                candidates_a = search_a._generate_de_candidates(4)  # type: ignore[attr-defined]
            with fork_rng():
                seed_rng(seed)
                candidates_b = search_b._generate_de_candidates(4)  # type: ignore[attr-defined]
            self.assertEqual(
                candidates_a, candidates_b, f"Mismatch at round {round_idx}"
            )

    @staticmethod
    def _create_deterministic_benchmark_fns() -> tuple[
        Callable[..., float], Callable[..., list[float]]
    ]:
        """Create deterministic benchmark functions that return consistent times.

        Uses the compiled function's code hash to generate deterministic times.
        The same config always produces the same benchmark time, regardless of
        call order. This eliminates the need for counter synchronization.

        Returns:
            A tuple of (do_bench_fn, interleaved_bench_fn) where:
            - do_bench_fn: Mock for do_bench (single benchmark)
            - interleaved_bench_fn: Mock for interleaved_bench (batch benchmark)
        """

        def _get_code_hash(fn: Callable[..., Any]) -> int:
            """Get a deterministic hash from the function's compiled code."""
            # fn is a functools.partial wrapping the compiled kernel
            if isinstance(fn, functools.partial):
                fn = fn.func
            # Hash the bytecode - different configs produce different bytecode
            return hash(fn.__code__.co_code)

        def _hash_to_time(code_hash: int) -> float:
            """Convert a code hash to a deterministic benchmark time."""
            # Map hash to range 0.5 to 5.0 ms
            return 0.5 + (abs(code_hash) * 17 % 45) / 10.0

        def deterministic_do_bench(
            fn: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
        ) -> float:
            # Generate a deterministic time based on the config's code hash
            return _hash_to_time(_get_code_hash(fn))

        def deterministic_interleaved_bench(
            fns: list[Callable[..., Any]],
            *args: Any,
            **kwargs: Any,
        ) -> list[float]:
            # Generate deterministic times for each function based on its code hash
            return [_hash_to_time(_get_code_hash(fn)) for fn in fns]

        return deterministic_do_bench, deterministic_interleaved_bench

    @parametrize(
        "autotuner",
        (
            "PatternSearch",
            "DifferentialEvolutionSearch",
            "LFBOPatternSearch",
            "DESurrogateHybrid",
            "LFBOTreeSearch",
        ),
    )
    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint(self, autotuner: str) -> None:
        config = self._get_autotuner_configs()[autotuner]
        search_class = config["class"]

        kernel, args = self._create_kernel_and_args()

        # Create deterministic benchmarks (same config always gets same time)
        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bound_kernel = kernel.bind(args)
            bound_kernel.settings.autotune_precompile = None
            bound_kernel.settings.autotune_progress_bar = False
            bound_kernel.settings.autotune_checkpoint_dir = tmpdir

            # Capture snapshots of the checkpoint file after each save
            # (since the file gets overwritten each generation)
            checkpoint_snapshots: list[tuple[int, bytes]] = []
            original_save_checkpoint = search_class.save_checkpoint

            def tracking_save_checkpoint(self: BaseSearch) -> Path | None:
                path = original_save_checkpoint(self)
                if path is not None and path.exists():
                    checkpoint_snapshots.append(
                        (self._current_generation, path.read_bytes())
                    )
                return path

            # Track set_generation() calls to verify generations progress
            set_generation_calls: list[int] = []
            original_set_generation = search_class.set_generation

            def tracking_set_generation(self: BaseSearch, generation: int) -> None:
                set_generation_calls.append(generation)
                return original_set_generation(self, generation)

            # Run autotuner A with patched methods
            search_a = search_class(bound_kernel, args, **config["kwargs"])
            with (
                patch.object(search_class, "save_checkpoint", tracking_save_checkpoint),
                patch.object(search_class, "set_generation", tracking_set_generation),
                # Prevent cleanup so we can inspect the file
                patch.object(search_class, "_cleanup_checkpoint", lambda self: None),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                search_a.autotune()

            # Verify real autotuning happened
            self.assertGreater(search_a.counters["benchmark"], 0)
            self.assertTrue(math.isfinite(search_a.best_perf_so_far))
            self.assertGreater(len(search_a.population), 0)

            # Verify set_generation() was called at least once
            self.assertGreater(
                len(set_generation_calls),
                0,
                f"{autotuner} did not call set_generation() during autotune",
            )
            # Verify generations are sequential (non-decreasing)
            for i in range(1, len(set_generation_calls)):
                prev_gen = set_generation_calls[i - 1]
                curr_gen = set_generation_calls[i]
                self.assertGreaterEqual(
                    curr_gen,
                    prev_gen,
                    f"{autotuner} generation went backwards: {prev_gen} -> {curr_gen}",
                )

            # Auto-checkpointing should have saved at least one checkpoint
            self.assertGreater(
                len(checkpoint_snapshots),
                0,
                "No checkpoints were saved during autotune",
            )

            # Use a MID-RUN checkpoint snapshot to properly test resumption semantics.
            mid_index = len(checkpoint_snapshots) // 2
            mid_gen, mid_bytes = checkpoint_snapshots[mid_index]

            # Record final result from search_a for comparison
            best_config_a = search_a.best.config

            # Place the mid-run snapshot into the checkpoint dir for run B
            stable_hash = search_a._get_stable_hash()
            checkpoint_file = Path(tmpdir) / f"{stable_hash}.pt"
            checkpoint_file.write_bytes(mid_bytes)

            # Create new search with same checkpoint dir - tests auto-discover
            bound_kernel_b = kernel.bind(args)
            bound_kernel_b.settings.autotune_precompile = None
            bound_kernel_b.settings.autotune_progress_bar = False
            bound_kernel_b.settings.autotune_checkpoint_dir = tmpdir

            search_b = search_class(bound_kernel_b, args, **config["kwargs"])

            # Track whether _init_search was called (it should NOT be when resuming)
            init_search_called = False
            original_init_search = search_class._init_search

            def tracking_init_search(self: BaseSearch) -> None:
                nonlocal init_search_called
                init_search_called = True
                return original_init_search(self)

            with (
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
                patch.object(search_class, "_init_search", tracking_init_search),
            ):
                best_config_b = search_b.autotune()

            # Verify search_b did NOT call _init_search (checkpoint loaded)
            self.assertFalse(
                init_search_called,
                "Resumed search should not call _init_search. "
                "Checkpoint was not loaded correctly.",
            )

            # Verify search_b's generation is at or beyond the checkpoint
            self.assertGreaterEqual(
                search_b._current_generation,
                mid_gen,
                f"Resumed search should be at or beyond checkpoint generation. "
                f"Checkpoint generation: {mid_gen}, "
                f"search_b generation: {search_b._current_generation}",
            )

            # Verify restored search produces same final result as uninterrupted run.
            self.assertEqual(
                best_config_a,
                best_config_b,
                f"Restored search should produce same best config as uninterrupted run.\n"
                f"search_a best: {best_config_a}\n"
                f"search_b best: {best_config_b}\n"
                f"Checkpoint used: snapshot {mid_index} of {len(checkpoint_snapshots)}",
            )

            # Verify restored search completed successfully
            self.assertGreater(len(search_b.population), 0)
            self.assertTrue(math.isfinite(search_b.best_perf_so_far))

            # Verify state_dict roundtrip
            self._verify_state_dict_save_restore(search_a, autotuner)

            # Verify algorithm-specific state
            config["verify_fn"](search_a, search_b)

            # Verify kernel functions were recompiled and work
            for member in search_b.population:
                if member.status == "ok":
                    self.assertIsNot(member.fn, _unset_fn)
                    result = member.fn(*args)
                    self.assertIsNotNone(result)

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_different_kernel_starts_fresh(self) -> None:
        """Test that a different kernel starts fresh when sharing a checkpoint dir.

        Different kernels have different stable hashes, so they write to
        different files in the same directory. No interference, no errors.
        """

        @helion.kernel(autotune_log_level=0)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        @helion.kernel(autotune_log_level=0)
        def mul_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] * 3.0
            return out

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # === FIRST RUN: add_kernel autotunes and saves checkpoint ===
            args_add = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_add = add_kernel.bind(args_add)
            bound_add.settings.autotune_precompile = None
            bound_add.settings.autotune_progress_bar = False
            bound_add.settings.autotune_checkpoint_dir = tmpdir

            search_add = PatternSearch(
                bound_add, args_add, initial_population=3, max_generations=1, copies=1
            )

            with (
                # Prevent cleanup so checkpoint remains for inspection
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                search_add.autotune()

            # Verify add_kernel's checkpoint exists
            add_hash = search_add._get_stable_hash()
            self.assertTrue((Path(tmpdir) / f"{add_hash}.pt").exists())

            # === SECOND RUN: mul_kernel with same dir should start fresh ===
            args_mul = (torch.randn([64], device=DEVICE),)
            bound_mul = mul_kernel.bind(args_mul)
            bound_mul.settings.autotune_precompile = None
            bound_mul.settings.autotune_progress_bar = False
            bound_mul.settings.autotune_checkpoint_dir = tmpdir

            search_mul = PatternSearch(
                bound_mul, args_mul, initial_population=3, max_generations=1, copies=1
            )

            # Track whether _init_search was called (it SHOULD be for fresh start)
            init_search_called = False
            original_init_search = PatternSearch._init_search

            def tracking_init_search(self: BaseSearch) -> None:
                nonlocal init_search_called
                init_search_called = True
                return original_init_search(self)

            with (
                patch.object(PatternSearch, "_init_search", tracking_init_search),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                result_mul = search_mul.autotune()

            # mul_kernel should have started fresh (no matching checkpoint)
            self.assertTrue(
                init_search_called,
                "mul_kernel should start fresh when no matching checkpoint exists",
            )
            self.assertIsNotNone(result_mul)

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_empty_dir_starts_fresh(self) -> None:
        """Test that an empty checkpoint dir means fresh start (no error)."""
        kernel, args = self._create_kernel_and_args()

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bound = kernel.bind(args)
            bound.settings.autotune_precompile = None
            bound.settings.autotune_progress_bar = False
            bound.settings.autotune_checkpoint_dir = tmpdir

            search = PatternSearch(
                bound, args, initial_population=3, max_generations=1, copies=1
            )

            init_search_called = False
            original_init_search = PatternSearch._init_search

            def tracking_init_search(self: BaseSearch) -> None:
                nonlocal init_search_called
                init_search_called = True
                return original_init_search(self)

            with (
                patch.object(PatternSearch, "_init_search", tracking_init_search),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                result = search.autotune()

            # Should complete successfully with a fresh start
            self.assertTrue(init_search_called)
            self.assertIsNotNone(result)

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_same_kernel_resumes(self) -> None:
        """Test that using a checkpoint from the same kernel resumes correctly."""

        @helion.kernel(autotune_log_level=0)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # === FIRST RUN: autotune and save checkpoint ===
            args_add = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_add = add_kernel.bind(args_add)
            bound_add.settings.autotune_precompile = None
            bound_add.settings.autotune_progress_bar = False
            bound_add.settings.autotune_checkpoint_dir = tmpdir

            search_add = PatternSearch(
                bound_add, args_add, initial_population=3, max_generations=1, copies=1
            )

            with (
                # Prevent cleanup so checkpoint remains for run B
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                search_add.autotune()

            # Verify checkpoint was saved
            stable_hash = search_add._get_stable_hash()
            checkpoint_file = Path(tmpdir) / f"{stable_hash}.pt"
            self.assertTrue(checkpoint_file.exists())

            # === SECOND RUN: same kernel should resume from checkpoint ===
            add_kernel.reset()
            args_add_v2 = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_add_v2 = add_kernel.bind(args_add_v2)
            bound_add_v2.settings.autotune_precompile = None
            bound_add_v2.settings.autotune_progress_bar = False
            bound_add_v2.settings.autotune_checkpoint_dir = tmpdir

            search_add_v2 = PatternSearch(
                bound_add_v2,
                args_add_v2,
                initial_population=3,
                max_generations=1,
                copies=1,
            )

            init_search_called = False
            original_init_search = search_add_v2._init_search

            def track_init_search() -> None:
                nonlocal init_search_called
                init_search_called = True
                original_init_search()

            with (
                patch.object(search_add_v2, "_init_search", track_init_search),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                result_add = search_add_v2.autotune()

            # Verify checkpoint was restored (not started fresh)
            self.assertFalse(
                init_search_called,
                "add_kernel should resume from checkpoint, not call _init_search",
            )
            self.assertIsNotNone(result_add)

    def test_checkpoint_env_var_propagates_to_kernel_settings(self) -> None:
        """Test that HELION_AUTOTUNE_CHECKPOINT_DIR env var is picked up by kernel settings."""
        checkpoint_dir = "/tmp/test_checkpoint_dir"

        with patch.dict(
            os.environ,
            {"HELION_AUTOTUNE_CHECKPOINT_DIR": checkpoint_dir},
            clear=False,
        ):

            @helion.kernel()
            def test_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] * 2.0
                return out

            self.assertEqual(
                test_kernel.settings.autotune_checkpoint_dir,
                checkpoint_dir,
                "Kernel settings should pick up HELION_AUTOTUNE_CHECKPOINT_DIR env var",
            )

        # Verify kernel defined outside the patch does NOT have the dir
        @helion.kernel()
        def test_kernel_no_env(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] * 2.0
            return out

        self.assertIsNone(
            test_kernel_no_env.settings.autotune_checkpoint_dir,
            "Kernel defined without env var should have checkpoint_dir=None",
        )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_corrupted_file_raises_error(self) -> None:
        """Test that a corrupted checkpoint file raises CheckpointError."""
        kernel, args = self._create_kernel_and_args()
        bound_kernel = kernel.bind(args)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a search to get the stable hash
            search = PatternSearch(bound_kernel, args, initial_population=2)
            stable_hash = search._get_stable_hash()

            # Write corrupted data to the checkpoint file
            checkpoint_file = Path(tmpdir) / f"{stable_hash}.pt"
            checkpoint_file.write_bytes(b"this is not a valid pickle file")

            # Configure kernel to use this checkpoint dir
            bound_kernel.settings.autotune_checkpoint_dir = tmpdir

            search = PatternSearch(bound_kernel, args, initial_population=2)
            with self.assertRaises(exc.CheckpointError) as ctx:
                search.autotune()

            # Verify error message is helpful
            error_msg = str(ctx.exception).lower()
            self.assertTrue(
                "corrupted" in error_msg or "failed to load" in error_msg,
                f"Error message should mention corruption: {ctx.exception}",
            )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_missing_keys_raises_error(self) -> None:
        """Test that checkpoint with missing keys raises CheckpointError."""
        kernel, args = self._create_kernel_and_args()

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bound_kernel = kernel.bind(args)
            bound_kernel.settings.autotune_precompile = None
            bound_kernel.settings.autotune_progress_bar = False
            bound_kernel.settings.autotune_checkpoint_dir = tmpdir

            # First, create a valid checkpoint
            search_a = PatternSearch(
                bound_kernel, args, initial_population=2, max_generations=1
            )
            with (
                # Prevent cleanup so checkpoint remains
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                search_a.autotune()

            stable_hash = search_a._get_stable_hash()
            checkpoint_path = Path(tmpdir) / f"{stable_hash}.pt"
            self.assertTrue(checkpoint_path.exists())

            # Load the checkpoint and remove a required key
            with safe_globals(Config, OrderedSet):
                state = torch.load(checkpoint_path, weights_only=True)
            del state["counters"]  # Remove required key
            torch.save(state, checkpoint_path)

            # Try to load the corrupted checkpoint
            bound_kernel_b = kernel.bind(args)
            bound_kernel_b.settings.autotune_precompile = None
            bound_kernel_b.settings.autotune_progress_bar = False
            bound_kernel_b.settings.autotune_checkpoint_dir = tmpdir

            search_b = PatternSearch(bound_kernel_b, args, initial_population=2)
            with self.assertRaises(exc.CheckpointError) as ctx:
                search_b.autotune()

            # Verify error message mentions the missing field
            error_msg = str(ctx.exception)
            self.assertIn(
                "counters",
                error_msg,
                f"Error message should mention missing field 'counters': {ctx.exception}",
            )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_truncated_file_raises_error(self) -> None:
        """Test that a truncated checkpoint file raises CheckpointError."""
        kernel, args = self._create_kernel_and_args()

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bound_kernel = kernel.bind(args)
            bound_kernel.settings.autotune_precompile = None
            bound_kernel.settings.autotune_progress_bar = False
            bound_kernel.settings.autotune_checkpoint_dir = tmpdir

            # First, create a valid checkpoint
            search_a = PatternSearch(
                bound_kernel, args, initial_population=2, max_generations=1
            )
            with (
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                search_a.autotune()

            stable_hash = search_a._get_stable_hash()
            checkpoint_path = Path(tmpdir) / f"{stable_hash}.pt"
            self.assertTrue(checkpoint_path.exists())

            # Truncate the checkpoint file
            full_bytes = checkpoint_path.read_bytes()
            truncated_bytes = full_bytes[: len(full_bytes) // 2]
            checkpoint_path.write_bytes(truncated_bytes)

            # Try to load the truncated checkpoint
            bound_kernel_b = kernel.bind(args)
            bound_kernel_b.settings.autotune_precompile = None
            bound_kernel_b.settings.autotune_progress_bar = False
            bound_kernel_b.settings.autotune_checkpoint_dir = tmpdir

            search_b = PatternSearch(bound_kernel_b, args, initial_population=2)
            with self.assertRaises(exc.CheckpointError) as ctx:
                search_b.autotune()

            # Verify error message is helpful
            error_msg = str(ctx.exception).lower()
            self.assertTrue(
                "corrupted" in error_msg or "failed to load" in error_msg,
                f"Error message should indicate file corruption: {ctx.exception}",
            )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_stopped_search_copy_not_serialized(self) -> None:
        """Test that stopped search copies are excluded from checkpoint serialization."""
        kernel, args = self._create_kernel_and_args()
        bound_kernel = kernel.bind(args)
        bound_kernel.settings.autotune_precompile = None
        bound_kernel.settings.autotune_progress_bar = False
        bound_kernel.settings.autotune_checkpoint_dir = None
        bound_kernel.settings.autotune_ignore_errors = True

        search = PatternSearch(
            bound_kernel,
            args,
            initial_population=3,
            copies=2,
            max_generations=1,
        )
        search._prepare()

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with (
            patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
            patch(
                "helion.autotuner.base_search.interleaved_bench",
                deterministic_interleaved_bench,
            ),
        ):
            # Initialize the search (creates population and search_copies)
            search._init_search()

        # Verify we have the expected setup
        self.assertEqual(len(search.search_copies), 2)
        self.assertGreaterEqual(len(search.population), 2)

        # Initially, no copies are stopped - both should be serialized
        state_before = search.state_dict()
        self.assertEqual(
            len(state_before["search_copies"]),
            2,
            "Both copies should be serialized when none are stopped",
        )

        # Now mark the second copy as stopped
        search.search_copies[1].stopped = True

        # Get state_dict - should only contain the non-stopped copy
        state_after = search.state_dict()
        self.assertIn("search_copies", state_after)
        self.assertEqual(
            len(state_after["search_copies"]),
            1,
            "Only non-stopped copies should be serialized",
        )

        # The serialized copy should be the first one (not stopped)
        serialized_copy = state_after["search_copies"][0]
        self.assertIn("current_index", serialized_copy)
        self.assertLess(
            serialized_copy["current_index"],
            len(search.population),
            "current_index should be valid",
        )
        self.assertFalse(
            serialized_copy["stopped"],
            "Serialized copy should not be stopped",
        )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_not_saved_when_dir_unset(self) -> None:
        """Test that no checkpoints are saved when autotune_checkpoint_dir is None."""
        kernel, args = self._create_kernel_and_args()
        bound_kernel = kernel.bind(args)
        bound_kernel.settings.autotune_precompile = None
        bound_kernel.settings.autotune_progress_bar = False
        bound_kernel.settings.autotune_checkpoint_dir = None  # explicitly unset

        save_results: list[Path | None] = []
        original_save = PatternSearch.save_checkpoint

        def tracking_save(self: BaseSearch) -> Path | None:
            result = original_save(self)
            save_results.append(result)
            return result

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        search = PatternSearch(
            bound_kernel, args, initial_population=3, max_generations=1, copies=1
        )
        with (
            patch.object(PatternSearch, "save_checkpoint", tracking_save),
            patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
            patch(
                "helion.autotuner.base_search.interleaved_bench",
                deterministic_interleaved_bench,
            ),
        ):
            search.autotune()

        # All save_checkpoint calls should have returned None (no-op)
        self.assertTrue(
            all(r is None for r in save_results),
            "save_checkpoint should return None when dir is unset",
        )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_cleanup_on_completion(self) -> None:
        """Test that checkpoint file is deleted on successful autotune completion."""
        kernel, args = self._create_kernel_and_args()

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bound_kernel = kernel.bind(args)
            bound_kernel.settings.autotune_precompile = None
            bound_kernel.settings.autotune_progress_bar = False
            bound_kernel.settings.autotune_checkpoint_dir = tmpdir

            search = PatternSearch(
                bound_kernel, args, initial_population=3, max_generations=1, copies=1
            )

            with (
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                search.autotune()

            # After completion, checkpoint file should be cleaned up
            stable_hash = search._get_stable_hash()
            checkpoint_file = Path(tmpdir) / f"{stable_hash}.pt"
            self.assertFalse(
                checkpoint_file.exists(),
                "Checkpoint file should be deleted after successful completion",
            )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_multiple_kernels_same_dir(self) -> None:
        """Two kernels using same checkpoint dir write different files, no interference."""

        @helion.kernel(autotune_log_level=0)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        @helion.kernel(autotune_log_level=0)
        def mul_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] * 3.0
            return out

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run add_kernel
            args_add = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_add = add_kernel.bind(args_add)
            bound_add.settings.autotune_precompile = None
            bound_add.settings.autotune_progress_bar = False
            bound_add.settings.autotune_checkpoint_dir = tmpdir

            search_add = PatternSearch(
                bound_add, args_add, initial_population=3, max_generations=1, copies=1
            )
            with (
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                search_add.autotune()

            # Run mul_kernel with same dir
            args_mul = (torch.randn([64], device=DEVICE),)
            bound_mul = mul_kernel.bind(args_mul)
            bound_mul.settings.autotune_precompile = None
            bound_mul.settings.autotune_progress_bar = False
            bound_mul.settings.autotune_checkpoint_dir = tmpdir

            search_mul = PatternSearch(
                bound_mul, args_mul, initial_population=3, max_generations=1, copies=1
            )
            with (
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                result_mul = search_mul.autotune()

            # Both should complete successfully
            self.assertIsNotNone(result_mul)

            # Verify they wrote different checkpoint files
            add_hash = search_add._get_stable_hash()
            mul_hash = search_mul._get_stable_hash()
            self.assertNotEqual(add_hash, mul_hash)
            self.assertTrue((Path(tmpdir) / f"{add_hash}.pt").exists())
            self.assertTrue((Path(tmpdir) / f"{mul_hash}.pt").exists())

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_checkpoint_force_autotune_still_resumes(self) -> None:
        """force_autotune + checkpoint dir still resumes from checkpoint.

        force_autotune only skips the result cache, not checkpoints.
        """
        kernel, args = self._create_kernel_and_args()

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run A: generate a checkpoint
            bound_a = kernel.bind(args)
            bound_a.settings.autotune_precompile = None
            bound_a.settings.autotune_progress_bar = False
            bound_a.settings.autotune_checkpoint_dir = tmpdir

            search_a = PatternSearch(
                bound_a, args, initial_population=3, max_generations=1, copies=1
            )
            with (
                # Prevent cleanup so checkpoint remains
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                search_a.autotune()

            # Verify checkpoint exists
            stable_hash = search_a._get_stable_hash()
            self.assertTrue((Path(tmpdir) / f"{stable_hash}.pt").exists())

            # Run B: force_autotune + same checkpoint dir
            kernel.reset()
            bound_b = kernel.bind(args)
            bound_b.settings.autotune_precompile = None
            bound_b.settings.autotune_progress_bar = False
            bound_b.settings.autotune_checkpoint_dir = tmpdir
            # force_autotune is handled at the cache layer, not in BaseSearch.autotune()
            # But we verify that _try_load_checkpoint doesn't check force_autotune

            search_b = PatternSearch(
                bound_b, args, initial_population=3, max_generations=1, copies=1
            )

            init_search_called = False
            original_init_search = PatternSearch._init_search

            def tracking_init_search(self: BaseSearch) -> None:
                nonlocal init_search_called
                init_search_called = True
                return original_init_search(self)

            with (
                patch.object(PatternSearch, "_init_search", tracking_init_search),
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                search_b.autotune()

            self.assertFalse(
                init_search_called,
                "force_autotune should not prevent checkpoint resumption",
            )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_lfbo_checkpoint_current_filtered_by_surrogate(self) -> None:
        """Regression test: save_checkpoint must not KeyError when
        _surrogate_select filters search_copy.current out of candidates.

        The bug: LFBOPatternSearch._autotune() builds new_population from
        candidates returned by generate_candidates(), but _surrogate_select
        can drop self.current from the candidate list. When set_generation()
        triggers save_checkpoint() → state_dict() → to_dict(), it builds
        member_id_to_idx from self.population (which doesn't include the
        dropped current member), causing KeyError on id(self.current).
        """
        kernel, args = self._create_kernel_and_args()

        deterministic_do_bench, deterministic_interleaved_bench = (
            self._create_deterministic_benchmark_fns()
        )

        from helion.autotuner import LFBOPatternSearch

        with tempfile.TemporaryDirectory() as tmpdir:
            bound = kernel.bind(args)
            bound.settings.autotune_precompile = None
            bound.settings.autotune_progress_bar = False
            bound.settings.autotune_checkpoint_dir = tmpdir

            search = LFBOPatternSearch(
                bound,
                args,
                initial_population=3,
                max_generations=3,
                copies=2,
            )

            with (
                patch("helion.autotuner.base_search.do_bench", deterministic_do_bench),
                patch(
                    "helion.autotuner.base_search.interleaved_bench",
                    deterministic_interleaved_bench,
                ),
            ):
                # This must not raise KeyError in to_dict()
                result = search.autotune()

            self.assertIsNotNone(result)


instantiate_parametrized_tests(TestAutotunerCheckpoint)


if __name__ == "__main__":
    import unittest

    unittest.main()
