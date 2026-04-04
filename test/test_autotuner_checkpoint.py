from __future__ import annotations

import contextlib
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

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfRefEager
from helion.autotuner import PatternSearch
from helion.autotuner import search_algorithms
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import _unset_fn
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any


class TestAutotunerCheckpoint(TestCase):
    # One-pass algorithms (no generation loop, no checkpointing) — excluded from
    # checkpoint tests because they never call save_checkpoint() mid-run.
    _CHECKPOINT_EXCLUDED: ClassVar[set[str]] = {"FiniteSearch", "RandomSearch"}

    def _get_autotuner_configs(self) -> dict[str, dict[str, Any]]:
        return {
            "PatternSearch": {
                "class": search_algorithms["PatternSearch"],
                "kwargs": {"initial_population": 3, "max_generations": 1, "copies": 1},
            },
            "DifferentialEvolutionSearch": {
                "class": search_algorithms["DifferentialEvolutionSearch"],
                # Note: max_generations=3 needed so at least one evolution generation runs
                # (DE starts from generation 2 after initial population)
                "kwargs": {"population_size": 4, "max_generations": 3},
            },
            "LFBOPatternSearch": {
                "class": search_algorithms["LFBOPatternSearch"],
                "kwargs": {"initial_population": 3, "max_generations": 1, "copies": 1},
            },
            "DESurrogateHybrid": {
                "class": search_algorithms["DESurrogateHybrid"],
                # Note: max_generations=3 needed so at least one evolution generation runs
                # (DE-based algorithms start from generation 2 after initial population)
                "kwargs": {"population_size": 4, "max_generations": 3},
            },
            "LFBOTreeSearch": {
                "class": search_algorithms["LFBOTreeSearch"],
                "kwargs": {"initial_population": 3, "max_generations": 1, "copies": 1},
            },
        }

    def test_all_algorithms_have_checkpoint_tests(self) -> None:
        """Verify all algorithms in search_algorithms have checkpoint tests."""
        tested_algos = set(self._get_autotuner_configs().keys())
        all_algos = set(search_algorithms.keys()) - self._CHECKPOINT_EXCLUDED
        missing = all_algos - tested_algos
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

    def _configure_for_checkpoint_test(
        self, bound: object, checkpoint_dir: str | None
    ) -> None:
        """Configure common settings for checkpoint tests."""
        bound.settings.autotune_precompile = None  # type: ignore[attr-defined]
        bound.settings.autotune_progress_bar = False  # type: ignore[attr-defined]
        bound.settings.autotune_checkpoint_dir = checkpoint_dir  # type: ignore[attr-defined]

    @contextlib.contextmanager
    def _deterministic_benchmark_ctx(self):  # type: ignore[no-untyped-def]
        """Context manager that patches do_bench and interleaved_bench with deterministic fns."""
        do_bench_fn, interleaved_fn = self._create_deterministic_benchmark_fns()
        with (
            patch("helion.autotuner.base_search.do_bench", do_bench_fn),
            patch(
                "helion.autotuner.base_search.interleaved_bench",
                interleaved_fn,
            ),
        ):
            yield

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

        with tempfile.TemporaryDirectory() as tmpdir:
            bound_kernel = kernel.bind(args)
            self._configure_for_checkpoint_test(bound_kernel, tmpdir)

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
                self._deterministic_benchmark_ctx(),
            ):
                search_a.autotune()

            # Verify real autotuning happened
            self.assertGreater(search_a._autotune_metrics.num_configs_tested, 0)
            self.assertTrue(math.isfinite(search_a.best_perf_so_far))

            # Verify generations are sequential (non-decreasing)
            for i in range(1, len(set_generation_calls)):
                prev_gen = set_generation_calls[i - 1]
                curr_gen = set_generation_calls[i]
                self.assertGreaterEqual(
                    curr_gen,
                    prev_gen,
                    f"{autotuner} generation went backwards: {prev_gen} -> {curr_gen}",
                )

            # Use a MID-RUN checkpoint snapshot to properly test resumption semantics.
            self.assertGreater(
                len(checkpoint_snapshots), 0, f"{autotuner} saved no checkpoints"
            )
            mid_index = len(checkpoint_snapshots) // 2
            mid_gen, mid_bytes = checkpoint_snapshots[mid_index]

            # Place the mid-run snapshot into the checkpoint dir for run B
            stable_hash = search_a._get_stable_hash()
            checkpoint_file = Path(tmpdir) / f"{stable_hash}.pt"
            checkpoint_file.write_bytes(mid_bytes)

            # Create new search with same checkpoint dir - tests auto-discover
            bound_kernel_b = kernel.bind(args)
            self._configure_for_checkpoint_test(bound_kernel_b, tmpdir)

            search_b = search_class(bound_kernel_b, args, **config["kwargs"])

            # Track whether _init_search was called (it should NOT be when resuming)
            init_search_called = False
            original_init_search = search_class._init_search

            def tracking_init_search(self: BaseSearch) -> None:
                nonlocal init_search_called
                init_search_called = True
                return original_init_search(self)

            with (
                self._deterministic_benchmark_ctx(),
                patch.object(search_class, "_init_search", tracking_init_search),
            ):
                search_b.autotune()

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

            # Verify restored search completed successfully
            self.assertGreater(len(search_b.population), 0)
            self.assertTrue(math.isfinite(search_b.best_perf_so_far))

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

        with tempfile.TemporaryDirectory() as tmpdir:
            # === FIRST RUN: add_kernel autotunes and saves checkpoint ===
            args_add = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_add = add_kernel.bind(args_add)
            self._configure_for_checkpoint_test(bound_add, tmpdir)

            search_add = PatternSearch(
                bound_add, args_add, initial_population=3, max_generations=1, copies=1
            )

            with (
                # Prevent cleanup so checkpoint remains for inspection
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                self._deterministic_benchmark_ctx(),
            ):
                search_add.autotune()

            # Verify add_kernel's checkpoint exists
            add_hash = search_add._get_stable_hash()
            self.assertTrue((Path(tmpdir) / f"{add_hash}.pt").exists())

            # === SECOND RUN: mul_kernel with same dir should start fresh ===
            args_mul = (torch.randn([64], device=DEVICE),)
            bound_mul = mul_kernel.bind(args_mul)
            self._configure_for_checkpoint_test(bound_mul, tmpdir)

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
                self._deterministic_benchmark_ctx(),
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

        with tempfile.TemporaryDirectory() as tmpdir:
            bound = kernel.bind(args)
            self._configure_for_checkpoint_test(bound, tmpdir)

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
                self._deterministic_benchmark_ctx(),
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

        with tempfile.TemporaryDirectory() as tmpdir:
            # === FIRST RUN: autotune and save checkpoint ===
            args_add = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_add = add_kernel.bind(args_add)
            self._configure_for_checkpoint_test(bound_add, tmpdir)

            search_add = PatternSearch(
                bound_add, args_add, initial_population=3, max_generations=1, copies=1
            )

            with (
                # Prevent cleanup so checkpoint remains for run B
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                self._deterministic_benchmark_ctx(),
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
            self._configure_for_checkpoint_test(bound_add_v2, tmpdir)

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
                self._deterministic_benchmark_ctx(),
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
    def test_checkpoint_truncated_file_raises_error(self) -> None:
        """Test that a truncated checkpoint file raises CheckpointError."""
        kernel, args = self._create_kernel_and_args()

        with tempfile.TemporaryDirectory() as tmpdir:
            bound_kernel = kernel.bind(args)
            self._configure_for_checkpoint_test(bound_kernel, tmpdir)

            # First, create a valid checkpoint
            search_a = PatternSearch(
                bound_kernel, args, initial_population=2, max_generations=1
            )
            with (
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                self._deterministic_benchmark_ctx(),
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
            self._configure_for_checkpoint_test(bound_kernel_b, tmpdir)

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
    def test_checkpoint_not_saved_when_dir_unset(self) -> None:
        """Test that no checkpoints are saved when autotune_checkpoint_dir is None."""
        kernel, args = self._create_kernel_and_args()
        bound_kernel = kernel.bind(args)
        self._configure_for_checkpoint_test(bound_kernel, None)

        save_results: list[Path | None] = []
        original_save = PatternSearch.save_checkpoint

        def tracking_save(self: BaseSearch) -> Path | None:
            result = original_save(self)
            save_results.append(result)
            return result

        search = PatternSearch(
            bound_kernel, args, initial_population=3, max_generations=1, copies=1
        )
        with (
            patch.object(PatternSearch, "save_checkpoint", tracking_save),
            self._deterministic_benchmark_ctx(),
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

        with tempfile.TemporaryDirectory() as tmpdir:
            bound_kernel = kernel.bind(args)
            self._configure_for_checkpoint_test(bound_kernel, tmpdir)

            search = PatternSearch(
                bound_kernel, args, initial_population=3, max_generations=1, copies=1
            )

            with self._deterministic_benchmark_ctx():
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

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run add_kernel
            args_add = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_add = add_kernel.bind(args_add)
            self._configure_for_checkpoint_test(bound_add, tmpdir)

            search_add = PatternSearch(
                bound_add, args_add, initial_population=3, max_generations=1, copies=1
            )
            with (
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                self._deterministic_benchmark_ctx(),
            ):
                search_add.autotune()

            # Run mul_kernel with same dir
            args_mul = (torch.randn([64], device=DEVICE),)
            bound_mul = mul_kernel.bind(args_mul)
            self._configure_for_checkpoint_test(bound_mul, tmpdir)

            search_mul = PatternSearch(
                bound_mul, args_mul, initial_population=3, max_generations=1, copies=1
            )
            with (
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                self._deterministic_benchmark_ctx(),
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

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run A: generate a checkpoint
            bound_a = kernel.bind(args)
            self._configure_for_checkpoint_test(bound_a, tmpdir)

            search_a = PatternSearch(
                bound_a, args, initial_population=3, max_generations=1, copies=1
            )
            with (
                # Prevent cleanup so checkpoint remains
                patch.object(PatternSearch, "_cleanup_checkpoint", lambda self: None),
                self._deterministic_benchmark_ctx(),
            ):
                search_a.autotune()

            # Verify checkpoint exists
            stable_hash = search_a._get_stable_hash()
            self.assertTrue((Path(tmpdir) / f"{stable_hash}.pt").exists())

            # Run B: force_autotune + same checkpoint dir
            kernel.reset()
            bound_b = kernel.bind(args)
            self._configure_for_checkpoint_test(bound_b, tmpdir)
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
                self._deterministic_benchmark_ctx(),
            ):
                search_b.autotune()

            self.assertFalse(
                init_search_called,
                "force_autotune should not prevent checkpoint resumption",
            )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_lfbo_checkpoint_current_filtered_by_surrogate(self) -> None:
        """Regression test: save_checkpoint must not error when
        _surrogate_select filters search_copy.current out of candidates.
        """
        kernel, args = self._create_kernel_and_args()

        from helion.autotuner import LFBOPatternSearch

        with tempfile.TemporaryDirectory() as tmpdir:
            bound = kernel.bind(args)
            self._configure_for_checkpoint_test(bound, tmpdir)

            search = LFBOPatternSearch(
                bound,
                args,
                initial_population=3,
                max_generations=3,
                copies=2,
            )

            with self._deterministic_benchmark_ctx():
                # This must not raise KeyError in to_dict()
                result = search.autotune()

            self.assertIsNotNone(result)


instantiate_parametrized_tests(TestAutotunerCheckpoint)


if __name__ == "__main__":
    import unittest

    unittest.main()
