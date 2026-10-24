from __future__ import annotations

import csv
import itertools
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
from typing import Callable
from typing import Sequence
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test._autotuner_test_utils import AutotunerTestCase

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import assert_close_with_mismatch_tolerance
from helion._testing import get_test_float32_matmul_precision
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfRefEager
from helion._testing import skipIfXPU
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import _MAX_REFERENCE_BASELINE_ATTEMPTS
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.random_search import RandomSearch
import helion.language as hl
from helion.runtime.settings import Settings
from helion.runtime.settings import _get_backend


@onlyBackends(["triton", "cute"])
class TestAutotunerAccuracy(AutotunerTestCase):
    def test_accuracy_check_filters_bad_config_wrong_output(self) -> None:
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(configs=[bad_config, good_config], autotune_log_level=0)
        def add_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(b.size()):
                b[tile] = a[tile] + b[tile]
            return b

        self._check_accuracy_filters_bad_config(
            add_inplace,
            bad_config,
            good_config,
            lambda fn: lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs) + 1,
            exact_final_failure_count=True,
        )

    def test_autotune_noncontiguous_arg(self) -> None:
        """Autotuning a kernel bound on a non-contiguous arg must succeed.

        A kernel bound on a non-contiguous arg hardcodes that arg's load strides
        into the compiled kernel as compile-time constants. The autotuner reruns
        that compiled kernel on a clone of the args to compute its accuracy
        baseline, so the clone must keep the same layout. If the clone is made
        contiguous, those hardcoded strides address the wrong (or out-of-bounds)
        memory and read garbage, so every config mismatches the baseline and
        autotuning fails to select one.

        The arg is a strided slice (``buf[::2]``): its contiguous clone compacts
        the storage, so the hardcoded stride-2 loads read the wrong elements.
        """
        config_a = helion.Config(block_sizes=[32], num_warps=4)
        config_b = helion.Config(block_sizes=[64], num_warps=4)

        @helion.kernel(configs=[config_a, config_b], autotune_log_level=0)
        def double(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2.0
            return out

        # Strided 1-D view: size 1024, stride 2 over a 2048-element buffer.
        x = torch.arange(2048, device=DEVICE, dtype=torch.float32)[::2]
        self.assertFalse(x.is_contiguous())
        self.assertEqual(x.stride(), (2,))

        bound_kernel = double.bind((x,))
        search = FiniteSearch(bound_kernel, (x,), configs=[config_a, config_b])
        best = search.autotune()
        self.assertIn(best, (config_a, config_b))

        # Compile the winning config directly; calling double(x) would rerun
        # the whole FiniteSearch since the manual search above is not attached
        # to the bound kernel.
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(x), x * 2.0)

    def test_autotune_broadcast_arg(self) -> None:
        """Autotuning a kernel bound on a broadcast/expanded arg must succeed.

        Same accuracy-baseline issue as the strided-slice case, but for a
        ``stride-0`` view from ``expand``. Here ``b`` is ``[M, 1]`` expanded to
        ``[M, N]`` (stride ``(1, 0)``); the kernel indexes ``b[ti, tj]`` so the
        stride-0 column load is hardcoded into the compiled kernel. A clone that
        materialized ``b`` to a contiguous ``[M, N]`` (stride ``(N, 1)``) would
        make that hardcoded load read a different element per column, so the
        baseline diverges and every config is rejected.
        """
        config_a = helion.Config(block_sizes=[32, 32], num_warps=4)
        config_b = helion.Config(block_sizes=[64, 64], num_warps=4)

        @helion.kernel(configs=[config_a, config_b], autotune_log_level=0)
        def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_i, tile_j in hl.tile(x.size()):
                out[tile_i, tile_j] = x[tile_i, tile_j] + b[tile_i, tile_j]
            return out

        m, n = 128, 128
        x = torch.randn(m, n, device=DEVICE)
        # Broadcast/expanded view: [m, 1] -> [m, n], stride (1, 0).
        b = torch.randn(m, 1, device=DEVICE).expand(m, n)
        self.assertFalse(b.is_contiguous())
        self.assertEqual(b.stride(), (1, 0))

        bound_kernel = add_bias.bind((x, b))
        search = FiniteSearch(bound_kernel, (x, b), configs=[config_a, config_b])
        best = search.autotune()
        self.assertIn(best, (config_a, config_b))

        # Compile the winning config directly; calling add_bias(x, b) would
        # rerun the whole FiniteSearch since the manual search above is not
        # attached to the bound kernel.
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(x, b), x + b)

    def test_accuracy_check_filters_bad_config_wrong_arg_mutation(self) -> None:
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(configs=[bad_config, good_config], autotune_log_level=0)
        def add_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(b.size()):
                b[tile] = a[tile] + b[tile]
            return b

        def wrap_bad_fn(fn):
            def wrong_fn(*fn_args, **fn_kwargs):
                result = fn(*fn_args, **fn_kwargs)
                # Introduce an extra mutation so inputs differ from baseline
                fn_args[1].add_(1)
                return result

            return wrong_fn

        self._check_accuracy_filters_bad_config(
            add_inplace,
            bad_config,
            good_config,
            wrap_bad_fn,
            exact_final_failure_count=False,
        )

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

        # The custom baseline fn is invoked on the parent-side baseline path;
        # skip the benchmark worker subprocess (seconds of startup overhead).
        @helion.kernel(
            configs=[config1, config2],
            autotune_baseline_fn=custom_baseline,
            autotune_log_level=0,
            autotune_benchmark_subprocess=False,
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

        # The custom baseline fn is invoked on the parent-side baseline path;
        # skip the benchmark worker subprocess (seconds of startup overhead).
        @helion.kernel(
            configs=[bad_config, good_config],
            autotune_baseline_fn=custom_baseline,
            autotune_log_level=0,
            autotune_benchmark_subprocess=False,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        self._check_accuracy_filters_bad_config(
            add,
            bad_config,
            good_config,
            lambda fn: lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs) + 1,
            exact_final_failure_count=True,
            check_spawn=False,
        )

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

    def test_reference_baseline_shrinks_after_hardware_rejection(self) -> None:
        """A reference config the device rejects backs off instead of aborting.

        The reference config only exists to produce baseline outputs, so a
        hardware limit Helion cannot model at config-generation time must not
        cost the whole autotune.  A promoted execution default goes first and
        backs off to the reference config.
        """
        attempted: list[int] = []
        search, args, reference_block, promoted = self._rejecting_baseline_search(
            lambda block, reference: block > reference // 4, attempted
        )
        self.assertGreaterEqual(reference_block, 4)

        search._prepare()

        # Halve once per failure until the tile is accepted.
        self.assertEqual(
            attempted,
            [*promoted, reference_block, reference_block // 2, reference_block // 4],
        )
        torch.testing.assert_close(
            search.benchmark_provider._baseline_output, args[0] + args[1]
        )

    def test_reference_baseline_aborts_when_every_block_size_fails(self) -> None:
        """Shrinking is bounded: a config nothing fixes still reports clearly."""
        attempted: list[int] = []
        search, _, reference_block, promoted = self._rejecting_baseline_search(
            lambda block, reference: True, attempted
        )

        with (
            patch.object(search.log, "warning") as warn,
            self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                "Autotuning reference config failed while computing baseline",
            ),
        ):
            search._prepare()

        shrinking = attempted[len(promoted) :]
        self.assertEqual(attempted[: len(promoted)], promoted)
        self.assertEqual(len(shrinking), _MAX_REFERENCE_BASELINE_ATTEMPTS)
        # Each attempt halves the previous one, starting from the reference
        # config that the error message reports.
        self.assertEqual(shrinking[0], reference_block)
        for previous, block in itertools.pairwise(shrinking):
            self.assertEqual(block, previous // 2)
        # A retry is announced only when it actually runs.
        retries = [
            c for c in warn.call_args_list if "retrying the baseline" in str(c.args[0])
        ]
        self.assertEqual(len(retries), len(shrinking) - 1)

    def test_reference_baseline_raises_errors_shrinking_cannot_fix(self) -> None:
        """Bugs and unrecoverable runtime errors are not retried.

        A smaller config would fail the same way, so the error surfaces on the
        reference config right away, as it did before the backoff existed.
        """
        errors = [RuntimeError("CUDA error: an illegal memory access was encountered")]
        if _get_backend() == "cute":
            # CuteBackend.classify_autotune_exception skips (rather than
            # raises) every non-fatal config failure, so the backoff keeps
            # shrinking past one and only unrecoverable runtime errors stop it.
            attempted: list[int] = []
            search, _, reference_block, promoted = self._rejecting_baseline_search(
                lambda block, reference: True, attempted, RuntimeError("unsupported op")
            )
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                "Autotuning reference config failed while computing baseline",
            ):
                search._prepare()
            self.assertEqual(
                len(attempted), len(promoted) + _MAX_REFERENCE_BASELINE_ATTEMPTS
            )
        else:
            errors.insert(0, RuntimeError("unsupported op"))
        for error in errors:
            with self.subTest(error=str(error)):
                attempted: list[int] = []
                search, _, reference_block, promoted = self._rejecting_baseline_search(
                    lambda block, reference: True, attempted, error
                )
                # An unrecoverable runtime error surfaces on the first config
                # (the promoted default when there is one); a bug on the
                # promoted default still backs off to the reference config,
                # which then raises it.  The error names the config it reports.
                on_promoted = bool(promoted) and "CUDA error" in str(error)
                role = (
                    "Execution default"
                    if on_promoted
                    else "Autotuning reference config"
                )

                with self.assertRaisesRegex(
                    helion.exc.InvalidConfig,
                    f"{role} failed while computing baseline",
                ) as ctx:
                    search._prepare()

                if on_promoted:
                    self.assertEqual(attempted, promoted)
                else:
                    self.assertEqual(attempted, [*promoted, reference_block])
                self.assertIs(ctx.exception.__cause__, error)
                # Raised as is: the cause keeps the traceback of where it failed.
                self.assertIsNotNone(error.__traceback__)

    def test_autotune_baseline_tolerance(self) -> None:
        cfg1 = helion.Config(block_sizes=[1], num_warps=4)
        cfg2 = helion.Config(block_sizes=[1], num_warps=8)
        a, b = torch.randn([32], device=DEVICE), torch.randn([32], device=DEVICE)

        # Baseline that returns slightly incorrect result (1e-4 error)
        def incorrect_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b + 1e-4

        # Test both strict (1e-5) and lenient (1e-3) tolerances
        for tol, expect_reject in [(1e-5, True), (1e-3, False)]:
            # The tolerance plumbing under test is shared with the in-process
            # accuracy path; skip the benchmark worker subprocess (seconds of
            # startup overhead per kernel).
            @helion.kernel(
                configs=[cfg1, cfg2],
                autotune_baseline_fn=incorrect_baseline,
                autotune_baseline_atol=tol,
                autotune_baseline_rtol=tol,
                autotune_benchmark_subprocess=False,
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

        # Test with float8_e4m3fn as a representative fp8 dtype. The automatic
        # tolerance selection under test is parent-side; skip the benchmark
        # worker subprocess (seconds of startup overhead).
        @helion.kernel(configs=[cfg1, cfg2], autotune_benchmark_subprocess=False)
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

        # The fp8 handling under test lives in the shared accuracy.assert_close;
        # skip the benchmark worker subprocess (seconds of startup overhead).
        @helion.kernel(configs=[cfg1, cfg2], autotune_benchmark_subprocess=False)
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
        #   + 2 benchmark runs = 4 total. The benchmark runs happen in the
        #   killable worker when the subprocess benchmark is enabled, where
        #   this patch does not reach: the parent then clones it twice.
        expected_mutated_clones = (
            2 if inplace_add.settings.autotune_benchmark_subprocess else 4
        )
        self.assertEqual(
            mutated_clones[0],
            expected_mutated_clones,
            f"Mutated tensor cloned {mutated_clones[0]} times, "
            f"expected {expected_mutated_clones}.",
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
        import helion.autotuner.accuracy as _accuracy

        numel = 2**22  # 4M float32 elements (~16 MB each)
        # The default chunk_size (2**22) would not chunk a tensor this small,
        # so pass a smaller one through the patched helper below; 4 chunks is
        # enough to exercise the chunked path and its memory bound.
        chunk_size = 2**20

        config1 = helion.Config(block_sizes=[128], num_warps=4)
        config2 = helion.Config(block_sizes=[256], num_warps=4)

        # Pin the accuracy check to the parent so the patched chunked helper
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

        # Patch the moved chunked helper to record peak memory delta per call.
        real_chunked_assert_close = _accuracy._chunked_assert_close
        peaks: list[int] = []

        def measuring_chunked_assert_close(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            real_chunked_assert_close(*args, chunk_size=chunk_size, **kwargs)
            peak = torch.cuda.max_memory_allocated() - before
            peaks.append(peak)

        with patch.object(
            _accuracy, "_chunked_assert_close", measuring_chunked_assert_close
        ):
            out = vec_add(a, b)

        # Accuracy check was called at least once
        self.assertGreater(len(peaks), 0, "Expected _chunked_assert_close to be called")

        # Every call's peak memory should be less than naive peak
        for i, p in enumerate(peaks):
            self.assertLess(
                p,
                naive_peak * 0.5,
                f"Call {i}: peak {p} should be < 50% of naive {naive_peak}",
            )

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

        # Custom accuracy-check fns always run in-process; skip the benchmark
        # worker subprocess (seconds of startup overhead).
        @helion.kernel(
            configs=[cfg1, cfg2],
            autotune_log_level=0,
            autotune_baseline_accuracy_check_fn=strict_check,
            autotune_benchmark_subprocess=False,
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


class TestMismatchTolerance(TestCase):
    def test_get_test_float32_matmul_precision_pallas_interpret(self) -> None:
        with (
            patch("helion._testing._get_backend", return_value="pallas"),
            patch("helion._testing.is_pallas_interpret", return_value=True),
        ):
            self.assertEqual(get_test_float32_matmul_precision(), "high")

    def test_get_test_float32_matmul_precision_real_pallas(self) -> None:
        with (
            patch("helion._testing._get_backend", return_value="pallas"),
            patch("helion._testing.is_pallas_interpret", return_value=False),
        ):
            self.assertEqual(get_test_float32_matmul_precision(), "medium")

    def test_assert_close_with_mismatch_tolerance_bounds_mismatches(self) -> None:
        with self.assertRaisesRegex(
            AssertionError, "Mismatched absolute diff too large"
        ):
            assert_close_with_mismatch_tolerance(
                torch.tensor([10.0, 1.0, 1.0, 1.0], device=DEVICE),
                torch.tensor([1.0, 1.0, 1.0, 1.0], device=DEVICE),
                max_mismatch_pct=0.5,
                max_mismatched_abs_diff=5.0,
            )

    def test_assert_close_with_mismatch_tolerance_non_finite(self) -> None:
        # Matching infinities (e.g. -inf outside an attention window) are not
        # mismatches, so only the 1.5 counts against the 30% budget here ...
        inf = float("inf")
        expected = torch.tensor([-inf, 1.0, 1.0, 1.0], device=DEVICE)
        assert_close_with_mismatch_tolerance(
            torch.tensor([-inf, 1.0, 1.0, 1.5], device=DEVICE),
            expected,
            max_mismatch_pct=0.3,
            max_mismatched_abs_diff=1.0,
        )
        # ... while a one-sided inf or NaN is a mismatch with infinite diff.
        for bad in (-inf, inf, float("nan")):
            with self.assertRaisesRegex(
                AssertionError, "Mismatched absolute diff too large"
            ):
                assert_close_with_mismatch_tolerance(
                    torch.tensor([-inf, 1.0, 1.0, bad], device=DEVICE),
                    expected,
                    max_mismatch_pct=0.3,
                    max_mismatched_abs_diff=1.0,
                )
        # A matching pair of infinities (or NaNs) has a NaN raw difference,
        # which must not hide the 0.5 miss from the whole-tensor bounds.
        for pad in (-inf, float("nan")):
            actual = torch.tensor([pad, 1.0, 1.0, 1.5], device=DEVICE)
            expected = torch.tensor([pad, 1.0, 1.0, 1.0], device=DEVICE)
            assert_close_with_mismatch_tolerance(
                actual,
                expected,
                max_mismatch_pct=0.3,
                max_abs_diff=0.5,
                max_rel_diff=0.5,
            )
            with self.assertRaisesRegex(AssertionError, "Absolute diff too large"):
                assert_close_with_mismatch_tolerance(
                    actual, expected, max_mismatch_pct=0.3, max_abs_diff=0.25
                )
            with self.assertRaisesRegex(AssertionError, "Relative diff too large"):
                assert_close_with_mismatch_tolerance(
                    actual, expected, max_mismatch_pct=0.3, max_rel_diff=0.25
                )

    def test_assert_close_with_mismatch_tolerance_mixed_dtypes(self) -> None:
        # torch.isclose rejects differing dtypes; the helper must still judge
        # such inputs numerically and raise only AssertionError, which is all
        # the autotuner's accuracy check catches.
        # CPU tensors: MPS has no float64, and the helper is device-agnostic.
        actual = torch.tensor([1.0, 1.0, 1.0, 1.5])
        expected = torch.ones(4, dtype=torch.float64)
        assert_close_with_mismatch_tolerance(actual, expected, max_mismatch_pct=0.3)
        with self.assertRaisesRegex(AssertionError, "Too many mismatches"):
            assert_close_with_mismatch_tolerance(actual, expected, max_mismatch_pct=0.1)

    def test_accuracy_scaled_atol_tolerates_reduction_noise(self) -> None:
        from helion.autotuner.accuracy import assert_close

        # A large-K reduction output: elements have scale ~200, and near-zero
        # elements legitimately differ across accumulation orders by amounts
        # far above a fixed elementwise atol.
        torch.manual_seed(0)
        expected = torch.randn(64, 64, device=DEVICE) * 200.0
        expected[0, 0] = 0.05  # near-zero element from cancellation
        actual = expected.clone()
        actual[0, 0] = -0.05  # reorder-scale wobble on the near-zero element

        # The fixed elementwise default rejects the wobble...
        with self.assertRaises(AssertionError):
            assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        # ...the scaled floor (atol * max(1, rms(expected))) accepts it...
        assert_close(
            actual, expected, atol=1e-2, rtol=1e-2, scale_atol_by_expected_rms=True
        )
        # ...but a genuinely wrong element at the output's own scale still fails.
        actual[1, 1] = expected[1, 1] + 100.0
        with self.assertRaises(AssertionError):
            assert_close(
                actual, expected, atol=1e-2, rtol=1e-2, scale_atol_by_expected_rms=True
            )

    def test_accuracy_scaled_atol_survives_inf_in_baseline(self) -> None:
        from helion.autotuner.accuracy import assert_close

        # An inf element in the baseline (log-space outputs, masked rows) must
        # not blow the RMS up to inf and disable the gate for the rest of the
        # tensor.
        expected = torch.randn(64, device=DEVICE)
        expected[0] = float("-inf")
        actual = expected.clone()
        actual[1] = expected[1] + 1e6
        with self.assertRaises(AssertionError):
            assert_close(
                actual, expected, atol=1e-2, rtol=1e-2, scale_atol_by_expected_rms=True
            )

    def test_accuracy_scaled_atol_no_op_for_small_scale_outputs(self) -> None:
        from helion.autotuner.accuracy import assert_close

        # Outputs with rms <= 1 (e.g. probabilities) keep the exact tolerance:
        # max(1, rms) == 1, so scaling never loosens well-scaled checks.
        expected = torch.full((32,), 0.5, device=DEVICE)
        actual = expected.clone()
        actual[0] += 0.05
        for scale in (False, True):
            with self.assertRaises(AssertionError):
                assert_close(
                    actual,
                    expected,
                    atol=1e-2,
                    rtol=1e-2,
                    scale_atol_by_expected_rms=scale,
                )


@onlyBackends(["triton", "cute"])
class TestAutotuneIgnoreErrors(TestCase):
    def _make_search(
        self, settings: Settings, *, args: tuple[object, ...] = ()
    ) -> BaseSearch:
        # NOTE: construct via __init__ (mock kernel) instead of hand-mirroring
        # its attributes, so new __init__ fields don't need to be added here.
        config_spec = SimpleNamespace(
            default_config=lambda: helion.Config(block_sizes=[1]),
            cute_flash_search_enabled=False,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                supports_precompile=lambda: True,
                autotune_config_is_viable=lambda _config_spec, _config: True,
                should_deduplicate_generated_sources=lambda config_spec: False,
                get_do_bench=lambda: None,
                classify_autotune_exception=lambda error: None,
                probe_long_autotune_kernels=lambda _config_spec: False,
            ),
        )
        kernel = SimpleNamespace(
            settings=settings,
            config_spec=config_spec,
            format_kernel_decorator=lambda config, s: "decorator",
            to_triton_code=lambda config: "code",
            maybe_log_repro=lambda log_func, args, config=None: None,
            supports_subprocess_benchmark=lambda: False,
            env=SimpleNamespace(process_group_name=None),
        )
        search = BaseSearch(kernel, args)
        with patch.object(
            LocalBenchmarkProvider,
            "_compute_baseline",
            return_value=(None, [], None),
        ):
            search._prepare()
        return search

    def _make_compile_failure_search(self) -> BaseSearch:
        search = self._make_search(
            Settings(
                autotune_precompile=None,
                autotune_log_level=logging.CRITICAL,
            )
        )
        search.kernel.env = SimpleNamespace(process_group_name=None)
        search.kernel.compile_config = None
        return search

    def _run_late_compile_failures(
        self,
        late_configs: Sequence[str],
        error_for: Callable[[str], Exception],
    ) -> tuple[list[object], list[object]]:
        search = self._make_compile_failure_search()

        def compile_config(config: str, **_kwargs: object) -> Callable[..., None]:
            if config == "valid":
                return lambda *args, **kwargs: None
            raise error_for(config)

        with (
            patch.object(search.kernel, "compile_config", side_effect=compile_config),
            patch.object(
                search.benchmark_provider,
                "_benchmark_function",
                return_value=1.0,
            ),
        ):
            initial = search.benchmark_batch(["valid"], desc="initial")
            late = search.benchmark_batch(list(late_configs), desc="late")
        return initial, late

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
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)

    def test_initial_compile_failures_raise(self) -> None:
        cases: tuple[tuple[str, type[Exception], str, Exception], ...] = (
            (
                "invalid",
                exc.InvalidConfig,
                "invalid initial config",
                exc.InvalidConfig("invalid initial config"),
            ),
            (
                "runtime",
                RuntimeError,
                "initial compiler failure",
                RuntimeError("initial compiler failure"),
            ),
        )
        for config, error_type, message, error in cases:
            with self.subTest(config=config):
                search = self._make_compile_failure_search()
                with (
                    patch.object(
                        search.kernel,
                        "compile_config",
                        side_effect=error,
                    ),
                    self.assertRaisesRegex(error_type, message),
                ):
                    search.benchmark_batch([config], desc="initial")

    def test_initial_compile_failures_skipped_when_fallback_available(self) -> None:
        search = self._make_compile_failure_search()
        with patch.object(
            search.kernel,
            "compile_config",
            side_effect=exc.BackendUnsupported("cute", "seed config unsupported"),
        ):
            results = search.benchmark_batch(
                ["seed"], desc="initial", raise_if_no_viable_config=False
            )
        self.assertEqual([result.perf for result in results], [float("inf")])
        self.assertEqual([result.status for result in results], ["error"])
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)

    def test_late_compile_failures_are_skipped(self) -> None:
        cases = (
            (
                "invalid",
                lambda config: exc.InvalidConfig(f"invalid late neighbor: {config}"),
            ),
            ("runtime", lambda config: RuntimeError("late compiler failure")),
            (
                "unsupported",
                lambda config: exc.BackendUnsupported(
                    "cute", f"unsupported late neighbor {config}"
                ),
            ),
        )
        for prefix, error_for in cases:
            with self.subTest(prefix=prefix):
                initial, late = self._run_late_compile_failures(
                    (f"{prefix}_a", f"{prefix}_b"), error_for
                )
                self.assertEqual(initial[0].perf, 1.0)
                self.assertEqual([result.perf for result in late], [float("inf")] * 2)
                self.assertEqual([result.status for result in late], ["error"] * 2)

    def test_autotune_log_sink_writes_csv_and_log(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune_run"
        settings = Settings(
            autotune_log=str(base_path),
            autotune_log_level=logging.CRITICAL,
        )
        logger = AutotuningLogger(settings)
        config = helion.Config(block_sizes=[32], num_warps=4)
        with logger.autotune_logging():
            config_id = logger.register_config(config)
            entry = AutotuneLogEntry(
                generation=5,
                status="ok",
                perf_ms=1.234,
                compile_time=0.5,
                config_id=config_id,
                config=config,
                source_hash="source-abc",
            )
            logger.record_autotune_entry(entry)
            logger("finalized entry", level=logging.CRITICAL)

        csv_path = base_path.with_suffix(".csv")
        log_path = base_path.with_suffix(".log")
        self.assertTrue(csv_path.exists())
        self.assertTrue(log_path.exists())
        rows = list(csv.reader(csv_path.read_text().splitlines()))
        self.assertEqual(len(rows), 2)  # header + exactly the one recorded entry
        header, row = rows[0], rows[1]
        self.assertEqual(
            header,
            [
                "run_id",
                "timestamp_s",
                "config_id",
                "generation",
                "status",
                "perf_ms",
                "compile_time_s",
                "config",
            ],
        )

        def cell(name: str) -> str:
            return row[header.index(name)]

        # No metadata supplied here, so the run_id join key is empty.
        self.assertEqual(cell("run_id"), "")
        self.assertEqual(cell("config_id"), config_id)
        self.assertEqual(cell("generation"), "5")
        self.assertEqual(cell("status"), "ok")
        self.assertEqual(cell("perf_ms"), "1.234000")
        self.assertEqual(cell("compile_time_s"), "0.50")
        self.assertEqual(cell("config"), str(config))
        source_rows = list(
            csv.reader(base_path.with_suffix(".sources.csv").read_text().splitlines())
        )
        self.assertEqual(source_rows[0][-2:], ["status", "source_hash"])
        self.assertEqual(source_rows[1][-2:], ["ok", "source-abc"])
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

    def test_benchmarked_members_survive_config_mutation(self):
        # Regression test: _benchmarked_members / _pinned_finalist_members must
        # not be keyed by Config objects. Config is mutable and hashes on its
        # contents, so mutating a recorded config in place (as normalize /
        # neighbor generation does during the search) corrupted the bookkeeping
        # -> _prune_benchmarked_members raised KeyError, and final verification
        # could select a config that was never accuracy-validated.
        search = DifferentialEvolutionSearch.__new__(DifferentialEvolutionSearch)
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        def make_member(perf: float, num_warps: int) -> PopulationMember:
            return PopulationMember(
                lambda *a, **k: None,
                [perf],
                [],
                helion.Config(block_sizes=[64], num_warps=num_warps),
                status="ok",
            )

        fast = make_member(1.0, num_warps=4)
        slow = make_member(2.0, num_warps=8)
        search._record_best_member_for_config(
            search._benchmarked_members, fast.config, fast
        )
        search._record_best_member_for_config(
            search._benchmarked_members, slow.config, slow
        )

        # Mutate the original configs in place *after* they were recorded.
        fast.config.config["num_warps"] = 999
        slow.config.config["num_warps"] = 999

        # Pruning to the single fastest config must not raise (the old code did
        # `del dict[config]` on a config whose hash had changed -> KeyError) and
        # must keep the fast member with the config it was actually benchmarked
        # with (snapshot), not the later in-place mutation.
        search._prune_benchmarked_members(top_k=1)
        self.assertEqual(len(search._benchmarked_members), 1)
        kept = next(iter(search._benchmarked_members.values()))
        self.assertEqual(kept.perfs[0], 1.0)
        self.assertEqual(kept.config["num_warps"], 4)

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork start method is unavailable on this platform",
    )
    def test_fork_precompile_avoids_device_reinit(self):
        settings = Settings(
            autotune_precompile="fork",
            autotune_log_level=logging.CRITICAL,
            autotune_compile_timeout=5,
        )
        search = self._make_search(settings, args=("arg0",))

        parent_pid = os.getpid()
        lazy_calls: list[int] = []
        device_module = torch.get_device_module(DEVICE)

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
            device_module._lazy_init()  # pyrefly: ignore[missing-attribute]
            _launcher("fake_kernel", (1,), *fn_args)

        with (
            patch(
                "helion.autotuner.precompile_future.make_precompiler",
                side_effect=fake_make_precompiler,
            ),
            patch.object(device_module, "_lazy_init", side_effect=fake_lazy_init),
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
            # started/completed entries are recorded in the parent's benchmark
            # loop either way; skip the benchmark worker subprocess (several
            # seconds of interpreter+CUDA startup per search).
            @helion.kernel(autotune_benchmark_subprocess=False)
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
            ("RandomSearch", lambda kernel, args: RandomSearch(kernel, args, count=2)),
            (
                "PatternSearch",
                lambda kernel, args: PatternSearch(
                    kernel,
                    args,
                    initial_population=2,
                    max_generations=1,
                    copies=1,
                    num_neighbors_cap=2,
                ),
            ),
            (
                "DifferentialEvolutionSearch",
                lambda kernel, args: DifferentialEvolutionSearch(
                    kernel, args, population_size=2, max_generations=1
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
        ``force_autotune``) is a biased slice excluded from the dataset: even
        with the dataset flag on, no ``.meta.jsonl`` is written. The debug
        ``.csv`` is still written, since it is gated only by the log path
        (PRD FR8/FR9)."""
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
                "HELION_AUTOTUNE_LOG_DETAILS": "1",
                "HELION_AUTOTUNE_LOG_LEVEL": "0",
            },
        ):
            # The CSV/sidecar gating under test is parent-side logging; skip
            # the benchmark worker subprocess (seconds of startup overhead).
            @helion.kernel(configs=configs, autotune_benchmark_subprocess=False)
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

        # Restricted search -> debug CSV written, but no dataset sidecar.
        self.assertTrue(base_path.with_suffix(".csv").exists())
        self.assertFalse(base_path.with_suffix(".meta.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
