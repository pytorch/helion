from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import datetime
import functools
import inspect
from itertools import count
from itertools import starmap
import logging
import math
from math import inf
import multiprocessing as mp
from multiprocessing import connection
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import time
import traceback
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import NoReturn
from typing import cast
from unittest.mock import patch
import uuid

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map
from torch.utils._pytree import tree_map_only
from triton.testing import do_bench

from .. import exc
from ..runtime.kernel import BoundKernel
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import make_precompiler
from .benchmarking import interleaved_bench
from .config_generation import ConfigGeneration
from .config_generation import FlatConfig
from .logger import SUPPRESSED_TRITON_CODE_MSG
from .logger import LambdaLogger
from .logger import classify_triton_exception
from .logger import format_triton_compile_failure
from .logger import log_generated_triton_code_debug
from .logger import match_unrecoverable_runtime_error
from .progress_bar import iter_with_progress

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import CompiledConfig
    from ..runtime.settings import Settings
    from . import ConfigSpec

log = logging.getLogger(__name__)


class BaseAutotuner(abc.ABC):
    """
    Abstract base class for all autotuners and classes that wrap autotuners, like caching.
    """

    @abc.abstractmethod
    def autotune(self, *, skip_cache: bool = False) -> Config:
        raise NotImplementedError


class BaseSearch(BaseAutotuner):
    """
    Base class for search algorithms. This class defines the interface and utilities for all
    search algorithms.

    Attributes:
        kernel (BoundKernel): The kernel to be tuned.
        settings (Settings): The settings associated with the kernel.
        config_spec (ConfigSpec): The configuration specification for the kernel.
        args (Sequence[object]): The arguments to be passed to the kernel.
        counters (collections.Counter): A counter to track various metrics during the search.
    """

    _baseline_output: object
    _kernel_mutates_args: bool
    _baseline_post_args: Sequence[object] | None
    _jobs: int
    _precompile_result_counter: count[int]

    def __init__(self, kernel: BoundKernel, args: Sequence[object]) -> None:
        """
        Initialize the BaseSearch object.

        Args:
            kernel: The kernel to be tuned.
            args: The arguments to be passed to the kernel.
        """
        super().__init__()
        self.kernel = kernel
        self.settings: Settings = kernel.settings
        self.config_spec: ConfigSpec = kernel.config_spec
        self.args: Sequence[object] = args
        self.counters: collections.Counter[str] = collections.Counter()
        self.log = LambdaLogger(self.settings.autotune_log_level)
        self.best_perf_so_far = inf
        seed = self.settings.autotune_random_seed
        random.seed(seed)
        self.log(f"Autotune random seed: {seed}")
        self._original_args: Sequence[object] = self._clone_args(self.args)
        self._precompile_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._precompile_args_path: str | None = None
        self._precompile_result_counter = count()
        (
            self._baseline_output,
            self._kernel_mutates_args,
            self._baseline_post_args,
        ) = self._compute_baseline()
        self._jobs = self._decide_num_jobs()

    def _next_precompile_result_path(self) -> str:
        assert self._precompile_tmpdir is not None
        return os.path.join(
            self._precompile_tmpdir.name,
            f"result_{next(self._precompile_result_counter)}.pkl",
        )

    def cleanup(self) -> None:
        if self._precompile_tmpdir is not None:
            self._precompile_tmpdir.cleanup()
            self._precompile_tmpdir = None
        self._precompile_args_path = None
        self._precompile_result_counter = count()

    def _clone_args(self, args: Sequence[object]) -> Sequence[object]:
        def _clone_leaf(leaf: object) -> object:
            if isinstance(leaf, torch.Tensor):
                clone = leaf.detach().clone()
                clone.requires_grad_(leaf.requires_grad)
                return clone
            return leaf

        return tree_map(_clone_leaf, args)

    def _compute_baseline(self) -> tuple[object, bool, Sequence[object] | None]:
        """
        Return output and post-run input arguments of the default-config kernel.
        Also detect if the kernel mutates any of its input arguments.
        """
        new_args = self._clone_args(self._original_args)
        baseline_config = self.config_spec.default_config()
        try:
            baseline_output = self.kernel.compile_config(
                baseline_config, allow_print=False
            )(*new_args)
            torch.accelerator.synchronize()
        except Exception as e:
            decorator = self.kernel.format_kernel_decorator(
                baseline_config, self.settings
            )
            log_generated_triton_code_debug(
                self.log,
                self.kernel,
                baseline_config,
                prefix=f"Generated Triton code for {decorator}:",
            )
            raise exc.InvalidConfig(
                "Default config failed while computing baseline.\n"
                f"Default config: {decorator}\n"
                f"{SUPPRESSED_TRITON_CODE_MSG}\n"
            ) from e
        original_args_flat, _ = tree_flatten(self._original_args)
        new_args_flat, _ = tree_flatten(new_args)
        mutated = False
        for old, new in zip(original_args_flat, new_args_flat, strict=False):
            if (
                isinstance(old, torch.Tensor)
                and isinstance(new, torch.Tensor)
                and (not torch.equal(new, old))
            ):
                mutated = True
                break
        baseline_post_args = self._clone_args(new_args)
        return baseline_output, mutated, baseline_post_args

    def _decide_num_jobs(self) -> int:
        if not self.settings.autotune_precompile:
            return 1

        jobs = self.settings.autotune_precompile_jobs
        if not jobs:
            jobs = os.cpu_count() or 1

        if self.settings.autotune_precompile != "spawn":
            return jobs

        memory_per_job = _estimate_tree_bytes(self.args) + _estimate_tree_bytes(
            self._baseline_output
        )
        memory_per_job *= 2  # safety factor
        if memory_per_job <= 0:
            return jobs

        device = self.kernel.env.device
        if device.type != "cuda":
            # TODO(jansel): support non-cuda devices
            return jobs

        available_memory, _ = torch.cuda.mem_get_info(device)
        jobs_by_memory = available_memory // memory_per_job
        if jobs_by_memory < jobs:
            gib_per_job = memory_per_job / (1024**3)
            available_gib = available_memory / (1024**3)
            if jobs_by_memory > 0:
                self.log.warning(
                    f"Reducing autotune precompile spawn jobs from {jobs} to {jobs_by_memory} "
                    f"due to limited GPU memory (estimated {gib_per_job:.2f} GiB per job, "
                    f"{available_gib:.2f} GiB free). "
                    f"Set HELION_AUTOTUNE_PRECOMPILE_JOBS={jobs_by_memory} "
                    "to make this lower cap persistent, "
                    'set HELION_AUTOTUNE_PRECOMPILE="fork" to disable spawning, or reduce GPU memory usage.'
                )
            else:
                raise exc.AutotuneError(
                    "Autotune precompile spawn mode requires at least one job, but estimated "
                    "memory usage exceeds available GPU memory."
                    f"Estimated {gib_per_job:.2f} GiB per job, but only "
                    f"{available_gib:.2f} GiB free. "
                    'Set HELION_AUTOTUNE_PRECOMPILE="fork" to disable spawning, or reduce GPU memory usage.'
                )
            jobs = jobs_by_memory

        return jobs

    def _validate_against_baseline(
        self, config: Config, output: object, args: Sequence[object]
    ) -> bool:
        try:
            torch.testing.assert_close(
                output, self._baseline_output, atol=1e-2, rtol=1e-2
            )
            if self._kernel_mutates_args:
                torch.testing.assert_close(
                    args, self._baseline_post_args, atol=1e-2, rtol=1e-2
                )
        except AssertionError as e:
            self.counters["accuracy_mismatch"] += 1
            if not self.settings.autotune_ignore_errors:
                self.log.warning(
                    f"Skipping config with accuracy mismatch: {config!r}\n{e!s}\nUse HELION_AUTOTUNE_ACCURACY_CHECK=0 to disable this check.\n"
                )
            return False
        return True

    def benchmark(self, config: Config) -> tuple[Callable[..., object], float]:
        """
        Benchmark a specific configuration.

        This method compiles the kernel with the given configuration and measures its performance.

        Args:
            config: The configuration to benchmark.

        Returns:
            The function and performance of the configuration in ms.
        """
        fn = self.kernel.compile_config(config, allow_print=False)
        if self.start_precompile_and_check_for_hangs(config, fn)():
            return fn, self.benchmark_function(config, fn)
        return fn, inf

    def benchmark_function(self, config: Config, fn: CompiledConfig) -> float:
        """
        Benchmark a compiled function.  This function is called by the autotuner to measure the
        performance of a specific configuration.

        Args:
            config: The configuration to benchmark.
            fn: A precompiled version of config.

        Returns:
            The performance of the configuration in ms.
        """
        self.counters["benchmark"] += 1
        self.log.debug(lambda: f"Running benchmark for {config!r}")
        try:
            # TODO(jansel): early exit with fewer trials if early runs are slow
            self.log.debug(lambda: f"Running {config} at {datetime.datetime.now()}")
            t0 = time.perf_counter()
            if self._kernel_mutates_args:
                self.args = self._clone_args(self._original_args)
            torch.accelerator.synchronize()
            output = fn(*self.args)  # make sure the kernel is compiled
            torch.accelerator.synchronize()
            if (
                self.settings.autotune_accuracy_check
                and not self._validate_against_baseline(config, output, self.args)
            ):
                # Accuracy check failed; reject this config
                return inf
            t1 = time.perf_counter()
            res = do_bench(
                functools.partial(fn, *self.args),
                return_mode="median",
                warmup=1,  # we are already warmed up above
                rep=50,
            )
            t2 = time.perf_counter()
            assert isinstance(res, float)
            self.log.debug(
                lambda: f"result: {res:.4f}ms (took {t1 - t0:.1f}s + {t2 - t1:.1f}s)",
            )
            if res < self.best_perf_so_far:
                self.best_perf_so_far = res
            return res
        except Exception as e:
            if match_unrecoverable_runtime_error(e):
                raise exc.TritonUnrecoverableRuntimeError(
                    reason=str(e),
                    decorator=self.kernel.format_kernel_decorator(
                        config, self.settings
                    ),
                    error=f"{type(e).__qualname__}: {e}",
                ) from e
            action = classify_triton_exception(e)
            if self.settings.autotune_ignore_errors:
                pass
            elif action == "raise":
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                raise exc.TritonError(
                    error=f"{type(e).__qualname__}: {e}",
                    decorator=decorator,
                    code=SUPPRESSED_TRITON_CODE_MSG,
                ) from e
            elif action == "warn":
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.log.warning(format_triton_compile_failure(config, e, self.kernel))
            else:
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.log.debug(f"Benchmarking failed: {type(e).__name__}: {e}")
            return inf

    def start_precompile_and_check_for_hangs(
        self, config: Config, fn: CompiledConfig
    ) -> PrecompileFuture:
        """
        Run the kernel in a spawned subprocess to detect hangs during compilation or execution.
        We use the subprocess timeout to guard against Triton kernels that never finish.
        We also do this in parallel (when called from parallel_benchmark) to do faster autotuning.
        Note that we compile in parallel, but we benchmark one-by-one to avoid noisy results.

        Args:
            config: The config that generated fn.
            fn: The function to be precompiled.

        Returns:
            True if the compilation was successful, False if it hung.
        """
        if not self.settings.autotune_precompile:
            return PrecompileFuture.skip(self, config, True)
        mode = self.settings.autotune_precompile
        if mode not in {"fork", "spawn"}:
            raise exc.InvalidAPIUsage("autotune_precompile must be 'fork' or 'spawn'")
        if self._kernel_mutates_args:
            device_args = self._clone_args(self._original_args)
        else:
            device_args = self.args

        decorator = self.kernel.format_kernel_decorator(config, self.settings)

        if mode == "spawn":
            ctx = mp.get_context("spawn")
            assert self._precompile_args_path is not None
            try:
                fn_spec = _serialize_compiled_fn(fn)
            except RuntimeError as err:
                raise exc.AutotuneError(
                    "Failed to serialize compiled kernel for spawn precompile."
                    ' Set HELION_AUTOTUNE_PRECOMPILE="fork" to fall back to fork mode.'
                ) from err
            result_path = self._next_precompile_result_path()
            process = cast(
                "mp.Process",
                ctx.Process(
                    target=_run_kernel_in_subprocess_spawn,
                    args=(fn_spec, self._precompile_args_path, result_path, decorator),
                ),
            )
            process.daemon = True
        else:
            precompiler = _prepare_precompiler_for_fork(
                fn, device_args, config, self.kernel, decorator
            )
            if precompiler is None:
                return PrecompileFuture.skip(self, config, True)
            ctx = mp.get_context("fork")
            result_path = self._next_precompile_result_path()
            process = cast(
                "mp.Process",
                ctx.Process(
                    target=_run_kernel_in_subprocess_fork,
                    args=(precompiler, config, self.kernel, result_path, decorator),
                ),
            )
            process.daemon = True
        return PrecompileFuture(
            search=self,
            config=config,
            process=process,
            timeout=self.settings.autotune_compile_timeout,
            result_path=result_path,
        )

    def parallel_benchmark(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[
        tuple[
            Config,
            Callable[..., object],
            float,
            Literal["ok", "error", "timeout"],
        ]
    ]:
        """
        Benchmark multiple configurations in parallel.

        Args:
            configs: A list of configurations to benchmark.
            desc: Description for the progress bar.

        Returns:
            A list of tuples containing configurations and their performance.
        """
        fns = [self.kernel.compile_config(c, allow_print=False) for c in configs]
        precompile_status: list[Literal["ok", "error", "timeout"]]
        if self.settings.autotune_precompile:
            futures = [
                *starmap(
                    self.start_precompile_and_check_for_hangs,
                    zip(configs, fns, strict=True),
                )
            ]
            is_workings = PrecompileFuture.wait_for_all(
                futures,
                desc=f"{desc} precompiling"
                if self.settings.autotune_progress_bar
                else None,
            )
            precompile_status = []
            for future, ok in zip(futures, is_workings, strict=True):
                reason = future.failure_reason
                if ok:
                    precompile_status.append("ok")
                elif reason == "timeout":
                    precompile_status.append("timeout")
                else:
                    precompile_status.append("error")
        else:
            is_workings = [True] * len(configs)
            precompile_status = ["ok"] * len(configs)
        results: list[
            tuple[
                Config, Callable[..., object], float, Literal["ok", "error", "timeout"]
            ]
        ] = []

        # Render a progress bar only when the user requested it.
        iterator = iter_with_progress(
            zip(configs, fns, is_workings, precompile_status, strict=True),
            total=len(configs),
            description=f"{desc} exploring neighbors",
            enabled=self.settings.autotune_progress_bar,
        )
        for config, fn, is_working, reason in iterator:
            status: Literal["ok", "error", "timeout"]
            if is_working:
                # benchmark one-by-one to avoid noisy results
                perf = self.benchmark_function(config, fn)
                status = "ok" if math.isfinite(perf) else "error"
                results.append((config, fn, perf, status))
            else:
                status = "timeout" if reason == "timeout" else "error"
                results.append((config, fn, inf, status))
        return results

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """
        Perform autotuning to find the best configuration.

        This method searches for the optimal configuration by benchmarking multiple configurations.

        Returns:
            The best configuration found during autotuning.
        """
        start = time.perf_counter()
        self.log.reset()
        exit_stack = contextlib.ExitStack()
        with exit_stack:
            # Autotuner triggers bugs in remote triton compile service
            exit_stack.enter_context(
                patch.dict(os.environ, {"TRITON_LOCAL_BUILD": "1"}, clear=False)
            )
            assert self._precompile_tmpdir is None
            tempdir = tempfile.TemporaryDirectory()
            self._precompile_tmpdir = tempdir
            if self.settings.autotune_precompile == "spawn":
                args_path = os.path.join(tempdir.name, "args.pt")
                torch.save(self.args, args_path)
                self._precompile_args_path = args_path
            exit_stack.callback(self.cleanup)
            best = self._autotune()
        end = time.perf_counter()
        kernel_decorator = self.kernel.format_kernel_decorator(best, self.settings)
        self.log(
            f"Autotuning complete in {end - start:.1f}s after searching {self.counters['benchmark']} configs.\n"
            "One can hardcode the best config and skip autotuning with:\n"
            f"    {kernel_decorator}\n",
            level=logging.INFO + 5,
        )
        if self.settings.print_output_code:
            triton_code = self.kernel.to_triton_code(best)
            print(triton_code, file=sys.stderr)
        return best

    def _autotune(self) -> Config:
        """
        Abstract method to perform the actual autotuning.

        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError


@dataclasses.dataclass
class PopulationMember:
    """
    Represents a member of the population in population-based search algorithms.

    Attributes:
        perfs (list[float]): The performance of the configuration, accumulated over multiple benchmarks.
        flat_values (FlatConfig): The flat representation of the configuration values.
        config (Config): The full configuration object.
    """

    fn: Callable[..., object]
    perfs: list[float]
    flat_values: FlatConfig
    config: Config
    status: Literal["ok", "error", "timeout", "unknown"] = "unknown"

    @property
    def perf(self) -> float:
        return self.perfs[-1]


def performance(member: PopulationMember) -> float:
    """
    Retrieve the performance of a population member.  Used as a sort key.

    Args:
        member: The population member.

    Returns:
        The performance of the member.
    """
    return member.perf


def _estimate_tree_bytes(obj: object) -> int:
    """Estimate the memory usage of a pytree of objects, counting shared storage only once."""
    total = 0
    seen_ptrs: set[int] = set()

    def _accumulate(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        size = tensor.element_size() * tensor.numel()
        try:
            storage = tensor.untyped_storage()
        except RuntimeError:
            pass
        else:
            ptr = storage.data_ptr()
            if ptr in seen_ptrs:
                return tensor
            seen_ptrs.add(ptr)
            size = storage.nbytes()
        total += size
        return tensor

    tree_map_only(torch.Tensor, _accumulate, obj)
    return total


class PopulationBasedSearch(BaseSearch):
    """
    Base class for search algorithms that use a population of configurations.

    Attributes:
        population (list[PopulationMember]): The current population of configurations.
        flat_spec (list[ConfigSpecFragment]): The flattened configuration specification.
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
    ) -> None:
        """
        Initialize the PopulationBasedSearch object.

        Args:
            kernel: The kernel to be tuned.
            args: The arguments to be passed to the kernel.
        """
        super().__init__(kernel, args)
        self.population: list[PopulationMember] = []
        overrides = self.settings.autotune_config_overrides or None
        self.config_gen: ConfigGeneration = ConfigGeneration(
            self.config_spec,
            overrides=overrides,
        )

    @property
    def best(self) -> PopulationMember:
        """
        Retrieve the best configuration in the population.

        Returns:
            The best population member.
        """
        return min(self.population, key=performance)

    def benchmark_flat(self, flat_values: FlatConfig) -> PopulationMember:
        """
        Benchmark a flat configuration.

        Args:
            flat_values: The flat configuration values.

        Returns:
            A population member with the benchmark results.
        """
        config = self.config_gen.unflatten(flat_values)
        fn, perf = self.benchmark(config)
        status: Literal["ok", "error"] = "ok" if math.isfinite(perf) else "error"
        return PopulationMember(fn, [perf], flat_values, config, status=status)

    def parallel_benchmark_flat(
        self, to_check: list[FlatConfig]
    ) -> list[PopulationMember]:
        """
        Benchmark multiple flat configurations in parallel.

        Args:
            to_check: A list of flat configurations to benchmark.

        Returns:
            A list of population members with the benchmark results.
        """
        result = [*map(self.make_unbenchmarked, to_check)]
        return self.parallel_benchmark_population(result)

    def make_unbenchmarked(self, flat_values: FlatConfig) -> PopulationMember:
        """
        Create a population member with unbenchmarked configuration.  You
        should pass the result of this to parallel_benchmark_population.

        Args:
            flat_values: The flat configuration values.

        Returns:
            A population member with undefined performance.
        """
        config = self.config_gen.unflatten(flat_values)
        return PopulationMember(_unset_fn, [], flat_values, config)

    def parallel_benchmark_population(
        self, members: list[PopulationMember], *, desc: str = "Benchmarking"
    ) -> list[PopulationMember]:
        """
        Benchmark multiple population members in parallel.  Members should be created with make_unbenchmarked.

        Args:
            members: The list of population members to benchmark.
            desc: Description for the progress bar.
        """
        for member, (config_out, fn, perf, status) in zip(
            members,
            self.parallel_benchmark([m.config for m in members], desc=desc),
            strict=True,
        ):
            assert config_out is member.config
            member.perfs.append(perf)
            member.fn = fn
            member.status = status
        return members

    def compare(self, a: PopulationMember, b: PopulationMember) -> int:
        """
        Compare two population members based on their performance, possibly with re-benchmarking.

        Args:
            a: The first population member.
            b: The second population member.

        Returns:
            -1 if a is better than b, 1 if b is better than a, 0 if they are equal.
        """
        if self.should_rebenchmark(a) and self.should_rebenchmark(b):
            self.rebenchmark([a, b])
        return (a.perf > b.perf) - (a.perf < b.perf)

    def should_rebenchmark(self, member: PopulationMember) -> bool:
        """
        Determine if a population member should be re-benchmarked to avoid outliers.

        Args:
            member: The population member to check.

        Returns:
            True if the member should be re-benchmarked, False otherwise.
        """
        threshold = self.settings.get_rebenchmark_threshold()
        return member.perf < threshold * self.best_perf_so_far and math.isfinite(
            member.perf
        )

    def rebenchmark(
        self, members: list[PopulationMember], *, desc: str = "Rebenchmarking"
    ) -> None:
        """
        Re-benchmark a list of population members to avoid outliers.

        Args:
            members: The list of population members to rebenchmark.
            desc: Description for the progress bar.
        """
        if len(members) < 2:
            return

        # Calculate repeat count based on best performance
        base_repeat = (
            int(200 / self.best_perf_so_far)
            if math.isfinite(self.best_perf_so_far) and self.best_perf_so_far > 0
            else 1000
        )
        repeat = min(1000, max(3, base_repeat))
        iterator = [functools.partial(m.fn, *self.args) for m in members]
        if self.settings.autotune_progress_bar:
            new_timings = interleaved_bench(iterator, repeat=repeat, desc=desc)
        else:
            new_timings = interleaved_bench(iterator, repeat=repeat)
        for m, t in zip(members, new_timings, strict=True):
            m.perfs.append(t)
            if t < self.best_perf_so_far:
                self.best_perf_so_far = t

    def rebenchmark_population(
        self,
        members: list[PopulationMember] | None = None,
        *,
        desc: str = "Rebenchmarking",
    ) -> None:
        """
        Re-benchmark the entire population to avoid outliers.

        Args:
            members: The list of population members to rebenchmark.
            desc: Description for the progress bar.
        """
        if members is None:
            members = self.population
        self.rebenchmark([p for p in members if self.should_rebenchmark(p)], desc=desc)

    def statistics(self) -> str:
        """
        Generate statistics for the current population.

        Returns:
            A string summarizing the population performance.
        """
        return population_statistics(self.population)


def population_statistics(population: list[PopulationMember]) -> str:
    """
    Create a summary of the population performance.

    Args:
        population: The population of configurations.

    Returns:
        A string summarizing the performance of the population.
    """
    population = sorted(population, key=performance)
    status_counts: collections.Counter[str] = collections.Counter()
    working: list[PopulationMember] = []
    for member in population:
        status = member.status
        if math.isfinite(member.perf):
            working.append(member)
            if status not in {"ok", "error", "timeout"}:
                status = "ok"
        else:
            if status not in {"error", "timeout"}:
                status = "error"
        if status == "timeout":
            status_counts["timeout"] += 1
        elif status == "error":
            status_counts["error"] += 1
        else:
            status_counts["ok"] += 1
    if len(working) == 0:
        raise exc.NoConfigFound
    parts: list[str] = []
    for label in ("error", "timeout", "ok"):
        count = status_counts.get(label, 0)
        if count:
            parts.append(f"{label}={count}")
    parts.extend(
        (
            f"min={working[0].perf:.4f}",
            f"mid={working[len(working) // 2].perf:.4f}",
            f"max={working[-1].perf:.4f}",
            f"best={population[0].config!s}",
        )
    )
    return " ".join(parts)


@dataclasses.dataclass
class PrecompileFuture:
    """
    Wraps a child process where we are precompiling a kernel.

    Attributes:
        search (BaseSearch): The search object that initiated the precompilation.
        config (Config): The configuration to be precompiled.
        process (mp.Process | None): The process running the precompilation.
        timeout (float): The timeout for the precompilation.
        start_time (float): The time when the precompilation started.
        end_time (float | None): The time when the precompilation ended.
        ok (bool | None): The result of the precompilation (True if successful, False otherwise).
    """

    search: BaseSearch
    config: Config
    process: mp.Process | None
    timeout: float
    # Set when the process is actually started. For queued futures this is None.
    start_time: float | None = None
    end_time: float | None = None
    ok: bool | None = None
    result_path: str | None = None
    _result_received: bool = False
    remote_error: RemoteError | None = None
    _remote_error_handled: bool = False
    failure_reason: Literal["ok", "error", "timeout"] | None = None

    @property
    def elapsed(self) -> float:
        """Return the elapsed time since the start of the precompilation."""
        if self.start_time is None:
            return 0.0
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def seconds_left(self) -> float:
        """Return the number of seconds left before the timeout."""
        if self.end_time is not None:
            return 0
        if self.start_time is None:
            return self.timeout
        return self.timeout - (time.time() - self.start_time)

    def is_alive(self) -> bool:
        """Check if the precompilation process is still alive."""
        if (p := self.process) is None:
            return False
        return p.is_alive()

    @property
    def started(self) -> bool:
        """Whether the process has been started."""
        return self.start_time is not None

    def start(self) -> None:
        """Start the underlying process and set the timer if not already started."""
        if self.process is None or self.started:
            return
        self.start_time = time.time()
        self.process.start()

    @staticmethod
    def skip(search: BaseSearch, config: Config, ok: bool) -> PrecompileFuture:
        """Dummy precompile future that is already done."""
        ts = time.time()
        return PrecompileFuture(
            search=search,
            config=config,
            process=None,
            timeout=0,
            ok=ok,
            start_time=ts,
            end_time=ts,
            result_path=None,
            _result_received=True,
            remote_error=None,
            _remote_error_handled=True,
            failure_reason="ok" if ok else "error",
        )

    def __call__(self) -> bool:
        """Wait for the precompilation to finish and return true on success."""
        if self.ok is not None:
            return self.ok
        process = self.process
        assert process is not None
        try:
            # Start now if not already started (single-future path)
            if not self.started:
                self.start()
            process.join(self.seconds_left())
        finally:
            self._mark_complete()
        self._consume_result(raise_on_raise=True)
        assert self.ok is not None
        return self.ok

    @staticmethod
    def wait_for_all(
        futures: list[PrecompileFuture],
        desc: str | None = None,
    ) -> list[bool]:
        """
        Wait for all precompile futures to complete.

        Args:
            futures: A list of PrecompileFuture objects.
            desc: Optional description used for the progress display.

        Returns:
            A list of boolean values indicating completion status.
        """
        progress = iter_with_progress(
            range(len(futures)),
            total=len(futures),
            description=desc,
            enabled=desc is not None,
        )
        next(progress, None)  # display the progress bar immediately
        progress_left = len(futures)
        remaining = [f for f in futures if f.ok is None]
        try:
            while remaining:
                remaining = PrecompileFuture._wait_for_all_step(remaining)
                while progress_left > len(remaining):
                    next(progress, None)
                    progress_left -= 1
        except BaseException:
            PrecompileFuture._cancel_all(futures)
            raise
        result = []
        for f in futures:
            assert f.ok is not None
            if f.failure_reason is None:
                f.failure_reason = "ok" if f.ok else "error"
            result.append(f.ok)
        return result

    @staticmethod
    def _wait_for_all_step(
        futures: list[PrecompileFuture],
    ) -> list[PrecompileFuture]:
        """Start up to the concurrency cap, wait for progress, and return remaining futures."""
        cap = futures[0].search._jobs if futures else 1
        running = [f for f in futures if f.started and f.ok is None and f.is_alive()]

        # Start queued futures up to the cap
        queued = collections.deque(f for f in futures if not f.started and f.ok is None)
        while len(running) < cap and queued:
            job = queued.popleft()
            job.start()
            if job.is_alive():
                running.append(job)

        # Wait for at least one to finish or time out
        timeout = min([f.seconds_left() for f in running], default=0.0)
        handles = [f.process.sentinel for f in running]  # pyright: ignore[reportOptionalMemberAccess]
        if handles and timeout > 0:
            connection.wait(handles, timeout)
        remaining: list[PrecompileFuture] = []
        for f in futures:
            if f.ok is not None:
                continue
            if f.started and (not f.is_alive() or f.seconds_left() <= 0):
                f._mark_complete()
                f._consume_result(raise_on_raise=True)
            else:
                remaining.append(f)
        return remaining

    @staticmethod
    def _cancel_all(futures: Iterable[PrecompileFuture]) -> None:
        """Cancel any futures that have not completed."""
        active = [future for future in futures if future.ok is None]
        for future in active:
            with contextlib.suppress(Exception):
                future._kill_without_wait()
        for future in active:
            with contextlib.suppress(Exception):
                future.cancel()

    def _kill_without_wait(self) -> None:
        """Issue a hard kill to the underlying process without waiting for exit."""
        process = self.process
        if process is None or not self.started:
            return
        if process.is_alive():
            with contextlib.suppress(Exception):
                process.kill()

    def cancel(self) -> None:
        """Terminate the underlying process (if any) without waiting for success."""
        self.end_time = time.time()
        process = self.process
        if process is not None:
            if self.started:
                with contextlib.suppress(Exception):
                    if process.is_alive():
                        process.kill()
                    process.join()
        if self.ok is None:
            self.ok = False
        if self.failure_reason is None:
            self.failure_reason = "error"
        self._consume_result(raise_on_raise=False)

    def _mark_complete(self) -> bool:
        """
        Mark the precompile future as complete and kill the process if needed.

        Returns:
            True if the precompilation was successful, False otherwise.
        """
        self.end_time = time.time()
        process = self.process
        assert process is not None
        # If the process hasn't been started yet (shouldn't happen in normal flow),
        # start and immediately terminate to maintain invariants.
        if not self.started:
            self.start()
        if not process.is_alive():
            self.ok = process.exitcode == 0
            self._consume_result(raise_on_raise=False)
            if self.ok:
                self.failure_reason = "ok"
            elif self.failure_reason is None:
                self.failure_reason = "error"
            return self.ok
        process.terminate()
        process.join(10)
        msg = f"Timeout after {self.elapsed:.0f}s compiling {self.config}"
        if process.is_alive():
            if not self.search.settings.autotune_ignore_errors:
                self.search.log.warning(
                    msg,
                    "(SIGKILL required)",
                )
            process.kill()
            process.join()
        else:
            if not self.search.settings.autotune_ignore_errors:
                self.search.log.warning(msg)

        self.ok = False
        self.failure_reason = "timeout"
        self._consume_result(raise_on_raise=False)
        return False

    def _consume_result(self, *, raise_on_raise: bool) -> None:
        if not self._result_received and self.result_path is not None:
            message_data: dict[str, object] | None = None
            try:
                with open(self.result_path, "rb") as f:
                    message_data = pickle.load(f)
            except FileNotFoundError:
                message_data = None
            except Exception as err:
                if self.remote_error is None:
                    self.remote_error = RemoteError(
                        exc_type=type(err).__name__,
                        exc_module=type(err).__module__,
                        exc_args=(str(err),),
                        traceback=None,
                        classification="warn",
                    )
            finally:
                with contextlib.suppress(Exception):
                    os.remove(self.result_path)
            if message_data is None:
                if self.failure_reason == "timeout":
                    # Timeout warnings have already been emitted; suppress secondary EOF logs.
                    self.remote_error = None
                    self._remote_error_handled = True
                elif self.remote_error is None:
                    self.remote_error = RemoteError(
                        exc_type="EOFError",
                        exc_module=__name__,
                        exc_args=("No result received from subprocess.",),
                        traceback=None,
                        classification="debug",
                    )
            elif message_data["status"] == "ok":
                if self.ok is None:
                    self.ok = True
                assert self.remote_error is None
            else:
                exc_args_obj = message_data["exc_args"]
                if isinstance(exc_args_obj, tuple):
                    exc_args_tuple: tuple[object, ...] = exc_args_obj
                else:
                    exc_args_tuple = tuple(cast("Iterable[object]", exc_args_obj))
                self.remote_error = RemoteError(
                    exc_type=cast("str", message_data["exc_type"]),
                    exc_module=cast("str | None", message_data["exc_module"]),
                    exc_args=exc_args_tuple,
                    traceback=cast("str | None", message_data["traceback"]),
                    classification=cast("str | None", message_data["classification"]),
                )
                self.ok = False
            self.result_path = None
            self._result_received = True

        error = self.remote_error
        if error is None or self._remote_error_handled:
            return
        exc_obj = error.to_exception()
        classification = error.classification or classify_triton_exception(exc_obj)
        if ignore_errors := self.search.settings.autotune_ignore_errors:
            classification = "debug"
        if classification == "raise":
            if raise_on_raise:
                self._remote_error_handled = True
                decorator = self.search.kernel.format_kernel_decorator(
                    self.config, self.search.settings
                )
                log_generated_triton_code_debug(
                    self.search.log,
                    self.search.kernel,
                    self.config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                raise exc.TritonError(
                    error=f"{type(exc_obj).__qualname__}: {exc_obj}",
                    decorator=decorator,
                    code=SUPPRESSED_TRITON_CODE_MSG,
                ) from exc_obj
            return

        decorator = self.search.kernel.format_kernel_decorator(
            self.config, self.search.settings
        )
        log_generated_triton_code_debug(
            self.search.log,
            self.search.kernel,
            self.config,
            prefix=f"Generated Triton code for {decorator}:",
        )
        formatted = format_triton_compile_failure(
            self.config, exc_obj, self.search.kernel
        )
        if error.traceback:
            formatted = (
                f"{formatted}\nRemote traceback (spawned process):\n{error.traceback}"
            )
        if classification == "warn":
            self.search.log.warning(formatted)
        elif not ignore_errors:
            self.search.log.debug(formatted)
        self._remote_error_handled = True


def _clone_tree(tree: object) -> object:
    def _clone(leaf: object) -> object:
        if isinstance(leaf, torch.Tensor):
            clone = leaf.detach().clone()
            clone.requires_grad_(leaf.requires_grad)
            return clone
        return leaf

    return tree_map(_clone, tree)


def _assert_args_close(actual: Sequence[object], expected: Sequence[object]) -> None:
    actual_flat, _ = tree_flatten(actual)
    expected_flat, _ = tree_flatten(expected)
    for act, exp in zip(actual_flat, expected_flat, strict=False):
        if isinstance(act, torch.Tensor) and isinstance(exp, torch.Tensor):
            torch.testing.assert_close(act, exp, atol=1e-2, rtol=1e-2)


def _write_result_file(result_path: str, message: dict[str, object]) -> None:
    tmp_path = f"{result_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(message, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, result_path)


def _run_kernel_in_subprocess_spawn(
    fn_spec: SerializedCompiledFunction,
    args_path: str,
    result_path: str,
    decorator: str,
) -> None:
    status = 0
    try:
        fn = _load_compiled_fn(fn_spec)
        args = torch.load(args_path)
        assert isinstance(args, (tuple, list))
        torch.accelerator.synchronize()
        fn(*args)
        torch.accelerator.synchronize()
        _write_result_file(result_path, {"status": "ok"})
    except Exception as exc:
        status = 1
        with contextlib.suppress(Exception):
            try:
                exc_args = tuple(exc.args)
            except Exception:
                exc_args = (str(exc),)
            try:
                classification = classify_triton_exception(exc)
            except Exception:
                classification = None
            _write_result_file(
                result_path,
                {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                    "decorator": decorator,
                    "exc_type": type(exc).__name__,
                    "exc_module": type(exc).__module__,
                    "exc_args": exc_args,
                    "classification": classification,
                },
            )
    finally:
        os._exit(status)


def _prepare_precompiler_for_fork(
    fn: CompiledConfig,
    args: Sequence[object],
    config: Config,
    kernel: BoundKernel,
    decorator: str,
) -> Callable[[], None] | None:
    def extract_launcher(
        triton_kernel: object,
        grid: tuple[int, ...],
        *launch_args: object,
        **launch_kwargs: object,
    ) -> NoReturn:
        raise _ExtractedLaunchArgs(triton_kernel, grid, launch_args, launch_kwargs)

    try:
        fn(*tuple(args), _launcher=extract_launcher)
        raise RuntimeError("Expected _ExtractedLaunchArgs to be raised")
    except _ExtractedLaunchArgs as extracted:
        precompiler_factory = make_precompiler(
            cast("Any", extracted.kernel),
            config,
            kernel,
        )
        precompiler = precompiler_factory(*extracted.args, **extracted.kwargs)
        if precompiler is already_compiled:
            return None
        return precompiler
    except Exception:
        log_generated_triton_code_debug(
            log,
            kernel,
            config,
            prefix=f"Generated Triton code for {decorator}:",
        )
        log.warning(
            "Helion autotuner precompile error for %s. %s",
            decorator,
            SUPPRESSED_TRITON_CODE_MSG,
            exc_info=True,
        )
        raise


def _run_kernel_in_subprocess_fork(
    precompiler: Callable[[], None],
    config: Config,
    kernel: BoundKernel,
    result_path: str,
    decorator: str,
) -> None:
    status = 0
    try:
        precompiler()
        _write_result_file(result_path, {"status": "ok"})
    except Exception as exc:
        status = 1
        with contextlib.suppress(Exception):
            try:
                exc_args = tuple(exc.args)
            except Exception:
                exc_args = (str(exc),)
            try:
                classification = classify_triton_exception(exc)
            except Exception:
                classification = None
            _write_result_file(
                result_path,
                {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                    "decorator": decorator,
                    "exc_type": type(exc).__name__,
                    "exc_module": type(exc).__module__,
                    "exc_args": exc_args,
                    "classification": classification,
                },
            )
    finally:
        os._exit(status)


class _ExtractedLaunchArgs(Exception):
    """Exception that carries kernel launch arguments for precompiler extraction."""

    def __init__(
        self,
        kernel: object,
        grid: tuple[int, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.grid = grid
        self.args = args
        self.kwargs = kwargs


def _unset_fn(*args: object) -> NoReturn:
    raise RuntimeError("Uninitialized function")


@dataclasses.dataclass
class SerializedCompiledFunction:
    function_name: str
    source_code: str
    filename: str | None
    module_name: str | None


@dataclasses.dataclass
class RemoteError:
    exc_type: str
    exc_module: str | None
    exc_args: tuple[object, ...]
    traceback: str | None
    classification: str | None

    def to_exception(self) -> Exception:
        exc_cls = types.new_class(self.exc_type, (Exception,))
        exc_cls.__module__ = self.exc_module or __name__
        exc_obj = exc_cls(*self.exc_args)
        exc_obj.remote_traceback = self.traceback
        return exc_obj


def _serialize_compiled_fn(fn: CompiledConfig) -> SerializedCompiledFunction:
    if "<locals>" in getattr(fn, "__qualname__", ""):
        raise RuntimeError("Unable to serialize nested compiled functions")
    module_name = getattr(fn, "__module__", None)
    module = sys.modules.get(module_name) if module_name is not None else None
    filename: str | None = None
    source_code: str | None = None
    if module is not None:
        filename = getattr(module, "__file__", None)
        if filename is not None and os.path.exists(filename):
            source_code = Path(filename).read_text(encoding="utf-8")
        if source_code is None:
            with contextlib.suppress(OSError, TypeError):
                source_code = inspect.getsource(module)
    if source_code is None:
        raise RuntimeError("Unable to capture source for compiled kernel")
    return SerializedCompiledFunction(
        function_name=fn.__name__,
        source_code=source_code,
        filename=filename,
        module_name=module_name,
    )


def _load_compiled_fn(fn_spec: SerializedCompiledFunction) -> CompiledConfig:
    module_name = f"_helion_autotune_subprocess_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.__file__ = fn_spec.filename or "<helion-autotune-subprocess>"
    module.__loader__ = None
    module.__package__ = None
    sys.modules[module_name] = module
    exec(
        compile(fn_spec.source_code, module.__file__, "exec"),
        module.__dict__,
    )
    fn = getattr(module, fn_spec.function_name, None)
    if fn is None:
        raise RuntimeError(
            f"Unable to locate compiled kernel '{fn_spec.function_name}' in generated module"
        )
    return fn
