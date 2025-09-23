from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import functools
import logging
import math
from math import inf
from multiprocessing import connection
import os
import pathlib
import random
import shutil
import sys
import tempfile
import time
import uuid
import weakref
from itertools import starmap
import importlib.util
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import NoReturn

if TYPE_CHECKING:
    from triton.runtime.jit import JITFunction

import torch
import torch.multiprocessing as mp
from torch.utils._pytree import tree_map
from triton.testing import do_bench

from .. import exc
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import make_precompiler
from .config_generation import ConfigGeneration
from .config_generation import FlatConfig
from .logger import LambdaLogger
from .logger import classify_triton_exception
from .logger import format_triton_compile_failure

log = logging.getLogger(__name__)


def _env_flag(name: str, default: str) -> bool:
    value = os.environ.get(name, default)
    return value.lower() not in {"0", "false", "no"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


_USE_PERSISTENT_BENCHMARK_WORKER = _env_flag(
    "HELION_AUTOTUNE_PERSISTENT_WORKER",
    "1",
)

_PERSISTENT_WORKER_POOL_SIZE = _env_int(
    "HELION_AUTOTUNE_PERSISTENT_WORKER_POOL_SIZE",
    1,
    minimum=1,
)

_AUTOTUNE_TIMING = _env_flag(
    "HELION_AUTOTUNE_TIMING",
    "0",
)


@dataclasses.dataclass
class _WorkerTimingStats:
    spawn_time: float = 0.0
    spawn_count: int = 0
    restart_time: float = 0.0
    restart_count: int = 0


@dataclasses.dataclass
class _PersistentTimingStats:
    handle_time: float = 0.0
    send_time: float = 0.0
    worker_load_time: float = 0.0
    worker_warmup_time: float = 0.0
    worker_bench_time: float = 0.0
    worker_postprocess_time: float = 0.0
    worker_send_time: float = 0.0
    wait_startup_time: float = 0.0
    wait_idle_time: float = 0.0
    call_count: int = 0
    startup_call_count: int = 0
    worker_spawn_time: float = 0.0
    worker_spawn_count: int = 0
    worker_restart_time: float = 0.0
    worker_restart_count: int = 0

    def merge_worker(self, stats: _WorkerTimingStats) -> None:
        self.worker_spawn_time += stats.spawn_time
        self.worker_spawn_count += stats.spawn_count
        self.worker_restart_time += stats.restart_time
        self.worker_restart_count += stats.restart_count


@dataclasses.dataclass
class _BaselineTimingStats:
    warmup_time: float = 0.0
    bench_time: float = 0.0
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0

if TYPE_CHECKING:
    from collections.abc import Sequence

    import triton

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import CompiledConfig
    from ..runtime.settings import Settings
    from . import ConfigSpec


class BaseAutotuner(abc.ABC):
    """
    Abstract base class for all autotuners and classes that wrap autotuners, like caching.
    """

    @abc.abstractmethod
    def autotune(self) -> Config:
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
        # Store args as an immutable tuple so we can ship them to subprocesses safely.
        self.args = tuple(args)
        self.counters: collections.Counter[str] = collections.Counter()
        self.log = LambdaLogger(self.settings.autotune_log_level)
        random.seed(self.settings.autotune_random_seed)
        self._use_persistent_worker = _USE_PERSISTENT_BENCHMARK_WORKER
        self._timing_enabled = _AUTOTUNE_TIMING
        if self._timing_enabled:
            self._persistent_timing_stats = _PersistentTimingStats()
            self._baseline_timing_stats = _BaselineTimingStats()
        else:
            self._persistent_timing_stats = None
            self._baseline_timing_stats = None
        if self._use_persistent_worker:
            # Worker processes must never see tensors that require grad; detach eagerly.
            self._worker_args = _sanitize_worker_args(self.args)
            self._worker_pool_size = max(1, _PERSISTENT_WORKER_POOL_SIZE)
            self._benchmark_workers: list[_BenchmarkWorker | None] = [
                None
                for _ in range(self._worker_pool_size)
            ]
            self._worker_finalizers: list[weakref.finalize | None] = [
                None
                for _ in range(self._worker_pool_size)
            ]
            self._worker_index = 0
            self._active_worker_index: int | None = None
        else:
            self._worker_args = self.args
            self._worker_pool_size = 0
            self._benchmark_workers = []
            self._worker_finalizers = []
            self._worker_index = 0
            self._active_worker_index = None

    def benchmark(self, config: Config) -> float:
        """
        Benchmark a specific configuration.

        This method compiles the kernel with the given configuration and measures its performance.

        Args:
            config: The configuration to benchmark.

        Returns:
            The performance of the configuration in seconds.
        """
        fn = self.kernel.compile_config(config, allow_print=False)
        if self.start_precompile_and_check_for_hangs(config, fn)():
            return self.benchmark_function(config, fn)
        return inf

    def _ensure_worker_spawned(self, index: int) -> "_BenchmarkWorker":
        worker: _BenchmarkWorker | None = None
        if 0 <= index < len(self._benchmark_workers):
            worker = self._benchmark_workers[index]
        if worker is None or not worker.is_alive():
            if worker is not None:
                self._drain_worker_timing(worker)
                worker.close()
            finalizer = None
            if 0 <= index < len(self._worker_finalizers):
                finalizer = self._worker_finalizers[index]
            if finalizer is not None:
                finalizer.detach()
            worker = _BenchmarkWorker(
                self._worker_args,
                timing_enabled=self._timing_enabled,
            )
            if 0 <= index < len(self._benchmark_workers):
                self._benchmark_workers[index] = worker
            if 0 <= index < len(self._worker_finalizers):
                self._worker_finalizers[index] = weakref.finalize(
                    self,
                    _close_worker,
                    worker,
                )
        worker.schedule_warmup()
        return worker

    def _get_benchmark_worker(self) -> tuple[int, "_BenchmarkWorker"]:
        if self._worker_pool_size == 0:
            raise RuntimeError("Persistent worker requested while disabled")
        index = self._worker_index
        worker = self._ensure_worker_spawned(index)
        worker.drain_warmup_ack()
        next_index = (index + 1) % self._worker_pool_size
        if self._worker_pool_size > 1:
            self._ensure_worker_spawned(next_index)
        self._worker_index = next_index
        return index, worker

    def _restart_benchmark_worker(self, index: int | None = None) -> None:
        if self._worker_pool_size == 0:
            return
        target_index = index
        if target_index is None:
            target_index = self._active_worker_index
        if target_index is None:
            return
        worker: _BenchmarkWorker | None = None
        if 0 <= target_index < len(self._benchmark_workers):
            worker = self._benchmark_workers[target_index]
        if worker is None:
            self._ensure_worker_spawned(target_index)
            return
        # Kill the existing subprocess to recover from CUDA failures or broken pipes.
        worker.restart()

    def _close_benchmark_worker(self) -> None:
        if not self._use_persistent_worker:
            return
        for index, worker in enumerate(self._benchmark_workers):
            if worker is None:
                continue
            worker.drain_warmup_ack(block=False)
            # Used at the end of autotune to ensure no stray subprocesses linger.
            self._drain_worker_timing(worker)
            worker.close()
            self._benchmark_workers[index] = None
        for index, finalizer in enumerate(self._worker_finalizers):
            if finalizer is None:
                continue
            finalizer.detach()
            self._worker_finalizers[index] = None
        self._active_worker_index = None
        self._worker_index = 0

    def _drain_worker_timing(self, worker: "_BenchmarkWorker") -> None:
        stats = worker.consume_timing_stats()
        if not self._timing_enabled:
            return
        if self._persistent_timing_stats is None:
            return
        self._persistent_timing_stats.merge_worker(stats)

    def _log_persistent_worker_summary(self) -> None:
        if not self._timing_enabled:
            return
        stats = self._persistent_timing_stats
        if stats is None:
            return
        calls = stats.call_count
        total_spawn = stats.worker_spawn_count
        total_restart = stats.worker_restart_count
        if calls == 0 and total_spawn == 0 and total_restart == 0:
            return

        def ms(value: float) -> float:
            return value * 1e3

        parts: list[str] = []
        if calls:
            handle_total_ms = ms(stats.handle_time)
            send_total_ms = ms(stats.send_time)
            load_total_ms = ms(stats.worker_load_time)
            warmup_total_ms = ms(stats.worker_warmup_time)
            bench_total_ms = ms(stats.worker_bench_time)
            postprocess_total_ms = ms(stats.worker_postprocess_time)
            worker_send_total_ms = ms(stats.worker_send_time)
            startup_wait_total_ms = ms(stats.wait_startup_time)
            wait_idle_total_ms = ms(stats.wait_idle_time)
            transport_total_ms = (
                send_total_ms
                + load_total_ms
                + warmup_total_ms
                + postprocess_total_ms
                + worker_send_total_ms
                + startup_wait_total_ms
                + wait_idle_total_ms
            )
            total_ms = handle_total_ms + transport_total_ms + bench_total_ms
            parts.append(f"calls={calls}")
            parts.append(
                f"handle_avg={handle_total_ms / calls:.3f}ms (total={handle_total_ms:.1f}ms)"
            )
            parts.append(
                f"send_avg={send_total_ms / calls:.3f}ms (total={send_total_ms:.1f}ms)"
            )
            parts.append(
                f"load_avg={load_total_ms / calls:.3f}ms (total={load_total_ms:.1f}ms)"
            )
            parts.append(
                f"warmup_avg={warmup_total_ms / calls:.3f}ms (total={warmup_total_ms:.1f}ms)"
            )
            parts.append(
                f"bench_avg={bench_total_ms / calls:.3f}ms (total={bench_total_ms:.1f}ms)"
            )
            parts.append(
                f"postprocess_avg={postprocess_total_ms / calls:.3f}ms (total={postprocess_total_ms:.1f}ms)"
            )
            parts.append(
                f"worker_send_avg={worker_send_total_ms / calls:.3f}ms (total={worker_send_total_ms:.1f}ms)"
            )
            if stats.startup_call_count:
                parts.append(
                    f"startup_wait_avg={startup_wait_total_ms / stats.startup_call_count:.3f}ms "
                    f"(total={startup_wait_total_ms:.1f}ms, events={stats.startup_call_count})"
                )
            else:
                parts.append("startup_wait_avg=0.000ms (total=0.0ms, events=0)")
            parts.append(
                f"idle_wait_avg={wait_idle_total_ms / calls:.3f}ms (total={wait_idle_total_ms:.1f}ms)"
            )
            parts.append(
                f"transport_avg={transport_total_ms / calls:.3f}ms (total={transport_total_ms:.1f}ms)"
            )
            parts.append(
                f"total_avg={total_ms / calls:.3f}ms (total={total_ms:.1f}ms)"
            )
            for label, value in (
                ("handle", handle_total_ms),
                ("send", send_total_ms),
                ("load", load_total_ms),
                ("warmup", warmup_total_ms),
                ("bench", bench_total_ms),
                ("postprocess", postprocess_total_ms),
                ("worker_send", worker_send_total_ms),
                ("startup_wait", startup_wait_total_ms),
                ("idle_wait", wait_idle_total_ms),
            ):
                if total_ms > 0 and value > 0:
                    pct = 100.0 * value / total_ms
                    parts.append(f"{label}_pct={pct:.1f}%")
        if total_spawn:
            parts.append(
                f"spawn_total={stats.worker_spawn_time:.3f}s (count={total_spawn})"
            )
        if total_restart:
            parts.append(
                f"restart_total={stats.worker_restart_time:.3f}s (count={total_restart})"
            )
        if not parts:
            return
        summary = "Persistent worker timing summary: " + ", ".join(parts)
        # Surface timing breakdown at an info-adjacent level so it shows up with the
        # default autotune log level (INFO) whenever timing is enabled.
        self.log(summary, level=logging.INFO + 1)
        self._persistent_timing_stats = _PersistentTimingStats()

    def _log_baseline_timing_summary(self) -> None:
        if not self._timing_enabled:
            return
        stats = self._baseline_timing_stats
        if stats is None:
            return
        if stats.call_count == 0:
            return
        warmup_ms = stats.warmup_time * 1e3
        bench_ms = stats.bench_time * 1e3
        total_ms = warmup_ms + bench_ms
        parts: list[str] = [f"calls={stats.call_count}"]
        if stats.success_count:
            parts.append(f"successes={stats.success_count}")
        if stats.failure_count:
            parts.append(f"failures={stats.failure_count}")
        if stats.success_count:
            parts.append(
                f"warmup_avg={warmup_ms / stats.success_count:.3f}ms (total={warmup_ms:.1f}ms)"
            )
            parts.append(
                f"bench_avg={bench_ms / stats.success_count:.3f}ms (total={bench_ms:.1f}ms)"
            )
            parts.append(
                f"total_avg={total_ms / stats.success_count:.3f}ms (total={total_ms:.1f}ms)"
            )
            if total_ms > 0:
                warmup_pct = 100.0 * warmup_ms / total_ms
                bench_pct = 100.0 * bench_ms / total_ms
                parts.append(f"warmup_pct={warmup_pct:.1f}%")
                parts.append(f"bench_pct={bench_pct:.1f}%")
        else:
            parts.append(f"warmup_total={warmup_ms:.1f}ms")
            parts.append(f"bench_total={bench_ms:.1f}ms")
            parts.append(f"total_total={total_ms:.1f}ms")
        summary = "Baseline benchmark timing summary: " + ", ".join(parts)
        self.log(summary, level=logging.INFO + 1)
        self._baseline_timing_stats = _BaselineTimingStats()

    def benchmark_function(self, config: Config, fn: CompiledConfig) -> float:
        """
        Benchmark a compiled function.  This function is called by the autotuner to measure the
        performance of a specific configuration.

        Args:
            config: The configuration to benchmark.
            fn: A precompiled version of config.

        Returns:
            The performance of the configuration in seconds.
        """
        self.counters["benchmark"] += 1
        self.log.debug(lambda: f"Running benchmark for {config!r}")
        if not self._use_persistent_worker:
            try:
                # TODO(jansel): early exit with fewer trials if early runs are slow
                t0 = time.perf_counter()
                fn(*self.args)  # make sure the kernel is compiled
                t1 = time.perf_counter()
                res = do_bench(
                    functools.partial(fn, *self.args),
                    return_mode="median",
                )
                t2 = time.perf_counter()
                if self._baseline_timing_stats is not None:
                    self._baseline_timing_stats.warmup_time += t1 - t0
                    self._baseline_timing_stats.bench_time += t2 - t1
                    self._baseline_timing_stats.call_count += 1
                    self._baseline_timing_stats.success_count += 1
                self.log.debug(
                    lambda: f"result: {res:.4f}ms (took {t1 - t0:.1f}s + {t2 - t1:.1f}s)",
                )
                return res  # pyright: ignore[reportReturnType]
            except Exception as e:
                action = classify_triton_exception(e)
                if action == "raise":
                    raise exc.TritonError(
                        f"{type(e).__qualname__}: {e}",
                        self.kernel.format_kernel_decorator(config, self.settings),
                    ) from e
                if action == "warn":
                    self.log.warning(format_triton_compile_failure(config, e))
                else:
                    self.log.debug(f"Benchmarking failed: {type(e).__name__}: {e}")
                if self._baseline_timing_stats is not None:
                    self._baseline_timing_stats.failure_count += 1
                    self._baseline_timing_stats.call_count += 1
                return inf

        else:
            worker_index, worker = self._get_benchmark_worker()
            self._active_worker_index = worker_index
            timing_enabled = self._timing_enabled
            total_start = time.perf_counter() if timing_enabled else None
            handle = _CompiledFnHandle.from_callable(fn)
            handle_ready = time.perf_counter() if timing_enabled else None
            is_cold_start = getattr(worker, "awaiting_first_request", False)
            # Offload the execution to the persistent subprocess so the main process never touches CUDA.
            (
                status,
                payload,
                send_elapsed,
                wait_elapsed,
                worker_send_elapsed,
            ) = worker.benchmark(handle)
            total_end = time.perf_counter() if timing_enabled else None
            bench_elapsed: float | None = None
            load_elapsed: float | None = None
            warmup_elapsed: float | None = None
            postprocess_elapsed: float | None = None
            result: float | None = None
            if status == "ok":
                (
                    result,
                    bench_elapsed,
                    load_elapsed,
                    warmup_elapsed,
                    postprocess_elapsed,
                    _unused_worker_send,
                ) = payload
                result = float(result)
            if (
                timing_enabled
                and self._persistent_timing_stats is not None
                and total_start is not None
                and handle_ready is not None
                and total_end is not None
            ):
                stats = self._persistent_timing_stats
                stats.handle_time += handle_ready - total_start
                if send_elapsed is not None:
                    stats.send_time += send_elapsed
                if load_elapsed is not None:
                    stats.worker_load_time += load_elapsed
                if warmup_elapsed is not None:
                    stats.worker_warmup_time += warmup_elapsed
                if bench_elapsed is not None:
                    stats.worker_bench_time += bench_elapsed
                if postprocess_elapsed is not None:
                    stats.worker_postprocess_time += postprocess_elapsed
                if worker_send_elapsed is not None:
                    stats.worker_send_time += worker_send_elapsed
                if wait_elapsed is not None:
                    deducted = 0.0
                    for piece in (
                        load_elapsed,
                        warmup_elapsed,
                        bench_elapsed,
                        postprocess_elapsed,
                        worker_send_elapsed,
                    ):
                        if piece is not None:
                            deducted += piece
                    overhead = wait_elapsed - deducted
                    if is_cold_start:
                        stats.startup_call_count += 1
                        if overhead > 0:
                            stats.wait_startup_time += overhead
                    elif overhead > 0:
                        stats.wait_idle_time += overhead
                elif is_cold_start:
                    stats.startup_call_count += 1
                stats.call_count += 1
            if status == "ok" and result is not None:
                self.log.debug(lambda: f"result: {result:.4f}ms")
                self._active_worker_index = worker_index
                return result

            action, message = payload
            # Reset the worker so subsequent benchmarks get a fresh CUDA context without any CUDA errors.
            self._restart_benchmark_worker(worker_index)
            if self._worker_pool_size > 0:
                self._ensure_worker_spawned(worker_index)
            self._active_worker_index = worker_index
            if action == "raise":
                raise exc.TritonError(
                    message,
                    self.kernel.format_kernel_decorator(config, self.settings),
                )
            if action == "warn":
                self.log.warning("Benchmarking failed for %s: %s", config, message)
            else:
                self.log.debug(lambda: f"Benchmarking failed: {message}")
            return inf

    def start_precompile_and_check_for_hangs(
        self, config: Config, fn: CompiledConfig
    ) -> PrecompileFuture:
        """
        Unfortunately, Triton can hang when compiling a kernel. This function tries to
        compile the kernel with the given configuration and checks if it hangs in a subprocess.
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
        ctx = mp.get_context("fork")

        def extract_launcher(
            triton_kernel: triton.JITFunction,
            grid: tuple[int, ...],
            *args: object,
            **kwargs: object,
        ) -> NoReturn:
            """Custom launcher that extracts arguments instead of executing."""
            raise _ExtractedLaunchArgs(triton_kernel, grid, args, kwargs)

        try:
            # Call main function with extraction launcher to extract arguments
            fn(*self.args, _launcher=extract_launcher)
            # Should not reach here
            raise RuntimeError("Expected _ExtractedLaunchArgs exception")
        except _ExtractedLaunchArgs as e:
            precompiler = make_precompiler(e.kernel, config)(*e.args, **e.kwargs)
            if precompiler is already_compiled:
                return PrecompileFuture.skip(self, config, True)
        except Exception:
            log.warning(
                "Helion autotuner precompile error for %s",
                self.kernel.format_kernel_decorator(config, self.settings),
                exc_info=True,
            )
            raise
        process: mp.Process = ctx.Process(target=precompiler)  # pyright: ignore[reportAssignmentType]
        return PrecompileFuture(
            search=self,
            config=config,
            process=process,
            timeout=self.settings.autotune_compile_timeout,
        )

    def parallel_benchmark(self, configs: list[Config]) -> list[tuple[Config, float]]:
        """
        Benchmark multiple configurations in parallel.

        Args:
            configs: A list of configurations to benchmark.

        Returns:
            A list of tuples containing configurations and their performance.
        """
        fns = [self.kernel.compile_config(c, allow_print=False) for c in configs]
        if self.settings.autotune_precompile:
            is_workings = PrecompileFuture.wait_for_all(
                [
                    *starmap(
                        self.start_precompile_and_check_for_hangs,
                        zip(configs, fns, strict=True),
                    )
                ]
            )
        else:
            is_workings = [True] * len(configs)
        results = []
        for config, fn, is_working in zip(configs, fns, is_workings, strict=True):
            if is_working:
                # benchmark one-by-one to avoid noisy results
                results.append((config, self.benchmark_function(config, fn)))
            else:
                results.append((config, inf))
        return results

    def autotune(self) -> Config:
        """
        Perform autotuning to find the best configuration.

        This method searches for the optimal configuration by benchmarking multiple configurations.

        Returns:
            The best configuration found during autotuning.
        """
        start = time.perf_counter()
        self.log.reset()
        best = self._autotune()
        end = time.perf_counter()
        kernel_decorator = self.kernel.format_kernel_decorator(best, self.settings)
        self.log(
            f"Autotuning complete in {end - start:.1f}s after searching {self.counters['benchmark']} configs.\n"
            "One can hardcode the best config and skip autotuning with:\n"
            f"    {kernel_decorator}\n",
            level=logging.INFO + 5,
        )

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except torch.cuda.CudaError as err:
                log.debug(
                    "Ignoring CUDA error while flushing after autotune: %s",
                    err,
                )

        if self.settings.print_output_code:
            triton_code = self.kernel.to_triton_code(best)
            print(triton_code, file=sys.stderr)
        self._close_benchmark_worker()
        self._log_baseline_timing_summary()
        self._log_persistent_worker_summary()
        return best

    def _autotune(self) -> Config:
        """
        Abstract method to perform the actual autotuning.

        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError


_PERSISTENT_MODULE_ROOT = pathlib.Path(tempfile.gettempdir()) / "helion_autotune_modules"
_PERSISTED_MODULES: dict[str, str] = {}


def _persist_compiled_module(module_path: str) -> str:
    cached = _PERSISTED_MODULES.get(module_path)
    if cached and pathlib.Path(cached).exists():
        return cached

    src_path = pathlib.Path(module_path)
    if not src_path.exists():
        if cached:
            return cached
        raise FileNotFoundError(module_path)

    # Snapshot the Triton-generated cache directory so repeated loads keep working even if
    # the original file is cleaned up (e.g. by torch.compile self-cleaning).
    _PERSISTENT_MODULE_ROOT.mkdir(parents=True, exist_ok=True)
    dest_dir = pathlib.Path(
        tempfile.mkdtemp(prefix="module_", dir=_PERSISTENT_MODULE_ROOT)
    )
    for sibling in src_path.parent.iterdir():
        if sibling.is_file():
            shutil.copy2(sibling, dest_dir / sibling.name)
    dest_file = dest_dir / src_path.name
    result = str(dest_file)
    _PERSISTED_MODULES[module_path] = result
    return result


def _detach_tensor_if_needed(value: object) -> object:
    if isinstance(value, torch.Tensor) and value.requires_grad:
        # Preserve the requires_grad flag so the benchmark sees the same autograd behavior
        # without trying to serialize a view that tracks gradients across process boundaries.
        return value.detach().requires_grad_(True)
    return value


def _sanitize_worker_args(args: tuple[object, ...]) -> tuple[object, ...]:
    """Detach any requires_grad tensors (even inside nested pytree structures) before sending to workers."""
    return tuple(tree_map(_detach_tensor_if_needed, arg) for arg in args)


###############################################################################
# Compiled function handles
###############################################################################


@dataclasses.dataclass(frozen=True)
class _CompiledFnHandle:
    module_path: str
    function_name: str

    @staticmethod
    def from_callable(fn: "CompiledConfig") -> "_CompiledFnHandle":
        module = sys.modules.get(fn.__module__)
        if module is None:
            raise RuntimeError(
                f"Compiled config module {fn.__module__!r} was not found in sys.modules"
            )
        module_path = getattr(module, "__file__", None)
        if not module_path:
            raise RuntimeError(
                f"Compiled config module {fn.__module__!r} does not have a __file__ attribute"
            )
        # The handle stores a stable path we control instead of the potentially ephemeral torch cache.
        persisted_path = _persist_compiled_module(module_path)
        return _CompiledFnHandle(persisted_path, fn.__name__)

    def load(self) -> "CompiledConfig":
        # Some PyTorch builds lazily delete cached files, so ensure the module is
        # persisted for the lifetime of the benchmark.
        module_path = pathlib.Path(self.module_path)
        if not module_path.exists():
            module_path = pathlib.Path(_persist_compiled_module(self.module_path))
        spec = importlib.util.spec_from_file_location(
            f"helion_autotune_worker_{uuid.uuid4().hex}",
            str(module_path),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Failed to create module spec for compiled function at {module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None
        loader.exec_module(module)
        try:
            fn = getattr(module, self.function_name)
        except AttributeError as err:  # pragma: no cover - best effort error message
            raise RuntimeError(
                f"Compiled function {self.function_name!r} not found in {module_path}"
            ) from err
        return fn


def _run_benchmark(
    fn_handle: "_CompiledFnHandle",
    args: tuple[object, ...],
    *,
    record_time: bool,
    ) -> tuple[
    float,
    float | None,
    float | None,
    float | None,
]:
    # Load the callable on-demand inside the worker process and measure it in isolation.
    load_start = time.perf_counter() if record_time else None
    fn = fn_handle.load()
    load_elapsed = (
        time.perf_counter() - load_start if record_time and load_start is not None else None
    )

    warmup_start = time.perf_counter() if record_time else None
    fn(*args)
    warmup_elapsed = (
        time.perf_counter() - warmup_start
        if record_time and warmup_start is not None
        else None
    )

    bench_start = time.perf_counter() if record_time else None
    res = do_bench(
        functools.partial(fn, *args),
        return_mode="median",
    )
    bench_elapsed = (
        time.perf_counter() - bench_start
        if record_time and bench_start is not None
        else None
    )

    # Caller will measure postprocess/send times separately.
    return float(res), bench_elapsed, load_elapsed, warmup_elapsed


def _benchmark_worker_entry(
    conn: connection.Connection,
    args: tuple[object, ...],
    timing_enabled: bool,
) -> None:
    # Simple message protocol: parent sends either ("benchmark", handle) or ("close", None).
    while True:
        try:
            message, payload = conn.recv()
        except EOFError:
            break
        if message == "close":
            break
        if message != "benchmark":
            if message == "warmup":
                warmup_elapsed = None
                if timing_enabled:
                    start = time.perf_counter()
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except torch.cuda.CudaError:
                        pass
                    warmup_elapsed = time.perf_counter() - start
                conn.send(("warmup_ack", None))
                if timing_enabled:
                    conn.send(("timing", warmup_elapsed))
                continue
            continue
        handle: _CompiledFnHandle = payload
        try:
            result, bench_elapsed, load_elapsed, warmup_elapsed = _run_benchmark(
                handle,
                args,
                record_time=timing_enabled,
            )
        except Exception as e:  # pragma: no cover - GPU errors are hard to simulate in CI
            conn.send(
                (
                    "error",
                    (
                        classify_triton_exception(e),
                        f"{type(e).__qualname__}: {e}",
                    ),
                )
            )
            conn.send(("timing", None))
        else:
            postprocess_elapsed = None
            worker_send_elapsed = None
            send_start = None
            if timing_enabled:
                postprocess_start = time.perf_counter()
            else:
                postprocess_start = None
            payload = (
                result,
                bench_elapsed,
                load_elapsed,
                warmup_elapsed,
                postprocess_elapsed,
                None,
            )
            if timing_enabled and postprocess_start is not None:
                send_start = time.perf_counter()
                postprocess_elapsed = send_start - postprocess_start
                payload = (
                    result,
                    bench_elapsed,
                    load_elapsed,
                    warmup_elapsed,
                    postprocess_elapsed,
                    None,
                )
            if send_start is None and timing_enabled:
                send_start = time.perf_counter()
            conn.send(("ok", payload))
            if timing_enabled and send_start is not None:
                worker_send_elapsed = time.perf_counter() - send_start
            conn.send(("timing", worker_send_elapsed))
    conn.close()


class _BenchmarkWorker:
    """Persistent benchmark subprocess that executes configurations sequentially.

    The worker stores no global CUDA state in the parent and communicates strictly via pipes,
    which lets us reuse the same process across many benchmark calls while keeping the main
    process CUDA-free.
    """

    def __init__(self, args: tuple[object, ...], *, timing_enabled: bool) -> None:
        self.args = args
        self.timing_enabled = timing_enabled
        # Spawn isolates CUDA initialization to the child process.
        self.ctx = mp.get_context("spawn")
        self.parent_conn: connection.Connection | None = None
        self.process: mp.Process | None = None
        self._timing_stats = _WorkerTimingStats()
        self._awaiting_first_request = False
        self._warmup_pending = False
        self._start()

    def _start(self) -> None:
        start = time.perf_counter() if self.timing_enabled else None
        parent_conn, child_conn = self.ctx.Pipe(duplex=True)
        process = self.ctx.Process(
            target=_benchmark_worker_entry,
            args=(child_conn, self.args, self.timing_enabled),
        )
        # Child will loop forever handling benchmark requests until we send "close".
        process.start()
        child_conn.close()
        self.parent_conn = parent_conn
        self.process = process
        self._awaiting_first_request = True
        self._warmup_pending = False
        if self.timing_enabled and start is not None:
            elapsed = time.perf_counter() - start
            self._timing_stats.spawn_time += elapsed
            self._timing_stats.spawn_count += 1

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    @property
    def awaiting_first_request(self) -> bool:
        return self._awaiting_first_request

    def benchmark(
        self, handle: "_CompiledFnHandle"
    ) -> tuple[str, object, float | None, float | None, float | None]:
        self.drain_warmup_ack()
        attempts = 0
        while attempts < 2:
            if not self.is_alive():
                self._restart_internal()
            try:
                assert self.parent_conn is not None
                # Each request contains only the compiled handle; args were fixed at construction.
                send_start = time.perf_counter() if self.timing_enabled else None
                self.parent_conn.send(("benchmark", handle))
                send_end = (
                    time.perf_counter()
                    if self.timing_enabled and send_start is not None
                    else None
                )
                result = self.parent_conn.recv()
                timing_msg = self.parent_conn.recv()
                send_elapsed: float | None = None
                wait_elapsed: float | None = None
                if (
                    self.timing_enabled
                    and send_start is not None
                    and send_end is not None
                ):
                    recv_end = time.perf_counter()
                    send_elapsed = send_end - send_start
                    wait_elapsed = recv_end - send_end
                status, payload = result
                timing_status, worker_send_elapsed = timing_msg
                if timing_status != "timing":
                    raise RuntimeError("Unexpected timing message from benchmark worker")
                self._awaiting_first_request = False
                return status, payload, send_elapsed, wait_elapsed, worker_send_elapsed
            except (BrokenPipeError, EOFError, OSError):
                # Broken connection: restart and retry once before giving up.
                attempts += 1
                self._restart_internal()
        raise RuntimeError("Benchmark worker repeatedly terminated")

    def restart(self) -> None:
        self._restart_internal()

    def _restart_internal(self) -> None:
        restart_start = time.perf_counter() if self.timing_enabled else None
        self.close()
        self._start()
        if self.timing_enabled and restart_start is not None:
            elapsed = time.perf_counter() - restart_start
            self._timing_stats.restart_time += elapsed
            self._timing_stats.restart_count += 1

    def schedule_warmup(self) -> None:
        if not self._awaiting_first_request:
            return
        if self._warmup_pending:
            return
        if not self.is_alive():
            self._restart_internal()
        conn = self.parent_conn
        if conn is None:
            return
        try:
            conn.send(("warmup", None))
            self._warmup_pending = True
        except (BrokenPipeError, EOFError, OSError):
            self._warmup_pending = False
            self._restart_internal()

    def drain_warmup_ack(self, *, block: bool = True) -> None:
        if not self._warmup_pending:
            return
        conn = self.parent_conn
        if conn is None:
            self._warmup_pending = False
            return
        if not block and not conn.poll():
            return
        try:
            status, _payload = conn.recv()
        except (BrokenPipeError, EOFError, OSError):
            self._warmup_pending = False
            self._restart_internal()
            return
        if status != "warmup_ack":
            raise RuntimeError("Unexpected warmup response from benchmark worker")
        if self.timing_enabled:
            timing_status, _elapsed = conn.recv()
            if timing_status != "timing":
                raise RuntimeError("Unexpected timing response during warmup")
        self._warmup_pending = False
        self._awaiting_first_request = False

    def consume_timing_stats(self) -> _WorkerTimingStats:
        if not self.timing_enabled:
            return _WorkerTimingStats()
        snapshot = dataclasses.replace(self._timing_stats)
        self._timing_stats = _WorkerTimingStats()
        return snapshot

    def close(self) -> None:
        if self.parent_conn is not None:
            # Politely ask the worker to exit before joining.
            with contextlib.suppress(Exception):
                self.parent_conn.send(("close", None))
            self.parent_conn.close()
        if self.process is not None:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
        self.parent_conn = None
        self.process = None


def _close_worker(worker: _BenchmarkWorker) -> None:
    # Used by weakref finalizers to ensure we do not leak subprocesses.
    worker.close()


class PopulationMember(NamedTuple):
    """
    Represents a member of the population in population-based search algorithms.

    Attributes:
        perf (float): The performance of the configuration.
        flat_values (FlatConfig): The flat representation of the configuration values.
        config (Config): The full configuration object.
    """

    perf: float
    flat_values: FlatConfig
    config: Config


def performance(member: PopulationMember) -> float:
    """
    Retrieve the performance of a population member.  Used as a sort key.

    Args:
        member: The population member.

    Returns:
        The performance of the member.
    """
    return member.perf


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
        self.config_gen: ConfigGeneration = ConfigGeneration(self.config_spec)

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
        return PopulationMember(self.benchmark(config), flat_values, config)

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
        configs = [*map(self.config_gen.unflatten, to_check)]
        result = []
        for flat_values, config_in, (config_out, perf) in zip(
            to_check, configs, self.parallel_benchmark(configs), strict=True
        ):
            assert config_in is config_out
            result.append(PopulationMember(perf, flat_values, config_in))
        return result

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
    if math.isinf(population[-1].perf):
        working = [x for x in population if not math.isinf(x.perf)]
        if len(working) == 0:
            raise exc.NoConfigFound
        return (
            f"failed={len(population) - len(working)} "
            f"min={working[0].perf:.4f} "
            f"mid={working[len(working) // 2].perf:.4f} "
            f"max={working[-1].perf:.4f} "
            f"best={population[0].config!s}"
        )
    return (
        f"min={population[0].perf:.4f} "
        f"mid={population[len(population) // 2].perf:.4f} "
        f"max={population[-1].perf:.4f} "
        f"best={population[0].config!s}"
    )


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
        assert self.ok is not None
        return self.ok

    @staticmethod
    def wait_for_all(
        futures: list[PrecompileFuture],
    ) -> list[bool]:
        """
        Wait for all precompile futures to complete.

        Args:
            futures: A list of PrecompileFuture objects.

        Returns:
            A list of boolean values indicating completion status.
        """
        remaining = [f for f in futures if f.ok is None]
        try:
            while remaining:
                remaining = PrecompileFuture._wait_for_all_step(remaining)
        except Exception:
            for f in remaining:
                if (p := f.process) is not None:
                    with contextlib.suppress(Exception):
                        p.terminate()
            raise
        result = []
        for f in futures:
            assert f.ok is not None
            result.append(f.ok)
        return result

    @staticmethod
    def _wait_for_all_step(
        futures: list[PrecompileFuture],
    ) -> list[PrecompileFuture]:
        """Start up to the concurrency cap, wait for progress, and return remaining futures."""
        # Concurrency cap from the settings of the first future's search
        cap = futures[0].search.settings.autotune_precompile_jobs or os.cpu_count() or 1
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
            else:
                remaining.append(f)
        return remaining

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
            return self.ok
        process.terminate()
        process.join(10)
        msg = f"Timeout after {self.elapsed:.0f}s compiling {self.config}"
        if process.is_alive():
            self.search.log.warning(
                msg,
                "(SIGKILL required)",
            )
            process.kill()
            process.join()
        else:
            self.search.log.warning(msg)

        self.ok = False
        return False


class _ExtractedLaunchArgs(Exception):
    """Exception that carries kernel launch arguments for precompiler extraction."""

    kernel: JITFunction[object]
    grid: object
    args: tuple[object, ...]
    kwargs: dict[str, object]

    def __init__(
        self,
        triton_kernel: JITFunction[object],
        grid: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        super().__init__()
        self.kernel = triton_kernel
        self.grid = grid
        self.args = args
        self.kwargs = kwargs
