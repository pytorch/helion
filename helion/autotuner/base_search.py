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

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design overview
# ---------------------------------------------------------------------------
#
# Kernel autotuning has two phases that can stress CUDA:
#   1. Compiling candidate configs (Triton can hang or crash the driver).
#   2. Benchmarking those configs to pick the fastest one.
#
# To avoid poisoning the main Python process with CUDA state—or worse, hanging it—we perform
# both phases in child processes. Compilation uses short-lived "fork" workers (safer on Linux and
# keeps the same CUDA context), while benchmarking runs inside a long-lived "spawn"ed worker.
#
# The benchmark worker acts as a tiny RPC server: the parent process sends messages containing a
# handle to a compiled Triton kernel, the worker loads and times the kernel on the GPU, and then
# replies with either the median runtime or a classified error. If CUDA faults or the worker dies,
# we tear it down and recreate it transparently.
#
# Keeping the worker alive across configs removes the heavy cost of re-initializing CUDA for every
# run while still isolating potential crashes. A weakref finalizer and `atexit` handler ensure no
# extra processes linger if autotuning exits early.

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
        # Worker processes must never see tensors that require grad; detach eagerly.
        self._worker_args = _sanitize_worker_args(self.args)
        self.counters: collections.Counter[str] = collections.Counter()
        self.log = LambdaLogger(self.settings.autotune_log_level)
        self._benchmark_worker: _BenchmarkWorker | None = None
        self._worker_finalizer: weakref.finalize | None = None

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

    def _get_benchmark_worker(self) -> "_BenchmarkWorker":
        worker = self._benchmark_worker
        if worker is None or not worker.is_alive():
            # Clean up any stale worker state before we spin up a new process.
            if worker is not None:
                worker.close()
            if self._worker_finalizer is not None:
                self._worker_finalizer.detach()
            worker = _BenchmarkWorker(self._worker_args)
            self._benchmark_worker = worker
            # Register a finalizer so the worker is torn down if this search object is GC'd.
            self._worker_finalizer = weakref.finalize(self, _close_worker, worker)
        return worker

    def _restart_benchmark_worker(self) -> None:
        worker = self._benchmark_worker
        if worker is None:
            worker = self._get_benchmark_worker()
        else:
            # Kill the existing subprocess to recover from CUDA failures or broken pipes.
            worker.restart()

    def _close_benchmark_worker(self) -> None:
        worker = self._benchmark_worker
        if worker is not None:
            # Used at the end of autotune to ensure no stray subprocesses linger.
            worker.close()
            self._benchmark_worker = None
        if self._worker_finalizer is not None:
            self._worker_finalizer.detach()
            self._worker_finalizer = None

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
        worker = self._get_benchmark_worker()
        # Offload the execution to the persistent subprocess so the main process never touches CUDA.
        status, payload = worker.benchmark(_CompiledFnHandle.from_callable(fn))
        if status == "ok":
            result = float(payload)
            self.log.debug(lambda: f"result: {result:.4f}ms")
            return result

        action, message = payload
        # Reset the worker so subsequent benchmarks get a fresh CUDA context without any CUDA errors.
        self._restart_benchmark_worker()
        if action == "raise":
            raise exc.TritonError(message, config)
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
        except Exception as e:
            log.warning(
                "Helion autotuner precompile error for config %r, error: %s",
                config,
                e,
                exc_info=True,
            )
            return PrecompileFuture.skip(self, config, False)
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
        self.log(
            f"Autotuning complete in {end - start:.1f}s after searching {self.counters['benchmark']} configs.\n"
            "One can hardcode the best config and skip autotuning with:\n"
            f"    @helion.kernel(config={best!r})\n",
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
) -> float:
    # Load the callable on-demand inside the worker process and measure it in isolation.
    fn = fn_handle.load()
    fn(*args)
    res = do_bench(
        functools.partial(fn, *args),
        return_mode="median",
    )
    return float(res)


def _benchmark_worker_entry(
    conn: connection.Connection,
    args: tuple[object, ...],
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
            continue
        handle: _CompiledFnHandle = payload
        try:
            result = _run_benchmark(handle, args)
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
        else:
            conn.send(("ok", result))
    conn.close()


class _BenchmarkWorker:
    """Persistent benchmark subprocess that executes configurations sequentially.

    The worker stores no global CUDA state in the parent and communicates strictly via pipes,
    which lets us reuse the same process across many benchmark calls while keeping the main
    process CUDA-free.
    """

    def __init__(self, args: tuple[object, ...]) -> None:
        self.args = args
        # Spawn isolates CUDA initialization to the child process.
        self.ctx = mp.get_context("spawn")
        self.parent_conn: connection.Connection | None = None
        self.process: mp.Process | None = None
        self._start()

    def _start(self) -> None:
        parent_conn, child_conn = self.ctx.Pipe(duplex=True)
        process = self.ctx.Process(
            target=_benchmark_worker_entry,
            args=(child_conn, self.args),
        )
        # Child will loop forever handling benchmark requests until we send "close".
        process.start()
        child_conn.close()
        self.parent_conn = parent_conn
        self.process = process

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def benchmark(self, handle: "_CompiledFnHandle") -> tuple[str, object]:
        attempts = 0
        while attempts < 2:
            if not self.is_alive():
                self._restart_internal()
            try:
                assert self.parent_conn is not None
                # Each request contains only the compiled handle; args were fixed at construction.
                self.parent_conn.send(("benchmark", handle))
                return self.parent_conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                # Broken connection: restart and retry once before giving up.
                attempts += 1
                self._restart_internal()
        raise RuntimeError("Benchmark worker repeatedly terminated")

    def restart(self) -> None:
        self._restart_internal()

    def _restart_internal(self) -> None:
        self.close()
        self._start()

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
