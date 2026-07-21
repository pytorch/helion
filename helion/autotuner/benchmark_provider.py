from __future__ import annotations

import abc
import copy
import dataclasses
import datetime
import functools
from itertools import count
from itertools import starmap
import math
from math import inf
import os
import tempfile
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import NoReturn
from typing import cast

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map_only
from torch.utils._pytree import tree_unflatten

from .. import exc
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import already_compiled_fail
from ..runtime.precompile_shim import make_precompiler
from .accuracy import assert_close as _assert_close
from .accuracy import is_fp8_dtype
from .benchmark_job import AccuracyCheckJob
from .benchmark_job import AccuracyCheckResult
from .benchmark_job import BenchmarkJob
from .benchmark_worker import BenchmarkSubprocessError
from .benchmark_worker import BenchmarkWorker
from .benchmarking import clear_jit_fast_path_caches
from .benchmarking import do_bench
from .benchmarking import do_bench_generic
from .benchmarking import synchronize_device
from .logger import SUPPRESSED_TRITON_CODE_MSG
from .logger import AutotuneLogEntry
from .logger import capture_output
from .logger import classify_triton_exception
from .logger import format_triton_compile_failure
from .logger import log_generated_triton_code_debug
from .logger import match_unrecoverable_runtime_error
from .logger import maybe_dump_triton_failure
from .precompile_future import PrecompileContext
from .precompile_future import PrecompileFuture
from .precompile_future import _ExtractedLaunchArgs
from .precompile_future import _serialize_compiled_fn
from .progress_bar import iter_with_progress
from helion._dist_utils import _clone_symm_mem_tensor
from helion._dist_utils import all_gather_object
from helion._dist_utils import get_signal_pad_ptrs_dev
from helion._dist_utils import is_symm_mem_tensor
from helion._dist_utils import sync_object

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import CompiledConfig
    from ..runtime.settings import Settings
    from . import ConfigSpec
    from .base_search import _AutotunableKernel
    from .logger import AutotuningLogger
    from .metrics import AutotuneMetrics


MultiShapeAggregation = Literal["geomean", "max"]
MultiShapeReference = Literal["default", "baseline"] | None


@dataclasses.dataclass
class _MultiShapeAutotuneArgs:
    """Private carrier that keeps existing autotuners anchored to one args tuple."""

    cases: tuple[tuple[BoundKernel, tuple[object, ...]], ...]
    aggregation: MultiShapeAggregation
    relative_to: MultiShapeReference
    cache_tag: str | None
    workload_key: tuple[object, ...]
    reference_latencies: tuple[float, ...] | None = None
    measurements: dict[str, tuple[tuple[float, ...], float, tuple[str, ...]]] = (
        dataclasses.field(default_factory=dict)
    )
    defer_selected_log: bool = False
    search_started: bool = False
    found_valid_config: bool = False

    def __len__(self) -> int:
        return len(self.cases[0][1])

    def __getitem__(self, index: int | slice) -> object:
        return self.cases[0][1][index]


def _aggregate_values(
    values: Sequence[float], aggregation: MultiShapeAggregation
) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        return inf
    if aggregation == "max":
        return max(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _aggregate_multi_shape_timings(
    timings: Sequence[float],
    *,
    aggregation: MultiShapeAggregation,
    references: Sequence[float] | None,
) -> float:
    """Reduce per-shape timings to the scalar objective used by searches."""
    raw = _aggregate_values(timings, aggregation)
    if references is None or not math.isfinite(raw):
        return raw
    if len(timings) != len(references) or not math.isfinite(
        _aggregate_values(references, aggregation)
    ):
        return inf
    ratios = [
        timing / reference
        for timing, reference in zip(timings, references, strict=True)
    ]
    return _aggregate_values(ratios, aggregation)


def _format_multi_shape_measurement(
    args: _MultiShapeAutotuneArgs,
    config: Config,
    timings: Sequence[float],
    aggregate: float,
    *,
    selected: bool,
    statuses: Sequence[str] | None = None,
) -> str:
    """Format one joint result with enough detail to explain its objective."""
    references = args.reference_latencies
    rows = []
    for index, ((_, case_args), timing) in enumerate(
        zip(args.cases, timings, strict=True)
    ):
        leaves, _ = tree_flatten(case_args)
        tensor_shapes = [
            tuple(value.shape) for value in leaves if torch.is_tensor(value)
        ]
        shape_text = f" tensor_shapes={tensor_shapes}" if tensor_shapes else ""
        status = statuses[index] if statuses is not None else "ok"
        if status != "ok" or not math.isfinite(timing) or timing <= 0:
            row = f"arg_sets[{index}]{shape_text}: status={status}"
            if math.isfinite(timing):
                row += f", latency={timing:.6f} ms"
        else:
            row = f"arg_sets[{index}]{shape_text}: latency={timing:.6f} ms"
        if (
            references is not None
            and status == "ok"
            and math.isfinite(timing)
            and timing > 0
        ):
            reference = references[index]
            row += (
                f", {args.relative_to} reference={reference:.6f} ms"
                f", ratio={timing / reference:.6f}x"
            )
        rows.append(row)

    if not math.isfinite(aggregate):
        objective = "rejected"
    elif references is None:
        objective = f"{args.aggregation}(latencies)={aggregate:.6f} ms"
    else:
        objective = (
            f"{args.aggregation}(latency ratios vs {args.relative_to})={aggregate:.6f}x"
        )
    if selected:
        details = "\n  ".join([*rows, f"objective: {objective}"])
        return f"Selected multi-shape config {config}:\n  {details}"
    return f"Multi-shape candidate {config}: {'; '.join([*rows, f'objective: {objective}'])}"


def _format_selected_multi_shape_measurement(
    args: _MultiShapeAutotuneArgs, config: Config
) -> str | None:
    measurement = args.measurements.get(repr(config))
    if measurement is None:
        return None
    timings, aggregate, statuses = measurement
    return _format_multi_shape_measurement(
        args,
        config,
        timings,
        aggregate,
        selected=True,
        statuses=statuses,
    )


def _materialize_multi_shape_config(config_spec: ConfigSpec, config: Config) -> Config:
    """Normalize a detached config against the anchor shape."""
    result = copy.deepcopy(config)
    config_spec.normalize(result)
    return result


def _has_valid_multi_shape_measurement(
    args: _MultiShapeAutotuneArgs,
    config_spec: ConfigSpec,
    config: Config,
) -> bool:
    materialized = _materialize_multi_shape_config(config_spec, config)
    measurement = args.measurements.get(repr(materialized))
    return measurement is not None and math.isfinite(measurement[1])


def _clone_args(
    args: Sequence[object],
    process_group_name: str | None,
    idx_to_clone: Sequence[int] | None = None,
) -> Sequence[object]:
    """
    Clone the given arguments, but cloning only the tensors specified by
      idx_to_clone. If idx_to_clone is None, clone all tensors.
    """

    def _should_clone(idx: int) -> bool:
        return idx_to_clone is None or idx in idx_to_clone

    args_flat, tree_spec = tree_flatten(args)
    old_arg_to_new_arg = {}

    for i, arg in enumerate(args_flat):
        if _should_clone(i) and is_symm_mem_tensor(arg, process_group_name):
            new_arg = _clone_symm_mem_tensor(arg, process_group_name)
            old_arg_to_new_arg[get_signal_pad_ptrs_dev(arg, process_group_name)] = (
                get_signal_pad_ptrs_dev(new_arg, process_group_name)
            )
            old_arg_to_new_arg[arg] = new_arg  # pyrefly: ignore[unsupported-operation]

    for i, arg in enumerate(args_flat):
        if arg in old_arg_to_new_arg:
            args_flat[i] = old_arg_to_new_arg[arg]
            continue
        if not isinstance(arg, torch.Tensor):
            continue
        if _should_clone(i):
            if arg.is_contiguous():
                clone = arg.detach().clone()
            else:
                # A kernel bound on a non-contiguous arg hardcodes that arg's
                # load strides into the compiled kernel as compile-time
                # constants. ``arg.detach().clone()`` returns a contiguous
                # tensor with a different layout and smaller storage, so those
                # hardcoded strides would address the wrong (or out-of-bounds)
                # memory when the autotuner accuracy baseline reruns the kernel
                # on the clone. ``copy.deepcopy`` does a storage-level copy that
                # reproduces the original size, stride, and offset, and also
                # handles broadcast/expanded views.
                clone = copy.deepcopy(arg.detach())
            clone.requires_grad_(arg.requires_grad)
            args_flat[i] = clone

    return tree_unflatten(args_flat, tree_spec)


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


def _triton_compile(
    fn: CompiledConfig,
    args: Sequence[object],
    config: Config,
    kernel: _AutotunableKernel,
) -> bool:
    """Trigger Triton JIT compilation without running the kernel.

    Extracts the Triton kernel and its launch arguments from fn, then
    invokes the precompiler so the compiled binary is cached before the
    actual benchmark run.

    The function requires the availability of CUDA.
    """

    def extract_launcher(
        triton_kernel: object,
        grid: tuple[int, ...],
        *launch_args: object,
        **launch_kwargs: object,
    ) -> NoReturn:
        raise _ExtractedLaunchArgs(triton_kernel, grid, launch_args, launch_kwargs)

    try:
        fn(*args, _launcher=extract_launcher)
        raise RuntimeError("Expected _ExtractedLaunchArgs to be raised")
    except _ExtractedLaunchArgs as extracted:
        precompiler = make_precompiler(
            cast("Any", extracted.kernel),
            config,
            cast("BoundKernel", kernel),
        )(*extracted.args, **extracted.kwargs)
        if precompiler is already_compiled:
            return True
        if precompiler is already_compiled_fail:
            return False
        return precompiler(False)  # pyrefly: ignore[bad-argument-count]
    except Exception:
        return False


class BenchmarkResult(NamedTuple):
    """Result of benchmarking a single configuration."""

    config: Config
    fn: Callable[..., object]
    perf: float
    status: Literal["ok", "error", "timeout", "peer_compilation_fail", "filtered"]
    compile_time: float | None


def _unset_fn(*args: object) -> NoReturn:
    raise RuntimeError("Uninitialized function")


def _never_exceeded() -> bool:
    return False


class BenchmarkProvider(abc.ABC):
    """Abstract interface for benchmarking kernel configurations.

    Search algorithms access this via ``self.benchmark_provider``.
    Subclass this to provide alternative benchmarking strategies
    (e.g. cross-node precompilation, overlapped precompile+benchmark).

    Lifecycle::

        provider = LocalBenchmarkProvider(...)
        provider.setup()
        try:
            provider.benchmark(configs)
        finally:
            provider.cleanup()

    ``BaseSearch`` manages this lifecycle automatically.

    ``budget_exceeded_fn`` is the local wall-clock budget check that
    ``BaseSearch._prepare`` installs via ``set_budget_exceeded_fn``.
    Subclasses should not call this hook directly inside loops that
    participate in distributed collectives; use a sync wrapper that
    agrees the cutoff across the process group so peers do not deadlock
    on an unmatched collective. Default no-op so providers built before
    the search installs the hook stay unchanged.
    """

    mutated_arg_indices: Sequence[int]
    # ``staticmethod`` prevents Python from binding the class-level
    # default to ``self`` when an instance reads it before any caller
    # has installed a real hook via ``set_budget_exceeded_fn``.
    budget_exceeded_fn: Callable[[], bool] = staticmethod(_never_exceeded)

    @abc.abstractmethod
    def __init__(
        self,
        kernel: _AutotunableKernel,
        settings: Settings,
        config_spec: ConfigSpec,
        args: Sequence[object],
        log: AutotuningLogger,
        autotune_metrics: AutotuneMetrics,
    ) -> None:
        """Initialize the provider with kernel context and benchmarking state."""
        ...

    def set_budget_exceeded_fn(self, fn: Callable[[], bool]) -> None:
        """Install the search's budget-check hook on this provider."""
        self.budget_exceeded_fn = fn

    @abc.abstractmethod
    def benchmark(
        self,
        configs: list[Config],
        *,
        desc: str = "Benchmarking",
    ) -> list[BenchmarkResult]:
        """Compile, precompile, validate, and time a batch of configs.

        Handles the full benchmark flow: compilation, optional subprocess
        precompilation, accuracy validation, timing, error classification,
        and progress reporting.

        Returns one ``BenchmarkResult`` per input config, in the same order.
        """
        ...

    def benchmark_isolated(
        self,
        fns: list[Callable[..., object]],
        *,
        warmup: int,
        rep: int,
        desc: str = "Benchmarking",
    ) -> list[float | None] | None:
        """Benchmark already-validated functions in an isolated subprocess.

        Return ``None`` when the provider cannot support the isolated path or
        per-function ``None`` when a timing could not be confirmed and callers
        should keep the prior timing for that function.
        """
        return None

    @abc.abstractmethod
    def setup(self) -> None:
        """Prepare resources needed before benchmarking begins (e.g. tmpdir)."""
        ...

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Release resources (tmpdir, subprocesses, etc.)."""
        ...


class LocalBenchmarkProvider(BenchmarkProvider):
    """Local single-machine benchmark provider.

    Compiles kernels locally, optionally precompiles in subprocesses
    (fork/spawn), and benchmarks on the local GPU.  This is the default
    provider created by ``BaseSearch._prepare()``.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        settings: Settings,
        config_spec: ConfigSpec,
        args: Sequence[object],
        log: AutotuningLogger,
        autotune_metrics: AutotuneMetrics,
    ) -> None:
        if isinstance(args, _MultiShapeAutotuneArgs):
            raise TypeError(
                "LocalBenchmarkProvider cannot benchmark multi-shape args directly"
            )
        self.kernel = kernel
        self.settings = settings
        self.config_spec = config_spec
        self.args = args
        self.log = log
        self._autotune_metrics = autotune_metrics
        self._accuracy_failure_config_ids: list[int] = []
        self._compile_failure_config_ids: list[int] = []
        self._precompile_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._precompile_args_path: str | None = None
        self._precompile_baseline_path: str | None = None
        self._precompile_result_counter: count[int] = count()
        self._benchmark_worker: BenchmarkWorker | None = None
        # budget_exceeded_fn inherits the class-level _never_exceeded default
        # until BaseSearch._prepare installs the search's real hook.

        # TODO(hinriksnaer): baseline computation is expensive (compiles and runs
        # the kernel). Currently safe because the provider is only constructed
        # from _prepare() during active autotuning, but ideally __init__ should
        # be cheap and expensive work deferred to setup().
        # Compute baseline and derived state
        (
            self._baseline_output,
            self.mutated_arg_indices,
            self._baseline_post_args,
        ) = self._compute_baseline()
        self._effective_atol, self._effective_rtol = (
            self._compute_effective_tolerances()
        )
        self._jobs = self._decide_num_jobs()

    def _record_accuracy_failure(self, config: Config) -> None:
        self._autotune_metrics.num_accuracy_failures += 1
        self._accuracy_failure_config_ids.append(id(config))

    def _record_compile_failure(self, config: Config) -> None:
        self._autotune_metrics.num_compile_failures += 1
        self._compile_failure_config_ids.append(id(config))

    def _compute_baseline(
        self,
    ) -> tuple[object, Sequence[int], Sequence[object] | None]:
        """
        Compute baseline output for accuracy validation during autotuning.
        Also detect if the kernel mutates any of its input arguments.

        The baseline is computed in one of two ways:
        - If settings.autotune_baseline_fn is provided, use that custom function
        - Otherwise, run the kernel with the default config
        """
        new_args = _clone_args(self.args, self.kernel.env.process_group_name)

        # Use custom baseline function if provided
        if self.settings.autotune_baseline_fn is not None:
            try:
                baseline_output = self.settings.autotune_baseline_fn(*new_args)
                synchronize_device()
            except Exception as e:
                raise exc.AutotuneError(
                    "Custom baseline function failed while computing baseline.\n"
                    f"Baseline function: {self.settings.autotune_baseline_fn}\n"
                ) from e
        else:
            # Use default config
            baseline_config = self.config_spec.default_config()
            try:
                baseline_output = self.kernel.compile_config(
                    baseline_config, allow_print=False
                )(*new_args)
                synchronize_device()
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
                self.kernel.maybe_log_repro(self.log.error, new_args, baseline_config)
                raise exc.InvalidConfig(
                    "Default config failed while computing baseline.\n"
                    f"Default config: {decorator}\n"
                    f"{SUPPRESSED_TRITON_CODE_MSG}\n"
                    "To work around this error, you could set `@helion.kernel(autotune_baseline_fn=...)` "
                    "to provide a custom baseline function (e.g. PyTorch eager implementation of your kernel)."
                ) from e

        original_args_flat, _ = tree_flatten(self.args)
        new_args_flat, _ = tree_flatten(new_args)
        mutated_arg_idxs = []
        # we should only count tensors, since they won't be bound or removed
        arg_idx = 0
        for old, new in zip(original_args_flat, new_args_flat, strict=False):
            if not (isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor)):
                arg_idx += 1
                continue
            try:
                equal = torch.equal(new, old)
            except RuntimeError:
                # torch.equal and device-to-host copies can fail on some
                # devices (e.g., TPU for large tensors).  Conservatively
                # assume the argument was not mutated.
                equal = True
            if not equal:
                mutated_arg_idxs.append(arg_idx)
            arg_idx += 1
        baseline_post_args = _clone_args(
            new_args,
            self.kernel.env.process_group_name,
            idx_to_clone=mutated_arg_idxs,
        )
        return baseline_output, mutated_arg_idxs, baseline_post_args

    def _compute_effective_tolerances(self) -> tuple[float, float]:
        """
        Compute effective tolerances based on the dtypes in the baseline output.

        For low-precision dtypes (fp8), we need stricter tolerances to ensure
        bitwise comparison works correctly. This method automatically detects
        such dtypes and adjusts tolerances accordingly.

        Returns:
            A tuple of (atol, rtol) to use for accuracy validation.
        """
        # Default tolerance when not user-specified
        DEFAULT_TOL = 1e-2

        # Get user-specified or default tolerances
        atol = self.settings.autotune_baseline_atol
        rtol = self.settings.autotune_baseline_rtol

        # Collect all dtypes from baseline output and mutated args
        dtypes = set()

        def collect_dtypes(obj: object) -> object:
            if isinstance(obj, torch.Tensor):
                dtypes.add(obj.dtype)
            return obj

        tree_map_only(torch.Tensor, collect_dtypes, self._baseline_output)
        if len(self.mutated_arg_indices) > 0 and self._baseline_post_args is not None:
            tree_map_only(torch.Tensor, collect_dtypes, self._baseline_post_args)

        # Only apply strict tolerances if ALL dtypes are fp8
        # Mixed dtypes (fp8 + fp32) would be too strict with atol=0.0, rtol=0.0
        all_dtypes_are_fp8 = dtypes and all(is_fp8_dtype(dtype) for dtype in dtypes)

        if all_dtypes_are_fp8:
            # All dtypes are fp8 - use bitwise comparison
            # unless the user explicitly set either tolerance value (i.e., not None)
            if atol is None and rtol is None:
                self.log(
                    f"Detected fp8 dtype(s) in output: {dtypes}. "
                    "Using bitwise comparison (atol=0.0, rtol=0.0) for autotuning accuracy check."
                )
                return 0.0, 0.0

        # Use user-specified values or defaults
        return (
            atol if atol is not None else DEFAULT_TOL,
            rtol if rtol is not None else DEFAULT_TOL,
        )

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

    def _precompile_context(self) -> PrecompileContext:
        """Build the narrow context that PrecompileFuture needs."""
        return PrecompileContext(
            settings=self.settings,
            log=self.log,
            kernel=self.kernel,
            args=self.args,
            jobs=self._jobs,
        )

    def setup(self) -> None:
        """Prepare precompile tmpdir and args for spawn mode."""
        if self._precompile_tmpdir is None:
            self._precompile_tmpdir = tempfile.TemporaryDirectory()
        if (
            self.settings.autotune_precompile == "spawn"
            or self._subprocess_benchmark_enabled()
        ):
            args_path = os.path.join(self._precompile_tmpdir.name, "args.pt")
            torch.save(self.args, args_path)
            self._precompile_args_path = args_path
        if self._subprocess_accuracy_check_enabled():
            baseline_path = os.path.join(self._precompile_tmpdir.name, "baseline.pt")
            torch.save(self._baseline_output, baseline_path)
            self._precompile_baseline_path = baseline_path

    def _next_precompile_result_path(self) -> str:
        """Return a fresh path for a precompile result file."""
        if self._precompile_tmpdir is None:
            self._precompile_tmpdir = tempfile.TemporaryDirectory()
        return os.path.join(
            self._precompile_tmpdir.name,
            f"result_{next(self._precompile_result_counter)}.pkl",
        )

    def cleanup(self) -> None:
        """Release precompile tmpdir, baseline tensors, and the budget hook.

        The budget hook installed by ``BaseSearch._prepare`` is a bound
        method that holds a reference to the search; without resetting it
        the search ``->`` provider ``->`` bound-method ``->`` search
        reference cycle keeps both alive until the next cyclic GC pass.
        Dropping the baseline tensors here lets refcount reclaim the
        cloned-arg GPU memory deterministically as soon as the provider
        loses its last external reference.
        """
        if self._benchmark_worker is not None:
            self._benchmark_worker.shutdown()
            self._benchmark_worker = None
        if self._precompile_tmpdir is not None:
            self._precompile_tmpdir.cleanup()
            self._precompile_tmpdir = None
        self._precompile_args_path = None
        self._precompile_baseline_path = None
        self._precompile_result_counter = count()
        # Drop the baseline tensors (GPU memory) so refcount frees them
        # the moment the provider loses its last external reference.
        self._baseline_output = None
        self._baseline_post_args = None
        # Restore the class-level no-op default so the provider no longer
        # holds a bound-method reference back to the search. Without this
        # the search-provider-budget-hook cycle (search -> provider ->
        # bound-method -> search) survives until cyclic GC runs.
        self.budget_exceeded_fn = _never_exceeded

    def _subprocess_benchmark_uses_wall_clock(self) -> bool:
        backend = getattr(self.config_spec, "backend", None)
        if backend is None:
            return False
        custom_bench = backend.get_do_bench()
        return backend.name == "cute" and custom_bench is do_bench_generic

    def _subprocess_benchmark_enabled(self) -> bool:
        """Subprocess benchmark path is opt-in and skipped for distributed /
        mutated-arg kernels where the worker's simple job shape doesn't fit."""
        if not self.settings.autotune_benchmark_subprocess:
            return False
        if dist.is_initialized():
            return False
        if len(self.mutated_arg_indices) > 0:
            return False
        if not self.kernel.supports_subprocess_benchmark():
            return False
        backend = getattr(self.config_spec, "backend", None)
        if backend is None or backend.get_do_bench() is None:
            return True
        return self._subprocess_benchmark_uses_wall_clock()

    def _subprocess_accuracy_check_enabled(self) -> bool:
        """Default accuracy checks can run in the same killable worker.

        Custom accuracy callbacks and mutated-argument checks can close over
        arbitrary process-local state, so those remain on the in-process path.
        """
        return (
            self.settings.autotune_accuracy_check
            and self.settings.autotune_baseline_accuracy_check_fn is None
            and len(self.mutated_arg_indices) == 0
            and self._subprocess_benchmark_enabled()
        )

    def _validate_against_baseline(
        self, config: Config, output: object, args: Sequence[object]
    ) -> bool:
        try:
            custom_check = self.settings.autotune_baseline_accuracy_check_fn
            if custom_check is not None:
                custom_check(output, self._baseline_output)
                if len(self.mutated_arg_indices) > 0:
                    custom_check(args, self._baseline_post_args)
            else:
                _assert_close(
                    output,
                    self._baseline_output,
                    atol=self._effective_atol,
                    rtol=self._effective_rtol,
                )
                if os.getenv("CHECK_INPUT_ACCURACY", "1") == "1":
                    if len(self.mutated_arg_indices) > 0:
                        # For distributed kernel, group_name may also be a argument.
                        # torch.testing.assert_close does not handle str argument.
                        # Filter needed.
                        assert self._baseline_post_args is not None
                        _assert_close(
                            args,
                            self._baseline_post_args,
                            atol=self._effective_atol,
                            rtol=self._effective_rtol,
                        )
        except AssertionError as e:
            if not self.settings.autotune_ignore_errors:
                self.log.warning(
                    f"Skipping config with accuracy mismatch: {config!r}\n{e!s}\nUse HELION_AUTOTUNE_ACCURACY_CHECK=0 to disable this check.\n"
                )
            return False
        return True

    def _create_precompile_future(
        self, config: Config, fn: CompiledConfig
    ) -> PrecompileFuture:
        """Create a subprocess to precompile the kernel and detect hangs."""
        ctx = self._precompile_context()
        if not self.settings.autotune_precompile:
            return PrecompileFuture.skip(ctx, config, True)
        mode = self.settings.autotune_precompile
        if mode not in {"fork", "spawn"}:
            raise exc.InvalidAPIUsage("autotune_precompile must be 'fork' or 'spawn'")
        if len(self.mutated_arg_indices) > 0:
            args = _clone_args(
                self.args,
                self.kernel.env.process_group_name,
                idx_to_clone=self.mutated_arg_indices,
            )
        else:
            args = self.args
        return PrecompileFuture.create(
            ctx=ctx,
            config=config,
            fn=fn,
            args=args,
            result_path=self._next_precompile_result_path(),
            args_path=self._precompile_args_path,
        )

    def _budget_exceeded_synced(self) -> bool:
        """Return whether any rank reports the autotune budget exhausted.

        The compile and benchmark loops below participate in distributed
        collectives. The cutoff decision must be agreed across the
        process group; otherwise one rank breaks early while peers keep
        entering collectives and the group deadlocks. Single-process
        autotune skips the all-gather because
        ``all_gather_object`` returns ``[obj]`` when distributed is not
        initialized.
        """
        local = self.budget_exceeded_fn()
        return any(
            all_gather_object(
                local,
                process_group_name=self.kernel.env.process_group_name,
            )
        )

    def benchmark(
        self,
        configs: list[Config],
        *,
        desc: str = "Benchmarking",
    ) -> list[BenchmarkResult]:
        """Compile, precompile, validate, and time a batch of configs.

        When ``budget_exceeded_fn`` reports the autotune wall-clock
        budget exhausted, the compile and benchmark loops short-circuit
        and leave any not-yet-handled slots at the default
        ``perf=inf, status="error"`` so the caller still receives one
        ``BenchmarkResult`` per input config in positional order. The
        cutoff is synchronized across the process group via
        ``_budget_exceeded_synced``.

        The precompile-phase wait (``PrecompileFuture.wait_for_all``)
        is not budget-aware: once the compile loop has queued
        precompile futures it drains all of them, so the effective
        wall-clock may overrun the configured budget by up to
        ``len(queued) * autotune_compile_timeout`` seconds. Backends
        that disable ``autotune_precompile`` (e.g. cute) skip that
        wait entirely.
        """
        all_configs = configs
        compiled: dict[int, Callable[..., object]] = {}
        futures: list[PrecompileFuture] | None = None

        # Compilation phase
        for i, config in enumerate(all_configs):
            if self._budget_exceeded_synced():
                break
            # Defensive: if `capture_output().__enter__` itself raises, the
            # except handler below still needs `captured` bound.
            captured: list[str] = [""]
            try:
                with capture_output() as captured:
                    compiled[i] = self.kernel.compile_config(config, allow_print=False)
            except Exception as e:
                if not compiled and i == len(all_configs) - 1:
                    raise
                maybe_dump_triton_failure(
                    self.kernel, config, e, captured_output=captured[0] or None
                )
                self.log.warning(
                    "Skipping config that failed to compile: "
                    f"{self.kernel.format_kernel_decorator(config, self.settings)}",
                    exc_info=True,
                )
        fns = list(compiled.values())
        valid_indices = list(compiled.keys())
        configs = [all_configs[i] for i in valid_indices]

        # Precompile phase
        if self.settings.autotune_precompile:
            futures = list(
                starmap(
                    self._create_precompile_future,
                    zip(configs, fns, strict=True),
                )
            )
            precompile_desc = (
                f"{desc} precompiling" if self.settings.autotune_progress_bar else None
            )
            is_workings = PrecompileFuture.wait_for_all(futures, desc=precompile_desc)
            precompile_status: list[Literal["ok", "error", "timeout"]] = []
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

        # Initialize results with defaults
        results: list[BenchmarkResult] = [
            BenchmarkResult(
                config=c, fn=_unset_fn, perf=inf, status="error", compile_time=None
            )
            for c in all_configs
        ]

        # Benchmark loop with progress reporting
        iterator = iter_with_progress(
            enumerate(zip(fns, is_workings, precompile_status, strict=True)),
            total=len(configs),
            description=f"{desc} exploring neighbors",
            enabled=self.settings.autotune_progress_bar,
        )
        for index, (fn, is_working, reason) in iterator:
            if self._budget_exceeded_synced():
                break
            config = configs[index]
            if futures is not None:
                future = futures[index]
                compile_time = (
                    future.elapsed
                    if future.process is not None and future.started
                    else None
                )
            else:
                compile_time = None
            status: Literal[
                "ok", "error", "timeout", "peer_compilation_fail", "filtered"
            ]
            # config_id is None when no log sink is active (skip recording). The
            # started and result rows share it so they join to one config, and
            # every config that reaches the benchmark loop is logged -- including
            # ones that never benchmark because they (or a peer) failed to compile.
            config_id = self.log.register_config(config)
            if config_id is not None:
                self.log.record_autotune_entry(
                    AutotuneLogEntry(
                        generation=self._autotune_metrics.num_generations,
                        status="started",
                        perf_ms=None,
                        compile_time=compile_time,
                        config_id=config_id,
                        config=config,
                    )
                )
            if all(
                all_gather_object(
                    is_working,
                    process_group_name=self.kernel.env.process_group_name,
                )
            ):
                perf = self._benchmark_function(config, fn)
                status = "ok" if math.isfinite(perf) else "error"
                recorded_perf = perf if math.isfinite(perf) else None
                results[valid_indices[index]] = BenchmarkResult(
                    config=config,
                    fn=fn,
                    perf=perf,
                    status=status,
                    compile_time=compile_time,
                )
            else:
                status = "timeout" if reason == "timeout" else "error"
                if is_working:
                    status = "peer_compilation_fail"
                recorded_perf = None
                results[valid_indices[index]] = BenchmarkResult(
                    config=config,
                    fn=fn,
                    perf=inf,
                    status=status,
                    compile_time=compile_time,
                )
            if config_id is not None:
                self.log.record_autotune_entry(
                    AutotuneLogEntry(
                        generation=self._autotune_metrics.num_generations,
                        status=status,
                        perf_ms=recorded_perf,
                        compile_time=compile_time,
                        config_id=config_id,
                        config=config,
                    )
                )
        return results

    def _clear_jit_fast_path_caches(self, fn: CompiledConfig) -> None:
        """Clear Triton JIT fast-path caches for this generated wrapper.

        Without this, tensors passed to the companion Triton JIT function can
        remain pinned in GPU memory by its _last_call cache across config
        benchmarks.
        """
        clear_jit_fast_path_caches(fn, self.log)

    def _benchmark_function(self, config: Config, fn: CompiledConfig) -> float:
        """Benchmark a single compiled function.  Returns time in ms or inf."""
        self._autotune_metrics.num_configs_tested += 1
        self.log.debug(lambda: f"Running benchmark for {config!r}")

        if self._subprocess_benchmark_enabled():
            result = self._benchmark_function_subprocess(config, fn)
            if result is not None:
                return result
            # None means the subprocess path could not handle this config
            # (e.g., serialization failed); fall through to in-process.

        if len(self.mutated_arg_indices) > 0:
            working_args = _clone_args(
                self.args,
                self.kernel.env.process_group_name,
                idx_to_clone=self.mutated_arg_indices,
            )
        else:
            working_args = self.args

        # precompile in the current process for distributed kernels.
        # The reason we need this is due to some tricky distributed kernels
        # like https://gist.github.com/shunting314/81f13ce00f835b21ab6466e21454b7c5 .
        # We specialize the RANK argument for each GPU,
        # some rank may get out of resource errors while others don't
        # due to the specialization.
        #
        # Without precompilation here, some rank may fail and skip running
        # the kernel while outer ranks waiting for its peers. It
        # results in a stuck job.
        #
        # Precompiilation happening in child process is not enough because
        # CUDA is not available there. We can not check resource usage
        # like shared-memory, tmem, max-threads etc.
        #
        # This precompilation has overhead. Only do it if distributed is
        # initialized.

        if dist.is_initialized():
            # Trigger Triton JIT compilation before running the kernel
            compile_success = _triton_compile(fn, working_args, config, self.kernel)
            compile_success_all = all(
                all_gather_object(
                    compile_success,
                    process_group_name=self.kernel.env.process_group_name,
                )
            )

            if not compile_success_all:
                return inf

        # Defensive: if `capture_output().__enter__` itself raises, the except
        # handler below still needs `_captured_output` bound.
        _captured_output: list[str] = [""]
        try:
            # TODO(jansel): early exit with fewer trials if early runs are slow
            self.log.debug(lambda: f"Running {config} at {datetime.datetime.now()}")
            t0 = time.perf_counter()

            # Narrow capture: only the kernel compile+launch sites (which can
            # emit Triton C-level MLIR diagnostics). Leave the accuracy check
            # and Python-level logging outside so warnings still reach stderr.
            with capture_output() as _captured_output:
                synchronize_device()
                output = fn(*working_args)  # make sure the kernel is compiled
                synchronize_device()

            pass_accuracy_check = (
                not self.settings.autotune_accuracy_check
                or self._validate_against_baseline(config, output, working_args)
            )
            if not pass_accuracy_check:
                self._record_accuracy_failure(config)
            if not all(
                all_gather_object(
                    pass_accuracy_check,
                    process_group_name=self.kernel.env.process_group_name,
                )
            ):
                # for distributed kernels like matmul-reduce-scatter, different ranks compute
                # a different chunk. It's possible that some ranks pass the accuracy check while
                # others don't. Skip the config if any rank fails the accuracy check.
                # Without this synchronization, some ranks go on to call the benchmark function
                # while other ranks return immediately, this will cause stuck jobs!
                return inf

            with capture_output() as _captured_output:
                benchmark_function = self.kernel.bench_compile_config(
                    config, allow_print=False
                )
                benchmark_function(*working_args)  # warmup benchmark kernel

                t1 = time.perf_counter()
                _backend = getattr(getattr(self, "config_spec", None), "backend", None)
                benchmark_runner = (
                    _backend.get_do_bench() if _backend is not None else None
                ) or do_bench
                res = benchmark_runner(
                    functools.partial(benchmark_function, *working_args),
                    return_mode="median",
                    warmup=1,  # we are already warmed up above
                    rep=50,
                    process_group_name=self.kernel.env.process_group_name,
                )
            res = sync_object(
                res, process_group_name=self.kernel.env.process_group_name
            )
            t2 = time.perf_counter()
            assert isinstance(res, float)

            self.log.debug(
                lambda: f"result: {res:.4f}ms (took {t1 - t0:.1f}s + {t2 - t1:.1f}s)",
            )
            return res
        except Exception as e:
            # e.__traceback__ holds references to all local variables in the call stack frames.
            # When a Triton kernel fails, the output tensors allocated by the Helion kernel function
            # were being held by the traceback, preventing them from being freed.
            e.__traceback__ = None
            maybe_dump_triton_failure(
                self.kernel,
                config,
                e,
                captured_output=_captured_output[0] or None,
            )
            if match_unrecoverable_runtime_error(e):
                self.kernel.maybe_log_repro(self.log.error, self.args, config)
                raise exc.TritonUnrecoverableRuntimeError(
                    reason=str(e),
                    decorator=self.kernel.format_kernel_decorator(
                        config, self.settings
                    ),
                    error=f"{type(e).__qualname__}: {e}",
                ) from e
            _backend = getattr(getattr(self, "config_spec", None), "backend", None)
            action = (
                _backend.classify_autotune_exception(e)
                if _backend is not None
                else None
            ) or classify_triton_exception(e)
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
                self.kernel.maybe_log_repro(self.log.error, self.args, config)
                raise exc.TritonError(
                    error=f"{type(e).__qualname__}: {e}",
                    decorator=decorator,
                    code=SUPPRESSED_TRITON_CODE_MSG,
                ) from e
            elif action == "warn":
                # Compile-related failures (e.g., PTXASError, PassManager
                # failed). Message is debug to keep autotune output quiet;
                # repro is warning so HELION_PRINT_REPRO=1 stays visible.
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.log.debug(format_triton_compile_failure(config, e, self.kernel))
                self.kernel.maybe_log_repro(self.log.warning, self.args, config)
            else:  # action == "debug" (CUDA OOM, expected runtime failures)
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.log.debug(f"Benchmarking failed: {type(e).__name__}: {e}")
                self.kernel.maybe_log_repro(self.log.debug, self.args, config)

            self._record_compile_failure(config)
            return inf
        finally:
            self._clear_jit_fast_path_caches(fn)

    def _benchmark_function_subprocess(
        self, config: Config, fn: CompiledConfig
    ) -> float | None:
        """Benchmark ``fn`` in a long-lived spawn subprocess with a per-call
        timeout. Returns the measured latency in ms, ``inf`` for a failure
        we classified and handled, or ``None`` if the subprocess path cannot
        handle this config and the caller should fall back to in-process.
        """
        try:
            latency = self._run_subprocess_benchmark_job(fn, warmup=1, rep=50)
            if latency is None:
                return None
        except BenchmarkSubprocessError as e:
            # Timeout or unexpected worker exit; skip config and continue.
            self.log.warning(f"Benchmark subprocess failed for {config!r}: {e}")
            self._record_compile_failure(config)
            return inf
        except Exception as e:
            e.__traceback__ = None
            if match_unrecoverable_runtime_error(e):
                # Worker is already killed; parent CUDA is unaffected.
                # Skip this config and continue the search.
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                self.log.warning(
                    f"Skipping config that triggered an unrecoverable runtime "
                    f"error in the benchmark subprocess: "
                    f"{type(e).__qualname__}: {e}\n  Config: {decorator}"
                )
                self.kernel.maybe_log_repro(self.log.warning, self.args, config)
                self._record_compile_failure(config)
                return inf
            self.log.debug(
                f"Benchmark subprocess raised for {config!r}: {type(e).__name__}: {e}"
            )
            self._record_compile_failure(config)
            return inf

        if self.settings.autotune_accuracy_check:
            try:
                accuracy_result = self._run_subprocess_accuracy_check_job(fn)
            except BenchmarkSubprocessError as e:
                self.log.warning(
                    f"Accuracy check subprocess failed for {config!r}: {e}"
                )
                self._record_compile_failure(config)
                return inf
            except Exception as e:
                e.__traceback__ = None
                if match_unrecoverable_runtime_error(e):
                    # Worker is already killed; parent CUDA is unaffected.
                    # Skip this config and continue the search.
                    decorator = self.kernel.format_kernel_decorator(
                        config, self.settings
                    )
                    self.log.warning(
                        f"Skipping config that triggered an unrecoverable runtime "
                        f"error in the accuracy check subprocess: "
                        f"{type(e).__qualname__}: {e}\n  Config: {decorator}"
                    )
                    self.kernel.maybe_log_repro(self.log.warning, self.args, config)
                    self._record_compile_failure(config)
                    return inf
                self.log.debug(
                    f"Accuracy check subprocess raised for {config!r}: "
                    f"{type(e).__name__}: {e}"
                )
                self._record_compile_failure(config)
                return inf

            if accuracy_result is not None:
                if not accuracy_result.ok:
                    if not self.settings.autotune_ignore_errors:
                        self.log.warning(
                            f"Skipping config with accuracy mismatch: {config!r}\n"
                            f"{accuracy_result.message}\n"
                            "Use HELION_AUTOTUNE_ACCURACY_CHECK=0 to disable this check.\n"
                        )
                    self._record_accuracy_failure(config)
                    return inf
                return float(latency)

            # None means a custom check fn or uncommon kernel can't run in the
            # worker; validate in-process instead.
            try:
                with capture_output():
                    output = fn(*self.args)
                    synchronize_device()
                if not self._validate_against_baseline(config, output, self.args):
                    self._record_accuracy_failure(config)
                    return inf
            except Exception as e:
                e.__traceback__ = None
                if match_unrecoverable_runtime_error(e):
                    # This ran in the parent process, so the IMA poisoned
                    # the parent CUDA context; the search cannot continue.
                    self.kernel.maybe_log_repro(self.log.error, self.args, config)
                    raise exc.TritonUnrecoverableRuntimeError(
                        reason=str(e),
                        decorator=self.kernel.format_kernel_decorator(
                            config, self.settings
                        ),
                        error=f"{type(e).__qualname__}: {e}",
                    ) from e
                self.log.debug(
                    f"Accuracy check raised for {config!r}: {type(e).__name__}: {e}"
                )
                self._record_compile_failure(config)
                return inf
            finally:
                # Same as the in-process path: drop JIT fast-path caches so
                # this fn's tensors aren't pinned in GPU memory across configs.
                self._clear_jit_fast_path_caches(fn)

        return float(latency)

    def _run_subprocess_accuracy_check_job(
        self, fn: CompiledConfig
    ) -> AccuracyCheckResult | None:
        if not self._subprocess_accuracy_check_enabled():
            return None
        if self._precompile_args_path is None or self._precompile_baseline_path is None:
            return None
        try:
            fn_spec = _serialize_compiled_fn(fn)
        except RuntimeError:
            return None

        if self._benchmark_worker is None:
            self._benchmark_worker = BenchmarkWorker(device=None)

        job = AccuracyCheckJob(
            fn_spec=fn_spec,
            args_path=self._precompile_args_path,
            baseline_path=self._precompile_baseline_path,
            atol=self._effective_atol,
            rtol=self._effective_rtol,
        )
        return cast(
            "AccuracyCheckResult",
            self._benchmark_worker.run(
                job,
                timeout=float(self.settings.autotune_benchmark_timeout),
            ),
        )

    def _run_subprocess_benchmark_job(
        self,
        fn: CompiledConfig,
        *,
        warmup: int,
        rep: int,
    ) -> float | None:
        if self._precompile_args_path is None:
            return None
        try:
            fn_spec = _serialize_compiled_fn(fn)
        except RuntimeError:
            return None

        if self._benchmark_worker is None:
            self._benchmark_worker = BenchmarkWorker(device=None)

        job = BenchmarkJob(
            fn_spec=fn_spec,
            args_path=self._precompile_args_path,
            warmup=warmup,
            rep=rep,
            use_wall_clock=self._subprocess_benchmark_uses_wall_clock(),
        )
        return float(
            self._benchmark_worker.run(
                job,
                timeout=float(self.settings.autotune_benchmark_timeout),
            )
        )

    def benchmark_isolated(
        self,
        fns: list[Callable[..., object]],
        *,
        warmup: int,
        rep: int,
        desc: str = "Benchmarking",
    ) -> list[float | None] | None:
        if not self._subprocess_benchmark_enabled():
            return None
        if self.settings.autotune_benchmark_fn is not None:
            return None

        timings: list[float | None] = []
        for fn in fns:
            try:
                timing = self._run_subprocess_benchmark_job(
                    cast("CompiledConfig", fn),
                    warmup=warmup,
                    rep=rep,
                )
            except BenchmarkSubprocessError as e:
                self.log.warning(f"{desc} subprocess failed: {e}")
                timing = None
            except Exception as e:
                e.__traceback__ = None
                if match_unrecoverable_runtime_error(e):
                    self.log.warning(f"{desc} sticky CUDA error skipped: {e}")
                    # The confirmation re-ran a previously accepted candidate in
                    # an isolated worker; a sticky CUDA error means that config is
                    # still unsafe, so remove it from contention.
                    timing = inf
                else:
                    self.log.debug(f"{desc} subprocess raised: {type(e).__name__}: {e}")
                    timing = None
            timings.append(None if timing is None else float(timing))
        return timings


class MultiShapeBenchmarkProvider(BenchmarkProvider):
    """Compose ordinary local providers and expose one scalar per config."""

    mutated_arg_indices: Sequence[int] = ()

    def __init__(
        self,
        kernel: _AutotunableKernel,
        settings: Settings,
        config_spec: ConfigSpec,
        args: Sequence[object],
        log: AutotuningLogger,
        autotune_metrics: AutotuneMetrics,
    ) -> None:
        if not isinstance(args, _MultiShapeAutotuneArgs):
            raise TypeError("MultiShapeBenchmarkProvider requires multi-shape args")
        from .metrics import AutotuneMetrics

        self.kernel = kernel
        self.settings = settings
        self.config_spec = config_spec
        self.args = args
        self.log = log
        self._autotune_metrics = autotune_metrics
        self.args.search_started = True
        self.children: list[LocalBenchmarkProvider] = []
        child_log = copy.copy(log)
        child_log._log_sink = None
        case_index = 0
        try:
            for index, (case_kernel, case_args) in enumerate(args.cases):
                case_index = index
                child = LocalBenchmarkProvider(
                    kernel=case_kernel,
                    settings=case_kernel.settings,
                    config_spec=case_kernel.config_spec,
                    args=case_args,
                    # Child rows are implementation details, so only their text logs
                    # are forwarded to the aggregate logger.
                    log=child_log,
                    autotune_metrics=AutotuneMetrics(),
                )
                child.set_budget_exceeded_fn(_never_exceeded)
                self.children.append(child)
        except Exception as error:
            for child in reversed(self.children):
                child.cleanup()
            if f"arg_sets[{case_index}]" in str(error):
                raise
            raise exc.AutotuneError(
                "Failed to prepare multi-shape autotune "
                f"arg_sets[{case_index}]: {error}"
            ) from error

    def set_budget_exceeded_fn(self, fn: Callable[[], bool]) -> None:
        self.budget_exceeded_fn = fn
        for child in self.children:
            child.set_budget_exceeded_fn(_never_exceeded)

    def setup(self) -> None:
        case_index = 0
        try:
            for index, child in enumerate(self.children):
                case_index = index
                child.setup()
            if (
                self.args.relative_to is not None
                and self.args.reference_latencies is None
            ):
                reference_values = []
                for case_index, child in enumerate(self.children):
                    reference_values.append(self._measure_reference(child, case_index))
                references = tuple(reference_values)
                if _aggregate_values(references, self.args.aggregation) == inf:
                    raise exc.AutotuneError(
                        "Multi-shape reference timings must be finite and positive"
                    )
                self.args.reference_latencies = references
        except Exception as error:
            for child in reversed(self.children):
                child.cleanup()
            if f"arg_sets[{case_index}]" in str(error):
                raise
            raise exc.AutotuneError(
                f"Failed to set up multi-shape autotune arg_sets[{case_index}]: {error}"
            ) from error

    def cleanup(self) -> None:
        for child in reversed(self.children):
            child.cleanup()
        self.budget_exceeded_fn = _never_exceeded

    def _measure_reference(
        self, child: LocalBenchmarkProvider, case_index: int
    ) -> float:
        if self.args.relative_to == "default":
            result = child.benchmark(
                [child.config_spec.default_config()],
                desc="Benchmarking default reference",
            )[0]
            if (
                result.status != "ok"
                or not math.isfinite(result.perf)
                or result.perf <= 0
            ):
                raise exc.AutotuneError(
                    "Default config reference benchmark failed for "
                    f"arg_sets[{case_index}]"
                )
            return result.perf
        baseline_fn = child.settings.autotune_baseline_fn
        if baseline_fn is None:
            raise exc.AutotuneError(
                "relative_to='baseline' requires autotune_baseline_fn for "
                f"arg_sets[{case_index}]"
            )
        if child.mutated_arg_indices:
            reference_args = _clone_args(
                child.args,
                child.kernel.env.process_group_name,
                idx_to_clone=child.mutated_arg_indices,
            )
        else:
            reference_args = child.args
        backend = getattr(child.config_spec, "backend", None)
        benchmark_runner = (
            backend.get_do_bench() if backend is not None else None
        ) or do_bench
        try:
            timing = benchmark_runner(
                functools.partial(baseline_fn, *reference_args),
                return_mode="median",
                warmup=1,
                rep=50,
                process_group_name=child.kernel.env.process_group_name,
            )
        except Exception as error:
            raise exc.AutotuneError(
                f"Baseline reference benchmark failed for arg_sets[{case_index}]"
            ) from error
        if isinstance(timing, tuple):
            timing = timing[0]
        timing = float(timing)
        if not math.isfinite(timing) or timing <= 0:
            raise exc.AutotuneError(
                "Baseline reference benchmark returned invalid timing "
                f"{timing!r} for arg_sets[{case_index}]"
            )
        return timing

    def benchmark(
        self,
        configs: list[Config],
        *,
        desc: str = "Benchmarking",
    ) -> list[BenchmarkResult]:
        return self._benchmark(configs, desc=desc, record_results=True)

    def _benchmark(
        self,
        configs: list[Config],
        *,
        desc: str,
        record_results: bool,
        check_budget: bool = True,
    ) -> list[BenchmarkResult]:
        if not configs:
            return []
        if check_budget and self.budget_exceeded_fn():
            return [
                BenchmarkResult(config, _unset_fn, inf, "error", None)
                for config in configs
            ]

        materialized: list[Config] = []
        valid_indices: list[int] = []
        results = [
            BenchmarkResult(config, _unset_fn, inf, "error", None) for config in configs
        ]
        for config_index, config in enumerate(configs):
            try:
                materialized.append(
                    _materialize_multi_shape_config(self.config_spec, config)
                )
            except exc.InvalidConfig as error:
                self.log.debug(
                    f"Skipping config that is invalid for the anchor shape: {error}"
                )
                if record_results:
                    self._record_aggregate_result(config, results[config_index])
            else:
                valid_indices.append(config_index)
        if not materialized:
            return results
        materialized_keys = [repr(config) for config in materialized]
        executed_configs = [copy.deepcopy(config) for config in materialized]

        child_failure_snapshots = [
            (
                len(child._accuracy_failure_config_ids),
                len(child._compile_failure_config_ids),
            )
            for child in self.children
        ]
        child_results = [
            self._benchmark_child(
                child,
                materialized,
                desc=f"{desc} shape {index + 1}",
                case_index=index,
            )
            for index, child in enumerate(self.children)
        ]
        if record_results:
            accuracy_failure_ids: set[int] = set()
            compile_failure_ids: set[int] = set()
            for child, (before_accuracy, before_compile) in zip(
                self.children, child_failure_snapshots, strict=True
            ):
                accuracy_failure_ids.update(
                    child._accuracy_failure_config_ids[before_accuracy:]
                )
                compile_failure_ids.update(
                    child._compile_failure_config_ids[before_compile:]
                )
            self._autotune_metrics.num_accuracy_failures += len(accuracy_failure_ids)
            self._autotune_metrics.num_compile_failures += len(compile_failure_ids)
        for child_config_index, config_index in enumerate(valid_indices):
            original = configs[config_index]
            row = [child[child_config_index] for child in child_results]
            timings = [result.perf for result in row]
            valid = all(
                result.status == "ok" and math.isfinite(result.perf) and result.perf > 0
                for result in row
            )
            perf = (
                _aggregate_multi_shape_timings(
                    timings,
                    aggregation=self.args.aggregation,
                    references=self.args.reference_latencies,
                )
                if valid
                else inf
            )
            if valid:
                timing_tuple = tuple(timings)
                statuses = tuple(result.status for result in row)
                self.args.measurements[materialized_keys[child_config_index]] = (
                    timing_tuple,
                    perf,
                    statuses,
                )
                self.log.debug(
                    lambda config=original, values=timing_tuple, objective=perf: (
                        _format_multi_shape_measurement(
                            self.args,
                            config,
                            values,
                            objective,
                            selected=False,
                        )
                    )
                )
            else:
                statuses = tuple(result.status for result in row)
                timing_tuple = tuple(timings)
                measurement_key = materialized_keys[child_config_index]
                self.args.measurements[measurement_key] = (
                    timing_tuple,
                    inf,
                    statuses,
                )
                self.log.debug(
                    lambda config=original, values=timing_tuple, states=statuses: (
                        _format_multi_shape_measurement(
                            self.args,
                            config,
                            values,
                            inf,
                            selected=False,
                            statuses=states,
                        )
                    )
                )
            timed_out = any(result.status == "timeout" for result in row)
            if timed_out:
                status: Literal["ok", "error", "timeout"] = "timeout"
            else:
                status = "ok" if math.isfinite(perf) else "error"
            compile_times = [
                result.compile_time for result in row if result.compile_time is not None
            ]
            compile_time = max(compile_times, default=None)
            anchor_fn = row[0].fn
            result = BenchmarkResult(
                config=original,
                fn=anchor_fn,
                perf=perf,
                status=status,
                compile_time=compile_time,
            )
            results[config_index] = result
            if math.isfinite(perf):
                self.args.found_valid_config = True
            if record_results:
                self._record_aggregate_result(
                    executed_configs[child_config_index],
                    result,
                    timings=timings,
                )
        return results

    def _benchmark_child(
        self,
        child: LocalBenchmarkProvider,
        configs: list[Config],
        *,
        desc: str,
        case_index: int,
    ) -> list[BenchmarkResult]:
        try:
            return child.benchmark(configs, desc=desc)
        except Exception as error:
            if not self._is_skippable_child_failure(child, error):
                raise
            self.log.debug(
                "Skipping all configs for "
                f"arg_sets[{case_index}] after {type(error).__name__}: {error}"
            )
            return [
                BenchmarkResult(config, _unset_fn, inf, "error", None)
                for config in configs
            ]

    @staticmethod
    def _is_skippable_child_failure(
        child: LocalBenchmarkProvider, error: Exception
    ) -> bool:
        if match_unrecoverable_runtime_error(error):
            return False
        if isinstance(error, exc.InvalidConfig):
            return True
        backend = getattr(child.config_spec, "backend", None)
        action = (
            backend.classify_autotune_exception(error) if backend is not None else None
        ) or classify_triton_exception(error)
        return child.settings.autotune_ignore_errors or action == "debug"

    def _record_aggregate_result(
        self,
        config: Config,
        result: BenchmarkResult,
        *,
        timings: Sequence[float] | None = None,
    ) -> None:
        self._autotune_metrics.num_configs_tested += 1
        config_id = self.log.register_config(config)
        if config_id is None:
            return
        self.log.record_autotune_entry(
            AutotuneLogEntry(
                generation=self._autotune_metrics.num_generations,
                status=result.status,
                perf_ms=(
                    _aggregate_values(timings, self.args.aggregation)
                    if timings is not None and math.isfinite(result.perf)
                    else None
                ),
                compile_time=result.compile_time,
                config_id=config_id,
                config=config,
            )
        )

    def raw_latency(self, config: Config) -> float:
        materialized = _materialize_multi_shape_config(self.config_spec, config)
        measurement = self.args.measurements.get(repr(materialized))
        if measurement is None:
            return inf
        timings, _, _ = measurement
        return _aggregate_values(timings, self.args.aggregation)

    def has_valid_measurement(self, config: Config) -> bool:
        return _has_valid_multi_shape_measurement(self.args, self.config_spec, config)

    def log_selected(self, config: Config) -> None:
        materialized = _materialize_multi_shape_config(self.config_spec, config)
        summary = _format_selected_multi_shape_measurement(self.args, materialized)
        if summary is not None:
            self.log(summary)

    def rebenchmark(
        self,
        configs: list[Config],
        previous_timings: list[float],
        *,
        desc: str,
    ) -> list[float]:
        if self.budget_exceeded_fn():
            return list(previous_timings)
        results = self._benchmark(
            configs,
            desc=desc,
            record_results=False,
            check_budget=False,
        )
        return [result.perf for result in results]
