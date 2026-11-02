from __future__ import annotations

import abc
import contextlib
import contextvars
import copy
import dataclasses
import datetime
import functools
import hashlib
from itertools import count
from itertools import starmap
import json
import logging
import math
from math import inf
import os
import pickle
import tempfile
import time
from types import MappingProxyType
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

from .. import exc
from ..runtime.config import Config
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import already_compiled_fail
from ..runtime.precompile_shim import make_precompiler
from .accuracy import assert_close as _assert_close
from .accuracy import effective_fp8_accuracy
from .accuracy import is_fp8_dtype
from .accuracy import poison_cached_device_memory
from .accuracy import relative_l2_disagreement
from .benchmark_job import AccuracyCheckJob
from .benchmark_job import AccuracyCheckResult
from .benchmark_job import BenchmarkJob
from .benchmark_job import CompiledFunctionLoadError
from .benchmark_worker import BenchmarkCompileTimeout
from .benchmark_worker import BenchmarkSubprocessError
from .benchmark_worker import BenchmarkTimeout
from .benchmark_worker import BenchmarkWorker
from .benchmark_worker import BenchmarkWorkerUnkillable
from .benchmarking import clear_jit_fast_path_caches
from .benchmarking import do_bench
from .benchmarking import do_bench_generic
from .benchmarking import synchronize_device
from .kernel_args import _clone_args
from .kernel_args import make_portable_args
from .kernel_args import release_private_kernel_args
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
from .render_pool import DEFAULT_MAX_JOBS as _RENDER_POOL_DEFAULT_MAX_JOBS
from .render_pool import CudaDeviceFacts
from .render_pool import RenderAhead
from helion._dist_utils import all_gather_object
from helion._dist_utils import sync_object

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import CompiledConfig
    from ..runtime.settings import Settings
    from . import ConfigSpec
    from .base_search import _AutotunableKernel
    from .logger import AutotuningLogger
    from .metrics import AutotuneMetrics


# True while the autotuner compiles configs to measure them (a search's
# reference baseline and candidates, a best-of-K rebench, AOT measurement).  A
# BoundKernel keys the specializations a compiled config's code needs (the
# hint-sized CuTe reduction extents) only outside it, so configs that are
# measured but not picked add no cache-key facts.
_SEARCHING_CANDIDATES: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_SEARCHING_CANDIDATES", default=False
)


@contextlib.contextmanager
def _searching_candidates() -> Iterator[None]:
    token = _SEARCHING_CANDIDATES.set(True)
    try:
        yield
    finally:
        _SEARCHING_CANDIDATES.reset(token)


MultiShapeAggregation = Literal["geomean", "max"]
MultiShapeReference = Literal["default", "baseline"] | None
_SUCCESSFUL_BENCHMARK_STATUSES = frozenset(("ok", "deduplicated"))
_COMPILER_SEED_TIMEOUT_RETRY_LIMIT = 1
# Measured launches of the one retry a timed-out default config gets when
# the kernel's ConfigSpec sets no compiler-seed retry count of its own.
_DEFAULT_CONFIG_TIMEOUT_RETRY_REPETITIONS = 3
_COMPILE_CONFIG_FAILURE_SOURCE_DOMAIN = b"helion.compile_config_failure.v1\0"
# How many configs the accuracy baseline may try, counting the autotuning
# reference config itself, before giving up (see _compute_reference_baseline).
_MAX_REFERENCE_BASELINE_ATTEMPTS = 4
# Configs sampled from the search space and tried for the accuracy baseline
# when the compiler refuses the autotuning reference config (and every shrink
# of it): a kernel whose conservative lowering a backend declines can still
# have configs that run.
_MAX_SAMPLED_BASELINE_ATTEMPTS = 8
# The relative L2 difference the execution default's outputs (and the
# arguments it mutates) may show against the autotuning reference config's
# before the search is refused (``_compute_reference_baseline``): two correct
# lowerings differ in accumulation order and bf16 roundings (1e-3 or so), a
# lowering that lost or misread a term by about 1.
BASELINE_CONSISTENCY_MAX_RELATIVE_L2 = 0.05


def _benchmark_status_succeeded(status: str) -> bool:
    return status in _SUCCESSFUL_BENCHMARK_STATUSES


def _baseline_failure_is_unrecoverable(error: BaseException) -> bool:
    """Whether a baseline attempt failed with an unrecoverable runtime error
    (an illegal instruction, an illegal memory access, ...): the device may
    be in a bad state, so the search stops instead of trying another
    config.  ``_run_baseline_config`` reports such an error as
    ``exc.InvalidConfig`` *from* it; every other ``exc.InvalidConfig`` -- a
    config the compiler refuses (the cute backend declines a config at
    compile time for a resident-storage or vector-width rule, as the
    candidate path skips it) -- is a failure the next config recovers from."""
    if match_unrecoverable_runtime_error(error):
        return True
    cause = error.__cause__
    return isinstance(cause, BaseException) and match_unrecoverable_runtime_error(cause)


def _compile_config_failure_source_hash(config: Config) -> str:
    """Return a stable ledger identity when compilation produced no callable."""
    canonical = json.dumps(config.config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(
        _COMPILE_CONFIG_FAILURE_SOURCE_DOMAIN + canonical.encode("utf-8")
    ).hexdigest()


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
        succeeded = _benchmark_status_succeeded(status)
        if not succeeded or not math.isfinite(timing) or timing <= 0:
            row = f"arg_sets[{index}]{shape_text}: status={status}"
            if math.isfinite(timing):
                row += f", latency={timing:.6f} ms"
        else:
            row = f"arg_sets[{index}]{shape_text}: latency={timing:.6f} ms"
            if status != "ok":
                row += f", status={status}"
        if (
            references is not None
            and succeeded
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


class UnmeasuredConfigValue(NamedTuple):
    """A categorical config value no measured candidate carried: every
    attempted config with ``key=value`` failed to compile (``count`` of them,
    ``first_error`` the first failure's ``Type: message``)."""

    key: str
    value: object
    count: int
    first_error: str


def _categorical_config_values(config: Config) -> list[tuple[str, object]]:
    """The ``(key, value)`` pairs of a config's categorical knobs: booleans and
    strings, with list-valued knobs expanded per position (``key[i]``). The
    numeric knobs (block sizes, stage counts) are left out: a size that fails
    for resource reasons is not a lowering the search could not reach."""
    pairs: list[tuple[str, object]] = []
    if not isinstance(config, Config):
        return pairs
    for key, value in config.config.items():
        if isinstance(value, (bool, str)):
            pairs.append((key, value))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, (bool, str)):
                    pairs.append((f"{key}[{index}]", item))
    return pairs


def _first_line(error: BaseException, limit: int = 240) -> str:
    text = f"{type(error).__name__}: {error}".strip().splitlines()
    first = text[0] if text else type(error).__name__
    return first if len(first) <= limit else first[: limit - 3] + "..."


def _unmeasured_config_values(
    measured: set[tuple[str, object]],
    failures: Mapping[tuple[str, object], tuple[int, str]],
) -> list[UnmeasuredConfigValue]:
    """The compile-failure pairs no measured candidate carried, most frequent
    first (see :meth:`BenchmarkProvider.unmeasured_config_values`)."""
    facts = [
        UnmeasuredConfigValue(key, value, count, first)
        for (key, value), (count, first) in failures.items()
        if (key, value) not in measured
    ]
    facts.sort(key=lambda fact: (-fact.count, fact.key, repr(fact.value)))
    return facts


class BenchmarkResult(NamedTuple):
    """Result of benchmarking a single configuration."""

    config: Config
    fn: Callable[..., object]
    perf: float
    # "skipped": the autotune budget ran out before this config was compiled
    # (or, with a precompile pool, timed); it was never attempted.
    status: Literal[
        "ok",
        "error",
        "timeout",
        "peer_compilation_fail",
        "filtered",
        "deduplicated",
        "accuracy_error",
        "source_rejected",
        "skipped",
    ]
    compile_time: float | None
    # Monotonic completion time, captured before later trials in the batch.
    # Optional for compatibility with custom benchmark providers.
    completed_at: float | None = None
    # Per-result evidence survives later measurements of normalized aliases.
    per_shape: tuple[float, ...] = ()
    per_shape_statuses: tuple[str, ...] = ()


# Private per-finalist argument copies are only taken when they leave most of
# the device memory untouched (the worker holds the originals as well).
_PRIVATE_ARGUMENT_COPIES_MEMORY_FRACTION = 0.5


class IsolatedBenchmarkFailure(NamedTuple):
    """A conclusive failure while rechecking an already-validated function.

    ``None`` remains the provider's backward-compatible signal that isolated
    timing was unavailable and the caller should retain its prior timing.
    """

    status: Literal["error", "timeout"]


IsolatedBenchmarkTiming = float | None | IsolatedBenchmarkFailure


@dataclasses.dataclass(frozen=True)
class _PendingEffectiveSourceFailure:
    config: Config
    config_id: str | None
    compile_time: float | None


def _unset_fn(*args: object) -> NoReturn:
    raise RuntimeError("Uninitialized function")


def _never_exceeded() -> bool:
    return False


# Once the budget hook says stop, a batch still attempts at most this many
# further candidates (compiles, and with a precompile pool timings) while the
# search has no successful measurement, so a budget that runs out before the
# first measurement returns a config when one of them works instead of
# ``NoConfigFound``, and overruns the budget by a bounded amount otherwise.
_BUDGET_EXEMPT_CANDIDATES = 3

# Render workers that fail where the parent's own render of the same config
# succeeds (code generation that reads tensor data, say) are given up after
# this many in a row: they would only add fork overhead to the serial path.
_RENDER_POOL_MAX_FORK_ONLY_FAILURES = 3


def _local_rank_count() -> int:
    """Autotuning processes sharing this host's CPUs (1 outside a process group)."""
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size:
        return max(1, int(local_world_size))
    if dist.is_initialized():
        return max(1, dist.get_world_size())
    return 1


def _available_cpus() -> int:
    """CPUs this process may run on (its affinity mask where there is one)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


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
    # The first compile error of the batch in progress (``None`` when every
    # config compiled); kept for the search's batch summary and surfaced as
    # the no-viable-config error when a budget cut ends the batch early.
    batch_first_compile_error: Exception | None = None

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

    def set_compiler_seed_configs(self, configs: Sequence[Config]) -> None:
        """Register normalized compiler-owned seeds, if the provider needs them."""
        return None

    def set_config_roles(self, roles: Mapping[Config, str]) -> None:
        """Register the configs the search benchmarks for a reason of their own.

        ``roles`` maps each such config, in the normalized form the search
        hands to :meth:`benchmark`, to the name of its role ("default config",
        "compiler seed", ...): a benchmark timeout of one of them is a search
        fact (the search goes on from random configs) and is reported by role.
        """
        return None

    def has_measured_source_hash(self, source_hash: str) -> bool:
        """Return whether a successful measurement produced ``source_hash``."""
        return False

    def take_effective_source_repairs(self) -> dict[Config, BenchmarkResult]:
        """Return and clear source-equivalent repairs discovered since last read."""
        return {}

    def take_no_viable_config_error(self) -> Exception | None:
        """Return and clear the compile error deferred by the last batch.

        A ``raise_if_no_viable_config=False`` batch that would otherwise have
        re-raised its compile error keeps it here (see :meth:`benchmark`).
        """
        return None

    def unmeasured_config_values(self) -> list[UnmeasuredConfigValue]:
        """Categorical config values the search could never measure (see
        :class:`UnmeasuredConfigValue`), most frequent first; empty when the
        provider does not track them."""
        return []

    def invalidate_effective_source_hash(self, source_hash: str) -> None:
        """Prevent a failed rebenchmark source from being reused as an alias."""
        return None

    @abc.abstractmethod
    def benchmark(
        self,
        configs: list[Config],
        *,
        desc: str = "Benchmarking",
        raise_if_no_viable_config: bool = True,
    ) -> list[BenchmarkResult]:
        """Compile, precompile, validate, and time a batch of configs.

        Handles the full benchmark flow: compilation, optional subprocess
        precompilation, accuracy validation, timing, error classification,
        and progress reporting.

        When nothing has been measured yet and every config in the batch
        fails to compile, the compile error is re-raised so a broken kernel
        surfaces its real cause instead of ``NoConfigFound``. Callers that
        hold a fallback population (a seed-only initial batch) pass
        ``raise_if_no_viable_config=False`` to receive ``perf=inf`` rows
        instead; the error is kept for :meth:`take_no_viable_config_error` so
        the caller can still raise it when its fallback turns out empty.

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
        fresh_process: bool = False,
        l2_flush_mode: str | None = None,
    ) -> list[IsolatedBenchmarkTiming] | None:
        """Benchmark already-validated functions in an isolated subprocess.

        With ``fresh_process``, discard earlier worker state and use one new
        process per function. This isolates cached arguments and allocations
        during final selection while retaining worker reuse during search.
        ``l2_flush_mode`` names the L2 regime of the timed calls (the final
        selection's); ``None`` keeps the search regime.

        Return ``None`` when the provider cannot support the isolated path or
        per-function ``None`` when a timing could not be confirmed and callers
        should keep the prior timing for that function. A returned
        :class:`IsolatedBenchmarkFailure` is conclusive and lets search policy
        decide whether that candidate must be invalidated.
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

    # Class-level default: tests construct partially-initialized providers, so
    # the picklability flag must resolve even before setup()/__init__ set it.
    _args_unpicklable: bool = False
    _subprocess_wrapper_unloadable: bool = False
    # Search facts (see ``unmeasured_config_values``): the categorical
    # (key, value) pairs some measured candidate carried, per pair of a
    # candidate that failed to compile its failure count and first error, and
    # the configs the search registered by role (``set_config_roles``, for
    # ``_timed_out_config_role``). Created on first use so a provider built
    # without ``__init__`` keeps working.
    _measured_config_values: set[tuple[str, object]] | None = None
    _compile_failure_values: dict[tuple[str, object], tuple[int, str]] | None = None
    _config_roles: Mapping[Config, str] = MappingProxyType({})
    _compiler_seed_configs: set[Config] | frozenset[Config] = frozenset()
    # The configs behind the functions of the isolated batch in progress,
    # by ``id(fn)`` (``register_isolated_candidates``), so a failed
    # re-timing can name its candidate.
    _isolated_candidate_configs: Mapping[int, Config] = MappingProxyType({})

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
        self._worker_failure_config_ids: list[int] = []
        self._precompile_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._precompile_args_path: str | None = None
        self._args_unpicklable: bool = False
        self._subprocess_wrapper_unloadable: bool = False
        self._precompile_baseline_path: str | None = None
        self._precompile_baseline_post_args_path: str | None = None
        self._precompile_result_counter: count[int] = count()
        self._benchmark_worker: BenchmarkWorker | None = None
        self._last_benchmark_failure_status: Literal["error", "timeout"] | None = None
        self._no_viable_config_error: Exception | None = None
        # Configs of the batch in progress whose compile has started; read by
        # the budget hook ``PopulationBasedSearch._initial_population_budget``
        # installs to price the initial population.
        self.batch_configs_started: int = 0
        # Render workers (see ``_render_ahead``): the CUDA facts they serve,
        # captured once; whether they were given up (a fork failed, or they
        # kept failing where the parent renders fine); the fork-only failure
        # streak; and whether a worker timeout was already warned about.
        self._cuda_facts: CudaDeviceFacts | None = None
        self._render_pool_unavailable: bool = False
        self._render_fork_only_failures: int = 0
        self._render_timeout_warned: bool = False
        self._effective_source_hashes: set[str] = set()
        self._effective_source_results: dict[str, BenchmarkResult] = {}
        self._invalid_effective_source_hashes: set[str] = set()
        self._pending_effective_source_failures: dict[
            str, list[_PendingEffectiveSourceFailure]
        ] = {}
        self._effective_source_repairs: dict[Config, BenchmarkResult] = {}
        self._compiler_seed_configs: set[Config] = set()
        self._compiler_seed_source_hashes: set[str] = set()
        self._compiler_seed_timeout_retry_claims: set[tuple[str, object]] = set()
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
        # Scale the atol floor per tensor only when the user did not pin an
        # explicit absolute tolerance (see accuracy.assert_close).
        self._scale_atol = self.settings.autotune_baseline_atol is None
        self._jobs = self._decide_num_jobs()

    @property
    def _fp8_accuracy(self) -> str:
        """How fp8 leaves are compared: the ``autotune_fp8_accuracy`` setting,
        or ``"bitwise"`` when explicit zero tolerances ask for exact bytes."""
        return effective_fp8_accuracy(self.settings)

    def _record_accuracy_failure(self, config: Config) -> None:
        self._autotune_metrics.num_accuracy_failures += 1
        self._accuracy_failure_config_ids.append(id(config))

    def has_measured_source_hash(self, source_hash: str) -> bool:
        result = self._effective_source_results.get(source_hash)
        return (
            result is not None
            and result.status in ("ok", "deduplicated")
            and math.isfinite(result.perf)
        )

    def take_effective_source_repairs(self) -> dict[Config, BenchmarkResult]:
        repairs = self._effective_source_repairs
        self._effective_source_repairs = {}
        return repairs

    def take_no_viable_config_error(self) -> Exception | None:
        error = self._no_viable_config_error
        self._no_viable_config_error = None
        return error

    def invalidate_effective_source_hash(self, source_hash: str) -> None:
        self._invalid_effective_source_hashes.add(source_hash)
        self._effective_source_results.pop(source_hash, None)
        self._pending_effective_source_failures.pop(source_hash, None)
        backend = self.config_spec.backend
        for config, repair in list(self._effective_source_repairs.items()):
            if backend.generated_source_hash(repair.fn) == source_hash:
                del self._effective_source_repairs[config]

    def _remember_effective_source_failure(
        self,
        source_hash: str,
        config: Config,
        config_id: str | None,
        compile_time: float | None,
    ) -> None:
        stored_config = copy.deepcopy(config)
        self._pending_effective_source_failures.setdefault(source_hash, []).append(
            _PendingEffectiveSourceFailure(
                config=stored_config,
                config_id=config_id,
                compile_time=compile_time,
            )
        )

    def _resolve_effective_source_failures(
        self,
        source_hash: str,
        source_result: BenchmarkResult,
    ) -> dict[Config, BenchmarkResult]:
        repaired: dict[Config, BenchmarkResult] = {}
        for pending in self._pending_effective_source_failures.pop(source_hash, []):
            repair = BenchmarkResult(
                config=pending.config,
                fn=source_result.fn,
                perf=source_result.perf,
                status="deduplicated",
                compile_time=None,
                completed_at=source_result.completed_at,
            )
            repaired[pending.config] = repair
            self._effective_source_repairs[pending.config] = repair
            self._autotune_metrics.num_source_deduplications += 1
            if pending.config_id is not None:
                self.log.record_autotune_entry(
                    AutotuneLogEntry(
                        # The original failure already has its own terminal row.
                        # Attribute this repair to the generation whose successful
                        # alias proved the source, including across benchmark batches.
                        generation=self._autotune_metrics.num_generations,
                        status="deduplicated",
                        perf_ms=source_result.perf,
                        compile_time=None,
                        config_id=pending.config_id,
                        config=pending.config,
                        source_hash=source_hash,
                        completed_at=repair.completed_at,
                    )
                )
        return repaired

    def set_compiler_seed_configs(self, configs: Sequence[Config]) -> None:
        self._compiler_seed_source_hashes = set()
        self._compiler_seed_timeout_retry_claims = set()
        if self._compiler_seed_timeout_retry_repetitions() is None:
            self._compiler_seed_configs = set()
            return
        self._compiler_seed_configs = {copy.deepcopy(config) for config in configs}

    def _compiler_seed_timeout_retry_repetitions(self) -> int | None:
        repetitions = self.config_spec.compiler_seed_timeout_retry_repetitions
        return repetitions if type(repetitions) is int and repetitions > 0 else None

    def _is_compiler_seed_config(self, config: Config) -> bool:
        return (
            self._compiler_seed_timeout_retry_repetitions() is not None
            and config in self._compiler_seed_configs
        )

    def _timeout_retry_repetitions(
        self, config: Config, effective_source_hash: str | None
    ) -> int | None:
        """Measured launches for the one retry a timed-out candidate gets, or
        None for a candidate that is not retried.

        A compiler seed (or a source alias of one) is retried with the count
        its ConfigSpec sets; the search's default config is retried with that
        count too, or with ``_DEFAULT_CONFIG_TIMEOUT_RETRY_REPETITIONS`` when
        the spec sets none: the default anchors the search (the finishing
        phase moves toward it, and it is a pinned finalist), so one coarse
        measurement beats dropping it on a transient timeout.
        """
        repetitions = self._compiler_seed_timeout_retry_repetitions()
        if repetitions is not None and (
            config in self._compiler_seed_configs
            or (
                effective_source_hash is not None
                and effective_source_hash in self._compiler_seed_source_hashes
            )
        ):
            return repetitions
        if self._config_roles.get(config) == "default config":
            return (
                repetitions
                if repetitions is not None
                else _DEFAULT_CONFIG_TIMEOUT_RETRY_REPETITIONS
            )
        return None

    def _claim_compiler_seed_timeout_retry(
        self,
        config: Config,
        effective_source_hash: str | None,
    ) -> bool:
        """Claim one budget-gated retry for this distinct seed source."""
        key: tuple[str, object] = (
            ("source", effective_source_hash)
            if effective_source_hash is not None
            else ("config", config)
        )
        if key in self._compiler_seed_timeout_retry_claims:
            return False
        if self.budget_exceeded_fn():
            return False
        self._compiler_seed_timeout_retry_claims.add(key)
        return True

    def _record_compile_failure(
        self, config: Config, error: BaseException | None = None
    ) -> None:
        self._autotune_metrics.num_compile_failures += 1
        self._compile_failure_config_ids.append(id(config))
        if error is None:
            return
        pairs = _categorical_config_values(config)
        if not pairs:
            return
        if self._compile_failure_values is None:
            self._compile_failure_values = {}
        message = _first_line(error)
        for pair in pairs:
            count, first = self._compile_failure_values.get(pair, (0, message))
            self._compile_failure_values[pair] = (count + 1, first)

    def _record_measured_configs(self, results: Sequence[BenchmarkResult]) -> None:
        """Note the categorical values of a batch's measured candidates: a
        timed result, a source alias of one, or a failure a later alias
        repaired -- every result the batch ends with a finite reading."""
        for result in results:
            if not math.isfinite(result.perf):
                continue
            pairs = _categorical_config_values(result.config)
            if not pairs:
                continue
            if self._measured_config_values is None:
                self._measured_config_values = set()
            self._measured_config_values.update(pairs)

    def unmeasured_config_values(self) -> list[UnmeasuredConfigValue]:
        return _unmeasured_config_values(
            self._measured_config_values or set(),
            self._compile_failure_values or {},
        )

    def set_config_roles(self, roles: Mapping[Config, str]) -> None:
        self._config_roles = dict(roles)

    def _timed_out_config_role(self, config: Config) -> str | None:
        """How a config that timed out in the benchmark worker mattered to the
        search: the role the search registered for it (``set_config_roles``:
        its default config, a compiler or user seed), a compiler seed kept for
        the timeout retry, or None for an ordinary candidate."""
        role = self._config_roles.get(config)
        if role is not None:
            return role
        if config in self._compiler_seed_configs:
            return "compiler seed"
        return None

    def _record_worker_failure(
        self,
        config: Config,
        status: Literal["error", "timeout"],
    ) -> None:
        self._last_benchmark_failure_status = status
        self._autotune_metrics.num_worker_failures += 1
        self._worker_failure_config_ids.append(id(config))

    def _warn_timed_out_role(
        self,
        config: Config,
        error: BenchmarkTimeout | BenchmarkCompileTimeout,
        *,
        retried_repetitions: int | None = None,
    ) -> None:
        """Name the search role of a config a worker deadline dropped.

        The default config or a compiler seed timing out is a search fact,
        not one slow candidate among many: the search then starts from random
        configs (and ends in ``NoConfigFound`` when none of them runs), which
        the generic per-config warning does not explain. A compile deadline
        names its own knob: the candidate was never timed.
        """
        role = self._timed_out_config_role(config)
        if role is None:
            return
        if isinstance(error, BenchmarkCompileTimeout):
            self.log.warning(
                f"The {role} did not finish compiling in the benchmark worker "
                f"({error}; HELION_AUTOTUNE_COMPILE_TIMEOUT), so the search "
                "continues without it from the other seeds or random configs. "
                "Raise the compile timeout, or provide another seed, if the "
                "search ends with no config found."
            )
            return
        retried = (
            f", also with {retried_repetitions} fixed measured repetitions"
            if retried_repetitions is not None
            else ""
        )
        self.log.warning(
            f"The {role} timed out in the benchmark worker ({error}; "
            f"HELION_AUTOTUNE_BENCHMARK_TIMEOUT{retried}): its per-launch time "
            "is too long for the measurement window, so the search continues "
            "without it from the other seeds or random configs. Raise the "
            "timeout, or provide a faster seed, if the search ends with no "
            "config found."
        )

    def _record_compile_timeout(
        self, config: Config, error: BenchmarkCompileTimeout
    ) -> None:
        """A candidate whose compile (and first call) outran the worker's
        compile deadline: a compile failure with the deadline as its reason
        (it never produced a kernel the search could time), not a slow
        kernel. The line names the config so the pathological candidate can
        be reproduced without the DEBUG log."""
        self.log(
            f"Compile of {config!r} exceeded the benchmark worker's compile "
            f"deadline ({error}; HELION_AUTOTUNE_COMPILE_TIMEOUT); recorded as "
            "a compile failure"
        )
        self._warn_timed_out_role(config, error)
        self._last_benchmark_failure_status = "timeout"
        self._record_compile_failure(config, error)

    def register_isolated_candidates(
        self, fns: Sequence[Callable[..., object]], configs: Sequence[Config]
    ) -> None:
        """Name the configs behind the functions of the next isolated batch
        (``benchmark_isolated`` logs them when a re-timing fails)."""
        self._isolated_candidate_configs = {
            id(fn): config for fn, config in zip(fns, configs, strict=True)
        }

    def _isolated_failure_subject(self, fn: Callable[..., object], desc: str) -> str:
        config = self._isolated_candidate_configs.get(id(fn))
        return f"{desc} subprocess failed" + (
            f" for {config!r}" if config is not None else ""
        )

    def _compute_baseline(
        self,
    ) -> tuple[object, Sequence[int], Sequence[object] | None]:
        """
        Compute baseline output for accuracy validation during autotuning.
        Also detect if the kernel mutates any of its input arguments.

        The baseline is computed in one of two ways:
        - If settings.autotune_baseline_fn is provided, use that custom function
        - Otherwise, run the kernel with its execution default, backing off to
          the conservative autotuning reference (see _compute_reference_baseline)
        """
        # Use custom baseline function if provided
        if self.settings.autotune_baseline_fn is not None:
            new_args = _clone_args(self.args, self.kernel.env.process_group_name)
            try:
                baseline_output = self.settings.autotune_baseline_fn(*new_args)
                synchronize_device()
            except Exception as e:
                raise exc.AutotuneError(
                    "Custom baseline function failed while computing baseline.\n"
                    f"Baseline function: {self.settings.autotune_baseline_fn}\n"
                ) from e
        else:
            baseline_output, new_args = self._compute_reference_baseline()

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

    def _compute_reference_baseline(self) -> tuple[object, Sequence[object]]:
        """Run a config to produce the accuracy baseline output.

        Returns the output and the (cloned) arguments it ran on.

        The execution default goes first when it differs from the autotuning
        reference config: a compiler heuristic that promotes it (a tensor-core
        lowering) gives the fast candidates its arithmetic -- the accumulation
        order of an MMA -- which the conservative reference's SIMT code does
        not share, and a kernel whose bf16 intermediates amplify that
        difference would otherwise fail every promoted candidate against the
        reference.  The reference config backs the default off; hardware
        limits Helion cannot model -- Intel per-thread scratch space, shared
        memory, register pressure -- can reject it on a kernel whose smaller
        configs are fine, so such failures back off to smaller block sizes
        instead of aborting the search.  Failures a smaller config cannot fix
        are raised right away (see _can_back_off_baseline).  Each attempt
        re-clones the arguments so a kernel that mutates its inputs before
        failing cannot poison the baseline.

        A promoted default that runs is checked against the reference: both
        run, and their outputs and mutated arguments must agree to
        ``BASELINE_CONSISTENCY_MAX_RELATIVE_L2`` in relative L2, or the
        search is refused (``exc.AutotuneError``) -- the two are lowerings
        of one kernel over one input, so a difference beyond accumulation
        order means one of them is wrong, and the accuracy check cannot
        tell which.
        """
        reference = self.config_spec.autotune_reference_config()
        execution = self.config_spec.default_config()
        if execution.config == reference.config:
            return self._run_reference_or_sampled_baseline(reference)
        try:
            promoted = self._run_baseline_config(
                execution, "Execution default", shrink=False
            )
        except Exception as e:
            if _baseline_failure_is_unrecoverable(e):
                # The device may be in a bad state.
                raise
            # The execution default failed -- a config the compiler refused
            # (``exc.InvalidConfig``) or one the device rejected: the
            # reference config exists for exactly this.  The traceback holds
            # this attempt's cloned args and outputs; drop it so they are
            # freed before the next attempt.
            e.__traceback__ = None
            self.log.warning(
                f"The execution default {execution} failed to compute the "
                f"accuracy baseline ({e}); retrying with the autotuning "
                "reference config"
            )
            return self._run_reference_or_sampled_baseline(reference)
        # The default's output is the baseline, once the reference agrees
        # with it: the accuracy check judges every candidate against the
        # baseline, so a default whose lowering is wrong would reject the
        # right candidates and keep the wrong ones (full_gla's dqkg at
        # 9534981b9: the promoted tcgen05 default dropped dk's intra-chunk
        # dot, 58 correct warp configs were rejected and a tcgen05 winner
        # shipped a 0.997 relative error).  A reference config the device
        # rejects where the default ran is reported and skips the check, as
        # it never ran before.
        try:
            conservative = self._run_baseline_config(
                reference, "Autotuning reference config"
            )
        except exc.InvalidConfig as e:
            if _baseline_failure_is_unrecoverable(e):
                raise
            self.log.warning(
                "The autotuning reference config failed to run; the execution "
                f"default's output is the accuracy baseline unchecked ({e})"
            )
            return promoted
        reason = relative_l2_disagreement(
            promoted, conservative, BASELINE_CONSISTENCY_MAX_RELATIVE_L2
        )
        if reason is not None:
            self.kernel.maybe_log_repro(self.log.error, self.args, execution)
            raise exc.AutotuneError(
                "The execution default and the autotuning reference config "
                f"disagree on the kernel's outputs ({reason}): one of the two "
                "lowerings is wrong, so neither output can be the accuracy "
                "baseline the candidates are judged against.\n"
                "Execution default: "
                f"{self.kernel.format_kernel_decorator(execution, self.settings)}\n"
                "Autotuning reference config: "
                f"{self.kernel.format_kernel_decorator(reference, self.settings)}\n"
                "Set HELION_AUTOTUNE_ACCURACY_CHECK=0 to skip every accuracy check, "
                "or `@helion.kernel(autotune_baseline_fn=...)` to provide the "
                "baseline (e.g. a PyTorch eager implementation of your kernel)."
            )
        return promoted

    def _run_reference_or_sampled_baseline(
        self, reference: Config
    ) -> tuple[object, Sequence[object]]:
        """The autotuning reference config's baseline run, or, when the
        compiler refuses it and every shrink of it, the first of a few configs
        sampled from the search space that runs (``_sampled_baseline_configs``).

        A backend can decline a kernel's conservative lowering for a reason no
        block size fixes (the cute backend refuses an ``hl.split`` exchange
        its lane-loop proof cannot settle under the default layouts) while
        other points of the search space compile and run; without a baseline
        the search never starts.  Any config that runs is as good an oracle
        as the reference was: both are lowerings of the kernel over its
        input.  An unrecoverable runtime error, or a failure the backend
        classifies as a bug rather than a config's (``_can_back_off_baseline``),
        still stops the search right away."""
        try:
            return self._run_baseline_config(reference, "Autotuning reference config")
        except exc.InvalidConfig as error:
            cause = error.__cause__
            if _baseline_failure_is_unrecoverable(error) or (
                isinstance(cause, Exception) and not self._can_back_off_baseline(cause)
            ):
                # The device is in a bad state, or a bug no config avoids.
                raise
            error.__traceback__ = None
            reference_error = error
        for config in self._sampled_baseline_configs(_MAX_SAMPLED_BASELINE_ATTEMPTS):
            try:
                result = self._run_baseline_config(
                    config, "Sampled baseline config", shrink=False
                )
            except Exception as e:
                if _baseline_failure_is_unrecoverable(e):
                    raise
                e.__traceback__ = None
                self.log.warning(f"Sampled baseline config {config} failed ({e})")
                continue
            cause = reference_error.__cause__ or reference_error
            self.log.warning(
                f"Using the sampled config {config} for the accuracy baseline; "
                f"the autotuning reference config {reference} failed to run ({cause})"
            )
            return result
        raise reference_error

    def _sampled_baseline_configs(self, count: int) -> list[Config]:
        """Up to ``count`` random configs of the search space to try for the
        baseline; none when the spec cannot generate them."""
        from .config_generation import ConfigGeneration

        try:
            generation = ConfigGeneration(
                self.config_spec,
                process_group_name=self.kernel.env.process_group_name,
            )
            return [generation.random_config() for _ in range(count)]
        except Exception as e:
            self.log.warning(f"No sampled baseline configs: {e}")
            return []

    def _run_baseline_config(
        self, config: Config, role: str, *, shrink: bool = True
    ) -> tuple[object, Sequence[object]]:
        """Compile and run ``config`` on cloned arguments for the accuracy
        baseline: ``(output, args)``, the arguments as the run left them.
        With ``shrink`` (the reference config) a failure a smaller config
        might avoid backs off the block sizes up to
        ``_MAX_REFERENCE_BASELINE_ATTEMPTS`` times; a failure nothing fixes
        is raised as ``exc.InvalidConfig`` naming ``role``.  Without it
        (the execution default) every failure but an unrecoverable runtime
        error is raised as is -- a compile-time ``exc.InvalidConfig``
        included -- for the caller to fall back; the unrecoverable error is
        raised as ``exc.InvalidConfig`` *from* it
        (``_baseline_failure_is_unrecoverable``)."""
        failure: tuple[Exception, Config, str] | None = None
        shrinks = 0
        while True:
            new_args = _clone_args(self.args, self.kernel.env.process_group_name)
            try:
                with _searching_candidates():
                    baseline_fn = self.kernel.compile_config(config, allow_print=False)
                # The baseline's fresh outputs must not inherit a previous
                # run's values either (see poison_cached_device_memory).
                if self.settings.autotune_accuracy_check:
                    poison_cached_device_memory(self.kernel.env.device)
                baseline_output = baseline_fn(*new_args)
                synchronize_device()
            except Exception as e:
                if not shrink and not match_unrecoverable_runtime_error(e):
                    raise
                if not self._can_back_off_baseline(e):
                    # A smaller config cannot fix this; report it as is, with
                    # its traceback.
                    failure = (e, config, role)
                    break
                # The traceback holds this attempt's cloned args and outputs;
                # drop it so they are freed before the next attempt.
                e.__traceback__ = None
                if failure is None:
                    failure = (e, config, role)
                shrinks += 1
                if shrinks >= _MAX_REFERENCE_BASELINE_ATTEMPTS:
                    break
                smaller = self.config_spec.shrink_block_sizes_once(config)
                if smaller is None:
                    break
                self.log.warning(
                    f"Autotuning reference config {config} failed ({e}); retrying "
                    f"the baseline with block sizes {smaller.config.get('block_sizes')}"
                )
                config = smaller
                continue
            if failure is not None:
                self.log.warning(
                    f"Using block sizes {config.config.get('block_sizes')} for the "
                    "accuracy baseline; the autotuning reference config "
                    f"{failure[1]} failed to run"
                )
            return baseline_output, new_args

        assert failure is not None
        error, baseline_config, role = failure
        decorator = self.kernel.format_kernel_decorator(baseline_config, self.settings)
        log_generated_triton_code_debug(
            self.log,
            self.kernel,
            baseline_config,
            prefix=f"Generated Triton code for {decorator}:",
        )
        self.kernel.maybe_log_repro(self.log.error, self.args, baseline_config)
        raise exc.InvalidConfig(
            f"{role} failed while computing baseline.\n"
            f"{role}: {decorator}\n"
            f"{SUPPRESSED_TRITON_CODE_MSG}\n"
            "To work around this error, you could set `@helion.kernel(autotune_baseline_fn=...)` "
            "to provide a custom baseline function (e.g. PyTorch eager implementation of your kernel)."
        ) from error

    def _can_back_off_baseline(self, error: Exception) -> bool:
        """Whether a smaller config might avoid *error* from a baseline attempt.

        Uses the same classification as benchmarking a candidate: failures it
        skips as config-specific (e.g. out of resources) may go away with
        smaller block sizes, while unrecoverable runtime errors and errors it
        raises (bugs) will not.
        """
        if match_unrecoverable_runtime_error(error):
            return False
        action = self.config_spec.backend.classify_autotune_exception(
            error
        ) or classify_triton_exception(error)
        return action != "raise"

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
            if self._fp8_accuracy == "bitwise":
                # All dtypes are fp8 - use bitwise comparison unless the user
                # explicitly set either tolerance value (i.e., not None); an
                # explicit 0/0 selected this mode and is returned below.
                if atol is None and rtol is None:
                    self.log(
                        f"Detected fp8 dtype(s) in output: {dtypes}. "
                        "Using bitwise comparison (atol=0.0, rtol=0.0) for autotuning accuracy check."
                    )
                    return 0.0, 0.0
            else:
                # The dequantized comparison admits one fp8 ulp on top of the
                # ordinary tolerances (see accuracy._assert_close_fp8_ulp).
                self.log(
                    f"Detected fp8 dtype(s) in output: {dtypes}. "
                    "Comparing dequantized values within one fp8 ulp plus the "
                    "tolerances for the autotuning accuracy check "
                    "(autotune_fp8_accuracy='ulp')."
                )

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
            error = self._save_kernel_args(self.args, args_path)
            if error is None:
                self._precompile_args_path = args_path
            else:
                # Spawn-mode precompile keeps its hard error since it cannot
                # run at all without the args file.
                if self.settings.autotune_precompile == "spawn":
                    raise error
                self._benchmark_in_process(error)
        if self._subprocess_accuracy_check_enabled():
            baseline_path = os.path.join(self._precompile_tmpdir.name, "baseline.pt")
            torch.save(self._baseline_output, baseline_path)
            self._precompile_baseline_path = baseline_path
            if len(self.mutated_arg_indices) > 0:
                # The worker compares a candidate's post-call mutated
                # arguments against the baseline's, as the in-process path
                # does with ``self._baseline_post_args``.  They are the
                # whole argument tree with the mutated tensors cloned, so a
                # callable argument is in them too and takes the same
                # serialization (and the same fallback) as the arguments.
                post_args_path = os.path.join(
                    self._precompile_tmpdir.name, "baseline_post_args.pt"
                )
                assert self._baseline_post_args is not None
                error = self._save_kernel_args(self._baseline_post_args, post_args_path)
                if error is None:
                    self._precompile_baseline_post_args_path = post_args_path
                else:
                    self._benchmark_in_process(error)

    def _benchmark_in_process(self, error: Exception) -> None:
        """Keep the benchmark and accuracy check in this process (the
        arguments cannot cross the spawn boundary) instead of failing the
        whole autotune."""
        self._args_unpicklable = True
        self.log.warning(
            f"Autotune args are not picklable ({error}); benchmarking "
            "in-process instead of in a killable subprocess."
        )

    def _save_kernel_args(self, args: Sequence[object], path: str) -> Exception | None:
        """Write an argument tree for the benchmark subprocess; the pickling
        error when it cannot be written.

        Kernel args holding lambdas/closures (e.g. an epilogue callable)
        cannot cross a spawn boundary as they are.  The worker rebuilds such
        functions from their code (``PortableFunction``): benchmarking must
        stay out of this process, because an in-process cuda-graph benchmark
        leaves the process in a state where every later CUDA graph runs each
        kernel node of a Helion-launched kernel ~0.35 us slower (measured on
        B200: a 2.8 us two-kernel call reads 3.5 us afterwards, for the cute
        and the triton backends alike).  Only when that fails too (a function
        that cannot be serialized by code, or another argument that still
        refuses to pickle) does the caller fall back to the in-process
        benchmark/accuracy path.
        """
        try:
            torch.save(args, path)
            return None
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            portable = make_portable_args(args)
            if portable is None:
                return error
            try:
                torch.save(portable, path)
            except (pickle.PicklingError, AttributeError, TypeError):
                return error
            self.log.debug(
                f"Autotune args hold functions standard pickling refuses ({error}); "
                "serialized them by code for the benchmark subprocess."
            )
            return None

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
        self._precompile_baseline_post_args_path = None
        self._precompile_result_counter = count()
        self._cuda_facts = None
        self._effective_source_hashes.clear()
        self._effective_source_results.clear()
        self._invalid_effective_source_hashes.clear()
        self._pending_effective_source_failures.clear()
        self._effective_source_repairs.clear()
        self._compiler_seed_source_hashes.clear()
        self._compiler_seed_timeout_retry_claims.clear()
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
        # Always False for the stock cute backend since it inherits the
        # default (event-timed) get_do_bench; this hook remains for a backend
        # that opts into wall-clock timing while still supporting the simple
        # subprocess benchmark job shape.
        backend = getattr(self.config_spec, "backend", None)
        if backend is None:
            return False
        custom_bench = backend.get_do_bench()
        return backend.name == "cute" and custom_bench is do_bench_generic

    def _probe_long_kernel(self) -> bool:
        """Whether timing should stop its estimate after one long launch."""
        if self.config_spec.cute_flash_search_enabled:
            return True
        return self.config_spec.backend.probe_long_autotune_kernels(self.config_spec)

    def _effective_source_dedup_enabled(self) -> bool:
        """Whether this provider may collapse source-identical candidates.

        Rank-specialized distributed code can produce a different source
        partition on each rank. Keeping every rank's benchmark loop identical
        is more important than deduplicating those uncommon searches.
        """
        return (
            self.config_spec.backend.should_deduplicate_generated_sources(
                self.config_spec
            )
            and not dist.is_initialized()
        )

    def _subprocess_benchmark_enabled(self) -> bool:
        """Whether candidates are timed in the killable benchmark worker.

        Enabled by default (``autotune_benchmark_subprocess``); skipped for
        distributed kernels, where the worker's simple job shape doesn't fit,
        and when the arguments or the generated wrapper cannot cross the
        process boundary. A kernel that writes into its inputs runs there
        too: the job clones the mutated arguments per candidate, as the
        in-process path does (``BenchmarkJob.mutated_arg_indices``). In the
        worker a candidate's device-side fault (an illegal instruction, an
        illegal address) kills only the worker; in-process it poisons the
        autotuning process's CUDA context and ends the whole search.
        """
        if not self.settings.autotune_benchmark_subprocess:
            return False
        if self._args_unpicklable:
            return False
        if self._subprocess_wrapper_unloadable:
            return False
        if dist.is_initialized():
            return False
        if getattr(getattr(self.kernel.env, "device", None), "type", None) == "mtia":
            # The MTIA runtime reserves the whole device for its first client,
            # so a spawned worker cannot open it while this process holds it.
            return False
        if not self.kernel.supports_subprocess_benchmark():
            return False
        backend = getattr(self.config_spec, "backend", None)
        if backend is None or backend.get_do_bench() is None:
            return True
        return self._subprocess_benchmark_uses_wall_clock()

    def _in_process_benchmark_reason(self) -> str:
        """Why candidates run in this process instead of the killable
        benchmark worker (for the message of a fatal device-side fault)."""
        if not self.settings.autotune_benchmark_subprocess:
            return "autotune_benchmark_subprocess is off (HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0)"
        if self._args_unpicklable:
            return "the kernel arguments are not picklable"
        if self._subprocess_wrapper_unloadable:
            return "the generated wrapper could not be loaded in the worker"
        if dist.is_initialized():
            return "distributed kernels run in-process"
        if getattr(getattr(self.kernel.env, "device", None), "type", None) == "mtia":
            return "the MTIA runtime admits one process per device"
        if not self.kernel.supports_subprocess_benchmark():
            return "this kernel does not support the benchmark worker"
        if not self._subprocess_benchmark_enabled():
            return "the backend's timing function runs in-process"
        if self.settings.autotune_baseline_accuracy_check_fn is not None:
            return "a custom autotune_baseline_accuracy_check_fn validates in-process"
        return "the candidate's benchmark job could not be sent to the worker"

    def _subprocess_accuracy_check_enabled(self) -> bool:
        """Default accuracy checks can run in the same killable worker.

        Custom accuracy callbacks can close over arbitrary process-local
        state, so those remain on the in-process path. A kernel's mutated
        arguments are checked in the worker against the baseline run's
        (``AccuracyCheckJob.baseline_post_args_path``).
        """
        return (
            self.settings.autotune_accuracy_check
            and self.settings.autotune_baseline_accuracy_check_fn is None
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
                    scale_atol_by_expected_rms=self._scale_atol,
                    fp8_accuracy=self._fp8_accuracy,
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
                            scale_atol_by_expected_rms=self._scale_atol,
                            fp8_accuracy=self._fp8_accuracy,
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

    def _budget_step_synced(self, *, protected: bool) -> tuple[bool, bool]:
        """Ask the budget hook before a compile or benchmark step.

        Returns ``(exceeded, stop)``. ``exceeded`` is whether any rank's hook
        reports the budget exhausted; ``stop`` is whether this step stops for
        it. A ``protected`` step runs past an exhausted budget: it is one of
        the few attempts the batch still makes while the search has no
        successful measurement (see :meth:`benchmark`), so a budget that runs
        out inside the first batch still yields a measured config instead of
        ``NoConfigFound``. The compile and benchmark loops take part in
        distributed collectives, so the decision is agreed across the process
        group (one rank breaking early while its peers keep entering
        collectives would deadlock the group); a protected step still asks
        the hook (a hook that counts or logs sees every step) and still takes
        part in the collective so every rank runs the same number of them.
        Single-process autotune skips the all-gather because
        ``all_gather_object`` returns ``[obj]`` when distributed is not
        initialized.
        """
        exceeded = any(
            all_gather_object(
                self.budget_exceeded_fn(),
                process_group_name=self.kernel.env.process_group_name,
            )
        )
        return exceeded, exceeded and not protected

    def _precompile_pool_enabled(self) -> bool:
        """Whether batches precompile in a Triton-style subprocess pool."""
        return bool(
            self.settings.autotune_precompile
            and self.config_spec.backend.supports_precompile()
        )

    @property
    def compiles_ahead(self) -> bool:
        """Whether a batch is compiled in full before any config is timed.

        True with a precompile pool; the streaming path (no pool, with or
        without render workers) compiles and times one config at a time.
        """
        return self._precompile_pool_enabled()

    def _render_pool_enabled(self) -> bool:
        """Whether candidates render in forked workers (``render_pool``).

        Fork mode on a backend without a precompile pool whose code
        generation is fork-safe (``supports_parallel_render``), for a real
        bound kernel, until a fork fails.
        """
        if (
            self.settings.autotune_precompile != "fork"
            or self._precompile_pool_enabled()
            or self._render_pool_unavailable
        ):
            return False
        from ..runtime.kernel import BoundKernel

        return (
            isinstance(self.kernel, BoundKernel)
            and self.config_spec.backend.supports_parallel_render()
        )

    def _render_jobs(self) -> int:
        """Render workers alive at once.

        The precompile job setting when given; otherwise the CPUs this
        process may run on, shared between the autotuning processes of the
        host when there is a process group, capped at
        ``render_pool.DEFAULT_MAX_JOBS``.
        """
        jobs = self.settings.autotune_precompile_jobs
        if jobs:
            return max(1, jobs)
        jobs = _available_cpus() // _local_rank_count()
        return max(1, min(jobs, _RENDER_POOL_DEFAULT_MAX_JOBS))

    def _render_ahead(self, configs: list[Config]) -> RenderAhead | None:
        """Start rendering ``configs`` in forked workers, or None for inline."""
        if len(configs) < 2 or not self._render_pool_enabled():
            return None
        from ..runtime.kernel import BoundKernel

        kernel = self.kernel
        assert isinstance(kernel, BoundKernel)
        if self._cuda_facts is None:
            self._cuda_facts = CudaDeviceFacts.capture(kernel.env.device)
        return RenderAhead(
            kernel,
            configs,
            jobs=self._render_jobs(),
            timeout=self.settings.autotune_compile_timeout,
            facts=self._cuda_facts,
            result_path=self._next_precompile_result_path,
            log=self.log,
            already_compiled=kernel.has_compiled_config,
        )

    def benchmark(
        self,
        configs: list[Config],
        *,
        desc: str = "Benchmarking",
        raise_if_no_viable_config: bool = True,
    ) -> list[BenchmarkResult]:
        """Compile, precompile, validate, and time a batch of configs.

        With a precompile pool (``autotune_precompile`` set on a backend that
        supports it) the whole batch is generated first, precompiled in
        parallel, then timed. Without one (CuTe, or precompile disabled)
        nothing is gained by compiling ahead, so each config is compiled and
        timed in turn ("streaming"): results exist as soon as the first
        config finishes, and a budget that runs out mid-batch leaves at most
        one compiled-but-untimed config. A backend whose code generation is
        fork-safe (``supports_parallel_render``, CuTe) renders the streamed
        configs in forked workers ahead of their turn in fork mode (see
        :mod:`helion.autotuner.render_pool`); the parent then only loads the
        source, JIT-compiles at first call and times. Whatever a worker's
        outcome (source, error, timeout past ``autotune_compile_timeout``, no
        worker at all) the parent compiles the config in-process, loading
        the worker's source when there is one and generating it otherwise,
        so a worker never fails a config and the per-config control flow,
        including the distributed collectives, is the same on every rank.
        Workers that keep failing where the parent renders fine are given
        up for the rest of the search.

        The budget hook (``budget_exceeded_fn``) is asked before each
        config's compile and, with a pool, before each config's timing; a
        streamed config is timed right after its compile without a second
        check. Once the hook says stop, no further config is compiled or
        timed, and the configs the batch never got to come back as
        ``perf=inf, status="skipped"`` in positional order, so the caller
        still receives one ``BenchmarkResult`` per input config. The one
        exception: while the search has no successful measurement, the
        batch still attempts up to ``_BUDGET_EXEMPT_CANDIDATES`` more
        candidates (compiles; with a pool also that many timings of already
        compiled configs, and no further compile once something in the batch
        is compiled) so a budget that runs out early returns a config when
        one of those works instead of ``NoConfigFound``; a runtime failure
        or a compile failure uses one of those attempts. The cutoff is
        synchronized across the process group via ``_budget_step_synced``.
        Without a pool a compiler seed's source is known only once the seed
        itself is compiled: a representative of that source timed earlier
        is not treated as a seed (no timeout retry), the seed is when its
        turn comes.

        The precompile-phase wait (``PrecompileFuture.wait_for_all``)
        is not budget-aware: once the compile loop has queued
        precompile futures it drains all of them, so the effective
        wall-clock may overrun the configured budget by up to
        ``len(queued) * autotune_compile_timeout`` seconds.

        ``compile_time`` is the precompile future's wall time with a pool;
        when streaming it is the in-process ``compile_config`` wall time
        (code generation and module load, not a later JIT at first call).
        """
        self._no_viable_config_error = None
        self.batch_configs_started = 0
        self.batch_first_compile_error = None
        all_configs = configs
        compiled: dict[int, Callable[..., object]] = {}
        inline_compile_times: dict[int, float] = {}
        metrics = self._autotune_metrics
        # Candidates attempted past the budget hook's stop (see
        # ``_BUDGET_EXEMPT_CANDIDATES``) and the batch's first compile error,
        # updated from the nested loops below.
        exemptions = {"compile": 0, "timing": 0}
        first_error: dict[str, Exception | None] = {"value": None}

        # Initialize results before source deduplication so source aliases and
        # unhandled budget-tail entries can be filled positionally.
        results: list[BenchmarkResult] = [
            BenchmarkResult(
                config=c, fn=_unset_fn, perf=inf, status="error", compile_time=None
            )
            for c in all_configs
        ]

        deduplicate_sources = self._effective_source_dedup_enabled()
        backend = self.config_spec.backend if deduplicate_sources else None
        source_hashes: dict[int, str] = {}
        seen_sources = self._effective_source_hashes
        deduplicated_indices: set[int] = set()
        recorded_deduplicated_indices: set[int] = set()
        batch_repairs: dict[Config, BenchmarkResult] = {}
        cached_source_results = self._effective_source_results
        invalid_sources = self._invalid_effective_source_hashes

        streaming = not self._precompile_pool_enabled()
        units: list[list[int]] = (
            [[i] for i in range(len(all_configs))]
            if streaming
            else [list(range(len(all_configs)))]
        )
        # Forked workers render the streamed configs ahead of their compile.
        renders = self._render_ahead(all_configs) if streaming else None

        def record_final_result(
            index: int,
            config_id: str | None,
            compile_time: float | None,
        ) -> None:
            if config_id is None:
                return
            result = results[index]
            self.log.record_autotune_entry(
                AutotuneLogEntry(
                    generation=metrics.num_generations,
                    status=result.status,
                    perf_ms=result.perf if math.isfinite(result.perf) else None,
                    compile_time=(
                        None if result.status == "deduplicated" else compile_time
                    ),
                    config_id=config_id,
                    config=all_configs[index],
                    source_hash=source_hashes.get(index),
                    completed_at=result.completed_at,
                )
            )

        def record_started_result(
            index: int,
            config_id: str | None,
            compile_time: float | None,
        ) -> None:
            if config_id is None:
                return
            self.log.record_autotune_entry(
                AutotuneLogEntry(
                    generation=metrics.num_generations,
                    status="started",
                    perf_ms=None,
                    compile_time=compile_time,
                    config_id=config_id,
                    config=all_configs[index],
                    source_hash=source_hashes.get(index),
                )
            )

        def record_compile_failure_row(index: int) -> None:
            """Source ledger and trace rows for a config that never compiled."""
            config = all_configs[index]
            source_hash = None
            if deduplicate_sources:
                # No callable exists to carry generated-source identity, but
                # strict source ledgers still need an auditable terminal row.
                source_hash = _compile_config_failure_source_hash(config)
                if source_hash not in seen_sources:
                    seen_sources.add(source_hash)
                    metrics.num_unique_sources += 1
            config_id = (
                self.log.register_config(config)
                if deduplicate_sources or self.log.trace_enabled
                else None
            )
            if config_id is not None:
                self.log.record_autotune_entry(
                    AutotuneLogEntry(
                        generation=metrics.num_generations,
                        status="error",
                        perf_ms=None,
                        compile_time=None,
                        config_id=config_id,
                        config=config,
                        source_hash=source_hash,
                        completed_at=results[index].completed_at,
                    )
                )

        def compile_unit(unit: list[int]) -> None:
            """Compile (code generation, or the load of a worker's render) one
            unit's configs in this process."""
            for i in unit:
                # Past the budget, a few more candidates still compile while
                # the search has no successful measurement (streaming times
                # each at once; with a pool, only while nothing in the batch
                # is compiled yet, since compiled configs are timed first).
                protected = (
                    metrics.num_successful_candidate_measurements == 0
                    and exemptions["compile"] < _BUDGET_EXEMPT_CANDIDATES
                    and (streaming or not compiled)
                )
                exceeded, stop = self._budget_step_synced(protected=protected)
                if stop:
                    break
                if exceeded:
                    exemptions["compile"] += 1
                self.batch_configs_started += 1
                config = all_configs[i]
                # Whatever the worker's outcome, the config is compiled here
                # (its source loaded when the worker produced one, generated
                # otherwise), so the per-config control flow and the
                # collectives below never depend on worker outcomes.
                rendered = renders.take(i) if renders is not None else None
                worker_failed = rendered is not None and rendered.status in (
                    "error",
                    "timeout",
                )
                if rendered is not None and rendered.status == "ok":
                    assert renders is not None and rendered.code is not None
                    renders.kernel.prime_rendered_code(config, rendered.code)
                    self._render_fork_only_failures = 0
                elif rendered is not None and rendered.status == "timeout":
                    # Rendering in-process now costs the same render again; a
                    # timeout alone never fails the config.
                    self.log(
                        f"Render worker exceeded {rendered.timeout:.0f}s "
                        f"(autotune_compile_timeout) for {config!r}; rendering "
                        "in-process.",
                        level=logging.WARNING
                        if not self._render_timeout_warned
                        else logging.DEBUG,
                    )
                    self._render_timeout_warned = True
                elif rendered is not None and rendered.status == "error":
                    # The parent's render raises the real error, if any.
                    self.log.debug(
                        lambda config=config, rendered=rendered: (
                            f"Render worker failed for {config!r}; rendering "
                            f"in-process:\n{rendered.error}"
                        )
                    )
                # Defensive: if `capture_output().__enter__` itself raises, the
                # except handler below still needs `captured` bound.
                captured: list[str] = [""]
                compile_started = time.perf_counter()
                try:
                    with capture_output() as captured, _searching_candidates():
                        compiled[i] = self.kernel.compile_config(
                            config, allow_print=False
                        )
                    if rendered is not None and rendered.status == "ok":
                        assert rendered.elapsed is not None
                        inline_compile_times[i] = rendered.elapsed
                    else:
                        inline_compile_times[i] = time.perf_counter() - compile_started
                    if worker_failed and renders is not None:
                        # The worker failed where the parent did not: a
                        # fork-only failure (code generation reading tensor
                        # data, say). After a few in a row the workers only
                        # cost fork overhead; render in-process from here on.
                        assert rendered is not None
                        self._render_fork_only_failures += 1
                        if (
                            self._render_fork_only_failures
                            >= _RENDER_POOL_MAX_FORK_ONLY_FAILURES
                        ):
                            self._render_pool_unavailable = True
                            renders.close()
                            lines = (rendered.error or "").strip().splitlines()
                            reason = lines[-1] if lines else rendered.status
                            self.log.warning(
                                "Render workers fail for this kernel where the "
                                f"parent renders fine ({self._render_fork_only_failures}"
                                f" in a row, last: {reason}); rendering autotune "
                                "candidates in-process."
                            )
                except Exception as e:
                    if worker_failed:
                        # The parent fails too: a real failure of the config.
                        self._render_fork_only_failures = 0
                    results[i] = results[i]._replace(completed_at=time.perf_counter())
                    no_viable_config = (
                        not compiled
                        and i == len(all_configs) - 1
                        and metrics.num_successful_candidate_measurements == 0
                    )
                    self._record_compile_failure(config, e)
                    record_compile_failure_row(i)
                    maybe_dump_triton_failure(
                        self.kernel, config, e, captured_output=captured[0] or None
                    )
                    self.log.warning(
                        "Skipping config that failed to compile: "
                        f"{self.kernel.format_kernel_decorator(config, self.settings)}",
                        exc_info=True,
                    )
                    if first_error["value"] is None:
                        first_error["value"] = e
                    if no_viable_config:
                        if raise_if_no_viable_config:
                            raise
                        self._no_viable_config_error = e

        def deduplicate_unit(unit: list[int]) -> dict[int, Callable[..., object]]:
            """Split one unit's compiled configs into unique sources and aliases."""
            unique_compiled: dict[int, Callable[..., object]] = {}
            for index in unit:
                fn = compiled.get(index)
                if fn is None:
                    continue
                source_hash = (
                    backend.generated_source_hash(fn)
                    if deduplicate_sources and backend is not None
                    else None
                )
                if source_hash is None:
                    unique_compiled[index] = fn
                    continue
                source_hashes[index] = source_hash
                if self._is_compiler_seed_config(all_configs[index]):
                    self._compiler_seed_source_hashes.add(source_hash)
                if source_hash not in seen_sources:
                    seen_sources.add(source_hash)
                    metrics.num_unique_sources += 1
                cached_result = cached_source_results.get(source_hash)
                if cached_result is not None:
                    results[index] = BenchmarkResult(
                        config=all_configs[index],
                        fn=cached_result.fn,
                        perf=cached_result.perf,
                        status="deduplicated",
                        compile_time=None,
                        completed_at=time.perf_counter(),
                    )
                    deduplicated_indices.add(index)
                    metrics.num_source_deduplications += 1
                    continue
                unique_compiled[index] = fn
            return unique_compiled

        def benchmark_unit(unique_compiled: dict[int, Callable[..., object]]) -> None:
            """Precompile (with a pool) and time one unit's unique configs."""
            fns = list(unique_compiled.values())
            valid_indices = list(unique_compiled.keys())
            unit_configs = [all_configs[i] for i in valid_indices]
            futures: list[PrecompileFuture] | None = None

            # Precompile phase
            if self._precompile_pool_enabled():
                futures = list(
                    starmap(
                        self._create_precompile_future,
                        zip(unit_configs, fns, strict=True),
                    )
                )
                precompile_desc = (
                    f"{desc} precompiling"
                    if self.settings.autotune_progress_bar
                    else None
                )
                is_workings = PrecompileFuture.wait_for_all(
                    futures, desc=precompile_desc
                )
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
                is_workings = [True] * len(unit_configs)
                precompile_status = ["ok"] * len(unit_configs)

            # Benchmark loop with progress reporting (per config with a pool;
            # the streaming path reports per unit instead).
            iterator = iter_with_progress(
                enumerate(zip(fns, is_workings, precompile_status, strict=True)),
                total=len(unit_configs),
                description=f"{desc} exploring neighbors",
                enabled=self.settings.autotune_progress_bar and not streaming,
            )
            for index, (fn, is_working, reason) in iterator:
                # A streamed config was compiled moments ago under this unit's
                # budget check; its timing is the cheap half of the unit, so
                # the budget is rechecked at the next unit instead. With a
                # pool, a few already compiled configs are still timed past
                # the budget while the search has no successful measurement.
                if not streaming:
                    protected = (
                        metrics.num_successful_candidate_measurements == 0
                        and exemptions["timing"] < _BUDGET_EXEMPT_CANDIDATES
                    )
                    exceeded, stop = self._budget_step_synced(protected=protected)
                    if stop:
                        break
                    if exceeded:
                        exemptions["timing"] += 1
                config = unit_configs[index]
                result_index = valid_indices[index]
                if futures is not None:
                    future = futures[index]
                    compile_time = (
                        future.elapsed
                        if future.process is not None and future.started
                        else None
                    )
                else:
                    compile_time = inline_compile_times.get(result_index)
                status: Literal[
                    "ok",
                    "error",
                    "timeout",
                    "peer_compilation_fail",
                    "filtered",
                    "deduplicated",
                    "accuracy_error",
                    "source_rejected",
                ]
                # config_id is None when no log sink is active. A started row
                # is emitted only for an actual benchmark attempt; source
                # aliases that reuse a completed result get a standalone
                # deduplicated row.
                config_id = self.log.register_config(config)
                source_hash = source_hashes.get(result_index)
                current_accuracy_failure = False
                source_result = (
                    cached_source_results.get(source_hash)
                    if source_hash is not None
                    else None
                )
                if source_hash is not None and source_hash in invalid_sources:
                    status = "source_rejected"
                    results[result_index] = BenchmarkResult(
                        config=config,
                        fn=fn,
                        perf=inf,
                        status=status,
                        compile_time=compile_time,
                        completed_at=time.perf_counter(),
                    )
                    metrics.num_source_deduplications += 1
                    record_final_result(result_index, config_id, compile_time)
                elif source_result is not None:
                    status = "deduplicated"
                    results[result_index] = BenchmarkResult(
                        config=config,
                        fn=source_result.fn,
                        perf=source_result.perf,
                        status=status,
                        compile_time=None,
                        completed_at=time.perf_counter(),
                    )
                    deduplicated_indices.add(result_index)
                    metrics.num_source_deduplications += 1
                    self.log.debug(
                        lambda config=config, source_hash=source_hash: (
                            f"Reusing effective generated source {source_hash} "
                            f"for {config!r}"
                        )
                    )
                    record_final_result(result_index, config_id, compile_time)
                    recorded_deduplicated_indices.add(result_index)
                elif all(
                    all_gather_object(
                        is_working,
                        process_group_name=self.kernel.env.process_group_name,
                    )
                ):
                    record_started_result(result_index, config_id, compile_time)
                    self._last_benchmark_failure_status = None
                    accuracy_failure_count = len(self._accuracy_failure_config_ids)
                    compiler_seed_source = self._is_compiler_seed_config(config) or (
                        source_hash is not None
                        and source_hash in self._compiler_seed_source_hashes
                    )
                    if compiler_seed_source and source_hash is not None:
                        perf = self._benchmark_function(
                            config,
                            fn,
                            effective_source_hash=source_hash,
                        )
                    else:
                        perf = self._benchmark_function(config, fn)
                    current_accuracy_failure = (
                        deduplicate_sources
                        and len(self._accuracy_failure_config_ids)
                        > accuracy_failure_count
                    )
                    status = (
                        "ok"
                        if math.isfinite(perf)
                        else (
                            "accuracy_error"
                            if current_accuracy_failure
                            else self._last_benchmark_failure_status or "error"
                        )
                    )
                    results[result_index] = BenchmarkResult(
                        config=config,
                        fn=fn,
                        perf=perf,
                        status=status,
                        compile_time=compile_time,
                        completed_at=time.perf_counter(),
                    )
                    # Keep the actual terminal outcome even when a later source
                    # alias repairs this candidate. The repair is logged as a
                    # separate deduplication event below.
                    record_final_result(result_index, config_id, compile_time)
                else:
                    status = "timeout" if reason == "timeout" else "error"
                    if is_working:
                        status = "peer_compilation_fail"
                    results[result_index] = BenchmarkResult(
                        config=config,
                        fn=fn,
                        perf=inf,
                        status=status,
                        compile_time=compile_time,
                        completed_at=time.perf_counter(),
                    )
                    record_final_result(result_index, config_id, compile_time)
                result = results[result_index]
                if result.status == "ok" and math.isfinite(result.perf):
                    metrics.num_successful_candidate_measurements += 1
                if (
                    source_hash is not None
                    and result.status == "ok"
                    and math.isfinite(result.perf)
                ):
                    cached_source_results[source_hash] = result
                    batch_repairs.update(
                        self._resolve_effective_source_failures(source_hash, result)
                    )
                if source_hash is not None and current_accuracy_failure:
                    invalid_sources.add(source_hash)
                    cached_source_results.pop(source_hash, None)
                    self._pending_effective_source_failures.pop(source_hash, None)
                elif source_hash is not None and result.status in (
                    "error",
                    "timeout",
                    "peer_compilation_fail",
                ):
                    self._remember_effective_source_failure(
                        source_hash, config, config_id, compile_time
                    )
            # A budget cut leaves the loop early; end the bar here.
            iterator.close()

        # Streaming reports progress per unit (the pool path per config inside
        # ``benchmark_unit``); the bar is closed below whichever way the loop
        # ends.
        unit_progress = iter_with_progress(
            units,
            total=len(units),
            description=f"{desc} exploring neighbors",
            enabled=self.settings.autotune_progress_bar and streaming,
        )
        try:
            for unit in unit_progress:
                compile_unit(unit)
                benchmark_unit(deduplicate_unit(unit))
                if streaming and not any(results[i].completed_at is None for i in unit):
                    continue
                # The budget stopped this unit short (nothing is compiled
                # ahead when streaming, so the rest of the batch is skipped
                # too); with a pool there is only one unit.
                break
        finally:
            unit_progress.close()
            if renders is not None:
                # Renders still in flight are for configs this batch will not
                # compile (budget cut or an error); their workers go.
                renders.close()
                if renders.unavailable:
                    self._render_pool_unavailable = True

        # A budget cut can end the batch before its last config, where the
        # compile error of an all-failing batch would have surfaced; the first
        # error still does, as the no-viable-config error and in summaries.
        self.batch_first_compile_error = first_error["value"]
        if (
            first_error["value"] is not None
            and not compiled
            and metrics.num_successful_candidate_measurements == 0
            and self._no_viable_config_error is None
        ):
            if raise_if_no_viable_config:
                raise first_error["value"]
            self._no_viable_config_error = first_error["value"]

        # Repairs discovered in this batch update its positional results. The
        # provider also retains them for the search to apply to members and
        # surrogate targets produced by earlier benchmark batches.
        for index, result in enumerate(results):
            repair = batch_repairs.get(result.config)
            if repair is not None and not math.isfinite(result.perf):
                results[index] = repair
        self._record_measured_configs(results)

        for index in deduplicated_indices - recorded_deduplicated_indices:
            result = results[index]
            config = all_configs[index]
            source_hash = source_hashes[index]
            if result.status == "deduplicated":
                self.log.debug(
                    lambda config=config, source_hash=source_hash: (
                        f"Reusing effective generated source {source_hash} "
                        f"for {config!r}"
                    )
                )
            config_id = self.log.register_config(config)
            if config_id is not None:
                self.log.record_autotune_entry(
                    AutotuneLogEntry(
                        generation=metrics.num_generations,
                        status=result.status,
                        perf_ms=result.perf if math.isfinite(result.perf) else None,
                        compile_time=None,
                        config_id=config_id,
                        config=config,
                        source_hash=source_hash,
                        completed_at=result.completed_at,
                    )
                )
        # Configs the budget cutoff never got to (never compiled, or compiled
        # but not timed) become terminal "skipped" rows when the batch ends.
        completed_at = time.perf_counter()
        return [
            result
            if result.completed_at is not None
            else result._replace(status="skipped", completed_at=completed_at)
            for result in results
        ]

    def _clear_jit_fast_path_caches(self, fn: CompiledConfig) -> None:
        """Clear Triton JIT fast-path caches for this generated wrapper.

        Without this, tensors passed to the companion Triton JIT function can
        remain pinned in GPU memory by its _last_call cache across config
        benchmarks.
        """
        clear_jit_fast_path_caches(fn, self.log)

    def _benchmark_function(
        self,
        config: Config,
        fn: CompiledConfig,
        *,
        effective_source_hash: str | None = None,
    ) -> float:
        """Benchmark a single compiled function.  Returns time in ms or inf."""
        self._autotune_metrics.num_configs_tested += 1
        self.log.debug(lambda: f"Running benchmark for {config!r}")

        if self._subprocess_benchmark_enabled():
            if effective_source_hash is not None:
                result = self._benchmark_function_subprocess(
                    config,
                    fn,
                    effective_source_hash=effective_source_hash,
                )
            else:
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
                if self.settings.autotune_accuracy_check:
                    # The run the accuracy check judges must not find a
                    # previous run's output in its recycled buffers.
                    poison_cached_device_memory(self.kernel.env.device)
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
                with _searching_candidates():
                    benchmark_function = self.kernel.bench_compile_config(
                        config, allow_print=False
                    )
                benchmark_function(*working_args)  # warmup benchmark kernel

                t1 = time.perf_counter()
                _backend = getattr(getattr(self, "config_spec", None), "backend", None)
                benchmark_runner = (
                    _backend.get_do_bench() if _backend is not None else None
                ) or do_bench
                benchmark_callable = functools.partial(
                    benchmark_function, *working_args
                )
                if benchmark_runner in (do_bench, do_bench_generic):
                    # The exact callable ran immediately above.
                    res = benchmark_runner(
                        benchmark_callable,
                        return_mode="median",
                        warmup=1,
                        rep=50,
                        process_group_name=self.kernel.env.process_group_name,
                        probe_long_kernel=self._probe_long_kernel(),
                        pre_warmed=True,
                    )
                else:
                    res = benchmark_runner(
                        benchmark_callable,
                        return_mode="median",
                        warmup=1,
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
                    isolation=self._in_process_benchmark_reason(),
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

            self._record_compile_failure(config, e)
            return inf
        finally:
            self._clear_jit_fast_path_caches(fn)

    def _benchmark_function_subprocess(
        self,
        config: Config,
        fn: CompiledConfig,
        *,
        effective_source_hash: str | None = None,
    ) -> float | None:
        """Benchmark ``fn`` in a long-lived spawn subprocess with a per-call
        timeout. Returns the measured latency in ms, ``inf`` for a failure
        we classified and handled, or ``None`` if the subprocess path cannot
        handle this config and the caller should fall back to in-process. An
        eligible timed-out compiler seed, or the search's default config, gets
        one budget-gated retry with a fixed repetition count
        (``_timeout_retry_repetitions``). A candidate past the worker's
        compile deadline is a compile failure, never retried.
        """
        retry_repetitions = self._timeout_retry_repetitions(
            config, effective_source_hash
        )
        retried_repetitions: int | None = None
        try:
            try:
                latency = self._run_subprocess_benchmark_job(fn, warmup=1, rep=50)
            except BenchmarkTimeout:
                if (
                    retry_repetitions is None
                    or not self._claim_compiler_seed_timeout_retry(
                        config, effective_source_hash
                    )
                ):
                    raise
                retried_repetitions = retry_repetitions
                self.log.debug(
                    lambda: (
                        "Retrying timed-out "
                        f"{self._timed_out_config_role(config) or 'candidate'} "
                        f"with {retry_repetitions} measured "
                        f"repetitions: {config!r}"
                    )
                )
                latency = self._run_subprocess_benchmark_job(
                    fn,
                    warmup=1,
                    rep=50,
                    fixed_repetitions=retry_repetitions,
                )
            if latency is None:
                return None
        except BenchmarkWorkerUnkillable:
            raise
        except BenchmarkCompileTimeout as e:
            self._record_compile_timeout(config, e)
            return inf
        except BenchmarkSubprocessError as e:
            # Timeout or unexpected worker exit; skip config and continue.
            self.log.warning(f"Benchmark subprocess failed for {config!r}: {e}")
            if isinstance(e, BenchmarkTimeout):
                self._warn_timed_out_role(
                    config, e, retried_repetitions=retried_repetitions
                )
            self._record_worker_failure(
                config,
                "timeout" if isinstance(e, BenchmarkTimeout) else "error",
            )
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
                self._record_compile_failure(config, e)
                return inf
            self.log.debug(
                f"Benchmark subprocess raised for {config!r}: {type(e).__name__}: {e}"
            )
            self._record_compile_failure(config, e)
            return inf

        if self.settings.autotune_accuracy_check:
            try:
                accuracy_result = self._run_subprocess_accuracy_check_job(fn)
            except BenchmarkWorkerUnkillable:
                raise
            except BenchmarkSubprocessError as e:
                self.log.warning(
                    f"Accuracy check subprocess failed for {config!r}: {e}"
                )
                self._record_worker_failure(
                    config,
                    "timeout" if isinstance(e, BenchmarkTimeout) else "error",
                )
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
                    self._record_compile_failure(config, e)
                    return inf
                self.log.debug(
                    f"Accuracy check subprocess raised for {config!r}: "
                    f"{type(e).__name__}: {e}"
                )
                self._record_compile_failure(config, e)
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
            # worker; validate in-process instead, on fresh copies of the
            # arguments the kernel writes into.
            if len(self.mutated_arg_indices) > 0:
                working_args = _clone_args(
                    self.args,
                    self.kernel.env.process_group_name,
                    idx_to_clone=self.mutated_arg_indices,
                )
            else:
                working_args = self.args
            try:
                with capture_output():
                    output = fn(*working_args)
                    synchronize_device()
                if not self._validate_against_baseline(config, output, working_args):
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
                        isolation=self._in_process_benchmark_reason(),
                    ) from e
                self.log.debug(
                    f"Accuracy check raised for {config!r}: {type(e).__name__}: {e}"
                )
                self._record_compile_failure(config, e)
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

        if (
            len(self.mutated_arg_indices) > 0
            and self._precompile_baseline_post_args_path is None
        ):
            return None
        job = AccuracyCheckJob(
            fn_spec=fn_spec,
            args_path=self._precompile_args_path,
            baseline_path=self._precompile_baseline_path,
            atol=self._effective_atol,
            rtol=self._effective_rtol,
            scale_atol=self._scale_atol,
            fp8_accuracy=self._fp8_accuracy,
            mutated_arg_indices=tuple(self.mutated_arg_indices),
            baseline_post_args_path=self._precompile_baseline_post_args_path,
            check_input_accuracy=os.getenv("CHECK_INPUT_ACCURACY", "1") == "1",
        )
        try:
            return cast(
                "AccuracyCheckResult",
                self._benchmark_worker.run(
                    job,
                    timeout=float(self.settings.autotune_benchmark_timeout),
                ),
            )
        except CompiledFunctionLoadError as error:
            self._disable_unloadable_benchmark_worker(error)
            return None

    def _run_subprocess_benchmark_job(
        self,
        fn: CompiledConfig,
        *,
        warmup: int,
        rep: int,
        fixed_repetitions: int | None = None,
        args_slot: int | None = None,
        timeout: float | None = None,
        l2_flush_mode: str | None = None,
    ) -> float | None:
        if self._precompile_args_path is None or self._subprocess_wrapper_unloadable:
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
            probe_long_kernel=self._probe_long_kernel(),
            fixed_repetitions=fixed_repetitions,
            args_slot=args_slot,
            mutated_arg_indices=tuple(self.mutated_arg_indices),
            l2_flush_mode=l2_flush_mode,
        )
        if timeout is None:
            timeout = float(self.settings.autotune_benchmark_timeout)
        try:
            return float(
                self._benchmark_worker.run(
                    job,
                    timeout=timeout,
                    compile_timeout=self._worker_compile_timeout(),
                )
            )
        except CompiledFunctionLoadError as error:
            self._disable_unloadable_benchmark_worker(error)
            return None

    def _worker_compile_timeout(self) -> float:
        """Deadline for a candidate's compile and first call in the benchmark
        worker (``autotune_compile_timeout``), never below the benchmark
        timeout: before the phases were separated the whole job had that
        long, and a fresh worker's spawn, imports and argument load count
        against this phase too."""
        return max(
            float(self.settings.autotune_compile_timeout),
            float(self.settings.autotune_benchmark_timeout),
        )

    def _disable_unloadable_benchmark_worker(
        self, error: CompiledFunctionLoadError
    ) -> None:
        # A generated wrapper can depend on a source module that exists only in
        # the parent process (for example a module loaded under a synthetic
        # namespace). This says nothing about whether the config itself is valid,
        # so let callers use their existing in-process fallback and stop sending
        # later candidates through the incompatible worker.
        self._subprocess_wrapper_unloadable = True
        if self._benchmark_worker is not None:
            self._benchmark_worker.shutdown()
            self._benchmark_worker = None
        self.log.debug(
            f"Benchmark worker could not load the generated wrapper; "
            f"falling back in-process: {error}"
        )

    def steady_window_benchmark_available(self) -> bool:
        """Whether finalists can be timed in the killable benchmark worker."""
        return (
            self._subprocess_benchmark_enabled()
            and self.settings.autotune_benchmark_fn is None
        )

    def private_argument_copies_affordable(self, count: int) -> bool:
        """Whether ``count`` private copies of the kernel arguments fit on the
        device next to the originals (CUDA tensors only; other devices share)."""
        tensors = [
            leaf
            for leaf in tree_flatten(self.args)[0]
            if isinstance(leaf, torch.Tensor) and leaf.device.type != "cpu"
        ]
        if not tensors:
            return True
        if not all(tensor.is_cuda for tensor in tensors):
            return False
        total_bytes = 0
        seen_storages: set[int] = set()
        for tensor in tensors:
            storage = tensor.untyped_storage()
            if storage.data_ptr() in seen_storages:
                continue
            seen_storages.add(storage.data_ptr())
            total_bytes += storage.nbytes()
        free_bytes, _total = torch.cuda.mem_get_info(tensors[0].device)
        return (
            count * total_bytes <= _PRIVATE_ARGUMENT_COPIES_MEMORY_FRACTION * free_bytes
        )

    def release_private_argument_copies(self) -> None:
        """Free the finalists' private argument copies held by the worker.

        Best-effort: the selection that used the copies is complete, so a
        release that fails is logged and left for the worker's next job. A
        worker that times out or dies on the release is killed (the copies
        die with it) and respawns on that job; a job that raises in the
        worker (a sticky CUDA error from a finalist's kernel, surfaced by
        the cache release as a ``RuntimeError``) has dropped the copies
        from the cache before the error, and ``BenchmarkWorker.run`` kills
        the worker when the error is unrecoverable. Only those two kinds
        are logged (the ``BenchmarkSubprocessError`` family and
        ``RuntimeError``); a worker that survived its kill still raises
        ``BenchmarkWorkerUnkillable`` and any other error propagates.
        """
        if self._benchmark_worker is None or not self._benchmark_worker.alive():
            return
        try:
            self._benchmark_worker.run(
                release_private_kernel_args,
                timeout=float(self.settings.autotune_benchmark_timeout),
            )
        except BenchmarkWorkerUnkillable:
            raise
        except (RuntimeError, BenchmarkSubprocessError) as error:
            self.log.warning(
                "Releasing the final selection's private argument copies failed; "
                f"the benchmark worker's next job proceeds without them: {error!r}"
            )

    def benchmark_isolated(
        self,
        fns: list[Callable[..., object]],
        *,
        warmup: int,
        rep: int,
        desc: str = "Benchmarking",
        fresh_process: bool = False,
        args_slots: Sequence[int | None] | None = None,
        timeout: float | None = None,
        l2_flush_mode: str | None = None,
    ) -> list[IsolatedBenchmarkTiming] | None:
        if not self._subprocess_benchmark_enabled():
            return None
        if self.settings.autotune_benchmark_fn is not None:
            return None
        if args_slots is None:
            args_slots = [None] * len(fns)

        timings: list[IsolatedBenchmarkTiming] = []
        for index, (fn, args_slot) in enumerate(zip(fns, args_slots, strict=True)):
            if fresh_process and self._benchmark_worker is not None:
                self._shutdown_isolated_worker(index - 1 if index else None)
            try:
                try:
                    timing = self._run_subprocess_benchmark_job(
                        cast("CompiledConfig", fn),
                        warmup=warmup,
                        rep=rep,
                        args_slot=args_slot,
                        timeout=timeout,
                        l2_flush_mode=l2_flush_mode,
                    )
                except BenchmarkWorkerUnkillable as error:
                    # the worker that could not be reaped ran this candidate
                    error.fn_index = index
                    raise
                except BenchmarkTimeout as e:
                    self.log.warning(f"{self._isolated_failure_subject(fn, desc)}: {e}")
                    self._autotune_metrics.num_isolated_rebenchmark_timeouts += 1
                    timings.append(IsolatedBenchmarkFailure("timeout"))
                    continue
                except BenchmarkSubprocessError as e:
                    # A compile deadline or a dead worker: the candidate was
                    # measured before, so its earlier timing is kept.
                    self.log.warning(f"{self._isolated_failure_subject(fn, desc)}: {e}")
                    timing = None
                except Exception as e:
                    e.__traceback__ = None
                    if match_unrecoverable_runtime_error(e):
                        self.log.warning(f"{desc} sticky CUDA error skipped: {e}")
                        # The confirmation re-ran a previously accepted candidate in
                        # an isolated worker; a sticky CUDA error means that config is
                        # still unsafe, so remove it from contention.
                        timings.append(IsolatedBenchmarkFailure("error"))
                        continue
                    self.log.debug(f"{desc} subprocess raised: {type(e).__name__}: {e}")
                    timing = None
                # A wrapper-load failure disables the worker because later
                # candidates may depend on the same unavailable source module.
                # Treat the whole isolated batch as unavailable so the caller
                # rebenchmarks every finalist in-process instead of mixing partial
                # fresh timings with stale population measurements.
                if self._subprocess_wrapper_unloadable:
                    return None
                timings.append(None if timing is None else float(timing))
            finally:
                if fresh_process and self._benchmark_worker is not None:
                    self._shutdown_isolated_worker(index)
        return timings

    def _shutdown_isolated_worker(self, fn_index: int | None) -> None:
        """Shut the isolated worker down; a worker that cannot be reaped names
        the batch position of the candidate it last ran."""
        assert self._benchmark_worker is not None
        try:
            self._benchmark_worker.shutdown()
        except BenchmarkWorkerUnkillable as error:
            error.fn_index = fn_index
            raise
        finally:
            self._benchmark_worker = None


class MultiShapeBenchmarkProvider(BenchmarkProvider):
    """Compose ordinary local providers and expose one scalar per config."""

    mutated_arg_indices: Sequence[int] = ()
    # Candidates attempted past the budget while nothing was measured yet
    # (see ``_BUDGET_EXEMPT_CANDIDATES``), across this provider's batches.
    _budget_exempt_used: int = 0

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
        self._collect_effective_source_repairs = config_spec.cute_flash_search_enabled
        self._original_configs_by_materialized_key: dict[str, list[Config]] = {}
        self._anchor_fns_by_materialized_key: dict[str, Callable[..., object]] = {}
        self._effective_source_repairs: dict[Config, BenchmarkResult] = {}
        child_log = copy.copy(log)
        child_log._log_sink = None
        child_log._trace_sink = None
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
            self._cleanup_children()
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

    def set_compiler_seed_configs(self, configs: Sequence[Config]) -> None:
        for child in self.children:
            child.set_compiler_seed_configs(configs)

    def set_config_roles(self, roles: Mapping[Config, str]) -> None:
        for child in self.children:
            child.set_config_roles(roles)

    def unmeasured_config_values(self) -> list[UnmeasuredConfigValue]:
        # A value measured on any shape was reachable; the failure counts of
        # the shapes add up, the first shape's first error is kept.
        measured: set[tuple[str, object]] = set()
        failures: dict[tuple[str, object], tuple[int, str]] = {}
        for child in self.children:
            measured.update(child._measured_config_values or set())
            for pair, (failed, first) in (child._compile_failure_values or {}).items():
                total, kept = failures.get(pair, (0, first))
                failures[pair] = (total + failed, kept)
        return _unmeasured_config_values(measured, failures)

    def take_effective_source_repairs(self) -> dict[Config, BenchmarkResult]:
        repairs = self._effective_source_repairs
        self._effective_source_repairs = {}
        return repairs

    def take_no_viable_config_error(self) -> Exception | None:
        # Mirror _benchmark_child: a failure that would have been skipped for
        # its shape must not surface as the search's fatal compile error.
        result: Exception | None = None
        for child in self.children:
            error = child.take_no_viable_config_error()
            if (
                result is None
                and error is not None
                and not self._is_skippable_child_failure(child, error)
            ):
                result = error
        return result

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
            self._cleanup_children()
            if f"arg_sets[{case_index}]" in str(error):
                raise
            raise exc.AutotuneError(
                f"Failed to set up multi-shape autotune arg_sets[{case_index}]: {error}"
            ) from error

    def cleanup(self) -> None:
        try:
            self._cleanup_children()
        finally:
            self._original_configs_by_materialized_key.clear()
            self._anchor_fns_by_materialized_key.clear()
            self._effective_source_repairs.clear()
            self._budget_exempt_used = 0
            self.budget_exceeded_fn = _never_exceeded

    def _cleanup_children(self) -> None:
        first_error: Exception | None = None
        fatal_error: BenchmarkWorkerUnkillable | None = None
        for child in reversed(self.children):
            try:
                child.cleanup()
            except BenchmarkWorkerUnkillable as error:
                if fatal_error is None:
                    fatal_error = error
            except Exception as error:
                if first_error is None:
                    first_error = error
        if fatal_error is not None:
            raise fatal_error
        if first_error is not None:
            raise first_error

    def _measure_reference(
        self, child: LocalBenchmarkProvider, case_index: int
    ) -> float:
        if self.args.relative_to == "default":
            result = child.benchmark(
                [child.config_spec.default_config()],
                desc="Benchmarking default reference",
            )[0]
            if (
                not _benchmark_status_succeeded(result.status)
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
        raise_if_no_viable_config: bool = True,
    ) -> list[BenchmarkResult]:
        return self._benchmark(
            configs,
            desc=desc,
            record_results=True,
            raise_if_no_viable_config=raise_if_no_viable_config,
        )

    def _benchmark(
        self,
        configs: list[Config],
        *,
        desc: str,
        record_results: bool,
        check_budget: bool = True,
        raise_if_no_viable_config: bool = True,
    ) -> list[BenchmarkResult]:
        if not configs:
            return []
        if check_budget and self.budget_exceeded_fn():
            # Like the single-shape provider: past the budget a few more
            # candidates are still attempted while nothing has been measured.
            exempt = 0
            if self._autotune_metrics.num_successful_candidate_measurements == 0:
                exempt = max(0, _BUDGET_EXEMPT_CANDIDATES - self._budget_exempt_used)
            exempt = min(exempt, len(configs))
            self._budget_exempt_used += exempt
            attempted = (
                self._benchmark(
                    configs[:exempt],
                    desc=desc,
                    record_results=record_results,
                    check_budget=False,
                    raise_if_no_viable_config=raise_if_no_viable_config,
                )
                if exempt
                else []
            )
            return [
                *attempted,
                *(
                    BenchmarkResult(
                        config, _unset_fn, inf, "skipped", None, time.perf_counter()
                    )
                    for config in configs[exempt:]
                ),
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
                results[config_index] = results[config_index]._replace(
                    completed_at=time.perf_counter()
                )
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

        child_metric_snapshots = [
            (
                len(child._accuracy_failure_config_ids),
                len(child._compile_failure_config_ids),
                len(child._worker_failure_config_ids),
                child._autotune_metrics.num_unique_sources,
                child._autotune_metrics.num_source_deduplications,
            )
            for child in self.children
        ]
        child_results = [
            self._benchmark_child(
                child,
                materialized,
                desc=f"{desc} shape {index + 1}",
                case_index=index,
                raise_if_no_viable_config=raise_if_no_viable_config,
            )
            for index, child in enumerate(self.children)
        ]
        # Child retries can repair an aggregate result from an earlier batch.
        # Consume those repairs before the current row replaces the stored
        # multi-shape measurement for the same materialized config.
        if self._collect_effective_source_repairs:
            self._collect_child_effective_source_repairs()
        if record_results:
            accuracy_failure_ids: set[int] = set()
            compile_failure_ids: set[int] = set()
            worker_failure_ids: set[int] = set()
            for child, (
                before_accuracy,
                before_compile,
                before_worker,
                before_unique_sources,
                before_source_deduplications,
            ) in zip(self.children, child_metric_snapshots, strict=True):
                accuracy_failure_ids.update(
                    child._accuracy_failure_config_ids[before_accuracy:]
                )
                compile_failure_ids.update(
                    child._compile_failure_config_ids[before_compile:]
                )
                worker_failure_ids.update(
                    child._worker_failure_config_ids[before_worker:]
                )
                self._autotune_metrics.num_unique_sources += (
                    child._autotune_metrics.num_unique_sources - before_unique_sources
                )
                self._autotune_metrics.num_source_deduplications += (
                    child._autotune_metrics.num_source_deduplications
                    - before_source_deduplications
                )
            self._autotune_metrics.num_accuracy_failures += len(accuracy_failure_ids)
            self._autotune_metrics.num_compile_failures += len(compile_failure_ids)
            self._autotune_metrics.num_worker_failures += len(worker_failure_ids)
        for child_config_index, config_index in enumerate(valid_indices):
            original = configs[config_index]
            row = [child[child_config_index] for child in child_results]
            timings = [result.perf for result in row]
            valid = all(
                _benchmark_status_succeeded(result.status)
                and math.isfinite(result.perf)
                and result.perf > 0
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
                # A joint trial completes when its last shape completes, not
                # when the provider finishes aggregating the entire batch.
                completed_at=(
                    max(cast("float", result.completed_at) for result in row)
                    if all(result.completed_at is not None for result in row)
                    else time.perf_counter()
                ),
                per_shape=timing_tuple,
                per_shape_statuses=statuses,
            )
            results[config_index] = result
            if self._collect_effective_source_repairs and not math.isfinite(perf):
                measurement_key = materialized_keys[child_config_index]
                originals = self._original_configs_by_materialized_key.setdefault(
                    measurement_key, []
                )
                if original not in originals:
                    originals.append(copy.deepcopy(original))
                self._anchor_fns_by_materialized_key[measurement_key] = anchor_fn
            if math.isfinite(perf):
                self.args.found_valid_config = True
            if record_results:
                self._record_aggregate_result(
                    executed_configs[child_config_index],
                    result,
                    timings=timings,
                )
        return results

    def _collect_child_effective_source_repairs(self) -> None:
        """Promote child source repairs once every shape has a finite timing."""
        for case_index, child in enumerate(self.children):
            for materialized, repair in child.take_effective_source_repairs().items():
                key = repr(materialized)
                measurement = self.args.measurements.get(key)
                if measurement is None:
                    continue
                if case_index == 0:
                    self._anchor_fns_by_materialized_key[key] = repair.fn
                timings, previous_perf, statuses = measurement
                repaired_timings = list(timings)
                repaired_statuses = list(statuses)
                repaired_timings[case_index] = repair.perf
                repaired_statuses[case_index] = repair.status
                valid = all(
                    _benchmark_status_succeeded(status)
                    and math.isfinite(timing)
                    and timing > 0
                    for timing, status in zip(
                        repaired_timings, repaired_statuses, strict=True
                    )
                )
                perf = (
                    _aggregate_multi_shape_timings(
                        repaired_timings,
                        aggregation=self.args.aggregation,
                        references=self.args.reference_latencies,
                    )
                    if valid
                    else inf
                )
                self.args.measurements[key] = (
                    tuple(repaired_timings),
                    perf,
                    tuple(repaired_statuses),
                )
                if math.isfinite(previous_perf) or not math.isfinite(perf):
                    continue
                originals = self._original_configs_by_materialized_key.get(key)
                anchor_fn = self._anchor_fns_by_materialized_key.get(key)
                if not originals or anchor_fn is None:
                    continue
                for original in originals:
                    self._effective_source_repairs[original] = BenchmarkResult(
                        config=original,
                        fn=anchor_fn,
                        perf=perf,
                        status="deduplicated",
                        compile_time=None,
                        completed_at=repair.completed_at,
                        per_shape=tuple(repaired_timings),
                        per_shape_statuses=tuple(repaired_statuses),
                    )
                self._original_configs_by_materialized_key.pop(key, None)
                self._anchor_fns_by_materialized_key.pop(key, None)
                self.args.found_valid_config = True

    def _benchmark_child(
        self,
        child: LocalBenchmarkProvider,
        configs: list[Config],
        *,
        desc: str,
        case_index: int,
        raise_if_no_viable_config: bool = True,
    ) -> list[BenchmarkResult]:
        try:
            return child.benchmark(
                configs,
                desc=desc,
                raise_if_no_viable_config=raise_if_no_viable_config,
            )
        except Exception as error:
            if not self._is_skippable_child_failure(child, error):
                raise
            self.log.debug(
                "Skipping all configs for "
                f"arg_sets[{case_index}] after {type(error).__name__}: {error}"
            )
            return [
                BenchmarkResult(
                    config, _unset_fn, inf, "error", None, time.perf_counter()
                )
                for config in configs
            ]

    @staticmethod
    def _is_skippable_child_failure(
        child: LocalBenchmarkProvider, error: Exception
    ) -> bool:
        if isinstance(error, BenchmarkWorkerUnkillable):
            return False
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
        if result.status == "ok" and math.isfinite(result.perf):
            self._autotune_metrics.num_successful_candidate_measurements += 1
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
                objective=result.perf,
                objective_unit="ratio" if self.args.relative_to is not None else "ms",
                completed_at=result.completed_at,
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

    def _record_rebenchmark_result(self, result: BenchmarkResult) -> None:
        config_id = self.log.register_config(result.config)
        assert config_id is not None
        self.log.record_trace_entry(
            AutotuneLogEntry(
                generation=self._autotune_metrics.num_generations,
                status=result.status,
                perf_ms=(
                    _aggregate_values(result.per_shape, self.args.aggregation)
                    if math.isfinite(result.perf)
                    else None
                ),
                compile_time=result.compile_time,
                config_id=config_id,
                config=result.config,
                objective=result.perf,
                objective_unit="ratio" if self.args.relative_to is not None else "ms",
                completed_at=result.completed_at,
            ),
            event="rebenchmark",
        )

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
        if self.log.trace_enabled:
            for result in results:
                self._record_rebenchmark_result(result)
        return [result.perf for result in results]
