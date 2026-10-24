from __future__ import annotations

import contextlib
import dataclasses
import functools
import glob
import logging
import math
import os
import statistics
import sys
import tempfile
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import TypeVar
from typing import cast

import torch

from ..runtime.settings import _env_get_bool
from ..runtime.settings import is_pallas_interpret
from .progress_bar import iter_with_progress
from helion._dist_utils import all_ranks_agree
from helion._dist_utils import sync_object

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .logger import AutotuningLogger

T = TypeVar("T")

_log = logging.getLogger(__name__)
_BENCHMARK_CUDAGRAPH_ENV = "HELION_BENCHMARK_CUDAGRAPH"
# HELION_AUTOTUNE_L2_FLUSH=0 times candidates with their operands L2-resident
# (back-to-back calls, no cache clear between them), the regime of a kernel
# sequence whose small operands were just produced; the default flushes L2
# before every timed call (the streamed-once regime tritonbench measures).
_L2_FLUSH_ENV = "HELION_AUTOTUNE_L2_FLUSH"
_L2_FLUSH_REPLAY_LITERALS = frozenset({"replay", "batch", "burst"})
# ``replay`` mode sizes a batch to about this much back-to-back kernel time,
# the sweep harness's 16-call burst for kernels up to ~64 us ...
_REPLAY_BATCH_TARGET_US = 1024.0
# A batched graph keeps every call's outputs alive (a real sequence of calls
# writes distinct tensors; reusing one output buffer keeps the writes
# L2-resident and under-reads store-bound kernels), so its pool is ``calls``
# times one call's allocations and is capped to this share of free memory.
_BATCH_MEMORY_FRACTION = 0.25
# ``replay`` mode then grows the batch by powers of two until one replay takes
# at least this long (the catalogue's ``graph_long_ms`` rule: CUDA event times
# are quantized to 1.953 us on B200, so a 48 us replay of sixteen 3 us calls
# reads in 4% steps and its first call's cold start is a sixteenth of the
# reading), at most ``_REPLAY_BATCH_MAX_CALLS`` calls and within the memory
# guard.
_REPLAY_BATCH_MIN_REPLAY_US = 400.0
_REPLAY_BATCH_MAX_CALLS = 256


# The regime of the final selection's steady windows; unset derives it from
# the search regime (see ``final_selection_l2_flush_mode``).
_FINAL_L2_FLUSH_ENV = "HELION_AUTOTUNE_FINAL_L2_FLUSH"
_L2_FLUSH_MODES = ("call", "off", "replay")


def _l2_flush_mode() -> str:
    """How timed replays treat L2 (``HELION_AUTOTUNE_L2_FLUSH``).

    ``call`` (the default, any true literal): flush before every timed call,
    inside batches too, and subtract the flush-only graph. ``off`` (any false
    literal): never flush; the operands stay L2-resident. ``replay``: flush
    once before each timed replay of a batch of back-to-back calls and never
    inside it — the regime of a kernel sequence that starts cold, which is
    what the sweep harness times (a 16-call graph after an L2 flush) and
    where pipeline depth, epilogue ring depth and dependent-launch overlap
    separate candidates that read alike resident or per-call flushed.
    """
    return _l2_flush_mode_from_env(_L2_FLUSH_ENV) or "call"


def _l2_flush_mode_from_env(name: str) -> str | None:
    """The regime an environment variable names, ``None`` when it is unset."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    lowered = raw.strip().lower()
    if lowered in _L2_FLUSH_REPLAY_LITERALS:
        return "replay"
    if lowered == "call":
        return "call"
    return "call" if _env_get_bool(name, default=True) else "off"


def _resolve_l2_flush_mode(override: str | None) -> str:
    """The regime a timer runs in: ``override`` when the caller names one
    (the final selection), else ``HELION_AUTOTUNE_L2_FLUSH``."""
    if override is None:
        return _l2_flush_mode()
    if override not in _L2_FLUSH_MODES:
        raise ValueError(
            f"l2_flush_mode must be one of {_L2_FLUSH_MODES}, got {override!r}"
        )
    return override


def final_selection_l2_flush_mode() -> str:
    """The regime the final selection times its finalists in.

    The search's candidate timings rank the population in the regime of
    ``HELION_AUTOTUNE_L2_FLUSH``. Its resident regime (``off``) is quick and
    tells most candidates apart, but configs that tie with their operands
    L2-resident separate once each burst of back-to-back calls starts cold
    (the first call streams the operands in: pipeline and epilogue ring
    depth, launch geometry and dependent-launch overlap all show), and that
    is the regime a deployed kernel sequence and the sweep harness's
    flushed 16-call graph run in. The final selection therefore times its
    shortlist and finalists in the ``replay`` regime whenever the search
    runs resident; the per-call flush (default) and ``replay`` searches keep
    their own regime. ``HELION_AUTOTUNE_FINAL_L2_FLUSH`` (``0``/``1``/``replay``)
    names the final selection's regime explicitly.
    """
    explicit = _l2_flush_mode_from_env(_FINAL_L2_FLUSH_ENV)
    if explicit is not None:
        return explicit
    search = _l2_flush_mode()
    return "replay" if search == "off" else search


_MIRRORED_BENCH_MAX_SWEEPS = 64
_INTERLEAVED_EVENT_PAIRS_CAP = 1024


@dataclasses.dataclass(frozen=True)
class MirroredBenchmarkTrace:
    """Bounded wall-clock samples from deterministic mirrored benchmark sweeps."""

    orders: list[list[int]]
    elapsed_ms: list[list[float]]
    medians_ms: list[float]
    target_ms: float | None = None
    repeat_reference_perf_ms: float | None = None
    sweep_count: int | None = None
    calls_per_sample: int | None = None
    total_calls: int | None = None


def _mirrored_bench_call_layout(desired_calls: int) -> tuple[int, int, int]:
    """Return balanced sweep, batch, and actual timed-call counts."""
    desired_calls = max(2, desired_calls)
    if desired_calls % 2:
        desired_calls += 1
    calls_per_sample = max(1, math.ceil(desired_calls / _MIRRORED_BENCH_MAX_SWEEPS))
    sweep_count = math.ceil(desired_calls / calls_per_sample)
    if sweep_count % 2:
        sweep_count += 1
    total_calls = sweep_count * calls_per_sample
    return sweep_count, calls_per_sample, total_calls


def _make_l2_cache_clearer() -> Callable[[], None]:
    """Return a callable that flushes the GPU L2 cache, or a no-op.

    The generic (wall-clock) bench for backends without event timing
    otherwise times kernels with the operands resident in L2 (warm), biasing
    the autotuner toward shallow-prefetch configs that starve the pipeline in
    the cold-L2 / streamed-once regime that deployment and tritonbench (which
    clears L2 by default) actually measure. Flushing L2 between timed calls
    makes the autotune regime match the deployment regime.

    Uses Triton's CUDA-only cache-clear primitive. Returns a no-op when not on
    CUDA (e.g. TPU/Pallas backends that also use the generic bench).
    """
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None:
        return lambda: None
    from triton import runtime

    active = runtime.driver.active  # type: ignore[attr-defined]
    cache = active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]

    def clear() -> None:
        active.clear_cache(cache)  # type: ignore[attr-defined]

    return clear


def _cudagraph_unavailable_reason() -> str | None:
    if getattr(torch.version, "hip", None) is not None:
        return "CUDA graph benchmarking is only enabled for NVIDIA CUDA"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.cuda.is_current_stream_capturing():
        return "the current CUDA stream is already capturing"
    return None


# The CUDA graph dispatch state of a process (``steady_graph_dispatch``).
_GRAPH_WARMUP_ENV = "HELION_BENCHMARK_GRAPH_WARMUP"
_GRAPH_WARMUP_LAUNCHES = 16
_GRAPH_WARMUP_NOTE = (
    f"{_GRAPH_WARMUP_LAUNCHES} eager launches on side streams before the first "
    "timed graph (helion.autotuner.benchmarking.steady_graph_dispatch)"
)
_GRAPH_WARMUP_OPTED_OUT = f"skipped: {_GRAPH_WARMUP_ENV}=0"
_graph_warmup_kept: list[object] = []
# Device index -> the note of the warm-up that ran there.
_graph_warmup_notes: dict[int, str] = {}


def _graph_warmup_launch(x: torch.Tensor) -> object:
    """One eager launch on a fresh side stream of ``x``'s device, ordered
    after the device's current stream (``x`` is filled there, and side
    streams do not synchronize with it implicitly) and before its next
    work.  Returns what must stay alive."""
    side = torch.cuda.Stream(device=x.device)
    side.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(side):
        y = x * 2
    torch.cuda.current_stream(x.device).wait_stream(side)
    torch.cuda.synchronize(x.device)
    return y


def _graph_warmup_enabled() -> bool:
    """``HELION_BENCHMARK_GRAPH_WARMUP``; a value that is not a boolean
    literal keeps the default (on) with a warning instead of raising: the
    read happens inside every graph capture and every worker job, where an
    exception would silently drop CUDA-graph timing (in-process callers
    swallow it) or fail every candidate (the worker reports it per job)."""
    try:
        return _env_get_bool(_GRAPH_WARMUP_ENV, default=True)
    except ValueError as error:
        _log.warning(f"{error}; the CUDA graph dispatch warm-up stays on")
        return True


def _graph_warmup_device_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    index = torch.device(device).index
    return torch.cuda.current_device() if index is None else index


def steady_graph_dispatch(device: torch.device | str | int | None = None) -> str:
    """Drive this process into its steady CUDA graph dispatch state, once.

    A fresh process replays CUDA graphs with a shorter gap between kernel
    nodes (0.37 us against 0.55 us on B200, torch 2.16 / CUDA 13) until it
    has launched about ten kernels on streams other than the default one;
    from then on every graph of the process, old or new, replays in the
    slower steady state for the life of the process (destroying graphs,
    ``empty_cache``, the replay stream, the number of graphs captured and
    the memory they hold make no difference; eight launches on one side
    stream flip it, eight event waits without launches do not).  Every
    graph capture here warms its callable up on a side stream first, so
    without this the first two candidates of a benchmark are captured in
    one state and the rest in another, and a harness's second graph reads
    faster than its third (a 16-call graph of a 7.7 us call read 7.8 us per
    call, the 64-call graph after it 9.8).  Any long-running PyTorch process
    (autograd, data loading, collectives, ``torch.compile`` graph trees all
    launch on side streams) runs in the steady state, so that is the state
    to rank candidates in.

    Issues ``_GRAPH_WARMUP_LAUNCHES`` eager launches on fresh side streams of
    ``device`` (the current device by default; the state was measured per
    process on one device, so a benchmark whose arguments live elsewhere
    passes their device and the warm-up runs once per device), keeps their
    outputs alive, and returns a note describing the warm-up; later calls
    for that device return the same note without doing anything.  Always
    returns a description of the state the caller's graphs will run in:
    the note, ``skipped: HELION_BENCHMARK_GRAPH_WARMUP=0`` (the opt-out, for
    measuring the fast state itself), ``skipped: <why CUDA graphs are
    unavailable here>``, or ``failed: <error>`` when the launches raised
    (never fatal: the process stays in whatever state it is in and the next
    call tries again, the failure being cheap and immediate).  Every CUDA
    graph the benchmarking helpers capture (``do_bench``,
    ``interleaved_bench``, ``do_bench_cuda_graph``) is preceded by this
    call, as is the catalogue harness's first timed graph.
    """
    if not _graph_warmup_enabled():
        return _GRAPH_WARMUP_OPTED_OUT
    reason = _cudagraph_unavailable_reason()
    if reason is not None:
        return f"skipped: {reason}"
    try:
        index = _graph_warmup_device_index(device)
        note = _graph_warmup_notes.get(index)
        if note is not None:
            return note
        x = torch.ones(1024, device=torch.device("cuda", index))
        kept = [_graph_warmup_launch(x) for _ in range(_GRAPH_WARMUP_LAUNCHES)]
    except Exception as error:
        _log.debug("CUDA graph dispatch warm-up failed; skipping", exc_info=True)
        return f"failed: {type(error).__name__}: {error}"
    _graph_warmup_kept.extend(kept)
    _graph_warmup_notes[index] = _GRAPH_WARMUP_NOTE
    return _GRAPH_WARMUP_NOTE


def _reset_graph_warmup_for_tests() -> None:
    _graph_warmup_kept.clear()
    _graph_warmup_notes.clear()


def _make_cudagraph_replay(fn: Callable[[], T]) -> Callable[[], T]:
    from ..runtime import cute_cuda_graph

    steady_graph_dispatch()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    static_output: list[T] = []
    with cute_cuda_graph() as graph:
        static_output.append(fn())
    torch.cuda.synchronize()

    def replay() -> T:
        graph.replay()
        return static_output[0]

    return replay


# A single CUDA-graph replay per event pair carries the graph launch latency
# (~4-6 us on B200) quantized to ~2 us steps, so kernels shorter than a few
# tens of microseconds all read the same value.  Below this per-call estimate
# the interleaved bench times one graph holding ``calls`` x [L2 flush; call]
# and subtracts a graph holding the same flushes, which cancels the launch
# latency and keeps every call cold in L2.
_BATCHED_REPLAY_BELOW_US = 64.0
_BATCHED_REPLAY_MAX_CALLS = 16
# do_bench times the flush-only graph once per this many batched samples.
_BATCHED_FLUSH_EVERY = 4


def _no_cache_clear() -> None:
    """Cache clearer for the L2-resident regime (HELION_AUTOTUNE_L2_FLUSH=0)."""


def _batched_replay_calls(estimate_us: float) -> int:
    """How many flushed calls to fold into one timed graph replay."""
    if not math.isfinite(estimate_us) or estimate_us <= 0:
        return 1
    if estimate_us >= _BATCHED_REPLAY_BELOW_US:
        return 1
    return max(
        1,
        min(
            _BATCHED_REPLAY_MAX_CALLS, math.ceil(_BATCHED_REPLAY_BELOW_US / estimate_us)
        ),
    )


def _back_to_back_batch_calls(
    fns: Sequence[Callable[[], object]],
    clear_cache: Callable[[], None],
    di: Any,  # noqa: ANN401
    calls: int,
    *,
    mode: str,
) -> int:
    """Re-size a back-to-back batch from a probe batch's per-call time.

    ``_batched_replay_calls`` sizes the batch from a one-call graph replay,
    which carries the graph's launch latency: a 3 us kernel estimates near
    11 us and gets a 6-call batch in which that latency still dominates the
    sample, so configs whose kernels differ by a few percent read the same.
    Deployment (and the sweep harness) run calls back to back, where those
    differences, dependent-launch overlap and tails show. When the calls of
    a batch run back to back (``off`` and ``replay`` modes) capture a probe
    batch, take its per-call time and size the batch from that; ``replay``
    also fills the batch towards ``_REPLAY_BATCH_TARGET_US`` of kernel time
    so the kernels the harness times as 16-call bursts are timed the same
    way, then doubles it until one replay lasts ``_REPLAY_BATCH_MIN_REPLAY_US``
    (up to ``_REPLAY_BATCH_MAX_CALLS`` calls, within ``_batch_memory_calls``),
    the graph the harness's ``graph_long_ms`` times. The per-call flushed
    regime keeps its sizing: its calls are separated by flushes.
    """
    if mode == "call" or (mode == "off" and calls >= _BATCHED_REPLAY_MAX_CALLS):
        return calls
    if mode == "off" and calls <= 1:
        return calls
    probe = _batched_flushed_replays(list(fns), _no_cache_clear, max(calls, 1))
    if probe is None:
        return calls
    probe_functions, _flush_replay = probe
    per_call_us = min(
        _single_replay_estimate_us(replay, clear_cache, di)
        for replay in probe_functions
    ) / max(calls, 1)
    sized = max(calls, _batched_replay_calls(per_call_us))
    if mode == "replay" and math.isfinite(per_call_us) and per_call_us > 0:
        sized = max(
            sized,
            min(
                _BATCHED_REPLAY_MAX_CALLS,
                math.ceil(_REPLAY_BATCH_TARGET_US / per_call_us),
            ),
        )
        while (
            sized * per_call_us < _REPLAY_BATCH_MIN_REPLAY_US
            and sized * 2 <= _REPLAY_BATCH_MAX_CALLS
        ):
            sized *= 2
    return sized


def _batch_memory_calls(fns: Sequence[Callable[[], object]], calls: int) -> int:
    """The largest batch of at most ``calls`` whose graph pools fit: a batch
    that keeps every call's outputs (``_make_batched_cudagraph_replay``) holds
    ``calls`` times what one call allocates, every candidate of an interleaved
    bench holds its own batch at the same time, and together they must stay
    within ``_BATCH_MEMORY_FRACTION`` of the free device memory. Runs one
    call of every ``fn`` (the same count on every rank)."""
    if calls <= 1 or not torch.cuda.is_available():
        return calls
    per_call = 0
    for fn in fns:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        kept = fn()
        per_call = max(per_call, torch.cuda.memory_allocated() - before)
        del kept
    if per_call <= 0:
        return calls
    free, _total = torch.cuda.mem_get_info()
    budget = int(free * _BATCH_MEMORY_FRACTION) // max(len(fns), 1)
    return max(1, min(calls, budget // per_call))


def _batched_repetitions(warmup: float, rep: float, batch_ms: float) -> tuple[int, int]:
    """Warm-up and timed sample counts for a batched timer: one sample is one
    replay of the batch (``batch_ms``, flush included where the regime
    flushes before a replay), not ``calls`` single flushed replays."""
    if not math.isfinite(batch_ms) or batch_ms <= 0:
        return 1, 1
    return max(1, int(warmup / batch_ms)), max(1, int(rep / batch_ms))


def _build_batched_replays(
    fns: list[Callable[[], object]],
    benchmark_functions: list[Callable[[], object]],
    clear_cache: Callable[[], None],
    di: Any,  # noqa: ANN401
    *,
    mode: str,
    process_group_name: str | None,
) -> tuple[int, list[Callable[[], object]], Callable[[], None] | None]:
    """Fold the graph-replayed ``fns`` into batches: ``(calls, replays,
    flush_replay)`` with ``calls == 1`` and the single replays when nothing
    is batched.

    Every rank agrees on each size before any kernel runs on it (a kernel
    holding a collective hangs when ranks launch it different numbers of
    times): the single-replay estimate's sizing is agreed, then the probe
    batch it sizes runs, its sizing is agreed, then the memory guard (one
    call per candidate) runs and its result is agreed. Only the ``off`` and
    ``replay`` regimes keep every call's outputs (the per-call flush between
    the calls of a ``call`` batch already evicts the previous call's stores),
    so only they pay the guard's extra call and the per-call pool. A capture
    that fails on any rank halves the batch and says so; the last resort is
    the single replays, announced as such.
    """
    keep_outputs = mode != "call"
    batch_clear_cache = clear_cache if mode == "call" else _no_cache_clear
    calls = sync_object(
        _batched_replay_calls(
            min(
                _single_replay_estimate_us(bf, clear_cache, di)
                for bf in benchmark_functions
            )
        ),
        process_group_name=process_group_name,
    )
    calls = sync_object(
        _back_to_back_batch_calls(fns, clear_cache, di, calls, mode=mode),
        process_group_name=process_group_name,
    )
    if keep_outputs and calls > 1:
        calls = sync_object(
            _batch_memory_calls(fns, calls), process_group_name=process_group_name
        )
    while calls > 1:
        batched = _batched_flushed_replays(
            fns, batch_clear_cache, calls, keep_outputs=keep_outputs
        )
        # a capture that failed on one rank drops the batch on every rank
        if all_ranks_agree(batched is not None, process_group_name):
            assert batched is not None
            replays, flush_replay = batched
            return calls, list(replays), flush_replay
        smaller = calls // 2
        if smaller > 1:
            _log.warning(
                "Batched CUDA graph capture of %d calls failed; retrying with %d",
                calls,
                smaller,
            )
        else:
            _log.warning(
                "Batched CUDA graph capture of %d calls failed; timing single "
                "flushed replays instead (graph launch latency included, "
                "readings of this pass are not comparable to batched ones)",
                calls,
            )
        calls = smaller
    return 1, list(benchmark_functions), None


def _make_batched_cudagraph_replay(
    fn: Callable[[], object] | None,
    clear_cache: Callable[[], None],
    calls: int,
    *,
    keep_outputs: bool = False,
) -> Callable[[], None]:
    """Capture ``calls`` x [``clear_cache``; ``fn``] (``fn`` may be None for the
    flush-only reference graph) and return its replay.

    With ``keep_outputs`` every call's outputs stay alive with the replay: a
    sequence of calls that dropped each output would let the graph pool hand
    the same buffer to every call, keeping the writes L2-resident across the
    batch, which under-reads store-bound kernels (on B200 a 3 us rope kernel
    with narrow store packets read 2.4 us per call that way against 3.2 us
    for the harness's graph of distinct outputs) and not evenly across
    configs. A kernel that writes into an argument (``out=``) returns that
    same tensor from every call, so its batch still stores into one buffer;
    the under-read persists for those kernels in the ``off`` and ``replay``
    regimes (rotating per-call clones of the mutated arguments would need
    the benchmark callable to expose them).
    """
    from ..runtime import cute_cuda_graph

    steady_graph_dispatch()
    if fn is not None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    prior_stream = torch.cuda.current_stream()
    outputs: list[object] = []
    try:
        with cute_cuda_graph() as graph:
            for _ in range(calls):
                clear_cache()
                if fn is not None:
                    output = fn()
                    if keep_outputs:
                        outputs.append(output)
    except Exception:
        # An invalidated capture leaves the capture stream current.
        torch.cuda.set_stream(prior_stream)
        raise
    torch.cuda.synchronize()

    def replay() -> None:
        graph.replay()

    # the outputs live exactly as long as the replay does
    replay.outputs = outputs  # type: ignore[attr-defined]
    return replay


def _batched_flushed_replays(
    fns: list[Callable[[], object]],
    clear_cache: Callable[[], None],
    calls: int,
    *,
    keep_outputs: bool = False,
) -> tuple[list[Callable[[], None]], Callable[[], None] | None] | None:
    """Batched graphs for ``fns`` plus the flush-only graph (None when the
    batch does not flush: there is nothing to subtract and an empty graph is
    not worth capturing), or None when a capture fails (the caller decides
    what to fall back to)."""
    try:
        batched = [
            _make_batched_cudagraph_replay(
                fn, clear_cache, calls, keep_outputs=keep_outputs
            )
            for fn in fns
        ]
        flush_replay = (
            None
            if clear_cache is _no_cache_clear
            else _make_batched_cudagraph_replay(None, clear_cache, calls)
        )
    except Exception as error:
        _log.warning(
            "Batched CUDA graph capture of %d calls failed: %s: %s",
            calls,
            type(error).__name__,
            error,
        )
        _log.debug("Batched CUDA graph capture failed", exc_info=True)
        return None
    return batched, flush_replay


def _positive_median(samples: list[float]) -> float:
    """Median of the samples; a non-positive median (possible only when the
    flush subtraction is swamped by interference) falls back to the positive
    samples, and to ``inf`` when there are none, so a poisoned candidate is
    never ranked best."""
    median = statistics.median(samples)
    if median > 0:
        return median
    positive = [sample for sample in samples if sample > 0]
    return statistics.median(positive) if positive else math.inf


def _positive_mean(samples: list[float]) -> float:
    """Mean of the positive samples (a non-positive flush-corrected sample is
    interference, not kernel time); ``inf`` when there are none."""
    positive = [sample for sample in samples if sample > 0]
    return statistics.mean(positive) if positive else math.inf


def _maybe_cudagraph_replay(
    fn: Callable[[], T], *, default_enabled: bool = False
) -> Callable[[], T]:
    if not _env_get_bool(_BENCHMARK_CUDAGRAPH_ENV, default=default_enabled):
        return fn

    reason = _cudagraph_unavailable_reason()
    if reason is not None:
        _log.debug("Skipping CUDA graph benchmarking: %s", reason)
        return fn

    prior_stream = torch.cuda.current_stream()
    try:
        return _make_cudagraph_replay(fn)
    except Exception:
        _log.debug("CUDA graph benchmark capture failed; falling back", exc_info=True)
        torch.cuda.set_stream(prior_stream)
        return fn


def clear_jit_fast_path_caches(
    fn: Callable[..., object],
    log: logging.Logger | AutotuningLogger | None = None,
) -> None:
    """Clear Triton JIT fast-path caches for a generated Helion wrapper."""
    try:
        fn_name = getattr(fn, "__name__", None)
        fn_globals = getattr(fn, "__globals__", None)
        if fn_name is None or fn_globals is None:
            return
        kernels = getattr(fn, "_helion_cute_kernels", None)
        if not isinstance(kernels, tuple):
            kernels = (fn_globals.get(f"_helion_{fn_name}"),)
        for kernel in kernels:
            clear = getattr(kernel, "clear_fast_path_caches", None)
            if clear is not None:
                clear()
    except Exception:
        if log is not None:
            log.debug("Failed to clear Triton JIT fast-path cache.", exc_info=True)


def synchronize_device() -> None:
    """Wait for device computation to complete."""
    if not is_pallas_interpret() and torch.accelerator.is_available():
        torch.accelerator.synchronize()


def compute_repeat(
    fn: Callable[[], object],
    *,
    target_ms: float = 100.0,
    min_repeat: int = 10,
    max_repeat: int = 1000,
    estimate_runs: int = 5,
    default_cudagraph: bool = False,
) -> int:
    """
    Estimate how many repetitions are needed to collect a stable benchmark for a
    single function call, mirroring Triton's ``do_bench`` heuristic while
    clamping the result between ``min_repeat`` and ``max_repeat``.
    """
    from triton import runtime

    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]
    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]

    # Warm the pipeline once before collecting timing samples.
    fn()
    di.synchronize()
    benchmark_function = _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph)

    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(estimate_runs):
        runtime.driver.active.clear_cache(cache)  # type: ignore[attr-defined]
        benchmark_function()
    end_event.record()
    di.synchronize()

    estimate_ms = start_event.elapsed_time(end_event) / max(estimate_runs, 1)
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        return max_repeat

    repeat = int(target_ms / estimate_ms)
    return max(min_repeat, min(max_repeat, max(1, repeat)))


def compute_repeat_generic(
    fn: Callable[[], object],
    *,
    target_ms: float = 100.0,
    min_repeat: int = 10,
    max_repeat: int = 1000,
    estimate_runs: int = 5,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
) -> int:
    """
    Estimate how many repetitions are needed using wall-clock timing.
    Used for backends that don't have Triton's event-based timing (e.g., Pallas/TPU).
    """
    # Warm the pipeline once before collecting timing samples.
    _output = fn()
    synchronize_device()

    clear_l2 = _make_l2_cache_clearer()
    start = time.perf_counter()
    for _ in range(estimate_runs):
        clear_l2()
        # Keep the latest asynchronous output alive through synchronization.
        _output = fn()
    synchronize_device()
    end = time.perf_counter()

    estimate_ms = (end - start) * 1000 / max(estimate_runs, 1)
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        return max_repeat

    repeat = int(target_ms / estimate_ms)
    return max(min_repeat, min(max_repeat, max(1, repeat)))


def _interleaved_repeat_cap(
    fns: list[Callable[[], object]],
    clear_cache: Callable[[], None],
    max_total_ms: float,
) -> int:
    """Bound interleaved repeats by the measured cost of one full sweep.

    Repeat counts are sized from candidate kernel time alone, but for
    microsecond kernels the fixed per-call cost (L2 flush, launch and wrapper
    overhead) dominates, so an unbounded repeat can take minutes of wall clock
    on slow hosts. One timed sweep captures the true per-iteration cost.
    """
    synchronize_device()
    start = time.perf_counter()
    for fn in fns:
        clear_cache()
        fn()
    synchronize_device()
    sweep_ms = (time.perf_counter() - start) * 1000
    if not math.isfinite(sweep_ms) or sweep_ms <= 0:
        return sys.maxsize
    return max(3, int(max_total_ms / sweep_ms))


def interleaved_bench(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,
    max_total_ms: float | None = None,
    process_group_name: str | None = None,
    l2_flush_mode: str | None = None,
) -> list[float]:
    """
    Benchmark multiple functions at once, interleaving their executions to reduce
    the impact of external factors (e.g., load, temperature) on the
    measurements.

    Args:
        fns: List of functions to benchmark
        repeat: Number of times to repeat each benchmark
        desc: Optional description for progress bar
        max_total_ms: Optional wall-clock budget for the whole timed loop;
            ``repeat`` is lowered so one measured sweep times ``repeat`` fits
        process_group_name: Distributed group whose ranks run this bench in
            lockstep; the graph batch size is agreed across it so every rank
            launches the same number of kernels
        l2_flush_mode: L2 regime of the timed calls (``call``, ``off`` or
            ``replay``, see ``_l2_flush_mode``); ``None`` reads
            ``HELION_AUTOTUNE_L2_FLUSH``
    """
    from triton import runtime

    # warmup
    for fn in fns:
        fn()
    flush_mode = _resolve_l2_flush_mode(l2_flush_mode)
    clear_cache: Callable[[], None]
    if flush_mode != "off":
        clear_cache = functools.partial(
            runtime.driver.active.clear_cache,  # type: ignore[attr-defined]
            runtime.driver.active.get_empty_cache_for_benchmark(),  # type: ignore[attr-defined]
        )
    else:
        clear_cache = _no_cache_clear
    clear_cache()
    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]
    if max_total_ms is not None:
        repeat = min(repeat, _interleaved_repeat_cap(fns, clear_cache, max_total_ms))
    di.synchronize()
    benchmark_functions = [
        _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph) for fn in fns
    ]

    calls = 1
    flush_replay: Callable[[], None] | None = None
    if fns and all(
        bf is not fn for bf, fn in zip(benchmark_functions, fns, strict=True)
    ):
        # Every function replays a CUDA graph: fold short calls into batches
        # (sizes agreed across ranks, see _build_batched_replays). Without a
        # per-call flush there is nothing to subtract: the batch is ``calls``
        # back-to-back replays whose one graph launch latency is amortized
        # (``replay`` mode flushes once before each replay).
        calls, benchmark_functions, flush_replay = _build_batched_replays(
            fns,
            benchmark_functions,
            clear_cache,
            di,
            mode=flush_mode,
            process_group_name=process_group_name,
        )
        if calls > 1:
            # Each sample now averages ``calls`` kernel executions.
            repeat = max(min(repeat, 20), math.ceil(repeat / calls))

    # Large finalist passes can create hundreds of thousands of live timing
    # events outstanding at once. On ROCm that crashes in
    # hipEventCreateWithFlags. On XPU, each Event.record() submits a real
    # profiling-tag command to the queue, and the Level Zero backend only
    # reclaims completed events near a synchronization point, so per-event
    # cost grows superlinearly with the backlog (flat ~0.3ms/event when
    # synchronized every ~1000 events, tens of ms/event past ~100k
    # unsynchronized events), turning a single rebenchmark pass into a
    # multi-minute stall. Reuse a bounded set after collecting each batch's
    # timings, preserving the full sample count and interleaving.
    batch_size = repeat
    if torch.version.hip is not None or torch.xpu.is_available():
        batch_size = min(
            repeat, max(1, _INTERLEAVED_EVENT_PAIRS_CAP // max(1, len(fns)))
        )
    start_events = [
        [di.Event(enable_timing=True) for _ in range(batch_size)]
        for _ in range(len(fns))
    ]
    end_events = [
        [di.Event(enable_timing=True) for _ in range(batch_size)]
        for _ in range(len(fns))
    ]
    flush_events = (
        [
            (di.Event(enable_timing=True), di.Event(enable_timing=True))
            for _ in range(batch_size)
        ]
        if flush_replay is not None
        else None
    )

    # When a description is supplied we show a progress bar so the user can
    # track the repeated benchmarking loop.
    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    timings: list[list[float]] = [[] for _ in fns]
    for i in iterator:
        slot = i % batch_size
        for j in range(len(benchmark_functions)):
            if flush_replay is None:
                clear_cache()
            start_events[j][slot].record()
            benchmark_functions[j]()
            end_events[j][slot].record()
        if flush_events is not None:
            assert flush_replay is not None
            flush_events[slot][0].record()
            flush_replay()
            flush_events[slot][1].record()
        if slot + 1 == batch_size or i + 1 == repeat:
            di.synchronize()
            for j in range(len(fns)):
                if flush_events is None:
                    timings[j].extend(
                        start_events[j][k].elapsed_time(end_events[j][k]) / calls
                        for k in range(slot + 1)
                    )
                else:
                    timings[j].extend(
                        (
                            start_events[j][k].elapsed_time(end_events[j][k])
                            - flush_events[k][0].elapsed_time(flush_events[k][1])
                        )
                        / calls
                        for k in range(slot + 1)
                    )

    if flush_events is None:
        return [statistics.median(samples) for samples in timings]
    return [_positive_median(samples) for samples in timings]


def _single_replay_estimate_us(
    replay: Callable[[], object],
    clear_cache: Callable[[], None],
    di: Any,  # noqa: ANN401
    samples: int = 3,
) -> float:
    """Best of ``samples`` flushed single replays, in microseconds."""
    best = math.inf
    for _ in range(samples):
        clear_cache()
        start = di.Event(enable_timing=True)
        end = di.Event(enable_timing=True)
        start.record()
        replay()
        end.record()
        di.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0)
    return best


def interleaved_bench_generic(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
    max_total_ms: float | None = None,
) -> list[float]:
    """
    Benchmark multiple functions using wall-clock timing.
    Used for backends that don't have Triton's event-based timing (e.g., Pallas/TPU).
    """
    # warmup
    _output: object = None
    for fn in fns:
        _output = fn()
    synchronize_device()

    clear_l2 = _make_l2_cache_clearer()
    if max_total_ms is not None:
        repeat = min(repeat, _interleaved_repeat_cap(fns, clear_l2, max_total_ms))
    all_times: list[list[float]] = [[] for _ in range(len(fns))]

    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    for _i in iterator:
        for j in range(len(fns)):
            clear_l2()
            synchronize_device()
            start = time.perf_counter()
            # Dropping an asynchronous output before the sync can trigger device
            # buffer destruction inside the timed region.
            _output = fns[j]()
            synchronize_device()
            end = time.perf_counter()
            all_times[j].append((end - start) * 1000)  # convert to ms

    return [statistics.median(times) for times in all_times]


def mirrored_bench_generic(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    after_call: Callable[[int], None] | None = None,
) -> MirroredBenchmarkTrace:
    """Benchmark functions in bounded, rotated forward/reverse sample pairs."""
    if not fns:
        return MirroredBenchmarkTrace([], [], [])

    # Every forward sweep is paired with its exact reverse so each function has
    # the same mean position within a pair. Rotate successive pairs to spread
    # any nonlinear thermal drift across the candidate set. Fast kernels batch
    # multiple individually timed calls into each retained sample so the trace is
    # bounded without reducing the requested timing work.
    sweep_count, calls_per_sample, total_calls = _mirrored_bench_call_layout(repeat)

    _output: object = None
    for index, fn in enumerate(fns):
        try:
            _output = fn()
            synchronize_device()
        finally:
            if after_call is not None:
                after_call(index)

    clear_l2 = _make_l2_cache_clearer()
    all_times: list[list[float]] = [[] for _ in fns]
    orders: list[list[int]] = []
    elapsed_ms: list[list[float]] = []
    indices = list(range(len(fns)))
    iterator = iter_with_progress(
        range(sweep_count),
        total=sweep_count,
        description=desc,
        enabled=desc is not None,
    )
    for sweep in iterator:
        offset = (sweep // 2) % len(indices)
        rotated = indices[offset:] + indices[:offset]
        order = rotated if sweep % 2 == 0 else list(reversed(rotated))
        sweep_times: list[float] = []
        for index in order:
            elapsed = 0.0
            for _ in range(calls_per_sample):
                clear_l2()
                synchronize_device()
                start = time.perf_counter()
                try:
                    output = fns[index]()
                    synchronize_device()
                    elapsed += (time.perf_counter() - start) * 1000
                    _output = output
                finally:
                    if after_call is not None:
                        after_call(index)
            sample = elapsed / calls_per_sample
            all_times[index].append(sample)
            sweep_times.append(sample)
        orders.append(order)
        elapsed_ms.append(sweep_times)

    return MirroredBenchmarkTrace(
        orders,
        elapsed_ms,
        [statistics.median(times) for times in all_times],
        sweep_count=sweep_count,
        calls_per_sample=calls_per_sample,
        total_calls=total_calls,
    )


def paired_device_micros_bench(
    fns: list[Callable[..., object]],
    reference_fn: Callable[..., object],
    *,
    device_micros_fn: Callable[[Callable[[], object]], float],
    desc: str | None = None,
) -> list[tuple[float, float]]:
    """Paired device-µs timing for the final-pick re-rank.

    Times each candidate and ``reference_fn`` via ``device_micros_fn`` and returns
    ``(candidate_micros, candidate_micros - reference_micros)`` per candidate (negative delta
    = faster on-device). Unlike single-call wall-clock µs, which on small Pallas
    matmuls is ~96-98% dispatch overhead, ``device_micros_fn`` reports on-chip µs so
    configs differing by a few µs are separable. An unusable trace yields
    ``(inf, inf)`` so the caller deranks it.
    """
    iterator = iter_with_progress(
        range(len(fns)),
        total=len(fns),
        description=desc,
        enabled=desc is not None,
    )
    results: list[tuple[float, float]] = []
    for j in iterator:
        candidate_micros = device_micros_fn(fns[j])
        reference_micros = device_micros_fn(reference_fn)
        if math.isfinite(candidate_micros) and math.isfinite(reference_micros):
            results.append((candidate_micros, candidate_micros - reference_micros))
        else:
            results.append((math.inf, math.inf))
    return results


# 100 calls/trace keeps the per-event average stable to ~0.1µs (measured), well
# below the multi-µs deltas the re-rank separates.
_PALLAS_AUTOTUNE_DEVICE_MICROS_DEFAULT_N_CALLS = 100
_PALLAS_AUTOTUNE_DEVICE_MICROS_DEFAULT_N_WARMUP = 5
# Min event count for a ``/device:TPU:0`` line to count as a kernel sample: above
# the DVFS counter lines (<=~17 events), below the kernel band (>=~40). Per-call
# us divides by the actual count, so partial-flush traces stay unbiased.
_PALLAS_AUTOTUNE_DEVICE_MICROS_MIN_TRACE_EVENTS = 20


def _autotune_rank_by_device_micros() -> bool:
    """True unless ``HELION_AUTOTUNE_PALLAS_RANK_BY`` opts out of device-µs ranking.

    Default ``device_time``; any other value (e.g. ``wall_time``) falls back to the
    legacy wall-clock paired-sample ranking. Pallas-only (the device-µs path is
    jax.profiler-based); inert on other backends.
    """
    value = (
        os.environ.get("HELION_AUTOTUNE_PALLAS_RANK_BY", "device_time").strip().lower()
    )
    return value == "device_time"


def _pallas_device_micros_for_fn(
    fn: Callable[[], object],
    *,
    n_calls: int,
    n_warmup: int,
) -> float:
    """Per-call on-device µs for ``fn`` under a single ``jax.profiler`` trace.

    Wraps ``n_calls`` invocations in one ``start_trace``/``stop_trace`` window,
    parses the ``.xplane.pb``, and returns ``total_ns / count / 1000`` for the
    dominant ``/device:TPU:0`` event (count >=
    ``_PALLAS_AUTOTUNE_DEVICE_MICROS_MIN_TRACE_EVENTS``, which excludes the DVFS
    counter line). Returns ``+inf`` when the trace is unusable (no ``jax``, no
    xplane/TPU plane, too few events). Kernel exceptions from ``fn`` propagate.
    """
    try:
        import jax  # pyrefly: ignore[missing-module-attribute]
    except ImportError:
        return math.inf

    # Warmup outside the trace window so first-call compile doesn't pollute it.
    for _ in range(n_warmup):
        fn()

    with tempfile.TemporaryDirectory(prefix="helion_autotune_device_micros_") as td:
        jax.profiler.start_trace(td)
        last_out: object = None
        try:
            for _ in range(n_calls):
                last_out = fn()
        finally:
            # Block on the last output so stop_trace sees every device event.
            if last_out is not None:
                with contextlib.suppress(TypeError, AttributeError):
                    jax.block_until_ready(last_out)
            jax.profiler.stop_trace()

        matches = glob.glob(os.path.join(td, "**", "*.xplane.pb"), recursive=True)
        if not matches:
            return math.inf

        pd = jax.profiler.ProfileData.from_file(matches[0])
        best_total_ns = 0
        best_count = 0
        for plane in pd.planes:
            if plane.name != "/device:TPU:0":
                continue
            for line in plane.lines:
                totals: dict[str, int] = {}
                counts: dict[str, int] = {}
                for ev in line.events:
                    totals[ev.name] = totals.get(ev.name, 0) + ev.duration_ns
                    counts[ev.name] = counts.get(ev.name, 0) + 1
                for name, total in totals.items():
                    if (
                        counts[name] >= _PALLAS_AUTOTUNE_DEVICE_MICROS_MIN_TRACE_EVENTS
                        and total > best_total_ns
                    ):
                        best_total_ns, best_count = total, counts[name]
        if best_total_ns == 0:
            return math.inf
        return best_total_ns / best_count / 1000.0


def make_pallas_paired_device_micros_bench(
    *,
    n_calls: int = _PALLAS_AUTOTUNE_DEVICE_MICROS_DEFAULT_N_CALLS,
    n_warmup: int = _PALLAS_AUTOTUNE_DEVICE_MICROS_DEFAULT_N_WARMUP,
) -> Callable[..., list[tuple[float, float]]] | None:
    """Paired device-µs bench closure for the Pallas backend, or None.

    Returns None when the user opted out via ``HELION_AUTOTUNE_PALLAS_RANK_BY=wall_time``.
    Otherwise the closure has the :func:`paired_device_micros_bench` signature and
    captures ``n_calls`` / ``n_warmup``.
    """
    if not _autotune_rank_by_device_micros():
        return None

    def _bench(
        fns: list[Callable[..., object]],
        reference_fn: Callable[..., object],
        *,
        desc: str | None = None,
    ) -> list[tuple[float, float]]:
        def _device_micros_fn(fn: Callable[[], object]) -> float:
            return _pallas_device_micros_for_fn(fn, n_calls=n_calls, n_warmup=n_warmup)

        return paired_device_micros_bench(
            fns,
            reference_fn,
            device_micros_fn=_device_micros_fn,
            desc=desc,
        )

    return _bench


def _summarize_statistics_fallback(
    times: list[float],
    quantiles: list[float] | None,
    return_mode: str,
) -> float | tuple[float, ...]:
    """Fallback statistics summarizer when triton.testing._summarize_statistics is unavailable."""
    if return_mode == "min":
        return min(times)
    if return_mode == "max":
        return max(times)
    if return_mode == "mean":
        return statistics.mean(times)
    if return_mode == "median":
        return statistics.median(times)
    # "all" mode
    if quantiles is not None:
        sorted_times = sorted(times)
        n = len(sorted_times)
        result = []
        for q in quantiles:
            idx = min(int(q * n), n - 1)
            result.append(sorted_times[idx])
        return tuple(result)
    return statistics.median(times)


def _estimate_runtime_and_warmup(
    run_batch: Callable[[int], float],
    *,
    warmup: int,
    rep: int,
    process_group_name: str | None,
) -> tuple[float, int, bool]:
    """Estimate one launch and identify a probe that is already a valid sample."""
    first_elapsed_ms = run_batch(1)
    first_estimate_ms = sync_object(
        first_elapsed_ms, process_group_name=process_group_name
    )
    if first_estimate_ms >= max(warmup, rep):
        # The setup call warmed the kernel, and this cache-cleared probe already
        # exceeds the complete timing window. Reuse it as the sole sample.
        return first_estimate_ms, 0, True

    remaining_elapsed_ms = run_batch(4)
    estimate_ms = sync_object(
        (first_elapsed_ms + remaining_elapsed_ms) / 5,
        process_group_name=process_group_name,
    )
    return estimate_ms, max(1, int(warmup / estimate_ms)), False


# This function is copied from triton._testing.do_bench with modification
# to make sure different ranks run the benchmark for the same number
# of times.
def do_bench(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    process_group_name: str | None = None,
    *,
    default_cudagraph: bool = False,
    fixed_repetitions: int | None = None,
    probe_long_kernel: bool = False,
    pre_warmed: bool = False,
    l2_flush_mode: str | None = None,
) -> float | tuple[float, ...]:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    :param fixed_repetitions: Skip adaptive estimation and time exactly this many
        calls after the initial setup call.
    :param probe_long_kernel: Estimate from a single call first and reuse that
        probe as the timing sample when it already exceeds both timing windows;
        CuTe flash enables it for multi-second attention candidates.
    :param pre_warmed: Skip the setup launch when the caller has already run
        and synchronized this exact benchmark callable.
    :param l2_flush_mode: L2 regime of the timed calls (``call``, ``off`` or
        ``replay``, see ``_l2_flush_mode``); ``None`` reads
        ``HELION_AUTOTUNE_L2_FLUSH``.
    """
    from triton import runtime
    from triton.testing import _summarize_statistics

    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()  # pyrefly: ignore

    if not pre_warmed:
        fn()
    di.synchronize()
    # Backward benchmarks mutate grad fields between iterations, so keep their
    # existing launch path.
    benchmark_function = (
        fn
        if grad_to_none is not None
        else _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph)
    )

    flush_mode = _resolve_l2_flush_mode(l2_flush_mode)
    clear_cache: Callable[[], None] = _no_cache_clear
    if flush_mode != "off":
        cache = runtime.driver.active.get_empty_cache_for_benchmark()  # pyrefly: ignore
        clear_cache = functools.partial(runtime.driver.active.clear_cache, cache)  # pyrefly: ignore
    calls = 1
    flush_replay: Callable[[], None] | None = None
    if benchmark_function is not fn:
        # The first L2 flush and the first replay of a fresh graph pay their
        # one-time costs; keep those out of the estimate that sizes the
        # repetitions and the batch.
        clear_cache()
        benchmark_function()
        di.synchronize()

    if fixed_repetitions is None and probe_long_kernel:

        def run_estimate_batch(count: int) -> float:
            batch_start = di.Event(enable_timing=True)
            batch_end = di.Event(enable_timing=True)
            batch_start.record()
            for _ in range(count):
                clear_cache()
                benchmark_function()
            batch_end.record()
            di.synchronize()
            return float(batch_start.elapsed_time(batch_end))

        estimate_ms, n_warmup, probe_is_sample = _estimate_runtime_and_warmup(
            run_estimate_batch,
            warmup=warmup,
            rep=rep,
            process_group_name=process_group_name,
        )
        if probe_is_sample:
            return cast(
                "float | tuple[float, ...]",
                _summarize_statistics([estimate_ms], quantiles, return_mode),
            )
        n_repeat = max(1, int(rep / estimate_ms))
    elif fixed_repetitions is None:
        # Estimate the runtime of the function
        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            clear_cache()
            benchmark_function()
        end_event.record()
        di.synchronize()
        estimate_ms = sync_object(
            start_event.elapsed_time(end_event) / 5,
            process_group_name=process_group_name,
        )

        # compute number of warmup and repeat
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
    else:
        if fixed_repetitions < 1:
            raise ValueError("fixed_repetitions must be at least 1")
        n_warmup = 0
        n_repeat = fixed_repetitions
        estimate_ms = math.inf
        if benchmark_function is not fn:
            # One flushed replay decides whether the fixed repetitions are
            # batched (the final verification pins the sample count).
            estimate_ms = (
                _single_replay_estimate_us(
                    benchmark_function,
                    clear_cache,
                    di,
                    samples=1,
                )
                / 1000.0
            )

    # A short graph-replayed kernel is timed as a batched graph (see
    # _BATCHED_REPLAY_BELOW_US; sizes agreed across ranks in
    # _build_batched_replays). In the per-call flush regime the flush-only
    # graph is timed every few samples so the subtraction never spans a clock
    # or thermal drift; without a per-call flush there is nothing to subtract.
    if benchmark_function is not fn:
        calls, (benchmark_function,), flush_replay = _build_batched_replays(
            [fn],
            [benchmark_function],
            clear_cache,
            di,
            mode=flush_mode,
            process_group_name=process_group_name,
        )
        if calls > 1 and fixed_repetitions is None:
            # One sample is one replay of the batch: size the windows from
            # its own time (a flush before it where the regime flushes), not
            # from ``calls`` single flushed replays.
            batch_ms = sync_object(
                _single_replay_estimate_us(
                    benchmark_function,
                    clear_cache if flush_replay is None else _no_cache_clear,
                    di,
                    samples=2,
                )
                / 1000.0,
                process_group_name=process_group_name,
            )
            n_warmup, n_repeat = _batched_repetitions(warmup, rep, batch_ms)
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    flush_events: list[tuple[Any, Any]] = []
    # Warm-up
    for _ in range(n_warmup):
        benchmark_function()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        if flush_replay is None:
            # we clear the L2 cache before each run (a no-op when the
            # L2-resident regime is selected)
            clear_cache()
        elif i % _BATCHED_FLUSH_EVERY == 0:
            # the batched graph flushes before every call it holds; time the
            # matching flush-only graph next to the samples it corrects
            flush_start = di.Event(enable_timing=True)
            flush_end = di.Event(enable_timing=True)
            flush_start.record()
            flush_replay()
            flush_end.record()
            flush_events.append((flush_start, flush_end))
        # record time of `fn`
        start_event[i].record()
        benchmark_function()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event, strict=True)]
    if flush_replay is None and calls > 1:
        times = [t / calls for t in times]
    if flush_replay is not None:
        flushes = [start.elapsed_time(end) for start, end in flush_events]
        times = [
            (t - flushes[i // _BATCHED_FLUSH_EVERY]) / calls
            for i, t in enumerate(times)
        ]
        if return_mode == "median" and statistics.median(times) <= 0:
            return _positive_median(times)
        if return_mode == "mean":
            return _positive_mean(times)
    return _summarize_statistics(times, quantiles, return_mode)  # pyrefly: ignore


def do_bench_cuda_graph(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    process_group_name: str | None = None,
    *,
    fixed_repetitions: int | None = None,
    pre_warmed: bool = False,
    reset: Callable[[], None] | None = None,
    phase: Callable[[str], None] | None = None,
) -> float | tuple[float, ...]:
    """Time GPU execution with external event nodes inside one CUDA graph.

    Input reset and L2 clearing precede the start event. Python dispatch and
    graph submission are excluded. ``reset`` must contain capturable device
    operations; it runs before every invocation, including capture and warmup.

    Warmup/repetition windows bound sampling wall time; reported samples are
    device milliseconds. Capture errors propagate without a timing fallback.
    """
    from triton.testing import _summarize_statistics

    from ..runtime import cute_cuda_graph

    reason = _cudagraph_unavailable_reason()
    if reason is not None:
        raise RuntimeError(f"GPU-only CUDA graph timing is unavailable: {reason}")
    # The handoff evaluation times every candidate through this function in
    # one process; without this its first ~ten candidates would be captured
    # in the fast dispatch state and the rest in the steady one.
    steady_graph_dispatch()
    if grad_to_none is not None:
        raise ValueError("CUDA graph timing requires an explicit device reset callback")
    if fixed_repetitions is not None and fixed_repetitions < 1:
        raise ValueError("fixed_repetitions must be at least 1")
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be nonnegative and rep must be positive")
    if return_mode not in ("min", "max", "mean", "median", "all"):
        raise ValueError(f"Invalid return_mode: {return_mode}")

    clear_l2 = _make_l2_cache_clearer()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.stream(stream):
        if reset is not None:
            reset()
        if not pre_warmed:
            fn()
        # Materialize event handles before capture.
        start.record()
        end.record()
    stream.synchronize()
    if phase is not None:
        phase("capture")
    with cute_cuda_graph(stream=stream) as graph:
        if reset is not None:
            reset()
        clear_l2()
        start.record()
        # Keep the captured output alive until all replays finish.
        _output = fn()
        end.record()

    def sample() -> float:
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        elapsed = float(start.elapsed_time(end))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Invalid GPU-only timing: {elapsed}")
        return elapsed

    if phase is not None:
        phase("warmup")
    if fixed_repetitions is None:
        began = time.perf_counter()
        sample()
        # Count complete replays, including cache clearing and synchronization,
        # to avoid excessive sampling work for microsecond kernels.
        iteration_ms = sync_object(
            max((time.perf_counter() - began) * 1000, 1e-6), process_group_name
        )
        n_warmup = max(0, int(warmup / iteration_ms))
        n_repeat = max(1, int(rep / iteration_ms))
    else:
        n_warmup = 0
        n_repeat = fixed_repetitions
    for _ in range(n_warmup):
        sample()
    if phase is not None:
        phase("measure")
    samples = [sample() for _ in range(n_repeat)]
    return cast(
        "float | tuple[float, ...]",
        _summarize_statistics(samples, quantiles, return_mode),
    )


def do_bench_generic(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    process_group_name: str | None = None,
    *,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
    fixed_repetitions: int | None = None,
    probe_long_kernel: bool = False,
    pre_warmed: bool = False,
    l2_flush_mode: str | None = None,  # accepted for API symmetry; always flushes
) -> float | tuple[float, ...]:
    """
    Benchmark using wall-clock timing for backends without Triton event timing.

    ``fixed_repetitions`` skips adaptive estimation and times exactly that many
    calls after the initial setup call. ``probe_long_kernel`` reuses the first
    probe as the timing sample when it already exceeds both timing windows;
    CuTe flash enables it for multi-second attention candidates.
    ``pre_warmed`` skips that setup call when the caller already ran and
    synchronized the same callable.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    _output = None if pre_warmed else fn()
    synchronize_device()

    clear_l2 = _make_l2_cache_clearer()

    if fixed_repetitions is None and probe_long_kernel:

        def run_estimate_batch(count: int) -> float:
            nonlocal _output
            synchronize_device()
            start = time.perf_counter()
            for _ in range(count):
                clear_l2()
                # Keep the latest asynchronous output alive through synchronization.
                _output = fn()
            synchronize_device()
            end = time.perf_counter()
            return (end - start) * 1000

        estimate_ms, n_warmup, probe_is_sample = _estimate_runtime_and_warmup(
            run_estimate_batch,
            warmup=warmup,
            rep=rep,
            process_group_name=process_group_name,
        )
        if probe_is_sample:
            return _summarize_statistics_fallback([estimate_ms], quantiles, return_mode)
        n_repeat = max(1, int(rep / estimate_ms))
    elif fixed_repetitions is None:
        synchronize_device()
        start = time.perf_counter()
        for _ in range(5):
            clear_l2()
            _output = fn()
        synchronize_device()
        end = time.perf_counter()
        estimate_ms = sync_object(
            (end - start) * 1000 / 5,
            process_group_name=process_group_name,
        )
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
    else:
        if fixed_repetitions < 1:
            raise ValueError("fixed_repetitions must be at least 1")
        n_warmup = 0
        n_repeat = fixed_repetitions
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    times: list[float] = []
    for _i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        clear_l2()
        synchronize_device()
        t0 = time.perf_counter()
        _output = fn()
        synchronize_device()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convert to ms
    return _summarize_statistics_fallback(times, quantiles, return_mode)
