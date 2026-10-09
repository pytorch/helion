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
    from .logger import AutotuningLogger

T = TypeVar("T")

_log = logging.getLogger(__name__)
_BENCHMARK_CUDAGRAPH_ENV = "HELION_BENCHMARK_CUDAGRAPH"
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


def _make_cudagraph_replay(fn: Callable[[], T]) -> Callable[[], T]:
    from ..runtime import cute_cuda_graph

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


def _make_batched_cudagraph_replay(
    fn: Callable[[], object] | None, clear_cache: Callable[[], None], calls: int
) -> Callable[[], None]:
    """Capture ``calls`` x [``clear_cache``; ``fn``] (``fn`` may be None for the
    flush-only reference graph) and return its replay."""
    from ..runtime import cute_cuda_graph

    if fn is not None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    prior_stream = torch.cuda.current_stream()
    try:
        with cute_cuda_graph() as graph:
            for _ in range(calls):
                clear_cache()
                if fn is not None:
                    fn()
    except Exception:
        # An invalidated capture leaves the capture stream current.
        torch.cuda.set_stream(prior_stream)
        raise
    torch.cuda.synchronize()
    return graph.replay


def _batched_flushed_replays(
    fns: list[Callable[[], object]],
    clear_cache: Callable[[], None],
    calls: int,
) -> tuple[list[Callable[[], None]], Callable[[], None]] | None:
    """Batched graphs for ``fns`` plus the flush-only graph, or None when a
    capture fails (the caller keeps the single-replay path)."""
    try:
        batched = [_make_batched_cudagraph_replay(fn, clear_cache, calls) for fn in fns]
        flush_replay = _make_batched_cudagraph_replay(None, clear_cache, calls)
    except Exception:
        _log.debug("Batched CUDA graph capture failed; falling back", exc_info=True)
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
    """
    from triton import runtime

    # warmup
    for fn in fns:
        fn()
    clear_cache = functools.partial(
        runtime.driver.active.clear_cache,  # type: ignore[attr-defined]
        runtime.driver.active.get_empty_cache_for_benchmark(),  # type: ignore[attr-defined]
    )
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
        # Every function replays a CUDA graph: fold short calls into batches.
        # Every rank must agree on the batch: a kernel holding a collective
        # hangs when one rank replays it a different number of times.
        calls = sync_object(
            _batched_replay_calls(
                min(
                    _single_replay_estimate_us(bf, clear_cache, di)
                    for bf in benchmark_functions
                )
            ),
            process_group_name=process_group_name,
        )
        if calls > 1:
            batched = _batched_flushed_replays(fns, clear_cache, calls)
            if not all_ranks_agree(batched is not None, process_group_name):
                batched = None
            if batched is None:
                calls = 1
            else:
                benchmark_functions, flush_replay = batched
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
                        start_events[j][k].elapsed_time(end_events[j][k])
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

    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # pyrefly: ignore
    calls = 1
    flush_replay: Callable[[], None] | None = None
    if benchmark_function is not fn:
        # The first L2 flush and the first replay of a fresh graph pay their
        # one-time costs; keep those out of the estimate that sizes the
        # repetitions and the batch.
        runtime.driver.active.clear_cache(cache)  # pyrefly: ignore
        benchmark_function()
        di.synchronize()

    if fixed_repetitions is None and probe_long_kernel:

        def run_estimate_batch(count: int) -> float:
            batch_start = di.Event(enable_timing=True)
            batch_end = di.Event(enable_timing=True)
            batch_start.record()
            for _ in range(count):
                runtime.driver.active.clear_cache(cache)  # pyrefly: ignore
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
            runtime.driver.active.clear_cache(cache)  # pyrefly: ignore
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
                    functools.partial(runtime.driver.active.clear_cache, cache),  # pyrefly: ignore
                    di,
                    samples=1,
                )
                / 1000.0
            )

    # A short graph-replayed kernel is timed as a batched flushed graph (see
    # _BATCHED_REPLAY_BELOW_US); every rank must agree on the batch.  The
    # batch is sized from the kernel's own replay time (the estimate above
    # includes the L2 flush).  The flush-only graph is timed every few samples
    # so the subtraction never spans a clock or thermal drift.
    if benchmark_function is not fn:
        clear_cache = functools.partial(runtime.driver.active.clear_cache, cache)  # pyrefly: ignore
        calls = sync_object(
            _batched_replay_calls(
                _single_replay_estimate_us(benchmark_function, clear_cache, di)
            ),
            process_group_name=process_group_name,
        )
        if calls > 1:
            batched = _batched_flushed_replays([fn], clear_cache, calls)
            # a capture that failed on one rank drops the batch on every rank
            if not all_ranks_agree(batched is not None, process_group_name):
                batched = None
            if batched is None:
                calls = 1
            else:
                (benchmark_function,), flush_replay = batched
                if fixed_repetitions is None:
                    # Each sample now covers ``calls`` executions.
                    n_warmup = max(1, n_warmup // calls)
                    n_repeat = max(1, n_repeat // calls)
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
            # we clear the L2 cache before each run
            runtime.driver.active.clear_cache(cache)  # pyrefly: ignore
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
