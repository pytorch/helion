from __future__ import annotations

import contextlib
import dataclasses
import functools
import glob
import logging
import math
import os
import statistics
import tempfile
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import TypeVar

import numpy as np
import torch

from ..runtime.settings import _env_get_bool
from ..runtime.settings import is_pallas_interpret
from .progress_bar import iter_with_progress
from helion._dist_utils import sync_object

if TYPE_CHECKING:
    from .logger import AutotuningLogger

T = TypeVar("T")

_log = logging.getLogger(__name__)
_BENCHMARK_CUDAGRAPH_ENV = "HELION_BENCHMARK_CUDAGRAPH"


def _make_l2_cache_clearer() -> Callable[[], None]:
    """Return a callable that flushes the GPU L2 cache, or a no-op.

    The generic (wall-clock) bench used by the CuTe backend otherwise times
    kernels with the operands resident in L2 (warm), biasing the autotuner
    toward shallow-prefetch configs that starve the pipeline in the cold-L2 /
    streamed-once regime that deployment and tritonbench (which clears L2 by
    default) actually measure. Flushing L2 between timed calls makes the
    autotune regime match the deployment regime.

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


@dataclasses.dataclass(frozen=True)
class PerfStats:
    """Per-config latency statistics for the autotune dataset.

    A successful benchmark yields all-real fields (``n_samples >= 1``). The
    default-constructed ``PerfStats()`` is the null sentinel for a config that
    was registered but never produced a good measurement (all ``None``,
    ``n_samples == 0``).
    """

    min: float | None = None
    median: float | None = None
    mean: float | None = None
    p90: float | None = None
    std: float | None = None
    n_samples: int = 0

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


# Return type of the benchmark helpers: a scalar, the multi-quantile tuple, or the
# full PerfStats record (``return_mode="stats"``).
BenchTimes = float | tuple[float, ...] | PerfStats


def _compute_perf_stats(times: list[float]) -> PerfStats:
    # Callers (do_bench / do_bench_generic) always build ``times`` from an
    # ``n_repeat = max(1, ...)`` loop, so an empty list is a programming error,
    # not a "no samples" state -- that state is ``PerfStats()`` at the
    # sink boundary.
    assert times, "_compute_perf_stats requires at least one timing sample"
    n = len(times)
    arr = np.asarray(times, dtype=float)
    return PerfStats(
        min=float(arr.min()),
        median=float(np.median(arr)),
        mean=float(arr.mean()),
        p90=float(np.percentile(arr, 90.0)),
        std=float(arr.std(ddof=1)) if n > 1 else 0.0,
        n_samples=n,
    )


def _cudagraph_unavailable_reason() -> str | None:
    if getattr(torch.version, "hip", None) is not None:
        return "CUDA graph benchmarking is only enabled for NVIDIA CUDA"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.cuda.is_current_stream_capturing():
        return "the current CUDA stream is already capturing"
    return None


def _make_cudagraph_replay(fn: Callable[[], T]) -> Callable[[], T]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_output: list[T] = []
    with torch.cuda.graph(graph):
        static_output.append(fn())
    torch.cuda.synchronize()

    def replay() -> T:
        graph.replay()
        return static_output[0]

    return replay


def _maybe_cudagraph_replay(
    fn: Callable[[], T], *, default_enabled: bool = False
) -> Callable[[], T]:
    if not _env_get_bool(_BENCHMARK_CUDAGRAPH_ENV, default=default_enabled):
        return fn

    reason = _cudagraph_unavailable_reason()
    if reason is not None:
        _log.debug("Skipping CUDA graph benchmarking: %s", reason)
        return fn

    try:
        return _make_cudagraph_replay(fn)
    except Exception:
        _log.debug("CUDA graph benchmark capture failed; falling back", exc_info=True)
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
        triton_jit_fn = fn_globals.get(f"_helion_{fn_name}")
        clear = getattr(triton_jit_fn, "clear_fast_path_caches", None)
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
    fn()
    synchronize_device()

    clear_l2 = _make_l2_cache_clearer()
    start = time.perf_counter()
    for _ in range(estimate_runs):
        clear_l2()
        fn()
    synchronize_device()
    end = time.perf_counter()

    estimate_ms = (end - start) * 1000 / max(estimate_runs, 1)
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        return max_repeat

    repeat = int(target_ms / estimate_ms)
    return max(min_repeat, min(max_repeat, max(1, repeat)))


def interleaved_bench(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,
) -> list[float]:
    """
    Benchmark multiple functions at once, interleaving their executions to reduce
    the impact of external factors (e.g., load, temperature) on the
    measurements.

    Args:
        fns: List of functions to benchmark
        repeat: Number of times to repeat each benchmark
        desc: Optional description for progress bar
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
    start_events = [
        [di.Event(enable_timing=True) for _ in range(repeat)] for _ in range(len(fns))
    ]
    end_events = [
        [di.Event(enable_timing=True) for _ in range(repeat)] for _ in range(len(fns))
    ]

    di.synchronize()
    benchmark_functions = [
        _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph) for fn in fns
    ]

    # When a description is supplied we show a progress bar so the user can
    # track the repeated benchmarking loop.
    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    for i in iterator:
        for j in range(len(benchmark_functions)):
            clear_cache()
            start_events[j][i].record()
            benchmark_functions[j]()
            end_events[j][i].record()
    di.synchronize()

    return [
        statistics.median(
            [
                s.elapsed_time(e)
                for s, e in zip(start_events[j], end_events[j], strict=True)
            ]
        )
        for j in range(len(fns))
    ]


def interleaved_bench_generic(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
) -> list[float]:
    """
    Benchmark multiple functions using wall-clock timing.
    Used for backends that don't have Triton's event-based timing (e.g., Pallas/TPU).
    """
    # warmup
    for fn in fns:
        fn()
    synchronize_device()

    clear_l2 = _make_l2_cache_clearer()
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
            fns[j]()
            synchronize_device()
            end = time.perf_counter()
            all_times[j].append((end - start) * 1000)  # convert to ms

    return [statistics.median(times) for times in all_times]


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
        import jax  # pyrefly: ignore[missing-import, missing-module-attribute]
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
) -> BenchTimes:
    """Fallback statistics summarizer when triton.testing._summarize_statistics is unavailable."""
    if return_mode == "min":
        return min(times)
    if return_mode == "max":
        return max(times)
    if return_mode == "mean":
        return statistics.mean(times)
    if return_mode == "median":
        return statistics.median(times)
    if return_mode == "stats":
        return _compute_perf_stats(times)
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
) -> BenchTimes:
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
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", "all", or "stats". Default is "mean".
    :type return_mode: str
    """
    from triton import runtime
    from triton.testing import _summarize_statistics

    assert return_mode in ["min", "max", "mean", "median", "all", "stats"]

    di = runtime.driver.active.get_device_interface()  # pyrefly: ignore

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
        start_event.elapsed_time(end_event) / 5, process_group_name=process_group_name
    )

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
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
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)  # pyrefly: ignore
        # record time of `fn`
        start_event[i].record()
        benchmark_function()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event, strict=True)]
    # "stats" precedes triton's helper, which doesn't know that mode.
    if return_mode == "stats":
        return _compute_perf_stats(times)
    return _summarize_statistics(times, quantiles, return_mode)  # pyrefly: ignore


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
) -> BenchTimes:
    """
    Benchmark using wall-clock timing for backends without Triton event timing.
    """
    assert return_mode in ["min", "max", "mean", "median", "all", "stats"]

    fn()
    synchronize_device()

    clear_l2 = _make_l2_cache_clearer()

    # Estimate the runtime of the function
    synchronize_device()
    start = time.perf_counter()
    for _ in range(5):
        clear_l2()
        fn()
    synchronize_device()
    end = time.perf_counter()
    estimate_ms = sync_object(
        (end - start) * 1000 / 5, process_group_name=process_group_name
    )

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
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
        fn()
        synchronize_device()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convert to ms
    return _summarize_statistics_fallback(times, quantiles, return_mode)
