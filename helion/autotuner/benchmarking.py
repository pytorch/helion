from __future__ import annotations

import functools
import math
import statistics
from typing import Any
from typing import Callable

from triton import runtime
from triton.testing import _summarize_statistics

from .progress_bar import iter_with_progress


def compute_repeat(
    fn: Callable[[], object],
    *,
    target_ms: float = 100.0,
    min_repeat: int = 10,
    max_repeat: int = 1000,
    estimate_runs: int = 5,
) -> int:
    """
    Estimate how many repetitions are needed to collect a stable benchmark for a
    single function call, mirroring Triton's ``do_bench`` heuristic while
    clamping the result between ``min_repeat`` and ``max_repeat``.
    """
    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]
    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]

    # Warm the pipeline once before collecting timing samples.
    fn()
    di.synchronize()

    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(estimate_runs):
        runtime.driver.active.clear_cache(cache)  # type: ignore[attr-defined]
        fn()
    end_event.record()
    di.synchronize()

    estimate_ms = start_event.elapsed_time(end_event) / max(estimate_runs, 1)
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        return max_repeat

    repeat = int(target_ms / estimate_ms)
    return max(min_repeat, min(max_repeat, max(1, repeat)))


def interleaved_bench(
    fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
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

    # When a description is supplied we show a progress bar so the user can
    # track the repeated benchmarking loop.
    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    for i in iterator:
        for j in range(len(fns)):
            clear_cache()
            start_events[j][i].record()
            fns[j]()
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


def do_bench_with_early_exit(
    fn: Callable[[], object],
    best_so_far: float,
    warmup: float = 25.0,
    rep: float = 100.0,
    grad_to_none: Any = None,
    quantiles: Any = None,
    return_mode: str = "median",
    early_exit_threshold: float = 2.0,
) -> tuple[float, int, bool]:
    """
    Benchmark a function with early exit if clearly slower than best_so_far.

    Identical to triton.testing.do_bench except adds early exit capability.
    Uses the 5-iteration estimation phase to determine if config is too slow.

    Args:
        fn: Function to benchmark
        best_so_far: Current best performance for early exit comparison
        warmup: Warmup time (in ms)
        rep: Repetition time (in ms)
        grad_to_none: Reset the gradient of the provided tensor to None
        quantiles: Performance percentile to return in addition to the median
        return_mode: The statistical measure to return ("min", "max", "mean", "median", or "all")
        early_exit_threshold: Exit if estimate > best_so_far * threshold

    Returns:
        (result, actual_reps, early_exited) where result is from _summarize_statistics
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]

    # Run 5 estimation iterations with individual event recording
    estimate_runs = 5
    start_events = [di.Event(enable_timing=True) for _ in range(estimate_runs)]
    end_events = [di.Event(enable_timing=True) for _ in range(estimate_runs)]
    for i in range(estimate_runs):
        runtime.driver.active.clear_cache(cache)  # type: ignore[attr-defined]
        start_events[i].record()
        fn()
        end_events[i].record()
    di.synchronize()

    estimate_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    estimate_ms = statistics.median(estimate_times)

    # Early exit if median exceeds threshold
    if math.isfinite(best_so_far) and estimate_ms > best_so_far * early_exit_threshold:
        return estimate_ms, estimate_runs, True

    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    start_event = [di.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for _ in range(n_repeat)]

    for _ in range(n_warmup):
        fn()

    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        runtime.driver.active.clear_cache(cache)  # type: ignore[attr-defined]
        start_event[i].record()
        fn()
        end_event[i].record()

    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]

    # Return result using same statistics function as original do_bench
    return _summarize_statistics(times, quantiles, return_mode), len(times), False
