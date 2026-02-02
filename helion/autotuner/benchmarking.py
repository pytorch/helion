from __future__ import annotations

import functools
import math
import statistics
from typing import Callable

from triton import runtime

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
    *,
    best_so_far: float,
    early_exit_threshold: float = 2.0,
    min_reps: int = 10,
    target_reps: int = 50,
    warmup: int = 1,
) -> tuple[float, int, bool]:
    """
    Benchmark a function with early exit if clearly slower than best_so_far.

    Args:
        fn: Function to benchmark
        best_so_far: Current best performance
        early_exit_threshold: Exit if median > best_so_far * threshold
        min_reps: Minimum reps before considering early exit
        target_reps: Target total repetitions if competitive
        warmup: Number of warmup runs

    Returns:
        (median_ms, actual_reps, early_exited)
    """

    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]
    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]

    # Warmup
    for _ in range(warmup):
        fn()
    di.synchronize()

    # Collect timing samples with progressive early exit
    times: list[float] = []
    events = []

    for i in range(target_reps):
        runtime.driver.active.clear_cache(cache)  # type: ignore[attr-defined]
        start = di.Event(enable_timing=True)
        end = di.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        events.append((start, end))

        # Check ONCE after min_reps
        if i == min_reps - 1:  # Only at rep 10
            di.synchronize()
            times = [s.elapsed_time(e) for s, e in events]
            current_median = statistics.median(times)

            # Early exit if clearly slower than best
            if math.isfinite(best_so_far) and current_median > best_so_far * early_exit_threshold:
                return current_median, len(times), True
        

    di.synchronize()
    if not times:
        times = [s.elapsed_time(e) for s, e in events]

    final_median = statistics.median(times)
    return final_median, len(times), False
