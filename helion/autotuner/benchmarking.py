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
