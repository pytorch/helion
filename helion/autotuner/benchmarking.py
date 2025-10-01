from __future__ import annotations

import functools
import statistics
from typing import Callable

from tqdm.auto import tqdm
from triton import runtime


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
    iterator = range(repeat)
    if desc is not None:
        iterator = tqdm(iterator, desc=desc, total=repeat, unit="round")
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
