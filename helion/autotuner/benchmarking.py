from __future__ import annotations

import functools
import math
import statistics
import time
from typing import Any
from typing import Callable
from typing import TypeVar

import torch

from ..runtime.settings import _get_backend
from ..runtime.settings import is_pallas_interpret
from .progress_bar import iter_with_progress
from helion._dist_utils import sync_object

T = TypeVar("T")


def synchronize_device(result: object = None) -> None:
    """Wait for device computation to complete.

    For TPU tensors, uses ``torch_tpu``'s tensor-level sync which truly
    blocks until the device finishes (``torch.accelerator.synchronize()``
    does not reliably wait on ``torch_tpu``).  For all other cases, falls
    back to ``torch.accelerator.synchronize()``.
    """
    if isinstance(result, torch.Tensor) and result.device.type == "tpu":
        try:
            from torch_tpu._internal.sync import (  # pyrefly: ignore[missing-import]
                synchronize as tpu_sync,
            )

            tpu_sync(result, wait=True)
            return
        except ImportError:
            raise ImportError(
                "torch_tpu is required for reliable device synchronization on TPU. "
                "Install torch_tpu or torch.accelerator.synchronize() will return "
                "before device computation finishes, producing incorrect benchmarks."
            ) from None
    if (
        not is_pallas_interpret()
        and _get_backend() != "pallas"
        and torch.accelerator.is_available()
    ):
        torch.accelerator.synchronize()


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
    from triton import runtime

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


def compute_repeat_generic(
    fn: Callable[[], object],
    *,
    target_ms: float = 100.0,
    min_repeat: int = 10,
    max_repeat: int = 1000,
    estimate_runs: int = 5,
) -> int:
    """
    Estimate how many repetitions are needed using wall-clock timing.
    Used for backends that don't have Triton's event-based timing (e.g., Pallas/TPU).
    """
    # Warm the pipeline once before collecting timing samples.
    out = fn()
    synchronize_device(out)

    start = time.perf_counter()
    for _ in range(estimate_runs):
        out = fn()
    synchronize_device(out)
    end = time.perf_counter()

    estimate_ms = (end - start) * 1000 / max(estimate_runs, 1)
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


def interleaved_bench_generic(
    fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
) -> list[float]:
    """
    Benchmark multiple functions using wall-clock timing.
    Used for backends that don't have Triton's event-based timing (e.g., Pallas/TPU).
    """
    # warmup
    out: object = None
    for fn in fns:
        out = fn()
    synchronize_device(out)

    all_times: list[list[float]] = [[] for _ in range(len(fns))]

    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    for _i in iterator:
        for j in range(len(fns)):
            synchronize_device(out)
            start = time.perf_counter()
            out = fns[j]()
            synchronize_device(out)
            end = time.perf_counter()
            all_times[j].append((end - start) * 1000)  # convert to ms

    return [statistics.median(times) for times in all_times]


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
    """
    from triton import runtime
    from triton.testing import _summarize_statistics

    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()  # pyrefly: ignore

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # pyrefly: ignore

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)  # pyrefly: ignore
        fn()
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
        fn()
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
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event, strict=True)]
    return _summarize_statistics(times, quantiles, return_mode)  # pyrefly: ignore


def do_bench_generic(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    process_group_name: str | None = None,
) -> float | tuple[float, ...]:
    """
    Benchmark using wall-clock timing for backends without Triton event timing.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    out = fn()
    synchronize_device(out)

    # Estimate the runtime of the function
    synchronize_device(out)
    start = time.perf_counter()
    for _ in range(5):
        out = fn()
    synchronize_device(out)
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
        synchronize_device(out)
        t0 = time.perf_counter()
        out = fn()
        synchronize_device(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convert to ms
    return _summarize_statistics_fallback(times, quantiles, return_mode)
