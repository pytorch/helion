from __future__ import annotations

from typing import Callable
from typing import Sequence

import torch
import triton


def _summarize_statistics(
    times: torch.Tensor,
    quantiles: Sequence[float] | None,
    return_mode: str,
) -> float | list[float]:
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(torch, return_mode)(times).item()


def do_bench_cudagraph_with_cache_clear(
    fn: Callable[[], object],
    rep: int = 20,
    grad_to_none: Sequence[torch.Tensor] | None = None,
    quantiles: Sequence[float] | None = None,
    return_mode: str = "mean",
) -> float | list[float]:
    """
    Clone of triton.testing.do_bench_cudagraph with explicit L2 cache clearing.

    NOTE: We will switch to use triton.testing.do_bench_cudagraph once it has explicit L2 cache clearing.

    Args:
        fn: Function to benchmark
        rep: Target total measurement time in milliseconds
        grad_to_none: Tensors whose gradients should be cleared before each measurement
        quantiles: Quantiles to compute from the timing measurements
        return_mode: "min", "max", "mean", "median", or "all"

    Returns:
        Timing measurement(s) in milliseconds according to return_mode
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    # Get a cache tensor and function to zero it for L2 cache clearing
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]
    clear_cache_fn = cache.zero_

    # Use a separate CUDA stream for all benchmark operations
    with torch.cuda.stream(torch.cuda.Stream()):
        # Warmup: clear cache and run function once to ensure it's compiled
        clear_cache_fn()
        fn()

        # Reset gradients if needed (for autograd-enabled benchmarks)
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None

        # Estimate execution time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            clear_cache_fn()
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        # Calculate number of repetitions needed to reach target measurement time (rep)
        n_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))

        # Create a CUDA graph for the actual kernel execution + cache clearing
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                clear_cache_fn()
                fn()
        torch.cuda.synchronize()

        # Create a separate CUDA graph for just cache clearing
        cache_clear_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cache_clear_graph):
            for _ in range(n_repeat):
                clear_cache_fn()
        torch.cuda.synchronize()

        # Run multiple retries to get stable measurements
        n_retries = 10
        cache_clear_times = []
        total_times = []
        for _ in range(n_retries):
            # Measure time for cache clearing only
            cache_clear_start_event = torch.cuda.Event(enable_timing=True)
            cache_clear_end_event = torch.cuda.Event(enable_timing=True)
            cache_clear_start_event.record()
            cache_clear_graph.replay()
            cache_clear_end_event.record()
            torch.cuda.synchronize()
            cache_clear_times.append(
                cache_clear_start_event.elapsed_time(cache_clear_end_event) / n_repeat
            )

            # Measure total time (cache clearing + kernel execution)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            total_times.append(start_event.elapsed_time(end_event) / n_repeat)

    # Subtract cache clearing overhead to get pure kernel execution time
    all_kernel_times = []
    for total_time, cache_clear_time in zip(
        total_times, cache_clear_times, strict=True
    ):
        kernel_time = total_time - cache_clear_time
        all_kernel_times.append(kernel_time)

    # Compute the requested statistic
    times = torch.tensor(all_kernel_times, dtype=torch.float)
    return _summarize_statistics(times, quantiles, return_mode)
