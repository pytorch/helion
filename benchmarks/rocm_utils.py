from __future__ import annotations

import math
from typing import Callable

import torch
import triton

_GRAPH_REPEATS = 256


def do_bench_cudagraph_with_cache_clear(
    fn: Callable[[], object],
    rep: int | None = 20,
    grad_to_none: list[torch.Tensor] | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    skip_cache_clearing: bool = False,
) -> float | list[float]:
    """TritonBench's cold-cache graph timer with bounded HIP submissions.

    Large graphs overflow HIP's recursive graph scheduler. Even smaller graphs
    can fail when many replays are queued without waiting. Keep the requested
    measurement duration by replaying bounded graphs and waiting between them.
    Cache clearing and subtraction match TritonBench's pinned timer.
    """
    # TritonBench is an optional benchmark dependency, installed by the caller.
    from tritonbench.components.do_bench.common import (  # pyrefly: ignore [missing-import]
        summarize_statistics,
    )
    from tritonbench.components.do_bench.utils import (  # pyrefly: ignore [missing-import]
        resolve_warmup_and_rep,
    )

    cache = (
        triton.runtime.driver.active.get_empty_cache_for_benchmark()  # pyrefly: ignore
        if not skip_cache_clearing
        else None
    )

    def clear_cache() -> None:
        if cache is not None:
            cache.zero_()

    with torch.cuda.stream(torch.cuda.Stream()):
        clear_cache()
        fn()
        if grad_to_none is not None:
            for tensor in grad_to_none:
                tensor.detach_()
                tensor.requires_grad_(True)
                tensor.grad = None

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            clear_cache()
            fn()
        end.record()
        torch.cuda.synchronize()
        estimate_ms = start.elapsed_time(end) / 5
        _, rep = resolve_warmup_and_rep(None, rep, estimate_ms)
        requested_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))
        graph_repeat = min(requested_repeat, _GRAPH_REPEATS)
        replay_count = math.ceil(requested_repeat / graph_repeat)
        total_repeat = graph_repeat * replay_count

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_repeat):
                if grad_to_none is not None:
                    for tensor in grad_to_none:
                        tensor.grad = None
                clear_cache()
                fn()
        torch.cuda.synchronize()

        cache_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cache_graph):
            for _ in range(graph_repeat):
                clear_cache()
        torch.cuda.synchronize()

        def measure(captured: torch.cuda.CUDAGraph) -> float:
            elapsed_ms = 0.0
            for _ in range(replay_count):
                start.record()
                captured.replay()
                end.record()
                # Bound outstanding HIP submissions. Place the wait outside
                # the event interval so CPU waiting time is not measured.
                end.synchronize()
                elapsed_ms += start.elapsed_time(end)
            return elapsed_ms / total_repeat

        times = []
        for _ in range(10):
            cache_ms = measure(cache_graph)
            times.append(measure(graph) - cache_ms)

    return summarize_statistics(
        torch.tensor(times, dtype=torch.float), quantiles, return_mode
    )
