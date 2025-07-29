from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch


def benchmark_distributed(
    fn: Callable[[], None],
    warmup: int = 50,
    rep: int = 100,
    profile_ranks: list[int] | None = None,
    cuda_graph: bool = False,
    return_mode: str = "mean",
) -> float:
    import torch.distributed as dist

    if cuda_graph:
        fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()

        def replay_target_fn() -> None:
            g.replay()

        fn = replay_target_fn

    rank = dist.get_rank() if dist.is_initialized() else 0
    if profile_ranks is None:
        profile_ranks = [0]

    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(warmup):
        fn()

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()

    begin_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    if rank in profile_ranks:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
    else:
        prof = nullcontext()

    with prof:
        torch.cuda._sleep(int(2e7))
        for i in range(rep):
            # flush L2 cache
            cache.zero_()
            begin_events[i].record()
            fn()
            end_events[i].record()
        torch.cuda.synchronize()

    latencies = [
        b.elapsed_time(e) for b, e in zip(begin_events, end_events, strict=False)
    ]

    if return_mode == "mean":
        return torch.tensor(latencies).mean().item() * 1000
    return torch.tensor(latencies).median().item() * 1000
