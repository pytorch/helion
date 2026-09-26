"""Shared four-rank setup and timing for distributed pretuned kernels."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import TypeVar

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from collections.abc import Callable


_T = TypeVar("_T")


def initialize(
    *, kernel_name: str, world_size: int, signal_pad_bytes: int
) -> tuple[int, int, dist.ProcessGroup]:
    """Initialize the fixed-size local NVSHMEM process group used by a probe."""
    if not torch.cuda.is_available():
        raise RuntimeError(f"{kernel_name} is pretuned only for NVIDIA SM100")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    actual_world_size = int(os.environ["WORLD_SIZE"])
    if actual_world_size != world_size:
        raise RuntimeError(f"{kernel_name} requires TP={world_size}")
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError(f"{kernel_name} is pretuned only for NVIDIA SM100")
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("default process group was not initialized")
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem.set_backend("NVSHMEM")
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), signal_pad_bytes))
    return rank, local_rank, group


def _capture(launch: Callable[[], _T]) -> tuple[torch.cuda.CUDAGraph, _T]:
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.graph(graph, stream=stream):
            output = launch()
    torch.cuda.synchronize()
    dist.barrier()
    return graph, output


def _max_rank_elapsed_us(graph: torch.cuda.CUDAGraph, calls: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed = torch.tensor(
        start.elapsed_time(end) * 1000.0 / calls,
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def benchmark(
    launches: dict[str, Callable[[], _T]],
    *,
    validate: Callable[[str, _T], None],
) -> dict[str, tuple[float, float]]:
    """Return paired max-rank warm and cold-L2 medians in microseconds."""
    captures = {name: _capture(launch) for name, launch in launches.items()}
    for graph, _output in captures.values():
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    for name, (_graph, output) in captures.items():
        validate(name, output)
    cache_scrub = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    warm_samples = {name: [] for name in launches}
    cold_samples = {name: [] for name in launches}
    names = tuple(launches)
    for trial in range(31):
        order = names if trial % 2 == 0 else tuple(reversed(names))
        for name in order:
            warm_samples[name].append(_max_rank_elapsed_us(captures[name][0], 100))
            cache_scrub.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            cold_samples[name].append(_max_rank_elapsed_us(captures[name][0], 1))
    return {
        name: (
            float(torch.tensor(warm_samples[name]).median()),
            float(torch.tensor(cold_samples[name]).median()),
        )
        for name in names
    }
