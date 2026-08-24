"""Shared CUDA graph and benchmark helpers for standalone probes."""

from __future__ import annotations

import os
import statistics
import subprocess
from typing import Callable
from typing import TypeVar

import torch

T = TypeVar("T")


def capture(fn: Callable[[], T]) -> tuple[torch.cuda.CUDAGraph, T]:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            output = fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = fn()
    torch.cuda.synchronize()
    return graph, output


def benchmark_interleaved(
    entries: dict[str, Callable[[], object]],
    repeats: int,
    batch_replays: int,
) -> dict[str, dict[str, float]]:
    samples = {name: [] for name in entries}
    names = list(entries)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample in range(repeats):
        order = names[sample % len(names) :] + names[: sample % len(names)]
        for name in order:
            start.record()
            for _ in range(batch_replays):
                entries[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0 / batch_replays)
    return {
        name: {
            "median_us": statistics.median(values),
            "mean_us": statistics.fmean(values),
            "p90_us": sorted(values)[min(len(values) - 1, int(0.9 * len(values)))],
        }
        for name, values in samples.items()
    }


def visible_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if "," in visible:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one idle GPU")
    return visible


def visible_gpu_pids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible_gpu(),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}


def require_idle_visible_gpu() -> None:
    visible = visible_gpu()
    memory_limit = int(os.environ.get("MEGAKERNEL_IDLE_MEMORY_LIMIT_MB", "256"))
    pids = visible_gpu_pids()
    if pids:
        raise RuntimeError(f"GPU {visible} has compute processes {sorted(pids)}")
    state = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible,
            "--query-gpu=utilization.gpu,utilization.memory,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gpu_util, memory_util, memory_used = (
        int(field.strip()) for field in state.split(",")
    )
    if gpu_util != 0 or memory_util != 0 or memory_used > memory_limit:
        raise RuntimeError(f"GPU {visible} is not idle: {state}")


def make_fp8_random(shape: tuple[int, ...], scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale).to(
        torch.float8_e4m3fn
    )
