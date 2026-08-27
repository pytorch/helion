"""Shared CUDA graph and benchmark helpers for standalone probes."""

from __future__ import annotations

import hashlib
import os
import random
import statistics
import subprocess
from typing import Callable
from typing import TypeVar

import torch

T = TypeVar("T")


def make_l2_cache_clearer() -> Callable[[], None] | None:
    """Return Triton's L2 flush primitive when cold-cache timing is requested."""
    if os.environ.get("MEGAKERNEL_CLEAR_L2") != "1":
        return None
    from triton import runtime

    active = runtime.driver.active  # type: ignore[attr-defined]
    cache = active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]

    def clear() -> None:
        active.clear_cache(cache)  # type: ignore[attr-defined]

    return clear


def lowered_triton_summary(source: str) -> dict[str, object]:
    """Return a compact, reproducible fingerprint of lowered Triton code."""
    lines = source.splitlines()
    constexpr_indices = [
        index
        for index, line in enumerate(lines)
        if line.startswith("_BLOCK_SIZE_") and " = tl.constexpr(" in line
    ]
    normalized_lines = list(lines)
    for index, declaration in zip(
        constexpr_indices,
        sorted(lines[index] for index in constexpr_indices),
        strict=True,
    ):
        normalized_lines[index] = declaration
    normalized = "\n".join(normalized_lines)
    if source.endswith("\n"):
        normalized += "\n"
    return {
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "line_count": len(lines),
        "jit_function_count": sum(line.strip() == "@triton.jit" for line in lines),
        "dependency_wait_loop_count": sum(
            line.lstrip().startswith("while tile_dependency_") for line in lines
        ),
        "atomic_add_count": source.count("tl.atomic_add("),
        "atomic_exchange_count": source.count("tl.atomic_xchg("),
        "debug_barrier_count": source.count("tl.debug_barrier("),
        "dot_count": source.count("tl.dot("),
        "load_count": source.count("tl.load("),
        "store_count": source.count("tl.store("),
    }


def flush_l2_exact(*, flush_mib: int = 256) -> None:
    """Flush exactly ``flush_mib`` MiB through Triton's benchmark buffer."""
    import triton

    driver = triton.runtime.driver.active
    flush_buffer = driver.get_empty_cache_for_benchmark()
    expected_bytes = flush_mib * 1024 * 1024
    actual_bytes = flush_buffer.numel() * flush_buffer.element_size()
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Triton L2 flush buffer is {actual_bytes} bytes, expected {expected_bytes}"
        )
    driver.clear_cache(flush_buffer)
    driver.get_device_interface().synchronize()


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


def capture_with_reset(
    fn: Callable[[], T],
    reset: Callable[[], None],
    *,
    warmups: int = 3,
) -> tuple[torch.cuda.CUDAGraph, T]:
    """Capture only ``fn`` while restoring mutable inputs between warmups."""
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(warmups):
            reset()
            output = fn()
        reset()
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
    clear_l2 = make_l2_cache_clearer()
    for sample in range(repeats):
        order = names[sample % len(names) :] + names[: sample % len(names)]
        for name in order:
            if clear_l2 is None:
                start.record()
                for _ in range(batch_replays):
                    entries[name]()
                end.record()
                end.synchronize()
                elapsed_us = start.elapsed_time(end) * 1000.0 / batch_replays
            else:
                elapsed_us = 0.0
                for _ in range(batch_replays):
                    clear_l2()
                    torch.cuda.synchronize()
                    start.record()
                    entries[name]()
                    end.record()
                    end.synchronize()
                    elapsed_us += start.elapsed_time(end) * 1000.0
                elapsed_us /= batch_replays
            samples[name].append(elapsed_us)
    return {
        name: {
            "median_us": statistics.median(values),
            "mean_us": statistics.fmean(values),
            "p90_us": sorted(values)[min(len(values) - 1, int(0.9 * len(values)))],
        }
        for name, values in samples.items()
    }


def benchmark_graphs_cold_l2(
    entries: dict[str, tuple[Callable[[], object], Callable[[], None]]],
    repeats: int,
    *,
    flush_mib: int = 256,
    order_seed: int = 0,
) -> dict[str, dict[str, object]]:
    """Time each graph replay after an identical, verified L2 flush."""
    if not entries:
        return {}
    if repeats <= 0 or repeats % 2:
        raise ValueError("repeats must be positive and even")
    names = list(entries)
    rng = random.Random(order_seed)
    samples: dict[str, list[float]] = {name: [] for name in names}
    import triton

    driver = triton.runtime.driver.active
    flush_buffer = driver.get_empty_cache_for_benchmark()
    expected_bytes = flush_mib * 1024 * 1024
    actual_bytes = flush_buffer.numel() * flush_buffer.element_size()
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Triton L2 flush buffer is {actual_bytes} bytes, expected {expected_bytes}"
        )
    device_interface = driver.get_device_interface()
    flush_l2_exact(flush_mib=flush_mib)

    while len(samples[names[0]]) < repeats:
        order = list(names)
        rng.shuffle(order)
        pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        for name in (*order, *reversed(order)):
            replay, reset = entries[name]
            reset()
            driver.clear_cache(flush_buffer)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            replay()
            end.record()
            pending.append((name, start, end))
        device_interface.synchronize()
        for name, start, end in pending:
            samples[name].append(start.elapsed_time(end) * 1000.0)

    result: dict[str, dict[str, object]] = {}
    for name, values in samples.items():
        values = values[:repeats]
        ordered = sorted(values)
        result[name] = {
            "median_us": statistics.median(values),
            "mean_us": statistics.fmean(values),
            "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_us": ordered[0],
            "p10_us": ordered[int(0.1 * (len(ordered) - 1))],
            "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
            "max_us": ordered[-1],
            "samples_us": values,
        }
    return result


def profile_cuda_timeline(
    fn: Callable[[], object],
) -> list[dict[str, float | int | str]]:
    """Return one replay's CUDA activities with normalized start times."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        fn()
        torch.cuda.synchronize()
    events = [
        event
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    if not events:
        return []
    origin = min(event.time_range.start for event in events)
    return [
        {
            "name": event.name,
            "start_us": event.time_range.start - origin,
            "end_us": event.time_range.end - origin,
            "duration_us": event.self_device_time_total,
            "stream": event.device_resource_id,
        }
        for event in sorted(events, key=lambda event: event.time_range.start)
    ]


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = (actual_f - expected_f).abs()
    denominator = actual_f.norm() * expected_f.norm()
    cosine = (
        float((actual_f.flatten() @ expected_f.flatten() / denominator).item())
        if float(denominator.item()) != 0.0
        else 1.0
    )
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "cosine": cosine,
    }


def clone_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in tensors.items()}


def gpu_snapshot() -> dict[str, str | int]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible_gpu(),
            (
                "--query-gpu=name,uuid,temperature.gpu,pstate,clocks.sm,"
                "clocks.mem,power.draw,memory.used,utilization.gpu"
            ),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [field.strip() for field in output.split(",")]
    keys = (
        "name",
        "uuid",
        "temperature_c",
        "pstate",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_w",
        "memory_used_mb",
        "utilization_pct",
    )
    integer_keys = {
        "temperature_c",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "memory_used_mb",
        "utilization_pct",
    }
    return {
        key: int(value) if key in integer_keys else value
        for key, value in zip(keys, fields, strict=True)
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
