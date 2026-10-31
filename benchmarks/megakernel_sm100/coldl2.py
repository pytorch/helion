"""Cold-L2 GPU timing shared by every system in this benchmark.

Each timed call runs right after an L2 flush and is measured on the GPU (CUPTI,
through ``torch.profiler``): the span from the start of the call's first kernel to
the end of its last kernel.  The flush (``L2Flusher``) moves several times the
size of L2 through it and ends in a read, so L2 holds only clean lines of the
flush's buffer when the call starts and the call pays for no write-backs.  A
call is captured in a CUDA graph and replayed, so a multi-kernel call (a
framework's decoder layer) is not slowed down by host launch gaps, as in a
CUDA-graph decode step.

B200's L2 resists streaming eviction: lines a call reads more than once can
survive any streaming flush (a 2 GiB read included), so a call's re-read data
may start L2-warm.  Rotating through 8 deep copies of the inputs (each last
touched 8 calls earlier) instead made Helion's kernels 0.6-1.9 us faster and
FlashInfer's MLA 1 us slower, so the harness times one set of inputs.

The call's kernels are found by correlation id: the host-side launches inside the
call's ``coldl2_call:<name>`` annotation (a graph launch or kernel launches) link
to the device kernels they started, so the flush is never part of a span.

A process gets one profiling session: from its third ``torch.profiler`` session
on, CUPTI leaves multi-kernel CUDA graphs measurably slower (about 4 us on a
12-kernel decoder layer, and it persists after profiling stops), which would
penalize multi-kernel baselines against single-kernel megakernels.
``time_cold_many`` therefore times every call of a process in one session, with
the calls interleaved, and a second session raises.

Even one session adds a little to each span (CUPTI's per-kernel bookkeeping:
more on a many-kernel graph).  Before profiling, ``time_cold_many`` also measures
each call without the profiler (``loop_us``): cudaEvents around N back-to-back
(reset, flush, call) iterations, minus the same loop with a one-kernel graph in
place of the call.  It is the call's increment over a trivial graph, so it is
offset from the span by about one tiny kernel, the same for every system.

Only ``torch`` is required (any version this benchmark's environments use), so the
same module times Helion, vLLM, SGLang, Mirage and ThunderKittens.
"""

from __future__ import annotations

import json
import operator
import os
from pathlib import Path
import statistics
import tempfile
from typing import TYPE_CHECKING
from typing import Any

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function

if TYPE_CHECKING:
    from collections.abc import Callable

# B200: 126 MB of L2.  Reading 4x that leaves none of the timed call's data.
FLUSH_BYTES = 512 << 20
_ANNOTATION = "coldl2_call:"
_SESSIONS = 0
# Host-side calls that start device work, by name (runtime and driver APIs).
_LAUNCHES = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cudaLaunchCooperativeKernel",
    "cudaGraphLaunch",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cuLaunchCooperativeKernel",
    "cuGraphLaunch",
    "cudaMemcpyAsync",
    "cudaMemsetAsync",
)
_DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


class L2Flusher:
    """Evicts L2: copies an int32 buffer of ``nbytes`` into an int64 buffer twice
    its size, then sums that (2.5x ``nbytes`` of DRAM traffic, ending in a read).

    L2 then holds only clean lines of the int64 buffer: the call pays for no
    write-backs.  A pure read of the buffer (an FP32 sum) leaves the next call
    up to 2.5 us slower for a few us (and the slowdown differs between kernels);
    after this flush a call starts as it would after a 50 us idle gap.
    """

    def __init__(
        self, device: torch.device | str | None = None, nbytes: int = FLUSH_BYTES
    ) -> None:
        self.buffer = torch.ones(nbytes // 4, dtype=torch.int32, device=device)
        self.wide = torch.empty(nbytes // 4, dtype=torch.int64, device=device)
        self.sink = torch.empty((), dtype=torch.int64, device=device)

    def __call__(self) -> None:
        self.wide.copy_(self.buffer)
        torch.sum(self.wide, dim=0, out=self.sink)


def graph_of(fn: Callable[[], object], warmup: int = 3) -> torch.cuda.CUDAGraph:
    """Capture ``fn`` as a CUDA graph after ``warmup`` eager calls.

    The warmup and the capture run on one side stream, as a framework warms up
    and captures on its stream: state a launcher caches per stream (Helion's
    retained launch state) is created by the warmup instead of being captured
    as a zero-fill that every replay repeats.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    return graph


class Recorded:
    """A call that keeps its latest outputs.

    Time a ``Recorded`` call and its ``outputs`` are those of the graph's last
    replay afterwards, so a runner can check them again after the timed reps.
    """

    def __init__(self, call: Callable[[], Any]) -> None:
        self.call = call
        self.outputs: Any = None

    def __call__(self) -> object:
        self.outputs = self.call()
        return self.outputs


def _spans_from_trace(path: str) -> dict[str, list[float]]:
    """Per-call device spans (us) by call name, in call order."""
    events = json.loads(Path(path).read_text())["traceEvents"]
    calls = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") == "user_annotation"
            and e.get("name", "").startswith(_ANNOTATION)
        ),
        key=operator.itemgetter("ts"),
    )
    launches = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
        and e.get("name") in _LAUNCHES
    ]
    device = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in _DEVICE_CATEGORIES:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                device.setdefault(corr, []).append(e)
    spans: dict[str, list[float]] = {}
    for call in calls:
        lo, hi = call["ts"], call["ts"] + call["dur"]
        corrs = {
            e["args"]["correlation"]
            for e in launches
            if e["pid"] == call["pid"]
            and e["tid"] == call["tid"]
            and lo <= e["ts"] <= hi
        }
        work = [k for c in corrs for k in device.get(c, ())]
        if not work:
            raise RuntimeError("a timed call launched no device work the profiler saw")
        spans.setdefault(call["name"][len(_ANNOTATION) :], []).append(
            max(k["ts"] + k["dur"] for k in work) - min(k["ts"] for k in work)
        )
    return spans


def _loop_us(
    run: Callable[[], object],
    reset: Callable[[], object],
    flusher: Callable[[], object],
    null: Callable[[], object],
    iters: int = 200,
    trials: int = 7,
) -> float:
    """Profiler-free per-call time: median over trials of (loop of run) - (loop of null)."""

    def loop(fn: Callable[[], object]) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            reset()
            flusher()
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / iters

    loop(run)
    loop(null)
    return statistics.median(loop(run) - loop(null) for _ in range(trials))


def _stats(spans: list[float]) -> dict[str, object]:
    ordered = sorted(spans)
    return {
        "median_us": statistics.median(spans),
        "p10_us": ordered[len(ordered) // 10],
        "p90_us": ordered[(len(ordered) * 9) // 10],
        "reps": len(spans),
        "spans_us": [round(s, 3) for s in spans],
    }


def time_cold_many(
    calls: dict[str, Callable[[], object]],
    *,
    reps: int = 200,
    warmup: int = 5,
    graph: bool = True,
    flusher: Callable[[], object] | None = None,
    resets: dict[str, Callable[[], object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Cold-L2 device time of each call: median (and spread) of ``reps`` spans in us.

    Every call is captured first; then each of ``reps`` rounds runs every call
    once (the order rotates between rounds), each right after its own flush.
    With ``graph`` (default) a call is captured once and its graph replayed;
    pass ``graph=False`` for a call that manages its own launches (it must still
    run all of its device work on the current stream's device).  ``resets[name]``
    restores state a call updates in place (a residual, a KV-cache row); it runs
    before the flush, so its writes are evicted too.  ``flusher`` defaults to an
    ``L2Flusher``; a distributed caller passes one that also aligns the ranks.
    Each result also has ``loop_us``, the profiler-free cross-check (see the
    module docstring), measured before the profiled reps.
    """
    global _SESSIONS
    if _SESSIONS:
        raise RuntimeError(
            "coldl2 times once per process (see the module docstring); pass every "
            "call to one time_cold_many"
        )
    _SESSIONS += 1
    flusher = flusher or L2Flusher(torch.device("cuda", torch.cuda.current_device()))
    runs = {name: graph_of(fn).replay if graph else fn for name, fn in calls.items()}
    resets = resets or {}
    names = list(runs)
    for _ in range(warmup):
        for name in names:
            resets.get(name, lambda: None)()
            flusher()
            runs[name]()
    torch.cuda.synchronize()
    counter = torch.zeros(1, device="cuda")
    null = graph_of(lambda: counter.add_(1)).replay
    loops = {
        name: _loop_us(runs[name], resets.get(name, lambda: None), flusher, null)
        for name in names
    }
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for rep in range(reps):
            for index in range(len(names)):
                name = names[(rep + index) % len(names)]
                resets.get(name, lambda: None)()
                flusher()
                with record_function(_ANNOTATION + name):
                    runs[name]()
                torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        spans = _spans_from_trace(path)
    finally:
        os.unlink(path)
    for name in names:
        if len(spans.get(name, ())) != reps:
            raise RuntimeError(f"{name}: expected {reps} spans in the trace")
    return {name: {**_stats(spans[name]), "loop_us": loops[name]} for name in names}


def time_cold(
    fn: Callable[[], object],
    *,
    reps: int = 200,
    warmup: int = 5,
    graph: bool = True,
    flusher: Callable[[], object] | None = None,
    reset: Callable[[], object] | None = None,
) -> dict[str, object]:
    """``time_cold_many`` of one call."""
    return time_cold_many(
        {"call": fn},
        reps=reps,
        warmup=warmup,
        graph=graph,
        flusher=flusher,
        resets={"call": reset} if reset else None,
    )["call"]


def gpu_info() -> dict[str, object]:
    """Name, SM count, versions, and (when NVML is available) clock and temperature."""
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    info: dict[str, object] = {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        info["sm_clock_mhz"] = torch.cuda.clock_rate(device)
        info["temperature_c"] = torch.cuda.temperature(device)
    except Exception:  # NVML is optional; the timings do not depend on it.
        pass
    return info
