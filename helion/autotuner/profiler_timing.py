"""Opt-in kernel-work collection, independent of autotuner measurement policy.

This module does not estimate repetitions, warm up, allocate a cache-clear buffer,
capture a CUDA graph, or select a search score. Its caller supplies the already
resolved call count, clear/synchronization operations, and compiled-code identity.
The metric is summed kernel execution time per call, not end-to-end latency.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import itertools
from typing import TYPE_CHECKING
from typing import Protocol

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

CALLABLE_WINDOW_NAME = "__helion_profiler_timing_callable__"
_VERSION = 1
_CPU = "DeviceType.CPU"
_CUDA = "DeviceType.CUDA"
_LAUNCHES = frozenset(("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch"))
_KERNELS = frozenset(("kernel", "concurrent_kernel"))


class ProfilerTimingCapabilityError(RuntimeError):
    """The captured work cannot establish this timing protocol; no score exists."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfilerTimingCapabilityError(message)


def _positive_int(value: int, name: str) -> None:
    _require(type(value) is int and value > 0, f"{name} must be a positive integer")


@dataclass(frozen=True)
class ProfilerTimingPolicy:
    """Versioned capture protocol, not a duration or repetition estimator."""

    chunk_size: int = 128
    version: int = _VERSION

    def __post_init__(self) -> None:
        _positive_int(self.chunk_size, "chunk_size")
        _require(
            type(self.version) is int and self.version == _VERSION, "Unknown version"
        )


_DEFAULT_POLICY = ProfilerTimingPolicy()


@dataclass(frozen=True)
class CallableIdentity:
    """Caller-owned compiled callable/source identifiers, never a kernel allowlist."""

    callable_key: str
    source_digest: str

    def __post_init__(self) -> None:
        for value in (self.callable_key, self.source_digest):
            _require(
                type(value) is str and bool(value), "Empty callable/source identity"
            )


@dataclass(frozen=True)
class ProfilerEvent:
    """The exact Kineto fields needed for independent launch attribution."""

    name: str
    device_type: str
    activity_type: str
    start_ns: int
    end_ns: int
    correlation_id: int
    linked_correlation_id: int
    device_index: int
    device_resource_id: int

    def __post_init__(self) -> None:
        for value in (self.name, self.device_type, self.activity_type):
            _require(type(value) is str and bool(value), "Unsupported event string")
        for value in (
            self.start_ns,
            self.end_ns,
            self.correlation_id,
            self.linked_correlation_id,
            self.device_index,
            self.device_resource_id,
        ):
            _require(type(value) is int, "Event fields require exact integer values")
        _require(0 <= self.start_ns <= self.end_ns, "Invalid event interval")
        _require(
            self.correlation_id >= 0 and self.linked_correlation_id >= 0, "Invalid ID"
        )


class _KinetoEvent(Protocol):
    def name(self) -> str: ...
    def device_type(self) -> object: ...
    def activity_type(self) -> str: ...
    def start_ns(self) -> int: ...
    def end_ns(self) -> int: ...
    def correlation_id(self) -> int: ...
    def linked_correlation_id(self) -> int: ...
    def device_index(self) -> int: ...
    def device_resource_id(self) -> int: ...


def read_profiler_events(events: Iterable[_KinetoEvent]) -> tuple[ProfilerEvent, ...]:
    """Reject an unsupported installed profiler schema rather than infer fields."""
    try:
        return tuple(
            ProfilerEvent(
                event.name(),
                str(event.device_type()),
                event.activity_type(),
                event.start_ns(),
                event.end_ns(),
                event.correlation_id(),
                event.linked_correlation_id(),
                event.device_index(),
                event.device_resource_id(),
            )
            for event in events
        )
    except (AttributeError, TypeError) as error:
        raise ProfilerTimingCapabilityError(
            "Unsupported Kineto event schema"
        ) from error


@dataclass(frozen=True)
class ProfilerTrace:
    """A bounded, immutable raw chunk; sinks receive it before attribution."""

    identity: CallableIdentity
    call_count: int
    events: tuple[ProfilerEvent, ...]


@dataclass(frozen=True)
class ProfilerChunk:
    call_count: int
    total_ns: int
    kernel_count: int
    launch_count: int
    device_index: int
    stream_id: int
    # Runtime API and ordered native entries per launch, discovered from the trace.
    work_signature: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class ProfilerTimingObservation:
    policy: ProfilerTimingPolicy
    identity: CallableIdentity
    chunks: tuple[ProfilerChunk, ...]

    @property
    def call_count(self) -> int:
        return sum(chunk.call_count for chunk in self.chunks)

    @property
    def total_ns(self) -> int:
        return sum(chunk.total_ns for chunk in self.chunks)

    @property
    def mean_ms(self) -> float:
        """Weighted mean kernel-work milliseconds; no fabricated quantiles."""
        return self.total_ns / self.call_count / 1_000_000


def attribute_profiler_trace(trace: ProfilerTrace) -> ProfilerChunk:
    """Attribute every measured launch through its *runtime* correlation ID.

    External IDs are checked separately and cannot select device work. Eager
    calls may issue multiple kernels; their durations are summed. The initial
    graph capability is one launch/one kernel with an exact GPU window witness.
    Multi-kernel graphs, transfers inside a call, and cross-stream work reject.
    """
    _positive_int(trace.call_count, "call_count")
    events = trace.events
    windows = sorted(
        (
            event
            for event in events
            if event.device_type == _CPU and event.name == CALLABLE_WINDOW_NAME
        ),
        key=lambda event: event.start_ns,
    )
    _require(len(windows) == trace.call_count, "Incorrect callable window count")
    _require(
        all(
            event.activity_type == "user_annotation" and event.end_ns > event.start_ns
            for event in windows
        ),
        "Invalid callable window",
    )
    _require(
        all(
            left.end_ns <= right.start_ns for left, right in itertools.pairwise(windows)
        ),
        "Overlapping callable windows",
    )
    _require(
        len({event.device_resource_id for event in windows}) == 1,
        "Multiple CPU threads",
    )
    starts = [event.start_ns for event in windows]

    def owner(event: ProfilerEvent, *, runtime: bool = False) -> int | None:
        index = bisect_right(starts, event.start_ns) - 1
        if index >= 0:
            window = windows[index]
            if event.start_ns < window.end_ns:
                contained = (
                    event.end_ns <= window.end_ns
                    and event.device_resource_id == window.device_resource_id
                )
                if runtime:
                    _require(contained, "Runtime launch crosses callable thread/window")
                return index if contained else None
        if runtime and index + 1 < len(windows):
            _require(
                event.end_ns <= windows[index + 1].start_ns, "Runtime overlaps callable"
            )
        return None

    runtime_ids: dict[int, ProfilerEvent] = {}
    external_ids: dict[int, ProfilerEvent] = {}
    for event in events:
        if event.device_type != _CPU:
            continue
        _require(
            not (
                event.activity_type.startswith("cuda_")
                and event.activity_type != "cuda_runtime"
                and owner(event, runtime=True) is not None
            ),
            "Unsupported CUDA activity inside callable",
        )
        if event.activity_type == "cuda_runtime":
            target = runtime_ids
        elif event.activity_type in ("cpu_op", "user_annotation"):
            target = external_ids
        else:
            continue
        _require(
            event.correlation_id > 0 and event.correlation_id not in target,
            "Ambiguous correlation namespace",
        )
        target[event.correlation_id] = event

    window_ids = {event.correlation_id: index for index, event in enumerate(windows)}
    _require(len(window_ids) == len(windows), "Duplicate callable window ID")
    launch_owners: dict[int, int | None] = {}

    def check_external(event: ProfilerEvent, expected: int | None) -> None:
        if event.linked_correlation_id:
            linked = external_ids.get(event.linked_correlation_id)
            _require(linked is not None, "Missing linked external ID")
            assert linked is not None
            _require(owner(linked) == expected, "External/runtime owner disagreement")

    for correlation, event in runtime_ids.items():
        index = owner(event, runtime=True)
        launch_owners[correlation] = index
        check_external(event, index)
        if index is not None:
            _require(event.end_ns > event.start_ns, "Nonpositive runtime interval")
            _require(
                not event.name.startswith(
                    ("cudaMemcpy", "cudaMemset", "cuMemcpy", "cuMemset")
                ),
                "Transfer inside callable is unsupported",
            )
            if event.name.startswith(
                ("cudaLaunch", "cudaGraphLaunch", "cuLaunch", "cuGraphLaunch")
            ):
                _require(event.name in _LAUNCHES, "Unsupported CUDA launch API")

    work: dict[int, list[ProfilerEvent]] = {}
    annotations: dict[int, ProfilerEvent] = {}
    seen: set[ProfilerEvent] = set()
    all_device_work: list[ProfilerEvent] = []
    for event in events:
        if event.device_type != _CUDA:
            _require(event.device_type == _CPU, "Unsupported device activity")
            continue
        if event.activity_type == "gpu_user_annotation":
            if event.name == CALLABLE_WINDOW_NAME:
                index = window_ids.get(event.correlation_id)
                _require(
                    index is not None and index not in annotations,
                    "Ambiguous GPU annotation",
                )
                assert index is not None
                annotations[index] = event
            continue
        _require(event not in seen, "Duplicate device work")
        seen.add(event)
        _require(event.end_ns > event.start_ns, "Nonpositive device work")
        launch = runtime_ids.get(event.correlation_id)
        _require(launch is not None, "Device work has no runtime launch")
        assert launch is not None
        _require(
            launch.device_resource_id == windows[0].device_resource_id,
            "Foreign-thread device work",
        )
        # CUPTI interpolates GPU timestamps into CPU time; cross-clock ordering
        # is not exact. Runtime correlation and CPU-window ownership establish
        # causality; GPU witnesses and stream ordering are checked below.
        index = launch_owners[event.correlation_id]
        check_external(event, index)
        all_device_work.append(event)
        if index is None:
            continue  # The caller's clear is deliberately outside callable windows.
        _require(launch.name in _LAUNCHES, "Device work uses an unsupported launch")
        _require(event.activity_type in _KERNELS, "Non-kernel work inside callable")
        work.setdefault(event.correlation_id, []).append(event)

    calls: list[list[tuple[ProfilerEvent, list[ProfilerEvent]]]] = [[] for _ in windows]
    for correlation, launch in runtime_ids.items():
        index = launch_owners[correlation]
        if index is not None and launch.name in _LAUNCHES:
            kernels = sorted(
                work.get(correlation, []), key=lambda event: event.start_ns
            )
            _require(
                len(kernels) == 1, "Each admitted launch must have exactly one kernel"
            )
            calls[index].append((launch, kernels))

    signatures = []
    ordered_work: list[ProfilerEvent] = []
    for index, launches in enumerate(calls):
        launches.sort(key=lambda item: item[0].start_ns)
        _require(bool(launches), "Callable has no complete kernel work")
        if any(launch.name == "cudaGraphLaunch" for launch, _ in launches):
            _require(
                len(launches) == 1 and index in annotations,
                "Ambiguous graph launch or missing GPU witness",
            )
        kernels = [kernel for _, items in launches for kernel in items]
        _require(
            all(a.end_ns <= b.start_ns for a, b in itertools.pairwise(kernels)),
            "Overlapping or reordered kernels",
        )
        signatures.append(
            tuple(
                (launch.name, tuple(event.name for event in items))
                for launch, items in launches
            )
        )
        ordered_work.extend(kernels)
        annotation = annotations.get(index)
        if annotation is not None:
            _require(
                (
                    annotation.start_ns,
                    annotation.end_ns,
                    annotation.device_index,
                    annotation.device_resource_id,
                )
                == (
                    kernels[0].start_ns,
                    kernels[-1].end_ns,
                    kernels[0].device_index,
                    kernels[0].device_resource_id,
                ),
                "GPU annotation does not match actual kernel interval",
            )

    _require(
        not annotations or len(annotations) == len(windows), "Partial GPU annotations"
    )
    _require(
        all(signature == signatures[0] for signature in signatures),
        "Per-call work changed",
    )
    streams = {
        (event.device_index, event.device_resource_id) for event in all_device_work
    }
    _require(len(streams) == 1, "Work/clear spans multiple devices or streams")
    device, stream = streams.pop()
    _require(device >= 0 and stream >= 0, "Unknown device/stream")
    all_device_work.sort(
        key=lambda event: (runtime_ids[event.correlation_id].start_ns, event.start_ns)
    )
    _require(
        all(a.end_ns <= b.start_ns for a, b in itertools.pairwise(all_device_work)),
        "Clear/work ordering or stream serialization is incomplete",
    )
    _require(
        all(a.end_ns <= b.start_ns for a, b in itertools.pairwise(ordered_work)),
        "Calls have overlapping or reordered device work",
    )
    return ProfilerChunk(
        trace.call_count,
        sum(event.end_ns - event.start_ns for event in ordered_work),
        len(ordered_work),
        sum(len(call) for call in calls),
        device,
        stream,
        signatures[0],
    )


def collect_profiler_timing(
    fn: Callable[[], object],
    *,
    identity: CallableIdentity,
    sample_count: int,
    clear: Callable[[], object],
    synchronize: Callable[[], object],
    policy: ProfilerTimingPolicy = _DEFAULT_POLICY,
    trace_sink: Callable[[ProfilerTrace], None] | None = None,
) -> ProfilerTimingObservation:
    """Capture exactly ``sample_count`` calls; all durations returned are means.

    Caller setup/warmup must already have finished. Clear runs outside each CPU
    window, on the same stream as work. Outputs are retained until chunk-end
    synchronization, including on a callable exception. No global observer is
    replaced, and no event-based or annotation-only fallback exists.
    """
    _positive_int(sample_count, "sample_count")
    chunks: list[ProfilerChunk] = []
    remaining = sample_count
    while remaining:
        _require(
            not torch.autograd._profiler_enabled(),
            "Cannot collect timing with an existing active profiler",
        )
        count = min(remaining, policy.chunk_size)
        outputs: list[object] = []
        synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
        ) as captured:
            try:
                for _ in range(count):
                    clear()
                    with torch.profiler.record_function(CALLABLE_WINDOW_NAME):
                        outputs.append(fn())
            finally:
                synchronize()
        # The explicit synchronization above retires every output's async use.
        outputs.clear()
        profiler = captured.profiler
        _require(
            profiler is not None and profiler.kineto_results is not None,
            "Missing Kineto capture",
        )
        assert profiler is not None and profiler.kineto_results is not None
        trace = ProfilerTrace(
            identity, count, read_profiler_events(profiler.kineto_results.events())
        )
        if trace_sink is not None:
            trace_sink(trace)
        chunk = attribute_profiler_trace(trace)
        if chunks:
            first = chunks[0]
            _require(
                (chunk.work_signature, chunk.device_index, chunk.stream_id)
                == (first.work_signature, first.device_index, first.stream_id),
                "Work identity/device/stream changed between chunks",
            )
        chunks.append(chunk)
        remaining -= count
    result = ProfilerTimingObservation(policy, identity, tuple(chunks))
    _require(result.call_count == sample_count, "Incomplete sample count")
    return result
