from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace

import pytest

from test.test_profiler_timing import _trace

from helion.autotuner import profiler_timing as subject


def _shift(
    trace: subject.ProfilerTrace, *, gpu_offset: int, drift: int = 0
) -> subject.ProfilerTrace:
    # Apply the same drift to a fixture call's clear, kernels and entire GPU
    # annotation, including the gaps between its multiple kernels.
    gpu_groups = sorted(e.start_ns for e in trace.events if e.name == "clear_kernel")
    shifted = []
    for event in trace.events:
        offset = 2_000_000
        if event.device_type == "DeviceType.CUDA":
            offset += (
                gpu_offset + (bisect_right(gpu_groups, event.start_ns) - 1) * drift
            )
        shifted.append(
            replace(
                event, start_ns=event.start_ns + offset, end_ns=event.end_ns + offset
            )
        )
    return replace(trace, events=tuple(shifted))


@pytest.mark.parametrize(
    "graph,annotations,kernels",
    [(False, False, 1), (False, True, 3), (True, True, 1)],
)
@pytest.mark.parametrize("gpu_offset", [-1_000_000, 0, 1_000_000])
def test_gpu_epoch_offset_preserves_correlated_work(
    graph: bool, annotations: bool, kernels: int, gpu_offset: int
) -> None:
    trace = _trace(graph=graph, annotations=annotations, kernels=kernels)
    expected = subject.attribute_profiler_trace(trace)
    shifted = _shift(trace, gpu_offset=gpu_offset)
    assert subject.attribute_profiler_trace(shifted) == expected
    assert [e.end_ns - e.start_ns for e in shifted.events] == [
        e.end_ns - e.start_ns for e in trace.events
    ]


@pytest.mark.parametrize(
    "graph,annotations,kernels",
    [(False, False, 1), (False, True, 3), (True, True, 1)],
)
def test_monotone_cross_clock_drift_preserves_gpu_intervals(
    graph: bool, annotations: bool, kernels: int
) -> None:
    trace = _trace(count=4, graph=graph, annotations=annotations, kernels=kernels)
    shifted = _shift(trace, gpu_offset=-1_000_000, drift=7)
    assert subject.attribute_profiler_trace(
        shifted
    ) == subject.attribute_profiler_trace(trace)


@pytest.mark.parametrize(
    "case",
    [
        "annotation_mismatch",
        "gpu_overlap",
        "gpu_reordering",
        "missing_runtime",
        "wrong_external",
        "wrong_stream",
        "nonpositive_duration",
    ],
)
def test_clock_offset_does_not_authorize_invalid_work(case: str) -> None:
    trace = _shift(_trace(graph=True), gpu_offset=-1_000_000)
    events = list(trace.events)
    kernels = [
        i
        for i, e in enumerate(events)
        if e.device_type == "DeviceType.CUDA" and e.name == "arbitrary_entry_0"
    ]
    first, second = kernels
    kernel = events[first]
    if case == "annotation_mismatch":
        events[first] = replace(
            kernel, start_ns=kernel.start_ns - 1, end_ns=kernel.end_ns - 1
        )
    elif case == "gpu_overlap":
        clear = next(e for e in events if e.name == "clear_kernel")
        events[first] = replace(kernel, start_ns=clear.end_ns - 1)
    elif case == "gpu_reordering":
        events[first] = replace(
            kernel, start_ns=events[second].start_ns, end_ns=events[second].end_ns
        )
        events[second] = replace(
            events[second], start_ns=kernel.start_ns, end_ns=kernel.end_ns
        )
    elif case == "missing_runtime":
        events = [
            e
            for e in events
            if not (
                e.activity_type == "cuda_runtime"
                and e.correlation_id == kernel.correlation_id
            )
        ]
    elif case == "wrong_external":
        events[first] = replace(
            kernel, linked_correlation_id=events[second].linked_correlation_id
        )
    elif case == "wrong_stream":
        events[first] = replace(kernel, device_resource_id=8)
    elif case == "nonpositive_duration":
        events[first] = replace(kernel, end_ns=kernel.start_ns)
    with pytest.raises(subject.ProfilerTimingCapabilityError):
        subject.attribute_profiler_trace(replace(trace, events=tuple(events)))
