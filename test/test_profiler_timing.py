from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from dataclasses import fields
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import weakref

import pytest
import torch

from helion.autotuner import profiler_timing as subject

IDENTITY = subject.CallableIdentity("compiled-fixture", "source-fixture-digest")


def _event(
    name: str,
    activity: str,
    start: int,
    end: int,
    correlation: int,
    *,
    cuda: bool = False,
    linked: int = 0,
    resource: int | None = None,
) -> subject.ProfilerEvent:
    return subject.ProfilerEvent(
        name,
        "DeviceType.CUDA" if cuda else "DeviceType.CPU",
        activity,
        start,
        end,
        correlation,
        linked,
        0 if cuda else 42,
        (7 if cuda else 42) if resource is None else resource,
    )


def _trace(
    count: int = 2,
    *,
    graph: bool = False,
    kernels: int = 1,
    annotations: bool = True,
    duration: int = 10,
) -> subject.ProfilerTrace:
    events = []
    for index in range(count):
        base = index * 1000
        corr = index * 100
        # The outside external ID deliberately collides with an inside runtime ID.
        events.extend(
            (
                _event("clear", "cpu_op", base + 1, base + 9, corr + 20),
                _event(
                    "cudaLaunchKernel",
                    "cuda_runtime",
                    base + 2,
                    base + 3,
                    corr + 5,
                    linked=corr + 20,
                ),
                _event(
                    "clear_kernel",
                    "kernel",
                    base + 50,
                    base + 60,
                    corr + 5,
                    cuda=True,
                    linked=corr + 20,
                ),
                _event(
                    subject.CALLABLE_WINDOW_NAME,
                    "user_annotation",
                    base + 100,
                    base + 200,
                    corr + 10,
                ),
            )
        )
        for kernel in range(kernels):
            runtime_id = corr + 20 + kernel
            launch_start = base + 110 + kernel * 10
            gpu_start = base + 300 + kernel * 100
            if not graph or kernel == 0:
                events.append(
                    _event(
                        "cudaGraphLaunch" if graph else "cudaLaunchKernelExC",
                        "cuda_runtime",
                        launch_start,
                        launch_start + 2,
                        runtime_id,
                        linked=corr + 10,
                    )
                )
            events.append(
                _event(
                    f"arbitrary_entry_{kernel}",
                    "kernel",
                    gpu_start,
                    gpu_start + duration,
                    corr + 20 if graph else runtime_id,
                    cuda=True,
                    linked=corr + 10,
                )
            )
        if annotations:
            events.append(
                _event(
                    subject.CALLABLE_WINDOW_NAME,
                    "gpu_user_annotation",
                    base + 300,
                    base + 300 + (kernels - 1) * 100 + duration,
                    corr + 10,
                    cuda=True,
                )
            )
    return subject.ProfilerTrace(IDENTITY, count, tuple(events))


def _mutate(
    trace: subject.ProfilerTrace,
    predicate: Any,
    **changes: Any,
) -> subject.ProfilerTrace:
    found = False
    events = []
    for event in trace.events:
        if predicate(event) and not found:
            events.append(replace(event, **changes))
            found = True
        else:
            events.append(event)
    assert found
    return replace(trace, events=tuple(events))


@pytest.mark.parametrize(
    "graph,annotations,kernels",
    [
        (False, False, 1),
        (False, True, 1),
        (False, False, 3),
        (False, True, 3),
        (True, True, 1),
    ],
)
def test_exact_eager_and_graph_attribution(
    graph: bool, annotations: bool, kernels: int
) -> None:
    trace = _trace(graph=graph, annotations=annotations, kernels=kernels)
    chunk = subject.attribute_profiler_trace(trace)
    assert chunk.call_count == 2
    assert chunk.total_ns == 20 * kernels
    assert chunk.kernel_count == chunk.launch_count == 2 * kernels
    assert chunk.stream_id == 7 and chunk.device_index == 0
    # Native entry names are discovered, never supplied as an allowlist.
    assert tuple(
        name for _, entries in chunk.work_signature for name in entries
    ) == tuple(f"arbitrary_entry_{i}" for i in range(kernels))


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_ns", float("nan")),
        ("end_ns", float("inf")),
        ("correlation_id", True),
        ("device_resource_id", 7.0),
        ("start_ns", -1),
        ("linked_correlation_id", -1),
        ("name", ""),
    ],
)
def test_event_schema_rejects_nonfinite_or_coerced_fields(
    field: str, value: Any
) -> None:
    with pytest.raises(subject.ProfilerTimingCapabilityError):
        replace(_trace().events[0], **{field: value})


@pytest.mark.parametrize("value", [0, -1, True, 1.0, float("inf")])
def test_count_policy_rejects_bad_values(value: Any) -> None:
    with pytest.raises(subject.ProfilerTimingCapabilityError):
        subject.ProfilerTimingPolicy(chunk_size=value)
    with pytest.raises(subject.ProfilerTimingCapabilityError):
        subject.attribute_profiler_trace(replace(_trace(), call_count=value))


def test_immutable_versioned_records() -> None:
    with pytest.raises(subject.ProfilerTimingCapabilityError):
        subject.ProfilerTimingPolicy(version=2)
    with pytest.raises(subject.ProfilerTimingCapabilityError):
        subject.CallableIdentity("", "source")
    for name in ("callable_key", "source_digest"):
        with pytest.raises(FrozenInstanceError):
            setattr(IDENTITY, name, "changed")


@pytest.mark.parametrize(
    "case",
    [
        "missing_kernel",
        "missing_runtime",
        "duplicate_runtime",
        "duplicate_external",
        "duplicate_kernel",
        "missing_annotation",
        "partial_annotation",
        "wrong_annotation",
        "external_disagreement",
        "missing_external",
        "copy",
        "memset",
        "window_overlap",
        "window_thread",
        "crossing_launch",
        "missing_launch_work",
        "stream",
        "device",
        "foreign_thread",
        "zero_kernel",
        "entry_change",
        "unknown_launch",
        "unknown_device",
        "clear_overlap",
        "gpu_before_launch",
        "annotation_only",
    ],
)
def test_attribution_fail_closed(case: str) -> None:
    trace = _trace(graph=case == "missing_annotation")

    def kernel(e):
        return e.name == "arbitrary_entry_0"

    def launch(e):
        return e.name in ("cudaLaunchKernelExC", "cudaGraphLaunch")

    def annotation(e):
        return e.activity_type == "gpu_user_annotation"

    def window(e):
        return e.activity_type == "user_annotation"

    if case == "missing_kernel":
        trace = replace(trace, events=tuple(e for e in trace.events if not kernel(e)))
    elif case == "missing_runtime":
        trace = replace(trace, events=tuple(e for e in trace.events if not launch(e)))
    elif case in ("duplicate_runtime", "duplicate_external", "duplicate_kernel"):
        predicate = (
            launch
            if case == "duplicate_runtime"
            else window
            if case == "duplicate_external"
            else kernel
        )
        trace = replace(
            trace, events=(*trace.events, next(e for e in trace.events if predicate(e)))
        )
    elif case in ("missing_annotation", "partial_annotation"):
        trace = replace(
            trace,
            events=tuple(
                e
                for e in trace.events
                if not annotation(e)
                or (case == "partial_annotation" and e.start_ns < 1000)
            ),
        )
    elif case == "wrong_annotation":
        trace = _mutate(trace, annotation, start_ns=301, end_ns=311)
    elif case == "external_disagreement":
        trace = _mutate(trace, kernel, linked_correlation_id=20)
    elif case == "missing_external":
        trace = _mutate(trace, kernel, linked_correlation_id=9999)
    elif case in ("copy", "memset"):
        trace = _mutate(
            trace,
            kernel,
            activity_type="gpu_memcpy" if case == "copy" else "gpu_memset",
        )
    elif case == "window_overlap":
        trace = _mutate(trace, window, end_ns=1200)
    elif case == "window_thread":
        trace = _mutate(trace, window, device_resource_id=43)
    elif case == "crossing_launch":
        trace = _mutate(trace, launch, end_ns=201)
    elif case == "missing_launch_work":
        trace = replace(
            trace,
            events=(
                *trace.events,
                _event("cudaLaunchKernel", "cuda_runtime", 150, 152, 99),
            ),
        )
    elif case == "stream":
        trace = _mutate(trace, kernel, device_resource_id=8)
    elif case == "device":
        trace = _mutate(trace, kernel, device_index=1)
    elif case == "foreign_thread":
        trace = _mutate(
            trace, lambda e: e.name == "cudaLaunchKernel", device_resource_id=43
        )
    elif case == "zero_kernel":
        trace = _mutate(trace, kernel, end_ns=300)
    elif case == "entry_change":
        trace = _mutate(trace, kernel, name="different_entry")
    elif case == "unknown_launch":
        trace = _mutate(trace, launch, name="cudaLaunchUnknown")
    elif case == "unknown_device":
        trace = _mutate(trace, kernel, device_type="DeviceType.XPU")
    elif case == "clear_overlap":
        trace = _mutate(
            trace, lambda e: e.name == "clear_kernel", start_ns=299, end_ns=309
        )
    elif case == "gpu_before_launch":
        trace = _mutate(trace, kernel, start_ns=100, end_ns=110)
    elif case == "annotation_only":
        trace = replace(
            trace,
            events=tuple(
                e
                for e in trace.events
                if e.device_type != "DeviceType.CUDA" or annotation(e)
            ),
        )
    with pytest.raises(subject.ProfilerTimingCapabilityError):
        subject.attribute_profiler_trace(trace)


def test_multikernel_graph_is_explicitly_unsupported() -> None:
    with pytest.raises(subject.ProfilerTimingCapabilityError, match="exactly one"):
        subject.attribute_profiler_trace(_trace(graph=True, kernels=2))


def test_copy_runtime_inside_without_device_trace_rejects() -> None:
    trace = _trace()
    trace = replace(
        trace,
        events=(*trace.events, _event("cudaMemcpyAsync", "cuda_runtime", 150, 152, 99)),
    )
    with pytest.raises(subject.ProfilerTimingCapabilityError, match="Transfer"):
        subject.attribute_profiler_trace(trace)


@pytest.mark.parametrize(
    "name,activity",
    [
        ("cudaLaunchUnrecognized", "cuda_runtime"),
        ("cuLaunchKernel", "cuda_driver"),
        ("cuLaunchKernel", "cuda_runtime"),
    ],
)
def test_extra_unattributed_launch_cannot_hide_in_valid_call(
    name: str, activity: str
) -> None:
    trace = _trace()
    trace = replace(trace, events=(*trace.events, _event(name, activity, 150, 152, 99)))
    with pytest.raises(subject.ProfilerTimingCapabilityError, match="Unsupported CUDA"):
        subject.attribute_profiler_trace(trace)


@pytest.mark.parametrize(
    "activity,name",
    [("gpu_memcpy", "cudaMemcpyAsync"), ("gpu_memset", "cudaMemsetAsync")],
)
def test_outside_clear_transfer_is_not_measured(activity: str, name: str) -> None:
    trace = _trace()
    trace = replace(
        trace,
        events=tuple(
            replace(event, name=name)
            if event.name == "cudaLaunchKernel"
            else replace(event, activity_type=activity)
            if event.name == "clear_kernel"
            else event
            for event in trace.events
        ),
    )
    assert subject.attribute_profiler_trace(trace).total_ns == 20


def test_multiple_graph_launches_in_one_window_are_rejected() -> None:
    trace = _trace(kernels=2)
    trace = replace(
        trace,
        events=tuple(
            replace(event, name="cudaGraphLaunch")
            if event.name == "cudaLaunchKernelExC"
            else event
            for event in trace.events
        ),
    )
    with pytest.raises(subject.ProfilerTimingCapabilityError, match="Ambiguous graph"):
        subject.attribute_profiler_trace(trace)


class _RawEvent:
    def __init__(self, event: subject.ProfilerEvent) -> None:
        self.event = event

    def __getattr__(self, name: str) -> Any:
        return lambda: getattr(self.event, name)


class _Output:
    pass


def _capture_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    changed_chunk: bool = False,
    fail_call: bool = False,
) -> tuple[Any, dict[str, Any]]:
    state: dict[str, Any] = {
        "active": False,
        "window": False,
        "calls": 0,
        "clears": 0,
        "syncs": 0,
        "captures": 0,
        "output_refs": [],
        "traces": [],
    }

    @contextmanager
    def profile(**kwargs: Any) -> Any:
        assert kwargs["activities"] == [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        assert not state["active"]
        state["active"] = True
        state["captures"] += 1
        before_calls = state["calls"]
        holder = SimpleNamespace(profiler=None)
        try:
            yield holder
        finally:
            state["active"] = False
            count = state["calls"] - before_calls
            if count:
                trace = _trace(count, duration=10 if state["captures"] == 1 else 30)
                if changed_chunk and state["captures"] == 2:
                    trace = replace(
                        trace,
                        events=tuple(
                            replace(e, device_resource_id=8)
                            if e.device_type == "DeviceType.CUDA"
                            else e
                            for e in trace.events
                        ),
                    )
                holder.profiler = SimpleNamespace(
                    kineto_results=SimpleNamespace(
                        events=lambda: tuple(_RawEvent(e) for e in trace.events)
                    )
                )

    @contextmanager
    def window(name: str) -> Any:
        assert state["active"] and not state["window"]
        assert name == subject.CALLABLE_WINDOW_NAME
        state["window"] = True
        try:
            yield
        finally:
            state["window"] = False

    def clear() -> None:
        assert state["active"] and not state["window"]
        state["clears"] += 1

    def call() -> object:
        assert state["active"] and state["window"]
        if fail_call and state["calls"] == 1:
            raise LookupError("callable failure")
        result = _Output()
        state["output_refs"].append(weakref.ref(result))
        state["calls"] += 1
        return result

    def synchronize() -> None:
        assert not state["window"]
        state["syncs"] += 1
        if state["active"]:
            assert all(ref() is not None for ref in state["output_refs"])
            state["output_refs"].clear()

    monkeypatch.setattr(torch.profiler, "profile", profile)
    monkeypatch.setattr(torch.profiler, "record_function", window)

    def invoke() -> subject.ProfilerTimingObservation:
        return subject.collect_profiler_timing(
            call,
            identity=IDENTITY,
            sample_count=5,
            clear=clear,
            synchronize=synchronize,
            policy=subject.ProfilerTimingPolicy(chunk_size=2),
            trace_sink=state["traces"].append,
        )

    return invoke, state


def test_chunks_exact_count_weighted_sum_and_output_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke, state = _capture_harness(monkeypatch)
    result = invoke()
    assert result.call_count == state["calls"] == state["clears"] == 5
    assert tuple(chunk.call_count for chunk in result.chunks) == (2, 2, 1)
    assert result.total_ns == 2 * 10 + 3 * 30
    assert result.mean_ms == pytest.approx(22 / 1_000_000)
    assert result.identity == IDENTITY and result.policy.version == 1
    assert state["syncs"] == 6 and state["captures"] == 3
    assert len(state["traces"]) == 3
    assert not state["active"] and not state["window"]


def test_cross_chunk_identity_rejects_and_retains_raw_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke, state = _capture_harness(monkeypatch, changed_chunk=True)
    with pytest.raises(subject.ProfilerTimingCapabilityError, match="between chunks"):
        invoke()
    assert len(state["traces"]) == 2
    assert not state["active"] and not state["window"]


def test_callable_failure_syncs_live_outputs_and_restores_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke, state = _capture_harness(monkeypatch, fail_call=True)
    with pytest.raises(LookupError, match="callable failure"):
        invoke()
    assert state["calls"] == 1 and state["syncs"] == 2
    assert not state["active"] and not state["window"]


@pytest.mark.parametrize("sample_count", [0, -1, True, 1.5])
def test_invalid_count_never_enters_capture(sample_count: Any) -> None:
    with (
        patch.object(
            torch.profiler, "profile", side_effect=AssertionError("capture forbidden")
        ),
        pytest.raises(subject.ProfilerTimingCapabilityError),
    ):
        subject.collect_profiler_timing(
            lambda: None,
            identity=IDENTITY,
            sample_count=sample_count,
            clear=lambda: None,
            synchronize=lambda: None,
        )


def test_real_installed_cpu_profiler_schema_does_not_initialize_cuda() -> None:
    before = torch.cuda.is_initialized()
    with (
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
        ),
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        with (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as prof,
            torch.profiler.record_function(subject.CALLABLE_WINDOW_NAME),
        ):
            torch.ones(2).add_(1)
        assert prof.profiler is not None and prof.profiler.kineto_results is not None
        events = subject.read_profiler_events(prof.profiler.kineto_results.events())
        assert events and all(
            type(getattr(event, field.name)) in (int, str)
            for event in events
            for field in fields(event)
        )
        with pytest.raises(
            subject.ProfilerTimingCapabilityError, match="no complete kernel work"
        ):
            subject.attribute_profiler_trace(subject.ProfilerTrace(IDENTITY, 1, events))
    assert torch.cuda.is_initialized() == before


def test_missing_installed_schema_is_typed_error() -> None:
    unsupported: Any = SimpleNamespace()
    with pytest.raises(subject.ProfilerTimingCapabilityError, match="schema"):
        subject.read_profiler_events([unsupported])


def test_existing_cpu_profiler_is_preserved_before_any_callback() -> None:
    before = torch.cuda.is_initialized()
    counts = {"call": 0, "clear": 0, "sync": 0}

    def count(name: str) -> None:
        counts[name] += 1

    with (
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
        ),
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as outer:
            with torch.profiler.record_function("outer_before_collector"):
                torch.ones(2).add_(1)
            assert torch.autograd._profiler_enabled()
            with (
                patch.object(
                    torch.profiler,
                    "profile",
                    side_effect=AssertionError("nested capture forbidden"),
                ),
                pytest.raises(
                    subject.ProfilerTimingCapabilityError,
                    match="existing active profiler",
                ),
            ):
                subject.collect_profiler_timing(
                    lambda: count("call"),
                    identity=IDENTITY,
                    sample_count=1,
                    clear=lambda: count("clear"),
                    synchronize=lambda: count("sync"),
                )
            assert torch.autograd._profiler_enabled()
            with torch.profiler.record_function("outer_after_collector"):
                torch.ones(2).mul_(2)
        assert {"outer_before_collector", "outer_after_collector"} <= {
            event.name for event in outer.events()
        }
    assert counts == {"call": 0, "clear": 0, "sync": 0}
    assert not torch.autograd._profiler_enabled()
    assert torch.cuda.is_initialized() == before


def test_external_profiler_between_chunks_rejects_before_next_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = torch.cuda.is_initialized()
    profile = torch.profiler.profile
    record_function = torch.profiler.record_function
    invoke, state = _capture_harness(monkeypatch)
    outer = profile(activities=[torch.profiler.ProfilerActivity.CPU])
    with (
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
        ),
        patch.object(torch.cuda, "is_available", return_value=False),
        ExitStack() as stack,
    ):

        class ActivatingSink(list[subject.ProfilerTrace]):
            def append(self, trace: subject.ProfilerTrace) -> None:
                super().append(trace)
                stack.enter_context(outer)
                with record_function("external_before_next_chunk"):
                    torch.ones(2).add_(1)

        state["traces"] = ActivatingSink()
        with pytest.raises(
            subject.ProfilerTimingCapabilityError, match="existing active profiler"
        ):
            invoke()
        assert torch.autograd._profiler_enabled()
        assert state["calls"] == state["clears"] == 2
        assert state["syncs"] == 2 and state["captures"] == 1
        assert len(state["traces"]) == 1
        with record_function("external_after_next_chunk"):
            torch.ones(2).mul_(2)
    assert {"external_before_next_chunk", "external_after_next_chunk"} <= {
        event.name for event in outer.events()
    }
    assert not torch.autograd._profiler_enabled()
    assert torch.cuda.is_initialized() == before
