from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import statistics
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

    import torch


GraphPolicy = Literal["required", "disabled"]
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineWorkload:
    """Strongly owned pipeline call and in-place reset of its input state.

    Both callbacks enqueue work on the current stream, or join any streams they
    use before returning. Reset preserves every address captured by ``run``.
    The callbacks retain their input/module owners. The caller owns the full
    raw-input boundary and accuracy contract.
    """

    run: Callable[[], object]
    reset: Callable[[], None]


class ReplayGraph(Protocol):
    def replay(self) -> None: ...


class TimerRuntime(Protocol):
    """CUDA operations are injectable so scheduling can be tested on the CPU."""

    def current_stream(self) -> object: ...

    def new_stream(self, caller: object) -> object: ...

    def set_current_stream(self, stream: object) -> None: ...

    def stream_scope(self, stream: object) -> AbstractContextManager[object]: ...

    def wait_stream(self, stream: object, other: object) -> None: ...

    def synchronize(self, stream: object) -> None: ...

    def new_event(self) -> object: ...

    def record_event(self, event: object, stream: object) -> None: ...

    def elapsed_ms(self, start: object, end: object) -> float: ...

    def capture(self, stream: object) -> AbstractContextManager[ReplayGraph]: ...


class CudaEventRuntime:
    """Lazy real adapter; constructing this class may initialize CUDA.

    CPU tests supply TimerRuntime instead and never construct this adapter.
    The existing CuTe graph context retains its raw-pointer resource owners.
    """

    def __init__(self) -> None:
        import torch

        if torch.version.hip is not None or not torch.cuda.is_available():
            raise RuntimeError("The pipeline event timer requires NVIDIA CUDA")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("The pipeline event timer cannot nest graph capture")
        self._torch = torch

    def current_stream(self) -> object:
        return self._torch.cuda.current_stream()

    def new_stream(self, caller: object) -> object:
        return self._torch.cuda.Stream(device=cast("torch.cuda.Stream", caller).device)

    def set_current_stream(self, stream: object) -> None:
        self._torch.cuda.set_stream(cast("torch.cuda.Stream", stream))

    def stream_scope(self, stream: object) -> AbstractContextManager[object]:
        return self._torch.cuda.stream(cast("torch.cuda.Stream", stream))

    def wait_stream(self, stream: object, other: object) -> None:
        cast("torch.cuda.Stream", stream).wait_stream(cast("torch.cuda.Stream", other))

    def synchronize(self, stream: object) -> None:
        cast("torch.cuda.Stream", stream).synchronize()

    def new_event(self) -> object:
        return self._torch.cuda.Event(enable_timing=True)

    def record_event(self, event: object, stream: object) -> None:
        cast("torch.cuda.Event", event).record(cast("torch.cuda.Stream", stream))

    def elapsed_ms(self, start: object, end: object) -> float:
        return float(
            cast("torch.cuda.Event", start).elapsed_time(cast("torch.cuda.Event", end))
        )

    def capture(self, stream: object) -> AbstractContextManager[ReplayGraph]:
        from helion.runtime import cute_cuda_graph

        return cute_cuda_graph(stream=cast("torch.cuda.Stream", stream))


@dataclass(frozen=True)
class WorkloadTiming:
    """Milliseconds, in input-workload order; estimates are not score samples."""

    samples_ms: tuple[float, ...]
    estimate_ms: float | None
    warmup_calls: int
    graph_policy: GraphPolicy

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)


@dataclass
class _Prepared:
    workload: PipelineWorkload
    graph: ReplayGraph | None = None
    captured_output: object = None
    last_output: object = None

    def invoke(self) -> None:
        if self.graph is None:
            self.last_output = self.workload.run()
        else:
            self.graph.replay()
            self.last_output = self.captured_output


def _reset_and_clear(prepared: _Prepared, clear_cache: Callable[[], None]) -> None:
    prepared.workload.reset()
    clear_cache()


def _timed_invocation(
    prepared: _Prepared,
    clear_cache: Callable[[], None],
    runtime: TimerRuntime,
    stream: object,
) -> float:
    start = runtime.new_event()
    end = runtime.new_event()
    _reset_and_clear(prepared, clear_cache)
    runtime.record_event(start, stream)
    prepared.invoke()
    runtime.record_event(end, stream)
    runtime.synchronize(stream)
    elapsed = runtime.elapsed_ms(start, end)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError(f"Expected a finite positive event duration, got {elapsed}")
    return elapsed


def _drain_and_restore(
    runtime: TimerRuntime,
    streams: Sequence[object],
    caller: object,
    active_error: BaseException | None,
) -> None:
    errors: list[BaseException] = []
    for stream in streams:
        try:
            runtime.synchronize(stream)
        except BaseException as error:
            errors.append(error)
    try:
        runtime.set_current_stream(caller)
    except BaseException as error:
        errors.append(error)
    if active_error is not None:
        for error in errors:
            _log.warning("Pipeline timer cleanup also failed: %r", error)
    elif errors:
        for error in errors[1:]:
            _log.warning("Pipeline timer cleanup also failed: %r", error)
        raise errors[0]


def benchmark_workloads(
    workloads: Sequence[PipelineWorkload],
    *,
    clear_cache: Callable[[], None],
    graph_policy: GraphPolicy,
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
    samples: int | None = None,
    runtime: TimerRuntime | None = None,
) -> tuple[WorkloadTiming, ...]:
    """Time explicit pipelines, one invocation per event pair, in round robin.

    Setup is one eager invocation per workload. Required graph mode then makes
    one run-only capture on a side stream of the caller's device, then one
    untimed caller-stream replay to finish fresh-graph setup. Every setup,
    capture, estimate, warmup and sample has a preceding reset and cache clear.
    Only the run/replay is between sample events. Samples replay on the original
    caller stream; reset and clear must obey that stream's ordering contract.

    A positive event estimate determines ceil(warmup_ms / estimate) warmups and
    ceil(rep_ms / estimate) samples, with no upper cap. Explicit ``samples``
    replaces the latter count exactly. No estimate is made if both a fixed
    count and zero warmup are requested. Estimates and setup are not returned
    as samples. Graph errors propagate, with no eager fallback or batching.

    This opt-in prototype does not change do_bench, choose a search objective,
    check accuracy, own a cache-clear allocation, or qualify a benchmark.
    """
    owned_workloads = tuple(workloads)
    if not owned_workloads:
        raise ValueError("At least one pipeline workload is required")
    if graph_policy not in ("required", "disabled"):
        raise ValueError("graph_policy must be required or disabled")
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("warmup_ms must be finite and nonnegative")
    if not math.isfinite(rep_ms) or rep_ms < 0:
        raise ValueError("rep_ms must be finite and nonnegative")
    if samples is not None and (type(samples) is not int or samples <= 0):
        raise ValueError("samples must be a positive integer")
    if samples is None and rep_ms <= 0:
        raise ValueError("Natural repetition requires positive rep_ms")

    runtime = CudaEventRuntime() if runtime is None else runtime
    caller = runtime.current_stream()
    streams = [caller]
    prepared = [_Prepared(workload) for workload in owned_workloads]
    active_error: BaseException | None = None
    try:
        for item in prepared:
            if graph_policy == "required":
                side = runtime.new_stream(caller)
                streams.append(side)
                runtime.wait_stream(side, caller)
                with runtime.stream_scope(side):
                    _reset_and_clear(item, clear_cache)
                    item.last_output = item.workload.run()
                    runtime.synchronize(side)
                    _reset_and_clear(item, clear_cache)
                    runtime.synchronize(side)
                    with runtime.capture(side) as graph:
                        item.graph = graph
                        item.captured_output = item.workload.run()
                    runtime.synchronize(side)
                runtime.wait_stream(caller, side)
                with runtime.stream_scope(caller):
                    _reset_and_clear(item, clear_cache)
                    item.invoke()
                    runtime.synchronize(caller)
            else:
                with runtime.stream_scope(caller):
                    _reset_and_clear(item, clear_cache)
                    item.last_output = item.workload.run()
                    runtime.synchronize(caller)

        estimates: list[float | None] = []
        warmup_counts: list[int] = []
        sample_counts: list[int] = []
        with runtime.stream_scope(caller):
            for item in prepared:
                estimate = (
                    _timed_invocation(item, clear_cache, runtime, caller)
                    if samples is None or warmup_ms > 0
                    else None
                )
                estimates.append(estimate)
                warmup_counts.append(
                    math.ceil(warmup_ms / estimate)
                    if estimate is not None and warmup_ms > 0
                    else 0
                )
                if samples is None:
                    assert estimate is not None
                    sample_counts.append(max(1, math.ceil(rep_ms / estimate)))
                else:
                    sample_counts.append(samples)

            for index in range(max(warmup_counts)):
                for item, count in zip(prepared, warmup_counts, strict=True):
                    if index < count:
                        _reset_and_clear(item, clear_cache)
                        item.invoke()
                        runtime.synchronize(caller)

            measurements: list[list[float]] = [[] for item in prepared]
            for index in range(max(sample_counts)):
                for item, count, values in zip(
                    prepared, sample_counts, measurements, strict=True
                ):
                    if index < count:
                        values.append(
                            _timed_invocation(item, clear_cache, runtime, caller)
                        )

        return tuple(
            WorkloadTiming(tuple(values), estimate, count, graph_policy)
            for values, estimate, count in zip(
                measurements, estimates, warmup_counts, strict=True
            )
        )
    except BaseException as error:
        active_error = error
        raise
    finally:
        # Keep workloads, captured outputs, graphs and last eager outputs alive
        # through all stream drains, including a failed capture or event call.
        _drain_and_restore(runtime, streams, caller, active_error)
