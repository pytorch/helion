"""Stand-ins for the autotuner's timers in CPU tests, the graph replay and L2
regime settings the timers take from the environment, and the marks of the
tests that read the CUDA activity trace (``test_benchmark_prelude``,
``test_benchmarking_batched_replay``, ``test_l2_flush_sweep``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from torch.profiler import ProfilerActivity
from torch.profiler import supported_activities

from helion._testing import DEVICE
from helion._testing import skipIfRocm
from helion.autotuner import benchmarking

if TYPE_CHECKING:
    from collections.abc import Callable

# The graph replay and L2 regime settings the timers and the final selection
# take from the environment: whether a call is timed as a graph replay, the L2
# regime of the search, and the regime and the target regime of the final
# selection. (Not everything the timers read there: the graph dispatch
# warm-up has HELION_BENCHMARK_GRAPH_WARMUP.)
TIMER_SETTINGS = (
    benchmarking._BENCHMARK_CUDAGRAPH_ENV,
    benchmarking._L2_FLUSH_ENV,
    benchmarking._FINAL_L2_FLUSH_ENV,
    benchmarking._TARGET_REGIME_ENV,
)


def stand_in_event_timer(
    monkeypatch: pytest.MonkeyPatch, *, sweep: str = "sweep"
) -> list[str]:
    """The event timers on a stand-in Triton driver: the log receives every
    flush, sweep of pinned lines (``_release_l2_pins``, logged as ``sweep``),
    event record and synchronization."""
    runtime = pytest.importorskip("triton.runtime")
    log: list[str] = []

    class Event:
        def __init__(self, enable_timing: bool) -> None:
            assert enable_timing

        def record(self) -> None:
            log.append("event")

        def elapsed_time(self, other: Event) -> float:
            return 2.0

    interface = SimpleNamespace(Event=Event, synchronize=lambda: log.append("sync"))
    driver = SimpleNamespace(
        get_device_interface=lambda: interface,
        get_empty_cache_for_benchmark=lambda: "buffer",
        clear_cache=lambda buffer: log.append("flush"),
    )
    monkeypatch.setattr(runtime.driver, "_active", driver)
    monkeypatch.setattr(
        benchmarking, "_release_l2_pins", lambda buffer: log.append(sweep)
    )
    for name in TIMER_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    return log


class LoggedPrelude:
    """A prelude as a timer sees one: running it logs ``prelude``."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def run(self) -> None:
        self.log.append("prelude")


def stand_in_trace(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """``do_bench``'s kernel timer on a stand-in CUDA activity trace: every
    launch is a kernel of its own, 10 ns after the one before. ``launch(name,
    ns)`` makes a kernel, ``log`` holds the launches of the window being
    traced, ``device`` and ``flush`` (1 ms) stand in for the timer's."""
    pytest.importorskip("triton")
    log: list[tuple[str, int]] = []

    def launch(name: str, nanoseconds: int) -> Callable[[], None]:
        return lambda: log.append((name, nanoseconds))

    def traced(run: Callable[[], None]) -> list[benchmarking._TraceEvent]:
        log.clear()
        run()
        events, clock = [], 0
        for name, nanoseconds in log:
            events.append(benchmarking._TraceEvent(name, clock, clock + nanoseconds))
            clock += nanoseconds + 10
        return events

    monkeypatch.setattr(benchmarking, "_kernel_timer_preconditions", lambda group: True)
    monkeypatch.setattr(benchmarking, "_flush_trace", None)
    monkeypatch.setattr(benchmarking, "_flush_sweep_trace", None)
    monkeypatch.setattr(
        benchmarking, "_trace_marker", launch(benchmarking._TRACE_MARKER, 1)
    )
    monkeypatch.setattr(benchmarking, "_cuda_kernel_events", traced)
    event = SimpleNamespace(record=lambda: None, elapsed_time=lambda end: 1.0)
    device = SimpleNamespace(
        Event=lambda enable_timing: event, synchronize=lambda: None
    )
    return SimpleNamespace(
        launch=launch, log=log, device=device, flush=launch("flush", 1_000_000)
    )


TRACE_MARKS = (
    skipIfRocm("the CUDA activity trace is NVIDIA-only"),
    pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA"),
    pytest.mark.skipif(
        ProfilerActivity.CUDA not in supported_activities(),
        reason="requires CUDA activity tracing",
    ),
)


def trace_test(fn: Callable[..., None]) -> Callable[..., None]:
    """``fn`` with ``TRACE_MARKS``: it reads the CUDA activity trace."""
    for mark in TRACE_MARKS:
        fn = mark(fn)
    return fn
