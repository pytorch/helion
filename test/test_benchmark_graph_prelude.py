from __future__ import annotations

import contextlib
import gc
import weakref

import pytest
import torch

from helion.autotuner import benchmarking
import helion.runtime as runtime
from helion.runtime.cute import launch_caches

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def mock_graph_timer(monkeypatch):
    """Install the CPU-only CUDA shell; each test owns its graph and events."""
    pytest.importorskip("triton")

    def install(capture, record, synchronize):
        class Stream:
            def wait_stream(self, other):
                pass

            def synchronize(self):
                synchronize()

        class Event:
            def __init__(self, *, enable_timing, external):
                assert enable_timing and external

            def record(self):
                record()

            def elapsed_time(self, other):
                return 3.0

        monkeypatch.setattr(benchmarking, "_cudagraph_unavailable_reason", lambda: None)
        monkeypatch.setattr(benchmarking, "steady_graph_dispatch", lambda: None)
        monkeypatch.setattr(torch.cuda, "Stream", Stream)
        monkeypatch.setattr(
            torch.cuda, "stream", lambda stream: contextlib.nullcontext()
        )
        monkeypatch.setattr(torch.cuda, "current_stream", Stream)
        monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
        monkeypatch.setattr(torch.cuda, "Event", Event)
        monkeypatch.setattr(runtime, "cute_cuda_graph", capture)

    return install


@pytest.mark.parametrize("mode", [None, "replay", "off"])
@pytest.mark.parametrize("pre_warmed", [False, True])
def test_graph_prelude_order_reset_and_output_lifetime(
    monkeypatch, mock_graph_timer, mode, pre_warmed
):
    """Two distinct outputs survive replay; reset precedes each callback."""
    pytest.importorskip("triton")
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "0")
    events = []
    outputs = []
    phase = "setup"
    value = 0

    class Output:
        pass

    class Graph:
        def reset(self):
            captured = [ref for where, ref in outputs if where == "capture"]
            assert all(ref() is not None for ref in captured)
            assert events[-1][1] == "sync"
            events.append((phase, "graph_reset"))

        def replay(self):
            gc.collect()
            captured = [ref for where, ref in outputs if where == "capture"]
            assert len(captured) == 2
            assert all(ref() is not None for ref in captured)

    def callback(name):
        nonlocal value
        assert value == 0, "reset must run separately before primer and timed callback"
        value += 1
        events.append((phase, name))
        result = Output()
        outputs.append((phase, weakref.ref(result)))
        return result

    def reset():
        nonlocal value
        value = 0
        events.append((phase, "reset"))

    def change_phase(name):
        nonlocal phase
        phase = name

    monkeypatch.setattr(
        benchmarking,
        "_make_l2_cache_clearer",
        lambda: lambda: events.append((phase, "clear")),
    )
    mock_graph_timer(
        lambda **kwargs: contextlib.nullcontext(Graph()),
        lambda: events.append((phase, "event")),
        lambda: events.append((phase, "sync")),
    )
    kwargs = {} if mode is None else {"l2_flush_mode": mode}
    assert (
        benchmarking.do_bench_cuda_graph(
            lambda: callback("timed"),
            pre_timing=lambda: callback("prime"),
            reset=reset,
            phase=change_phase,
            fixed_repetitions=2,
            pre_warmed=pre_warmed,
            **kwargs,
        )
        == 3.0
    )
    expected = ["reset"]
    if mode != "off":
        expected.append("clear")
    expected.extend(["prime", "reset", "event", "timed", "event"])
    assert [event for where, event in events if where == "capture"] == expected
    assert events[-1] == ("measure", "graph_reset")
    setup_calls = [
        event for where, event in events if where == "setup" and event != "sync"
    ]
    assert setup_calls == (
        ["reset", "event", "event"]
        if pre_warmed
        else ["reset", "prime", "reset", "timed", "event", "event"]
    )


def test_graph_timer_rejects_per_call_bank_flush_before_capture(monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setattr(benchmarking, "_cudagraph_unavailable_reason", lambda: None)
    monkeypatch.setattr(benchmarking, "steady_graph_dispatch", lambda: None)
    with pytest.raises(ValueError, match="replay or off"):
        benchmarking.do_bench_cuda_graph(lambda: None, l2_flush_mode="call")


@pytest.mark.parametrize("fails", ["begin", "body", "end", "interrupted end"])
def test_capture_that_raises_leaves_the_callers_stream_current(
    monkeypatch, mock_graph_timer, fails
):
    """``torch.cuda.graph`` enters its capture stream ahead of the capture's
    begin and skips its stream exit when the begin or the end raises (an
    invalidated capture). ``cute_cuda_graph`` puts the caller's stream back,
    so the timer that captures through it leaves it current: a benchmark
    worker would otherwise run every later job on the failed job's stream.
    That holds for whatever is raised, an interrupt of the end included."""
    current = ["caller's"]
    raised = KeyboardInterrupt if fails == "interrupted end" else RuntimeError

    class Graph:
        def reset(self):
            pass

    class TorchGraph:
        def __init__(self, graph, *, stream, **options):
            self.stream = stream

        def __enter__(self):
            current.append(self.stream)
            if fails == "begin":
                raise RuntimeError("capture_begin failed")

        def __exit__(self, *exc_info):
            if fails in ("end", "interrupted end"):
                raise raised("operation failed due to a previous error during capture")
            # the capture ended: the capture stream's context exits
            current.pop()

    def fn():
        if fails == "body" and len(current) > 1:
            raise RuntimeError("the captured call failed")

    mock_graph_timer(launch_caches.cute_cuda_graph, lambda: None, lambda: None)
    monkeypatch.setattr(launch_caches, "CuteCUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", TorchGraph)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current[-1])
    monkeypatch.setattr(torch.cuda, "set_stream", current.append)
    with pytest.raises(raised, match="capture|captured call"):
        benchmarking.do_bench_cuda_graph(
            fn, pre_warmed=True, l2_flush_mode="off", fixed_repetitions=1
        )
    assert current[-1] == "caller's"
    assert launch_caches.ACTIVE_CUDA_GRAPH.get() is None


def test_capture_whose_stream_read_raises_marks_no_graph_as_captured(monkeypatch):
    """``cute_cuda_graph`` reads the caller's stream, to put it back after a
    capture that raises, before it names its graph as the one being
    captured: where that read raises (an interrupt, a broken device) no
    capture began, and no later launch of the context may read as inside
    one."""

    class Graph:
        pass

    def current_stream():
        raise RuntimeError("the device is gone")

    monkeypatch.setattr(launch_caches, "CuteCUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    assert launch_caches.ACTIVE_CUDA_GRAPH.get() is None
    with (
        pytest.raises(RuntimeError, match="the device is gone"),
        launch_caches.cute_cuda_graph(),
    ):
        raise AssertionError("the capture's body must not run")
    assert launch_caches.ACTIVE_CUDA_GRAPH.get() is None


@pytest.mark.parametrize(
    "failure", ["capture_body", "capture_end", "replay", "measure"]
)
def test_graph_resources_reset_after_failures(mock_graph_timer, failure):
    """Cleanup follows capture exit and synchronization, with outputs still live."""
    pytest.importorskip("triton")
    captured = []
    events = []

    class Output:
        pass

    class Graph:
        def replay(self):
            raise RuntimeError("replay")

        def reset(self):
            assert events[-1] == "sync"
            assert "capture_exit" in events
            assert all(ref() is not None for ref in captured)
            events.append("reset")

    @contextlib.contextmanager
    def capture(**kwargs):
        try:
            yield Graph()
            if failure == "capture_end":
                raise RuntimeError(failure)
        finally:
            events.append("capture_exit")

    def callback(*, timed=False):
        if timed and failure == "capture_body":
            raise RuntimeError(failure)
        output = Output()
        captured.append(weakref.ref(output))
        return output

    def phase(name):
        if name == "measure" and failure == "measure":
            raise RuntimeError(failure)

    mock_graph_timer(capture, lambda: None, lambda: events.append("sync"))
    with pytest.raises(RuntimeError, match=failure):
        benchmarking.do_bench_cuda_graph(
            lambda: callback(timed=True),
            pre_timing=callback,
            phase=phase,
            pre_warmed=True,
            l2_flush_mode="off",
            fixed_repetitions=1,
        )
    assert events[-1] == "reset"
    assert len(captured) == (1 if failure == "capture_body" else 2)
