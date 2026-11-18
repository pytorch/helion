"""Stand-ins for the interleavings of a ``cute_cuda_graph`` capture's startup
with code that asks whether a capture is underway and then calls the driver
(``helion.runtime.cute.launch_caches.outside_captures``), without a device.

``capture_stand_ins`` puts a recording stand-in in ``torch.cuda.graph``'s
place and a :class:`WatchedGate` in the capture gate's, so that a test can
start a capture on another thread from inside a driver call's stand-in and
see whether that capture began before the call returned.
"""

from __future__ import annotations

import contextlib
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

from helion.runtime.cute import launch_caches

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

# How long a stand-in waits for another thread before the test fails: far
# longer than any step here takes, so only a deadlock reaches it.
STEP_TIMEOUT_S = 30.0


class WatchedGate:
    """The capture gate, which tells a test what happens at it.

    ``waited`` is set when a thread starts to wait for the gate. ``taken``,
    when the test gives one, runs on the thread that has just taken the gate
    with a ``with`` statement (a capture's count), before that thread does
    anything under it. A thread that waits for the gate it holds itself, or
    for longer than ``STEP_TIMEOUT_S``, is recorded in ``deadlocks`` and
    goes on without the gate, so that the test fails instead of hanging. So
    is a thread that releases the gate without holding it: the lock this
    stands in for would give up another thread's hold there.
    """

    def __init__(self, taken: Callable[[], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._holder: int | None = None
        self.taken = taken
        self.waited = threading.Event()
        self.deadlocks: list[str] = []

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        if self._lock.acquire(blocking=False):
            self._holder = me
            return True
        if not blocking:
            return False
        if self._holder == me:
            self.deadlocks.append("a thread waited for the gate it holds")
            return False
        self.waited.set()
        if not self._lock.acquire(timeout=STEP_TIMEOUT_S):
            self.deadlocks.append("a thread waited for the gate and never got it")
            return False
        self._holder = me
        return True

    def release(self) -> None:
        if self._holder != threading.get_ident():
            self.deadlocks.append("a thread released the gate without holding it")
            return
        self._holder = None
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> bool:
        acquired = self.acquire()
        if acquired and self.taken is not None:
            self.taken()
        return acquired

    def __exit__(self, *exc_info: object) -> None:
        self.release()


class StandInGraph:
    """``torch.cuda.graph`` as ``cute_cuda_graph`` uses it: the entry records
    ``capture_begin`` among ``events`` and the exit ``capture_end``. With
    ``fails`` the entry raises before the capture begins."""

    def __init__(self, events: list[str], *, fails: bool = False) -> None:
        self.events = events
        self.fails = fails
        self.begun = threading.Event()

    def __call__(self, graph: object, **_: object) -> StandInGraph:
        return self

    def __enter__(self) -> None:
        if self.fails:
            raise RuntimeError("capture_begin failed")
        self.events.append("capture_begin")
        self.begun.set()

    def __exit__(self, *exc_info: object) -> None:
        self.events.append("capture_end")


@contextlib.contextmanager
def capture_stand_ins(
    events: list[str],
    *,
    fails: bool = False,
    taken: Callable[[], None] | None = None,
) -> Iterator[tuple[WatchedGate, StandInGraph]]:
    """Run ``cute_cuda_graph`` over a :class:`StandInGraph` and a
    :class:`WatchedGate` until the block ends, on whatever thread calls it."""
    gate = WatchedGate(taken)
    graph = StandInGraph(events, fails=fails)
    with (
        patch.object(launch_caches, "_CAPTURE_GATE", gate),
        patch.object(launch_caches, "CuteCUDAGraph", SimpleNamespace),
        patch.object(torch.cuda, "graph", graph),
        patch.object(torch.cuda, "current_stream", lambda: None),
        patch.object(torch.cuda, "set_stream", lambda stream: None),
    ):
        yield gate, graph
    assert not gate.locked()


class CaptureOnAnotherThread:
    """A ``cute_cuda_graph`` capture that another thread starts when the
    test says so, from inside the stand-in of a driver call that a thread
    makes after it found no capture underway.

    ``start_inside_a_driver_call`` starts the thread and returns once the
    capture has either begun (nothing kept it from beginning inside the
    call) or its thread waits for the gate (it begins behind the call).
    The capture stays open until ``finish``, so that every driver call the
    test's thread still makes lands inside a capture that began.
    """

    def __init__(self, gate: WatchedGate, graph: StandInGraph) -> None:
        self.gate = gate
        self.graph = graph
        self._release = threading.Event()
        self._errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._capture, daemon=True)

    def _capture(self) -> None:
        try:
            with launch_caches.cute_cuda_graph():
                assert self._release.wait(STEP_TIMEOUT_S)
        except BaseException as error:
            self._errors.append(error)

    def start(self) -> None:
        self._thread.start()

    def start_inside_a_driver_call(self) -> None:
        self.start()
        deadline = time.monotonic() + STEP_TIMEOUT_S
        while not (self.graph.begun.is_set() or self.gate.waited.is_set()):
            assert time.monotonic() < deadline, "the capture's thread did not start"
            time.sleep(0.001)

    def finish(self) -> None:
        self._release.set()
        if self._thread.ident is not None:
            self._thread.join(STEP_TIMEOUT_S)
            assert not self._thread.is_alive()
        assert not self._errors, self._errors
        assert not self.gate.deadlocks, self.gate.deadlocks
