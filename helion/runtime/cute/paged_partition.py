"""Launch workspace of the balanced paged-attention partition.

A balanced launch splits some units among CTAs, which exchange each part's
unnormalized result through this buffer: one record per resident CTA, two
arrival counters and four partial slots each
(``helion._compiler.cute._paged_attention_partition``). The generated host
function asks for it on every launch, ahead of the launcher.

The counters are zero whenever no launch that uses the buffer is in flight:
it is zeroed once, and the part that merges a unit resets the unit's
counter. Partials are only read behind the arrival of the part that wrote
them.

Two things break that: inputs that change while a launch reads them (CTAs
that read different lengths plan different splits, and count parts that
never arrive or miss parts that do), and a launch that never completes.
Either can leave a counter at a count ``L`` above zero, and nothing zeroes
it again: this module cannot see either happen. Every later launch that
splits a unit at that counter, in ``n`` parts, then goes wrong at the unit:

- ``L < n``. The counter reaches ``n - 1`` after ``n - 1 - L`` arrivals, so
  a part merges without the last ``L`` parts. It folds what their slots
  hold then, an earlier launch's partials or stores still in flight, and
  stores the result. The late parts count on from the merger's reset and
  leave ``L`` behind again (less only by an arrival that falls between a
  merger's read of the counter and its reset), so wrong values can persist
  across launches of the same split. When the early merger is the lowest
  part, which reads the counter and then resets it, a part that arrives in
  between reads the same count: two parts merge and store the unit.
- ``L >= n``. No part reads ``n - 1``, so nothing merges: the unit's
  outputs keep the previous contents of their memory, and the counter grows
  by ``n`` with every launch.

One buffer serves the launches of one stream, which are ordered, so kernels
of the same geometry (record count and record size) share it and a launch on
another stream gets another buffer: the sixteen streams that launched most
recently keep theirs, per device and geometry. The geometry is the owner, not
the byte count: records of another size keep their counters where this
geometry's partials lie, which no launch leaves zero. A kernel launched under
dependent launch touches the buffer only behind its dependency wait, so it
may still overlap its predecessor on the stream.

A CUDA graph gets a buffer of its own per captured stream: its replays run
on whatever stream is current then. That buffer must not add a fill to the
graph, which every replay would repeat, so an eager launch keeps one spare
buffer zeroed and a capture adopts it; the next eager launch replaces it.
The spare's fill has completed before a capture can find it. It runs on a
stream this module keeps for these fills, one per device, so the launch that
makes a spare waits for that fill and not for the work queued on the stream
it launches on. CUDA creates that stream for this module (``_private_stream``):
a stream of PyTorch's pool is handed to other code too, and a thread that
captured a graph on it would record the fill as a node of its graph and have
its capture invalidated by the wait. The launch makes the spare with the
captures of other threads ignored (``_other_threads_captures_ignored``): in
a thread's default capture mode CUDA refuses it the spare's allocation and
the wait while any thread captures a graph in the ``global`` mode, the
default of ``torch.cuda.graph``, and the refused call invalidates that
capture. A capture of the launching thread itself bars those calls in every
mode but ``relaxed`` (by CUDA's documentation of the modes; no run here used
that one), so no spare is made while that thread's current stream captures
on any device, nor while a ``cute_cuda_graph`` capture is underway
in the process, and no such capture begins while one is made
(``_restore_spare`` says how). A capture that finds no spare allocates
inside the graph and zeroes the counters there: one small node ahead of its
first attention launch.

A graph's buffer lives as long as the graph, whose nodes hold its address.
``cute_cuda_graph`` releases it when its graph is reset or collected. A raw
``torch.cuda.graph`` has no owner the launcher can see, so its buffer is
leased on the graph being captured (``graph_lease``): CUDA ends the lease
when the graph is destroyed and its replays have completed, and the next
launch that captures, or that makes an eager buffer or a spare, returns the
buffers whose leases have ended to the allocator. A launch that finds its
eager buffer and a spare does not look, so that a relaunch polls nothing.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import ctypes
import threading
from typing import TYPE_CHECKING

import torch

from ..._compiler.cute.paged_attention_partition_bounds import RECORD_COUNTER_BYTES
from .graph_lease import cuda_driver
from .graph_lease import graph_lease_ended
from .graph_lease import lease_capturing_graph
from .launch_caches import ACTIVE_CUDA_GRAPH
from .launch_caches import cuda_stream_capture_context
from .launch_caches import outside_captures
from .launch_caches import track_cuda_graph_cache_entry

if TYPE_CHECKING:
    from collections.abc import Generator

# Eager buffers kept per device and geometry: those of the streams that
# launched most recently, one each.
EAGER_STREAMS = 16
# CU_STREAM_NON_BLOCKING: no implicit synchronization with the legacy
# default stream, as for the streams PyTorch creates.
_NON_BLOCKING = 1
# CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: the calling thread is barred from
# calls a capture cannot record only by a capture of its own; "concurrent
# capture sequences in other threads are ignored".
_THREAD_LOCAL = 1


# Records and the bytes of each: what lays a buffer out.
_Geometry = tuple[int, int]
# A capture's buffer: device, stream, capture id and geometry.
_Captured = tuple[int, int, int, _Geometry]


class _Workspaces:
    """The buffers by owner: eager streams, captures, and the spares."""

    def __init__(self) -> None:
        # Held for dictionary updates and the looks at leases, which wait
        # for nothing: never across an allocation, a fill, a stream
        # synchronization or the release of a buffer. Lock order: the
        # capture gate (``launch_caches.outside_captures``, held while a
        # spare is made), then ``fill_stream_lock``, then this lock; nothing
        # waits for the gate under either of the two.
        self.lock = threading.Lock()
        # Keys end in the geometry, ``(ctas, record_bytes)``. In order of
        # last use, the most recent last.
        self.eager: OrderedDict[tuple[int, int, _Geometry], torch.Tensor] = (
            OrderedDict()
        )
        # A captured graph embeds the buffer's address, so the entry lives
        # as long as the graph: ``cute_cuda_graph`` removes it with its
        # graph, and a raw ``torch.cuda.graph``'s goes when its lease ends.
        self.captured: dict[_Captured, torch.Tensor] = {}
        # The leases of raw captures on their graphs, each with its entry.
        self.leases: dict[int, _Captured] = {}
        self.spares: dict[tuple[int, _Geometry], torch.Tensor] = {}
        # The stream of each device on which its spares are zeroed, and
        # the lock held while one is created, so that a device gets one.
        self.fill_streams: dict[int, torch.cuda.Stream] = {}
        self.fill_stream_lock = threading.Lock()


_WORKSPACES = _Workspaces()


def _zeroed(device: torch.device, geometry: _Geometry) -> torch.Tensor:
    workspace = torch.zeros(geometry[0] * geometry[1], dtype=torch.uint8, device=device)
    assert workspace.data_ptr() % 128 == 0
    return workspace


def _eager_workspace(
    state: _Workspaces, key: tuple[int, int, _Geometry], device: torch.device
) -> torch.Tensor:
    """A zeroed buffer for the stream of ``key``, which has none yet; the
    streams used least recently beyond ``EAGER_STREAMS`` lose theirs."""
    # The fill is enqueued on the launching stream, ahead of the launch.
    fresh = _zeroed(device, key[2])
    with state.lock:
        workspace = state.eager.setdefault(key, fresh)
        state.eager.move_to_end(key)
        same = [other for other in state.eager if other[::2] == key[::2]]
        # An evicted buffer returns to the allocator's pool of the stream it
        # was used on, so its reuse is ordered behind any launch still
        # running there.
        evicted = [
            state.eager.pop(other)
            for other in same[: max(0, len(same) - EAGER_STREAMS)]
        ]
    # The evicted buffers are freed here, behind the lock.
    evicted.clear()
    return workspace


def _private_stream(device: torch.device) -> torch.cuda.Stream | None:
    """A stream of ``device`` that no other code holds, or ``None`` if CUDA
    creates none.

    ``torch.cuda.Stream`` hands out the streams of a pool in turn, so the
    stream it returns can be one another thread holds, or captures a CUDA
    graph on. The driver's ``cuStreamCreate`` returns a stream of its own,
    which PyTorch selects, allocates on and waits for as an
    ``ExternalStream``. Nothing destroys it: it serves the device's spares
    for as long as the process lives, PyTorch's allocator keeps the blocks
    freed on it under its handle, and CUDA frees it with the context.
    """
    handle = ctypes.c_void_p()
    with torch.cuda.device(device):
        # The driver creates the stream in the context that is current on
        # this thread, and a thread that has not used the device yet has
        # none. A memory query makes it current without waiting for the
        # device and without touching a stream.
        torch.cuda.mem_get_info()
        if cuda_driver().library.cuStreamCreate(ctypes.byref(handle), _NON_BLOCKING):
            return None
    return torch.cuda.ExternalStream(handle.value, device=device)


def _fill_stream(
    state: _Workspaces, index: int, device: torch.device
) -> torch.cuda.Stream | None:
    """The stream of device ``index`` on which its spare buffers are
    allocated and zeroed, created on first use (``_private_stream``);
    ``None`` while CUDA creates none. One stream for all of them: the
    allocator reuses a freed block only on the stream it was allocated on,
    so the spare a released graph frees serves the next one."""
    with state.lock:
        stream = state.fill_streams.get(index)
    if stream is None:
        with state.fill_stream_lock:
            with state.lock:
                stream = state.fill_streams.get(index)
            if stream is None:
                stream = _private_stream(device)
                if stream is not None:
                    with state.lock:
                        state.fill_streams[index] = stream
    return stream


@contextmanager
def _other_threads_captures_ignored() -> Generator[bool, None, None]:
    """Put the calling thread into CUDA's ``THREAD_LOCAL`` stream capture
    interaction mode for the block and give it back the mode it had, as
    CUDA asks of ``cuThreadExchangeStreamCaptureMode`` ("in a push-pop
    fashion"); ``False`` where CUDA refuses the exchange, and the block
    then runs in the mode the thread had.

    A thread is in the ``GLOBAL`` mode unless it says otherwise, and CUDA
    then refuses it every call a capture cannot record (an allocation, a
    stream synchronization) while any thread of the process captures a
    graph in that mode, the default of ``torch.cuda.graph``; the refused
    call invalidates that capture. In the ``THREAD_LOCAL`` mode only a
    capture of the calling thread bars such calls, so a block that has
    excluded those may make them beside the captures of other threads.
    Measured on a B200 beside a default-mode ``torch.cuda.graph`` held open
    on another thread: an allocation that PyTorch has to ask CUDA for, on a
    stream of the caller's own, and ``cuStreamSynchronize`` of that stream
    succeed and leave the capture valid. In the ``GLOBAL`` mode the
    synchronization is refused with ``CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED``
    and the capture is lost; the allocation is refused there too ("operation
    not permitted when stream is capturing"), in a run that also made the
    refused synchronization, so the capture's loss was not seen for the
    allocation alone. ``cuCtxSynchronize`` is refused in either mode: it
    waits for the capturing stream as well.
    """
    exchange = cuda_driver().library.cuThreadExchangeStreamCaptureMode
    # In: the mode to set. Out: the mode the thread had.
    mode = ctypes.c_int(_THREAD_LOCAL)
    exchanged = not exchange(ctypes.byref(mode))
    try:
        yield exchanged
    finally:
        if exchanged:
            exchange(ctypes.byref(mode))


def _restore_spare(
    state: _Workspaces, index: int, geometry: _Geometry, device: torch.device
) -> None:
    """Zero a spare buffer for the next capture, which adopts it on any
    stream without a fill of its own: the fill has completed before the
    spare is published. The fill runs on the device's fill stream, so this
    launch waits for the fill alone: synchronizing the stream it launches
    on would wait for everything the caller queued there. Other threads'
    launches do not wait for that synchronization; two threads that both
    find no spare each make one, and one of the two is kept.

    No other code holds the fill stream (``_private_stream``), so no
    thread captures a graph on it: the fill is recorded into no graph, and
    the wait is not a synchronization of a capturing stream.

    What CUDA lets a thread do beside a capture is the thread's stream
    capture interaction mode. In the ``GLOBAL`` mode, a thread's default,
    CUDA refuses it the spare's allocation and the wait while any thread
    of the process captures a graph in the ``global`` mode, the default of
    ``torch.cuda.graph`` and ``cute_cuda_graph``, whatever stream the call
    names, and the refused call invalidates that capture. The spare is
    therefore made in the ``THREAD_LOCAL`` mode
    (``_other_threads_captures_ignored``), in which the captures of other
    threads do not bar this one. Measured on a B200: beside a default-mode
    ``torch.cuda.graph`` that another thread holds open, an eager launch
    then makes its spare and the capture replays right, where the wait was
    refused, the launch raised and that capture was lost.

    A capture of this thread bars those calls in either mode, so no spare
    is made while the thread captures: not while its current stream
    captures on any device the process has a context on
    (``_capturing_on_any_device``; measured on two B200s for a raw
    ``torch.cuda.graph`` on another device than the launch's, where making
    the spare was refused in both modes: its zeroing, ``torch.zeros``, in
    the two runs of the launch as it was and, where that passed, in the
    device test with this check removed, the wait for the fill, "API call
    conflicts with a stream capture sequence initiated from the calling
    thread"), and not while a ``cute_cuda_graph`` capture is underway
    anywhere in the process. No
    ``cute_cuda_graph`` capture begins while a spare is made either: the
    questions, the fill and the wait are one step under the capture gate
    (``launch_caches.outside_captures``), which the startup of such a
    capture waits for. This launch does not wait for the gate: while
    another thread holds it (it counts its capture, unloads a library or
    makes a spare) no spare is made.

    One capture is not seen: a raw ``torch.cuda.graph`` of this thread on a
    stream that is no longer current on its device. No run has that case:
    from the measurement on two devices it follows that the spare is refused
    there and the capture lost, and from CUDA's documentation of the modes
    that a thread which switches streams inside its own raw capture and
    launches eagerly there has to capture in the ``relaxed`` mode.

    Where no spare is made, and where CUDA creates no fill stream, this
    launch goes on with its eager buffer, a capture that finds no spare
    zeroes its counters inside its graph, as when no spare ever existed,
    and the next eager launch tries again."""
    with outside_captures(wait=False) as safe:
        if not safe or _capturing_on_any_device():
            return
        with _other_threads_captures_ignored() as exchanged:
            if not exchanged:
                return
            stream = _fill_stream(state, index, device)
            if stream is None:
                return
            with torch.cuda.stream(stream):
                spare = _zeroed(device, geometry)
            stream.synchronize()
        with state.lock:
            state.spares.setdefault((index, geometry), spare)


def _release_destroyed(state: _Workspaces) -> int:
    """Return the buffers of the raw captures whose graphs CUDA has
    destroyed to the allocator, and how many those were. CUDA ends a lease
    only when the launches of its graph have completed, so nothing still
    runs on them."""
    with state.lock:
        ended = [lease for lease in state.leases if graph_lease_ended(lease)]
        buffers = [state.captured.pop(state.leases.pop(lease), None) for lease in ended]
    # The buffers are freed on return, behind the lock.
    return sum(buffer is not None for buffer in buffers)


def _capturing(index: int) -> bool:
    """Whether the current stream of device ``index`` captures a CUDA graph."""
    if index == torch.cuda.current_device():
        return torch.cuda.is_current_stream_capturing()
    with torch.cuda.device(index):
        return torch.cuda.is_current_stream_capturing()


def _capturing_on_any_device() -> bool:
    """Whether the calling thread's current stream captures a CUDA graph on
    any device the process has a context on: a device without one has no
    stream that could, and asking would create its context."""
    return any(
        torch._C._cuda_hasPrimaryContext(index) and _capturing(index)
        for index in range(torch.cuda.device_count())
    )


def paged_partition_workspace(
    like: torch.Tensor, *, ctas: int, record_bytes: int
) -> torch.Tensor:
    """The workspace of a balanced launch on ``like``'s device and stream.

    ``ctas`` records of ``record_bytes`` each, as flat bytes.
    """
    device = like.device
    nbytes = ctas * record_bytes
    if device.type != "cuda":
        return torch.zeros(nbytes, dtype=torch.uint8, device=device)
    index = device.index
    assert index is not None
    state = _WORKSPACES
    geometry = (ctas, record_bytes)
    if not _capturing(index):
        # The raw handle: constructing a Stream object costs more than the
        # rest of this lookup.
        key = (index, torch._C._cuda_getCurrentRawStream(index), geometry)
        with state.lock:
            workspace = state.eager.get(key)
            if workspace is not None:
                state.eager.move_to_end(key)
            spare = (index, geometry) in state.spares
        if workspace is None or not spare:
            # A launch that allocates also returns what destroyed graphs
            # used. The next spare reuses a spare returned that way: both
            # are allocated on the fill stream. The eager buffer made here
            # is allocated on the launching stream and reuses none of it.
            _release_destroyed(state)
        if workspace is None:
            workspace = _eager_workspace(state, key, device)
        if not spare:
            _restore_spare(state, index, geometry, device)
        return workspace
    stream, capture_id = cuda_stream_capture_context(device)
    assert capture_id is not None
    captured = (index, stream, capture_id, geometry)
    workspace = state.captured.get(captured)
    if workspace is not None:
        return workspace
    _release_destroyed(state)
    # ``cute_cuda_graph`` owns what its capture uses; a raw graph takes a
    # lease, ahead of the buffer so that a lease CUDA refuses leaves none.
    managed = ACTIVE_CUDA_GRAPH.get() is not None
    lease = None if managed else lease_capturing_graph(device)
    with state.lock:
        if lease is not None:
            state.leases[lease] = captured
        workspace = state.captured.get(captured)
        adopted = workspace is None
        if adopted:
            workspace = state.spares.pop((index, geometry), None)
            if workspace is not None:
                state.captured[captured] = workspace
    if workspace is None:
        # No spare: this capture zeroes the counters inside its graph. Like
        # the eager buffers, outside the lock; of two threads in one
        # capture, one buffer is kept.
        fresh = torch.empty(nbytes, dtype=torch.uint8, device=device)
        assert fresh.data_ptr() % 128 == 0
        fresh.view(ctas, record_bytes)[:, :RECORD_COUNTER_BYTES].zero_()
        with state.lock:
            workspace = state.captured.setdefault(captured, fresh)
        adopted = workspace is fresh
    if adopted and managed:
        track_cuda_graph_cache_entry(
            state, "captured", captured, workspace, (workspace,)
        )
    return workspace
