"""Launch-time caches of the CuTe launcher and their CUDA-graph ownership.

The launcher keeps three kinds of per-kernel state between launches: the
marshalled launch arguments (:class:`LaunchArgCacheEntry`, keyed by the
argument signature), the grouped-GEMM scheduler metadata copied from the
layout tensors, and launcher-owned exchange workspaces.  A CUDA graph captured
through :func:`cute_cuda_graph` embeds the raw pointers of those entries, so
:class:`CudaGraphResources` ties them to the graph's lifetime; raw
``torch.cuda.graph`` captures are retained conservatively.
:func:`cuda_stream_capture_context` is the ``(stream, capture id)`` pair that
isolates mutable or capture-owned launch tensors per stream.
:func:`drop_relaunch_state` forgets what a kernel keeps of its last launch.
:func:`capture_may_be_active` tells code that runs at arbitrary points (a
finalizer) whether a driver call could land in a capture, and
:func:`outside_captures` keeps a capture from beginning between that
question and the driver calls that follow it.  Leaf module: the launcher
imports it, never the reverse.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextlib import suppress
import contextvars
from dataclasses import dataclass
import gc
import importlib
import threading
from typing import TYPE_CHECKING
from typing import cast
import weakref

import torch

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch.cuda import _POOL_HANDLE


# Keep the per-kernel launch-argument cache small: production kernels normally
# relaunch one or two stable tensor signatures, while autotune may probe many.
LAUNCH_ARG_CACHE_LIMIT = 8
GROUPED_STATIC_METADATA_CACHE_LIMIT = 8
DYNAMIC_TENSORMAP_WORKSPACE_CACHE_LIMIT = 8

GroupedLaunchContext = tuple[str, int | None, int, int | None]

# The attributes of a kernel object in which the launcher keeps the state of
# its fast relaunch: the last launch's marshalled arguments with the compiled
# function they were made for, and that function's executor with its packed
# C arguments. Both refer to a compiled function, so to its CUDA library.
# The launcher sets and reads the state under these names and no spelling of
# its own, so what ``drop_relaunch_state`` forgets is what the launcher keeps.
LAST_LAUNCH_CACHE_ATTRIBUTE = "_helion_cute_last_launch_cache"
FASTPATH_ATTRIBUTE = "_helion_cute_fastpath"
RELAUNCH_STATE_ATTRIBUTES = (LAST_LAUNCH_CACHE_ATTRIBUTE, FASTPATH_ATTRIBUTE)


def drop_relaunch_state(cute_kernel: object) -> None:
    """Forget the fast relaunch state of ``cute_kernel``: its next launch
    marshals its arguments again and builds the state anew. For an owner
    that unloads the kernel's compiled functions
    (``CompiledCuteLauncher.release_kernel``): a relaunch from that state
    would run code that is no longer loaded."""
    for name in RELAUNCH_STATE_ATTRIBUTES:
        with suppress(AttributeError):
            delattr(cute_kernel, name)


@dataclass
class CudaGraphResources:
    cache_entries: dict[tuple[int, str, object], tuple[object, str, object, object]]
    tensors: dict[int, torch.Tensor]
    devices: set[torch.device]
    streams: set[torch.cuda.Stream]

    def retain(
        self,
        owner: object,
        attribute: str,
        key: object,
        value: object,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        self.cache_entries[(id(owner), attribute, key)] = (
            owner,
            attribute,
            key,
            value,
        )
        for tensor in tensors:
            self.tensors[id(tensor)] = tensor
            self.devices.add(tensor.device)
            stream = torch.cuda.current_stream(tensor.device)
            tensor.record_stream(stream)
            self.streams.add(stream)

    def record_replay_streams(self) -> None:
        for device in self.devices:
            stream = torch.cuda.current_stream(device)
            if stream not in self.streams:
                for tensor in self.tensors.values():
                    if tensor.device == device:
                        tensor.record_stream(stream)
                self.streams.add(stream)

    def release(self) -> None:
        for owner, attribute, key, value in self.cache_entries.values():
            cache = getattr(owner, attribute, None)
            if isinstance(cache, dict) and cache.get(key) is value:
                cache.pop(key)
        self.cache_entries.clear()
        self.tensors.clear()
        self.devices.clear()
        self.streams.clear()


class CuteCUDAGraph(torch.cuda.CUDAGraph):
    def __init__(self) -> None:
        super().__init__()
        self._helion_resources = CudaGraphResources({}, {}, set(), set())
        self._helion_resource_finalizer = weakref.finalize(
            self, self._helion_resources.release
        )

    def replay(self) -> None:
        self._helion_resources.record_replay_streams()
        super().replay()

    def reset(self) -> None:
        # Descriptor storage must outlive every graph node that holds its raw
        # address, so destroy the graph before dropping retained ownership.
        super().reset()
        self._helion_resources.release()


ACTIVE_CUDA_GRAPH: contextvars.ContextVar[CuteCUDAGraph | None] = (
    contextvars.ContextVar("helion_cute_active_cuda_graph", default=None)
)


# ``cute_cuda_graph`` captures underway in the process, on any thread, stream
# or device: counted from ahead of torch's entry, which begins the capture,
# until its end has run (``capture_may_be_active``).
_CAPTURES_UNDERWAY = 0
# The count's own lock, held for one addition or subtraction and nothing
# else. The end of a capture lowers the count under this lock alone: it also
# runs where the collector finalizes a capture that was entered and then
# abandoned, on any thread at any allocation, and on the thread that holds
# the gate it would wait for itself there. Re-entrant, because that
# finalization can interrupt a thread inside this lock's own region too.
_CAPTURE_COUNT_LOCK = threading.RLock()
# The capture gate: held while the count is raised and while code that asked
# ``capture_may_be_active`` makes the driver calls its answer allowed
# (``outside_captures``), so that no capture begins between such a question
# and those calls. A capture's startup waits for it; a finalizer, which the
# collector runs on any thread at any allocation, also on the thread that
# holds the gate, only tries it; a capture's end does not take it. A count
# lowered beside a holder cannot make its answer wrong: a block that saw a
# count has declined, and one that saw none holds the gate against every
# raise. Never held across torch's entry or exit, a launch of a kernel the
# caller asked for or the body of a capture: the one launch under it is a
# spare workspace's fill, on a stream no other code holds.
#
# Lock order: the gate comes first. The partition workspaces take their own
# locks under it (``paged_partition._restore_spare``), and nothing waits for
# the gate while it holds one of those: the one acquisition that can happen
# there is a finalizer's, which does not wait.
_CAPTURE_GATE = threading.Lock()


@contextmanager
def _capture_underway() -> Generator[None, None, None]:
    """Count a capture as underway from here to the end of the block."""
    global _CAPTURES_UNDERWAY
    # Waits for a holder's driver calls to return (``outside_captures``): the
    # capture begins behind them.
    with _CAPTURE_GATE, _CAPTURE_COUNT_LOCK:
        _CAPTURES_UNDERWAY += 1
    try:
        yield
    finally:
        # Without the gate: this also runs in a finalizer, on a thread that
        # may hold it (``_CAPTURE_COUNT_LOCK``).
        with _CAPTURE_COUNT_LOCK:
            _CAPTURES_UNDERWAY -= 1


def capture_may_be_active() -> bool:
    """Whether a driver call issued now could land in a CUDA graph capture.

    CUDA refuses a call a capture cannot record, such as
    ``cudaLibraryUnload``, while a capture is underway
    (``CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED``): in the default capture mode
    (``"global"``, which ``torch.cuda.graph`` and :func:`cute_cuda_graph`
    use unless told otherwise) from every thread of the process, in
    ``"thread_local"`` mode from the capturing thread only.  Refused, on
    the capturing thread or (measured for a stream and for a device
    synchronization) on another thread in the global mode, the call also
    invalidates the capture: every later call on its stream, the capture's
    end included, then fails.  Code that runs wherever the collector
    happens to run asks here before it calls the driver.  True from ahead
    of the begin to the
    end of a :func:`cute_cuda_graph` capture anywhere in the process,
    whatever its mode (for a ``"thread_local"`` or ``"relaxed"`` one this is
    more than CUDA asks of the other threads), or while this thread's
    current stream on the current device is capturing (a raw
    ``torch.cuda.graph``); a query that fails counts as capturing.  A raw
    capture on another thread or device is not seen.

    The answer holds for the caller's next driver call only inside
    :func:`outside_captures`: elsewhere another thread's capture can begin
    behind it.
    """
    if _CAPTURES_UNDERWAY:
        return True
    if not torch.cuda.is_initialized():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return True


@contextmanager
def outside_captures(*, wait: bool) -> Generator[bool, None, None]:
    """Whether the block may make driver calls a capture cannot record: no
    capture is underway (``capture_may_be_active``) and, while the block
    runs, no :func:`cute_cuda_graph` capture begins. Such a capture's
    startup waits for the block's end, as does every block that waits, so
    the block holds the gate for those calls alone and begins no capture.
    What runs under it: the wait of an unload for the devices its launcher
    launched on and the unloads behind it
    (``CompiledCuteLauncher.unload_libraries``), a finalizer's unload, and
    the making of a spare partition workspace, which is its allocation, the
    fill it launches on a stream of the workspaces' own and the wait for
    that fill (``paged_partition._restore_spare``).

    ``False`` leaves the block without those calls: a capture may be
    underway, or, without ``wait``, another thread is in such a block or
    raising the count for its capture. With ``wait`` the block waits for
    that thread. A finalizer never waits: the collector may have
    interrupted the thread that holds the gate, which then waits for nobody
    but the finalizer. A module that the collector finalizes while another
    thread is in such a block, in the wait of an unload or for a spare's
    fill for instance, therefore keeps its library loaded for the life of
    the process, as one finalized inside a capture does, and is counted and
    warned about with those although no capture ran
    (``compile_cache.skipped_capture_unloads``).
    """
    if not _CAPTURE_GATE.acquire(blocking=wait):
        yield False
        return
    try:
        yield not capture_may_be_active()
    finally:
        _CAPTURE_GATE.release()


def track_cuda_graph_cache_entry(
    owner: object,
    attribute: str,
    key: object,
    value: object,
    tensors: tuple[torch.Tensor, ...],
) -> None:
    graph = ACTIVE_CUDA_GRAPH.get()
    if graph is None:
        # Raw torch.cuda.graph callers are conservatively retained because the
        # CUDA capture ID does not expose the owning Python CUDAGraph lifetime.
        return
    graph._helion_resources.retain(
        owner,
        attribute,
        key,
        value,
        tensors,
    )


@contextmanager
def cute_cuda_graph(
    *,
    pool: _POOL_HANDLE | None = None,
    stream: torch.cuda.Stream | None = None,
    capture_error_mode: str = "global",
) -> Generator[CuteCUDAGraph, None, None]:
    """Capture CuTe launches while tying raw-pointer owners to ``graph``.

    Raw ``torch.cuda.graph`` captures cannot expose their Python graph owner to
    the launcher, so their resources are conservatively retained without bound.

    Captured native paired-attention launches may use a dependency proof to
    overlap reads of independent inputs. Such graphs must retain their captured
    node topology and parameter addresses. Updating tensor contents before
    replay is supported; replacing pointer arguments or rewiring nodes through
    raw CUDA graph handles requires a new capture. ``replay()`` accepts no new
    arguments, and ``reset()`` releases the old capture proof.

    A capture that raises leaves the caller's stream current.
    ``torch.cuda.graph`` enters the capture stream ahead of the capture's
    begin and skips its stream exit when the begin or the end raises (an
    invalidated capture), which would leave the capture stream current for
    everything the thread runs afterwards.
    """
    if torch.compiler.config.force_cudagraph_gc:
        # torch's entry collects before it begins the capture
        # (``TORCH_CUDAGRAPH_GC``), by then with the capture counted: a dead
        # module finalized there would keep its library loaded, as one
        # finalized inside the capture does. Collected here, ahead of the
        # count, it is unloaded as outside any capture.
        gc.collect()
    # Read ahead of the context variable: a raise from here (an interrupt, a
    # broken device) leaves no graph marked as the one being captured.
    prior_stream = torch.cuda.current_stream()
    graph = CuteCUDAGraph()
    token = ACTIVE_CUDA_GRAPH.set(graph)
    try:
        # Counted (``capture_may_be_active``) from ahead of torch's entry,
        # which begins the capture, until torch's exit has ended it: the
        # count is the outer context, so an entry that raises lowers it
        # again. Counting takes the capture gate, which waits for the
        # driver calls of a thread that found no capture underway
        # (``outside_captures``): those calls return before this capture
        # begins, and every later question finds it counted. The ``with``
        # statement pairs torch's entry with its exit: a capture that began
        # is ended whatever is raised afterwards.
        with (
            _capture_underway(),
            torch.cuda.graph(
                graph,
                pool=pool,
                stream=stream,
                capture_error_mode=capture_error_mode,
            ),
        ):
            yield graph
    except BaseException:
        torch.cuda.set_stream(prior_stream)
        raise
    finally:
        ACTIVE_CUDA_GRAPH.reset(token)


@dataclass(frozen=True)
class GroupedStaticMetadataResult:
    problem_sizes: torch.Tensor | None
    starts: torch.Tensor | None
    total_clusters: int
    real_groups: torch.Tensor | None = None
    runtime_tile_records: torch.Tensor | None = None
    direct_pointers: torch.Tensor | None = None
    direct_strides: torch.Tensor | None = None

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            *((self.problem_sizes,) if self.problem_sizes is not None else ()),
            *((self.starts,) if self.starts is not None else ()),
            *((self.real_groups,) if self.real_groups is not None else ()),
            *(
                (self.runtime_tile_records,)
                if self.runtime_tile_records is not None
                else ()
            ),
            *((self.direct_pointers,) if self.direct_pointers is not None else ()),
            *((self.direct_strides,) if self.direct_strides is not None else ()),
        )


@dataclass(frozen=True)
class GroupedStaticMetadataCacheEntry:
    layout_ref: weakref.ReferenceType[torch.Tensor]
    n_sizes_ref: weakref.ReferenceType[torch.Tensor] | None
    k_sizes_ref: weakref.ReferenceType[torch.Tensor] | None
    has_m_tail: bool
    has_n_tail: bool
    result: GroupedStaticMetadataResult

    def matches(
        self,
        layout: torch.Tensor,
        n_sizes: torch.Tensor | None,
        k_sizes: torch.Tensor | None,
    ) -> bool:
        if self.layout_ref() is not layout:
            return False
        if self.n_sizes_ref is None:
            if n_sizes is not None:
                return False
        elif self.n_sizes_ref() is not n_sizes:
            return False
        if self.k_sizes_ref is None:
            return k_sizes is None
        return self.k_sizes_ref() is k_sizes


@dataclass(frozen=True)
class LaunchArgCacheEntry:
    schema: tuple[tuple[object, ...], ...]
    launch_args: tuple[object, ...]
    grouped_static_metadata: tuple[GroupedStaticMetadataCacheEntry, ...]
    owned_tensors: tuple[torch.Tensor, ...]


def cuda_stream_capture_context(device: torch.device) -> tuple[int, int | None]:
    """Return the current stream and its active capture ID, if any."""
    cuda_runtime = importlib.import_module("cuda.bindings.runtime")
    with torch.cuda.device(device):
        stream_handle = int(torch.cuda.current_stream().cuda_stream)
        capture_info = cuda_runtime.cudaStreamGetCaptureInfo(
            cuda_runtime.cudaStream_t(stream_handle)
        )
    error = capture_info[0]
    status = capture_info[1]
    if error != cuda_runtime.cudaError_t.cudaSuccess:
        raise exc.BackendUnsupported(
            "cute",
            f"failed to query the CUDA stream capture context: {error}",
        )
    if status == cuda_runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone:
        return stream_handle, None
    if status == cuda_runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return stream_handle, int(capture_info[2])
    raise exc.BackendUnsupported(
        "cute", "the current CUDA stream capture has been invalidated"
    )


MMA_DAG_FIXUP_WORKSPACE_CACHE_LIMIT = 8
SPLIT_K_WORKSPACE_CACHE_LIMIT = 8


def cached_exchange_workspace(
    cute_kernel: object,
    attribute: str,
    cache_key: tuple[object, ...],
    *,
    device: torch.device,
    nbytes: int,
    zero_from: int,
    capture_id: int | None,
    limit: int,
) -> torch.Tensor:
    """A launcher-owned exchange workspace cached on ``cute_kernel`` under
    ``attribute`` per ``cache_key`` (which ends in the capture id): bytes
    from ``zero_from`` on are zeroed once at allocation; a captured graph's
    entry is tracked with the graph, eager entries are bounded by ``limit``.
    """
    try:
        cache = cast(
            "OrderedDict[tuple[object, ...], torch.Tensor]",
            getattr(cute_kernel, attribute),
        )
    except AttributeError:
        cache = OrderedDict()
        setattr(cute_kernel, attribute, cache)
    workspace = cache.get(cache_key)
    if workspace is not None:
        return workspace
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    workspace[zero_from:].zero_()
    cache[cache_key] = workspace
    if capture_id is not None:
        track_cuda_graph_cache_entry(
            cute_kernel, attribute, cache_key, workspace, (workspace,)
        )
    else:
        eager_keys = [key for key in cache if key[-1] is None]
        while len(eager_keys) > limit:
            cache.pop(eager_keys.pop(0))
    return workspace
