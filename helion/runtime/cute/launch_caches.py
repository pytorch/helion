"""Launch-time caches of the CuTe launcher and their CUDA-graph ownership.

The launcher keeps three kinds of per-kernel state between launches: the
marshalled launch arguments (:class:`LaunchArgCacheEntry`, keyed by the
argument signature), the grouped-GEMM scheduler metadata copied from the
layout tensors, and launcher-owned exchange workspaces.  A CUDA graph captured
through :func:`cute_cuda_graph` embeds the raw pointers of those entries, so
:class:`CudaGraphResources` ties them to the graph's lifetime; raw
``torch.cuda.graph`` captures are retained conservatively.
:func:`cuda_stream_capture_context` is the ``(stream, capture id)`` pair that
isolates mutable or capture-owned launch tensors per stream.  Leaf module:
the launcher imports it, never the reverse.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import importlib
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
    """
    graph = CuteCUDAGraph()
    token = ACTIVE_CUDA_GRAPH.set(graph)
    try:
        with torch.cuda.graph(
            graph,
            pool=pool,
            stream=stream,
            capture_error_mode=capture_error_mode,
        ):
            yield graph
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
