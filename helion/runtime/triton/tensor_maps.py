"""Tensor maps (TMA descriptors) in global memory for the tensor form of
``hl.prefetch``: ``prefetch_map`` encodes and caches one per prefetched tensor
and box, and the generated host code passes its address to the kernel."""

from __future__ import annotations

import collections
import ctypes
import functools
import threading

import torch


class _PrefetchMap:
    """A cached tensor map.  A plain class, not a dataclass: Helion-free
    output inlines this module, and a dataclass resolves its annotations
    through sys.modules, which a module loaded without registering it there
    lacks."""

    def __init__(self, host: torch.Tensor, device: torch.Tensor) -> None:
        self.host = host  # pinned
        self.device = device
        # Recorded after the first eager launch's copy, on that launch's stream.
        self.copied: torch.cuda.Event | None = None
        # That copy completed: every stream reads the map.
        self.written = False
        # A CUDA graph captured a launch reading the map: it holds the address.
        self.captured = False
        # The streams eager launches read the map on (recorded on its memory).
        self.streams: set[int] = set()

    @property
    def address(self) -> int:
        return self.device.data_ptr()


# Tensor maps of tensor-form L2 prefetches, by (address, dtype, shape, strides,
# box, device): built once per tensor (a host launch builds no descriptor), and
# kept with their pinned host copies while launches and captured graphs read
# them.  Their device memory comes from a pool of their own: under capture,
# not from the capture's pool, which CUDA graph trees reset under the maps
# they keep.  A map a graph captured stays; the others are dropped past
# _MAX_PREFETCH_MAPS of them on a device (prefetched tensors whose address
# changes every call).  A launch's host code holds its maps until the launch
# is submitted, and a freed map's memory waits for the streams that read it.
_PREFETCH_MAPS: dict[tuple[object, ...], _PrefetchMap] = {}
_MAX_PREFETCH_MAPS = 4096
_UNCAPTURED_PREFETCH_MAPS: collections.Counter[torch.device] = collections.Counter()
_CAPTURED_PREFETCH_MAPS: list[_PrefetchMap] = []
_PREFETCH_MAP_POOLS: dict[torch.device, torch.cuda.MemPool] = {}
_PREFETCH_MAP_LOCK = threading.Lock()


def prefetch_map(base: torch.Tensor, box: list[int]) -> _PrefetchMap:
    """A tensor map (TMA descriptor) of ``base`` with box ``box`` in global
    memory, for ``cp.async.bulk.prefetch.tensor`` (a prefetch only reads the
    map, so it carries no shared layout: no swizzle, the element type only
    sizes the dims); its ``address`` is the kernel argument, and the launch
    holds the map until it is submitted.  A map an eager launch builds is
    copied on that launch's stream before another thread can see it; one
    built under CUDA graph capture, by the first eager launch.  A launch on
    another stream waits for that copy until it completes.  Until a copy
    exists each capture records its own (a replay may come before any other
    graph's)."""
    key = (
        base.data_ptr(),
        base.dtype,
        tuple(base.shape),
        tuple(base.stride()),
        tuple(box),
        base.device,
    )
    if (entry := _PREFETCH_MAPS.get(key)) is None:
        _drop_uncaptured_prefetch_maps(base.device)
        host = _encode_tensor_map(base, box).pin_memory()
        with _PREFETCH_MAP_LOCK:
            if (pool := _PREFETCH_MAP_POOLS.get(base.device)) is None:
                pool = _PREFETCH_MAP_POOLS[base.device] = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool, base.device):
            fresh = _PrefetchMap(host, torch.empty_like(host, device=base.device))
        if not torch.cuda.is_current_stream_capturing():
            # Written on the allocating stream before it is published: the
            # allocator may hand out memory that work queued earlier on this
            # stream still reads, which a copy on another stream would not
            # wait for.
            fresh.device.copy_(host, non_blocking=True)
            fresh.copied = torch.cuda.Event()
            fresh.copied.record(torch.cuda.current_stream(base.device))
        # A concurrent miss on another thread may have published first: use
        # the entry that stays in the cache.
        with _PREFETCH_MAP_LOCK:
            entry = _PREFETCH_MAPS.setdefault(key, fresh)
            if entry is fresh:
                _UNCAPTURED_PREFETCH_MAPS[base.device] += 1
    if torch.cuda.is_current_stream_capturing():
        if not entry.captured:
            with _PREFETCH_MAP_LOCK:
                if not entry.captured:
                    entry.captured = True
                    # Kept for good, even if a drop took it since the lookup.
                    _CAPTURED_PREFETCH_MAPS.append(entry)
                    if _PREFETCH_MAPS.get(key) is entry:
                        _UNCAPTURED_PREFETCH_MAPS[base.device] -= 1
        if not entry.written:
            entry.device.copy_(entry.host, non_blocking=True)
        return entry
    stream = torch.cuda.current_stream(base.device)
    if stream.stream_id not in entry.streams:
        # Freed, the map's memory waits for the work queued on this stream.
        entry.device.record_stream(stream)
        entry.streams.add(stream.stream_id)
    if not entry.written:
        if entry.copied is None:
            entry.device.copy_(entry.host, non_blocking=True)
            # Recorded before it is published: another thread may query it.
            copied = torch.cuda.Event()
            copied.record(stream)
            entry.copied = copied
        elif entry.copied.query():
            entry.written = True
        else:
            stream.wait_event(entry.copied)
    return entry


def _drop_uncaptured_prefetch_maps(device: torch.device) -> None:
    """Past _MAX_PREFETCH_MAPS maps on ``device`` no graph captured, drop them
    on an eager miss; a map no launch holds is freed at once."""
    if (
        torch.cuda.is_current_stream_capturing()
        or _UNCAPTURED_PREFETCH_MAPS[device] < _MAX_PREFETCH_MAPS
    ):
        return
    with _PREFETCH_MAP_LOCK:
        for key, entry in list(_PREFETCH_MAPS.items()):
            if key[-1] == device and not entry.captured:
                del _PREFETCH_MAPS[key]
                _UNCAPTURED_PREFETCH_MAPS[device] -= 1


@functools.cache
def _libcuda() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


def _encode_tensor_map(base: torch.Tensor, box: list[int]) -> torch.Tensor:
    rank, size = base.ndim, base.element_size()
    # CU_TENSOR_MAP_DATA_TYPE_UINT8 / UINT16 / UINT32 / UINT64 by element size.
    dtype = {1: 0, 2: 1, 4: 2, 8: 4}[size]
    # Innermost dim first; strides in bytes, of the dims past the innermost.
    dims = (ctypes.c_uint64 * rank)(*reversed(base.shape))
    strides = (ctypes.c_uint64 * (rank - 1))(
        *[s * size for s in reversed(base.stride()[:-1])]
    )
    boxes = (ctypes.c_uint32 * rank)(*reversed(box))
    ones = (ctypes.c_uint32 * rank)(*([1] * rank))
    # The encoder wants a 64-byte aligned map.
    storage = (ctypes.c_uint8 * (128 + 63))()
    tensor_map = (ctypes.addressof(storage) + 63) & ~63
    encode = _libcuda().cuTensorMapEncodeTiled
    # interleave none, swizzle none, L2 promotion none, OOB fill none
    result = encode(
        ctypes.c_void_p(tensor_map),
        dtype,
        rank,
        ctypes.c_void_p(base.data_ptr()),
        dims,
        strides,
        boxes,
        ones,
        0,
        0,
        0,
        0,
    )
    if result != 0:
        raise RuntimeError(
            f"cuTensorMapEncodeTiled failed ({result}) for a {list(base.shape)} "
            f"{base.dtype} tensor with strides {list(base.stride())}, box {box}"
        )
    raw = bytearray(ctypes.string_at(tensor_map, 128))
    # Triton's (CUTLASS's) fix for drivers up to 13.1: a tensor spanning under
    # 128 KB clears bit 21 of the map's second word.
    span = 1 + sum(
        (n - 1) * s * size for n, s in zip(base.shape, base.stride(), strict=True)
    )
    if _driver_version() <= 13010 and span < 128 * 1024:
        word = int.from_bytes(raw[8:16], "little") & ~(1 << 21)
        raw[8:16] = word.to_bytes(8, "little")
    return torch.frombuffer(raw, dtype=torch.uint8)


@functools.cache
def _driver_version() -> int:
    version = ctypes.c_int()
    if _libcuda().cuDriverGetVersion(ctypes.byref(version)) != 0:
        raise RuntimeError("cuDriverGetVersion failed")
    return version.value
