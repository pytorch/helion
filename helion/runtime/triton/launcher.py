"""Helion-dependency-free runtime launch helpers for the Triton backend.

This module holds the small set of runtime symbols that Helion's *generated*
Triton code depends on at execution time:

* :func:`default_launcher` -- invokes a compiled ``triton.jit`` kernel.
* :func:`get_num_sm` -- persistent-kernel grid size (host statement).
* :func:`set_triton_allocator` -- installs the scratch allocator used by TMA /
  tensor-descriptor kernels (device-function prefix statement).

It depends only on ``torch`` and ``triton`` -- no other ``helion`` module -- so
the ahead-of-time precompiler can bulk-export this file verbatim into a
standalone kernel with zero Helion runtime dependency.

Helion-specific behavior that is only meaningful in-process (translating
Triton's opaque shape errors into :class:`helion.exc.ShapeMismatch`, and the
CPU/TPU cases of :func:`get_num_sm`) lives in thin wrappers in
:mod:`helion.runtime`, not here.
"""

from __future__ import annotations

import contextlib
import contextvars
import ctypes
import functools
import math
from typing import TYPE_CHECKING
import weakref

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    import triton
except ImportError:
    triton = None  # type: ignore[assignment]


if triton is not None:

    def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
        # Dynamically get device from Triton backend
        current_target = triton.runtime.driver.active.get_current_target()
        if current_target is None:
            raise RuntimeError("No active Triton target available")
        backend = current_target.backend
        return torch.empty(size, device=backend, dtype=torch.int8)

    def set_triton_allocator() -> None:
        try:
            from triton import set_allocator
            from triton.runtime._allocation import NullAllocator
            from triton.runtime._allocation import _allocator
        except ImportError:
            return
        if isinstance(_allocator, contextvars.ContextVar):
            existing = _allocator.get()
        else:  # older versions of Triton
            existing = _allocator
        # if allocator isn't NullAllocator, we assume it is set by the user
        if isinstance(existing, NullAllocator):
            set_allocator(_alloc_fn)

else:

    def set_triton_allocator() -> None:  # type: ignore[misc]
        pass


def get_num_sm(device: torch.device, *, reserved_sms: int = 0) -> int:
    """
    Get the number of streaming multiprocessors (SMs) for the specified GPU.

    Args:
        device: Device to query. Must be a GPU device (``cuda``/``xpu``/``mps``/
            ``mtia``); CPU/TPU handling lives in :func:`helion.runtime.get_num_sm`.
        reserved_sms: Number of SMs to keep free for other work (e.g., communication
            kernels). Defaults to 0 meaning all device SMs are available to Helion.

    Returns:
        Grid size to use for a persistent kernel on the device after accounting
        for any reserved SMs. Always at least 1.
    """
    available_sms: int
    assert device.type in [
        "cuda",
        "xpu",
        "mtia",
        "mps",
    ], "TODO: implement for other devices"
    if device.type == "cuda":
        available_sms = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count
    # TODO(EikanWang): gpu_subslice_count is an out-of-date term. we change update it to XeCore number.
    elif device.type == "xpu":
        available_sms = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    elif device.type == "mps":
        available_sms = torch.backends.mps.get_core_count()
    elif device.type == "mtia":
        device_props = torch.mtia.get_device_properties(device.index)
        if "max_grid_height" in device_props and "max_grid_width" in device_props:
            available_sms = (
                device_props["max_grid_height"] * device_props["max_grid_width"]
            )
        else:
            raise RuntimeError(
                f"Unable to determine SM count for MTIA device. "
                f"Available properties: {list(device_props.keys())}"
            )
    else:
        raise NotImplementedError(
            f"get_num_sm not implemented for device type: {device.type}"
        )

    if reserved_sms <= 0:
        return available_sms
    return max(available_sms - reserved_sms, 1)


# CUs per XCD by base CDNA architecture.  Used to derive the live,
# partition-visible XCD count from the observed CU count (see get_num_xcd).
_CUS_PER_XCD: dict[str, int] = {
    "gfx942": 38,  # CDNA3 (MI300)
    "gfx950": 32,  # CDNA4 (MI350)
    "gfx951": 32,  # CDNA4 (MI355)
}


def get_num_xcd(device: torch.device | int | None = None) -> int:
    """Number of XCDs visible for ``device`` on AMD CDNA, else ``1``.

    Derived from the live, partition-visible compute-unit count rather than the
    architecture name, so MI300A (6 XCDs) and compute-partition modes such as CPX
    (which expose a single XCD) are handled correctly.  Returns ``1`` -- which
    disables xcd_remap -- for unknown architectures or a CU count that does not
    look like an integer number of XCDs.
    """
    if not torch.cuda.is_available():
        return 1
    try:
        props = torch.cuda.get_device_properties(
            device if device is not None else torch.cuda.current_device()
        )
    except Exception:
        return 1
    arch = getattr(props, "gcnArchName", None)
    if not arch:
        return 1
    cus_per_xcd = _CUS_PER_XCD.get(arch.split(":")[0])
    if cus_per_xcd is None:
        return 1
    cu_count = props.multi_processor_count
    num_xcd = round(cu_count / cus_per_xcd)
    # Tolerate harvested parts, but bail out (return 1) if the live CU count does
    # not look like an integer number of XCDs.
    if num_xcd < 1 or abs(num_xcd * cus_per_xcd - cu_count) > cus_per_xcd // 4:
        return 1
    return num_xcd


def default_launcher(
    triton_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    _remote_copy_signal_dst: torch.Tensor | None = None,
    _remote_copy_signal_slots_per_program: int = 0,
    _remote_copy_process_group_name: str | None = None,
    _remote_barrier_signal_slots_per_program: int = 0,
    _remote_barrier_process_group_name: str | None = None,
    _remote_copy_scratch_specs: tuple[tuple[torch.Tensor, int], ...] = (),
    _persistent_state_specs: tuple[
        tuple[torch.Tensor, int, torch.dtype, bool], ...
    ] = (),
    _persistent_state_process_group_name: str | None = None,
    _persistent_state_rank_digest: str | None = None,
    _minimum_resident_programs: int = 0,
    _ttgir_weight_prefetch: bool = False,
    _ttgir_weight_ring: bool = False,
    _tensor_storage_aliases: tuple[tuple[int, ...], ...] = (),
    _check_storage_aliasing: bool = False,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    # The schedule of a multi-root kernel (cross-loop state, or the phases of
    # fused roots) orders accesses through tensors that share a storage only
    # if it was compiled for that sharing.  A helion.Kernel call recompiles
    # for each sharing pattern; code run without the compiler (exported, AOT,
    # ``bound._run``) is checked here.
    checked = bool(_persistent_state_specs) or _check_storage_aliasing
    shared_storages = (
        _shared_storages(args) if checked or _ttgir_weight_prefetch else ()
    )
    if checked and shared_storages:
        _check_compiled_storage_sharing(
            triton_kernel, shared_storages, _tensor_storage_aliases
        )
    if _remote_copy_signal_slots_per_program:
        if _remote_copy_signal_dst is None or _remote_copy_process_group_name is None:
            raise RuntimeError(
                "remote-copy completion storage requires a symmetric destination "
                "and process group"
            )
        signal = _get_remote_copy_signal(
            triton_kernel,
            _remote_copy_signal_dst,
            _remote_copy_process_group_name,
            math.prod(grid) * _remote_copy_signal_slots_per_program,
        )
        # Allocation zeroes new pads and receive waits reset consumed slots.
        # Clearing here could erase a completion sent before this rank launches.
        args = (*args, signal)
    if _remote_barrier_signal_slots_per_program:
        if _remote_barrier_process_group_name is None:
            raise RuntimeError(
                "remote-barrier completion storage requires a process group"
            )
        signal = _get_remote_barrier_signal(
            triton_kernel,
            _remote_barrier_process_group_name,
            math.prod(grid) * _remote_barrier_signal_slots_per_program,
        )
        args = (*args, signal)
    for slot, (scratch_like, numel_per_program) in enumerate(
        _remote_copy_scratch_specs
    ):
        scratch = _get_remote_copy_scratch(
            triton_kernel,
            scratch_like,
            slot,
            math.prod(grid) * numel_per_program,
        )
        args = (*args, scratch)
    if _persistent_state_specs:
        persistent_state_namespace = (
            tuple(grid),
            num_warps,
            num_stages,
            ptx_options,
            launch_cooperative_grid,
            tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
            tuple(spec[1:] for spec in _persistent_state_specs),
            # Symmetric state exchanges the namespace, so ranks compare digests.
            _persistent_state_rank_digest,
        )
        for slot, (state_like, numel, dtype, symmetric) in enumerate(
            _persistent_state_specs
        ):
            if _persistent_state_process_group_name is None:
                state_args = (
                    _get_persistent_state(
                        triton_kernel,
                        state_like,
                        persistent_state_namespace,
                        slot,
                        numel,
                        dtype,
                    ),
                )
            else:
                state_args = _get_process_group_state(
                    triton_kernel,
                    state_like,
                    persistent_state_namespace,
                    slot,
                    numel,
                    dtype,
                    symmetric=symmetric,
                    process_group_name=_persistent_state_process_group_name,
                )
            args = (*args, *state_args)
    # For both CUDA and MTIA, use the same kernel execution.
    run_kwargs: dict = {
        "grid": grid,
        "warmup": False,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "launch_cooperative_grid": launch_cooperative_grid,
        **kwargs,
    }
    if ptx_options is not None:
        run_kwargs["ptx_options"] = ptx_options
    # The pass is part of the kernel's cache key, so it covers every compile
    # and lookup of this kernel, including the occupancy probe.  It tells
    # written from read-only tensors by argument, writing together arguments
    # the compiler saw share a storage (e.g. a tensor and its TMA descriptor);
    # a launch whose arguments share storage otherwise runs without it.
    alias_classes = (
        prefetch_alias_classes(
            triton_kernel, args, _tensor_storage_aliases, shared_storages
        )
        if _ttgir_weight_prefetch
        else None
    )
    with ttgir_weight_prefetch_context(
        alias_classes is not None, alias_classes or (), ring=_ttgir_weight_ring
    ):
        if _minimum_resident_programs:
            # ``triton_kernel`` is a JITFunction.  Resource information belongs to
            # its exact compiled specialization, so compile (but do not launch)
            # that specialization before asking CUDA for its occupancy.
            compiled_kernel = triton_kernel.run(  # type: ignore[union-attr]
                *args,
                **{**run_kwargs, "warmup": True},
            )
            _validate_resident_program_capacity(
                compiled_kernel,
                args,
                num_warps=num_warps,
                required_programs=_minimum_resident_programs,
            )
        return triton_kernel.run(  # type: ignore[union-attr]
            *args,
            **run_kwargs,
        )


def _shared_storages(args: tuple[object, ...]) -> tuple[tuple[int, ...], ...]:
    """Positions of the tensor (and tensor descriptor) arguments on each
    storage that more than one of them views.  Storages whose
    ``[data_ptr, data_ptr + nbytes)`` ranges overlap (separately constructed
    over one buffer, e.g. by ``from_dlpack``) count as one."""
    spans: list[tuple[int, int, int]] = []
    for index, arg in enumerate(args):
        tensor = getattr(arg, "base", arg)  # TensorDescriptor -> its tensor
        if isinstance(tensor, torch.Tensor):
            storage = tensor.untyped_storage()
            start = storage.data_ptr()
            nbytes = storage.nbytes()
            if start and nbytes:  # empty tensors share the null pointer
                spans.append((start, start + nbytes, index))
    if len(spans) < 2:
        return ()
    spans.sort()
    groups: list[tuple[int, ...]] = []
    members: list[int] = []
    reach = 0
    for start, end, index in spans:
        if start >= reach:  # disjoint from every earlier storage
            if len(members) > 1:
                groups.append(tuple(sorted(members)))
            members = []
        members.append(index)
        reach = max(reach, end)
    if len(members) > 1:
        groups.append(tuple(sorted(members)))
    return tuple(sorted(groups))


def _distinct_storages(args: tuple[object, ...]) -> bool:
    return not _shared_storages(args)


def prefetch_alias_classes(
    triton_kernel: object,
    args: tuple[object, ...],
    compiled: tuple[tuple[int, ...], ...],
    shared: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[tuple[str, ...], ...] | None:
    """The storage classes the TTGIR pass compiles for (argument names of the
    positions that share a storage at compile time, ``compiled``), or None if
    the launch's sharing (``shared``, by default that of ``args``) is not
    within them: then the pass must not run.  Arguments the compiler saw
    share a storage need not share it at launch; the pass then only writes
    more than it must."""
    if shared is None:
        shared = _shared_storages(args)
    group_of = {
        index: group for group, members in enumerate(compiled) for index in members
    }
    for positions in shared:
        groups = {group_of.get(index) for index in positions}
        if len(groups) != 1 or None in groups:
            return None
    if not compiled:
        return ()
    names = getattr(triton_kernel, "arg_names", None)
    if not isinstance(names, (list, tuple)) or any(
        index >= len(names) for members in compiled for index in members
    ):
        return None
    return tuple(tuple(str(names[index]) for index in members) for members in compiled)


def _check_compiled_storage_sharing(
    triton_kernel: object,
    shared: tuple[tuple[int, ...], ...],
    compiled: tuple[tuple[int, ...], ...],
) -> None:
    """Refuse a launch whose arguments share a storage the compiler did not
    see them share (``compiled``: positions sharing one at compile time)."""
    group_of = {
        index: group for group, members in enumerate(compiled) for index in members
    }
    for positions in shared:
        if len({group_of.get(index, -1 - index) for index in positions}) == 1:
            continue
        names = getattr(triton_kernel, "arg_names", None)
        described = ", ".join(
            repr(names[index]) if names and index < len(names) else str(index)
            for index in positions
        )
        raise RuntimeError(
            f"Cross-loop launch aliasing: arguments {described} share one "
            "storage, but the kernel was compiled for tensors that do not share "
            "it, and its cross-loop schedule orders accesses only between "
            "tensors the compiler saw share storage.  Call the helion.Kernel "
            "(its cache key includes the storage aliasing of tensor inputs), or "
            "bind it with arguments that alias like these."
        )


def ttgir_weight_prefetch_context(
    enable: bool,
    alias_classes: tuple[tuple[str, ...], ...] = (),
    ring: bool = False,
) -> contextlib.AbstractContextManager[None]:
    """Compile kernels with the TTGIR weight prefetch pass inside the context;
    ``alias_classes``: argument names that may share a storage; ``ring``: the
    pass tries its warp-specialized weight ring first (the kernel's
    ``ttgir_weight_ring`` setting, emitted as ``_ttgir_weight_ring``)."""
    if enable:
        try:
            from .ttgir_weight_prefetch import enabled
        except ImportError:  # a standalone export runs without the pass
            pass
        else:
            return enabled(alias_classes, ring=ring)
    return contextlib.nullcontext()


def compile_only_launch_args(
    *args: object,
    _remote_copy_signal_slots_per_program: int = 0,
    _remote_barrier_signal_slots_per_program: int = 0,
    _remote_copy_scratch_specs: tuple[tuple[torch.Tensor, int], ...] = (),
    _persistent_state_specs: tuple[
        tuple[torch.Tensor, int, torch.dtype, bool], ...
    ] = (),
    **kwargs: object,
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Return ``triton_kernel.run`` arguments for compiling without launching.

    The arguments ``default_launcher`` appends become empty tensors of the same
    dtypes, which is all compilation reads, and launcher-only options drop.
    """
    dtypes = [torch.int64] * (
        bool(_remote_copy_signal_slots_per_program)
        + bool(_remote_barrier_signal_slots_per_program)
    )
    dtypes += [like.dtype for like, _ in _remote_copy_scratch_specs]
    for _, _, dtype, symmetric in _persistent_state_specs:
        dtypes += [dtype, torch.int64] if symmetric else [dtype]
    kwargs = {name: value for name, value in kwargs.items() if not name.startswith("_")}
    if not dtypes:
        return args, kwargs
    device = next(arg.device for arg in args if isinstance(arg, torch.Tensor))
    placeholders = [torch.empty(0, dtype=dtype, device=device) for dtype in dtypes]
    return (*args, *placeholders), kwargs


def _get_remote_copy_signal(
    triton_kernel: object,
    dst: torch.Tensor,
    process_group_name: str,
    required_slots: int,
) -> torch.Tensor:
    """Return compiler-owned completion slots from ``dst``'s signal pad."""
    import torch.distributed._symmetric_memory as symm_mem

    cache = vars(triton_kernel).setdefault("_helion_remote_copy_signal_cache", {})

    key = (id(dst), process_group_name)
    entry = cache.get(key)
    if entry is not None and entry[0]() is dst:
        signal_pad = entry[1]
    else:
        handle = symm_mem.rendezvous(
            dst,
            group=process_group_name,  # pyrefly: ignore[bad-argument-type]
        )
        signal_pad = handle.get_signal_pad(handle.rank, dtype=torch.int64)

        def remove_from_cache(_ref: object) -> None:
            cache.pop(key, None)

        cache[key] = (weakref.ref(dst, remove_from_cache), signal_pad)

    capacity = signal_pad.numel()
    if required_slots > capacity:
        raise RuntimeError(
            "Helion remote copies require "
            f"{required_slots} int64 completion slots, but the symmetric-memory "
            f"signal pad has capacity {capacity}. Increase the signal pad size "
            "before allocating symmetric tensors."
        )
    # Reserve from the end so Helion's slots do not overlap PyTorch's standard
    # low-offset signal-pad protocols.
    return signal_pad.narrow(0, capacity - required_slots, required_slots)


def _get_remote_barrier_signal(
    triton_kernel: object,
    process_group_name: str,
    required_slots: int,
) -> torch.Tensor:
    """Return compiler-owned peer-barrier counters from a group workspace."""
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.device("cuda", torch.cuda.current_device())
    cache = vars(triton_kernel).setdefault("_helion_remote_barrier_signal_cache", {})
    key = (device, process_group_name)
    entry = cache.get(key)
    if entry is None:
        workspace = symm_mem.empty(1, dtype=torch.uint8, device=device)
        handle = symm_mem.rendezvous(
            workspace,
            group=process_group_name,  # pyrefly: ignore[bad-argument-type]
        )
        cache[key] = (workspace, handle)
    else:
        _, handle = entry
    signal_pad = handle.get_signal_pad(handle.rank, dtype=torch.int64)
    capacity = signal_pad.numel()
    if required_slots > capacity:
        raise RuntimeError(
            "Helion remote barriers require "
            f"{required_slots} int64 completion slots, but the symmetric-memory "
            f"signal pad has capacity {capacity}. Increase the signal pad size "
            "before launching the kernel."
        )
    return signal_pad.narrow(0, capacity - required_slots, required_slots)


def _get_remote_copy_scratch(
    triton_kernel: object,
    like: torch.Tensor,
    slot: int,
    required_numel: int,
) -> torch.Tensor:
    """Return stream-local global scratch for one computed DMA source."""
    if like.device.type != "cuda":
        raise RuntimeError("NVSHMEM remote-copy scratch requires a CUDA tensor")
    stream = torch.cuda.current_stream(like.device)
    cache = vars(triton_kernel).setdefault("_helion_remote_copy_scratch_cache", {})
    key = (like.device, like.dtype, stream.cuda_stream, slot)
    scratch = cache.get(key)
    if scratch is None or scratch.numel() < required_numel:
        scratch = torch.empty(
            required_numel,
            dtype=like.dtype,
            device=like.device,
        )
        cache[key] = scratch
    return scratch


def _get_persistent_state(
    triton_kernel: object,
    like: torch.Tensor,
    namespace: tuple[object, ...],
    slot: int,
    required_numel: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return stream-local compiler state retained across kernel launches."""
    if like.device.type != "cuda":
        raise RuntimeError("persistent Triton state requires a CUDA tensor")
    stream = torch.cuda.current_stream(like.device)
    cache = vars(triton_kernel).setdefault("_helion_persistent_state_cache", {})
    key = (like.device, dtype, stream.cuda_stream, namespace, slot)
    state = cache.get(key)
    if state is None or state.numel() < required_numel:
        state = _new_zeroed_state(required_numel, dtype, like.device)
        cache[key] = state
    return state


_CU_STREAM_CAPTURE_MODE_RELAXED = 2


@functools.cache
def _cuda_driver() -> object | None:
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuThreadExchangeStreamCaptureMode.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuThreadExchangeStreamCaptureMode.restype = ctypes.c_int
    except (OSError, AttributeError):
        return None
    return lib


@contextlib.contextmanager
def _relaxed_stream_capture_mode(lib: object) -> Iterator[None]:
    """Allow this thread "potentially unsafe" CUDA calls (an allocation or a
    host wait on a stream that is not capturing) during a capture, as
    PyTorch's caching allocator does for its own allocations
    (``CUDAStreamCaptureModeGuard``)."""
    mode = ctypes.c_int(_CU_STREAM_CAPTURE_MODE_RELAXED)
    if lib.cuThreadExchangeStreamCaptureMode(ctypes.byref(mode)) != 0:  # type: ignore[attr-defined]
        raise RuntimeError("cuThreadExchangeStreamCaptureMode failed")
    try:
        yield
    finally:
        lib.cuThreadExchangeStreamCaptureMode(ctypes.byref(mode))  # type: ignore[attr-defined]


def _is_capturing(device: torch.device) -> bool:
    """Whether ``device``'s current stream (the stream the state is keyed by
    and the launch runs on) is capturing; not the current device's."""
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


_STATE_INIT_STREAMS: dict[torch.device, torch.cuda.Stream] = {}
# Retained state allocated during a capture, kept alive for the life of the
# process: graphs that captured its launches address it, but (unlike the
# graph's private pool) nothing else ties its memory to those graphs, and the
# launcher's cache entry may go (a kernel object freed or regenerated).
_CAPTURED_STATES: list[torch.Tensor] = []


def _new_zeroed_state(
    numel: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Zeroed retained state, allocated and zeroed outside any ongoing capture.

    Retained state only has to be zero before its first launch: launches read
    their epoch from it and leave epochs and counters consistent for the next
    launch. Requested while the current stream is capturing, ``torch.zeros``
    would instead record the allocation (from the graph's private pool) and
    its fill kernel into the graph, so every replay would re-zero the state:
    one more kernel per replay, in front of the kernel's first launch, which
    also cuts that launch's programmatic (PDL) edge to its predecessor. So the
    state is allocated on a side stream that is not capturing (the regular
    pool), zeroed there, and the host waits for the zeroes, which are then in
    place before the graph is first launched.
    """
    lib = _cuda_driver() if device.type == "cuda" else None
    if lib is None or not _is_capturing(device):
        return torch.zeros(numel, dtype=dtype, device=device)
    with _relaxed_stream_capture_mode(lib), torch.cuda.device(device):
        side = _STATE_INIT_STREAMS.get(device)
        if side is None:
            side = _STATE_INIT_STREAMS[device] = torch.cuda.Stream(device)
        with torch.cuda.stream(side):
            state = torch.zeros(numel, dtype=dtype, device=device)
        side.synchronize()
    _CAPTURED_STATES.append(state)
    return state


def _get_process_group_state(
    triton_kernel: object,
    like: torch.Tensor,
    namespace: tuple[object, ...],
    slot: int,
    required_numel: int,
    dtype: torch.dtype,
    *,
    symmetric: bool,
    process_group_name: str,
) -> tuple[object, ...]:
    """Return the kernel arguments of retained state shared by every stream.

    One epoch then covers the dispatch ticket and peer counters, so launches of
    one kernel must not overlap across streams. A symmetric state is followed by
    its per-rank base pointer table.
    """
    if like.device.type != "cuda":
        raise RuntimeError("persistent Triton state requires a CUDA tensor")
    cache = vars(triton_kernel).setdefault("_helion_persistent_state_cache", {})
    # The numel is part of the namespace, so a cached entry always fits.
    key = (like.device, dtype, process_group_name, namespace, slot)
    entry = cache.get(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Helion allocates distributed launch state on first use; run the "
                "kernel once before CUDA graph capture"
            )
        if symmetric:
            entry = _new_symmetric_state(
                like.device,
                (
                    triton_kernel.__name__,  # type: ignore[attr-defined]
                    namespace,
                    slot,
                    required_numel,
                    str(dtype),
                ),
                required_numel,
                dtype,
                process_group_name,
            )
        else:
            entry = (torch.zeros(required_numel, dtype=dtype, device=like.device),)
        cache[key] = entry
    return entry[:2]


def _new_symmetric_state(
    device: torch.device,
    fingerprint: tuple[object, ...],
    numel: int,
    dtype: torch.dtype,
    process_group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """Allocate zeroed symmetric state once every rank agrees on the launch.

    Returns the state, its per-rank base pointer table followed by this rank,
    and the owning handle.
    """
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from torch.distributed.distributed_c10d import _resolve_process_group

    group = _resolve_process_group(
        process_group_name  # pyrefly: ignore[bad-argument-type]
    )
    with torch.cuda.device(device):
        fingerprint = (*fingerprint, torch.cuda.get_device_capability(device))
        fingerprints: list[object] = [None] * dist.get_world_size(group)
        dist.all_gather_object(fingerprints, fingerprint, group=group)
        if any(other != fingerprint for other in fingerprints):
            raise RuntimeError(
                "Helion distributed kernels require the same launch on every "
                f"rank; got {fingerprints!r}"
            )
        state = symm_mem.empty(numel, dtype=dtype, device=device)
        state.zero_()
        # Zero before rendezvous, since peers may publish right after it.
        torch.cuda.current_stream(device).synchronize()
        handle = symm_mem.rendezvous(
            state,
            group=process_group_name,  # pyrefly: ignore[bad-argument-type]
        )
    ptrs = torch.tensor(
        [*handle.buffer_ptrs, handle.rank], dtype=torch.int64, device=device
    )
    return state, ptrs, handle


_TMEM_COLUMNS_PER_SM = 512


def _tcgen05_blocks_per_sm(
    cuda_driver: object,
    function: object,
    device: torch.device,
    num_warps: int,
    shared: int,
    tmem_columns: int,
) -> int:
    """Resident blocks per SM from registers, shared memory, threads and TMEM."""
    attr = cuda_driver.CUfunction_attribute  # type: ignore[attr-defined]
    dev_attr = cuda_driver.CUdevice_attribute  # type: ignore[attr-defined]
    fn = cuda_driver.CUfunction(int(function))  # type: ignore[attr-defined]
    regs = cuda_driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)[1]  # type: ignore[attr-defined]
    static_smem = cuda_driver.cuFuncGetAttribute(  # type: ignore[attr-defined]
        attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn
    )[1]
    index = device.index if device.index is not None else torch.cuda.current_device()
    cu_device = cuda_driver.cuDeviceGet(index)[1]  # type: ignore[attr-defined]

    def device_attr(name: str) -> int:
        return int(
            cuda_driver.cuDeviceGetAttribute(getattr(dev_attr, name), cu_device)[1]  # type: ignore[attr-defined]
        )

    threads = num_warps * 32
    # Registers are allocated per warp in units of 256.
    regs_per_warp = -(-max(int(regs), 1) * 32 // 256) * 256
    by_regs = device_attr("CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR") // (
        regs_per_warp * num_warps
    )
    reserved = device_attr("CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK")
    smem_per_block = int(shared) + int(static_smem) + reserved
    by_smem = device_attr(
        "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"
    ) // max(smem_per_block, 1)
    by_threads = (
        device_attr("CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR") // threads
    )
    by_blocks = device_attr("CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR")
    by_tmem = _TMEM_COLUMNS_PER_SM // int(tmem_columns)
    return max(1, min(by_regs, by_smem, by_threads, by_blocks, by_tmem))


def _validate_resident_program_capacity(
    compiled_kernel: object,
    args: tuple[object, ...],
    *,
    num_warps: int,
    required_programs: int,
) -> None:
    """Reject a polling schedule whose required CTA cohort cannot be resident."""
    import importlib

    tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if tensor is None or tensor.device.type != "cuda":
        raise RuntimeError("cross-loop residency checks require a CUDA tensor")

    if compiled_kernel is None:
        raise RuntimeError("unable to compile cross-loop scheduled kernel")

    # Accessing ``run`` initializes Triton's module/function handles without
    # launching the kernel.  Cache the exact driver result on the compiled
    # specialization because this wrapper is also called during graph capture.
    _run = compiled_kernel.run  # type: ignore[attr-defined]
    function = getattr(compiled_kernel, "function", None)
    metadata = getattr(compiled_kernel, "metadata", None)
    shared = getattr(metadata, "shared", None)
    if function is None or not isinstance(shared, int):
        raise RuntimeError("unable to query cross-loop kernel occupancy")

    device = tensor.device
    # A warp-specialized kernel launches its partitions' warps too: Triton
    # records the total in the compiled metadata.
    num_warps = max(num_warps, int(getattr(metadata, "num_warps", 0) or 0))
    cache = vars(compiled_kernel).setdefault(
        "_helion_resident_program_capacity_cache", {}
    )
    key = (device, num_warps, shared)
    capacity = cache.get(key)
    if capacity is None:
        cuda_driver = importlib.import_module("cuda.bindings.driver")
        with torch.cuda.device(device):
            error, blocks_per_sm = (
                cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    cuda_driver.CUfunction(int(function)),
                    num_warps * 32,
                    shared,
                )
            )
        if error != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                f"CUDA occupancy query failed for cross-loop kernel: {error}"
            )
        tmem_columns = getattr(metadata, "tmem_size", 0) or 0
        if tmem_columns > 0:
            # The occupancy API reports one block per SM for any kernel ptxas
            # tags as using tcgen05 (EIATTR_TCGEN05_1CTA_USED), however small;
            # the hardware co-schedules such CTAs while registers, shared
            # memory, threads and TMEM columns allow.
            blocks_per_sm = max(
                int(blocks_per_sm),
                _tcgen05_blocks_per_sm(
                    cuda_driver, function, device, num_warps, shared, tmem_columns
                ),
            )
        properties = torch.cuda.get_device_properties(device)
        capacity = int(blocks_per_sm) * int(properties.multi_processor_count)
        cache[key] = capacity
    if required_programs > capacity:
        raise RuntimeError(
            "Cross-loop scheduling requires "
            f"{required_programs} concurrently resident programs, but this "
            f"kernel/device can residently execute only {capacity}. Choose a "
            "lower-resource configuration, a smaller ready prefix, or a "
            "root barrier."
        )
