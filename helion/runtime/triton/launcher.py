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

import contextvars
import hashlib
from itertools import starmap
import math
from typing import cast
import weakref

import torch

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
    _distributed_readiness_device_anchor: torch.Tensor | None = None,
    _distributed_readiness_signal_slots: int = 0,
    _distributed_readiness_world_size: int = 0,
    _distributed_readiness_process_group_name: str | None = None,
    _remote_copy_scratch_specs: tuple[tuple[torch.Tensor, int], ...] = (),
    _persistent_state_specs: tuple[tuple[torch.Tensor, int, torch.dtype], ...] = (),
    _minimum_resident_programs: int = 0,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    original_args = args
    remote_copy_signal_slots = 0
    if _remote_copy_signal_slots_per_program:
        if _remote_copy_signal_dst is None or _remote_copy_process_group_name is None:
            raise RuntimeError(
                "remote-copy completion storage requires a symmetric destination "
                "and process group"
            )
        remote_copy_signal_slots = (
            math.prod(grid) * _remote_copy_signal_slots_per_program
        )
        signal = _get_remote_copy_signal(
            triton_kernel,
            _remote_copy_signal_dst,
            _remote_copy_process_group_name,
            remote_copy_signal_slots,
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
    distributed_launch_fingerprint: str | None = None
    if _distributed_readiness_signal_slots:
        if (
            _distributed_readiness_device_anchor is None
            or _distributed_readiness_process_group_name is None
            or _distributed_readiness_world_size <= 0
        ):
            raise RuntimeError(
                "distributed readiness requires a CUDA tensor and process group"
            )
        distributed_launch_fingerprint = _distributed_launch_fingerprint(
            triton_kernel,
            grid,
            original_args,
            process_group_name=_distributed_readiness_process_group_name,
            num_warps=num_warps,
            num_stages=num_stages,
            ptx_options=ptx_options,
            launch_cooperative_grid=launch_cooperative_grid,
            launch_options=kwargs,
            state_schema=tuple(
                (numel, str(dtype)) for _, numel, dtype in _persistent_state_specs
            ),
            remote_copy_slots=remote_copy_signal_slots,
            remote_barrier_slots=(
                math.prod(grid) * _remote_barrier_signal_slots_per_program
            ),
            readiness_slots=_distributed_readiness_signal_slots,
        )
        signal, signal_ptrs, signal_offset = _get_distributed_readiness_signal(
            triton_kernel,
            _distributed_readiness_device_anchor,
            _distributed_readiness_process_group_name,
            _distributed_readiness_signal_slots,
            expected_world_size=_distributed_readiness_world_size,
            launch_fingerprint=distributed_launch_fingerprint,
        )
        args = (*args, signal, signal_ptrs, signal_offset)
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
            ("distributed_readiness", distributed_launch_fingerprint)
            if distributed_launch_fingerprint is not None
            else (
                tuple(grid),
                num_warps,
                num_stages,
                ptx_options,
                launch_cooperative_grid,
                tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
                tuple((numel, dtype) for _, numel, dtype in _persistent_state_specs),
            )
        )
        for slot, (state_like, numel, dtype) in enumerate(_persistent_state_specs):
            state = _get_persistent_state(
                triton_kernel,
                state_like,
                persistent_state_namespace,
                slot,
                numel,
                dtype,
            )
            args = (*args, state)
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


def _distributed_launch_fingerprint(
    triton_kernel: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    *,
    process_group_name: str,
    num_warps: int,
    num_stages: int,
    ptx_options: str | None,
    launch_cooperative_grid: bool,
    launch_options: dict,
    state_schema: tuple[tuple[int, str], ...] = (),
    remote_copy_slots: int = 0,
    remote_barrier_slots: int = 0,
    readiness_slots: int = 0,
) -> str:
    """Fingerprint the compiled schedule while ignoring dynamic tensor extents."""

    params = getattr(triton_kernel, "params", ())

    def argument_signature(index: int, arg: object) -> object:
        param = params[index] if index < len(params) else None
        if getattr(param, "is_constexpr", False):
            return ("constexpr", type(arg).__qualname__, repr(arg))
        if isinstance(arg, torch.Tensor):
            return (
                "pointer",
                str(arg.dtype),
                arg.device.type,
            )
        if isinstance(arg, int) and not isinstance(arg, bool):
            # Triton's runtime specialization distinguishes one-valued and
            # 16-byte-divisible integers; preserve those classes without
            # freezing ordinary dynamic shape values into this fingerprint.
            return ("runtime_int", arg == 1, arg % 16 == 0)
        return ("runtime", type(arg).__module__, type(arg).__qualname__)

    payload = (
        getattr(
            triton_kernel,
            "cache_key",
            getattr(triton_kernel, "src", None),
        ),
        process_group_name,
        tuple(grid),
        tuple(starmap(argument_signature, enumerate(args))),
        num_warps,
        num_stages,
        ptx_options,
        launch_cooperative_grid,
        tuple(sorted((name, repr(value)) for name, value in launch_options.items())),
        state_schema,
        remote_copy_slots,
        remote_barrier_slots,
        readiness_slots,
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


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


def _get_distributed_readiness_signal(
    triton_kernel: object,
    dst: torch.Tensor,
    process_group_name: str,
    required_slots: int,
    *,
    expected_world_size: int | None = None,
    launch_fingerprint: str | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Return dedicated symmetric readiness state for one protocol stream.

    First use is an SPMD collective: every rank must initialize protocol/stream
    instances in the same order.  Cache hits have no host synchronization and
    are safe for CUDA graph replay.
    """
    if dst.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError(
            "compiler-derived distributed readiness requires NVIDIA CUDA"
        )
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed.distributed_c10d as c10d

    group = c10d._resolve_process_group(
        process_group_name  # pyrefly: ignore[bad-argument-type]
    )
    actual_world_size = dist.get_world_size(group)
    if expected_world_size is not None and actual_world_size != expected_world_size:
        raise RuntimeError(
            "distributed readiness was compiled for "
            f"world size {expected_world_size}, but process group "
            f"{process_group_name!r} has world size {actual_world_size}"
        )
    cache = vars(triton_kernel).setdefault(
        "_helion_distributed_readiness_signal_cache", {}
    )
    stream = torch.cuda.current_stream(dst.device)
    key = (
        dst.device,
        process_group_name,
        id(group),
        launch_fingerprint,
        required_slots,
        stream.cuda_stream,
    )
    entry = cache.get(key)
    if entry is not None:
        _group, _workspace, handle, signal_pad, offset = entry
        return signal_pad, handle.signal_pad_ptrs_dev, offset

    with torch.cuda.device(dst.device):
        allocation_device = dst.device
        # PyTorch currently keys its symmetric pool by the spelling originally
        # passed to symm_mem.empty (for example ``cuda`` versus ``cuda:0``), while
        # NVSHMEM's process-global team manager requires that spelling to remain
        # stable. Reuse the established spelling for this physical device.
        for pool_device in getattr(symm_mem, "_symm_mem_pools", {}):
            candidate = torch.device(pool_device)
            candidate_index = (
                torch.cuda.current_device()
                if candidate.index is None
                else candidate.index
            )
            if (
                candidate.type == dst.device.type
                and candidate_index == dst.device.index
            ):
                allocation_device = candidate
                break
        workspace = symm_mem.empty(1, dtype=torch.uint8, device=allocation_device)
        handle = symm_mem.rendezvous(
            workspace,
            group=process_group_name,  # pyrefly: ignore[bad-argument-type]
        )
    with torch.cuda.device(dst.device):
        signal_pad = handle.get_signal_pad(handle.rank, dtype=torch.uint64)
        capacity = signal_pad.numel()
        capacity_ok = required_slots <= capacity
        offset = capacity - required_slots if capacity_ok else 0
        if capacity_ok:
            signal_pad.narrow(0, offset, required_slots).zero_()
        # Complete initialization before any peer may publish into this workspace.
        stream.synchronize()
        statuses: list[tuple[str | None, int] | None] = [None] * actual_world_size
        dist.all_gather_object(
            statuses,
            (launch_fingerprint, capacity),
            group=group,
        )
    if any(status is None for status in statuses):
        raise RuntimeError("distributed readiness initialization did not complete")
    concrete_statuses = cast("list[tuple[str | None, int]]", statuses)
    fingerprints = [fingerprint for fingerprint, _capacity in concrete_statuses]
    if fingerprints != fingerprints[:1] * len(fingerprints):
        raise RuntimeError(
            "distributed readiness requires identical kernel schedules and "
            f"launch geometry on every rank; got {fingerprints!r}"
        )
    capacities = [peer_capacity for _fingerprint, peer_capacity in concrete_statuses]
    if capacities != capacities[:1] * len(capacities):
        raise RuntimeError(
            "distributed readiness requires identical signal pad capacities "
            f"on every rank; got {capacities!r}"
        )
    if any(required_slots > peer_capacity for peer_capacity in capacities):
        raise RuntimeError(
            "Helion distributed readiness requires "
            f"{required_slots} uint64 signal slots per stream, but the symmetric-memory "
            f"signal pad capacities are {capacities!r}. Increase the signal pad size "
            "before launching the kernel."
        )
    cache[key] = (group, workspace, handle, signal_pad, offset)
    return signal_pad, handle.signal_pad_ptrs_dev, offset


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
        state = torch.zeros(required_numel, dtype=dtype, device=like.device)
        cache[key] = state
    return state


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
