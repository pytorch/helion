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
import math
import threading
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


def _runtime_lock(owner: object) -> threading.RLock:
    """Return the per-kernel lock guarding compiler-owned runtime state."""
    values = vars(owner)
    lock = values.get("_helion_runtime_lock")
    if lock is None:
        lock = values.setdefault("_helion_runtime_lock", threading.RLock())
    return lock


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
    _persistent_state_specs: tuple[tuple[torch.Tensor, int, torch.dtype], ...] = (),
    _minimum_resident_programs: int = 0,
    _target_resident_programs_per_sm: int = 0,
    _requires_clc: bool = False,
    _cross_loop_dispatch_kind: str | None = None,
    _cross_loop_fallback_reason: str | None = None,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately.

    Persistent compiler state is keyed by kernel specialization and capture
    stream while a launch is assembled. Captured CUDA Graphs retain those state
    pointers, so every launch or replay sharing that state, including distinct
    graph instances captured on the same stream, must be serialized.
    """
    if _cross_loop_dispatch_kind is not None:
        vars(triton_kernel)["_helion_cross_loop_dispatch"] = (
            _cross_loop_dispatch_kind,
            _cross_loop_fallback_reason,
        )
    if _requires_clc:
        tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        if tensor is None or tensor.device.type != "cuda":
            raise RuntimeError("CLC dispatch requires a CUDA tensor argument")
        major, _minor = torch.cuda.get_device_capability(tensor.device)
        if major < 10:
            raise RuntimeError("CLC dispatch requires CUDA compute capability sm_100+")
        if len(grid) != 1:
            raise RuntimeError("CLC dispatch currently requires a one-dimensional grid")
        if kwargs.get("launch_pdl") is not True:
            raise RuntimeError("CLC dispatch requires launch_pdl=True")
        if kwargs.get("num_ctas", 1) != 1:
            raise RuntimeError("CLC dispatch currently requires one-CTA clusters")
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
            _requires_clc,
            _target_resident_programs_per_sm,
            tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
            tuple((numel, dtype) for _, numel, dtype in _persistent_state_specs),
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
    if _target_resident_programs_per_sm:
        compiled_kernel = triton_kernel.run(  # type: ignore[union-attr]
            *args,
            **{**run_kwargs, "warmup": True},
        )
        # The launch metadata is mutable on Triton's compiled specialization.
        # Keep target selection and the corresponding launch atomic with
        # respect to another host thread using the same specialization.
        with _runtime_lock(compiled_kernel):
            _limit_resident_programs_per_sm(
                compiled_kernel,
                args,
                num_warps=num_warps,
                target_programs=_target_resident_programs_per_sm,
            )
            return triton_kernel.run(  # type: ignore[union-attr]
                *args,
                **run_kwargs,
            )
    return triton_kernel.run(  # type: ignore[union-attr]
        *args,
        **run_kwargs,
    )


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
    with _runtime_lock(triton_kernel):
        cache = vars(triton_kernel).setdefault("_helion_persistent_state_cache", {})
        key = (like.device, dtype, stream.cuda_stream, namespace, slot)
        state = cache.get(key)
        if state is None or state.numel() < required_numel:
            _reject_allocation_during_cuda_graph_capture(
                like.device,
                "persistent Triton state",
            )
            state = torch.zeros(required_numel, dtype=dtype, device=like.device)
            cache[key] = state
        return state


def _reject_allocation_during_cuda_graph_capture(
    device: torch.device,
    allocation: str,
) -> None:
    """Require compiler-owned buffers to be initialized before graph capture."""
    with torch.cuda.device(device):
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        raise RuntimeError(
            f"{allocation} must be initialized before CUDA Graph capture; "
            "warm up the compiled Helion kernel on the capture stream first"
        )


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
        registers = getattr(compiled_kernel, "n_regs", "unknown")
        spills = getattr(compiled_kernel, "n_spills", "unknown")
        raise RuntimeError(
            "Cross-loop scheduling requires "
            f"{required_programs} concurrently resident programs, but this "
            f"kernel/device can residently execute only {capacity} "
            f"(registers={registers}, spills={spills}, "
            f"dynamic_shared_bytes={shared}). Choose a "
            "lower-resource configuration, an earlier dependency frontier, "
            "or root completion."
        )


def _limit_resident_programs_per_sm(
    compiled_kernel: object,
    args: tuple[object, ...],
    *,
    num_warps: int,
    target_programs: int,
) -> None:
    """Cap CLC residency without exposing backend scratch bytes to scheduling."""
    if target_programs <= 0:
        raise ValueError("the resident-program target must be positive")
    tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if tensor is None or tensor.device.type != "cuda":
        raise RuntimeError("CLC residency limiting requires a CUDA tensor")
    if compiled_kernel is None:
        raise RuntimeError("unable to compile CLC kernel")

    with _runtime_lock(compiled_kernel):
        # Initialize the CUDA function using Triton's original dynamic-shared
        # size. The generated launcher reads the actual launch size from
        # packed_metadata, which is restored for each cached target below.
        _run = compiled_kernel.run  # type: ignore[attr-defined]
        function = getattr(compiled_kernel, "function", None)
        metadata = getattr(compiled_kernel, "metadata", None)
        shared = getattr(metadata, "shared", None)
        if function is None or not isinstance(shared, int):
            raise RuntimeError("unable to query CLC kernel occupancy")

        values = vars(compiled_kernel)
        base_shared = values.setdefault("_helion_clc_base_shared", shared)
        if not isinstance(base_shared, int):
            raise RuntimeError("invalid cached CLC shared-memory baseline")
        cache = values.setdefault("_helion_clc_residency_limits", {})
        cache_key = (tensor.device, num_warps, target_programs)
        padded_shared = cache.get(cache_key)
        cache_miss = padded_shared is None
        if padded_shared is None:
            _reject_allocation_during_cuda_graph_capture(
                tensor.device,
                "CLC residency metadata",
            )
            padded_shared = _compute_clc_resident_shared_bytes(
                function,
                tensor.device,
                num_warps=num_warps,
                base_shared=base_shared,
                target_programs=target_programs,
            )
        current_metadata = compiled_kernel.metadata  # type: ignore[attr-defined]
        if current_metadata.shared == padded_shared:
            if cache_miss:
                cache[cache_key] = padded_shared
            return
        updated_metadata = current_metadata._replace(shared=padded_shared)
        from triton.compiler.compiler import make_backend

        packed_metadata = make_backend(updated_metadata.target).pack_metadata(
            updated_metadata
        )
        compiled_kernel.metadata = updated_metadata  # type: ignore[attr-defined]
        compiled_kernel.packed_metadata = packed_metadata  # type: ignore[attr-defined]
        if cache_miss:
            cache[cache_key] = padded_shared


def _compute_clc_resident_shared_bytes(
    function: object,
    device: torch.device,
    *,
    num_warps: int,
    base_shared: int,
    target_programs: int,
) -> int:
    """Return dynamic shared bytes that cap occupancy at ``target_programs``."""
    import importlib

    with torch.cuda.device(device):
        cuda_driver = importlib.import_module("cuda.bindings.driver")
        cuda_function = cuda_driver.CUfunction(int(function))

        def blocks_per_sm(dynamic_shared: int) -> int:
            error, blocks = cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                cuda_function,
                num_warps * 32,
                dynamic_shared,
            )
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"CUDA occupancy query failed for CLC kernel: {error}"
                )
            return int(blocks)

        if blocks_per_sm(base_shared) <= target_programs:
            return base_shared

        error, static_shared = cuda_driver.cuFuncGetAttribute(
            cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            cuda_function,
        )
        if error != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                f"CUDA shared-memory query failed for CLC kernel: {error}"
            )
        properties = torch.cuda.get_device_properties(device)
        max_dynamic_shared = int(properties.shared_memory_per_block_optin) - int(
            static_shared
        )
        error = cuda_driver.cuFuncSetAttribute(
            cuda_function,
            cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            max_dynamic_shared,
        )[0]
        if error != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                f"CUDA shared-memory opt-in failed for CLC kernel: {error}"
            )
        alignment = 256
        low = max(base_shared, alignment)
        high = max_dynamic_shared
        low_units = (low + alignment - 1) // alignment
        high_units = high // alignment
        if (
            high_units < low_units
            or blocks_per_sm(high_units * alignment) > target_programs
        ):
            raise RuntimeError(
                f"unable to limit CLC occupancy to {target_programs} programs per SM"
            )
        while low_units < high_units:
            midpoint = (low_units + high_units) // 2
            if blocks_per_sm(midpoint * alignment) <= target_programs:
                high_units = midpoint
            else:
                low_units = midpoint + 1
        return low_units * alignment
