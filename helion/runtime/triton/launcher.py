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
from typing import TYPE_CHECKING
from typing import Protocol
from typing import cast
import weakref

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


class _CompiledKernelWithPackedMetadata(Protocol):
    packed_metadata: tuple[object, ...]


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
    _persistent_state_specs: tuple[tuple[torch.Tensor, int, torch.dtype], ...] = (),
    _minimum_resident_programs: int = 0,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
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
        _configure_resident_program_capacity(
            compiled_kernel,
            args,
            num_warps=num_warps,
            required_programs=_minimum_resident_programs,
            grid_programs=math.prod(grid),
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
    cache = vars(triton_kernel).setdefault("_helion_persistent_state_cache", {})
    key = (like.device, dtype, stream.cuda_stream, namespace, slot)
    state = cache.get(key)
    if state is None or state.numel() < required_numel:
        state = torch.zeros(required_numel, dtype=dtype, device=like.device)
        cache[key] = state
    return state


def _select_residency_shared_memory(
    minimum: int,
    maximum: int,
    target_blocks: int,
    occupancy: Callable[[int], int],
) -> tuple[int, int]:
    """Choose the least shared memory that reaches the nearest safe occupancy."""
    if not 0 <= minimum <= maximum:
        raise ValueError("invalid dynamic shared-memory search range")
    if target_blocks <= 0:
        raise ValueError("target blocks per SM must be positive")

    cache: dict[int, int] = {}

    def cached_occupancy(shared: int) -> int:
        result = cache.get(shared)
        if result is None:
            result = occupancy(shared)
            cache[shared] = result
        return result

    base_blocks = cached_occupancy(minimum)
    if base_blocks <= target_blocks:
        return minimum, base_blocks

    def first_at_most(limit: int) -> int | None:
        if cached_occupancy(maximum) > limit:
            return None
        low = minimum
        high = maximum
        while low < high:
            candidate = (low + high) // 2
            if cached_occupancy(candidate) <= limit:
                high = candidate
            else:
                low = candidate + 1
        return low

    selected = first_at_most(target_blocks)
    if selected is None:
        nearest_higher = cached_occupancy(maximum)
        selected = first_at_most(nearest_higher)
        assert selected is not None
        return selected, nearest_higher

    selected_blocks = cached_occupancy(selected)
    if selected_blocks == target_blocks:
        return selected, selected_blocks

    # An occupancy cliff may skip the requested value. Keep the nearest higher
    # resident count because excess residency is safe while too little can
    # deadlock a polling schedule.
    nearest_higher = cached_occupancy(selected - 1)
    selected = first_at_most(nearest_higher)
    assert selected is not None
    return selected, nearest_higher


def _configure_resident_program_capacity(
    compiled_kernel: object,
    args: tuple[object, ...],
    *,
    num_warps: int,
    required_programs: int,
    grid_programs: int,
) -> None:
    """Realize and validate the requested resident CTA cohort.

    A CLC launch with a cancelable tail uses unused dynamic shared memory to
    cap physical residency at the smallest whole-device wave that contains the
    requested cohort. This keeps the policy in logical CTA units while deriving
    the hardware allocation from the compiled specialization and target GPU.
    """
    import importlib

    tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if tensor is None or tensor.device.type != "cuda":
        raise RuntimeError("cross-loop residency checks require a CUDA tensor")

    if compiled_kernel is None:
        raise RuntimeError("unable to compile cross-loop scheduled kernel")

    device = tensor.device
    configuration_key = (device, num_warps, required_programs, grid_programs)
    kernel_state = vars(compiled_kernel)
    configured = kernel_state.get("_helion_resident_program_configuration")
    if configured is not None:
        configured_key, _launch_shared, _capacity = configured
        if configured_key != configuration_key:
            raise RuntimeError(
                "one compiled Triton specialization received conflicting "
                "cross-loop residency requests"
            )
        return

    lock = kernel_state.setdefault(
        "_helion_resident_program_configuration_lock",
        threading.Lock(),
    )
    with lock:
        configured = kernel_state.get("_helion_resident_program_configuration")
        if configured is not None:
            configured_key, _launch_shared, _capacity = configured
            if configured_key != configuration_key:
                raise RuntimeError(
                    "one compiled Triton specialization received conflicting "
                    "cross-loop residency requests"
                )
            return

        # Accessing ``run`` initializes Triton's exact module/function handles
        # without launching it. Configure this cached specialization once,
        # before it can be captured into a CUDA graph.
        _run = compiled_kernel.run  # type: ignore[attr-defined]
        function = getattr(compiled_kernel, "function", None)
        metadata = getattr(compiled_kernel, "metadata", None)
        shared = getattr(metadata, "shared", None)
        packed_metadata = getattr(compiled_kernel, "packed_metadata", None)
        if (
            function is None
            or not isinstance(shared, int)
            or not isinstance(packed_metadata, tuple)
            or len(packed_metadata) < 3
        ):
            raise RuntimeError("unable to query cross-loop kernel occupancy")

        properties = torch.cuda.get_device_properties(device)
        sm_count = int(properties.multi_processor_count)
        if sm_count <= 0:
            raise RuntimeError("cross-loop residency checks require at least one SM")
        cuda_driver = importlib.import_module("cuda.bindings.driver")
        function_handle = cuda_driver.CUfunction(int(function))

        def blocks_per_sm(dynamic_shared: int) -> int:
            error, blocks = cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                function_handle,
                num_warps * 32,
                dynamic_shared,
            )
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"CUDA occupancy query failed for cross-loop kernel: {error}"
                )
            return int(blocks)

        with torch.cuda.device(device):
            launch_shared = shared
            resident_blocks = blocks_per_sm(launch_shared)
            if grid_programs > required_programs:
                target_blocks = math.ceil(required_programs / sm_count)
                if resident_blocks > target_blocks:
                    error, current_maximum = cuda_driver.cuFuncGetAttribute(
                        cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        function_handle,
                    )
                    if error != cuda_driver.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(
                            "CUDA dynamic shared-memory query failed for "
                            f"cross-loop kernel: {error}"
                        )
                    current_maximum = max(shared, int(current_maximum))
                    launch_shared, resident_blocks = _select_residency_shared_memory(
                        shared,
                        current_maximum,
                        target_blocks,
                        blocks_per_sm,
                    )
                    if resident_blocks > target_blocks:
                        error, static_shared = cuda_driver.cuFuncGetAttribute(
                            cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            function_handle,
                        )
                        if error != cuda_driver.CUresult.CUDA_SUCCESS:
                            raise RuntimeError(
                                "CUDA static shared-memory query failed for "
                                f"cross-loop kernel: {error}"
                            )
                        maximum_dynamic = int(
                            properties.shared_memory_per_block_optin
                        ) - int(static_shared)
                        if maximum_dynamic > current_maximum:
                            (error,) = cuda_driver.cuFuncSetCacheConfig(
                                function_handle,
                                cuda_driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_SHARED,
                            )
                            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                                raise RuntimeError(
                                    "CUDA shared-memory cache setup failed for "
                                    f"cross-loop kernel: {error}"
                                )
                            (error,) = cuda_driver.cuFuncSetAttribute(
                                function_handle,
                                cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                maximum_dynamic,
                            )
                            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                                raise RuntimeError(
                                    "CUDA dynamic shared-memory limit setup failed "
                                    f"for cross-loop kernel: {error}"
                                )
                            launch_shared, resident_blocks = (
                                _select_residency_shared_memory(
                                    shared,
                                    maximum_dynamic,
                                    target_blocks,
                                    blocks_per_sm,
                                )
                            )

            capacity = resident_blocks * sm_count
            if required_programs > capacity:
                raise RuntimeError(
                    "Cross-loop scheduling requires "
                    f"{required_programs} concurrently resident programs, but "
                    f"this kernel/device can residently execute only {capacity}. "
                    "Choose a smaller resident cohort or a lower-resource "
                    "kernel configuration."
                )
            if launch_shared != shared:
                packed = list(packed_metadata)
                packed[2] = launch_shared
                cast(
                    "_CompiledKernelWithPackedMetadata",
                    compiled_kernel,
                ).packed_metadata = tuple(packed)
            kernel_state["_helion_required_dynamic_shared_bytes"] = shared
            kernel_state["_helion_launch_dynamic_shared_bytes"] = launch_shared
            kernel_state["_helion_resident_blocks_per_sm"] = resident_blocks
            kernel_state["_helion_resident_program_configuration"] = (
                configuration_key,
                launch_shared,
                capacity,
            )
