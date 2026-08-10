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
    _remote_copy_scratch_specs: tuple[tuple[torch.Tensor, int], ...] = (),
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
