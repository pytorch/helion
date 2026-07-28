"""Runtime launch helpers for the Triton backend.

This module holds the small set of runtime symbols that Helion's *generated*
Triton code depends on at execution time:

* :func:`default_launcher` -- invokes a compiled ``triton.jit`` kernel.
* :func:`get_num_sm` -- persistent-kernel grid size (host statement).
* :func:`set_triton_allocator` -- installs the scratch allocator used by TMA /
  tensor-descriptor kernels (device-function prefix statement).

Keeping these in a single dedicated module (rather than scattered through
``helion.runtime``) lets the ahead-of-time precompiler bulk-export exactly this
file into a standalone kernel. It is intended to depend only on ``torch`` and
``triton`` (that guarantee is enforced in a follow-up change; today it still
reaches back into a couple of ``helion`` helpers).
"""

from __future__ import annotations

import contextvars

import torch

from ... import exc
from ..._utils import triton_is_available
from ..settings import is_pallas_interpret as _module_is_pallas_interpret

if triton_is_available():
    import triton

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
    Get the number of streaming multiprocessors (SMs) for the specified device.

    Args:
        device: Device to query.
        reserved_sms: Number of SMs to keep free for other work (e.g., communication
            kernels). Defaults to 0 meaning all device SMs are available to Helion.

    Returns:
        Grid size to use for a persistent kernel on the device after accounting
        for any reserved SMs. Always at least 1.
    """
    available_sms: int
    if device.type == "cpu":
        if not _module_is_pallas_interpret():
            raise AssertionError("TODO: implement for other devices")
        return 1
    if device.type == "tpu":
        return 1

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


def default_launcher(
    triton_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    # For both CUDA and MTIA, use the same kernel execution
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
    try:
        return triton_kernel.run(  # type: ignore[union-attr]
            *args,
            **run_kwargs,
        )
    except Exception as error:
        message = str(error)
        if "Cannot make_shape_compatible: incompatible dimensions" in message:
            raise exc.ShapeMismatch("kernel operands", message) from error
        raise
