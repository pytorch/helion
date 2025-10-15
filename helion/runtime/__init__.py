from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

import torch

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel
from .triton_helpers import triton_send_signal as triton_send_signal
from .triton_helpers import triton_wait_multiple_signal as triton_wait_multiple_signal
from .triton_helpers import triton_wait_signal as triton_wait_signal

if TYPE_CHECKING:
    import triton


def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


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


def get_num_sm(device: torch.device) -> int:
    """
    Get the number of streaming multiprocessors (SMs) for the specified device.

    Args:
        device: Device to query.

    Returns:
        Grid size to use for a persistent kernel on the device.
    """
    assert device.type in ["cuda", "xpu"], "TODO: implement for other devices"
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device.index).multi_processor_count
    # TODO(EikanWang): gpu_subslice_count is an out-of-date term. we change update it to XeCore number.
    return torch.xpu.get_device_properties(device.index).gpu_subslice_count


def default_launcher(
    triton_kernel: triton.JITFunction,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    return triton_kernel.run(
        *args,
        grid=grid,
        warmup=False,
        num_warps=num_warps,
        num_stages=num_stages,
        **kwargs,
    )
