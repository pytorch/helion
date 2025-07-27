from __future__ import annotations

import functools

import torch
import triton

from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel
from .triton_helpers import triton_send_signal as triton_send_signal
from .triton_helpers import triton_wait_multiple_signal as triton_wait_multiple_signal
from .triton_helpers import triton_wait_signal as triton_wait_signal


def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    # Ensure minimum alignment of 16 bytes for GPU operations
    # B200 appears to have stricter alignment requirements than H100
    alignment = max(alignment, 16)
    
    # Allocate extra space to ensure we can align the pointer
    padded_size = size + alignment - 1
    raw_tensor = torch.empty(padded_size, device="cuda", dtype=torch.int8)
    
    # Get the raw pointer and align it
    ptr = raw_tensor.data_ptr()
    aligned_ptr = (ptr + alignment - 1) & ~(alignment - 1)
    offset = aligned_ptr - ptr
    
    # Return a view starting at the aligned offset
    return raw_tensor[offset:offset + size]


@functools.cache
def set_triton_allocator() -> None:
    try:
        from triton.runtime._allocation import NullAllocator
        from triton.runtime._allocation import _allocator

        if not isinstance(_allocator, NullAllocator):
            return
    except ImportError:
        pass
    triton.set_allocator(_alloc_fn)


def get_num_sm(device: torch.device) -> int:
    """
    Get the number of streaming multiprocessors (SMs) for the specified device.

    Args:
        device: Device to query.

    Returns:
        Grid size to use for a persistent kernel on the device.
    """
    assert device.type == "cuda", "TODO: implement for other devices"
    return torch.cuda.get_device_properties(device.index).multi_processor_count


def default_launcher(
    triton_kernel: triton.JITFunction,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    return triton_kernel.run(
        *args, grid=grid, warmup=False, num_warps=num_warps, num_stages=num_stages
    )
