from __future__ import annotations

import contextvars
import os
from typing import TYPE_CHECKING

import torch

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel
from .triton_helpers import triton_send_signal as triton_send_signal
from .triton_helpers import triton_wait_multiple_signal as triton_wait_multiple_signal
from .triton_helpers import triton_wait_signal as triton_wait_signal
import os

if TYPE_CHECKING:
    import triton


def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    # Dynamically get device from Triton backend
    backend = triton.runtime.driver.active.get_current_target().backend  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
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
    assert device.type in ["cuda", "xpu", "cpu"], "TODO: implement for other devices"
    available_sms: int
    if device.type == "cpu":
        try:
            num_threads = int(torch.get_num_threads())
        except Exception:
            num_threads = 0
        available_sms = num_threads if num_threads > 0 else int(os.cpu_count() or 1)
    elif device.type == "cuda":
        available_sms = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count
    # TODO(EikanWang): gpu_subslice_count is an out-of-date term. we change update it to XeCore number.
    elif device.type == "xpu":
        available_sms = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    elif device.type == "mtia":
        try:
            from triton_mtia.backend.compiler import get_num_sm_for_arch
            return get_num_sm_for_arch(device.backend.arch)
        except ImportError:
            raise RuntimeError("MTIA backend selected, but not available.")
    else:
        raise NotImplementedError(f"get_num_sm not implemented for device type: {device.type}")

    if reserved_sms <= 0:
        return available_sms
    return max(available_sms - reserved_sms, 1)

def default_launcher(
    triton_kernel: triton.JITFunction,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    # Get current backend from Triton
    import triton
    backend = triton.runtime.driver.active.get_current_target().backend  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    if backend == "mtia":
        # MTIA-specific initialization
        try:
            from mtia.re.re_unittest_lib import init_mtia_device
            from triton_mtia.python.mtia.eager import mtia_triton_launcher

            init_mtia_device()
            # Ignore disk cache. Kernels will still keep an in-memory cache.
            os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
            mtia_triton_launcher.init()
        except ImportError as e:
            raise RuntimeError(f"MTIA backend selected but required modules not available: {e}")
    
    # For both CUDA and MTIA, use the same kernel execution
    return triton_kernel.run(
        *args,
        grid=grid,
        warmup=False,
        num_warps=num_warps,
        num_stages=num_stages,
        **kwargs,
    )
