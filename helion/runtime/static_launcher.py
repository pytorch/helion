"""Static launcher support for Helion kernels.

On first call, compiles via Triton (warmup only) and wraps the compiled binary
with PyTorch's StaticallyLaunchedTritonKernel. All launches (including the
first) go through the static C++ path, bypassing Triton's Python runtime.
Supports CUDA, ROCm, and XPU via PyTorch's StaticallyLaunchedTritonKernel.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import torch

from .._utils import triton_is_available

if triton_is_available():
    import triton


@functools.cache
def _check_static_launcher_available() -> bool:
    if os.environ.get("HELION_STATIC_LAUNCHER", "1") == "0":
        return False
    try:
        # Could use the `torch._inductor.config` directly in the test, 
        # which is controlled by the PyTorch's behavior.
        return torch._inductor.config.use_static_triton_launcher
    except AttributeError:
        return False


# Device types that support static launching
_STATIC_LAUNCHER_DEVICES = {"cuda", "xpu"}

_BIN_EXTS = {".cubin", ".hsaco", ".zebin"}

_STATIC_LAUNCHER_ATTR = "_helion_static_launcher"


def _static_launch(
    triton_kernel: Any,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    device_type: str,
    *,
    num_warps: int,
    num_stages: int,
    launch_cooperative_grid: bool = False,
    **kwargs: object,
) -> None:
    """Launch via PyTorch's static C++ launcher, compiling on first call."""
    from torch._inductor.runtime.static_triton_launcher import (
        statically_launched_kernel_by_device,
    )

    entry = getattr(triton_kernel, _STATIC_LAUNCHER_ATTR, None)

    if entry is None:
        # First call: compile only (warmup=True), then wrap with static launcher.
        compiled_kernel = triton_kernel.run(
            *args,
            grid=grid,
            warmup=True,
            num_warps=num_warps,
            num_stages=num_stages,
            generate_native_code=True,
            launch_cooperative_grid=launch_cooperative_grid,
            **kwargs,
        )
        # StaticallyLaunchedTritonKernel.__init__ reads _cubin_path.
        # Triton stores binary paths in metadata_group, so bridge the gap.
        for fname, fpath in compiled_kernel.metadata_group.items():
            if any(fname.endswith(ext) for ext in _BIN_EXTS):
                compiled_kernel._cubin_path = fpath
                break
        static = statically_launched_kernel_by_device(compiled_kernel, device_type)
        # static.run() expects only non-constexpr args, so build a mask.
        n_args = len(static.arg_names)
        const_set = set(static.full_constexprs)
        keep_mask = [i not in const_set for i in range(n_args)]
        device = triton.runtime.driver.active.get_current_device()
        static.load_kernel(device)
        setattr(triton_kernel, _STATIC_LAUNCHER_ATTR, (static, keep_mask))
    else:
        static, keep_mask = entry

    # All launches go through the static C++ path.
    gx = grid[0] if len(grid) > 0 else 1
    gy = grid[1] if len(grid) > 1 else 1
    gz = grid[2] if len(grid) > 2 else 1
    device = triton.runtime.driver.active.get_current_device()
    stream = triton.runtime.driver.active.get_current_stream(device)
    filtered = tuple(a for a, keep in zip(args, keep_mask, strict=True) if keep)
    static.run(gx, gy, gz, stream, *filtered)
