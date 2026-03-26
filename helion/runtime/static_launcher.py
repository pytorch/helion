"""Static launcher support for Helion kernels.

On first call, compiles via Triton (normal path).
On subsequent calls, dispatches through PyTorch's static launcher C++ path,
bypassing Triton's Python runtime entirely.
Supports CUDA, ROCm, and XPU via PyTorch's StaticallyLaunchedTritonKernel.
"""

from __future__ import annotations

import functools
import os

import torch

from .._compile_time import measure
from .._utils import triton_is_available

if triton_is_available():
    import triton


@functools.cache
def _check_static_launcher_available() -> bool:
    if os.environ.get("HELION_STATIC_LAUNCHER", "1") == "0":
        return False
    try:
        from torch._inductor.runtime.static_triton_launcher import (  # noqa: F401
            statically_launched_kernel_by_device,
        )

        return True
    except ImportError:
        return False


# Device types that support static launching
_STATIC_LAUNCHER_DEVICES = {"cuda", "xpu"}


def _static_launch(
    triton_kernel: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    device_type: str,
    *,
    num_warps: int,
    num_stages: int,
    launch_cooperative_grid: bool = False,
    _cache: dict[tuple[int, tuple[object, ...]], object] = {},
    **kwargs: object,
) -> object:
    """Launch via static launcher, falling back to Triton on first call."""
    from torch._inductor.runtime.static_triton_launcher import (
        statically_launched_kernel_by_device,
    )

    fingerprint = tuple(
        a.dtype if isinstance(a, torch.Tensor) else type(a) for a in args
    )
    cache_key = (id(triton_kernel), fingerprint)
    static = _cache.get(cache_key)

    if static is None:
        # First call: compile via Triton.
        with measure("static_launch.first_call"):
            try:
                compiled_kernel = triton_kernel.run(  # type: ignore[union-attr]
                    *args,
                    grid=grid,
                    warmup=False,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    generate_native_code=True,
                    launch_cooperative_grid=launch_cooperative_grid,
                    **kwargs,
                )
            except Exception:
                return None
            try:
                # PyTorch's StaticallyLaunchedTritonKernel expects _cubin_path on
                # the compiled kernel (normally set by Inductor). Triton's
                # CompiledKernel stores binary paths in metadata_group instead,
                # so we bridge the gap here.
                _BIN_EXTS = {".cubin", ".hsaco", ".zebin"}
                for fname, fpath in compiled_kernel.metadata_group.items():  # type: ignore[union-attr]
                    if any(fname.endswith(ext) for ext in _BIN_EXTS):
                        compiled_kernel._cubin_path = fpath  # type: ignore[union-attr]
                        break
                static = statically_launched_kernel_by_device(compiled_kernel, device_type)
                # Build keep mask from full_constexprs (already computed by
                # StaticallyLaunchedTritonKernel) so we don't duplicate logic.
                n_args = len(static.arg_names)  # type: ignore[attr-defined]
                const_set = set(static.full_constexprs)  # type: ignore[attr-defined]
                static._keep = [i not in const_set for i in range(n_args)]  # type: ignore[attr-defined]
                device = triton.runtime.driver.active.get_current_device()  # type: ignore[name-defined]
                static.load_kernel(device)
                _cache[cache_key] = static
            except Exception:
                pass
        return compiled_kernel

    # Hot path: bypass Triton entirely
    with measure("static_launch.hot_path"):
        gx = grid[0] if len(grid) > 0 else 1
        gy = grid[1] if len(grid) > 1 else 1
        gz = grid[2] if len(grid) > 2 else 1
        device = triton.runtime.driver.active.get_current_device()  # type: ignore[name-defined]
        stream = triton.runtime.driver.active.get_current_stream(device)  # type: ignore[name-defined]
        filtered = tuple(
            a
            for a, keep in zip(args, static._keep)
            if keep  # type: ignore[attr-defined]
        )
        static.run(gx, gy, gz, stream, *filtered)
