"""Helpers for Metal backend tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.codecache import PyCodeCache

if TYPE_CHECKING:
    import helion


def run_with_msl(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    config: helion.Config | None = None,
    module_key: str = "",
) -> tuple[object, str, str]:
    """Compile and run ``kernel`` on ``args`` at ``config`` (None: as
    ``BoundKernel.to_code`` picks one, the kernel's own config if it has
    one) through the normal Helion pipeline;
    return its result, its generated Python and the MSL it compiled.  A new
    ``module_key`` loads a fresh module, whose MSL is generated anew."""
    # Imported here, on macOS: Linux CI imports the Metal test modules only to
    # skip them, and they import metal_jit behind a darwin check.
    from .._compiler.metal.metal_jit import _MetalKernel

    # Raw generated entrypoints take every argument explicitly, defaults
    # included: bind and run the same normalized arguments.
    args = kernel.normalize_args(*args)
    bound = kernel.bind(args)
    code = bound.to_code(config)
    module = PyCodeCache.load(code, module_key)
    result = getattr(module, kernel.fn.__name__)(*args)
    torch.mps.synchronize()
    (msl,) = [
        obj.msl_source
        for obj in vars(module).values()
        if isinstance(obj, _MetalKernel) and obj.msl_source is not None
    ]
    return result, code, msl
