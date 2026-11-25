"""Helpers for Metal backend tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion._compiler.metal.export import record_launches
import helion.language as hl

if TYPE_CHECKING:
    from types import ModuleType

    from helion._compiler.autotuner_heuristics.registry import AutotunerHeuristic
    from helion.runtime.kernel import BoundKernel


def _load(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    config: helion.Config | None,
    module_key: str = "",
) -> tuple[ModuleType, str, tuple[object, ...]]:
    """The module ``kernel`` generates for ``args`` at ``config``, loaded, its
    source, and the normalized arguments its entrypoint takes: raw generated
    entrypoints take every argument explicitly, defaults included, so they
    bind and run the same normalized arguments."""
    args = kernel.normalize_args(*args)
    code = kernel.bind(args).to_code(config)
    return PyCodeCache.load(code, module_key), code, args


def _compiled_msl(module: ModuleType) -> str:
    """The MSL the generated ``module``'s Metal kernel compiled."""
    # Imported here, on macOS: Linux CI imports the Metal test modules only to
    # skip them, and they import metal_jit behind a darwin check.
    from .._compiler.metal.metal_jit import _MetalKernel

    (msl,) = [
        obj.msl_source
        for obj in vars(module).values()
        if isinstance(obj, _MetalKernel) and obj.msl_source is not None
    ]
    return msl


def heuristic_seeds(
    heuristic: type[AutotunerHeuristic], bound: BoundKernel
) -> list[helion.Config]:
    """``heuristic``'s seed configs for a bound kernel, as the autotuner
    plants them at bind: its list, or its one seed where it gives none."""
    assert bound.host_function is not None
    with bound.env:
        device_ir = bound.host_function.device_ir
        seeds = heuristic.get_seed_configs(bound.env, device_ir)
        if seeds is None:
            seed = heuristic.get_seed_config(bound.env, device_ir)
            seeds = [] if seed is None else [seed]
    return seeds


def render_msl(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    config: helion.Config,
    num_sm: int = 16,
) -> str:
    """The MSL ``kernel`` generates for ``args`` at ``config``, without an
    Apple GPU: its generated module runs on ``args``, CPU tensors whose
    dtypes and shapes are all the MSL depends on, with a launcher that
    records its launch (``metal.export.record_launches``), so nothing
    compiles or launches.  A persistent kernel bakes the GPU's core count,
    which CPU tensors don't have, into its MSL: ``num_sm``.

    The module is generated anew, not taken from the bound kernel's compile
    cache, so a render under a patched heuristic sees the code the patch
    generates.  Each launch renders its MSL again, so one module serves
    every core count."""
    module, _, args = _load(kernel, args, config, "render_msl")
    entry = getattr(module, kernel.fn.__name__)
    (launch,) = record_launches(entry, args, num_sm=num_sm)
    return launch.source


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
    module, code, args = _load(kernel, args, config, module_key)
    result = getattr(module, kernel.fn.__name__)(*args)
    torch.mps.synchronize()
    return result, code, _compiled_msl(module)


@helion.kernel(backend="metal", static_shapes=True)
def addmm_k_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A K loop accumulating ``torch.addmm``: every Metal matmul lowering
    takes it."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


def mpp_only(
    kernel: helion.Kernel, args: tuple[object, ...], config: helion.Config | None
) -> helion.Config:
    """``config`` (the default for None) naming MPP's lowering, which a
    config without ``metal_matmul`` falls back from
    (``search_surface.MatmulFallback``): its refusals raise."""
    if config is None:
        config = kernel.bind(args).config_spec.default_config()
    return helion.Config.from_dict({"metal_matmul": "mpp", **config.config})


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def scale_by_0dim(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``x * s`` for a 0-dim ``s``, which binds as a buffer."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * s[()]
    return out
