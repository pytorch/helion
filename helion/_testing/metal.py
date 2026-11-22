"""Helpers for Metal backend tests."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from unittest import mock

import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion._compiler.metal import c10_preamble
import helion.language as hl

if TYPE_CHECKING:
    from types import ModuleType

    from helion._compiler.autotuner_heuristics.registry import AutotunerHeuristic
    from helion.runtime.kernel import BoundKernel


class _NoShader:
    """A compiled library stand-in: every kernel in it launches nothing."""

    def __getattr__(self, name: str) -> object:
        return lambda *args, **kwargs: None


def _whole_c10(whole: bool) -> contextlib.AbstractContextManager[object]:
    """With ``whole``, Metal kernels compiled inside take the whole c10
    headers, ``c10_preamble`` off."""
    if not whole:
        return contextlib.nullcontext()
    return mock.patch.object(c10_preamble, "prune", lambda msl: msl)


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
    module_key: str = "",
    whole_c10: bool = False,
) -> str:
    """The MSL ``kernel`` generates for ``args`` at ``config``, without an
    Apple GPU: its generated module runs on ``args``, CPU tensors whose
    dtypes and shapes are all the MSL depends on, with
    ``torch.mps.compile_shader`` stubbed out, so nothing compiles or
    launches.  A persistent kernel bakes the GPU's core count, which CPU
    tensors don't have, into its MSL: ``num_sm``.

    The module is loaded under its own key, apart from the one a GPU run of
    the same code loads (:func:`run_with_msl`): a module's kernel keeps its
    launch plans by everything but the tensors' device, so a shared module
    would hand this render's no-op plan to a GPU run, or a GPU run's shader
    to this render.  The key also holds ``num_sm``, which the generated
    code doesn't, so each core count renders its own MSL; a new
    ``module_key`` renders it anew.  ``whole_c10`` renders it with the whole
    c10 headers, in place of the declarations it uses (``c10_preamble``)."""
    key = f"render_msl:{num_sm}:{module_key}:{whole_c10}"
    module, _, args = _load(kernel, args, config, key)
    with (
        _whole_c10(whole_c10),
        mock.patch.object(
            torch.mps, "compile_shader", lambda source: _NoShader(), create=True
        ),
        mock.patch.object(
            helion.runtime,
            "get_num_sm",
            lambda device, *, reserved_sms=0: num_sm - reserved_sms,
        ),
    ):
        getattr(module, kernel.fn.__name__)(*args)
    return _compiled_msl(module)


def run_with_msl(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    config: helion.Config | None = None,
    module_key: str = "",
    whole_c10: bool = False,
) -> tuple[object, str, str]:
    """Compile and run ``kernel`` on ``args`` at ``config`` (None: as
    ``BoundKernel.to_code`` picks one, the kernel's own config if it has
    one) through the normal Helion pipeline;
    return its result, its generated Python and the MSL it compiled.  A new
    ``module_key`` loads a fresh module, whose MSL is generated anew;
    ``whole_c10`` compiles it with the whole c10 headers (``c10_preamble``
    off)."""
    module, code, args = _load(kernel, args, config, f"{module_key}:{whole_c10}")
    with _whole_c10(whole_c10):
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


@helion.kernel(
    backend="metal", static_shapes=True, configs=[helion.Config(block_sizes=[256])]
)
def literal_math(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``x * y`` with Python-number literals, which lowering rounds to a
    16-bit x's dtype on the host: added on either side, subtracted, compared
    and divided into by remainder."""
    out = torch.empty([5, x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        u = x[tile] * y[tile]
        out[0, tile] = u + 1.1
        out[1, tile] = 1.1 + u
        out[2, tile] = u - 1.1
        out[3, tile] = torch.where(u > 0.3, u, -u)
        out[4, tile] = torch.remainder(u, 0.7)
    return out
