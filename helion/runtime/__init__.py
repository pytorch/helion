from __future__ import annotations

import logging

import torch

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .. import exc
from .config import Config as Config
from .cute.launcher import _CUTE_LAUNCH_ARG_CACHE_LIMIT as _CUTE_LAUNCH_ARG_CACHE_LIMIT
from .cute.launcher import (
    _TCGEN05_DYNAMIC_TENSORMAP_WORKSPACE_CACHE_LIMIT as _TCGEN05_DYNAMIC_TENSORMAP_WORKSPACE_CACHE_LIMIT,
)
from .cute.launcher import _append_cute_wrapper_plan as _append_cute_wrapper_plan
from .cute.launcher import (
    _build_cached_cute_schema_and_args as _build_cached_cute_schema_and_args,
)
from .cute.launcher import _build_cute_schema_and_args as _build_cute_schema_and_args
from .cute.launcher import (
    _build_tcgen05_grouped_static_metadata as _build_tcgen05_grouped_static_metadata,
)
from .cute.launcher import _create_cute_wrapper as _create_cute_wrapper
from .cute.launcher import _cuda_stream_capture_context as _cuda_stream_capture_context
from .cute.launcher import _cute_cluster_shape as _cute_cluster_shape
from .cute.launcher import (
    _cute_cluster_shape_from_wrapper_plans as _cute_cluster_shape_from_wrapper_plans,
)
from .cute.launcher import (
    _cute_dynamic_tensormap_contexts as _cute_dynamic_tensormap_contexts,
)
from .cute.launcher import (
    _cute_grouped_launch_contexts as _cute_grouped_launch_contexts,
)
from .cute.launcher import _cute_last_launch_arg_guard as _cute_last_launch_arg_guard
from .cute.launcher import (
    _cute_last_launch_cache_entry as _cute_last_launch_cache_entry,
)
from .cute.launcher import _cute_launch_arg_cache_key as _cute_launch_arg_cache_key
from .cute.launcher import _CuteLaunchArgCacheEntry as _CuteLaunchArgCacheEntry
from .cute.launcher import _ensure_cute_dsl_arch_env as _ensure_cute_dsl_arch_env
from .cute.launcher import _get_compiled_cute_launcher as _get_compiled_cute_launcher
from .cute.launcher import (
    _record_cute_owned_launch_tensors as _record_cute_owned_launch_tensors,
)
from .cute.launcher import (
    _tcgen05_grouped_dynamic_tensormap_workspace as _tcgen05_grouped_dynamic_tensormap_workspace,
)
from .cute.launcher import (
    _tcgen05_grouped_tensor_mutation_key as _tcgen05_grouped_tensor_mutation_key,
)
from .cute.launcher import cute_cuda_graph as cute_cuda_graph
from .cute.launcher import default_cute_launcher as default_cute_launcher
from .kernel import Kernel as Kernel
from .kernel import OutputCodeOptions as OutputCodeOptions
from .kernel import kernel as kernel
from .pallas.launcher import default_pallas_launcher as default_pallas_launcher
from .settings import is_pallas_interpret as _module_is_pallas_interpret
from .triton.launcher import default_launcher as _triton_default_launcher
from .triton.launcher import get_num_sm as _triton_get_num_sm
from .triton.launcher import get_num_xcd as get_num_xcd
from .triton.launcher import set_triton_allocator as set_triton_allocator

log: logging.Logger = logging.getLogger(__name__)


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
    """Thin in-process wrapper over the dependency-free
    :func:`helion.runtime.triton.launcher.default_launcher` that translates
    Triton's opaque "incompatible dimensions" error into
    :class:`helion.exc.ShapeMismatch`.
    """
    try:
        return _triton_default_launcher(
            triton_kernel,
            grid,
            *args,
            num_warps=num_warps,
            num_stages=num_stages,
            ptx_options=ptx_options,
            launch_cooperative_grid=launch_cooperative_grid,
            **kwargs,
        )
    except Exception as error:
        message = str(error)
        if "Cannot make_shape_compatible: incompatible dimensions" in message:
            raise exc.ShapeMismatch("kernel operands", message) from error
        raise


def get_num_sm(device: torch.device, *, reserved_sms: int = 0) -> int:
    """Number of SMs (persistent-kernel grid size) for any Helion device.

    Adds the CPU (Pallas-interpret) and TPU cases on top of the dependency-free
    GPU helper :func:`helion.runtime.triton.launcher.get_num_sm`. See that
    function for argument/return semantics.
    """
    if device.type == "cpu":
        if not _module_is_pallas_interpret():
            raise AssertionError("TODO: implement for other devices")
        return 1
    if device.type == "tpu":
        return 1
    return _triton_get_num_sm(device, reserved_sms=reserved_sms)


def default_metal_launcher(
    metal_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _block_dims: tuple[int, int, int] = (256, 1, 1),
    **kwargs: object,
) -> None:
    """Default launcher for Metal kernels on Apple MPS devices.

    The ``metal_kernel`` is a ``@metal_jit`` decorated function that
    translates its Python AST body to MSL and compiles it via
    ``torch.mps.compile_shader`` on each call.
    This launcher dispatches the compiled kernel with the given grid and
    threadgroup dimensions.

    Uses a 3D threadgroup dispatch model: ``_block_dims`` specifies the
    threadgroup size as ``(x, y, z)``.  The grid specifies the number of
    threadgroups per dimension.
    """
    kwargs.pop("num_warps", None)
    kwargs.pop("num_stages", None)
    if kwargs:
        raise exc.BackendUnsupported(
            "metal", f"unexpected launcher kwargs: {sorted(kwargs)}"
        )

    from .._compiler.metal.metal_launcher import set_required_threads_per_threadgroup

    set_required_threads_per_threadgroup(metal_kernel, _block_dims)
    lib, kernel_name = metal_kernel(*args)  # type: ignore[operator]

    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    dispatch_fn = getattr(lib, kernel_name)
    bx, by, bz = _block_dims
    # Pad grid to 3D
    gx = grid[0] if len(grid) > 0 else 1
    gy = grid[1] if len(grid) > 1 else 1
    gz = grid[2] if len(grid) > 2 else 1
    total_threads = (gx * bx, gy * by, gz * bz)
    group_size = (bx, by, bz)
    dispatch_fn(*tensor_args, threads=total_threads, group_size=group_size)


_flydsl_jit_cache: dict = {}
_flydsl_compiled_cache: dict = {}  # cache_key -> flyc.CompiledFunction
_flydsl_stream: object = None  # one persistent HIP stream, reused across launches


def default_flydsl_launcher(
    flydsl_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _num_threads: int = 64,
    **kwargs: object,
) -> None:
    """Default launcher for FlyDSL kernels on ROCm devices.

    @flyc.kernel can only be called inside @flyc.jit, so we generate a temp
    Python file with a @flyc.jit wrapper that has explicit fx.Tensor params.
    """
    import importlib.util
    import tempfile

    kwargs.pop("num_warps", None)
    kwargs.pop("num_stages", None)
    if kwargs:
        from .. import exc

        raise exc.BackendUnsupported(
            "flydsl", f"unexpected launcher kwargs: {sorted(kwargs)}"
        )

    n = len(args)
    gx = grid[0] if len(grid) > 0 else 1
    gy = grid[1] if len(grid) > 1 else 1
    gz = grid[2] if len(grid) > 2 else 1

    # _num_threads is supplied by FlyDSLBackend.launcher_keyword_args as 64*bm
    # (bm warps, one warp per row). Default 64 = single-warp (bm=1) fallback.

    cache_key = (id(flydsl_kernel), n, gx, gy, gz, _num_threads)
    if cache_key not in _flydsl_jit_cache:
        params = ", ".join(f"_a{i}: fx.Tensor" for i in range(n))
        call = ", ".join(f"_a{i}" for i in range(n))
        src = f"""
import flydsl.expr as fx
import flydsl.compiler as flyc

def _make(kernel):
    @flyc.jit
    def _launch({params}, _s: fx.Stream):
        kernel({call}).launch(
            grid=({gx}, {gy}, {gz}),
            block=({_num_threads}, 1, 1),
            stream=_s,
        )
    return _launch
"""
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, prefix="_flydsl_jit_"
        ) as tf:
            tf.write(src)
            fname = tf.name

        spec = importlib.util.spec_from_file_location(
            f"_flydsl_jit_{abs(hash(cache_key))}", fname
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _flydsl_jit_cache[cache_key] = mod._make(flydsl_kernel)

    # Persistent null stream (fx.Stream(None) -> stream 0), reused across launches.
    global _flydsl_stream
    if _flydsl_stream is None:
        import flydsl.expr as _fx

        _flydsl_stream = _fx.Stream(None)

    # Cache a directly-callable flyc.CompiledFunction instead of re-entering the
    # @flyc.jit wrapper on every launch. CompiledFunction.__call__ only refreshes
    # ctypes arg storage (data ptrs / scalars / stream) and invokes the JIT-ed
    # function pointer (~5 us), skipping inspect.Signature.bind and cache-key
    # resolution in JitFunction.__call__ (the dominant per-launch CPU cost).
    compiled = _flydsl_compiled_cache.get(cache_key)
    if compiled is None:
        import flydsl.compiler as _flyc

        compiled = _flyc.compile(_flydsl_jit_cache[cache_key], *args, _flydsl_stream)
        _flydsl_compiled_cache[cache_key] = compiled
    compiled(*args, _flydsl_stream)
