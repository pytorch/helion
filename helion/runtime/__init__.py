from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .. import exc
from .._compiler.metal.metal_launcher import (
    default_metal_launcher as default_metal_launcher,
)
from .config import Config as Config
from .cute.launcher import cute_cuda_graph as cute_cuda_graph
from .cute.launcher import default_cute_launcher as default_cute_launcher
from .cute.occupancy import get_max_active_clusters as get_max_active_clusters
from .kernel import Kernel as Kernel
from .kernel import OutputCodeOptions as OutputCodeOptions
from .kernel import kernel as kernel
from .pallas.launcher import default_pallas_launcher as default_pallas_launcher
from .settings import is_pallas_interpret as _module_is_pallas_interpret
from .triton.launcher import default_launcher as _triton_default_launcher
from .triton.launcher import get_num_sm as _triton_get_num_sm
from .triton.launcher import get_num_xcd as get_num_xcd
from .triton.launcher import set_triton_allocator as set_triton_allocator

if TYPE_CHECKING:
    import torch

log: logging.Logger = logging.getLogger(__name__)


def default_launcher(
    triton_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    _remote_copy_signal_dst: torch.Tensor | None = None,
    _remote_copy_signal_slots_per_program: int = 0,
    _remote_copy_process_group_name: str | None = None,
    _remote_barrier_signal_slots_per_program: int = 0,
    _remote_barrier_process_group_name: str | None = None,
    _remote_copy_scratch_specs: tuple[tuple[torch.Tensor, int], ...] = (),
    _persistent_state_specs: tuple[
        tuple[torch.Tensor, int, torch.dtype, bool], ...
    ] = (),
    _persistent_state_process_group_name: str | None = None,
    _persistent_state_rank_digest: str | None = None,
    _persistent_state_rank_shares: tuple[int, ...] = (),
    _minimum_resident_programs: int = 0,
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
            _remote_copy_signal_dst=_remote_copy_signal_dst,
            _remote_copy_signal_slots_per_program=_remote_copy_signal_slots_per_program,
            _remote_copy_process_group_name=_remote_copy_process_group_name,
            _remote_barrier_signal_slots_per_program=_remote_barrier_signal_slots_per_program,
            _remote_barrier_process_group_name=_remote_barrier_process_group_name,
            _remote_copy_scratch_specs=_remote_copy_scratch_specs,
            _persistent_state_specs=_persistent_state_specs,
            _persistent_state_process_group_name=_persistent_state_process_group_name,
            _persistent_state_rank_digest=_persistent_state_rank_digest,
            _persistent_state_rank_shares=_persistent_state_rank_shares,
            _minimum_resident_programs=_minimum_resident_programs,
            ptx_options=ptx_options,
            launch_cooperative_grid=launch_cooperative_grid,
            **kwargs,
        )
    except Exception as error:
        message = str(error)
        if "Cannot make_shape_compatible: incompatible dimensions" in message:
            raise exc.ShapeMismatch("kernel operands", message) from error
        if message.startswith("Cross-loop scheduling requires "):
            raise exc.InvalidConfig(message) from error
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


# cache_key -> @flyc.jit-wrapped launcher / flyc.CompiledFunction. Both grow
# unbounded across distinct (kernel, nargs, grid, threads) keys; acceptable for
# the experimental backend, but revisit with eviction if a long-lived process
# autotunes many shapes.
_flydsl_jit_cache: dict = {}
_flydsl_compiled_cache: dict = {}
_flydsl_stream: object = None  # one persistent HIP stream, reused across launches


def default_flydsl_launcher(
    flydsl_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _num_threads: int = 64,
    **kwargs: object,
) -> None:
    """Default launcher for FlyDSL kernels on ROCm devices.

    @flyc.kernel can only run inside @flyc.jit, so build a launcher wrapping it in
    @flyc.jit with explicit fx.Tensor params. flyc's ASTRewriter re-parses the
    wrapper via ``inspect.getsource``, so its source must be retrievable -- we
    register it in ``linecache`` under a synthetic name rather than writing a temp
    file (no disk I/O, nothing to leak if the process dies mid-compile).
    """
    import linecache

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

    # _num_threads = 64*bm from launcher_keyword_args; default 64 = bm=1.
    # id(flydsl_kernel) is safe as a key component only because the cached closure
    # (_make(flydsl_kernel)) keeps the kernel object alive, so its id can't be
    # recycled for a different kernel while the entry lives.
    cache_key = (id(flydsl_kernel), n, gx, gy, gz, _num_threads)
    if cache_key not in _flydsl_jit_cache:
        params = ", ".join(f"_a{i}: fx.Tensor" for i in range(n))
        call = ", ".join(f"_a{i}" for i in range(n))
        src = f"""import flydsl.expr as fx
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
        fname = f"<flydsl_jit_{abs(hash(cache_key))}>"
        # Register the source so inspect.getsource (used by flyc's ASTRewriter)
        # can retrieve it from the in-memory string.
        linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
        mod_ns: dict = {}
        exec(compile(src, fname, "exec"), mod_ns)
        _flydsl_jit_cache[cache_key] = mod_ns["_make"](flydsl_kernel)

    # Persistent null stream (fx.Stream(None) -> stream 0), reused across launches.
    global _flydsl_stream
    if _flydsl_stream is None:
        import flydsl.expr as _fx  # pyrefly: ignore[missing-import]

        _flydsl_stream = _fx.Stream(None)

    # Cache a directly-callable flyc.CompiledFunction instead of re-entering the
    # @flyc.jit wrapper on every launch.
    compiled = _flydsl_compiled_cache.get(cache_key)
    if compiled is None:
        import flydsl.compiler as _flyc  # pyrefly: ignore[missing-import]

        compiled = _flyc.compile(_flydsl_jit_cache[cache_key], *args, _flydsl_stream)
        _flydsl_compiled_cache[cache_key] = compiled
    compiled(*args, _flydsl_stream)
