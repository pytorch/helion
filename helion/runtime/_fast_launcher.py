"""Minimal static-kernel launcher (Inductor-style).

Public entry points re-exported from :mod:`helion.runtime`:

* :func:`default_launcher` — Triton's full per-call pipeline. Used as
  the fallback when the static launcher can't engage (and as the
  codegen wrapper's ``_launcher`` kwdefault before
  :meth:`BoundKernel.set_config` installs the static launcher).
* :class:`StaticLauncher` — lazy-priming launcher that builds PyTorch
  Inductor's ``StaticallyLaunchedCudaKernel`` on first call and
  dispatches every subsequent call straight to ``cuLaunchKernel`` via
  ``torch._C._StaticCudaLauncher``. Skips Triton's Python-side
  ``CudaLauncher.__call__`` wrapper.
* :func:`_build_fast_launcher` — factory used by
  ``BoundKernel._install_fast_launcher`` (private; pulls config /
  env from the bound kernel and constructs a ``StaticLauncher``).

Falls back to :func:`default_launcher` on any construction failure:
older PyTorch without ``_StaticCudaLauncher``, missing cubin path,
hooks installed at prime time, kernel with ``num_ctas != 1`` /
``launch_pdl`` / global+profile scratch, etc.

Two correctness guards run on every call (matching Inductor's
static-launcher invariants):

* **Alignment**: per-call check on the tensor pointer positions
  captured at prime time. Any unaligned arg falls back to
  ``default_launcher`` (Triton's full path), which compiles a
  matching spec. Inductor clones unaligned inputs and copies writes
  back using IR metadata; Helion has no such metadata, so a clone
  here could silently drop writes to in-place / output args.
* **``used_global_vals`` mutation**: snapshot taken at prime time;
  any per-call mismatch falls back to ``default_launcher`` so Triton's
  own ``RuntimeError`` surfaces instead of the static path silently
  using the stale binary.

The launcher does NOT carry multi-spec dispatch, per-call hook
re-reads, a multi-device guard, or debug-knob retracking. If your
workload swaps CUDA devices mid-process, attaches launch hooks AFTER
the first call, or relies on per-knob recompilation, set
``HELION_SKIP_FAST_LAUNCHER=1`` and Triton's default path will handle
those edges.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

import torch

from .. import exc

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from .._compiler.compile_environment import CompileEnvironment
    from .config import Config

log: logging.Logger = logging.getLogger(__name__)

# Sentinel for "key not present" lookups — distinguishes a missing
# entry from a legitimately-stored ``None`` value, which the pool's
# global-mutation check would otherwise conflate.
_MISSING: object = object()

# Optional imports for the static-launcher path. If any fails (older
# PyTorch without ``_StaticCudaLauncher``, MTIA build, etc.),
# ``_STATIC_LAUNCHER_AVAILABLE`` stays False and every call routes
# through :func:`default_launcher`.
try:
    from torch._inductor.runtime.cache_dir_utils import triton_cache_dir
    from torch._inductor.runtime.runtime_utils import triton_hash_to_path_key
    from torch._inductor.runtime.static_triton_launcher import (
        statically_launched_kernel_by_device,
    )
    from triton.runtime.driver import driver

    _STATIC_LAUNCHER_AVAILABLE = True
except ImportError:
    _STATIC_LAUNCHER_AVAILABLE = False


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
    """Triton's full per-call pipeline; fallback for the static path."""
    run_kwargs: dict = {
        "grid": grid,
        "warmup": False,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "launch_cooperative_grid": launch_cooperative_grid,
        **kwargs,
    }
    if ptx_options is not None:
        run_kwargs["ptx_options"] = ptx_options
    try:
        return triton_kernel.run(  # type: ignore[union-attr]
            *args,
            **run_kwargs,
        )
    except Exception as error:
        message = str(error)
        if "Cannot make_shape_compatible: incompatible dimensions" in message:
            raise exc.ShapeMismatch("kernel operands", message) from error
        raise


class StaticLauncher:
    """Lazy-priming static-kernel launcher.

    First call:
      - Run Triton warmup compile to obtain a ``CompiledKernel``.
      - Locate the on-disk cubin (``triton_cache_dir`` + hash + name).
      - Build ``StaticallyLaunchedCudaKernel`` and ``load_kernel``.
      - Compute the constexpr-arg filter (the static launcher's
        ``arg_tys`` only enumerates runtime params; Helion's wrapper
        passes constexpr args inline too).

    Subsequent calls:
      - Unpack ``grid`` into 3 ints.
      - Fetch the current CUDA stream.
      - Filter out constexpr args.
      - ``static_kernel.run(grid_x, grid_y, grid_z, stream, *args)``
        jumps straight into ``cuLaunchKernel`` via PyTorch's C++
        extension — no Triton Python wrapper, no per-call guards.

    On any priming failure: ``self._static_kernel`` stays ``None`` and
    every call routes through :func:`default_launcher`. The escape
    hatch ``HELION_SKIP_STATIC_LAUNCHER=1`` forces this state.
    """

    __slots__ = (
        "_device",
        "_device_index",
        "_get_stream",
        "_keep_indices",
        "_prime_lock",
        "_primed",
        "_ptr_indices",
        "_run_kwargs",
        "_static_kernel",
        "_expected_globals",
    )

    def __init__(
        self,
        *,
        device: torch.device,
        num_warps: int,
        num_stages: int,
        launch_cooperative_grid: bool = False,
        ptx_options: str | None = None,
        extra_kwargs: dict | None = None,
    ) -> None:
        run_kwargs: dict = {
            "num_warps": num_warps,
            "num_stages": num_stages,
            "launch_cooperative_grid": launch_cooperative_grid,
        }
        if extra_kwargs:
            run_kwargs.update(extra_kwargs)
        if ptx_options is not None:
            run_kwargs["ptx_options"] = ptx_options
        self._run_kwargs = run_kwargs
        self._primed = False
        self._prime_lock = threading.Lock()
        self._static_kernel: object | None = None
        self._keep_indices: tuple[int, ...] | None = None
        # Positions in the ORIGINAL ``*args`` that are tensor pointers
        # — used per-call for the 16-byte alignment check (see the
        # alignment guard in :meth:`__call__`).
        self._ptr_indices: tuple[int, ...] = ()
        # Expected values for ``triton_kernel.used_global_vals``,
        # snapshotted at prime time. Keyed by ``(name, gid)`` (the same
        # composite key Triton uses) so we can look up the live
        # ``globals_dict`` from ``triton_kernel.used_global_vals`` on
        # each call without holding our own reference to it. Mutation
        # between calls => fall back to ``default_launcher`` so
        # Triton's own guard raises rather than our launcher silently
        # dispatching the stale binary.
        #
        # NOTE: ``gid`` is ``id(globals_dict)``. If a module is
        # reloaded between prime and a later call, the old ``id`` may
        # be reused by an unrelated dict, and the live ``ugv`` lookup
        # for that ``(name, gid)`` could match a different module's
        # globals. Vanishingly unlikely in practice (Triton's own
        # ``used_global_vals`` is the source of truth and isn't
        # mutated by user code), but worth knowing.
        self._expected_globals: dict[tuple[str, int], object] = {}
        self._get_stream: object | None = None
        # Device the bound kernel was created for (carried from
        # ``BoundKernel._env.device``, which itself comes from the
        # first tensor arg at bind time). ``device.index`` is what
        # Triton's driver APIs (``load_kernel``, ``get_current_stream``)
        # take — they expect an int, not a ``torch.device``. The
        # bound-kernel cache key is keyed on the full device (incl.
        # index), so two GPUs don't collide here.
        self._device: torch.device = device
        self._device_index: int = device.index if device.index is not None else 0

    def _device_guard(self) -> AbstractContextManager[None]:
        """Context manager that makes the launcher's bound device current.

        Device-type-aware: ``torch.cuda.device`` covers CUDA *and* ROCm
        (PyTorch reports both as ``cuda``); ``torch.xpu.device`` covers
        Intel XPU. Using ``torch.cuda.device`` unconditionally would raise
        ``RuntimeError: PyTorch was compiled without CUDA support`` on an
        XPU-only build, even though Inductor's static launcher supports
        ``xpu``.
        """
        if self._device.type == "xpu":
            return torch.xpu.device(self._device_index)
        return torch.cuda.device(self._device_index)

    def _try_prime(
        self,
        triton_kernel: object,
        grid: tuple[int, ...],
        args: tuple[object, ...],
    ) -> None:
        """Best-effort: warmup compile + build the static kernel.
        Leaves ``self._static_kernel = None`` on any failure so the
        caller falls through to ``default_launcher``.
        """
        if not _STATIC_LAUNCHER_AVAILABLE:
            return
        if os.environ.get("HELION_SKIP_STATIC_LAUNCHER", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            return
        # Inductor's static launcher dispatches by device type and only
        # supports ``cuda`` (incl. ROCm — PyTorch reports both as
        # ``cuda``), ``hip``, and ``xpu``. Anything else (cpu, mps,
        # ...) → bail to ``default_launcher``.
        if self._device.type not in ("cuda", "hip", "xpu"):
            return
        try:
            warmup_kwargs = {**self._run_kwargs, "grid": grid, "warmup": True}
            compiled_kernel = triton_kernel.run(  # type: ignore[union-attr]
                *args, **warmup_kwargs
            )
            if compiled_kernel is None:
                return
            device_index = self._device_index
            cubin_path = os.path.join(
                triton_cache_dir(device_index),
                triton_hash_to_path_key(compiled_kernel.hash),
                f"{compiled_kernel.src.fn.__name__}.cubin",
            )
            if not os.path.exists(cubin_path):
                return
            compiled_kernel._cubin_path = cubin_path
            static_kernel = statically_launched_kernel_by_device(
                compiled_kernel, device_type=self._device.type
            )
            static_kernel.load_kernel(device_index)
            # Helion's generated wrapper passes ALL positional args
            # (including ``tl.constexpr`` parameters); the static
            # launcher's ``arg_tys`` only encodes runtime params, so
            # we pre-compute which positions to KEEP. ``None`` means
            # no filtering needed.
            full_constexprs = getattr(static_kernel, "full_constexprs", None)
            if full_constexprs:
                constexpr_set = set(full_constexprs)
                n_args = len(static_kernel.arg_names)
                self._keep_indices = tuple(
                    i for i in range(n_args) if i not in constexpr_set
                )
            # Map pointer positions back to ORIGINAL arg positions so
            # the per-call alignment check walks just those tensors.
            # ``arg_tys`` is in filtered (non-constexpr) order; the
            # ``_keep_indices`` table translates filtered → original.
            keep = self._keep_indices
            arg_tys = static_kernel.arg_tys
            if keep is None:
                self._ptr_indices = tuple(i for i, t in enumerate(arg_tys) if t == "O")
            else:
                self._ptr_indices = tuple(
                    keep[i] for i, t in enumerate(arg_tys) if t == "O"
                )
            # Snapshot expected values for ``used_global_vals`` so
            # per-call mutation surfaces as a fallback to
            # ``default_launcher`` (which itself raises Triton's
            # RuntimeError). We keep only the (name, gid) → expected
            # mapping — the live ``globals_dict`` is fetched from
            # ``triton_kernel.used_global_vals`` per call, so this
            # launcher doesn't pin module dicts independently of
            # Triton's own ownership.
            ugv = getattr(triton_kernel, "used_global_vals", None)
            if ugv:
                self._expected_globals = {
                    (name, gid): val for (name, gid), (val, _gdict) in ugv.items()
                }
            self._static_kernel = static_kernel
            # Cache the bound method to skip Triton's ``LazyProxy``
            # resolution on every call: ``driver.active`` is a property
            # that calls ``_active_driver_proxy()`` each access
            # (~150-400 ns), and the launcher is pinned to one device
            # so the binding never needs to be re-resolved.
            self._get_stream = driver.active.get_current_stream  # pyrefly: ignore
        except Exception:
            # Surface the underlying cause at DEBUG so users diagnosing
            # silent fallbacks (HELION_LOGS=all) can see why the static
            # path is disabled. Any failure here just means we route
            # through ``default_launcher`` — semantically equivalent,
            # only slower.
            log.debug("StaticLauncher prime failed", exc_info=True)
            self._static_kernel = None

    def __call__(
        self,
        triton_kernel: object,
        grid: tuple[int, ...],
        *args: object,
        # Spelled-out kwargs so CPython binds the wrapper's kwargs
        # into local slots rather than packing a per-call **kwargs
        # dict. Values are unused — the launcher's ``_run_kwargs``
        # has them baked in.
        num_warps: int | None = None,
        num_stages: int | None = None,
        launch_cooperative_grid: bool = False,
        **kwargs: object,
    ) -> object:
        # Under ``torch.compile``, dynamo traces through this
        # ``__call__`` and chokes on the ``getattr(triton_kernel,
        # "used_global_vals", ...)`` below ("Unsupported hasattr call"
        # on ``TritonKernelVariable``). Route compile-time tracing
        # straight through ``default_launcher`` (= Triton's
        # ``JITFunction.run``), which dynamo already knows how to
        # handle via its Triton-kernel-variable rules.
        if torch.compiler.is_compiling():
            return default_launcher(
                triton_kernel,
                grid,
                *args,
                **self._run_kwargs,
            )
        if not self._primed:
            with self._prime_lock:
                if not self._primed:
                    # Prime under the launcher's bound device so the
                    # static kernel's cubin is loaded into the right
                    # CUDA context (``cuModuleLoad`` uses the current
                    # context; loading in cuda:0's context but launching
                    # on cuda:1's stream would yield "invalid argument"
                    # from ``cuLaunchKernel``).
                    with self._device_guard():
                        self._try_prime(triton_kernel, grid, args)
                    self._primed = True
        static_kernel = self._static_kernel
        if static_kernel is None:
            return default_launcher(
                triton_kernel,
                grid,
                *args,
                **self._run_kwargs,
            )
        # ``used_global_vals`` mutation check: walk the prime-time
        # snapshot and look up each entry's live ``globals_dict`` via
        # ``triton_kernel.used_global_vals``. Iterating the snapshot
        # (not the live mapping) catches both value mutations AND
        # entries removed from ``used_global_vals`` between prime and
        # call. The ``_MISSING`` sentinel disambiguates a removed key
        # from a value that's legitimately ``None``.
        if self._expected_globals:
            ugv = getattr(triton_kernel, "used_global_vals", None) or {}
            for (name, gid), expected in self._expected_globals.items():
                entry = ugv.get((name, gid), _MISSING)
                if entry is _MISSING:
                    return default_launcher(
                        triton_kernel,
                        grid,
                        *args,
                        **self._run_kwargs,
                    )
                _val, gdict = entry
                if gdict.get(name, _MISSING) != expected:
                    return default_launcher(
                        triton_kernel,
                        grid,
                        *args,
                        **self._run_kwargs,
                    )
        # Alignment guard: the compiled binary was specialized for
        # whichever alignment we saw at prime time (usually all
        # 16-byte aligned, since the CUDA caching allocator emits
        # aligned blocks). A misaligned arg here would hit
        # ``CUDA error: misaligned address``.
        #
        # Inductor's ``align_inputs_from_check_idxs`` clones the
        # unaligned tensor to an aligned copy, relying on its IR
        # metadata to know which clones need to be copied back after
        # the launch (writes to outputs). Helion doesn't track
        # read/write status per arg, so a clone would silently drop
        # any kernel writes to that position — wrong-result bug for
        # in-place / output kernels. Fall back to ``default_launcher``
        # instead; Triton's full path will compile a new spec for the
        # new alignment if needed.
        #
        # The ``isinstance`` check guards against optional / union-typed
        # positions where ``arg_tys == "O"`` at prime time but a later
        # call passes ``None`` or a non-tensor — ``.data_ptr()`` would
        # raise. Fall through to ``default_launcher`` so Triton can
        # produce a clearer error or compile a new spec.
        for i in self._ptr_indices:
            arg = args[i]
            if not isinstance(arg, torch.Tensor) or arg.data_ptr() & 15:
                return default_launcher(
                    triton_kernel,
                    grid,
                    *args,
                    **self._run_kwargs,
                )
        try:
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            # pyrefly: ignore[not-callable]
            stream = self._get_stream(self._device_index)
            keep = self._keep_indices
            # Wrap the launch in the launcher's bound device so
            # ``cuLaunchKernel`` runs in the matching CUDA context. The
            # cost is two C calls (``cuCtxGetCurrent`` /
            # ``cuCtxSetCurrent``) per call — small, and necessary for
            # correctness when the caller's current device differs from
            # the launcher's bound device (e.g. a kernel pinned to
            # cuda:1 called while the default cuda:0 is current).
            with self._device_guard():
                if keep is None:
                    # pyrefly: ignore[missing-attribute]
                    return static_kernel.run(grid_0, grid_1, grid_2, stream, *args)
                # pyrefly: ignore[missing-attribute]
                return static_kernel.run(
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    *(args[i] for i in keep),
                )
        except Exception as error:
            message = str(error)
            if "Cannot make_shape_compatible: incompatible dimensions" in message:
                raise exc.ShapeMismatch("kernel operands", message) from error
            raise


def _build_fast_launcher(
    *,
    config: Config,
    env: CompileEnvironment,
) -> StaticLauncher:
    """Build a :class:`StaticLauncher` for the given config + env.

    Invoked from :meth:`BoundKernel.set_config` after the generated
    Triton kernel is compiled. The returned object's ``__call__``
    signature matches :func:`default_launcher`, so the generated
    wrapper can use it as a drop-in replacement (threaded through as
    ``_launcher=`` by ``set_config``).

    Consumes :meth:`TritonBackend.launcher_runtime_kwargs` to extract
    the raw runtime values from the config (the same source of truth
    ``launcher_keyword_args`` uses for codegen), then splits the dict
    into the named ``StaticLauncher`` parameters plus any leftover
    backend-tunable keys (``maxnreg``, ``_triton_config_*``-stripped
    keys, etc.) which are folded into ``extra_kwargs``. ``extra_kwargs``
    is forwarded to ``default_launcher`` on the fallback path; the
    static-launcher hot path ignores it because the cubin produced at
    prime time has those values baked in already.
    """
    backend = env.backend
    kwargs = backend.launcher_runtime_kwargs(  # type: ignore[attr-defined]
        config, has_barrier=env.has_barrier
    )
    num_warps = kwargs.pop("num_warps")
    num_stages = kwargs.pop("num_stages")
    launch_cooperative_grid = kwargs.pop("launch_cooperative_grid", False)
    ptx_options = kwargs.pop("ptx_options", None)
    return StaticLauncher(
        device=env.device,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_cooperative_grid=launch_cooperative_grid,
        ptx_options=ptx_options,
        extra_kwargs=kwargs or None,
    )
