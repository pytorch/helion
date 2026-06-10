"""Per-spec fast launcher that bypasses Triton's ``JITFunction.run``.

Public entry points re-exported from :mod:`helion.runtime`:

* :func:`default_launcher` — Triton's full per-call pipeline. Used as
  the fallback when the fast path can't (or shouldn't) engage, and as
  the codegen wrapper's ``_launcher`` kwdefault before
  :meth:`BoundKernel.set_config` installs a fast launcher.
* :class:`_FastLauncher` — closure that caches Triton's per-call
  bookkeeping across multiple ``_spec`` keys.
* :func:`build_fast_launcher` — factory used by
  ``BoundKernel._install_fast_launcher``.

The ``_FastLauncher`` docstring explains the multi-spec cache and
which parts of Triton's per-call pipeline it skips.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch

from .. import exc

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Hashable


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
    """Default launcher function that executes the kernel immediately."""
    # For both CUDA and MTIA, use the same kernel execution
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


class _SpecEntry:  # noqa: B903 - explicit __slots__ for hot-path attribute access
    """Per-Triton-spec compiled-kernel state.

    One instance lives in ``_FastLauncher._spec_cache`` for each distinct
    Triton specialization (pointer alignment + scalar specialization +
    binary-affecting knob state) we have ever launched. Each entry holds
    everything the C launcher needs plus a snapshot of
    ``used_global_vals`` taken at compile time for this binary.
    """

    __slots__ = (
        "compiled_run",
        "kernel_launch_metadata",
        "packed_metadata",
        "triton_function",
        "used_global_checks",
    )

    def __init__(
        self,
        *,
        compiled_run: Callable[..., object],
        triton_function: object,
        packed_metadata: object,
        kernel_launch_metadata: Callable[..., object],
        used_global_checks: tuple[tuple[dict, str, object], ...],
    ) -> None:
        self.compiled_run = compiled_run
        self.triton_function = triton_function
        self.packed_metadata = packed_metadata
        self.kernel_launch_metadata = kernel_launch_metadata
        self.used_global_checks = used_global_checks


class _FastLauncher:
    """Multi-spec fast launcher that bypasses Triton's ``JITFunction.run``.

    The first call primes the launcher: captures Triton's active driver,
    device, knob namespaces, and identifies which positional args are
    tensor pointers (so we can compute alignment specialization inline
    on every subsequent call instead of invoking Triton's binder).

    Every call computes a tiny *spec key* — alignment bits + the
    binary-affecting knob state (``debug``, ``instrumentation_mode``,
    ``add_stages_inspection_hook``) — and dict-looks-up a cached
    :class:`_SpecEntry`. On hit, dispatches straight into the C launcher.
    On miss, compiles (via Triton) for that spec, caches the resulting
    entry, and dispatches.

    Helion's :class:`BoundKernel` already specializes on the *Helion*
    axes (dtype, shape, stride, device type), so the only per-call
    specialization left to Triton is pointer alignment and per-call
    knob state — which we encode in the spec key.

    Compared to a full ``JITFunction.run`` round-trip we skip:
    * Triton's binder (we compute alignment inline ~6x cheaper)
    * ``compute_cache_key`` + ``kernel_cache.get`` (replaced by a tuple
      hash + dict lookup on our own ``_spec_cache``)
    * The always-on ``kernel.launch_metadata`` call when no profiler
      hooks are attached
    * The whole ``JITFunction.run`` Python frame, kwargs munging, and
      per-call device/binder unpacks.

    If priming or any per-spec compile fails, we transparently fall
    back to :func:`default_launcher` for that launch so the kernel
    still runs.
    """

    __slots__ = (
        "_active_driver",
        "_device",
        "_get_current_stream",
        "_init_failed",
        "_knobs_compilation",
        "_knobs_runtime",
        "_launch_cooperative_grid",
        "_num_warps",
        "_num_stages",
        "_prime_lock",
        "_primed",
        "_ptx_options",
        "_run_kwargs",
        "_spec_cache",
        "_tensor_arg_indices",
    )

    def __init__(
        self,
        *,
        num_warps: int,
        num_stages: int,
        launch_cooperative_grid: bool = False,
        ptx_options: str | None = None,
        extra_kwargs: dict | None = None,
    ) -> None:
        self._num_warps = num_warps
        self._num_stages = num_stages
        self._launch_cooperative_grid = launch_cooperative_grid
        self._ptx_options = ptx_options
        # Pre-built kwargs dict for warmup AND the fallback path. Stored
        # once so the hot path doesn't have to build a fresh kwargs dict
        # per launch.
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

        # Serializes both the one-time priming (see :meth:`_prime`) AND
        # the per-spec compile-on-miss in :meth:`__call__`. Concurrent
        # first-of-spec callers must not redundantly compile, so they
        # use double-checked locking against ``_spec_cache``.
        self._prime_lock = threading.Lock()
        # Per-launcher state, set once by :meth:`_prime` on the first call.
        self._primed = False
        self._init_failed = False
        self._active_driver: object = None
        self._device: object = None
        self._get_current_stream: Callable[[object], object] | None = None
        self._knobs_runtime: object = None
        self._knobs_compilation: object = None
        # Positions of ``*args`` that are torch.Tensor pointers — used by
        # the hot path to compute the alignment portion of the spec key
        # without invoking Triton's binder.
        self._tensor_arg_indices: tuple[int, ...] = ()
        # Per-spec compiled-kernel cache, keyed on
        # ``(align_bits, debug, instrumentation_mode, stages_hook_id)``.
        # Each entry is a :class:`_SpecEntry` holding the compiled
        # binary's ``run``, function handle, packed metadata,
        # launch_metadata builder, and ``used_global_vals`` snapshot.
        self._spec_cache: dict[Hashable, _SpecEntry] = {}

    def _prime(self, triton_kernel: object, args: tuple[object, ...]) -> None:
        """One-time launcher setup. Captures driver/knob references and
        identifies which positional args are tensor pointers.

        Called once on the first invocation, under ``self._prime_lock``.
        Also runs Triton's binder ONCE to verify that
        ``tuple(bound_args.values()) == args`` (so we can pass ``args``
        directly to the C launcher in every subsequent call, skipping
        the binder entirely on the hot path).

        On any failure we set ``_init_failed = True`` so subsequent
        calls fall through to :func:`default_launcher`. ``_primed=True``
        is set last via the outer ``finally`` so any caller waiting on
        the prime lock sees either fully-published state or a clean
        ``_init_failed=True`` sentinel — never a half-written mix.
        """
        try:
            try:
                import triton
                from triton.runtime.driver import driver
            except ImportError:
                self._init_failed = True
                return

            try:
                active_driver = driver.active
                # pyrefly: ignore[missing-attribute]
                device = active_driver.get_current_device()
                # Unpack mirrors jit.py:720; we consume only the binder
                # for the one-time bind-skip-safe probe below.
                (
                    _kernel_cache,
                    _kernel_key_cache,
                    _target,
                    _backend,
                    binder,
                    # pyrefly: ignore[missing-attribute]
                ) = triton_kernel.device_caches[device]
                # Verify that ``tuple(bound_args.values()) == args``. This
                # has to hold for every call we want to keep on the fast
                # path, since the hot path passes ``args`` straight to
                # the C launcher without re-invoking the binder. Helion's
                # generated wrapper hands the launcher the kernel's
                # positional args in declaration order, so this is the
                # common case; if it doesn't hold for *this* kernel
                # signature, we permanently fall back.
                bound_args, _spec, _opts = binder(*args, **self._run_kwargs)
                bound_values = tuple(bound_args.values())
                if not (
                    len(bound_values) == len(args)
                    and all(b is a for b, a in zip(bound_values, args, strict=True))
                ):
                    self._init_failed = True
                    return

                # Per-launcher state — constant across every launch.
                tensor_indices = tuple(
                    i for i, a in enumerate(args) if isinstance(a, torch.Tensor)
                )
                self._active_driver = active_driver
                self._device = device
                # pyrefly: ignore[missing-attribute]
                self._get_current_stream = active_driver.get_current_stream
                self._knobs_runtime = triton.knobs.runtime
                self._knobs_compilation = triton.knobs.compilation
                self._tensor_arg_indices = tensor_indices
            except Exception:
                self._init_failed = True
        finally:
            # Publish ``_primed`` last so threads waiting on the lock
            # either see a fully-set-up launcher or a clean
            # ``_init_failed`` sentinel — never a half-written mix.
            self._primed = True

    def _materialize_spec(
        self,
        triton_kernel: object,
        grid: tuple[int, ...],
        args: tuple[object, ...],
    ) -> _SpecEntry | None:
        """Compile (or fetch from Triton's cache) the binary for the
        current spec and return a fully-populated :class:`_SpecEntry`.

        Called by :meth:`__call__` on a ``_spec_cache`` miss, under
        ``self._prime_lock``. Invokes Triton's full pipeline once: this
        is the "cold" path where we accept the binder + compile + cache
        lookup costs, knowing every subsequent call with this spec_key
        will dispatch directly via the cached entry.
        """
        try:
            warmup_kwargs = {**self._run_kwargs, "grid": grid, "warmup": True}
            # pyrefly: ignore[missing-attribute]
            compiled_kernel = triton_kernel.run(*args, **warmup_kwargs)
            if compiled_kernel is None:
                return None
            # Access ``.run`` FIRST so ``_init_handles()`` populates
            # ``.function`` etc. before we read them.
            run_fn = compiled_kernel.run
            triton_function = compiled_kernel.function
            packed_metadata = compiled_kernel.packed_metadata
            kernel_launch_metadata = compiled_kernel.launch_metadata
            # Snapshot ``used_global_vals`` at this binary's compile
            # time. Each spec entry has its own snapshot — different
            # binaries may pin different globals.
            ugv = getattr(triton_kernel, "used_global_vals", None)
            used_global_checks: tuple[tuple[dict, str, object], ...] = ()
            if ugv:
                used_global_checks = tuple(
                    (gdict, name, val) for (name, _gid), (val, gdict) in ugv.items()
                )
            return _SpecEntry(
                compiled_run=run_fn,
                triton_function=triton_function,
                packed_metadata=packed_metadata,
                kernel_launch_metadata=kernel_launch_metadata,
                used_global_checks=used_global_checks,
            )
        except Exception:
            return None

    def __call__(
        self,
        triton_kernel: object,
        grid: tuple[int, ...],
        *args: object,
        # Spell out the kwargs the generated wrapper always passes so
        # CPython binds them positionally into local slots rather than
        # packing them into a per-call **kwargs dict. The closure already
        # has the runtime values baked in; these parameter names exist
        # solely to absorb the wrapper's keyword arguments cheaply, and
        # are otherwise unused.
        num_warps: int | None = None,
        num_stages: int | None = None,
        launch_cooperative_grid: bool = False,
        **kwargs: object,
    ) -> object:
        # Under torch.compile / Dynamo tracing, defer to ``default_launcher``
        # which routes through ``triton_kernel.run`` and is captured by
        # Dynamo's ``triton_kernel_wrapper_mutation`` HOP handler. (Same
        # reason as in the single-spec design — the fast path's direct
        # ``_cuda_getCurrentRawStream`` call returns ``int`` and trips
        # Dynamo's "non-Tensor return" check.)
        if torch.compiler.is_compiling():
            return default_launcher(
                triton_kernel,
                grid,
                *args,
                **self._run_kwargs,
            )

        if not self._primed:
            # Double-checked locking: the unsynchronized read above is
            # the fast path; the lock-guarded re-check below serializes
            # concurrent first calls so only one thread runs ``_prime``.
            with self._prime_lock:
                if not self._primed:
                    self._prime(triton_kernel, args)
        if self._init_failed:
            return default_launcher(
                triton_kernel,
                grid,
                *args,
                **self._run_kwargs,
            )

        # Multi-device guard. The cached driver/stream getter were
        # captured against the priming device; if the caller has since
        # switched CUDA devices, fall back to ``default_launcher`` which
        # dispatches via Triton's per-device ``device_caches``.
        active_driver = self._active_driver
        if (
            active_driver is not None
            # pyrefly: ignore[missing-attribute]
            and active_driver.get_current_device() != self._device
        ):
            return default_launcher(
                triton_kernel,
                grid,
                *args,
                **self._run_kwargs,
            )

        # Read knob state once. The binary-affecting knobs go into the
        # spec key (so toggling them just lands on a different cache
        # entry rather than forcing a fallback). The launch hooks are
        # passed straight through to the C launcher per-call so a
        # profiler attaching after priming still fires.
        knobs_runtime = self._knobs_runtime
        knobs_compilation = self._knobs_compilation
        runtime_debug = getattr(knobs_runtime, "debug", False)
        instr_mode = getattr(knobs_compilation, "instrumentation_mode", "")
        stages_hook = getattr(knobs_runtime, "add_stages_inspection_hook", None)

        # Compute the spec key INLINE — no Triton binder call. Helion's
        # ``BoundKernel`` already specializes on dtype/shape/stride/device
        # via its own bind cache, so the remaining Triton-level
        # specialization for this kernel is pointer alignment and the
        # binary-affecting knob state. ``_tensor_arg_indices`` is the
        # precomputed list of positions of tensor pointer args
        # (captured at priming), so the loop walks just those args.
        align_bits = 0
        for i in self._tensor_arg_indices:
            # pyrefly: ignore[missing-attribute]
            if args[i].data_ptr() & 15 == 0:
                align_bits |= 1 << i
        spec_key = (
            align_bits,
            runtime_debug,
            instr_mode,
            id(stages_hook) if stages_hook is not None else 0,
        )

        # Cache lookup. On miss we acquire the lock to serialize
        # compile-on-miss for concurrent first-of-spec callers, then
        # re-check the cache (another thread may have compiled while
        # we waited) before invoking the cold path.
        spec_cache = self._spec_cache
        entry = spec_cache.get(spec_key)
        if entry is None:
            with self._prime_lock:
                entry = spec_cache.get(spec_key)
                if entry is None:
                    entry = self._materialize_spec(triton_kernel, grid, args)
                    if entry is None:
                        return default_launcher(
                            triton_kernel,
                            grid,
                            *args,
                            **self._run_kwargs,
                        )
                    spec_cache[spec_key] = entry

        # Per-spec ``used_global_vals`` check. Each spec entry has its
        # own snapshot; a mutation that would have caused Triton to
        # raise ``RuntimeError`` is detected here and we fall back so
        # the user sees Triton's own error from ``JITFunction.run``.
        for gdict, name, expected in entry.used_global_checks:
            if gdict.get(name) != expected:
                return default_launcher(
                    triton_kernel,
                    grid,
                    *args,
                    **self._run_kwargs,
                )

        # All fallback conditions are now ruled out — commit to the
        # fast path. Fire ``pre_run_hooks`` inline (mirrors
        # jit.py:717-718). Hooks don't affect the binary, so doing this
        # here is safe; doing it later than the fallback conditions
        # avoids double-firing if any condition above had tripped.
        pre_run_hooks = getattr(triton_kernel, "pre_run_hooks", None)
        if pre_run_hooks:
            hook_kwargs = {
                **self._run_kwargs,
                "debug": runtime_debug,
                "instrumentation_mode": instr_mode,
            }
            for hook in pre_run_hooks:
                hook(*args, **hook_kwargs)

        try:
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            # pyrefly: ignore[not-callable]
            stream = self._get_current_stream(self._device)
            # Re-read launch hooks per-call so a profiler attached
            # after priming still fires. Skip ``launch_metadata`` when
            # no hook will actually consume it (the common case; this
            # is one of our biggest wins vs ``JITFunction.run``).
            enter_hook = getattr(knobs_runtime, "launch_enter_hook", None)
            exit_hook = getattr(knobs_runtime, "launch_exit_hook", None)
            if (enter_hook is None or getattr(enter_hook, "calls", None) == []) and (
                exit_hook is None or getattr(exit_hook, "calls", None) == []
            ):
                launch_metadata = None
            else:
                launch_metadata = entry.kernel_launch_metadata(grid, stream, *args)
            return entry.compiled_run(
                grid_0,
                grid_1,
                grid_2,
                stream,
                entry.triton_function,
                entry.packed_metadata,
                launch_metadata,
                enter_hook,
                exit_hook,
                *args,
            )
        except Exception as error:
            message = str(error)
            if "Cannot make_shape_compatible: incompatible dimensions" in message:
                raise exc.ShapeMismatch("kernel operands", message) from error
            raise


def build_fast_launcher(
    *,
    num_warps: int,
    num_stages: int,
    launch_cooperative_grid: bool = False,
    ptx_options: str | None = None,
    extra_kwargs: dict | None = None,
) -> _FastLauncher:
    """Build a :class:`_FastLauncher` closure with config baked in.

    Invoked from :meth:`BoundKernel.set_config` after the generated Triton
    kernel is compiled. The returned object's ``__call__`` signature
    matches :func:`default_launcher`, so the generated wrapper can use it
    as a drop-in replacement (threaded through as ``_launcher=`` by the
    bound-kernel-level wrapper installed in ``set_config``).
    """
    return _FastLauncher(
        num_warps=num_warps,
        num_stages=num_stages,
        launch_cooperative_grid=launch_cooperative_grid,
        ptx_options=ptx_options,
        extra_kwargs=extra_kwargs,
    )
