"""Fused fast launch: collapse ``Kernel.__call__`` → generated ``_run`` wrapper →
``default_launcher`` → ``CompiledKernel.run`` into one cached step.

On a repeat call every value the generated ``_run`` wrapper computes -- the grid
and the constant scalar launch arguments -- is a pure function of the argument
metadata (exact shapes/strides/dtypes and scalar values), and which launch-arg
slot comes from which input tensor is fixed per specialization.  So after one
*priming* launch we compile a "launch closure" and later calls launch the
compiled kernel directly, skipping the wrapper frame, the launcher frame, and
the second key build.

Hot path (see ``Kernel.__call__``): :meth:`Kernel._fused_dispatch_key` maps the
call straight to a cached closure, which -- in one Python frame -- verifies the
kernel's captured globals, builds the ``CompiledKernel.run`` argument list, and
launches.  That key is ``_fast_dispatch_key`` (exact per-argument metadata,
scalar values, and the user ``key=`` function) plus a 16-byte pointer-alignment
bit per tensor.  It is built per call by the same type-dispatched code the normal
dispatch uses -- *not* an expression baked to the primed argument layout -- so a
call whose layout differs (e.g. a tensor where the prime saw ``None``) yields a
different key and re-primes, never a false hit.

Correctness.  The fused path bypasses ``default_launcher``'s direct-launch key,
which is where the two *uncatchable* launch hazards are normally guarded -- a
misaligned pointer into an alignment-specialized binary (async CUDA error) and a
stale captured global (silently wrong numerics).  The alignment bit is folded
into the key, so an unaligned pointer arriving after an aligned prime is a new
key -> its own closure.  Each closure re-checks its specialization's captured
globals against their primed values (a deleted global reads a ``_MISSING``
sentinel and so compares unequal) and raises :class:`GlobalsChanged` on any
mismatch; the caller catches it and re-validates via the slow path, which raises
Triton's own error.

Because the Triton CUDA launcher reads only ``.data_ptr()`` from a pointer
argument (sizes/strides reach the kernel as separate explicit scalar args), a
launch-arg tensor can be replaced by any input tensor sharing its ``data_ptr``
-- which is why a zero-storage-offset view (e.g. ``x.view(...)``) is served by
its base input.

Fusion is disabled (permanently, per kernel) whenever the recipe cannot be
reproduced this way: a host wrapper that does side-effecting work beyond
launch + return (checked once by inspecting the wrapper source -- see
:func:`_wrapper_body_is_fuseable`), a non-``None`` wrapper return (covers kernels
that allocate their output), multiple device launches, a launch arg that is
neither a plain tensor nor an ``int``/``float``/``bool``/``None`` constant, a
tensor arg with no matching input pointer, cooperative-grid launches, extra
launcher kwargs, or any active launch hook/knob.
"""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from .kernel import BoundKernel
    from .kernel import Kernel


# Sentinel for a captured global that was deleted since a fused recipe was
# primed: the recipe's inline guard reads it via ``.get`` so a deletion compares
# unequal (raising GlobalsChanged) rather than throwing an uncaught KeyError.
_MISSING_GLOBAL = object()


class GlobalsChanged(Exception):
    """Raised by a launch closure when a captured Triton global was mutated
    since the recipe was primed; the caller falls back to the slow path."""


@functools.cache
def _triton_driver() -> object:
    """The Triton ``driver`` module, cached so the fused-launch hot path avoids a
    fresh ``from triton.runtime import driver`` import lookup on every call."""
    from triton.runtime import driver

    return driver


def _triton_direct_launch_ok(triton_kernel: object) -> bool:
    """One-time (per specialization) check that a raw launch bypassing
    ``JITFunction.run`` is safe.

    Verifies no hook that ``JITFunction.run`` would have invoked is active and no
    knob that feeds into launch behavior is set.  Called by the fused-launch
    prime path when caching a compiled specialization, never on the per-launch
    hot path.
    """
    # pyrefly: ignore [missing-attribute]
    if triton_kernel.pre_run_hooks or triton_kernel.debug:
        return False
    import triton.knobs as knobs

    rt = knobs.runtime
    if rt.add_stages_inspection_hook is not None or rt.debug:
        return False
    for hook in (rt.launch_enter_hook, rt.launch_exit_hook):
        # HookChain with empty ``calls`` is inactive; a plain user callable
        # is always treated as active.
        if hook is not None and getattr(hook, "calls", True):
            return False
    return not knobs.compilation.instrumentation_mode


def build_fused_key_fn(
    args: tuple[object, ...],
    user_key_fn: Callable[..., object] | None,
) -> Callable[[tuple[object, ...]], object]:
    """Compile a flat dispatch-key builder for the fused cache, from a sample
    argument tuple (the layout seen on the first prime for that layout).

    The returned lambda encodes, per position, the same information as
    ``Kernel._fast_dispatch_key`` -- ``type(arg)`` and, for a tensor,
    dtype/shape/stride/device and (frozenset-normalized) dynamo static indices,
    for a scalar its value -- plus a 16-byte pointer-alignment bit per tensor,
    and finally the user ``key=`` function's result.  Compiling one flat
    expression (as Triton's ``binder`` does) is markedly faster on the hot path
    than the generic per-call ``_fast_dispatch_key`` builder for many-argument
    kernels.

    Every position embeds ``a[i].__class__`` first, so a call whose layout
    differs from this sample -- e.g. a tensor where the sample had ``None`` --
    produces a different key (never a false hit).  If a differing layout makes an
    expression raise (``.dtype`` on an int, ``.data_ptr`` on ``None``), the
    caller catches it and rebuilds the expression for the new layout.
    """
    parts: list[str] = []
    for i, a in enumerate(args):
        t = type(a)
        if t is torch.Tensor or t is torch.nn.Parameter:
            # frozenset(...) matches _fast_dispatch_key's normalization so a
            # set/list _dynamo_static_indices stays hashable.
            parts.append(
                f"a[{i}].__class__, a[{i}].dtype, a[{i}].shape, a[{i}].stride(), "
                f"a[{i}].device, _fs(getattr(a[{i}], '_dynamo_static_indices', None)), "
                f"not a[{i}].data_ptr() & 15"
            )
        elif t is int or t is float or t is bool or t is str:
            parts.append(f"a[{i}].__class__, a[{i}]")
        elif a is None:
            # NoneType marker; a tensor arriving here yields Tensor != NoneType.
            parts.append(f"a[{i}].__class__")
        else:  # torch.dtype / torch.device -- value is its own key
            parts.append(f"a[{i}].__class__, a[{i}]")
    namespace: dict[str, object] = {
        "_fs": lambda si: None if si is None else frozenset(si)
    }
    meta = f"({', '.join(parts)},)" if parts else "()"
    if user_key_fn is not None:
        namespace["_kf"] = user_key_fn
        return eval(f"lambda a: ({meta}, _kf(*a))", namespace)
    return eval(f"lambda a: {meta}", namespace)


def _wrapper_body_is_fuseable(bound: BoundKernel) -> bool:
    """True iff the generated host wrapper is a plain launch + return.

    The fused recipe reproduces only the launcher call and the return value.  A
    wrapper that also runs host-side statements with observable effects -- a bare
    tensor-method call like ``out.zero_()``, an augmented assignment, a loop, a
    conditional -- would have those effects silently dropped on replay (e.g. a
    pre-launch ``zero_`` leaving stale data in a region the device loop does not
    cover).  Such wrappers only emit a ``TensorOperationInWrapper`` warning, which
    a kernel may suppress, so they cannot be told apart at runtime by behavior;
    inspect the wrapper source.

    Allowed statements: ``assert`` (specialization guards), plain ``x = ...``
    and tuple-unpacking assignments (constexpr block sizes, shape unpacking,
    ``x = x.view(...)`` reshapes -- pure, no side effect), the single
    ``_launcher(...)`` call expressed as a bare ``Expr``, and the ``return``.
    Anything else (a non-launcher ``Expr`` call, ``AugAssign``, ``For``, ``If``,
    ``With``, ...) makes the kernel non-fuseable.
    """
    run = bound._run
    if run is None:
        return False
    try:
        source = textwrap.dedent(inspect.getsource(run))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return False
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        return False
    saw_launch = False
    for stmt in func.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Assert)):
            continue
        if isinstance(stmt, ast.Return):
            continue
        if isinstance(stmt, ast.Expr):
            # The only bare expression statement allowed is the launcher call.
            call = stmt.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_launcher"
            ):
                saw_launch = True
                continue
            return False
        # For / If / With / AugAssign / While / etc. -- host-side control flow
        # or side effects the recipe cannot reproduce.
        return False
    return saw_launch


def _disable(kernel: Kernel, result: object) -> object:
    kernel._fused_disabled = True
    return result


def fused_prime(kernel: Kernel, bound: BoundKernel, args: tuple[object, ...]) -> object:
    """Run the kernel once through the wrapper while recording its single
    launcher call, then compile and cache a launch closure (or permanently
    disable fusion for this kernel).  Returns the wrapper's result.
    """
    from . import default_launcher

    # One-time source inspection: reject wrappers that do host-side work beyond
    # launch + return (their side effects would be dropped on replay).
    if kernel._fused_wrapper_ok is None:
        kernel._fused_wrapper_ok = _wrapper_body_is_fuseable(bound)
    if not kernel._fused_wrapper_ok:
        # pyrefly: ignore [not-callable]
        return _disable(kernel, bound._run(*args))

    captured: dict[str, Any] = {"n": 0}

    def recorder(
        triton_kernel: object,
        grid: tuple[int, ...],
        *largs: object,
        num_warps: int,
        num_stages: int,
        ptx_options: str | None = None,
        launch_cooperative_grid: bool = False,
        **kwargs: object,
    ) -> object:
        captured["n"] += 1
        captured["triton_kernel"] = triton_kernel
        captured["grid"] = grid
        captured["largs"] = largs
        captured["ptx_options"] = ptx_options
        captured["lcg"] = launch_cooperative_grid
        captured["kwargs"] = kwargs
        compiled = default_launcher(
            triton_kernel,
            grid,
            *largs,
            num_warps=num_warps,
            num_stages=num_stages,
            ptx_options=ptx_options,
            launch_cooperative_grid=launch_cooperative_grid,
            # pyrefly: ignore [bad-argument-type]
            **kwargs,
        )
        captured["compiled"] = compiled
        return compiled

    # pyrefly: ignore [not-callable]
    result = bound._run(*args, _launcher=recorder)

    # Only single-launch, out-arg / in-place kernels are fuseable.  A non-None
    # return means the wrapper produced a value we cannot reproduce without
    # re-running it (typically an allocated output tensor).
    if (
        result is not None
        or captured["n"] != 1
        or captured["kwargs"]
        or captured["ptx_options"] is not None
        or captured["lcg"]
        or type(captured["grid"]) is not tuple
    ):
        return _disable(kernel, result)

    triton_kernel = captured["triton_kernel"]
    compiled = captured["compiled"]
    if compiled is None or not _triton_direct_launch_ok(triton_kernel):
        return _disable(kernel, result)

    # Map each input tensor's data_ptr -> its position.
    ptr_to_idx: dict[int, int] = {}
    for i, a in enumerate(args):
        ta = type(a)
        if ta is torch.Tensor or ta is torch.nn.Parameter:
            # pyrefly: ignore [missing-attribute]
            ptr_to_idx.setdefault(a.data_ptr(), i)

    # Resolve each launcher arg to an input-tensor index or a baked constant.
    arg_parts: list[str] = []
    namespace: dict[str, object] = {}
    for j, a in enumerate(captured["largs"]):
        ta = type(a)
        if ta is torch.Tensor or ta is torch.nn.Parameter:
            # pyrefly: ignore [missing-attribute]
            i = ptr_to_idx.get(a.data_ptr())
            if i is None:
                return _disable(kernel, result)
            arg_parts.append(f"a[{i}]")
        elif ta is int or ta is float or ta is bool or a is None:
            namespace[f"_c{j}"] = a
            arg_parts.append(f"_c{j}")
        else:
            return _disable(kernel, result)

    grid = captured["grid"]
    grid_size = len(grid)
    namespace["_gx"] = grid[0]
    namespace["_gy"] = grid[1] if grid_size > 1 else 1
    namespace["_gz"] = grid[2] if grid_size > 2 else 1
    namespace["_run"] = compiled.run
    namespace["_fn"] = compiled.function
    namespace["_pm"] = compiled.packed_metadata
    namespace["_compiled"] = compiled  # keep alive so _run/_fn stay valid
    # pyrefly: ignore [missing-attribute]
    namespace["_active"] = _triton_driver().active
    namespace["_Changed"] = GlobalsChanged

    # Inline captured-globals verification: a mutated OR deleted global compares
    # unequal (deletion reads the _MISSING sentinel via .get), raising
    # GlobalsChanged so the caller takes the slow path (which re-validates and
    # raises Triton's own error).
    namespace["_MISSING"] = _MISSING_GLOBAL
    guard_conds: list[str] = []
    for gi, ((name, _), (val, gdict)) in enumerate(
        # pyrefly: ignore [missing-attribute]
        triton_kernel.used_global_vals.items()
    ):
        namespace[f"_g{gi}"] = gdict
        namespace[f"_gv{gi}"] = val
        guard_conds.append(f"_g{gi}.get({name!r}, _MISSING) != _gv{gi}")
    guard = ""
    if guard_conds:
        guard = f"    if {' or '.join(guard_conds)}: raise _Changed\n"

    launch_args = ", ".join(arg_parts)
    src = (
        "def _launch(a):\n"
        f"{guard}"
        "    _run(_gx, _gy, _gz, "
        "_active.get_current_stream(_active.get_current_device()), "
        "_fn, _pm, None, None, None"
        f"{', ' + launch_args if launch_args else ''})\n"
    )
    exec(src, namespace)
    launch = cast("Callable[[tuple[object, ...]], None]", namespace["_launch"])

    # Key on the same per-call, type-dispatched key the hot path uses.  It is
    # strictly finer than the specialization key (includes exact metadata,
    # scalar values, the user key= function, and the alignment bit), so a hit is
    # always the specialization a real bind() would have chosen.  A key of None
    # (unhandled arg type / no tensor) means "don't fuse this call".
    fused_key = kernel._fused_dispatch_key(args)
    if fused_key is not None:
        kernel._fused_recipes[fused_key] = launch
    return result
