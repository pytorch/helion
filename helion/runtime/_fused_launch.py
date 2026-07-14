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
launches.  That key holds the same information as ``_fast_dispatch_key`` (exact
per-argument metadata, scalar values, and the user ``key=`` function) plus a
16-byte pointer-alignment bit per tensor.  Both keys are built by the shared
:func:`build_key_fn` -- a flat expression compiled (baked) for the primed
argument layout, but carrying a per-position class guard, so a call whose layout
differs (e.g. a tensor where the prime saw ``None``) raises :class:`_LayoutChanged`
and re-primes for the new layout, never a false hit.

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


_BLOCK_SIZE_GLOBAL_PREFIX = "_BLOCK_SIZE_"


def _is_helion_block_size_global(name: object) -> bool:
    """True for Helion-generated block-size globals safe to trust after compile."""
    return (
        type(name) is str
        and name.startswith(_BLOCK_SIZE_GLOBAL_PREFIX)
        and name[len(_BLOCK_SIZE_GLOBAL_PREFIX) :].isdigit()
    )


# Raised by a baked key builder when the call's argument *layout* differs from
# the sample it was compiled for (a differing type at some position); the caller
# catches it and rebuilds the builder for the new layout.  This is the single
# mechanism that guarantees a layout change is never a false hit -- see
# build_key_fn's per-position ``a[i].__class__ is <C> or _raise()`` guard.
class _LayoutChanged(Exception):
    pass


def _raise() -> object:
    raise _LayoutChanged


def build_key_fn(
    args: tuple[object, ...],
    user_key_fn: Callable[..., object] | None,
    *,
    include_alignment: bool,
    bail_when_no_tensor: bool,
) -> Callable[[tuple[object, ...]], object]:
    """Compile a flat dispatch-key builder from a sample argument tuple (the
    layout seen when this builder was compiled).  This is the single source of
    truth for both dispatch keys:

    * ``Kernel._fast_dispatch_key`` -- ``include_alignment=False,
      bail_when_no_tensor=True``: keys the ``_dispatch_cache`` (BoundKernel),
      which does not specialize on pointer alignment, and returns ``None`` when
      no tensor pins the device so the caller takes the slow ``bind()`` path.
    * the fused-launch cache -- ``include_alignment=True,
      bail_when_no_tensor=False``: adds a 16-byte pointer-alignment bit per
      tensor (the fused path bypasses ``default_launcher``'s direct-launch key
      where alignment is otherwise guarded) and always builds a key.

    The returned lambda encodes one layout token plus, per tensor,
    dtype/shape/stride/device and (frozenset-normalized) dynamo static indices,
    per scalar its value, and finally the user ``key=`` result. Compiling one
    flat expression (as Triton's ``binder`` does) is markedly faster on the hot
    path than a generic per-call loop for many-argument kernels.

    Layout safety: every position first checks ``a[i].__class__ is <C>`` for the
    sampled class ``C`` and calls ``_raise()`` (-> :class:`_LayoutChanged`) on a
    mismatch.  So a call whose layout differs -- e.g. a tensor where the sample
    had ``None`` or a scalar -- always raises rather than silently emitting a
    truncated (collision-prone) key; the caller rebuilds for the new layout.  The
    layout token is what prevents keys from different baked layouts from
    colliding in the shared recipe cache.
    """
    # A guard for every position (including bail positions), so a builder that
    # bails to None for *this* layout still raises _LayoutChanged -- and is
    # rebuilt -- when a later call's layout differs.  Otherwise a cached bail
    # lambda would wrongly return None for every future layout.  The leading
    # guard checks the arg *count*, so a differing length (which would otherwise
    # IndexError or silently drop trailing args) also forces a rebuild.
    #
    # The key carries one prebuilt layout token instead of emitting each class
    # guard into the tuple. The guards still make layout changes rebuild the
    # key function, while the token prevents keys from different layouts from
    # colliding in the shared recipe cache.
    len_guard = f"(len(a) == {len(args)} or _raise())"
    parts: list[str] = []
    guards: list[str] = [len_guard]
    namespace: dict[str, object] = {
        "_fs": lambda si: None if si is None else frozenset(si),
        "_layout": object(),
        "_raise": _raise,
    }
    has_tensor = False
    bail = False
    for i, a in enumerate(args):
        t = type(a)
        # Per-position layout guard: raises _LayoutChanged on a class mismatch,
        # so a differing type at this position forces a rebuild instead of
        # silently emitting a truncated (collision-prone) key.
        namespace[f"_C{i}"] = t
        guard = f"(a[{i}].__class__ is _C{i} or _raise())"
        guards.append(guard)
        if t is torch.Tensor or t is torch.nn.Parameter:
            has_tensor = True
            # frozenset(...) keeps a set/list _dynamo_static_indices hashable.
            align = f", not a[{i}].data_ptr() & 15" if include_alignment else ""
            parts.append(
                f"a[{i}].dtype, a[{i}].shape, a[{i}].stride(), "
                f"a[{i}].device, _fs(getattr(a[{i}], '_dynamo_static_indices', None))"
                f"{align}"
            )
        elif t is int or t is float or t is bool or t is str:
            parts.append(f"a[{i}]")
        elif a is None:
            pass
        elif t is torch.dtype or t is torch.device:  # value is its own key
            parts.append(f"a[{i}]")
        else:
            # Unhandled arg type (container, tensor subclass, ...): the caller
            # takes the slow path (fast) / skips fusion (fused) via a None key.
            bail = True
    if bail or (bail_when_no_tensor and not has_tensor):
        # Evaluate every guard (so a layout change raises) then yield None.
        return eval(f"lambda a: ({' and '.join(guards)}) and None", namespace)
    meta_parts = ["_layout", *parts]
    meta = f"(({' and '.join(guards)}) and ({', '.join(meta_parts)},))"
    if user_key_fn is not None:
        namespace["_kf"] = user_key_fn
        return eval(f"lambda a: ({meta}, _kf(*a))", namespace)
    return eval(f"lambda a: {meta}", namespace)


# Calls the fused recipe may safely skip on replay because they mutate no
# existing (input/out) tensor: pure shape/view ops, allocations (an allocated
# launch arg is caught downstream by arg resolution / the non-None return
# check), the host grid-topology helpers, and pure builtins used in shape math.
# Anything not listed -- notably every in-place op (``zero_``/``fill_``/... by
# torch's trailing-underscore convention) -- makes an assignment non-fuseable.
_PURE_CALL_NAMES = frozenset(
    {
        # shape / view ops -- no data mutation
        "view", "reshape", "size", "stride", "expand", "expand_as",
        "broadcast_to", "broadcast_tensors", "permute", "transpose", "t",
        "contiguous", "flatten", "unflatten", "unsqueeze", "squeeze",
        "as_strided", "narrow", "select", "movedim", "to", "type_as",
        "dim", "numel", "element_size",
        # allocations (pure w.r.t. existing tensors)
        "tensor", "scalar_tensor", "empty", "empty_like", "empty_strided",
        "zeros", "zeros_like", "ones", "ones_like", "new_empty", "new_zeros",
        # helion host-side grid-topology helpers
        "get_num_sm", "get_num_xcd",
        # pure builtins used in shape/grid arithmetic
        "min", "max", "len", "int", "float", "bool", "abs",
    }
)  # fmt: skip


def _expr_is_pure(node: ast.expr | None) -> bool:
    """True iff evaluating ``node`` has no observable side effect on an existing
    tensor -- i.e. every ``Call`` it contains targets a name in
    :data:`_PURE_CALL_NAMES`.  A call through anything but a plain name or
    attribute (e.g. a subscript or an arbitrary expression) is treated as impure.
    """
    for sub in ast.walk(node) if node is not None else ():
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            else:
                return False
            if name not in _PURE_CALL_NAMES:
                return False
    return True


# Name of the module-proxy the generated wrapper reads a kernel's module-scope
# globals through (helion/_compiler/output_header.py::SOURCE_MODULE); a launch
# arg spelled ``_source_module.<attr>`` is re-read from live module state on
# every call, so a fused recipe (which bakes scalar launch args at prime time)
# must not fuse it -- see _wrapper_reads_module_global.
_SOURCE_MODULE = "_source_module"


def _wrapper_reads_module_global(node: ast.expr | None) -> bool:
    """True iff ``node`` reads a kernel module-scope global via the wrapper's
    module proxy (an ``ast.Attribute`` on ``_source_module``).

    The wrapper re-reads such a global on every call and passes it to the
    launcher as a scalar arg; the fused recipe instead bakes each scalar launch
    arg as a constant at prime time, so a later mutation of the global would be
    silently ignored (the non-fused path re-reads it and stays correct).  Fusing
    such a kernel is therefore unsafe.
    """
    for sub in ast.walk(node) if node is not None else ():
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == _SOURCE_MODULE
        ):
            return True
    return False


def _wrapper_body_is_fuseable(bound: BoundKernel) -> bool:
    """True iff the generated host wrapper is a plain launch + return.

    The fused recipe reproduces only the launcher call and the return value.  A
    wrapper that also runs host-side statements with observable effects -- a bare
    tensor-method call like ``out.zero_()``, an augmented assignment, a loop, a
    conditional, *or an assignment whose RHS mutates a tensor* (e.g.
    ``out = out.zero_()``) -- would have those effects silently dropped on replay
    (e.g. a pre-launch ``zero_`` leaving stale data in a region the device loop
    does not cover).  Such wrappers only emit a ``TensorOperationInWrapper``
    warning, which a kernel may suppress, so they cannot be told apart at runtime
    by behavior; inspect the wrapper source.

    Allowed statements: ``assert`` (specialization guards), plain ``x = ...`` and
    tuple-unpacking assignments *whose RHS is pure* (constexpr block sizes, shape
    unpacking, ``x = x.view(...)`` reshapes -- see :func:`_expr_is_pure`), the
    single ``_launcher(...)`` call expressed as a bare ``Expr``, and the
    ``return``.  Anything else (a non-launcher ``Expr`` call, an assignment with a
    side-effecting RHS, ``AugAssign``, ``For``, ``If``, ``With``, ...) makes the
    kernel non-fuseable.

    A launcher call that reads a module-scope global (``_source_module.<attr>``,
    passed as a scalar launch arg re-read every call) is also rejected: the fused
    recipe bakes scalar launch args at prime time, so a later mutation of the
    global would be silently ignored -- see :func:`_wrapper_reads_module_global`.
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
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            # A pure RHS that does not read a module global (a side-effecting
            # call like ``out = out.zero_()`` would be dropped on replay; a
            # ``s = _source_module.SCALE`` read would be frozen at prime time).
            if _expr_is_pure(stmt.value) and not _wrapper_reads_module_global(
                stmt.value
            ):
                continue
            return False
        if isinstance(stmt, ast.Assert):
            continue
        if isinstance(stmt, ast.Return):
            continue
        if isinstance(stmt, ast.Expr):
            # The only bare expression statement allowed is the launcher call,
            # and only if none of its args re-read a module global.
            call = stmt.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_launcher"
                and not _wrapper_reads_module_global(call)
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
    direct_arg_parts: list[str] = []
    tensor_direct_arg_parts: list[str] = []
    tensor_arg_names: dict[int, str] = {}
    tensor_direct_params: list[str] = []
    for i, a in enumerate(args):
        ta = type(a)
        if ta is torch.Tensor or ta is torch.nn.Parameter:
            tensor_arg_names[i] = f"a{len(tensor_direct_params)}"
            tensor_direct_params.append(tensor_arg_names[i])
    namespace: dict[str, object] = {}
    for j, a in enumerate(captured["largs"]):
        ta = type(a)
        if ta is torch.Tensor or ta is torch.nn.Parameter:
            # pyrefly: ignore [missing-attribute]
            i = ptr_to_idx.get(a.data_ptr())
            if i is None:
                return _disable(kernel, result)
            arg_parts.append(f"a[{i}]")
            direct_arg_parts.append(f"a{i}")
            tensor_direct_arg_parts.append(tensor_arg_names[i])
        elif ta is int or ta is float or ta is bool or a is None:
            namespace[f"_c{j}"] = a
            arg_parts.append(f"_c{j}")
            direct_arg_parts.append(f"_c{j}")
            tensor_direct_arg_parts.append(f"_c{j}")
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
    active = _triton_driver().active
    namespace["_active"] = active
    namespace["_Changed"] = GlobalsChanged

    launch_device: int | None = None
    for a in args:
        ta = type(a)
        if ta is torch.Tensor or ta is torch.nn.Parameter:
            # pyrefly: ignore [missing-attribute]
            launch_device = a.device.index
            break
    if launch_device is not None:
        namespace["_stream"] = active.get_current_stream
        namespace["_dev"] = launch_device
        stream_expr = "_stream(_dev)"
    else:
        namespace["_get_stream"] = active.get_current_stream
        namespace["_get_dev"] = active.get_current_device
        stream_expr = "_get_stream(_get_dev())"

    # Inline captured-globals verification: a mutated OR deleted global compares
    # unequal (deletion reads the _MISSING sentinel via .get), raising
    # GlobalsChanged so the caller takes the slow path (which re-validates and
    # raises Triton's own error).
    namespace["_MISSING"] = _MISSING_GLOBAL
    guard_conds: list[str] = []
    trusted_no_guard = True
    for gi, ((name, _), (val, gdict)) in enumerate(
        # pyrefly: ignore [missing-attribute]
        triton_kernel.used_global_vals.items()
    ):
        trusted_no_guard = trusted_no_guard and _is_helion_block_size_global(name)
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
        f"{stream_expr}, "
        "_fn, _pm, None, None, None"
        f"{', ' + launch_args if launch_args else ''})\n"
    )
    exec(src, namespace)
    launch = cast("Callable[[tuple[object, ...]], None]", namespace["_launch"])
    direct_params = ", ".join(f"a{i}" for i in range(len(args)))
    direct_launch_args = ", ".join(direct_arg_parts)
    direct_src = (
        f"def _direct_launch({direct_params}):\n"
        f"{guard}"
        "    _run(_gx, _gy, _gz, "
        f"{stream_expr}, "
        "_fn, _pm, None, None, None"
        f"{', ' + direct_launch_args if direct_launch_args else ''})\n"
    )
    exec(direct_src, namespace)
    direct_launch = namespace["_direct_launch"]
    launch._helion_direct_launch = direct_launch  # type: ignore[attr-defined]
    trusted_direct_launch = direct_launch
    if trusted_no_guard and guard:
        trusted_direct_src = (
            f"def _trusted_direct_launch({direct_params}):\n"
            "    _run(_gx, _gy, _gz, "
            f"{stream_expr}, "
            "_fn, _pm, None, None, None"
            f"{', ' + direct_launch_args if direct_launch_args else ''})\n"
        )
        exec(trusted_direct_src, namespace)
        trusted_direct_launch = namespace["_trusted_direct_launch"]
        launch._helion_trusted_direct_launch = trusted_direct_launch  # type: ignore[attr-defined]
    elif trusted_no_guard:
        launch._helion_trusted_direct_launch = direct_launch  # type: ignore[attr-defined]
    tensor_direct_src = (
        f"def _tensor_direct_launch({', '.join(tensor_direct_params)}):\n"
        f"{guard}"
        "    _run(_gx, _gy, _gz, "
        f"{stream_expr}, "
        "_fn, _pm, None, None, None"
        f"{', ' + ', '.join(tensor_direct_arg_parts) if tensor_direct_arg_parts else ''})\n"
    )
    exec(tensor_direct_src, namespace)
    direct_launch._helion_tensor_direct_launch = namespace[  # type: ignore[attr-defined]
        "_tensor_direct_launch"
    ]
    launch._helion_tensor_direct_launch = namespace[  # type: ignore[attr-defined]
        "_tensor_direct_launch"
    ]
    if trusted_no_guard:
        trusted_tensor_direct_launch = namespace["_tensor_direct_launch"]
        if guard:
            trusted_tensor_direct_src = (
                f"def _trusted_tensor_direct_launch({', '.join(tensor_direct_params)}):\n"
                "    _run(_gx, _gy, _gz, "
                f"{stream_expr}, "
                "_fn, _pm, None, None, None"
                f"{', ' + ', '.join(tensor_direct_arg_parts) if tensor_direct_arg_parts else ''})\n"
            )
            exec(trusted_tensor_direct_src, namespace)
            trusted_tensor_direct_launch = namespace["_trusted_tensor_direct_launch"]
        direct_launch._helion_trusted_tensor_direct_launch = (  # type: ignore[attr-defined]
            trusted_tensor_direct_launch
        )
        trusted_direct_launch._helion_trusted_tensor_direct_launch = (  # type: ignore[attr-defined]
            trusted_tensor_direct_launch
        )
        launch._helion_trusted_tensor_direct_launch = (  # type: ignore[attr-defined]
            trusted_tensor_direct_launch
        )

    # Key on the same per-call, type-dispatched key the hot path uses.  It is
    # strictly finer than the specialization key (includes exact metadata,
    # scalar values, the user key= function, and the alignment bit), so a hit is
    # always the specialization a real bind() would have chosen.  A key of None
    # (unhandled arg type / no tensor) means "don't fuse this call".
    fused_key = kernel._fused_dispatch_key(args)
    if fused_key is not None:
        kernel._fused_recipes[fused_key] = launch
    return result
