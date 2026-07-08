"""Fused fast launch: collapse ``Kernel.__call__`` → generated ``_run`` wrapper →
``default_launcher`` → ``CompiledKernel.run`` into one cached step.

On a repeat call every value the generated ``_run`` wrapper computes -- the grid
and the constant scalar launch arguments -- is a pure function of the argument
metadata (exact shapes/strides/dtypes and scalar values), and which launch-arg
slot comes from which input tensor is fixed per specialization.  So after one
*priming* launch we compile a "launch closure" and later calls launch the
compiled kernel directly, skipping the wrapper frame, the launcher frame, and
the second key build.

Hot path (see ``Kernel.__call__``): a single flat, ``eval``-compiled key (built
once per kernel) maps straight to that closure, which -- in one Python frame --
verifies the kernel's captured globals, builds the ``CompiledKernel.run``
argument list, and launches.  The flat key encodes the same per-argument
metadata as :meth:`Kernel._fast_dispatch_key` plus a 16-byte pointer-alignment
bit per input tensor.

Correctness.  The fused path bypasses ``default_launcher``'s direct-launch key,
which is where the two *uncatchable* launch hazards are normally guarded -- a
misaligned pointer into an alignment-specialized binary (async CUDA error) and a
stale captured global (silently wrong numerics).  The alignment bit is folded
into the flat key, so an unaligned pointer arriving after an aligned prime is a
new key -> its own closure.  Each closure re-checks its specialization's
captured globals against their primed values and raises :class:`_GlobalsChanged`
(deletion raises ``KeyError``) on any mismatch; the caller treats either as a
miss and re-primes via the slow path, which raises Triton's own error.

Because the Triton CUDA launcher reads only ``.data_ptr()`` from a pointer
argument (sizes/strides reach the kernel as separate explicit scalar args), a
launch-arg tensor can be replaced by any input tensor sharing its ``data_ptr``
-- which is why a zero-storage-offset view (e.g. ``x.view(...)``) is served by
its base input.

Output-allocating kernels are supported: the generated wrapper is codegen'd over
fake tensors, so every host-side allocation is a pure function of input metadata
and scalar values (never of tensor data), all of which are in the fused key.
The recipe records each allocated tensor's ``(shape, stride, dtype, device)``
and rebuilds it with ``torch.empty_strided`` on replay; the caching allocator
always returns >=16-byte-aligned storage, matching the alignment the binary was
specialized for.  The wrapper's return value (``None``, a tensor, or a
tuple/list of tensors and simple constants) is reproduced from inputs,
allocations, and baked constants.

Fusion is disabled (permanently, per kernel) whenever the recipe cannot be
reproduced this way: multiple device launches, a launch/return tensor with a
nonzero storage offset or a view layout over storage it shares with another
(unreconstructable blind), a launch arg that is neither a plain tensor nor an
``int``/``float``/``bool``/``None`` constant, a return value outside the
supported shapes, cooperative-grid launches, extra launcher kwargs, or any
active launch hook/knob.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from .kernel import BoundKernel
    from .kernel import Kernel


class _GlobalsChanged(Exception):
    """Raised by a launch closure when a captured Triton global was mutated."""


def _build_flat_key_fn(
    args: tuple[object, ...],
) -> Callable[[tuple[object, ...]], tuple[object, ...]]:
    """Compile the kernel-wide flat dispatch key from a sample argument tuple.

    Encodes, per position, the same metadata as ``Kernel._fast_dispatch_key``
    (dtype/shape/stride/device + dynamo static indices for tensors, class+value
    for scalars) plus a 16-byte pointer-alignment bit for each tensor.  Baking
    one flat expression avoids the per-call type dispatch and nested-tuple
    construction of the method-based key.  A later call whose argument layout
    differs from the primed one makes the expression raise (e.g. ``.dtype`` on
    an int), which the caller treats as a miss and re-primes.
    """
    parts: list[str] = []
    for i, a in enumerate(args):
        t = type(a)
        if t is torch.Tensor or t is torch.nn.Parameter:
            parts.append(
                f"a[{i}].dtype, a[{i}].shape, a[{i}].stride(), a[{i}].device, "
                f"getattr(a[{i}], '_dynamo_static_indices', None), "
                f"not a[{i}].data_ptr() & 15"
            )
        elif t is int or t is float or t is bool or t is str:
            parts.append(f"a[{i}].__class__, a[{i}]")
        elif a is None:
            parts.append("None")
        else:  # torch.dtype / torch.device
            parts.append(f"a[{i}]")
    return eval(f"lambda a: ({', '.join(parts)},)")


def _disable(kernel: Kernel, result: object) -> object:
    kernel._fused_disabled = True
    return result


def fused_prime(kernel: Kernel, bound: BoundKernel, args: tuple[object, ...]) -> object:
    """Run the kernel once through the wrapper while recording its single
    launcher call, then compile and cache a launch closure (or permanently
    disable fusion for this kernel).  Returns the wrapper's result.
    """
    from . import _triton_direct_launch_ok
    from . import _triton_driver
    from . import default_launcher

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

    # Fuseable kernels issue exactly one plain device launch.
    if (
        captured["n"] != 1
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

    namespace: dict[str, object] = {}

    # Tensors the wrapper allocated on the host (outputs / scratch): every one
    # is a pure function of input metadata + scalars (the wrapper is codegen'd
    # over fake tensors, so an allocation never depends on tensor data), and all
    # of those are in the fused key.  Record each distinct one's storage layout
    # and rebuild it with ``empty_strided`` in the closure; the caching
    # allocator always returns >=16B-aligned storage, matching the alignment the
    # binary was specialized for.  ``alloc`` returns the source expression for a
    # tensor (an input ``a[i]`` or an allocation ``_o{k}``) or None if it cannot
    # be reproduced -- a nonzero storage offset, or the same storage reused with
    # a different layout (a view we cannot reconstruct blind).
    alloc_slots: dict[int, tuple[int, tuple[int, ...], tuple[int, ...]]] = {}

    def alloc(t: torch.Tensor, *, launch_arg: bool) -> str | None:
        dp = t.data_ptr()
        i = ptr_to_idx.get(dp)
        if i is not None:
            # The C launcher reads only a pointer from a tensor arg, so any input
            # sharing this storage works as a launch arg; a returned value must
            # match the input exactly (else it is a view we would return wrong).
            if launch_arg:
                return f"a[{i}]"
            src_t = cast("torch.Tensor", args[i])
            same = (
                t.shape == src_t.shape
                and t.stride() == src_t.stride()
                and t.storage_offset() == src_t.storage_offset()
                and t.dtype == src_t.dtype
            )
            return f"a[{i}]" if same else None
        slot = alloc_slots.get(dp)
        if slot is None:
            if t.storage_offset() != 0:
                return None
            k = len(alloc_slots)
            alloc_slots[dp] = (k, tuple(t.shape), t.stride())
            namespace[f"_shape{k}"] = tuple(t.shape)
            namespace[f"_stride{k}"] = t.stride()
            namespace[f"_dt{k}"] = t.dtype
            namespace[f"_dev{k}"] = t.device
            return f"_o{k}"
        k, shape, stride = slot
        if tuple(t.shape) != shape or t.stride() != stride or t.storage_offset():
            return None  # same storage, different layout -> unreconstructable view
        return f"_o{k}"

    # Resolve each launcher arg to an input, an allocation, or a baked constant.
    arg_parts: list[str] = []
    for j, a in enumerate(captured["largs"]):
        ta = type(a)
        if ta is torch.Tensor or ta is torch.nn.Parameter:
            expr = alloc(cast("torch.Tensor", a), launch_arg=True)
            if expr is None:
                return _disable(kernel, result)
            arg_parts.append(expr)
        elif ta is int or ta is float or ta is bool or a is None:
            namespace[f"_c{j}"] = a
            arg_parts.append(f"_c{j}")
        else:
            return _disable(kernel, result)

    # Reproduce the wrapper's return value from inputs / allocations / constants.
    ret_expr = _return_expr(result, alloc, namespace)
    if ret_expr is None:
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
    namespace["_Changed"] = _GlobalsChanged
    namespace["_empty_strided"] = torch.empty_strided

    # Inline captured-globals verification: any mutated value raises, a deleted
    # global raises KeyError; both are caught by the caller as a miss.
    guard_conds: list[str] = []
    for gi, ((name, _), (val, gdict)) in enumerate(
        # pyrefly: ignore [missing-attribute]
        triton_kernel.used_global_vals.items()
    ):
        namespace[f"_g{gi}"] = gdict
        namespace[f"_gv{gi}"] = val
        guard_conds.append(f"_g{gi}[{name!r}] != _gv{gi}")
    guard = ""
    if guard_conds:
        guard = f"    if {' or '.join(guard_conds)}: raise _Changed\n"

    allocs = "".join(
        f"    _o{k} = _empty_strided(_shape{k}, _stride{k}, "
        f"dtype=_dt{k}, device=_dev{k})\n"
        for k in range(len(alloc_slots))
    )
    launch_args = ", ".join(arg_parts)
    src = (
        "def _launch(a):\n"
        f"{guard}"
        f"{allocs}"
        "    _run(_gx, _gy, _gz, "
        "_active.get_current_stream(_active.get_current_device()), "
        "_fn, _pm, None, None, None"
        f"{', ' + launch_args if launch_args else ''})\n"
        f"    return {ret_expr}\n"
    )
    exec(src, namespace)
    launch = cast("Callable[[tuple[object, ...]], object]", namespace["_launch"])

    if kernel._fused_key_fn is None:
        kernel._fused_key_fn = _build_flat_key_fn(args)
    kernel._fused_recipes[kernel._fused_key_fn(args)] = launch
    return result


def _return_expr(
    result: object,
    alloc: Callable[..., str | None],
    namespace: dict[str, object],
) -> str | None:
    """Source expression that reproduces the wrapper's return value, or None if
    it cannot be reproduced (a returned view, or an unsupported container).

    Supports ``None``, a single tensor, and a ``tuple``/``list`` of tensors and
    simple constants -- the return shapes Helion host wrappers actually emit.
    """
    if result is None:
        return "None"
    tr = type(result)
    if tr is torch.Tensor or tr is torch.nn.Parameter:
        return alloc(cast("torch.Tensor", result), launch_arg=False)
    if tr is tuple or tr is list:
        parts: list[str] = []
        for n, elem in enumerate(cast("tuple[object, ...]", result)):
            te = type(elem)
            if te is torch.Tensor or te is torch.nn.Parameter:
                expr = alloc(cast("torch.Tensor", elem), launch_arg=False)
                if expr is None:
                    return None
                parts.append(expr)
            elif te is int or te is float or te is bool or elem is None:
                namespace[f"_r{n}"] = elem
                parts.append(f"_r{n}")
            else:
                return None
        if tr is list:
            return f"[{', '.join(parts)}]"
        return f"({', '.join(parts)},)" if len(parts) == 1 else f"({', '.join(parts)})"
    return None
