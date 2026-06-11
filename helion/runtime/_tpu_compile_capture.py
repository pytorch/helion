"""Make a Helion Pallas kernel capturable by ``torch.compile(backend="tpu")``.

Wrapping it as a ``torch.library.custom_op`` (+ ``register_fake`` meta) makes it
opaque to Dynamo, so torch_xla captures one XLA node instead of tracing the eager
torch_tpu -> jax bridge (which breaks). Same mechanism as torch_tpu's ``jax_op``.

Two registration paths (opt in with ``HELION_TPU_COMPILE_CAPTURE=1``):

* **Decoration-time** -- annotated, functional, benchmark-free kernels (``config=``
  / ``autotune_effort="none"`` / ``aot_kernel``) register at definition: zero
  warm-up, schema from annotations, scalars folded to ``hl.constexpr`` per call.
* **First-call** (fallback) -- unannotated, mutating, or ``configs=[...]`` kernels
  register on one eager warm-up; a miss under compile raises.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import typing
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import torch

from .._compiler._dynamo.variables import _detect_mutated_inputs
from .._compiler._dynamo.variables import infer_output_spec
from ..language import constexpr

if TYPE_CHECKING:
    from .kernel import Kernel

_capture_lib = torch.library.Library("helion_capture", "FRAGMENT")
_op_counter = 0
# {(id(kernel), signature) -> op or None}. A flat dict keyed by id() keeps the
# under-compile lookup Dynamo-traceable (a WeakIdKeyDictionary thrashes it);
# id() reuse is moot for module-scope kernels, which live for the process.
_op_cache: dict[Any, Callable[..., Any] | None] = {}

# Python scalar types that map directly to a torch.library schema type.
_SCHEMA_TYPES: dict[type, str] = {float: "float", int: "int", bool: "bool"}

# Sentinel: tells Kernel.__call__ to run its normal dispatch path.
RUN_NORMAL = object()

_MISS_MESSAGE = (
    "Helion kernel {name!r} could not be captured under torch.compile on the "
    "Pallas backend. Run it once eagerly (this shape) before compiling. Kernels "
    "with input mutation or aliasing are not capturable."
)
_NOT_CAPTURABLE_MESSAGE = (
    "Capture supports Tensor/tuple outputs with no input mutation or aliasing; "
    "non-Tensor args (scalars, dtypes, flags) are baked in per value. These inputs "
    "are not capturable."
)


def _define_op(
    kernel: Kernel,
    in_decl: str,
    n_out: int,
    fake: Callable[..., Any],
    impl: Callable[..., Any],
) -> Callable[..., Any]:
    """Define + register one uniquely named ``helion_capture`` op and return it.

    The fake is registered before the impl so a fake failure can't leave a live
    impl with no meta. Outputs are Tensors (contiguous).
    """
    global _op_counter
    _op_counter += 1
    name = f"{kernel.name}_{_op_counter}"
    out_decl = "Tensor" if n_out == 1 else "(" + ", ".join(["Tensor"] * n_out) + ")"
    _capture_lib.define(f"{name}({in_decl}) -> {out_decl}")
    torch.library.register_fake(f"helion_capture::{name}", fake)
    _capture_lib.impl(name, impl, "CompositeExplicitAutograd")
    return getattr(torch.ops.helion_capture, name)


def _fake_and_impl(
    kernel: Kernel,
    n_out: int,
    reconstruct: Callable[[tuple[Any, ...]], tuple[Any, ...]],
    *,
    validate: bool,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build the (fake, impl) for a capture op. ``reconstruct`` maps the op's
    incoming args to the kernel's full arg tuple (scalars folded to constexpr). The
    fake infers output shapes per call (torch_xla recompiles per static shape) and,
    when ``validate``, rejects input mutation/aliasing -- the decoration path never
    traced at registration, while the first-call path already validated up front.
    """

    def fake(*args: object) -> torch.Tensor | tuple[torch.Tensor, ...]:
        spec = infer_output_spec(kernel, reconstruct(args))
        leaf = spec["leaf_specs"]
        if validate and (
            spec.get("mutated_inputs")
            or spec.get("direct_aliases")
            or any(s["type"] != "tensor" for s in leaf)
        ):
            raise RuntimeError(_NOT_CAPTURABLE_MESSAGE)
        outs = [
            torch.empty(s["shape"], dtype=s["dtype"], device=s["device"]) for s in leaf
        ]
        return outs[0] if n_out == 1 else tuple(outs)

    def impl(*args: object) -> torch.Tensor | tuple[torch.Tensor, ...]:
        full = reconstruct(args)
        out = kernel.bind(full)(*full)
        return tuple(out) if n_out > 1 else out

    return fake, impl


# --------------------------------------------------------------------------- #
# Decoration-time path (the standard way): schema from annotations.
# --------------------------------------------------------------------------- #
def _decoration_schema(kernel: Kernel) -> tuple[list[type | None], int] | None:
    """Derive ``(kinds, n_out)`` from the kernel's type hints, or None if the return
    isn't ``Tensor``/``tuple[Tensor, ...]`` or a param isn't schema-able (dtype,
    list, ConstExpr, ...). ``kinds[i]`` is ``None`` for a Tensor param (annotated
    ``torch.Tensor`` or unannotated), else the scalar type. An annotation that won't
    resolve makes ``get_type_hints`` raise; ``register_decoration_op`` catches that.
    """
    hints = typing.get_type_hints(kernel.fn)
    ret = hints.get("return")
    if ret is torch.Tensor:
        n_out = 1
    elif (
        typing.get_origin(ret) is tuple
        and (args := typing.get_args(ret))
        and all(a is torch.Tensor for a in args)
    ):
        n_out = len(args)
    else:
        return None
    kinds: list[type | None] = []
    for name in kernel.signature.parameters:
        hint = hints.get(name)
        if hint is None or hint is torch.Tensor:
            kinds.append(None)
        elif hint in _SCHEMA_TYPES:
            kinds.append(hint)
        else:
            return None
    return kinds, n_out


def _is_functional(kernel: Kernel) -> bool:
    """True if the kernel body neither mutates an input nor returns one directly.

    AST-only (no trace), so it is safe to check at decoration time. Subtle view
    aliasing isn't detectable here; such kernels are rare in the pointwise/norm
    domain and would otherwise be caught by the first-call trace.
    """
    names = set(kernel.signature.parameters)
    src = textwrap.dedent(inspect.getsource(kernel.fn))
    body = ast.parse(src).body[0].body  # type: ignore[attr-defined]
    if _detect_mutated_inputs(body, names):
        return False
    return not any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id in names
        for node in ast.walk(ast.Module(body, []))
    )


def _resolves_without_benchmark(kernel: Kernel) -> bool:
    """True if the kernel resolves a config with no runtime benchmark, any shape.

    Mirrors the benchmark-free cases of the autotune dispatch (backend.autotune):
    a single ``config=``, ``autotune_effort="none"`` with at most one config, or an
    ``aot_kernel`` (heuristic cache, never benchmarks). ``force_autotune`` overrides
    all of these (it benchmarks regardless), and ``configs=[...]`` / open autotuning
    benchmark per shape -- those use the first-call warm-up path instead.
    """
    s = kernel.settings
    if s.force_autotune:
        return False
    if s.autotune_effort == "none":
        return len(kernel.configs) <= 1
    return len(kernel.configs) == 1 or s.autotune_cache == "AOTAutotuneCache"


def register_decoration_op(kernel: Kernel) -> Callable[..., Any] | None:
    """Register the kernel's custom_op at decoration time, or None if ineligible.

    Eligible kernels are functional, fully schema-derivable from annotations, and
    benchmark-free; the returned callable takes the kernel's own signature and
    routes through the op under torch.compile (scalars are folded to constexpr).
    """
    # Defensive: this runs in Kernel.__init__ (callers gate on the Pallas backend),
    # so an ineligible/exotic kernel must return None, never raise and break the
    # kernel's definition.
    try:
        parsed = _decoration_schema(kernel)
        if (
            parsed is None
            or not _resolves_without_benchmark(kernel)
            or not _is_functional(kernel)
        ):
            return None
        return _build_decoration_op(kernel, *parsed)
    except Exception:
        return None


def _build_decoration_op(
    kernel: Kernel, kinds: list[type | None], n_out: int
) -> Callable[..., Any]:
    decl = ", ".join(
        f"Tensor a{i}" if k is None else f"{_SCHEMA_TYPES[k]} a{i}"
        for i, k in enumerate(kinds)
    )

    def fold(args: tuple[Any, ...]) -> tuple[Any, ...]:
        # Scalars are constexpr (compile-time constants for the Pallas kernel);
        # otherwise Helion materializes them as device buffers that orphan under
        # capture (torch_tpu aborts: "argument not provided").
        return tuple(
            constexpr(a) if k is not None else a
            for a, k in zip(args, kinds, strict=True)
        )

    # validate=True: the AST eligibility check can't see view-aliasing or in-place
    # methods, so the fake re-validates per call and raises a clean compile error.
    fake, impl = _fake_and_impl(kernel, n_out, fold, validate=True)
    op = _define_op(kernel, decl, n_out, fake, impl)

    def captured(args: tuple[Any, ...]) -> object:
        return op(*kernel.normalize_args(*args))

    return captured


# --------------------------------------------------------------------------- #
# First-call path (fallback): schema from a trace, scalars baked per value.
# --------------------------------------------------------------------------- #
def _freeze(v: object) -> object:
    """Hashable view of a non-Tensor arg; raises TypeError if it holds a Tensor."""
    if isinstance(v, torch.Tensor):
        raise TypeError("Tensor inside a container arg")
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    return v


def _signature(args: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Cache key over the arg *structure*, or None if a non-Tensor arg isn't
    hashable. Tensors contribute only their position (generic over shape/dtype);
    each distinct scalar/flag value keys a distinct (baked) op."""
    try:
        key = tuple(
            ("tensor",)
            if isinstance(a, torch.Tensor)
            else ("const", type(a).__name__, _freeze(a))
            for a in args
        )
        hash(key)
    except TypeError:
        return None
    return key


def _tensors(args: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """The Tensor args, in order -- the actual arguments the captured op takes."""
    return tuple(a for a in args if isinstance(a, torch.Tensor))


def _const_scalar(a: object) -> object:
    """Fold an int/float (incl. bool) arg to ``hl.constexpr``; others pass through."""
    return constexpr(a) if isinstance(a, (int, float)) else a


def build_op(kernel: Kernel, args: tuple[Any, ...]) -> Callable[..., Any] | None:
    """Register (once, cached) a first-call custom_op for this kernel+signature.

    Returns the op, or None if not capturable. Registration is best-effort: it runs
    in the eager ``__call__`` path (result discarded, kernel then runs normally), so
    a failure must cache None and fall through, never break the eager call.
    ``_register_op`` returns None for the expected uncapturable cases (mutation/
    aliasing/non-Tensor outputs); the except catches the rest -- e.g. a return inside
    control flow, which ``infer_output_spec`` raises on -- and caches the miss so it
    isn't retried (re-running it would re-autotune every call).
    """
    sig = _signature(args)
    if sig is None:
        return None
    key = (id(kernel), sig)
    if key not in _op_cache:
        try:
            _op_cache[key] = _register_op(kernel, args)
        except Exception:
            _op_cache[key] = None
    return _op_cache[key]


def _register_op(kernel: Kernel, args: tuple[Any, ...]) -> Callable[..., Any] | None:
    # Fill omitted defaults (e.g. eps) so infer_output_spec's strict zip lines up
    # and the baked consts carry every non-Tensor value.
    args = kernel.normalize_args(*args)
    spec = infer_output_spec(kernel, args)
    leaf_specs = spec["leaf_specs"]
    if (
        spec.get("mutated_inputs")
        or spec.get("direct_aliases")
        or not leaf_specs
        or any(s["type"] != "tensor" for s in leaf_specs)
    ):
        return None

    consts = {
        i: _const_scalar(a)
        for i, a in enumerate(args)
        if not isinstance(a, torch.Tensor)
    }
    n_args = len(args)

    def rebuild(tensors: tuple[torch.Tensor, ...]) -> tuple[Any, ...]:
        it = iter(tensors)
        return tuple(consts[i] if i in consts else next(it) for i in range(n_args))

    # Autotune the folded specialization eagerly; the impl runs it under
    # torch.compile, where autotuning can't happen, so cache the config now.
    if any(isinstance(a, (int, float)) for a in args):
        full = rebuild(_tensors(args))
        kernel.bind(full)(*full)

    n_in, n_out = n_args - len(consts), len(leaf_specs)
    in_decl = ", ".join(f"Tensor a{i}" for i in range(n_in))
    # validate=False: capturability was already checked above on the real args.
    fake, impl = _fake_and_impl(kernel, n_out, rebuild, validate=False)
    return _define_op(kernel, in_decl, n_out, fake, impl)


def _lookup_under_compile(kernel: Kernel, args: tuple[Any, ...]) -> object:
    """Pure cache lookup + call -- the only Dynamo-traceable first-call path."""
    sig = _signature(args)
    op = _op_cache.get((id(kernel), sig)) if sig is not None else None
    if op is None:
        raise RuntimeError(_MISS_MESSAGE.format(name=kernel.name))
    return op(*_tensors(args))


# --------------------------------------------------------------------------- #
# Entry points.
# --------------------------------------------------------------------------- #
def auto_capture_call(kernel: Kernel, args: tuple[Any, ...]) -> object:
    """Pallas path for ``Kernel.__call__`` (opt-in ``HELION_TPU_COMPILE_CAPTURE``).

    A decoration-registered op (zero warm-up) is preferred; otherwise the first-call
    path registers on one eager warm-up and raises on a miss under compile.
    """
    cap = getattr(kernel, "_capture_op", None)
    if torch.compiler.is_compiling():
        if cap is not None:
            return cap(args)
        return _lookup_under_compile(kernel, args)
    if cap is None:
        build_op(kernel, args)
    return RUN_NORMAL
