"""Make a Helion Pallas/TPU kernel capturable by ``torch.compile(backend="tpu")``.

Wrapping a Helion kernel as a ``torch.library.custom_op`` makes it opaque to
Dynamo so torch_xla captures it as one XLA-graph node, instead of tracing into
the eager torch_tpu -> jax bridge (which breaks). Schema + jax-free fake come
from Helion's output-spec inference. The op is cached per (kernel, shape, dtype).

Two entry points:
  * ``tpu_compile_capture(kernel, example_inputs)`` -- explicit one-line wrap; robust
    (the user calls the returned op directly).
  * ``auto_capture_call`` -- called from ``Kernel.__call__`` to register the op as
    a side effect of an eager warm-up (opt-in ``HELION_TPU_COMPILE_CAPTURE=1``), so
    no wrap is needed. Static shapes only; registration must be single-threaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import torch

if TYPE_CHECKING:
    from .kernel import Kernel

_capture_lib = torch.library.Library("helion_capture", "FRAGMENT")
_op_counter = 0
# {(id(kernel), shapes, dtypes) -> op or None}. A flat dict keyed by id() keeps
# the under-compile lookup Dynamo-traceable (a WeakIdKeyDictionary thrashes it);
# id() reuse is moot for module-scope kernels, which live for the process.
_op_cache: dict[Any, Callable[..., Any] | None] = {}

# Sentinel: tells Kernel.__call__ to run its normal dispatch path.
RUN_NORMAL = object()


def _signature(args: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Cache key for an all-Tensor call, or None if any arg isn't a Tensor."""
    if not all(isinstance(a, torch.Tensor) for a in args):
        return None
    return tuple((tuple(a.shape), a.dtype) for a in args)


def build_op(kernel: Kernel, args: tuple[Any, ...]) -> Callable[..., Any] | None:
    """Register (once, cached) a custom_op for this kernel+signature.

    Returns the op, or ``None`` if the kernel/signature isn't capturable
    (non-tensor args/outputs, input mutation/aliasing, or any registration
    failure). Never raises -- callers fall back to normal dispatch.
    """
    sig = _signature(args)
    if sig is None:
        return None
    key = (id(kernel), sig)
    if key in _op_cache:
        return _op_cache[key]
    try:
        op = _register_op(kernel, args)
    except Exception:
        op = None
    _op_cache[key] = op
    return op


def _register_op(kernel: Kernel, args: tuple[Any, ...]) -> Callable[..., Any] | None:
    from .._compiler._dynamo.variables import infer_output_spec

    spec = infer_output_spec(kernel, args)
    leaf_specs = spec["leaf_specs"]
    if (
        spec.get("mutated_inputs")
        or spec.get("direct_aliases")
        or not leaf_specs
        or any(s["type"] != "tensor" for s in leaf_specs)
    ):
        return None

    global _op_counter
    _op_counter += 1
    name = f"{kernel.name}_{_op_counter}"
    n_in, n_out = len(args), len(leaf_specs)
    in_decl = ", ".join(f"Tensor a{i}" for i in range(n_in))
    out_decl = "Tensor" if n_out == 1 else "(" + ", ".join(["Tensor"] * n_out) + ")"
    _capture_lib.define(f"{name}({in_decl}) -> {out_decl}")

    # Register the fake before the impl so a fake failure can't leave a live
    # impl with no meta. Captured outputs are contiguous (torch.empty).
    def fake(*tensors: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        outs = [
            torch.empty(s["shape"], dtype=s["dtype"], device=s["device"])
            for s in leaf_specs
        ]
        return outs[0] if n_out == 1 else tuple(outs)

    torch.library.register_fake(f"helion_capture::{name}", fake)

    def impl(*tensors: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # bind() directly, not kernel(*tensors): the latter re-enters the
        # Kernel.__call__ capture hook and recurses into this op.
        out = kernel.bind(tensors)(*tensors)
        return tuple(out) if n_out > 1 else out

    _capture_lib.impl(name, impl, "CompositeExplicitAutograd")
    return getattr(torch.ops.helion_capture, name)


def tpu_compile_capture(
    kernel: Kernel, example_inputs: tuple[Any, ...]
) -> Callable[..., Any]:
    """Wrap ``kernel`` so ``torch.compile(backend="tpu")`` captures it as one node.

    Args:
        kernel: a ``@helion.kernel`` (Pallas/TPU backend).
        example_inputs: representative positional args (all Tensors, no scalars);
            used once (jax-free) to derive output shapes/dtypes. Output must be a
            Tensor or tuple of Tensors, with no input mutation/aliasing.
    """

    if kernel.settings.backend != "pallas":
        raise NotImplementedError(
            f"tpu_compile_capture targets the Pallas/TPU backend; {kernel.name!r} "
            f"uses backend {kernel.settings.backend!r}. On other backends Helion "
            "kernels are already captured by torch.compile via its HOP lowering "
            "(see the torch_compile_fusion setting); capture is unneeded there."
        )
    op = build_op(kernel, tuple(example_inputs))
    if op is None:
        raise NotImplementedError(
            "tpu_compile_capture supports all-Tensor inputs and Tensor/tuple outputs "
            "with no input mutation or aliasing"
        )
    return op


def auto_capture_call(kernel: Kernel, args: tuple[Any, ...]) -> object:
    """Pallas path for ``Kernel.__call__``.

    Eager: register the op as a side effect, then run the normal path. Under
    compile: a pure cache lookup (fullgraph-safe) returns the captured op; a miss
    raises a clear error rather than silently falling into the jax bridge (which
    Dynamo can't trace). Misses mean the kernel wasn't warmed up at this shape,
    has dynamic shapes, or isn't capturable.
    """
    if not torch.compiler.is_compiling():
        build_op(kernel, args)
        return RUN_NORMAL
    sig = _signature(args)
    op = _op_cache.get((id(kernel), sig)) if sig is not None else None
    if op is not None:
        return op(*args)
    raise RuntimeError(
        f"Helion kernel {kernel.name!r} could not be captured under torch.compile "
        "on the Pallas backend. Run it once eagerly at this shape before compiling, "
        "or use helion.tpu_compile_capture(kernel, example_inputs). Kernels with dynamic "
        "shapes, non-tensor args, or input mutation are not auto-capturable."
    )
