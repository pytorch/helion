"""Make a Helion Pallas/TPU kernel capturable by ``torch.compile(backend="tpu")``.

A ``@helion.kernel`` call on Pallas goes through the eager torch_tpu -> jax
bridge, paying a per-call dispatch cost; under ``torch.compile`` Dynamo would
trace *into* that bridge (into jax) and break. Wrapping the kernel as a
``torch.library.custom_op`` makes it opaque to Dynamo and lets torch_xla capture
it as a single XLA-graph node, collapsing the dispatch cost.

``tpu_compile_capture`` derives the op schema and fake (meta) impl from the example
inputs using Helion's jax-free shape inference, so the user writes no
boilerplate. The op is specialized to the example shapes/dtypes -- call it once
per (kernel, shape, dtype); a new signature registers a fresh op.
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
_op_cache: dict[Any, Callable[..., Any]] = {}


def tpu_compile_capture(
    kernel: Kernel, example_inputs: tuple[Any, ...]
) -> Callable[..., Any]:
    """Wrap ``kernel`` so ``torch.compile(backend="tpu")`` captures it as one node.

    Args:
        kernel: a ``@helion.kernel`` (Pallas/TPU backend).
        example_inputs: representative positional args (all Tensors, no scalars).
            Used once (jax-free) to derive output shapes/dtypes; the op is
            specialized to them. Output must be a Tensor or tuple of Tensors,
            with no input mutation/aliasing.
    """

    if kernel.settings.backend != "pallas":
        raise NotImplementedError(
            f"tpu_compile_capture targets the Pallas/TPU backend; {kernel.name!r} "
            f"uses backend {kernel.settings.backend!r}. On other backends Helion "
            "kernels are already captured by torch.compile via its HOP lowering "
            "(see the torch_compile_fusion setting); capture is unneeded there."
        )
    from .._compiler._dynamo.variables import infer_output_spec

    global _op_counter
    args = tuple(example_inputs)
    if not all(isinstance(a, torch.Tensor) for a in args):
        raise NotImplementedError(
            "tpu_compile_capture supports all-Tensor positional args (no scalars)"
        )

    cache_key = (
        id(kernel),
        tuple((tuple(a.shape), a.dtype, str(a.device)) for a in args),
    )
    cached = _op_cache.get(cache_key)
    if cached is not None:
        return cached

    spec = infer_output_spec(kernel, args)
    if spec.get("mutated_inputs") or spec.get("direct_aliases"):
        raise NotImplementedError(
            "tpu_compile_capture does not support kernels that mutate or alias inputs"
        )
    leaf_specs = spec["leaf_specs"]
    if not leaf_specs or any(s["type"] != "tensor" for s in leaf_specs):
        raise NotImplementedError(
            "tpu_compile_capture supports Tensor or tuple-of-Tensor outputs"
        )

    n_in, n_out = len(args), len(leaf_specs)
    _op_counter += 1
    name = f"{kernel.name}_{_op_counter}"
    in_decl = ", ".join(f"Tensor a{i}" for i in range(n_in))
    out_decl = "Tensor" if n_out == 1 else "(" + ", ".join(["Tensor"] * n_out) + ")"
    _capture_lib.define(f"{name}({in_decl}) -> {out_decl}")

    def impl(*tensors: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        out = kernel(*tensors)
        # A multi-tensor return is boxed as a tuple by the schema; normalize a
        # list return so eager and captured paths agree.
        return tuple(out) if n_out > 1 else out

    _capture_lib.impl(name, impl, "CompositeExplicitAutograd")

    def fake(*tensors: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        outs = [
            torch.empty(s["shape"], dtype=s["dtype"], device=s["device"])
            for s in leaf_specs
        ]
        return outs[0] if n_out == 1 else tuple(outs)

    torch.library.register_fake(f"helion_capture::{name}", fake)

    op = getattr(torch.ops.helion_capture, name)
    _op_cache[cache_key] = op
    return op
