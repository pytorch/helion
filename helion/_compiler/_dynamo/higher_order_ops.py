from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch._higher_order_ops import effects as hop_effects
from torch._higher_order_ops.utils import register_fake
from torch._library.effects import EffectType
from torch._ops import HigherOrderOperator
import torch.fx.experimental.proxy_tensor
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.experimental.proxy_tensor import track_tensor_tree
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

    from helion.runtime.kernel import Kernel


class HelionKernelWrapperMutation(HigherOrderOperator):
    """HOP that wraps a Helion kernel call, deferring compilation to codegen."""

    def __init__(self) -> None:
        super().__init__("helion_kernel_wrapper_mutation", cacheable=True)

    def __call__(
        self,
        *,
        kernel_idx: int,
        constant_args: dict[str, object],
        tensor_args: dict[str, object],
        output_spec: dict[str, object],
    ) -> tuple[object, ...]:
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,
            output_spec=output_spec,
        )


helion_kernel_wrapper_mutation = HelionKernelWrapperMutation()
hop_effects._register_effectful_op(helion_kernel_wrapper_mutation, EffectType.ORDERED)


class HelionKernelWrapperFunctional(HigherOrderOperator):
    """Functional version of Helion kernel wrapper.

    This HOP takes a tensors_to_clone parameter, clones specified inputs
    before mutation, and returns both the kernel outputs and the cloned
    tensors (for functionalization to track mutations).

    Returns:
        tuple of (kernel_outputs: tuple, cloned_tensors: dict[str, Tensor])
    """

    def __init__(self) -> None:
        super().__init__("helion_kernel_wrapper_functional", cacheable=True)

    def __call__(
        self,
        *,
        kernel_idx: int,
        constant_args: dict[str, object],
        tensor_args: dict[str, object],
        output_spec: dict[str, object],
        tensors_to_clone: list[str],
    ) -> tuple[tuple[object, ...], dict[str, torch.Tensor]]:
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,
            output_spec=output_spec,
            tensors_to_clone=tensors_to_clone,
        )


helion_kernel_wrapper_functional = HelionKernelWrapperFunctional()


def get_helion_kernel(kernel_idx: int) -> Kernel:
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    return cast("Kernel", kernel_side_table.get_kernel(kernel_idx))


def _clone_tensors(
    tensor_args: dict[str, torch.Tensor], tensors_to_clone: list[str]
) -> dict[str, torch.Tensor]:
    """Clone specified tensors from tensor_args."""
    cloned: dict[str, torch.Tensor] = {}
    for name in tensors_to_clone:
        tensor = tensor_args.get(name)
        if isinstance(tensor, torch.Tensor):
            # pyrefly: ignore [unsupported-operation]
            cloned[name] = tensor.clone()
    return cloned


# =============================================================================
# Mutation HOP dispatch implementations
# =============================================================================


@helion_kernel_wrapper_mutation.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def helion_kernel_wrapper_mutation_dense(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    kernel, all_args = get_helion_kernel(kernel_idx), {**constant_args, **tensor_args}
    args = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    result = kernel(*args)
    return (result,) if not isinstance(result, tuple) else result


@register_fake(helion_kernel_wrapper_mutation)
def helion_kernel_wrapper_mutation_fake(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    # Create output tensors/scalars from spec
    results: list[torch.Tensor | object] = []
    for spec in cast(
        "list[dict[str, object] | None]", output_spec.get("output_specs", [])
    ):
        if spec is None:
            results.append(None)
        elif "scalar_value" in spec:
            # Return None for scalars - they are handled at Dynamo level as constants.
            # The actual scalar value is returned directly from _build_return_value
            # as a ConstantVariable, so the HOP output is never used for scalars.
            results.append(None)
        else:
            # Use empty_strided to preserve stride from output_spec
            # This is important for non-contiguous outputs (e.g., transposed tensors)
            stride = spec.get("stride")
            shape = cast("list[int]", spec["shape"])
            dtype = cast("torch.dtype", spec["dtype"])
            device = cast("str", spec["device"])
            if stride:
                results.append(
                    torch.empty_strided(
                        shape,
                        cast("list[int]", stride),
                        dtype=dtype,
                        device=device,
                    )
                )
            else:
                results.append(torch.empty(shape, dtype=dtype, device=device))
    return tuple(results)


def _trace_helion_hop(
    mode: ProxyTorchDispatchMode,
    hop: HigherOrderOperator,
    node_kwargs: dict[str, object],
) -> object:
    """Trace a Helion HOP call through proxy mode."""
    with disable_proxy_modes_tracing():
        out = hop(**node_kwargs)  # pyrefly: ignore[bad-argument-type]
    proxy_kwargs = {
        # pyrefly: ignore[missing-attribute]
        k: pytree.tree_map(mode.tracer.unwrap_proxy, v) if k == "tensor_args" else v
        for k, v in node_kwargs.items()
    }
    out_proxy = mode.tracer.create_proxy(
        "call_function", hop, (), proxy_kwargs, name=hop._name
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@helion_kernel_wrapper_mutation.py_impl(
    torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
)
def helion_kernel_wrapper_mutation_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    return _trace_helion_hop(  # pyrefly: ignore[bad-return]
        mode,
        helion_kernel_wrapper_mutation,
        {
            "kernel_idx": kernel_idx,
            "constant_args": constant_args,
            "tensor_args": tensor_args,
            "output_spec": output_spec,
        },
    )


@helion_kernel_wrapper_mutation.py_functionalize_impl
def helion_kernel_wrapper_mutation_functionalize(
    ctx: BaseFunctionalizeAPI,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    """Convert mutation HOP to functional HOP during functionalization.

    1. Identify mutated inputs from output_spec
    2. Call the functional HOP which clones those inputs and runs the kernel
    3. Use ctx.replace() to propagate mutations back through functionalization
    """
    # pyrefly: ignore[bad-argument-type]
    unwrapped_tensor_args = ctx.unwrap_tensors(tensor_args)

    # Get mutated inputs from output_spec (already computed at Dynamo level)
    mutated_inputs = cast("list[str]", output_spec.get("mutated_inputs", []))
    tensors_to_clone = list(mutated_inputs)

    with ctx.redispatch_to_next():
        # Call functional HOP which clones inputs, runs kernel, returns both
        kernel_outputs, cloned_tensors = helion_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=unwrapped_tensor_args,
            output_spec=output_spec,
            tensors_to_clone=tensors_to_clone,
        )

    # Propagate mutations back through functionalization context
    for key, cloned_tensor in cloned_tensors.items():
        if not isinstance(cloned_tensor, torch.Tensor):
            continue
        input_tensor = tensor_args.get(key)
        if not isinstance(input_tensor, torch.Tensor):
            continue

        ctx.replace(input_tensor, cloned_tensor)
        ctx.mark_mutation_hidden_from_autograd(input_tensor)
        ctx.commit_update(input_tensor)
        ctx.sync(input_tensor)

    return ctx.wrap_tensors(kernel_outputs)


# =============================================================================
# Functional HOP dispatch implementations
# =============================================================================


@helion_kernel_wrapper_functional.py_impl(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
def helion_kernel_wrapper_functional_dense(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    """Clone specified inputs, call mutation HOP, return kernel outputs and cloned tensors."""
    cloned_tensors = _clone_tensors(tensor_args, tensors_to_clone)
    cloned_tensor_args = {
        key: cloned_tensors.get(key, val) for key, val in tensor_args.items()
    }

    kernel_outputs = helion_kernel_wrapper_mutation(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args=cloned_tensor_args,  # pyrefly: ignore[bad-argument-type]
        output_spec=output_spec,
    )

    return (kernel_outputs, cloned_tensors)


@register_fake(helion_kernel_wrapper_functional)
def helion_kernel_wrapper_functional_fake(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    """Create fake outputs and cloned fake tensors."""
    kernel_outputs = helion_kernel_wrapper_mutation_fake(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args=tensor_args,
        output_spec=output_spec,
    )
    return (kernel_outputs, _clone_tensors(tensor_args, tensors_to_clone))


@helion_kernel_wrapper_functional.py_impl(
    torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
)
def helion_kernel_wrapper_functional_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    """Trace the functional HOP call."""
    return _trace_helion_hop(  # pyrefly: ignore[bad-return]
        mode,
        helion_kernel_wrapper_functional,
        {
            "kernel_idx": kernel_idx,
            "constant_args": constant_args,
            "tensor_args": tensor_args,
            "output_spec": output_spec,
            "tensors_to_clone": tensors_to_clone,
        },
    )


@helion_kernel_wrapper_functional.py_functionalize_impl
def helion_kernel_wrapper_functional_functionalize(
    ctx: BaseFunctionalizeAPI,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    """Simple pass-through for functional HOP - just wrap/unwrap tensors."""
    # pyrefly: ignore[bad-argument-type]
    unwrapped_tensor_args = ctx.unwrap_tensors(tensor_args)
    with ctx.redispatch_to_next():
        kernel_outputs, cloned_tensors = helion_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=unwrapped_tensor_args,
            output_spec=output_spec,
            tensors_to_clone=tensors_to_clone,
        )
    wrapped_outputs = ctx.wrap_tensors(kernel_outputs)
    # pyrefly: ignore[bad-argument-type]
    wrapped_cloned = ctx.wrap_tensors(cloned_tensors)
    return (wrapped_outputs, wrapped_cloned)  # pyrefly: ignore[bad-return]


# =============================================================================
# Fallthrough for dispatch keys (both HOPs)
# =============================================================================
_FALLTHROUGH_KEYS = [
    torch._C.DispatchKey.PythonDispatcher,
    torch._C.DispatchKey.PythonTLSSnapshot,
    torch._C.DispatchKey.ADInplaceOrView,
    torch._C.DispatchKey.BackendSelect,
    torch._C.DispatchKey.AutocastCPU,
    torch._C.DispatchKey.AutocastCUDA,
    torch._C.DispatchKey.AutogradCUDA,
    torch._C.DispatchKey.AutogradCPU,
]
for key in _FALLTHROUGH_KEYS:
    helion_kernel_wrapper_mutation.fallthrough(key)
    helion_kernel_wrapper_functional.fallthrough(key)
