"""Helion kernel wrapper HOP for torch.compile integration."""

from __future__ import annotations

from typing import Any

import torch
from torch._higher_order_ops import effects as hop_effects
from torch._higher_order_ops.utils import register_fake
from torch._library.effects import EffectType
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
import torch.fx.experimental.proxy_tensor
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.experimental.proxy_tensor import track_tensor_tree
import torch.utils._pytree as pytree


class HelionKernelWrapperMutation(HigherOrderOperator):
    """HOP that wraps a Helion kernel call, deferring compilation to codegen."""

    def __init__(self) -> None:
        super().__init__("helion_kernel_wrapper_mutation", cacheable=True)

    def __call__(
        self,
        *,
        kernel_idx: int,
        constant_args: dict[str, Any],
        tensor_args: dict[str, Any],
        output_spec: dict[str, Any],
    ) -> tuple[Any, ...]:
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,
            output_spec=output_spec,
        )


helion_kernel_wrapper_mutation = HelionKernelWrapperMutation()
hop_effects._register_effectful_op(helion_kernel_wrapper_mutation, EffectType.ORDERED)


def get_helion_kernel(kernel_idx: int) -> Any:  # noqa: ANN401
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    return kernel_side_table.get_kernel(kernel_idx)


@helion_kernel_wrapper_mutation.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def helion_kernel_wrapper_mutation_dense(  # noqa: ANN202
    *,
    kernel_idx: Any,  # noqa: ANN401
    constant_args: Any,  # noqa: ANN401
    tensor_args: Any,  # noqa: ANN401
    output_spec: Any,  # noqa: ANN401
):
    kernel, all_args = get_helion_kernel(kernel_idx), {**constant_args, **tensor_args}
    args = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    result = kernel(*args)
    return (result,) if not isinstance(result, tuple) else result


@register_fake(helion_kernel_wrapper_mutation)
def helion_kernel_wrapper_mutation_fake(  # noqa: ANN202
    *,
    kernel_idx: Any,  # noqa: ANN401
    constant_args: Any,  # noqa: ANN401
    tensor_args: Any,  # noqa: ANN401
    output_spec: Any,  # noqa: ANN401
):
    # Create output tensors/scalars from spec
    results = []
    for spec in output_spec["output_specs"]:
        if "scalar_value" in spec:
            results.append(spec["scalar_value"])
        else:
            results.append(
                torch.empty(spec["shape"], dtype=spec["dtype"], device=spec["device"])
            )
    return tuple(results)


@helion_kernel_wrapper_mutation.py_impl(
    torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
)
def helion_kernel_wrapper_mutation_proxy(  # noqa: ANN202
    mode: Any,  # noqa: ANN401
    *,
    kernel_idx: Any,  # noqa: ANN401
    constant_args: Any,  # noqa: ANN401
    tensor_args: Any,  # noqa: ANN401
    output_spec: Any,  # noqa: ANN401
):
    with disable_proxy_modes_tracing():
        out = helion_kernel_wrapper_mutation(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,
            output_spec=output_spec,
        )
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, tensor_args)
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        helion_kernel_wrapper_mutation,
        (),
        {
            "kernel_idx": kernel_idx,
            "constant_args": constant_args,
            "tensor_args": proxy_args,
            "output_spec": output_spec,
        },
        name="helion_kernel_wrapper_mutation",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@helion_kernel_wrapper_mutation.py_functionalize_impl
def helion_kernel_wrapper_mutation_functionalize(  # noqa: ANN202
    ctx: Any,  # noqa: ANN401
    kernel_idx: Any,  # noqa: ANN401
    constant_args: Any,  # noqa: ANN401
    tensor_args: Any,  # noqa: ANN401
    output_spec: Any,  # noqa: ANN401
):
    unwrapped_tensor_args = ctx.unwrap_tensors(tensor_args)
    mutated = [
        name
        for name in output_spec.get("mutated_inputs", [])
        if name in unwrapped_tensor_args
    ]
    cloned_tensor_args = dict(unwrapped_tensor_args)
    for name in mutated:
        val = cloned_tensor_args.get(name)
        if isinstance(val, torch.Tensor):
            cloned_tensor_args[name] = clone_preserve_strides(val)

    with ctx.redispatch_to_next():
        outputs = helion_kernel_wrapper_mutation(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=cloned_tensor_args,
            output_spec=output_spec,
        )

    for name in mutated:
        new_arg = cloned_tensor_args.get(name)
        if not isinstance(new_arg, torch.Tensor):
            continue
        orig_arg = tensor_args[name]
        ctx.replace(orig_arg, new_arg)
        ctx.mark_mutation_hidden_from_autograd(orig_arg)
        ctx.commit_update(orig_arg)
        ctx.sync(orig_arg)

    return ctx.wrap_tensors(outputs)


# Fallthrough for dispatch keys
for key in [
    torch._C.DispatchKey.PythonDispatcher,
    torch._C.DispatchKey.PythonTLSSnapshot,
    torch._C.DispatchKey.ADInplaceOrView,
    torch._C.DispatchKey.BackendSelect,
    torch._C.DispatchKey.AutocastCPU,
    torch._C.DispatchKey.AutocastCUDA,
    torch._C.DispatchKey.AutogradCUDA,
    torch._C.DispatchKey.AutogradCPU,
]:
    helion_kernel_wrapper_mutation.fallthrough(key)
