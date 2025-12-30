"""Helion kernel wrapper HOP for torch.compile integration."""

from __future__ import annotations

from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
import torch.fx.experimental.proxy_tensor
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.experimental.proxy_tensor import track_tensor_tree


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


def get_helion_kernel(kernel_idx: int) -> Any:  # noqa: ANN401
    from helion._compiler._dynamo.variables import helion_kernel_side_table

    return helion_kernel_side_table.get_kernel(kernel_idx)


def _create_output_tensors(
    spec: dict[str, Any],
) -> tuple[torch.Tensor | int | float | None, ...]:
    """Create output tensors/values from the output spec.

    The spec contains:
    - num_outputs: number of outputs
    - output_specs: list of per-output specs, where each spec is either:
      - {"shape": [...], "dtype": ..., "device": ...} for tensor outputs
      - {"scalar_value": value} for constant scalar outputs
      - None for unknown non-tensor outputs
    """
    num_outputs = spec["num_outputs"]
    output_specs = spec["output_specs"]

    results: list[torch.Tensor | int | float | None] = []
    for i in range(num_outputs):
        if i < len(output_specs) and output_specs[i] is not None:
            out_spec = output_specs[i]
            if "scalar_value" in out_spec:
                # Scalar constant value
                results.append(out_spec["scalar_value"])
            else:
                # Tensor output
                results.append(
                    torch.empty(
                        out_spec["shape"],
                        dtype=out_spec["dtype"],
                        device=out_spec["device"],
                    )
                )
        else:
            # Unknown non-tensor output
            results.append(None)
    return tuple(results)


# Dispatch implementations


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


@helion_kernel_wrapper_mutation.py_impl(torch._subclasses.FakeTensorMode)
def helion_kernel_wrapper_mutation_fake(  # noqa: ANN202
    mode: Any,  # noqa: ANN401
    *,
    kernel_idx: Any,  # noqa: ANN401
    constant_args: Any,  # noqa: ANN401
    tensor_args: Any,  # noqa: ANN401
    output_spec: Any,  # noqa: ANN401
):
    with mode:
        return _create_output_tensors(output_spec)


@helion_kernel_wrapper_mutation.py_impl(torch._C.DispatchKey.Meta)
def helion_kernel_wrapper_mutation_meta(  # noqa: ANN202
    *,
    kernel_idx: Any,  # noqa: ANN401
    constant_args: Any,  # noqa: ANN401
    tensor_args: Any,  # noqa: ANN401
    output_spec: Any,  # noqa: ANN401
):
    return _create_output_tensors(output_spec)


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
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(
            helion_kernel_wrapper_mutation(
                kernel_idx=kernel_idx,
                constant_args=constant_args,
                tensor_args=ctx.unwrap_tensors(tensor_args),
                output_spec=output_spec,
            )
        )


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
