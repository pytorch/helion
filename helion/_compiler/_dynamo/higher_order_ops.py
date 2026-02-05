from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._higher_order_ops import effects as hop_effects
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch._higher_order_ops.utils import register_fake
from torch._library.effects import EffectType
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.experimental.proxy_tensor import track_tensor_tree
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

    from helion.runtime.kernel import Kernel


class HelionKernelWrapperMutation(HigherOrderOperator):
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


def get_helion_kernel(kernel_idx: int) -> Kernel:
    return cast("Kernel", kernel_side_table.get_kernel(kernel_idx))


@helion_kernel_wrapper_mutation.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def helion_kernel_wrapper_mutation_dense(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    kernel = get_helion_kernel(kernel_idx)
    all_args = {**constant_args, **tensor_args}
    args = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    result = kernel(*args)
    return result if isinstance(result, tuple) else (result,)


@register_fake(helion_kernel_wrapper_mutation)
def helion_kernel_wrapper_mutation_fake(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | int | float | None, ...]:
    """Create fake output tensors from spec."""
    num_outputs = cast("int", output_spec.get("num_outputs", 0))
    specs = cast("list[dict[str, object] | None]", output_spec.get("output_specs", []))

    if num_outputs <= 0:
        return (None,)

    results: list[torch.Tensor | int | float | None] = []
    for i in range(num_outputs):
        spec = specs[i] if i < len(specs) else None
        if spec is None or "shape" not in spec:
            # Scalar output
            results.append(spec.get("scalar_value") if spec else None)
        else:
            results.append(
                torch.empty_strided(
                    spec["shape"],  # pyrefly: ignore[bad-argument-type]
                    spec["stride"],  # pyrefly: ignore[bad-argument-type]
                    dtype=spec["dtype"],  # type: ignore[arg-type]  # pyrefly: ignore[bad-argument-type]
                    device=spec["device"],  # type: ignore[arg-type]
                )
            )
    return tuple(results)


@helion_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def helion_kernel_wrapper_mutation_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    with disable_proxy_modes_tracing():
        out = helion_kernel_wrapper_mutation(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,  # pyrefly: ignore[bad-argument-type]
            output_spec=output_spec,
        )
    # pyrefly: ignore[missing-attribute]
    proxy_tensor_args = pytree.tree_map(mode.tracer.unwrap_proxy, tensor_args)
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        helion_kernel_wrapper_mutation,
        (),
        {
            "kernel_idx": kernel_idx,
            "constant_args": constant_args,
            "tensor_args": proxy_tensor_args,
            "output_spec": output_spec,
        },
        name=helion_kernel_wrapper_mutation._name,
    )
    return track_tensor_tree(
        out, out_proxy, constant=None, tracer=mode.tracer
    )  # pyrefly: ignore[bad-return]


@helion_kernel_wrapper_mutation.py_functionalize_impl
def helion_kernel_wrapper_mutation_functionalize(
    ctx: BaseFunctionalizeAPI,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    if output_spec.get("mutated_inputs"):
        raise NotImplementedError(
            "Helion kernels that mutate inputs are not yet supported with "
            "torch.compile fusion. Use allow_fusion=False for mutating kernels."
        )
    unwrapped = ctx.unwrap_tensors(tensor_args)  # pyrefly: ignore[bad-argument-type]
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(
            helion_kernel_wrapper_mutation(
                kernel_idx=kernel_idx,
                constant_args=constant_args,
                tensor_args=unwrapped,
                output_spec=output_spec,
            )
        )


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
