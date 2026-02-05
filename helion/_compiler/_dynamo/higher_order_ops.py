from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
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


class HelionKernelWrapperFunctional(HigherOrderOperator):
    """Functional HOP for mutation: clones inputs, returns (outputs, cloned_tensors)."""

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
    return cast("Kernel", kernel_side_table.get_kernel(kernel_idx))


def _clone_tensors(
    tensor_args: dict[str, torch.Tensor], tensors_to_clone: list[str]
) -> dict[str, torch.Tensor]:
    return {
        name: tensor_args[name].clone()  # pyrefly: ignore[unsupported-operation]
        for name in tensors_to_clone
        if name in tensor_args and isinstance(tensor_args[name], torch.Tensor)
    }


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
    """Create fake output tensors/scalars from spec."""
    specs = cast("list[dict[str, object]]", output_spec.get("output_specs", []))
    if cast("int", output_spec.get("num_outputs", len(specs))) <= 0:
        return (None,)

    def make_output(s: dict[str, object]) -> torch.Tensor | int | float | None:
        if "shape" not in s:
            return cast("int | float | None", s.get("scalar_value"))
        return torch.empty_strided(
            s["shape"],  # pyrefly: ignore[bad-argument-type]
            s["stride"],  # pyrefly: ignore[bad-argument-type]
            dtype=s["dtype"],  # type: ignore[arg-type]  # pyrefly: ignore[bad-argument-type]
            device=s["device"],  # type: ignore[arg-type]
        )

    return tuple(make_output(s) for s in specs)


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
    unwrapped = ctx.unwrap_tensors(tensor_args)  # pyrefly: ignore[bad-argument-type]
    mutated_inputs = cast("list[str]", output_spec.get("mutated_inputs", []))
    with ctx.redispatch_to_next():
        kernel_outputs, cloned_tensors = helion_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=unwrapped,
            output_spec=output_spec,
            tensors_to_clone=list(mutated_inputs),
        )
    for key, cloned in cloned_tensors.items():
        if isinstance(cloned, torch.Tensor) and isinstance(
            tensor_args.get(key), torch.Tensor
        ):
            ctx.replace(tensor_args[key], cloned)
            ctx.mark_mutation_hidden_from_autograd(tensor_args[key])
            ctx.commit_update(tensor_args[key])
            ctx.sync(tensor_args[key])
    return ctx.wrap_tensors(kernel_outputs)


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
    cloned = _clone_tensors(tensor_args, tensors_to_clone)
    kernel_outputs = helion_kernel_wrapper_mutation(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args={
            k: cloned.get(k, v) for k, v in tensor_args.items()
        },  # pyrefly: ignore[bad-argument-type]
        output_spec=output_spec,
    )
    return (kernel_outputs, cloned)


@register_fake(helion_kernel_wrapper_functional)
def helion_kernel_wrapper_functional_fake(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    return (
        helion_kernel_wrapper_mutation_fake(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,
            output_spec=output_spec,
        ),
        _clone_tensors(tensor_args, tensors_to_clone),
    )


@helion_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def helion_kernel_wrapper_functional_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    with disable_proxy_modes_tracing():
        out = helion_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,  # pyrefly: ignore[bad-argument-type]
            output_spec=output_spec,
            tensors_to_clone=tensors_to_clone,
        )
    unwrap = mode.tracer.unwrap_proxy  # pyrefly: ignore[missing-attribute]
    proxy_kwargs = {
        "kernel_idx": kernel_idx,
        "constant_args": constant_args,
        "tensor_args": pytree.tree_map(unwrap, tensor_args),
        "output_spec": output_spec,
        "tensors_to_clone": tensors_to_clone,
    }
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        helion_kernel_wrapper_functional,
        (),
        proxy_kwargs,
        name=helion_kernel_wrapper_functional._name,
    )
    return track_tensor_tree(
        out, out_proxy, constant=None, tracer=mode.tracer
    )  # pyrefly: ignore[bad-return]


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
