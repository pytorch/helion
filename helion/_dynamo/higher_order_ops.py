"""Helion kernel wrapper HOP for torch.compile integration."""
from __future__ import annotations
from typing import Any
import torch
from torch._ops import HigherOrderOperator


class HelionKernelWrapperMutation(HigherOrderOperator):
    """HOP that wraps a Helion kernel call, deferring compilation to codegen."""
    def __init__(self) -> None: super().__init__("helion_kernel_wrapper_mutation", cacheable=True)
    def __call__(self, *, kernel_idx: int, constant_args: dict[str, Any],
                 tensor_args: dict[str, Any], output_spec: dict[str, Any]) -> tuple[Any, ...]:
        return super().__call__(kernel_idx=kernel_idx, constant_args=constant_args,
                               tensor_args=tensor_args, output_spec=output_spec)


helion_kernel_wrapper_mutation = HelionKernelWrapperMutation()


def get_helion_kernel(kernel_idx: int) -> Any:
    from helion._dynamo.variables import helion_kernel_side_table
    return helion_kernel_side_table.get_kernel(kernel_idx)


def _create_output_tensors(spec: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.empty(spec.get("shape", []), dtype=spec.get("dtype", torch.float32),
                             device=spec.get("device", "cuda")) for _ in range(spec.get("num_outputs", 1)))


# Dispatch implementations

@helion_kernel_wrapper_mutation.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def helion_kernel_wrapper_mutation_dense(*, kernel_idx, constant_args, tensor_args, output_spec):
    kernel, all_args = get_helion_kernel(kernel_idx), {**constant_args, **tensor_args}
    args = [all_args[n] if n in all_args else p.default for n, p in kernel.signature.parameters.items() if n in all_args or p.default is not p.empty]
    result = kernel(*args)
    return (result,) if not isinstance(result, tuple) else result

@helion_kernel_wrapper_mutation.py_impl(torch._subclasses.FakeTensorMode)
def helion_kernel_wrapper_mutation_fake(mode, *, kernel_idx, constant_args, tensor_args, output_spec):
    with mode: return _create_output_tensors(output_spec)

@helion_kernel_wrapper_mutation.py_impl(torch._C.DispatchKey.Meta)
def helion_kernel_wrapper_mutation_meta(*, kernel_idx, constant_args, tensor_args, output_spec):
    return _create_output_tensors(output_spec)

@helion_kernel_wrapper_mutation.py_impl(torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode)
def helion_kernel_wrapper_mutation_proxy(mode, *, kernel_idx, constant_args, tensor_args, output_spec):
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, track_tensor_tree
    import torch.utils._pytree as pytree
    with disable_proxy_modes_tracing():
        out = helion_kernel_wrapper_mutation(kernel_idx=kernel_idx, constant_args=constant_args, tensor_args=tensor_args, output_spec=output_spec)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, tensor_args)
    out_proxy = mode.tracer.create_proxy("call_function", helion_kernel_wrapper_mutation, (),
        {"kernel_idx": kernel_idx, "constant_args": constant_args, "tensor_args": proxy_args, "output_spec": output_spec}, name="helion_kernel_wrapper_mutation")
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

@helion_kernel_wrapper_mutation.py_functionalize_impl
def helion_kernel_wrapper_mutation_functionalize(ctx, kernel_idx, constant_args, tensor_args, output_spec):
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(helion_kernel_wrapper_mutation(kernel_idx=kernel_idx, constant_args=constant_args,
            tensor_args=ctx.unwrap_tensors(tensor_args), output_spec=output_spec))

# Fallthrough for dispatch keys
for key in [torch._C.DispatchKey.PythonDispatcher, torch._C.DispatchKey.PythonTLSSnapshot,
            torch._C.DispatchKey.ADInplaceOrView, torch._C.DispatchKey.BackendSelect,
            torch._C.DispatchKey.AutocastCPU, torch._C.DispatchKey.AutocastCUDA,
            torch._C.DispatchKey.AutogradCUDA, torch._C.DispatchKey.AutogradCPU]:
    helion_kernel_wrapper_mutation.fallthrough(key)
