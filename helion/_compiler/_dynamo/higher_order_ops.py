from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import cast

import torch
from torch._higher_order_ops import effects as hop_effects
from torch._higher_order_ops.utils import register_fake
from torch._library.effects import EffectType
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.experimental.proxy_tensor import get_proxy_slot
from torch.fx.experimental.proxy_tensor import track_tensor_tree
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

    from helion.runtime.kernel import Kernel


class HelionKernelSideTable:
    id_to_kernel: ClassVar[dict[int, Kernel]] = {}
    kernel_to_id: ClassVar[dict[Kernel, int]] = {}
    lock: ClassVar[threading.Lock] = threading.Lock()

    # Returns index on the table
    def add_kernel(self, kernel: Kernel) -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]

            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    # Returns the helion kernel at the given index
    def get_kernel(self, idx: int) -> Kernel:
        # No need to lock here as fetching from dict is atomic
        if idx not in self.id_to_kernel:
            raise AssertionError(f"Kernel index {idx} not found in id_to_kernel")
        return self.id_to_kernel[idx]

    # Resets the table (only meant to be used in unit tests)
    # This is only safe assuming single threaded execution
    def reset_table(self) -> None:
        self.id_to_kernel.clear()
        self.kernel_to_id.clear()


helion_kernel_side_table = HelionKernelSideTable()


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
    return helion_kernel_side_table.get_kernel(kernel_idx)


def _clone_tensors(
    tensor_args: dict[str, torch.Tensor], tensors_to_clone: list[str]
) -> dict[str, torch.Tensor]:
    return {
        name: tensor_args[name].clone()  # pyrefly: ignore[unsupported-operation]
        for name in tensors_to_clone
        if name in tensor_args
    }


def _rebuild_container_args(all_args: dict[str, object]) -> None:
    """Rebuild container params (tuple/list/dict) from flattened keys using pytree."""
    container_specs = all_args.pop("__container_specs", None)
    if not isinstance(container_specs, dict):
        return
    for name, spec_str in container_specs.items():
        spec = pytree.treespec_loads(spec_str)
        elements = [all_args.pop(f"{name}.{i}") for i in range(spec.num_leaves)]
        all_args[name] = pytree.tree_unflatten(elements, spec)


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
    _rebuild_container_args(all_args)
    args = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    result = kernel(*args)
    flat_leaves, _ = pytree.tree_flatten(result)
    return tuple(leaf for leaf in flat_leaves if isinstance(leaf, torch.Tensor))


@register_fake(helion_kernel_wrapper_mutation)
def helion_kernel_wrapper_mutation_fake(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor, ...]:
    """Create fake output tensors from spec."""
    specs = cast("list[dict[str, object]]", output_spec["leaf_specs"])
    result = []
    for spec in specs:
        if spec["type"] == "tensor":
            assert all(key in spec for key in ("shape", "dtype", "device")), (
                f"output_spec missing required keys: {spec}"
            )
            result.append(
                torch.empty(  # pyrefly: ignore[no-matching-overload]
                    spec["shape"],  # pyrefly: ignore[bad-argument-type]
                    dtype=spec["dtype"],  # type: ignore[arg-type]  # pyrefly: ignore[bad-argument-type]
                    device=spec["device"],  # type: ignore[arg-type]
                )
            )
    return tuple(result)


def _trace_hop_proxy(
    hop: HigherOrderOperator,
    mode: ProxyTorchDispatchMode,
    kwargs: dict[str, object],
) -> object:
    """Shared proxy tracing logic for mutation and functional HOPs."""
    with disable_proxy_modes_tracing():
        out = hop(**kwargs)

    _py_sym_types = (torch.SymInt, torch.SymFloat, torch.SymBool)

    def _unwrap_syms(val: object) -> object:
        if isinstance(val, _py_sym_types):
            return get_proxy_slot(  # pyrefly: ignore[no-matching-overload]
                val, mode.tracer, transform=lambda e: e.force()
            )
        return val

    proxy_kwargs = {}
    for k, v in kwargs.items():
        if k == "tensor_args":
            proxy_kwargs[k] = pytree.tree_map(
                mode.tracer.unwrap_proxy,  # pyrefly: ignore[missing-attribute]
                v,
            )
        elif k == "output_spec":
            proxy_kwargs[k] = pytree.tree_map(_unwrap_syms, v)
        else:
            proxy_kwargs[k] = v
    out_proxy = mode.tracer.create_proxy(
        "call_function", hop, (), proxy_kwargs, name=hop._name
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@helion_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def helion_kernel_wrapper_mutation_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    return _trace_hop_proxy(  # pyrefly: ignore[bad-return]
        helion_kernel_wrapper_mutation,
        mode,
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
    return _trace_hop_proxy(  # pyrefly: ignore[bad-return]
        helion_kernel_wrapper_functional,
        mode,
        {
            "kernel_idx": kernel_idx,
            "constant_args": constant_args,
            "tensor_args": tensor_args,
            "output_spec": output_spec,
            "tensors_to_clone": tensors_to_clone,
        },
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
    helion_kernel_wrapper_functional.fallthrough(key)
