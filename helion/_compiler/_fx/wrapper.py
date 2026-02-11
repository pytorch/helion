from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


def _is_fx_tracing(args: tuple[object, ...]) -> bool:
    for a in args:
        t = type(a)
        if t.__name__ == "Proxy" and t.__module__ == "torch.fx.proxy":
            return True
    return False


def create_fx_proxy_for_kernel(
    kernel: Kernel,
    kernel_idx: int | None,
    args: tuple[Any, ...],
    output_spec: dict[str, object],
) -> Any:
    """Create an FX proxy node for a Helion kernel call during symbolic tracing.

    Registers the kernel in the global ``HelionKernelSideTable``, partitions
    arguments into constants vs tensor proxies, and emits a
    ``helion_kernel_wrapper_mutation`` HOP node into the FX graph via
    ``tracer.create_proxy()``.  This bypasses the ``Proxy.__torch_function__``
    guard that rejects HigherOrderOperators.

    The returned proxy represents the HOP's output: a **flat tuple of tensor
    outputs only** (scalars are dropped by the dense execution impl).  Callers
    should index into this tuple to extract individual outputs, e.g.
    ``proxy[0]`` for a single-tensor-output kernel.

    Args:
        kernel: The ``helion.Kernel`` to register and trace.
        kernel_idx: Pre-computed side table index, or ``None`` to register now.
        args: Positional arguments (mix of ``Proxy`` and constant values).
        output_spec: Output specification with ``leaf_specs`` and
            ``tree_spec_str`` keys, as produced by ``_infer_output_spec`` or
            built from running fake tensors through the kernel's fake impl.

    Returns:
        A ``torch.fx.Proxy`` representing the flat tuple of tensor outputs.
    """
    from torch.fx import Proxy

    from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table
    from helion._compiler._dynamo.higher_order_ops import helion_kernel_wrapper_mutation

    proxy_arg: Proxy | None = None
    for a in args:
        if isinstance(a, Proxy):
            proxy_arg = a
            break
    assert proxy_arg is not None, "create_fx_proxy_for_kernel called without Proxy args"

    tracer = proxy_arg.tracer

    if kernel_idx is None:
        kernel_idx = helion_kernel_side_table.add_kernel(kernel)

    constant_args: dict[str, object] = {}
    tensor_args: dict[str, Proxy] = {}
    for name, val in zip(kernel.signature.parameters.keys(), args, strict=False):
        if isinstance(val, Proxy):
            tensor_args[name] = val
        else:
            constant_args[name] = val

    return tracer.create_proxy(
        "call_function",
        helion_kernel_wrapper_mutation,
        (),
        {
            "kernel_idx": kernel_idx,
            "constant_args": constant_args,
            "tensor_args": tensor_args,
            "output_spec": output_spec,
        },
    )
