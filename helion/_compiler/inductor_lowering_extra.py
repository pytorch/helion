from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generator

import sympy
import torch
from torch._inductor.ir import Buffer
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import InputBuffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TemplateBuffer
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import lowerings as original_lowerings
from torch._inductor.lowering import register_lowering
from torch._inductor.lowering import to_dtype

if TYPE_CHECKING:
    pass

from ._inductor.template_buffer import HelionTemplateBuffer
from ._inductor.template_buffer import is_trivial_view
from helion._compiler._dynamo.higher_order_ops import get_helion_kernel
from helion._compiler._dynamo.higher_order_ops import (
    helion_kernel_wrapper_mutation as _helion_hop,
)

inductor_lowering_dispatch: dict[Callable[..., Any] | str, Callable[..., Any]] = {}


def create_fp16_to_fp32_unary_fallback_lowering(
    original_op: Callable[..., object],
) -> Callable[..., object]:
    """Create a lowering that converts fp16/bfloat16 inputs to fp32 before calling the operation."""

    @functools.wraps(original_op)
    def fp32_fallback_lowering(x: object) -> object:
        if isinstance(x, TensorBox) and (original_dtype := x.get_dtype()) in (
            torch.float16,
            torch.bfloat16,
        ):
            x_fp32 = to_dtype(x, torch.float32)
            result_fp32 = original_op(x_fp32)
            assert isinstance(result_fp32, TensorBox)
            return to_dtype(result_fp32, original_dtype)
        return original_op(x)

    return fp32_fallback_lowering


# Operations that need fp32 fallbacks due to libdevice/tl_math limitations
FP32_FALLBACK_OPS_UNARY = [
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.log.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.log1p.default,
    torch.ops.aten.expm1.default,
    torch.ops.aten.exp.default,
]

# Register fp32 fallback lowerings for ops that don't support fp16/bfloat16
for op in FP32_FALLBACK_OPS_UNARY:
    inductor_lowering_dispatch[op] = create_fp16_to_fp32_unary_fallback_lowering(
        original_lowerings[op]
    )


@contextlib.contextmanager
def patch_inductor_lowerings() -> Generator[None, Any, Any]:
    """Context manager to temporarily patch the inductor lowering table.

    This is useful for overwriting specific Inductor lowerings without
    affecting the global state, especially in cases where Helion
    is missing support for a specific lowering.
    """
    # pyrefly: ignore [implicit-import]
    original_lowerings = torch._inductor.lowering.lowerings.copy()
    try:
        # pyrefly: ignore [implicit-import]
        torch._inductor.lowering.lowerings.update(inductor_lowering_dispatch)
        yield
    finally:
        # pyrefly: ignore [implicit-import]
        torch._inductor.lowering.lowerings = original_lowerings


# pyrefly: ignore [implicit-import]
register_inductor_lowering = torch._inductor.lowering.register_lowering


def var_mean_helper_(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    from torch._inductor.lowering import var_mean_sum_
    from torch._prims_common import get_computation_dtype

    out_dtype = x.get_dtype()
    compute_dtype = get_computation_dtype(out_dtype)

    x = to_dtype(x, compute_dtype, copy=False)

    kwargs = {
        "x": x,
        "axis": axis,
        "correction": correction,
        "keepdim": keepdim,
        "return_mean": return_mean,
    }
    # TODO(yf225): support Welford reduction in Helion, then switch back to use Inductor `var_mean_helper_()`.
    output = var_mean_sum_(**kwargs)
    output = tuple(to_dtype(o, out_dtype, copy=False) for o in output)
    # pyrefly: ignore [bad-return]
    return output[0] if not return_mean else output


@register_inductor_lowering(
    [torch.ops.aten.var.correction],
    lowering_dict=inductor_lowering_dispatch,
)
def var_(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=False,
    )


@register_inductor_lowering(
    torch.ops.aten.var_mean.correction,
    lowering_dict=inductor_lowering_dispatch,
)
def var_mean(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=True,
    )


@register_lowering(_helion_hop, type_promotion_kind=None)
def lower_helion_kernel(  # noqa: ANN202
    *,
    kernel_idx: Any,  # noqa: ANN401
    constant_args: Any,  # noqa: ANN401
    tensor_args: Any,  # noqa: ANN401
    output_spec: Any,  # noqa: ANN401
):
    """Lower a Helion kernel call to HelionTemplateBuffer."""

    def realize_input(tensor_box: Any) -> Any:  # noqa: ANN401
        """Realize a TensorBox to a buffer for use in HelionTemplateBuffer."""
        BUF_TYPES = (
            ComputedBuffer,
            InputBuffer,
            ReinterpretView,
            TemplateBuffer,
            Buffer,
        )
        if not isinstance(tensor_box, TensorBox):
            return tensor_box
        data = tensor_box.data
        if isinstance(data, ReinterpretView) and is_trivial_view(data):
            base = data.unwrap_view()
            if isinstance(base, StorageBox):
                base = base.data
            if isinstance(base, IRNode):
                return base
        if isinstance(data, StorageBox):
            if not isinstance(data.data, BUF_TYPES):
                data.realize()
            inner = data.data
            if isinstance(inner, ReinterpretView) and is_trivial_view(inner):
                base = inner.unwrap_view()
                if isinstance(base, StorageBox):
                    base = base.data
                if isinstance(base, IRNode):
                    return base
            return inner
        if isinstance(data, BUF_TYPES):
            return data
        tensor_box.realize()
        return (
            tensor_box.data.data
            if isinstance(tensor_box.data, StorageBox)
            else tensor_box.data
        )

    kernel = get_helion_kernel(kernel_idx)
    inputs, arg_names = (
        zip(  # noqa: B905
            *[
                ((realize_input(tb), name))
                for name, tb in tensor_args.items()
                if isinstance(tb, TensorBox)
            ]
        )
        if tensor_args
        else ([], [])
    )
    inputs, arg_names = list(inputs), list(arg_names)
    num_outputs = output_spec["num_outputs"]
    per_output_specs = output_spec["output_specs"]
    output_aliases = output_spec.get("output_aliases", [])
    output_alias_is_direct = output_spec.get("output_alias_is_direct", [])

    # Bind the kernel with fake tensors for code generation
    fake_tensors, sig = [], kernel.signature.parameters
    for name in sig:
        if name in arg_names:
            tb = tensor_args[name]
            assert isinstance(tb, TensorBox), (
                f"Expected TensorBox for {name}, got {type(tb)}"
            )
            size = [
                int(s) if isinstance(s, (int, sympy.Integer)) else 64
                for s in tb.get_size()
            ]
            fake_tensors.append(
                torch.empty(size, dtype=tb.get_dtype(), device=tb.get_device())
            )
        elif name in constant_args:
            fake_tensors.append(constant_args[name])
        elif sig[name].default is not sig[name].empty:
            fake_tensors.append(sig[name].default)
    bound = kernel.bind(tuple(fake_tensors))

    # Helper to get layout for a specific output index
    def get_output_layout(idx: int) -> FixedLayout | None:
        spec = per_output_specs[idx]
        if spec is not None and "shape" in spec:
            # Tensor output - has shape/dtype/device
            return FixedLayout(
                device=torch.device(spec["device"]),
                dtype=spec["dtype"],
                size=spec["shape"],
            )
        # Non-tensor output (scalar or unknown) - no layout needed
        return None

    # Helper to get scalar value for a specific output index
    def get_scalar_value(idx: int) -> int | float | None:
        spec = per_output_specs[idx]
        if spec is not None and "scalar_value" in spec:
            return spec["scalar_value"]
        return None

    # Determine layout based on number of outputs
    layout: FixedLayout | MultiOutputLayout
    if num_outputs == 1:
        single_layout = get_output_layout(0)
        if single_layout is None:
            raise ValueError(
                "Single-output kernel must return a tensor, not a scalar"
            )
        layout = single_layout
    else:
        # Get device from first tensor output for MultiOutputLayout
        multi_output_device = next(
            (
                torch.device(s["device"])
                for s in per_output_specs
                if s is not None and "device" in s
            ),
            torch.device("cuda"),
        )
        layout = MultiOutputLayout(device=multi_output_device)

    # Extract mutated inputs from output_spec
    mutated_input_names = output_spec.get("mutated_inputs", [])
    mutated_inputs = [
        inputs[arg_names.index(name)]
        for name in mutated_input_names
        if name in arg_names
    ]

    buf = HelionTemplateBuffer(
        layout=layout,
        inputs=inputs,
        kernel=kernel,
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_arg_names=arg_names,
        bound_kernel=bound,
        mutated_inputs=mutated_inputs or None,
        output_aliases=output_aliases,
        output_alias_is_direct=output_alias_is_direct,
        autotune_args=tuple(fake_tensors),
    )

    if num_outputs == 1:
        alias_name = output_aliases[0] if output_aliases else None
        is_direct = bool(output_alias_is_direct and output_alias_is_direct[0])
        if alias_name is not None and alias_name in arg_names and is_direct:
            alias_inp = inputs[arg_names.index(alias_name)]
            if isinstance(alias_inp, IRNode):
                return (TensorBox.create(alias_inp),)
        return (TensorBox(StorageBox(buf)),)
    # Create per-output nodes - tensor outputs get TensorBox, scalars get their value
    results: list[TensorBox | int | float | None] = []
    multi_output_nodes = []
    for i in range(num_outputs):
        alias_name = output_aliases[i] if i < len(output_aliases) else None
        is_direct = bool(
            output_alias_is_direct
            and i < len(output_alias_is_direct)
            and output_alias_is_direct[i]
        )
        if alias_name is not None and alias_name in arg_names and is_direct:
            alias_inp = inputs[arg_names.index(alias_name)]
            if isinstance(alias_inp, IRNode):
                results.append(TensorBox.create(alias_inp))
                continue
        output_layout = get_output_layout(i)
        if output_layout is not None:
            mo = MultiOutput(layout=output_layout, input=buf, indices=[(tuple, i)])
            multi_output_nodes.append(mo)
            results.append(TensorBox.create(mo))
        else:
            # Scalar output - return the scalar value if known
            results.append(get_scalar_value(i))
    buf.multi_output_nodes = multi_output_nodes
    if num_outputs > 1 and not multi_output_nodes:
        # All outputs are aliases/scalars; avoid MultiOutputLayout assertions.
        fallback_layout = get_output_layout(0)
        if fallback_layout is not None:
            buf.layout = fallback_layout
    return tuple(results)
