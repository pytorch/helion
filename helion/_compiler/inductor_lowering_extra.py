from __future__ import annotations

import contextlib
import functools
from typing import Any
from typing import Callable
from typing import Generator

import sympy
import torch
from torch._inductor.ir import ExternKernel
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import FlexibleLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import lowerings as original_lowerings
from torch._inductor.lowering import register_lowering
from torch._inductor.lowering import to_dtype
from torch._inductor.virtualized import V

from ._inductor.template_buffer import HelionTemplateBuffer
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
    kernel = get_helion_kernel(kernel_idx)

    # Realize inputs: convert TensorBox to buffer/ReinterpretView
    def realize(tb: Any) -> Any:  # noqa: ANN401
        if not isinstance(tb, TensorBox):
            return tb
        result = ExternKernel.realize_input(tb)
        if isinstance(result, StorageBox):
            result = result.data
        if isinstance(getattr(result, "layout", None), FlexibleLayout):
            result.freeze_layout()
        return result

    realized = {n: realize(tb) for n, tb in tensor_args.items() if isinstance(tb, TensorBox)}
    inputs, arg_names = list(realized.values()), list(realized.keys())

    # Extract output spec components
    num_outputs = output_spec["num_outputs"]
    specs = output_spec["output_specs"]
    aliases = output_spec.get("output_aliases", [])
    direct_flags = output_spec.get("output_alias_is_direct", [])

    # Build fake tensors for kernel binding
    fake_tensors = []
    for name, param in kernel.signature.parameters.items():
        if name in realized:
            inp = realized[name]
            size = [int(s) if isinstance(s, (int, sympy.Integer)) else 64 for s in inp.get_size()]
            stride = [int(s) if isinstance(s, (int, sympy.Integer)) else 1 for s in inp.get_stride()]
            fake_tensors.append(torch.empty_strided(size, stride, dtype=inp.get_dtype(), device=inp.get_device()))
        elif name in constant_args:
            fake_tensors.append(constant_args[name])
        elif param.default is not param.empty:
            fake_tensors.append(param.default)
    bound = kernel.bind(tuple(fake_tensors))

    def make_layout(idx: int) -> FixedLayout | None:
        """Create FixedLayout from output spec at given index, or None for scalars."""
        spec = specs[idx]
        if spec is None or "shape" not in spec:
            return None
        return FixedLayout(device=torch.device(spec["device"]), dtype=spec["dtype"], size=spec["shape"])

    # Determine buffer layout
    if num_outputs == 1:
        layout = make_layout(0)
        if layout is None:
            raise ValueError("Single-output kernel must return a tensor, not a scalar")
    else:
        device = next((torch.device(s["device"]) for s in specs if s and "device" in s), torch.device("cuda"))
        layout = MultiOutputLayout(device=device)

    # Build HelionTemplateBuffer
    mutated = [inputs[arg_names.index(n)] for n in output_spec.get("mutated_inputs", []) if n in arg_names]
    buf = HelionTemplateBuffer(
        layout=layout,
        inputs=inputs,
        kernel=kernel,
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_arg_names=arg_names,
        bound_kernel=bound,
        mutated_inputs=mutated or None,
        output_aliases=aliases,
        output_alias_is_direct=direct_flags,
        autotune_args=tuple(fake_tensors),
    )
    V.graph.no_fuse_buffer_names.add(buf.get_name())  # Disable fusion for now

    # Build output results
    results: list[TensorBox | int | float | None] = []
    multi_output_nodes: list[MultiOutput] = []

    for i in range(num_outputs):
        spec = specs[i]
        alias_name = aliases[i] if i < len(aliases) else None

        # Handle aliased outputs
        if alias_name and alias_name in arg_names:
            alias_inp = inputs[arg_names.index(alias_name)]
            if isinstance(alias_inp, IRNode):
                is_direct = i < len(direct_flags) and direct_flags[i]
                if is_direct:
                    results.append(TensorBox.create(alias_inp))
                    continue
                # Indirect alias: create ReinterpretView
                if spec and "stride" in spec:
                    view_layout = FixedLayout(
                        device=alias_inp.get_device(),
                        dtype=alias_inp.get_dtype(),
                        size=[sympy.Integer(s) for s in spec["shape"]],
                        stride=[sympy.Integer(s) for s in spec["stride"]],
                        offset=sympy.Integer(spec.get("storage_offset", 0)),
                    )
                    storage = alias_inp if isinstance(alias_inp, StorageBox) else StorageBox(alias_inp)
                    results.append(TensorBox.create(ReinterpretView(data=storage, layout=view_layout)))
                    continue

        # Handle non-alias tensor outputs
        out_layout = make_layout(i)
        if out_layout is not None:
            if num_outputs == 1:
                results.append(TensorBox(StorageBox(buf)))
            else:
                mo = MultiOutput(layout=out_layout, input=buf, indices=[(tuple, i)])
                multi_output_nodes.append(mo)
                results.append(TensorBox.create(mo))
        else:
            # Scalar output
            results.append(spec.get("scalar_value") if spec else None)

    if num_outputs > 1:
        buf.multi_output_nodes = multi_output_nodes
        if not multi_output_nodes:
            fallback = make_layout(0)
            if fallback:
                buf.layout = fallback

    return tuple(results)
