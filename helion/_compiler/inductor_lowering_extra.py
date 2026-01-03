from __future__ import annotations

import contextlib
import functools
from typing import Any
from typing import Callable
from typing import Generator
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node
from torch._inductor.dependencies import MemoryDep
from torch._inductor.dependencies import StarDep
from torch._inductor.dependencies import WeakDep
from torch._inductor.ir import Buffer
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import ExternKernel
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import InputBuffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import Pointwise
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TemplateBuffer
from torch._inductor.ir import TensorBox
from torch._inductor.ir import View
from torch._inductor.virtualized import V
from torch._inductor.lowering import lowerings as original_lowerings
from torch._inductor.lowering import register_lowering
from torch._inductor.lowering import to_dtype
from torch._inductor.lowering import var_mean_sum_
from torch._inductor.choices import InductorChoices
from torch._prims_common import get_computation_dtype

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode
    from torch._inductor.scheduler import Scheduler

from ._inductor.template_buffer import HelionTemplateBuffer
from ._inductor.indexing import compute_helion_transform
from ._inductor.indexing import has_unsafe_views
from ._inductor.indexing import has_unsafe_views_from_fx_nodes

_view_tracking_patched = False
_baseview_post_init = None


def _install_view_tracking() -> None:
    """Patch BaseView.__post_init__ to record view IR nodes by FX node."""
    global _view_tracking_patched, _baseview_post_init
    if _view_tracking_patched:
        return
    from torch._inductor.ir import BaseView

    _baseview_post_init = BaseView.__post_init__

    def _wrapped_post_init(self) -> None:  # noqa: ANN001
        assert _baseview_post_init is not None
        _baseview_post_init(self)
        try:
            graph = V.graph
        except Exception:
            return
        if not hasattr(graph, "helion_view_node_map"):
            return
        fx_node = getattr(graph, "current_node", None)
        if isinstance(fx_node, Node):
            graph.helion_view_node_map[fx_node] = self

    BaseView.__post_init__ = _wrapped_post_init  # type: ignore[assignment]
    _view_tracking_patched = True


class HelionFusionChoices(InductorChoices):
    """Allow Helion template prologue fusion even when MemoryDep indices differ."""

    @staticmethod
    def _helion_template_output_nodes(
        template: HelionTemplateBuffer,
    ) -> dict[str, IRNode]:
        output_nodes: dict[str, IRNode] = {template.get_name(): template}
        multi_outputs = getattr(template, "multi_output_nodes", [])
        for mo in multi_outputs:
            if isinstance(mo, IRNode):
                output_nodes[mo.get_name()] = mo
        return output_nodes

    @staticmethod
    def _helion_template_input_nodes(
        template: HelionTemplateBuffer,
    ) -> dict[str, IRNode]:
        input_nodes: dict[str, IRNode] = {}
        for inp in template.inputs:  # type: ignore[union-attr]
            if isinstance(inp, IRNode):
                input_nodes[inp.get_name()] = inp
        return input_nodes

    @staticmethod
    def can_fuse(
        scheduler: "Scheduler",
        node1: "BaseSchedulerNode",
        node2: "BaseSchedulerNode",
        shared_data_score: int,
    ) -> bool:
        """Allow fusion when nodes share buffer names (for Helion templates only)."""
        if node1.is_template() and not node2.is_template():
            template = node1.get_template_node()
            if isinstance(template, HelionTemplateBuffer):
                output_nodes = HelionFusionChoices._helion_template_output_nodes(
                    template
                )
                reads = {
                    dep.name
                    for dep in node2.read_writes.reads
                    if isinstance(dep, (MemoryDep, StarDep, WeakDep))
                    and dep.name in output_nodes
                }
                if reads:
                    acc_name = next(iter(reads))
                    kernel_output = output_nodes.get(acc_name)
                    epilogue_nodes = node2.get_nodes()
                    last_node = epilogue_nodes[-1] if epilogue_nodes else None
                    epilogue_ir = (
                        last_node.node if hasattr(last_node, "node") else None
                    )
                    if (
                        isinstance(kernel_output, IRNode)
                        and isinstance(epilogue_ir, IRNode)
                    ):
                        kernel_shape = list(kernel_output.get_size())
                        kernel_strides = (
                            list(kernel_output.get_stride())
                            if hasattr(kernel_output, "get_stride")
                            else None
                        )
                        epilogue_shape = list(epilogue_ir.get_size())
                        epilogue_layout = epilogue_ir.get_layout()
                        transform = compute_helion_transform(
                            epilogue_nodes,
                            acc_name,
                            kernel_shape,
                            epilogue_shape,
                            epilogue_layout,
                            kernel_strides,
                        )
                        if transform.unsupported or transform.broadcast_dims:
                            return False

        if node2.is_template() and not node1.is_template():
            template = node2.get_template_node()
            if isinstance(template, HelionTemplateBuffer):
                input_nodes = HelionFusionChoices._helion_template_input_nodes(template)
                if input_nodes:
                    prologue_nodes = node1.get_nodes()
                    last_node = prologue_nodes[-1] if prologue_nodes else None
                    prologue_ir = (
                        last_node.node if hasattr(last_node, "node") else None
                    )
                    if prologue_nodes and has_unsafe_views(prologue_nodes):
                        return False
                    if isinstance(prologue_ir, IRNode):
                        prologue_outs = [
                            out.get_name()
                            for out in last_node.get_outputs()
                            if hasattr(out, "get_name")
                        ]
                        acc_name = next(
                            (name for name in prologue_outs if name in input_nodes),
                            None,
                        )
                        if acc_name is not None:
                            fx_inputs = getattr(V.graph, "helion_input_fx_nodes", None)
                            if fx_inputs and acc_name in fx_inputs:
                                if has_unsafe_views_from_fx_nodes(
                                    [fx_inputs[acc_name]]
                                ):
                                    return False
                            existing = None
                            for snode in reversed(prologue_nodes):
                                if hasattr(snode, "_helion_prologue_transform"):
                                    existing = getattr(
                                        snode, "_helion_prologue_transform"
                                    )
                                    break
                            if existing is not None:
                                if existing.unsupported or existing.broadcast_dims:
                                    return False
                            else:
                                kernel_input = input_nodes[acc_name]
                                kernel_shape = list(kernel_input.get_size())
                                kernel_strides = (
                                    list(kernel_input.get_stride())
                                    if hasattr(kernel_input, "get_stride")
                                    else None
                                )
                                prologue_shape = list(prologue_ir.get_size())
                                prologue_layout = prologue_ir.get_layout()
                                transform = compute_helion_transform(
                                    prologue_nodes,
                                    acc_name,
                                    kernel_shape,
                                    prologue_shape,
                                    prologue_layout,
                                    kernel_strides,
                                )
                                if transform.unsupported or transform.broadcast_dims:
                                    return False
                                for snode in prologue_nodes:
                                    setattr(
                                        snode, "_helion_prologue_transform", transform
                                    )
                                setattr(node1, "_helion_prologue_transform", transform)
        # Check if node2 is a Helion template with zero shared_data_score
        if (
            shared_data_score == 0
            and node2.is_template()
            and isinstance(node2.get_template_node(), HelionTemplateBuffer)
            and node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
        ):
            return True

        return InductorChoices.can_fuse(scheduler, node1, node2, shared_data_score)


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
    # Register the Helion kernel lowering (done lazily to avoid circular imports)
    _register_helion_kernel_lowering()

    # pyrefly: ignore [implicit-import]
    original_lowerings = torch._inductor.lowering.lowerings.copy()

    # Set the custom choices handler to allow template prologue fusion
    # with different indexing (e.g., flat prologue fused into 2D kernel).
    #
    # IMPORTANT: We intentionally do NOT use the context manager returned by
    # set_choices_handler because the scheduler runs AFTER this context exits.
    # The flow is: torch.compile() -> trace -> Helion kernel call (this context)
    # -> context exit -> inductor scheduler (needs the custom handler).
    #
    # HelionFusionChoices is a safe superset of InductorChoices - it falls back
    # to default behavior for all non-Helion-template cases. The handler persists
    # for the duration of the process, which is acceptable since it only affects
    # template prologue fusion decisions.
    V.set_choices_handler(HelionFusionChoices())

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


_helion_kernel_lowering_registered = False


def _register_helion_kernel_lowering() -> None:
    """Register the Helion kernel lowering with Inductor.

    This is done in a function to avoid circular imports at module load time.
    """
    global _helion_kernel_lowering_registered
    if _helion_kernel_lowering_registered:
        return
    _helion_kernel_lowering_registered = True
    _install_view_tracking()

    @register_lowering(_helion_hop, type_promotion_kind=None)
    def lower_helion_kernel(  # noqa: ANN202
        *,
        kernel_idx: Any,  # noqa: ANN401
        constant_args: Any,  # noqa: ANN401
        tensor_args: Any,  # noqa: ANN401
        output_spec: Any,  # noqa: ANN401
    ):
        """Lower a Helion kernel call to HelionTemplateBuffer."""
        if not hasattr(V.graph, "helion_view_node_map"):
            V.graph.helion_view_node_map = {}  # type: ignore[attr-defined]
        if not hasattr(V.graph, "helion_kernel_fx_nodes"):
            V.graph.helion_kernel_fx_nodes = set()  # type: ignore[attr-defined]
        if isinstance(getattr(V.graph, "current_node", None), Node):
            V.graph.helion_kernel_fx_nodes.add(V.graph.current_node)  # type: ignore[attr-defined]

        def realize_input(tensor_box: Any) -> Any:  # noqa: ANN401
            """Realize TensorBox to buffer for prologue fusion."""
            BUF_TYPES = (ComputedBuffer, InputBuffer, ReinterpretView, TemplateBuffer, Buffer)
            FUSABLE_TYPES = (Pointwise, View)

            if not isinstance(tensor_box, TensorBox):
                return tensor_box
            data = tensor_box.data

            def needs_realize(inner: Any) -> bool:
                return isinstance(inner, FUSABLE_TYPES) or not isinstance(inner, BUF_TYPES)

            if isinstance(data, View):
                if isinstance(data.data, StorageBox) and needs_realize(data.data.data):
                    data.data.realize()
                return ExternKernel.convert_to_reinterpret_view(data)

            if isinstance(data, StorageBox):
                if needs_realize(data.data):
                    data.realize()
                return data.data

            if isinstance(data, BUF_TYPES):
                return data

            tensor_box.realize()
            return tensor_box.data.data if isinstance(tensor_box.data, StorageBox) else tensor_box.data

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

        if not hasattr(V.graph, "helion_input_fx_nodes"):
            V.graph.helion_input_fx_nodes = {}  # type: ignore[attr-defined]
        helion_fx_nodes = V.graph.helion_input_fx_nodes  # type: ignore[attr-defined]
        for inp in inputs:
            if not isinstance(inp, IRNode):
                continue
            origin = inp.get_origin_node()
            if isinstance(origin, Node):
                helion_fx_nodes[inp.get_name()] = origin
                continue
            origins = getattr(inp, "origins", None)
            if origins:
                for origin in origins:
                    if isinstance(origin, Node):
                        helion_fx_nodes[inp.get_name()] = origin
                        break

        # Bind the kernel with fake tensors for code generation
        fake_tensors, sig = [], kernel.signature.parameters
        for name in sig:
            if name in arg_names:
                tb = tensor_args[name]
                assert isinstance(
                    tb, TensorBox
                ), f"Expected TensorBox for {name}, got {type(tb)}"
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

        def get_output_layout(idx: int) -> FixedLayout | None:
            if (spec := per_output_specs[idx]) and "shape" in spec:
                return FixedLayout(device=torch.device(spec["device"]), dtype=spec["dtype"], size=spec["shape"])
            return None

        def get_scalar_value(idx: int) -> int | float | None:
            if spec := per_output_specs[idx]:
                return spec.get("scalar_value")
            return None

        # Determine layout based on number of outputs
        if num_outputs == 1:
            layout = get_output_layout(0)
            if layout is None:
                raise ValueError(
                    "Single-output kernel must return a tensor, not a scalar"
                )
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
            # pyrefly: ignore [bad-assignment]
            layout = MultiOutputLayout(device=multi_output_device)

        buf = HelionTemplateBuffer(
            layout=layout,
            inputs=inputs,
            kernel=kernel,
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_arg_names=arg_names,
            bound_kernel=bound,
        )

        if num_outputs == 1:
            return (TensorBox(StorageBox(buf)),)
        # Create per-output nodes - tensor outputs get TensorBox, scalars get their value
        results: list[TensorBox | int | float | None] = []
        multi_output_nodes = []
        for i in range(num_outputs):
            layout = get_output_layout(i)
            if layout is not None:
                mo = MultiOutput(layout=layout, input=buf, indices=[(tuple, i)])
                multi_output_nodes.append(mo)
                results.append(TensorBox.create(mo))
            else:
                # Scalar output - return the scalar value if known
                results.append(get_scalar_value(i))
        # pyrefly: ignore [missing-attribute]
        buf.multi_output_nodes = multi_output_nodes
        # pyrefly: ignore [bad-return]
        return tuple(results)
