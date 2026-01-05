from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generator

import sympy
import torch
from torch._inductor.choices import InductorChoices
from torch._inductor.dependencies import MemoryDep
from torch._inductor.dependencies import StarDep
from torch._inductor.dependencies import WeakDep
from torch._inductor.ir import BaseView
from torch._inductor.ir import Buffer
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import InputBuffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import MutationOutput
from torch._inductor.ir import OutputSpec
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TemplateBuffer
from torch._inductor.ir import TensorBox
import torch._inductor.lowering as _inductor_lowering
from torch._inductor.lowering import lowerings as original_lowerings
from torch._inductor.lowering import register_lowering
from torch._inductor.lowering import to_dtype
from torch._inductor.lowering import var_mean_sum_
from torch._inductor.virtualized import V
from torch._prims_common import get_computation_dtype

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode
    from torch._inductor.scheduler import Scheduler

from ._inductor.template_buffer import HelionTemplateBuffer
from ._inductor.template_buffer import _get_ir_node
from ._inductor.template_buffer import has_non_trivial_view
from ._inductor.template_buffer import has_view
from ._inductor.template_buffer import is_trivial_view
from ._inductor.template_buffer import same_shape_and_stride
from helion._compiler._dynamo.higher_order_ops import get_helion_kernel
from helion._compiler._dynamo.higher_order_ops import (
    helion_kernel_wrapper_mutation as _helion_hop,
)

inductor_lowering_dispatch: dict[Callable[..., Any] | str, Callable[..., Any]] = {}
_SCHEDULER_FUSION_PATCHED = False
_ORIG_SCHEDULER_FUSABLE_READ_AND_WRITE = None
_ORIG_SCHEDULER_CAN_FUSE = None


def _collect_mutated_names(template: HelionTemplateBuffer) -> set[str]:
    mutated_names: set[str] = set(
        getattr(template, "_helion_mutated_input_names", set())
    )
    if template.mutated_inputs:
        for buf in template.mutated_inputs:
            if isinstance(buf, BaseView):
                mutated_names.add(buf.get_name())
                base = buf.unwrap_view()
                if isinstance(base, StorageBox):
                    base = base.data
                if isinstance(base, IRNode):
                    mutated_names.add(base.get_name())
            elif isinstance(buf, IRNode):
                mutated_names.add(buf.get_name())
    for arg_name, buf in template.named_input_nodes.items():
        buf_name = buf.get_name()
        if buf_name in mutated_names:
            mutated_names.add(arg_name)
        if isinstance(buf, BaseView):
            base = buf.unwrap_view()
            if isinstance(base, StorageBox):
                base = base.data
            if isinstance(base, IRNode) and base.get_name() in mutated_names:
                mutated_names.add(arg_name)
    for out in template.outputs[1:]:
        if isinstance(out, MutationOutput):
            mutated_names.add(out.get_name())
            for buf in out.get_mutation_buffers():
                if isinstance(buf, IRNode):
                    mutated_names.add(buf.get_name())
    return mutated_names


def _mutated_base_names(template: HelionTemplateBuffer) -> set[str]:
    names: set[str] = set()
    if template.mutated_inputs:
        for buf in template.mutated_inputs:
            if isinstance(buf, BaseView):
                base = buf.unwrap_view()
                if isinstance(base, StorageBox):
                    base = base.data
                if isinstance(base, IRNode):
                    names.add(base.get_name())
            elif isinstance(buf, IRNode):
                names.add(buf.get_name())
    return names


def _block_multi_mutation_epilogue(
    template: HelionTemplateBuffer, node2: BaseSchedulerNode | None
) -> bool:
    """Block epilogue fusion when template has multiple mutated inputs."""
    if not node2 or not template.mutated_inputs:
        return False
    if isinstance(template.layout, MultiOutputLayout):
        return False
    if len(_mutated_base_names(template)) <= 1:
        return False
    if not node2.read_writes:
        return False
    mutated_names = _collect_mutated_names(template)
    return any(
        isinstance(dep, (MemoryDep, StarDep, WeakDep))
        and dep.name in mutated_names
        for dep in node2.read_writes.reads
    )


def _get_direct_aliases(template: HelionTemplateBuffer) -> set[str]:
    """Extract direct output aliases from template."""
    if not template._output_aliases or not template._output_alias_is_direct:
        return set()
    return {
        alias
        for alias, is_direct in zip(
            template._output_aliases,
            template._output_alias_is_direct,
            strict=False,
        )
        if is_direct and alias
    }


def _patch_scheduler_fusable_read_and_write() -> None:
    global _SCHEDULER_FUSION_PATCHED
    global _ORIG_SCHEDULER_FUSABLE_READ_AND_WRITE
    global _ORIG_SCHEDULER_CAN_FUSE

    if _SCHEDULER_FUSION_PATCHED:
        return

    from torch._inductor import dependencies
    from torch._inductor import scheduler as inductor_scheduler

    original_fusable = inductor_scheduler.Scheduler.fusable_read_and_write

    def helion_fusable_read_and_write(  # noqa: ANN202
        self, read: dependencies.Dep, write: dependencies.MemoryDep
    ) -> bool:
        def _normalize_name(name: str) -> str:
            name = self.mutation_renames.get(name, name)
            buf = V.graph.try_get_buffer(name)
            if isinstance(buf, MutationOutput):
                mutation_bufs = buf.get_mutation_buffers()
                if mutation_bufs:
                    buf = mutation_bufs[0]
                    name = buf.get_name()
            if isinstance(buf, BaseView) and is_trivial_view(buf):
                base = buf.unwrap_view()
                if isinstance(base, StorageBox):
                    base = base.data
                if isinstance(base, IRNode):
                    name = base.get_name()
            return name

        if isinstance(read, dependencies.MemoryDep):
            read_name = _normalize_name(read.name)
            write_name = _normalize_name(write.name)
            if (
                read_name == write_name
                and read.mode == write.mode
                and not read.is_indirect()
                and not write.is_indirect()
                and read.num_vars == write.num_vars
                and read.size == write.size
                and isinstance(read.index, sympy.Symbol)
                and isinstance(write.index, sympy.Symbol)
            ):
                return True
        if isinstance(read, dependencies.StarDep):
            read_name = _normalize_name(read.name)
            write_name = _normalize_name(write.name)
            if (
                read.mode is None
                and write.mode is None
                and read_name == write_name
                and not write.is_indirect()
            ):
                if V.graph.sizevars.statically_known_equals(
                    write.get_numel(), V.graph.get_numel(read.name)
                ):
                    return True
        if isinstance(read, dependencies.WeakDep):
            read_name = _normalize_name(read.name)
            write_name = _normalize_name(write.name)
            if read_name == write_name and not write.is_indirect():
                return True
        return original_fusable(self, read, write)

    inductor_scheduler.Scheduler.fusable_read_and_write = helion_fusable_read_and_write
    _ORIG_SCHEDULER_FUSABLE_READ_AND_WRITE = original_fusable

    original_can_fuse = inductor_scheduler.Scheduler.can_fuse

    def helion_can_fuse(  # noqa: ANN202
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        can_reorder: bool = False,
        allow_mix_order_reduction: bool = True,
    ) -> bool:
        if not HelionFusionChoices._is_helion_template_node(node1):
            return original_can_fuse(
                self,
                node1,
                node2,
                can_reorder=can_reorder,
                allow_mix_order_reduction=allow_mix_order_reduction,
            )

        def _normalize_name(name: str) -> str:
            name = self.mutation_renames.get(name, name)
            buf = V.graph.try_get_buffer(name)
            if isinstance(buf, MutationOutput):
                mutation_bufs = buf.get_mutation_buffers()
                if mutation_bufs:
                    buf = mutation_bufs[0]
                    name = buf.get_name()
            if isinstance(buf, BaseView) and is_trivial_view(buf):
                base = buf.unwrap_view()
                if isinstance(base, StorageBox):
                    base = base.data
                if isinstance(base, IRNode):
                    name = base.get_name()
            return name

        def _normalized_buffer_names(node: BaseSchedulerNode) -> set[str]:
            return {
                _normalize_name(n)
                for n in node.read_writes.buffer_names()
                if isinstance(n, str)
            }

        def _alias_shared_data() -> bool:
            if node1.get_device() != node2.get_device():
                return False
            if node2.is_reduction() or node2.is_template():
                return False
            if not node2.read_writes:
                return False
            n1 = _normalized_buffer_names(node1)
            n2 = _normalized_buffer_names(node2)
            return bool(n1 & n2)

        def _alias_override_safe() -> bool:
            if not _alias_shared_data():
                return False
            template = node1.get_template_node()
            if not isinstance(template, HelionTemplateBuffer):
                return False
            if _block_multi_mutation_epilogue(template, node2):
                return False
            if template.uses_atomics():
                return False
            if template.uses_internal_views() and template.mutated_inputs:
                return False
            direct_aliases = _get_direct_aliases(template)
            mutated_names = _collect_mutated_names(template)
            allow_mutation_epilogue = bool(direct_aliases & mutated_names)
            if node2.has_aliasing_or_mutation() and not allow_mutation_epilogue:
                return False
            epilogue_nodes = node2.get_nodes()
            if has_non_trivial_view(epilogue_nodes):
                return False
            epilogue_ir = _get_ir_node(epilogue_nodes[-1]) if epilogue_nodes else None
            if not isinstance(epilogue_ir, IRNode):
                return False
            if mutated_names and not (direct_aliases & mutated_names):
                if node2.read_writes and any(
                    isinstance(dep, (MemoryDep, StarDep, WeakDep))
                    and dep.name in mutated_names
                    for dep in node2.read_writes.reads
                ):
                    return False

            output_nodes = HelionFusionChoices._template_outputs(template)
            if template.mutated_inputs:
                for buf in template.mutated_inputs:
                    if isinstance(buf, IRNode):
                        output_nodes[buf.get_name()] = buf
            for out in template.outputs[1:]:
                if isinstance(out, MutationOutput):
                    mutation_bufs = out.get_mutation_buffers()
                    for buf in mutation_bufs:
                        if isinstance(buf, IRNode):
                            output_nodes[buf.get_name()] = buf
                            output_nodes[out.get_name()] = buf
                elif isinstance(out, IRNode):
                    output_nodes[out.get_name()] = out

            reads = {
                dep.name
                for dep in node2.read_writes.reads
                if isinstance(dep, (MemoryDep, StarDep, WeakDep))
                and dep.name in output_nodes
            }
            if not reads:
                return False
            for name in reads:
                kernel_out = output_nodes[name]
                if not same_shape_and_stride(kernel_out, epilogue_ir):
                    return False
            return True

        allow_mutation_epilogue = False
        if isinstance(node1.get_template_node(), HelionTemplateBuffer):
            template = node1.get_template_node()
            if _block_multi_mutation_epilogue(template, node2):
                allow_mutation_epilogue = False
            else:
                direct_aliases = _get_direct_aliases(template)
                mutated_names = _collect_mutated_names(template)
                if direct_aliases & mutated_names and node2.read_writes:
                    allow_mutation_epilogue = any(
                        isinstance(dep, (MemoryDep, StarDep, WeakDep))
                        and dep.name in mutated_names
                        for dep in node2.read_writes.reads
                    )
        orig_has_alias = None
        if node2.has_aliasing_or_mutation() and allow_mutation_epilogue:
            orig_has_alias = node2.has_aliasing_or_mutation
            node2.has_aliasing_or_mutation = lambda: False  # type: ignore[assignment]
        try:
            result = original_can_fuse(
                self,
                node1,
                node2,
                can_reorder=can_reorder,
                allow_mix_order_reduction=allow_mix_order_reduction,
            )
        finally:
            if orig_has_alias is not None:
                node2.has_aliasing_or_mutation = orig_has_alias  # type: ignore[assignment]
        if not result and _alias_override_safe():
            return True
        return result

    inductor_scheduler.Scheduler.can_fuse = helion_can_fuse
    _ORIG_SCHEDULER_CAN_FUSE = original_can_fuse
    _SCHEDULER_FUSION_PATCHED = True


class HelionFusionChoices(InductorChoices):
    """Disallow Helion fusion when view ops or shape/stride mismatches are present."""

    @staticmethod
    def _is_helion_template_node(node: BaseSchedulerNode) -> bool:
        return node.is_template() and isinstance(
            node.get_template_node(), HelionTemplateBuffer
        )

    @staticmethod
    def _template_outputs(template: HelionTemplateBuffer) -> dict[str, IRNode]:
        output_nodes: dict[str, IRNode] = {template.get_name(): template}
        # HelionTemplateBuffer always has multi_output_nodes (initialized to [] in __init__)
        for mo in template.multi_output_nodes:
            if isinstance(mo, IRNode):
                output_nodes[mo.get_name()] = mo
        return output_nodes

    @staticmethod
    def _template_inputs(template: HelionTemplateBuffer) -> dict[str, IRNode]:
        input_nodes: dict[str, IRNode] = {}
        for inp in template.inputs:  # type: ignore[union-attr]
            if isinstance(inp, IRNode):
                input_nodes[inp.get_name()] = inp
        return input_nodes

    @staticmethod
    def _is_supported_view_input(inp: IRNode) -> bool:
        if not isinstance(inp, BaseView):
            return True
        if isinstance(inp, ReinterpretView):
            base = inp.unwrap_view()
            if isinstance(base, IRNode):
                return same_shape_and_stride(inp, base)
        return False

    @staticmethod
    def can_fuse(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        if node1.is_template() and not node2.is_template():
            template = node1.get_template_node()
            if isinstance(template, HelionTemplateBuffer):
                if _block_multi_mutation_epilogue(template, node2):
                    return False
                if template.uses_atomics():
                    return False
                if template.uses_internal_views() and template.mutated_inputs:
                    return False
                direct_aliases = _get_direct_aliases(template)
                mutated_names = _collect_mutated_names(template)
                if mutated_names and not (direct_aliases & mutated_names):
                    if node2.read_writes and any(
                        isinstance(dep, (MemoryDep, StarDep, WeakDep))
                        and dep.name in mutated_names
                        for dep in node2.read_writes.reads
                    ):
                        return False
                output_nodes = HelionFusionChoices._template_outputs(template)
                reads = {
                    dep.name
                    for dep in node2.read_writes.reads
                    if isinstance(dep, (MemoryDep, StarDep, WeakDep))
                    and dep.name in output_nodes
                }
                if reads:
                    epilogue_nodes = node2.get_nodes()
                    if has_non_trivial_view(epilogue_nodes):
                        return False
                    epilogue_ir = (
                        _get_ir_node(epilogue_nodes[-1]) if epilogue_nodes else None
                    )
                    if isinstance(epilogue_ir, IRNode):
                        for name in reads:
                            # name is guaranteed to be in output_nodes (filtered above)
                            kernel_out = output_nodes[name]
                            if not isinstance(kernel_out, IRNode):
                                return False
                            if not same_shape_and_stride(kernel_out, epilogue_ir):
                                return False

        if node2.is_template() and not node1.is_template():
            template = node2.get_template_node()
            if isinstance(template, HelionTemplateBuffer):
                if template.uses_internal_views():
                    return False
                prologue_nodes = node1.get_nodes()
                for snode in prologue_nodes:
                    if not isinstance(_get_ir_node(snode), ComputedBuffer):
                        return False
                if has_view(prologue_nodes):
                    return False
                template_reads = node2.used_buffer_names()
                produced = node1.get_buffer_names() & template_reads
                if len(produced) != 1:
                    return False
                out_name = next(iter(produced))
                out_buf = V.graph.get_buffer(out_name)
                template_inputs = HelionFusionChoices._template_inputs(template)
                if any(
                    not HelionFusionChoices._is_supported_view_input(inp)
                    for inp in template_inputs.values()
                ):
                    return False
                if out_name not in template_inputs:
                    return False
                template_inp = template_inputs[out_name]
                if isinstance(out_buf, IRNode) and isinstance(template_inp, IRNode):
                    if not same_shape_and_stride(template_inp, out_buf):
                        return False
                    if node1.read_writes:
                        has_matching_input = False
                        for dep in node1.read_writes.reads:
                            if not isinstance(dep, (MemoryDep, StarDep, WeakDep)):
                                continue
                            if dep.name == out_name:
                                continue
                            buf = V.graph.get_buffer(dep.name)
                            if isinstance(buf, BaseView):
                                return False
                            if isinstance(buf, IRNode) and same_shape_and_stride(
                                buf, out_buf
                            ):
                                has_matching_input = True
                                break
                        if not has_matching_input:
                            return False
                    else:
                        return False

        if shared_data_score == 0 and (
            HelionFusionChoices._is_helion_template_node(node1)
            or HelionFusionChoices._is_helion_template_node(node2)
        ):
            shared_data_score = 1
        return InductorChoices.can_fuse(scheduler, node1, node2, shared_data_score)


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

    saved_lowerings = _inductor_lowering.lowerings.copy()
    # Ensure fusion decisions respect Helion view/shape constraints.
    V.set_choices_handler(HelionFusionChoices())
    try:
        _inductor_lowering.lowerings.update(inductor_lowering_dispatch)
        _patch_scheduler_fusable_read_and_write()
        yield
    finally:
        _inductor_lowering.lowerings = saved_lowerings


register_inductor_lowering = _inductor_lowering.register_lowering


def var_mean_helper_(
    x: TensorBox,
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
) -> TensorBox | tuple[TensorBox, ...]:
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
    return output[0] if not return_mean else output


@register_inductor_lowering(
    [torch.ops.aten.var.correction],
    lowering_dict=inductor_lowering_dispatch,
)
def var_(
    x: TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> TensorBox | tuple[TensorBox, ...]:
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
    x: TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> TensorBox | tuple[TensorBox, ...]:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=True,
    )


def _register_helion_kernel_lowering() -> None:
    """Register the Helion kernel lowering with Inductor.

    This is done in a function to avoid circular imports at module load time.
    """

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
        layout: OutputSpec
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
