from __future__ import annotations

import ast
import contextlib
from itertools import dropwhile
from typing import TYPE_CHECKING
from typing import Callable
from typing import Sequence
from typing import cast

import sympy
import torch
from torch._inductor import config as inductor_fusion_config
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.ir import ExternKernel
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import FlexibleLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import Layout
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import OutputSpec
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TensorBox
from torch._inductor.ir import TritonTemplateBuffer
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..output_header import get_needed_imports
from ..output_header import library_imports

if TYPE_CHECKING:
    from torch._inductor.codegen.simd import SIMDScheduling
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.scheduler import BaseSchedulerNode

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


class HelionTemplateBuffer(TritonTemplateBuffer):
    """Helion kernel IR node for torch.compile integration."""

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        kernel: Kernel,
        constant_args: dict[str, object],
        tensor_arg_names: list[str],
        bound_kernel: BoundKernel,
        autotune_args: tuple[object, ...] | None = None,
    ) -> None:
        # Required by Inductor scheduler
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()
        self.removed_buffers: OrderedSet[str] = OrderedSet()
        self.inplaced_to_remove: OrderedSet[str] = OrderedSet()

        self.named_input_nodes = dict(zip(tensor_arg_names, inputs, strict=True))
        self.kernel_name: str | None = None
        self.multi_output_nodes: list[MultiOutput] = []

        self._helion_kernel = kernel
        self._bound_kernel = bound_kernel
        self._constant_args_dict = constant_args
        self._autotune_args = autotune_args

        super().__init__(
            layout=cast("Layout", layout),
            inputs=inputs,
            make_kernel_render=lambda tb, hint_override=None: (self, self.render),
            mutated_inputs=None,
            allowed_prologue_inps=OrderedSet(),
        )

    def render(self) -> PartialRender:
        """Generate Triton code."""
        if not self._bound_kernel:
            return PartialRender("", {})
        # Ensure config is available (triggers autotuning if needed)
        if self._autotune_args:
            self._bound_kernel.ensure_config_exists(self._autotune_args)
        cfg = self._bound_kernel._config
        assert cfg is not None, "Config should be set after ensure_config_exists"
        host_fn = self._helion_kernel.name
        inner_fn = f"_helion_{host_fn}"
        inner_fn_placeholder = f"{inner_fn}_{Placeholder.KERNEL_NAME}"

        # Generate Python AST for Triton kernel
        with self._bound_kernel.env:
            host_function = self._bound_kernel.host_function
            assert host_function is not None, "BoundKernel must have a host_function"
            root = generate_ast(host_function, cfg, emit_repro_caller=False)

        # Rename functions and update references in a single AST walk
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                if node.name == host_fn:
                    node.name = str(Placeholder.KERNEL_NAME)
                elif node.name == inner_fn:
                    node.name = inner_fn_placeholder
            elif isinstance(node, ast.Name) and node.id == inner_fn:
                node.id = inner_fn_placeholder

        # Unparse AST to Triton source code
        triton_code = get_needed_imports(root) + unparse(
            root, output_origin_lines=self._bound_kernel.settings.output_origin_lines
        )
        return PartialRender(triton_code, {})

    def call_kernel(
        self, kernel_name: str, template_buffer: TritonTemplateBuffer | None = None
    ) -> None:
        """Emit the kernel call site."""
        wrapper = V.graph.wrapper_code
        output_name = self.get_name()
        reinterp_count = 0

        def get_input_expr(arg_name: str, inp: IRNode) -> str:
            nonlocal reinterp_count
            if not isinstance(inp, ReinterpretView):
                return inp.get_name()  # type: ignore[union-attr]
            expr = wrapper.codegen_reinterpret_view(
                inp.data,
                list(inp.get_size()),
                list(inp.get_stride()),
                inp.layout.offset,
                wrapper.writeline,
            )
            if expr != inp.data.get_name():
                wrapper.writeline(f"reinterp_{reinterp_count} = {expr}")
                expr = f"reinterp_{reinterp_count}"
                reinterp_count += 1
            return expr

        arg_inputs = {
            name: get_input_expr(name, inp)
            for name, inp in self.named_input_nodes.items()
        }
        sig = self._helion_kernel.signature.parameters
        args = [
            arg_inputs.get(n, repr(self._constant_args_dict.get(n, p.default)))
            for n, p in sig.items()
            if n in arg_inputs
            or n in self._constant_args_dict
            or p.default is not p.empty
        ]
        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

    def get_layout(self) -> Layout:
        """Get layout, handling multi-output case."""
        if isinstance(self.layout, MultiOutputLayout):
            mo = self.multi_output_nodes
            assert mo and isinstance(mo[0], MultiOutput), (
                "MultiOutputLayout without multi_output_nodes"
            )
            return cast("Layout", mo[0].layout)
        return super().get_layout()

    def codegen_template_override(
        self,
        scheduling: SIMDScheduling,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
        buf_name_to_prologue_group: dict[str, list[BaseSchedulerNode]],
        prologue_preserves_zero_mask_fn: Callable[[str], bool],
        render: Callable[[], PartialRender | str],
        only_gen_src_code: bool,
    ) -> HelionTemplateBuffer | str:
        """Entry point for template codegen called by Inductor scheduler."""
        with V.set_kernel_handler(self):
            if not only_gen_src_code:
                template_node.mark_run()
            partial_code = render()
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize_remaining()
            )
            if inductor_fusion_config.benchmark_kernel:
                src_code = f"\n{src_code}\n{IndentedBuffer().getvalue()}"
            if only_gen_src_code:
                return src_code
            self.kernel_name = scheduling.define_kernel(src_code, [template_node], self)
        return self

    def emit_kernel_override(
        self,
        wrapper: PythonWrapperCodegen,
        src_code: str,
        kernel_name: str,
        node_schedule: Sequence[BaseSchedulerNode | object],
        kernel_path: str,
        get_kernel_metadata: Callable[
            [Sequence[BaseSchedulerNode | object], PythonWrapperCodegen],
            tuple[str, str],
        ],
    ) -> bool:
        """Entry point for kernel emission."""
        required = ("triton", "tl", "_default_launcher")
        conditional = ("libdevice", "tl_math", "triton_helpers", "helion", "hl")
        for name in (*required, *(n for n in conditional if f"{n}." in src_code)):
            wrapper.add_import_once(library_imports[name])

        # Add imports for captured global variables (e.g., "import __main__ as _source_module")
        # These are tracked in HostFunction.global_imports during kernel compilation
        if self._bound_kernel.host_function is not None:
            for imp in self._bound_kernel.host_function.global_imports.values():
                wrapper.add_import_once(imp.codegen())

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        wrapper.header.writeline(f"# kernel path: {kernel_path}\n{origins}\n{detailed}")

        # Skip import lines at the beginning
        for line in dropwhile(
            lambda ln: (s := ln.strip()).startswith(
                ("from __future__", "import ", "from ")
            )
            or not s,
            src_code.split("\n"),
        ):
            wrapper.header.writeline(line)
        wrapper.header.writeline("")
        return True

    def set_current_node(self, node: BaseSchedulerNode) -> contextlib.nullcontext[None]:
        """Set current node for codegen context."""
        return contextlib.nullcontext()


@register_lowering(helion_kernel_wrapper_mutation, type_promotion_kind=None)
def lower_helion_kernel(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, object],
) -> tuple[TensorBox | int | float | None, ...]:
    """Lower a Helion kernel call to HelionTemplateBuffer."""
    kernel = get_helion_kernel(kernel_idx)

    # Extract output spec components
    num_outputs = cast("int", output_spec.get("num_outputs", 0))
    specs = cast("list[dict[str, object] | None]", output_spec.get("output_specs", []))

    # Realize inputs: convert TensorBox to buffer/ReinterpretView
    realized: dict[str, IRNode] = {}
    for n, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            result = ExternKernel.realize_input(tb)
            if isinstance(result, StorageBox):
                result = result.data
            if isinstance(result.layout, FlexibleLayout):  # type: ignore[union-attr]
                result.freeze_layout()
            realized[n] = result

    # Build fake tensors for kernel binding (sympy exprs -> concrete ints)
    def as_int(x: object, default: int) -> int:
        return int(x) if isinstance(x, (int, sympy.Integer)) else default

    fake_tensors: list[object] = [
        torch.empty_strided(
            [as_int(s, 64) for s in realized[n].get_size()],
            [as_int(s, 1) for s in realized[n].get_stride()],
            dtype=realized[n].get_dtype(),
            device=realized[n].get_device(),
        )
        if n in realized
        else constant_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in realized or n in constant_args or p.default is not p.empty
    ]
    bound = kernel.bind(tuple(fake_tensors))

    inputs = list(realized.values())

    def make_layout(idx: int) -> FixedLayout | None:
        """Create FixedLayout from output spec at given index, or None for scalars."""
        spec = specs[idx] if idx < len(specs) else None
        if spec is None or "shape" not in spec:
            return None
        # Include stride if available to preserve non-contiguous layouts
        stride = cast("list[int] | None", spec.get("stride"))
        return FixedLayout(
            device=torch.device(  # pyrefly: ignore[no-matching-overload]
                spec["device"]
            ),
            dtype=spec["dtype"],  # pyrefly: ignore[bad-argument-type]
            size=cast("list[sympy.Expr]", spec["shape"]),
            stride=[sympy.Integer(s) for s in stride] if stride else None,
        )

    # Determine buffer layout
    if num_outputs <= 0:
        # Void-returning kernel: use a minimal layout.
        # The buffer still executes mutations but produces no output tensor.
        # num_outputs is -1 when kernel has no return statement.
        device = torch.device("cuda")  # Default device
        if inputs:
            inp_device = inputs[0].get_device()
            if inp_device is not None:
                device = inp_device
        layout = FixedLayout(
            device=device,
            dtype=torch.float32,
            size=[sympy.Integer(1)],  # Minimal size
        )
    elif num_outputs == 1:
        layout = make_layout(0)
        if layout is None:
            raise ValueError("Single-output kernel must return a tensor, not a scalar")
    else:
        # Check if all outputs are scalars (no tensor outputs)
        has_tensor_output = any(make_layout(i) is not None for i in range(num_outputs))
        if not has_tensor_output:
            raise ValueError(
                "Kernels that return only scalars (no tensors) are not supported "
                "with torch.compile. Return at least one tensor, or use the kernel "
                "outside of torch.compile."
            )
        device = next(
            (
                torch.device(s["device"])  # pyrefly: ignore[no-matching-overload]
                for s in specs
                if s and "device" in s
            ),
            torch.device("cuda"),
        )
        layout = MultiOutputLayout(device=device)

    buf = HelionTemplateBuffer(
        layout=layout,
        inputs=inputs,
        kernel=kernel,
        constant_args=constant_args,
        tensor_arg_names=list(realized.keys()),
        bound_kernel=bound,
        autotune_args=tuple(fake_tensors),
    )
    V.graph.no_fuse_buffer_names.add(buf.get_name())  # Disable fusion for now

    # Build output results
    results: list[TensorBox | int | float | None] = []
    multi_output_nodes: list[MultiOutput] = []

    for i in range(num_outputs):
        out_layout = make_layout(i)
        if out_layout is not None:
            if num_outputs == 1:
                results.append(TensorBox(StorageBox(buf)))
            else:
                mo = MultiOutput(layout=out_layout, input=buf, indices=[(tuple, i)])
                multi_output_nodes.append(mo)
                results.append(TensorBox.create(mo))
        else:
            # Scalar output - return None since scalars are handled at Dynamo level
            # as ConstantVariable. The HOP output is never used for scalar positions.
            results.append(None)

    if num_outputs > 1:
        buf.multi_output_nodes = multi_output_nodes
        if not multi_output_nodes:
            # All tensor outputs are aliases - find first tensor output's layout
            # (skip scalars which return None from make_layout)
            for i in range(num_outputs):
                fallback = make_layout(i)
                if fallback:
                    buf.layout = fallback
                    break

    if num_outputs <= 0:
        # Void-returning kernel: return (None,) to indicate no return value.
        # The buffer is still created above for mutation side effects.
        # num_outputs is -1 when kernel has no return statement.
        return (None,)

    return tuple(results)
