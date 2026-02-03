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
from torch._inductor.fx_passes.post_grad import PatternMatcherPass

# pyrefly: ignore [missing-module-attribute]
from torch._inductor.fx_passes.post_grad import apply_pass_to_subgraphs
from torch._inductor.fx_passes.post_grad import register_graph_pattern
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
from torch._inductor.pattern_matcher import CallFunctionVarArgs
from torch._inductor.pattern_matcher import Match
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
import torch.fx
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree

from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional_dense
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
        mutated_inputs: Sequence[IRNode] | None = None,
        autotune_args: tuple[object, ...] | None = None,
    ) -> None:
        # Required by PyTorch inductor's scheduler (matches Kernel class in common.py)
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()
        self.removed_buffers: OrderedSet[str] = OrderedSet()
        self.inplaced_to_remove: OrderedSet[str] = OrderedSet()

        self.named_input_nodes = {
            name: inputs[i]
            for i, name in enumerate(tensor_arg_names)
            if i < len(inputs)
        }
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
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=OrderedSet(),
        )
        if mutated_inputs:
            never_reuse = V.graph.never_reuse_buffers
            never_reuse.add(self.get_name())
            never_reuse.update(
                buf.get_name() for buf in mutated_inputs if isinstance(buf, IRNode)
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
        # Use KERNEL_NAME placeholder for inner kernel so it gets a unique name
        inner_fn_placeholder = f"_helion_{Placeholder.KERNEL_NAME}"

        # Generate Python AST for Triton kernel
        with self._bound_kernel.env:
            host_function = self._bound_kernel.host_function
            assert host_function is not None, "BoundKernel must have a host_function"
            root = generate_ast(host_function, cfg, emit_repro_caller=False)

        # Rename host function to kernel name placeholder
        fn_node = next(
            (
                n
                for n in ast.walk(root)
                if isinstance(n, ast.FunctionDef) and n.name == host_fn
            ),
            None,
        )
        if fn_node:
            fn_node.name = str(Placeholder.KERNEL_NAME)

        # Rename inner Triton kernel function to avoid name collisions
        # when the same kernel is called multiple times with different shapes
        inner_fn_node = next(
            (
                n
                for n in ast.walk(root)
                if isinstance(n, ast.FunctionDef) and n.name == inner_fn
            ),
            None,
        )
        if inner_fn_node:
            inner_fn_node.name = inner_fn_placeholder

        # Update references to the inner kernel in the host function
        for node in ast.walk(root):
            if isinstance(node, ast.Name) and node.id == inner_fn:
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
        wrapper, output_name, reinterp_idx = V.graph.wrapper_code, self.get_name(), [0]

        def get_input_expr(arg_name: str, inp: IRNode) -> str:
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
                wrapper.writeline(f"reinterp_{reinterp_idx[0]} = {expr}")
                expr = f"reinterp_{reinterp_idx[0]}"
                reinterp_idx[0] += 1
            return expr

        arg_inputs = {
            name: get_input_expr(name, inp)
            for name, inp in self.named_input_nodes.items()
        }
        sig = self._helion_kernel.signature.parameters
        args = [
            arg_inputs.get(n) or repr(self._constant_args_dict.get(n, sig[n].default))
            for n in sig
            if n in arg_inputs
            or n in self._constant_args_dict
            or sig[n].default is not sig[n].empty
        ]
        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

        # Handle multi-output unpacking
        if isinstance(self.layout, MultiOutputLayout):
            for mo in self.multi_output_nodes:
                if isinstance(mo, MultiOutput) and mo.indices:
                    idx_str = output_name + "".join(f"[{idx}]" for _, idx in mo.indices)
                    wrapper.writeline(f"{mo.get_name()} = {idx_str}")

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
        # Extract MultiOutput nodes from epilogue
        mo_nodes: list[MultiOutput] = [
            cast("MultiOutput", getattr(n, "node", n))
            for n in (epilogue_nodes or [])
            if isinstance(getattr(n, "node", n), MultiOutput)
        ]
        if mo_nodes:
            self.multi_output_nodes = mo_nodes

        with V.set_kernel_handler(self):
            if not only_gen_src_code:
                for node in [
                    template_node,
                    *(
                        n
                        for n in epilogue_nodes or []
                        if getattr(n, "node", n) in mo_nodes
                    ),
                ]:
                    node.mark_run()
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

        # Add typing import for type annotations used in function signatures
        # This is needed because Inductor doesn't use 'from __future__ import annotations'
        wrapper.add_import_once("from typing import Optional")

        # Add imports for captured global variables (e.g., "import __main__ as _source_module")
        # These are tracked in HostFunction.global_imports during kernel compilation
        if self._bound_kernel.host_function is not None:
            for (
                global_import
            ) in self._bound_kernel.host_function.global_imports.values():
                wrapper.add_import_once(global_import.codegen())

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


def decompose_helion_kernel_wrapper_functional(graph: torch.fx.Graph) -> None:
    """Decomposes helion_kernel_wrapper_functional nodes into clones and mutation HOP.

    This pass converts functional HOP nodes to the dense implementation which
    handles cloning of mutated inputs and calls the underlying mutation HOP.
    """
    # First, recursively process any subgraphs
    apply_pass_to_subgraphs(decompose_helion_kernel_wrapper_functional, graph)

    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(helion_kernel_wrapper_functional),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args: object, **kwargs: object) -> None:
        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is because replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args: object) -> tuple[object, ...]:
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            return (helion_kernel_wrapper_functional_dense(*args, **kwargs),)

        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    graph_pass.apply(graph)

    for _ in graph.find_nodes(
        op="call_function", target=helion_kernel_wrapper_functional
    ):
        raise AssertionError("helion_kernel_wrapper_functional was not removed")


# Install Helion's decompose pass via Inductor's post_grad_custom_post_pass config.
# This uses the official Inductor extension point to run Helion's decompose pass,
# converting helion_kernel_wrapper_functional to clones + mutation HOP.
_original_post_pass = inductor_fusion_config.post_grad_custom_post_pass


def _helion_post_grad_pass(graph: torch.fx.Graph) -> None:
    """Run Helion's passes, chaining with any existing custom pass."""
    if _original_post_pass is not None:
        _original_post_pass(graph)
    decompose_helion_kernel_wrapper_functional(graph)


inductor_fusion_config.post_grad_custom_post_pass = _helion_post_grad_pass


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
    aliases = cast("list[str | None]", output_spec.get("output_aliases", []))
    direct_flags = cast("list[bool]", output_spec.get("output_alias_is_direct", []))

    # Realize inputs: convert TensorBox to buffer/ReinterpretView
    def realize(tb: TensorBox) -> IRNode:
        result = ExternKernel.realize_input(tb)
        if isinstance(result, StorageBox):
            result = result.data
        if isinstance(getattr(result, "layout", None), FlexibleLayout):
            result.freeze_layout()
        return result

    realized = {
        n: realize(tb) for n, tb in tensor_args.items() if isinstance(tb, TensorBox)
    }

    # Build ordered arg_names and inputs lists from realized
    arg_names = list(realized.keys())
    inputs = list(realized.values())

    # Build fake tensors for kernel binding
    fake_tensors = []
    for name, param in kernel.signature.parameters.items():
        if name in realized:
            inp = realized[name]
            size = [
                int(s) if isinstance(s, (int, sympy.Integer)) else 64
                for s in inp.get_size()
            ]
            stride = [
                int(s) if isinstance(s, (int, sympy.Integer)) else 1
                for s in inp.get_stride()
            ]
            fake_tensors.append(
                torch.empty_strided(
                    size, stride, dtype=inp.get_dtype(), device=inp.get_device()
                )
            )
        elif name in constant_args:
            fake_tensors.append(constant_args[name])
        elif param.default is not param.empty:
            fake_tensors.append(param.default)
    bound = kernel.bind(tuple(fake_tensors))

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

    # Build HelionTemplateBuffer
    mutated = [
        inputs[arg_names.index(n)]
        for n in cast("list[str]", output_spec.get("mutated_inputs", []))
        if n in arg_names
    ]
    buf = HelionTemplateBuffer(
        layout=layout,
        inputs=inputs,
        kernel=kernel,
        constant_args=constant_args,
        tensor_arg_names=arg_names,
        bound_kernel=bound,
        mutated_inputs=mutated or None,
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
                    alias_device = alias_inp.get_device()
                    assert alias_device is not None
                    view_layout = FixedLayout(
                        device=alias_device,
                        dtype=alias_inp.get_dtype(),
                        size=[
                            sympy.Integer(s)
                            for s in spec["shape"]  # pyrefly: ignore[not-iterable]
                        ],
                        stride=[
                            sympy.Integer(s)
                            for s in spec["stride"]  # pyrefly: ignore[not-iterable]
                        ],
                        offset=sympy.Integer(spec.get("storage_offset", 0)),
                    )
                    storage = (
                        alias_inp
                        if isinstance(alias_inp, StorageBox)
                        else StorageBox(alias_inp)
                    )
                    results.append(
                        TensorBox.create(
                            ReinterpretView(data=storage, layout=view_layout)
                        )
                    )
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
