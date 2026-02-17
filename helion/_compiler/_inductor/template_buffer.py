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
from torch._inductor import dependencies
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.ir import ExternKernel
from torch._inductor.ir import FallbackKernel
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
import torch.utils._pytree as pytree

from .._dynamo.higher_order_ops import _rebuild_container_args
from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional
from .._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from .._dynamo.variables import _get_flat_output
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


class _CodeExpr(str):
    """A str whose repr() returns itself, for embedding variable names in generated code.

    When generating a kernel call like ``kernel(x, (a, b))``, container args are
    rebuilt via pytree into e.g. ``(_CodeExpr("a"), _CodeExpr("b"))``.  Python's
    built-in ``repr()`` on that tuple then produces ``(a, b)`` instead of
    ``('a', 'b')``, giving us correct code for free.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


class HelionTemplateBuffer(TritonTemplateBuffer):
    """Inductor template buffer for Helion kernel."""

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        kernel: Kernel,
        constant_args: dict[str, object],
        tensor_arg_names: list[str],
        bound_kernel: BoundKernel,
        mutated_input_names: list[str] | None = None,
        autotune_args: tuple[object, ...] | None = None,
    ) -> None:
        # Required by Inductor scheduler
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()
        self.removed_buffers: OrderedSet[str] = OrderedSet()
        self.inplaced_to_remove: OrderedSet[str] = OrderedSet()

        self.named_input_nodes = dict(zip(tensor_arg_names, inputs, strict=True))
        self.kernel_name: str | None = None
        self._helion_kernel = kernel
        self._bound_kernel = bound_kernel
        self._constant_args_dict = constant_args
        self._autotune_args = autotune_args

        mutated_inputs_irnodes = [
            self.named_input_nodes[n]
            for n in (mutated_input_names or [])
            if n in self.named_input_nodes
        ] or None

        super().__init__(
            layout=cast("Layout", layout),
            inputs=inputs,
            make_kernel_render=lambda tb, hint_override=None: (self, self.render),
            mutated_inputs=mutated_inputs_irnodes,
            allowed_prologue_inps=OrderedSet(),
        )

        for inp in mutated_inputs_irnodes or []:
            if hasattr(inp, "get_name"):
                V.graph.never_reuse_buffers.add(inp.get_name())

    # Layout is always MultiOutputLayout: reads from inputs only,
    # writes go through MultiOutput children, no allocation needed.

    def extract_read_writes(self, normalize: bool = False) -> dependencies.ReadWrites:
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        for inp in self.inputs:
            name = inp.get_name()  # pyrefly: ignore[missing-attribute]
            reads.add(dependencies.StarDep(name))
        return dependencies.ReadWrites(
            reads=reads,
            writes=OrderedSet(),
            index_exprs=OrderedSet(),
            range_vars=None,
            var_ranges=None,
        )

    def should_allocate(self) -> bool:
        return False

    def get_size(self) -> Sequence[sympy.Expr]:
        return []

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

        all_args: dict[str, object] = {n: _CodeExpr(v) for n, v in arg_inputs.items()}
        for n, v in self._constant_args_dict.items():
            if n not in all_args:
                all_args[n] = v if n == "__container_specs" else _CodeExpr(repr(v))
        _rebuild_container_args(all_args)

        sig = self._helion_kernel.signature.parameters
        args = [
            repr(all_args[n]) if n in all_args else repr(p.default)
            for n, p in sig.items()
            if n in all_args or p.default is not p.empty
        ]
        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

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
            lambda ln: (
                (s := ln.strip()).startswith(("from __future__", "import ", "from "))
                or not s
            ),
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
) -> tuple[TensorBox, ...]:
    """Lower a Helion kernel call to HelionTemplateBuffer."""
    kernel = get_helion_kernel(kernel_idx)
    mutated_inputs_list = cast("list[str]", output_spec.get("mutated_inputs", []))

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

    all_args: dict[str, object] = {**constant_args}
    for n, r in realized.items():
        all_args[n] = torch.empty_strided(
            [as_int(s, 64) for s in r.get_size()],
            [as_int(s, 1) for s in r.get_stride()],
            dtype=r.get_dtype(),
            device=r.get_device(),
        )
    _rebuild_container_args(all_args)

    fake_tensors: list[object] = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    bound = kernel.bind(tuple(fake_tensors))
    inputs = list(realized.values())

    # Derive output structure from bound kernel using inductor-time input layouts.
    # This gives correct strides even when inductor changes input memory layouts.
    flat_leaves, tree_spec, _ = _get_flat_output(bound.host_function)
    example_outputs = [leaf for leaf in flat_leaves if isinstance(leaf, torch.Tensor)]

    # Create buffer for scheduling
    dev = (
        example_outputs[0].device
        if example_outputs
        else inputs[0].get_device()
        if inputs
        else torch.device("cuda")
    )
    assert dev is not None
    buf = HelionTemplateBuffer(
        layout=MultiOutputLayout(device=dev),
        inputs=inputs,
        kernel=kernel,
        constant_args=constant_args,
        tensor_arg_names=list(realized.keys()),
        bound_kernel=bound,
        mutated_input_names=mutated_inputs_list or None,
        autotune_args=tuple(fake_tensors),
    )
    V.graph.no_fuse_buffer_names.add(buf.get_name())

    if not example_outputs:
        return ()

    # Direct alias lookup: leaf_index -> input_name (for outputs identical to inputs)
    direct_alias_at_leaf = {
        i: name
        for i, name in cast(
            "dict[int, str]", output_spec.get("direct_aliases", {})
        ).items()
        if name in realized
    }

    # Reconstruct structured output and create MultiOutput nodes
    # (same pattern as FallbackKernel.generate_output in torch/_inductor/ir.py)
    assert tree_spec is not None
    structured = pytree.tree_unflatten(flat_leaves, tree_spec)

    # Walk structured output creating MultiOutput nodes
    leaf_counter = [0]
    # Track seen tensors by identity so duplicates reuse the same MultiOutput
    seen_outputs: dict[int, TensorBox] = {}

    def collect_tensor_outputs(
        output: object, indices: list[tuple[type, int]]
    ) -> list[TensorBox]:
        if isinstance(output, (list, tuple)):
            return [
                r
                for i in range(len(output))
                for r in collect_tensor_outputs(
                    output[i], [*indices, (type(output), i)]
                )
            ]
        leaf_idx = leaf_counter[0]
        leaf_counter[0] += 1
        if isinstance(output, torch.Tensor):
            if leaf_idx in direct_alias_at_leaf:
                return [TensorBox.create(realized[direct_alias_at_leaf[leaf_idx]])]
            tid = id(output)
            if tid in seen_outputs:
                return [seen_outputs[tid]]
            mo = MultiOutput(FallbackKernel.tensor_to_layout(output), buf, indices)
            tb = TensorBox(mo)
            seen_outputs[tid] = tb
            return [tb]
        return []

    return tuple(collect_tensor_outputs(structured, []))


@register_lowering(helion_kernel_wrapper_functional, type_promotion_kind=None)
def lower_helion_kernel_functional(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[TensorBox, ...], dict[str, TensorBox]]:
    from torch._inductor.lowering import clone

    cloned = {
        n: clone(tb) if n in tensors_to_clone and isinstance(tb, TensorBox) else tb
        for n, tb in tensor_args.items()
    }
    outputs = lower_helion_kernel(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args=cloned,
        output_spec=output_spec,
    )
    return (outputs, {n: cloned[n] for n in tensors_to_clone if n in cloned})
