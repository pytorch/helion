"""HelionTemplateBuffer - IR node for Helion kernels in Inductor."""

from __future__ import annotations

import ast
import contextlib
from itertools import dropwhile
from typing import Any, Sequence, cast

from torch._inductor import config as inductor_fusion_config
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.ir import (
    IRNode, Layout, MultiOutput, MultiOutputLayout,
    OutputSpec, ReinterpretView, TritonTemplateBuffer,
)
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..output_header import get_needed_imports, library_imports


class HelionTemplateBuffer(TritonTemplateBuffer):
    """Helion kernel IR node for torch.compile integration."""

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        kernel: Any,  # noqa: ANN401
        kernel_idx: int,
        constant_args: dict[str, Any],
        tensor_arg_names: list[str],
        bound_kernel: Any,  # noqa: ANN401
        mutated_inputs: Sequence[IRNode] | None = None,
        output_aliases: list[str | None] | None = None,
        output_alias_is_direct: list[bool] | None = None,
        autotune_args: tuple[Any, ...] | None = None,
    ) -> None:
        # Required by PyTorch inductor's scheduler (matches Kernel class in common.py)
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()
        self.removed_buffers: OrderedSet[str] = OrderedSet()
        self.inplaced_to_remove: OrderedSet[str] = OrderedSet()

        self.named_input_nodes = {name: inputs[i] for i, name in enumerate(tensor_arg_names) if i < len(inputs)}
        self.kernel_name: str | None = None
        self.multi_output_nodes: list[MultiOutput] = []

        self._helion_kernel = kernel
        self._tensor_arg_names = tensor_arg_names
        self._bound_kernel = bound_kernel
        self._constant_args_dict = constant_args
        self._output_aliases = output_aliases or []
        self._output_alias_is_direct = output_alias_is_direct or []
        self._autotune_args = autotune_args

        super().__init__(
            layout=cast("Layout", layout), inputs=inputs,
            make_kernel_render=lambda tb, hint_override=None: (self, self.render),
            mutated_inputs=mutated_inputs, allowed_prologue_inps=OrderedSet(),
        )
        if mutated_inputs:
            never_reuse = V.graph.never_reuse_buffers
            never_reuse.add(self.get_name())
            never_reuse.update(buf.get_name() for buf in mutated_inputs if isinstance(buf, IRNode))

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

        # Generate Python AST for Triton kernel
        with self._bound_kernel.env:
            root = generate_ast(
                self._bound_kernel.host_function, cfg, emit_repro_caller=False
            )

        # Rename host function to kernel name placeholder
        fn_node = next((n for n in ast.walk(root) if isinstance(n, ast.FunctionDef) and n.name == host_fn), None)
        if fn_node:
            fn_node.name = str(Placeholder.KERNEL_NAME)

        # Unparse AST to Triton source code
        triton_code = get_needed_imports(root) + unparse(
            root, output_origin_lines=self._bound_kernel.settings.output_origin_lines
        )
        return PartialRender(triton_code, {})

    def call_kernel(self, kernel_name: str, template_buffer: Any = None) -> None:  # noqa: ANN401
        """Emit the kernel call site."""
        wrapper, output_name, reinterp_idx = V.graph.wrapper_code, self.get_name(), [0]

        def get_input_expr(arg_name: str, inp: IRNode) -> str:
            if not isinstance(inp, ReinterpretView):
                return inp.get_name()  # type: ignore[union-attr]
            expr = wrapper.codegen_reinterpret_view(
                inp.data, list(inp.get_size()), list(inp.get_stride()), inp.layout.offset, wrapper.writeline
            )
            if expr != inp.data.get_name():
                wrapper.writeline(f"reinterp_{reinterp_idx[0]} = {expr}")
                expr = f"reinterp_{reinterp_idx[0]}"
                reinterp_idx[0] += 1
            return expr

        arg_inputs = {name: get_input_expr(name, inp) for name, inp in self.named_input_nodes.items()}
        sig = self._helion_kernel.signature.parameters
        args = [
            arg_inputs.get(n) or repr(self._constant_args_dict.get(n, sig[n].default))
            for n in sig if n in arg_inputs or n in self._constant_args_dict or sig[n].default is not sig[n].empty
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
        scheduling: Any,
        template_node: Any,
        epilogue_nodes: Any,
        prologue_nodes: Any,
        buf_name_to_prologue_group: Any,
        prologue_preserves_zero_mask_fn: Any,
        render: Any,
        only_gen_src_code: Any,
    ):
        """Entry point for template codegen called by Inductor scheduler."""
        # Extract MultiOutput nodes from epilogue (may be pre-populated in inductor_lowering_extra.py)
        mo_nodes = [
            getattr(n, "node", n) for n in (epilogue_nodes or [])
            if isinstance(getattr(n, "node", n), MultiOutput)
        ]
        if mo_nodes:
            self.multi_output_nodes = mo_nodes

        with V.set_kernel_handler(self):
            if not only_gen_src_code:
                for node in [template_node, *(n for n in epilogue_nodes or [] if getattr(n, "node", n) in mo_nodes)]:
                    node.mark_run()
            partial_code = render()
            src_code = partial_code if isinstance(partial_code, str) else partial_code.finalize_remaining()
            if inductor_fusion_config.benchmark_kernel:
                src_code = f"\n{src_code}\n{IndentedBuffer().getvalue()}"
            if only_gen_src_code:
                return src_code
            self.kernel_name = scheduling.define_kernel(src_code, [template_node], self)
        return self

    def emit_kernel_override(
        self,
        wrapper: Any,
        src_code: Any,
        kernel_name: Any,
        node_schedule: Any,
        kernel_path: Any,
        get_kernel_metadata: Any,
    ):
        """Entry point for kernel emission."""
        required = ("triton", "tl", "_default_launcher")
        conditional = ("libdevice", "tl_math", "triton_helpers", "helion")
        for name in (*required, *(n for n in conditional if f"{n}." in src_code)):
            wrapper.add_import_once(library_imports[name])

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        wrapper.header.writeline(f"# kernel path: {kernel_path}\n{origins}\n{detailed}")

        # Skip import lines at the beginning
        for line in dropwhile(
            lambda l: (s := l.strip()).startswith(("from __future__", "import ", "from ")) or not s,
            src_code.split("\n"),
        ):
            wrapper.header.writeline(line)
        wrapper.header.writeline("")
        return True

    def set_current_node(self, node: Any) -> contextlib.nullcontext[None]:
        """Set current node for codegen context."""
        return contextlib.nullcontext()
