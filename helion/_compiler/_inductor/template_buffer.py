"""HelionTemplateBuffer - IR node for Helion kernels in Inductor."""

from __future__ import annotations

import ast
import contextlib
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
from typing import cast
from typing_extensions import Self

import sympy
import torch
from torch._inductor import config as inductor_fusion_config
from torch._inductor import dependencies
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.dependencies import Dep
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import BaseView
from torch._inductor.ir import Buffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import FlexibleLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import InputBuffer
from torch._inductor.ir import Layout
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import MutationOutput
from torch._inductor.ir import OutputSpec
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TritonTemplateBuffer
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops
from torch.utils._ordered_set import OrderedSet

from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..output_header import get_needed_imports
from ..output_header import library_imports

if TYPE_CHECKING:
    from types import TracebackType


def _get_ir_node(n: Any) -> Any:  # noqa: ANN401
    """Extract the IR node from a scheduler node."""
    return n.node if isinstance(n, BaseSchedulerNode) else n


def _get_name(node: object) -> str:
    """Get the name of an IR node (typed wrapper for missing stubs)."""
    return node.get_name()  # type: ignore[union-attr]


def is_trivial_view(node: IRNode) -> bool:
    """Check if a view is a trivial reinterpret of its base."""
    if not isinstance(node, ReinterpretView):
        return False
    if node.layout.offset != 0:
        return False
    base = node.unwrap_view()
    return isinstance(base, IRNode) and same_shape_and_stride(node, base)


def same_shape_and_stride(lhs: IRNode, rhs: IRNode) -> bool:
    """Check if two IR nodes have the same shape and stride."""
    sizevars = V.graph.sizevars
    if not sizevars.statically_known_list_equals(
        list(lhs.get_size()), list(rhs.get_size())
    ):
        return False
    if isinstance(lhs, Buffer) and isinstance(rhs, Buffer):
        if not sizevars.statically_known_list_equals(
            list(lhs.get_stride()), list(rhs.get_stride())
        ):
            return False
    return True


class HelionTemplateBuffer(TritonTemplateBuffer):
    """Helion kernel IR node for torch.compile integration."""

    multi_output_nodes: list[MultiOutput]
    _helion_epilogue_aliases: list[str]

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
        # Required by PyTorch inductor's scheduler
        self.prologue_fused_inputs: set[str] = set()
        self.prologue_fused_inputs_preserve_zero: set[str] = set()
        self.removed_buffers: set[str] = set()
        self.inplaced_to_remove: set[str] = set()

        self.named_input_nodes = {
            name: inputs[i]
            for i, name in enumerate(tensor_arg_names)
            if i < len(inputs)
        }
        self.kernel_name: str | None = None

        self._helion_kernel = kernel
        self._tensor_arg_names = tensor_arg_names
        self._bound_kernel = bound_kernel
        self._constant_args_dict = constant_args
        self._output_aliases = output_aliases or []
        self._output_alias_is_direct = output_alias_is_direct or []
        self._autotune_args = autotune_args

        self.multi_output_nodes = []
        self._helion_epilogue_aliases = []

        # Fusion state (populated during enable_fusion context)
        self._captured_buffers: dict[str, Any] = {}
        self._fusion_stored_info: dict[str, Any] = {}
        self._epilogue_specs: dict[str, list[Any]] = {}  # accumulator_name -> nodes
        self._multi_dep_epilogue_specs: list[tuple[list[Any], set[str]]] = []
        self._uses_atomics_cache: bool | None = None

        super().__init__(
            layout=cast("Layout", layout),
            inputs=inputs,
            make_kernel_render=lambda tb, hint_override=None: (self, self.render),
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=OrderedSet(
                inp.get_name() for inp in inputs if isinstance(inp, IRNode)
            ),
        )
        if mutated_inputs:
            V.graph.never_reuse_buffers.add(self.get_name())
            for buf in mutated_inputs:
                if isinstance(buf, BaseView):
                    base = buf.unwrap_view()
                    if isinstance(base, IRNode):
                        V.graph.never_reuse_buffers.add(base.get_name())
                    V.graph.never_reuse_buffers.add(buf.get_name())
                elif isinstance(buf, IRNode):
                    V.graph.never_reuse_buffers.add(buf.get_name())

    def render(self) -> PartialRender:
        """Generate Triton code, optionally with fusion enabled."""
        if not self._bound_kernel:
            return PartialRender("", {})
        # Ensure config is available (triggers autotuning if needed)
        if self._autotune_args:
            self._bound_kernel.ensure_config_exists(self._autotune_args)
        cfg = self._bound_kernel.normalize_config()
        host_fn = self._helion_kernel.name

        # Check if fusion is enabled via the experimental flag
        kernel_settings = self._helion_kernel.settings
        use_fusion = getattr(
            kernel_settings, "_experimental_allow_torch_compile_fusion", False
        )

        # Generate AST with optional fusion
        with self._bound_kernel.env as env:
            if use_fusion:
                with env.enable_fusion(template_buffer=self):
                    root = generate_ast(
                        self._bound_kernel.host_function, cfg, emit_repro_caller=False
                    )
            else:
                root = generate_ast(
                    self._bound_kernel.host_function, cfg, emit_repro_caller=False
                )

        # Rename host function to kernel name placeholder
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef) and node.name == host_fn:
                node.name = str(Placeholder.KERNEL_NAME)
                break

        # Unparse AST to Triton source code
        triton_code = get_needed_imports(root) + unparse(
            root, output_origin_lines=self._bound_kernel.settings.output_origin_lines
        )
        return PartialRender(triton_code, {})

    def call_kernel(self, kernel_name: str, template_buffer: Any = None) -> None:  # noqa: ANN401
        """Emit the kernel call site."""
        wrapper, output_name = V.graph.wrapper_code, self.get_name()
        reinterpret_counter = 0

        def emit_reinterpret(
            base: str, size: tuple[int, ...], stride: tuple[int, ...], offset: int = 0
        ) -> str:
            nonlocal reinterpret_counter
            name = f"reinterp_{reinterpret_counter}"
            reinterpret_counter += 1
            wrapper.writeline(
                f"{name} = reinterpret_tensor({base}, {size}, {stride}, {offset})"
            )
            return name

        # Build arg-name -> input-name mapping
        arg_inputs: dict[str, str] = {}
        for arg_name, inp in self.named_input_nodes.items():
            if isinstance(inp, ReinterpretView):
                storage_name = _get_name(inp.data)
                if is_trivial_view(inp):
                    inp_name = storage_name
                else:
                    view_size = tuple(int(s) for s in inp.get_size())
                    view_stride = tuple(int(s) for s in inp.get_stride())
                    view_offset = int(inp.layout.offset)
                    inp_name = emit_reinterpret(
                        storage_name, view_size, view_stride, view_offset
                    )
            elif isinstance(inp, BaseView):
                storage_name = _get_name(inp.data)
                view_size = tuple(int(s) for s in inp.get_size())
                view_stride = tuple(
                    int(s) for s in FlexibleLayout.contiguous_strides(view_size)
                )
                inp_name = emit_reinterpret(storage_name, view_size, view_stride, 0)
            else:
                inp_name = _get_name(inp)
            arg_inputs[arg_name] = inp_name

        # Build args from signature
        args, sig = [], self._helion_kernel.signature.parameters
        for name in sig:
            if name in arg_inputs:
                args.append(arg_inputs[name])
            elif name in self._constant_args_dict:
                args.append(repr(self._constant_args_dict[name]))
            elif sig[name].default is not sig[name].empty:
                args.append(repr(sig[name].default))
        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

        # Handle multi-output unpacking
        if isinstance(self.layout, MultiOutputLayout):
            for mo in self.multi_output_nodes:
                if isinstance(mo, MultiOutput) and mo.indices:
                    n = mo.get_name()
                    idx_str = output_name
                    for _, idx in mo.indices:
                        idx_str = f"{idx_str}[{idx}]"
                    wrapper.writeline(f"{n} = {idx_str}")

    def get_layout(self) -> Layout:
        """Get layout, handling multi-output case."""
        if isinstance(self.layout, MultiOutputLayout):
            mo = self.multi_output_nodes
            assert mo and isinstance(mo[0], MultiOutput), (
                "MultiOutputLayout without multi_output_nodes"
            )
            return cast("Layout", mo[0].layout)
        return super().get_layout()

    @property
    def _fusion_store_map(self) -> dict[int, str]:
        """Compute fusion store map from multi_output_nodes."""
        if isinstance(self.layout, MultiOutputLayout):
            return {
                i: mo.get_name()
                for i, mo in enumerate(self.multi_output_nodes)
                if isinstance(mo, MultiOutput)
            }
        return {}

    def uses_atomics(self) -> bool:
        """Check if this kernel uses atomic operations.

        Atomics prevent epilogue fusion because the store order matters.
        Detects atomics by checking the FX graph for atomic operation nodes.
        """
        if self._uses_atomics_cache is not None:
            return self._uses_atomics_cache
        if not self._bound_kernel:
            self._uses_atomics_cache = True
            return True
        try:
            from ..host_function import HostFunction

            graph = self._bound_kernel.host_function._call_graph
            for n in graph.nodes:
                if n.op == "call_function" and hasattr(n.target, "__name__"):
                    if "atomic" in n.target.__name__.lower():
                        self._uses_atomics_cache = True
                        return True
        except Exception:
            # If we can't determine, assume no atomics
            pass
        self._uses_atomics_cache = False
        return self._uses_atomics_cache

    def extract_read_writes(self, normalize: bool = False):  # noqa: ANN201
        """Extract read/write dependencies for scheduling."""
        deps = super().extract_read_writes(normalize=normalize)

        # Add write deps for mutation outputs
        if self.mutated_inputs:
            mutation_outputs = self.outputs[1:] if len(self.outputs) > 1 else []
            for buf, mut_out in zip(self.mutated_inputs, mutation_outputs, strict=False):
                if not isinstance(buf, IRNode) or not isinstance(mut_out, Buffer):
                    continue
                layout = buf.get_layout()
                if not isinstance(layout, Layout):
                    continue
                indexer = layout.make_indexer()

                def _mut_out_dummy(
                    index, rindex, _name=mut_out.get_name(), _indexer=indexer
                ):
                    assert len(rindex) == 0
                    return ops.store(_name, _indexer(index), "fake")

                deps.writes |= dependencies.extract_read_writes(
                    _mut_out_dummy, buf.get_size(), (), normalize=normalize
                ).writes

        # For multi-output templates, add write dependencies for ALL outputs
        if isinstance(self.layout, MultiOutputLayout) and self.multi_output_nodes:
            for mo_node in self.multi_output_nodes:
                if not isinstance(mo_node, MultiOutput):
                    continue
                mo_layout = mo_node.layout
                if not isinstance(mo_layout, FixedLayout):
                    continue
                mo_name = mo_node.get_name()
                mo_size = mo_layout.size
                mo_indexer = mo_layout.make_indexer()

                def _dummy(index, rindex, _name=mo_name, _indexer=mo_indexer):
                    assert len(rindex) == 0
                    return ops.store(_name, _indexer(index), "fake")

                mo_deps = dependencies.extract_read_writes(
                    _dummy, mo_size, (), normalize=normalize
                )
                deps.writes |= mo_deps.writes

        return deps

    def get_multi_output_write_dep(
        self,
        output_name: str,
        template_writes: OrderedSet[Dep],
    ) -> MemoryDep:
        """Get the write dependency for a specific output buffer."""
        for w in template_writes:
            if isinstance(w, MemoryDep) and w.name == output_name:
                return w
        raise AssertionError(
            f"No write dependency found for output '{output_name}'. "
            f"Available writes: {[w.name for w in template_writes if isinstance(w, MemoryDep)]}"
        )

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
        multi_output_nodes = [
            n
            for n in (epilogue_nodes or [])
            if isinstance(_get_ir_node(n), MultiOutput)
        ]
        if multi_output_nodes:
            self.multi_output_nodes = [_get_ir_node(n) for n in multi_output_nodes]

        with self:
            if not only_gen_src_code:
                for node in [template_node, *multi_output_nodes]:
                    node.mark_run()
            partial_code = render()

        with V.set_kernel_handler(self):
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize_remaining()
            )
            node_schedule = [template_node]
            if inductor_fusion_config.benchmark_kernel:
                src_code = f"\n{src_code}\n{IndentedBuffer().getvalue()}"
            if only_gen_src_code:
                return src_code
            self.kernel_name = scheduling.define_kernel(src_code, node_schedule, self)
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
        wrapper.add_import_once(library_imports["triton"])
        wrapper.add_import_once(library_imports["tl"])
        wrapper.add_import_once(library_imports["_default_launcher"])

        for name in ("libdevice", "tl_math", "triton_helpers", "helion"):
            if f"{name}." in src_code:
                wrapper.add_import_once(library_imports[name])

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        wrapper.header.writeline(f"# kernel path: {kernel_path}\n{origins}\n{detailed}")

        in_imports = True
        for line in src_code.split("\n"):
            s = line.strip()
            if in_imports and (
                s.startswith(("from __future__", "import ", "from ")) or not s
            ):
                continue
            in_imports = False
            wrapper.header.writeline(line)
        wrapper.header.writeline("")
        return True

    def __enter__(self) -> Self:
        """Enter kernel handler context."""
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.__enter__()
        self._exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit kernel handler context."""
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def set_current_node(self, node: Any) -> contextlib.nullcontext[None]:
        """Set current node for codegen context."""
        return contextlib.nullcontext()

    def split_and_set_ranges(
        self, lengths: Sequence[Sequence[sympy.Expr]]
    ) -> list[list[sympy.Expr]]:
        """Split ranges for prologue codegen."""
        result, idx = [], 0
        for g in lengths:
            result.append([sympy.Symbol(f"idx{idx + i}") for i in range(len(g))])
            idx += len(g)
        return result
