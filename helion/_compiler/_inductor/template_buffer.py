"""HelionTemplateBuffer - IR node for Helion kernels in Inductor."""

from __future__ import annotations

import ast
import contextlib
from typing import Any
from typing import Callable
from typing import Sequence
from typing_extensions import Self

import sympy
from torch._inductor import config as inductor_fusion_config
from torch._inductor import dependencies
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.dependencies import MemoryDep
from torch._inductor.dependencies import StarDep
from torch._inductor.dependencies import WeakDep
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import Layout
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import TritonTemplateBuffer
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops
from torch.utils._ordered_set import OrderedSet

from ...runtime.config import Config
from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..output_header import get_needed_imports
from ..output_header import library_imports


class HelionTemplateBuffer(TritonTemplateBuffer):
    """Helion kernel IR node with fusion support."""

    multi_output_nodes: list[MultiOutput]
    _helion_epilogue_aliases: list[str]

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        kernel: Any,  # noqa: ANN401
        kernel_idx: int,  # noqa: ARG002
        constant_args: dict[str, Any],
        tensor_arg_names: list[str],
        bound_kernel: Any,  # noqa: ANN401
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
        self._epilogue_specs: dict[str, list] = {}  # accumulator_name -> nodes
        self._prologue_specs: dict[
            str, tuple[list, str]
        ] = {}  # arg_name -> (nodes, buffer_name)
        self._constant_args_dict = constant_args
        self._multi_dep_epilogue_specs: list[
            tuple[list, set[str]]
        ] = []  # (nodes, accumulator_names)
        self._uses_atomics_cache: bool | None = None

        # Fusion state (used during code generation)
        self._captured_buffers: dict[str, tuple[str, bool]] = {}
        self._fusion_stored_info: dict[str, ast.expr] = {}

        self.multi_output_nodes = []
        self._helion_epilogue_aliases = []

        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=lambda tb, hint_override=None: (self, self.render),
            mutated_inputs=None,
            allowed_prologue_inps=OrderedSet(
                inp.get_name() for inp in inputs if isinstance(inp, IRNode)
            ),
        )
        if self.uses_atomics():
            V.graph.no_fuse_buffer_names.add(self.get_name())

    def render(self) -> PartialRender:
        """Generate Triton code with fusion applied."""
        if not self._bound_kernel:
            return PartialRender("", {})
        cfg = self._bound_kernel._require_implicit_config()
        if not isinstance(cfg, Config):
            cfg = Config(**cfg)
        self._bound_kernel.env.config_spec.normalize(cfg)
        host_fn, triton_fn = (
            self._helion_kernel.name,
            f"_helion_{self._helion_kernel.name}",
        )

        # Generate AST - fusion is applied during this step via memory_ops.py
        with self._bound_kernel.env as env:
            env.set_template_buffer(self)
            root = generate_ast(
                self._bound_kernel.host_function, cfg, emit_repro_caller=False
            )
            all_captured = {
                buf_name: param_name
                for buf_name, (param_name, _) in self._captured_buffers.items()
            }

        # Inject captured buffer parameters into the AST
        if all_captured:
            params = list(all_captured.values())
            tensor_ann = ast.Attribute(
                value=ast.Name(id="torch", ctx=ast.Load()),
                attr="Tensor",
                ctx=ast.Load(),
            )

            def _get_name(a: ast.AST) -> str | None:
                if isinstance(a, ast.arg):
                    return a.arg
                if isinstance(a, ast.Name):
                    return a.id
                return None

            def _insert_params(
                args: list[ast.AST],
                new_args: list[str],
                make: Callable[[str], ast.AST],
            ) -> None:
                existing = {_get_name(a) for a in args}
                to_add = [make(p) for p in new_args if p not in existing]
                if to_add:
                    idx = next(
                        (
                            i
                            for i, a in enumerate(args)
                            if (_get_name(a) or "").startswith("_BLOCK_SIZE")
                        ),
                        len(args),
                    )
                    args[idx:idx] = to_add

            for node in ast.walk(root):
                if isinstance(node, ast.FunctionDef) and node.name == triton_fn:
                    _insert_params(
                        node.args.args,  # pyrefly: ignore [bad-argument-type]
                        params,
                        lambda p: ast.arg(arg=p),
                    )
                elif isinstance(node, ast.FunctionDef) and node.name == host_fn:
                    _insert_params(
                        node.args.args,  # pyrefly: ignore [bad-argument-type]
                        params,
                        lambda p: ast.arg(arg=p, annotation=tensor_ann),
                    )
                    node.name = str(Placeholder.KERNEL_NAME)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_launcher"
                ):
                    if (
                        len(node.args) >= 2
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == triton_fn
                    ):
                        _insert_params(
                            node.args,  # pyrefly: ignore [bad-argument-type]
                            params,
                            lambda p: ast.Name(id=p, ctx=ast.Load()),
                        )
        else:
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
        """Emit the kernel call site.

        Generates code like: output = kernel_name(arg1, arg2, ...)
        Handles argument mapping and multi-output unpacking.
        """
        wrapper, output_name = V.graph.wrapper_code, self.get_name()

        # Build input names, replacing fused inputs with captured buffers
        input_names = []
        for inp in self.inputs:
            inp_name = inp.get_name()  # pyrefly: ignore [missing-attribute]
            if (
                inp_name not in self.prologue_fused_inputs
                or not self._captured_buffers
            ):
                input_names.append(inp_name)
                continue
            # Find captured buffer with matching shape
            matched = None
            inp_size = inp.get_size()  # pyrefly: ignore [missing-attribute]
            for buf_name in self._captured_buffers:
                buf = V.graph.get_buffer(buf_name)
                if isinstance(buf, IRNode) and buf.get_size() == inp_size:
                    matched = buf_name
                    break
            input_names.append(
                matched or next(iter(self._captured_buffers), None) or inp_name
            )

        # Build args from signature
        args, input_idx, sig = [], 0, self._helion_kernel.signature.parameters
        for name in sig:
            if name in self._tensor_arg_names and input_idx < len(input_names):
                args.append(input_names[input_idx])
                input_idx += 1
            elif name in self._constant_args_dict:
                args.append(repr(self._constant_args_dict[name]))
            elif sig[name].default is not sig[name].empty:
                args.append(repr(sig[name].default))
        args.extend(self._captured_buffers)
        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

        # Handle multi-output unpacking
        mo_names, mo_list = set(), []
        if isinstance(self.layout, MultiOutputLayout):
            for mo in self.multi_output_nodes:
                if isinstance(mo, MultiOutput) and mo.indices:
                    n = mo.get_name()
                    mo_names.add(n)
                    mo_list.append(n)
                    idx_str = output_name
                    for _, idx in mo.indices:
                        idx_str = f"{idx_str}[{idx}]"
                    wrapper.writeline(f"{n} = {idx_str}")
        for i, a in enumerate(
            a
            for a in self._helion_epilogue_aliases
            if a != output_name and a not in mo_names
        ):
            wrapper.writeline(
                f"{a} = {mo_list[i] if i < len(mo_list) else output_name}"
            )

    def capture_buffer(self, buffer_name: str, epilogue: bool = True) -> str:
        """Register a captured buffer and return its parameter name.

        Called by codegen.py when fusion needs to access an external buffer.
        """
        count = sum(
            1 for _, is_epi in self._captured_buffers.values() if is_epi == epilogue
        )
        param_name = f"{'epilogue' if epilogue else 'prologue'}_input_{count}"
        self._captured_buffers[buffer_name] = (param_name, epilogue)
        return param_name

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
        """
        if self._uses_atomics_cache is not None:
            return self._uses_atomics_cache
        if not self._bound_kernel:
            self._uses_atomics_cache = True
            return True

        cfg = self._bound_kernel._require_implicit_config()
        if not isinstance(cfg, Config):
            cfg = Config(**cfg)
        self._bound_kernel.env.config_spec.normalize(cfg)
        with self._bound_kernel.env as env:
            env.set_template_buffer()
            self._uses_atomics_cache = "tl.atomic_" in unparse(
                generate_ast(
                    self._bound_kernel.host_function, cfg, emit_repro_caller=False
                ),
                output_origin_lines=False,
            )
        return self._uses_atomics_cache

    def supports_epilogue_fusion(self) -> bool:
        """Check if this kernel supports epilogue fusion."""
        return not self.uses_atomics()

    def supports_multi_outputs(self) -> bool:
        """Check if this kernel supports multi-output fusion."""
        return isinstance(self.layout, MultiOutputLayout)

    def can_fuse_multi_output(self, node2: Any) -> bool:  # noqa: ANN401
        """Check if this multi-output template can fuse with node2."""
        return (
            isinstance(self.layout, MultiOutputLayout)
            and isinstance(node2.node, MultiOutput)
            and len(node2.node.inputs) == 1
            and node2.node.inputs[0].get_name()  # pyrefly: ignore
            == self.get_name()
        )

    def get_layout(self) -> Layout:
        """Get layout, handling multi-output case."""
        if isinstance(self.layout, MultiOutputLayout):
            mo = self.multi_output_nodes
            assert mo and isinstance(mo[0], MultiOutput), (
                "MultiOutputLayout without multi_output_nodes"
            )
            return mo[0].layout
        return super().get_layout()

    def extract_read_writes(self, normalize: bool = False):  # noqa: ANN201
        """Extract read/write dependencies for scheduling.

        For multi-output templates with different output shapes (e.g., [128] vs [64, 2]),
        we add a write dependency for EACH output with its correct index pattern.
        This enables the scheduler to fuse epilogues with each output independently.
        """
        # Get base dependencies from parent
        deps = super().extract_read_writes(normalize=normalize)

        # For multi-output templates, add write dependencies for ALL outputs
        if isinstance(self.layout, MultiOutputLayout) and self.multi_output_nodes:
            for mo_node in self.multi_output_nodes:
                if not isinstance(mo_node, MultiOutput):
                    continue

                # Get the MultiOutput's layout and create its indexer
                mo_layout = mo_node.layout
                if not isinstance(mo_layout, FixedLayout):
                    continue

                mo_name = mo_node.get_name()
                mo_size = mo_layout.size
                mo_indexer = mo_layout.make_indexer()

                # Create index variables for this output's shape
                def _dummy(index, rindex):  # noqa: ANN001, ANN202
                    assert len(rindex) == 0
                    return ops.store(mo_name, mo_indexer(index), "fake")

                # Extract write dependency with correct index pattern
                mo_deps = dependencies.extract_read_writes(
                    _dummy, mo_size, (), normalize=normalize
                )

                # Add the write dependency for this output
                deps.writes |= mo_deps.writes

        return deps

    def get_multi_output_write_dep(
        self,
        output_name: str,
        template_writes: OrderedSet[MemoryDep | StarDep | WeakDep],
    ) -> MemoryDep:
        """Get the write dependency for a specific output buffer.

        Overrides base class to find the MemoryDep matching output_name, enabling
        fusion when outputs have different shapes (and thus different index patterns).
        """
        for w in template_writes:
            if isinstance(w, MemoryDep) and w.name == output_name:
                return w
        raise AssertionError(
            f"No write dependency found for output '{output_name}'. "
            f"Available writes: {[w.name for w in template_writes if isinstance(w, MemoryDep)]}"
        )

    def codegen_template_override(
        self,
        scheduling: Any,  # noqa: ANN401
        template_node: Any,  # noqa: ANN401
        epilogue_nodes: Any,  # noqa: ANN401
        prologue_nodes: Any,  # noqa: ANN401
        buf_name_to_prologue_group: Any,  # noqa: ANN401
        prologue_preserves_zero_mask_fn: Any,  # noqa: ANN401
        render: Any,  # noqa: ANN401
        only_gen_src_code: Any,  # noqa: ANN401
    ):
        """Entry point for template codegen called by Inductor scheduler."""

        def get_node(n: Any) -> Any:  # noqa: ANN401
            return n.node if isinstance(n, BaseSchedulerNode) else n

        multi_output_nodes = [n for n in (epilogue_nodes or []) if isinstance(get_node(n), MultiOutput)]
        fusable_epilogue_nodes = [n for n in (epilogue_nodes or []) if not isinstance(get_node(n), MultiOutput)]

        if fusable_epilogue_nodes and not self.uses_atomics():
            outputs = {self.get_name()} | {
                o.get_name() for o in self.multi_output_nodes if isinstance(o, IRNode)
            }
            for ep in fusable_epilogue_nodes:
                if not (isinstance(ep, BaseSchedulerNode) and ep.read_writes):
                    continue
                reads = {
                    d.name for d in ep.read_writes.reads
                    if isinstance(d, (MemoryDep, StarDep, WeakDep)) and d.name in outputs
                }
                if len(reads) > 1:
                    self._multi_dep_epilogue_specs.append(([ep], reads))
                elif len(reads) == 1:
                    self._epilogue_specs.setdefault(next(iter(reads)), []).append(ep)

        for name, buf in self.named_input_nodes.items():
            if p := buf_name_to_prologue_group.get(buf.get_name()):
                self._prologue_specs[name] = (list(p), buf.get_name())
        fusable_prologue_nodes = [
            n for nodes in buf_name_to_prologue_group.values() for n in nodes
        ]

        with self:
            if not only_gen_src_code:
                for node in [template_node, *fusable_epilogue_nodes, *multi_output_nodes]:
                    node.mark_run()
            self._helion_epilogue_aliases = [
                ep.node.get_name()
                for nodes in self._epilogue_specs.values()
                for ep in nodes
                if isinstance(ep, BaseSchedulerNode) and isinstance(ep.node, IRNode)
            ]
            partial_code = render()
            for buffer in self.named_input_nodes.values():
                for pn in buf_name_to_prologue_group.get(buffer.get_name(), []):
                    pn.codegen(self.split_and_set_ranges(pn.get_ranges()))

        with V.set_kernel_handler(self):
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize_remaining()
            )
            node_schedule = [
                *fusable_prologue_nodes,
                template_node,
                *fusable_epilogue_nodes,
            ]
            if inductor_fusion_config.benchmark_kernel:
                # Benchmark stubs inlined - Helion doesn't support benchmarking
                src_code = f"\n{src_code}\n{IndentedBuffer().getvalue()}"
            if only_gen_src_code:
                return src_code
            self.kernel_name = scheduling.define_kernel(src_code, node_schedule, self)
        return self

    def emit_kernel_override(
        self,
        wrapper: Any,  # noqa: ANN401
        src_code: Any,  # noqa: ANN401
        kernel_name: Any,  # noqa: ANN401
        node_schedule: Any,  # noqa: ANN401
        kernel_path: Any,  # noqa: ANN401
        get_kernel_metadata: Any,  # noqa: ANN401
    ):
        """Entry point for kernel emission.

        Writes the Triton source code to the wrapper, adding necessary imports.
        """
        wrapper.add_import_once(library_imports["triton"])
        wrapper.add_import_once(library_imports["tl"])
        wrapper.add_import_once(library_imports["_default_launcher"])

        for name in ("libdevice", "tl_math", "triton_helpers"):
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

        aliases = []
        for sn in node_schedule:
            node = sn.node if isinstance(sn, BaseSchedulerNode) else sn
            if isinstance(node, IRNode):
                n = node.get_name()
                if n and n != self.get_name() and n not in self._captured_buffers:
                    aliases.append(n)
        self._helion_epilogue_aliases = aliases
        return True

    def __enter__(self) -> Self:
        """Enter kernel handler context."""
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.__enter__()
        self._exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, *args: object) -> None:
        """Exit kernel handler context."""
        # pyrefly: ignore [bad-argument-type]
        self._exit_stack.__exit__(*args)

    def set_current_node(self, node: Any) -> contextlib.nullcontext[None]:  # noqa: ANN401, ARG002
        """Set current node for codegen context (no-op, required by Inductor scheduler)."""
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


