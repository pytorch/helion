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
from torch._inductor.ir import FlexibleLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import Layout
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import TritonTemplateBuffer
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops
from torch.utils._ordered_set import OrderedSet

from ...runtime.config import Config
from .indexing import compute_helion_transform
from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..output_header import get_needed_imports
from ..output_header import library_imports


class HelionTemplateBuffer(TritonTemplateBuffer):
    """Helion kernel IR node with fusion support."""

    multi_output_nodes: list[MultiOutput]

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        kernel: Any,  # noqa: ANN401
        kernel_idx: int,
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
        # Maps prologue output buffer name -> source buffer name (for call_kernel)
        # Populated during prologue codegen when capture_buffer is called
        self._prologue_to_source: dict[str, str] = {}
        # Context: current prologue output being processed (set during prologue codegen)
        self._current_prologue_output: str | None = None
        # Store-shape recording (used to match view-store subscripts)
        self._recording_store_shapes = False
        self._recorded_store_shapes: dict[str, tuple[list, list]] = {}

        self.multi_output_nodes = []

        # Enable prologue fusion for all tensor inputs
        # The scheduler will check if each input has fusable prologue ops
        allowed_prologue_inps = OrderedSet(
            inp.get_name() for inp in inputs if isinstance(inp, IRNode)
        )
        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=lambda tb, hint_override=None: (self, self.render),
            mutated_inputs=None,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        if self.uses_atomics():
            V.graph.no_fuse_buffer_names.add(self.get_name())

    def render(self) -> PartialRender:
        """Generate Triton code with fusion applied."""
        if not self._bound_kernel:
            return PartialRender("", {})
        cfg = self._normalized_config()
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

    def _normalized_config(self) -> Config:
        cfg = self._bound_kernel._require_implicit_config()
        if not isinstance(cfg, Config):
            cfg = Config(**cfg)
        self._bound_kernel.env.config_spec.normalize(cfg)
        return cfg

    def _record_store_shapes(self) -> None:
        if not self._bound_kernel:
            return
        cfg = self._normalized_config()

        saved_captured = self._captured_buffers
        saved_fusion_stored_info = self._fusion_stored_info
        saved_prologue_to_source = self._prologue_to_source
        saved_current_prologue_output = self._current_prologue_output
        saved_recording = self._recording_store_shapes

        self._captured_buffers = {}
        self._fusion_stored_info = {}
        self._prologue_to_source = {}
        self._current_prologue_output = None
        self._recorded_store_shapes = {}
        self._recording_store_shapes = True
        try:
            with self._bound_kernel.env as env:
                env.set_template_buffer(self)
                generate_ast(
                    self._bound_kernel.host_function, cfg, emit_repro_caller=False
                )
        finally:
            self._recording_store_shapes = saved_recording
            self._captured_buffers = saved_captured
            self._fusion_stored_info = saved_fusion_stored_info
            self._prologue_to_source = saved_prologue_to_source
            self._current_prologue_output = saved_current_prologue_output

    def _record_store_shape(self, output_name: str, shape: list, strides: list) -> None:
        if output_name not in self._recorded_store_shapes:
            self._recorded_store_shapes[output_name] = (shape, strides)

    def call_kernel(self, kernel_name: str, template_buffer: Any = None) -> None:  # noqa: ANN401
        """Emit the kernel call site.

        Generates code like: output = kernel_name(arg1, arg2, ...)
        Handles argument mapping and multi-output unpacking.
        """
        wrapper, output_name = V.graph.wrapper_code, self.get_name()
        reinterpret_counter = 0

        def emit_reinterpret(
            base: str, size: tuple[int, ...], stride: tuple[int, ...], offset: int = 0
        ) -> str:
            """Emit reinterpret_tensor and return the variable name."""
            nonlocal reinterpret_counter
            name = f"reinterp_{reinterpret_counter}"
            reinterpret_counter += 1
            wrapper.writeline(
                f"{name} = reinterpret_tensor({base}, {size}, {stride}, {offset})"
            )
            return name

        # Build input names, handling ReinterpretView and fused inputs
        input_names = []
        for inp in self.inputs:
            inp_name = inp.get_name()  # pyrefly: ignore [missing-attribute]

            # Handle ReinterpretView: emit reinterpret_tensor so kernel gets correct shape
            if isinstance(inp, ReinterpretView):
                storage_name = inp.data.get_name()  # pyrefly: ignore [missing-attribute]
                # If storage was fused as prologue, use captured source buffer instead
                base_name = self._prologue_to_source.get(storage_name, storage_name)

                view_size = tuple(int(s) for s in inp.get_size())
                view_stride = tuple(int(s) for s in inp.get_stride())
                view_offset = int(inp.layout.offset)
                input_names.append(
                    emit_reinterpret(base_name, view_size, view_stride, view_offset)
                )
                continue

            if inp_name not in self.prologue_fused_inputs or not self._captured_buffers:
                input_names.append(inp_name)
                continue

            # Fused prologue: use captured source buffer
            source = self._prologue_to_source.get(inp_name)
            assert source is not None

            # Check if shapes match; if not, emit reinterpret_tensor for host code
            inp_size = tuple(V.graph.sizevars.size_hint(s) for s in inp.get_size())
            source_buf = V.graph.get_buffer(source)
            if source_buf is not None:
                source_size = tuple(
                    V.graph.sizevars.size_hint(s) for s in source_buf.get_size()
                )
                if inp_size != source_size:
                    # Compute contiguous strides for the target size
                    strides = tuple(int(s) for s in FlexibleLayout.contiguous_strides(inp_size))
                    input_names.append(emit_reinterpret(source, inp_size, strides))
                    continue

            input_names.append(source)

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
        output_names = set()
        if isinstance(self.layout, MultiOutputLayout):
            for mo in self.multi_output_nodes:
                if isinstance(mo, MultiOutput) and mo.indices:
                    n = mo.get_name()
                    output_names.add(n)
                    idx_str = output_name
                    for _, idx in mo.indices:
                        idx_str = f"{idx_str}[{idx}]"
                    wrapper.writeline(f"{n} = {idx_str}")

        # Assign epilogue aliases: each output gets aliased to its first epilogue result.
        # Skip captured buffers (reshaped epilogues store directly to their output).
        captured = set(self._captured_buffers.keys())
        for out_name in output_names or [output_name]:
            for ep in self._epilogue_specs.get(out_name, []):
                if isinstance(ep, BaseSchedulerNode) and isinstance(ep.node, IRNode):
                    alias = ep.node.get_name()
                    if alias not in captured and alias != output_name and alias not in output_names:
                        wrapper.writeline(f"{alias} = {out_name}")
                        break  # Only first alias per output

    def capture_buffer(self, buffer_name: str, epilogue: bool = True) -> str:
        """Register a captured buffer and return its parameter name.

        Called by codegen.py when fusion needs to access an external buffer.
        """
        count = sum(
            1 for _, is_epi in self._captured_buffers.values() if is_epi == epilogue
        )
        param_name = f"{'epilogue' if epilogue else 'prologue'}_input_{count}"
        self._captured_buffers[buffer_name] = (param_name, epilogue)

        # For prologue: record mapping from prologue output -> source buffer
        # Used by call_kernel to find the correct substitute
        if not epilogue and self._current_prologue_output is not None:
            if self._current_prologue_output not in self._prologue_to_source:
                self._prologue_to_source[self._current_prologue_output] = buffer_name

        return param_name

    @contextlib.contextmanager
    def prologue_context(self, prologue_output: str) -> contextlib.Iterator[None]:
        """Context manager to track which prologue is being codegen'd.

        This enables capture_buffer to build _prologue_to_source mapping,
        used by call_kernel to substitute fused prologue outputs.
        """
        self._current_prologue_output = prologue_output
        yield
        self._current_prologue_output = None

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

    def check_prologue_fusion_heuristics(
        self, prologue_node: BaseSchedulerNode
    ) -> bool | None:
        """Always allow prologue fusion (bypass default read/write byte heuristics)."""
        return True

    def can_fuse_vertical_override(self, producer_node: BaseSchedulerNode) -> bool | None:
        """Allow prologue fusion for allowed inputs."""
        writes = {d.name for d in producer_node.read_writes.writes if hasattr(d, "name")}
        if writes & self.allowed_prologue_inps:
            return True
        return None

    def supports_multi_outputs(self) -> bool:
        """Check if this kernel supports multi-output fusion."""
        return isinstance(self.layout, MultiOutputLayout)

    def _get_output_shapes(self) -> dict[str, list]:
        """Get shapes of all outputs (main + multi-outputs)."""
        shapes: dict[str, list] = {self.get_name(): list(self.get_size())}
        for mo in self.multi_output_nodes:
            if isinstance(mo, IRNode):
                shapes[mo.get_name()] = list(mo.get_size())
        return shapes

    def can_fuse_multi_output(self, node2: Any) -> bool:  # noqa: ANN401
        """Check if node2 can fuse: MultiOutput extraction or epilogue consuming outputs."""
        node2_ir = node2.node if hasattr(node2, "node") else node2

        # Case 1: MultiOutput extraction nodes - only for multi-output kernels
        if isinstance(node2_ir, MultiOutput):
            if not isinstance(self.layout, MultiOutputLayout):
                return False
            return (
                len(node2_ir.inputs) == 1
                and node2_ir.inputs[0].get_name() == self.get_name()
            )

        # Case 2: Epilogue ComputedBuffers that read template outputs
        if not (isinstance(node2_ir, IRNode) and hasattr(node2, "read_writes")):
            return False

        output_names = set(self._get_output_shapes().keys())
        reads = {
            d.name for d in node2.read_writes.reads
            if isinstance(d, (MemoryDep, StarDep, WeakDep)) and d.name in output_names
        }
        if not reads:
            return False

        output_nodes: dict[str, IRNode] = {self.get_name(): self}
        for mo in self.multi_output_nodes:
            if isinstance(mo, IRNode):
                output_nodes[mo.get_name()] = mo

        epilogue_nodes = node2.get_nodes()
        last_node = epilogue_nodes[-1] if epilogue_nodes else None
        epilogue_ir = last_node.node if hasattr(last_node, "node") else None
        if isinstance(epilogue_ir, IRNode):
            epilogue_shape = list(epilogue_ir.get_size())
            epilogue_layout = epilogue_ir.get_layout()
            for read_name in reads:
                kernel_output = output_nodes.get(read_name)
                if not isinstance(kernel_output, IRNode):
                    continue
                kernel_shape = list(kernel_output.get_size())
                kernel_strides = (
                    list(kernel_output.get_stride())
                    if hasattr(kernel_output, "get_stride")
                    else None
                )
                transform = compute_helion_transform(
                    epilogue_nodes,
                    read_name,
                    kernel_shape,
                    epilogue_shape,
                    epilogue_layout,
                    kernel_strides,
                )
                if transform.unsupported or transform.broadcast_dims:
                    return False
        return True

    def get_layout(self) -> Layout:
        """Get layout, handling multi-output case."""
        if isinstance(self.layout, MultiOutputLayout) and self.multi_output_nodes:
            return self.multi_output_nodes[0].layout
        return super().get_layout()

    def extract_read_writes(self, normalize: bool = False):  # noqa: ANN201
        """Extract read/write dependencies, adding writes for each multi-output."""
        deps = super().extract_read_writes(normalize=normalize)

        if isinstance(self.layout, MultiOutputLayout) and self.multi_output_nodes:
            for mo_node in self.multi_output_nodes:
                if not isinstance(mo_node, MultiOutput):
                    continue
                mo_layout = mo_node.layout
                if not isinstance(mo_layout, FixedLayout):
                    continue

                mo_name, mo_size = mo_node.get_name(), mo_layout.size
                mo_indexer = mo_layout.make_indexer()

                def _dummy(index, rindex):  # noqa: ANN001, ANN202
                    assert len(rindex) == 0
                    return ops.store(mo_name, mo_indexer(index), "fake")

                mo_deps = dependencies.extract_read_writes(
                    _dummy, mo_size, (), normalize=normalize
                )
                deps.writes |= mo_deps.writes

        return deps

    def get_multi_output_write_dep(
        self,
        output_name: str,
        template_writes: OrderedSet[MemoryDep | StarDep | WeakDep],
    ) -> MemoryDep:
        """Get write dependency matching output_name (supports different output shapes)."""
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
        self._record_store_shapes()

        def get_ir(n: Any) -> Any:
            return n.node if isinstance(n, BaseSchedulerNode) else n

        multi_output_nodes = [
            n for n in (epilogue_nodes or []) if isinstance(get_ir(n), MultiOutput)
        ]
        fusable_epilogue_nodes = [
            n for n in (epilogue_nodes or []) if not isinstance(get_ir(n), MultiOutput)
        ]

        # Process epilogue nodes for fusion
        actually_fused_epilogues: list = []
        if fusable_epilogue_nodes and not self.uses_atomics():
            output_names = set(self._get_output_shapes().keys())
            output_nodes: dict[str, IRNode] = {self.get_name(): self}
            for mo in self.multi_output_nodes:
                if isinstance(mo, IRNode):
                    output_nodes[mo.get_name()] = mo

            for ep in fusable_epilogue_nodes:
                if not (isinstance(ep, BaseSchedulerNode) and ep.read_writes):
                    continue
                reads = {
                    d.name
                    for d in ep.read_writes.reads
                    if isinstance(d, (MemoryDep, StarDep, WeakDep))
                    and d.name in output_names
                }
                if not reads:
                    continue
                acc_name = next(iter(reads))
                kernel_output = output_nodes.get(acc_name)
                if not isinstance(kernel_output, IRNode):
                    continue
                epilogue_nodes = ep.get_nodes()
                if not epilogue_nodes:
                    continue
                last_node = epilogue_nodes[-1]
                epilogue_ir = last_node.node if hasattr(last_node, "node") else None
                if not isinstance(epilogue_ir, IRNode):
                    continue
                if override := self._recorded_store_shapes.get(acc_name):
                    kernel_shape, kernel_strides = override
                else:
                    kernel_shape = list(kernel_output.get_size())
                    kernel_strides = (
                        list(kernel_output.get_stride())
                        if hasattr(kernel_output, "get_stride")
                        else None
                    )
                epilogue_shape = list(epilogue_ir.get_size())
                epilogue_layout = epilogue_ir.get_layout()

                transform = None
                for snode in reversed(epilogue_nodes):
                    if hasattr(snode, "_helion_epilogue_transform"):
                        transform = getattr(snode, "_helion_epilogue_transform")
                        break
                if transform is None:
                    transform = compute_helion_transform(
                        epilogue_nodes,
                        acc_name,
                        kernel_shape,
                        epilogue_shape,
                        epilogue_layout,
                        kernel_strides,
                    )
                if transform.unsupported:
                    continue
                for snode in epilogue_nodes:
                    setattr(snode, "_helion_epilogue_transform", transform)
                setattr(ep, "_helion_epilogue_transform", transform)

                actually_fused_epilogues.append(ep)
                if len(reads) > 1:
                    self._multi_dep_epilogue_specs.append(([ep], reads))
                else:
                    self._epilogue_specs.setdefault(next(iter(reads)), []).append(ep)

        # Collect prologues for fusion
        fused_prologue_bufs: set[str] = set()
        fusable_prologue_nodes: list[BaseSchedulerNode] = []
        for name, buf in self.named_input_nodes.items():
            buf_name = buf.get_name()
            prologue_group = buf_name_to_prologue_group.get(buf_name)
            if not prologue_group:
                continue
            prologue_nodes = list(prologue_group)
            last_node = prologue_nodes[-1]
            prologue_ir = last_node.node if hasattr(last_node, "node") else None
            if not isinstance(prologue_ir, IRNode):
                continue
            kernel_shape = list(buf.get_size())
            kernel_strides = (
                list(buf.get_stride()) if hasattr(buf, "get_stride") else None
            )
            prologue_shape = list(prologue_ir.get_size())
            prologue_layout = prologue_ir.get_layout()

            transform = None
            for snode in reversed(prologue_nodes):
                if hasattr(snode, "_helion_prologue_transform"):
                    transform = getattr(snode, "_helion_prologue_transform")
                    break
            if transform is None:
                transform = compute_helion_transform(
                    prologue_nodes,
                    buf_name,
                    kernel_shape,
                    prologue_shape,
                    prologue_layout,
                    kernel_strides,
                )
            if transform.unsupported or transform.broadcast_dims:
                raise AssertionError(
                    f"Unsupported Helion prologue fusion for input '{buf_name}'."
                )
            for snode in prologue_nodes:
                setattr(snode, "_helion_prologue_transform", transform)

            self._prologue_specs[name] = (prologue_nodes, buf_name)
            fused_prologue_bufs.add(buf_name)
            fusable_prologue_nodes.extend(prologue_nodes)

        with self:
            if not only_gen_src_code:
                # Only mark actually fused epilogues as run.
                for node in [
                    template_node,
                    *actually_fused_epilogues,
                    *multi_output_nodes,
                ]:
                    node.mark_run()
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
                *actually_fused_epilogues,
            ]
            if inductor_fusion_config.benchmark_kernel:
                # Benchmark stubs inlined - Helion doesn't support benchmarking
                src_code = f"\n{src_code}\n{IndentedBuffer().getvalue()}"
            if only_gen_src_code:
                return src_code

            # Mark fused prologue buffers as removed (not allocated, skip 'del buf')
            self.removed_buffers.update(fused_prologue_bufs)

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
        """Emit Triton source code with imports."""
        for key in ("triton", "tl", "_default_launcher"):
            wrapper.add_import_once(library_imports[key])
        for name in ("libdevice", "tl_math", "triton_helpers"):
            if f"{name}." in src_code:
                wrapper.add_import_once(library_imports[name])

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        wrapper.header.writeline(f"# kernel path: {kernel_path}\n{origins}\n{detailed}")

        in_imports = True
        for line in src_code.split("\n"):
            s = line.strip()
            if in_imports and (s.startswith(("from __future__", "import ", "from ")) or not s):
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

    def __exit__(self, *args: object) -> None:
        """Exit kernel handler context."""
        # pyrefly: ignore [bad-argument-type]
        self._exit_stack.__exit__(*args)

    def set_current_node(self, node: Any) -> contextlib.nullcontext[None]:  # noqa: ANN401
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
