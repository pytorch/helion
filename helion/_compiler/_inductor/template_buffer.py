from __future__ import annotations

import ast
import contextlib
from itertools import dropwhile
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence
from typing import cast

from torch._inductor import config as inductor_fusion_config
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.dependencies import MemoryDep
from torch._inductor.dependencies import StarDep
from torch._inductor.dependencies import WeakDep
from torch._inductor.ir import BaseView
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
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
from torch.utils._ordered_set import OrderedSet

from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..output_header import get_needed_imports
from ..output_header import library_imports

if TYPE_CHECKING:
    from torch._inductor.codegen.simd import SIMDScheduling
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


def _get_ir_node(n: Any) -> Any:  # noqa: ANN401
    """Extract the IR node from a scheduler node."""
    return n.node if isinstance(n, BaseSchedulerNode) else n


def has_non_trivial_view(nodes: Sequence[BaseSchedulerNode]) -> bool:
    """Check if any node is a non-trivial view (unsupported for fusion)."""
    for n in nodes:
        ir = _get_ir_node(n)
        if isinstance(ir, BaseView) and not is_trivial_view(ir):
            return True
    return False


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
        kernel: Kernel,
        kernel_idx: int,
        constant_args: dict[str, object],
        tensor_arg_names: list[str],
        bound_kernel: BoundKernel,
        mutated_inputs: Sequence[IRNode] | None = None,
        output_aliases: list[str | None] | None = None,
        output_alias_is_direct: list[bool] | None = None,
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

        self._helion_kernel = kernel
        self._tensor_arg_names = tensor_arg_names
        self._bound_kernel = bound_kernel
        self._constant_args_dict = constant_args
        self._output_aliases = output_aliases or []
        self._output_alias_is_direct = output_alias_is_direct or []
        self._autotune_args = autotune_args

        # Epilogue fusion tracking
        self._epilogue_specs: dict[str, list] = {}  # accumulator_name -> nodes
        self._multi_dep_epilogue_specs: list[
            tuple[list, set[str]]
        ] = []  # (nodes, accumulator_names)
        self._uses_atomics_cache: bool | None = None
        self._helion_alias_map: dict[str, str] = {}
        self._helion_mutated_input_names: set[str] = set()

        # Fusion state (used during code generation)
        self._captured_buffers: dict[str, tuple[str, bool]] = {}
        self._fusion_stored_info: dict[str, ast.expr] = {}

        self.multi_output_nodes = []
        self._helion_epilogue_aliases = []

        super().__init__(
            layout=cast("Layout", layout),
            inputs=inputs,
            make_kernel_render=lambda tb, hint_override=None: (self, self.render),
            mutated_inputs=mutated_inputs,
            # Disable prologue fusion - Helion kernels don't support it
            allowed_prologue_inps=OrderedSet(),
        )
        if mutated_inputs:
            never_reuse = V.graph.never_reuse_buffers
            never_reuse.add(self.get_name())
            for buf in mutated_inputs:
                if isinstance(buf, BaseView):
                    base = buf.unwrap_view()
                    if isinstance(base, IRNode):
                        never_reuse.add(base.get_name())
                    never_reuse.add(buf.get_name())
                elif isinstance(buf, IRNode):
                    never_reuse.add(buf.get_name())
        if self.uses_atomics():
            V.graph.no_fuse_buffer_names.add(self.get_name())

    def supports_multi_outputs(self) -> bool:
        """Return True if this template supports multi-output fusion."""
        return isinstance(self.layout, MultiOutputLayout)

    def can_fuse_multi_output(self, node2: object) -> bool:
        """Return True if this template can fuse with its MultiOutput node."""
        # Only allow fusion with MultiOutput nodes (the nodes that wrap template outputs)
        if not isinstance(self.layout, MultiOutputLayout) or self.uses_atomics():
            return False
        # Check if node2 is a scheduler node wrapping a MultiOutput
        if isinstance(node2, BaseSchedulerNode) and isinstance(node2.node, MultiOutput):
            return True
        return False

    def multi_output_should_allocate(self) -> bool:
        """Return True if MultiOutput nodes wrapping this template should allocate.

        When True, each MultiOutput gets its own buffer instead of just indexing
        into the template's return value. This is needed for Helion multi-output
        templates to ensure proper buffer allocation.
        """
        return isinstance(self.layout, MultiOutputLayout)

    def multi_output_prevents_buffer_reuse(self) -> bool:
        """Return True if MultiOutput results should not be reused for other buffers.

        When True, signals that MultiOutput buffers alias this template and
        should not be added to the buffer reuse pool. This prevents Inductor
        from incorrectly reusing a MultiOutput buffer for an unrelated operation,
        which can corrupt results when outputs share underlying storage references.
        """
        return isinstance(self.layout, MultiOutputLayout)

    def render(self) -> PartialRender:
        """Generate Triton code with fusion applied."""
        if not self._bound_kernel:
            return PartialRender("", {})
        # Ensure config is available (triggers autotuning if needed)
        if self._autotune_args:
            self._bound_kernel.ensure_config_exists(self._autotune_args)
        cfg = self._bound_kernel._config
        assert cfg is not None, "Config should be set after ensure_config_exists"
        host_fn = self._helion_kernel.name
        # The AST is generated with this original triton function name
        old_triton_fn = f"_helion_{self._helion_kernel.name}"
        # Use output buffer name to make Triton function name unique per kernel instance
        # This prevents name collisions when the same kernel is called multiple times
        # with different epilogue fusion settings
        output_name = self.get_name()
        triton_fn = f"_helion_{self._helion_kernel.name}_{output_name}"

        # Generate AST - fusion is applied during this step via memory_ops.py
        with self._bound_kernel.env as env, env.enable_fusion(template_buffer=self):
            host_function = self._bound_kernel.host_function
            assert host_function is not None, "BoundKernel must have a host_function"
            root = generate_ast(host_function, cfg, emit_repro_caller=False)
            all_captured = {
                buf_name: param_name
                for buf_name, (param_name, _) in self._captured_buffers.items()
            }

        def _get_name_from_ast(a: ast.AST) -> str | None:
            if isinstance(a, ast.arg):
                return a.arg
            if isinstance(a, ast.Name):
                return a.id
            return None

        def _insert_params(
            args: list[Any],
            new_args: list[str],
            make: Callable[[str], ast.AST],
        ) -> None:
            existing = {_get_name_from_ast(a) for a in args}
            to_add = [make(p) for p in new_args if p not in existing]
            if to_add:
                idx = next(
                    (
                        i
                        for i, a in enumerate(args)
                        if (_get_name_from_ast(a) or "").startswith("_BLOCK_SIZE")
                    ),
                    len(args),
                )
                args[idx:idx] = to_add

        # Rename Triton function to unique name and inject captured buffer parameters
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef) and node.name == old_triton_fn:
                # Rename to unique name
                node.name = triton_fn
                if all_captured:
                    _insert_params(
                        node.args.args,
                        list(all_captured.values()),
                        lambda p: ast.arg(arg=p),
                    )
            elif isinstance(node, ast.FunctionDef) and node.name == host_fn:
                if all_captured:
                    tensor_ann = ast.Attribute(
                        value=ast.Name(id="torch", ctx=ast.Load()),
                        attr="Tensor",
                        ctx=ast.Load(),
                    )
                    _insert_params(
                        node.args.args,
                        list(all_captured.values()),
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
                    and node.args[0].id == old_triton_fn
                ):
                    # Update launcher call to use new triton function name
                    node.args[0].id = triton_fn
                    if all_captured:
                        _insert_params(
                            node.args,
                            list(all_captured.values()),
                            lambda p: ast.Name(id=p, ctx=ast.Load()),
                        )

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
        # Add captured buffers as arguments
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
                    idx_str = output_name + "".join(f"[{idx}]" for _, idx in mo.indices)
                    wrapper.writeline(f"{n} = {idx_str}")

        # Handle epilogue aliases
        aliases = [
            a
            for a in self._helion_epilogue_aliases
            if a != output_name and a not in mo_names
        ]
        epilogue_capture_names = {
            name for name, (_, is_epi) in self._captured_buffers.items() if is_epi
        }
        if self.mutated_inputs:
            for alias in aliases:
                V.graph.never_reuse_buffers.add(alias)
        # Pre-populate alias_map with default target for all aliases
        default_target = mo_list[0] if mo_list else output_name
        alias_map: dict[str, str] = dict.fromkeys(aliases, default_target)
        # Override with specific mappings from epilogue specs
        for acc_name, nodes in self._epilogue_specs.items():
            if acc_name in mo_names:
                target = acc_name
            else:
                target = default_target
            for ep in nodes:
                if isinstance(ep, BaseSchedulerNode) and isinstance(ep.node, IRNode):
                    ep_name = ep.node.get_name()
                    if ep_name in alias_map and ep_name not in epilogue_capture_names:
                        alias_map[ep_name] = target
        for alias in epilogue_capture_names:
            if alias in alias_map:
                alias_map[alias] = alias
        merged_alias_map = dict(alias_map)
        if self._helion_alias_map:
            merged_alias_map.update(self._helion_alias_map)
        self._helion_alias_map = merged_alias_map

        for a in aliases:
            wrapper.writeline(f"{a} = {alias_map[a]}")

    def get_layout(self) -> Layout:
        """Get layout, handling multi-output case."""
        if isinstance(self.layout, MultiOutputLayout):
            mo = self.multi_output_nodes
            assert mo and isinstance(mo[0], MultiOutput), (
                "MultiOutputLayout without multi_output_nodes"
            )
            return cast("Layout", mo[0].layout)
        return super().get_layout()

    def capture_buffer(self, buffer_name: str, epilogue: bool = True) -> str:
        """Register a captured buffer and return its parameter name.

        Called by codegen.py when fusion needs to access an external buffer.
        """
        # Return existing param name if buffer already captured
        if buffer_name in self._captured_buffers:
            return self._captured_buffers[buffer_name][0]
        if epilogue and self.mutated_inputs:
            for buf in self.mutated_inputs:
                if not isinstance(buf, IRNode):
                    continue
                if buf.get_name() != buffer_name:
                    continue
                for name, inp in self.named_input_nodes.items():
                    if isinstance(inp, IRNode) and inp.get_name() == buf.get_name():
                        return name
            mutation_outputs = self.outputs[1:] if len(self.outputs) > 1 else []
            for buf, mut_out in zip(
                self.mutated_inputs, mutation_outputs, strict=False
            ):
                if not isinstance(buf, IRNode) or not isinstance(mut_out, Buffer):
                    continue
                if mut_out.get_name() != buffer_name:
                    continue
                for name, inp in self.named_input_nodes.items():
                    if isinstance(inp, IRNode) and inp.get_name() == buf.get_name():
                        return name
        count = sum(
            1 for _, is_epi in self._captured_buffers.values() if is_epi == epilogue
        )
        param_name = f"{'epilogue' if epilogue else 'prologue'}_input_{count}"
        self._captured_buffers[buffer_name] = (param_name, epilogue)
        return param_name

    @property
    def _fusion_store_map(self) -> dict[int, str]:
        """Compute fusion store map from multi_output_nodes.

        Maps store_index -> buffer_name. The store_index is the order of store
        operations in codegen. For non-mutation outputs, this should match the
        output_index (position in return tuple), which we extract from
        MultiOutput.indices.

        Note: Mutations don't create MultiOutput nodes, so they won't be in this
        map, which is correct since mutations shouldn't have epilogues applied.
        """
        if isinstance(self.layout, MultiOutputLayout):
            result = {}
            for mo in self.multi_output_nodes:
                if isinstance(mo, MultiOutput) and mo.indices:
                    # indices is [(tuple, output_index)] - extract output_index
                    output_index = mo.indices[0][1]
                    result[output_index] = mo.get_name()
            return result
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

        # Check the FX graphs for atomic operations
        from ...language import atomic_ops

        atomic_funcs = {getattr(atomic_ops, name) for name in atomic_ops.__all__}

        device_ir = self._bound_kernel.host_function.device_ir
        for graph_info in device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.op == "call_function" and node.target in atomic_funcs:
                    self._uses_atomics_cache = True
                    return True

        self._uses_atomics_cache = False
        return self._uses_atomics_cache

    def supports_epilogue_fusion(self) -> bool:
        """Check if this kernel supports epilogue fusion."""
        return not self.uses_atomics()

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
        multi_output_nodes = [
            n
            for n in (epilogue_nodes or [])
            if isinstance(_get_ir_node(n), MultiOutput)
        ]
        if os.environ.get("HELION_DEBUG_FUSION"):
            print(f"DEBUG multi_output_nodes (len={len(multi_output_nodes)}):")
            for i, mo in enumerate(multi_output_nodes):
                ir = _get_ir_node(mo)
                mo_index = getattr(ir, "index", "N/A")
                print(f"  [{i}] name={ir.get_name()}, index={mo_index}")
        if multi_output_nodes:
            self.multi_output_nodes = [_get_ir_node(n) for n in multi_output_nodes]

        fusable_epilogue_nodes = [
            n
            for n in (epilogue_nodes or [])
            if not isinstance(_get_ir_node(n), MultiOutput)
        ]

        # Collect mutated input names
        mutated_input_names: set[str] = set()
        if self.mutated_inputs:
            for buf in self.mutated_inputs:
                if isinstance(buf, BaseView):
                    mutated_input_names.add(buf.get_name())
                    base = buf.unwrap_view()
                    if isinstance(base, StorageBox):
                        base = base.data
                    if isinstance(base, IRNode):
                        mutated_input_names.add(base.get_name())
                elif isinstance(buf, IRNode):
                    mutated_input_names.add(buf.get_name())
        for arg_name, buf in self.named_input_nodes.items():
            buf_name = buf.get_name()
            if buf_name in mutated_input_names:
                mutated_input_names.add(arg_name)
            if isinstance(buf, BaseView):
                base = buf.unwrap_view()
                if isinstance(base, StorageBox):
                    base = base.data
                if isinstance(base, IRNode) and base.get_name() in mutated_input_names:
                    mutated_input_names.add(arg_name)
        self._helion_mutated_input_names = set(mutated_input_names)

        # Populate epilogue specs
        fused_epilogue_nodes: list[BaseSchedulerNode] = []
        if fusable_epilogue_nodes and self.supports_epilogue_fusion():
            mutation_outputs = {
                o.get_name() for o in self.outputs[1:] if isinstance(o, Buffer)
            }
            mutation_output_names = {
                o.get_name() for o in self.outputs[1:] if isinstance(o, MutationOutput)
            }
            outputs = {self.get_name()} | {
                o.get_name() for o in self.multi_output_nodes if isinstance(o, IRNode)
            }
            if self._output_aliases and self._output_alias_is_direct:
                direct_aliases = {
                    alias
                    for alias, is_direct in zip(
                        self._output_aliases, self._output_alias_is_direct, strict=False
                    )
                    if is_direct and alias
                }
                if direct_aliases & mutated_input_names:
                    outputs |= mutation_outputs | mutated_input_names

            for ep in fusable_epilogue_nodes:
                if not (isinstance(ep, BaseSchedulerNode) and ep.read_writes):
                    continue
                reads: set[str] = set()
                for d in ep.read_writes.reads:
                    if not isinstance(d, (MemoryDep, StarDep, WeakDep)):
                        continue
                    if (
                        isinstance(d, (StarDep, WeakDep))
                        and d.name in mutation_output_names
                    ):
                        continue
                    if d.name in outputs:
                        reads.add(d.name)
                        continue
                    buf = V.graph.get_buffer(d.name)
                    if isinstance(buf, BaseView) and is_trivial_view(buf):
                        base = buf.unwrap_view()
                        if isinstance(base, StorageBox):
                            base = base.data
                        if isinstance(base, IRNode) and base.get_name() in outputs:
                            reads.add(d.name)
                epilogue_sched_nodes = ep.get_nodes()
                if len(epilogue_sched_nodes) > 1:
                    node_set = set(epilogue_sched_nodes)
                    filtered: list[BaseSchedulerNode] = []
                    for snode in epilogue_sched_nodes:
                        outs = snode.get_outputs()
                        if any(
                            user.node not in node_set
                            for out in outs
                            for user in out.users
                            if hasattr(user, "node")
                        ):
                            filtered.append(snode)
                    if filtered:
                        epilogue_sched_nodes = filtered
                    graph_outputs = set(V.graph.get_output_names())
                    filtered_outputs = [
                        snode
                        for snode in epilogue_sched_nodes
                        if any(
                            o.get_name() in graph_outputs for o in snode.get_outputs()
                        )
                    ]
                    if filtered_outputs:
                        epilogue_sched_nodes = filtered_outputs
                if not epilogue_sched_nodes or has_non_trivial_view(
                    epilogue_sched_nodes
                ):
                    continue
                last_node = epilogue_sched_nodes[-1]
                epilogue_ir = _get_ir_node(last_node)
                if not isinstance(epilogue_ir, IRNode):
                    continue
                if len(reads) > 1:
                    ok = True
                    for name in reads:
                        kernel_out = V.graph.get_buffer(name)
                        if isinstance(kernel_out, MutationOutput):
                            mutation_bufs = kernel_out.get_mutation_buffers()
                            kernel_out = mutation_bufs[0] if mutation_bufs else None
                        if not isinstance(
                            kernel_out, IRNode
                        ) or not same_shape_and_stride(kernel_out, epilogue_ir):
                            ok = False
                            break
                    if ok:
                        self._multi_dep_epilogue_specs.append(([ep], reads))
                        fused_epilogue_nodes.append(ep)
                elif len(reads) == 1:
                    acc_name = next(iter(reads))
                    kernel_out = V.graph.get_buffer(acc_name)
                    if isinstance(kernel_out, MutationOutput):
                        mutation_bufs = kernel_out.get_mutation_buffers()
                        kernel_out = mutation_bufs[0] if mutation_bufs else None
                    if isinstance(kernel_out, IRNode) and same_shape_and_stride(
                        kernel_out, epilogue_ir
                    ):
                        self._epilogue_specs.setdefault(acc_name, []).append(ep)
                        fused_epilogue_nodes.append(ep)
                        if os.environ.get("HELION_DEBUG_FUSION"):
                            ep_name = (
                                ep.node.get_name() if hasattr(ep, "node") else str(ep)
                            )
                            print(
                                f"DEBUG epilogue registered: acc_name={acc_name} -> ep={ep_name}"
                            )

        with V.set_kernel_handler(self):
            if not only_gen_src_code:
                for node in [template_node, *fused_epilogue_nodes, *multi_output_nodes]:
                    node.mark_run()
            self._helion_epilogue_aliases = [
                ep.node.get_name()
                for nodes in self._epilogue_specs.values()
                for ep in nodes
                if isinstance(ep, BaseSchedulerNode) and isinstance(ep.node, IRNode)
            ]
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
        conditional = ("libdevice", "tl_math", "triton_helpers", "helion")
        for name in (*required, *(n for n in conditional if f"{n}." in src_code)):
            wrapper.add_import_once(library_imports[name])

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
