"""HelionTemplateBuffer - IR node for Helion kernels in Inductor."""

from __future__ import annotations

import ast
import contextlib
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
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
from torch._inductor.dependencies import StarDep
from torch._inductor.dependencies import WeakDep
from torch._inductor.ir import BaseView
from torch._inductor.ir import Buffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import FlexibleLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import Layout
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import OutputSpec
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import TritonTemplateBuffer
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops
from torch.utils._ordered_set import OrderedSet

from ...language import atomic_ops
from ...runtime.config import Config
from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..output_header import get_needed_imports
from ..output_header import library_imports

if TYPE_CHECKING:
    from types import TracebackType

    from ..device_ir import DeviceIR


# Set of atomic operation functions for checking device IR
_ATOMIC_OPS: set[Callable[..., Any]] = {
    getattr(atomic_ops, name)
    for name in atomic_ops.__all__
    if callable(getattr(atomic_ops, name, None))
}


def _device_ir_uses_atomics(device_ir: DeviceIR) -> bool:
    """Check if a device IR contains any atomic operations.

    This checks the FX graph nodes directly without needing to generate code,
    allowing us to determine atomics usage without requiring a config.
    """
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op == "call_function" and node.target in _ATOMIC_OPS:
                return True
    return False


def _get_ir_node(n: Any) -> Any:  # noqa: ANN401
    """Extract the IR node from a scheduler node."""
    return n.node if isinstance(n, BaseSchedulerNode) else n


def _get_name(node: object) -> str:
    """Get the name of an IR node (typed wrapper for missing stubs)."""
    return node.get_name()  # type: ignore[union-attr]


def has_view(nodes: Sequence[BaseSchedulerNode]) -> bool:
    """Check if any node in the list is a view operation."""
    return any(isinstance(_get_ir_node(n), BaseView) for n in nodes)


def same_shape_and_stride(lhs: IRNode, rhs: IRNode) -> bool:
    """Check if two IR nodes have the same shape and stride."""
    sizevars = V.graph.sizevars
    if not sizevars.statically_known_list_equals(
        list(lhs.get_size()), list(rhs.get_size())
    ):
        return False
    # Buffer and its subclasses (ComputedBuffer, InputBuffer, etc.) have get_stride()
    if isinstance(lhs, Buffer) and isinstance(rhs, Buffer):
        if not sizevars.statically_known_list_equals(
            list(lhs.get_stride()), list(rhs.get_stride())
        ):
            return False
    return True


def _rewrite_dtype_references(
    root: ast.AST, dtype_map: dict[str, torch.dtype]
) -> ast.AST:
    """Rewrite dtype references when prologue changes tensor dtype.

    Handles two patterns:
    - torch.*_like(x) -> torch.*(x.shape, dtype=new, device=x.device)
    - dtype=x.dtype -> dtype=torch.{new}
    """
    like_funcs = {
        "empty_like",
        "zeros_like",
        "ones_like",
        "full_like",
        "randn_like",
        "rand_like",
    }

    def make_dtype_attr(dt: torch.dtype) -> ast.Attribute:
        return ast.Attribute(
            value=ast.Name(id="torch", ctx=ast.Load()),
            attr=str(dt).split(".")[-1],
            ctx=ast.Load(),
        )

    class DtypeRewriter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            node = cast("ast.Call", self.generic_visit(node))
            func = node.func
            # torch.*_like(x) -> torch.*(x.shape, dtype=new, device=x.device)
            if (
                isinstance(func, ast.Attribute)
                and func.attr in like_funcs
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and (arg := node.args[0].id) in dtype_map
            ):
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="torch", ctx=ast.Load()),
                        attr=func.attr.replace("_like", ""),
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Attribute(
                            value=ast.Name(id=arg, ctx=ast.Load()),
                            attr="shape",
                            ctx=ast.Load(),
                        ),
                        *node.args[1:],
                    ],
                    keywords=[
                        ast.keyword(arg="dtype", value=make_dtype_attr(dtype_map[arg])),
                        ast.keyword(
                            arg="device",
                            value=ast.Attribute(
                                value=ast.Name(id=arg, ctx=ast.Load()),
                                attr="device",
                                ctx=ast.Load(),
                            ),
                        ),
                        *node.keywords,
                    ],
                )
            # dtype=x.dtype -> dtype=torch.{new}
            new_kw = [
                ast.keyword(
                    arg="dtype", value=make_dtype_attr(dtype_map[kw.value.value.id])
                )
                if (
                    kw.arg == "dtype"
                    and isinstance(kw.value, ast.Attribute)
                    and kw.value.attr == "dtype"
                    and isinstance(kw.value.value, ast.Name)
                    and kw.value.value.id in dtype_map
                )
                else kw
                for kw in node.keywords
            ]
            return (
                ast.Call(func=node.func, args=node.args, keywords=new_kw)
                if new_kw != node.keywords
                else node
            )

    return DtypeRewriter().visit(root)


class HelionTemplateBuffer(TritonTemplateBuffer):
    """Helion kernel IR node with fusion support."""

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
        autotune_args: Sequence[Any] | None = None,
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
        self._autotune_args = autotune_args  # For autotuning during render()

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

        # Try to get implicit config, or trigger autotuning if none available
        cfg = self._bound_kernel._implicit_config()
        if cfg is None:
            # No implicit config available - trigger autotuning
            if self._autotune_args is not None:
                # Discover captured buffers via dry-run and create dummy tensors.
                # This solves the chicken-and-egg problem: we need to generate code
                # to know which buffers are captured, but we need captured buffer
                # tensors to benchmark the fused code.
                captured_buffer_dummies = self._create_captured_buffer_dummies()

                # Augment autotune args with dummy captured buffer tensors
                augmented_args = tuple(self._autotune_args) + captured_buffer_dummies

                # Now autotune WITH fusion context so benchmarks use fused code
                cfg = self._bound_kernel.autotune(
                    augmented_args,
                    force=False,
                    template_buffer=self,
                )
            else:
                raise RuntimeError(
                    "No config available and no autotune args provided. "
                    "Either provide a config via @helion.kernel(configs=[...]) "
                    "or use autotune_effort='none' for default config."
                )
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

            for node in ast.walk(root):
                if isinstance(node, ast.FunctionDef) and node.name == triton_fn:
                    _insert_params(
                        node.args.args,
                        params,
                        lambda p: ast.arg(arg=p),
                    )
                elif isinstance(node, ast.FunctionDef) and node.name == host_fn:
                    _insert_params(
                        node.args.args,
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
                            node.args,
                            params,
                            lambda p: ast.Name(id=p, ctx=ast.Load()),
                        )
        else:
            for node in ast.walk(root):
                if isinstance(node, ast.FunctionDef) and node.name == host_fn:
                    node.name = str(Placeholder.KERNEL_NAME)
                    break

        # Fix torch.*_like(x) when prologue fusion changes the dtype.
        # The call site passes the original tensor (e.g., fp16), but the prologue
        # computes a different dtype (fp32). We need the prologue output dtype.
        prologue_dtype_changes: dict[str, torch.dtype] = {}
        for arg_name, (nodes, _) in self._prologue_specs.items():
            if not nodes:
                continue
            prologue_out = _get_ir_node(nodes[-1])
            if not isinstance(prologue_out, IRNode):
                continue
            new_dtype = prologue_out.get_dtype()
            # Find original input dtype from first prologue node's reads
            first_node = nodes[0]
            if hasattr(first_node, "read_writes") and first_node.read_writes:
                for dep in first_node.read_writes.reads:
                    if isinstance(dep, MemoryDep):
                        orig_buf = V.graph.get_buffer(dep.name)
                        if isinstance(orig_buf, IRNode):
                            orig_dtype = orig_buf.get_dtype()
                            if orig_dtype != new_dtype:
                                prologue_dtype_changes[arg_name] = new_dtype
                            break

        if prologue_dtype_changes:
            root = _rewrite_dtype_references(root, prologue_dtype_changes)
            ast.fix_missing_locations(root)

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

        # Build arg-name -> input-name mapping, replacing fused inputs with captured buffers
        arg_inputs: dict[str, str] = {}
        for arg_name, inp in self.named_input_nodes.items():
            if isinstance(inp, ReinterpretView):
                storage_name = _get_name(inp.data)
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
                view_offset = 0
                inp_name = emit_reinterpret(
                    storage_name, view_size, view_stride, view_offset
                )
            else:
                inp_name = _get_name(inp)

            if (
                inp_name in self.prologue_fused_inputs
                and self._captured_buffers
                and isinstance(inp, IRNode)
            ):
                matched = None
                inp_size = inp.get_size()
                for buf_name in self._captured_buffers:
                    buf = V.graph.get_buffer(buf_name)
                    if isinstance(buf, IRNode) and buf.get_size() == inp_size:
                        matched = buf_name
                        break
                if matched is not None:
                    inp_name = matched

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
        aliases = [
            a
            for a in self._helion_epilogue_aliases
            if a != output_name and a not in mo_names
        ]
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
                    if ep_name in alias_map:
                        alias_map[ep_name] = target

        for a in aliases:
            wrapper.writeline(f"{a} = {alias_map[a]}")

    def capture_buffer(self, buffer_name: str, epilogue: bool = True) -> str:
        """Register a captured buffer and return its parameter name.

        Called by codegen.py when fusion needs to access an external buffer.
        """
        # Return existing param name if buffer already captured
        if buffer_name in self._captured_buffers:
            return self._captured_buffers[buffer_name][0]
        count = sum(
            1 for _, is_epi in self._captured_buffers.values() if is_epi == epilogue
        )
        param_name = f"{'epilogue' if epilogue else 'prologue'}_input_{count}"
        self._captured_buffers[buffer_name] = (param_name, epilogue)
        return param_name

    def _create_captured_buffer_dummies(self) -> tuple[torch.Tensor, ...]:
        """Discover captured buffers via dry-run code generation and create dummy tensors.

        This solves the chicken-and-egg problem: we need to generate code to know
        which buffers are captured, but we need captured buffer tensors to benchmark
        the fused code. Solution: do a discovery pass, create dummies, then autotune.
        """
        assert self._bound_kernel is not None

        # Save current state
        saved_captured = self._captured_buffers.copy()
        self._captured_buffers.clear()

        # Do a dry-run code generation to discover captured buffers
        cfg = self._bound_kernel.env.config_spec.default_config()
        with self._bound_kernel.env as env:
            env.set_template_buffer(self)
            try:
                generate_ast(
                    self._bound_kernel.host_function, cfg, emit_repro_caller=False
                )
            except Exception:
                pass  # Ignore errors, we just want to discover captured buffers

        # Create dummy tensors for discovered captured buffers
        dummy_tensors = []
        for buffer_name in self._captured_buffers:
            buf = V.graph.get_buffer(buffer_name)
            if buf is not None:
                dummy = self._create_dummy_tensor(buf)
                dummy_tensors.append(dummy)

        # Restore state (will be repopulated during actual autotuning)
        self._captured_buffers = saved_captured

        return tuple(dummy_tensors)

    def _create_dummy_tensor(self, buf: IRNode) -> torch.Tensor:
        """Create a dummy tensor from an IRNode buffer using empty_strided."""
        from torch._inductor import config as inductor_config

        # Get concrete sizes (convert symbolic to int)
        sizes = tuple(
            V.graph.sizevars.size_hint(
                s, fallback=inductor_config.unbacked_symint_fallback
            )
            for s in buf.get_size()
        )

        # Get concrete strides
        strides = tuple(
            V.graph.sizevars.size_hint(
                s, fallback=inductor_config.unbacked_symint_fallback
            )
            for s in buf.get_stride()
        )

        return torch.empty_strided(
            sizes,
            strides,
            dtype=buf.get_dtype(),
            device=buf.get_device(),
        )

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

        This method checks the device IR directly without generating code,
        allowing it to work without requiring a config. This is important
        because uses_atomics() is called during __init__ before the scheduler
        makes fusion decisions and before autotuning can happen.
        """
        if self._uses_atomics_cache is not None:
            return self._uses_atomics_cache
        if not self._bound_kernel:
            self._uses_atomics_cache = True
            return True

        # Check the device IR directly without needing a config
        device_ir = self._bound_kernel.host_function.device_ir
        self._uses_atomics_cache = _device_ir_uses_atomics(device_ir)
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
            and _get_name(node2.node.inputs[0]) == self.get_name()
        )

    def get_layout(self) -> Layout:
        """Get layout, handling multi-output case."""
        if isinstance(self.layout, MultiOutputLayout):
            mo = self.multi_output_nodes
            assert mo and isinstance(mo[0], MultiOutput), (
                "MultiOutputLayout without multi_output_nodes"
            )
            return cast("Layout", mo[0].layout)
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
                # Capture loop variables to avoid late binding issues
                def _dummy(index, rindex, _name=mo_name, _indexer=mo_indexer):  # noqa: ANN001, ANN202
                    assert len(rindex) == 0
                    return ops.store(_name, _indexer(index), "fake")

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
        template_writes: OrderedSet[Dep],
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

    def codegen_template_override(  # noqa: ANN201
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
        prologue_groups = buf_name_to_prologue_group or {}

        multi_output_nodes = [
            n
            for n in (epilogue_nodes or [])
            if isinstance(_get_ir_node(n), MultiOutput)
        ]
        # Populate self.multi_output_nodes with the actual MultiOutput IR nodes
        # Only update if we found MultiOutput nodes in epilogue_nodes
        if multi_output_nodes:
            self.multi_output_nodes = [_get_ir_node(n) for n in multi_output_nodes]
        fusable_epilogue_nodes = [
            n
            for n in (epilogue_nodes or [])
            if not isinstance(_get_ir_node(n), MultiOutput)
        ]

        fused_epilogue_nodes: list[BaseSchedulerNode] = []
        if fusable_epilogue_nodes and not self.uses_atomics():
            outputs = {self.get_name()} | {
                o.get_name() for o in self.multi_output_nodes if isinstance(o, IRNode)
            }
            for ep in fusable_epilogue_nodes:
                if not (isinstance(ep, BaseSchedulerNode) and ep.read_writes):
                    continue
                reads = {
                    d.name
                    for d in ep.read_writes.reads
                    if isinstance(d, (MemoryDep, StarDep, WeakDep))
                    and d.name in outputs
                }
                epilogue_nodes = ep.get_nodes()
                if not epilogue_nodes or has_view(epilogue_nodes):
                    continue
                last_node = epilogue_nodes[-1]
                epilogue_ir = _get_ir_node(last_node)
                if not isinstance(epilogue_ir, IRNode):
                    continue
                if len(reads) > 1:
                    ok = True
                    for name in reads:
                        kernel_out = V.graph.get_buffer(name)
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
                    if isinstance(kernel_out, IRNode) and same_shape_and_stride(
                        kernel_out, epilogue_ir
                    ):
                        self._epilogue_specs.setdefault(acc_name, []).append(ep)
                        fused_epilogue_nodes.append(ep)

        for name, buf in self.named_input_nodes.items():
            buf_name = buf.get_name()
            if buf_name in prologue_groups:
                p = prologue_groups[buf_name]
                if has_view(p):
                    continue
                self._prologue_specs[name] = (list(p), buf_name)
        fusable_prologue_nodes = [
            n
            for nodes in prologue_groups.values()
            if not has_view(nodes)
            for n in nodes
        ]

        with self:
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
            for buffer in self.named_input_nodes.values():
                buf_name = buffer.get_name()
                if buf_name in buf_name_to_prologue_group:
                    for pn in buf_name_to_prologue_group[buf_name]:
                        if pn in fusable_prologue_nodes:
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
                *fused_epilogue_nodes,
            ]
            if inductor_fusion_config.benchmark_kernel:
                # Benchmark stubs inlined - Helion doesn't support benchmarking
                src_code = f"\n{src_code}\n{IndentedBuffer().getvalue()}"
            if only_gen_src_code:
                return src_code
            self.kernel_name = scheduling.define_kernel(src_code, node_schedule, self)
        return self

    def emit_kernel_override(  # noqa: ANN201
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit kernel handler context."""
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

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
