from __future__ import annotations

import ast
import contextlib
import dataclasses
from itertools import dropwhile
import types
from typing import TYPE_CHECKING
from typing import Callable
from typing import Sequence
from typing import cast

import sympy
import torch
from torch._inductor import config as inductor_fusion_config
from torch._inductor import dependencies
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.ir import Buffer
from torch._inductor.ir import ComputedBuffer
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
from torch._inductor.ops_handler import AddParenHandler
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree

from ._patches import apply_patches as _apply_patches

_apply_patches()
del _apply_patches

from .._dynamo.higher_order_ops import _rebuild_container_args
from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional
from .._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from .._dynamo.variables import _get_flat_output
from ..ast_extension import unparse
from ..ast_read_writes import ast_rename
from ..generate_ast import generate_ast
from ..indexing_strategy import SubscriptIndexing
from ..output_header import get_needed_imports
from ..output_header import library_imports

if TYPE_CHECKING:
    from torch._inductor.codegen.simd import SIMDScheduling
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.scheduler import BaseSchedulerNode

    from ..inductor_lowering import CodegenState
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


@dataclasses.dataclass
class _FusionResult:
    """Result of replaying a pointwise op through Inductor's ops handler.

    When we want to fuse an adjacent pointwise op (e.g. relu, add) into a
    Helion kernel, we "replay" it: run its inner_fn with a custom handler that
    records what it loads, computes, and stores.  This dataclass captures those
    recordings — the fused Triton expression, any extra loads it needs, and
    where it writes its output.
    """

    # Additional tl.load() calls needed by the fused expression.
    # Maps generated var name (e.g. "_epi_0") to (buffer_param_name, index_expr).
    fusion_loads: dict[str, tuple[str, str]] = dataclasses.field(default_factory=dict)
    # The fused Triton expression string, e.g. "relu(_kernel_val_0) + _epi_0"
    fused_expr: str | None = None
    # Buffer name the epilogue writes to. When this differs from the kernel's
    # original output buffer, the kernel's store is redirected to this buffer
    # and the original output buffer is removed.
    store_target: str | None = None


@dataclasses.dataclass
class EpilogueSpec:
    """Specification for epilogue fusion on a single hl.store."""

    # The fused Triton expression string that replaces the original store value,
    # e.g. "relu(_kernel_val_0) + _epi_0".
    fused_expr: str
    # Additional tl.load() calls needed by the fused expression.
    # Maps generated var name (e.g. "_epi_0") to (kernel_param_name, index_expr).
    fusion_loads: dict[str, tuple[str, str]]


@dataclasses.dataclass
class PrologueSpec:
    """Specification for prologue fusion on a single hl.load."""

    # The fused Triton expression string with a "_load_val" placeholder that
    # gets substituted with the actual hl.load result at codegen time,
    # e.g. "(_load_val).to(tl.float32)".
    fused_expr: str
    # Original source buffer name used by call_kernel to rewrite the input
    # argument. When set, the kernel reads from this buffer instead of the
    # prologue's output buffer, effectively bypassing the intermediate alloc.
    source_buf: str | None = None


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

        # Required by TritonOverrides ops (V.kernel.cse, V.kernel.min_elem_per_thread)
        self.cse = types.SimpleNamespace(varname_map={})
        self.min_elem_per_thread = 0

        # Maps output buffer name -> (kernel param name, MultiOutput indices).
        self._output_buf_to_param: dict[
            str, tuple[str | None, list[tuple[type, int]]]
        ] = {}
        # Extra kernel params for epilogue fusion (e.g. redirected outputs, extra inputs).
        self._epilogue_extra_params: list[tuple[str, str]] = []
        # Param renames when an epilogue stores to a different buffer than the kernel output.
        self._epilogue_renames: dict[str, str] = {}
        # True when kernel returns symbolic non-tensor values (e.g., SymInts
        # from x.size()), which produce unsupported ops in epilogue consumers.
        self._has_symbolic_return_values = False
        # Per-output-param fusion specs populated by _build_epilogue_specs.
        self._epilogue_specs: dict[str, EpilogueSpec] = {}
        # Per-input-param fusion specs populated by _build_prologue_specs.
        self._prologue_specs: dict[str, PrologueSpec] = {}

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
            # Mark all inputs as eligible for prologue fusion;
            # the scheduler decides which ones actually get fused.
            allowed_prologue_inps=OrderedSet(
                inp.get_name()
                for inp in inputs  # type: ignore[union-attr]
            ),
        )

        for inp in mutated_inputs_irnodes or []:
            if hasattr(inp, "get_name"):
                V.graph.never_reuse_buffers.add(inp.get_name())

    def _codegen_with_fusion(
        self,
        prologue_nodes: Sequence[BaseSchedulerNode],
        epilogue_nodes: list[BaseSchedulerNode],
        buf_name_to_prologue_group: dict[str, list[BaseSchedulerNode]],
    ) -> str:
        """Build fusion specs and regenerate kernel code.

        Called from codegen_template_override. Two phases:

        codegen_template_override
          └→ _codegen_with_fusion
               Phase 1 — trace fusion expressions:
               ├→ _build_prologue_specs
               │    └→ _extract_fusion_expr (per node) → self._prologue_specs
               ├→ _build_epilogue_specs
               │    └→ _extract_fusion_expr (per node) → self._epilogue_specs
               │
               Phase 2 — regenerate kernel (specs are now populated):
               └→ _generate_triton_ast
                    └→ generate_ast → Helion kernel codegen
                         ├→ hl.store (memory_ops.py) → codegen_epilogue_fusion
                         └→ hl.load  (memory_ops.py) → codegen_prologue_fusion
        """
        if prologue_nodes:
            self._build_prologue_specs(buf_name_to_prologue_group)
        if epilogue_nodes:
            self._build_epilogue_specs(epilogue_nodes)

        # Re-run Helion's generate_ast() to regenerate the Triton kernel AST.
        # During codegen, hl.store and hl.load (in memory_ops.py) check that
        # V.kernel is a HelionTemplateBuffer with non-empty fusion specs
        # and emit fused expressions inline.
        root = self._generate_triton_ast()

        assert root is not None

        # Inject extra params and apply redirect renames
        if self._epilogue_extra_params:
            # Helion always generates: inner function first, then host function.
            funcs = [
                n for n in ast.iter_child_nodes(root) if isinstance(n, ast.FunctionDef)
            ]
            assert len(funcs) >= 2
            inner_func, host_func = funcs[0], funcs[1]

            # Find the launcher call in the host function
            launcher_call = next(
                (
                    n
                    for n in ast.walk(host_func)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in ("_launcher", "_default_launcher")
                ),
                None,
            )
            # Add extra parameters to inner function, host function, and launcher.
            extra_param_names = [p for p, _ in self._epilogue_extra_params]
            for name in extra_param_names:
                inner_func.args.args.append(ast.arg(arg=name))
                host_func.args.args.append(ast.arg(arg=name))
                if launcher_call is not None:
                    launcher_call.args.append(ast.Name(id=name, ctx=ast.Load()))

            # Apply redirect renames so that tensor_descriptor params
            # (used in make_tensor_descriptor) are renamed correctly.
            if self._epilogue_renames:
                for orig_param, new_param in self._epilogue_renames.items():
                    ast_rename(inner_func, {orig_param: new_param})

        return self._ast_to_source(root)

    # Layout is always MultiOutputLayout: reads from inputs only,
    # writes go through MultiOutput children, no allocation needed.

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype for prologue-fusion heuristic checks.

        The parent TemplateBuffer.dtype does ``self.get_layout().dtype``,
        but our layout is MultiOutputLayout which has no dtype attribute.
        We override to infer dtype from the first input tensor instead.
        """
        if self.inputs:
            return self.inputs[0].get_dtype()  # type: ignore[union-attr]
        return torch.float32

    def extract_read_writes(self, normalize: bool = False) -> dependencies.ReadWrites:
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        for inp in self.inputs:
            if isinstance(inp, (ReinterpretView, Buffer)) and isinstance(
                inp.layout, Layout
            ):
                indexer = inp.layout.make_indexer()

                # Simulate a load through the input's layout indexer so that
                # extract_read_writes produces MemoryDep (with stride/offset info)
                # instead of a plain StarDep. The scheduler needs these richer
                # dependencies for correct buffer-grouping and fusion decisions.
                def dummy(
                    index: Sequence[object],
                    rindex: Sequence[object],
                    indexer: Callable[..., object] = indexer,
                    inp: IRNode = inp,
                ) -> object:
                    assert len(rindex) == 0
                    return ops.load(inp.get_name(), indexer(index))  # type: ignore[union-attr]

                reads |= dependencies.extract_read_writes(
                    dummy,
                    inp.get_size(),
                    (),
                    normalize=normalize,  # type: ignore[union-attr]
                ).reads
            else:
                reads.add(dependencies.StarDep(inp.get_name()))  # type: ignore[union-attr]
        # MemoryDep write needed for scheduler buffer-grouping and
        # FusedSchedulerNode.fuse() StarDep -> MemoryDep rewriting.
        writes: OrderedSet[dependencies.Dep] = OrderedSet()
        writes.add(
            dependencies.MemoryDep(
                self.get_name(),
                sympy.Integer(0),
                var_names=(),
                size=(),
            )
        )
        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=OrderedSet(),
            range_vars=None,
            var_ranges=None,
        )

    def can_fuse_multi_output_epilogue(self, snode: object) -> bool:
        """Control which scheduler nodes can fuse as an epilogue of this template."""
        n2_node = getattr(snode, "node", None)
        if isinstance(n2_node, MultiOutput):
            return (
                len(n2_node.inputs) == 1
                and n2_node.inputs[0].get_name()  # pyrefly: ignore[missing-attribute]
                == self.name
            )
        if not isinstance(n2_node, ComputedBuffer):
            return False

        # Check if snode can be fused as an epilogue (reads from kernel output)
        output_reads = {
            dep.name
            for dep in snode.read_writes.reads  # type: ignore[union-attr]
            if dep.name in self._output_buf_to_param
        }
        if not output_reads or len(output_reads) > 1:
            return False
        # Epilogue fusion requires a tl.store to fuse into, so reject buffers
        # without a kernel parameter mapping (which only tl.store outputs have).
        output_buf = next(iter(output_reads))
        val = self._output_buf_to_param.get(output_buf)
        if val is None or val[0] is None or self._has_symbolic_return_values:
            return False

        # Shape check: fusion reuses the kernel's store offset, only correct
        # when all involved buffers share the same shape.
        buf = V.graph.get_buffer(output_buf)
        if buf is None:
            return True
        output_shape = tuple(buf.get_size())
        for dep in snode.read_writes.writes:  # type: ignore[union-attr]
            b = V.graph.get_buffer(dep.name)
            if b is not None and tuple(b.get_size()) != output_shape:
                return False
        # Reject if output shape doesn't match any template input shape,
        # which indicates a shape-changing wrapper operation (e.g. reduction).
        input_shapes = {tuple(inp.get_size()) for inp in self.inputs}  # type: ignore[union-attr]
        return not input_shapes or output_shape in input_shapes

    def should_allocate(self) -> bool:
        return False

    def get_size(self) -> Sequence[sympy.Expr]:
        return []

    def _generate_triton_ast(self) -> ast.Module | None:
        """Generate and rename the Triton kernel AST.

        Returns the AST with function names replaced by Placeholder.KERNEL_NAME,
        or None if the bound kernel is not available.
        """
        if not self._bound_kernel:
            return None
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

        # Collect module-level variable names that need uniquification
        # (constexpr assignments like _BLOCK_SIZE_0 = tl.constexpr(32))
        assert isinstance(root, ast.Module)
        module_level_vars: dict[str, str] = {}
        for node in root.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_level_vars[target.id] = (
                            f"{target.id}_{Placeholder.KERNEL_NAME}"
                        )

        # Rename functions, module-level vars, and update references
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                if node.name == host_fn:
                    node.name = str(Placeholder.KERNEL_NAME)
                elif node.name == inner_fn:
                    node.name = inner_fn_placeholder
            elif isinstance(node, ast.Name):
                if node.id == inner_fn:
                    node.id = inner_fn_placeholder
                elif node.id in module_level_vars:
                    node.id = module_level_vars[node.id]

        return root  # pyrefly: ignore[bad-return]

    def _ast_to_source(self, root: ast.Module) -> str:
        """Convert AST to source code with imports."""
        return get_needed_imports(root) + unparse(
            root, output_origin_lines=self._bound_kernel.settings.output_origin_lines
        )

    def render(self) -> PartialRender:
        """Generate Triton code."""
        root = self._generate_triton_ast()
        if root is None:
            return PartialRender("", {})
        return PartialRender(self._ast_to_source(root), {})

    @staticmethod
    def _extract_fusion_expr(
        ir_node: ComputedBuffer,
        intercept_bufs: set[str],
        intercept_value: str,
        get_extra_input: Callable[[str], str],
        var_prefix: str,
    ) -> _FusionResult:
        """Trace a ComputedBuffer's inner expression to extract a fused Triton expression.

        Replays the buffer's ops through a custom handler that intercepts loads
        (replacing the kernel's own output with ``intercept_value`` and recording
        extra loads from other buffers) and captures the final store expression.
        Returns a ``_FusionResult`` with the fused expression string, any extra
        loads, and the store target.
        """
        result = _FusionResult()

        class _Handler(AddParenHandler):
            def load(self, name: str, index: object) -> str:
                if name in intercept_bufs:
                    # Epilogue fusion only handles elementwise ComputedBuffer ops.
                    # Match Inductor's convention: upcast fp16/bf16 to fp32 for
                    # pointwise computation (TritonOverrides assumes fp32 inputs).
                    dtype = V.graph.get_dtype(name)
                    if dtype in (torch.float16, torch.bfloat16):
                        return f"({intercept_value}).to(tl.float32)"
                    return intercept_value
                param = get_extra_input(name)
                var_name = f"{var_prefix}{len(result.fusion_loads)}"
                result.fusion_loads[var_name] = (param, str(index))
                return var_name

            def store(
                self, name: str, index: object, value: object, mode: object = None
            ) -> None:
                result.fused_expr = str(value)
                result.store_target = name

        handler = _Handler(TritonOverrides())
        pw = ir_node.data
        # Use symbolic indices so extra loads capture correct index expressions
        index = [sympy.Symbol(f"_fidx_{i}") for i in range(len(pw.ranges))]
        with V.set_ops_handler(handler):  # pyrefly: ignore[bad-argument-type]
            value = pw.inner_fn(index)
            handler.store(ir_node.get_name(), index, value)

        return result

    def _build_epilogue_specs(
        self,
        epilogue_nodes: Sequence[BaseSchedulerNode],
    ) -> None:
        """Trace each epilogue node to build per-output fusion specs.

        For each epilogue node, extracts the fused Triton expression via
        _extract_fusion_expr and records it in self._epilogue_specs.  When an
        epilogue stores to a different buffer than the kernel's original output,
        adds a redirect rename (self._epilogue_renames) and an extra output
        parameter (self._epilogue_extra_params).  Extra input buffers referenced
        by the epilogue expression are also added to self._epilogue_extra_params.
        """
        extra_inputs: dict[str, str] = {}  # buf_name -> param_name

        for epi_idx, epi_node in enumerate(epilogue_nodes):
            kernel_output_buf = next(
                d.name
                for d in epi_node.read_writes.reads
                if d.name in self._output_buf_to_param
            )
            target_param = self._output_buf_to_param[kernel_output_buf][0]
            assert target_param is not None

            fusion_result = self._extract_fusion_expr(
                epi_node.node,  # pyrefly: ignore[bad-argument-type]
                intercept_bufs=set(self._output_buf_to_param),
                intercept_value=f"_kernel_val_{epi_idx}",
                get_extra_input=lambda name: extra_inputs.setdefault(
                    name, f"_epi_input_{len(extra_inputs)}"
                ),
                var_prefix="_epi_",
            )

            # Handle epilogue output redirect (stores to different buffer)
            if (
                fusion_result.store_target is not None
                and fusion_result.store_target != kernel_output_buf
            ):
                epi_out_param = f"_epi_out_{len(self._epilogue_extra_params)}"
                self._epilogue_renames[target_param] = epi_out_param
                self._epilogue_extra_params.append(
                    (epi_out_param, fusion_result.store_target)
                )
                self.removed_buffers.add(kernel_output_buf)

            self._epilogue_specs[target_param] = EpilogueSpec(
                fused_expr=fusion_result.fused_expr,  # pyrefly: ignore[bad-argument-type]
                fusion_loads=fusion_result.fusion_loads,
            )

        # Add extra input parameters
        for buf_name, param_name in extra_inputs.items():
            self._epilogue_extra_params.append((param_name, buf_name))

    def _build_prologue_specs(
        self,
        buf_name_to_prologue_group: dict[str, list[BaseSchedulerNode]],
    ) -> None:
        """Trace each prologue node to build per-input fusion specs.

        For each entry in buf_name_to_prologue_group (scheduler-provided mapping
        of input buffer name → prologue nodes), extracts the fused Triton
        expression via _extract_fusion_expr and records it in self._prologue_specs.
        The fused expression uses a ``_load_val`` placeholder that gets replaced
        with the actual hl.load result at codegen time.
        """
        input_buf_to_params: dict[str, list[str]] = {}
        for param_name, inp in self.named_input_nodes.items():
            buf_name = inp.get_name()  # type: ignore[union-attr]
            input_buf_to_params.setdefault(buf_name, []).append(param_name)

        for fused_buf_name, pro_nodes in buf_name_to_prologue_group.items():
            if fused_buf_name not in input_buf_to_params:
                continue

            for pro_node in pro_nodes:
                source_buf_names = {dep.name for dep in pro_node.read_writes.reads}

                fusion_result = self._extract_fusion_expr(
                    pro_node.node,  # pyrefly: ignore[bad-argument-type]
                    intercept_bufs=source_buf_names,
                    intercept_value="_load_val",
                    get_extra_input=lambda name: name,
                    var_prefix="_pro_",
                )

                source_buf = next(iter(source_buf_names)) if source_buf_names else None
                for input_param in input_buf_to_params[fused_buf_name]:
                    self._prologue_specs[input_param] = PrologueSpec(
                        fused_expr=fusion_result.fused_expr,  # pyrefly: ignore[bad-argument-type]
                        source_buf=source_buf,
                    )

    def call_kernel(
        self, kernel_name: str, template_buffer: TritonTemplateBuffer | None = None
    ) -> None:
        """Emit the kernel call site."""
        wrapper = V.graph.wrapper_code
        output_name = self.get_name()
        reinterp_count = 0

        def get_input_expr(arg_name: str, inp: IRNode) -> str:
            nonlocal reinterp_count
            buf_name = inp.get_name()  # type: ignore[union-attr]
            pro_spec = self._prologue_specs.get(arg_name)
            source_buf = pro_spec.source_buf if pro_spec is not None else None

            if source_buf is not None:
                # This input's buffer is prologue-fused: use source buffer.
                if isinstance(inp, ReinterpretView):
                    # Preserve the view (strides/offsets) but point to source
                    sizes = tuple(inp.get_size())
                    strides = tuple(inp.get_stride())
                    offset = inp.layout.offset
                    expr = f"reinterpret_tensor({source_buf}, {sizes}, {strides}, {offset})"
                    wrapper.writeline(f"reinterp_{reinterp_count} = {expr}")
                    result = f"reinterp_{reinterp_count}"
                    reinterp_count += 1
                    return result
                return source_buf

            if not isinstance(inp, ReinterpretView):
                return buf_name
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

        # Add epilogue extra parameters (outputs and inputs)
        args.extend(buf_name for _, buf_name in self._epilogue_extra_params)
        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

        # Emit MultiOutput extraction code for each output buffer.
        # MultiOutput nodes are marked as run by codegen_template_override,
        # so their separate codegen is suppressed. We must emit the extraction
        # here so that downstream consumers can reference the buffer names.
        for mo_name, (_param, indices) in sorted(self._output_buf_to_param.items()):
            if mo_name not in self.removed_buffers:
                idx_str = output_name
                for _, idx in indices:
                    idx_str = f"{idx_str}[{idx}]"
                wrapper.writeline(f"{mo_name} = {idx_str}")

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
            partial_code = render()
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize_remaining()
            )

            # Separate MultiOutput nodes (tuple extraction) from actual epilogue ops
            multi_output_nodes = [
                n
                for n in epilogue_nodes
                if isinstance(getattr(n, "node", None), MultiOutput)
            ]
            epilogue_nodes_no_mo = [
                n
                for n in epilogue_nodes
                if not isinstance(getattr(n, "node", None), MultiOutput)
            ]

            if prologue_nodes or epilogue_nodes_no_mo:
                src_code = self._codegen_with_fusion(
                    prologue_nodes, epilogue_nodes_no_mo, buf_name_to_prologue_group
                )
            if inductor_fusion_config.benchmark_kernel:
                src_code = f"\n{src_code}\n{IndentedBuffer().getvalue()}"

            if only_gen_src_code:
                return src_code

            # Build node schedule, mark all as run, and define kernel
            node_schedule: list[BaseSchedulerNode] = [
                template_node,
                *multi_output_nodes,
                *prologue_nodes,
                *epilogue_nodes_no_mo,
            ]
            for node in node_schedule:
                node.mark_run()
            self.kernel_name = scheduling.define_kernel(src_code, node_schedule, self)
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
            if isinstance(tb.data, MultiOutput):
                # MultiOutput nodes (outputs of other template buffers) already
                # have FixedLayout with the correct strides.  realize_input()
                # falls through to copy_input() for these, creating a new buffer
                # with FlexibleLayout that defaults to contiguous strides -- this
                # loses the original (potentially non-contiguous) stride info.
                # Use the MultiOutput directly to preserve its layout.
                result = tb.data
            else:
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
    flat_leaves, tree_spec, return_ast = _get_flat_output(bound.host_function)
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
        output: object,
        indices: list[tuple[type, int]],
        ast_node: ast.expr | None,
    ) -> list[TensorBox]:
        if isinstance(output, (list, tuple)):
            elts = (
                ast_node.elts if isinstance(ast_node, (ast.Tuple, ast.List)) else None
            )  # type: ignore[union-attr]
            return [
                r
                for i in range(len(output))
                for r in collect_tensor_outputs(
                    output[i],
                    [*indices, (type(output), i)],
                    elts[i] if elts is not None else None,
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
            # Map output buffer name to (kernel param name, extraction indices)
            buf._output_buf_to_param[mo.get_name()] = (
                ast_node.id if isinstance(ast_node, ast.Name) else None,
                indices,
            )
            tb = TensorBox(mo)
            seen_outputs[tid] = tb
            return [tb]
        # Non-tensor leaf with a non-constant AST node (e.g., x.size(0))
        # indicates a symbolic return value that would produce unsupported
        # ops (index_expr) in epilogue consumers.
        if ast_node is not None and not isinstance(ast_node, ast.Constant):
            buf._has_symbolic_return_values = True
        return []

    return tuple(collect_tensor_outputs(structured, [], return_ast))


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


def codegen_epilogue_fusion(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object],
    value: ast.AST,
    extra_mask: ast.AST | None,
) -> ast.AST:
    """Emit fused epilogue code inline within an hl.store during Triton codegen.

    Called from memory_ops.py when epilogue fusion is enabled.  Assigns the
    original store value to a temp variable (_kernel_val_N), emits any extra
    tl.load statements needed by the epilogue expression (with broadcast-aware
    offset substitution), and returns the fused expression AST that replaces the
    original store value.
    """
    kernel = V.kernel
    assert isinstance(kernel, HelionTemplateBuffer) and kernel._epilogue_specs
    param_name = state.device_function.tensor_arg(tensor).name
    spec = kernel._epilogue_specs.get(param_name)
    if spec is None:
        return value

    # Emit: _kernel_val_N = <original_value>
    # Unique per-epilogue name avoids Triton type conflicts across branches.
    epi_idx = list(kernel._epilogue_specs.keys()).index(param_name)
    kernel_val_name = f"_kernel_val_{epi_idx}"
    state.add_statement(
        ast.Assign(
            targets=[
                ast.Name(id=kernel_val_name, ctx=ast.Store())
            ],  # pyrefly: ignore[missing-attribute]
            value=value,  # pyrefly: ignore[bad-argument-type]
            lineno=0,
        )
    )

    # Emit tl.load statements for extra loads (broadcast-aware)
    if spec.fusion_loads:  # pyrefly: ignore[missing-attribute]
        indexing = SubscriptIndexing.create(state, tensor, [*subscript], extra_mask)
        offset_str = ast.unparse(indexing.index_expr)
        mask_str = ast.unparse(indexing.mask_expr)
        has_mask = indexing.has_mask()

        for var_name, (
            param_name,
            sympy_idx,
        ) in spec.fusion_loads.items():  # pyrefly: ignore[missing-attribute]
            if sympy_idx and indexing.dim_index_exprs:
                # Substitute _fidx_i with actual Triton index expressions.
                # Reverse order avoids substring collisions (_fidx_1 vs _fidx_10).
                load_offset = sympy_idx
                for i in range(len(indexing.dim_index_exprs) - 1, -1, -1):
                    load_offset = load_offset.replace(
                        f"_fidx_{i}", f"({indexing.dim_index_exprs[i]})"
                    )
                load_offset = load_offset.strip()
            else:
                load_offset = offset_str
            ptr = param_name if load_offset == "0" else f"{param_name} + {load_offset}"
            mask_part = f", {mask_str}, other=0" if has_mask else ""
            state.add_statement(f"{var_name} = tl.load({ptr}{mask_part})")

    # Parse the final fused value expression
    return ast.parse(
        spec.fused_expr, mode="eval"
    ).body  # pyrefly: ignore[no-matching-overload]


def codegen_prologue_fusion(
    state: CodegenState,
    tensor: torch.Tensor,
    value: ast.AST,
) -> ast.AST:
    """Emit fused prologue code inline within an hl.load during Triton codegen.

    Called from memory_ops.py when prologue fusion is enabled.  Substitutes the
    ``_load_val`` placeholder in the prologue spec's fused expression with the
    actual load AST, effectively inlining the prologue op (e.g. dtype cast)
    into the load site.
    """
    kernel = V.kernel
    assert isinstance(kernel, HelionTemplateBuffer) and kernel._prologue_specs
    param_name = state.device_function.tensor_arg(tensor).name
    spec = kernel._prologue_specs.get(param_name)
    if spec is None:
        return value
    load_str = ast.unparse(value)
    return ast.parse(spec.fused_expr.replace("_load_val", load_str), mode="eval").body
