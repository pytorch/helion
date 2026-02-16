from __future__ import annotations

import ast
import contextlib
from itertools import chain
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
from torch._inductor.ir import BaseView
from torch._inductor.ir import ExternKernel
from torch._inductor.ir import FallbackKernel
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
from torch.fx.experimental.symbolic_shapes import CallMethodKey
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



def _size_hint(expr: sympy.Expr | int, fallback: int | None = None) -> int | None:
    """Resolve a sympy expression to a concrete int hint."""
    if isinstance(expr, (int, sympy.Integer)):
        return int(expr)
    result = V.graph.sizevars.shape_env.size_hint(expr, allow_none=True)
    return int(result) if result is not None else fallback


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
        # Maps unbacked stride symbols created for outputs to their KeyPath
        # bindings (for use with codegen_unbacked_symbol_defs_for_outputs).
        self._output_stride_defs: dict[sympy.Symbol, pytree.KeyPath] = {}
        # Maps unbacked stride symbols created for inputs to (arg_name, dim).
        # arg_name is the kernel parameter name (not the graph input name),
        # so call_kernel() can resolve it via arg_inputs to get the correct
        # codegen expression (e.g. a reinterpret_tensor variable for views).
        # The helion node "owns" these symbols — it reports them via
        # get_unbacked_symbol_defs() and emits their definitions in call_kernel().
        self._input_stride_defs: dict[sympy.Symbol, tuple[str, int]] = {}

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

    @property
    def unbacked_bindings(self) -> dict[sympy.Symbol, pytree.KeyPath] | None:
        """Combined unbacked bindings for scheduler CUDA graph partitioning.

        The Inductor scheduler checks this attribute to identify nodes with
        unbacked symbol definitions.  Returns output stride defs (already
        KeyPaths) merged with input stride defs (converted to KeyPaths).
        """
        if not self._output_stride_defs and not self._input_stride_defs:
            return None
        bindings: dict[sympy.Symbol, pytree.KeyPath] = dict(self._output_stride_defs)
        for sym, (_arg_name, dim) in self._input_stride_defs.items():
            bindings[sym] = (CallMethodKey("stride"), pytree.SequenceKey(dim))
        return bindings

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

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        """Return unbacked symbols defined by this kernel.

        Includes both output stride symbols (from _output_stride_defs) and
        input stride symbols (from _input_stride_defs).  Reporting them
        here lets the scheduler know the helion node defines these symbols
        so they don't need to come from an external scope.
        """
        return OrderedSet(self._output_stride_defs.keys()) | OrderedSet(self._input_stride_defs.keys())

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

        # Emit definitions for output stride symbols via the wrapper's
        # WrapperLine abstraction (needed for CUDA graph / FX IR codegen).
        if self._output_stride_defs:
            wrapper.codegen_unbacked_symbol_defs_for_outputs(
                output_name, [], self._output_stride_defs)

        # Emit definitions for input stride symbols, also via WrapperLine.
        # Group by resolved codegen name (e.g. reinterp_0 for views).
        if self._input_stride_defs:
            input_groups: dict[str, dict[sympy.Symbol, pytree.KeyPath]] = {}
            for sym, (arg_name, dim) in self._input_stride_defs.items():
                codegen_name = arg_inputs.get(arg_name, arg_name)
                if codegen_name not in input_groups:
                    input_groups[codegen_name] = {}
                input_groups[codegen_name][sym] = (
                    CallMethodKey("stride"), pytree.SequenceKey(dim))
            for codegen_name, bindings in input_groups.items():
                wrapper.codegen_unbacked_symbol_defs_for_outputs(
                    codegen_name, [], bindings)

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



def _realize_inputs(tensor_args: dict[str, TensorBox]) -> dict[str, IRNode]:
    """Realize TensorBox inputs to buffer/ReinterpretView IR nodes.

    Results are cached on TensorBox objects so that subsequent kernel
    lowerings for the same input get the original (pre-relaxation) node.
    """
    realized: dict[str, IRNode] = {}
    for n, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            cached = getattr(tb, "_helion_realized", None)
            if cached is not None:
                realized[n] = cached
                continue
            result = ExternKernel.realize_input(tb)
            if isinstance(result, StorageBox):
                result = result.data
            if isinstance(result.layout, FlexibleLayout):  # type: ignore[union-attr]
                result.freeze_layout()
            realized[n] = result
            tb._helion_realized = result  # type: ignore[attr-defined]
    return realized


def _relax_input_strides(
    tensor_args: dict[str, TensorBox],
    realized: dict[str, IRNode],
    kernel: Kernel,
) -> dict[sympy.Symbol, tuple[str, int]]:
    """Create per-kernel unbacked stride symbols and mutate tb.data for downstream ops.

    Two concerns:
    1. Per-kernel stride defs: each kernel creates its own unbacked stride
       symbols for its own codegen (needed for CUDA graph partitioning).
    2. Downstream relaxation: mutate tb.data ONCE per TensorBox so
       downstream non-Helion ops see unbacked strides and generate
       stride-parameterized code (compensates for relaxed stride guards).
    """
    input_stride_defs: dict[sympy.Symbol, tuple[str, int]] = {}
    if kernel.settings.static_shapes:
        return input_stride_defs

    shape_env = V.graph.sizevars.shape_env
    for n, tb in tensor_args.items():
        if not isinstance(tb, TensorBox) or n not in realized:
            continue
        inner = realized[n]
        if isinstance(inner, BaseView) and not isinstance(inner, ReinterpretView):
            continue
        input_name = inner.get_name()
        if input_name not in V.graph.graph_inputs:
            continue
        graph_input = V.graph.graph_inputs[input_name]
        if isinstance(inner, ReinterpretView) and len(inner.get_size()) != len(graph_input.get_size()):
            continue

        # Concern 1: Create per-kernel unbacked stride symbols.
        new_strides: list[sympy.Expr] = []
        with shape_env.ignore_fresh_unbacked_symbols():
            for dim, s in enumerate(inner.get_stride()):
                u = shape_env.create_unbacked_symint()
                shape_env.set_real_tensor_prop_unbacked_vals(
                    u.node.expr, int(_size_hint(s, fallback=1)))
                new_strides.append(u.node.expr)
                input_stride_defs[u.node.expr] = (n, dim)

        # Concern 2: Mutate tb.data once for downstream non-Helion ops.
        if not getattr(tb, "_helion_relaxed", False):
            tb._helion_relaxed = True  # type: ignore[attr-defined]
            inner_offset = inner.layout.offset if isinstance(inner, ReinterpretView) else sympy.Integer(0)
            new_layout = FixedLayout(inner.get_device(), inner.get_dtype(), inner.get_size(), new_strides, offset=inner_offset)
            inner_for_rv = StorageBox(inner) if isinstance(inner, ReinterpretView) else inner
            tb.data = StorageBox(ReinterpretView(data=inner_for_rv, layout=new_layout))

    return input_stride_defs


def _check_specialized_inputs(
    kernel: Kernel, bound: BoundKernel, realized: dict[str, IRNode]
) -> None:
    """Raise ValueError if a specialized kernel receives unbacked symbolic inputs.

    A kernel that specializes on sizes/strides bakes concrete values into the
    Triton code.  If any input has unbacked symbols (from an upstream dynamic
    kernel's output), those baked-in values will be wrong at runtime.
    """
    if not (
        kernel.settings.static_shapes
        or bound.env.specialized_vars
        or bound.env.specialized_strides
    ):
        return
    reason = (
        "static_shapes=True"
        if kernel.settings.static_shapes
        else "hl.specialize()"
    )
    backed = V.graph.sizevars.shape_env.backed_var_to_val
    for arg_name, r in realized.items():
        for s in (*r.get_size(), *r.get_stride()):
            if isinstance(s, sympy.Expr) and any(
                sym not in backed for sym in s.free_symbols
            ):
                raise ValueError(
                    f"Helion kernel '{kernel.name}' has {reason} but received "
                    f"input '{arg_name}' with unbacked symbolic size/stride "
                    f"from an upstream dynamic kernel.  A specialized kernel "
                    f"bakes concrete values into the generated Triton code, "
                    f"which produces wrong results when shapes change.\n\n"
                    f"Fix: set static_shapes=False on '{kernel.name}', e.g.:\n"
                    f"  @helion.kernel(static_shapes=False)\n"
                    f"  def {kernel.name}(...):"
                )


def _build_helion_sym_remap(
    bound: BoundKernel, realized: dict[str, IRNode]
) -> dict[sympy.Expr, sympy.Expr]:
    """Build a mapping from Helion ShapeEnv symbols to Inductor IR expressions.

    The kernel.bind() call creates output FakeTensors using Helion's own ShapeEnv
    (for static_shapes=False).  These symbols must not leak into Inductor's IR.
    We remap them using the correspondence between Helion's input FakeTensors and
    the Inductor realized IR nodes.
    """
    helion_shape_env = bound.env.shape_env
    helion_sym_remap: dict[sympy.Expr, sympy.Expr] = {}
    helion_params = bound.host_function.flat_tensor_params()
    for name, ind_node in realized.items():
        if (helion_param := helion_params.get(name)) is None:
            continue
        for helion_s, ind_s in chain(
            zip(helion_param.shape, ind_node.get_size(), strict=True),
            zip(helion_param.stride(), ind_node.get_stride(), strict=True),
        ):
            if isinstance(helion_s, torch.SymInt) and helion_s.node.shape_env is helion_shape_env and helion_s.node.expr.is_Symbol:
                helion_sym_remap.setdefault(helion_s.node.expr, ind_s)
    return helion_sym_remap


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

    realized = _realize_inputs(tensor_args)
    input_stride_defs = _relax_input_strides(tensor_args, realized, kernel)
    stride_relaxed = not kernel.settings.static_shapes

    all_args: dict[str, object] = {**constant_args}
    for n, r in realized.items():
        all_args[n] = torch.empty_strided(
            [_size_hint(s, fallback=64) for s in r.get_size()],
            [_size_hint(s, fallback=1) for s in r.get_stride()],
            dtype=r.get_dtype(), device=r.get_device(),
        )
    _rebuild_container_args(all_args)

    fake_tensors: list[object] = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    bound = kernel.bind(tuple(fake_tensors))

    _check_specialized_inputs(kernel, bound, realized)

    inputs = list(realized.values())

    flat_leaves, tree_spec = _get_flat_output(bound.host_function)
    example_outputs = [leaf for leaf in flat_leaves if isinstance(leaf, torch.Tensor)]

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
    buf._input_stride_defs = input_stride_defs
    V.graph.no_fuse_buffer_names.add(buf.get_name())

    if not example_outputs:
        return ()

    direct_alias_at_leaf = {
        i: name
        for i, name in cast(
            "dict[int, str]", output_spec.get("direct_aliases", {})
        ).items()
        if name in realized
    }

    helion_sym_remap = _build_helion_sym_remap(bound, realized)

    def _remap_helion_syms(vals: list[sympy.Expr]) -> list[sympy.Expr]:
        """Replace any Helion ShapeEnv symbols with Inductor equivalents."""
        return [v.xreplace(helion_sym_remap) for v in vals] if helion_sym_remap else vals

    all_output_stride_defs: dict[sympy.Symbol, pytree.KeyPath] = {}

    assert tree_spec is not None
    structured = pytree.tree_unflatten(flat_leaves, tree_spec)

    leaf_counter = [0]
    seen_outputs: dict[int, TensorBox] = {}

    def collect_tensor_outputs(
        output: object, indices: list[tuple[type, int]]
    ) -> list[TensorBox]:
        if isinstance(output, (list, tuple)):
            return [
                r
                for i, item in enumerate(output)
                for r in collect_tensor_outputs(item, [*indices, (type(output), i)])
            ]
        leaf_idx = leaf_counter[0]
        leaf_counter[0] += 1
        if isinstance(output, torch.Tensor):
            if leaf_idx in direct_alias_at_leaf:
                return [TensorBox.create(realized[direct_alias_at_leaf[leaf_idx]])]
            tid = id(output)
            if tid in seen_outputs:
                return [seen_outputs[tid]]
            layout = FallbackKernel.tensor_to_layout(output)
            layout.size = _remap_helion_syms(layout.size)

            if stride_relaxed:
                backed_strides = _remap_helion_syms(layout.stride)
                shape_env = V.graph.sizevars.shape_env
                base_keypath: tuple[pytree.KeyEntry, ...] = tuple(
                    pytree.SequenceKey(idx) for _, idx in indices
                )
                new_strides: list[sympy.Expr] = []
                with shape_env.ignore_fresh_unbacked_symbols():
                    for d, backed_s in enumerate(backed_strides):
                        u = shape_env.create_unbacked_symint()
                        h = _size_hint(backed_s, fallback=1)
                        if h is not None:
                            shape_env.set_real_tensor_prop_unbacked_vals(u.node.expr, int(h))
                        new_strides.append(u.node.expr)
                        all_output_stride_defs[u.node.expr] = base_keypath + (
                            CallMethodKey("stride"), pytree.SequenceKey(d),
                        )
                layout.stride = new_strides
                mo = MultiOutput(layout, buf, indices, skip_size_stride_alignment_checks=True)
                tb = TensorBox(StorageBox(mo))
            else:
                layout.stride = _remap_helion_syms(layout.stride)
                mo = MultiOutput(layout, buf, indices)
                tb = TensorBox(StorageBox(mo))
            seen_outputs[tid] = tb
            return [tb]
        return []

    result = tuple(collect_tensor_outputs(structured, []))

    if all_output_stride_defs:
        buf._output_stride_defs = all_output_stride_defs

    return result


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
