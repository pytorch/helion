from __future__ import annotations

import ast
import contextlib
import dataclasses
from itertools import dropwhile
import logging
from typing import TYPE_CHECKING
from typing import Callable
from typing import Sequence
from typing import cast

import sympy
import torch
from torch._inductor import config as inductor_fusion_config
from torch._inductor import dependencies
from torch._inductor.codegen.common import IndentedBuffer
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
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree

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

log = logging.getLogger(__name__)


class _MockCSE:
    """Minimal mock of Triton's CSE for ops that access V.kernel.cse."""

    def __init__(self) -> None:
        self.varname_map: dict[str, object] = {}
        self._cache: dict[object, object] = {}

    def generate(self, *args: object, **kwargs: object) -> str:
        return str(args[1]) if len(args) > 1 else "_cse_tmp"


class _FusionOpsHandler:
    """Ops handler that generates Triton code strings for fusion operations.

    Used for both epilogue and prologue fusion. Delegates compute ops
    (add, mul, relu, tanh, sigmoid, etc.) to TritonOverrides. Overrides
    load/store to intercept memory ops.

    Parameters:
        intercept_bufs: buf names to intercept in load() (kernel outputs
            for epilogue, source inputs for prologue)
        intercept_value: string to return for intercepted bufs
        get_extra_input: callback to get/create a parameter for external bufs
        make_load_expr: callback (param) -> Triton load expression string
        var_prefix: "_epi_" or "_pro_"
    """

    def __init__(
        self,
        intercept_bufs: set[str],
        intercept_value: str,
        get_extra_input: Callable[[str], str],
        make_load_expr: Callable[[str], str],
        var_prefix: str,
    ) -> None:
        self._intercept_bufs = intercept_bufs
        self._intercept_value = intercept_value
        self._get_extra_input = get_extra_input
        self._make_load_expr = make_load_expr
        self._var_prefix = var_prefix
        self.statements: list[str] = []
        self.final_value: str | None = None
        self.final_store_name: str | None = None
        self._counter = 0

    def _new_var(self, expr_str: str) -> str:
        name = f"{self._var_prefix}{self._counter}"
        self._counter += 1
        self.statements.append(f"{name} = {expr_str}")
        return name

    def load(self, name: str, index: object) -> str:
        if name in self._intercept_bufs:
            return self._intercept_value
        param = self._get_extra_input(name)
        return self._new_var(self._make_load_expr(param))

    def store(
        self, name: str, index: object, value: object, mode: object = None
    ) -> None:
        self.final_value = str(value)
        self.final_store_name = name

    def reduction(
        self,
        dtype: object,
        src_dtype: object,
        reduction_type: object,
        value: object,
    ) -> object:
        raise RuntimeError("Reductions not supported in fusion")

    def store_reduction(self, name: str, index: object, value: object) -> None:
        raise RuntimeError("Store reductions not supported in fusion")

    def run(self, ir_node: ComputedBuffer) -> None:
        """Execute this handler on a ComputedBuffer node."""
        pw = ir_node.data
        inner_fn = pw.inner_fn
        index = [sympy.Integer(0)] * len(pw.ranges)
        with V.set_ops_handler(self):  # pyrefly: ignore[bad-argument-type]
            value = inner_fn(index)
            self.store(ir_node.get_name(), index, value)
        if self.final_value is None:
            raise RuntimeError("Fusion handler did not capture a store")

    def __getattr__(self, name: str) -> object:
        # Delegate compute ops (add, mul, relu, sigmoid, tanh, maximum,
        # constant, to_dtype, etc.) to TritonOverrides.
        # Wrap string results in parentheses to avoid operator precedence
        # issues in chained expressions (e.g., "a + b * c" vs "(a + b) * c").
        from torch._inductor.codegen.triton import TritonOverrides

        attr = getattr(TritonOverrides, name, None)
        if attr is not None:
            return lambda *a, **kw: (
                f"({r})" if isinstance(r := attr(*a, **kw), str) else r
            )
        raise AttributeError(f"_FusionOpsHandler has no attribute '{name}'")


@dataclasses.dataclass
class EpilogueSpec:
    """Specification for epilogue fusion on a single store."""

    final_value: str  # fused expression (e.g., "(relu(_kv_0) + _epi_0)")
    statements: list[str]  # assignment statements from handler
    kernel_val_placeholder: str  # e.g., "_kernel_val_0"
    extra_load_vars: dict[str, str]  # var_name -> param_name for load substitution
    redirect_output: tuple[str, str, str] | None  # (orig_param, new_param, new_buf)


@dataclasses.dataclass
class PrologueSpec:
    """Specification for prologue fusion on a single load."""

    final_value: str  # expression with "_pro_load" placeholder


def _find_funcs(
    root: ast.Module,
) -> tuple[ast.FunctionDef, ast.FunctionDef]:
    """Find the @triton.jit inner function and host function in the AST.

    Helion always generates the inner function (with @triton.jit) first,
    followed by the host function.
    """
    funcs = [n for n in ast.iter_child_nodes(root) if isinstance(n, ast.FunctionDef)]
    assert len(funcs) >= 2, f"Expected at least 2 functions, found {len(funcs)}"
    inner_func, host_func = funcs[0], funcs[1]
    assert any(
        isinstance(dec, ast.Attribute)
        and isinstance(dec.value, ast.Name)
        and dec.value.id == "triton"
        and dec.attr == "jit"
        for dec in inner_func.decorator_list
    ), "First function must have @triton.jit decorator"
    return inner_func, host_func


def _inject_extra_params(
    extra_params: list[tuple[str, str]],
    inner_func: ast.FunctionDef,
    host_func: ast.FunctionDef,
) -> None:
    """Add extra parameters to inner function, host function, and launcher."""
    for param_name, _buf_name in extra_params:
        # Find position of first constexpr param (to insert before it)
        insert_pos = len(inner_func.args.args)
        for i, arg in enumerate(inner_func.args.args):
            ann = arg.annotation
            if (
                ann is not None
                and isinstance(ann, ast.Attribute)
                and ann.attr == "constexpr"
            ):
                insert_pos = i
                break
        inner_func.args.args.insert(insert_pos, ast.arg(arg=param_name))
        host_func.args.args.append(ast.arg(arg=param_name))
        # Insert parameter into the _launcher call matching inner function order.
        # The launcher call args layout is: kernel_fn, grid, <kernel_params...>,
        # where kernel_params must match the inner function's parameter order.
        for node in ast.walk(host_func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("_launcher", "_default_launcher")
            ):
                inner_pos = next(
                    (
                        i
                        for i, a in enumerate(inner_func.args.args)
                        if a.arg == param_name
                    ),
                    len(inner_func.args.args) - 1,
                )
                launcher_insert_pos = 2 + inner_pos
                node.args.insert(
                    launcher_insert_pos,
                    ast.Name(id=param_name, ctx=ast.Load()),
                )
                break


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

        # Mock CSE for TritonOverrides ops that access V.kernel.cse
        self.cse = _MockCSE()
        # Required by TritonOverrides.to_dtype for fp8 dtype conversions
        self.min_elem_per_thread = 0

        # Epilogue fusion state (populated during codegen)
        self._output_buf_names: set[str] = set()
        self._epilogue_extra_params: list[
            tuple[str, str]
        ] = []  # (param_name, buf_name)
        # MultiOutput IR nodes for proper tuple extraction in call_kernel
        self._multi_output_ir_nodes: list[MultiOutput] = []

        # Prologue fusion state (populated during codegen)
        self._prologue_replacements: dict[str, str] = {}  # input_param -> source_param

        # Whether the kernel has wrapper-level outputs (e.g., return x + y).
        # Such outputs are computed by the host function, not by tl.store in
        # the inner kernel. Epilogue fusion requires a tl.store to fuse into,
        # so it's impossible for wrapper-level outputs.
        self._has_wrapper_level_outputs = False

        # Fusion specs: populated during codegen_template_override, read by
        # hl.store/hl.load codegen in memory_ops.py via V.kernel
        self._epilogue_specs: dict[str, EpilogueSpec] = {}  # keyed by kernel param name
        self._prologue_specs: dict[str, PrologueSpec] = {}  # keyed by kernel param name
        self._fusion_extra_params: list[tuple[str, str]] = []  # (param_name, buf_name)

        mutated_inputs_irnodes = [
            self.named_input_nodes[n]
            for n in (mutated_input_names or [])
            if n in self.named_input_nodes
        ] or None

        # Allow all non-mutated inputs to be prologue-fused.  The scheduler
        # checks this set when deciding whether a pointwise producer can be
        # folded into the template's loads.  Mutated inputs are excluded
        # because the template writes back to them, and post-mutation
        # consumers need the materialized buffer.
        _mutated_buf_names = {
            self.named_input_nodes[n].get_name()  # type: ignore[union-attr]
            for n in (mutated_input_names or [])
            if n in self.named_input_nodes
        }
        _allowed_inps = OrderedSet(
            inp.get_name()  # type: ignore[union-attr]
            for inp in inputs
            if inp.get_name() not in _mutated_buf_names  # type: ignore[union-attr]
        )
        super().__init__(
            layout=cast("Layout", layout),
            inputs=inputs,
            make_kernel_render=lambda tb, hint_override=None: (self, self.render),
            mutated_inputs=mutated_inputs_irnodes,
            allowed_prologue_inps=_allowed_inps,
        )

        for inp in mutated_inputs_irnodes or []:
            if hasattr(inp, "get_name"):
                V.graph.never_reuse_buffers.add(inp.get_name())

    # Layout is always MultiOutputLayout: reads from inputs only,
    # writes go through MultiOutput children, no allocation needed.

    def is_multi_outputs_template(self) -> bool:
        return isinstance(self.layout, MultiOutputLayout)

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype for prologue-fusion heuristic checks.

        MultiOutputLayout doesn't carry a dtype, so we infer one from the
        first input tensor.
        """
        if self.inputs:
            return self.inputs[0].get_dtype()  # type: ignore[union-attr]
        return torch.float32

    def extract_read_writes(self, normalize: bool = False) -> dependencies.ReadWrites:
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        for inp in self.inputs:
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

    def can_fuse_multi_outputs_template(self, node1: object, node2: object) -> bool:
        """Control which scheduler nodes can fuse with this template."""
        n2_node = getattr(node2, "node", None)
        if isinstance(n2_node, MultiOutput):
            return super().can_fuse_multi_outputs_template(  # pyrefly: ignore[missing-attribute]
                node1, node2
            )
        if not isinstance(n2_node, ComputedBuffer):
            return False
        return self._can_fuse_as_epilogue(node2) or self._can_fuse_as_prologue(node2)

    def _can_fuse_as_epilogue(self, node2: object) -> bool:
        """Check if node2 can be fused as an epilogue (reads from kernel output)."""
        output_reads = {
            dep.name
            for dep in node2.read_writes.reads  # type: ignore[union-attr]
            if dep.name in self._output_buf_names
        }
        if not output_reads or self._has_wrapper_level_outputs or len(output_reads) > 1:
            return False
        from torch._inductor.ir import Reduction

        if isinstance(getattr(getattr(node2, "node", None), "data", None), Reduction):
            return False

        # Shape check: fusion reuses the kernel's store offset, only correct
        # when all involved buffers share the same shape.
        def _buf_shape(name: str) -> tuple[object, ...] | None:
            buf = V.graph.get_buffer(name)
            return tuple(buf.get_size()) if buf is not None else None

        output_shape = _buf_shape(next(iter(output_reads)))
        if output_shape is not None:
            shapes_to_check = [
                _buf_shape(dep.name)
                for dep in node2.read_writes.writes  # type: ignore[union-attr]
            ] + [
                _buf_shape(dep.name)
                for dep in node2.read_writes.reads  # type: ignore[union-attr]
                if dep.name not in self._output_buf_names
            ]
            for s in shapes_to_check:
                if s is not None and s != output_shape:
                    return False
            # Reject if output shape doesn't match any template input shape,
            # which indicates a shape-changing wrapper operation (e.g. reduction).
            input_shapes = {tuple(inp.get_size()) for inp in self.inputs}  # type: ignore[union-attr]
            if input_shapes and output_shape not in input_shapes:
                return False
        return True

    def _can_fuse_as_prologue(self, node2: object) -> bool:
        """Check if node2 can be fused as a prologue (writes to template input).

        Uses allowed_prologue_inps which already excludes mutated inputs.
        Only fuses single-source prologues since multi-source prologues
        can't map all sources through the _pro_load placeholder correctly.
        """
        allowed = self.get_allowed_prologue_inps()
        if not allowed:
            return False
        source_bufs = {d.name for d in node2.read_writes.reads}  # type: ignore[union-attr]
        if len(source_bufs) > 1:
            return False
        for dep in node2.read_writes.writes:  # type: ignore[union-attr]
            if dep.name in allowed:
                return all(
                    len(out_buf.users) == 1
                    for out_buf in node2.get_outputs()  # type: ignore[union-attr]
                )
        return False

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

    # ------------------------------------------------------------------
    # Epilogue fusion helpers
    # ------------------------------------------------------------------

    def _classify_fused_nodes(
        self,
        epilogue_nodes: Sequence[BaseSchedulerNode],
    ) -> tuple[
        list[BaseSchedulerNode],
        list[BaseSchedulerNode],
        list[BaseSchedulerNode],
    ]:
        """Classify epilogue_nodes into three categories.

        Returns (multi_output_nodes, actual_epilogue, misclassified_prologues).

        Prologues that entered via can_fuse_multi_outputs_template end up in
        the epilogue list; we detect them here by checking if they write to
        a template input buffer.
        """
        input_buf_names = {
            inp.get_name()  # pyrefly: ignore[missing-attribute]
            for inp in self.inputs  # type: ignore[union-attr]
        }
        multi_output_nodes: list[BaseSchedulerNode] = []
        actual_epilogue: list[BaseSchedulerNode] = []
        misclassified_prologues: list[BaseSchedulerNode] = []

        for n in epilogue_nodes:
            node_ir = getattr(n, "node", None)
            if isinstance(node_ir, MultiOutput):
                multi_output_nodes.append(n)
            elif isinstance(node_ir, ComputedBuffer) and any(
                dep.name in input_buf_names for dep in n.read_writes.writes
            ):
                misclassified_prologues.append(n)
            else:
                actual_epilogue.append(n)

        return multi_output_nodes, actual_epilogue, misclassified_prologues

    def _build_output_buf_to_param(self) -> dict[str, str]:
        """Map output buffer names to kernel parameter names.

        Uses the host function's return AST and MultiOutput indices to trace
        which output buffer corresponds to which kernel parameter.
        """
        host_function = self._bound_kernel.host_function
        if host_function is None:
            return {}
        return_value = next(
            (
                s.value
                for s in host_function.body
                if isinstance(s, ast.Return) and s.value
            ),
            None,
        )
        output_buf_to_param: dict[str, str] = {}
        if return_value is not None:
            for mo in self._multi_output_ir_nodes:
                node = return_value
                try:
                    for _, idx in mo.indices:
                        node = node.elts[idx]  # type: ignore[union-attr]
                except (AttributeError, IndexError, TypeError):
                    continue
                if isinstance(node, ast.Name):
                    output_buf_to_param[mo.get_name()] = node.id
        return output_buf_to_param

    def _build_epilogue_specs(
        self,
        epilogue_nodes: Sequence[BaseSchedulerNode],
    ) -> None:
        """Build epilogue fusion specs from scheduler nodes.

        Runs _FusionOpsHandler on each epilogue node and stores the result
        as an EpilogueSpec keyed by kernel parameter name.
        """
        output_buf_to_param = self._build_output_buf_to_param()

        extra_inputs: dict[str, str] = {}  # buf_name -> param_name

        def get_extra_input(buf_name: str) -> str:
            if buf_name not in extra_inputs:
                extra_inputs[buf_name] = f"_epi_input_{len(extra_inputs)}"
            return extra_inputs[buf_name]

        for epi_counter, epi_node in enumerate(epilogue_nodes):
            kernel_output_buf = next(
                (
                    d.name
                    for d in epi_node.read_writes.reads
                    if d.name in self._output_buf_names
                ),
                None,
            )
            if kernel_output_buf is None:
                raise RuntimeError("Epilogue node doesn't read from any kernel output")

            target_param = output_buf_to_param.get(kernel_output_buf)
            if target_param is None:
                # Fallback: find a param name that isn't an input
                for mo in self._multi_output_ir_nodes:
                    if mo.get_name() == kernel_output_buf:
                        # Try to find a unique non-input param
                        break

            kernel_val_name = f"_kernel_val_{epi_counter}"

            # Handler uses param name directly as placeholder for extra loads;
            # actual tl.load() calls are generated at codegen time.
            handler = _FusionOpsHandler(
                intercept_bufs=self._output_buf_names,
                intercept_value=kernel_val_name,
                get_extra_input=get_extra_input,
                make_load_expr=lambda p: p,
                var_prefix="_epi_",
            )

            ir_node = epi_node.node
            if not isinstance(ir_node, ComputedBuffer):
                raise RuntimeError(f"Unsupported epilogue node type: {type(ir_node)}")
            handler.run(ir_node)

            # Track extra load vars: handler statements like "_epi_0 = _epi_input_0"
            # need their RHS (param name) to be replaced with actual tl.load() at codegen
            extra_load_vars: dict[str, str] = {}
            for stmt_str in handler.statements:
                # Statements look like "_epi_0 = _epi_input_0"
                parts = stmt_str.split(" = ", 1)
                if len(parts) == 2 and parts[1].strip() in extra_inputs.values():
                    extra_load_vars[parts[0].strip()] = parts[1].strip()

            # Handle epilogue output redirect
            redirect_output: tuple[str, str, str] | None = None
            epilogue_output = handler.final_store_name
            if epilogue_output is not None and epilogue_output != kernel_output_buf:
                epi_out_param = f"_epi_out_{len(self._fusion_extra_params)}"
                assert target_param is not None
                redirect_output = (target_param, epi_out_param, epilogue_output)
                self._fusion_extra_params.append((epi_out_param, epilogue_output))
                self.removed_buffers.add(kernel_output_buf)

            assert target_param is not None
            self._epilogue_specs[target_param] = EpilogueSpec(
                final_value=handler.final_value,  # pyrefly: ignore[bad-argument-type]
                statements=handler.statements,
                kernel_val_placeholder=kernel_val_name,
                extra_load_vars=extra_load_vars,
                redirect_output=redirect_output,
            )

        # Add extra input parameters
        for buf_name, param_name in extra_inputs.items():
            self._fusion_extra_params.append((param_name, buf_name))

        # Also add to _epilogue_extra_params for call_kernel
        self._epilogue_extra_params.extend(self._fusion_extra_params)

    def _build_prologue_specs(
        self,
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> None:
        """Build prologue fusion specs from scheduler nodes.

        Runs _FusionOpsHandler on each prologue node and stores the result
        as a PrologueSpec keyed by kernel parameter name.
        """
        input_buf_to_params: dict[str, list[str]] = {}
        for param_name, inp in self.named_input_nodes.items():
            buf_name = inp.get_name()  # type: ignore[union-attr]
            input_buf_to_params.setdefault(buf_name, []).append(param_name)

        for pro_node in prologue_nodes:
            ir_node = pro_node.node
            if not isinstance(ir_node, ComputedBuffer):
                raise RuntimeError(f"Unsupported prologue node type: {type(ir_node)}")

            fused_buf_name = next(
                (
                    dep.name
                    for dep in pro_node.read_writes.writes
                    if dep.name in input_buf_to_params
                ),
                None,
            )
            if fused_buf_name is None:
                raise RuntimeError("Prologue doesn't write to any template input")

            source_buf_names = {dep.name for dep in pro_node.read_writes.reads}
            if len(source_buf_names) > 1:
                raise RuntimeError(
                    "Multi-source prologues are not supported by _FusionOpsHandler"
                )

            source_needs_upcast = (
                inductor_fusion_config.triton.codegen_upcast_to_fp32
                and any(
                    V.graph.get_dtype(s) in (torch.float16, torch.bfloat16)
                    for s in source_buf_names
                )
            )
            intercept_val = (
                "_pro_load.to(tl.float32)" if source_needs_upcast else "_pro_load"
            )
            handler = _FusionOpsHandler(
                intercept_bufs=source_buf_names,
                intercept_value=intercept_val,
                get_extra_input=lambda name: name,
                make_load_expr=lambda param: param,
                var_prefix="_pro_",
            )
            handler.run(ir_node)

            for input_param in input_buf_to_params[fused_buf_name]:
                self._prologue_specs[input_param] = PrologueSpec(
                    final_value=handler.final_value,  # pyrefly: ignore[bad-argument-type]
                )

            if source_buf_names:
                self._prologue_replacements[fused_buf_name] = next(
                    iter(source_buf_names)
                )

    # ------------------------------------------------------------------
    # Codegen entry points
    # ------------------------------------------------------------------

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
            source_buf = self._prologue_replacements.get(buf_name)

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

        sig = self._helion_kernel.signature.parameters
        args = [
            arg_inputs.get(n, repr(self._constant_args_dict.get(n, p.default)))
            for n, p in sig.items()
            if n in arg_inputs
            or n in self._constant_args_dict
            or p.default is not p.empty
        ]

        # Add epilogue extra parameters (outputs and inputs)
        for _param_name, buf_name in self._epilogue_extra_params:
            args.append(buf_name)

        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

        # Emit MultiOutput extraction code for each output buffer.
        # MultiOutput nodes are marked as run by codegen_template_override,
        # so their separate codegen is suppressed. We must emit the extraction
        # here so that downstream consumers can reference the buffer names.
        if self._multi_output_ir_nodes:
            # Use indices from MultiOutput IR nodes for proper nested extraction
            for mo in self._multi_output_ir_nodes:
                mo_name = mo.get_name()
                if mo_name not in self.removed_buffers:
                    idx_str = output_name
                    for _, idx in mo.indices:
                        idx_str = f"{idx_str}[{idx}]"
                    wrapper.writeline(f"{mo_name} = {idx_str}")
        else:
            # Fallback for single-output kernels
            for mo_name in sorted(self._output_buf_names):
                if mo_name not in self.removed_buffers:
                    wrapper.writeline(f"{mo_name} = {output_name}")

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

            node_schedule: list[BaseSchedulerNode] = [template_node]

            multi_output_nodes, actual_epilogue, misclassified_prologues = (
                self._classify_fused_nodes(epilogue_nodes)
            )

            # Mark MultiOutput nodes as run and include in schedule
            if not only_gen_src_code:
                for n in multi_output_nodes:
                    n.mark_run()
            node_schedule.extend(multi_output_nodes)

            # Store MultiOutput IR nodes for proper tuple extraction in call_kernel
            self._multi_output_ir_nodes = [  # pyrefly: ignore[bad-assignment]
                getattr(n, "node", None)
                for n in multi_output_nodes
                if isinstance(getattr(n, "node", None), MultiOutput)
            ]

            # Combine actual prologue_nodes with misclassified ones
            actual_prologue = list(prologue_nodes) + misclassified_prologues

            if actual_prologue or actual_epilogue:
                try:
                    # Build fusion specs
                    if actual_prologue:
                        self._build_prologue_specs(actual_prologue)
                    if actual_epilogue:
                        self._build_epilogue_specs(actual_epilogue)

                    # Re-run generate_ast() — hl.store/hl.load codegen checks
                    # V.kernel for specs and emits fused code directly.
                    # Save/restore generated_kernel_count because generate_ast()
                    # internally creates a TritonKernel that increments it.
                    from torch._inductor import metrics as _metrics

                    cfg = self._bound_kernel._config
                    assert cfg is not None
                    host_fn = self._helion_kernel.name
                    inner_fn = f"_helion_{host_fn}"
                    inner_fn_placeholder = f"{inner_fn}_{Placeholder.KERNEL_NAME}"

                    saved_count = _metrics.generated_kernel_count
                    with self._bound_kernel.env:
                        host_function = self._bound_kernel.host_function
                        assert host_function is not None
                        root = generate_ast(host_function, cfg, emit_repro_caller=False)
                    _metrics.generated_kernel_count = saved_count

                    # Rename functions
                    for node in ast.walk(root):
                        if isinstance(node, ast.FunctionDef):
                            if node.name == host_fn:
                                node.name = str(Placeholder.KERNEL_NAME)
                            elif node.name == inner_fn:
                                node.name = inner_fn_placeholder
                        elif isinstance(node, ast.Name) and node.id == inner_fn:
                            node.id = inner_fn_placeholder

                    # Inject extra params and apply redirect renames
                    if self._fusion_extra_params or any(
                        s.redirect_output is not None
                        for s in self._epilogue_specs.values()
                    ):
                        inner_func, host_func = _find_funcs(
                            root  # pyrefly: ignore[bad-argument-type]
                        )
                        if self._fusion_extra_params:
                            _inject_extra_params(
                                self._fusion_extra_params, inner_func, host_func
                            )
                        # Apply redirect renames on the entire inner function
                        # so that tensor_descriptor params (used in
                        # make_tensor_descriptor) are renamed correctly.
                        from ..ast_read_writes import ast_rename

                        for spec in self._epilogue_specs.values():
                            if spec.redirect_output is not None:
                                orig_param, new_param, _new_buf = spec.redirect_output
                                ast_rename(inner_func, {orig_param: new_param})

                    src_code = get_needed_imports(root) + unparse(
                        root,
                        output_origin_lines=self._bound_kernel.settings.output_origin_lines,
                    )

                    # Mark nodes as run only after successful fusion
                    if not only_gen_src_code:
                        for node in actual_prologue:
                            node.mark_run()
                        for node in actual_epilogue:
                            node.mark_run()
                    node_schedule.extend(actual_prologue)
                    node_schedule.extend(actual_epilogue)
                except Exception:
                    log.warning(
                        "Prologue/epilogue fusion failed, falling back to unfused codegen",
                        exc_info=True,
                    )
                    # Reset fusion state on failure
                    self._epilogue_extra_params = []
                    self._prologue_replacements = {}
                    self._epilogue_specs = {}
                    self._prologue_specs = {}
                    self._fusion_extra_params = []
                    self.removed_buffers = OrderedSet()

            if inductor_fusion_config.benchmark_kernel:
                src_code = f"\n{src_code}\n{IndentedBuffer().getvalue()}"

            if only_gen_src_code:
                return src_code

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
            # MultiOutput nodes (outputs of other template buffers) already
            # have FixedLayout with the correct strides.  realize_input()
            # falls through to copy_input() for these, creating a new buffer
            # with FlexibleLayout that defaults to contiguous strides -- this
            # loses the original (potentially non-contiguous) stride info.
            # Use the MultiOutput directly to preserve its layout.
            if isinstance(tb.data, MultiOutput):
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

    fake_tensors: list[object] = [
        torch.empty_strided(
            [as_int(s, 64) for s in realized[n].get_size()],
            [as_int(s, 1) for s in realized[n].get_stride()],
            dtype=realized[n].get_dtype(),
            device=realized[n].get_device(),
        )
        if n in realized
        else constant_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in realized or n in constant_args or p.default is not p.empty
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

    # Detect wrapper-level outputs: if the return AST contains operations
    # (not just simple names, tuples, or lists), some outputs are computed
    # by the host wrapper, not by tl.store in the inner kernel.
    if return_ast is not None:
        buf._has_wrapper_level_outputs = any(
            not isinstance(n, (ast.Name, ast.Constant, ast.Tuple, ast.List))
            for n in ast.walk(return_ast)
            if isinstance(n, ast.expr)
        )

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
            # Track MultiOutput buffer names for epilogue fusion matching
            buf._output_buf_names.add(mo.get_name())
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
