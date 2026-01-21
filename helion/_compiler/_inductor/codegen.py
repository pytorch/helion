"""Triton codegen for Inductor prologue/epilogue fusion."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Callable

import sympy
import torch
from torch._inductor import config as inductor_config
from torch._inductor.bounds import ValueRanges
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.simd import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.ir import Buffer
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import Pointwise
from torch._inductor.scheduler import BaseSchedulerNode
import torch._inductor.lowering  # noqa: F401
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.virtualized import V

from ...language.tile_proxy import Tile
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..utils import get_broadcast_slice

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _broadcast_name(name: str, dim_index: int, ndim: int) -> str:
    """Apply broadcast slicing to a name if multi-dimensional."""
    if ndim > 1:
        return f"{name}{get_broadcast_slice(dim_index, ndim)}"
    return name


class FusionOpsHandler(DefaultHandler):
    """OpsHandler that delegates to TritonOverrides to generate Triton code strings."""

    _parent_handler = TritonOverrides()

    def __init__(
        self,
        accumulator_map: dict[str, str],  # buffer_name -> variable_name string
        index_symbols: list[sympy.Symbol],  # Symbols passed to inner_fn
        capture_buffer_fn: Callable[[str], str],
    ) -> None:
        super().__init__()
        self.accumulator_map = accumulator_map
        self.index_symbols = index_symbols
        self.capture_buffer = capture_buffer_fn

    @staticmethod
    def _wrap_arg(a: object) -> object:
        """Wrap string args as CSEVariable for TritonOverrides compatibility."""
        if isinstance(a, CSEVariable):
            return a
        if isinstance(a, str):
            return CSEVariable(a, ValueRanges.unknown())
        return a

    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> CSEVariable:
        """Delegate to TritonOverrides for most ops, wrapping result in parens for precedence."""
        wrapped_args = tuple(self._wrap_arg(a) for a in args)
        wrapped_kwargs = {k: self._wrap_arg(v) for k, v in kwargs.items()}
        result = str(
            getattr(self._parent_handler, name)(*wrapped_args, **wrapped_kwargs)
        )
        return CSEVariable(f"({result})", ValueRanges.unknown())

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        """Handle load ops - return accumulator or generate tl.load."""
        buf_dtype = V.graph.get_dtype(name)
        if name in self.accumulator_map:
            return CSEVariable(
                self.accumulator_map[name], ValueRanges.unknown(), dtype=buf_dtype
            )
        ptr_name = self.capture_buffer(name)
        return CSEVariable(
            f"tl.load({ptr_name} + {self._sympy_to_string(index)})",
            ValueRanges.unknown(),
            dtype=buf_dtype,
        )

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        """Convert sympy index expression to Triton code."""
        return CSEVariable(
            self._sympy_to_string(expr), ValueRanges.unknown(), dtype=dtype
        )

    def _sympy_to_string(self, expr: sympy.Expr) -> str:
        """Convert sympy expression to string with broadcasting applied to index symbols."""
        ndim = len(self.index_symbols)
        if ndim <= 1:
            return str(expr)
        # Apply broadcasting to index symbols
        replacements = {
            sym: sympy.Symbol(_broadcast_name(sym.name, i, ndim))
            for i, sym in enumerate(self.index_symbols)
        }
        return str(expr.xreplace(replacements))


def codegen_epilogue_fusion(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    value: ast.expr,
    store_index: int,
) -> tuple[ast.expr, list[ast.expr]]:
    """Apply epilogue fusion to a store operation.

    This is the main entry point called from memory_ops.py store codegen.

    Args:
        state: The CodegenState
        subscript: The subscript indices
        value: The AST expression for the value being stored
        store_index: The store operation index

    Returns:
        Tuple of (transformed_value, extra_store_statements)
    """
    env = CompileEnvironment.current()
    tb = env._template_buffer
    if not tb:
        return value, []

    subscript_names = _get_subscript_names(state, subscript)
    if not subscript_names:
        return value, []

    original_value = value
    # Capture tb with narrowed type for use in closures
    template_buffer = tb

    def capture_epilogue(n: str) -> str:
        return template_buffer.capture_buffer(n, epilogue=True)

    # Get epilogue items: list of (accumulator_name, nodes) for this store
    if store_index in tb._fusion_store_map:
        acc_name = tb._fusion_store_map[store_index]
        # Only include if this accumulator has epilogue specs registered
        if acc_name in tb._epilogue_specs:
            nodes = tb._epilogue_specs[acc_name]
            epilogue_items = [(acc_name, nodes)] if nodes else []
        else:
            epilogue_items = []
    elif not tb._fusion_store_map:
        epilogue_items = list(tb._epilogue_specs.items())
        if len(epilogue_items) > 1:
            epilogue_items = []
    else:
        epilogue_items = []

    extra_stores: list[ast.expr] = []
    for acc_name, nodes in epilogue_items:
        if nodes:
            ep_nodes = list(nodes)
            if len(ep_nodes) == 1 and isinstance(ep_nodes[0], BaseSchedulerNode):
                ep_nodes = list(ep_nodes[0].get_nodes())
            if len(ep_nodes) > 1:
                graph_outputs = set(V.graph.get_output_names())
                filtered = [
                    n
                    for n in ep_nodes
                    if any(o.get_name() in graph_outputs for o in n.get_outputs())
                ]
                if filtered:
                    ep_nodes = filtered
            value_str = ast.unparse(value)
            acc_map = {acc_name: value_str}
            # Add aliases to accumulator map for epilogue fusion
            alias_map = getattr(template_buffer, "_helion_alias_map", None)
            if alias_map:
                for alias in alias_map:
                    acc_map.setdefault(alias, value_str)
            if template_buffer._output_aliases:
                for alias in template_buffer._output_aliases:
                    acc_map.setdefault(alias, value_str)
            value = _invoke_pointwise_with_ops_handler(
                ep_nodes, acc_map, subscript_names, capture_epilogue, "epilogue"
            )
            # If epilogue changes dtype, store to epilogue output buffer instead
            epilogue_out = ep_nodes[-1].node
            if isinstance(epilogue_out, ComputedBuffer):
                if epilogue_out.get_dtype() != V.graph.get_dtype(acc_name):
                    param = capture_epilogue(epilogue_out.get_name())
                    strides = (
                        list(epilogue_out.get_stride())
                        if isinstance(epilogue_out, Buffer)
                        else []
                    )
                    if len(strides) == len(subscript_names):
                        ndim = len(subscript_names)
                        offset = (
                            " + ".join(
                                f"{_broadcast_name(n, i, ndim)} * {s}"
                                for i, (n, s) in enumerate(
                                    zip(subscript_names, strides, strict=True)
                                )
                            )
                            or "0"
                        )
                        extra_stores.append(
                            expr_from_string(
                                f"tl.store({param} + {{offset}}, {{value}})",
                                offset=ast.parse(offset, mode="eval").body,
                                value=ast.parse(ast.unparse(value), mode="eval").body,
                            )
                        )
                        value = original_value  # Keep original for kernel output

    if tb._fusion_store_map and store_index in tb._fusion_store_map:
        tb._fusion_stored_info[tb._fusion_store_map[store_index]] = original_value
    if tb._fusion_store_map and store_index == max(tb._fusion_store_map.keys()):
        for nodes, acc_names in (
            [] if tb.uses_atomics() else tb._multi_dep_epilogue_specs
        ):
            if not nodes:
                continue
            # Collect stored values for accumulators that have been processed
            acc_values = {
                n: tb._fusion_stored_info[n]
                for n in acc_names
                if n in tb._fusion_stored_info
            }
            if not acc_values:
                continue

            epilogue_output = nodes[-1].node
            epilogue_buf_name = (
                epilogue_output.get_name()
                if isinstance(epilogue_output, IRNode)
                else None
            )
            if not epilogue_buf_name:
                continue

            acc_map = {n: ast.unparse(v) for n, v in acc_values.items()}
            result_ast = _invoke_pointwise_with_ops_handler(
                nodes, acc_map, subscript_names, capture_epilogue, "epilogue"
            )
            param_name = tb.capture_buffer(epilogue_buf_name, epilogue=True)

            # Get strides from epilogue output or template buffer
            strides = (
                list(epilogue_output.get_stride())
                if isinstance(epilogue_output, Buffer)
                else None
            )
            if not strides or len(strides) != len(subscript_names):
                for buf_name in tb._fusion_store_map.values():
                    buf = V.graph.get_buffer(buf_name)
                    if isinstance(buf, Buffer) and len(
                        s := list(buf.get_stride())
                    ) == len(subscript_names):
                        strides = s
                        break
            assert strides and len(strides) == len(subscript_names), (
                f"Cannot determine strides: subscript_names={subscript_names}, strides={strides}"
            )

            ndim = len(subscript_names)
            offset_parts = []
            for i, (n, stride) in enumerate(zip(subscript_names, strides, strict=True)):
                n_broadcasted = _broadcast_name(n, i, ndim)
                offset_parts.append(f"{n_broadcasted} * {stride}")
            offset_expr = " + ".join(offset_parts) or "0"

            extra_stores.append(
                expr_from_string(
                    f"tl.store({param_name} + {{offset}}, {{value}})",
                    offset=ast.parse(offset_expr, mode="eval").body,
                    value=ast.parse(ast.unparse(result_ast), mode="eval").body,
                )
            )

    return value, extra_stores


def codegen_prologue_fusion(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    value: ast.expr,
    input_name: str,
) -> ast.expr:
    """Apply prologue fusion to a load operation.

    This is the main entry point called from memory_ops.py load codegen.

    Args:
        state: The CodegenState
        subscript: The subscript indices
        value: The AST expression for the loaded value
        input_name: The name of the input tensor being loaded

    Returns:
        The transformed value after prologue fusion is applied
    """
    env = CompileEnvironment.current()
    tb = env._template_buffer
    if not tb:
        return value

    if input_name not in tb._prologue_specs:
        return value
    nodes, buffer_name = tb._prologue_specs[input_name]

    if not nodes:
        return value

    if input_name in tb._helion_mutated_input_names:
        if input_name in tb._prologue_fused_once:
            return value
        tb._prologue_fused_once.add(input_name)

    subscript_names = _get_subscript_names(state, subscript)
    if not subscript_names:
        return value

    input_str = ast.unparse(value)
    value = _invoke_pointwise_with_ops_handler(
        nodes,
        {buffer_name: input_str},
        subscript_names,
        lambda n: tb.capture_buffer(n, epilogue=False),
        "prologue",
    )

    return value  # noqa: RET504


def _invoke_pointwise_with_ops_handler(
    nodes: list,
    accumulator_map: dict[str, str],  # buffer_name -> variable_name string
    subscript_names: list[str],
    capture_buffer_fn: Callable[[str], str],
    node_type: str,
) -> ast.expr:
    """Execute Pointwise.inner_fn with FusionOpsHandler to generate Triton code AST."""
    last_node = nodes[-1].node
    assert isinstance(last_node, ComputedBuffer) and isinstance(
        last_node.data, Pointwise
    ), f"Expected Pointwise {node_type}, got {type(last_node)}"

    pw = last_node.data
    # Create symbols for inner_fn - must match len(pw.ranges)
    # Use names from subscript_names where available
    num_indices = len(pw.ranges)
    index_symbols = [
        sympy.Symbol(subscript_names[i] if i < len(subscript_names) else f"idx_{i}")
        for i in range(num_indices)
    ]
    handler = FusionOpsHandler(accumulator_map, index_symbols, capture_buffer_fn)

    # Disable load-time upcast so FusionOpsHandler can track dtype via CSEVariable
    with (
        inductor_config.patch({"triton.codegen_upcast_to_fp32": False}),
        V.set_ops_handler(handler),
        V.set_kernel_handler(
            TritonKernel({}, features=SIMDKernelFeatures([], sympy.S.One))
        ),
    ):
        result_str = str(pw.inner_fn(index_symbols))

    return expr_from_string(result_str)


def _get_subscript_names(
    state: CodegenState, subscript: list[object] | tuple[object, ...]
) -> list[str]:
    """Get index variable names for subscript dimensions.

    This function generates index variable names for each dimension in the subscript.
    For Tile and SymInt items, it uses the actual index variable names from the codegen.
    For slice(None) items (full-range slices), it finds the corresponding reduction
    dimension's block_id and uses its actual index variable name.
    """
    names: list[str] = []
    env = CompileEnvironment.current()
    # Collect block_ids used by Tile and SymInt items
    used_block_ids: set[int] = set()
    for item in subscript:
        if isinstance(item, Tile):
            used_block_ids.add(item.block_id)
        elif isinstance(item, torch.SymInt):
            if (block_id := env.get_block_id(item)) is not None:
                used_block_ids.add(block_id)

    # Find reduction block_ids that are not used by tiles
    # These are the block_ids for slice(None) dimensions
    reduction_block_ids: list[int] = []
    for info in env.block_sizes:
        if info.reduction and info.block_id not in used_block_ids:
            reduction_block_ids.append(info.block_id)

    reduction_idx = 0
    for i, item in enumerate(subscript):
        if isinstance(item, Tile):
            names.append(state.codegen.index_var(item.block_id))
        elif isinstance(item, torch.SymInt):
            if (block_id := env.get_block_id(item)) is not None:
                names.append(state.codegen.index_var(block_id))
        elif isinstance(item, int):
            pass
        elif item == slice(None):
            # Use the reduction dimension's actual index variable
            assert reduction_idx < len(reduction_block_ids), (
                f"More slice(None) items than reduction dimensions: {subscript}"
            )
            block_id = reduction_block_ids[reduction_idx]
            names.append(state.codegen.index_var(block_id))
            reduction_idx += 1
        else:
            raise ValueError(
                f"Unhandled subscript type {type(item).__name__} at index {i}"
            )
    return names
