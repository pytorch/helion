"""Triton codegen for Inductor prologue/epilogue fusion."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Callable

import sympy
import torch
import torch._inductor.lowering  # noqa: F401
from torch._inductor import config as inductor_config
from torch._inductor.bounds import ValueRanges
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.simd import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.ir import Buffer
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import Pointwise
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
    _dtype_handler = DtypePropagationOpsHandler()

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
        """Wrap string args as CSEVariable for dtype tracking."""
        if isinstance(a, CSEVariable):
            return a
        if isinstance(a, str):
            return CSEVariable(a, ValueRanges.unknown())
        return a

    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> CSEVariable:
        """Delegate to TritonOverrides for most ops, wrapping result in parens for precedence.

        Returns a CSEVariable with dtype tracking so subsequent ops can auto-upcast.
        """
        wrapped_args = tuple(self._wrap_arg(a) for a in args)
        wrapped_kwargs = {k: self._wrap_arg(v) for k, v in kwargs.items()}
        result = str(
            getattr(self._parent_handler, name)(*wrapped_args, **wrapped_kwargs)
        )
        result_str = f"({result})"

        # Use DtypePropagationOpsHandler to get proper result dtype
        if hasattr(self._dtype_handler, name):
            result_dtype = getattr(self._dtype_handler, name)(
                *wrapped_args, **wrapped_kwargs
            )
            return CSEVariable(result_str, ValueRanges.unknown(), dtype=result_dtype)
        return CSEVariable(result_str, ValueRanges.unknown())

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

    def store(
        self, name: str, index: sympy.Expr, value: str, mode: object = None
    ) -> str:
        """Handle store ops - just return the value being stored."""
        return str(value)

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
    capture_epilogue = lambda n: tb.capture_buffer(n, epilogue=True)

    # Get epilogue items: list of (accumulator_name, nodes) for this store
    if store_index in tb._fusion_store_map:
        acc_name = tb._fusion_store_map[store_index]
        nodes = tb._epilogue_specs.get(acc_name, [])
        epilogue_items = [(acc_name, nodes)] if nodes else []
    elif not tb._fusion_store_map:
        epilogue_items = list(tb._epilogue_specs.items())
    else:
        epilogue_items = []

    for acc_name, nodes in epilogue_items:
        if nodes:
            acc_map = {acc_name: ast.unparse(value)}
            value = _invoke_pointwise_with_ops_handler(
                nodes, acc_map, subscript_names, capture_epilogue, "epilogue"
            )

    if tb._fusion_store_map and store_index in tb._fusion_store_map:
        tb._fusion_stored_info[tb._fusion_store_map[store_index]] = original_value

    extra_stores: list[ast.expr] = []
    if tb._fusion_store_map and store_index == max(tb._fusion_store_map.keys()):
        for nodes, acc_names in (
            [] if tb.uses_atomics() else tb._multi_dep_epilogue_specs
        ):
            if not nodes:
                continue
            acc_values = {
                n: v for n in acc_names if (v := tb._fusion_stored_info.get(n))
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

    spec = tb._prologue_specs.get(input_name)
    if not spec:
        return value
    nodes, buffer_name = spec

    if not nodes:
        return value

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

    return value


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

    This extracts the variable names used for indexing (e.g., ["indices_0", "indices_1"])
    which are needed for fusion transformations.
    """
    names: list[str] = []
    env = CompileEnvironment.current()
    dim_idx = 0
    for i, item in enumerate(subscript):
        if isinstance(item, Tile):
            names.append(state.codegen.index_var(item.block_id))
            dim_idx += 1
        elif isinstance(item, torch.SymInt):
            if (block_id := env.get_block_id(item)) is not None:
                names.append(state.codegen.index_var(block_id))
            dim_idx += 1
        elif isinstance(item, int):
            pass
        elif isinstance(item, slice) and item == slice(None):
            names.append(f"indices_{dim_idx}")
            dim_idx += 1
        else:
            raise ValueError(
                f"Unhandled subscript type {type(item).__name__} at index {i}"
            )
    return names
