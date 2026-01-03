"""Triton codegen for Inductor prologue/epilogue fusion."""

from __future__ import annotations

import ast
import itertools
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
from torch._inductor.codegen.triton import texpr
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.ir import Buffer
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import Pointwise
import torch._inductor.lowering  # noqa: F401
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.utils import sympy_subs
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
        accumulator_map: dict[str, str],
        index_symbols: list[sympy.Symbol],
        capture_buffer_fn: Callable[[str], str],
        broadcast_names: list[str],
    ) -> None:
        super().__init__()
        self.accumulator_map = accumulator_map
        self.index_symbols = index_symbols
        self.capture_buffer = capture_buffer_fn
        self.broadcast_names = broadcast_names

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
        """Delegate to TritonOverrides for ops, wrapping result for dtype tracking."""
        wrapped_args = tuple(self._wrap_arg(a) for a in args)
        wrapped_kwargs = {k: self._wrap_arg(v) for k, v in kwargs.items()}
        result = str(
            getattr(self._parent_handler, name)(*wrapped_args, **wrapped_kwargs)
        )
        result_str = f"({result})"

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
        """Convert sympy expression to Triton code with broadcasting.

        Uses texpr to handle ModularIndexing, FloorDiv, etc.
        """
        return _sympy_to_triton(expr, self.broadcast_names)


def _sympy_to_triton(
    expr: sympy.Expr,
    broadcast_names: list[str],
) -> str:
    """Convert sympy expression to Triton code, applying broadcast/offsets."""
    ndim = len(broadcast_names)
    replacements: dict[sympy.Symbol, sympy.Expr] = {}

    for i, name in enumerate(broadcast_names):
        old_sym = sympy.Symbol(name)
        new_name = _broadcast_name(name, i, ndim) if ndim > 1 else name
        new_sym: sympy.Expr = sympy.Symbol(new_name)
        replacements[old_sym] = new_sym

    return texpr(expr.xreplace(replacements))


def _get_helion_transform(nodes: list, attr: str) -> object | None:
    for snode in reversed(nodes):
        if hasattr(snode, attr):
            return getattr(snode, attr)
    return None


def _build_store_expr(
    param_name: str,
    index_exprs: list[sympy.Expr],
    strides: list,
    result_ast: ast.expr,
    broadcast_names: list[str],
    mask_expr: str | None = None,
    extra_mask: ast.AST | None = None,
    offset: sympy.Expr | None = None,
) -> ast.expr:
    """Build a tl.store expression with computed offset and optional mask."""
    assert len(index_exprs) == len(strides), (len(index_exprs), len(strides))
    offset_parts = []
    for expr, stride in zip(index_exprs, strides, strict=True):
        expr_str = _sympy_to_triton(expr, broadcast_names)
        offset_parts.append(f"({expr_str}) * {stride}")

    if offset is not None and offset != 0:
        offset_parts.append(str(texpr(sympy.sympify(offset))))

    offset_expr = " + ".join(offset_parts) or "0"
    if broadcast_names:
        used_symbols = {sym.name for expr in index_exprs for sym in expr.free_symbols}
        missing = [
            (name, i)
            for i, name in enumerate(broadcast_names)
            if name not in used_symbols
        ]
        if missing:
            extra_terms = [
                f"(0 * {_broadcast_name(name, dim, len(broadcast_names))})"
                for name, dim in missing
            ]
            offset_expr = f"({offset_expr}) + " + " + ".join(extra_terms)

    mask_ast: ast.AST | None = None
    if mask_expr:
        mask_ast = ast.parse(mask_expr, mode="eval").body
    if extra_mask is not None:
        mask_ast = (
            extra_mask
            if mask_ast is None
            else expr_from_string("({mask}) & ({extra})", mask=mask_ast, extra=extra_mask)
        )

    template = (
        f"tl.store({param_name} + {{offset}}, {{value}}, mask={{mask}})"
        if mask_ast is not None
        else f"tl.store({param_name} + {{offset}}, {{value}})"
    )
    kwargs: dict[str, ast.expr] = {
        "offset": ast.parse(offset_expr, mode="eval").body,
        "value": ast.parse(ast.unparse(result_ast), mode="eval").body,
    }
    if mask_ast is not None:
        kwargs["mask"] = mask_ast
    return expr_from_string(template, **kwargs)


def _build_store_exprs(
    param_name: str,
    index_exprs: list[sympy.Expr],
    strides: list,
    result_ast: ast.expr,
    broadcast_names: list[str],
    mask_expr: str | None,
    broadcast_dims: list[tuple[int, int]] | None,
    extra_mask: ast.AST | None,
    offset: sympy.Expr | None,
) -> list[ast.expr]:
    if not broadcast_dims:
        return [
            _build_store_expr(
                param_name,
                index_exprs,
                strides,
                result_ast,
                broadcast_names,
                mask_expr,
                extra_mask,
                offset,
            )
        ]

    dims = list(broadcast_dims)
    ranges = [range(size) for _, size in dims]
    stores: list[ast.expr] = []
    for combo in itertools.product(*ranges):
        idx_exprs = list(index_exprs)
        for (dim, _), value in zip(dims, combo, strict=True):
            idx_exprs[dim] = sympy.Integer(value)
        stores.append(
            _build_store_expr(
                param_name,
                idx_exprs,
                strides,
                result_ast,
                broadcast_names,
                mask_expr,
                extra_mask,
                offset,
            )
        )
    return stores


def _process_epilogue(
    nodes: list,
    acc_map: dict[str, str],
    subscript_names: list[str],
    capture_buffer_fn: Callable[[str], str],
) -> tuple[
    ast.expr | None,
    str,
    list[sympy.Expr] | None,
    list | None,
    str | None,
    list[tuple[int, int]] | None,
    sympy.Expr | None,
    bool,
]:
    """Process epilogue nodes and return store info.

    Unified path for both single-output and multi-output epilogues.

    Returns:
        (result_ast, param_name, index_exprs, strides, mask, broadcast_dims, offset, unsupported)
        If unsupported=True, the epilogue cannot be fused and should be skipped.
    """
    epilogue_output = nodes[-1].node
    epilogue_buf_name = (
        epilogue_output.get_name() if isinstance(epilogue_output, IRNode) else None
    )
    epilogue_shape = (
        list(epilogue_output.get_size()) if isinstance(epilogue_output, IRNode) else None
    )

    kernel_symbols = [sympy.Symbol(n) for n in subscript_names]

    # Get epilogue strides for layout-aware stores
    epilogue_ndim = len(epilogue_shape) if epilogue_shape else len(subscript_names)
    epilogue_strides = None
    if epilogue_buf_name:
        buf = V.graph.get_buffer(epilogue_buf_name)
        if isinstance(buf, Buffer):
            strides = list(buf.get_stride())
            if len(strides) == epilogue_ndim:
                epilogue_strides = strides

    transform = _get_helion_transform(nodes, "_helion_epilogue_transform")
    if transform is None or transform.unsupported:
        return None, "", None, None, None, None, None, True
    if len(transform.kernel_symbols) != len(subscript_names):
        raise AssertionError(
            "Helion epilogue transform/kernel index rank mismatch; "
            "store-shape recording did not run or was incomplete."
        )

    repl = dict(zip(transform.kernel_symbols, kernel_symbols, strict=True))
    index_exprs = [sympy_subs(expr, repl) for expr in transform.index_exprs]
    mask_parts: list[str] = []
    for expr in transform.mod_masks:
        expr_str = _sympy_to_triton(sympy_subs(expr, repl), subscript_names)
        mask_parts.append(f"({expr_str} == 0)")

    # Invoke pointwise with transform
    result_ast = _invoke_pointwise_with_ops_handler(
        nodes, acc_map, subscript_names, capture_buffer_fn, transform=transform
    )

    # Determine strides for store (epilogue layout)
    strides = epilogue_strides
    assert strides is not None, f"Cannot determine strides for epilogue {epilogue_buf_name}"
    offset = None
    if isinstance(epilogue_output, IRNode):
        try:
            offset = epilogue_output.get_layout().offset
        except Exception:
            offset = None

    param_name = capture_buffer_fn(epilogue_buf_name) if epilogue_buf_name else ""
    if epilogue_shape is not None:
        for idx_expr, size in zip(index_exprs, epilogue_shape, strict=True):
            idx_str = _sympy_to_triton(idx_expr, subscript_names)
            size_str = texpr(sympy.sympify(size))
            mask_parts.append(f"(({idx_str} >= 0) & ({idx_str} < {size_str}))")
    mask = " & ".join(mask_parts) if mask_parts else None
    broadcast_dims = transform.broadcast_dims
    return result_ast, param_name, index_exprs, strides, mask, broadcast_dims, offset, False


def codegen_epilogue_fusion(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    value: ast.expr,
    extra_mask: ast.AST | None,
    store_index: int,
    indexing_idx: int,
) -> list[ast.expr]:
    """Apply epilogue fusion to a store operation and generate all stores.

    Main entry point called from memory_ops.py store codegen.
    """
    device_fn = state.device_function

    def generate_primary_store(val: ast.expr) -> ast.expr:
        return device_fn.get_indexing_strategy(indexing_idx).codegen_store(
            state, tensor, [*subscript], val, extra_mask
        )

    env = CompileEnvironment.current()
    tb = env._template_buffer
    if not tb:
        return [generate_primary_store(value)]

    subscript_names = _get_subscript_names(state, subscript)
    if not subscript_names:
        return [generate_primary_store(value)]
    if tb._recording_store_shapes:
        acc_name = None
        if store_index in tb._fusion_store_map:
            acc_name = tb._fusion_store_map[store_index]
        elif not tb._fusion_store_map:
            acc_name = tb.get_name()
        if acc_name and isinstance(tensor, torch.Tensor):
            tb._record_store_shape(
                acc_name,
                list(tensor.size()),
                list(tensor.stride()),
            )
        return [generate_primary_store(value)]
    original_value = value

    def capture_epilogue(n: str) -> str:
        return tb.capture_buffer(n, epilogue=True)

    # Get epilogue items for this store
    if store_index in tb._fusion_store_map:
        acc_name = tb._fusion_store_map[store_index]
        nodes = tb._epilogue_specs.get(acc_name, [])
        epilogue_items = [(acc_name, nodes)] if nodes else []
    elif not tb._fusion_store_map:
        epilogue_items = list(tb._epilogue_specs.items())
    else:
        epilogue_items = []

    redirect_stores: list[ast.expr] | None = None
    for acc_name, nodes in epilogue_items:
        if not nodes:
            continue

        # Always build redirect store when epilogue is fusable
        acc_map = {acc_name: ast.unparse(value)}
        (
            result_ast,
            param_name,
            index_exprs,
            strides,
            mask,
            broadcast_dims,
            offset,
            unsupported,
        ) = _process_epilogue(
            nodes,
            acc_map,
            subscript_names,
            capture_epilogue,
        )
        # If unsupported, skip fusion - Inductor will generate a separate kernel
        if not unsupported:
            redirect_stores = _build_store_exprs(
                param_name,
                index_exprs,
                strides,
                result_ast,
                subscript_names,
                mask,
                broadcast_dims,
                extra_mask,
                offset,
            )

    if tb._fusion_store_map and store_index in tb._fusion_store_map:
        tb._fusion_stored_info[tb._fusion_store_map[store_index]] = original_value

    # Handle multi-dep epilogues
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
                epilogue_output.get_name() if isinstance(epilogue_output, IRNode) else None
            )
            if not epilogue_buf_name:
                continue

            acc_map = {n: ast.unparse(v) for n, v in acc_values.items()}
            (
                result_ast,
                param_name,
                index_exprs,
                strides,
                mask,
                broadcast_dims,
                offset,
                unsupported,
            ) = _process_epilogue(
                nodes,
                acc_map,
                subscript_names,
                capture_epilogue,
            )
            # Skip unsupported multi-dep epilogues
            if not unsupported:
                extra_stores.extend(
                    _build_store_exprs(
                        param_name,
                        index_exprs,
                        strides,
                        result_ast,
                        subscript_names,
                        mask,
                        broadcast_dims,
                        extra_mask,
                        offset,
                    )
                )

    primary_stores = redirect_stores or [generate_primary_store(value)]
    return [*primary_stores, *extra_stores]


def codegen_prologue_fusion(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    value: ast.expr,
    input_name: str,
) -> ast.expr:
    """Apply prologue fusion to a load operation.

    Main entry point called from memory_ops.py load codegen.
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
    if tb._recording_store_shapes:
        return value

    input_str = ast.unparse(value)
    transform = _get_helion_transform(nodes, "_helion_prologue_transform")
    if transform is None or transform.unsupported:
        raise AssertionError(
            f"Missing Helion prologue transform for '{buffer_name}'."
        )

    with tb.prologue_context(buffer_name):
        return _invoke_pointwise_with_ops_handler(
            nodes,
            {buffer_name: input_str},
            subscript_names,
            lambda n: tb.capture_buffer(n, epilogue=False),
            transform=transform,
        )


def _invoke_pointwise_with_ops_handler(
    nodes: list,
    accumulator_map: dict[str, str],
    subscript_names: list[str],
    capture_buffer_fn: Callable[[str], str],
    transform: object | None = None,
) -> ast.expr:
    """Execute Pointwise.inner_fn with FusionOpsHandler to generate Triton code AST."""
    last_node = nodes[-1].node
    assert isinstance(last_node, ComputedBuffer) and isinstance(last_node.data, Pointwise)

    pw = last_node.data
    pw_size = list(pw.ranges)
    kernel_symbols = [sympy.Symbol(n) for n in subscript_names]

    if transform is not None:
        repl = dict(zip(transform.kernel_symbols, kernel_symbols, strict=True))
        index_symbols = [sympy_subs(expr, repl) for expr in transform.index_exprs]
        broadcast_names = subscript_names
    else:
        broadcast_names = [
            subscript_names[i] if i < len(subscript_names) else f"idx_{i}"
            for i in range(len(pw_size))
        ]
        index_symbols = [sympy.Symbol(n) for n in broadcast_names]

    handler = FusionOpsHandler(
        accumulator_map, index_symbols, capture_buffer_fn, broadcast_names
    )

    with (
        inductor_config.patch({"triton.codegen_upcast_to_fp32": False}),
        V.set_ops_handler(handler),
        V.set_kernel_handler(TritonKernel({}, features=SIMDKernelFeatures([], sympy.S.One))),
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
    for item in subscript:
        if isinstance(item, Tile):
            names.append(state.codegen.index_var(item.block_id))
        elif isinstance(item, torch.SymInt):
            if (block_id := env.get_block_id(item)) is not None:
                names.append(state.codegen.index_var(block_id))
        elif isinstance(item, int):
            pass
        elif item == slice(None):
            # Use the reduction dimension's actual index variable
            if reduction_idx < len(reduction_block_ids):
                block_id = reduction_block_ids[reduction_idx]
                names.append(state.codegen.index_var(block_id))
                reduction_idx += 1
            else:
                # Fallback: find a unique name (shouldn't happen normally)
                used_names = set(names)
                for info in env.block_sizes:
                    try:
                        used_names.add(state.codegen.index_var(info.block_id))
                    except (KeyError, IndexError):
                        pass
                next_idx = 0
                while f"indices_{next_idx}" in used_names:
                    next_idx += 1
                names.append(f"indices_{next_idx}")
    return names
