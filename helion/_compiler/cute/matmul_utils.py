from __future__ import annotations

import ast
from dataclasses import dataclass
import enum
from itertools import starmap
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
import torch
from torch.fx.node import Node

from ... import exc
from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _for_loop_step
from ...language._tracing_ops import _if
from ...language._tracing_ops import _while_loop
from ...language.atomic_ops import ATOMIC_OPS
from ...language.atomic_ops import atomic_add
from ...language.memory_ops import load
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..dtype_utils import cast_ast
from ..matmul_utils import _needs_f32_accumulator
from .indexing import CutePackedAffineLoad
from .indexing import CutePackedTerms
from .indexing import match_cute_stack_reshape_rhs

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ...autotuner.config_spec import MatmulFact
    from ..aten_lowering import LoweringContext
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..helper_function import CodegenInterface


def cute_f32_mma_matches_dot(
    lhs_dtype: torch.dtype,
    rhs_dtype: torch.dtype,
    acc_dtype: torch.dtype | None,
    out_dtype: torch.dtype | None,
) -> bool:
    """Whether a matmul unit accumulating in fp32 matches ``hl.dot`` semantics.

    A 16-bit float accumulator or result asks for a narrowed dot; with
    sub-32-bit operands, or a narrower float accumulator than the fp32
    operands, the scalar fallback provides it instead.
    """
    if not _needs_f32_accumulator(lhs_dtype, rhs_dtype) and not any(
        dtype is not None and dtype.is_floating_point and dtype.itemsize < 4
        for dtype in (acc_dtype, out_dtype)
    ):
        return True
    return out_dtype in (None, torch.float32) and acc_dtype in (None, torch.float32)


def cute_scalar_reduction_dtype(
    lhs_dtype: torch.dtype | None,
    rhs_dtype: torch.dtype | None,
    requested: torch.dtype | None,
) -> torch.dtype | None:
    """The dtype a scalar matmul sums ``requested``-typed products in.

    Sub-32-bit operands, and a sub-32-bit float accumulator, sum in fp32 and
    narrow the result once: rounding every product to fp16/bf16 would lose
    precision that ``hl.dot`` (and Triton's fp32 ``tl.dot`` tile) does not.
    """
    if (
        lhs_dtype is not None
        and rhs_dtype is not None
        and _needs_f32_accumulator(lhs_dtype, rhs_dtype)
    ) or (
        requested is not None and requested.is_floating_point and requested.itemsize < 4
    ):
        return torch.float32
    return requested


def cute_matmul_root_placements(
    env: CompileEnvironment, device_ir: DeviceIR
) -> tuple[tuple[MatmulFact, tuple[int, ...]], ...]:
    """Return proven contraction/root pairs for the active device regions.

    A fissioned DeviceIR retains kernel-wide facts but owns only a subset of
    root grids. Match those roots by the recorded graph attribution, never by
    the order of parallel matmul and grid lists. Callers supply their own dtype,
    layout, and shared-knob eligibility restrictions.
    """
    spec = env.config_spec
    if len(spec.matmul_facts) == 1 and len(device_ir.grid_block_ids) == 1:
        return ((spec.matmul_facts[0], tuple(device_ir.grid_block_ids[0])),)
    matmul = spec.kernel_matmul_fact
    grid = spec.kernel_grid_fact
    if matmul is None or grid is None or not matmul.attribution_complete:
        return ()
    active_roots = {tuple(root) for root in device_ir.grid_block_ids}
    if not active_roots.issubset(grid.grid_groups):
        return ()
    placed = []
    for resolved in matmul.matmuls:
        root = grid.group_for_graph(resolved.site.graph_id)
        if not root:
            return ()
        if root in active_roots:
            placed.append((resolved.fact, root))
    return tuple(placed)


@dataclass(frozen=True)
class DirectGroupedNLoadPlan:
    lhs_k_offset: int
    rhs_k_offset: int
    rhs_n_offset: int


def _cute_static_int_extent(size: object) -> int | None:
    if not isinstance(size, (int, torch.SymInt, sympy.Expr)):
        return None
    expr = sympy.sympify(size)
    if CompileEnvironment.has_current():
        expr = CompileEnvironment.current().specialize_expr(expr)
    if getattr(expr, "free_symbols", None):
        return None
    try:
        return int(expr)
    except TypeError:
        return None


def _cute_hinted_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt) and CompileEnvironment.has_current():
        return CompileEnvironment.current().size_hint(value)
    return None


def _cute_mask_to_preserves_k_invariance(node: torch.fx.Node, k_dim: int) -> bool:
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return False
    if not _cute_k_invariant_tensor_node(source, k_dim):
        return False
    source_val = source.meta.get("val")
    if not isinstance(source_val, torch.Tensor):
        return False
    if not CompileEnvironment.has_current():
        return False
    normalized_k_dim = k_dim % source_val.ndim
    return (
        CompileEnvironment.current().resolve_block_id(
            source_val.shape[normalized_k_dim]
        )
        is None
    )


def _cute_k_invariant_tensor_node(node: torch.fx.Node, k_dim: int) -> bool:
    if node.op != "call_function":
        return False

    target = node.target
    if target in {
        torch.ops.aten.full.default,
        torch.ops.aten.full_like.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.ones_like.default,
    }:
        return True

    from ...language._decorators import is_api_func
    from ...language._tracing_ops import _mask_to

    if is_api_func(target) and getattr(target, "__name__", "") in {
        "full",
        "zeros",
    }:
        return True

    if target == _mask_to:
        return _cute_mask_to_preserves_k_invariance(node, k_dim)

    unary_passthrough_targets = {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.prims.convert_element_type.default,
    }
    if target in unary_passthrough_targets:
        source = node.args[0] if node.args else None
        return isinstance(source, torch.fx.Node) and _cute_k_invariant_tensor_node(
            source,
            k_dim,
        )

    pointwise_targets = {
        torch.ops.aten.abs.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.pow.Tensor_Tensor,
        torch.ops.aten.relu.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.to.dtype,
    }
    if target in pointwise_targets:
        for arg in [*node.args, *node.kwargs.values()]:
            if isinstance(arg, torch.fx.Node):
                val = arg.meta.get("val")
                if isinstance(val, torch.Tensor) and not _cute_k_invariant_tensor_node(
                    arg,
                    k_dim,
                ):
                    return False
        return True

    return False


def cute_static_k_invariant_extent(
    lhs_node: torch.fx.Node | None,
    rhs_node: torch.fx.Node | None,
) -> int | None:
    if lhs_node is None or rhs_node is None:
        return None
    lhs_val = lhs_node.meta.get("val")
    rhs_val = rhs_node.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return None
    if lhs_val.ndim < 2 or rhs_val.ndim < 2:
        return None
    if not (
        _cute_k_invariant_tensor_node(lhs_node, -1)
        and _cute_k_invariant_tensor_node(rhs_node, -2)
    ):
        return None
    k_extent = _cute_static_int_extent(lhs_val.shape[-1])
    if k_extent is None or k_extent <= 1:
        return k_extent
    rhs_k_extent = _cute_static_int_extent(rhs_val.shape[-2])
    if rhs_k_extent != k_extent:
        return None
    return k_extent


def cute_static_serial_matmul_k_extent(
    lhs_node: torch.fx.Node | None,
    rhs_node: torch.fx.Node | None,
) -> int | None:
    def serial_extent(size: object) -> int | None:
        if (extent := _cute_static_int_extent(size)) is not None:
            return extent
        if not CompileEnvironment.has_current():
            return None
        env = CompileEnvironment.current()
        if not env.settings.static_shapes:
            return None
        # A ``:`` slice's size is its reduction block's symbol, whose hint is
        # the padded power of two; the loop must stop at the block's true size.
        block_id = env.resolve_block_id(size)
        if block_id is not None and env.block_sizes[block_id].reduction:
            return _cute_static_int_extent(env.block_sizes[block_id].size)
        return _cute_hinted_int(size)

    if lhs_node is None or rhs_node is None:
        return None
    lhs_val = lhs_node.meta.get("val")
    rhs_val = rhs_node.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return None
    if lhs_val.ndim != 2 or rhs_val.ndim != 2:
        return None
    k_extent = serial_extent(lhs_val.shape[-1])
    if k_extent is None or k_extent <= 1:
        return k_extent
    rhs_k_extent = serial_extent(rhs_val.shape[-2])
    if rhs_k_extent != k_extent:
        return None
    return k_extent


def emit_cute_serial_scalar_mm_from_loads(
    ctx: LoweringContext,
    lhs_node: torch.fx.Node,
    rhs_node: torch.fx.Node,
    *,
    k_extent: int | None,
    out_dtype: torch.dtype | None,
) -> ast.AST | None:
    def active_index_var(block_id: int) -> str | None:
        active_device_loops = getattr(ctx.cg, "active_device_loops", {})
        loops = active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.index_var(block_id)
        grid_state = getattr(ctx.cg, "current_grid_state", None)
        if grid_state is not None and block_id in grid_state.block_ids:
            return grid_state.strategy.index_var(block_id)
        return None

    def active_mask_var(block_id: int) -> str | None:
        active_device_loops = getattr(ctx.cg, "active_device_loops", {})
        loops = active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.mask_var(block_id)
        grid_state = getattr(ctx.cg, "current_grid_state", None)
        if grid_state is not None and block_id in grid_state.block_ids:
            return grid_state.strategy.mask_var(block_id)
        return None

    def add_offset(index_expr: str, offset: int) -> str:
        if offset == 0:
            return index_expr
        return f"({index_expr}) + {offset}"

    def tensor_ast_and_dtype(source: object) -> tuple[ast.AST, torch.dtype] | None:
        if isinstance(source, torch.Tensor):
            return (
                expr_from_string(ctx.cg.device_function.tensor_arg(source).name),
                source.dtype,
            )
        if isinstance(source, torch.fx.Node):
            val = source.meta.get("val")
            if isinstance(val, torch.Tensor):
                return (
                    expr_from_string(ctx.cg.device_function.tensor_arg(val).name),
                    val.dtype,
                )
        return None

    if k_extent is None:
        return None
    if lhs_node.target is not load or rhs_node.target is not load:
        return None
    if len(lhs_node.args) < 2 or len(rhs_node.args) < 2:
        return None
    # Each thread re-reads whole operand rows and columns at the matmul.
    if _cute_matmul_reread_unsafe(ctx.cg, lhs_node) or _cute_matmul_reread_unsafe(
        ctx.cg, rhs_node
    ):
        return None

    lhs_source = lhs_node.args[0]
    rhs_source = rhs_node.args[0]
    lhs_info = tensor_ast_and_dtype(lhs_source)
    rhs_info = tensor_ast_and_dtype(rhs_source)
    if lhs_info is None or rhs_info is None:
        return None
    lhs_tensor, lhs_dtype = lhs_info
    rhs_tensor, rhs_dtype = rhs_info
    rhs_val = rhs_node.meta.get("val")
    n_extent = (
        _cute_hinted_int(rhs_val.shape[1])
        if isinstance(rhs_val, torch.Tensor)
        else None
    )
    load_plan = analyze_direct_grouped_n_loads(
        lhs_node,
        rhs_node,
        k_extent=k_extent,
        n_extent=n_extent,
    )
    if load_plan is None:
        return None

    m_block_id = cute_resolve_active_block_id(ctx.cg, lhs_node.meta["val"].shape[0])
    n_block_id = cute_resolve_active_block_id(ctx.cg, rhs_node.meta["val"].shape[-1])
    if m_block_id is None or n_block_id is None:
        return None
    m_index = active_index_var(m_block_id)
    n_index = active_index_var(n_block_id)
    if m_index is None or n_index is None:
        return None
    m_mask = active_mask_var(m_block_id)
    n_mask = active_mask_var(n_block_id)
    reduction_dtype = cute_scalar_reduction_dtype(lhs_dtype, rhs_dtype, out_dtype)
    backend = CompileEnvironment.current().backend
    result_var = ctx.cg.device_function.new_var("dot_serial_result")

    def masked_scalar_load(
        tensor_value: ast.AST,
        row_expr: str,
        col_expr: str,
        *,
        source_dtype: torch.dtype,
        mask_expr: str | None,
    ) -> ast.AST:
        value = expr_from_string(
            f"{{tensor}}[{row_expr}, {col_expr}]",
            tensor=tensor_value,
        )
        if mask_expr is not None:
            zero = expr_from_string(f"{backend.dtype_str(source_dtype)}(0)")
            value = expr_from_string(
                "({value} if {mask} else {zero})",
                value=value,
                mask=expr_from_string(mask_expr),
                zero=zero,
            )
        if reduction_dtype is not None:
            value = cast_ast(value, reduction_dtype)
        return value

    def term_at(k_expr: str) -> ast.AST:
        lhs_value = masked_scalar_load(
            lhs_tensor,
            m_index,
            add_offset(k_expr, load_plan.lhs_k_offset),
            source_dtype=lhs_dtype,
            mask_expr=m_mask,
        )
        rhs_value = masked_scalar_load(
            rhs_tensor,
            add_offset(k_expr, load_plan.rhs_k_offset),
            add_offset(n_index, load_plan.rhs_n_offset),
            source_dtype=rhs_dtype,
            mask_expr=n_mask,
        )
        return expr_from_string("{lhs} * {rhs}", lhs=lhs_value, rhs=rhs_value)

    ctx.cg.add_statement(
        statement_from_string(f"{result_var} = {{term}}", term=term_at("0"))
    )
    if k_extent > 1:
        k_var = ctx.cg.device_function.new_var("serial_k")
        ctx.cg.add_statement(
            statement_from_string(
                f"for {k_var} in range(1, {k_extent}):\n"
                f"    {result_var} = {result_var} + {{term}}",
                term=term_at(k_var),
            )
        )
    result = expr_from_string(result_var)
    if out_dtype is not None and reduction_dtype != out_dtype:
        result = cast_ast(result, out_dtype)
    return result


def analyze_direct_grouped_n_loads(
    lhs_load: Node,
    rhs_load: Node,
    *,
    k_extent: int | None,
    n_extent: int | None = None,
) -> DirectGroupedNLoadPlan | None:
    def is_full_slice(index: object) -> bool:
        return (
            isinstance(index, slice)
            and index.start is None
            and index.stop is None
            and index.step is None
        )

    def contiguous_index_offset(
        index: object,
        *,
        required_extent: int | None = None,
    ) -> int | None:
        if is_full_slice(index):
            return 0
        if isinstance(index, Node):
            if index.target is not torch.ops.prims.iota.default:
                return None
            if index.kwargs.get("step", 1) != 1:
                return None
            fake = index.meta.get("val")
            if not isinstance(fake, torch.Tensor) or fake.ndim != 1:
                return None
            extent = _cute_hinted_int(fake.shape[0])
            if extent is None:
                return None
            if required_extent is not None and extent != required_extent:
                return None
            start = _cute_hinted_int(index.kwargs.get("start", 0))
            return 0 if start is None else start
        if not isinstance(index, slice):
            return None
        if index.step not in (None, 1):
            return None
        start = _cute_hinted_int(index.start) or 0
        stop = _cute_hinted_int(index.stop)
        if stop is None:
            return None
        if required_extent is not None and stop - start != required_extent:
            return None
        return start

    if lhs_load.target is not load or rhs_load.target is not load:
        return None
    if len(lhs_load.args) < 2 or len(rhs_load.args) < 2:
        return None
    lhs_index = lhs_load.args[1]
    rhs_index = rhs_load.args[1]
    if not isinstance(lhs_index, list) or not isinstance(rhs_index, list):
        return None
    if len(lhs_index) != 2 or len(rhs_index) != 2:
        return None

    lhs_k_offset = contiguous_index_offset(lhs_index[1], required_extent=k_extent)
    rhs_k_offset = contiguous_index_offset(rhs_index[0], required_extent=k_extent)
    rhs_n_offset = contiguous_index_offset(rhs_index[1], required_extent=n_extent)
    if lhs_k_offset is None or rhs_k_offset is None or rhs_n_offset is None:
        return None
    if lhs_k_offset < 0 or rhs_k_offset < 0 or rhs_n_offset < 0:
        return None
    return DirectGroupedNLoadPlan(
        lhs_k_offset=lhs_k_offset,
        rhs_k_offset=rhs_k_offset,
        rhs_n_offset=rhs_n_offset,
    )


def _cute_matmul_reread_unsafe(cg: CodegenInterface, load_node: Node) -> bool:
    """Whether a matmul lowering may not re-read ``load_node``'s tensor.

    The serial-K, synthetic-lane fold and operand-rematerializing lowerings
    read the operand again at the matmul, at elements other threads loaded,
    so the tensor must not be written anywhere in the kernel.
    """
    from ..generate_ast import GenerateAST
    from .view_ops import _tensor_written_anywhere

    return not isinstance(cg, GenerateAST) or _tensor_written_anywhere(cg, load_node)


@dataclass(frozen=True)
class CuteSerialOperand:
    """A matmul operand that a serial K loop can re-read from its ``load``.

    ``positions[d]`` is the subscript position of ``load_node`` that operand
    dim ``d`` reads; ``casts`` are the element-type conversions applied between
    the load and the operand, innermost first.
    """

    load_node: Node
    positions: tuple[int, ...]
    casts: tuple[torch.dtype, ...]


def cute_trace_serial_operand(
    cg: CodegenInterface, node: Node, consumer: Node
) -> CuteSerialOperand | None:
    """Trace a matmul operand through permutes, copies and casts to a ``load``.

    The load and its views must sit in ``consumer``'s graph, and the loaded
    tensor must not be written anywhere in the kernel: the matmul re-reads it
    at elements other threads loaded, after the load.  Every subscript entry
    must be a scalar, a tile index, a step-1 slice or a 1-d tensor; the
    caller decides which of the last three it can address.
    """
    from ..indexing_strategy import _get_tile_with_offset_info
    from .memory_ops import _cute_tile_axis_block_id

    operand_val = node.meta.get("val")
    if not isinstance(operand_val, torch.Tensor):
        return None
    # ``dims[d]``: the dim of ``current`` that operand dim ``d`` reads.
    dims = list(range(operand_val.ndim))
    casts: list[torch.dtype] = []
    current = node
    while current.target is not load:
        if current.op != "call_function" or not current.args:
            return None
        source = current.args[0]
        if not isinstance(source, Node):
            return None
        ndim = current.meta["val"].ndim
        target = current.target
        if target is torch.ops.aten.permute.default:
            # Output dim ``i`` of a permute is input dim ``order[i]``.
            order = current.args[1]
            assert isinstance(order, (list, tuple))
            dims = [cast("int", order[d]) % ndim for d in dims]
        elif target is torch.ops.aten.transpose.int:
            first = cast("int", current.args[1]) % ndim
            second = cast("int", current.args[2]) % ndim
            swap = {first: second, second: first}
            dims = [swap.get(d, d) for d in dims]
        elif target is torch.ops.aten.t.default:
            dims = [1 - d for d in dims]
        elif target is torch.ops.prims.convert_element_type.default:
            dtype = current.args[1]
            assert isinstance(dtype, torch.dtype)
            casts.append(dtype)
        elif target not in (
            torch.ops.aten.clone.default,
            torch.ops.aten.detach.default,
            torch.ops.aten.alias.default,
        ):
            return None
        current = source
    if current.graph is not consumer.graph or _cute_matmul_reread_unsafe(cg, current):
        return None
    if len(current.args) < 2 or (len(current.args) > 2 and current.args[2] is not None):
        return None
    subscript = current.args[1]
    load_val = current.meta.get("val")
    if not isinstance(subscript, (list, tuple)) or not isinstance(
        load_val, torch.Tensor
    ):
        return None
    positions: list[int] = []
    for position, index in enumerate(subscript):
        if isinstance(index, slice):
            if index.step not in (None, 1):
                return None
            positions.append(position)
        elif isinstance(index, Node):
            index_val = index.meta.get("val")
            if _get_tile_with_offset_info(index_val, current, position) is not None:
                return None
            if isinstance(index_val, torch.Tensor):
                if index_val.ndim != 1:
                    return None
                positions.append(position)
            elif isinstance(index_val, torch.SymInt):
                if _cute_tile_axis_block_id(index_val) is not None:
                    positions.append(position)
            elif not isinstance(index_val, int):
                return None
        elif type(index) is not int:
            return None
    if len(positions) != load_val.ndim:
        return None
    return CuteSerialOperand(
        load_node=current,
        positions=tuple(positions[d] for d in dims),
        casts=tuple(reversed(casts)),
    )


def emit_cute_serial_mm_from_load_views(
    cg: CodegenInterface,
    env: Mapping[Node, object],
    node: Node,
    lhs_node: Node,
    rhs_node: Node,
    *,
    acc: ast.AST | None,
    acc_dtype: torch.dtype | None,
    out_dtype: torch.dtype | None,
) -> ast.AST | None:
    """Lower a matmul whose K axis has no thread axis of its own as a serial loop.

    K sharing its block id with M or N (``carry[:, :] @ w`` with a square
    ``carry``: both full slices dedup onto one reduction block) cannot be
    reduced across threads, since the SIMT lowering gives that block one lane
    coordinate.  Each thread instead loops over the static K extent and
    re-reads both operands from global memory at its own (batch, M) / (batch,
    N) coordinates, through the permutes, copies and casts recorded by
    ``cute_trace_serial_operand``.  Returns None when an operand is not such a
    load view, K is not static, or the free axes are not distinct active
    blocks.  ``lhs_node``/``rhs_node`` are the operands of ``node`` (an mm,
    bmm or hl.dot, or the operands after an addmm/baddbmm accumulator).
    """
    from ...language.memory_ops import _cute_index_exprs
    from ...language.memory_ops import _cute_scalar_load_expr
    from ..inductor_lowering import CodegenState
    from .memory_ops import _cute_tile_axis_block_id

    lhs_val = lhs_node.meta["val"]
    rhs_val = rhs_node.meta["val"]
    ndim = lhs_val.ndim
    if ndim not in (2, 3) or rhs_val.ndim != ndim:
        return None
    lhs = cute_trace_serial_operand(cg, lhs_node, node)
    rhs = cute_trace_serial_operand(cg, rhs_node, node)
    if lhs is None or rhs is None:
        return None

    compile_env = CompileEnvironment.current()
    canonical = compile_env.canonical_block_id
    # The free axes (batch, M, N), each a distinct active block whose lane
    # coordinate is this thread's output element.
    free_sizes = [lhs_val.shape[-2], rhs_val.shape[-1]]
    if ndim == 3:
        free_sizes.append(lhs_val.shape[0])
    free_blocks: list[int] = []
    for size in free_sizes:
        block_id = cute_resolve_active_block_id(cg, size)
        if block_id is None:
            return None
        free_blocks.append(block_id)
    if len({canonical(block_id) for block_id in free_blocks}) != len(free_blocks):
        return None
    m_block, n_block = free_blocks[:2]
    if ndim == 3:
        batch_block = free_blocks[2]
        rhs_batch = cute_resolve_active_block_id(cg, rhs_val.shape[0])
        if rhs_batch is None or canonical(rhs_batch) != canonical(batch_block):
            return None
        lhs_roles: tuple[int | None, ...] = (batch_block, m_block, None)
        rhs_roles: tuple[int | None, ...] = (batch_block, None, n_block)
    else:
        lhs_roles = (m_block, None)
        rhs_roles = (None, n_block)

    k_var = cg.device_function.new_var("serial_k", dce=False)

    def element(
        operand: CuteSerialOperand, roles: tuple[int | None, ...]
    ) -> tuple[ast.AST, int, list[str]] | None:
        """The operand element at ``k_var``, its K extent, and its masks."""
        load_node = operand.load_node
        source = load_node.args[0]
        tensor = source.meta.get("val") if isinstance(source, Node) else source
        # fp8 / fp4 storage needs a raw-byte load and a decode, not ``.load()``.
        if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            return None
        subscript = load_node.args[1]
        assert isinstance(subscript, (list, tuple))
        index_exprs: dict[int, str] = {}
        masks: list[str] = []
        k_extent: int | None = None
        for position, role in zip(operand.positions, roles, strict=True):
            index = subscript[position]
            if role is None:
                # The contraction axis: a static range read at ``k_var``.
                if isinstance(index, slice):
                    full = _cute_static_int_extent(tensor.shape[position])
                    start = _cute_static_int_extent(index.start or 0)
                    stop = (
                        full
                        if index.stop is None
                        else _cute_static_int_extent(index.stop)
                    )
                    if full is None or start is None or stop is None:
                        return None
                    stop = min(stop, full)
                    k_extent = stop - start
                elif (
                    isinstance(index, Node)
                    and index.target is torch.ops.prims.iota.default
                    and index.kwargs.get("step", 1) == 1
                ):
                    start = _cute_static_int_extent(index.kwargs.get("start", 0))
                    k_extent = _cute_static_int_extent(index.meta["val"].shape[0])
                    full = _cute_static_int_extent(tensor.shape[position])
                    if (
                        start is None
                        or k_extent is None
                        or full is None
                        or start + k_extent > full
                    ):
                        return None
                else:
                    return None
                if start < 0:
                    return None
                index_exprs[position] = k_var if start == 0 else f"({k_var}) + {start}"
                continue
            index_var = _cute_active_index_var(cg, role)
            if index_var is None:
                return None
            if isinstance(index, slice):
                start = _cute_static_int_extent(index.start or 0)
                if start is None or start < 0:
                    return None
                index_exprs[position] = (
                    index_var if start == 0 else f"({index_var}) + {start}"
                )
            elif (
                isinstance(index, Node)
                and isinstance(index_val := index.meta.get("val"), torch.SymInt)
                and (tile_block := _cute_tile_axis_block_id(index_val)) is not None
                and canonical(tile_block) == canonical(role)
            ):
                index_exprs[position] = index_var
            else:
                return None
            if (mask_var := _cute_active_mask_var(cg, role)) is not None:
                masks.append(mask_var)
        # Scalar entries (constants, ``tile.id``, ``hl.grid`` indices) lower as
        # in the load itself; the axes above are stood in for by ``0``.
        state = CodegenState(
            cast("Any", cg),
            fx_node=load_node,
            env=cast("dict[Node, Any]", env),
        )
        scalar_exprs = _cute_index_exprs(
            state,
            [
                0
                if position in index_exprs
                else (index.meta["val"] if isinstance(index, Node) else index)
                for position, index in enumerate(subscript)
            ],
            tensor=tensor,
        )
        value = expr_from_string(
            _cute_scalar_load_expr(
                cg.device_function.tensor_arg(tensor).name,
                list(starmap(index_exprs.get, enumerate(scalar_exprs))),
                tensor.dtype,
            )
        )
        for dtype in operand.casts:
            value = cast_ast(value, dtype)
        assert k_extent is not None
        return value, k_extent, masks

    lhs_element = element(lhs, lhs_roles)
    rhs_element = element(rhs, rhs_roles)
    if lhs_element is None or rhs_element is None:
        return None
    lhs_value, lhs_k_extent, lhs_masks = lhs_element
    rhs_value, rhs_k_extent, rhs_masks = rhs_element
    if lhs_k_extent != rhs_k_extent or lhs_k_extent < 1:
        return None

    reduction_dtype = (
        cute_scalar_reduction_dtype(
            lhs_val.dtype, rhs_val.dtype, acc_dtype or out_dtype
        )
        or torch.float32
    )
    reduction_str = CompileEnvironment.current().backend.dtype_str(reduction_dtype)
    acc_var = cg.device_function.new_var("serial_dot_acc")
    cg.add_statement(statement_from_string(f"{acc_var} = {reduction_str}(0.0)"))
    body = f"{acc_var} = {acc_var} + {{lhs}} * {{rhs}}"
    mask_vars = list(dict.fromkeys([*lhs_masks, *rhs_masks]))
    if mask_vars:
        guard = " and ".join(mask_vars)
        body = f"if {guard}:\n        {body}"
    cg.add_statement(
        statement_from_string(
            f"for {k_var} in range({lhs_k_extent}):\n    {body}",
            lhs=cast_ast(lhs_value, reduction_dtype),
            rhs=cast_ast(rhs_value, reduction_dtype),
        )
    )
    result: ast.AST = expr_from_string(acc_var)
    final_dtype = out_dtype
    if acc is not None:
        base = acc
        if acc_dtype is not None and acc_dtype != reduction_dtype:
            base = cast_ast(base, reduction_dtype)
        result = expr_from_string("{acc} + {prod}", acc=base, prod=result)
        final_dtype = acc_dtype
    if final_dtype is not None and final_dtype != reduction_dtype:
        result = cast_ast(result, final_dtype)
    return result


def strip_new_vars(node: Node) -> Node:
    """``node`` without the ``_new_var`` copies the tracer wraps it in."""
    from ...language._tracing_ops import _new_var

    while node.op == "call_function" and node.target is _new_var:
        (source,) = node.args
        if not isinstance(source, Node):
            break
        node = source
    return node


def loop_carries(
    graph: torch.fx.Graph, graphs: Sequence[GraphInfo]
) -> dict[Node, Node] | None:
    """The carries of a ``for`` loop body among ``graphs``.

    Maps each body output that the phi after the loop merges with an initial
    value to the placeholder receiving that initial value.  None when
    ``graph`` is not such a body or its loop statement is not among
    ``graphs``.
    """
    from ...language._tracing_ops import _phi
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import control_flow_parent_entries

    loops = [
        graph_info
        for graph_info in graphs
        if isinstance(graph_info, ForLoopGraphInfo) and graph_info.graph is graph
    ]
    if len(loops) != 1:
        return None
    parent = control_flow_parent_entries(graphs).get(loops[0].graph_id)
    if parent is None:
        return None
    loop_node, args_slot = parent
    outer_args = loop_node.args[args_slot]
    (output,) = graph.find_nodes(op="output")
    results = output.args[0]
    if not isinstance(results, (list, tuple)) or not isinstance(
        outer_args, (list, tuple)
    ):
        return None
    placeholders = list(graph.find_nodes(op="placeholder"))
    carries: dict[Node, Node] = {}
    for user in loop_node.users:
        index = user.args[1] if user.target is operator.getitem else None
        if not isinstance(index, int) or not isinstance(results[index], Node):
            continue
        for phi in user.users:
            if phi.target is not _phi or phi.args[1] is not user:
                continue
            for placeholder, outer in zip(placeholders, outer_args, strict=True):
                if outer is phi.args[0]:
                    carries[results[index]] = placeholder
    return carries


# Reads of a tensor's metadata, not its values.
SHAPE_QUERY_TARGETS = frozenset(
    {
        torch.ops.aten.sym_size.int,
        torch.ops.aten.sym_stride.int,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.sym_storage_offset.default,
    }
)


def reads_value_of(node: Node, sources: set[Node]) -> bool:
    """Whether ``node``'s value depends on one of ``sources`` (shape queries aside)."""
    stack = [node]
    seen: set[Node] = set()
    while stack:
        current = stack.pop()
        if current in sources:
            return True
        if current not in seen and current.target not in SHAPE_QUERY_TARGETS:
            seen.add(current)
            stack.extend(current.all_input_nodes)
    return False


def reads_loop_carry(node: Node, carries: dict[Node, Node]) -> bool:
    """Whether ``node`` depends on a placeholder that receives a loop carry."""
    return reads_value_of(node, set(carries.values()))


def cute_outer_accumulates_result(
    fx_node: torch.fx.Node | None,
    *,
    is_acc_none: bool,
) -> bool:
    return cute_outer_accumulator_node(fx_node, is_acc_none=is_acc_none) is not None


# Pure dtype / layout changes and lane-invariant scaling between a matmul and
# the ``hl.atomic_add`` that consumes it.  Each is linear in the matmul result,
# so adding the per-K-lane partial products separately gives the same total.
_CUTE_ATOMIC_LINEAR_PASSTHROUGH_TARGETS = frozenset(
    {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.expand.default,
    }
)
_CUTE_ATOMIC_LINEAR_SCALE_TARGETS = frozenset(
    {
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
    }
)
# Device IR control flow: ``(graph id argument, subgraph argument list)``
# positions whose list entries line up with the subgraph's placeholders.
_CUTE_SUBGRAPH_ARG_SLOTS: dict[object, tuple[tuple[int, int], ...]] = {
    _for_loop: ((0, 3),),
    _for_loop_step: ((0, 3),),
    _while_loop: ((0, 2), (1, 2)),
    _if: ((1, 3), (2, 4)),
}


class CuteAtomicLaneRoute(enum.Enum):
    """How a matmul whose K axis is split across lanes reaches ``hl.atomic_*``."""

    NONE = "none"
    """No atomic consumes the result inside the K lane loop: keep the running sum."""
    PER_LANE = "per_lane"
    """Each K lane's partial product sum is atomically added on its own."""
    OWNED = "owned"
    """Complete the K sum with the owned product-sum marker before the atomic."""


def cute_atomic_consumer_lane_route(
    fx_node: torch.fx.Node | None,
    *,
    is_acc_none: bool,
    get_graph: Callable[[int], GraphInfo],
) -> CuteAtomicLaneRoute:
    """Pick how a lane-split matmul result that reaches ``hl.atomic_*`` lowers.

    Only meaningful when the matmul's K axis really is split across a serial
    lane loop: the per-thread ``dot_acc`` running sum then exposes a prefix
    of the contraction to every lane iteration, and every consumer inside the
    lane loop (the matmul's own graph plus the loop and branch subgraphs
    nested in it, reached through ``get_graph``) sees that prefix.  A
    thread-mapped K axis is complete at every thread and needs none of this,
    so callers decide after resolving the lane variable.

    ``PER_LANE``: ``hl.atomic_add`` is the only consumer of a bare fp32/fp64
    matmul result, separated from it by nothing but linear, lane-invariant
    transforms (views, widening float casts, multiplication or division by a
    Python number).  Each lane's partial sum is then atomically added on its
    own.

    ``OWNED``: the same chain, but the matmul result is half precision, so
    every per-lane partial would be rounded before its add.  The owned
    product-sum marker completes the K sum first (or its lane scheduler
    declines the lowering).

    Any other route to an atomic op (a second user, a tensor or
    kernel-argument scale, a non-linear op, a second matmul fed by the
    prefix, ``atomic_max`` and friends, a loop-carried accumulator, or an
    atomic inside a nested loop or branch) cannot be split per lane, and the
    running sum is wrong for it too, so it is rejected loudly.  A result that
    only leaves the K loop through the loop's outputs is complete there and
    is not followed.
    """
    if fx_node is None:
        return CuteAtomicLaneRoute.NONE
    if is_acc_none and _linear_atomic_add_consumer(fx_node) is not None:
        val = fx_node.meta["val"]
        assert isinstance(val, torch.Tensor)
        # Per-lane partials are exact for integer results and carry one fp32
        # rounding each; half-precision results would stack one rounding per
        # K lane, so they take the owned product-sum route instead.
        if val.dtype in (torch.float32, torch.float64) or not val.is_floating_point():
            return CuteAtomicLaneRoute.PER_LANE
        return CuteAtomicLaneRoute.OWNED
    if _reaches_atomic_value(fx_node, set(), get_graph):
        raise exc.BackendUnsupported(
            "cute",
            "hl.atomic_* consumes a matmul result whose K axis is split across "
            "lanes through a chain the scalar fallback cannot split per lane "
            "(shared result, non-linear op, second matmul, tensor scale, "
            "loop-carried accumulator, or an atomic in a nested loop or branch)",
        )
    return CuteAtomicLaneRoute.NONE


def _atomic_value_argument(user: torch.fx.Node) -> object:
    # ``hl.atomic_add(target, index, value, sem=...)``: the value is the third
    # positional argument.  ``atomic_cas`` puts ``expected`` there instead,
    # so this helper is only for ``atomic_add``.
    assert user.target is atomic_add
    if "value" in user.kwargs:
        return user.kwargs["value"]
    return user.args[2] if len(user.args) > 2 else None


def cute_per_lane_atomic_consumer(fx_node: torch.fx.Node) -> torch.fx.Node:
    """The ``hl.atomic_add`` node a ``PER_LANE``-routed matmul's partials reach."""
    consumer = _linear_atomic_add_consumer(fx_node)
    assert consumer is not None
    return consumer


def _linear_atomic_add_consumer(fx_node: torch.fx.Node) -> torch.fx.Node | None:
    """The ``hl.atomic_add`` whose value is *fx_node* through linear ops alone, or None."""
    node = fx_node
    while True:
        users = [user for user in node.users if isinstance(user, torch.fx.Node)]
        if len(users) != 1:
            return None
        (user,) = users
        if user.op != "call_function":
            return None
        if user.target is atomic_add:
            return user if _atomic_value_argument(user) is node else None
        if user.target is torch.ops.prims.convert_element_type.default:
            # Each lane's partial is cast before its atomic add, so only a
            # cast that keeps every fp32 partial exact may stay in the chain.
            if user.args[1] not in (torch.float32, torch.float64):
                return None
            node = user
            continue
        if user.target in _CUTE_ATOMIC_LINEAR_PASSTHROUGH_TARGETS:
            node = user
            continue
        if user.target in _CUTE_ATOMIC_LINEAR_SCALE_TARGETS and len(user.args) == 2:
            lhs, rhs = user.args
            other = rhs if lhs is node else lhs
            is_division = user.target in (
                torch.ops.aten.div.Tensor,
                torch.ops.aten.div.Scalar,
            )
            if is_division and lhs is not node:
                return None
            if not isinstance(other, (int, float)) or isinstance(other, bool):
                return None
            node = user
            continue
        return None


def _subgraph_placeholders(
    user: torch.fx.Node,
    node: torch.fx.Node,
    get_graph: Callable[[int], GraphInfo],
) -> Iterator[torch.fx.Node]:
    """Yield the subgraph placeholders that *user* binds to *node*."""
    for graph_id_index, args_index in _CUTE_SUBGRAPH_ARG_SLOTS[user.target]:
        graph_id = user.args[graph_id_index]
        outer_args = user.args[args_index]
        assert isinstance(graph_id, int)
        assert isinstance(outer_args, (list, tuple))
        placeholders = get_graph(graph_id).graph.find_nodes(op="placeholder")
        for placeholder, outer_arg in zip(placeholders, outer_args, strict=True):
            if outer_arg is node:
                yield placeholder


def _reaches_atomic_value(
    node: torch.fx.Node,
    visited: set[torch.fx.Node],
    get_graph: Callable[[int], GraphInfo],
) -> bool:
    """Whether *node* flows into an atomic op inside the K lane loop.

    Every op of the graph propagates the per-lane prefix of a running sum,
    including a second matmul fed by it, so the walk follows all users and
    descends into the loop and branch subgraphs that take *node* as an
    argument; every value such a subgraph returns counts as tainted too.  The
    walk does not climb out through a graph's outputs: the running sum is
    only used when the K lane loop is the innermost open scope, so the only
    outputs the matmul's own graph can reach are the K loop's, and a value
    that leaves the K loop is complete once the loop has finished.
    """
    if node in visited:
        return False
    visited.add(node)
    for user in node.users:
        if not isinstance(user, torch.fx.Node) or user.op != "call_function":
            continue
        if user.target in ATOMIC_OPS:
            # ``user`` reads ``node`` in some argument (value, expected or an
            # index), and every one of them is wrong for a prefix.
            return True
        if user.target in _CUTE_SUBGRAPH_ARG_SLOTS and any(
            _reaches_atomic_value(placeholder, visited, get_graph)
            for placeholder in _subgraph_placeholders(user, node, get_graph)
        ):
            return True
        if _reaches_atomic_value(user, visited, get_graph):
            return True
    return False


def cute_outer_accumulator_node(
    fx_node: torch.fx.Node | None,
    *,
    is_acc_none: bool,
) -> torch.fx.Node | None:
    if not is_acc_none or fx_node is None:
        return None
    users = [user for user in fx_node.users if isinstance(user, torch.fx.Node)]
    if len(users) != 1:
        return None
    (user,) = users
    if user.target is not torch.ops.aten.add.Tensor or len(user.args) < 2:
        return None
    lhs, rhs = user.args[:2]
    if lhs is fx_node:
        other_arg = rhs
    elif rhs is fx_node:
        other_arg = lhs
    else:
        return None
    if not isinstance(other_arg, torch.fx.Node):
        return None
    stack_trace = user.meta.get("stack_trace")
    if not isinstance(stack_trace, str):
        source_line = None
    else:
        source_lines = [
            line.strip() for line in stack_trace.splitlines() if line.strip()
        ]
        source_line = source_lines[-1] if source_lines else None
    if source_line is not None:
        if "+=" in source_line:
            return other_arg
        try:
            parsed = ast.parse(source_line, mode="exec")
        except SyntaxError:
            parsed = None
        if (
            parsed is not None
            and len(parsed.body) == 1
            and isinstance(parsed.body[0], ast.Assign)
        ):
            assign = parsed.body[0]
            if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
                return None
            target_name = assign.targets[0].id
            value = assign.value
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                operands = (value.left, value.right)
                if any(
                    isinstance(operand, ast.Name) and operand.id == target_name
                    for operand in operands
                ):
                    return other_arg
                return None
    from ...language._tracing_ops import _new_var

    if other_arg.target is not _new_var or len(other_arg.args) != 1:
        return None
    source = other_arg.args[0]
    if not isinstance(source, torch.fx.Node) or source.op != "placeholder":
        return None
    output_nodes = [node for node in user.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        return None
    (output_vals,) = output_nodes[0].args
    if isinstance(output_vals, torch.fx.Node):
        return other_arg if output_vals is user else None
    if not isinstance(output_vals, (list, tuple)):
        return None
    if user not in output_vals:
        return None
    return other_arg


def cute_outer_accumulator_dtype(
    fx_node: torch.fx.Node | None,
    *,
    is_acc_none: bool,
) -> torch.dtype | None:
    outer_acc = cute_outer_accumulator_node(fx_node, is_acc_none=is_acc_none)
    if outer_acc is None:
        return None
    val = outer_acc.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val.dtype
    return None


def cute_supports_scalar_matmul_fallback(
    cg: CodegenInterface,
    lhs_val: torch.Tensor,
    rhs_val: torch.Tensor,
    out_val: torch.Tensor,
    *,
    k_block_id: int | None,
) -> bool:
    if lhs_val.ndim != 2 or rhs_val.ndim != 2 or out_val.ndim != 2:
        return True
    if k_block_id is not None:
        return True
    grid_state = getattr(cg, "current_grid_state", None)
    if grid_state is None:
        return True
    if len(grid_state.block_ids) >= 2:
        return True
    n_block_id = CompileEnvironment.current().resolve_block_id(out_val.shape[-1])
    if n_block_id is not None:
        return True
    return all(size <= 1 for size in grid_state.thread_axis_sizes.values())


def cute_resolve_active_block_id(
    cg: CodegenInterface,
    size: int | torch.SymInt,
) -> int | None:
    cg_any = cast("Any", cg)
    env = CompileEnvironment.current()
    canonical_block_id = getattr(env, "canonical_block_id", lambda block_id: block_id)
    block_id = env.resolve_block_id(size)
    if block_id is None:
        return None
    canonical_candidate = canonical_block_id(block_id)
    active_block_ids: set[int] = set()
    if cg_any.current_grid_state is not None:
        active_block_ids.update(cg_any.current_grid_state.block_ids)
    for loops in cg_any.active_device_loops.values():
        for loop_state in loops:
            active_block_ids.update(loop_state.block_ids)
    matches = [
        active_block_id
        for active_block_id in active_block_ids
        if canonical_block_id(active_block_id) == canonical_candidate
    ]
    if not matches:
        return None
    if block_id in matches:
        return block_id
    if len(matches) != 1:
        return None
    return matches[0]


def cute_resolve_active_matmul_k_block_id(
    cg: CodegenInterface,
    lhs_k_size: int | torch.SymInt,
    rhs_k_size: int | torch.SymInt,
    rhs_n_size: int | torch.SymInt,
    *,
    lhs_m_size: int | torch.SymInt | None = None,
) -> int | None:
    env = CompileEnvironment.current()
    canonical_block_id = getattr(env, "canonical_block_id", lambda block_id: block_id)
    lhs_k_block_id = cute_resolve_active_block_id(cg, lhs_k_size)
    rhs_k_block_id = cute_resolve_active_block_id(cg, rhs_k_size)
    if lhs_k_block_id is None or rhs_k_block_id is None:
        return None
    if canonical_block_id(lhs_k_block_id) != canonical_block_id(rhs_k_block_id):
        return None
    # K must be a block of its own: a free (M or N) axis bound to the same
    # block would make the cross-thread K reduction sum the free axis too
    # (one lane coordinate per block id), broadcasting one value per row.
    k_canonical = canonical_block_id(lhs_k_block_id)
    for free_size in (rhs_n_size, lhs_m_size):
        if free_size is None:
            continue
        free_block_id = cute_resolve_active_block_id(cg, free_size)
        if (
            free_block_id is not None
            and canonical_block_id(free_block_id) == k_canonical
        ):
            return None
    return lhs_k_block_id


def cute_rematerialize_rhs_at_contraction_block(
    ctx: LoweringContext,
    lhs_node: torch.fx.Node,
    rhs_node: torch.fx.Node,
) -> tuple[ast.AST, int] | None:
    """Re-lower a matmul ``rhs`` whose contraction axis is the *wrong* block.

    Handles the case where ``lhs.shape[-1]`` (the contraction) and
    ``rhs.shape[-2]`` (the contraction) resolve to *different* active block
    ids that happen to share the same size (e.g. ``mask[b, q, k] @ q[b, q, h]``
    where the lhs contracts on the inner ``tile_k`` loop but the rhs's middle
    dim is indexed by the outer ``tile_q``).  The originally-lowered ``rhs``
    load is then loop-invariant in the contraction loop and the matmul would
    read ``rhs[..., q, ...]`` every step instead of ``rhs[..., k, ...]``.

    Re-codegens the ``rhs`` load with the contraction block remapped from its
    original (loop-invariant) block to the active contraction block, so each
    contraction step reads the correct element.  Returns
    ``(new_rhs_ast, k_block_id)`` on success, or ``None`` when the case does
    not apply (caller keeps its existing behaviour).
    """
    env = CompileEnvironment.current()
    canonical_block_id = getattr(env, "canonical_block_id", lambda block_id: block_id)
    lhs_val = lhs_node.meta.get("val")
    rhs_val = rhs_node.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return None
    if lhs_val.ndim < 2 or rhs_val.ndim < 2:
        return None
    lhs_k_block_id = cute_resolve_active_block_id(ctx.cg, lhs_val.shape[-1])
    rhs_k_block_id = cute_resolve_active_block_id(ctx.cg, rhs_val.shape[-2])
    if lhs_k_block_id is None or rhs_k_block_id is None:
        return None
    if canonical_block_id(lhs_k_block_id) == canonical_block_id(rhs_k_block_id):
        # Same block already - no remap needed; the standard resolver handles it.
        return None
    # The two contraction axes must have the same extent for the matmul to be
    # well-defined; require the same static size before remapping.
    if not env.known_equal(lhs_val.shape[-1], rhs_val.shape[-2]):
        return None
    # Only re-materialize a plain ``load`` whose middle (contraction) dim is the
    # block we need to remap.  Anything more complex (computed operands,
    # permutes) is left to the existing path.
    if rhs_node.target is not load or _cute_matmul_reread_unsafe(ctx.cg, rhs_node):
        return None
    if len(rhs_node.args) < 2 or not isinstance(rhs_node.args[1], (list, tuple)):
        return None
    # The remapped rhs would now contract on ``lhs_k_block_id``; reject if its
    # free (N) axis collides with that same block.
    rhs_n_block_id = cute_resolve_active_block_id(ctx.cg, rhs_val.shape[-1])
    if rhs_n_block_id is not None and canonical_block_id(
        rhs_n_block_id
    ) == canonical_block_id(lhs_k_block_id):
        return None

    new_rhs = _cute_recodegen_load_with_block_remap(
        ctx, rhs_node, {rhs_k_block_id: lhs_k_block_id}
    )
    if new_rhs is None:
        return None
    return new_rhs, lhs_k_block_id


def _cute_recodegen_load_with_block_remap(
    ctx: LoweringContext,
    load_node: torch.fx.Node,
    remap: dict[int, int],
) -> ast.AST | None:
    """Re-run the CuTe ``load`` codegen for *load_node* under a block remap."""
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState

    cg = ctx.cg
    if not isinstance(cg, GenerateAST):
        return None
    cute_state = cg.device_function.cute_state

    def env_arg(arg: object) -> object:
        if isinstance(arg, torch.fx.Node):
            return ctx.env[arg]
        if isinstance(arg, (list, tuple)):
            return type(arg)(env_arg(a) for a in arg)
        return arg

    def proxy_arg(arg: object) -> object:
        if isinstance(arg, torch.fx.Node):
            return arg.meta["val"]
        if isinstance(arg, (list, tuple)):
            return type(arg)(proxy_arg(a) for a in arg)
        return arg

    proxy_args = [proxy_arg(a) for a in load_node.args]
    ast_args = [env_arg(a) for a in load_node.args]
    state = CodegenState(
        cg,
        fx_node=load_node,
        env=ctx.env,
        proxy_args=list(proxy_args),
        ast_args=list(ast_args),
    )
    previous = dict(cute_state.matmul_operand_block_remap)
    cute_state.matmul_operand_block_remap = dict(remap)
    load_codegen = load._codegen["cute"]  # pyrefly: ignore[missing-attribute]
    try:
        result = load_codegen(state)
    finally:
        cute_state.matmul_operand_block_remap = previous
    if isinstance(result, ast.AST):
        return result
    return None


# fx view ops whose scalar-fallback lowering is a no-op (the underlying load
# value is unchanged) - peeling them reaches the rhs's underlying ``load``.
_CUTE_RHS_VIEW_PASSTHROUGH_TARGETS = (
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.reshape.default,
    torch.ops.prims.convert_element_type.default,
)


def cute_static_mn_collapse_n_block_id(
    cg: CodegenInterface,
    lhs_node: torch.fx.Node,
    rhs_node: torch.fx.Node,
) -> int | None:
    """Detect the static-M==N collapse of a matmul reduced over N.

    Returns the shared (M and N) block id when a matmul's lhs free axis
    (``lhs.shape[-2]`` = M) and rhs free axis (``rhs.shape[-1]`` = N) resolve to
    the *same* active block id (the collapse: M and N are the same static-size
    dim).  In this case the standard resolver indexes the rhs N axis at the M
    thread index, computing only ``out[m, m]`` (the diagonal).  Returns ``None``
    when M and N are distinct blocks (every ordinary matmul/bmm), so the caller
    keeps its existing diagonal-free behaviour.
    """
    env = CompileEnvironment.current()
    canonical_block_id = getattr(env, "canonical_block_id", lambda block_id: block_id)
    lhs_val = lhs_node.meta.get("val")
    rhs_val = rhs_node.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return None
    if lhs_val.ndim < 2 or rhs_val.ndim < 2:
        return None
    m_block_id = cute_resolve_active_block_id(cg, lhs_val.shape[-2])
    n_block_id = cute_resolve_active_block_id(cg, rhs_val.shape[-1])
    if m_block_id is None or n_block_id is None:
        return None
    if canonical_block_id(m_block_id) != canonical_block_id(n_block_id):
        return None
    # The contraction (K) axis must be a *different* block from the shared M/N
    # block - otherwise this is not a well-formed matmul.
    k_block_id = cute_resolve_active_block_id(cg, lhs_val.shape[-1])
    if k_block_id is not None and canonical_block_id(k_block_id) == canonical_block_id(
        m_block_id
    ):
        return None
    return n_block_id


def cute_underlying_load_node(node: torch.fx.Node) -> torch.fx.Node | None:
    """Peel no-op view ops off *node* to reach its underlying ``load`` node."""
    cur: torch.fx.Node | None = node
    for _ in range(16):
        if cur is None or cur.op != "call_function":
            return None
        if cur.target is load:
            return cur
        if cur.target not in _CUTE_RHS_VIEW_PASSTHROUGH_TARGETS:
            return None
        source = cur.args[0] if cur.args else None
        cur = source if isinstance(source, torch.fx.Node) else None
    return None


def cute_rematerialize_rhs_at_index_override(
    ctx: LoweringContext,
    rhs_node: torch.fx.Node,
    n_block_id: int,
    index_expr: str,
) -> ast.AST | None:
    """Re-lower the rhs of a static-M==N-collapse matmul at a serial N index.

    The rhs's free (N) axis shares a block id with the lhs M axis, so the
    standard load indexes it at the M thread index.  Re-codegen the rhs's
    underlying ``load`` with that block's index overridden to *index_expr* (a
    serial N-loop variable) and masking for it suppressed, so the load reads
    ``rhs[..., n, ...]`` for the loop's current n instead of the diagonal.
    """
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState

    cg = ctx.cg
    if not isinstance(cg, GenerateAST):
        return None
    load_node = cute_underlying_load_node(rhs_node)
    if load_node is None or _cute_matmul_reread_unsafe(cg, load_node):
        return None
    cute_state = cg.device_function.cute_state

    def env_arg(arg: object) -> object:
        if isinstance(arg, torch.fx.Node):
            return ctx.env[arg]
        if isinstance(arg, (list, tuple)):
            return type(arg)(env_arg(a) for a in arg)
        return arg

    def proxy_arg(arg: object) -> object:
        if isinstance(arg, torch.fx.Node):
            return arg.meta["val"]
        if isinstance(arg, (list, tuple)):
            return type(arg)(proxy_arg(a) for a in arg)
        return arg

    proxy_args = [proxy_arg(a) for a in load_node.args]
    ast_args = [env_arg(a) for a in load_node.args]
    state = CodegenState(
        cg,
        fx_node=load_node,
        env=ctx.env,
        proxy_args=list(proxy_args),
        ast_args=list(ast_args),
    )
    previous = dict(cute_state.matmul_operand_index_override)
    cute_state.matmul_operand_index_override = {n_block_id: index_expr}
    load_codegen = load._codegen["cute"]  # pyrefly: ignore[missing-attribute]
    try:
        result = load_codegen(state)
    finally:
        cute_state.matmul_operand_index_override = previous
    if isinstance(result, ast.AST):
        return result
    return None


def _cute_matmul_operand_indices() -> dict[object, tuple[int, int]]:
    """``(lhs_arg_index, rhs_arg_index)`` for each matmul op the CuTe backend
    lowers through the scalar fallback.

    The contraction (K) axis is ``lhs.shape[-1]`` / ``rhs.shape[-2]``; the free
    (output) axes are ``lhs.shape[-2]`` (M) and ``rhs.shape[-1]`` (N).  Includes
    both the aten matmul family and Helion's device-IR ``hl.dot`` / ``dot_scaled``
    ops, which appear *before* aten lowering when the tile strategies are built.
    """
    from ...language import matmul_ops

    return {
        torch.ops.aten.mm.default: (0, 1),
        torch.ops.aten.bmm.default: (0, 1),
        torch.ops.aten.bmm.dtype: (0, 1),
        torch.ops.aten.addmm.default: (1, 2),
        torch.ops.aten.baddbmm.default: (1, 2),
        matmul_ops.dot: (0, 1),
        matmul_ops.dot_scaled: (0, 3),
    }


def _cute_kernel_matmul_operands() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """The fake ``(lhs, rhs)`` operand values of every matmul in the kernel's
    device IR (the ops of ``_cute_matmul_operand_indices``)."""
    from ..host_function import HostFunction
    from ..host_function import NoCurrentFunction

    if not CompileEnvironment.has_current():
        return
    try:
        hf = HostFunction.current()
    except NoCurrentFunction:
        return
    # Guard via getattr so a test double exposing only the public ``device_ir``
    # (without the private ``_device_ir`` backing field) is handled too; the
    # real HostFunction.device_ir property asserts when ``_device_ir`` is None,
    # so we must avoid touching the property in that case.
    if getattr(hf, "_device_ir", "unset") is None:
        return
    device_ir = hf.device_ir
    operand_indices_by_target = _cute_matmul_operand_indices()
    for graph_info in getattr(device_ir, "graphs", ()):
        graph = getattr(graph_info, "graph", None)
        if not isinstance(graph, torch.fx.Graph):
            continue
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            operand_indices = operand_indices_by_target.get(node.target)
            if operand_indices is None:
                continue
            lhs_idx, rhs_idx = operand_indices
            args = node.args
            if len(args) <= max(lhs_idx, rhs_idx):
                continue
            lhs_arg = args[lhs_idx]
            rhs_arg = args[rhs_idx]
            if not isinstance(lhs_arg, Node) or not isinstance(rhs_arg, Node):
                continue
            lhs_val = lhs_arg.meta.get("val")
            rhs_val = rhs_arg.meta.get("val")
            if not isinstance(lhs_val, torch.Tensor) or not isinstance(
                rhs_val, torch.Tensor
            ):
                continue
            if lhs_val.ndim < 1 or rhs_val.ndim < 2:
                continue
            yield lhs_val, rhs_val


def cute_matmul_contraction_block_ids() -> set[int]:
    """Return block ids that are the *contraction* (K) axis of some matmul.

    A matmul's contraction axis is ``lhs.shape[-1]`` (equivalently
    ``rhs.shape[-2]``).  When such an axis is also a reduction block that the
    CuTe backend would otherwise split into ``N`` hardware threads x a synthetic
    per-thread lane loop, the cross-thread matmul reduction only sums the ``N``
    thread lanes - never the synthetic-lane partials - so each contracted dot
    product covers only ``N`` of the ``K`` elements.  The thread-budget planner
    uses this set to give the contraction axis priority for real threads (so the
    already-landed cross-warp shared-memory reduction can sum the full ``K``)
    and pushes the synthetic lanes onto the free / output tile axes instead,
    where a lane loop is correct.

    Returns the *canonical* block ids so callers can compare against either the
    raw or canonical form.
    """
    result: set[int] = set()
    env = CompileEnvironment.current()
    for lhs_val, rhs_val in _cute_kernel_matmul_operands():
        for k_size in (lhs_val.shape[-1], rhs_val.shape[-2]):
            block_id = env.resolve_block_id(k_size)
            if block_id is not None:
                result.add(env.canonical_block_id(block_id))
    return result


def cute_kernel_has_half_precision_matmul() -> bool:
    """Whether some matmul of the kernel has a 16-bit float operand, which a
    warp-MMA plan (the direct grouped-N path) may lower with its own thread
    layout instead of the scalar fallback."""
    return any(
        val.dtype in (torch.float16, torch.bfloat16)
        for operands in _cute_kernel_matmul_operands()
        for val in operands
    )


def cute_outer_accumulator_out_dtype(
    resolved_out_dtype: torch.dtype,
    outer_acc_dtype: torch.dtype | None,
) -> torch.dtype:
    """Return a safe CuTe outer-add result dtype.

    Only adopt the outer accumulator dtype when it exactly matches PyTorch's
    promotion result for `outer_acc + matmul_result`. This preserves mixed-kind
    cases like `int32 + fp16 -> fp16` while still allowing numerically useful
    `bf16/fp16 + fp32 -> fp32`.
    """

    if outer_acc_dtype is None:
        return resolved_out_dtype
    promoted = torch.promote_types(resolved_out_dtype, outer_acc_dtype)
    if promoted == outer_acc_dtype:
        return outer_acc_dtype
    return resolved_out_dtype


@dataclass(frozen=True)
class CuteFoldLoad:
    """A matmul operand traced back to a direct memory load.

    The contraction (K) axis is the load's dimension ``k_position`` and
    ``free_sizes`` are the symbolic sizes of its other dimensions in storage
    order. ``index_offsets`` is the start of each dimension's slice (0 for a
    full slice or a tile), added to the block index or K variable the fold
    re-reads that dimension with. ``scale`` folds in any scalar multipliers
    seen on the way to the load. Any ``permute`` chain is accepted: the fold
    re-reads each dimension through its own block index variable (K through
    the fold's loop variable) at the tensor's own stride, so neither the order
    the operand presents its dimensions in nor the position of K matters
    (``w[:, :]`` as a matmul rhs contracts its leading dimension).
    """

    load_node: torch.fx.Node
    tensor: torch.Tensor
    free_sizes: tuple[int | torch.SymInt, ...]
    k_size: int | torch.SymInt
    k_position: int
    index_offsets: tuple[int, ...]
    scale: float


def _cute_trace_matmul_operand_load(
    cg: CodegenInterface,
    node: torch.fx.Node,
    *,
    contraction_dim: int,
) -> CuteFoldLoad | None:
    """Trace a matmul operand back to a single direct ``load``.

    ``contraction_dim`` is the operand's K axis (``-1`` for the lhs, ``-2`` for
    the rhs). Follows ``mul``-by-scalar (scaling), ``permute`` (tracking which
    source dim the contraction axis comes from), element-type conversions and
    ``_new_var``/placeholder pass-throughs, crossing subgraph boundaries via
    ``placeholder_to_outer_arg``. Returns None when the operand is not a simple
    scaled/permuted direct load.
    """
    from ...language._tracing_ops import _new_var
    from ..device_ir import NodeArgsGraphInfo

    def graph_info_for(graph: torch.fx.Graph) -> object | None:
        for graph_info in getattr(cg, "codegen_graphs", ()):
            if getattr(graph_info, "graph", None) is graph:
                return graph_info
        return None

    operand_val = node.meta.get("val")
    if not isinstance(operand_val, torch.Tensor) or operand_val.ndim < 2:
        return None
    # Source dim of the operand's contraction axis, updated through permutes.
    k_dim = contraction_dim % operand_val.ndim
    scale = 1.0
    current: object = node
    seen: set[int] = set()
    while isinstance(current, torch.fx.Node) and id(current) not in seen:
        seen.add(id(current))
        target = current.target
        if target is load:
            val = current.meta.get("val")
            source = current.args[0]
            source_val = (
                source.meta.get("val") if isinstance(source, torch.fx.Node) else source
            )
            if not isinstance(val, torch.Tensor) or not isinstance(
                source_val, torch.Tensor
            ):
                return None
            if val.ndim < 2 or val.ndim != source_val.ndim:
                return None
            # The re-read rebuilds each address from a block index (or the K
            # variable) plus the dim's slice start; a masked load, a gather and
            # an offset tile index have no such form.
            if len(current.args) > 2 and current.args[2] is not None:
                return None
            if _cute_matmul_reread_unsafe(cg, current):
                return None
            subscript = current.args[1]
            if not isinstance(subscript, (list, tuple)):
                return None
            index_offsets = []
            for position, index in enumerate(subscript):
                offset = _cute_fold_index_offset(current, position, index)
                if offset is None:
                    return None
                index_offsets.append(offset)
            return CuteFoldLoad(
                load_node=current,
                tensor=source_val,
                free_sizes=tuple(
                    size for dim, size in enumerate(val.shape) if dim != k_dim
                ),
                k_size=val.shape[k_dim],
                k_position=k_dim,
                index_offsets=tuple(index_offsets),
                scale=scale,
            )
        if target is torch.ops.aten.mul.Tensor and len(current.args) == 2:
            lhs_arg, rhs_arg = current.args
            if isinstance(lhs_arg, (int, float)) and isinstance(rhs_arg, torch.fx.Node):
                scale *= float(lhs_arg)
                current = rhs_arg
                continue
            if isinstance(rhs_arg, (int, float)) and isinstance(lhs_arg, torch.fx.Node):
                scale *= float(rhs_arg)
                current = lhs_arg
                continue
            return None
        if target is torch.ops.aten.permute.default and len(current.args) == 2:
            order = current.args[1]
            ndim = current.meta["val"].ndim
            if not isinstance(order, (list, tuple)) or len(order) != ndim:
                return None
            # Output dim ``i`` of a permute is input dim ``order[i]``: follow
            # the contraction axis back to the permute's source. Free dims may
            # land anywhere (e.g. ``q[tile, :, :].transpose(0, 1)``) since the
            # fold indexes each by its own block variable.
            k_dim = order[k_dim] % ndim
            current = current.args[0]
            continue
        if target is torch.ops.prims.convert_element_type.default or target is _new_var:
            current = current.args[0]
            continue
        if current.op == "placeholder":
            graph_info = graph_info_for(current.graph)
            if isinstance(graph_info, NodeArgsGraphInfo):
                current = graph_info.placeholder_to_outer_arg(current)
                continue
            return None
        return None
    return None


def _cute_fold_index_offset(
    load_node: torch.fx.Node, position: int, index: object
) -> int | None:
    """The start a fold re-read adds to its index for this subscript entry.

    A step-1 slice starts at its static start; a tile index (its block-size
    symbol) is the block's global index itself.
    """
    from ..indexing_strategy import _get_tile_with_offset_info
    from .memory_ops import _cute_tile_axis_block_id

    if isinstance(index, slice):
        start = _cute_static_int_extent(index.start or 0)
        if index.step not in (None, 1) or start is None or start < 0:
            return None
        return start
    if isinstance(index, torch.fx.Node):
        index_val = index.meta.get("val")
        if (
            isinstance(index_val, torch.SymInt)
            and _cute_tile_axis_block_id(index_val) is not None
            and _get_tile_with_offset_info(index_val, load_node, position) is None
        ):
            return 0
    return None


def cute_has_synthetic_lane_k(
    cg: CodegenInterface,
    k_block_id: int | None,
) -> bool:
    """Whether K is split across a lane loop, independent of a static numel."""
    if k_block_id is None:
        return False
    cg_any = cast("Any", cg)
    loops = cg_any.active_device_loops.get(k_block_id)
    loop_state = loops[-1] if loops else None
    strategy = getattr(loop_state, "strategy", None)
    if strategy is None:
        return False
    lane_var = getattr(strategy, "_synthetic_cute_lane_var", None)
    lane_extent = getattr(strategy, "_synthetic_cute_lane_extent", 1)
    return lane_var is not None and isinstance(lane_extent, int) and lane_extent > 1


def cute_synthetic_lane_k_extent(
    cg: CodegenInterface,
    k_block_id: int | None,
) -> int | None:
    """Return the static K extent when ``k_block_id`` is reduced by a synthetic
    lane loop (i.e. the K axis is split across a Python lane loop rather than
    fully mapped to threads).

    A cross-thread warp reduction over such a K only covers the live-thread
    fraction of K, so a matmul that reduces over it must instead fold the full
    extent itself (see ``emit_cute_synthetic_lane_fold_mm``).

    ``None`` can also mean that the numel is dynamic. Call
    ``cute_has_synthetic_lane_k`` when deciding whether a scalar reduction is
    safe, rather than treating an unknown extent as absence of a lane loop.
    """
    if not cute_has_synthetic_lane_k(cg, k_block_id):
        return None
    assert k_block_id is not None
    env = CompileEnvironment.current()
    numel = env.block_sizes[k_block_id].numel
    return _cute_static_int_extent(numel)


def emit_cute_synthetic_lane_fold_mm(
    ctx: LoweringContext,
    lhs_node: torch.fx.Node,
    rhs_node: torch.fx.Node,
    *,
    k_block_id: int,
    k_extent: int,
    acc: ast.AST | None,
    out_dtype: torch.dtype | None,
    acc_dtype: torch.dtype | None,
    lhs_dtype: torch.dtype,
    rhs_dtype: torch.dtype,
) -> ast.AST | None:
    """Lower a matmul whose K axis is reduced by a synthetic lane loop.

    The default cute scalar fallback reduces the matmul's K across live threads
    only; when K is split into a wrapping per-thread lane loop that warp
    reduction silently drops every lane but the current one. Instead emit a
    self-contained serial loop over the full K extent that re-reads both
    operands directly from their tensors, so each (free-dim) thread computes the
    complete dot product. Returns None when either operand is not a simple
    scaled/permuted direct load that can be re-read this way.
    """
    from ...language.memory_ops import _cute_scalar_pointer_expr

    cg = ctx.cg
    lhs_fold = _cute_trace_matmul_operand_load(cg, lhs_node, contraction_dim=-1)
    rhs_fold = _cute_trace_matmul_operand_load(cg, rhs_node, contraction_dim=-2)
    if lhs_fold is None or rhs_fold is None:
        return None

    def k_size_matches(fold: CuteFoldLoad) -> bool:
        static_k = _cute_static_int_extent(fold.k_size)
        if static_k is not None:
            return static_k == k_extent
        # A full-slice load over the rdim may carry the reduction block's size
        # symbol rather than the static dim; it must then be the K block.
        return cute_resolve_active_block_id(cg, fold.k_size) == k_block_id

    if not (k_size_matches(lhs_fold) and k_size_matches(rhs_fold)):
        return None

    def free_index_and_mask(
        fold: CuteFoldLoad,
    ) -> tuple[list[str], list[str]] | None:
        indices: list[str] = []
        masks: list[str] = []
        for size in fold.free_sizes:
            block_id = cute_resolve_active_block_id(cg, size)
            if block_id is None:
                return None
            index_var = _cute_active_index_var(cg, block_id)
            if index_var is None:
                return None
            indices.append(index_var)
            mask_var = _cute_active_mask_var(cg, block_id)
            if mask_var is not None:
                masks.append(mask_var)
        return indices, masks

    lhs_indexed = free_index_and_mask(lhs_fold)
    rhs_indexed = free_index_and_mask(rhs_fold)
    if lhs_indexed is None or rhs_indexed is None:
        return None
    lhs_indices, lhs_masks = lhs_indexed
    rhs_indices, rhs_masks = rhs_indexed

    backend = CompileEnvironment.current().backend
    reduction_dtype = (
        cute_scalar_reduction_dtype(lhs_dtype, rhs_dtype, acc_dtype or out_dtype)
        or torch.float32
    )
    k_var = cg.device_function.new_var("mm_fold_k", dce=False)

    def load_expr(fold: CuteFoldLoad, indices: list[str]) -> str:
        tensor_name = cg.device_function.tensor_arg(fold.tensor).name
        index_vars = list(indices)
        index_vars.insert(fold.k_position, k_var)
        index_exprs = [
            index_var if offset == 0 else f"({index_var}) + {offset}"
            for index_var, offset in zip(index_vars, fold.index_offsets, strict=True)
        ]
        # The kernel's index type: Int64 once an input passes 2**31 elements.
        return f"{_cute_scalar_pointer_expr(tensor_name, index_exprs)}.load()"

    lhs_load = load_expr(lhs_fold, lhs_indices)
    rhs_load = load_expr(rhs_fold, rhs_indices)
    fold_scale = lhs_fold.scale * rhs_fold.scale
    reduction_str = backend.dtype_str(reduction_dtype)
    term = f"{reduction_str}({lhs_load}) * {reduction_str}({rhs_load})"
    if fold_scale != 1.0:
        term = f"{term} * {reduction_str}({fold_scale!r})"
    mask_vars = list(dict.fromkeys([*lhs_masks, *rhs_masks]))
    acc_var = cg.device_function.new_var("mm_fold_acc")
    cg.add_statement(f"{acc_var} = {reduction_str}(0.0)")
    body = f"{acc_var} = {acc_var} + {term}"
    if mask_vars:
        guard = " and ".join(mask_vars)
        loop = f"for {k_var} in range({k_extent}):\n    if {guard}:\n        {body}"
    else:
        loop = f"for {k_var} in range({k_extent}):\n    {body}"
    cg.add_statement(statement_from_string(loop))

    result: ast.AST = expr_from_string(acc_var)
    if acc is not None:
        base = acc
        if acc_dtype is not None and acc_dtype != reduction_dtype:
            base = cast_ast(base, reduction_dtype)
        result = expr_from_string("{acc} + {prod}", acc=base, prod=result)
        final_dtype = acc_dtype
    else:
        final_dtype = out_dtype
    if final_dtype is not None and final_dtype != reduction_dtype:
        result = cast_ast(result, final_dtype)
    return result


def _cute_active_index_var(cg: CodegenInterface, block_id: int) -> str | None:
    cg_any = cast("Any", cg)
    loops = cg_any.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy.index_var(block_id)
    grid_state = cg_any.current_grid_state
    if grid_state is not None and block_id in grid_state.block_ids:
        return grid_state.strategy.index_var(block_id)
    return None


def _cute_active_mask_var(cg: CodegenInterface, block_id: int) -> str | None:
    cg_any = cast("Any", cg)
    loops = cg_any.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy.mask_var(block_id)
    grid_state = cg_any.current_grid_state
    if grid_state is not None and block_id in grid_state.block_ids:
        return grid_state.strategy.mask_var(block_id)
    return None


def cute_lower_rhs_for_matmul(
    env: Mapping[torch.fx.Node, object],
    lhs: ast.AST | CutePackedAffineLoad | CutePackedTerms,
    rhs_node: torch.fx.Node,
    rhs_fallback: ast.AST | CutePackedTerms,
) -> tuple[ast.AST | CutePackedTerms, tuple[tuple[torch.fx.Node, ...], int] | None]:
    rhs: ast.AST | CutePackedTerms = rhs_fallback
    packed_rhs = None
    if isinstance(lhs, (CutePackedAffineLoad, CutePackedTerms)):
        packed_rhs = match_cute_stack_reshape_rhs(rhs_node)
        if packed_rhs is not None:
            packed_nodes, _ = packed_rhs
            rhs = CutePackedTerms(
                tuple(cast("ast.AST", env[packed_node]) for packed_node in packed_nodes)
            )
    return rhs, packed_rhs
