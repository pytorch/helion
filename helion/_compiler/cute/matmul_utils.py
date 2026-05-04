from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
import torch
from torch.fx.node import Node

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
    from collections.abc import Mapping

    from ..aten_lowering import LoweringContext
    from ..helper_function import CodegenInterface


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
        size_hint = getattr(env, "size_hint", None)
        if not callable(size_hint):
            return None
        hinted_size = size_hint(size)
        return hinted_size if isinstance(hinted_size, int) else None

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

    def hinted_int(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        size_hint = getattr(CompileEnvironment.current(), "size_hint", None)
        if callable(size_hint):
            hinted = size_hint(value)
            if isinstance(hinted, int):
                return hinted
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
        hinted_int(rhs_val.shape[1]) if isinstance(rhs_val, torch.Tensor) else None
    )
    load_plan = analyze_direct_grouped_n_loads(
        lhs_node,
        rhs_node,
        k_extent=k_extent,
        n_extent=n_extent,
    )
    if load_plan is None:
        return None
    lhs_k_offset_int = load_plan.lhs_k_offset
    rhs_k_offset_int = load_plan.rhs_k_offset
    rhs_n_offset_int = load_plan.rhs_n_offset

    m_block_id = cute_resolve_active_block_id(ctx.cg, lhs_node.meta["val"].shape[0])
    n_block_id = cute_resolve_active_block_id(ctx.cg, rhs_node.meta["val"].shape[-1])
    if m_block_id is None or n_block_id is None:
        return None
    m_index = active_index_var(m_block_id)
    n_index = active_index_var(n_block_id)
    if m_index is None or n_index is None:
        return None
    m_index_str = m_index
    n_index_str = n_index
    m_mask = active_mask_var(m_block_id)
    n_mask = active_mask_var(n_block_id)
    reduction_dtype = out_dtype
    if _needs_f32_accumulator(lhs_dtype, rhs_dtype):
        reduction_dtype = torch.float32
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
            m_index_str,
            add_offset(k_expr, lhs_k_offset_int),
            source_dtype=lhs_dtype,
            mask_expr=m_mask,
        )
        rhs_value = masked_scalar_load(
            rhs_tensor,
            add_offset(k_expr, rhs_k_offset_int),
            add_offset(n_index_str, rhs_n_offset_int),
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

    def hinted_int(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        size_hint = getattr(CompileEnvironment.current(), "size_hint", None)
        if callable(size_hint):
            hinted = size_hint(value)
            if isinstance(hinted, int):
                return hinted
        return None

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
            extent = hinted_int(fake.shape[0])
            if extent is None:
                return None
            if required_extent is not None and extent != required_extent:
                return None
            start = hinted_int(index.kwargs.get("start", 0))
            return 0 if start is None else start
        if not isinstance(index, slice):
            return None
        if index.step not in (None, 1):
            return None
        start = hinted_int(index.start) or 0
        stop = hinted_int(index.stop)
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


def cute_outer_accumulates_result(
    fx_node: torch.fx.Node | None,
    *,
    is_acc_none: bool,
    add_targets: tuple[object, ...] = (torch.ops.aten.add.Tensor,),
) -> bool:
    return (
        cute_outer_accumulator_node(
            fx_node,
            is_acc_none=is_acc_none,
            add_targets=add_targets,
        )
        is not None
    )


def cute_outer_accumulator_node(
    fx_node: torch.fx.Node | None,
    *,
    is_acc_none: bool,
    add_targets: tuple[object, ...] = (torch.ops.aten.add.Tensor,),
) -> torch.fx.Node | None:
    if not is_acc_none or fx_node is None:
        return None
    users = [user for user in fx_node.users if isinstance(user, torch.fx.Node)]
    if len(users) != 1:
        return None
    (user,) = users
    if user.target not in add_targets or len(user.args) < 2:
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

                def is_target_name(expr: ast.expr) -> bool:
                    return isinstance(expr, ast.Name) and expr.id == target_name

                if is_target_name(value.left) or is_target_name(value.right):
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
    add_targets: tuple[object, ...] = (torch.ops.aten.add.Tensor,),
) -> torch.dtype | None:
    outer_acc = cute_outer_accumulator_node(
        fx_node,
        is_acc_none=is_acc_none,
        add_targets=add_targets,
    )
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
) -> int | None:
    env = CompileEnvironment.current()
    canonical_block_id = getattr(env, "canonical_block_id", lambda block_id: block_id)
    lhs_k_block_id = cute_resolve_active_block_id(cg, lhs_k_size)
    rhs_k_block_id = cute_resolve_active_block_id(cg, rhs_k_size)
    if lhs_k_block_id is None or rhs_k_block_id is None:
        return None
    if canonical_block_id(lhs_k_block_id) != canonical_block_id(rhs_k_block_id):
        return None
    rhs_n_block_id = cute_resolve_active_block_id(cg, rhs_n_size)
    if rhs_n_block_id is not None and canonical_block_id(
        rhs_n_block_id
    ) == canonical_block_id(lhs_k_block_id):
        return None
    return lhs_k_block_id


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


def cute_lower_rhs_for_matmul(
    env: Mapping[torch.fx.Node, object],
    lhs: ast.AST | CutePackedAffineLoad,
    rhs_node: torch.fx.Node,
    rhs_fallback: ast.AST,
) -> tuple[ast.AST | CutePackedTerms, tuple[tuple[torch.fx.Node, ...], int] | None]:
    rhs: ast.AST | CutePackedTerms = rhs_fallback
    packed_rhs = None
    if isinstance(lhs, CutePackedAffineLoad):
        packed_rhs = match_cute_stack_reshape_rhs(rhs_node)
        if packed_rhs is not None:
            packed_nodes, _ = packed_rhs
            rhs = CutePackedTerms(
                tuple(cast("ast.AST", env[packed_node]) for packed_node in packed_nodes)
            )
    return rhs, packed_rhs
