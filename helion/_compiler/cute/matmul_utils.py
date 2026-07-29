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
        if not CompileEnvironment.current().settings.static_shapes:
            return None
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


def cute_outer_accumulates_result(
    fx_node: torch.fx.Node | None,
    *,
    is_acc_none: bool,
) -> bool:
    return cute_outer_accumulator_node(fx_node, is_acc_none=is_acc_none) is not None


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
    if rhs_node.target is not load:
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
    if load_node is None:
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
    from ..host_function import HostFunction
    from ..host_function import NoCurrentFunction

    if not CompileEnvironment.has_current():
        return set()
    env = CompileEnvironment.current()
    canonical_block_id = getattr(env, "canonical_block_id", lambda block_id: block_id)
    try:
        hf = HostFunction.current()
    except NoCurrentFunction:
        return set()
    # Guard via getattr so a test double exposing only the public ``device_ir``
    # (without the private ``_device_ir`` backing field) is handled too; the
    # real HostFunction.device_ir property asserts when ``_device_ir`` is None,
    # so we must avoid touching the property in that case.
    if getattr(hf, "_device_ir", "unset") is None:
        return set()
    device_ir = hf.device_ir
    operand_indices_by_target = _cute_matmul_operand_indices()
    result: set[int] = set()
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
            for k_size in (lhs_val.shape[-1], rhs_val.shape[-2]):
                block_id = env.resolve_block_id(k_size)
                if block_id is not None:
                    result.add(canonical_block_id(block_id))
    return result


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

    The contraction (K) axis is always the load's last dimension and
    ``free_sizes`` are the symbolic sizes of the load's leading dimensions in
    storage order. ``scale`` folds in any scalar multipliers seen on the way to
    the load. A ``permute`` that only swaps the trailing two dims (turning a
    ``[..., n, k]`` load into the ``[..., k, n]`` view a bmm contracts) is a
    no-op for the underlying load, so it is accepted and ignored.
    """

    load_node: torch.fx.Node
    tensor: torch.Tensor
    free_sizes: tuple[int | torch.SymInt, ...]
    k_size: int | torch.SymInt
    scale: float


def _cute_trace_matmul_operand_load(
    cg: CodegenInterface,
    node: torch.fx.Node,
) -> CuteFoldLoad | None:
    """Trace a matmul operand back to a single direct ``load``.

    Follows ``mul``-by-scalar (scaling), a trailing-dim ``permute`` (the
    transpose a bmm contracts over), element-type conversions and
    ``_new_var``/placeholder pass-throughs, crossing subgraph boundaries via
    ``placeholder_to_outer_arg``. Returns None when the operand is not a simple
    scaled/transposed direct load whose contraction axis is the load's trailing
    dimension.
    """
    from ...language._tracing_ops import _new_var
    from ..device_ir import NodeArgsGraphInfo

    def graph_info_for(graph: torch.fx.Graph) -> object | None:
        for graph_info in getattr(cg, "codegen_graphs", ()):
            if getattr(graph_info, "graph", None) is graph:
                return graph_info
        return None

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
            return CuteFoldLoad(
                load_node=current,
                tensor=source_val,
                free_sizes=tuple(val.shape[: val.ndim - 1]),
                k_size=val.shape[-1],
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
            if not isinstance(order, (list, tuple)):
                return None
            normalized = [o % ndim for o in order]
            # Only a trailing-dim swap (the bmm's K/N transpose) leaves the
            # underlying load's storage order — and thus its trailing K axis —
            # untouched. Anything else would move the contraction axis.
            if normalized != [*range(ndim - 2), ndim - 1, ndim - 2]:
                return None
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
    """
    if k_block_id is None:
        return None
    cg_any = cast("Any", cg)
    loops = cg_any.active_device_loops.get(k_block_id)
    loop_state = loops[-1] if loops else None
    strategy = getattr(loop_state, "strategy", None)
    if strategy is None:
        return None
    lane_var = getattr(strategy, "_synthetic_cute_lane_var", None)
    lane_extent = getattr(strategy, "_synthetic_cute_lane_extent", 1)
    if lane_var is None or not isinstance(lane_extent, int) or lane_extent <= 1:
        return None
    env = CompileEnvironment.current()
    numel = env.block_sizes[k_block_id].numel
    return _cute_static_int_extent(numel)


def emit_cute_synthetic_lane_fold_mm(
    ctx: LoweringContext,
    lhs_node: torch.fx.Node,
    rhs_node: torch.fx.Node,
    *,
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
    scaled/transposed direct load that can be re-read this way.
    """
    cg = ctx.cg
    lhs_fold = _cute_trace_matmul_operand_load(cg, lhs_node)
    rhs_fold = _cute_trace_matmul_operand_load(cg, rhs_node)
    if lhs_fold is None or rhs_fold is None:
        return None
    if _cute_static_int_extent(lhs_fold.k_size) != k_extent:
        return None
    if _cute_static_int_extent(rhs_fold.k_size) != k_extent:
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
    reduction_dtype = acc_dtype or out_dtype or torch.float32
    if _needs_f32_accumulator(lhs_dtype, rhs_dtype):
        reduction_dtype = torch.float32
    k_var = cg.device_function.new_var("mm_fold_k", dce=False)

    def load_expr(fold: CuteFoldLoad, indices: list[str]) -> str:
        tensor_name = cg.device_function.tensor_arg(fold.tensor).name
        terms = [f"{tensor_name}.iterator"]
        ndim = fold.tensor.ndim
        for index_var, dim in zip(indices, range(ndim - 1), strict=True):
            terms.append(
                f"cutlass.Int32({index_var}) "
                f"* cutlass.Int32({tensor_name}.layout.stride[{dim}])"
            )
        terms.append(
            f"cutlass.Int32({k_var}) "
            f"* cutlass.Int32({tensor_name}.layout.stride[{ndim - 1}])"
        )
        return "(" + " + ".join(terms) + ").load()"

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
