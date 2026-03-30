from __future__ import annotations

import ast

import sympy
import torch

from ..compile_environment import CompileEnvironment


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
