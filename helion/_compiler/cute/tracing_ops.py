"""CuTe-backend codegen for ops defined in ``helion.language._tracing_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``_tracing_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codegen.simd import constant_repr

from ...language import _decorators
from ...language._tracing_ops import _get_symnode
from ...language._tracing_ops import _if
from ...language._tracing_ops import _mask_to
from ...language._tracing_ops import _val_to_sympy
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..dtype_utils import cast_ast
from ..host_function import HostFunction
from ..variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(_get_symnode, "cute")
def _(state: CodegenState) -> ast.AST:
    # pyrefly: ignore [missing-attribute]
    val = state.fx_node.meta["val"]
    if isinstance(val, int):
        return expr_from_string(str(val))

    assert isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)), val
    sym_expr = _val_to_sympy(val)
    origin_info = HostFunction.current().expr_to_origin.get(sym_expr)
    if origin_info is not None and isinstance(origin_info.origin, BlockSizeOrigin):
        block_size_var = state.device_function.block_size_var(
            origin_info.origin.block_id
        )
        if block_size_var is None:
            return expr_from_string("1")
        return expr_from_string(block_size_var)
    return state.codegen.lift_symnode(
        expr_from_string(state.sympy_expr(sym_expr)),
        sym_expr,
        dce=True,
        prefix="symnode",
    )


@_decorators.codegen(_if, "cute")
def _(state: CodegenState) -> list[object]:
    """Emit dynamic if-conditions for the CuTe DSL backend.

    CuTe DSL forbids referencing a variable after a dynamic if/else when the
    variable is first defined inside the branches. Pre-declare any such output
    in the outer scope before emitting the if so both branches reassign it.
    """
    from ..ast_extension import create
    from ..device_ir import ElseGraphInfo
    from ..device_ir import IfGraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import codegen_call_with_graph

    graph_info = state.get_graph(state.proxy_arg(1))
    assert isinstance(graph_info, IfGraphInfo)
    assert isinstance(state.codegen, GenerateAST)

    test = state.ast_arg(0)
    if_args = state.ast_args[3]
    else_args = state.ast_args[4]
    assert isinstance(if_args, list)
    assert isinstance(else_args, list)
    assert all(isinstance(x, ast.AST) for x in if_args)
    assert all(isinstance(x, ast.AST) for x in else_args)

    # Tag each branch with the dynamic ``_if`` node identity so synthetic
    # ``hl.arange`` axes allocated in mutually-exclusive branches can share a
    # single thread axis (only one branch runs per program instance).
    if_node_id = id(state.fx_node)

    if_body_stmts: list[ast.AST] = []
    with (
        state.codegen.set_statements(if_body_stmts),
        state.codegen.cute_branch_scope(if_node_id, 0),
    ):
        if_outputs = codegen_call_with_graph(
            state.codegen, graph_info.graph, [*if_args]
        )

    assert graph_info.else_branch is not None
    else_graph = state.get_graph(graph_info.else_branch)
    assert isinstance(else_graph, ElseGraphInfo)
    else_body_stmts: list[ast.AST] = []
    with (
        state.codegen.set_statements(else_body_stmts),
        state.codegen.cute_branch_scope(if_node_id, 1),
    ):
        else_outputs = codegen_call_with_graph(
            state.codegen, else_graph.graph, [*else_args]
        )

    # Pre-declare any variable that is first defined inside both branches in the
    # outer scope so CuTe DSL can resolve it after the if/else. The phi pass
    # later renames the else-branch's name to match the if-branch's name, so we
    # use the if-branch output name as the canonical pre-declared name.
    if graph_info.branches_outputs is not None:
        if_output_node = graph_info.graph.find_nodes(op="output")[0]
        if_graph_outputs = cast("tuple[object, ...]", if_output_node.args[0])
        backend = CompileEnvironment.current().backend
        for if_entry, else_entry in graph_info.branches_outputs:
            if not (isinstance(if_entry, int) and isinstance(else_entry, int)):
                continue
            if_name_node = if_outputs[if_entry]
            assert isinstance(if_name_node, ast.Name)
            fx_out = if_graph_outputs[if_entry]
            if not isinstance(fx_out, torch.fx.Node):
                continue
            val = fx_out.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue
            dtype_str = backend.dtype_str(val.dtype)
            state.add_statement(
                statement_from_string(f"{if_name_node.id} = {dtype_str}(0)")
            )

    if not if_body_stmts:
        if_body_stmts.append(ast.Pass())
    if not else_body_stmts:
        else_body_stmts.append(ast.Pass())
    if_ast_node = create(ast.If, test=test, body=if_body_stmts, orelse=else_body_stmts)
    state.add_statement(if_ast_node)

    if_return_names, else_return_names = graph_info.get_branches_return_names(
        state, if_outputs, else_outputs
    )
    return cast(
        "list[object]",
        [expr_from_string(n) for n in if_return_names]
        + [expr_from_string(n) for n in else_return_names],
    )


@_decorators.codegen(_mask_to, "cute")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))

    mask_exprs: list[str] = []
    input_sizes = [*tensor.size()]
    for dim, size in enumerate(input_sizes):
        if (
            index := CompileEnvironment.current().resolve_block_id(size)
        ) is not None and (mask_var := state.codegen.mask_var(index)) is not None:
            expand = state.tile_strategy.expand_str(input_sizes, dim)
            expr = f"({mask_var}{expand})"
            if expr not in mask_exprs:
                mask_exprs.append(expr)
    if not mask_exprs:
        return state.ast_arg(0)
    mask_expr = " and ".join(mask_exprs)
    input_dtype = tensor.dtype
    expr_typed = cast_ast(state.ast_arg(0), input_dtype)
    other_typed = CompileEnvironment.current().backend.cast_ast(
        expr_from_string(constant_repr(other)),
        input_dtype,
    )
    return expr_from_string(
        "({expr} if {mask} else {other})",
        expr=expr_typed,
        mask=expr_from_string(mask_expr),
        other=other_typed,
    )
