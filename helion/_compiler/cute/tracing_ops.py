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
from ...language._tracing_ops import _if
from ...language._tracing_ops import _mask_select
from ...language._tracing_ops import _mask_to
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_BLOCK_UNIFORM_ATTR
from ..compile_environment import CompileEnvironment
from ..dtype_utils import cast_ast
from .block_uniform import block_uniform
from .synthetic_axes import branch_scope

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection

    from ..inductor_lowering import CodegenState


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
    assert graph_info.else_branch is not None
    else_graph = state.get_graph(graph_info.else_branch)
    assert isinstance(else_graph, ElseGraphInfo)

    # As in ``IfGraphInfo.codegen``, a condition that depends only on block
    # sizes is a constant for this config.  Emit only the live branch, inline:
    # CuTe DSL stages even ``if False:`` bodies, so the dead branch (whose
    # shapes and tile plans need not fit this config) must not be emitted.
    constexpr_test = state.device_function.evaluate_constexpr_condition(
        state.proxy_arg(0)
    )
    if constexpr_test is not None:
        live_graph, live_args = (
            (graph_info.graph, if_args)
            if constexpr_test
            else (else_graph.graph, else_args)
        )
        live_outputs = codegen_call_with_graph(state.codegen, live_graph, [*live_args])
        # The ``_phi`` nodes merging the two sides see the live names on both.
        live_return_names = graph_info.get_branch_return_names(
            state, live_outputs, 0 if constexpr_test else 1
        )
        return [expr_from_string(n) for n in live_return_names * 2]

    # Tag each branch with the dynamic ``_if`` node identity so synthetic
    # ``hl.arange`` axes allocated in mutually-exclusive branches can share a
    # single thread axis (only one branch runs per program instance).
    assert state.fx_node is not None
    if_node_id = id(state.fx_node)
    # A condition every thread of the CTA evaluates alike sends the whole
    # block down one side, where a block-wide barrier is convergent.
    uniform = block_uniform(state.fx_node.args[0])

    if_body_stmts: list[ast.AST] = []
    with (
        state.codegen.set_statements(if_body_stmts),
        branch_scope(state.codegen, if_node_id, 0, divergent=not uniform),
    ):
        if_outputs = codegen_call_with_graph(
            state.codegen, graph_info.graph, [*if_args]
        )
        if_outputs = graph_info.copy_outer_outputs(state, if_outputs, 0)

    else_body_stmts: list[ast.AST] = []
    with (
        state.codegen.set_statements(else_body_stmts),
        branch_scope(state.codegen, if_node_id, 1, divergent=not uniform),
    ):
        else_outputs = codegen_call_with_graph(
            state.codegen, else_graph.graph, [*else_args]
        )
        else_outputs = graph_info.copy_outer_outputs(state, else_outputs, 1)

    # CuTe DSL requires a variable assigned in a branch to keep the type it
    # had before the if, but emitted values keep their DSL type (index math
    # on Int64 shape arguments, fp32 math on 16-bit tensors), which can
    # differ from the FX dtype.  A variable first defined inside both
    # branches is pre-declared with its FX dtype in the outer scope (so CuTe
    # DSL can also resolve it after the if/else) and both branches cast to
    # that dtype; the phi pass later renames the else-branch's name to match
    # the if-branch's name, so the if-branch name is the canonical one.  A
    # variable one branch leaves unchanged keeps its type, which is only
    # known at trace time, so the other branch casts its new value to the
    # type of the value from before the if.  Each cast binds a fresh name:
    # later passes read ``x = f(x)`` as a value carried across lanes.
    assert graph_info.branches_outputs is not None
    backend = CompileEnvironment.current().backend
    graph_outputs = [
        cast("tuple[object, ...]", graph.find_nodes(op="output")[0].args[0])
        for graph in (graph_info.graph, else_graph.graph)
    ]
    outputs = ([*if_outputs], [*else_outputs])
    statements = (if_body_stmts, else_body_stmts)
    if_return_names, else_return_names = graph_info.get_branches_return_names(
        state, if_outputs, else_outputs
    )
    for entries, if_name, else_name in zip(
        graph_info.branches_outputs, if_return_names, else_return_names, strict=True
    ):
        indices = {
            branch: entry
            for branch, entry in enumerate(entries)
            if isinstance(entry, int)
        }
        if not indices:
            continue
        branch, index = next(iter(indices.items()))
        fx_out = graph_outputs[branch][index]
        val = fx_out.meta.get("val") if isinstance(fx_out, torch.fx.Node) else None
        if not isinstance(val, torch.Tensor):
            continue
        if len(indices) == 2:
            values = {
                side: backend.cast_ast(cast("ast.AST", outputs[side][entry]), val.dtype)
                for side, entry in indices.items()
            }
        else:
            live_in = (if_name, else_name)[1 - branch]
            before = state.device_function.new_var(f"{live_in}_before_if")
            state.add_statement(statement_from_string(f"{before} = {live_in}"))
            values = {
                branch: expr_from_string(
                    f"_cute_join_cast({{value}}, {before})",
                    value=cast("ast.AST", outputs[branch][index]),
                )
            }
        for branch, value in values.items():
            joined = state.device_function.new_var((if_name, else_name)[branch])
            with state.codegen.set_statements(statements[branch]):
                state.codegen.add_statement(
                    statement_from_string(f"{joined} = {{value}}", value=value)
                )
            outputs[branch][indices[branch]] = expr_from_string(joined)
        if len(indices) == 2:
            joined = cast("ast.Name", outputs[0][indices[0]]).id
            dtype_str = backend.dtype_str(val.dtype)
            state.add_statement(statement_from_string(f"{joined} = {dtype_str}(0)"))
    if_return_names, else_return_names = graph_info.get_branches_return_names(
        state, *outputs
    )

    if not if_body_stmts:
        if_body_stmts.append(ast.Pass())
    if not else_body_stmts:
        else_body_stmts.append(ast.Pass())
    if_ast_node = create(ast.If, test=test, body=if_body_stmts, orelse=else_body_stmts)
    if uniform:
        setattr(if_ast_node, HELION_BLOCK_UNIFORM_ATTR, True)
    state.add_statement(if_ast_node)

    return cast(
        "list[object]",
        [expr_from_string(n) for n in if_return_names]
        + [expr_from_string(n) for n in else_return_names],
    )


def _tile_mask_expr(
    state: CodegenState,
    tensor: torch.Tensor,
    block_ids: Collection[int] | None = None,
) -> str | None:
    """The mask of the active tiles over ``tensor``'s dims (of the canonical
    ``block_ids`` only, if given), or None when no such dim has a masked
    tile."""
    env = CompileEnvironment.current()
    mask_exprs: list[str] = []
    input_sizes = [*tensor.size()]
    for dim, size in enumerate(input_sizes):
        if (
            (index := env.resolve_block_id(size)) is not None
            and (block_ids is None or env.canonical_block_id(index) in block_ids)
            and (mask_var := state.codegen.mask_var(index)) is not None
        ):
            expand = state.tile_strategy.expand_str(input_sizes, dim)
            expr = f"({mask_var}{expand})"
            if expr not in mask_exprs:
                mask_exprs.append(expr)
    if not mask_exprs:
        return None
    return " and ".join(mask_exprs)


def _masked_select(
    state: CodegenState,
    tensor: torch.Tensor,
    other: Callable[[torch.dtype], ast.AST],
    block_ids: Collection[int] | None = None,
) -> ast.AST:
    """``(tensor if mask else other)`` for the active tiles' mask (of
    ``block_ids`` only, if given), at ``tensor``'s dtype; ``other(dtype)``
    renders the fallback at the dtype the select runs in.  ``tensor`` itself
    where no such tile is masked."""
    mask_expr = _tile_mask_expr(state, tensor, block_ids)
    if mask_expr is None:
        return state.ast_arg(0)
    input_dtype = tensor.dtype
    # CuTe cannot select between float8 scalars; they convert to float32 and
    # back exactly.
    select_dtype = (
        torch.float32
        if input_dtype.itemsize == 1 and input_dtype.is_floating_point
        else input_dtype
    )
    result = expr_from_string(
        "({expr} if {mask} else {other})",
        expr=cast_ast(state.ast_arg(0), select_dtype),
        mask=expr_from_string(mask_expr),
        other=other(select_dtype),
    )
    if select_dtype != input_dtype:
        result = cast_ast(result, input_dtype)
    return result


def _constant_other(value: float | bool) -> Callable[[torch.dtype], ast.AST]:
    """A masked select's constant ``value`` in the select's dtype."""
    return lambda dtype: CompileEnvironment.current().backend.cast_ast(
        expr_from_string(constant_repr(value)), dtype
    )


@_decorators.codegen(_mask_to, "cute")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))
    return _masked_select(state, tensor, _constant_other(other))


@_decorators.codegen(_mask_select, "cute")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    block_ids = state.proxy_arg(2)
    assert isinstance(block_ids, tuple)
    other = state.proxy_arg(1)
    if isinstance(other, (int, float, bool)):
        return _masked_select(state, tensor, _constant_other(other), block_ids)
    return _masked_select(
        state, tensor, lambda dtype: cast_ast(state.ast_arg(1), dtype), block_ids
    )
