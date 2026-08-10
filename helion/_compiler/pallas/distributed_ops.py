"""Pallas lowering for Helion's one-sided communication primitives."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.distributed_ops import start_async_remote_copy
from ...language.distributed_ops import wait_async_remote_copy
from ...language.distributed_ops import wait_recv_async_remote_copy
from ...language.distributed_ops import wait_send_async_remote_copy
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


_PALLAS_OP_META = "_helion_pallas_remote_copy_op"


def _at_expr(index_ast: object, prefix: str) -> tuple[str, dict[str, ast.AST]]:
    assert isinstance(index_ast, (list, tuple))
    placeholders: dict[str, ast.AST] = {}
    parts: list[str] = []
    for position, index in enumerate(index_ast):
        name = f"{prefix}{position}"
        if isinstance(index, int):
            placeholders[name] = expr_from_string(repr(index))
        else:
            assert isinstance(index, ast.AST)
            placeholders[name] = index
        parts.append(f"{{{name}}}")
    return ", ".join(parts), placeholders


def _ref_expr(
    name: str, index_ast: object, prefix: str
) -> tuple[str, dict[str, ast.AST]]:
    index, placeholders = _at_expr(index_ast, prefix)
    if not index:
        return name, placeholders
    return f"{name}.at[{index}]", placeholders


@_decorators.codegen(start_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    src_name = device_fn.tensor_arg(src).name
    dst_name = device_fn.tensor_arg(dst).name
    send_sem = device_fn.register_dma_semaphore(name_hint="remote_send_sem")
    recv_sem = device_fn.register_dma_semaphore(name_hint="remote_recv_sem")
    op_name = device_fn.new_var("remote_copy", dce=False)

    assert state.fx_node is not None
    state.fx_node.meta[_PALLAS_OP_META] = op_name

    src_ref, src_placeholders = _ref_expr(
        src_name, state.ast_args[1], "_remote_src_index"
    )
    dst_ref, dst_placeholders = _ref_expr(
        dst_name, state.ast_args[4], "_remote_dst_index"
    )
    device_id = state.ast_args[2]
    if isinstance(device_id, int):
        device_id = expr_from_string(repr(device_id))
    assert isinstance(device_id, ast.AST)

    state.codegen.add_statement(
        statement_from_string(
            f"{op_name} = pltpu.make_async_remote_copy("
            f"{src_ref}, {dst_ref}, {send_sem}, {recv_sem}, "
            "device_id={device_id}, "
            "device_id_type=pl.DeviceIdType.LOGICAL)",
            device_id=device_id,
            **src_placeholders,
            **dst_placeholders,
        )
    )
    state.codegen.add_statement(statement_from_string(f"{op_name}.start()"))
    return expr_from_string(op_name)


def _paired_op_name(state: CodegenState) -> str:
    assert state.fx_node is not None
    descriptor = state.fx_node.args[0]
    if not isinstance(descriptor, torch.fx.Node):
        raise exc.InternalError(
            RuntimeError("remote-copy wait is not paired with a start operation")
        )
    op_name = descriptor.meta.get(_PALLAS_OP_META)
    if not isinstance(op_name, str):
        raise exc.InternalError(
            RuntimeError("remote-copy wait could not resolve its start operation")
        )
    return op_name


def _emit_wait(state: CodegenState, method: str) -> ast.AST:
    state.codegen.add_statement(
        statement_from_string(f"{_paired_op_name(state)}.{method}()")
    )
    return expr_from_string("None")


@_decorators.codegen(wait_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "wait")


@_decorators.codegen(wait_send_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "wait_send")


@_decorators.codegen(wait_recv_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "wait_recv")
