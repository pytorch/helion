"""Triton/NVSHMEM lowering for Helion communication primitives."""

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
    from ..device_function import DeviceFunction
    from ..inductor_lowering import CodegenState


_TRITON_SIGNAL_META = "_helion_triton_remote_copy_signal"


def _index_expr(
    index_ast: object,
    tensor: torch.Tensor,
    device_fn: DeviceFunction,
    prefix: str,
) -> tuple[str, dict[str, ast.AST]]:
    assert isinstance(index_ast, (list, tuple))
    placeholders: dict[str, ast.AST] = {}
    terms: list[str] = []
    for position, index in enumerate(index_ast):
        if isinstance(index, int):
            index_expr = repr(index)
        else:
            assert isinstance(index, ast.AST)
            name = f"{prefix}{position}"
            placeholders[name] = index
            index_expr = f"{{{name}}}"
        stride = device_fn.tensor_stride(tensor, position).name
        terms.append(f"({index_expr}) * {stride}")
    return " + ".join(terms) or "0", placeholders


def _region_ptr(
    tensor: torch.Tensor,
    index_ast: object,
    device_fn: DeviceFunction,
    prefix: str,
) -> tuple[str, dict[str, ast.AST], str]:
    base = device_fn.tensor_arg(tensor).name
    offset, placeholders = _index_expr(index_ast, tensor, device_fn, prefix)
    assert isinstance(index_ast, (list, tuple))
    suffix_sizes = [
        device_fn.tensor_size(tensor, dim).name
        for dim in range(len(index_ast), tensor.ndim)
    ]
    numel = " * ".join(suffix_sizes) or "1"
    return f"{base} + {offset}", placeholders, numel


def _has_receive_wait(node: torch.fx.Node) -> bool:
    receive_waits = {wait_async_remote_copy, wait_recv_async_remote_copy}
    return any(
        user.op == "call_function" and user.target in receive_waits
        for user in node.users
    )


@_decorators.codegen(start_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    signal = state.proxy_arg(5)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    device_fn.requires_nvshmem = True
    src_ptr, src_placeholders, numel = _region_ptr(
        src, state.ast_args[1], device_fn, "_remote_src_index"
    )
    dst_ptr, dst_placeholders, _ = _region_ptr(
        dst, state.ast_args[4], device_fn, "_remote_dst_index"
    )
    device_id = state.ast_args[2]
    if isinstance(device_id, int):
        device_id = expr_from_string(repr(device_id))
    assert isinstance(device_id, ast.AST)

    assert state.fx_node is not None
    if _has_receive_wait(state.fx_node):
        if not isinstance(signal, torch.Tensor):
            raise exc.BackendUnsupported(
                "triton",
                "remote-copy wait()/wait_recv() without a symmetric int64 "
                "signal tensor and scalar signal_index",
            )
        signal_ptr, signal_placeholders, _ = _region_ptr(
            signal, state.ast_args[6], device_fn, "_remote_signal_index"
        )
        signal_name = device_fn.new_var("remote_signal", dce=False)
        state.fx_node.meta[_TRITON_SIGNAL_META] = signal_name
        state.codegen.add_statement(
            statement_from_string(
                f"{signal_name} = {signal_ptr}", **signal_placeholders
            )
        )
        state.codegen.add_statement(
            statement_from_string(
                "nvshmem.putmem_signal_block("
                f"{dst_ptr}, {src_ptr}, ({numel}) * {src.element_size()}, "
                f"{signal_name}, 1, 0, {{device_id}})",
                device_id=device_id,
                **src_placeholders,
                **dst_placeholders,
            )
        )
    else:
        state.codegen.add_statement(
            statement_from_string(
                f"nvshmem.put({dst_ptr}, {src_ptr}, {numel}, {{device_id}})",
                device_id=device_id,
                **src_placeholders,
                **dst_placeholders,
            )
        )
    return expr_from_string("None")


def _paired_signal(state: CodegenState) -> str:
    assert state.fx_node is not None
    descriptor = state.fx_node.args[0]
    if not isinstance(descriptor, torch.fx.Node):
        raise exc.InternalError(
            RuntimeError("remote-copy wait is not paired with a start operation")
        )
    signal = descriptor.meta.get(_TRITON_SIGNAL_META)
    if not isinstance(signal, str):
        raise exc.InternalError(
            RuntimeError("remote-copy receive wait could not resolve its signal")
        )
    return signal


def _emit_receive_wait(state: CodegenState) -> ast.AST:
    signal = _paired_signal(state)
    state.codegen.add_statement(
        statement_from_string(f"nvshmem.signal_wait_until({signal}, 0, 1)")
    )
    state.codegen.add_statement(statement_from_string(f"tl.store({signal}, 0)"))
    return expr_from_string("None")


@_decorators.codegen(wait_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    state.codegen.add_statement(statement_from_string("nvshmem.quiet()"))
    return _emit_receive_wait(state)


@_decorators.codegen(wait_send_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    state.codegen.add_statement(statement_from_string("nvshmem.quiet()"))
    return expr_from_string("None")


@_decorators.codegen(wait_recv_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_receive_wait(state)
