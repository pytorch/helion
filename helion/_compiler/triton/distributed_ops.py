"""Triton/NVSHMEM lowering for Helion communication primitives."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.distributed_ops import _REMOTE_COPY_DESCRIPTOR_ID_META
from ...language.distributed_ops import _REMOTE_COPY_DESCRIPTOR_OPS_META
from ...language.distributed_ops import make_async_remote_copy
from ...language.distributed_ops import remote_barrier
from ...language.distributed_ops import start_async_remote_copy_descriptor
from ...language.distributed_ops import wait_async_remote_copy
from ...language.distributed_ops import wait_recv_async_remote_copy
from ...language.distributed_ops import wait_send_async_remote_copy
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..inductor_lowering import CodegenState


_FLAT_PROGRAM_ID = (
    "tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)"
    " + tl.program_id(1) * tl.num_programs(0) + tl.program_id(0)"
)
_NUM_PROGRAMS = "tl.num_programs(2) * tl.num_programs(1) * tl.num_programs(0)"
# Values from NVSHMEM's nvshmemx_signal_op_t and nvshmem_cmp_t.
_NVSHMEM_SIGNAL_ADD = 10
_NVSHMEM_CMP_NE = 1


@dataclass
class _TritonRemoteCopyInfo:
    start_statement: ast.stmt
    signal: str | None


@_decorators.codegen(remote_barrier, "triton")
def _(state: CodegenState) -> ast.AST:
    from ..compile_environment import CompileEnvironment

    device_fn = state.device_function
    device_fn.requires_nvshmem = True
    CompileEnvironment.current().has_barrier = True
    state.codegen.add_statement(statement_from_string("nvshmem.barrier_all()"))
    return expr_from_string("None")


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
    descriptor_ops = node.meta.get(_REMOTE_COPY_DESCRIPTOR_OPS_META, set())
    assert isinstance(descriptor_ops, set)
    return bool(descriptor_ops & {wait_async_remote_copy, wait_recv_async_remote_copy})


def _reserve_signal_slot(
    device_fn: DeviceFunction, dst: torch.Tensor
) -> tuple[str, int]:
    signal_arg = device_fn.triton_remote_copy_signal_arg
    if signal_arg is None:
        try:
            dst_host_str = device_fn.tensor_arg(dst).host_str()
        except (KeyError, RuntimeError) as error:
            raise exc.BackendUnsupported(
                "triton",
                "remote-copy receive completion requires a host-provided "
                "symmetric destination tensor",
            ) from error
        signal_arg = device_fn.new_var("remote_copy_signal")
        device_fn.wrapper_only_params.append(signal_arg)
        device_fn.triton_remote_copy_signal_arg = signal_arg
        device_fn.triton_remote_copy_signal_dst = dst_host_str

    slot = device_fn.triton_remote_copy_signal_slots
    device_fn.triton_remote_copy_signal_slots += 1
    return signal_arg, slot


def _prepare_remote_copy(state: CodegenState) -> ast.AST:
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    device_fn.requires_nvshmem = True
    device_fn.requires_remote_copy = True
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
    descriptor_id = state.fx_node.meta.get(_REMOTE_COPY_DESCRIPTOR_ID_META)
    if not isinstance(descriptor_id, int):
        raise exc.InternalError(RuntimeError("remote-copy descriptor has no ID"))
    signal_name = None
    if _has_receive_wait(state.fx_node):
        signal_arg, signal_slot = _reserve_signal_slot(device_fn, dst)
        signal_name = device_fn.new_var("remote_signal", dce=False)
        state.codegen.add_statement(
            statement_from_string(
                f"{signal_name} = {signal_arg} + "
                f"({signal_slot}) * ({_NUM_PROGRAMS}) + ({_FLAT_PROGRAM_ID})"
            )
        )
        start_statement = statement_from_string(
            "nvshmem.putmem_signal_block("
            f"{dst_ptr}, {src_ptr}, "
            f"tl.cast(({numel}) * {src.element_size()}, tl.int64), "
            f"{signal_name}, tl.cast(1, tl.uint64), "
            f"{_NVSHMEM_SIGNAL_ADD}, {{device_id}})",
            device_id=device_id,
            **src_placeholders,
            **dst_placeholders,
        )
    else:
        start_statement = statement_from_string(
            f"nvshmem.put({dst_ptr}, {src_ptr}, {numel}, {{device_id}})",
            device_id=device_id,
            **src_placeholders,
            **dst_placeholders,
        )
    device_fn.remote_copy_descriptors[descriptor_id] = _TritonRemoteCopyInfo(
        start_statement=start_statement,
        signal=signal_name,
    )
    return expr_from_string("None")


@_decorators.codegen(make_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _prepare_remote_copy(state)


def _paired_copy_info(state: CodegenState) -> _TritonRemoteCopyInfo:
    descriptor_id = state.proxy_arg(0)
    if not isinstance(descriptor_id, int):
        raise exc.InternalError(
            RuntimeError("remote-copy lifecycle operation has no descriptor ID")
        )
    info = state.device_function.remote_copy_descriptors.get(descriptor_id)
    if not isinstance(info, _TritonRemoteCopyInfo):
        raise exc.InternalError(
            RuntimeError(
                "remote-copy lifecycle operation could not resolve its descriptor"
            )
        )
    return info


def _emit_statements(
    state: CodegenState,
    statements: list[ast.stmt],
) -> ast.AST:
    for statement in statements:
        state.codegen.add_statement(statement)
    return expr_from_string("None")


@_decorators.codegen(start_async_remote_copy_descriptor, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_statements(state, [_paired_copy_info(state).start_statement])


def _paired_signal(state: CodegenState) -> str:
    signal = _paired_copy_info(state).signal
    if not isinstance(signal, str):
        raise exc.InternalError(
            RuntimeError("remote-copy receive wait could not resolve its signal")
        )
    return signal


def _receive_wait_statements(state: CodegenState) -> list[ast.stmt]:
    signal = _paired_signal(state)
    return [
        statement_from_string(
            f"nvshmem.signal_wait_until({signal}, {_NVSHMEM_CMP_NE}, 0)"
        ),
        statement_from_string(
            f"tl.atomic_add({signal}, -1, sem='relaxed', scope='sys')"
        ),
    ]


@_decorators.codegen(wait_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_statements(
        state,
        [statement_from_string("nvshmem.quiet()"), *_receive_wait_statements(state)],
    )


@_decorators.codegen(wait_send_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_statements(state, [statement_from_string("nvshmem.quiet()")])


@_decorators.codegen(wait_recv_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_statements(state, _receive_wait_statements(state))
