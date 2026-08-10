"""Pallas lowering for Helion's one-sided communication primitives."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import sympy
import torch

from ... import exc
from ...language import _decorators
from ...language.distributed_ops import make_async_remote_copy
from ...language.distributed_ops import remote_barrier
from ...language.distributed_ops import start_async_remote_copy
from ...language.distributed_ops import start_async_remote_copy_descriptor
from ...language.distributed_ops import start_async_remote_copy_descriptor_if
from ...language.distributed_ops import wait_async_remote_copy
from ...language.distributed_ops import wait_async_remote_copy_if
from ...language.distributed_ops import wait_recv_async_remote_copy
from ...language.distributed_ops import wait_recv_async_remote_copy_if
from ...language.distributed_ops import wait_send_async_remote_copy
from ...language.distributed_ops import wait_send_async_remote_copy_if
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from . import codegen as pallas_codegen
from .plan_tiling import REMOTE_DST_INDEXING_PATTERNS
from .plan_tiling import REMOTE_SRC_INDEXING_PATTERNS

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


_PALLAS_OP_META = "_helion_pallas_remote_copy_op"
_PALLAS_SRC_MATERIALIZATION_META = "_helion_pallas_remote_copy_src_materialization"


def _fori_semaphore_slot(
    state: CodegenState,
) -> tuple[tuple[int, ...], ast.AST | None]:
    """Give each iteration of a one-dimensional fori loop its own semaphore.

    Balanced remote pushes can arrive out of order when the peer changes between
    iterations. Reusing one receive semaphore would let a later notification
    satisfy an earlier wait, making the earlier destination unsafe to read.
    """
    from ..compile_environment import CompileEnvironment
    from ..tile_strategy import ForiLoopState

    active_loops = {
        id(loop): loop
        for loops in state.codegen.active_device_loops.values()
        for loop in loops
        if isinstance(loop, ForiLoopState)
    }
    if len(active_loops) != 1:
        return (), None
    loop = next(iter(active_loops.values()))
    if len(loop.block_ids) != 1:
        return (), None

    block_id = loop.block_ids[0]
    info = loop.block_id_to_info.get(block_id)
    block_size = state.device_function.resolved_block_size(block_id)
    if info is None or info.end_expr is None or not isinstance(block_size, int):
        return (), None

    env = CompileEnvironment.current()
    begin_expr = sympy.Integer(0) if info.begin_expr is None else info.begin_expr
    begin_expr = env.specialize_expr(begin_expr)
    end_expr = env.specialize_expr(info.end_expr)
    if begin_expr.free_symbols or end_expr.free_symbols:
        return (), None
    trip_count = max(
        1, (int(end_expr) - int(begin_expr) + block_size - 1) // block_size
    )
    if trip_count == 1:
        return (), None
    return (trip_count,), expr_from_string(loop.loop_var_name)


@_decorators.codegen(remote_barrier, "pallas")
def _(state: CodegenState) -> ast.AST:
    proxy_device_ids = state.proxy_arg(0)
    ast_device_ids = state.ast_args[0]
    device_ids: list[int | ast.AST]
    if isinstance(proxy_device_ids, list):
        assert isinstance(ast_device_ids, list)
        device_ids = ast_device_ids
    elif isinstance(proxy_device_ids, torch.Tensor):
        assert isinstance(ast_device_ids, ast.AST)
        if proxy_device_ids.ndim == 0:
            device_ids = [ast_device_ids]
        elif proxy_device_ids.ndim == 1 and isinstance(proxy_device_ids.shape[0], int):
            device_ids = [
                expr_from_string(f"{{device_ids}}[{index}]", device_ids=ast_device_ids)
                for index in range(proxy_device_ids.shape[0])
            ]
        else:
            raise exc.TypeInferenceError(
                "remote_barrier expects a scalar or statically sized 1-D peer tensor"
            )
    elif isinstance(proxy_device_ids, int):
        device_ids = [proxy_device_ids]
    else:
        raise exc.TypeInferenceError(
            "remote_barrier expects a logical peer, a 1-D peer tensor, or a list"
        )

    if not device_ids:
        return expr_from_string("None")

    device_fn = state.device_function
    device_fn.requires_collective_id = True
    barrier = device_fn.new_var("remote_barrier", dce=False)
    state.codegen.add_statement(
        statement_from_string(f"{barrier} = pltpu.get_barrier_semaphore()")
    )
    peers: list[ast.AST] = []
    for device_id in device_ids:
        if isinstance(device_id, int):
            device_id = expr_from_string(repr(device_id))
        assert isinstance(device_id, ast.AST)
        peers.append(device_id)
        state.codegen.add_statement(
            statement_from_string(
                f"pltpu.semaphore_signal({barrier}, inc=1, "
                "device_id={device_id}, "
                "device_id_type=pl.DeviceIdType.LOGICAL)",
                device_id=device_id,
            )
        )
    state.codegen.add_statement(
        statement_from_string(f"pltpu.semaphore_wait({barrier}, {len(peers)})")
    )

    second_barrier_fn = device_fn.new_var("remote_barrier_phase_2", dce=False)
    placeholders = {f"device_id_{index}": peer for index, peer in enumerate(peers)}
    body = [f"def {second_barrier_fn}(second_barrier):"]
    for index in range(len(peers)):
        body.extend(
            (
                "    pltpu.semaphore_signal(",
                "        second_barrier, inc=1,",
                f"        device_id={{device_id_{index}}},",
                "        device_id_type=pl.DeviceIdType.LOGICAL,",
                "    )",
            )
        )
    body.append(f"    pltpu.semaphore_wait(second_barrier, {len(peers)})")
    state.codegen.add_statement(statement_from_string("\n".join(body), **placeholders))
    state.codegen.add_statement(
        statement_from_string(
            f"pl.run_scoped({second_barrier_fn}, pltpu.SemaphoreType.REGULAR)"
        )
    )
    return expr_from_string("None")


def _direct_ref_expr(tensor: ast.AST, ast_index: list[object]) -> ast.AST:
    if not ast_index:
        return tensor
    placeholders: dict[str, ast.AST] = {"tensor": tensor}
    parts = []
    for index, value in enumerate(ast_index):
        if isinstance(value, int):
            value = expr_from_string(repr(value))
        assert isinstance(value, ast.AST)
        key = f"index_{index}"
        placeholders[key] = value
        parts.append(f"{{{key}}}")
    return expr_from_string(f"{{tensor}}.at[{', '.join(parts)}]", **placeholders)


def _remote_ref_expr(
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_ast: object,
    proxy_index: object,
    ast_index: object,
    metadata_key: str,
    *,
    materialize_device_value: bool = False,
) -> ast.AST:
    """Resolve a logical remote-copy region against the active VMEM block."""
    assert isinstance(proxy_index, (list, tuple))
    assert isinstance(ast_index, list)
    try:
        name = state.device_function.tensor_arg(tensor).name
    except KeyError:
        if not materialize_device_value:
            raise exc.TypeInferenceError(
                "Pallas remote-copy destinations must be kernel tensor arguments"
            ) from None
        # A computed tile is a JAX value, while TPU DMA requires a Ref. Allocate
        # one reusable VMEM slot now, but defer populating it until descriptor
        # start so a preceding wait can safely release the previous transfer.
        assert isinstance(tensor_ast, ast.AST)
        name = state.device_function.register_scratch(
            tuple(tensor.shape), tensor.dtype, name_hint="remote_src"
        )
        assert state.fx_node is not None
        state.fx_node.meta[_PALLAS_SRC_MATERIALIZATION_META] = (name, tensor_ast)
        return _direct_ref_expr(expr_from_string(name), ast_index)

    from ..device_function import PallasMemorySpace

    active_name = pallas_codegen.vmem_name(state, name)
    if (
        active_name == name
        and state.device_function.pallas_memory_space.get(id(tensor))
        == PallasMemorySpace.HBM
    ):
        return _direct_ref_expr(expr_from_string(name), ast_index)
    name = active_name
    if not proxy_index:
        return expr_from_string(name)

    assert state.fx_node is not None
    patterns = state.fx_node.meta.get(metadata_key)
    assert isinstance(patterns, list)
    parts, none_dims = pallas_codegen.index_parts(
        state,
        proxy_index,
        tensor,
        indexing_patterns=patterns,
        ast_subscripts=ast_index,
        # Remote-copy indices may select an arbitrary slot within a full VMEM
        # block. Only dimensions tied to the active tile become local offsets.
        pipeline_scalar_indices_local=False,
        tensor_indices_are_scalars=True,
    )
    assert not none_dims
    return expr_from_string(f"{name}.at[{', '.join(parts)}]")


def _make_remote_copy(state: CodegenState, *, start: bool) -> ast.AST:
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    # Explicit descriptors intentionally carry one reusable completion slot
    # across loop iterations. Create-and-start copies remain independent so a
    # later peer's notification cannot satisfy an earlier receive wait.
    semaphore_shape, semaphore_slot = (
        _fori_semaphore_slot(state) if start else ((), None)
    )
    send_sem = device_fn.register_dma_semaphore(
        name_hint="remote_send_sem", shape=semaphore_shape
    )
    recv_sem = device_fn.register_dma_semaphore(
        name_hint="remote_recv_sem", shape=semaphore_shape
    )
    if semaphore_slot is not None:
        slot = ast.unparse(semaphore_slot)
        send_sem = f"{send_sem}.at[{slot}]"
        recv_sem = f"{recv_sem}.at[{slot}]"
    op_name = device_fn.new_var("remote_copy", dce=False)

    assert state.fx_node is not None
    state.fx_node.meta[_PALLAS_OP_META] = op_name

    src_ref = _remote_ref_expr(
        state,
        src,
        state.ast_args[0],
        state.proxy_arg(1),
        state.ast_args[1],
        REMOTE_SRC_INDEXING_PATTERNS,
        materialize_device_value=True,
    )
    dst_ref = _remote_ref_expr(
        state,
        dst,
        state.ast_args[3],
        state.proxy_arg(4),
        state.ast_args[4],
        REMOTE_DST_INDEXING_PATTERNS,
    )
    device_id = state.ast_args[2]
    if isinstance(device_id, int):
        device_id = expr_from_string(repr(device_id))
    assert isinstance(device_id, ast.AST)

    state.codegen.add_statement(
        statement_from_string(
            f"{op_name} = pltpu.make_async_remote_copy("
            "{src_ref}, {dst_ref}, "
            f"{send_sem}, {recv_sem}, "
            "device_id={device_id}, "
            "device_id_type=pl.DeviceIdType.LOGICAL)",
            src_ref=src_ref,
            dst_ref=dst_ref,
            device_id=device_id,
        )
    )
    if start:
        _emit_source_materialization(state, state.fx_node)
        state.codegen.add_statement(statement_from_string(f"{op_name}.start()"))
    return expr_from_string(op_name)


@_decorators.codegen(make_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _make_remote_copy(state, start=False)


@_decorators.codegen(start_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _make_remote_copy(state, start=True)


def _paired_op_node(state: CodegenState) -> torch.fx.Node:
    assert state.fx_node is not None
    descriptor = state.fx_node.args[0]
    if not isinstance(descriptor, torch.fx.Node):
        raise exc.InternalError(
            RuntimeError("remote-copy wait is not paired with a start operation")
        )
    return descriptor


def _paired_op_name(state: CodegenState) -> str:
    op_name = _paired_op_node(state).meta.get(_PALLAS_OP_META)
    if not isinstance(op_name, str):
        raise exc.InternalError(
            RuntimeError("remote-copy wait could not resolve its start operation")
        )
    return op_name


def _emit_source_materialization(
    state: CodegenState, descriptor: torch.fx.Node
) -> bool:
    materialization = descriptor.meta.get(_PALLAS_SRC_MATERIALIZATION_META)
    if materialization is None:
        return False
    scratch, value = materialization
    assert isinstance(scratch, str)
    assert isinstance(value, ast.AST)
    state.codegen.add_statement(
        statement_from_string(f"{scratch}[...] = {{value}}", value=value)
    )
    return True


def _emit_wait(state: CodegenState, method: str) -> ast.AST:
    if method == "start":
        _emit_source_materialization(state, _paired_op_node(state))
    state.codegen.add_statement(
        statement_from_string(f"{_paired_op_name(state)}.{method}()")
    )
    return expr_from_string("None")


def _emit_wait_if(state: CodegenState, method: str) -> ast.AST:
    predicate = state.proxy_arg(1)
    if isinstance(predicate, bool):
        return _emit_wait(state, method) if predicate else expr_from_string("None")
    predicate_ast = state.ast_args[1]
    assert isinstance(predicate_ast, ast.AST)
    descriptor = _paired_op_node(state)
    materialization = descriptor.meta.get(_PALLAS_SRC_MATERIALIZATION_META)
    if method == "start" and materialization is not None:
        scratch, value = materialization
        assert isinstance(scratch, str)
        assert isinstance(value, ast.AST)
        function_name = state.device_function.new_var("start_remote_copy")
        state.codegen.add_statement(
            statement_from_string(
                f"@pl.when({{predicate}})\n"
                f"def {function_name}():\n"
                f"    {scratch}[...] = {{value}}\n"
                f"    {_paired_op_name(state)}.start()",
                predicate=predicate_ast,
                value=value,
            )
        )
        return expr_from_string("None")
    state.codegen.add_statement(
        statement_from_string(
            f"pl.when({{predicate}})(lambda: {_paired_op_name(state)}.{method}())",
            predicate=predicate_ast,
        )
    )
    return expr_from_string("None")


@_decorators.codegen(start_async_remote_copy_descriptor, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "start")


@_decorators.codegen(start_async_remote_copy_descriptor_if, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait_if(state, "start")


@_decorators.codegen(wait_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "wait")


@_decorators.codegen(wait_async_remote_copy_if, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait_if(state, "wait")


@_decorators.codegen(wait_send_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "wait_send")


@_decorators.codegen(wait_send_async_remote_copy_if, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait_if(state, "wait_send")


@_decorators.codegen(wait_recv_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "wait_recv")


@_decorators.codegen(wait_recv_async_remote_copy_if, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait_if(state, "wait_recv")
