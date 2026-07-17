"""Pallas-backend codegen for ops defined in ``helion.language.distributed_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
registrations; it is listed in ``helion/_compiler/pallas/_codegen_modules.py``
so the backend registry imports it with the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.distributed_ops import start_async_remote_copy
from ...language.distributed_ops import sync_barrier
from ...language.distributed_ops import wait_async_remote_copy
from ...language.distributed_ops import wait_async_remote_recv
from ...language.distributed_ops import wait_send_async_remote_copy
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _codegen_at_expr(
    index_ast: object, prefix: str, block_rows: int = 1
) -> tuple[str, dict[str, ast.AST]]:
    """Build a ``.at[...]`` subscript string + placeholder map from an index
    list.  Each element is either an ``int`` literal (emitted inline) or an
    already-lowered AST node (passed through as a placeholder).  ``prefix``
    namespaces the placeholders so src and dst indices never collide.

    When ``block_rows > 1`` the *last* index element is a block start: it is
    emitted as ``pl.ds(start, block_rows)`` so the copy selects a contiguous
    block of ``block_rows`` rows on that dim (a strided DMA) instead of one row.
    """
    assert isinstance(index_ast, (list, tuple)), index_ast
    placeholders: dict[str, ast.AST] = {}
    parts: list[str] = []
    last = len(index_ast) - 1
    for i, elt in enumerate(index_ast):
        name = f"{prefix}{i}"
        if isinstance(elt, int):
            placeholders[name] = expr_from_string(repr(elt))
        else:
            assert isinstance(elt, ast.AST), elt
            placeholders[name] = elt
        if block_rows > 1 and i == last:
            parts.append(f"pl.ds({{{name}}}, {block_rows})")
        else:
            parts.append(f"{{{name}}}")
    return ", ".join(parts), placeholders


@_decorators.codegen(start_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``_op = pltpu.make_async_remote_copy(...); _op.start()``.

    Emits two *independent* refs (``src.at[index]`` -> ``dst.at[dst_index]``)
    so the copy can be asymmetric.  ``device_id`` may be a runtime scalar.
    Stashes the ``_op`` variable name on the FX node's meta dict so the paired
    ``wait_async_remote_copy`` codegen can look it up.
    """
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    src_name = device_fn.tensor_arg(src).name
    dst_name = device_fn.tensor_arg(dst).name

    send_sem = device_fn.register_dma_semaphore(name_hint="send_sem")
    recv_sem = device_fn.register_dma_semaphore(name_hint="recv_sem")
    # Stash the receive semaphore so a later ``wait_async_remote_recv`` drain
    # (the receive-side counterpart of ``wait_send``) can reference the same
    # semaphore that inbound peer pushes into ``dst`` increment.
    device_fn._distributed_recv_sem = recv_sem  # type: ignore[attr-defined]
    op_var = device_fn.new_var("_remote_copy", dce=False)

    # Stash the op variable name on the FX node so wait can find it.
    assert state.fx_node is not None
    state.fx_node.meta["_pallas_async_copy_op"] = op_var

    # block_rows (arg 5): copy a contiguous block of this many rows starting at
    # the last index element (default 1 = single-row copy).
    block_rows = state.proxy_arg(5) if len(state.proxy_args) > 5 else 1
    assert isinstance(block_rows, int)
    src_at, src_ph = _codegen_at_expr(state.ast_args[1], "_sidx", block_rows)
    dst_at, dst_ph = _codegen_at_expr(state.ast_args[4], "_didx", block_rows)

    device_id_ast = state.ast_args[2]
    if isinstance(device_id_ast, int):
        device_id_ast = expr_from_string(repr(device_id_ast))
    assert isinstance(device_id_ast, ast.AST)

    state.codegen.add_statement(
        statement_from_string(
            f"{op_var} = pltpu.make_async_remote_copy("
            f"{src_name}.at[{src_at}], {dst_name}.at[{dst_at}], "
            f"{send_sem}, {recv_sem}, "
            f"device_id={{device_id}}, "
            f"device_id_type=pl.DeviceIdType.LOGICAL)",
            device_id=device_id_ast,
            **src_ph,
            **dst_ph,
        )
    )
    state.codegen.add_statement(statement_from_string(f"{op_var}.start()"))
    return expr_from_string(op_var)


@_decorators.codegen(wait_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``<op_var>.wait()`` by looking up the op var stashed by
    the paired ``start_async_remote_copy`` on its FX node.
    """
    assert state.fx_node is not None
    descriptor_arg = state.fx_node.args[0]
    assert isinstance(descriptor_arg, torch.fx.Node), (
        "wait_async_remote_copy argument must be an FX node returned by "
        "start_async_remote_copy"
    )
    op_var = descriptor_arg.meta.get("_pallas_async_copy_op")
    if op_var is None:
        raise exc.InternalError(
            RuntimeError(
                "wait_async_remote_copy could not find the op variable name "
                "on the descriptor's source node — check that codegen for "
                "start_async_remote_copy ran first."
            )
        )
    state.codegen.add_statement(statement_from_string(f"{op_var}.wait()"))
    return expr_from_string("None")


@_decorators.codegen(wait_send_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``<op_var>.wait_send()`` (drain the local send) using the op var
    stashed by the paired ``start_async_remote_copy``."""
    assert state.fx_node is not None
    descriptor_arg = state.fx_node.args[0]
    assert isinstance(descriptor_arg, torch.fx.Node), (
        "wait_send_async_remote_copy argument must be an FX node returned by "
        "start_async_remote_copy"
    )
    op_var = descriptor_arg.meta.get("_pallas_async_copy_op")
    if op_var is None:
        raise exc.InternalError(
            RuntimeError(
                "wait_send_async_remote_copy could not find the op variable name "
                "on the descriptor's source node."
            )
        )
    state.codegen.add_statement(statement_from_string(f"{op_var}.wait_send()"))
    return expr_from_string("None")


@_decorators.codegen(wait_async_remote_recv, "pallas")
def _(state: CodegenState) -> object:
    """Emit the receive-side drain::

        pltpu.make_async_copy(dst[0:count], dst[0:count], recv_sem).wait()

    Waits for ``count`` row-sized increments on the receive semaphore stashed
    by ``start_async_remote_copy`` (the copies that filled ``dst`` allocated
    it).  ``src == dst`` so no data moves -- this is a pure semaphore drain.
    """
    dst = state.proxy_arg(0)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    dst_name = device_fn.tensor_arg(dst).name
    recv_sem = getattr(device_fn, "_distributed_recv_sem", None)
    if recv_sem is None:
        raise exc.InternalError(
            RuntimeError(
                "wait_async_remote_recv found no receive semaphore on the "
                "device function — a start_async_remote_copy pushing into a "
                "local buffer must be codegen'd before the drain."
            )
        )

    # ``count`` names the leading-dim extent of the drain slice.  It appears
    # twice (src and dst are the same slice), so a compile-time count is inlined
    # as a literal and a runtime scalar is bound to a fresh variable first --
    # either way the ``.at[...]`` slice is referenced by a plain name string, so
    # no AST node is shared between the two occurrences.
    count_ast = state.ast_args[1]
    count_literal: int | None = None
    if isinstance(count_ast, int):
        count_literal = count_ast
    elif isinstance(count_ast, ast.Constant) and isinstance(count_ast.value, int):
        count_literal = count_ast.value

    if count_literal is not None:
        count_ref = repr(count_literal)
    else:
        assert isinstance(count_ast, ast.AST)
        count_var = device_fn.new_var("_recv_drain_count", dce=False)
        state.codegen.add_statement(
            statement_from_string(f"{count_var} = {{count}}", count=count_ast)
        )
        count_ref = count_var

    # Match dst rank: dst[0:count] over the leading dim, full trailing dims.
    trailing = ", :" * (dst.ndim - 1)
    slab = f"{dst_name}.at[pl.ds(0, {count_ref}){trailing}]"
    state.codegen.add_statement(
        statement_from_string(
            f"pltpu.make_async_copy({slab}, {slab}, {recv_sem}).wait()"
        )
    )
    return expr_from_string("None")


@_decorators.codegen(sync_barrier, "pallas")
def _(state: CodegenState) -> object:
    """Emit the ``get_barrier_semaphore`` all-to-all barrier: signal every peer
    (self included) then wait for ``world_size`` signals.  ``world_size`` is a
    compile-time constant so the signal loop unrolls; peers are addressed by
    flat LOGICAL id, matching ``start_async_remote_copy``'s addressing.
    """
    ws_ast = state.ast_args[0]
    world_size: int | None = None
    if isinstance(ws_ast, int):
        world_size = ws_ast
    elif isinstance(ws_ast, ast.Constant) and isinstance(ws_ast.value, int):
        world_size = ws_ast.value
    if world_size is None:
        raise exc.InternalError(
            RuntimeError("sync_barrier world_size must be a compile-time int")
        )

    sem = state.device_function.new_var("_barrier_sem", dce=False)
    state.codegen.add_statement(
        statement_from_string(f"{sem} = pltpu.get_barrier_semaphore()")
    )
    for i in range(world_size):
        state.codegen.add_statement(
            statement_from_string(
                f"pltpu.semaphore_signal({sem}, device_id={i}, "
                f"device_id_type=pl.DeviceIdType.LOGICAL)"
            )
        )
    state.codegen.add_statement(
        statement_from_string(f"pltpu.semaphore_wait({sem}, {world_size})")
    )
    return expr_from_string("None")
