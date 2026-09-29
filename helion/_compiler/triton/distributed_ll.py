"""Triton codegen for in-band (LL) tile readiness.

See ``helion._compiler.distributed_ll`` for the planner and the protocol.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ...language.inline_asm_ops import inline_asm_elementwise
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..distributed_ll import TILE_ACCESS_KEY_META
from ..distributed_ll import plan_distributed_ll
from ..indexing_strategy import SubscriptIndexing

if TYPE_CHECKING:
    import torch

    from ...runtime.config import Config
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..distributed_ll import DistributedLLPlan
    from ..distributed_ll import LLAllocation
    from ..helper_function import CodegenInterface
    from ..inductor_lowering import CodegenState


def _push_asm(scope: str, masked: bool) -> str:
    """Relaxed store of one 8-byte word at ``scope`` (``sys`` or ``gpu``).

    Each word is single-copy atomic and carries its own epoch, so the consumer
    needs no fence or acquire.
    """
    store = f"st.relaxed.{scope}.global.u64 [$1], $2;"
    if masked:
        store = f"{{ .reg .pred p; setp.ne.b32 p, $3, 0; @p {store} }}"
    return f"{store} mov.u32 $0, 0;"


@dataclasses.dataclass(frozen=True)
class LLPendingPoll:
    """An issued mailbox load whose tag check is deferred to its first use."""

    node: torch.fx.Node
    word: str
    load: str
    # Elementwise int32 "tag not yet current" expression.
    stale: str
    block_shaped: bool
    # Block words with the same known shape share one reduction.
    shape: str | None
    result: ast.stmt


def register_distributed_ll(
    device_function: DeviceFunction,
    device_ir: DeviceIR,
    config: Config,
) -> None:
    """Plan LL readiness for this config and register its hidden kernel state."""
    env = CompileEnvironment.current()
    graph = device_ir.tile_dependency_graph
    if (
        env.backend_name != "triton"
        or not (env.settings.distributed_ll or env.settings.local_ll)
        or config.cross_loop_pipeline not in ("dynamic", "dynamic_exact")
        or not str(config.pid_type).startswith("persistent")
        or not device_ir.implicit_dependency_starts
        or graph is None
    ):
        return
    plan = plan_distributed_ll(
        graph,
        device_ir,
        config,
        symmetric=env.settings.distributed_ll and graph.has_cross_rank_dependencies(),
        local=env.settings.local_ll,
    )
    if plan is None:
        return
    device_function.distributed_ll_plan = plan
    if plan.mailbox_words:
        names = [
            device_function.new_var(hint, dce=False)
            for hint in (
                "tile_dependency_ll_mailbox",
                "tile_dependency_ll_mailbox_ptrs",
                "tile_dependency_ll_rank",
            )
        ]
        device_function.wrapper_only_params.extend(names)
        (
            device_function.triton_distributed_ll_mailbox_arg,
            device_function.triton_distributed_ll_mailbox_ptrs_arg,
            device_function.triton_distributed_ll_rank_arg,
        ) = names
        device_function.triton_distributed_ll_parity_var = device_function.new_var(
            "tile_dependency_ll_parity", dce=False
        )
        device_function.triton_distributed_ll_mailbox_words = plan.mailbox_words
    if plan.local_mailbox_words:
        # Bound as launcher state beside the dispatch ticket once kernel
        # arguments exist (emit_cross_loop_schedule).
        device_function.triton_distributed_ll_local_mailbox_arg = (
            device_function.new_var("tile_dependency_ll_local_mailbox", dce=False)
        )
    device_function.triton_distributed_ll_tag_var = device_function.new_var(
        "tile_dependency_ll_tag", dce=False
    )
    _hoist_ll_loads(device_function.codegen.codegen_graphs, plan)


def _reorderable(node: torch.fx.Node) -> bool:
    """Whether a node is pure and bumps no per-memory-op config counter."""
    # Impure asm (e.g. clock reads) is not marked has_side_effect, so skip all asm.
    return (
        node.op == "call_function"
        and TILE_ACCESS_KEY_META not in node.meta
        and node.target is not inline_asm_elementwise
        and not node.is_impure()
    )


def _hoist_ll_loads(graphs: list[GraphInfo], plan: DistributedLLPlan) -> None:
    """Move each LL load, with its pure inputs, up to the previous impure node.

    Loads then issue back to back and their deferred polls merge into one loop.
    Only pure nodes are crossed, so memory ops keep their order (and their
    indexing / eviction config slots).  Runs on the per-config codegen copies.
    """
    for graph_info in graphs:
        graph = graph_info.graph
        for node in list(graph.nodes):
            if node.meta.get(TILE_ACCESS_KEY_META) not in plan.loads:
                continue
            segment: list[torch.fx.Node] = []
            floor = node.prev
            while floor.op != "root" and _reorderable(floor):
                segment.append(floor)
                floor = floor.prev
            if not segment:
                continue
            needed = {node}
            for candidate in segment:
                if any(user in needed for user in candidate.users):
                    needed.add(candidate)
            anchor = floor
            for moved in [*reversed(segment), node]:
                if moved in needed:
                    anchor.append(moved)
                    anchor = moved


def ll_store_allocation(state: CodegenState) -> LLAllocation | None:
    plan = state.device_function.distributed_ll_plan
    if plan is None or state.fx_node is None:
        return None
    key = state.fx_node.meta.get(TILE_ACCESS_KEY_META)
    return None if key is None else plan.stores.get(key)


def ll_load_source(state: CodegenState) -> tuple[LLAllocation, int | None] | None:
    plan = state.device_function.distributed_ll_plan
    if plan is None or state.fx_node is None:
        return None
    key = state.fx_node.meta.get(TILE_ACCESS_KEY_META)
    return None if key is None else plan.loads.get(key)


def _ll_tag(device_function: DeviceFunction) -> str:
    tag = device_function.triton_distributed_ll_tag_var
    if tag is None:
        raise AssertionError("LL state was not registered")
    return tag


def _local_mailbox(device_function: DeviceFunction) -> str:
    mailbox = device_function.triton_distributed_ll_local_mailbox_arg
    if mailbox is None:
        raise AssertionError("local LL mailbox was not registered")
    return mailbox


def _symmetric_names(device_function: DeviceFunction) -> tuple[str, str, str, str]:
    mailbox = device_function.triton_distributed_ll_mailbox_arg
    mailbox_ptrs = device_function.triton_distributed_ll_mailbox_ptrs_arg
    rank = device_function.triton_distributed_ll_rank_arg
    parity = device_function.triton_distributed_ll_parity_var
    if mailbox is None or mailbox_ptrs is None or rank is None or parity is None:
        raise AssertionError("distributed LL state was not registered")
    return mailbox, mailbox_ptrs, rank, parity


def _bits_type(dtype: torch.dtype) -> str:
    return f"tl.uint{dtype.itemsize * 8}"


def codegen_ll_push(
    state: CodegenState,
    fake_tensor: torch.Tensor,
    subscript: list[object],
    value: ast.AST,
    extra_mask: ast.AST | None,
    allocation: LLAllocation,
) -> ast.AST:
    """Push each stored element, epoch tagged, into every reader's mailbox.

    Returns the (lifted) value so the ordinary store reuses it.
    """
    device_function = state.device_function
    tag = _ll_tag(device_function)
    backend = CompileEnvironment.current().backend
    indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
    output_size = SubscriptIndexing.compute_shape(fake_tensor, subscript, state)
    if not isinstance(value, ast.Constant):
        value = state.codegen.lift(value, dce=True, prefix="ll_value")
    word_value: ast.AST = value
    if not indexing.block_shaped_offset and output_size:
        word_value = expr_from_string(
            backend.reshape_expr("{value}", "[]"), value=value
        )
    offset: ast.AST = indexing.index_expr
    if indexing.needs_broadcast():
        shape_str = state.tile_strategy.shape_str(output_size)
        offset = expr_from_string(
            backend.broadcast_to_expr("{offset}", shape_str), offset=offset
        )
    offset = state.codegen.lift(offset, dce=True, prefix="ll_offset")
    dtype = backend.dtype_str(allocation.dtype)
    word = state.codegen.lift(
        expr_from_string(
            f"tl.cast(tl.cast(tl.cast({{value}}, {dtype}), "
            f"{_bits_type(allocation.dtype)}, bitcast=True), tl.uint64) "
            f"| ({tag} << 32)",
            value=word_value,
        ),
        dce=True,
        prefix="ll_word",
    )
    mask: ast.Name | None = None
    if indexing.has_mask():
        mask = state.codegen.lift(
            expr_from_string("tl.cast({mask}, tl.int32)", mask=indexing.mask_expr),
            dce=True,
            prefix="ll_mask",
        )
    if allocation.world_size is None:
        address = f"{_local_mailbox(device_function)} + {allocation.mailbox_offset}"
        _emit_push(state, "gpu", f"{address} + {offset.id}", word, mask)
        return value
    _mailbox, mailbox_ptrs, rank, parity = _symmetric_names(device_function)
    slot = (
        f"{allocation.mailbox_offset} + {rank} * {allocation.numel} + "
        f"{parity} * {allocation.slot_words()}"
    )
    for peer in range(allocation.world_size):
        base = device_function.new_var("ll_peer_mailbox", dce=True)
        state.add_statement(
            statement_from_string(
                f"{base} = tl.load(({mailbox_ptrs}).to(tl.pointer_type(tl.uint64)) "
                f"+ {peer}).to(tl.pointer_type(tl.uint64))"
            )
        )
        _emit_push(state, "sys", f"{base} + ({slot}) + {offset.id}", word, mask)
    return value


def _emit_push(
    state: CodegenState,
    scope: str,
    address: str,
    word: ast.Name,
    mask: ast.Name | None,
) -> None:
    sink = state.device_function.new_var("ll_push", dce=False)
    if mask is None:
        constraints, args = "=r,l,l", f"{address}, {word.id}"
    else:
        constraints, args = "=r,l,l,r", f"{address}, {word.id}, {mask.id}"
    state.add_statement(
        statement_from_string(
            f"{sink} = tl.inline_asm_elementwise("
            f"asm={_push_asm(scope, mask is not None)!r}, "
            f"constraints={constraints!r}, args=[{args}], dtype=tl.int32, "
            "is_pure=False, pack=1)"
        )
    )


def codegen_ll_poll(
    state: CodegenState,
    fake_tensor: torch.Tensor,
    subscript: list[object],
    extra_mask: ast.AST | None,
    allocation: LLAllocation,
    source_rank: int | None,
) -> ast.AST:
    """Replace an LL load with a poll of this rank's own mailbox."""
    device_function = state.device_function
    tag = _ll_tag(device_function)
    env = CompileEnvironment.current()
    backend = env.backend
    indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
    output_size = SubscriptIndexing.compute_shape(fake_tensor, subscript, state)
    offset: ast.AST = indexing.index_expr
    block_shaped = indexing.block_shaped_offset
    if indexing.has_mask() and not block_shaped:
        shape_str = state.tile_strategy.shape_str(output_size)
        offset = expr_from_string(
            f"{{offset}} + {backend.zeros_expr(shape_str, env.index_type())}",
            offset=offset,
        )
        block_shaped = True
    if source_rank is None:
        base = f"{_local_mailbox(device_function)} + {allocation.mailbox_offset}"
    else:
        mailbox, _mailbox_ptrs, _rank, parity = _symmetric_names(device_function)
        base = (
            f"{mailbox} + ({allocation.mailbox_offset} + "
            f"{source_rank * allocation.numel} + "
            f"{parity} * {allocation.slot_words()})"
        )
    address = state.codegen.lift(
        expr_from_string(f"{base} + {{offset}}", offset=offset),
        dce=True,
        prefix="ll_address",
    )
    # Masked lanes read a word that already carries this launch's tag.
    load = f"tl.load({address.id}, volatile=True)"
    if indexing.has_mask():
        mask = state.codegen.lift(indexing.mask_expr, dce=True, prefix="ll_mask")
        load = f"tl.load({address.id}, {mask.id}, other={tag} << 32, volatile=True)"
    word = device_function.new_var("ll_word", dce=False)
    state.add_statement(statement_from_string(f"{word} = {load}"))
    result = expr_from_string(
        f"tl.cast(tl.cast({word}, {_bits_type(allocation.dtype)}), "
        f"{backend.dtype_str(allocation.dtype)}, bitcast=True)"
    )
    shape = None
    if indexing.needs_broadcast():
        shape_str = state.tile_strategy.shape_str(output_size)
        result = expr_from_string(
            backend.broadcast_to_expr("{value}", shape_str), value=result
        )
    elif block_shaped:
        shape = state.tile_strategy.shape_str(output_size)
    value = device_function.new_var("ll_value", dce=False)
    assert state.fx_node is not None
    device_function.distributed_ll_pending_polls.append(
        LLPendingPoll(
            node=state.fx_node,
            word=word,
            load=load,
            stale=f"tl.cast(({word} >> 32) != {tag}, tl.int32)",
            block_shaped=block_shaped,
            shape=shape,
            result=statement_from_string(f"{value} = {{result}}", result=result),
        )
    )
    return create(ast.Name, id=value, ctx=ast.Load())


def flush_ll_polls(
    cg: CodegenInterface,
    graph: torch.fx.Graph,
    user: torch.fx.Node | None = None,
) -> None:
    """Wait for this graph's issued LL words, all in one reload loop.

    With ``user`` set, flush only if ``user`` reads one of the pending words.
    """
    device_function = cg.device_function
    pending = device_function.distributed_ll_pending_polls
    polls = [poll for poll in pending if poll.node.graph is graph]
    if not polls or (
        user is not None
        and not any(poll.node in user.all_input_nodes for poll in polls)
    ):
        return
    device_function.distributed_ll_pending_polls = [
        poll for poll in pending if poll.node.graph is not graph
    ]
    checks: list[str] = []
    by_shape: dict[str, list[str]] = {}
    for poll in polls:
        if not poll.block_shaped:
            checks.append(poll.stale)
        elif poll.shape is None:
            checks.append(f"tl.max({poll.stale})")
        else:
            by_shape.setdefault(poll.shape, []).append(poll.stale)
    checks.extend(f"tl.max({' | '.join(stale)})" for stale in by_shape.values())
    cg.add_statement(
        create(
            ast.While,
            test=expr_from_string(f"({' | '.join(checks)}) != 0"),
            body=[
                statement_from_string(f"{poll.word} = {poll.load}") for poll in polls
            ],
            orelse=[],
        )
    )
    for poll in polls:
        cg.add_statement(poll.result)
