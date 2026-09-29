"""Triton codegen for in-band (LL) cross-rank readiness.

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

# Relaxed system-scope stores: each 8-byte word is single-copy atomic and
# carries its own epoch, so the consumer needs no fence or acquire.
_PUSH_ASM = "st.relaxed.sys.global.u64 [$1], $2; mov.u32 $0, 0;"
_MASKED_PUSH_ASM = (
    "{ .reg .pred p; setp.ne.b32 p, $3, 0; "
    "@p st.relaxed.sys.global.u64 [$1], $2; } mov.u32 $0, 0;"
)
_MULTICAST_PUSH_ASM = "multimem.st.relaxed.sys.global.b64 [$1], $2; mov.u32 $0, 0;"
_MASKED_MULTICAST_PUSH_ASM = (
    "{ .reg .pred p; setp.ne.b32 p, $3, 0; "
    "@p multimem.st.relaxed.sys.global.b64 [$1], $2; } mov.u32 $0, 0;"
)


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
        or not env.settings.distributed_ll
        or config.cross_loop_pipeline != "dynamic"
        or not str(config.pid_type).startswith("persistent")
        or not device_ir.implicit_dependency_starts
        or graph is None
        or not graph.has_cross_rank_dependencies()
    ):
        return
    plan = plan_distributed_ll(
        graph,
        device_ir,
        pack_bf16=env.settings.distributed_ll_multicast,
    )
    if plan is None:
        return
    device_function.distributed_ll_plan = plan
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
    if env.settings.distributed_ll_multicast:
        multicast = device_function.new_var("tile_dependency_ll_multicast", dce=False)
        device_function.wrapper_only_params.append(multicast)
        device_function.triton_distributed_ll_multicast_arg = multicast
    device_function.triton_distributed_ll_tag_var = device_function.new_var(
        "tile_dependency_ll_tag", dce=False
    )
    device_function.triton_distributed_ll_parity_var = device_function.new_var(
        "tile_dependency_ll_parity", dce=False
    )
    device_function.triton_distributed_ll_mailbox_words = plan.mailbox_words
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


def ll_load_source(state: CodegenState) -> tuple[LLAllocation, int] | None:
    plan = state.device_function.distributed_ll_plan
    if plan is None or state.fx_node is None:
        return None
    key = state.fx_node.meta.get(TILE_ACCESS_KEY_META)
    return None if key is None else plan.loads.get(key)


def _ll_names(device_function: DeviceFunction) -> tuple[str, str, str, str, str]:
    mailbox = device_function.triton_distributed_ll_mailbox_arg
    mailbox_ptrs = device_function.triton_distributed_ll_mailbox_ptrs_arg
    rank = device_function.triton_distributed_ll_rank_arg
    tag = device_function.triton_distributed_ll_tag_var
    parity = device_function.triton_distributed_ll_parity_var
    if (
        mailbox is None
        or mailbox_ptrs is None
        or rank is None
        or tag is None
        or parity is None
    ):
        raise AssertionError("distributed LL state was not registered")
    return mailbox, mailbox_ptrs, rank, tag, parity


def _bits_type(dtype: torch.dtype) -> str:
    return f"tl.uint{dtype.itemsize * 8}"


def _pair_shapes(
    state: CodegenState,
    output_size: list[int | torch.SymInt],
) -> tuple[str, str, str]:
    """Return actual codegen numel, ``[pairs, 2]``, and ``[pairs]`` shapes."""
    dimensions = state.tile_strategy.shape_dims(output_size)
    numel = " * ".join(f"({dimension})" for dimension in dimensions) or "1"
    pair_count = f"({numel}) // 2"
    return numel, f"[{pair_count}, 2]", f"[{pair_count}]"


def _split_pairs(
    state: CodegenState,
    pairs: ast.Name,
    prefix: str,
) -> tuple[str, str]:
    """Split a final size-two dimension without unsupported tensor indexing."""
    low = state.device_function.new_var(f"{prefix}_low", dce=True)
    high = state.device_function.new_var(f"{prefix}_high", dce=True)
    state.add_statement(statement_from_string(f"{low}, {high} = tl.split({pairs.id})"))
    return low, high


def _push_statement(
    sink: str,
    address: str,
    word: str,
    mask: str | None,
    *,
    multicast: bool,
) -> ast.stmt:
    if multicast:
        asm = _MULTICAST_PUSH_ASM
        masked_asm = _MASKED_MULTICAST_PUSH_ASM
    else:
        asm = _PUSH_ASM
        masked_asm = _MASKED_PUSH_ASM
    if mask is None:
        constraints, args = "=r,l,l", f"{address}, {word}"
    else:
        asm = masked_asm
        constraints, args = "=r,l,l,r", f"{address}, {word}, {mask}"
    return statement_from_string(
        f"{sink} = tl.inline_asm_elementwise(asm={asm!r}, "
        f"constraints={constraints!r}, args=[{args}], dtype=tl.int32, "
        "is_pure=False, pack=1)"
    )


def _ll_push_transport(
    device_function: DeviceFunction,
    mailbox_ptrs: str,
    multicast_ptr: str | None,
    slot: str,
    offset: str,
    word: str,
    mask: str | None,
    world_size: int,
) -> list[ast.stmt]:
    """Emit one optional multicast push with the exact unicast fallback."""
    unicast: list[ast.stmt] = []
    for peer in range(world_size):
        base = device_function.new_var("ll_peer_mailbox", dce=True)
        sink = device_function.new_var("ll_push", dce=False)
        unicast.extend(
            (
                statement_from_string(
                    f"{base} = tl.load(({mailbox_ptrs}).to("
                    f"tl.pointer_type(tl.uint64)) + {peer}).to("
                    "tl.pointer_type(tl.uint64))"
                ),
                _push_statement(
                    sink,
                    f"{base} + ({slot}) + {offset}",
                    word,
                    mask,
                    multicast=False,
                ),
            )
        )
    if multicast_ptr is None:
        return unicast

    base = device_function.new_var("ll_multicast_mailbox", dce=True)
    sink = device_function.new_var("ll_multicast_push", dce=False)
    multicast = [
        statement_from_string(
            f"{base} = tl.cast({multicast_ptr}, tl.int64).to("
            "tl.pointer_type(tl.uint64))"
        ),
        _push_statement(
            sink,
            f"{base} + ({slot}) + {offset}",
            word,
            mask,
            multicast=True,
        ),
    ]
    return [
        create(
            ast.If,
            test=expr_from_string(f"{multicast_ptr} != 0"),
            body=multicast,
            orelse=unicast,
        )
    ]


def codegen_ll_push(
    state: CodegenState,
    fake_tensor: torch.Tensor,
    subscript: list[object],
    value: ast.AST,
    extra_mask: ast.AST | None,
    allocation: LLAllocation,
) -> ast.AST:
    """Push each stored element, epoch tagged, into every rank's mailbox.

    Returns the (lifted) value so the ordinary store reuses it.
    """
    device_function = state.device_function
    _mailbox, mailbox_ptrs, rank, tag, parity = _ll_names(device_function)
    multicast_ptr = device_function.triton_distributed_ll_multicast_arg
    backend = CompileEnvironment.current().backend
    indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
    output_size = SubscriptIndexing.compute_shape(fake_tensor, subscript, state)
    if not isinstance(value, ast.Constant):
        value = state.codegen.lift(value, dce=True, prefix="ll_value")
    offset: ast.AST = indexing.index_expr
    if indexing.needs_broadcast():
        shape_str = state.tile_strategy.shape_str(output_size)
        offset = expr_from_string(
            backend.broadcast_to_expr("{offset}", shape_str), offset=offset
        )
    offset = state.codegen.lift(offset, dce=True, prefix="ll_offset")
    dtype = backend.dtype_str(allocation.dtype)
    mask: ast.Name | None = None
    if indexing.has_mask():
        mask = state.codegen.lift(
            expr_from_string("tl.cast({mask}, tl.int32)", mask=indexing.mask_expr),
            dce=True,
            prefix="ll_mask",
        )
    if allocation.elements_per_word == 2:
        # A single atomic b64 store carries both BF16 payloads and the full tag:
        # [epoch:32 | odd payload:16 | even payload:16].
        block_numel, pair_shape, _packed_shape = _pair_shapes(state, output_size)
        state.add_statement(
            statement_from_string(f"tl.static_assert(({block_numel}) % 2 == 0)")
        )
        output_shape = state.tile_strategy.shape_str(output_size)
        expanded_value = expr_from_string(
            backend.broadcast_to_expr("{value}", output_shape), value=value
        )
        pairs = state.codegen.lift(
            expr_from_string(
                f"tl.reshape({{value}}, {pair_shape})", value=expanded_value
            ),
            dce=True,
            prefix="ll_pairs",
        )
        pair_low, pair_high = _split_pairs(state, pairs, "ll_pair")
        word = state.codegen.lift(
            expr_from_string(
                f"tl.cast(tl.cast(tl.cast({pair_low}, {dtype}), "
                "tl.uint16, bitcast=True), tl.uint64) | "
                f"(tl.cast(tl.cast(tl.cast({pair_high}, {dtype}), "
                "tl.uint16, bitcast=True), tl.uint64) << 16) | "
                f"({tag} << 32)"
            ),
            dce=True,
            prefix="ll_word",
        )
        offset_pairs = state.codegen.lift(
            expr_from_string(f"tl.reshape({{offset}}, {pair_shape})", offset=offset),
            dce=True,
            prefix="ll_offset_pairs",
        )
        offset_low, _offset_high = _split_pairs(state, offset_pairs, "ll_offset_pair")
        offset = state.codegen.lift(
            expr_from_string(f"{offset_low} // 2"),
            dce=True,
            prefix="ll_word_offset",
        )
        if mask is not None:
            mask_pairs = state.codegen.lift(
                expr_from_string(f"tl.reshape({{mask}}, {pair_shape})", mask=mask),
                dce=True,
                prefix="ll_mask_pairs",
            )
            mask_low, mask_high = _split_pairs(state, mask_pairs, "ll_mask_pair")
            mask = state.codegen.lift(
                expr_from_string(
                    f"tl.cast({mask_low} | {mask_high}, tl.int32)",
                ),
                dce=True,
                prefix="ll_word_mask",
            )
    else:
        word_value: ast.AST = value
        if not indexing.block_shaped_offset and output_size:
            word_value = expr_from_string(
                backend.reshape_expr("{value}", "[]"), value=value
            )
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
    slot = (
        f"{allocation.mailbox_offset} + {rank} * {allocation.words_per_source()} + "
        f"{parity} * {allocation.slot_words()}"
    )
    for statement in _ll_push_transport(
        device_function,
        mailbox_ptrs,
        multicast_ptr,
        slot,
        offset.id,
        word.id,
        None if mask is None else mask.id,
        allocation.world_size,
    ):
        state.add_statement(statement)
    return value


def codegen_ll_poll(
    state: CodegenState,
    fake_tensor: torch.Tensor,
    subscript: list[object],
    extra_mask: ast.AST | None,
    allocation: LLAllocation,
    source_rank: int,
) -> ast.AST:
    """Replace a peer-view load with a poll of this rank's own mailbox."""
    device_function = state.device_function
    mailbox, _mailbox_ptrs, _rank, tag, parity = _ll_names(device_function)
    env = CompileEnvironment.current()
    backend = env.backend
    indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
    output_size = SubscriptIndexing.compute_shape(fake_tensor, subscript, state)
    assert state.fx_node is not None
    access_key = state.fx_node.meta[TILE_ACCESS_KEY_META]
    plan = device_function.distributed_ll_plan
    assert plan is not None
    coalesce_pairs = (
        allocation.elements_per_word == 2
        and access_key in plan.packed_loads
        and not indexing.needs_broadcast()
    )
    offset: ast.AST = indexing.index_expr
    block_shaped = indexing.block_shaped_offset
    if indexing.has_mask() and not block_shaped:
        shape_str = state.tile_strategy.shape_str(output_size)
        offset = expr_from_string(
            f"{{offset}} + {backend.zeros_expr(shape_str, env.index_type())}",
            offset=offset,
        )
        block_shaped = True
    element_offset = offset
    packed_shape: str | None = None
    pair_shape: str | None = None
    if coalesce_pairs:
        block_numel, pair_shape, packed_shape = _pair_shapes(state, output_size)
        state.add_statement(
            statement_from_string(f"tl.static_assert(({block_numel}) % 2 == 0)")
        )
        element_offset = state.codegen.lift(
            offset,
            dce=True,
            prefix="ll_element_offset",
        )
        offset_pairs = state.codegen.lift(
            expr_from_string(
                f"tl.reshape({{offset}}, {pair_shape})", offset=element_offset
            ),
            dce=True,
            prefix="ll_offset_pairs",
        )
        offset_low, _offset_high = _split_pairs(state, offset_pairs, "ll_offset_pair")
        offset = expr_from_string(f"{offset_low} // 2")
    elif allocation.elements_per_word == 2:
        element_offset = state.codegen.lift(
            offset,
            dce=True,
            prefix="ll_element_offset",
        )
        offset = expr_from_string("{offset} // 2", offset=element_offset)
    address = state.codegen.lift(
        expr_from_string(
            f"{mailbox} + ({allocation.mailbox_offset} + "
            f"{source_rank * allocation.words_per_source()} + "
            f"{parity} * {allocation.slot_words()}) + {{offset}}",
            offset=offset,
        ),
        dce=True,
        prefix="ll_address",
    )
    # Masked lanes read a word that already carries this launch's tag.
    load = f"tl.load({address.id}, volatile=True)"
    if indexing.has_mask():
        mask = state.codegen.lift(indexing.mask_expr, dce=True, prefix="ll_mask")
        if coalesce_pairs:
            assert pair_shape is not None
            mask_pairs = state.codegen.lift(
                expr_from_string(
                    f"tl.reshape({{mask}}, {pair_shape})",
                    mask=mask,
                ),
                dce=True,
                prefix="ll_mask_pairs",
            )
            mask_low, mask_high = _split_pairs(state, mask_pairs, "ll_mask_pair")
            mask = state.codegen.lift(
                expr_from_string(f"{mask_low} | {mask_high}"),
                dce=True,
                prefix="ll_word_mask",
            )
        load = f"tl.load({address.id}, {mask.id}, other={tag} << 32, volatile=True)"
    word = device_function.new_var("ll_word", dce=False)
    state.add_statement(statement_from_string(f"{word} = {load}"))
    if coalesce_pairs:
        output_shape = state.tile_strategy.shape_str(output_size)
        low = (
            f"tl.cast(tl.cast({word}, tl.uint16), "
            f"{backend.dtype_str(allocation.dtype)}, bitcast=True)"
        )
        high = (
            f"tl.cast(tl.cast({word} >> 16, tl.uint16), "
            f"{backend.dtype_str(allocation.dtype)}, bitcast=True)"
        )
        result = expr_from_string(f"tl.reshape(tl.join({low}, {high}), {output_shape})")
    elif allocation.elements_per_word == 2:
        result = expr_from_string(
            f"tl.cast(tl.cast({word} >> "
            "(tl.cast({offset} & 1, tl.uint64) * 16), tl.uint16), "
            f"{backend.dtype_str(allocation.dtype)}, bitcast=True)",
            offset=element_offset,
        )
    else:
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
        shape = packed_shape or state.tile_strategy.shape_str(output_size)
    value = device_function.new_var("ll_value", dce=False)
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
