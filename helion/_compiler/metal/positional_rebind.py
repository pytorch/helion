"""Reject the positional merges that a permute's relabel would get wrong.

A Metal thread holds one element of each value, addressed through the block
ids of the value's dims, so ``aten.permute`` only relabels dims
(``codegen_permute_metal``): the thread holding ``x[i, j]`` holds
``x.T[j, i]``.  Helion's tiles are positional arrays, though.  A store, a
mask, an element-wise op, a ``where``, a matmul accumulator, tensor indexers
broadcast together and a value carried through a loop or a branch all meet
their operands' dims by position.  Where a position pairs two different
blocks, or a tile with a dim that has none, the thread would read another
element than the positional program does, so the kernel is rejected.

Ported from CuTe's ``rebound_block_dims`` / ``operand_positions`` and the
store and mask half of ``check_literal_slot_dims``.  CuTe exchanges a
rebound operand between threads; Metal refuses it instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node
from torch.fx.node import map_arg

from ... import exc
from ...language import _tracing_ops
from ...language.matmul_ops import dot
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from ..indexing_strategy import SubscriptIndexing
from ..inductor_lowering import PointwiseLowering

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from ..device_ir import GraphInfo

    Shape = Sequence[int | torch.SymInt]


def reject_positional_rebinds(graphs: list[GraphInfo]) -> None:
    """Raise ``BackendUnsupported`` where a kernel that permutes meets a dim by
    position with another block, or a tile with a dim that has none.

    Only kernels holding an ``aten.permute`` are checked.  A permute-free
    kernel can also meet dims on different blocks (``out[tm, tn] =
    y[tn, tm]``); ``literal_dims.reject_literal_dims``, which runs after this
    on every kernel, refuses those.
    """
    nodes = [
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.op == "call_function"
    ]
    if not any(node.target is torch.ops.aten.permute.default for node in nodes):
        return
    check_positional_merges(
        nodes, "in a kernel that permutes it would read another element"
    )


def check_positional_merges(nodes: Iterable[Node], why: str) -> None:
    """Raise ``BackendUnsupported`` where one of *nodes* meets a dim by
    position with another block, or a tile with a dim that has none.  *why*
    says what the by-block reading would get wrong there."""
    for node in nodes:
        _check_node(node, why)


def _check_node(node: Node, why: str) -> None:
    result = node.meta.get("val")
    if node.target is store:
        access = _access_shape(node, why)
        if access is None:
            return
        _check_operand("a store", node, node.args[2], access, why)
        _check_operand("a store mask", node, node.args[3], access, why)
    elif node.target is load:
        access = _access_shape(node, why)
        if access is None:
            return
        # A traced load's value has its access shape; a load a graph rewrite
        # inserts (MPP's reload of a materialized result) can carry another.
        assert isinstance(result, torch.Tensor)
        _check("a load result", node, result.shape, access, why)
        _check_operand("a load mask", node, node.args[2], access, why)
    elif isinstance(node.meta.get("lowering"), PointwiseLowering) and isinstance(
        result, torch.Tensor
    ):
        for operand in node.all_input_nodes:
            value = _tensor(operand)
            if value is None or value.ndim == 0:
                continue
            if value.ndim == result.ndim:
                _check("an element-wise operand", node, value.shape, result.shape, why)
            else:
                _check(
                    "a broadcast element-wise operand",
                    node,
                    value.shape,
                    result.shape,
                    why,
                    lower_rank=True,
                )
    elif node.target is torch.ops.aten.where.self:
        assert isinstance(result, torch.Tensor)
        for operand in node.all_input_nodes:
            _check_operand("a where operand", node, operand, result.shape, why)
    elif node.target is dot:
        assert isinstance(result, torch.Tensor)
        _check_operand("a matmul accumulator", node, node.args[2], result.shape, why)
    elif node.target in (torch.ops.aten.addmm.default, torch.ops.aten.baddbmm.default):
        assert isinstance(result, torch.Tensor)
        _check_operand("a matmul accumulator", node, node.args[0], result.shape, why)
    elif node.target is _tracing_ops._phi:
        # A loop's or branch's result has the fake value its body assigns, so
        # this compares the layout a carry leaves with the one it enters with
        # (for a value both branches assign, the two branches' layouts).  A
        # loop body is traced with the entering layout, so once they agree,
        # the checks inside the body hold on every iteration.
        initial, merged = node.args
        initial_value = _tensor(initial)
        if initial_value is not None:
            _check_operand(
                "a control-flow carry", node, merged, initial_value.shape, why
            )


def _access_shape(node: Node, why: str) -> list[int | torch.SymInt] | None:
    """The shape a load or store addresses, after checking the tensor
    indexers it broadcasts together; None for a target Metal's load and store
    reject anyway (a stack tensor)."""
    tensor, subscript = map_arg(node.args[:2], lambda arg: arg.meta["val"])
    if not isinstance(tensor, torch.Tensor):
        return None
    assert isinstance(subscript, (list, tuple))
    env = CompileEnvironment.current()
    indexers = [k for k in subscript if isinstance(k, torch.Tensor)]
    # All-1-D indexers form a Cartesian product, so they never meet.
    if (
        len(indexers) > 1
        and env.should_broadcast_tensor_indexers(subscript)
        and not all(indexer.ndim == 1 for indexer in indexers)
    ):
        shape = env.tensor_indexer_broadcast_shape(indexers)
        for indexer in indexers:
            _check("an index tensor", node, indexer.shape, shape, why)
    return SubscriptIndexing.compute_shape(tensor, list(subscript))


def _tensor(arg: object) -> torch.Tensor | None:
    if isinstance(arg, Node):
        value = arg.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value
    return None


def _check_operand(
    what: str, node: Node, operand: object, consumer: Shape, why: str
) -> None:
    value = _tensor(operand)
    if value is not None:
        _check(what, node, value.shape, consumer, why)


def _check(
    what: str,
    node: Node,
    value: Shape,
    consumer: Shape,
    why: str,
    *,
    lower_rank: bool = False,
) -> None:
    """Raise if a dim of *value* meets, at its position in *consumer*, another
    block, or a tile meets a dim without one."""
    env = CompileEnvironment.current()
    positions = _operand_positions(value, consumer, lower_rank=lower_rank)
    for dim, (size, position) in enumerate(zip(value, positions, strict=True)):
        if position < 0:
            continue
        consumer_size = consumer[position]
        block, consumer_block = canonical_block(size), canonical_block(consumer_size)
        if block is not None and consumer_block is not None:
            if block != consumer_block:
                raise _rebind_unsupported(
                    what,
                    node,
                    dim,
                    value,
                    consumer,
                    f"block {block} meets block {consumer_block}",
                    why,
                )
            continue
        # Two dims without a tile need equal sizes, which Helion's broadcast
        # check enforces at trace time.  A size-1 dim still broadcasts.
        if env.known_equal(size, 1) or env.known_equal(consumer_size, 1):
            continue
        if _is_tile(size) != _is_tile(consumer_size):
            raise _rebind_unsupported(
                what,
                node,
                dim,
                value,
                consumer,
                f"{_describe(size)} meets {_describe(consumer_size)}, one of them "
                "without a tile",
                why,
            )


def _operand_positions(value: Shape, consumer: Shape, *, lower_rank: bool) -> list[int]:
    """The consumer dim each dim of *value* meets, as in CuTe's
    ``operand_positions``.

    Right-aligned, so a negative position is a dim the consumer lacks.  With
    *lower_rank*, a lower-rank operand is placed the way
    ``TileStrategyDispatch.broadcast_expand_dims`` expands it: each dim at the
    consumer dim of its block, else at an unused dim without one of equal
    size, else at the last unused dim, then in the operand's own order.
    """
    env = CompileEnvironment.current()
    if not (lower_rank and 0 < len(value) < len(consumer)):
        offset = len(consumer) - len(value)
        return [dim + offset for dim in range(len(value))]
    consumer_blocks = [canonical_block(size) for size in consumer]
    matched: list[int] = []
    for size in value:
        block = canonical_block(size)
        free = [p for p in range(len(consumer)) if p not in matched]
        if block is not None:
            candidates = [p for p in free if consumer_blocks[p] == block]
        else:
            candidates = [
                p
                for p in free
                if consumer_blocks[p] is None and env.known_equal(size, consumer[p])
            ]
        matched.append(candidates[0] if candidates else free[-1])
    return sorted(matched)


def canonical_block(size: int | torch.SymInt) -> int | None:
    env = CompileEnvironment.current()
    block_id = env.get_block_id(size)
    return None if block_id is None else env.canonical_block_id(block_id)


def _is_tile(size: int | torch.SymInt) -> bool:
    """Whether *size* is a tile's dim.  A ``:`` slice is a plain int in traced
    values but its reduction block in an access shape, so a reduction block
    counts as a dim without a tile."""
    block = canonical_block(size)
    return (
        block is not None
        and not CompileEnvironment.current().block_sizes[block].reduction
    )


def _describe(size: int | torch.SymInt) -> str:
    block = canonical_block(size)
    if block is None:
        return f"a dimension of size {size}"
    info = CompileEnvironment.current().block_sizes[block]
    if info.reduction:
        return f"a dimension of size {info.size}"
    return f"block {block}"


def _rebind_unsupported(
    what: str,
    node: Node,
    dim: int,
    value: Shape,
    consumer: Shape,
    reason: str,
    why: str,
) -> exc.BackendUnsupported:
    return exc.BackendUnsupported(
        "metal",
        f"{what} ({node.name}) meets dimension {dim} of a value of shape "
        f"{list(value)} at the same position of {list(consumer)}: {reason}.  A "
        f"Metal thread holds values by block id, so {why}; index both sides "
        "with the same tiles in the same order",
    )
