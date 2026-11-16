"""Which dims of an op's operands meet which dims of its result.

An op that combines several tile values per element (a pointwise op, ``where``,
``stack``, ``hl.join``, ``hl.inline_asm_elementwise``, a tuple ``hl.reduce`` /
``hl.associative_scan``, a matmul's accumulator) pairs its operands' dims by
position.  ``rebound_operands`` names the operands it combines,
``rebound_consumer_dims`` the dims they are compared against, and
``operand_positions`` the result dim each operand dim meets.  Backend-neutral:
CuTe's reshape and view checks and the literal-dim analyses
(``literal_dims``) all read them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node

from ..language.inline_asm_ops import inline_asm_elementwise
from ..language.matmul_ops import dot as hl_dot
from ..language.reduce_ops import _reduce
from ..language.scan_ops import _associative_scan
from ..language.view_ops import join
from .inductor_lowering import size1_reduction_positions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .compile_environment import CompileEnvironment
    from .compile_environment import Config

# The reshape targets.
VIEW_TARGETS = frozenset(
    {
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.view.default,
    }
)


def operand_positions(
    env: CompileEnvironment,
    value: torch.Tensor,
    consumer_sizes: Sequence[int | torch.SymInt],
    consumer_block_ids: Sequence[int | None],
    *,
    pointwise_placement: bool,
) -> list[int]:
    """The consumer dim each dim of ``value`` meets (see ``rebound_block_dims``).

    Right-aligned, so a negative position is a dim the consumer does not have;
    a lower-rank pointwise operand is placed as
    ``TileDispatch.broadcast_expand_dims`` places it, and a higher-rank one
    (the input of a reduction over extent-1 dims) as
    ``size1_reduction_positions`` places it.  A higher-rank operand that is
    no such input stays right-aligned.
    """
    if pointwise_placement and value.ndim > len(consumer_block_ids):
        positions = size1_reduction_positions(value.shape, consumer_sizes)
        if positions is not None:
            return positions
    offset = len(consumer_block_ids) - value.ndim
    if not (pointwise_placement and 0 < value.ndim < len(consumer_block_ids)):
        return [dim + offset for dim in range(value.ndim)]
    consumer_canonical = [
        None if block_id is None else env.canonical_block_id(block_id)
        for block_id in consumer_block_ids
    ]
    matched: list[int] = []
    used: set[int] = set()
    for size in value.shape:
        block_id = env.get_block_id(size)
        canonical = None if block_id is None else env.canonical_block_id(block_id)
        match = None
        if canonical is not None:
            match = next(
                (
                    position
                    for position, candidate in enumerate(consumer_canonical)
                    if position not in used and candidate == canonical
                ),
                None,
            )
        else:
            match = next(
                (
                    position
                    for position, candidate in enumerate(consumer_canonical)
                    if position not in used
                    and candidate is None
                    and env.known_equal(size, consumer_sizes[position])
                ),
                None,
            )
        if match is None:
            match = max(
                position
                for position in range(len(consumer_block_ids))
                if position not in used
            )
        matched.append(match)
        used.add(match)
    return sorted(matched)


def block_extent(env: CompileEnvironment, config: Config, block_id: int) -> int | None:
    """``block_id``'s extent under ``config`` (``None`` while it is symbolic)."""
    extent = env.block_sizes[env.canonical_block_id(block_id)].from_config(config)
    return extent if isinstance(extent, int) else None


# Custom lowerings that combine several tile operands per thread and so need
# ``check_pointwise_rebound_block_ids`` like a PointwiseLowering: where, stack,
# hl.join, hl.inline_asm_elementwise, the tuple forms of hl.reduce /
# hl.associative_scan (whose combine function pairs the inputs' elements) and
# the matmuls with an accumulator, which the SIMT fallback adds per thread.
REBOUND_CHECK_TARGETS = frozenset(
    {
        torch.ops.aten.where.self,
        torch.ops.aten.stack.default,
        join,
        inline_asm_elementwise,
        _reduce,
        _associative_scan,
        torch.ops.aten.addmm.default,
        torch.ops.aten.baddbmm.default,
        hl_dot,
        torch.ops.aten.gather.default,
    }
)


_MATMUL_WITH_ACCUMULATOR = frozenset(
    {torch.ops.aten.addmm.default, torch.ops.aten.baddbmm.default, hl_dot}
)


def rebound_operands(node: Node) -> list[Node]:
    """The operands of ``node`` whose elements are combined per thread.

    A matmul's lhs / rhs have their own layout logic; only its accumulator
    (``addmm(acc, ...)``, ``baddbmm(acc, ...)``, ``hl.dot(..., acc=acc)``) is
    added element-wise to the product.
    """
    if node.target in _MATMUL_WITH_ACCUMULATOR:
        if node.target is hl_dot:
            acc = node.args[2] if len(node.args) > 2 else node.kwargs.get("acc")
        else:
            acc = node.args[0] if node.args else None
        return [acc] if isinstance(acc, Node) else []
    if node.target is torch.ops.aten.gather.default:
        # The index is read per thread and selects along ``dim`` of the input
        # the thread addresses by its own block coordinates.
        index = node.args[2] if len(node.args) > 2 else node.kwargs.get("index")
        return [index] if isinstance(index, Node) else []
    return list(node.all_input_nodes)


def rebound_consumer_dims(
    node: Node,
) -> tuple[list[int | torch.SymInt], bool] | None:
    """The dims an operand of ``node`` is compared against, and whether the
    operands are placed as a PointwiseLowering places them
    (``operand_positions``).

    Only a PointwiseLowering expands a lower-rank operand by block id
    (``TileDispatch.broadcast_expand_dims``) and reads a higher-rank one, the
    input of a reduction over extent-1 dims, without the reduced dims
    (``size1_reduction_positions``).  ``tl.where``, the inline asm,
    ``hl.join``, the combine functions of a tuple reduce / scan and the
    matmul accumulator receive their operands unexpanded and so right-align
    them: a stack's or join's operands meet the result without the stacked
    (last) dim, a tuple reduce or scan's inputs meet each other, a
    tuple-result inline asm's operands meet its (broadcast) outputs.  A
    single-input reduce or scan has nothing to re-bind against.
    """
    value = node.meta.get("val")
    target = node.target
    if target is torch.ops.aten.stack.default or target is join:
        if not isinstance(value, torch.Tensor):
            return None
        sizes: list[int | torch.SymInt] = [*value.shape]
        if target is join:
            dim = value.ndim - 1
        else:
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
            assert isinstance(dim, int)
            dim %= value.ndim
        del sizes[dim]
        return sizes, False
    if target is _reduce or target is _associative_scan:
        is_tuple_input = (
            node.args[4] if len(node.args) > 4 else node.kwargs.get("is_tuple_input")
        )
        if not is_tuple_input:
            return None
        first = next(
            (
                operand.meta["val"]
                for operand in node.all_input_nodes
                if isinstance(operand.meta.get("val"), torch.Tensor)
            ),
            None,
        )
        return None if first is None else ([*first.shape], False)
    if target is inline_asm_elementwise:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return ([*value.shape], False) if isinstance(value, torch.Tensor) else None
    if target is torch.ops.aten.gather.default:
        source = node.args[0] if node.args else None
        source_val = source.meta.get("val") if isinstance(source, Node) else None
        if not isinstance(source_val, torch.Tensor):
            return None
        return [*source_val.shape], False
    if not isinstance(value, torch.Tensor):
        return None
    if target is torch.ops.aten.where.self or target in _MATMUL_WITH_ACCUMULATOR:
        return [*value.shape], False
    return [*value.shape], True
