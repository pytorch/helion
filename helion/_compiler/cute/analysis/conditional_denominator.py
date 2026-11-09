"""The authored reduction tree shared by conditional attention certificates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


def require_denominator_tree(
    value: object,
    lanes: object,
    rows: int,
    *,
    call: Callable[[object, object, int], tuple],
    same: Callable[[object, object], None],
    require: Callable[[bool], None],
) -> None:
    """Check the exact tree using the owner's resolution and refusal rules.

    ``call`` must check the operation, arity and kwargs after resolving its
    input; ``same`` compares resolved identities. Those callbacks retain each
    owner's capture handling and record every node consumed by this proof.
    """
    pieces = []
    for group in (3, 2, 1):
        value, selected = call(value, torch.ops.aten.add.Tensor, 2)
        source, axis, index = call(selected, torch.ops.aten.select.int, 3)
        require((axis, index) == (1, group))
        pieces.append(source)
    source, axis, index = call(value, torch.ops.aten.select.int, 3)
    require((axis, index) == (1, 0))
    for piece in pieces:
        same(source, piece)
    left, right = call(source, torch.ops.aten.add.Tensor, 2)
    previous, axis, index = call(left, torch.ops.aten.select.int, 3)
    peer, peer_axis, peer_index = call(right, torch.ops.aten.select.int, 3)
    same(previous, peer)
    require((axis, index, peer_axis, peer_index) == (2, 0, 2, 1))
    for width in (2, 4, 8, 16):
        left, right = call(previous, torch.ops.aten.add.Tensor, 2)
        previous, axis, start, end = call(left, torch.ops.aten.slice.Tensor, 4)
        peer, peer_axis, peer_start, peer_end = call(
            right, torch.ops.aten.slice.Tensor, 4
        )
        same(previous, peer)
        require((axis, start, end) == (2, 0, width))
        require((peer_axis, peer_start, peer_end) == (2, width, 2**63 - 1))
    previous, sizes = call(previous, torch.ops.aten.view.default, 2)
    require(sizes == [rows, 4, 32])
    same(previous, lanes)
