"""Shared operation boundaries for row-wise sorting and prefix selection."""

from __future__ import annotations

import torch
from torch.fx import Node


def is_ordered_selection(node: Node) -> bool:
    return node.target in (torch.ops.aten.topk.default, torch.ops.aten.sort.default)


def selection_args(node: Node) -> tuple[Node, int | None, bool, bool] | None:
    """Return source, selected count, descending order, and stable tie policy.

    A missing count denotes a complete sort. Sorting always uses first-index
    ties, which also satisfies the default sort's unspecified tie ordering.
    Top-k may distinguish the two zero signs to permit exact value decoding.
    """
    if not is_ordered_selection(node) or not node.args:
        return None
    source = node.args[0]
    if not isinstance(source, Node):
        return None
    sorting = node.target is torch.ops.aten.sort.default
    if sorting:
        if len(node.args) > 3 or set(node.kwargs) - {"dim", "descending"}:
            return None
        k = None
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        descending = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
        )
    else:
        if not 2 <= len(node.args) <= 5 or node.kwargs:
            return None
        k = node.args[1]
        if type(k) is not int:
            return None
        dim = node.args[2] if len(node.args) > 2 else -1
        descending = node.args[3] if len(node.args) > 3 else True
    if type(dim) is not int or dim not in (-1, 1) or type(descending) is not bool:
        return None
    return source, k, descending, sorting
