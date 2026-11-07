"""Shared operation boundaries for row-wise sorting and prefix selection."""

from __future__ import annotations

import operator

import torch
from torch.fx import Node


def is_ordered_selection(node: Node) -> bool:
    return node.target in (
        torch.ops.aten.topk.default,
        torch.ops.aten.sort.default,
        torch.ops.aten.sort.stable,
    )


def ordered_selection_args(
    node: Node,
) -> tuple[Node, int | None, int, bool, bool] | None:
    """Return the shared arguments of sorting and prefix selection.

    A missing count denotes a complete sort. Sorting always uses first-index
    ties, which also satisfies the default sort's unspecified tie ordering.
    Top-k may distinguish the two zero signs to permit exact value decoding.
    """
    if not is_ordered_selection(node) or not node.args:
        return None
    source = node.args[0]
    if not isinstance(source, Node):
        return None
    sorting = node.target in (torch.ops.aten.sort.default, torch.ops.aten.sort.stable)
    if sorting:
        if node.target is torch.ops.aten.sort.stable:
            if (
                len(node.args) != 1
                or set(node.kwargs) - {"stable", "dim", "descending"}
                or "stable" not in node.kwargs
                or node.kwargs["stable"] is not None
                and type(node.kwargs["stable"]) is not bool
            ):
                return None
        elif len(node.args) > 3 or set(node.kwargs) - {"dim", "descending"}:
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
    if type(dim) is not int or type(descending) is not bool:
        return None
    return source, k, dim, descending, sorting


def selection_args(node: Node) -> tuple[Node, int | None, bool, bool] | None:
    """Restrict shared selection arguments to a complete row's last axis."""
    arguments = ordered_selection_args(node)
    if arguments is None:
        return None
    source, k, dim, descending, sorting = arguments
    if dim not in (-1, 1):
        return None
    return source, k, descending, sorting


def selection_values_only(node: Node) -> bool:
    """Prove that the selection's tuple and original indices are unobserved.

    Every direct user must be the value getter in this graph. Shape queries,
    tuple returns, index getters and cross-graph escapes decline. This proof
    is independent of the value dtype and bounded-axis admission.
    """
    return (
        ordered_selection_args(node) is not None
        and bool(node.users)
        and all(
            user.graph is node.graph
            and user.target is operator.getitem
            and len(user.args) == 2
            and user.args[0] is node
            and type(user.args[1]) is int
            and user.args[1] == 0
            and not user.kwargs
            for user in node.users
        )
    )


def unordered_topk_args(node: Node) -> tuple[Node, int, int, bool] | None:
    """Bind a literal unordered top-k without borrowing sorting admission.

    The ordinary sorting owners may emit ordered results for either top-k mode.
    A compaction owner needs the explicit False rather than a schema default.
    """
    if node.target is not torch.ops.aten.topk.default:
        return None
    schema = node.target._schema.arguments
    if len(node.args) > len(schema) or set(node.kwargs) - {
        argument.name for argument in schema
    }:
        return None
    arguments: dict[str, object] = {}
    for position, argument in enumerate(schema):
        if position < len(node.args):
            if argument.name in node.kwargs:
                return None
            arguments[argument.name] = node.args[position]
        elif argument.name in node.kwargs:
            arguments[argument.name] = node.kwargs[argument.name]
        elif argument.has_default_value():
            arguments[argument.name] = argument.default_value
        else:
            return None
    source = arguments["self"]
    k, dim, largest = (
        arguments["k"],
        arguments["dim"],
        arguments["largest"],
    )
    if (
        not isinstance(source, Node)
        or type(k) is not int
        or type(dim) is not int
        or type(largest) is not bool
        or arguments["sorted"] is not False
    ):
        return None
    return source, k, dim, largest
