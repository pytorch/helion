"""Admission for warp aggregation of relaxed integer fetch-add tickets."""

from __future__ import annotations

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import creation_ops


def _boolean_increment(value: object) -> bool:
    """Prove that each increment is either integer zero or integer one."""
    while isinstance(value, Node):
        fake = value.meta.get("val")
        if not isinstance(fake, torch.Tensor):
            return False
        if fake.dtype == torch.bool:
            return True
        if value.target in (
            _tracing_ops._new_var,
            torch.ops.aten.alias.default,
            torch.ops.prims.convert_element_type.default,
        ):
            value = value.args[0]
        elif value.target is creation_ops.full:
            value = value.args[1]
        else:
            return False
    return type(value) in (int, bool) and value in (0, 1)


def warp_atomic_candidate(node: Node | None) -> bool:
    """Only integer tickets with relaxed ordering may combine transactions.

    A group returns the old counter plus the number of earlier live increments,
    so every lane observes a valid serial ordering, including zero increments.
    Stronger memory ordering needs one fence-carrying transaction per lane.
    """
    if (
        node is None
        or node.target is not atomic_ops.atomic_add
        or len(node.args) != 4
        or node.args[3] != "relaxed"
        or not isinstance(node.args[0], Node)
    ):
        return False
    target = node.args[0].meta.get("val")
    return (
        isinstance(target, torch.Tensor)
        and target.dtype == torch.int32
        and _boolean_increment(node.args[2])
    )
