"""Typed current-call signature and simultaneous scalar carry contracts."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise exc.InvalidConfig(f"resident while plan: {reason}")


def _signature(value: Node) -> tuple[torch.dtype, tuple[int, ...]]:
    fake = value.meta.get("val")
    _require(isinstance(fake, torch.Tensor), "tensor metadata required")
    fake = cast("torch.Tensor", fake)
    _require(
        all(type(size) is int and size > 0 for size in fake.shape),
        "positive static physical shape required",
    )
    return fake.dtype, cast("tuple[int, ...]", tuple(fake.shape))


def _outputs(info: GraphInfo) -> tuple[Node, ...]:
    nodes = list(info.graph.find_nodes(op="output"))
    _require(len(nodes) == 1, "one graph output required")
    values = nodes[0].args[0]
    _require(
        isinstance(values, (list, tuple))
        and all(isinstance(value, Node) for value in values),
        "tensor output list required",
    )
    return tuple(cast("Sequence[Node]", values))


def _carry_map(
    call: Node, captures: tuple[Node, ...], outputs: tuple[Node, ...]
) -> tuple[tuple[int, int], ...]:
    slots: dict[int, int] = {}
    order = {node: index for index, node in enumerate(call.graph.nodes)}
    _require(call in order, "loop call is absent from caller graph")
    for item in call.users:
        _require(
            item.op == "call_function"
            and item.graph is call.graph
            and item in order
            and order[item] > order[call]
            and item.target is operator.getitem
            and len(item.args) == 2
            and not item.kwargs
            and item.args[0] is call
            and type(item.args[1]) is int
            and 0 <= item.args[1] < len(outputs),
            "invalid loop output projection",
        )
        index = cast("int", item.args[1])
        for phi in item.users:
            _require(
                phi.op == "call_function"
                and phi.graph is call.graph
                and phi in order
                and order[phi] > order[item]
                and phi.target is _tracing_ops._phi
                and len(phi.args) == 2
                and not phi.kwargs
                and phi.args[1] is item,
                "output must have an explicit initialized phi",
            )
            matches = [i for i, value in enumerate(captures) if value is phi.args[0]]
            _require(len(matches) == 1, "ambiguous phi entry capture")
            slot = matches[0]
            _require(index not in slots or slots[index] == slot, "conflicting phi")
            _require(
                _signature(outputs[index]) == _signature(captures[slot]),
                "carry physical shape or dtype changed",
            )
            _require(
                _signature(item) == _signature(outputs[index]),
                "projection physical shape or dtype changed",
            )
            _require(
                _signature(phi) == _signature(captures[slot]),
                "phi physical shape or dtype changed",
            )
            slots[index] = slot
    _require(set(slots) == set(range(len(outputs))), "uninitialized loop output")
    _require(len(set(slots.values())) == len(slots), "duplicate carry destination")
    return tuple(sorted(slots.items()))
