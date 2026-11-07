"""Shared exact signed-field FX fixtures for CPU materialization tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

import torch
from torch.fx import Graph
from torch.fx import Node

from ..language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Callable


def field_node(
    graph: Graph,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    dtype: torch.dtype,
    kwargs: dict[str, Any] | None = None,
) -> Node:
    node = graph.call_function(target, args, kwargs or {})
    node.meta["val"] = torch.empty((2, 8), dtype=dtype)
    return node


def signed_field_graph(
    offset: int = 0,
    width: int = 4,
    *,
    identity: bool = False,
    shift_only: bool = False,
    wide: torch.dtype = torch.int32,
) -> SimpleNamespace:
    graph = Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty((2, 8), dtype=torch.int8)
    output = graph.placeholder("destination")
    output.meta["val"] = torch.empty((2, 8), dtype=torch.bfloat16)
    load = field_node(graph, memory_ops.load, (source, [0, 0]), torch.int8)
    current = load
    if offset:
        current = field_node(
            graph,
            torch.ops.aten.bitwise_right_shift.Tensor_Scalar,
            (current, offset),
            torch.int8,
        )
    if not identity and not shift_only:
        # The original INT4 source masks in INT8 and then widens.
        if width == 8:
            current = field_node(
                graph,
                torch.ops.aten._to_copy.default,
                (current,),
                wide,
                {"dtype": wide},
            )
        current = field_node(
            graph,
            torch.ops.aten.bitwise_and.Scalar,
            (current, (1 << width) - 1),
            current.meta["val"].dtype,
        )
    current = field_node(
        graph, torch.ops.aten._to_copy.default, (current,), wide, {"dtype": wide}
    )
    if not identity and not shift_only:
        compare = field_node(
            graph, torch.ops.aten.ge.Scalar, (current, 1 << (width - 1)), torch.bool
        )
        subtract = field_node(
            graph, torch.ops.aten.sub.Tensor, (current, 1 << width), wide
        )
        current = field_node(
            graph, torch.ops.aten.where.self, (compare, subtract, current), wide
        )
    value = field_node(
        graph,
        torch.ops.aten._to_copy.default,
        (current,),
        torch.bfloat16,
        {"dtype": torch.bfloat16},
    )
    store = field_node(graph, memory_ops.store, (output, [0, 0], value), torch.bfloat16)
    graph.output(())
    return SimpleNamespace(
        graph=graph, source=source, output=output, load=load, value=value, store=store
    )
