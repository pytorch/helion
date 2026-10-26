"""Expose exact builtin combine functions to ordinary reduction planning."""

from __future__ import annotations

from itertools import starmap
import operator
from typing import TYPE_CHECKING

import torch
from torch._inductor.ir import Reduction

from ...language._tracing_ops import _mask_to
from ...language.reduce_ops import _reduce

if TYPE_CHECKING:
    from torch.fx.node import Argument

    from ..device_ir import DeviceIR


_BUILTINS = {
    torch.ops.aten.add.Tensor: torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.mul.Tensor: torch.ops.aten.prod.dim_int,
    torch.ops.aten.maximum.default: torch.ops.aten.amax.default,
    torch.ops.aten.minimum.default: torch.ops.aten.amin.default,
}

_REDUCTION_TYPES = {
    torch.ops.aten.sum.dim_IntList: "sum",
    torch.ops.aten.prod.dim_int: "prod",
    torch.ops.aten.amax.default: "max",
    torch.ops.aten.amin.default: "min",
}


def _has_identity_padding(value: torch.fx.Node, target: torch._ops.OpOverload) -> bool:
    fake = value.meta["val"]
    assert isinstance(fake, torch.Tensor)
    return (
        value.op == "call_function"
        and value.target is _mask_to
        and isinstance(value.args[1], (int, float, bool))
        and value.args[1]
        == Reduction.default_accumulator(_REDUCTION_TYPES[target], fake.dtype)
    )


def _independent_combines(
    graph: torch.fx.Graph, arity: int
) -> list[torch._ops.OpOverload] | None:
    placeholders = list(graph.find_nodes(op="placeholder"))
    outputs = list(graph.find_nodes(op="output"))
    if len(placeholders) != 2 * arity or len(outputs) != 1:
        return None
    result = outputs[0].args[0]
    values = list(result) if isinstance(result, (tuple, list)) else [result]
    if len(values) != arity:
        return None
    combines = []
    for index, value in enumerate(values):
        if (
            not isinstance(value, torch.fx.Node)
            or value.op != "call_function"
            or value.target not in _BUILTINS
            or value.kwargs
            or value.args
            not in (
                (placeholders[index], placeholders[index + arity]),
                (placeholders[index + arity], placeholders[index]),
            )
        ):
            return None
        combines.append(_BUILTINS[value.target])
    # Prove the complete graph, not just its final operator. Extra computation
    # can change the combine semantics or have side effects.
    if set(graph.nodes) != set(placeholders + values + outputs):
        return None
    return combines


def canonicalize_reductions(ir: DeviceIR) -> None:
    """Let builtin hl.reduce use the same masks, carries and layouts as aten."""
    for info in ir.graphs:
        graph = info.graph
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target is not _reduce:
                continue
            combine_id, inputs, dim, keep_dims, is_tuple = node.args
            if not isinstance(combine_id, int) or not isinstance(dim, int):
                continue
            values = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
            if any(not isinstance(value, torch.fx.Node) for value in values):
                continue
            combines = _independent_combines(ir.graphs[combine_id].graph, len(values))
            if combines is None:
                continue
            # hl.reduce's explicit `other` can intentionally contribute on
            # padded lanes. Ordinary reduction planning applies its identity;
            # only canonicalize when that preserves the existing padding.
            if not all(
                starmap(_has_identity_padding, zip(values, combines, strict=True))
            ):
                continue
            users = list(node.users)
            if is_tuple and any(
                user.op != "call_function"
                or user.target is not operator.getitem
                or len(user.args) != 2
                or user.args[0] is not node
                or not isinstance(user.args[1], int)
                or not 0 <= user.args[1] < len(values)
                for user in users
            ):
                continue
            result_metadata = (
                {user.args[1]: user.meta for user in users}
                if is_tuple
                else {0: node.meta}
            )
            replacements = {}
            with graph.inserting_before(node):
                for index, (value, target) in enumerate(
                    zip(values, combines, strict=True)
                ):
                    if index not in result_metadata:
                        continue
                    assert isinstance(value, torch.fx.Node)
                    fake = value.meta["val"]
                    assert isinstance(fake, torch.Tensor)
                    axis = dim if target is torch.ops.aten.prod.dim_int else [dim]
                    kwargs: dict[str, Argument] = (
                        {"dtype": fake.dtype}
                        if target
                        in (torch.ops.aten.sum.dim_IntList, torch.ops.aten.prod.dim_int)
                        else {}
                    )
                    replacement = graph.call_function(
                        target, (value, axis, keep_dims), kwargs
                    )
                    replacement.meta = dict(result_metadata[index])
                    replacement.meta["orig_node"] = replacement
                    replacement.meta["original_aten"] = target
                    replacements[index] = replacement
            if is_tuple:
                for user in users:
                    user.replace_all_uses_with(replacements[user.args[1]])
                    graph.erase_node(user)
            else:
                node.replace_all_uses_with(replacements[0])
            graph.erase_node(node)
        graph.lint()
