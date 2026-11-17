"""Fold a K loop's ``acc = acc + mm(a, b)`` into ``addmm(acc, a, b)``.

Metal's matmul planner accumulates a K loop's products into the loop's
carried acc (``mpp_matmul``): a matmul with no acc over a tiled K would give
each K tile's product alone.  A body that adds the product to the carry
itself, ``acc = acc + torch.matmul(a, b)``, means the same accumulation, so
the add and the product become one ``addmm`` (``baddbmm`` for a batched
product) before the graphs get their lowerings.

The fold applies only where it changes no value: the product, the carry and
the sum share a dtype and a shape, so no rounding of the product to a
narrower dtype is skipped and nothing is broadcast; and the product has no
other reader.  And only where the loop walks the product's K: a product
over a whole K in a loop over something else stays a sum of its own.
"""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo

if TYPE_CHECKING:
    from ..device_ir import DeviceIR

_ACCUMULATING = {
    torch.ops.aten.mm.default: torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default: torch.ops.aten.baddbmm.default,
}


def _is_carry(node: object) -> bool:
    """Whether ``node`` is a loop body's copy of a carried value."""
    return (
        isinstance(node, torch.fx.Node)
        and node.target is _tracing_ops._new_var
        and isinstance(node.args[0], torch.fx.Node)
        and node.args[0].op == "placeholder"
    )


def fold_accumulating_matmuls(device_ir: DeviceIR) -> None:
    for info in device_ir.graphs:
        if not isinstance(info, ForLoopGraphInfo):
            continue
        graph = info.graph
        for node in [*graph.nodes]:
            if node.target is not torch.ops.aten.add.Tensor or node.kwargs:
                continue
            if len(node.args) != 2:
                continue
            for acc, product in (node.args, node.args[::-1]):
                if (
                    _is_carry(acc)
                    and isinstance(product, torch.fx.Node)
                    and product.target in _ACCUMULATING
                    and len(product.users) == 1
                    and _same_value_type(node, acc, product)
                    and _walks_k(info, product)
                ):
                    with graph.inserting_before(node):
                        fused = graph.call_function(
                            _ACCUMULATING[product.target],
                            (acc, *product.args),
                        )
                    fused.meta = {**node.meta}
                    node.replace_all_uses_with(fused)
                    graph.erase_node(node)
                    graph.erase_node(product)
                    break


def _walks_k(info: ForLoopGraphInfo, product: torch.fx.Node) -> bool:
    """Whether the loop walks the product's K: K's extent is one of the
    loop's tiles, or a size computed from one (a widened slice).  A product
    over a K the loop doesn't tile (a ':' K in a loop over a batch) is a
    whole sum each step, added to the carry as it stands."""
    env = CompileEnvironment.current()
    lhs = product.args[0]
    assert isinstance(lhs, torch.fx.Node)
    k = lhs.meta["val"].shape[-1]
    if not isinstance(k, torch.SymInt):
        return False
    tiles = {env.block_sizes[b].var.node.expr for b in info.block_ids}
    return bool(k.node.expr.free_symbols & tiles)


def _same_value_type(*nodes: torch.fx.Node) -> bool:
    env = CompileEnvironment.current()
    values = [n.meta.get("val") for n in nodes]
    tensors = [v for v in values if isinstance(v, torch.Tensor)]
    if len(tensors) != len(values):
        return False
    first = tensors[0]
    return all(
        v.dtype == first.dtype
        and v.ndim == first.ndim
        and all(starmap(env.known_equal, zip(v.shape, first.shape, strict=True)))
        for v in tensors
    )
