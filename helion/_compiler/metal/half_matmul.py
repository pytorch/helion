"""Lower a half-precision matmul as a float32 product rounded once.

``torch.matmul`` of two float16 (or bfloat16) operands returns their dtype,
but accumulates in float32 and rounds the finished sum: the same value as
``hl.dot(a, b, out_dtype=torch.float32).to(torch.float16)``.  Metal's matmul
planner accumulates in float32 only (``mpp_matmul``), so before the graphs get
their lowerings each ``mm`` or ``bmm`` with a half-precision result becomes
its float32 product and the rounding cast after it: hl.dot's own float32
form for an ``mm``, and ``bmm.dtype`` (the batched form the planner takes
without an acc) for a ``bmm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language import matmul_ops

if TYPE_CHECKING:
    from ..device_ir import DeviceIR

_HALF = (torch.float16, torch.bfloat16)


def round_half_matmuls(device_ir: DeviceIR) -> None:
    for info in device_ir.graphs:
        graph = info.graph
        for node in [*graph.nodes]:
            if node.target not in (
                torch.ops.aten.mm.default,
                torch.ops.aten.bmm.default,
            ):
                continue
            value = node.meta.get("val")
            lhs, rhs = node.args
            assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype not in _HALF
                or lhs.meta["val"].dtype != value.dtype
                or rhs.meta["val"].dtype != value.dtype
            ):
                continue
            with graph.inserting_before(node):
                product = (
                    graph.call_function(
                        torch.ops.aten.bmm.dtype, (lhs, rhs, torch.float32)
                    )
                    if node.target is torch.ops.aten.bmm.default
                    else graph.call_function(
                        matmul_ops.dot, (lhs, rhs, None, torch.float32)
                    )
                )
                product.meta = {
                    **node.meta,
                    "val": value.new_empty(value.shape, dtype=torch.float32),
                }
                rounded = graph.call_function(
                    torch.ops.prims.convert_element_type.default,
                    (product, value.dtype),
                )
                rounded.meta = {**node.meta}
            node.replace_all_uses_with(rounded)
            graph.erase_node(node)
