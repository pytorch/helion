"""The existing closed additive helper grammar for computed-fragment scans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..device_ir import HelperFunctionGraphInfo


def additive_scan_helper(helper: HelperFunctionGraphInfo) -> bool:
    nodes = list(helper.graph.nodes)
    if len(nodes) != 4:
        return False
    lhs, rhs, add, output = nodes
    return not (
        lhs.op != "placeholder"
        or rhs.op != "placeholder"
        or add.target is not torch.ops.aten.add.Tensor
        or add.args not in ((lhs, rhs), (rhs, lhs))
        or add.kwargs.get("alpha", 1) != 1
        or output.op != "output"
        or output.args != (add,)
    )
