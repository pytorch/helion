"""Remove lossless operand conversions before fast-math dots on Hopper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language._tracing_ops import _mask_to
from ...language.matmul_ops import MATMUL_FACT_ID_META
from ...language.matmul_ops import dot

if TYPE_CHECKING:
    from ...autotuner.config_spec import MatmulFact

_NARROW_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
}


def _dtype(node: object) -> torch.dtype | None:
    if isinstance(node, torch.fx.Node):
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value.dtype
    return None


def _source(
    node: object, dot_node: torch.fx.Node
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node | None] | None:
    if not isinstance(node, torch.fx.Node):
        return None
    mask = None
    if node.op == "call_function" and node.target is _mask_to:
        if (
            len(node.args) != 2
            or node.args[1] != 0
            or node.kwargs
            or set(node.users) != {dot_node}
        ):
            return None
        mask = node
        node = node.args[0]
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target != torch.ops.prims.convert_element_type.default
        or len(node.args) != 2
        or node.args[1] != torch.float32
        or node.kwargs
    ):
        return None
    source = node.args[0]
    if isinstance(source, torch.fx.Node) and _dtype(source) in _NARROW_DTYPES:
        return source, node, mask
    return None


def narrow_dot_operands(graph: torch.fx.Graph, facts: list[MatmulFact]) -> int:
    """Use the original dtype for paired FP32 upcasts, retaining FP32 output."""
    changed = 0
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not dot
            or len(node.args) != 4
            or node.kwargs
            or _dtype(node) != torch.float32
            or node.args[3] not in (None, torch.float32)
            or (node.args[2] is not None and _dtype(node.args[2]) != torch.float32)
        ):
            continue
        lhs = _source(node.args[0], node)
        rhs = _source(node.args[1], node)
        fact_id = node.meta.get(MATMUL_FACT_ID_META)
        if lhs is None or rhs is None or _dtype(lhs[0]) != _dtype(rhs[0]):
            continue
        if not isinstance(fact_id, int) or not 0 <= fact_id < len(facts):
            continue
        dtype = _dtype(lhs[0])
        assert dtype is not None
        operands = []
        for source, _, mask in (lhs, rhs):
            if mask is not None:
                mask.args = (source, 0)
                mask.meta["val"] = source.meta["val"]
            operands.append(mask if mask is not None else source)
        node.args = (*operands, *node.args[2:])
        facts[fact_id] = facts[fact_id]._replace(lhs_dtype=dtype, rhs_dtype=dtype)
        for cast in {lhs[1], rhs[1]}:
            if not cast.users:
                graph.erase_node(cast)
        changed += 1
    return changed
