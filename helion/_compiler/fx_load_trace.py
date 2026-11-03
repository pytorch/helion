"""Analysis: trace an FX value through dtype casts back to the ``hl.load`` that
produced it.

Shared by the CuTe MMA emitters (operand provenance) and Metal's MPP matmul
rewrite, so neither backend imports the other.  Moved verbatim from
``helion/_compiler/cute/cute_mma.py``.
"""

from __future__ import annotations

import torch
from torch.fx.node import Node

TRACE_THROUGH_TARGETS = {
    torch.ops.prims.convert_element_type.default,
    # NOTE: permute is NOT included because the MMA pipeline reads
    # raw tensor data — tracing through permute would bypass the
    # data shuffle.  Permuted operands fall back to scalar codegen.
}


def trace_to_load(node: Node) -> Node | None:
    """Trace through dtype casts to the underlying load node."""
    from ..language import memory_ops

    cur = node
    while cur.op == "call_function" and cur.target is not memory_ops.load:
        if cur.target not in TRACE_THROUGH_TARGETS:
            return None
        input_nodes = [arg for arg in cur.args if isinstance(arg, Node)]
        if len(input_nodes) != 1:
            return None
        cur = input_nodes[0]
    if cur.op != "call_function" or cur.target is not memory_ops.load:
        return None
    return cur


def trace_to_load_tensor(node: Node) -> tuple[Node, str, torch.Tensor] | None:
    """Trace through dtype casts to find the underlying load tensor."""
    load_node = trace_to_load(node)
    if load_node is None:
        return None
    tensor_node = load_node.args[0]
    if not isinstance(tensor_node, Node):
        return None
    fake = tensor_node.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    return load_node, tensor_node.name, fake
