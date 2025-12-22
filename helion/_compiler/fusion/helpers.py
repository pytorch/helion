"""Helper utilities for fusion code."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch._inductor.ir import MultiOutput


def safe_get_name(node: Any) -> str | None:
    """Get buffer name from node if available.

    Args:
        node: Any object that might have a get_name() method.

    Returns:
        The buffer name string, or None if not available.
    """
    return node.get_name() if hasattr(node, "get_name") else None


def is_multi_output_node(node: Any) -> bool:
    """Check if a scheduler node wraps a MultiOutput IR node.

    Args:
        node: A scheduler node or IR node.

    Returns:
        True if the node is a MultiOutput node.
    """
    from torch._inductor.ir import MultiOutput

    inner = getattr(node, "node", node)
    return isinstance(inner, MultiOutput)


def partition_multi_output(nodes: list) -> tuple[list, list]:
    """Partition nodes into MultiOutput and non-MultiOutput groups.

    Args:
        nodes: List of scheduler nodes.

    Returns:
        Tuple of (multi_output_nodes, other_nodes).
    """
    multi_output = []
    other = []
    for n in nodes:
        if is_multi_output_node(n):
            multi_output.append(n)
        else:
            other.append(n)
    return multi_output, other


def get_node_buffer_names(nodes: list) -> set[str]:
    """Get all buffer names from a list of nodes.

    Args:
        nodes: List of nodes with potential get_name() methods.

    Returns:
        Set of buffer name strings.
    """
    names = set()
    for node in nodes:
        name = safe_get_name(node)
        if name:
            names.add(name)
    return names
