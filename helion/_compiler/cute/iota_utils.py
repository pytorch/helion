from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _tracing_ops

if TYPE_CHECKING:
    from torch.fx.node import Node

    from ..generate_ast import GenerateAST


def _is_atomic_tensor_index_iota_user(source_node: Node, user: Node) -> bool:
    if user.op != "call_function" or not callable(user.target):
        return False
    target_name = getattr(user.target, "__name__", "")
    if not target_name.startswith("atomic_") or len(user.args) < 2:
        return False
    index_arg = user.args[1]
    return (
        isinstance(index_arg, (list, tuple))
        and len(index_arg) == 1
        and index_arg[0] is source_node
    )


def cute_iota_has_atomic_tensor_index_only_users(
    source_node: Node,
    cg: GenerateAST,
    *,
    _visited: set[Node] | None = None,
) -> bool:
    from ..device_ir import ForLoopGraphInfo

    visited = set() if _visited is None else _visited
    if source_node in visited:
        return False
    visited.add(source_node)

    users = list(source_node.users)
    if len(users) != 1:
        return False
    (user,) = users
    if _is_atomic_tensor_index_iota_user(source_node, user):
        return True
    if (
        user.op != "call_function"
        or not _tracing_ops.is_for_loop_target(user.target)
        or not user.args
        or not isinstance(user.args[0], int)
    ):
        return False

    graph_info = cg.get_graph(user.args[0])
    if not isinstance(graph_info, ForLoopGraphInfo):
        return False

    matched_placeholders = [
        placeholder
        for placeholder, outer_node in zip(
            graph_info.graph.find_nodes(op="placeholder"),
            graph_info.node_args,
            strict=True,
        )
        if outer_node is source_node
    ]
    if len(matched_placeholders) != 1:
        return False
    return cute_iota_has_atomic_tensor_index_only_users(
        matched_placeholders[0],
        cg,
        _visited=visited,
    )
