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


# Per-thread pointwise / reshape ops that keep each thread's scalar element in
# place. A free ``hl.arange`` feeding such an op (e.g. ``start + arange`` or
# ``arange < end`` for the bounds mask) still describes a per-lane coordinate,
# so we look *through* these to confirm the arange ultimately lands in a
# load/store index (or its mask).
def _iota_index_passthrough_target(target: object) -> bool:
    import torch

    name = getattr(target, "__name__", "")
    if name == "getitem":
        return True
    target_str = str(target)
    return any(
        op in target_str
        for op in (
            "add.",
            "sub.",
            "mul.",
            "div.",
            "lt.",
            "le.",
            "gt.",
            "ge.",
            "eq.",
            "ne.",
            "remainder.",
            "bitwise_and.",
            "bitwise_or.",
            "__and__",
            "__or__",
            "expand.",
            "unsqueeze.",
            "view.",
            "reshape.",
            "_unsafe_view.",
            "convert_element_type.",
            "_to_copy.",
        )
    ) or target in (
        torch.ops.aten.expand.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    )


def _is_memory_op_index_user(source_node: Node, user: Node) -> bool:
    """True when ``source_node`` appears in a load/store's index list."""
    from ...language import memory_ops

    if user.op != "call_function" or user.target not in (
        memory_ops.load,
        memory_ops.store,
    ):
        return False
    if len(user.args) < 2:
        return False
    index_arg = user.args[1]
    if not isinstance(index_arg, (list, tuple)):
        return False
    return any(entry is source_node for entry in index_arg)


def cute_iota_is_free_memory_index(
    source_node: Node,
    cg: GenerateAST,
    *,
    _visited: set[Node] | None = None,
) -> bool:
    """True when a free ``hl.arange`` iota ultimately indexes a load/store.

    Recognizes the unbound-arange-index pattern: an iota whose value flows
    (possibly through per-lane pointwise/reshape ops, mask comparisons, or a
    for-loop placeholder) into a ``memory_ops.load``/``memory_ops.store`` index
    list. This is the gate for mapping the arange onto a synthetic thread axis.
    """

    visited = set() if _visited is None else _visited
    if source_node in visited:
        return False
    visited.add(source_node)

    for user in source_node.users:
        if _is_memory_op_index_user(source_node, user):
            return True
        if user.op != "call_function":
            continue
        if _is_for_loop_placeholder_index_user(source_node, user, cg, visited):
            return True
        if _iota_index_passthrough_target(
            user.target
        ) and cute_iota_is_free_memory_index(user, cg, _visited=visited):
            return True
    return False


def _is_for_loop_placeholder_index_user(
    source_node: Node,
    user: Node,
    cg: GenerateAST,
    visited: set[Node],
) -> bool:
    from ..device_ir import ForLoopGraphInfo

    if (
        not _tracing_ops.is_for_loop_target(user.target)
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
    return any(
        cute_iota_is_free_memory_index(placeholder, cg, _visited=visited)
        for placeholder in matched_placeholders
    )


def cute_free_arange_indexed_dim_key(
    source_node: Node,
    cg: GenerateAST,
    *,
    _visited: set[Node] | None = None,
) -> object | None:
    """Return a stable key for the tensor dim a free ``hl.arange`` indexes.

    The key is the (stringified) size of the tensor dimension the arange lands
    in. Two arange dims that address the *same* logical dimension (e.g. the load
    and store ``hl.arange(k)`` over a K-sized axis) yield the same key and
    therefore share one synthetic thread axis, while a cartesian ``row``/``col``
    pair addressing differently-sized dims gets distinct keys (distinct axes).
    Returns ``None`` when no load/store index consumer is found.
    """
    visited = set() if _visited is None else _visited
    if source_node in visited:
        return None
    visited.add(source_node)

    for user in source_node.users:
        key = _memory_op_indexed_dim_key(source_node, user)
        if key is not None:
            return key
        if user.op != "call_function":
            continue
        placeholder_key = _for_loop_placeholder_dim_key(source_node, user, cg, visited)
        if placeholder_key is not None:
            return placeholder_key
        if _iota_index_passthrough_target(user.target):
            downstream = cute_free_arange_indexed_dim_key(user, cg, _visited=visited)
            if downstream is not None:
                return downstream
    return None


def _memory_op_indexed_dim_key(source_node: Node, user: Node) -> object | None:
    import torch
    from torch.fx.node import Node as FxNode

    from ...language import memory_ops

    if user.op != "call_function" or user.target not in (
        memory_ops.load,
        memory_ops.store,
    ):
        return None
    if len(user.args) < 2:
        return None
    index_arg = user.args[1]
    if not isinstance(index_arg, (list, tuple)):
        return None
    tensor_node = user.args[0]
    if not isinstance(tensor_node, FxNode):
        return None
    tensor_val = tensor_node.meta.get("val")
    if not isinstance(tensor_val, torch.Tensor):
        return None
    tensor_dim = 0
    for entry in index_arg:
        if entry is None:
            # ``None`` introduces a new broadcast dim; it does not consume a
            # tensor dimension.
            continue
        if entry is source_node:
            if tensor_dim >= tensor_val.ndim:
                return None
            # Key on (index position, dim size). The position disambiguates two
            # distinct free arange nodes that co-occur as different entries of a
            # load/store index list but address equal-sized dims (e.g. a square
            # cartesian ``out[arange(N), arange(N)]``): without the position they
            # would share one synthetic thread axis and only the diagonal would
            # be written. A single arange node reused across a load and a store
            # still resolves to one key (one axis), preserving the roundtrip.
            return (tensor_dim, str(tensor_val.shape[tensor_dim]))
        tensor_dim += 1
    return None


def _for_loop_placeholder_dim_key(
    source_node: Node,
    user: Node,
    cg: GenerateAST,
    visited: set[Node],
) -> object | None:
    from ..device_ir import ForLoopGraphInfo

    if (
        not _tracing_ops.is_for_loop_target(user.target)
        or not user.args
        or not isinstance(user.args[0], int)
    ):
        return None
    graph_info = cg.get_graph(user.args[0])
    if not isinstance(graph_info, ForLoopGraphInfo):
        return None
    for placeholder, outer_node in zip(
        graph_info.graph.find_nodes(op="placeholder"),
        graph_info.node_args,
        strict=True,
    ):
        if outer_node is source_node:
            key = cute_free_arange_indexed_dim_key(placeholder, cg, _visited=visited)
            if key is not None:
                return key
    return None


def cute_free_arange_compacted_tile_begin_factor(
    source_node: Node,
    cg: GenerateAST,
) -> tuple[int, int] | None:
    """Detect ``out[tile.begin + hl.arange(block // F)] = compacted_tile``.

    Returns ``(block_id, factor)`` when ``source_node`` is a free ``hl.arange``
    whose only consumer is an ``add(arange, tile.begin)`` that feeds a load/store
    index list, and whose length is the tile's block size divided by a constexpr
    factor ``F`` (i.e. the arange addresses a *compacted* sub-block tile). The
    arange must then resolve to the tile-LOCAL lane ``lane // F`` rather than the
    global ``index_var // F``: the global form already folds the tile's offset in,
    so adding ``tile.begin`` again double-counts it.

    Returns ``None`` (a strict no-op) unless the whole pattern matches, so every
    already-supported arange keeps its existing resolution.
    """
    import torch

    factor = _arange_block_split_factor(source_node)
    if factor is None:
        return None
    block_id, _ = factor

    users = list(source_node.users)
    if len(users) != 1:
        return None
    (add_node,) = users
    if (
        add_node.op != "call_function"
        or add_node.target is not torch.ops.aten.add.Tensor
    ):
        return None

    other = _add_sibling(add_node, source_node)
    if other is None or not _is_tile_begin_node(other, block_id):
        return None
    if not _add_feeds_memory_index(add_node):
        return None
    return factor


def _arange_block_split_factor(source_node: Node) -> tuple[int, int] | None:
    """Return ``(block_id, factor)`` when the arange length is ``block // F``."""
    import sympy
    import torch
    from torch.utils._sympy.functions import FloorDiv

    from ..compile_environment import CompileEnvironment

    fake_val = source_node.meta.get("val")
    if not isinstance(fake_val, torch.Tensor) or fake_val.ndim != 1:
        return None
    length = fake_val.shape[0]
    if not isinstance(length, torch.SymInt):
        return None
    expr = length._sympy_()
    if not isinstance(expr, FloorDiv) or len(expr.args) != 2:
        return None
    base, divisor = expr.args
    if not isinstance(base, sympy.Symbol) or not isinstance(divisor, sympy.Integer):
        return None
    factor = int(divisor)
    if factor < 2:
        return None
    block_id = CompileEnvironment.current().get_block_id(base)
    if block_id is None:
        return None
    return block_id, factor


def _add_sibling(add_node: Node, source_node: Node) -> Node | None:
    from torch.fx.node import Node as FxNode

    siblings = [
        arg
        for arg in add_node.args
        if isinstance(arg, FxNode) and arg is not source_node
    ]
    if len(siblings) != 1:
        return None
    return siblings[0]


def _is_tile_begin_node(node: Node, block_id: int) -> bool:
    """True when ``node`` is (a scalar arithmetic derivative of) ``tile.begin``.

    The store base may pre-scale the begin to match a compacted output, e.g.
    ``out[tile.begin // F + arange]``. We unwrap integer ``floordiv``/``mul``/
    ``add`` chains whose tensor operand is the tile's ``tile_begin`` for the same
    block id so both ``tile.begin + arange`` and ``tile.begin // F + arange``
    qualify.
    """
    import operator

    import torch
    from torch.fx.node import Node as FxNode

    from ...language.tile_ops import tile_begin
    from ..compile_environment import CompileEnvironment

    if node.op != "call_function" or not node.args:
        return False

    if node.target is tile_begin:
        tile_arg = node.args[0]
        if not isinstance(tile_arg, FxNode):
            return False
        tile_val = tile_arg.meta.get("val")
        if not isinstance(tile_val, torch.SymInt):
            return False
        return CompileEnvironment.current().get_block_id(tile_val) == block_id

    scalar_arith = {
        torch.ops.aten.floor_divide.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        operator.floordiv,
        operator.mul,
        operator.add,
        operator.sub,
    }
    if node.target in scalar_arith:
        return any(
            isinstance(arg, FxNode) and _is_tile_begin_node(arg, block_id)
            for arg in node.args
        )
    return False


def _add_feeds_memory_index(add_node: Node) -> bool:
    from ...language import memory_ops

    for user in add_node.users:
        if (
            user.op == "call_function"
            and user.target in (memory_ops.load, memory_ops.store)
            and len(user.args) >= 2
            and isinstance(user.args[1], (list, tuple))
            and any(entry is add_node for entry in user.args[1])
        ):
            return True
    return False
