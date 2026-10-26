from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx.node import Node

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...language import view_ops
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_expr
from ..host_function import HostFunction
from ..variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST

    _Slot = tuple[Node, int]


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


def cute_free_arange_memory_index_positions(
    source_node: Node,
    *,
    _visited: set[Node] | None = None,
) -> list[tuple[Node, int]]:
    """Every ``(load/store node, index position)`` a free arange addresses."""
    visited = set() if _visited is None else _visited
    if source_node in visited:
        return []
    visited.add(source_node)
    positions: list[tuple[Node, int]] = []
    for user in source_node.users:
        if _is_memory_op_index_user(source_node, user):
            index_arg = user.args[1]
            assert isinstance(index_arg, (list, tuple))
            positions.extend(
                (user, position)
                for position, entry in enumerate(index_arg)
                if entry is source_node
            )
        elif user.op == "call_function" and _iota_index_passthrough_target(user.target):
            positions.extend(
                cute_free_arange_memory_index_positions(user, _visited=visited)
            )
    return positions


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


# Index-shape dims that are not an arange lane: a size-one dim, or any other
# axis (a tile, a slice, a gather).
_ONE = "one"
_AXIS = "axis"

_LANE_PASSTHROUGH_TARGETS = frozenset(
    {
        _tracing_ops._new_var,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
    }
)


class FreeArangeLanes:
    """Positional lane classes of the ``hl.arange`` dims of a kernel's values.

    Helion values are positional: a store writes element ``j`` of its value at
    the ``j``-th coordinate of its index, a load's value dims follow its index
    dims, and pointwise operands meet at equal positions counted from the
    right.  A free arange (bound to no tile axis) gets a synthetic thread
    axis, so every arange dim that meets another (a load's index dim carried
    to a store's index, two loaded values added, a permute in between) must
    take the same axis whatever its start and step, and two dims of one value
    distinct axes.  This unions the dims that meet through the ops it models
    (pointwise ops, ``[:, None]``-style subscripts, permute, unsqueeze, expand,
    loads and stores with one-dimensional arange indexes).  An arange-derived
    value reaching any other op leaves the kernel unmodeled: ``root`` is then
    ``None`` and callers keep their size-based keys.
    """

    def __init__(self, graphs: Iterable[GraphInfo]) -> None:
        self._parent: dict[_Slot, _Slot] = {}
        self._derived: set[Node] = set()
        self._store_lanes: list[list[_Slot]] = []
        self.complete = all(self._visit_graph(info.graph) for info in graphs)
        # Two dims of one value or one access in a single class need two lane
        # coordinates of one axis: ``r1``/``r2`` merged by ``a[r1] + a[r2]``
        # and then indexing as a cartesian pair (``x[p[r1], q[r2]]``), or one
        # arange gathering two dims (``x[p[r], q[r]]``).  The classes cannot
        # lower it, and the size-based keys get the merged statement wrong.
        self.shares_lane = self.complete and any(
            len({self._find(slot) for slot in slots}) != len(slots)
            for slots in self._lane_groups()
        )

    def root(self, node: Node) -> _Slot | None:
        """The class of the last dim of ``node`` (an iota or a value of one)."""
        slot = (node, -1)
        if not self.complete or slot not in self._parent:
            return None
        if self.shares_lane:
            raise exc.BackendUnsupported(
                "cute",
                "two dims of one value or load/store share a free hl.arange "
                "lane; the SIMT lowering addresses one lane coordinate per "
                "thread axis",
            )
        return self._find(slot)

    def _lane_groups(self) -> Iterator[list[_Slot]]:
        """The arange lanes of each derived value and of each store's index."""
        for node in self._derived:
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                yield [
                    (node, -i)
                    for i in range(1, value.ndim + 1)
                    if self._is_lane((node, -i))
                ]
        yield from self._store_lanes

    def _find(self, slot: _Slot) -> _Slot:
        parent = self._parent.setdefault(slot, slot)
        while parent != slot:
            grandparent = self._parent[parent]
            self._parent[slot] = grandparent
            slot, parent = parent, grandparent
        return slot

    def _union(self, a: _Slot, b: _Slot) -> None:
        self._parent[self._find(a)] = self._find(b)

    def _is_lane(self, slot: _Slot) -> bool:
        return slot in self._parent

    def _visit_graph(self, graph: torch.fx.Graph) -> bool:
        for node in graph.nodes:
            if node.op == "output":
                if any(arg in self._derived for arg in node.all_input_nodes):
                    return False
                continue
            if node.op != "call_function":
                continue
            if node.target is torch.ops.prims.iota.default:
                self._find((node, -1))
                self._derived.add(node)
                continue
            if not any(arg in self._derived for arg in node.all_input_nodes):
                continue
            if not self._visit(node):
                return False
        return True

    def _visit(self, node: Node) -> bool:
        target = node.target
        if target is memory_ops.load:
            return self._visit_load(node)
        if target is memory_ops.store:
            return self._visit_store(node)
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return False
        self._derived.add(node)
        if target in _LANE_PASSTHROUGH_TARGETS or (
            isinstance(target, torch._ops.OpOverload)
            and torch.Tag.pointwise in target.tags
        ):
            return self._visit_pointwise(node, value)
        source = node.args[0]
        if not isinstance(source, Node):
            return False
        source_value = source.meta.get("val")
        if not isinstance(source_value, torch.Tensor):
            return False
        pairs: list[tuple[int, int]] = []
        if target is view_ops.subscript:
            entries = node.args[1]
            if not isinstance(entries, (list, tuple)):
                return False
            source_dim = output_dim = 0
            for entry in entries:
                if entry is None:
                    output_dim += 1
                elif isinstance(entry, slice) and entry == slice(None):
                    pairs.append((source_dim, output_dim))
                    source_dim += 1
                    output_dim += 1
                else:
                    return False
            pairs.extend(
                (source_dim + i, output_dim + i)
                for i in range(source_value.ndim - source_dim)
            )
        elif target is torch.ops.aten.permute.default:
            dims = node.args[1]
            assert isinstance(dims, (list, tuple))
            pairs = []
            for output_dim, dim in enumerate(dims):
                assert isinstance(dim, int)
                pairs.append((dim % source_value.ndim, output_dim))
        elif target is torch.ops.aten.unsqueeze.default:
            new_dim = node.args[1]
            assert isinstance(new_dim, int)
            new_dim %= value.ndim
            pairs = [
                (source_dim, source_dim + (source_dim >= new_dim))
                for source_dim in range(source_value.ndim)
            ]
        elif target is torch.ops.aten.expand.default:
            offset = value.ndim - source_value.ndim
            pairs = [
                (source_dim, source_dim + offset)
                for source_dim in range(source_value.ndim)
            ]
        else:
            return False
        for source_dim, output_dim in pairs:
            source_slot = (source, source_dim - source_value.ndim)
            if self._is_lane(source_slot):
                self._union((node, output_dim - value.ndim), source_slot)
        return True

    def _visit_pointwise(self, node: Node, value: torch.Tensor) -> bool:
        env = CompileEnvironment.current()
        operands = [
            arg
            for arg in node.all_input_nodes
            if isinstance(arg.meta.get("val"), torch.Tensor)
        ]
        for i in range(1, value.ndim + 1):
            slots = [
                (operand, -i)
                for operand in operands
                if operand.meta["val"].ndim >= i
                and not env.known_equal(operand.meta["val"].shape[-i], 1)
            ]
            lanes = [slot for slot in slots if self._is_lane(slot)]
            if not lanes:
                continue
            if len(lanes) != len(slots):
                # An arange dim meeting a tile axis: not a synthetic lane.
                return False
            for slot in lanes:
                self._union((node, -i), slot)
        return True

    def _index_dims(self, index: object) -> list[_Slot | str] | None:
        """The dims of the shape a load at ``index`` produces, as ``compute_shape``.

        An arange-derived index entry contributes its lane; ``None`` when an
        arange-derived entry is not one-dimensional, or when tensor indexers of
        more dims broadcast together (one-dimensional ones form a cartesian
        product, one dim each in order, as without broadcasting).
        """
        if not isinstance(index, (list, tuple)):
            return None
        env = CompileEnvironment.current()
        values = [
            entry.meta.get("val") if isinstance(entry, Node) else entry
            for entry in index
        ]
        if (
            any(isinstance(entry, Node) and entry in self._derived for entry in index)
            and env.should_broadcast_tensor_indexers(values)
            and any(
                isinstance(value, torch.Tensor) and value.ndim != 1 for value in values
            )
        ):
            return None
        dims: list[_Slot | str] = []
        for entry, value in zip(index, values, strict=True):
            if value is None:
                dims.append(_ONE)
            elif isinstance(value, torch.SymInt):
                symbol = _symint_expr(value)
                origin = (
                    HostFunction.current().expr_to_origin.get(symbol)
                    if isinstance(symbol, sympy.Symbol)
                    else None
                )
                if origin is not None and isinstance(origin.origin, BlockSizeOrigin):
                    dims.append(_AXIS)
            elif isinstance(value, slice):
                dims.append(_AXIS)
            elif isinstance(value, torch.Tensor):
                if not (isinstance(entry, Node) and entry in self._derived):
                    dims.extend([_AXIS] * len(env.tensor_indexer_dims(value)))
                elif value.ndim == 1 and self._is_lane((entry, -1)):
                    dims.append((entry, -1))
                else:
                    return None
            elif not isinstance(value, int):
                return None
        return dims

    def _align(self, operand: object, dims: list[_Slot | str]) -> bool:
        """Union the dims of ``operand`` with ``dims``, aligned from the right."""
        if not isinstance(operand, Node):
            return True
        value = operand.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return True
        if value.ndim > len(dims):
            return False
        env = CompileEnvironment.current()
        for i in range(1, value.ndim + 1):
            if env.known_equal(value.shape[-i], 1):
                continue
            slot = (operand, -i)
            dim = dims[-i]
            if isinstance(dim, tuple) and self._is_lane(slot):
                self._union(slot, dim)
            elif isinstance(dim, tuple) or self._is_lane(slot):
                return False
        return True

    def _visit_load(self, node: Node) -> bool:
        if not isinstance(node.args[0], Node):
            return False
        dims = self._index_dims(node.args[1])
        value = node.meta.get("val")
        if dims is None or not isinstance(value, torch.Tensor):
            return False
        if len(dims) != value.ndim:
            return False
        for output_dim, dim in enumerate(dims):
            if isinstance(dim, tuple):
                self._union((node, output_dim - value.ndim), dim)
        self._derived.add(node)
        return len(node.args) < 3 or self._align(node.args[2], dims)

    def _visit_store(self, node: Node) -> bool:
        if not isinstance(node.args[0], Node):
            return False
        dims = self._index_dims(node.args[1])
        if dims is None:
            return False
        self._store_lanes.append([dim for dim in dims if isinstance(dim, tuple)])
        return all(self._align(operand, dims) for operand in node.args[2:4])


def cute_free_arange_lanes(cg: GenerateAST) -> FreeArangeLanes:
    """The kernel's :class:`FreeArangeLanes`, built on first use."""
    if cg.cute_free_arange_lanes is None:
        cg.cute_free_arange_lanes = FreeArangeLanes(cg.codegen_graphs)
    return cg.cute_free_arange_lanes
