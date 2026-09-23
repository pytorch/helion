"""Turn explicit preserve-output branches into masked stores.

``store(x, i, where(mask, value, load(x, i)))`` needlessly writes back the
old value where ``mask`` is false. Such writes also race active lanes when
inactive gather/scatter indices are clamped to another lane's address.

Only the direct, same-dtype pattern is accepted. Tensor/index references
must match exactly, and no write, control-flow call, or unknown effect may
occur between the load and store. This pass runs before lowering metadata
is prepared, so a new mask receives the ordinary pointwise lowering.
"""

from __future__ import annotations

from itertools import starmap
import operator
from typing import TYPE_CHECKING
from typing import TypeGuard
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language import view_ops

if TYPE_CHECKING:
    from torch.fx.node import Argument

_WHERE = torch.ops.aten.where.self
_AND = torch.ops.aten.logical_and.default
_PURE_FUNCTIONS = frozenset(
    {
        memory_ops.load,
        view_ops.subscript,
        view_ops.split,
        view_ops.join,
        tile_ops.tile_index,
        tile_ops.tile_begin,
        tile_ops.tile_end,
        tile_ops.tile_block_size,
        tile_ops.tile_count,
        tile_ops.tile_id,
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._constant_tensor,
        _tracing_ops._pre_broadcast_tile,
        _tracing_ops._mask_to,
        operator.getitem,
        operator.add,
        operator.sub,
        operator.mul,
        operator.floordiv,
        operator.mod,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.and_,
        operator.or_,
        operator.xor,
        operator.lshift,
        operator.rshift,
        operator.neg,
        operator.invert,
    }
)


def _is_read_only(node: torch.fx.Node) -> bool:
    if node.op in ("placeholder", "get_attr"):
        return True
    if node.op != "call_function" or node.is_impure():
        return False
    if node.target in _PURE_FUNCTIONS:
        return True
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace in ("aten", "prims")
        and not node.target._schema.is_mutable
    )


def _arguments(
    node: torch.fx.Node, names: tuple[str, ...], required: int
) -> tuple[object, ...] | None:
    if len(node.args) > len(names) or set(node.kwargs) - set(names):
        return None
    if set(node.kwargs).intersection(names[: len(node.args)]):
        return None
    result = list(node.args)
    for index in range(len(node.args), len(names)):
        name = names[index]
        if index < required and name not in node.kwargs:
            return None
        result.append(node.kwargs.get(name))
    return tuple(result)


def _same_index(left: object, right: object) -> bool:
    if left is right:
        return True
    if isinstance(left, torch.fx.Node) or isinstance(right, torch.fx.Node):
        return False
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            starmap(_same_index, zip(left, right, strict=True))
        )
    if isinstance(left, slice) and isinstance(right, slice):
        return all(
            starmap(
                _same_index,
                zip(
                    (left.start, left.stop, left.step),
                    (right.start, right.stop, right.step),
                    strict=True,
                ),
            )
        )
    if type(left) is int and type(right) is int:
        return left == right
    if isinstance(left, torch.SymInt) and isinstance(right, torch.SymInt):
        return left.node is right.node
    return False


def _tensor_value(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.fx.Node):
        result = value.meta.get("val")
        if isinstance(result, torch.Tensor):
            return result
    return None


def _is_bool_tensor(value: object) -> TypeGuard[torch.fx.Node]:
    tensor = _tensor_value(value)
    return tensor is not None and tensor.dtype is torch.bool


def _mask_shape(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[int | torch.SymInt, ...] | None:
    dims: list[int | torch.SymInt] = []
    lhs, rhs = list(left.shape), list(right.shape)
    lhs = [1] * max(0, len(rhs) - len(lhs)) + lhs
    rhs = [1] * max(0, len(lhs) - len(rhs)) + rhs
    for a, b in zip(lhs, rhs, strict=True):
        if _same_index(a, b) or type(b) is int and b == 1:
            dims.append(a)
        elif type(a) is int and a == 1:
            dims.append(b)
        else:
            return None
    return tuple(dims)


def _rewrite_store(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    epochs: dict[torch.fx.Node, int],
    epoch: int,
) -> bool:
    args = _arguments(node, ("tensor", "index", "value", "extra_mask"), 3)
    if args is None:
        return False
    tensor, indices, value, extra_mask = args
    if (
        not isinstance(tensor, torch.fx.Node)
        or not isinstance(indices, (list, tuple))
        or not isinstance(value, torch.fx.Node)
        or value.op != "call_function"
        or value.target is not _WHERE
        or len(value.args) != 3
        or value.kwargs
    ):
        return False
    condition, update, old = value.args
    if (
        not _is_bool_tensor(condition)
        or not isinstance(old, torch.fx.Node)
        or old.op != "call_function"
        or old.target is not memory_ops.load
        or epochs.get(old) != epoch
    ):
        return False
    load_args = _arguments(old, ("tensor", "index", "extra_mask", "eviction_policy"), 2)
    if (
        load_args is None
        or load_args[0] is not tensor
        or not _same_index(load_args[1], indices)
        or load_args[2] is not None
    ):
        return False
    tensors = [_tensor_value(item) for item in (tensor, value, update, old)]
    target_value, result_value, update_value, old_value = tensors
    if (
        target_value is None
        or result_value is None
        or update_value is None
        or old_value is None
        or any(
            item.dtype is not target_value.dtype for item in tensors if item is not None
        )
        # Removing where must not remove a broadcast of the stored value.
        or not _same_index(tuple(update_value.shape), tuple(result_value.shape))
        or not _same_index(tuple(old_value.shape), tuple(result_value.shape))
    ):
        return False

    mask = condition
    if extra_mask is not None and extra_mask is not condition:
        if not _is_bool_tensor(extra_mask):
            return False
        condition_value = _tensor_value(condition)
        extra_value = _tensor_value(extra_mask)
        assert condition_value is not None and extra_value is not None
        shape = _mask_shape(condition_value, extra_value)
        if shape is None:
            return False
        with graph.inserting_before(node):
            mask = graph.call_function(_AND, (condition, extra_mask))
            mask.meta = {
                **value.meta,
                "val": condition_value.new_empty(shape),
            }
    node.args = (tensor, cast("Argument", indices), update, mask)
    node.kwargs = {}
    if not value.users:
        graph.erase_node(value)
    if not old.users:
        graph.erase_node(old)
    return True


def fold_noop_stores(graph: torch.fx.Graph) -> int:
    """Fold direct preserve-output stores and return the number changed.

    Writes to any tensor invalidate earlier loads; no alias analysis is
    assumed. Control-flow regions are analyzed independently by DeviceIR.
    Unknown calls remain in the graph and prevent crossing their effects.
    """
    epochs: dict[torch.fx.Node, int] = {}
    epoch = 0
    changed = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is memory_ops.store:
            changed += int(_rewrite_store(graph, node, epochs, epoch))
        if not _is_read_only(node):
            epoch += 1
        epochs[node] = epoch
    if changed:
        graph.lint()
    return changed
