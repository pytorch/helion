"""Prove replication along a reduction axis without assigning data ownership."""

from __future__ import annotations

import torch
from torch.fx import Node

from ....language._tracing_ops import _mask_to
from ....language.view_ops import subscript
from ...utils import is_full_slice


def explicitly_replicated_axis(node: Node, dim: int) -> bool:
    """Whether ``dim`` comes from an explicit singleton expansion.

    Only coordinate-preserving pointwise operations and singleton-only views
    are transparent. In particular, a reshape of distributed source axes is
    not a replication proof, nor is an unrelated reduction with the same size.
    """

    # Postorder over the FX DAG keeps both work and Python stack depth bounded.
    # Shared predecessors are proved once per axis.
    root = (node, dim)
    proofs: dict[tuple[Node, int], bool] = {}
    dependencies: dict[tuple[Node, int], tuple[tuple[Node, int], ...]] = {}
    worklist = [(root, False)]
    while worklist:
        key, finish = worklist.pop()
        if key in proofs:
            continue
        if finish:
            proofs[key] = all(proofs[source] for source in dependencies.pop(key))
            continue
        result = _replicated_axis(*key)
        if isinstance(result, bool):
            proofs[key] = result
            continue
        dependencies[key] = result
        worklist.append((key, True))
        worklist.extend((source, False) for source in result if source not in proofs)
    return proofs[root]


def _replicated_axis(node: Node, dim: int) -> bool | tuple[tuple[Node, int], ...]:
    """A terminal proof result or the predecessor proofs it requires."""
    value = node.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim == 0
        or node.op != "call_function"
    ):
        return False
    dim %= value.ndim
    if not node.args or not isinstance(node.args[0], Node):
        return False
    source = node.args[0]
    source_value = source.meta.get("val")
    if not isinstance(source_value, torch.Tensor):
        return False
    if node.target is _mask_to:
        return ((source, dim),)
    if node.target is subscript:
        indices = node.args[1]
        if not isinstance(indices, (list, tuple)) or not all(
            index is None or is_full_slice(index) for index in indices
        ):
            return False
        source_dim = 0
        for output_dim, index in enumerate(indices):
            if output_dim == dim:
                return ((source, source_dim),) if index is not None else False
            source_dim += index is not None
        return False
    if node.target is torch.ops.aten.expand.default:
        source_dim = dim - (value.ndim - source_value.ndim)
        if source_dim < 0 or source_value.shape[source_dim] == 1:
            return True
        if source_value.shape[source_dim] != value.shape[dim]:
            return False
        return ((source, source_dim),)
    if node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
    ):
        source_axes = [i for i, size in enumerate(source_value.shape) if size != 1]
        output_axes = [i for i, size in enumerate(value.shape) if size != 1]
        if dim not in output_axes or [source_value.shape[i] for i in source_axes] != [
            value.shape[i] for i in output_axes
        ]:
            return False
        return ((source, source_axes[output_axes.index(dim)]),)
    if not isinstance(node.target, torch._ops.OpOverload) or (
        torch.Tag.pointwise not in node.target.tags
    ):
        return False
    inputs = [
        arg
        for arg in node.all_input_nodes
        if isinstance(arg.meta.get("val"), torch.Tensor)
    ]
    if not inputs or any(arg.meta["val"].shape != value.shape for arg in inputs):
        return False
    return tuple((arg, dim) for arg in inputs)
