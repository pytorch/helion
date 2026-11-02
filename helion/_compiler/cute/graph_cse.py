"""Merge duplicate reductions of a device-IR graph before reduction rolling.

Tracing builds one node per source expression, so ``softmax(x)`` and
``log_softmax(x)`` of one operand carry two ``amax`` nodes over the same
masked input, and once those are one node, two identical ``sub``/``exp``/
``_mask_to`` chains and two identical ``sum`` nodes follow.  The reduction
roller turns every reduction node into its own sweep with its own cross-thread
exchange, and the persistent strategy emits one collective per reduction node,
so each duplicate costs whole passes over the row rather than an instruction.

A duplicate is merged when it is a reduction, or when one of its inputs took
part in a merge earlier in the pass -- as the erased node or as its survivor,
since trace order decides which twin of a chain is seen first (the chains
between two merged reductions, whichever consumer was traced first).
Identical pure nodes whose inputs took no part in a merge stay as traced: the
backend's structural matchers key on the traced graph shape, and only the
reduction chains are the measured cost.  Two nodes computing the same pure
operator on the same inputs are bitwise identical, so a merge never changes a
result.

Only functional ``torch.ops`` overloads and the compiler's own ``_mask_to``
qualify.  Memory operations, loop and branch nodes, host tensors, symbolic
sizes, Python operators and anything FX reports as impure (mutation, random
number generation) are never merged, so memory order and every configuration
slot keyed to a load or store stay as traced.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import torch
from torch._ops import OpOverload

from ...language._tracing_ops import _mask_to

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..device_ir import DeviceIR

_HASHABLE_LEAVES = (
    int,
    bool,
    str,
    type(None),
    torch.dtype,
    torch.device,
    torch.layout,
    torch.memory_format,
)


class _Unkeyable(Exception):
    pass


def _key(obj: object) -> object:
    if isinstance(obj, torch.fx.Node):
        return ("node", id(obj))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(_key(item) for item in obj))
    if isinstance(obj, dict):
        return ("dict", tuple((k, _key(v)) for k, v in sorted(obj.items())))
    if isinstance(obj, slice):
        return ("slice", _key(obj.start), _key(obj.stop), _key(obj.step))
    if isinstance(obj, float):
        # Byte identity keeps -0.0 apart from 0.0 and one NaN payload from another.
        return ("float", struct.pack("<d", obj))
    if isinstance(obj, _HASHABLE_LEAVES):
        return (type(obj).__name__, obj)
    raise _Unkeyable


def _value_signature(node: torch.fx.Node) -> object:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        # Symbolic sizes are not hashable; their printed form names the symbol.
        return ("tensor", tuple(str(size) for size in val.shape), val.dtype)
    if isinstance(val, (int, float, bool)):
        return (type(val).__name__, val)
    raise _Unkeyable


# Pure by schema, yet two calls need not hold equal values: the contents of
# an uninitialized allocation are whatever the allocator left behind.
_VALUE_INDETERMINATE = frozenset(
    {
        "aten::empty",
        "aten::empty_like",
        "aten::empty_permuted",
        "aten::empty_quantized",
        "aten::empty_strided",
        "aten::new_empty",
        "aten::new_empty_strided",
    }
)


def _is_candidate(node: torch.fx.Node) -> bool:
    if node.op != "call_function" or node.is_impure():
        return False
    target = node.target
    if target is _mask_to:
        return True
    if not isinstance(target, OpOverload):
        return False
    schema = target._schema
    if schema.name in _VALUE_INDETERMINATE:
        return False
    if schema.is_mutable or any(arg.alias_info is not None for arg in schema.arguments):
        return False
    return not any(
        tag in target.tags
        for tag in (torch.Tag.nondeterministic_seeded, torch.Tag.inplace_view)
    )


def _is_reduction(node: torch.fx.Node) -> bool:
    from ..inductor_lowering import ReductionLowering

    return isinstance(node.meta.get("lowering"), ReductionLowering)


def cse_pure_nodes(
    graph: torch.fx.Graph,
    *,
    is_root: Callable[[torch.fx.Node], bool] = _is_reduction,
) -> dict[torch.fx.Node, torch.fx.Node]:
    """Replace duplicate roots and the duplicates they cause; ``{erased: survivor}``.

    ``is_root`` names the nodes whose duplicates merge on their own; any other
    duplicate merges only when one of its inputs took part in a merge earlier
    in the same pass -- as the erased node or as the survivor.  Both sides
    count because trace order decides which consumer is seen first: with
    duplicate roots ``r1``/``r2`` and consumers ``mul(r2, 2)`` traced before
    ``mul(r1, 2)``, the first consumer (an erased node's) survives its key and
    the second (the survivor's) is the duplicate to admit.
    """
    inputs_before = {node: list(node.all_input_nodes) for node in graph.nodes}
    seen: dict[object, torch.fx.Node] = {}
    replaced: dict[torch.fx.Node, torch.fx.Node] = {}
    survivors: set[torch.fx.Node] = set()
    for node, inputs in inputs_before.items():
        if not _is_candidate(node):
            continue
        try:
            key = (
                node.target,
                _key(node.args),
                _key(node.kwargs),
                _value_signature(node),
                type(node.meta.get("lowering")),
            )
        except _Unkeyable:
            continue
        survivor = seen.get(key)
        if survivor is None:
            seen[key] = node
            continue
        if not (
            is_root(node) or any(inp in replaced or inp in survivors for inp in inputs)
        ):
            continue
        node.replace_all_uses_with(survivor)
        graph.erase_node(node)
        replaced[node] = survivor
        survivors.add(survivor)
    return replaced


def cse_device_ir(device_ir: DeviceIR) -> int:
    """CSE every graph of ``device_ir``; subgraph argument lists follow."""
    from ..device_ir import NodeArgsGraphInfo

    replaced: dict[torch.fx.Node, torch.fx.Node] = {}
    for graph_info in device_ir.graphs:
        replaced.update(cse_pure_nodes(graph_info.graph))
    if replaced:
        for graph_info in device_ir.graphs:
            if isinstance(graph_info, NodeArgsGraphInfo):
                graph_info.node_args = [
                    replaced.get(node, node) for node in graph_info.node_args
                ]
    return len(replaced)
