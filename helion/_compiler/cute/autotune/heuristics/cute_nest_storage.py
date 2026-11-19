"""What the device IR shows of a kernel's memory accesses and of the values
its loops reduce: the facts the seed gates of the row-loop heuristics share
(``CuteLaneCarryRowHeuristic``, ``CuteResidentReductionHeuristic``).

A gate keeps a family from planting seeds on a kernel its lowering declines
at every config; the lowering still proves each config it is given.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from .....language import _tracing_ops
from .....language.memory_ops import load as language_load
from .....language.memory_ops import store as language_store
from ....runtime_tensor_guards import tensor_pairs_with_proven_disjoint_storage

if TYPE_CHECKING:
    import sympy

    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR


def block_dims(value: object, symbol: sympy.Expr, size: int | None) -> int:
    """The dims of a tensor value that are a block's: of its size symbol
    or, for a dim traced at its static length, of that ``size``."""
    if not isinstance(value, torch.Tensor):
        return 0
    return sum(
        dim._sympy_() == symbol if isinstance(dim, torch.SymInt) else dim == size
        for dim in value.shape
    )


def reduces(node: torch.fx.Node, symbol: sympy.Expr, size: int | None) -> bool:
    """Whether ``node``'s value has fewer dims of a block (``block_dims``)
    than one of its inputs: a reduction along that block.  A load indexes
    its tensor and reduces nothing, however many dims of the tensor's
    static shape have the block's length (a split of as many rows as the
    rows have columns)."""
    value = node.meta.get("val")
    if node.target is language_load:
        return False
    return isinstance(value, torch.Tensor) and any(
        block_dims(argument.meta.get("val"), symbol, size)
        > block_dims(value, symbol, size)
        for argument in node.all_input_nodes
    )


def inputs(node: torch.fx.Node) -> set[torch.fx.Node]:
    """``node`` and every node its value is computed from."""
    seen: set[torch.fx.Node] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(current.all_input_nodes)
    return seen


@dataclasses.dataclass(frozen=True)
class HostTensorAccesses:
    """The host tensors a kernel's device graphs load and store."""

    # Each tensor's traced value, by its name in the kernel.
    tensors: dict[str, torch.Tensor]
    loaded: frozenset[str]
    # ``(graph id, tensor name, store node)`` of every store, in graph order.
    stores: tuple[tuple[int, str, torch.fx.Node], ...]

    def stored(self, graph_id: int) -> list[str]:
        """The tensors the stores of one graph write, one per store."""
        return [name for graph, name, _node in self.stores if graph == graph_id]


def host_tensor_accesses(device_ir: DeviceIR) -> HostTensorAccesses | None:
    """The loads and stores of ``device_ir``'s graphs by host tensor;
    ``None`` when a store writes anything else, or an access names no
    traced tensor.  A load of a value (a tile's index vector) reads no
    memory."""
    tensors: dict[str, torch.Tensor] = {}
    loaded: set[str] = set()
    stores: list[tuple[int, str, torch.fx.Node]] = []
    for info in device_ir.graphs:
        for node in info.graph.nodes:
            if node.target not in (language_load, language_store):
                continue
            tensor = node.args[0]
            if not isinstance(tensor, torch.fx.Node):
                return None
            if tensor.target is not _tracing_ops._host_tensor:
                if node.target is language_store:
                    return None
                continue
            name, fake = tensor.args[0], tensor.meta.get("val")
            if not isinstance(name, str) or not isinstance(fake, torch.Tensor):
                return None
            tensors[name] = fake
            if node.target is language_load:
                loaded.add(name)
            else:
                stores.append((info.graph_id, name, node))
    return HostTensorAccesses(tensors, frozenset(loaded), tuple(stores))


def proven_apart(
    env: CompileEnvironment, device_ir: DeviceIR, tensors: dict[str, torch.Tensor]
) -> set[frozenset[str]] | None:
    """The pairs of ``tensors`` the kernel's storage rule proves apart, as
    the lowering applies it (``DeviceFunction.proven_disjoint_tensor_pairs``):
    with the facts of this binding's arguments, which its cache key holds.

    Seeds are planted with no host function current; the rule reads the
    tensors' origins from the device IR's.  ``None`` without one.
    """
    host = device_ir.host_function
    if host is None:
        return None
    with host:
        return tensor_pairs_with_proven_disjoint_storage(
            env, tensors, allow_unbound=True
        )
