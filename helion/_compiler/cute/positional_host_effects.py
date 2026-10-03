"""Host-memory ordering for the positional whole-root fallback.

``PositionalEmitter`` evaluates pure FX expressions lazily at each use site, in
whichever CTA-cooperative phase needs them.  A host ``load`` is the only
admitted expression that reads mutable state: re-evaluating it after a host
store that precedes the use in program order would observe that store.

Before every host-writing effect ``E`` (a ``store``, or a ``_for_loop`` or
output-free ``_if`` whose live regions transitively store) the emitter
materializes each host load ``L`` that

* precedes ``E`` in program order (in ``E``'s graph, or an enclosing graph
  reached through a loop or branch port alias),
* is still reachable through unmaterialized pure nodes from anything evaluated
  at or after ``E``: ``E``'s own operands (value and indices of a store, the
  arguments and bounds of a loop, the ports of a branch), every later node, and
  the graph output that seeds loop carries, and
* reads a host tensor that may alias a tensor ``E`` writes.

``E``'s own operands are included because a store phase is cooperative: one
thread's read of ``x[j, i]`` must not observe another thread's ``x[j, i]``
store in the same phase (``x[b] = x[b].transpose(-1, -2)``).

May-alias uses only existing provenance, never names, shapes, or FakeTensor
object identity:

1. equal FakeTensor storage (views, or one tensor passed in two argument slots)
   may alias;
2. different storages where one is a wrapper-fresh allocation
   (``CompileEnvironment.fresh_allocation_storages`` for ``empty`` factories,
   or ``fresh_initialized_storages`` for ``zeros``/``ones``/``full`` factories)
   are disjoint;
3. two other storages are disjoint only when every visible input ``Source`` pair
   is proven disjoint by the bound kernel's runtime storage matrix
   (``runtime_tensor_sources_are_proven_disjoint``).  That fact participates in
   the bound-kernel cache key, so an aliasing launch cannot reuse the code.
   Without that registered fact the tensors may alias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch._guards import Source

    from ..compile_environment import CompileEnvironment


class HostEffectProvenanceError(Exception):
    """A host effect lacks the provenance required to order it soundly."""


# Nested regions a node executes: a loop body, or the *live* arms of an
# output-free branch (the selected arm when static, both when runtime), each
# with the outer values bound positionally to its placeholders.
NestedGraphs = Callable[[Node], "Sequence[tuple[torch.fx.Graph, Sequence[object]]]"]
PhiSource = Callable[[Node], Node]


def resolve_host_tensor(node: object, aliases: Mapping[Node, Node]) -> Node | None:
    """Follow loop-port aliases to the originating ``_host_tensor`` node."""
    seen: set[Node] = set()
    while isinstance(node, Node) and node not in seen:
        seen.add(node)
        if node.op == "call_function" and node.target is _tracing_ops._host_tensor:
            return node
        if node not in aliases:
            return None
        node = aliases[node]
    return None


def _port_aliases(
    graph: torch.fx.Graph, ports: Sequence[object], aliases: Mapping[Node, Node]
) -> dict[Node, Node]:
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(ports):
        raise HostEffectProvenanceError("nested region ports do not match")
    result = dict(aliases)
    for placeholder, argument in zip(placeholders, ports, strict=True):
        if isinstance(argument, Node):
            result.setdefault(placeholder, argument)
    return result


def written_host_tensors(
    node: Node, aliases: Mapping[Node, Node], nested_graphs: NestedGraphs
) -> list[Node]:
    """Host tensors written by ``node``, including its live nested regions."""
    if node.op != "call_function":
        return []
    if node.target is memory_ops.store:
        target = resolve_host_tensor(node.args[0], aliases)
        if target is None:
            raise HostEffectProvenanceError(
                f"store {node.name} target has no host tensor provenance"
            )
        return [target]
    written: list[Node] = []
    for graph, ports in nested_graphs(node):
        inner_aliases = _port_aliases(graph, ports, aliases)
        for inner in graph.nodes:
            written += written_host_tensors(inner, inner_aliases, nested_graphs)
    return written


class HostAliasOracle:
    """Conservative may-alias relation over host tensor nodes."""

    def __init__(self, env: CompileEnvironment) -> None:
        self.env = env

    @staticmethod
    def _storage(host: Node) -> torch.UntypedStorage:
        value = host.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise HostEffectProvenanceError(
                f"host tensor {host.name} has no tensor metadata"
            )
        return value.untyped_storage()

    def _input_sources(self, storage: torch.UntypedStorage) -> list[Source]:
        return [
            source
            for tensor, source in self.env.input_sources.items()
            if tensor.untyped_storage()._cdata == storage._cdata
        ]

    def may_alias(self, left: Node, right: Node) -> bool:
        left_storage, right_storage = self._storage(left), self._storage(right)
        if left_storage._cdata == right_storage._cdata:
            return True
        fresh = {
            storage._cdata
            for storage in (
                self.env.fresh_allocation_storages | self.env.fresh_initialized_storages
            )
        }
        if left_storage._cdata in fresh or right_storage._cdata in fresh:
            return False
        from .memory_ops import runtime_tensor_sources_are_proven_disjoint

        left_sources = self._input_sources(left_storage)
        right_sources = self._input_sources(right_storage)
        if not left_sources or not right_sources:
            return True
        return not all(
            runtime_tensor_sources_are_proven_disjoint(
                self.env, left_source, right_source
            )
            for left_source in left_sources
            for right_source in right_sources
        )


def _live_host_loads(
    roots: Iterable[Node],
    *,
    later: set[Node],
    materialized: Mapping[Node, object],
    aliases: Mapping[Node, Node],
    phi_source: PhiSource,
) -> dict[Node, Node]:
    """Earlier host loads reachable through unmaterialized pure nodes."""
    found: dict[Node, Node] = {}
    seen: set[Node] = set()
    pending = list(roots)
    while pending:
        node = pending.pop()
        if node in seen or node in materialized:
            continue
        seen.add(node)
        if node in aliases:
            pending.append(aliases[node])
            continue
        if node.op != "call_function":
            continue
        if node.target is _tracing_ops._host_tensor:
            continue
        if node.target is _tracing_ops._phi:
            # The emitter binds a carry merge to its loop result only.
            pending.append(phi_source(node))
            continue
        if node.target is memory_ops.load and node not in later:
            host = resolve_host_tensor(node.args[0], aliases)
            if host is not None:
                found[node] = host
        pending.extend(node.all_input_nodes)
    return found


def loads_to_preserve(
    nodes: list[Node],
    index: int,
    *,
    materialized: Mapping[Node, object],
    aliases: Mapping[Node, Node],
    nested_graphs: NestedGraphs,
    phi_source: PhiSource,
    oracle: HostAliasOracle,
) -> list[Node]:
    """Host loads to materialize immediately before ``nodes[index]``.

    Returned in program order (enclosing-graph loads first), so emission is
    deterministic.
    """
    effect = nodes[index]
    writes = written_host_tensors(effect, aliases, nested_graphs)
    if not writes:
        return []
    following = nodes[index + 1 :]
    roots = list(effect.all_input_nodes)
    for node in following:
        roots += node.all_input_nodes
    live = _live_host_loads(
        roots,
        later=set(following),
        materialized=materialized,
        aliases=aliases,
        phi_source=phi_source,
    )
    position = {node: offset for offset, node in enumerate(nodes)}
    hazards = [
        load
        for load, host in live.items()
        if any(oracle.may_alias(host, written) for written in writes)
    ]
    return sorted(hazards, key=lambda load: (position.get(load, -1), load.name))
