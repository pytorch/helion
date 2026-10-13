"""Physical publication dependencies and original-scope read lifetimes.

This is planning, not a layout or completion proof. A published value stops
expression ancestry at its original full owner; its upstream inputs are not
read again. Existing emitters still prove coordinates, native visibility and
successful completion. The first client binds complete preparation owners in
one original role scope; no register image or cross-role alias is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from torch.fx import Node


class InvalidPhysicalUse(ValueError):
    pass


def _arguments(value: object) -> object:
    """Snapshot only FX arguments, never host plans, bindings or their graphs."""
    if isinstance(value, (tuple, list)):
        return tuple(_arguments(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _arguments(item)) for key, item in value.items())
    return value


def _node_facts(node: Node) -> tuple[object, ...]:
    value = node.meta.get("val")
    tensor = (
        (value.dtype, tuple(map(str, value.shape)), tuple(map(str, value.stride())))
        if isinstance(value, torch.Tensor)
        else None
    )
    return (
        node,
        node.op,
        node.target,
        _arguments(node.args),
        _arguments(node.kwargs),
        tuple(node.users),
        tensor,
    )


@dataclass(frozen=True)
class PhysicalPublication:
    """One complete typed value at an original owner, not a guessed subview."""

    node: Node
    owner: object = field(compare=False, repr=False)
    dtype: torch.dtype
    shape: tuple[int, ...]

    def facts(self) -> tuple[object, ...]:
        return self.node, id(self.owner), self.dtype, self.shape


@dataclass(frozen=True)
class PhysicalReadPoint:
    """Original planned read interval; it cannot mint an execution receipt."""

    scope: object = field(compare=False, repr=False)
    event: int
    through: int


@dataclass
class _OwnerReads:
    owner: object
    scope: object
    born: int
    through: int
    first: int | None = None


class PhysicalUseFrontier:
    """Resolve real publication stops and consume uses before request packing.

    Graph facts are checked at explicit planning boundaries, not recursively
    for each leaf. Resolution memoization is local to an unchanged publication
    environment. A newly materialized value invalidates that environment.
    """

    def __init__(self, nodes: tuple[Node, ...]) -> None:
        if not nodes or len(set(nodes)) != len(nodes):
            raise InvalidPhysicalUse("missing or duplicated use graph")
        self.nodes = nodes
        self.graph = nodes[0].graph
        if any(node.graph is not self.graph for node in nodes):
            raise InvalidPhysicalUse("foreign use graph")
        self._nodes = frozenset(nodes)
        self._facts = tuple(_node_facts(node) for node in nodes)
        self._environment: object = None
        self._memo: dict[Node, tuple[PhysicalPublication, ...]] = {}
        self._ancestors: dict[Node, frozenset[Node]] = {}
        self._owners: dict[int, _OwnerReads] = {}
        self._publications: dict[
            int, tuple[PhysicalPublication, tuple[object, ...]]
        ] = {}

    def check(self) -> None:
        if (
            tuple(self.graph.nodes) != self.nodes
            or tuple(_node_facts(node) for node in self.nodes) != self._facts
            or any(
                value.facts() != facts for value, facts in self._publications.values()
            )
        ):
            raise InvalidPhysicalUse("physical-use graph changed")

    def resolve(
        self,
        roots: Iterable[Node],
        publications: Mapping[Node, PhysicalPublication],
        *,
        external: frozenset[Node],
        required: frozenset[Node],
        traversable: frozenset[Node],
        expand: Node | None = None,
    ) -> tuple[PhysicalPublication, ...]:
        """Resolve all argument edges, including selectors, masks and kwargs.

        Explicit externally supplied values stop before required publications,
        matching the original preparation partition. Expanding a materializer's
        own root does not allow another missing materialization to be crossed.
        """
        for value in publications.values():
            previous = self._publications.setdefault(id(value), (value, value.facts()))
            if previous[0] is not value or previous[1] != value.facts():
                raise InvalidPhysicalUse("original publication changed")
        environment = (
            tuple((node, id(value)) for node, value in publications.items()),
            external,
            required,
            traversable,
            expand,
        )
        if environment != self._environment:
            self._environment = environment
            self._memo.clear()
        active: set[Node] = set()

        def visit(node: Node) -> tuple[PhysicalPublication, ...]:
            if node not in self._nodes:
                raise InvalidPhysicalUse("foreign physical-use input")
            if node in self._memo:
                return self._memo[node]
            if node in active:
                raise InvalidPhysicalUse("cyclic physical-use input")
            active.add(node)
            if node in external:
                result: tuple[PhysicalPublication, ...] = ()
            elif node in required and node is not expand:
                publication = publications.get(node)
                if publication is None or publication.node is not node:
                    raise InvalidPhysicalUse("missing original publication")
                value = node.meta.get("val")
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype != publication.dtype
                    or len(publication.shape) != value.ndim
                    or any(
                        type(size) is not int or size <= 0 for size in publication.shape
                    )
                    or any(
                        type(old) is int and old != new
                        for old, new in zip(value.shape, publication.shape, strict=True)
                    )
                ):
                    raise InvalidPhysicalUse("publication changed typed full value")
                self._owner(publication.owner)
                result = (publication,)
            else:
                if node not in traversable:
                    raise InvalidPhysicalUse("input escapes original partition")
                found: dict[int, PhysicalPublication] = {}
                for parent in node.all_input_nodes:
                    for publication in visit(parent):
                        found[id(publication)] = publication
                result = tuple(found.values())
            active.remove(node)
            self._memo[node] = result
            return result

        found: dict[int, PhysicalPublication] = {}
        for node in roots:
            for publication in visit(node):
                found[id(publication)] = publication
        return tuple(found.values())

    def register_owner(self, owner: object, scope: object, born: int) -> None:
        if id(owner) in self._owners or type(born) is not int or born < 0:
            raise InvalidPhysicalUse("invalid or duplicated original owner")
        self._owners[id(owner)] = _OwnerReads(owner, scope, born, born + 1)

    def register_live_in(self, owner: object, scope: object) -> None:
        """Bind the original carry present before event zero, not a new write.

        The recurrence caller retains its exact carry/storage/FP32 admission.
        Ordinary publication registration still rejects negative birth events.
        """
        if id(owner) in self._owners:
            raise InvalidPhysicalUse("duplicated original live-in owner")
        self._owners[id(owner)] = _OwnerReads(owner, scope, -1, 0)

    def _owner(self, owner: object) -> _OwnerReads:
        result = self._owners.get(id(owner))
        if result is None or result.owner is not owner:
            raise InvalidPhysicalUse("unbound original owner")
        return result

    def require_published(self, owner: object, point: PhysicalReadPoint) -> None:
        """Preserve read ordering even for a non-destructive resident seed.

        A resident TMEM seed does not extend a nonexistent shared-C lease.
        This check grants neither a shared read nor destination reuse.
        """
        current = self._owner(owner)
        if (
            point.scope is not current.scope
            or type(point.event) is not int
            or type(point.through) is not int
            or not current.born < point.event < point.through
        ):
            # Another role/generation needs its original ordering/join binding.
            # Never maximize event numbers from unrelated scopes.
            raise InvalidPhysicalUse("read lacks original publication ordering")

    def read(self, owner: object, point: PhysicalReadPoint) -> None:
        self.require_published(owner, point)
        current = self._owner(owner)
        current.first = (
            point.event if current.first is None else min(current.first, point.event)
        )
        current.through = max(current.through, point.through)

    def first_read(self, owner: object, scope: object) -> int:
        current = self._owner(owner)
        if scope is not current.scope or current.first is None:
            raise InvalidPhysicalUse("missing original-scope read")
        return current.first

    def through(self, owner: object, scope: object) -> int:
        current = self._owner(owner)
        if scope is not current.scope:
            raise InvalidPhysicalUse("foreign lifetime scope")
        return current.through

    def ancestors(self, roots: Iterable[Node]) -> frozenset[Node]:
        """Memoized complete argument closure, without inventing publications."""
        active: set[Node] = set()

        def ancestry(node: Node) -> frozenset[Node]:
            if node not in self._nodes or node in active:
                raise InvalidPhysicalUse("invalid exclusive-use dependency")
            if node not in self._ancestors:
                active.add(node)
                result = {node}
                for parent in node.all_input_nodes:
                    result.update(ancestry(parent))
                active.remove(node)
                self._ancestors[node] = frozenset(result)
            return self._ancestors[node]

        result: set[Node] = set()
        for root in roots:
            result.update(ancestry(root))
        return frozenset(result)

    def exclusive_use(self, source: Node, operand: Node, consumer: Node) -> bool:
        """Discover the complete original value-use frontier for a bridge.

        This selects semantic exclusivity, not a register image: the original
        binder must still prove same-thread coordinates, full native layout,
        narrowing, participation and the actual successful fragment receipt.
        """
        if not {source, operand, consumer} <= self._nodes:
            raise InvalidPhysicalUse("foreign exclusive-use boundary")
        allowed = self.ancestors((operand,))
        if source not in allowed:
            return False
        pending, visited = [source], set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            permitted = {consumer} if node is operand else allowed
            if any(user not in permitted for user in node.users):
                return False
            if node is not operand:
                pending.extend(node.users)
        return True
