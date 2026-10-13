"""Original contraction and state authority for a complete role epoch.

This is not an alternative two-dot continuation. Split-K segments belong to
one original ContractionSpec, and only their complete ordered prefix describes
that contraction. Native owners, role phases and successful body emission are
separate obligations of the enclosing BodyProgram adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from . import chained_matmul as chain
from .chained_completed_store import _records

if TYPE_CHECKING:
    from torch.fx import Node

    from .contraction_region import ContractionRegion
    from .contraction_region import ContractionSpec
    from .prepared_continuation import PreparedContinuation
    from .prepared_tcgen_binding import MatchedPreparedProjection


@dataclass(frozen=True)
class ContractionSegment:
    """One original K-atom interval, not another logical dot."""

    begin: int
    end: int
    commit: bool
    wait_after: bool = False


@dataclass(frozen=True)
class SegmentedContraction:
    region: ContractionRegion
    spec: ContractionSpec
    atom_k: int
    segments: tuple[ContractionSegment, ...]
    _k_extent: int = field(init=False, repr=False)
    _revision: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Use original config-resolved tile geometry, not logical-symbol
        # equality. The enclosing complete match supplies the full extent.
        shape, _ = self._geometry()
        object.__setattr__(self, "_k_extent", shape[1])
        self._validate()
        object.__setattr__(self, "_revision", self._current())

    def _validate(self) -> None:
        lhs, rhs = self._geometry()
        if lhs[1] != self._k_extent or rhs[0] != self._k_extent:
            raise chain._UnsupportedChain("contraction K geometry changed")
        if (
            not any(item is self.spec for item in self.region.contractions)
            or self.spec.accumulator is not None
            or self.spec.result_dtype is not torch.float32
            or type(self.atom_k) is not int
            or self.atom_k <= 0
            or not self.segments
        ):
            raise chain._UnsupportedChain("invalid segmented contraction")
        cursor = 0
        for segment in self.segments:
            if (
                type(segment.begin) is not int
                or type(segment.end) is not int
                or segment.begin != cursor
                or segment.end <= segment.begin
                or type(segment.commit) is not bool
                or type(segment.wait_after) is not bool
                or segment.wait_after
                and not segment.commit
            ):
                raise chain._UnsupportedChain("changed contraction K prefix")
            cursor = segment.end
        if not self.segments[-1].commit or cursor * self.atom_k != self._k_extent:
            raise chain._UnsupportedChain("incomplete contraction K prefix")

    def _geometry(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        lhs, rhs = (node.meta.get("val") for node in (self.spec.lhs, self.spec.rhs))
        if (
            not isinstance(lhs, torch.Tensor)
            or not isinstance(rhs, torch.Tensor)
            or lhs.ndim != 2
            or rhs.ndim != 2
        ):
            raise chain._UnsupportedChain("missing contraction tile geometry")
        # Re-evaluate original block bindings on every check. In particular a
        # new reduction_loops entry must invalidate the old split-K prefix.
        return chain._host_shape(lhs), chain._host_shape(rhs)

    def _current(self) -> object:
        return (
            self.region,
            id(self.region),
            self.region.graph,
            id(self.region.graph),
            self.spec,
            id(self.spec),
            tuple(id(item) for item in self.region.contractions),
            tuple(id(item) for item in self.segments),
            self._geometry(),
            _records((self.spec, self.atom_k, self.segments, self._k_extent)),
            tuple(
                (
                    node,
                    node.op,
                    node.target,
                    _records(node.args),
                    _records(node.kwargs),
                    _records(node.meta),
                    tuple(node.users),
                )
                for node in self.region.graph.nodes
            ),
        )

    def check(self) -> None:
        self._validate()
        if self._current() != self._revision:
            raise chain._UnsupportedChain("segmented contraction graph changed")

    def initialized(self, segment: ContractionSegment) -> bool:
        self.check()
        if not any(item is segment for item in self.segments):
            raise chain._UnsupportedChain("foreign contraction segment")
        return segment.begin != 0


@dataclass(frozen=True)
class PreparedEpoch:
    """The four specs from the original fully validated recurrence match.

    Retain the existing projection and Q/O proofs. No names or reconstructed
    stage plans stand in for these exact graph-owned Nodes and specs.
    """

    projection: MatchedPreparedProjection
    continuation: PreparedContinuation
    projected: SegmentedContraction
    update: ContractionSpec
    _identity: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "_identity", self._current())

    def _current(self) -> object:
        return tuple(
            (item, id(item))
            for item in (
                self.projection,
                self.continuation,
                self.projected,
                self.update,
            )
        )

    def _validate(self) -> None:
        self.projection.check()
        self.continuation.check()
        self.projected.check()
        region = self.continuation.region
        expected = (
            self.projected.spec,
            *self.continuation.specs,
            self.update,
        )
        if (
            self.projected.region is not region
            or region.graph is not self.projection.loop.graph
            or self.projected.spec.node is not self.projection.source
            or len(region.contractions) != len(expected)
            or {id(item) for item in region.contractions}
            != {id(item) for item in expected}
            or len({id(item) for item in expected}) != len(expected)
            or self.update.result_dtype is not torch.float32
            or self.projection.state not in region.nodes
            or self.projection.operand not in region.nodes
        ):
            raise chain._UnsupportedChain("foreign prepared epoch relation")

    def check(self) -> None:
        self._validate()
        if self._current() != self._identity:
            raise chain._UnsupportedChain("prepared epoch selection changed")


def epoch_for_nodes(
    projection: MatchedPreparedProjection,
    continuation: PreparedContinuation,
    update: Node,
) -> PreparedEpoch:
    """Called only after the original four-dot recurrence validator succeeds."""
    region = continuation.region
    selected = tuple(
        tuple(spec for spec in region.contractions if spec.node is node)
        for node in (projection.source, update)
    )
    if any(len(items) != 1 for items in selected):
        raise chain._UnsupportedChain("epoch contraction is not in region")
    return PreparedEpoch(
        projection,
        continuation,
        SegmentedContraction(
            region,
            selected[0][0],
            16,
            (ContractionSegment(0, 4, False), ContractionSegment(4, 8, True)),
        ),
        selected[1][0],
    )
