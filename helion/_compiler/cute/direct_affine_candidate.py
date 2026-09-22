# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Immutable semantic candidates for direct affine-scan lowering.

Discovery deliberately runs before CuTe layout planning.  A candidate owns no
schedule and changes no FX metadata; it is only a topology snapshot that a late
lowering can revalidate after ordinary code generation has exposed concrete
addresses and thread coordinates.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from .direct_affine_plan import SUPPORTED_FEATURE_EXTENT
from .direct_affine_plan import DirectAffineMma
from .direct_affine_plan import select_direct_affine_mma
from .short_affine_scan import ShortAffineScanRegion
from .short_affine_scan import discover_short_affine_scan_regions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


@dataclasses.dataclass(frozen=True)
class DirectAffineNodeFingerprint:
    """Identity of one node in a candidate's source interval."""

    node: torch.fx.Node = dataclasses.field(repr=False)
    op: str
    target: object = dataclasses.field(repr=False)
    args: object = dataclasses.field(repr=False)
    kwargs: object = dataclasses.field(repr=False)
    users: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)


def _freeze_argument(value: object) -> object:
    if isinstance(value, torch.fx.Node):
        return value
    if isinstance(value, tuple):
        return ("tuple", *(_freeze_argument(item) for item in value))
    if isinstance(value, list):
        return ("list", *(_freeze_argument(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            *(
                (key, _freeze_argument(item))
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ),
        )
    if isinstance(value, slice):
        return (
            "slice",
            _freeze_argument(value.start),
            _freeze_argument(value.stop),
            _freeze_argument(value.step),
        )
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype)):
        return value
    if isinstance(value, torch.device):
        return ("device", value.type, value.index)
    return (type(value), id(value))


def _fingerprint(
    nodes: tuple[torch.fx.Node, ...],
) -> tuple[DirectAffineNodeFingerprint, ...]:
    return tuple(
        DirectAffineNodeFingerprint(
            node=node,
            op=node.op,
            target=node.target,
            args=_freeze_argument(node.args),
            kwargs=_freeze_argument(node.kwargs),
            users=tuple(node.users),
        )
        for node in nodes
    )


@dataclasses.dataclass(frozen=True)
class DirectAffineCandidate:
    """A names-agnostic affine region awaiting physical safety proofs."""

    region: ShortAffineScanRegion = dataclasses.field(repr=False)
    source_fingerprint: tuple[DirectAffineNodeFingerprint, ...] = dataclasses.field(
        repr=False
    )

    @property
    def graph_id(self) -> int:
        return self.region.graph_id

    @property
    def smallest_mma(self) -> DirectAffineMma:
        return select_direct_affine_mma(self.region.step_count)

    def validate(self) -> None:
        """Raise if the snapshot cannot be a direct affine lowering."""

        region = self.region
        if (
            region.storage_dtype is not torch.bfloat16
            or region.feature_extent != SUPPORTED_FEATURE_EXTENT
            or region.row_extent <= 0
            or region.row_extent % 16
            or region.live_outs
            or self.smallest_mma is DirectAffineMma.ORDINARY
            or len(set(region.owned_nodes)) != len(region.owned_nodes)
            or len(set(region.producer_nodes)) != len(region.producer_nodes)
            or not set(region.owned_nodes).issubset(region.source_interval)
            or not set(region.producer_nodes).issubset(region.source_interval)
            or any(
                access.mask is not None or access.other is not None or access.kwargs
                for access in (*region.read_accesses, *region.write_accesses)
            )
            or self.source_fingerprint != _fingerprint(region.source_interval)
        ):
            raise ValueError("invalid direct affine candidate")


def _candidate(region: ShortAffineScanRegion) -> DirectAffineCandidate | None:
    mma = select_direct_affine_mma(region.step_count)
    if mma is DirectAffineMma.ORDINARY:
        return None
    candidate = DirectAffineCandidate(
        region=region,
        source_fingerprint=_fingerprint(region.source_interval),
    )
    try:
        candidate.validate()
    except ValueError:
        return None
    return candidate


def discover_direct_affine_candidates(
    graphs: Sequence[GraphInfo],
) -> tuple[DirectAffineCandidate, ...]:
    """Discover non-overlapping root candidates without mutating the graphs."""

    return tuple(
        candidate
        for region in discover_short_affine_scan_regions(graphs)
        if (candidate := _candidate(region)) is not None
    )


def revalidate_direct_affine_candidate(
    candidate: DirectAffineCandidate,
    graph_info: GraphInfo,
) -> DirectAffineCandidate | None:
    """Rediscover a candidate and reject any changed source edge or boundary."""

    if graph_info.graph_id != candidate.graph_id:
        return None
    rebuilt = discover_direct_affine_candidates((graph_info,))
    return next((item for item in rebuilt if item == candidate), None)


__all__ = [
    "DirectAffineCandidate",
    "DirectAffineNodeFingerprint",
    "discover_direct_affine_candidates",
    "revalidate_direct_affine_candidate",
]
