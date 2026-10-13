from __future__ import annotations

from dataclasses import replace

import pytest

from .test_cute_chained_frontier_materialization import completed as completed
from helion._compiler.cute.chained_preparation_transfers import (
    bind_preparation_transfer_span,
)
from helion._compiler.cute.chained_preparation_transfers import (
    preparation_transfer_owner,
)


def test_generic_transfer_preserves_complete_frontier_owners(completed):
    _, _, bound, receipt = completed
    physical = bound.physical
    action = next(item for item in physical.accepted.actions if item.proof is receipt)
    original = bound.frontier_transfers[0]
    transfer = bind_preparation_transfer_span(
        physical, original.first, original.stop, action.reads, action.writes
    )
    assert transfer is not None
    assert transfer.sources == original.sources
    assert transfer.destinations == original.destinations


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "negative",
        "before_owner",
        "after_owner",
        "duplicate_view",
        "duplicate_owner",
    ],
)
def test_physical_transfer_rejects_invalid_effective_accesses(completed, mutation):
    _, _, bound, receipt = completed
    physical = bound.physical
    action = next(item for item in physical.accepted.actions if item.proof is receipt)
    transfer = bound.frontier_transfers[0]
    name = action.reads[0]
    view = next(view for view in physical.views if view.original.semantic.name == name)
    owner = physical.layout.region(view.owner)
    if mutation == "duplicate_owner":
        physical = replace(
            physical,
            layout=replace(physical.layout, regions=(*physical.layout.regions, owner)),
        )
    elif mutation == "duplicate_view":
        physical = replace(physical, views=(*physical.views, view))
    else:
        accesses = {
            "empty": (),
            "negative": ((owner.byte_offset, -1),),
            "before_owner": ((owner.byte_offset - 1, 1),),
            "after_owner": ((owner.byte_end, 1),),
        }[mutation]
        physical = replace(
            physical,
            views=tuple(
                replace(item, accesses=accesses) if item is view else item
                for item in physical.views
            ),
        )
    assert (
        preparation_transfer_owner(physical, name, transfer.first, transfer.stop)
        is None
    )


@pytest.mark.parametrize("first,stop", [(True, 2), (0, False), (-1, 2), (2, 2), (3, 2)])
def test_physical_transfer_rejects_invalid_intervals_even_without_accesses(
    completed, first, stop
):
    _, _, bound, _ = completed
    assert bind_preparation_transfer_span(bound.physical, first, stop, (), ()) is None
