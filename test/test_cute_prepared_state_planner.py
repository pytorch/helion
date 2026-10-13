from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from test.test_cute_prepared_epoch_walk import _code as epoch_code
from test.test_cute_prepared_state_body import _selected as root_code

from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import prepared_state_planner as planner
from helion._compiler.cute.prepared_epoch_state import DescriptorStateBinding
from helion._compiler.cute.prepared_state_body import FragmentStateBinding


def _observe(build, records, requests, cuts):
    plan = build(requests, cuts)
    reverse = build(tuple(reversed(requests)), cuts)

    def signature(current):
        return tuple(
            tuple((effect.kind, id(effect.binding)) for effect in phase)
            for phase in current.phases
        )

    assert signature(plan) == signature(reverse)
    assert sum(effect.kind == "read" for effect in plan.actions) == len(plan.residency)
    request = requests[0]
    with pytest.raises(chain._UnsupportedChain, match="lost original native view"):
        build(
            (
                replace(
                    request,
                    source_view=replace(
                        request.source_view, offset=request.source_view.offset + 1
                    ),
                ),
                *requests[1:],
            ),
            cuts,
        )
    with pytest.raises(chain._UnsupportedChain, match="duplicated state publication"):
        build((*requests, request), cuts)
    if request.complete_before is not None:
        with pytest.raises(chain._UnsupportedChain, match="original .*cut"):
            build((replace(request, complete_before=None), *requests[1:]), cuts)
    foreign = object()
    clones = {
        id(cut): replace(cut, owner=foreign, scope=(id(foreign), *cut.scope[1:]))
        for cut in cuts
    }
    moved = tuple(
        replace(
            item,
            transform_before=clones[id(item.transform_before)],
            store_before=clones[id(item.store_before)],
            complete_before=None
            if item.complete_before is None
            else clones[id(item.complete_before)],
        )
        for item in requests
    )
    with pytest.raises(chain._UnsupportedChain, match="original .*cut"):
        build(moved, tuple(clones.values()))
    original = plan.phases
    plan.phases = tuple(() for _ in cuts)
    try:
        with pytest.raises(
            chain._UnsupportedChain, match="state transfer decision changed"
        ):
            plan.check()
    finally:
        plan.phases = original
    plan.check()
    records.append((type(request.read), len(requests), len(plan.residency)))
    return plan


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_original_root_panels_use_shared_state_decisions(dtype):
    build = planner.plan_state_transfers
    records = []
    with (
        patch.object(
            planner,
            "plan_state_transfers",
            lambda requests, cuts: _observe(build, records, requests, cuts),
        ),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
    ):
        source = root_code("overlap64", dtype, 32)
    assert records and all(
        kind is FragmentStateBinding and requests == reads
        for kind, requests, reads in records
    )
    assert "prepared_tcgen_edge.execute_prepared_read" in source


def test_original_dv2_dual_values_use_one_original_read():
    build = planner.plan_state_transfers
    records = []
    with (
        patch.object(
            planner,
            "plan_state_transfers",
            lambda requests, cuts: _observe(build, records, requests, cuts),
        ),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
    ):
        source = epoch_code()
    assert records.count((DescriptorStateBinding, 2, 1)) == 4
    assert len(records) == 7
    assert source.count("prepared_tcgen_edge.pack_layout_f_state(") == 4
    assert source.count("prepared_tcgen_edge.scale_layout_f_state(") == 4
