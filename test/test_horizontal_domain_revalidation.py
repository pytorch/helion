from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest
import torch

from .test_cute_horizontal_state_update import _with_region
from helion._compiler.cute.horizontal_state_update import InapplicableStateUpdate
from helion._compiler.cute.horizontal_state_update import PointwiseStateRegion
from helion._compiler.cute.horizontal_state_update import bind_state_task_domain
from helion.language import view_ops


def test_domain_check_validates_current_owners_once_per_call():
    def check(region, graphs):
        domain = bind_state_task_domain(region)
        original = PointwiseStateRegion.check_readonly_owners
        observed = []

        def checked(current):
            observed.append(current)
            return original(current)

        with patch.object(PointwiseStateRegion, "check_readonly_owners", checked):
            for _ in range(3):
                domain.check()
        assert observed == [region, region, region]

    _with_region(check)


@pytest.mark.parametrize(
    "mutation", ["store_index", "mask", "source", "dtype", "view", "extra_write"]
)
def test_domain_rejects_graph_mutation_after_success(mutation):
    def check(region, graphs):
        domain = bind_state_task_domain(region)
        domain.check()
        node = region.state_store
        if mutation == "store_index":
            args = list(node.args)
            selectors = list(args[1])
            selectors[0] = 0
            args[1] = selectors
            node.args = tuple(args)
        elif mutation == "mask":
            node.args = (*node.args[:3], True)
        elif mutation == "source":
            node.args = (region.output_store.args[0], *node.args[1:])
        elif mutation == "dtype":
            region.update.meta["val"] = torch.empty((64, 32), dtype=torch.float64)
        elif mutation == "view":
            view = next(
                n for n in graphs[0].graph.nodes if n.target is view_ops.subscript
            )
            view.args = (view.args[0], [slice(None, None, -1)])
        else:
            with graphs[0].graph.inserting_before(node):
                graphs[0].graph.call_function(node.target, node.args)
        with pytest.raises(InapplicableStateUpdate, match="graph changed"):
            domain.check()

    _with_region(check)


def test_domain_rejects_same_abi_storage_replacement_after_success():
    def check(region, graphs):
        domain = bind_state_task_domain(region)
        domain.check()
        tensor = region.state_load.args[0].meta["val"]
        old_owner = tensor.untyped_storage()._cdata
        with tensor.fake_mode:
            tensor.set_(
                torch.empty_strided(
                    tensor.shape,
                    tensor.stride(),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            )
        assert old_owner != tensor.untyped_storage()._cdata
        with pytest.raises(InapplicableStateUpdate, match="graph changed"):
            domain.check()

    _with_region(check)


def test_domain_rechecks_current_alias_authority_after_success():
    def check(region, graphs):
        domain = bind_state_task_domain(region)
        domain.check()
        with (
            patch(
                "helion._compiler.cute.memory_ops.runtime_tensor_sources_are_proven_disjoint",
                return_value=False,
            ),
            pytest.raises(InapplicableStateUpdate, match="alias"),
        ):
            domain.check()
        domain.check()

    _with_region(check)


@pytest.mark.parametrize(
    "changes",
    [
        {"grid_block_ids": (-1,)},
        {"grid_extents": (3, 2, 32)},
        {"row_block_id": -1},
        {"state_shape": (3, 2, 32, 32)},
        {"state_strides": (1, 1, 1, 1)},
        {"index_dtype": torch.float32},
        {"storage_dtype": torch.float64},
    ],
)
def test_domain_rejects_stale_fields_after_success(changes):
    def check(region, graphs):
        domain = bind_state_task_domain(region)
        domain.check()
        stale = dataclasses.replace(domain, **changes)
        with pytest.raises(InapplicableStateUpdate, match="domain changed"):
            stale.check()
        domain.check()

    _with_region(check)
