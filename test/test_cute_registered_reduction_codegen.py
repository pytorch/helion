"""An untiled registered block size still owns a full reduction extent."""

from __future__ import annotations

import dataclasses

import pytest
import torch

import helion
from helion._compiler.device_ir_analysis import DeviceIRAnalysis
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import stage_sources
from helion._testing.registered_reduction import partitioned_copy_sum

pytestmark = pytest.mark.cpu_only


def test_registered_extent_has_a_consumer_reduction_axis():
    with cpu_target():
        bound = cpu_bind(partitioned_copy_sum, (torch.ones(7, 11),))
        # The registered count has a full-domain reduction alias. Until
        # rolling tracks its original-symbol loads, no chunk is searchable.
        with bound.env:
            registered = bound.env.block_sizes[
                bound.config_spec.block_sizes[0].block_id
            ]
            alias = next(
                info
                for info in bound.env.block_sizes
                if info.reduction
                and isinstance(info.size, torch.SymInt)
                and info.size._sympy_() == registered.symbol()
            )
            # A literal carry extent of 64 must not match the alias's fake
            # tracing hint. Only symbolic provenance can name this axis.
            graph = torch.fx.Graph()
            carry = graph.placeholder("unrelated_carry")
            carry.meta["val"] = torch.empty(64)
            with bound.host_function:
                analysis = DeviceIRAnalysis.build(
                    bound.host_function.device_ir, bound.env
                )
                analysis = dataclasses.replace(analysis, accumulator_inputs=(carry,))
                assert analysis.accumulator_facts(bound.env)[0].dim_block_ids == (None,)
            for parts in (1, 2, 4, 8):
                selected = helion.Config.from_dict(
                    dict(bound.config_spec.default_config()) | {"block_sizes": [parts]}
                )
                assert alias.from_config(selected) == parts
            assert (
                alias.block_id
                not in bound.config_spec.reduction_loops.valid_block_ids()
            )
        # Flat selects the explicit materialized phases for the barrier.
        config = dict(bound.config_spec.default_config()) | {
            "pid_type": "flat",
            "block_sizes": [4],
        }
        source = bound.to_code(helion.Config.from_dict(config))
    stages = stage_sources(source)
    assert len(stages) == 2
    # Consumer coordinates must own scalar pointer loads, not leave the
    # host TensorView indexed as if it were a device fragment.
    assert "workspace.iterator" in stages[1]
    assert "workspace[" not in stages[1]
    assert "_RDIM_SIZE_" in stages[1]
