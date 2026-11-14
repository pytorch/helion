"""Partition counts tune internal workspaces while preserving final output."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_ordered_reference import local_histogram_phases
from helion._testing.cute_ordered_reference import partitioned_dot_sum


@skipIfNotCUDA()
@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("splits", [1, 2, 4, 8])
def test_partition_count_preserves_end_to_end_result(splits):
    x = (torch.arange(7 * 8, device=DEVICE).reshape(7, 8) % 4).to(torch.bfloat16)
    y = (torch.arange(8 * 8 * 11, device=DEVICE).reshape(8, 8, 11) % 3).to(
        torch.bfloat16
    )
    bound = partitioned_dot_sum.bind((x, y))
    split_spec = bound.config_spec.block_sizes[0]
    assert split_spec.min_size == 1 and split_spec.max_size == 8
    config = dict(bound.config_spec.autotune_reference_config())
    config["block_sizes"] = [splits, *config["block_sizes"][1:]]
    assert helion.Config.from_dict(config).pid_type == "flat"
    output = bound.compile_config(helion.Config.from_dict(config))(x, y)
    left = (x + x.amax(dim=0, keepdim=True)).float()
    expected = (left[None, :, :] @ y.float()).sum(dim=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@skipIfNotCUDA()
@skipUnlessCuteAvailable("requires CuTe DSL")
def test_local_atomic_reference_preserves_fresh_histograms():
    x = torch.arange(3 * 11, device=DEVICE).reshape(3, 11).to(torch.int32)
    bound = local_histogram_phases.bind((x,))
    compiled = bound.compile_config(bound.config_spec.autotune_reference_config())
    for delta in (0, 1):
        changed = x + delta
        expected = torch.nn.functional.one_hot((changed % 8).long(), 8).sum(dim=1)
        torch.testing.assert_close(
            compiled(changed), (expected * 2).float(), rtol=0, atol=0
        )
