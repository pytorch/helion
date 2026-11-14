"""A single MMA root resolves registered output strides from its config."""

from __future__ import annotations

import re

import pytest
import torch

from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_configured_strides import pitched_config
from helion._testing.cute_configured_strides import pitched_matmul

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("pitch", [64, 256])
def test_single_root_mma_output_pitch_matches_allocation(mode, pitch):
    with cpu_target():
        bound = cpu_bind(
            pitched_matmul,
            (
                torch.empty(2, 64, 16, dtype=torch.bfloat16),
                torch.empty(2, 16, 16, dtype=torch.bfloat16),
            ),
        )
        assert bound.config_spec.cute_mma_dag_phase_index is None
        assert len(bound.host_function.device_ir.root_ids) == 1
        source = bound.to_code(pitched_config(mode, pitch))
    pointer = next(
        line for line in source.splitlines() if "= out.iterator.toint()" in line
    )
    # Each group starts pitch rows apart, with 16 fp32 columns per row.
    assert re.search(rf"\* {pitch * 16 * 4}(?:\s|$)", pointer)
