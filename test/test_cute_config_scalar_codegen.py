"""Block-size scalars are configuration constants in uniform expressions."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_config_scalar import repeated_dot
from helion._testing.cute_config_scalar import tile_position_bound

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("repeats", [1, 8])
def test_registered_scalar_loop_bound_is_selected_from_config(mode, repeats):
    with cpu_target():
        bound = cpu_bind(
            repeated_dot,
            (
                torch.empty(64, 16, dtype=torch.bfloat16),
                torch.empty(16, 16, dtype=torch.bfloat16),
            ),
        )
        assert bound.config_spec.cute_mma_dag_facts is not None
        source = bound.to_code(
            helion.Config(
                block_sizes=[repeats, 64, 16],
                cute_mma_dag=mode,
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )
        assert "_helion_mma_dag" in source
        assert f"_BLOCK_SIZE_0 = {repeats}" in source
        assert "cutlass.range(0, _BLOCK_SIZE_0)" in source


def test_device_tile_position_is_not_a_configuration_scalar():
    with cpu_target():
        bound = cpu_bind(
            tile_position_bound,
            (
                torch.empty(64, 16, dtype=torch.bfloat16),
                torch.empty(16, 16, dtype=torch.bfloat16),
            ),
        )
        assert bound.config_spec.cute_mma_dag_facts is None
