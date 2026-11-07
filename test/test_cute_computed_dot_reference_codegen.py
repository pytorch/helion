"""Conservative CuTe lowering owns computed half contraction operands."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.computed_fragment import computed_fragment_supported
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_computed_dot_reference import computed_dot_chain
from helion._testing.cute_computed_dot_reference import pointwise_operand_dot

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("computed_rhs", [False, True])
@pytest.mark.parametrize("columns", [16, 128])
def test_reference_owns_completed_half_dot(computed_rhs, columns):
    args = (
        torch.empty((32, 64), dtype=torch.bfloat16),
        torch.empty((64, columns), dtype=torch.bfloat16),
        torch.empty((columns, 32), dtype=torch.bfloat16),
        computed_rhs,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(computed_dot_chain, backend="cute", static_shapes=True), args
        )
        config = bound.config_spec.autotune_reference_config()
        config.config["cute_fragment_half_mma"] = False
        config.config["cute_mma_dag"] = "off"
        source = bound.to_code(config)
    assert "fragment_dot" in source
    assert "_helion_mma_dag" not in source


def test_pointwise_only_half_operand_keeps_existing_owner():
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(pointwise_operand_dot, backend="cute", static_shapes=True),
            (
                torch.empty((32, 64), dtype=torch.float16),
                torch.empty((64, 128), dtype=torch.float16),
            ),
        )
        with bound.env, bound.host_function:
            assert not computed_fragment_supported(
                bound.env, bound.host_function.device_ir.graphs
            )
        source = bound.to_code(bound.config_spec.autotune_reference_config())
    assert "fragment_dot" not in source
