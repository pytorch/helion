from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

import helion
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.variable_origin import TileBeginOrigin
from helion._compiler.variable_origin import TileIdOrigin
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_mma_root_coordinates import root_coordinate_matmul

if TYPE_CHECKING:
    import sympy

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("kind", ["begin", "id", "mixed", "captured"])
def test_root_coordinates_in_mma_recipes(mode: str, kind: str) -> None:
    args = (
        torch.empty(128, 32, dtype=torch.bfloat16),
        torch.empty(32, 64, dtype=torch.bfloat16),
        kind,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(root_coordinate_matmul, backend="cute", static_shapes=True),
            args,
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            assert any(
                isinstance(
                    bound.host_function.expr_to_origin[
                        cast("sympy.Symbol", symbol)
                    ].origin,
                    (TileBeginOrigin, TileIdOrigin),
                )
                for expr in program.aux_indices.values()
                for symbol in expr.free_symbols
            )
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    marker = "mma_bf16" if mode == "warp" else "tcgen05"
    assert marker in source
    if kind == "mixed":
        assert "md_i" in source
