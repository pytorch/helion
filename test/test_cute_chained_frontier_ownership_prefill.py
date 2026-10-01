from __future__ import annotations

import pytest

from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from helion._testing import skipUnlessBackends


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("leaf_count", [1, 4])
@pytest.mark.parametrize("operand_retention", [False, True])
def test_retiled_frontier_preserves_original_ragged_prefill_oracle_gpu(
    leaf_count, operand_retention
):
    _run_preparation_prefill(
        True,
        128,
        4,
        rectangular_leaf=True,
        leaf_count=leaf_count,
        seed_columns=32,
        cache_layout="xor",
        vector_group=True,
        cohorts=3,
        preparation_unroll=1,
        register_islands=True,
        operand_retention=operand_retention,
        frontier_tile_columns=32,
    )
