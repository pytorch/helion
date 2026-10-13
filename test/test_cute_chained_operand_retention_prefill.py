from __future__ import annotations

import pytest

from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from helion._testing import skipUnlessBackends


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("leaf_count", [1, 4])
@pytest.mark.parametrize("collective_retention", [False, True])
@pytest.mark.parametrize("cohorts,register_islands", [(1, False), (3, True)])
def test_retained_operands_preserve_original_ragged_prefill_oracle_gpu(
    leaf_count, collective_retention, cohorts, register_islands
):
    # Original Triton reference, all tolerances, empty/ragged sequence inputs,
    # FP32 initial/final state and replay/input assertions remain unchanged.
    _run_preparation_prefill(
        True,
        128,
        4,
        rectangular_leaf=True,
        leaf_count=leaf_count,
        seed_columns=32,
        cache_layout="xor",
        # Do not request an ineffective CSE choice after retention removes
        # its last shared expression; ownership-only grouping is not CSE.
        vector_group=cohorts > 1 and not collective_retention,
        cohorts=cohorts,
        preparation_unroll=1,
        register_islands=register_islands,
        collective_retention=collective_retention,
        operand_retention=True,
    )
