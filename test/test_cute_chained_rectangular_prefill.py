from __future__ import annotations

import pytest

from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from helion._testing import skipUnlessBackends


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_rectangular_prefill_gpu_preserves_ragged_source_and_reused_slot(
    bt32, value_tile, consumer_warps
):
    _run_preparation_prefill(bt32, value_tile, consumer_warps, rectangular_leaf=True)
