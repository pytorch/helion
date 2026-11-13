from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_rowwise_native import check_rowwise_native


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "query_len,threshold,scoped",
    [(4, None, False), (5, None, False), (5, 16, False), (5, None, True)],
)
def test_rowwise_source_outputs_and_changed_graph(query_len, threshold, scoped):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("The attention helpers use SM100 tensor memory")
    check_rowwise_native(query_len, DEVICE, threshold=threshold, scoped=scoped)
