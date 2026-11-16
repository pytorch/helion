from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_value_metadata_tma import check_metadata_tma_slots


@skipIfNotCUDA()
@pytest.mark.parametrize("read_overlap", [False, True])
def test_copied_metadata_then_tma_fallback_and_replay(read_overlap):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired metadata/TMA slot fixture targets SM100")
    check_metadata_tma_slots(DEVICE, read_overlap=read_overlap)
