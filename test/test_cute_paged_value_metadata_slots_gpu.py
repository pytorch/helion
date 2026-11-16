from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_value_metadata import HIGH_PAGE_STRIDE
from helion._testing.cute_paged_value_metadata import check_metadata_slots


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "page_stride,sweep_words,read_overlap",
    [
        (384, True, False),
        (384, True, True),
        # This sparse view needs over 8 GiB of backing storage, exceeding a
        # default parallel GPU worker's budget once its context is included.
        pytest.param(HIGH_PAGE_STRIDE, False, True, marks=pytest.mark.slow),
    ],
)
def test_normalized_metadata_slots_and_capture_replay(
    page_stride: int, sweep_words: bool, read_overlap: bool
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired metadata slot fixture targets SM100")
    check_metadata_slots(
        page_stride, DEVICE, sweep_words=sweep_words, read_overlap=read_overlap
    )
