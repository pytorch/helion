from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def exact_fp32_sum_input(shape: Sequence[int]) -> torch.Tensor:
    """Make signed CPU inputs whose per-row sums are exact in every order.

    Values are integer multiples of 1/1024. The bound on the sum of absolute
    integer numerators keeps every partial sum below 2**24 in those units,
    so FP32 additions cannot round. A private CPU generator makes the data
    identical across backends without advancing their global RNG state.
    """
    assert math.prod(shape[1:]) * 2048 < 2**24
    generator = torch.Generator(device="cpu").manual_seed(0)
    numerators = torch.randint(
        -2048, 2049, tuple(shape), generator=generator, device="cpu"
    )
    return numerators.to(torch.float32) / 1024
