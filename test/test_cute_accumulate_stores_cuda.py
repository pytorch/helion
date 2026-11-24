"""GPU tests: ``cute_accumulate_stores`` stores the same values as the
knob-off code, bit for bit, and the same sums but for the rounding of the
products: the knob-off accumulate pass reads each product once, which
``fuse_fma`` contracts into the accumulating add, where the store reads it
too with the knob on."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.accumulate_store_kernels import SEED_CONFIG
from helion._testing.accumulate_store_kernels import store_and_column_sums
from helion._testing.accumulate_store_kernels import store_and_column_sums_reference


@pytest.mark.parametrize(
    ("shape", "dtype", "layout"),
    [
        ((8, 200, 64), torch.bfloat16, {}),
        ((3, 128, 40), torch.float32, {"cute_vector_widths": [1, 4, 1]}),
    ],
    ids=["bf16_rows_off_the_tile", "fp32_columns_off_the_tile"],
)
def test_stores_and_sums_match_the_knob_off_code(
    shape: tuple[int, int, int], dtype: torch.dtype, layout: dict[str, object]
) -> None:
    config = {**SEED_CONFIG, **layout}
    kernels = [
        helion.kernel(
            store_and_column_sums.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config.from_dict({**config, "cute_accumulate_stores": knob}),
        )
        for knob in (False, True)
    ]
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    (y_off, total_off), (y_on, total_on) = (kernel(x) for kernel in kernels)
    assert torch.equal(y_on, y_off)
    torch.testing.assert_close(total_on, total_off, rtol=1e-6, atol=1e-5)
    y_ref, total_ref = store_and_column_sums_reference(x)
    torch.testing.assert_close(y_on, y_ref)
    torch.testing.assert_close(total_on.double(), total_ref, rtol=1e-4, atol=1e-3)
