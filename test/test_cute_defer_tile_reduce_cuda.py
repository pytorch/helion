"""GPU tests: ``cute_defer_tile_reduce`` sums every element once, whichever
way the rows' threads are laid out: exactly where every partial sum is exact,
and within the rounding of another summation order otherwise."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.split_reduction_kernels import SPLIT_CONFIG
from helion._testing.split_reduction_kernels import cross_thread_combines
from helion._testing.split_reduction_kernels import split_column_sums
from helion._testing.split_reduction_kernels import split_column_sums_reference

# 600-row splits of 4700 rows: the last split is partial (500 rows), and so
# is the last 128-row tile of every split.
_ROWS, _ROWS_PER_SPLIT = 4700, 600


@pytest.mark.parametrize(
    "layout",
    [
        {},
        {
            "block_sizes": [1, 32, 512],
            "num_threads": [1, 4, 128],
            "cute_lane_unroll": 4,
        },
        {"cute_vloop_sink": False, "cute_lane_unroll": 1},
    ],
    ids=["one_warp", "shared_memory", "unsunk"],
)
def test_deferred_sums_add_every_row_once(layout: dict[str, object]) -> None:
    config = {**SPLIT_CONFIG, **layout}
    kernels = [
        helion.kernel(
            split_column_sums.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config.from_dict({**config, "cute_defer_tile_reduce": defer}),
        )
        for defer in (False, True)
    ]
    bound = kernels[1].bind(
        (
            torch.empty(_ROWS, 64, dtype=torch.bfloat16, device=DEVICE),
            torch.empty(_ROWS, 64, dtype=torch.bfloat16, device=DEVICE),
            _ROWS_PER_SPLIT,
        )
    )
    code = bound.to_code(bound.config_spec.normalized_config(kernels[1].configs[0]))
    # No cross-thread combine is left inside the row loop.
    assert cross_thread_combines(code)[0] == 0

    torch.manual_seed(0)
    # Small integers: every product and partial sum is exact in fp32, so
    # the order of the adds cannot show.
    x = torch.randint(-4, 5, (_ROWS, 64), device=DEVICE).to(torch.bfloat16)
    y = torch.randint(-4, 5, (_ROWS, 64), device=DEVICE).to(torch.bfloat16)
    expected = split_column_sums_reference(x, y, _ROWS_PER_SPLIT)
    for kernel in kernels:
        for actual, reference in zip(
            kernel(x, y, _ROWS_PER_SPLIT), expected, strict=True
        ):
            assert torch.equal(actual.double(), reference)
    # Random values: the deferred combine adds in another order.
    x = torch.randn(_ROWS, 64, device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn(_ROWS, 64, device=DEVICE, dtype=torch.bfloat16)
    expected = split_column_sums_reference(x, y, _ROWS_PER_SPLIT)
    for kernel in kernels:
        for actual, reference in zip(
            kernel(x, y, _ROWS_PER_SPLIT), expected, strict=True
        ):
            torch.testing.assert_close(actual.double(), reference, rtol=1e-4, atol=1e-3)
