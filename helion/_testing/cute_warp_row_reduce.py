"""Native fixture for exact FP32 warp-row reduction and changed-input replay."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch

from helion._compiler.cute.device.warp_reduce import warp_row_sum


def reference_row_sums(values: torch.Tensor) -> torch.Tensor:
    """Independent per-row ascending-XOR tree, without ownership transposes."""
    lanes = torch.arange(32, device=values.device)
    result = values.clone()
    for mask in (1, 2, 4, 8, 16):
        result = result + result[..., lanes ^ mask]
    return result[..., 0]


def assert_same_float32_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare non-NaN bits (including signed zero) and NaN classification."""
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    valid = ~torch.isnan(expected)
    assert torch.equal(
        actual[valid].view(torch.int32), expected[valid].view(torch.int32)
    )


@cute.kernel
def _reduce_rows(values: cute.Tensor, output: cute.Tensor) -> None:
    thread = cute.arch.thread_idx()[0]
    warp = thread // 32
    lane = thread % 32
    # One compilation covers every supported row-list size, each with separate
    # input/output slots. Every call is reached by all 32 lanes of each warp.
    for power in cutlass.range_constexpr(6):
        count = 1 << power
        offset = count - 1
        row, total = warp_row_sum(
            [
                cutlass.Float32(values[warp, offset + index, lane])
                for index in range(count)
            ],
            lane,
        )
        if lane < count:
            output[warp, offset + row] = total


@cute.jit
def launch(values: cute.Tensor, output: cute.Tensor, stream: object) -> None:
    _reduce_rows(values, output).launch(
        grid=(1, 1, 1), block=(128, 1, 1), stream=stream
    )
