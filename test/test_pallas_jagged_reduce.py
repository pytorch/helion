"""Stress tests for the per-item DMA jagged_reduce template.

Calls ``default_pallas_jagged_reduce_launcher`` directly with hand-built
inputs and compares against a torch reference. The launcher exercises
the template (``jagged_reduce_pallas``) end-to-end on TPU and through
the DLPack bridge in interpret mode.

These tests cover branches that ``test_examples.test_jagged_sum``
doesn't hit on its own — empty items, partial-tail tiles, lane padding
on M, bf16, single-item kernels — so they catch DMA / acc-init /
flush regressions even when the example test happens to pass.
"""

from __future__ import annotations

import unittest

import torch

from helion._testing import DEVICE
from helion._testing import LONG_INT_TYPE
from helion._testing import onlyBackends


def _torch_reference(jagged_data: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    num_items = offsets.shape[0] - 1
    out = torch.zeros(
        (num_items, jagged_data.shape[1]),
        dtype=jagged_data.dtype,
        device=jagged_data.device,
    )
    for i in range(num_items):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        if end > start:
            out[i] = (
                jagged_data[start:end]
                .sum(dim=0, dtype=torch.float32)
                .to(jagged_data.dtype)
            )
    return out


def _offsets_from_lengths(lengths: list[int]) -> torch.Tensor:
    cumsum = [0]
    for length in lengths:
        cumsum.append(cumsum[-1] + length)
    return torch.tensor(cumsum, dtype=LONG_INT_TYPE, device=DEVICE)


@onlyBackends(["pallas"])
class TestJaggedReduceTemplate(unittest.TestCase):
    """Direct stress tests for the jagged_reduce launcher + template."""

    def _check(
        self,
        lengths: list[int],
        m_actual: int,
        *,
        dtype: torch.dtype = torch.float32,
        jagged_tile_size: int = 64,
        atol: float = 1e-4,
        rtol: float = 1e-3,
    ) -> None:
        from helion.runtime import default_pallas_jagged_reduce_launcher

        offsets = _offsets_from_lengths(lengths)
        total_rows = int(offsets[-1])
        jagged_data = torch.randn(total_rows, m_actual, dtype=dtype, device=DEVICE)

        expected = _torch_reference(jagged_data, offsets)
        actual = default_pallas_jagged_reduce_launcher(
            jagged_data, offsets, jagged_tile_size=jagged_tile_size
        )

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_single_item(self) -> None:
        """One item, single-tile length — exercises prologue + epilogue
        without any inter-item ping-pong."""
        self._check([10], m_actual=128)

    def test_partial_tail_tile(self) -> None:
        """Item length not a multiple of jagged_tile_size — exercises
        the in-loop mask that zeroes out tail rows before the reduction."""
        self._check([70], m_actual=128, jagged_tile_size=64)

    def test_multi_tile_item(self) -> None:
        """Item spans multiple tile_size-row blocks — exercises the
        inner loop's prefetch chain across tiles within one item."""
        self._check([200], m_actual=128, jagged_tile_size=64)

    def test_many_items(self) -> None:
        """Many small items — exercises the per-item prefetch chain
        and the output-block ping-pong across item boundaries."""
        self._check([1, 2, 3, 17, 5, 8, 13, 21, 34, 55], m_actual=128)

    def test_lane_padded_m(self) -> None:
        """M_actual not a multiple of 128 — exercises the lane-padding
        in/out and the trailing-slice on the way back."""
        self._check([16, 32, 48], m_actual=80)

    def test_empty_item(self) -> None:
        """One item has length 0 — its row in the output must be zero
        (the caller-provided initial value), and the surrounding items
        must still produce correct sums."""
        self._check([8, 0, 16], m_actual=128)

    def test_bf16(self) -> None:
        """bf16 input/output — exercises the fp32-accumulator → o_dtype
        cast in the flush path."""
        self._check(
            [16, 32, 24], m_actual=128, dtype=torch.bfloat16, atol=1e-2, rtol=1e-2
        )

    def test_offsets_int64_accepted(self) -> None:
        """Launcher must accept int64 offsets and convert to int32
        before handing them to the TPU kernel."""
        from helion.runtime import default_pallas_jagged_reduce_launcher

        offsets = torch.tensor([0, 10, 25], dtype=torch.int64, device=DEVICE)
        jagged_data = torch.randn(25, 128, dtype=torch.float32, device=DEVICE)
        expected = _torch_reference(jagged_data, offsets)
        actual = default_pallas_jagged_reduce_launcher(jagged_data, offsets)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
