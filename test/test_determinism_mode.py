from __future__ import annotations

import torch

from helion.autotuner.base_search import _FillEmptyWithNaNMode


class TestFillEmptyWithNaNMode:
    def test_fills_float_empty_with_nan(self) -> None:
        with _FillEmptyWithNaNMode():
            t = torch.empty(4, 4, device="cuda")
        assert t.isnan().all()

    def test_fills_empty_like_with_nan(self) -> None:
        x = torch.zeros(4, 4, device="cuda")
        with _FillEmptyWithNaNMode():
            t = torch.empty_like(x)
        assert t.isnan().all()

    def test_fills_empty_strided_with_nan(self) -> None:
        with _FillEmptyWithNaNMode():
            t = torch.empty_strided((4, 4), (4, 1), device="cuda")
        assert t.isnan().all()

    def test_fills_new_empty_with_nan(self) -> None:
        x = torch.zeros(2, device="cuda")
        with _FillEmptyWithNaNMode():
            t = x.new_empty(4, 4)
        assert t.isnan().all()

    def test_does_not_fill_integer_empty(self) -> None:
        with _FillEmptyWithNaNMode():
            t = torch.empty(4, 4, dtype=torch.int32, device="cuda")
        # Integer tensors cannot hold NaN, should not be modified
        assert not t.isnan().any()

    def test_does_not_affect_zeros(self) -> None:
        with _FillEmptyWithNaNMode():
            t = torch.zeros(4, 4, device="cuda")
        assert (t == 0).all()

    def test_does_not_affect_ones(self) -> None:
        with _FillEmptyWithNaNMode():
            t = torch.ones(4, 4, device="cuda")
        assert (t == 1).all()

    def test_does_not_affect_full(self) -> None:
        with _FillEmptyWithNaNMode():
            t = torch.full((4, 4), 42.0, device="cuda")
        assert (t == 42.0).all()

    def test_equal_nan_comparison(self) -> None:
        """Both baseline and trial produce NaN in unwritten positions."""
        with _FillEmptyWithNaNMode():
            a = torch.empty(8, device="cuda")
        with _FillEmptyWithNaNMode():
            b = torch.empty(8, device="cuda")
        # Write to first 4 elements only
        a[:4] = 1.0
        b[:4] = 1.0
        # With equal_nan=True, both should match (NaN == NaN)
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0, equal_nan=True)
