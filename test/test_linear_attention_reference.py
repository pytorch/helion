"""``chunked_linear_attn_reference`` under deep intra-chunk decay: the pairwise
decay ``exp(g_i - g_j)`` is masked to ``j <= i`` before the exp, so a chunk
whose cumulated decay passes fp32's exp range (88) no longer turns the masked
upper triangle into ``inf * 0 = NaN`` (mamba2_ssd backward at the campaign
shapes).  The diagonal-decay branch keeps its factored ``exp(g_i) * exp(-g_j)``
form (the pairwise form costs the full_gla baselines 1.5-3x at the large campaign
shapes), so its cases stay below that range."""

from __future__ import annotations

import torch

from helion._testing import TestCase
from helion._testing.linear_attention_utils import chunked_linear_attn_reference
from helion._testing.linear_attention_utils import make_mamba2_inputs
from helion._testing.linear_attention_utils import naive_recurrent_reference
from helion._testing.linear_attention_utils import rel_error

# The reference is pure PyTorch: these tests run without a GPU.
CPU = torch.device("cpu")


def _deep_decay_inputs(
    *, diagonal: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    B, H, T, D, DV = 1, 2, 256, 16, 16
    q, k, v, _, _ = make_mamba2_inputs(B, H, T, D, DV, dtype=torch.bfloat16, device=CPU)
    # Scalar decay: a chunk of 64 steps at -3 cumulates to -192: exp(192)
    # overflows fp32, exp(-192) underflows it.  Diagonal decay keeps the
    # factored form, so its chunk cumulates to -64 (inside fp32's exp range).
    shape = (B, H, T, D) if diagonal else (B, H, T)
    g = torch.full(shape, -1.0 if diagonal else -3.0, dtype=torch.bfloat16)
    return q, k, v, g


class TestChunkedReferenceDeepDecay(TestCase):
    def test_scalar_decay_matches_the_recurrent_reference(self) -> None:
        q, k, v, g = _deep_decay_inputs(diagonal=False)
        out = chunked_linear_attn_reference(q, k, v, g, C=64)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(rel_error(out, naive_recurrent_reference(q, k, v, g)), 2e-2)

    def test_diagonal_decay_matches_the_recurrent_reference(self) -> None:
        q, k, v, g = _deep_decay_inputs(diagonal=True)
        out = chunked_linear_attn_reference(q, k, v, g, C=64)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(rel_error(out, naive_recurrent_reference(q, k, v, g)), 2e-2)

    def test_backward_through_deep_decay_is_finite(self) -> None:
        for diagonal in (False, True):
            with self.subTest(diagonal=diagonal):
                q, k, v, g = _deep_decay_inputs(diagonal=diagonal)
                q, k, v = (t.clone().requires_grad_(True) for t in (q, k, v))
                out = chunked_linear_attn_reference(q, k, v, g, C=64)
                torch.manual_seed(1)
                out.backward(torch.randn_like(out))
                for grad in (q.grad, k.grad, v.grad):
                    assert grad is not None
                    self.assertTrue(torch.isfinite(grad).all())

    def test_mamba2_campaign_shape_is_finite(self) -> None:
        # The campaign's shape 3 and its input seed: 192 NaN outputs before the
        # exponent was masked (two chunks cumulated past -88).
        torch.manual_seed(20260911 + 3)
        q, k, v, g, scale = make_mamba2_inputs(
            8, 8, 1024, 64, 64, dtype=torch.bfloat16, device=CPU
        )
        out = chunked_linear_attn_reference(q * scale, k, v, g, C=64)
        self.assertTrue(torch.isfinite(out).all())
