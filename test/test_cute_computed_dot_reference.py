"""Completed half dot values stay correct across dtype rounding and replay."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_computed_dot_reference import computed_dot_chain


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "dtype,computed_rhs,columns",
    [
        (torch.float16, False, 128),
        (torch.float16, True, 128),
        (torch.bfloat16, False, 128),
        (torch.bfloat16, True, 128),
        (torch.bfloat16, False, 16),
    ],
)
def test_computed_half_dot_reference(dtype, computed_rhs, columns):
    a = ((torch.arange(32 * 64, device=DEVICE).reshape(32, 64) % 7 + 9) / 16).to(dtype)
    b = (
        (torch.arange(64 * columns, device=DEVICE).reshape(64, columns) % 5 + 11) / 16
    ).to(dtype)
    c = (
        (torch.arange(columns * 32, device=DEVICE).reshape(columns, 32) % 3 - 1) / 8
    ).to(dtype)
    args = (a, b, c, computed_rhs)
    bound = helion.kernel(computed_dot_chain, backend="cute", static_shapes=True).bind(
        args
    )
    compiled = bound.compile_config(bound.config_spec.autotune_reference_config())
    for replay in range(2):
        if replay:
            a.neg_()
            b.copy_(b.roll(1, 0))
            c.copy_(c.roll(3, 1))
        unrounded = a.cpu().float() @ b.cpu().float()
        intermediate = unrounded.to(dtype).float()
        assert not torch.equal(intermediate, unrounded)
        expected = intermediate @ c.cpu().float()
        assert not torch.equal(expected, unrounded @ c.cpu().float())
        torch.testing.assert_close(compiled(*args).cpu(), expected, rtol=0, atol=0)
