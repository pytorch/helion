"""Inductor's lowerings compute host scalars as numbers, not symbols."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(static_shapes=True, autotune_effort="none")
def compare_to_scalar(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] > 0.1
    return out


@skipIfRefEager("renders generated code; ref mode generates none")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scalar_compared_with_a_16_bit_tensor_is_a_number(dtype: torch.dtype) -> None:
    """Inductor rounds the scalar of a comparison with a 16-bit tensor
    through ``torch.tensor(v, dtype).item()`` (``promote_constants``).  Run
    under Helion's fake mode, that ``item()`` was an unbacked SymFloat the
    generated code read without declaring (``zuf0``); outside it, it is the
    rounded number."""
    with cpu_render_target(fake_device_sm_count=148):
        code = cpu_bind(compare_to_scalar, (torch.randn(64, dtype=dtype),)).to_code(
            helion.Config(block_sizes=[32])
        )
    assert "zuf" not in code, code
    rounded = torch.tensor(0.1, dtype=dtype).item()
    assert repr(rounded) in code, code
