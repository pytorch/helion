from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.rng_utils import philox_int32_ref
from helion._compiler.rng_utils import philox_randint_ref
import helion.language as hl
from helion.runtime.settings import _get_backend


@pytest.mark.parametrize(
    ("low", "high"),
    [(-(2**31), 2**31), (-(2**31), 1), (-17, 23)],
)
def test_randint_reference_range(low: int, high: int) -> None:
    offsets = torch.arange(1024, dtype=torch.int64)
    signed = philox_int32_ref(42, offsets).to(torch.int64)
    if high - low > 1 << 31:
        magnitude = signed & 0xFFFFFFFF
    else:
        magnitude = torch.where(signed < 0, -signed, signed)
    expected = (low + magnitude % (high - low)).to(torch.int32)
    actual = philox_randint_ref(42, offsets, low, high)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bool(((actual >= low) & (actual.to(torch.int64) < high)).all())
    if (low, high) == (-(2**31), 2**31):
        assert bool((actual < 0).any())
        assert bool((actual >= 0).any())


@helion.kernel(static_shapes=True, autotune_effort="none")
def _full_range_kernel(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    for tile in hl.tile(x.numel(), block_size=256):
        result[tile] = hl.randint([tile], -(2**31), 2**31, seed=42)
    return result


@helion.kernel(static_shapes=True, autotune_effort="none")
def _narrow_range_kernel(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    for tile in hl.tile(x.numel(), block_size=256):
        result[tile] = hl.randint([tile], -17, 23, seed=42)
    return result


@pytest.mark.skipif(
    not torch.cuda.is_available() or _get_backend() != "triton",
    reason="requires the CUDA Triton backend",
)
@pytest.mark.parametrize(("low", "high"), [(-(2**31), 2**31), (-17, 23)])
def test_randint_device_range(low: int, high: int) -> None:
    x = torch.empty((1024,), dtype=torch.int32, device="cuda")
    kernel = _full_range_kernel if low == -(2**31) else _narrow_range_kernel
    actual = kernel(x)
    offsets = torch.arange(x.numel(), dtype=torch.int64, device=x.device)
    signed = philox_int32_ref(42, offsets).to(torch.int64)
    if high - low > 1 << 31:
        magnitude = signed & 0xFFFFFFFF
    else:
        magnitude = torch.where(signed < 0, -signed, signed)
    expected = (low + magnitude % (high - low)).to(torch.int32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
