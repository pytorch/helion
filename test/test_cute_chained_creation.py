from __future__ import annotations

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _filled_contraction(a, b, scale: float):
    m, k = a.shape
    n = b.shape[-1]
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for rows, columns in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        multiplier = hl.full([], scale * 0.5, dtype=torch.float16).float()
        accumulator = hl.full([rows, columns], scale, dtype=torch.bfloat16).float()
        weighted = (a[rows, kk].float() * multiplier).to(a.dtype)
        out[rows, columns] = hl.dot(
            weighted, b[kk, columns], acc=accumulator, out_dtype=torch.float32
        )
    return out


@pytest.mark.parametrize("schedule", ["cp_async_register", "tcgen05_tmem"])
def test_scalar_and_tile_full_use_common_lowering(schedule: str) -> None:
    with _cpu_codegen():
        bound = _filled_contraction._bind_isolated(
            (
                torch.empty((128, 16), dtype=torch.bfloat16),
                torch.empty((16, 32), dtype=torch.bfloat16),
                0.3333,
            )
        )
        source = bound.to_code(
            helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
        )
    assert "chain_0_mma" in source and "chain_0_seed" in source
    assert "cutlass.Float16(" in source and "cutlass.BFloat16(" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("schedule", ["cp_async_register", "tcgen05_tmem"])
@pytest.mark.parametrize("scale", [0.3333, -0.1271])
def test_filled_contraction_preserves_narrowing_gpu(
    schedule: str, scale: float
) -> None:
    torch.manual_seed(732)
    a = torch.randn((128, 16), dtype=torch.bfloat16, device=DEVICE) * 0.1
    b = torch.randn((16, 32), dtype=torch.bfloat16, device=DEVICE) * 0.1
    bound = _filled_contraction._bind_isolated((a, b, scale))
    config = helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
    source = bound.to_code(config)
    assert "chain_0_mma" in source
    actual = bound.compile_config(config)(a, b, scale)
    weight = torch.tensor(scale * 0.5, dtype=torch.float16, device=DEVICE).float()
    seed = torch.tensor(scale, dtype=torch.bfloat16, device=DEVICE).float()
    expected = (a.float() * weight).to(a.dtype).float() @ b.float() + seed
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)
