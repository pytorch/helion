"""Runtime numerics of comparison masks in the tcgen05 fused epilogue (sm_100 GPU).

``out = acc.to(fp16) * (c[tile] > 0.3)`` compares ``c`` against fp16(0.3), as
torch eager, the SIMT lowering and the Triton backend do; a row planted at
exactly fp16(0.3) separates that from a comparison against the wider 0.3.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="tcgen05 requires an sm_100 GPU",
    ),
]


def _masked_gemm_gt(x: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype) * (c[tile_m, tile_n] > 0.3)
    return out


def _masked_gemm_eq(x: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype) * (c[tile_m, tile_n] == 0.3)
    return out


@pytest.mark.parametrize(
    ("kernel_fn", "mask"),
    [
        (_masked_gemm_gt, lambda c: c > 0.3),
        (_masked_gemm_eq, lambda c: c == 0.3),
    ],
    ids=["gt", "eq"],
)
def test_fp16_comparison_mask_matches_eager(
    kernel_fn: Callable[..., torch.Tensor],
    mask: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    torch.manual_seed(0)
    m, k, n = 256, 128, 256
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    w = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
    c = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
    planted = torch.tensor(0.3, dtype=torch.float16)
    rows = {
        0: planted,
        1: torch.nextafter(planted, torch.tensor(0.0, dtype=torch.float16)),
        2: torch.nextafter(planted, torch.tensor(1.0, dtype=torch.float16)),
    }
    for row, value in rows.items():
        c[row, :64] = value.item()
    kernel = helion.kernel(
        kernel_fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel.bind((x, w, c))
    assert bound.config_spec.cute_tcgen05_search_enabled
    config = bound.config_spec.default_config()
    bound.set_config(config)
    source = bound.to_code(config)
    assert "tcgen05_aux" in source
    assert "0.300048828125" in source and " 0.3\n" not in source
    out = bound(x, w, c)
    torch.cuda.synchronize()
    expected = (x.float() @ w.float()).to(torch.float16) * mask(c)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)
    for row in rows:
        eager = bool(mask(c[row, :64]).any().item())
        assert bool(out[row, :64].ne(0).any().item()) == eager, row
