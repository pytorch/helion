"""TMA output origins for complete and partial two-CTA MXFP8 tiles."""

from __future__ import annotations

import pytest
import torch

from test.test_cute_mxfp8_mma import _config
from test.test_cute_mxfp8_packed_provenance import _packed_mxfp8

from helion._compiler.cute.block_scaled_config import MXFP8_EPILOGUE_MODES
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("m,n", [(128, 256), (257, 264)])
@pytest.mark.parametrize("mode", tuple(MXFP8_EPILOGUE_MODES)[1:])
@pytest.mark.parametrize("inplace", [False, True])
def test_cuda_two_cta_tma_output_origins(
    m: int, n: int, mode: str, inplace: bool
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native MXFP8 required")
    k = 512
    torch.manual_seed(5)
    a = (torch.randn((m, k), device=DEVICE) / 3).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device=DEVICE) / 3).to(torch.float8_e4m3fn)
    sa = torch.randint(125, 129, (m, k // 32), device=DEVICE, dtype=torch.uint8)
    sb = torch.randint(125, 129, (n, k // 32), device=DEVICE, dtype=torch.uint8)
    packed = [
        torch.empty_strided(
            (rows, k // 128),
            (1, (rows + 3) // 4 * 4),
            device=DEVICE,
            dtype=torch.int32,
        )
        for rows in (m, n)
    ]
    packed[0].copy_(sa.view(torch.int32))
    packed[1].copy_(sb.view(torch.int32))
    out = torch.full((m, n), float("nan"), device=DEVICE, dtype=torch.bfloat16)
    values = (a, b, *packed, out)
    bound = _packed_mxfp8._bind_isolated(values)
    config = _config(
        bound,
        cute_scaled_tile_n=128,
        cute_scaled_tile_k=128,
        cute_scaled_cluster_m=2,
        cute_scaled_stages=2,
        cute_scaled_fused_scales=True,
        cute_scaled_sf_tma=inplace,
        cute_scaled_sf_inplace=inplace,
        cute_scaled_epilogue_mode=mode,
    )
    actual = bound.compile_config(config)(*values)
    expected = (
        (a.double() * torch.exp2(sa.double() - 127).repeat_interleave(32, dim=1))
        @ (b.double() * torch.exp2(sb.double() - 127).repeat_interleave(32, dim=1)).T
    ).bfloat16()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.016)
