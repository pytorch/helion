"""N64 native NVFP4 tiles: two N64 tiles share one 128-row prepared SFB chunk."""

from __future__ import annotations

from examples.nvfp4_gemm import _nvfp4_w4a4_matmul_kernel
from examples.nvfp4_gemm import pack_fp4
from examples.nvfp4_gemm import pack_fp4_last_dim
from examples.nvfp4_gemm import quantize_fp4_e2m1
from examples.nvfp4_gemm import reference_nvfp4_w4a4_matmul
from examples.nvfp4_gemm import swizzle_fp8_scales
import pytest
import torch

from test.test_cute_block_scaled_mma import _native_config
from test.test_cute_block_scaled_mma import _wrapper_plan

import helion
from helion._compiler.cute.block_scaled_config import normalize_block_scaled_config
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def test_n64_is_an_nvfp4_tile_choice() -> None:
    config: dict[str, object] = {"cute_scaled_mma": True, "cute_scaled_tile_n": 64}
    normalize_block_scaled_config(
        config, available=True, capability=(10, 0), fix_invalid=False
    )
    assert config["cute_scaled_tile_n"] == 64


@pytest.mark.parametrize("cluster,persistent", [(1, False), (1, True), (2, False)])
def test_cuda_n64_observes_both_weight_scale_halves(
    cluster: int, persistent: bool
) -> None:
    """The weight scales differ between the two 64-row halves of every
    128-row scale chunk, so an N64 tile reading the wrong half is caught."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native NVFP4 required")
    m, k, n = 200, 512, 320
    torch.manual_seed(0)
    a_packed = pack_fp4_last_dim(
        quantize_fp4_e2m1(torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE))
    ).view(torch.float4_e2m1fn_x2)
    w_packed = pack_fp4(
        quantize_fp4_e2m1(torch.randn(k, n, dtype=torch.bfloat16, device=DEVICE))
    ).view(torch.float4_e2m1fn_x2)
    groups = torch.arange(k // 16, device=DEVICE)
    act_scale = swizzle_fp8_scales(
        (0.5 + ((torch.arange(m, device=DEVICE)[:, None] + groups) % 5) * 0.25).to(
            torch.float8_e4m3fn
        )
    )
    weight_scale = swizzle_fp8_scales(
        (
            0.5
            + ((torch.arange(n, device=DEVICE)[:, None] // 64 + groups * 3) % 4) * 0.5
        ).to(torch.float8_e4m3fn)
    )
    expected = reference_nvfp4_w4a4_matmul(a_packed, w_packed, act_scale, weight_scale)
    config = _native_config(
        cute_scaled_tile_n=64,
        cute_scaled_tile_k=256,
        cute_scaled_cluster_m=cluster,
        cute_scaled_stages=3,
        cute_scaled_persistent=persistent,
    )
    kernel = helion.kernel(
        _nvfp4_w4a4_matmul_kernel.fn, backend="cute", static_shapes=True, config=config
    )
    out = torch.empty(m, n, dtype=torch.bfloat16, device=DEVICE)
    args = (
        a_packed.view(m, k // 16, 8),
        w_packed.view(k // 16, 8, n),
        act_scale.reshape(-1),
        weight_scale.reshape(-1),
        out,
        1.0,
    )
    assert _wrapper_plan(kernel.bind(args).to_triton_code(config))["bn"] == 64
    kernel(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)
