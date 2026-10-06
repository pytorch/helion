from __future__ import annotations

from examples import nvfp4_gemv
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("packed_input", [False, True])
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
def test_nvfp4_gemv_search_boundaries_write_every_row(packed_input: bool) -> None:
    rows, groups = 64, 64
    weight = torch.randint(256, (rows, groups * 8), dtype=torch.uint8, device=DEVICE)
    scale = nvfp4_gemv.make_fp8_scales((rows, groups), "cuda")
    out = torch.empty(rows, dtype=torch.bfloat16, device=DEVICE)
    weight_arg = weight.view(torch.float4_e2m1fn_x2).view(rows, groups, 8)
    alpha = 1.25
    if packed_input:
        x = torch.randint(256, (groups * 8,), dtype=torch.uint8, device=DEVICE)
        x_scale = nvfp4_gemv.make_fp8_scales((groups,), "cuda")
        args = (
            weight_arg,
            x.view(torch.float4_e2m1fn_x2).view(groups, 8),
            scale.reshape(-1),
            x_scale.reshape(-1),
            out,
            alpha,
        )
        kernel = helion.kernel(
            nvfp4_gemv._nvfp4_gemv_fp4in_body, backend="cute", static_shapes=True
        )
    else:
        x = torch.randn(groups * 16, dtype=torch.bfloat16, device=DEVICE)
        args = (weight_arg, x.view(groups, 16), scale.reshape(-1), out, alpha)
        kernel = helion.kernel(
            nvfp4_gemv._nvfp4_gemv_bf16in_body, backend="cute", static_shapes=True
        )
    bound = kernel.bind(args)
    row_spec = bound.config_spec.block_sizes[0]
    # Exercise both ends of the advertised row search range. Previously its
    # upper bound was 8, but the kernel only stored the first row of each tile.
    for row_tile in {row_spec.min_size, row_spec.max_size}:
        for k_tile in (32, 64):
            compiled = bound.compile_config(
                helion.Config(block_sizes=[row_tile, k_tile])
            )
            for _ in range(3):
                weight.random_(0, 256)
                out.fill_(float("nan"))
                if packed_input:
                    x.random_(0, 256)
                    expected = nvfp4_gemv.reference_nvfp4_gemv_fp4in(
                        weight, x, scale, x_scale, alpha
                    )
                else:
                    x.normal_()
                    expected = nvfp4_gemv.reference_nvfp4_gemv_bf16in(
                        weight, x, scale, alpha
                    )
                torch.testing.assert_close(
                    compiled(*args), expected, atol=0.25, rtol=0.01
                )
