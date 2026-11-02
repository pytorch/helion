"""Native masked cluster-tail reductions with aligned, padded input rows."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl


def _fused_residual_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.shape
    output = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    residual_out = torch.empty_like(output)
    for row in hl.tile(rows):
        total = x[row, :].to(torch.float32) + residual[row, :].to(torch.float32)
        variance = torch.mean(total * total, dim=-1)
        inverse = torch.rsqrt(variance + 1e-6)
        output[row, :] = (total * inverse[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
        residual_out[row, :] = total.to(x.dtype)
    return output, residual_out


def _row_maximum(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.shape[0],), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.shape[0]):
        output[row] = torch.amax(x[row, :], dim=-1)
    return output


def _tail_config(vector_width: int) -> helion.Config:
    return helion.Config(
        block_sizes=[1],
        num_threads=[1, 128],
        reduction_loops=[2048],
        cute_vector_widths=[vector_width, 1],
        cute_reduction_reloads=["register_prefetch"],
        cute_cluster_n=4,
    )


@skipIfNotCUDA()
@skipUnlessCuteAvailable("CuTe DSL is required")
@skipIfCudaCapabilityLessThan((9, 0))
@pytest.mark.parametrize("columns", [6144, 7176, 7183])
@pytest.mark.parametrize(
    "dtype,vector_width,integer_dtype,rtol,atol",
    [
        (torch.bfloat16, 8, torch.int16, 1e-2, 1e-2),
        (torch.float16, 8, torch.int16, 2e-3, 2e-3),
        (torch.float32, 4, torch.int32, 1e-5, 1e-5),
    ],
)
def test_register_prefetch_fused_residual_norm_partial_cluster(
    columns: int,
    dtype: torch.dtype,
    vector_width: int,
    integer_dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    # 6144 leaves the fourth CTA inactive and needs a newly created mask.
    # The other sizes leave a partial CTA; 7183 also straddles a vector packet.
    # Padded storage aligns each input row without entering the logical norm.
    rows = 3
    stride = -(-columns // 8) * 8
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    x_storage = torch.full((rows, stride), 1000.0, dtype=dtype, device=DEVICE)
    residual_storage = torch.full_like(x_storage, -300.0)
    x = x_storage[:, :columns]
    residual = residual_storage[:, :columns]
    x.copy_(
        torch.randn((rows, columns), dtype=dtype, device=DEVICE, generator=generator)
    )
    residual.copy_(
        torch.randn((rows, columns), dtype=dtype, device=DEVICE, generator=generator)
    )
    # A large final valid value makes loss of a straddling packet visible even
    # at BF16 precision; the surrounding invalid padding has a distinct value.
    x[:, -1] = 50.0
    residual[:, -1] = 0.25
    weight = torch.rand(columns, dtype=dtype, device=DEVICE, generator=generator) + 0.5
    x_before = x_storage.clone()
    residual_before = residual_storage.clone()
    weight_before = weight.clone()
    total = x.float() + residual.float()
    inverse = torch.rsqrt(total.square().mean(dim=1, keepdim=True) + 1e-6)
    expected = (total * inverse * weight.float()).to(dtype)
    expected_residual = total.to(dtype)

    kernel = helion.kernel(
        _fused_residual_norm, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel.bind((x, residual, weight))
    config = bound.config_spec.normalized_config(_tail_config(vector_width))
    code = bound.to_code(config)
    assert "_cute_grouped_reduce_cluster(" in code
    assert "_fuse_cache_" in code
    assert "_consume_prefetch_" in code
    compiled = bound.compile_config(config)
    for _repeat in range(2):
        actual, stored_residual = compiled(x, residual, weight)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        assert torch.equal(
            stored_residual.view(integer_dtype), expected_residual.view(integer_dtype)
        )
        assert torch.equal(x_storage.view(integer_dtype), x_before.view(integer_dtype))
        assert torch.equal(
            residual_storage.view(integer_dtype), residual_before.view(integer_dtype)
        )
        assert torch.equal(
            weight.view(integer_dtype), weight_before.view(integer_dtype)
        )


@skipIfNotCUDA()
@skipUnlessCuteAvailable("CuTe DSL is required")
@skipIfCudaCapabilityLessThan((9, 0))
@pytest.mark.parametrize("columns", [6144, 7183])
def test_partial_cluster_float32_maximum_uses_negative_neutral(columns: int) -> None:
    stride = -(-columns // 8) * 8
    storage = torch.full((3, stride), 1000.0, dtype=torch.float32, device=DEVICE)
    x = storage[:, :columns]
    generator = torch.Generator(device=DEVICE).manual_seed(43)
    x.copy_(-torch.rand(x.shape, device=DEVICE, generator=generator) - 1.0)
    x[:, -1] = -0.125
    before = storage.clone()
    expected = x.amax(dim=1)
    kernel = helion.kernel(
        _row_maximum, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel.bind((x,))
    config = bound.config_spec.normalized_config(_tail_config(4))
    assert "_cute_grouped_reduce_cluster(" in bound.to_code(config)
    compiled = bound.compile_config(config)
    for _repeat in range(2):
        torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)
        assert torch.equal(storage.view(torch.int32), before.view(torch.int32))
