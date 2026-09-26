"""Full-width rolled reductions must preserve values across multiple sweeps."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import _get_backend
from helion._testing import skipIfNotCUDA
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = pytest.mark.skipif(
    _get_backend() != "cute", reason="CuTe backend coverage"
)


@helion.kernel(backend="cute", static_shapes=False)
def _centered_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.size()
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    means = torch.empty((m,), device=x.device, dtype=torch.float32)
    rstds = torch.empty_like(means)
    for rows in hl.tile(m):
        acc = x[rows, :].float()
        mean = acc.mean(-1)
        centered = acc - mean[:, None]
        variance = (centered * centered).mean(-1)
        rstd = torch.rsqrt(variance + 1e-5)
        out[rows, :] = (
            centered * rstd[:, None] * weight[:].float() + bias[:].float()
        ).to(x.dtype)
        means[rows] = mean
        rstds[rows] = rstd
    return out, means, rstds


@helion.kernel(backend="cute", static_shapes=False)
def _var_mean_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    for rows in hl.tile(m):
        acc = x[rows, :].float()
        variance, mean = torch.var_mean(acc, dim=-1, correction=0, keepdim=True)
        out[rows, :] = (
            (acc - mean) * torch.rsqrt(variance + 1e-5) * weight[:].float()
            + bias[:].float()
        ).to(x.dtype)
    return out


@pytest.mark.parametrize("reload", ["register", "gmem"])
@pytest.mark.parametrize("layout", ["aligned", "tail", "strided", "unaligned_rows"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kernel", [_centered_layer_norm, _var_mean_layer_norm])
@skipIfNotCUDA()
def test_full_chunk_multi_sweep(
    kernel, dtype: torch.dtype, layout: str, reload: str
) -> None:
    width = 256
    rows = 17 if layout == "tail" else 32
    if layout == "strided":
        x = torch.randn((rows, width * 2), device=DEVICE, dtype=dtype)[:, ::2]
    elif layout == "unaligned_rows":
        x = torch.randn((rows, width + 1), device=DEVICE, dtype=dtype)[:, :width]
    else:
        x = torch.randn((rows, width), device=DEVICE, dtype=dtype)
    # A nonzero mean makes stale values in later reduction sweeps conspicuous.
    x.mul_(0.5).sub_(2.3)
    weight = torch.randn(width, device=DEVICE)
    bias = torch.randn(width, device=DEVICE)
    config = helion.Config(
        block_sizes=[16],
        num_threads=[16, 16],
        reduction_loops=[width],
        cute_vector_widths=[16, 1],
        cute_lane_layouts=["strided", "blocked"],
        cute_reduction_reloads=[reload],
    )
    args = (x, weight, bias)
    bound = kernel.bind(args)
    actual = bound.compile_config(config)(*args)
    expected = torch.nn.functional.layer_norm(x.float(), (width,), weight, bias, 1e-5)
    if kernel is _centered_layer_norm:
        output, mean, rstd = actual
        torch.testing.assert_close(mean, x.float().mean(-1))
        torch.testing.assert_close(
            rstd, torch.rsqrt(x.float().var(-1, correction=0) + 1e-5)
        )
    else:
        output = actual
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(
        output, expected.to(dtype), rtol=tolerance, atol=tolerance
    )


@skipIfNotCUDA()
def test_full_chunk_seed_for_dynamic_shapes() -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("Full-width seed targets SM100 and newer")

    @helion.kernel(backend="cute", static_shapes=False)
    def normalize(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        m, n = x.size()
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        for rows in hl.tile(m):
            acc = x[rows, :].float()
            inv_rms = torch.rsqrt((acc * acc).mean(-1) + 1e-5)
            out[rows, :] = (acc * inv_rms[:, None] * weight[:].float()).to(x.dtype)
        return out

    x = torch.randn((32, 256), device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(256, device=DEVICE, dtype=x.dtype)
    bound = normalize.bind((x, weight))
    seeds = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("reduction_loops") == [256]
    ]
    assert len(seeds) == 1
    generation = ConfigGeneration(bound.config_spec)
    config = generation.unflatten(generation.flatten(seeds[0]))
    assert config.config["reduction_loops"] == [256]
    assert config.config["num_threads"] == [16, 16]
    actual = bound.compile_config(config)(x, weight)
    expected = torch.nn.functional.rms_norm(x, (256,), weight, eps=1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
