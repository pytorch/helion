from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_columns(
    x: torch.Tensor,
    tile_size: hl.constexpr,
    reverse: hl.constexpr,
    channel_tile: hl.constexpr,
) -> torch.Tensor:
    batches, length, channels = x.shape
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for batch, channel, row in hl.tile(
        [batches, channels, length], block_size=[1, channel_tile, tile_size]
    ):
        out[batch, row, channel] = hl.cumsum(
            x[batch, row, channel].float(),
            dim=1,
            # pyrefly: ignore [bad-argument-type]
            reverse=reverse,
        )
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "tile_size,length", [(32, 32), (64, 53), (128, 191), (256, 256)]
)
def test_parallel_scan(
    dtype: torch.dtype, reverse: bool, tile_size: int, length: int
) -> None:
    x = torch.empty((2, length, 3), device=DEVICE, dtype=dtype)
    args = (x, tile_size, reverse, 1)
    bound = _scan_columns.bind(args)
    config = bound.env.config_spec.default_config()
    code = bound.to_code(config)
    assert "scan_warp_prefix" in code
    assert "scan_initialized" not in code
    fn = bound.compile_config(config)
    for seed in range(5):
        torch.manual_seed(seed)
        x.normal_()
        original = x.clone()
        result = fn(*args)
        expected = torch.empty_like(result, dtype=torch.float64)
        for start in range(0, length, tile_size):
            part = x[:, start : start + tile_size].double()
            if reverse:
                part = part.flip(1).cumsum(1).flip(1)
            else:
                part = part.cumsum(1)
            expected[:, start : start + tile_size] = part
        torch.testing.assert_close(result.double(), expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(fn(*args), result, atol=0, rtol=0)
        torch.testing.assert_close(x, original, atol=0, rtol=0)


def test_parallel_scan_lane_split_fallback() -> None:
    x = torch.rand((2, 2048, 3), device=DEVICE)
    args = (x, 2048, False, 1)
    bound = _scan_columns.bind(args)
    config = bound.env.config_spec.default_config()
    code = bound.to_code(config)
    assert "for lane_" in code
    assert "scan_warp_prefix" not in code
    result = bound.compile_config(config)(*args)
    torch.testing.assert_close(result, x.cumsum(1), atol=2e-5, rtol=2e-5)


def test_parallel_scan_non_x_axis_fallback() -> None:
    x = torch.randn((2, 111, 3), device=DEVICE)
    args = (x, 128, False, 2)
    bound = _scan_columns.bind(args)
    config = bound.env.config_spec.default_config()
    code = bound.to_code(config)
    assert "scan_warp_prefix" not in code
    result = bound.compile_config(config)(*args)
    torch.testing.assert_close(result, x.cumsum(1), atol=2e-5, rtol=2e-5)


def _offset_sum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left + right + 1.0


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _custom_scan(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
        out[row, channel] = hl.associative_scan(_offset_sum, x[row, channel], dim=0)
    return out


def test_parallel_scan_custom_combine_fallback() -> None:
    x = torch.randn((128, 3), device=DEVICE)
    bound = _custom_scan.bind((x,))
    config = bound.env.config_spec.default_config()
    code = bound.to_code(config)
    assert "scan_warp_prefix" not in code
    result = bound.compile_config(config)(x)
    expected = x.cumsum(0) + torch.arange(128, device=DEVICE)[:, None]
    torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
        out[row, channel] = hl.cumsum(x[row, channel], dim=0)
    return out


def test_parallel_scan_output_alias_fallback() -> None:
    x = torch.randn((128, 3), device=DEVICE)
    output = torch.empty_like(x)
    bound = _scan_into.bind((x, output))
    config = bound.env.config_spec.default_config()
    assert "scan_warp_prefix" in bound.to_code(config)
    result = bound.compile_config(config)(x, output)
    torch.testing.assert_close(result, x.cumsum(0), atol=2e-5, rtol=2e-5)
    # Separate argument names and separately constructed views do not prove
    # disjointness. Both aliases must specialize away from the fast path.
    for aliased in (x, x.view_as(x)):
        alias_bound = _scan_into.bind((x, aliased))
        assert alias_bound is not bound
        assert "scan_warp_prefix" not in alias_bound.to_code(config)
