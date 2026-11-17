from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_fragment_index_domains import dot_then_reduce
from helion._testing.cute_fragment_index_domains import indexed_gemm
from helion._testing.cute_fragment_index_domains import reshaped_dot_then_max


@skipIfNotCUDA()
@pytest.mark.parametrize("rows", [17, 32])
def test_fragment_index_recipe_preserves_ragged_tile_domain(rows: int) -> None:
    # The host storage deliberately extends past the logical rows. Address
    # capacity alone must not authorize tail lanes to overwrite that region.
    values = torch.ones((2 * rows + 32) * 16, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.eye(16, device=DEVICE, dtype=torch.bfloat16)
    output = torch.full_like(values, -17)
    args = (values, weights, output, rows)
    bound = indexed_gemm.bind(args)
    config = helion.Config(block_sizes=[32])
    generated = bound.to_code(config)
    assert "fragment_tensor_index" in generated
    compiled = bound.compile_config(config)
    for changed in (False, True):
        if changed:
            values.fill_(3)
            output.fill_(-17)
        compiled(*args)
        torch.testing.assert_close(output[: 2 * rows * 16], values[: 2 * rows * 16])
        assert torch.all(output[2 * rows * 16 :] == -17).item()


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "kind,axis,rows,columns,half_mma",
    [
        ("max", 0, 7, 16, True),
        ("min", 1, 8, 9, True),
        ("prod", 0, 7, 9, True),
        ("max", 0, 7, 16, False),
    ],
)
def test_dot_reduction_uses_neutral_padding(
    kind, axis, rows, columns, half_mma
) -> None:
    # Four compilations cover each nonzero neutral value, both output axes,
    # mixed regular/ragged shapes and the serial dot path. Replay changes the result.
    x = torch.ones((2, rows, 8), device=DEVICE, dtype=torch.bfloat16)
    y = torch.full(
        (2, 8, columns),
        -0.125 if kind == "max" else 0.125,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    bound = dot_then_reduce.bind((x, y, kind, axis))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config())
        | {
            "pid_type": "flat",
            "cute_fragment_half_mma": half_mma,
        }
    )
    compiled = bound.compile_config(config)
    reduction = {"max": torch.amax, "min": torch.amin, "prod": torch.prod}[kind]
    for changed in (False, True):
        if changed:
            x[:, 0, :].mul_(-0.5)
        values = x.float() @ y.float()
        expected = values + reduction(values, dim=axis + 1, keepdim=True)
        torch.testing.assert_close(compiled(x, y, kind, axis), expected, rtol=0, atol=0)


@skipIfNotCUDA()
@pytest.mark.parametrize("half_mma", [False, True])
def test_reshaped_dot_reduction_uses_the_flat_logical_bound(half_mma):
    x = torch.ones((2, 56), device=DEVICE, dtype=torch.bfloat16)
    y = torch.full((8, 16), -0.125, device=DEVICE, dtype=torch.bfloat16)
    bound = reshaped_dot_then_max.bind((x, y))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config())
        | {"pid_type": "flat", "cute_fragment_half_mma": half_mma}
    )
    compiled = bound.compile_config(config)
    for changed in (False, True):
        if changed:
            x[:, :8].mul_(-0.5)
        value = x.float().reshape(2, 7, 8) @ y.float()
        expected = value + value.amax(dim=1, keepdim=True)
        torch.testing.assert_close(compiled(x, y), expected, rtol=0, atol=0)
