"""Native correctness of computed FP16/BF16 dots and their FP32 carries."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.cute_half_mma import batched_dot
from helion._testing.cute_half_mma import cast_dot
from helion._testing.cute_half_mma import computed_dot
from helion._testing.cute_half_mma import recurrent_dot

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 32, 32), (24, 19, 11)])
def test_half_mma_matches_fp32_accumulation(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(42)
    m, k, n = shape
    x = torch.randn(2, m, k, device=DEVICE, dtype=dtype) * 0.2
    y = torch.randn(2, k, n, device=DEVICE, dtype=dtype) * 0.2
    acc = torch.randn(2, m, n, device=DEVICE, dtype=torch.float32)
    bound = computed_dot.bind((x, y, acc))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {"cute_fragment_half_mma": True}
    )
    compiled = bound.compile_config(config)
    actual = compiled(x, y, acc)
    expected = torch.bmm(x.float() * 0.5, y.float()) + acc
    expected = expected + expected.amax(dim=1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_half_mma_keeps_a_computed_loop_accumulator() -> None:
    torch.manual_seed(43)
    x = torch.randn(2, 3, 32, 32, device=DEVICE, dtype=torch.bfloat16) * 0.2
    y = torch.randn(2, 3, 32, 8, device=DEVICE, dtype=torch.bfloat16) * 0.2
    bound = recurrent_dot.bind((x, y))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {"cute_fragment_half_mma": True}
    )
    actual = bound.compile_config(config)(x, y)
    expected = torch.zeros((2, 32, 8), device=DEVICE)
    for step in range(3):
        expected = expected * 0.5 + torch.bmm(
            x[:, step].float() * 0.5, y[:, step].float()
        )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_mma_predicates_an_odd_native_contraction(dtype: torch.dtype) -> None:
    torch.manual_seed(44)
    x = torch.randn(2, 19, 19, device=DEVICE, dtype=dtype) * 0.2
    y = torch.randn(2, 19, 11, device=DEVICE, dtype=dtype) * 0.2
    bound = batched_dot.bind((x, y))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {"cute_fragment_half_mma": True}
    )
    actual = bound.compile_config(config)(x, y)
    torch.testing.assert_close(
        actual, torch.bmm(x.float(), y.float()), rtol=1e-4, atol=1e-4
    )


def test_half_mma_partial_in_place_cast_matches_reference() -> None:
    torch.manual_seed(45)
    x = torch.randn(2, 8, 8, device=DEVICE, dtype=torch.bfloat16) * 0.2
    y = torch.randn(2, 8, 8, device=DEVICE, dtype=torch.float16) * 0.2
    bound = cast_dot.bind((x, y))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {"cute_fragment_half_mma": True}
    )
    actual = bound.compile_config(config)(x, y)
    torch.testing.assert_close(
        actual, torch.bmm(x.half().float(), y.float()), rtol=1e-4, atol=1e-4
    )
