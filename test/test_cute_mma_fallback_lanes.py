"""Keep scalar loads and coordinates when the native MMA path declines."""

from __future__ import annotations

from examples.matmul_split_k import matmul_split_k
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


def _config() -> helion.Config:
    return helion.Config(
        block_sizes=[64, 64, 32],
        num_threads=[2, 64, 1],
        split_k=16,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_reduction_keeps_scalar_operand_loads(dtype: torch.dtype) -> None:
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with _mock_cuda_unavailable(), _cpu_target():
        bound = kernel._bind_isolated(
            (
                torch.empty((64, 1024), dtype=dtype),
                torch.empty((1024, 128), dtype=dtype),
            )
        )
        code = bound.to_code(_config())
    # The ordinary native-MMA matcher rejects the extra reduction grid axis.
    # Its fallback still needs both inputs and every output coordinate.
    assert "x.iterator" in code
    assert "y.iterator" in code
    assert "indices_0 =" in code
    assert "indices_1 =" in code
    assert "atomic_add(" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_split_reduction_fallback_tails(dtype: torch.dtype, bias: bool) -> None:
    generator = torch.Generator(device=CUDA_DEVICE).manual_seed(4701)
    x = (
        torch.randn((97, 530), device=CUDA_DEVICE, dtype=dtype, generator=generator)
        * 0.05
    )
    y = (
        torch.randn((530, 71), device=CUDA_DEVICE, dtype=dtype, generator=generator)
        * 0.05
    )
    b = torch.randn((71,), device=CUDA_DEVICE, dtype=dtype, generator=generator) * 0.05
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        config=_config(),
    )
    args = (x, y, lambda acc, tiles: acc + b[tiles[1]]) if bias else (x, y)

    def check() -> None:
        expected = x.float() @ y.float()
        if bias:
            expected += b.float()
        torch.testing.assert_close(
            kernel(*args), expected.to(dtype), atol=2e-3, rtol=1e-2
        )

    check()
    for _ in range(2):
        x.copy_(
            torch.randn(x.shape, device=CUDA_DEVICE, dtype=dtype, generator=generator)
            * 0.05
        )
        y.copy_(
            torch.randn(y.shape, device=CUDA_DEVICE, dtype=dtype, generator=generator)
            * 0.05
        )
        check()
