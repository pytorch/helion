from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _matmul(
    x: torch.Tensor,
    y: torch.Tensor,
    transpose_lhs: hl.constexpr,
    transpose_rhs: hl.constexpr,
) -> torch.Tensor:
    m = x.size(1) if transpose_lhs else x.size(0)
    k = x.size(0) if transpose_lhs else x.size(1)
    n = y.size(0) if transpose_rhs else y.size(1)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            if transpose_lhs:
                lhs = x[reduction, row].T
            else:
                lhs = x[row, reduction]
            if transpose_rhs:
                rhs = y[column, reduction].T
            else:
                rhs = y[reduction, column]
            acc = torch.addmm(acc, lhs, rhs)
        out[row, column] = acc.to(x.dtype)
    return out


@pytest.mark.parametrize("swap_axes", [False, True])
@pytest.mark.parametrize("mma_impl", ["universal", "warp"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "transpose_lhs,transpose_rhs", [(False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("shape", [(64, 16, 64), (65, 19, 70)])
@skipIfNotCUDA()
def test_scalar_mma_producers_preserve_operand_order(
    swap_axes: bool,
    mma_impl: str,
    dtype: torch.dtype,
    transpose_lhs: bool,
    transpose_rhs: bool,
    shape: tuple[int, int, int],
) -> None:
    if mma_impl == "warp" and not get_cute_mma_support().warp_f16bf16:
        pytest.skip("warp MMA is unavailable")
    m, n, k = shape
    x = torch.empty((k, m) if transpose_lhs else (m, k), device=DEVICE, dtype=dtype)
    y = torch.empty((n, k) if transpose_rhs else (k, n), device=DEVICE, dtype=dtype)
    inputs = (x, y, transpose_lhs, transpose_rhs)
    with patch.dict("os.environ", {"HELION_CUTE_MMA_IMPL": mma_impl}):
        kernel = helion.kernel(
            _matmul, backend="cute", static_shapes=True, autotune_effort="none"
        )
        bound = kernel.bind(inputs)
        block_sizes = [32, 16, 32] if mma_impl == "universal" else [64, 8, 16]
        config = helion.Config(
            block_sizes=block_sizes,
            num_threads=[
                block_sizes[0] // 2 if swap_axes else block_sizes[0],
                block_sizes[1],
                block_sizes[2],
            ],
            pid_type="persistent_blocked",
            loop_orders=[[1, 0]] if swap_axes else [[0, 1]],
        )
        code = bound.to_code(config)
        expected_op = "MmaUniversalOp" if mma_impl == "universal" else "MmaF16BF16Op"
        assert expected_op in code
        compiled = bound.compile_config(config)
    for _ in range(3):
        x.random_(-3, 4)
        y.random_(-3, 4)
        lhs = x.T if transpose_lhs else x
        rhs = y.T if transpose_rhs else y
        torch.testing.assert_close(compiled(*inputs), lhs @ rhs, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled(*inputs)
    x.random_(-3, 4)
    y.random_(-3, 4)
    graph.replay()
    torch.testing.assert_close(actual, lhs @ rhs, atol=0, rtol=0)


@skipIfNotCUDA()
def test_warp_mma_rejects_missing_physical_warps() -> None:
    if not get_cute_mma_support().warp_f16bf16:
        pytest.skip("warp MMA is unavailable")
    x = torch.empty((64, 64), device=DEVICE, dtype=torch.float16)
    y = torch.empty((8, 64), device=DEVICE, dtype=torch.float16)
    with patch.dict("os.environ", {"HELION_CUTE_MMA_IMPL": "warp"}):
        kernel = helion.kernel(
            _matmul, backend="cute", static_shapes=True, autotune_effort="none"
        )
        bound = kernel.bind((x, y, False, True))
        config = helion.Config(
            block_sizes=[64, 8, 16],
            num_threads=[1, 8, 16],
            pid_type="persistent_blocked",
        )
        with pytest.raises(exc.BackendUnsupported, match="enough physical threads"):
            bound.to_code(config)
