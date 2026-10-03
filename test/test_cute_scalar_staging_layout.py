from __future__ import annotations

import pytest
import torch

from test.test_cute_lowerings import _make_tcgen05_persistent_config
from test.test_cute_scalar_staging_sync import _native_matmul

from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("transpose_b", (False, True))
@pytest.mark.parametrize("block_k", (64, 128))
def test_scalar_staging_covers_native_partition_remainders(
    dtype: torch.dtype, transpose_b: bool, block_k: int
) -> None:
    if not torch.cuda.is_available() or not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires native tcgen05 MMA")
    # Unaligned row strides force scalar staging. Each native atom covers only
    # part of K, and the final M/N/K tiles all require masks.
    m, n, k = 97, 17, 530
    a = torch.randn(m, k + 1, dtype=dtype, device=CUDA_DEVICE)[:, :k]
    b = (
        torch.randn(n, k, dtype=dtype, device=CUDA_DEVICE).T
        if transpose_b
        else torch.randn(k, n, dtype=dtype, device=CUDA_DEVICE)
    )
    config = _make_tcgen05_persistent_config(
        block_sizes=[64, 32, block_k],
        pid_type="flat",
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=1,
    )
    bound = _native_matmul._bind_isolated((a, b))
    bound.set_config(config)
    source = bound.to_code(config)
    assert "make_trivial_tiled_mma(" in source
    assert "'kind': 'tcgen05_ab_tma'" not in source
    for _repeat in range(3):
        a.normal_()
        b.normal_()
        actual = bound(a, b)
        torch.testing.assert_close(
            actual, (a.float() @ b.float()).to(dtype), rtol=2e-2, atol=3e-2
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(a, b)
    a.normal_()
    b.normal_()
    captured.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(
        captured, (a.float() @ b.float()).to(dtype), rtol=2e-2, atol=3e-2
    )
