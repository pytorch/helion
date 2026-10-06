from __future__ import annotations

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

import helion
from helion._compiler.autotuner_heuristics.cute import (
    grouped_row_union_paired_clc_carrier,
)
from helion._compiler.cute.grouped_row_union import STARTUP_PREFILL_KEY
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


@pytest.mark.parametrize("startup", [False, True])
@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial_g31"])
def test_ordinary_paired_profile_cuda(startup: bool, partial: bool) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired tcgen05 profile requires a Blackwell CUDA device")
    m, n, k = (640 if partial else 768), 256, 512
    offsets = [17, 17, 49, 113, 113] + [617] * 27 if partial else [0, m]
    generator = torch.Generator(device=CUDA_DEVICE).manual_seed(73453)
    a = torch.randn(
        (m, k), dtype=torch.bfloat16, device=CUDA_DEVICE, generator=generator
    )
    b = torch.randn(
        (k, n), dtype=torch.bfloat16, device=CUDA_DEVICE, generator=generator
    )
    groups = torch.tensor(offsets, dtype=torch.int32, device=CUDA_DEVICE)
    args = (a, b, groups)
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_segmented_matmul_tiling=True,
    )
    bound = kernel.bind(args)
    with bound.env:
        config = grouped_row_union_paired_clc_carrier(
            bound.env, bound.host_function.device_ir
        )
    assert config is not None
    config.config[STARTUP_PREFILL_KEY] = startup
    bound.set_config(config)
    # The public function allocates output on every invocation. The partial case
    # covers [17,617), with empty groups and uncovered prefix/suffix rows; its
    # final 192-row physical tile contains only 64 allocated rows. Unselected
    # rows remain unspecified, so compare only the original intervals' union.
    expected = a.float() @ b.float()
    for _ in range(2):
        result = bound(*args)
        assert result.shape == (m, n) and result.dtype == torch.bfloat16
        torch.testing.assert_close(
            result[offsets[0] : offsets[-1]].float(),
            expected[offsets[0] : offsets[-1]],
            atol=0.01,
            rtol=0.05,
        )
