from __future__ import annotations

import itertools

import pytest
import torch

from test.test_cute_grouped_gemm_split_sizes import _selected_config
from test.test_cute_shared_rhs_grouped import _offset_mm

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("column_major", [False, True])
def test_shared_rhs_device_offsets_and_graph_replay(
    dtype: torch.dtype, offset_dtype: torch.dtype, column_major: bool
) -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100-family TCGEN05 required")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(731)
    a = torch.randn((313, 128), device=device, dtype=dtype, generator=generator)
    b = torch.randn(
        (352, 128) if column_major else (128, 352),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if column_major:
        b = b.T
    offsets = torch.tensor([0, 0, 1, 129, 313], device=device, dtype=offset_dtype)
    args = (a, b, offsets)
    kernel = helion.kernel(
        _offset_mm,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_segmented_matmul_tiling=True,
    )
    bound = kernel.bind(args)
    config = _selected_config(64)
    config.config["tcgen05_ab_stages"] = 6
    source = bound.to_code(config)
    assert "'shared_rhs': True" in source
    assert "cute.gemm(" in source
    bound.set_config(config)

    def expected(routing: list[int]) -> torch.Tensor:
        result = torch.zeros((313, 352), device=device, dtype=dtype)
        product = (a.float() @ b.float()).to(dtype)
        for start, end in itertools.pairwise(routing):
            first, last = max(0, start), min(313, end)
            if last > first:
                result[first:last] = product[first:last]
        return result

    torch.testing.assert_close(
        bound(*args), expected([0, 0, 1, 129, 313]), rtol=3e-2, atol=3e-2
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    for routing in (
        [5, 171, 171, 9, 287],  # Empty, overlapping, and uncovered intervals.
        [-50, 363, 380, 380, 400],  # Negative prefix and an extent larger than M.
        [7, 7, 7, 7, 7],  # No work remains, including the final epilogue.
    ):
        a.normal_(generator=generator)
        b.normal_(generator=generator)
        offsets.copy_(offsets.new_tensor(routing))
        graph.replay()
        torch.testing.assert_close(captured, expected(routing), rtol=3e-2, atol=3e-2)
