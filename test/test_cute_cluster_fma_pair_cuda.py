"""Runtime composition of scaled FP32 FMA and the packed cluster exchange."""

from __future__ import annotations

from functools import partial

import pytest
import torch

from test.test_cute_cluster_online_pair import softmax_two_pass_kernel

import helion
from helion.autotuner.benchmarking import _make_cudagraph_replay

CUDA_DEVICE = "cuda"


def _fill_cluster_rows(x: torch.Tensor, cluster: int, generation: int) -> None:
    columns = x.shape[1]
    segment = columns // cluster
    x.normal_()
    x[1:5].fill_(-torch.inf)
    finite_segment = generation % cluster
    x[1, finite_segment * segment : (finite_segment + 1) * segment] = -1000.0
    x[2, :segment] = -1.0
    x[4, -1] = torch.inf
    x[5, generation % columns] = torch.nan
    x[6, columns - 1 - generation % columns] = torch.nan
    x[7].zero_()
    x[7, ::2] = -0.0


def _check_cluster_rows(actual: torch.Tensor, x: torch.Tensor) -> None:
    expected = x.float().softmax(dim=1).to(x.dtype)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3, equal_nan=True)
    # Empty CTA slices of otherwise finite rows must contribute exact zeros.
    empty = torch.isneginf(x[1:3])
    assert torch.equal(actual[1:3][empty], torch.zeros_like(actual[1:3][empty]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize(
    "dtype,columns,cluster", [(torch.bfloat16, 32768, 2), (torch.float16, 65536, 4)]
)
def test_cluster_fma_frames_fresh_pointers_and_mutated_graphs(
    dtype: torch.dtype, columns: int, cluster: int, fast_math: bool
) -> None:
    torch.manual_seed(20260913)
    kernel = helion.kernel(
        softmax_two_pass_kernel.fn,
        backend="cute",
        static_shapes=True,
        fast_math=fast_math,
        autotune_effort="none",
    )
    config = helion.Config(
        block_sizes=[1, columns],
        num_threads=[0, 256],
        cute_vector_widths=[1, 8],
        cute_lane_layouts=["blocked", "strided"],
        cute_cluster_n=cluster,
    )
    retained: list[torch.Tensor] = []
    for replacement in range(2):
        x = torch.empty((8, columns), dtype=dtype, device=CUDA_DEVICE)
        if retained:
            assert x.data_ptr() != retained[0].data_ptr()
        retained.append(x)
        _fill_cluster_rows(x, cluster, replacement)
        bound = kernel.bind((x,))
        source = bound.to_code(config)
        assert source.count("_cute_grouped_reduce_cluster_online_pair(") == 1
        assert "_cute_grouped_reduce_cluster(" not in source
        assert "cute.math.exp2(cute.math.fma(" in source
        assert "_pair_negative_inf_0" in source and "_pair_rescale_0" in source
        assert f"fastmath={fast_math}" in source
        assert source.index("operator.eq(mi,") > source.index("mi = _pair_gmax_0")
        bound.set_config(config)

        invoke = partial(bound, x)
        _check_cluster_rows(invoke(), x)
        replay = _make_cudagraph_replay(invoke)
        output = replay()
        for mutation in range(4):
            _fill_cluster_rows(x, cluster, replacement * 4 + mutation)
            snapshot = x.view(torch.uint8).clone()
            _check_cluster_rows(invoke(), x)
            output.fill_(torch.nan)
            actual = replay()
            _check_cluster_rows(actual, x)
            assert torch.equal(x.view(torch.uint8), snapshot)
