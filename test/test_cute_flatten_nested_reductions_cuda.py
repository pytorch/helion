from __future__ import annotations

from examples.jagged_layer_norm import jagged_layer_norm_kernel
from examples.jagged_layer_norm import reference_jagged_layer_norm_pytorch
import pytest
import torch

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("columns", [4, 69, 128])
@pytest.mark.parametrize("rows_per_tile", [1, 2])
@pytest.mark.parametrize("unaligned", [False, True])
def test_flattened_reductions_preserve_tails_empty_rows_and_graph_updates(
    columns: int, rows_per_tile: int, unaligned: bool
) -> None:
    lengths = torch.tensor([0, 1, 2, 9, 257, 0, 513, 7, 65], dtype=torch.int64)
    rows = int(lengths.sum())
    shift = int(unaligned)
    storage = torch.randn(rows * columns + shift, device=CUDA_DEVICE)
    values = storage[shift:].view(rows, columns)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]).cuda()
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        cute_flatten_nested_reductions=True,
        static_shapes=False,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((values, offsets, 1e-6))
    assert len(bound.config_spec.block_sizes) == 4
    config = helion.Config(
        block_sizes=[rows_per_tile, 512, 512, 512],
        num_threads=[rows_per_tile, 128, 128, 128],
        cute_vector_widths=[1, 4, 4, 4],
        cute_lane_layouts=["strided"] * 4,
    )
    bound.to_code(config)
    bound.set_config(config)
    args = (values, offsets, 1e-6)
    for _ in range(2):
        actual = bound(*args)
        expected = reference_jagged_layer_norm_pytorch(*args)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    for seed in range(3):
        values.copy_(torch.randn_like(values) * (seed + 1))
        changed_lengths = lengths.roll(seed + 1)
        offsets.copy_(
            torch.cat(
                [torch.zeros(1, dtype=torch.int64), changed_lengths.cumsum(0)]
            ).cuda()
        )
        graph.replay()
        torch.cuda.synchronize()
        expected = reference_jagged_layer_norm_pytorch(*args)
        torch.testing.assert_close(captured, expected, atol=1e-5, rtol=1e-4)
