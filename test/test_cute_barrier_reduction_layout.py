from __future__ import annotations

from examples.split_k_barrier import split_k_matmul
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("shape", [(16, 4096, 16), (13, 33, 17)])
@pytest.mark.parametrize("reduction_block", [None, 4096])
@skipIfNotCUDA()
def test_split_k_barrier_reference_thread_layout(
    shape: tuple[int, int, int], reduction_block: int | None
) -> None:
    m, k, n = shape
    a = torch.randn((m, k), device=DEVICE)
    b = torch.randn((k, n), device=DEVICE)
    kernel = helion.kernel(
        split_k_matmul.fn, backend="cute", static_shapes=True, dot_precision="ieee"
    )
    bound = kernel.bind((a, b))
    config = bound.config_spec.autotune_reference_config()
    config.config["block_sizes"] = [1, 1, 8, 1, 1]
    config.config["split_k"] = 64
    config.config["reduction_loops"] = [reduction_block]
    compiled = bound.compile_config(config)
    for _ in range(3):
        a.normal_()
        b.normal_()
        torch.testing.assert_close(compiled(a, b), a @ b, atol=1e-3, rtol=1e-3)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled(a, b)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = compiled(a, b)
    for _ in range(3):
        a.normal_()
        b.normal_()
        graph.replay()
        torch.testing.assert_close(actual, a @ b, atol=1e-3, rtol=1e-3)
