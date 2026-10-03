from __future__ import annotations

from examples.welford import welford
import pytest
import torch
import torch.nn.functional as F

from test.test_cute_sibling_layout_safety import _config

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("kind", "rows", "columns", "dtype"),
    (
        pytest.param(
            "scalar_stats", 16384, 1024, torch.float32, id="scalar-stats-original"
        ),
        pytest.param(
            "scalar_stats", 17, 1024, torch.float32, id="scalar-stats-row-tail"
        ),
        pytest.param(
            "broadcast_count", 16384, 1024, torch.float32, id="broadcast-count-original"
        ),
        pytest.param(
            "broadcast_count", 129, 17, torch.float32, id="broadcast-count-tail"
        ),
        pytest.param("row_tail", 17, 77, torch.float32, id="shared-axis-tail"),
        pytest.param("warp_per_row", 17, 80, torch.float32, id="warp-per-row-tail"),
        pytest.param("warp_per_row", 16, 256, torch.bfloat16, id="warp-per-row-bf16"),
        pytest.param(
            "warp_per_row", 17, 80, torch.bfloat16, id="warp-per-row-bf16-tail"
        ),
    ),
)
def test_sibling_layout_rejected_configs_and_valid_vector_paths(
    kind: str, rows: int, columns: int, dtype: torch.dtype
) -> None:
    kernel = helion.kernel(
        welford.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = (
        torch.empty(columns, dtype=dtype, device=CUDA_DEVICE),
        torch.empty(columns, dtype=dtype, device=CUDA_DEVICE),
        torch.empty((rows, columns), dtype=dtype, device=CUDA_DEVICE),
    )
    bound = kernel.bind(args)
    compiled = bound.compile_config(_config(kind))

    def mutate(values: tuple[torch.Tensor, ...], seed: int) -> None:
        torch.manual_seed(seed)
        for value in values:
            value.uniform_(-0.5, 0.5)

    def check(actual: torch.Tensor, values: tuple[torch.Tensor, ...]) -> None:
        weight, bias, x = values
        expected = F.layer_norm(
            x.float(), (columns,), weight.float(), bias.float(), eps=1e-5
        ).to(dtype)
        # Retain the benchmark's original tolerance. Poisoned graph outputs
        # additionally prove that every logical output is overwritten.
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.1)

    for seed in (901, 902, 903):
        mutate(args, seed)
        actual = compiled(*args)
        torch.cuda.synchronize()
        check(actual, args)
    replacements = tuple(torch.empty_like(value) for value in args)
    mutate(replacements, 904)
    actual = compiled(*replacements)
    torch.cuda.synchronize()
    check(actual, replacements)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    for seed in (905, 906, 907):
        mutate(args, seed)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(captured, args)
