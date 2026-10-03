from __future__ import annotations

import pytest
import torch

from .test_cute_chained_pipeline import args as _args
from .test_cute_chained_pipeline import leaf_config
from .test_cute_chained_pipeline import pair
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("schedule", ["serial64", "overlap64"])
@pytest.mark.parametrize("columns", [32, 64])
def test_root_seed_panels_paired_tma_gpu(dtype, schedule, columns):
    torch.manual_seed(887)
    args = tuple(
        torch.randn_like(arg, device=DEVICE) * 0.03 for arg in _args(dtype, n=128)
    )
    original_inputs = [arg.clone() for arg in args]
    config = leaf_config(schedule=schedule)
    before = pair._bind_isolated(args)
    before.set_config(config)
    after = pair._bind_isolated(args)
    after.set_config(
        helion.Config.from_dict(
            config.config | {"cute_chained_seed_tile_columns": columns}
        )
    )
    expected = before(*args)
    torch.testing.assert_close(after(*args), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = after(*args)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for actual, original in zip(args, original_inputs, strict=True):
        assert torch.equal(actual, original)
