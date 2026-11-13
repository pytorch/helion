from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args


@skipIfNotCUDA()
def test_empty_int64_attention_interval_precedes_subtraction() -> None:
    """Both original and replay intervals are empty; assert the native route.

    The sibling attention oracle covers nonempty intervals. This regression
    specifically proves the extreme ordered-comparison guard before subtract.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the initial paged pipeline uses SM100 tensor memory")
    args = list(paged_attention_args(device=DEVICE))
    for value in args[:7]:
        assert isinstance(value, torch.Tensor)
        value.zero_()
    bounds = torch.tensor([[1, -(2**63)], [2**63 - 1, -1]], device=DEVICE)
    args.append(bounds)
    bound = paged_online_attention.bind(tuple(args))
    config = helion.Config(
        block_sizes=[64],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention",
        cute_min_blocks_per_mp=1,
    )
    with bound.env:
        config = bound.config_spec.normalized_config(config)
    assert config.get("cute_mma_dag_pass_warps") == "attention"
    assert "_helion_paged_attention.paged_attention_body(" in bound.to_code(config)
    compiled = bound.compile_config(config)
    outputs = compiled(*args)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            replay = compiled(*args)
        for changed in (False, True):
            if changed:
                bounds.copy_(torch.tensor([[0, 0], [5, 2]], device=DEVICE))
                outputs = compiled(*args)
            graph.replay()
            for output, lse in (outputs, replay):
                assert torch.count_nonzero(output).item() == 0
                assert torch.isneginf(lse).all().item()
