from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_paged_attention_schedule import paged_attention_reference


@skipIfNotCUDA()
@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize("window", [-1, 31])
def test_paged_attention_owner_and_changed_data_replay(fast_math, window):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the initial paged pipeline uses SM100 tensor memory")
    args = list(paged_attention_args(device=DEVICE))
    args[-1] = window
    q, k, v, sk, sv, pages, lengths = args[:7]
    q.copy_(((torch.arange(q.numel(), device=DEVICE) % 19 - 9) / 32).reshape(q.shape))
    k.copy_((torch.arange(k.numel(), device=DEVICE) % 15 - 7).reshape(k.shape))
    v.copy_((torch.arange(v.numel(), device=DEVICE) % 17 - 8).reshape(v.shape))
    sk.copy_((torch.arange(sk.numel(), device=DEVICE) % 4 + 125).reshape(sk.shape))
    sv.copy_((torch.arange(sv.numel(), device=DEVICE) % 4 + 125).reshape(sv.shape))
    pages.copy_(
        (torch.arange(pages.numel(), device=DEVICE) * 7 % k.size(0)).reshape(
            pages.shape
        )
    )
    lengths.copy_(torch.tensor([0, 257], device=DEVICE))
    kernel = helion.kernel(
        paged_online_attention.fn,
        backend="cute",
        static_shapes=True,
        fast_math=fast_math,
    )
    bound = kernel.bind(tuple(args))
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                cute_mma_dag_pass_warps="attention",
                cute_min_blocks_per_mp=1,
            )
        )
    assert config.get("cute_mma_dag_pass_warps") == "attention"
    compiled = bound.compile_config(config)
    compiled(*args)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            captured = compiled(*args)
        for changed in (False, True):
            if changed:
                q.neg_()
                pages.add_(3).remainder_(k.size(0))
                lengths.copy_(torch.tensor([129, 0], device=DEVICE))
            expected = paged_attention_reference(args)
            actual = compiled(*args)
            graph.replay()
            for value, replay, reference in zip(
                actual, captured, expected, strict=True
            ):
                torch.testing.assert_close(value, reference, rtol=0.008, atol=0.004)
                torch.testing.assert_close(replay, reference, rtol=0.008, atol=0.004)
