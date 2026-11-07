from __future__ import annotations

from typing import cast

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
@pytest.mark.parametrize(
    "rows,query_columns,fast_math,min_blocks,window",
    [
        (5, 8, False, 1, -1),
        (17, 32, True, 1, 31),
        (40, 40, False, 2, 31),
        (8, 8, True, 2, -1),
    ],
)
def test_per_key_denominator_owner_and_replay(
    rows, query_columns, fast_math, min_blocks, window
):
    """Four configurations cover three query panels, math and occupancy.

    Each configuration is compiled once and replays changed metadata/data. The
    separate semantic CPU matrix proves denominator ordering; the oracle here
    retains the full two-dimensional state until its final reduction. Rows 5
    and 17 also cover padding to the 8- and 32-column native panels. The largest
    panel retains the two-CTA launch to cover its tighter register budget.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the paged pipeline uses SM100 tensor memory")
    args = list(paged_attention_args(device=DEVICE, rows=rows))
    args[-1] = window
    args.extend((None, True))
    q, k, v, sk, sv, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
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
    bound = helion.kernel(
        paged_online_attention.fn,
        backend="cute",
        static_shapes=True,
        fast_math=fast_math,
    ).bind(tuple(args))
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[1 << (rows - 1).bit_length()],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                cute_mma_dag_pass_warps="attention",
                cute_min_blocks_per_mp=min_blocks,
            )
        )
    assert config.get("cute_mma_dag_pass_warps") == "attention"
    generated = bound.to_code(config)
    assert "_helion_paged_partial.paged_attention_partial_body(" in generated
    assert f"query_columns={query_columns}" in generated
    assert "_helion_cute_preferred_smem_carveout" not in generated
    compiled = bound.compile_config(config)
    compiled(*args)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            captured = compiled(*args)
        initial_k = k.clone()
        for changed in (0, 1, 2, 3):
            if changed == 1:
                q.neg_()
                pages.add_(3).remainder_(k.size(0))
                lengths.copy_(torch.tensor([129, 0], device=DEVICE))
            if changed == 2:
                # Every visible score is zero. Later tiles therefore retain a
                # finite maximum and exercise the vote's unchanged-max path.
                k.zero_()
            if changed == 3:
                # Restore fractional probabilities as well as exact masked zeros.
                k.copy_(initial_k)
                # P=0 must still multiply decoded nonfinite V. These scales also
                # exercise the decoder's full-domain fallback, not its fast range.
                v[:, 0, :, 0, :32] = 0
                sv[:, 0, :, 0, 0] = 255
                q[0, 0, 0] = float("nan")
            expected = paged_attention_reference(args, denominator_per_key=True)
            actual = compiled(*args)
            graph.replay()
            for value, replay, reference in zip(
                actual, captured, expected, strict=True
            ):
                torch.testing.assert_close(
                    value, reference, rtol=0.002, atol=0.002, equal_nan=True
                )
                torch.testing.assert_close(
                    replay, reference, rtol=0.002, atol=0.002, equal_nan=True
                )
