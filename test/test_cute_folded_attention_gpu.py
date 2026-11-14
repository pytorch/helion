from __future__ import annotations

from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_folded_attention import folded_attention_reference
from helion._testing.cute_folded_attention import folded_paged_attention
from helion._testing.cute_optional_lse import initialize_paged_attention_inputs


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "query_len,fast_math,min_blocks,paired_retirement",
    [
        (1, False, 1, False),
        (4, False, 1, False),
        (5, True, 2, False),
        (4, False, 1, True),
    ],
)
def test_folded_attention_layout_and_replay(
    query_len, fast_math, min_blocks, paired_retirement
):
    """Replay ordinary roots and a paired serial folded-output retirement.

    The fourth case covers the seeded non-reference paired body with folded
    output addressing; all cases replay empty/changed/nonfinite inputs.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    args = folded_attention_args(query_len, device=DEVICE)
    q, k, v, _ks, vs, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    out = cast("torch.Tensor", args[-1])
    initialize_paged_attention_inputs(args)
    bound = helion.kernel(
        folded_paged_attention.fn,
        backend="cute",
        static_shapes=True,
        fast_math=fast_math,
    ).bind(args)
    config = helion.Config(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired"
        if paired_retirement
        else "attention",
        cute_min_blocks_per_mp=min_blocks,
        cute_pdl=False,
        cute_mma_dag_early_tmem_retirement=paired_retirement,
    )
    with bound.env:
        config = bound.config_spec.normalized_config(config)
    source = bound.to_code(config)
    assert source.count("@cute.kernel") == 1
    assert "io_layouts=" in source
    assert ("early_tmem_retirement=True" in source) == paired_retirement
    if paired_retirement:
        assert "paged_attention_paired_body(" in source
        assert "maximum_output=" not in source
        assert "probability_schedule=" not in source
    compiled = bound.compile_config(config)
    compiled(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = compiled(*args)
    # All holes and the storage prefix must remain untouched.
    storage = out.as_strided(
        (out.untyped_storage().nbytes() // out.element_size(),), (1,), 0
    )
    written = torch.zeros_like(storage, dtype=torch.bool)
    written.as_strided(out.shape, out.stride(), out.storage_offset()).fill_(True)
    for changed in range(3):
        if changed == 1:
            q.neg_()
            pages.add_(2).remainder_(k.size(0))
            lengths.copy_(torch.tensor([129, 0], device=DEVICE))
        elif changed == 2:
            v[:, :, 0, :32] = 0
            vs[:, :, 0, 0] = 255
            q[0, 0, 0, 0] = float("nan")
        expected = folded_attention_reference(args)
        out.fill_(float("nan"))
        eager = compiled(*args)
        for value, reference in zip(eager, expected, strict=True):
            torch.testing.assert_close(
                value, reference, rtol=0.002, atol=0.002, equal_nan=True
            )
        assert torch.all(storage[~written] == -17).item()
        # Eager and replay share the caller's output. Reset every logical
        # result so neither execution can borrow the other's valid stores.
        for value in replay:
            value.fill_(float("nan"))
        graph.replay()
        for value, reference in zip(replay, expected, strict=True):
            torch.testing.assert_close(
                value, reference, rtol=0.002, atol=0.002, equal_nan=True
            )
        assert torch.all(storage[~written] == -17).item()
