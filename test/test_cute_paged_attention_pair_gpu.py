from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_optional_lse import check_optional_lse_replay
from helion._testing.cute_optional_lse import initialize_paged_attention_inputs
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_optional_lse import optional_lse_fixture
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_pair import paired_bounds_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,query_len,group,heads,reference,fast_math,return_lse,pdl,sleep_waits,query_tma",
    [
        ("mapped", 1, 8, 2, False, False, True, False, True, True),
        ("mapped", 5, 8, 2, True, True, False, True, True, True),
        ("per_key", 5, 1, 4, False, True, True, False, False, False),
        ("per_key", 17, 1, 4, True, False, True, False, True, True),
        ("mapped", 4, 8, 2, False, True, True, True, False, False),
        ("per_key", 5, 8, 2, True, False, True, False, False, True),
    ],
)
def test_paired_schedule_and_replay(
    form,
    query_len,
    group,
    heads,
    reference,
    fast_math,
    return_lse,
    pdl,
    sleep_waits,
    query_tma,
):
    """Six compiles cover panels, ragged rows, heads, source forms, math,
    both wait policies, scalar query stores and TMA query copies.

    Each replays changed page mappings, empty requests, negative maxima and
    exceptional V/query inputs, preserving independent state for both heads.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    if form == "mapped":
        kernel, args = optional_lse_fixture(
            form, return_lse, DEVICE, query_len=query_len
        )
        args = (*args, reference)
    else:
        kernel = paged_online_attention
        values = list(
            paged_attention_args(rows=query_len * group, heads=heads, device=DEVICE)
        )
        values[7:9] = [query_len, group]
        if query_len == 17:
            values[9] = -1
        args = (*values, None, True, return_lse, reference)
    initialize_paged_attention_inputs(args)
    bound = helion.kernel(
        kernel.fn, backend="cute", static_shapes=True, fast_math=fast_math
    ).bind(args)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * group)],
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=pdl,
        cute_mma_dag_sleep_waits=sleep_waits,
        cute_mma_dag_query_tma=query_tma,
    )
    source = bound.to_code(config)
    assert "paged_attention_paired_body(" in source
    assert "block=(640, 1, 1)" in source
    check_optional_lse_replay(
        bound, config, args, form, return_lse, reference_scores=reference
    )


@skipIfNotCUDA()
def test_paired_empty_int64_endpoints_and_graph():
    """An empty extreme range must not overflow subtraction or doubled trips."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    values = list(paged_attention_args(rows=8, heads=4, device=DEVICE))
    values[7] = 1
    initialize_paged_attention_inputs(tuple(values))
    bounds = torch.tensor(
        [[1, -(2**63)], [2**63 - 1, -1]], dtype=torch.int64, device=DEVICE
    )
    args = (*values, bounds, True)
    bound = paired_bounds_attention("request").bind(args)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[8],
        cute_mma_dag_pass_warps="attention_paired",
        cute_mma_dag_sleep_waits=True,
        cute_mma_dag_query_tma=True,
    )
    assert "paged_attention_paired_body(" in bound.to_code(config)
    compiled = bound.compile_config(config)
    compiled(*args)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            captured = compiled(*args)
        for changed in (False, True):
            if changed:
                bounds.copy_(bounds.new_tensor([[2**31 + 1, -(2**31)], [0, -(2**63)]]))
            for result in (compiled(*args), captured):
                graph.replay()
                assert torch.count_nonzero(result[0]) == 0
                assert result[1] is not None and torch.isneginf(result[1]).all()
