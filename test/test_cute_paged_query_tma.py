from __future__ import annotations

import dataclasses

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.paged_attention_transport import paged_query_tma
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_optional_lse import optional_lse_fixture

pytestmark = pytest.mark.cpu_only
QUERY = "cute_mma_dag_query_tma"


def _bound(form, query_len, *, group=8, head_stride=128):
    kernel, initial = optional_lse_fixture(form, True, query_len=query_len)
    args = list(initial)
    if form == "mapped":
        q = args[0]
        assert isinstance(q, torch.Tensor)
        args[0] = torch.empty_strided(
            (q.size(0), query_len, group * 2, 128),
            (query_len * 4608 + 256, 4608, head_stride, 1),
            dtype=q.dtype,
        )
        args[10] = args[10][:, :, : group * 2, :]
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, tuple(args))
    assert bound.host_function is not None
    with cpu_render_target(), bound.env, bound.host_function:
        program = match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        semantic = match_paged_attention(program)
        assert semantic is not None
        transport = plan_paged_attention_transport(bound.env, semantic)
    assert transport is not None
    return bound, transport, args[0]


@pytest.mark.parametrize(
    "form,query_len", [("per_key", 1), ("mapped", 4), ("mapped", 5)]
)
def test_query_descriptor_addresses_match_original_view(form, query_len):
    bound, transport, query = _bound(form, query_len)
    facts = bound.config_spec.cute_mma_dag_facts.attention
    assert facts.query_tma_available
    descriptor = paged_query_tma(transport, facts.query_columns)
    assert descriptor is not None
    assert descriptor.box[0] == 64
    assert descriptor.box[1] * descriptor.box[2] == facts.query_columns
    # Compare every logical element to the original tensor view, independently
    # of the compiler's separable access map and its synthetic descriptor axes.
    for request in range(2):
        for head in range(2):
            for row in range(query_len * 8):
                for dim in range(128):
                    if form == "mapped":
                        original = (request, row // 8, head * 8 + row % 8, dim)
                        mapped = (dim, row % 8, row // 8, head, request)
                    else:
                        original = (request * 2 + head, row, dim)
                        mapped = (dim, row, 0, head, request)
                    assert sum(
                        i * s for i, s in zip(original, query.stride(), strict=True)
                    ) == sum(
                        i * s for i, s in zip(mapped, descriptor.strides, strict=True)
                    )


@pytest.mark.parametrize("group,head_stride", [(3, 128), (8, 129)])
def test_query_tma_unavailable_preserves_scalar_paired_schedule(group, head_stride):
    bound, transport, _ = _bound("mapped", 5, group=group, head_stride=head_stride)
    facts = bound.config_spec.cute_mma_dag_facts.attention
    assert facts.paired_items and not facts.query_tma_available
    assert paged_query_tma(transport, facts.query_columns) is None
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(5 * group)],
        cute_mma_dag_pass_warps="attention_paired",
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "paged_attention_paired_body(" in source
    assert "query_map=" not in source
    with bound.env, bound.host_function:
        assert QUERY not in bound.config_spec._flat_fields()
        baseline = bound.config_spec.normalized_config(config)
        config.config[QUERY] = False
        assert bound.config_spec.normalized_config(config) == baseline
        config.config[QUERY] = True
        with pytest.raises(helion.exc.InvalidConfig, match=QUERY):
            bound.config_spec.normalized_config(config)


def test_query_descriptor_declines_unrepresentable_extents_and_strides():
    _, transport, _ = _bound("mapped", 5)
    assert transport.direct_io is not None
    # The original direct-I/O proof permits these symbolic positive strides;
    # the hardware descriptor has narrower byte-stride and dimension fields.
    for stride in (0, -8, 2**39):
        direct = dataclasses.replace(transport.direct_io, query=(stride, 4608, 128, 1))
        assert (
            paged_query_tma(dataclasses.replace(transport, direct_io=direct), 40)
            is None
        )
    assert paged_query_tma(dataclasses.replace(transport, items=2**33), 40) is None
    assert paged_query_tma(transport, 32) is None
