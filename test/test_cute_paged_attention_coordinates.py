"""Paged masks preserve scalar tensor arithmetic before index promotion."""

from __future__ import annotations

import dataclasses
import functools

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.paged_attention_transport import (
    paged_attention_schedule_facts,
)
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(dtype, per_key, length_dtype=torch.int32):
    kernel = helion.kernel(
        paged_online_attention.fn,
        backend="cute",
        static_shapes=True,
        index_dtype=dtype,
    )
    args = list(paged_attention_args(rows=40 if per_key else 64))
    args[6] = args[6].to(dtype=length_dtype)
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(kernel, (*args, None, per_key))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("per_key", [False, True])
def test_attention_passes_proved_coordinate_width_to_native_body(dtype, per_key):
    bound = _bound(dtype, per_key)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env:
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
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    owner = "paged_attention_partial_body" if per_key else "paged_attention_body"
    assert owner + "(" in source
    width = "Int64" if dtype == torch.int64 else "Int32"
    assert f"coordinate_dtype=cutlass.{width}" in source


@pytest.mark.parametrize("unproved", ["mixed_row", "wide_literal"])
def test_attention_declines_unimplemented_coordinate_contract(unproved):
    bound = _bound(torch.int64, True)
    with bound.env, bound.host_function:
        program = match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        semantic = match_paged_attention(program)
        assert semantic is not None
        assert paged_attention_schedule_facts(bound.env, semantic) is not None
        if unproved == "mixed_row":
            coords = dataclasses.replace(semantic.coordinates, row_dtype=torch.int32)
        else:
            coords = dataclasses.replace(semantic.coordinates, query_len=2**31)
        changed = dataclasses.replace(semantic, coordinates=coords)
        assert plan_paged_attention_transport(bound.env, changed) is not None
        assert paged_attention_schedule_facts(bound.env, changed) is None


def test_attention_declines_wide_length_tensor():
    bound = _bound(torch.int64, True, torch.int64)
    with bound.env, bound.host_function:
        program = match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        semantic = match_paged_attention(program)
        assert semantic is not None
        assert semantic.coordinates.lengths.dtype == torch.int64
        assert semantic.coordinates.length_base_dtype == torch.int64
        assert plan_paged_attention_transport(bound.env, semantic) is not None
        assert paged_attention_schedule_facts(bound.env, semantic) is None


def test_large_meta_pool_automatically_selects_int64_attention():
    meta_device = torch.device("meta")
    args = list(paged_attention_args(rows=40, device=meta_device))
    pages, pitch = 65536, 33792
    for raw_index, scale_index in ((1, 3), (2, 4)):
        storage = torch.empty(pages * pitch, dtype=torch.int8, device=meta_device)
        args[raw_index] = storage.as_strided(
            (pages, 1, 128, 2, 128), (pitch, 32768, 256, 128, 1)
        )
        args[scale_index] = storage.view(torch.uint8).as_strided(
            (pages, 1, 128, 2, 4), (pitch, 1024, 4, 512, 1), 32768
        )
    args.extend((None, True))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(paged_online_attention, tuple(args))
    assert bound.env.index_dtype == torch.int64
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
