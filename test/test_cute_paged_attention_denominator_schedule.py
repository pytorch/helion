from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING
from typing import cast

import pytest

import helion
from helion._compiler.cute.cute_mma_dag import AXIS_LEAD
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_config import paged_attention_config_reason
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(rows, per_key=True):
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(
            cast("Kernel[object]", paged_online_attention),
            (*paged_attention_args(rows=rows), None, per_key),
        )


def _config(block, min_blocks):
    return helion.Config(
        block_sizes=[block],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention",
        cute_min_blocks_per_mp=min_blocks,
    )


@pytest.mark.parametrize(
    "rows,columns,min_blocks", [(8, 8, 1), (8, 8, 2), (40, 40, 1), (40, 40, 2)]
)
def test_partial_schedule_preserves_query_extent_and_launch(rows, columns, min_blocks):
    bound = _bound(rows)
    block = 1 << (rows - 1).bit_length()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.query_rows == rows
    assert facts.attention.query_block == block
    assert facts.attention.query_columns == columns
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        config = bound.config_spec.normalized_config(_config(block, min_blocks))
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "_helion_paged_partial.paged_attention_partial_body(" in source
    assert "_helion_cute_preferred_smem_carveout" not in source
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert len(calls) == 1
    launch_block = next(kw.value for kw in calls[0].keywords if kw.arg == "block")
    assert ast.literal_eval(launch_block) == (320, 1, 1)
    # Forty logical rows inside a 64-row root still produce one CTA per item.
    assert ast.literal_eval(calls[0].args[1]) == (4,)


def test_partial_schedule_seeds_supported_topologies_and_occupancies():
    bound = _bound(40)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        seeds = MmaDagSeedPlanner(
            facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
        ).configs([])
        assert [
            (seed.get("cute_mma_dag_pass_warps"), seed.get("cute_min_blocks_per_mp"))
            for seed in seeds
        ] == [
            ("attention", 1),
            ("attention", 2),
            ("attention_scoped", 2),
        ]
        for seed in seeds:
            config = bound.config_spec.normalized_config(seed)
            assert config.get("cute_mma_dag_pass_warps") == seed.get(
                "cute_mma_dag_pass_warps"
            )
            assert config.block_sizes == [64]
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, config
            )


def test_partial_query_padding_is_exact():
    bound = _bound(40)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    sizes = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
    }
    query = facts.axes[facts.attention.query_axis].block_id
    for incorrect in (32, 128):
        sizes[query] = incorrect
        assert paged_attention_config_reason(facts, sizes, _config(incorrect, 1))


def test_large_partial_queries_keep_the_ordinary_route():
    facts = _bound(64).config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is None


def test_reduced_denominator_keeps_its_original_topology():
    bound = _bound(64, False)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert not facts.attention.denominator_per_key
    sizes = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
    }
    assert paged_attention_config_reason(facts, sizes, _config(64, 2))


@pytest.mark.parametrize(
    "rows,columns",
    [(1, 8), (8, 8), (9, 32), (32, 32), (33, 40), (40, 40), (5, 8), (17, 32)],
)
def test_partial_schedule_selects_smallest_proved_query_panel(rows, columns):
    # The six endpoints pin each panel transition; 5 and 17 cover padded
    # extents exercised by the native module within the two smaller panels.
    bound = _bound(rows)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.query_columns == columns
    block = 1 << (rows - 1).bit_length()
    assert facts.attention.query_block == block
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        config = bound.config_spec.normalized_config(_config(block, 1))
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
    if rows in (5, 17):
        with cpu_render_target(fake_device_sm_count=148):
            source = bound.to_code(config)
        assert "_helion_paged_partial.paged_attention_partial_body(" in source
        assert f"query_columns={columns}" in source
