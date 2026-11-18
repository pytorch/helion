from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.analysis.paged_attention_pair import (
    paired_attention_item_axis,
)
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.paged_attention_transport import (
    paged_attention_schedule_facts,
)
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._compiler.cute.pdl_release import CUTE_PDL_CAPTURE_DEFAULT
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_pair import paired_bounds_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_paged_base2 import base2_attention_fixture
from helion._testing.cute_paged_reference import reference_attention_fixture
from helion._testing.cute_paged_v_lag import without_paired_head
from helion.autotuner.base_search import PopulationBasedSearch

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


def test_promoted_default_is_resolved_as_the_seed_a_search_times():
    # The CuTe backend has a promoted default run the final selection's
    # rounds. The search finds its finalist by transferring the default the
    # way it transfers the compiler seed it is, under its fixed overrides.
    args = (*paged_attention_args(rows=40), None, True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(paged_online_attention, args)
        spec = bound.config_spec
        assert spec.compiler_default_config == spec.compiler_seed_configs[0]
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            for overrides, policy in (
                (None, (True, True, "loader_tail")),
                # The opt-out declines the default's dependent launch, in the
                # seed and in the default alike.
                ({"cute_pdl": False}, (False, False, "auto")),
                # The paired schedule cannot run under this override: its
                # default is not in the search and defends nothing.
                ({"cute_min_blocks_per_mp": 0}, None),
            ):
                search = PopulationBasedSearch.__new__(PopulationBasedSearch)
                search.settings = bound.settings
                search.config_spec = spec
                search.config_gen = spec.create_config_generation(overrides=overrides)
                resolved = search._resolve_defending_default()
                seeds = search.config_gen.seed_flat_config_pairs()
                if policy is None:
                    assert resolved is None
                    assert all(
                        config.get("cute_mma_dag_pass_warps") != "attention_paired"
                        for _flat, config in seeds
                    )
                    continue
                assert resolved == seeds[0][1]
                assert resolved.get("cute_mma_dag_pass_warps") == "attention_paired"
                assert (
                    resolved.get("cute_pdl", False),
                    resolved.get("cute_pdl_capture_overlap", False),
                    resolved.get("cute_pdl_release", "auto"),
                ) == policy


@pytest.mark.parametrize(
    "heads,index,dtype,accepted",
    [
        (2, None, torch.int32, True),
        (4, None, torch.int32, True),
        (1, None, torch.int32, False),
        (3, None, torch.int32, False),
        (4, "request", torch.int64, True),
        (4, "item // 2", torch.int64, True),
        (4, "item", torch.int32, False),
        (4, "head", torch.int64, False),
    ],
)
def test_paired_range_proof(heads, index, dtype, accepted):
    kernel = paged_online_attention if index is None else paired_bounds_attention(index)
    bounds = None if index is None else torch.empty((2 * heads, 2), dtype=dtype)
    args = (*paged_attention_args(rows=40, heads=heads), bounds, True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            semantic = match_paged_attention(program)
            assert semantic is not None
            transport = plan_paged_attention_transport(bound.env, semantic)
            assert transport is not None
            axis = paired_attention_item_axis(bound.env, transport)
            assert (axis is not None) == accepted
            facts = paged_attention_schedule_facts(bound.env, semantic)
            assert facts is not None and facts.paired_items == accepted
            if not accepted:
                config = optional_lse_config()
                config.config.update(
                    block_sizes=[64], cute_mma_dag_pass_warps="attention_paired"
                )
                with pytest.raises(helion.exc.InvalidConfig, match="attention_paired"):
                    bound.config_spec.normalized_config(config)
            if axis is not None:
                assert axis.extent == 2 * heads
                # The grouping belongs to a complete root, not a projected
                # phase of a multi-root launch with sibling work.
                bound.config_spec.cute_mma_dag_phase_index = 0
                assert paired_attention_item_axis(bound.env, transport) is None


@pytest.mark.parametrize("heads,per_key", [(2, True), (1, True), (2, False)])
def test_no_config_default_is_the_paired_family_under_the_pair_proof(heads, per_key):
    args = (*paged_attention_args(rows=40 if per_key else 32, heads=heads), None)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(paged_online_attention, (*args, per_key))
        spec = bound.config_spec
        facts = spec.cute_mma_dag_facts
        assert facts is not None and facts.attention is not None
        proved = heads == 2 and per_key
        assert facts.attention.paired_items is proved
        seeds = spec.compiler_seed_configs
        promoted = spec.compiler_default_config
        assert promoted is not None and promoted == seeds[0]
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            default = spec.default_config()
        source = bound.to_code(default)
        if not proved:
            # One KV head, or a reduced denominator: no paired schedule, and
            # the default stays the first seed of the ordinary ranking.
            assert all(
                seed.get("cute_mma_dag_pass_warps") != "attention_paired"
                for seed in seeds
            )
            assert default.get("cute_mma_dag") == "warp"
            assert "_body(" not in source
            return
        assert without_paired_head(seeds, facts) == seeds[1:]
        # The schedule's own defaults, none of the base config's pins.
        assert default.get("cute_mma_dag_pass_warps") == "attention_paired"
        assert default.get("cute_mma_dag_v_lag") == 2
        assert default.get("cute_mma_dag_query_tma") is True
        assert default.get("cute_mma_dag_early_tmem_retirement") is True
        assert {key: default.get(key) for key in CUTE_PDL_CAPTURE_DEFAULT} == dict(
            CUTE_PDL_CAPTURE_DEFAULT
        )
        # The kernel without a config is the seed's own render: two requests
        # fit one wave, so the loader's tail releases the dependents.
        assert source == bound.to_code(helion.Config.from_dict(dict(promoted.config)))
        for text in (
            "paged_attention_paired_body(",
            "v_lag=2",
            "early_tmem_retirement=True",
            "capture_release_tail=True",
            "tma_load_hints=('evict_first', 'evict_first', 'evict_first', 'evict_first')",
            "query_map=",
        ):
            assert text in source


@pytest.mark.parametrize(
    "heads,rows,reference", [(2, 5, False), (4, 17, False), (2, 40, True)]
)
def test_paired_codegen_and_seed(heads, rows, reference):
    args = (*paged_attention_args(rows=rows, heads=heads), None, True, True, reference)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(paged_online_attention, args)
        config = optional_lse_config()
        config.config.update(
            block_sizes=[1 << (rows - 1).bit_length()],
            cute_mma_dag_pass_warps="attention_paired",
            cute_min_blocks_per_mp=1,
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            normalized = bound.config_spec.normalized_config(config)
            assert normalized.get("cute_mma_dag_pass_warps") == "attention_paired"
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, normalized
            )
        source = bound.to_code(normalized)
    assert "paged_attention_paired_body(" in source
    assert "block=(640, 1, 1)" in source
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert len(calls) == 1
    # Two requests and heads original CTAs are grouped in adjacent pairs.
    grid_expression = ast.Expression(calls[0].args[1])
    assert eval(
        compile(grid_expression, "<paired-grid>", "eval"), {"__builtins__": {}}, {}
    ) == (heads,)
    assert "_helion_cute_preferred_smem_carveout" not in source
    config.config["cute_min_blocks_per_mp"] = 2
    with (
        bound.env,
        bound.host_function,
        pytest.raises(helion.exc.InvalidConfig, match="min_blocks_per_mp"),
    ):
        bound.config_spec.normalized_config(config)


@pytest.mark.parametrize("base2,reverse", [(False, True), (True, False), (True, True)])
def test_paired_explicit_semantic_forms(base2, reverse):
    fixture = base2_attention_fixture if base2 else reference_attention_fixture
    kernel, args = fixture("per_key", 5, True)
    args = (*args, reverse)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(cast("Kernel[object]", kernel), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            semantic = match_paged_attention(program)
            assert semantic is not None
            assert (semantic.coordinates.reflected is not None) == reverse
            assert semantic.score_base2 == base2
            transport = plan_paged_attention_transport(bound.env, semantic)
            assert transport is not None
            assert paired_attention_item_axis(bound.env, transport) is not None
            facts = paged_attention_schedule_facts(bound.env, semantic)
            assert facts is not None and facts.paired_items
        config = optional_lse_config()
        config.config.update(
            block_sizes=[64], cute_mma_dag_pass_warps="attention_paired"
        )
        source = bound.to_code(config)
        assert "paged_attention_paired_body(" in source
        assert ("base2_scores=True" in source) == base2
        assert ("reflected_dtype=cutlass.Int32" in source) == reverse
