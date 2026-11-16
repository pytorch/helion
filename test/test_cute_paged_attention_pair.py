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
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_pair import paired_bounds_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_paged_base2 import base2_attention_fixture
from helion._testing.cute_paged_reference import reference_attention_fixture

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


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
