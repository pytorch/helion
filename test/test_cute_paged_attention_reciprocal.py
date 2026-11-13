from __future__ import annotations

import functools

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_reciprocal import RECIPROCAL_ATTENTION_CASES
from helion._testing.cute_paged_reciprocal import reciprocal_attention_fixture

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(form, query_len, lse, reference, base2, reverse, intrinsic=False):
    kernel, args = reciprocal_attention_fixture(
        form,
        query_len,
        lse,
        reference_scores=reference,
        base2_scores=base2,
        reverse_pages=reverse,
        intrinsic=intrinsic,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
    return bound, program


@pytest.mark.parametrize(
    "form,query_len,reference,base2,reverse,schedule,occupancy,fast_math,lse,pdl",
    RECIPROCAL_ATTENTION_CASES,
)
def test_explicit_reciprocal_output_source(
    form, query_len, reference, base2, reverse, schedule, occupancy, fast_math, lse, pdl
):
    bound, program = _bound(form, query_len, lse, reference, base2, reverse)
    plan = match_paged_attention(program)
    assert plan is not None and plan.output_reciprocal
    assert (plan.score_reference is not None) == reference
    assert plan.score_base2 == base2
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps=schedule,
        cute_min_blocks_per_mp=occupancy,
        cute_pdl=pdl,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "output_reciprocal=True" in source
    assert ("paged_attention_paired_body(" in source) == (
        schedule == "attention_paired"
    )


def _reciprocal_multiply(program):
    reciprocal = next(
        node
        for node in program.stores[0].value.graph.nodes
        if node.target is torch.ops.aten.reciprocal.default
    )
    (multiply,) = reciprocal.users
    assert multiply.target is torch.ops.aten.mul.Tensor
    return reciprocal, multiply


def test_direct_reciprocal_intrinsic_and_commuted_multiplication():
    _kernel, original = _bound("per_key", 4, True, False, False, False, True)
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None and plan.output_reciprocal
    normalized = program.stores[0].value.args[0]
    assert normalized.target is torch.ops.aten.mul.Tensor
    normalized.args = tuple(reversed(normalized.args))
    plan = match_paged_attention(program)
    assert plan is not None and plan.output_reciprocal

    # Torch lowers 1.0 / x to reciprocal(x) * 1.0; also cover the matcher arm
    # for a literal unit multiplier on the left of that row reciprocal.
    _kernel, unit_original = _bound("per_key", 4, True, False, False, False)
    program = copy_paged_attention_program(unit_original)
    reciprocal, multiply = _reciprocal_multiply(program)
    assert multiply.args == (reciprocal, 1.0)
    multiply.args = tuple(reversed(multiply.args))
    plan = match_paged_attention(program)
    assert plan is not None and plan.output_reciprocal


@pytest.mark.parametrize("mutation", ["numerator", "unsafe", "no_reciprocal", "bf16"])
def test_reciprocal_proof_rejects_different_source_math(mutation):
    _kernel, original = _bound("per_key", 4, True, False, False, False)
    program = copy_paged_attention_program(original)
    assert match_paged_attention(program) is not None
    reciprocal, multiply = _reciprocal_multiply(program)
    (safe,) = reciprocal.args
    if mutation == "numerator":
        multiply.args = (reciprocal, 3.0)
    elif mutation == "unsafe":
        reciprocal.args = (safe.args[2],)
    elif mutation == "no_reciprocal":
        multiply.args = (safe, 1.0)
    else:
        with reciprocal.graph.inserting_before(reciprocal):
            narrow = reciprocal.graph.call_function(
                torch.ops.prims.convert_element_type.default, (safe, torch.bfloat16)
            )
        narrow.meta["val"] = safe.meta["val"].to(torch.bfloat16)
        reciprocal.args = (narrow,)
        reciprocal.meta["val"] = reciprocal.meta["val"].to(torch.bfloat16)
    assert match_paged_attention(program) is None
