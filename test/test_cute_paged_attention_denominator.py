from __future__ import annotations

import dataclasses
import functools

import pytest
import torch

from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import Carry
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion.language._tracing_ops import _new_var

pytestmark = pytest.mark.cpu_only


@functools.cache
def _program(per_key=True, rows=40):
    args = (*paged_attention_args(rows=rows), None, per_key)
    with cpu_render_target():
        bound = cpu_bind(paged_online_attention, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            result = match_mma_dag(bound.host_function.device_ir.graphs)
    assert result is not None
    return result


@pytest.mark.parametrize(
    "per_key,rows", [(False, 40), (True, 40), (False, 64), (True, 64)]
)
def test_denominator_form_preserves_source_reduction_order(per_key, rows):
    program = _program(per_key, rows)
    plan = match_paged_attention(program)
    assert plan is not None
    assert plan.denominator_per_key is per_key
    assert plan.query.fake.shape[1] == rows
    if per_key:
        assert isinstance(plan.denominator, Carry)
        assert plan.denominator_final.target is torch.ops.aten.sum.dim_IntList
        assert plan.denominator_final.args[1] == [1]
    else:
        assert plan.denominator_final is plan.denominator.phi


# Distinct proof obligations of the matrix recurrence and final row reduction.
@pytest.mark.parametrize(
    "mutation",
    [
        "rounded_probability",
        "wrong_alpha",
        "nonunit_add",
        "wrong_final_axis",
        "wrong_final_fill",
        "nonzero_init",
        "wrong_lse",
        "subtract_probability",
    ],
)
def test_per_key_denominator_rejects_changed_semantics(mutation):
    program = copy_paged_attention_program(_program())
    plan = match_paged_attention(program)
    assert plan is not None and isinstance(plan.denominator, Carry)
    carry = plan.denominator
    value = program.resolve(carry.output)
    if mutation == "rounded_probability":
        value.args = tuple(
            program.dots[1].node.args[0] if arg is plan.probability else arg
            for arg in value.args
        )
    elif mutation == "wrong_alpha":
        rescale = next(arg for arg in value.args if arg is not plan.probability)
        rescale.args = (carry.placeholder, 1.0)
    elif mutation == "nonunit_add":
        value.kwargs = {**value.kwargs, "alpha": 2}
    elif mutation == "wrong_final_axis":
        plan.denominator_final.args = (plan.denominator_final.args[0], [0])
    elif mutation == "wrong_final_fill":
        masked = plan.denominator_final.args[0]
        masked.args = (carry.phi, 1)
    elif mutation == "nonzero_init":
        loop = dataclasses.replace(
            program.loops[0],
            carries=tuple(
                dataclasses.replace(item, init_value=item.placeholder)
                if item is carry
                else item
                for item in program.loops[0].carries
            ),
        )
        program = dataclasses.replace(program, loops=(loop,))
    elif mutation == "wrong_lse":
        assert plan.lse is not None
        logarithm = next(
            node
            for node in plan.lse.node.graph.nodes
            if node.target is torch.ops.aten.log.default
        )
        logarithm.args = (carry.phi,)
    else:
        value.target = torch.ops.aten.sub.Tensor
    assert match_paged_attention(program) is None


@pytest.mark.parametrize(
    "per_key,alias_condition",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_denominator_transparent_aliases(per_key, alias_condition):
    program = copy_paged_attention_program(_program(per_key))
    plan = match_paged_attention(program)
    assert plan is not None
    source = plan.denominator_final
    graph = plan.output.node.graph
    with graph.inserting_after(source):
        alias = graph.call_function(_new_var, (source,))
        alias.meta = dict(source.meta)
    safe = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.aten.where.self
        and program.resolve(node.args[2]) is source
    )
    condition = safe.args[0]
    condition.args = (alias if alias_condition else source, 0)
    safe.args = (condition, safe.args[1], source if alias_condition else alias)
    logarithm = next(
        node for node in graph.nodes if node.target is torch.ops.aten.log.default
    )
    logarithm.args = (alias if alias_condition else source,)
    graph.lint()
    matched = match_paged_attention(program)
    assert matched is not None and matched.denominator_per_key is per_key
    assert matched.denominator_final is source
