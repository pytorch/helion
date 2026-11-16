from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_base2 import base2_attention_fixture
from helion.language._tracing_ops import _new_var

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(form, return_lse):
    kernel, args = base2_attention_fixture(form, 4, return_lse)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(cast("Kernel[object]", kernel), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return bound, program


@pytest.mark.parametrize(
    "form,lse",
    [
        ("mapped", True),
        ("mapped", False),
        ("per_key", True),
        ("per_key", False),
    ],
)
def test_explicit_base2_source_keeps_natural_lse_and_separate_route(form, lse):
    bound, program = _bound(form, lse)
    plan = match_paged_attention(program)
    assert plan is not None and plan.score_base2
    with cpu_render_target(fake_device_sm_count=148):
        code = bound.to_code(optional_lse_config())
    assert "base2_scores=True" in code
    assert "reference_scores=True" in code


def _lse_terms(program):
    plan = match_paged_attention(program)
    assert plan is not None and plan.lse is not None
    lse = next(store.value for store in program.vec_stores if store.access is plan.lse)
    assert isinstance(lse, torch.fx.Node)
    maximum_term = next(
        arg
        for arg in lse.args
        if isinstance(arg, torch.fx.Node) and arg.target is torch.ops.aten.mul.Tensor
    )
    log_term = next(
        arg
        for arg in lse.args
        if isinstance(arg, torch.fx.Node) and arg.target is torch.ops.aten.log.default
    )
    return plan, lse, maximum_term, log_term


@pytest.mark.parametrize(
    "mutation",
    [
        "probability_exp",
        "alpha_exp",
        "unscaled_lse",
        "wrong_factor",
        "bf16_factor",
        "log2_lse",
        "lse_alpha",
    ],
)
def test_base2_proof_rejects_mixed_units_and_rounded_tensor_constants(mutation):
    _bound_kernel, original = _bound("mapped", True)
    program = copy_paged_attention_program(original)
    plan, lse, maximum_term, log_term = _lse_terms(program)
    if mutation == "probability_exp":
        plan.probability.target = torch.ops.aten.exp.default
    elif mutation == "alpha_exp":
        alpha = next(
            node
            for node in plan.probability.graph.nodes
            if node.target is torch.ops.aten.exp2.default and node.meta["val"].ndim == 1
        )
        alpha.target = torch.ops.aten.exp.default
    elif mutation == "unscaled_lse":
        lse.args = tuple(
            plan.maximum.phi if arg is maximum_term else arg for arg in lse.args
        )
    elif mutation == "wrong_factor":
        maximum_term.args = (plan.maximum.phi, 1.0)
    elif mutation == "bf16_factor":
        with lse.graph.inserting_before(maximum_term):
            constant = lse.graph.call_function(
                torch.ops.aten.scalar_tensor.default,
                (math.log(2),),
                {"dtype": torch.bfloat16},
            )
        constant.meta["val"] = torch.empty((), dtype=torch.bfloat16)
        maximum_term.args = (plan.maximum.phi, constant)
    elif mutation == "log2_lse":
        log_term.target = torch.ops.aten.log2.default
    else:
        lse.kwargs = {"alpha": 2}
    assert match_paged_attention(program) is None


def test_base2_lse_allows_a_named_scaled_maximum():
    _, original = _bound("mapped", True)
    program = copy_paged_attention_program(original)
    _plan, lse, maximum, _log_term = _lse_terms(program)
    with lse.graph.inserting_before(lse):
        named = lse.graph.call_function(_new_var, (maximum,))
    named.meta = dict(maximum.meta)
    lse.args = tuple(named if arg is maximum else arg for arg in lse.args)
    matched = match_paged_attention(program)
    assert matched is not None and matched.score_base2
