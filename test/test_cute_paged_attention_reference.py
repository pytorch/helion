from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._testing import cpu_render_target
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_reference import reference_attention_bound
from helion.language._tracing_ops import _new_var

pytestmark = pytest.mark.cpu_only


def _node(value: object) -> torch.fx.Node:
    assert isinstance(value, torch.fx.Node)
    return value


@pytest.mark.parametrize("form", ["mapped", "per_key"])
@pytest.mark.parametrize("query_len", [1, 4, 5])
def test_reference_source_selects_its_own_arithmetic(form, query_len):
    bound, program = reference_attention_bound(form, query_len)
    plan = match_paged_attention(program)
    assert plan is not None and plan.score_reference is not None
    assert not plan.score_base2
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps="attention_scoped",
        cute_min_blocks_per_mp=2,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "reference_scores=True" in source
    assert "paged_attention_partial_body(" in source


def test_reference_reduced_denominator_is_not_admitted():
    _kernel, program = reference_attention_bound("reduced", 4)
    assert match_paged_attention(program) is None


def test_reference_centering_accepts_commuted_addition():
    _, original = reference_attention_bound("mapped", 5)
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None
    centered = _node(plan.probability.args[0])
    centered.args = (centered.args[1], centered.args[0])
    commuted = match_paged_attention(program)
    assert commuted is not None
    assert commuted.score is plan.score
    assert commuted.score_reference is plan.score_reference
    # Commutativity does not permit a scaled shift or reversed subtraction.
    centered.kwargs = {"alpha": 2}
    assert match_paged_attention(program) is None
    centered.kwargs = {}
    centered.target = torch.ops.aten.sub.Tensor
    assert match_paged_attention(program) is None


# Each mutation removes one distinct dependency of the proved recurrence.
# These reuse one bound graph and perform no additional compiler launches.
@pytest.mark.parametrize(
    "mutation",
    [
        "negative_reference",
        "exclude_zero",
        "finite_cap",
        "reference_one",
        "unshifted_score",
        "unshifted_maximum",
        "wrong_shift",
        "shift_alpha",
        "probability_alpha",
        "maximum_alpha",
        "exp2",
        "wrong_alpha_source",
    ],
)
def test_reference_proof_rejects_changed_math(mutation):
    _, original = reference_attention_bound("mapped", 5)
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None and plan.score_reference is not None
    reference = plan.score_reference
    condition, old_max, zero = reference.args
    lower, upper = (_node(arg) for arg in _node(condition).args)
    centered = _node(plan.probability.args[0])
    shift = _node(_node(centered.args[1]).args[0])
    maximum = plan.maximum.output
    while maximum.target is _new_var:
        maximum = _node(maximum.args[0])
    maximum_add = _node(next(arg for arg in maximum.args if arg is not old_max))
    if mutation == "negative_reference":
        lower.args = (lower.args[0], -1.0)
    elif mutation == "exclude_zero":
        lower.target = torch.ops.aten.gt.Scalar
    elif mutation == "finite_cap":
        upper.args = (upper.args[0], 128.0)
    elif mutation == "reference_one":
        reference.args = (condition, old_max, 1.0)
    elif mutation == "unshifted_score":
        mask, score, empty = plan.score.args
        plan.score.args = (mask, _node(score).args[0], empty)
    elif mutation == "unshifted_maximum":
        maximum.args = tuple(
            maximum_add.args[0] if arg is maximum_add else arg for arg in maximum.args
        )
    elif mutation == "wrong_shift":
        shift.args = (shift.args[1], shift.args[0])
    elif mutation == "shift_alpha":
        shift.kwargs = {"alpha": 2}
    elif mutation == "probability_alpha":
        centered.kwargs = {"alpha": 2}
    elif mutation == "maximum_alpha":
        maximum_add.kwargs = {"alpha": 2}
    elif mutation == "exp2":
        plan.probability.target = torch.ops.aten.exp2.default
    else:
        alpha = next(
            node
            for node in plan.probability.graph.nodes
            if node.target is torch.ops.aten.exp.default
            and node is not plan.probability
        )
        difference = _node(alpha.args[0])
        difference.args = (reference, difference.args[1])
    assert match_paged_attention(program) is None
