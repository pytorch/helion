from __future__ import annotations

import ast
import dataclasses
import functools
import math

import pytest
import torch

import helion
from helion import exc
from helion._compiler.compile_environment import ConfigValueExpression
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_reference import reference_attention_bound
from helion._testing.cute_paged_rowwise_lazy import paged_query_first_direct_attention
from helion._testing.cute_paged_rowwise_native import ROW_CASES
from helion._testing.cute_paged_rowwise_native import configurable_rowwise_kernel
from helion._testing.cute_paged_rowwise_native import configure_rowwise_case
from helion._testing.cute_paged_rowwise_native import constexpr_rowwise_kernel
from helion._testing.cute_paged_rowwise_native import rowwise_config
from helion._testing.cute_paged_rowwise_native import rowwise_reference
from helion.language._tracing_ops import _new_var

pytestmark = pytest.mark.cpu_only


def _bound(query_len, *, form="literal", threshold=2.0, window=31):
    # Normalize defaults before entering the cache, including _bound(4) callers.
    return _cached_bound(
        query_len, form, threshold if form == "constexpr" else None, window
    )


@functools.cache
def _cached_bound(query_len, form, threshold, window):
    args = list(folded_attention_args(query_len))
    args[7] = window
    if form == "literal":
        kernel = paged_query_first_direct_attention
    elif form == "constexpr":
        kernel = constexpr_rowwise_kernel()
        args.append(threshold)
    else:
        assert form == "config"
        kernel = configurable_rowwise_kernel()
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, tuple(args))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return bound, program


def _render(query_len, form, threshold, *, window=31):
    return _cached_render(query_len, form, threshold, window)


@functools.cache
def _cached_render(query_len, form, threshold, window):
    bound, _ = _bound(query_len, form=form, threshold=threshold, window=window)
    config = rowwise_config(query_len)
    if form == "config":
        config.config["rescale_threshold"] = int(threshold)
    with cpu_render_target(fake_device_sm_count=148):
        return bound.to_code(config)


def _device(source):
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )


@pytest.mark.parametrize(
    "query_len,schedule",
    [(4, "attention_paired"), (5, "attention_paired"), (5, "attention_scoped")],
)
def test_authored_rowwise_recurrence_selects_its_own_constant(query_len, schedule):
    bound, program = _bound(query_len)
    plan = match_paged_attention(program)
    assert plan is not None and plan.rowwise_rescale is not None
    assert plan.rowwise_rescale.threshold == 2.0
    assert plan.score_base2 and plan.score_reference is not None
    config = rowwise_config(query_len, scoped=schedule == "attention_scoped")
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "rowwise_rescale_threshold=2.0" in source
    assert "reference_scores=True" in source and "base2_scores=True" in source
    expected = (
        "paged_attention_paired_body"
        if schedule == "attention_paired"
        else "paged_attention_partial_body"
    )
    assert expected + "(" in source


def _node(value):
    assert isinstance(value, torch.fx.Node)
    return value


def _copy(value):
    while value.target is _new_var:
        value = _node(value.args[0])
    return value


# Every mutation removes a distinct proof obligation from the same copied graph.
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_nan",
        "nan_old",
        "missing_invalid",
        "wrong_invalid",
        "extra_term",
        "wrong_row_max",
        "non_strict",
        "negative",
        "infinite",
        "nan_threshold",
        "huge_integer",
        "boolean_threshold",
        "inexact_constant",
        "false_proposed",
        "true_old",
        "wrong_proposed",
        "wrong_alpha",
    ],
)
def test_rowwise_proof_rejects_changed_predicate_or_recurrence(mutation):
    _, original = _bound(4)
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None and plan.rowwise_rescale is not None
    maximum = _copy(plan.maximum.output)
    predicate = _node(maximum.args[0])
    nested, nan = map(_node, predicate.args)
    greater, invalid = map(_node, nested.args)
    proposed = _node(maximum.args[1])
    old = plan.maximum.placeholder
    if mutation == "missing_nan":
        maximum.args = (nested, proposed, old)
    elif mutation == "nan_old":
        nan.args = (old,)
    elif mutation == "missing_invalid":
        nested.args = (greater, greater)
    elif mutation == "wrong_invalid":
        lower = _node(_node(invalid.args[0]).args[0])
        lower.args = (old, -1.0)
    elif mutation == "extra_term":
        predicate.args = (nested, nested)
    elif mutation == "wrong_row_max":
        greater.args = (old, 2.0)
    elif mutation == "non_strict":
        greater.target = torch.ops.aten.ge.Scalar
    elif mutation in (
        "negative",
        "infinite",
        "nan_threshold",
        "huge_integer",
        "boolean_threshold",
        "inexact_constant",
    ):
        greater.args = (
            greater.args[0],
            {
                "negative": -1.0,
                "infinite": math.inf,
                "nan_threshold": math.nan,
                "huge_integer": 10**1000,
                "boolean_threshold": True,
                "inexact_constant": 0.1,
            }[mutation],
        )
    elif mutation == "false_proposed":
        maximum.args = (predicate, proposed, proposed)
    elif mutation == "true_old":
        maximum.args = (predicate, old, old)
    elif mutation == "wrong_proposed":
        proposed.target = torch.ops.aten.minimum.default
    else:
        alpha = next(
            n
            for n in plan.probability.graph.nodes
            if n.target is torch.ops.aten.exp2.default and n is not plan.probability
        )
        difference = _node(alpha.args[0])
        difference.args = (plan.score_reference, difference.args[1])
    assert match_paged_attention(program) is None


def test_ordinary_reference_has_no_rowwise_constant():
    bound, program = reference_attention_bound("mapped", 4)
    plan = match_paged_attention(program)
    assert plan is not None and plan.rowwise_rescale is None
    config = optional_lse_config()
    config.config.update(cute_mma_dag_pass_warps="attention_paired")
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "rowwise_rescale_threshold" not in source


def test_native_case_oracle_shapes_and_nonfinite_classes():
    args = list(folded_attention_args(5))
    args[7] = 2047
    args = tuple(args)
    for case in ROW_CASES:
        configure_rowwise_case(args, case)
        out, lse = rowwise_reference(args)
        assert out.shape == (2, 5, 16, 128) and lse.shape == (2, 5, 16)
        if case in ("original", "restored"):
            assert (out[0] == 0).all() and torch.isneginf(lse[0]).all()
        elif case in ("nonfinite", "zero_v_nan_q"):
            assert torch.isnan(out[0, 0, 0]).all() and torch.isnan(lse[0, 0, 0])
        else:
            assert torch.isfinite(out).all()


def test_constexpr_threshold_two_preserves_literal_source_and_sixteen_is_distinct():
    literal = _render(4, "literal", 2.0)
    sources = {
        threshold: _render(4, "constexpr", threshold) for threshold in (2.0, 16.0)
    }
    for threshold in (2.0, 16.0):
        bound, program = _bound(4, form="constexpr", threshold=threshold)
        assert bound.env.settings.fast_math
        plan = match_paged_attention(program)
        assert plan is not None and plan.rowwise_rescale is not None
        assert plan.rowwise_rescale.threshold == threshold
        assert f"rowwise_rescale_threshold={threshold!r}" in sources[threshold]
    assert ast.dump(_device(literal)) == ast.dump(_device(sources[2.0]))
    assert ast.dump(_device(sources[16.0])) != ast.dump(_device(sources[2.0]))
    changed = sources[16.0].replace(
        "rowwise_rescale_threshold=16.0", "rowwise_rescale_threshold=2.0"
    )
    assert ast.dump(_device(changed)) == ast.dump(_device(sources[2.0]))


@pytest.mark.parametrize("query_len", [4, 5])
def test_exact_native_configuration_and_mixed_later_tile(query_len):
    args = list(folded_attention_args(query_len))
    args[7] = 2047
    args = tuple(args)
    configure_rowwise_case(args, "mixed_rows")
    q, k, _, ks, _, pages, lengths = args[:7]
    group = q.size(2) // k.size(2)
    query = q[0, -1, :group].float()
    length, width = int(lengths[0]), k.size(1)
    last_tile = (length - 1) // width

    def scores(tile):
        page = int(pages[0, tile])
        key = (
            k[page, :, 0].float()
            * torch.exp2(ks[page, :, 0].float() - 127).repeat_interleave(32, -1)
        ).to(q.dtype)
        return (
            query
            @ (key * 2.0**-6).float().T
            * (q.size(-1) ** -0.5 * math.log2(math.e) * 2.0**6)
        )

    # The final query sees one key on the first reversed page, then a full page.
    # Check its eight rows directly, without intercepting unrelated torch.where.
    assert last_tile > 0 and args[7] >= 2 * width
    maximum = scores(last_tile)[:, : length - last_tile * width].amax(1)
    tile_maximum = (scores(last_tile - 1) - maximum[:, None]).amax(1)
    valid = (maximum >= 0) & torch.isfinite(maximum)
    needs_rescale = (tile_maximum > 2.0) | ~valid | torch.isnan(tile_maximum)
    assert (needs_rescale & valid).any() and (~needs_rescale & valid).any(), (
        "Need a later tile with both source choices among valid old rows in one head"
    )
    source = _render(query_len, "literal", 2.0, window=2047)
    assert "rowwise_rescale_threshold=2.0" in source
    assert "value_metadata_scales=" in source and "query_map=" in source


def test_tunable_recognition_retains_config_identity_without_active_environment():
    bound, program = _bound(4, form="config")
    plan = match_paged_attention(program)
    assert plan is not None and plan.rowwise_rescale is not None
    assert plan.rowwise_rescale.threshold == ConfigValueExpression(
        "config", ("rescale_threshold",)
    )
    fragment = bound.config_spec.user_defined_tunables["rescale_threshold"]
    assert fragment.search_values() == [2, 16]
    assert (
        match_paged_attention(dataclasses.replace(program, config_scalars={})) is None
    )


@pytest.mark.parametrize("threshold", [2, 16])
def test_each_selected_config_emits_the_same_body_as_authored_constexpr(threshold):
    source = _render(4, "config", float(threshold))
    literal = _render(4, "constexpr", float(threshold))
    assert f"rowwise_rescale_threshold={float(threshold)!r}" in source
    assert ast.dump(ast.Module(body=_device(source).body, type_ignores=[])) == ast.dump(
        ast.Module(body=_device(literal).body, type_ignores=[])
    )


def test_large_integer_literal_preserves_previous_scalar_conversion():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            constexpr_rowwise_kernel(), (*folded_attention_args(4), 2**60 + 1)
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    plan = match_paged_attention(program)
    assert plan is not None and plan.rowwise_rescale is not None
    assert plan.rowwise_rescale.threshold == float(2**60 + 1)


@pytest.mark.parametrize("threshold", [-1, 2**24 + 1, True, 2.0])
def test_invalid_selected_threshold_is_rejected(threshold):
    _, program = _bound(4, form="config")
    plan = match_paged_attention(program)
    assert plan is not None and plan.rowwise_rescale is not None
    with pytest.raises(exc.InvalidConfig, match="rowwise rescale threshold"):
        plan.rowwise_rescale.threshold_for(helion.Config(rescale_threshold=threshold))
