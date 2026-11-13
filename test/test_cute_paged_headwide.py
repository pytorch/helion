"""Complete-head ownership and source semantics for the lazy-rescale lowering."""

from __future__ import annotations

import ast
import functools
import inspect
import math

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.cute_mma_dag import mma_dag_facts
from helion._compiler.cute.paged_attention_transport import (
    paged_attention_schedule_facts,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_headwide import configure_headwide_case
from helion._testing.cute_paged_headwide import fixed_root_matmul
from helion._testing.cute_paged_headwide import headwide_config
from helion._testing.cute_paged_headwide import headwide_kernel
from helion._testing.cute_paged_rowwise_lazy import paged_query_first_direct_attention
from helion._testing.cute_paged_rowwise_native import ROW_CASES
from helion._testing.cute_paged_rowwise_native import rowwise_config
from helion._testing.cute_paged_rowwise_native import rowwise_input_bytes
from helion._testing.cute_paged_rowwise_native import rowwise_reference
from helion._testing.cute_paged_rowwise_native import validate_rowwise_record
from helion.language._tracing_ops import _mask_to
from helion.language._tracing_ops import _new_var

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(query_len=4):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(headwide_kernel(), folded_attention_args(query_len))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return bound, program


@pytest.mark.parametrize("query_len", [4, 5])
def test_headwide_source_selects_exact_recurrence(query_len, tmp_path):
    bound, program = _bound(query_len)
    plan = match_paged_attention(program)
    assert plan is not None and plan.rowwise_rescale is not None
    assert plan.rowwise_rescale.headwide and plan.rowwise_rescale.threshold == 2.0
    axis = plan.query.tile_dims[0]
    assert axis.extent == query_len * 8
    assert axis.size == helion.next_power_of_2(query_len * 8)
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        facts = mma_dag_facts(program)
    assert facts.axes[axis.index].fixed == axis.size
    assert not facts.axes[axis.index].tunable
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(headwide_config(query_len))
    (tmp_path / "generated.py").write_text(source)
    assert "headwide_rescale=True" in source
    assert "rowwise_rescale_threshold=2.0" in source
    assert "paged_attention_paired_body(" in source
    assert "key_post_scale_log2=-6" in source
    assert "value_post_scale_log2=-22" in source
    assert "output_scale_log2=22" in source
    assert "reflected_dtype=" in source


def _copy(node):
    while node.target is _new_var:
        node = node.args[0]
    return node


# Each mutation removes a distinct proof obligation; one real traced graph is reused.
@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "multiple_heads",
        "no_fixed_owner",
        "padding_true",
        "no_mask",
        "wrong_axis",
        "wrong_cast",
        "non_boolean",
        "non_strict",
        "extra_predicate",
    ],
)
def test_headwide_certificate_refuses_incomplete_or_changed_vote(mutation):
    _, original = _bound()
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None
    selected = _copy(plan.maximum.output)
    predicate = selected.args[0]
    reduction = predicate.args[0]
    masked = reduction.args[0]
    converted = masked.args[0]
    rows = converted.args[0]
    if mutation in ("partial", "multiple_heads", "no_fixed_owner"):
        old = plan.query.tile_dims[0]
        updates = {
            "partial": {"size": 16},
            "multiple_heads": {"extent": 64, "size": 64},
            "no_fixed_owner": {"size": None},
        }[mutation]
        # This is a private copy of the real program; every shared axis reference
        # must observe the same changed ownership fact.
        for name, value in updates.items():
            object.__setattr__(old, name, value)
    elif mutation == "padding_true":
        masked.args = (converted, 1.0)
    elif mutation == "no_mask":
        reduction.args = (converted, [0])
    elif mutation == "wrong_axis":
        reduction.args = (masked, [1])
    elif mutation == "wrong_cast":
        converted.args = (rows, torch.float16)
    elif mutation == "non_boolean":
        with _bound()[0].env:
            rows.meta["val"] = torch.empty_like(rows.meta["val"], dtype=torch.float32)
    elif mutation == "non_strict":
        predicate.target = torch.ops.aten.ge.Scalar
    else:
        rows.args = (rows.args[0], rows.args[0])
    assert match_paged_attention(program) is None


def _evaluate_selection(plan, old, delta, valid_rows):
    """Execute the real traced recurrence slice, with its explicit logical mask."""
    certificate = plan.rowwise_rescale
    assert certificate is not None
    selected = _copy(plan.maximum.output)
    vote = selected.args[0]
    rows = vote.args[0].args[0].args[0].args[0]
    nan = rows.args[1]
    assert nan.target is torch.ops.aten.isnan.default
    tile_maximum = nan.args[0]
    values = {plan.maximum.placeholder: old, tile_maximum: delta}

    def evaluate(node):
        if not isinstance(node, torch.fx.Node):
            return node
        if node in values:
            return values[node]
        args = torch.fx.node.map_arg(node.args, evaluate)
        kwargs = dict(node.kwargs)
        if node.target is _new_var:
            result = args[0]
        elif node.target is _mask_to:
            result = torch.where(valid_rows, args[0], args[1])
        elif node.target is torch.ops.aten.scalar_tensor.default:
            kwargs["device"] = "cpu"
            result = node.target(*args, **kwargs)
        else:
            result = node.target(*args, **kwargs)
        values[node] = result
        return result

    return evaluate(plan.maximum.output)


@pytest.mark.parametrize("query_len", [4, 5])
def test_actual_source_vote_updates_other_rows_and_excludes_padding(query_len):
    _, program = _bound(query_len)
    plan = match_paged_attention(program)
    assert plan is not None
    rows = query_len * 8
    physical = helion.next_power_of_2(rows)
    valid = torch.arange(physical) < rows
    old = torch.full((physical,), -math.inf)
    old[:rows] = 1.0
    delta = torch.full((physical,), math.nan)
    delta[:rows] = 1.0
    # Invalid/NaN padding cannot trigger an update of otherwise quiescent rows.
    actual = _evaluate_selection(plan, old, delta, valid)
    torch.testing.assert_close(actual[:rows], old[:rows], rtol=0, atol=0)
    # One row's vote updates every other valid row, even below the threshold.
    delta[0] = 3.0
    actual = _evaluate_selection(plan, old, delta, valid)
    torch.testing.assert_close(actual[:rows], old[:rows] + delta[:rows], rtol=0, atol=0)
    # Strict threshold, invalid references, and NaNs use the unchanged source rule.
    for first_old, first_delta in [
        (1.0, 2.0),
        (1.0, float("inf")),
        (-math.inf, -math.inf),
        (math.inf, 0.0),
        (math.nan, 0.0),
        (1.0, math.nan),
    ]:
        old[0], delta[0] = first_old, first_delta
        actual = _evaluate_selection(plan, old, delta, valid)
        reference = torch.where(
            (old[:rows] >= 0) & (old[:rows] < math.inf), old[:rows], 0.0
        )
        needs = (
            (delta[:rows] > 2.0)
            | ~((old[:rows] >= 0) & (old[:rows] < math.inf))
            | torch.isnan(delta[:rows])
        )
        expected = (
            torch.maximum(old[:rows], delta[:rows] + reference)
            if bool(needs.any())
            else old[:rows]
        )
        torch.testing.assert_close(
            actual[:rows], expected, rtol=0, atol=0, equal_nan=True
        )


def test_existing_rowwise_source_does_not_select_headwide_policy():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(paged_query_first_direct_attention, folded_attention_args(4))
        source = bound.to_code(rowwise_config(4))
    assert "rowwise_rescale_threshold=2.0" in source
    assert "headwide_rescale" not in source


def test_authored_transform_changes_only_vote_and_complete_tile():
    original = ast.parse(inspect.getsource(paged_query_first_direct_attention.fn)).body[
        0
    ]
    actual = ast.parse(inspect.getsource(headwide_kernel().fn)).body[0]
    original.decorator_list = []
    actual.name = original.name
    rewrites = 0
    for node in ast.walk(actual):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "hl.tile":
            node.keywords[0].value = ast.parse("[1, None]", mode="eval").body
            rewrites += 1
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "torch.where"
            and ast.unparse(node.args[0]).startswith("torch.amax(needs_rescale.float()")
        ):
            node.args[0] = ast.Name("needs_rescale", ast.Load())
            rewrites += 1
    assert rewrites == 2
    assert ast.dump(original) == ast.dump(actual)


@pytest.mark.parametrize("rows,block", [(16, 16), (40, 64), (32, 16)])
def test_fixed_root_size_is_retained_or_partial_root_refused(rows, block):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            fixed_root_matmul,
            (
                torch.empty((rows * 2, 16), dtype=torch.bfloat16)[::2],
                torch.empty((16, 16), dtype=torch.bfloat16),
                block,
            ),
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
        if block < rows:
            assert program is None
            return
        assert program is not None
        source = bound.to_code(
            helion.Config(block_sizes=[], cute_mma_dag="warp", cute_mma_dag_warps=1)
        )
        assert "_helion_mma_dag" in source
        roots = [a for a in program.axes if a.kind == "root"]
        assert [(a.extent, a.size) for a in roots] == [(rows, block), (16, 16)]


def test_native_head_vote_refuses_extra_physical_register_rows():
    bound, program = _bound(2)
    plan = match_paged_attention(program)
    assert plan is not None and plan.rowwise_rescale is not None
    assert plan.rowwise_rescale.headwide
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        # Sixteen logical rows would occupy the native 32-row panel. Its
        # invalid-maximum vote has no padding mask, so this route must decline.
        assert paged_attention_schedule_facts(bound.env, plan) is None


def test_scoped_headwide_source_retains_single_head_ownership():
    bound, _ = _bound(5)
    config = headwide_config(5, scoped=True)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "paged_attention_partial_body(" in source
    assert "headwide_rescale=True" in source


def test_native_headwide_cases_keep_empty_nonfinite_and_head_boundaries():
    args_list = list(folded_attention_args(5))
    args_list[7] = 2047
    args = tuple(args_list)
    for case in ROW_CASES:
        configure_headwide_case(args, case)
        out, lse = rowwise_reference(args, headwide=True)
        if case in ("original", "restored"):
            assert (out[0] == 0).all() and torch.isneginf(lse[0]).all()
        elif case == "changed_q":
            assert args[6].tolist() == [128, 0]
            assert (out[1] == 0).all() and torch.isneginf(lse[1]).all()
        elif case in ("nonfinite", "zero_v_nan_q"):
            assert torch.isnan(out[0, 0, 0]).all() and torch.isnan(lse[0, 0, 0])
        else:
            assert torch.isfinite(out).all()
    configure_headwide_case(args, "mixed_rows")
    before = rowwise_reference(args, headwide=True)
    args[0][:, :, 8:].mul_(16)
    after = rowwise_reference(args, headwide=True)
    for old, new in zip(before, after, strict=True):
        torch.testing.assert_close(old[:, :, :8], new[:, :, :8], rtol=0, atol=0)


def test_full_record_revalidation_rejects_missing_lse_and_changed_input():
    actual = [
        torch.zeros((2, 4, 16, 128), dtype=torch.bfloat16),
        torch.zeros((2, 4, 16)),
    ]
    before = [torch.zeros(n, dtype=torch.uint8) for n in rowwise_input_bytes(4)]
    record = {
        "query_len": 4,
        "actual": actual,
        "expected": [x.clone() for x in actual],
        "input_before": before,
        "input_after": [x.clone() for x in before],
        "inputs_unchanged": [True] * 5,
    }
    assert validate_rowwise_record(record, 4) == 16512
    record.update(
        output_backing_before=torch.zeros(8, dtype=torch.uint8),
        output_backing_after=torch.zeros(8, dtype=torch.uint8),
        output_written_bytes=torch.tensor(
            [False, True, True, True, True, True, True, False]
        ),
    )
    record["output_backing_after"][-1] = 1
    with pytest.raises(AssertionError):
        validate_rowwise_record(record, 4)
    record["output_backing_after"][-1] = 0
    record["input_after"][0][-1] = 1
    with pytest.raises(AssertionError):
        validate_rowwise_record(record, 4)
    record["input_after"][0][-1] = 0
    record["actual"][1] = actual[1][..., :1]
    with pytest.raises(AssertionError):
        validate_rowwise_record(record, 4)
