from __future__ import annotations

import functools

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import _Matcher
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_positive_zero import POSITIVE_ZERO_ATTENTION_CASES
from helion._testing.cute_paged_positive_zero import positive_zero_attention_fixture
from helion._testing.cute_paged_postscale import postscale_attention_fixture

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "form,schedule,occupancy,query_len,post_scale,fast_math",
    POSITIVE_ZERO_ATTENTION_CASES,
)
def test_positive_zero_reaches_each_selected_value_decoder(
    form, schedule, occupancy, query_len, post_scale, fast_math
):
    kernel, args = positive_zero_attention_fixture(
        query_len, post_scale, "cpu", form=form, reference_scores=form == "mapped"
    )
    kernel = helion.kernel(
        kernel.fn, backend="cute", static_shapes=True, fast_math=fast_math
    )
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps=schedule,
        cute_min_blocks_per_mp=occupancy,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = cpu_bind(kernel, args).to_code(config)
    body = (
        "paged_attention_body"
        if form == "reduced"
        else "paged_attention_paired_body"
        if schedule == "attention_paired"
        else "paged_attention_partial_body"
    )
    assert f"{body}(" in source
    assert "value_positive_zero=True" in source
    assert "key_post_scale_log2=-6" in source
    if post_scale:
        assert f"value_post_scale_log2={post_scale}" in source
    else:
        assert "value_post_scale_log2=" not in source


def test_positive_zero_postscale_domain_and_native_identity():
    codes = torch.arange(65536, dtype=torch.int32)
    raw = (codes % 256).to(torch.uint8).view(torch.int8).float()
    scales = codes // 256
    decoded = (raw * torch.exp2(scales.float() - 127)).bfloat16()
    # Every INT8/scale pair and admitted exponent share CPU tensors; no GPU
    # compilation grid is needed to cover the underflow and native boundaries.
    for exponent in range(-22, 1):
        before = (decoded * 2.0**exponent).bfloat16()
        after = before + 0.0
        nonzero = before.isfinite() & (before != 0)
        assert torch.equal(
            before[nonzero].view(torch.int16), after[nonzero].view(torch.int16)
        )
        assert (after[before == 0].view(torch.int16) == 0).all()
        assert torch.equal(before.isnan(), after.isnan())
        native = (scales <= 247) & (scales >= max(0, -6 - exponent))
        assert torch.equal(
            before[native].view(torch.int16), after[native].view(torch.int16)
        )
        if exponent <= -14:
            zero = scales <= -14 - exponent
            assert (after[zero].view(torch.int16) == 0).all()
    boundary = (decoded * 2.0**-13).bfloat16() + 0.0
    assert boundary[(raw == -128) & (scales == 0)].ne(0).all()


@functools.cache
def _bound(post_scale=-22):
    kernel, args = positive_zero_attention_fixture(post_scale=post_scale)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = _Matcher(bound.host_function.device_ir.graphs).match()
    assert program is not None
    return bound, program


@pytest.mark.parametrize("post_scale", [-22, -14, -6, 0])
def test_written_positive_zero_is_a_separate_source_fact(post_scale):
    bound, program = _bound(post_scale)
    plan = match_paged_attention(program)
    assert plan is not None and plan.value.positive_zero
    assert not plan.key.positive_zero
    assert plan.value.post_scale_log2 == post_scale
    assert plan.value.value.target is torch.ops.aten.add.Tensor
    if post_scale == 0:
        decoded = plan.value.value.args[0]
        assert isinstance(decoded, torch.fx.Node)
        assert decoded.target is torch.ops.prims.convert_element_type.default
        assert decoded.args[1] == torch.bfloat16
    # The new source fact is also admitted by the transport proof.
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        assert plan_paged_attention_transport(bound.env, plan) is not None


def test_ordinary_source_remains_admitted_without_positive_zero():
    kernel, args = postscale_attention_fixture(
        "mapped",
        5,
        True,
        key=-6,
        value=-22,
        reference_scores=True,
        base2_scores=True,
        reverse_pages=True,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = _Matcher(bound.host_function.device_ir.graphs).match()
            assert program is not None
            plan = match_paged_attention(program)
            assert plan is not None and not plan.value.positive_zero
            assert plan_paged_attention_transport(bound.env, plan) is not None


def test_commuted_positive_zero_retains_the_source_node():
    _, original = _bound()
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None
    value = plan.value.value
    value.args = tuple(reversed(value.args))
    updated = match_paged_attention(program)
    assert updated is not None and updated.value.positive_zero
    assert updated.value.value is value


@pytest.mark.parametrize(
    "mutation",
    [
        "negative_zero",
        "nonzero",
        "integer_zero",
        "bool_zero",
        "alpha",
        "fp32_result",
        "fp32_input",
        "before_postscale",
        "duplicate_add",
        "key_add",
        "postscale_outside_domain",
    ],
)
def test_positive_zero_proof_declines_nearby_sources(mutation):
    # Pin the combined sign, dtype, placement and operand-role admission rules;
    # dtype mutations may also be refused by the underlying decode proof.
    _, original = _bound()
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None
    value = plan.value.value
    scaled, zero = value.args
    assert isinstance(scaled, torch.fx.Node)
    if mutation in ("negative_zero", "nonzero", "integer_zero", "bool_zero"):
        factor = {
            "negative_zero": -0.0,
            "nonzero": 1.0,
            "integer_zero": 0,
            "bool_zero": False,
        }[mutation]
        value.args = (scaled, factor)
    elif mutation == "alpha":
        value.kwargs = {"alpha": 2}
    elif mutation in ("fp32_result", "fp32_input"):
        node = value if mutation == "fp32_result" else scaled
        node.meta["val"] = node.meta["val"].to(torch.float32)
    elif mutation == "before_postscale":
        decoded, factor = scaled.args
        value.replace_all_uses_with(scaled)
        value.args = (decoded, zero)
        scaled.args = (value, factor)
    elif mutation == "duplicate_add":
        with value.graph.inserting_before(value):
            inner = value.graph.call_function(torch.ops.aten.add.Tensor, (scaled, 0.0))
            inner.meta = dict(value.meta)
        value.args = (inner, zero)
    elif mutation == "key_add":
        transposed = program.dots[0].node.args[1]
        assert isinstance(transposed, torch.fx.Node)
        key = transposed.args[0]
        assert isinstance(key, torch.fx.Node)
        with transposed.graph.inserting_before(transposed):
            added = transposed.graph.call_function(
                torch.ops.aten.add.Tensor, (key, 0.0)
            )
            added.meta = dict(key.meta)
        transposed.args = (added, *transposed.args[1:])
    elif mutation == "postscale_outside_domain":
        scaled.args = (scaled.args[0], 2.0**-23)
    else:
        raise AssertionError(mutation)
    assert match_paged_attention(program) is None
