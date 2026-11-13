from __future__ import annotations

import ast
import copy
import dataclasses
import functools

import pytest

import helion
from helion import exc
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_folded_attention import folded_attention_reference
from helion._testing.cute_paged_rowwise_native import configurable_rowwise_kernel
from helion._testing.cute_paged_rowwise_native import rowwise_config
from helion._testing.cute_source import transform_kernel

pytestmark = pytest.mark.cpu_only


class _Fragment(ast.NodeTransformer):
    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.changed = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_tunable"
        ):
            node.args[1] = ast.parse(self.expression, mode="eval").body
            self.changed += 1
        return node


@functools.cache
def _bound(expression: str):
    base = configurable_rowwise_kernel()
    transform = _Fragment(expression)
    changed = transform_kernel(
        base,
        transform,
        name=base.fn.__name__,
        filename=f"<rowwise-domain-{expression}>",
    )
    assert transform.changed == 1
    kernel = helion.kernel(changed.fn, settings=copy.deepcopy(base.settings))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, folded_attention_args(4))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    return bound, program


@pytest.mark.parametrize(
    "expression",
    [
        "helion.autotuner.IntegerFragment(-2, 8, -1)",
        "helion.autotuner.EnumFragment((2, -1), search_choices=(2,))",
        "helion.autotuner.EnumFragment((2, 2**24 + 1))",
        "helion.autotuner.EnumFragment((2, True))",
        "helion.autotuner.EnumFragment((2, 2.0))",
        "helion.autotuner.IntegerFragment(2**24, 2**24 + 2)",
        "helion.autotuner.EnumFragment((2, 2**60 + 1))",
        "helion.autotuner.PowerOfTwoFragment(2**24, 2**25, 2**24 + 1)",
    ],
)
def test_ineligible_declared_domain_preserves_ordinary_codegen(expression):
    _, program = _bound(expression)
    assert program is None


def test_ineligible_domain_renders_ordinary_codegen():
    # Domain-specific refusal is checked above; one full render covers fallback.
    bound, _ = _bound("helion.autotuner.IntegerFragment(-2, 8, -1)")
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(bound.config_spec.default_config())
    ast.parse(source)
    assert "paged_attention_paired_body(" not in source
    assert "paged_attention_partial_body(" not in source


@pytest.mark.parametrize(
    "expression",
    [
        "helion.autotuner.EnumFragment((2, 16, 32, 64))",
        "helion.autotuner.IntegerFragment(0, 64, 2)",
        "helion.autotuner.PowerOfTwoFragment(2, 64, 2)",
        "helion.autotuner.PowerOfTwoFragment(2, 64, 3)",
        "helion.autotuner.IntegerFragment(2**25, 2**25)",
    ],
)
def test_complete_exact_integer_domain_retains_paged_lowering(expression):
    bound, program = _bound(expression)
    assert program is not None
    plan = match_paged_attention(program)
    assert plan is not None and plan.rowwise_rescale is not None
    value = bound.config_spec.user_defined_tunables["rescale_threshold"].default()
    assert (
        plan.rowwise_rescale.threshold_for(helion.Config(rescale_threshold=value))
        == value
    )
    assert (
        match_paged_attention(dataclasses.replace(program, config_fragments={})) is None
    )
    config = rowwise_config(4)
    config.config["rescale_threshold"] = value
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert f"rowwise_rescale_threshold={float(value)!r}" in source


@pytest.mark.parametrize("threshold", [-1, 2**24 + 1])
@pytest.mark.parametrize("reference", [False, True])
def test_supplied_outside_domain_keeps_ordinary_codegen(threshold, reference):
    bound, _ = _bound("helion.autotuner.EnumFragment((2, 16))")
    config = (
        bound.config_spec.autotune_reference_config()
        if reference
        else helion.Config(block_sizes=[32])
    )
    config.config["rescale_threshold"] = threshold
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    ast.parse(source)
    assert config.config["rescale_threshold"] == threshold
    assert "paged_attention_paired_body(" not in source
    assert "paged_attention_partial_body(" not in source


@pytest.mark.parametrize("threshold", [-1, 2**24 + 1])
def test_forced_paged_outside_domain_fails_closed(threshold):
    bound, _ = _bound("helion.autotuner.EnumFragment((2, 16))")
    config = rowwise_config(4)
    config.config["rescale_threshold"] = threshold
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(exc.InvalidConfig, match="rowwise rescale threshold"),
    ):
        bound.to_code(config)


def test_exact_supplied_value_outside_choices_retains_paged_codegen():
    bound, _ = _bound("helion.autotuner.EnumFragment((2, 16))")
    config = rowwise_config(4)
    config.config["rescale_threshold"] = 32
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "rowwise_rescale_threshold=32.0" in source


def test_reference_rejects_threshold_without_relative_scores():
    with pytest.raises(AssertionError):
        folded_attention_reference(
            folded_attention_args(4), rowwise_rescale_threshold=2
        )
