from __future__ import annotations

import ast

from examples.int4_gemm import matmul_bf16_int4
import pytest
import torch

from test.test_cute_epilogue_fanout import cuda_trace  # noqa: F401

import helion
from helion._compiler.cute.packed_operand import KEY
from helion._compiler.cute.packed_operand import MODES
from helion._compiler.cute.packed_operand import prove_packed_operand
from helion._testing import skipUnlessBackends
from helion.autotuner.compiler_coverage import append_compiler_coverage

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("reduction", (64, 128, 256))
def test_original_packed_operand_reaches_ordinary_codegen(reduction: int) -> None:
    kernel = helion.kernel(
        matmul_bf16_int4.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        cute_materialize_transformed_operands=True,
    )
    arguments = (
        torch.empty((32, reduction), dtype=torch.bfloat16),
        torch.empty((reduction // 2, 12), dtype=torch.int8),
    )
    bound = kernel._bind_isolated(arguments)
    host = bound.host_function
    assert host is not None
    with bound.env, host, bound._runtime_arg_values_for_codegen():
        plan = prove_packed_operand(bound.env, host, binding=True)
    assert plan is not None
    assert (plan.rows, plan.columns, plan.reduction) == (32, 12, reduction)
    (group,) = [g for g in bound.config_spec.compiler_coverage_groups if g.key == KEY]
    assert group.domain == MODES
    pinned = []
    generation = bound.config_spec.create_config_generation()
    population, outcomes = append_compiler_coverage(
        [],
        generation,
        cached_configs=list,
        pin=pinned.append,
    )
    (selected,) = [
        item.effective
        for item in outcomes
        if item.mechanism == group.mechanism
        and item.effective is not None
        and item.effective.config.get(KEY) == "warp_narrow4"
    ]
    assert selected in pinned and generation.flatten(selected) in population
    source = bound.to_code(selected)
    tree = ast.parse(source)
    host_ast = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == host.name
    )
    launches = [
        node
        for node in ast.walk(host_ast)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_launcher"
    ]
    assert len(launches) == 1
    call = launches[0]
    assert ast.literal_eval(call.args[1]) == (1, 3, 1)
    assert [ast.unparse(argument) for argument in call.args[2:]] == list(plan.arguments)
    assert {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords} == {
        "block": (128, 1, 1)
    }
    assert not any(
        isinstance(node, ast.Name) and node.id == plan.materialized_name
        for node in ast.walk(host_ast)
    )


@pytest.mark.parametrize("reduction", (512, 1024))
def test_large_packed_operand_keeps_ordinary_fallback(reduction: int) -> None:
    kernel = helion.kernel(
        matmul_bf16_int4.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        cute_materialize_transformed_operands=True,
    )
    bound = kernel._bind_isolated(
        (
            torch.empty((32, reduction), dtype=torch.bfloat16),
            torch.empty((reduction // 2, 12), dtype=torch.int8),
        )
    )
    assert not bound.config_spec.cute_materialized_operand_schedule_available
    assert all(g.key != KEY for g in bound.config_spec.compiler_coverage_groups)
