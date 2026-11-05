"""Typed negative zero preserves bits across the real DSL literal-cache scope.

All IR tests are CPU-only construction; no compile, NVCC or GPU launch.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test.test_cute_shared_semantic_memo import arguments
from test.test_cute_shared_semantic_memo import configuration
from test.test_cute_shared_semantic_memo import repeated_prefix

import helion
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute import shared_epilogue_dag as shared
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def literal(dtype, value):
    expr = epi._ScalarConstantExpr(repr(value), dtype)
    source, text = epi._render_auxiliary_tensor_expr(
        expr, "unused", {}, lambda name: name, ""
    )
    assert not source
    return text


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("negative_first", (False, True))
def test_real_CUTLASS_constant_IR_retains_both_zero_signs_in_one_insertion_scope(
    dtype, negative_first
):
    import importlib

    import cutlass

    # The extension module exports its IR classes dynamically.
    ir = importlib.import_module("cutlass._mlir.ir")

    positive = literal(dtype, 0.0)
    negative = literal(dtype, -0.0)
    assert positive.endswith(("(0.0)", "(0.0))"))
    assert "bitcast" in negative and "Float32(-0.0)" not in negative
    assert (
        "Int32(-2147483648)" in negative
        if dtype is torch.float32
        else "Int16(-32768)" in negative
    )
    ordered = (negative, positive) if negative_first else (positive, negative)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            values = [eval(text, {"cutlass": cutlass}).ir_value() for text in ordered]
            # Compare with the base MLIR implementation, NOT overloaded numeric
            # __eq__ (which emits an arithmetic comparison instead of a bool).
            assert not ir.Value.__eq__(*values)
        text = str(module)
        assert "arith.bitcast" in text
        assert "-2147483648" in text if dtype is torch.float32 else "-32768" in text
        assert "arith.constant 0.000000e+00" in text


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_literal_repair_is_bit_exact_and_nonzero_constructor_text_is_unchanged(dtype):
    assert literal(dtype, 1e-6) == (
        f"cutlass.Float32(cutlass.{'Float16' if dtype is torch.float16 else 'BFloat16'}(1e-06))"
        if dtype is not torch.float32
        else "cutlass.Float32(1e-06)"
    )
    negative = literal(dtype, -0.0)

    class IntBits(int):
        def bitcast(self, target):
            source = torch.int32 if target is torch.float32 else torch.int16
            return torch.tensor(int(self), dtype=source).view(target)

    namespace = SimpleNamespace(
        Int32=IntBits,
        Int16=IntBits,
        Float32=lambda x: torch.as_tensor(x).to(torch.float32),
        Float16=torch.float16,
        BFloat16=torch.bfloat16,
    )
    # Only this test namespace maps the DSL result type; product helper/cache
    # implementation is neither patched nor replaced.
    text = negative.replace(".bitcast(cutlass.Float32)", ".bitcast(torch.float32)")
    got = eval(text, {"cutlass": namespace, "torch": torch})
    assert isinstance(got, torch.Tensor)
    assert got.dtype is torch.float32 and got.item() == 0.0 and torch.signbit(got)
    u = torch.tensor(-0.0)
    denominator = u + got
    output = (torch.tensor(1.0) / denominator) * torch.tensor(-0.625)
    assert output.item() == float("inf") and not torch.signbit(output)


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="full", fast_math=False
)
def weak_zero_prefix(a, b, u, out0, out1):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        positive = u[row] + 0.0
        negative = u[row] + -0.0
        out0[row, col] = acc / positive.reshape([row.index.size(0), 1])
        out1[row, col] = acc / negative.reshape([row.index.size(0), 1])


def test_already_admitted_weak_M_prefix_literals_use_the_same_typed_signed_storage_renderer():
    values = arguments(torch.float16)
    kernel = helion.kernel(weak_zero_prefix.fn, settings=weak_zero_prefix.settings)
    actual = (values[0], values[1], values[2], values[6], values[7])
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        bound = cpu_bind(kernel, actual)
    assert bound.config_spec.cute_tcgen05_search_enabled
    for config in (configuration(True), configuration(False)):
        code = bound.to_code(config)
        assert "cutlass.Int32(-2147483648).bitcast(cutlass.Float32)" in code
        assert "cutlass.Float32(-0.0)" not in code
        assert code.count("PipelineTmaStore.create(") == 2
        assert code.count("tcgen05_acc_pipeline.consumer_release(") == 1


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_authorized_CPU_adapter_reinterprets_integer_storage_before_float_widening(
    dtype,
):
    from test.test_cute_epilogue_scalar_constants import _evaluate

    got = _evaluate("", literal(dtype, -0.0), torch.tensor(1.0))
    assert got.dtype is torch.float32
    assert got.view(torch.int32).item() == -2147483648
    u = torch.tensor(-0.0)
    negative_denominator = u + got
    output = (torch.tensor(1.0) / negative_denominator) * torch.tensor(-0.625)
    assert output.view(torch.int32).item() == 2139095040


@pytest.mark.parametrize(
    "expression",
    (
        "cutlass.Int16(-32768).bitcast(cutlass.Float32)",
        "cutlass.Int32(-2147483648).bitcast(cutlass.Float16)",
        "cutlass.Int32(-2147483648.0).bitcast(cutlass.Float32)",
        "cutlass.Int32(0).bitcast(cutlass.Float32)",
        "cutlass.Int64(-9223372036854775808).bitcast(cutlass.Float32)",
        "cutlass.Int32(-2147483648)",
        "cutlass.Int32(-2147483648).bitcast(cutlass.Float64)",
    ),
)
def test_authorized_CPU_adapter_rejects_unknown_integer_values_and_widths(expression):
    from test.test_cute_epilogue_scalar_constants import _evaluate

    with pytest.raises(ValueError):
        _evaluate("", expression, torch.tensor(1.0))


def test_exact_managed_zero_fixture_preserves_distinct_prefixes_and_all_store_protocols():
    kernel = helion.kernel(repeated_prefix.fn, settings=repeated_prefix.settings)
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        bound = cpu_bind(kernel, (*arguments(torch.float16), "zero"))
    state = bound.config_spec._cute_tcgen05_config
    (plan,) = state.epilogue_fanout_plans
    assert isinstance(plan, shared.SharedDagFanoutPlan)
    assert plan.semantic_memo is not None
    sources = {
        name: bound.to_code(configuration(name == "shared"))
        for name in ("off", "shared")
    }
    for source in sources.values():
        divs = [
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
        ]
        assert len(divs) == 4
        assert "cutlass.Int32(-2147483648).bitcast(cutlass.Float32)" in source
        assert "cutlass.Float32(-0.0)" not in source
        assert source.count("PipelineTmaStore.create(") == 4
        assert source.count("producer_commit()") == source.count("producer_tail()") == 4
        assert (
            source.count("tcgen05_acc_pipeline.consumer_release(")
            == source.count("tcgen05_acc_consumer_state.advance()")
            == 1
        )
    left = plan.chains[1].steps[0].expr
    right = plan.chains[2].steps[0].expr
    assert isinstance(left, epi._PointwiseDagExpr)
    assert isinstance(right, epi._PointwiseDagExpr)
    assert shared.dag_keys(left) != shared.dag_keys(right)
    assert plan.semantic_memo.value_keys[1][-1] != plan.semantic_memo.value_keys[2][-1]
    key = "tensor_storage_disjoint_matrix_v1"
    saved = bound.env.bound_runtime_input_specialization_results.pop(key)
    try:
        with pytest.raises((helion.exc.BackendUnsupported, helion.exc.InvalidConfig)):
            bound.to_code(configuration())
        assert key not in bound.env.bound_runtime_input_specialization_results
    finally:
        bound.env.bound_runtime_input_specialization_results[key] = saved
