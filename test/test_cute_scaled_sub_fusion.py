from __future__ import annotations

import ast
import ctypes
import ctypes.util
from types import SimpleNamespace

from examples.aot_example import row_softmax
import numpy as np
import pytest
import torch

from .test_cute_bounded_cache_codegen import _config
from .test_cute_bounded_cache_codegen import _cpu_target
import helion
from helion._compiler.cute.fuse_fma import fuse_fma
from helion._compiler.cute.hoist_loop_invariant_recip import hoist_loop_invariant_recips
from helion._compiler.cute.scaled_sub_fusion import SCALED_SUBTRACTION_ATTR
from helion._compiler.cute.scaled_sub_fusion import (
    contract_distributed_scale_subtractions,
)
from helion._testing import skipUnlessBackends

SOURCE = """
offset = cutlass.Float32(other)
for lane in range(1):
    value = cutlass.Float32(input + lane)
    result = (value - offset) * 1.4426950408889634
"""


def _count_fmas(body):
    return sum(
        isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.math.fma"
        for statement in body
        for node in ast.walk(statement)
    )


def _marked_body(source):
    body = ast.parse(source).body
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                setattr(node, SCALED_SUBTRACTION_ATTR, True)
    return body


@pytest.mark.parametrize("reverse", [False, True])
def test_existing_scaled_sub_rewrite_contracts_explicit_fp32_only(reverse):
    source = SOURCE.replace("value - offset", "offset - value") if reverse else SOURCE
    body = hoist_loop_invariant_recips(ast.parse(source).body)
    assert _count_fmas(body) == 0
    marked = [
        node
        for statement in body
        for node in ast.walk(statement)
        if getattr(node, SCALED_SUBTRACTION_ATTR, False)
    ]
    assert len(marked) == 1
    body = fuse_fma(body)
    assert _count_fmas(body) == 1
    text = ast.unparse(ast.Module(body=body, type_ignores=[]))
    if reverse:
        assert "cute.math.fma(-value, 1.4426950408889634, _helion_scaled_0)" in text
    else:
        assert "cute.math.fma(value, 1.4426950408889634, -_helion_scaled_0)" in text


@pytest.mark.parametrize("dtype", ["Float16", "BFloat16", "Float64", "Int32", "Int64"])
def test_scale_marker_does_not_prove_other_arithmetic_is_fp32(dtype):
    body = _marked_body(
        f"value = cutlass.{dtype}(input)\nscaled = cutlass.{dtype}(other)\nresult = value * 1.25 - scaled\n"
    )
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    result = contract_distributed_scale_subtractions(body, {})
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == original


@pytest.mark.parametrize(
    "prefix",
    [
        "value = input",
        "value = cutlass.Float32(input)\nif condition:\n    value = cutlass.Int32(input)",
        "if condition:\n    value = cutlass.Float32(input)",
        "value = cutlass.Float32(input)\nvalue = opaque(input)",
    ],
)
def test_unknown_or_conditionally_rebound_input_keeps_original_arithmetic(prefix):
    body = _marked_body(
        prefix + "\nscaled = cutlass.Float32(other)\nresult = value * 1.25 - scaled\n"
    )
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    result = contract_distributed_scale_subtractions(body, {})
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == original


def test_definite_float32_type_on_both_branch_paths_is_sufficient():
    body = _marked_body(
        "if condition:\n    value = cutlass.Float32(input)\nelse:\n    value = cutlass.Float32(other)\nscaled = cutlass.Float32(other)\nresult = value * 1.25 - scaled\n"
    )
    result = contract_distributed_scale_subtractions(body, {})
    assert _count_fmas(result) == 1


def test_repeated_body_cannot_reuse_initializer_type_after_integer_rebinding():
    body = _marked_body(
        "value = cutlass.Float32(input)\nscaled = cutlass.Float32(other)\nfor lane in range(2):\n    result = value * 1.25 - scaled\n    value = cutlass.Int32(input)\n"
    )
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    result = contract_distributed_scale_subtractions(body, {})
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == original


def test_zero_trip_loop_does_not_establish_an_outer_float32_definition():
    body = _marked_body(
        "scaled = cutlass.Float32(other)\nfor lane in range(count):\n    value = cutlass.Float32(input)\nresult = value * 1.25 - scaled\n"
    )
    result = contract_distributed_scale_subtractions(body, {})
    assert _count_fmas(result) == 0


def test_post_rename_integer_rebinding_invalidates_the_float32_fact():
    body = _marked_body(
        "value = cutlass.Float32(input)\nscaled = cutlass.Float32(other)\nalias = cutlass.Int32(input)\nresult = value * 1.25 - scaled\n"
    )
    result = contract_distributed_scale_subtractions(body, {"alias": "value"})
    assert _count_fmas(result) == 0


def test_unmarked_nested_product_and_explicit_fma_are_unchanged():
    body = ast.parse(
        "value = cutlass.Float32(input)\nscaled = cutlass.Float32(other)\nfirst = value * 1.25 - scaled\nsecond = cute.math.fma(value, 1.25, -scaled)\n"
    ).body
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    result = contract_distributed_scale_subtractions(body, {})
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == original


def test_unknown_operand_call_is_not_reordered_past_the_scaled_value():
    body = _marked_body(
        "scaled = cutlass.Float32(other)\nresult = scaled - cutlass.Float32(opaque()) * 1.25\n"
    )
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    result = contract_distributed_scale_subtractions(body, {})
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == original


def test_direct_float32_scalar_cast_retains_the_explicit_conversion():
    body = _marked_body(
        "scaled = cutlass.Float32(other)\n"
        "result = cutlass.Float32(value) * 1.25 - scaled\n"
    )
    result = contract_distributed_scale_subtractions(body, {})
    assert _count_fmas(result) == 1
    assert "cute.math.fma(cutlass.Float32(value), 1.25, -scaled)" in ast.unparse(
        ast.Module(body=result, type_ignores=[])
    )


@pytest.mark.parametrize(
    "binding",
    [
        "def value():\n    return 1",
        "class value:\n    pass",
        "try:\n    value = cutlass.Int32(input)\nexcept Exception:\n    pass",
    ],
)
def test_opaque_binding_construct_drops_reaching_scalar_types(binding):
    body = _marked_body(
        "value = cutlass.Float32(input)\nscaled = cutlass.Float32(other)\n"
        + binding
        + "\nresult = value * 1.25 - scaled\n"
    )
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    result = contract_distributed_scale_subtractions(body, {})
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == original


def test_embedded_assignment_does_not_inherit_a_stale_type():
    body = _marked_body(
        "value = cutlass.Float32(input)\nscaled = cutlass.Float32(other)\n"
        "result = ((value := cutlass.Int32(input)), value * 1.25 - scaled)\n"
    )
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    result = contract_distributed_scale_subtractions(body, {})
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == original


def _libm_fmaf():
    """Correctly rounded fp32 FMA from the C math library, on Linux or macOS."""
    library = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    fma = library.fmaf
    fma.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fma.restype = ctypes.c_float
    return fma


def _execute_explicit(body, value, offset):
    fma = _libm_fmaf()
    calls = []

    def fp32_fma(left, right, addend):
        calls.append((left, right, addend))
        return np.float32(fma(float(left), float(right), float(addend)))

    namespace = {
        "input": value,
        "other": offset,
        "cutlass": SimpleNamespace(Float32=np.float32),
        "cute": SimpleNamespace(math=SimpleNamespace(fma=fp32_fma)),
    }
    with np.errstate(all="ignore"):
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                "<scaled-sub-fusion>",
                "exec",
            ),
            namespace,
        )
    return namespace["result"], calls


@pytest.mark.parametrize(
    "value,offset",
    [
        (2.208984375, 3.37890625),
        (0.125244140625, 3.37890625),
        (0.0, 0.0),
        (-0.0, 0.0),
        (0.0, -0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (float("inf"), float("inf")),
        (float("nan"), 0.0),
    ],
)
def test_explicit_contract_preserves_fp32_fma_width_and_nonfinite_classes(
    value, offset
):
    # Keep the signed-zero operands intact here. SOURCE's deliberate lane
    # dependency adds +0 before the cast, which changes a -0 input to +0.
    body = fuse_fma(
        _marked_body(
            "value = cutlass.Float32(input)\n"
            "scaled = cutlass.Float32(other) * 1.4426950408889634\n"
            "result = value * 1.4426950408889634 - scaled\n"
        )
    )
    result, calls = _execute_explicit(body, value, offset)
    assert len(calls) == 1
    left, scale, addend = calls[0]
    assert isinstance(left, np.float32)
    assert isinstance(addend, np.float32)
    fma = _libm_fmaf()
    expected = np.float32(
        fma(float(np.float32(value)), float(np.float32(scale)), float(addend))
    )
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result.tobytes() == expected.tobytes()


@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@skipUnlessBackends(["cute"])
def test_actual_masked_and_vector_sources_follow_the_same_fusion_policy(
    fast_math, aligned, dtype
):
    vector = 4 if dtype == torch.float32 else 8
    with _cpu_target():
        storage = torch.empty(17 * 513 + 1, dtype=dtype)
        x = storage.as_strided((17, 513), (513, 1), int(not aligned))
        kernel = helion.kernel(
            row_softmax.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
            fast_math=fast_math,
            cute_full_slice_matmul_tiling=True,
            cute_segmented_matmul_tiling=True,
            cute_flatten_nested_reductions=True,
        )
        bound = kernel._bind_isolated((x,))
        scalar = bound.to_code(_config("scalar", vector=vector))
        generated = bound.to_code(_config("bounded_layout", vector=vector))
    functions = {
        node.name: node
        for node in ast.parse(generated).body
        if isinstance(node, ast.FunctionDef)
    }
    original = next(
        node
        for node in ast.parse(scalar).body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion_row_softmax"
    )
    assert ast.dump(original) == ast.dump(functions["_helion_row_softmax"])
    assert _count_fmas(original.body) >= 2
    assert _count_fmas(functions["_helion_row_softmax_bounded"].body) >= 2
    assert ("cute.arch.load(" in scalar) is aligned
    assert ("cute.arch.load(" in generated) is aligned
