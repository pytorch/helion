from __future__ import annotations

import ast
import ctypes
import ctypes.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .test_cute_bounded_cache_codegen import _config
from .test_cute_bounded_cache_codegen import _cpu_target
from .test_cute_bounded_cache_cuda import CacheCase
from .test_cute_bounded_cache_cuda import run_guarded_cache_case
import helion
from helion._compiler.cute.fuse_fma import fuse_fma
from helion._compiler.cute.hoist_loop_invariant_recip import hoist_loop_invariant_recips
from helion._compiler.cute.scaled_sub_fusion import SCALED_SUBTRACTION_ATTR
from helion._compiler.cute.scaled_sub_fusion import (
    contract_distributed_scale_subtractions,
)
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import pairwise
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only

row_softmax = import_example("aot_example").row_softmax

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
    # Distributing the scale rounds differently, and only in loops, so
    # without fast_math the product keeps its exact form.
    exact = hoist_loop_invariant_recips(ast.parse(source).body, fast_math=False)
    assert ast.dump(ast.Module(body=exact, type_ignores=[])) == ast.dump(
        ast.parse(source)
    )
    body = hoist_loop_invariant_recips(ast.parse(source).body, fast_math=True)
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
            cute_region_fission=True,
            cute_materialize_transformed_operands=True,
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
    # The distributed scale (and so its contraction) needs fast_math.
    scaled = sum(
        isinstance(node, ast.Name) and node.id.startswith("_helion_scaled_")
        for node in ast.walk(original)
    )
    assert bool(scaled) is fast_math
    if fast_math:
        assert _count_fmas(original.body) >= 2
        assert _count_fmas(functions["_helion_row_softmax_bounded"].body) >= 2
    else:
        assert _count_fmas(original.body) == _count_fmas(
            functions["_helion_row_softmax_bounded"].body
        )
    assert ("cute.arch.load(" in scalar) is aligned
    assert ("cute.arch.load(" in generated) is aligned


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("fast_math", "dtype", "layout"),
    pairwise(
        fast_math=(False, True),
        dtype=(torch.float16, torch.bfloat16, torch.float32),
        layout=("dense", "unaligned"),
    ),
)
def test_scaled_sub_fusion_exact_masked_and_vector_replays(fast_math, dtype, layout):
    case = CacheCase(
        f"scaled-fusion-{dtype}-{layout}-fast{int(fast_math)}",
        (17, 513),
        dtype,
        False,
        4096,
        128,
        4 if dtype == torch.float32 else 8,
        layout=layout,
    )
    run_guarded_cache_case(case, fast_math=fast_math)


# One call's result is the product's left operand, so that call alone
# decides whether ``t = a * e; out = t + e`` contracts (``e`` is fp32).
_MATH_RESULT = """
i32 = cutlass.Int32(i)
j32 = cutlass.Int32(j)
f32 = cutlass.Float32(x)
g32 = cutlass.Float32(y)
e = cutlass.Float32(z)
a = {call}
t = a * e
out = t + e
"""


# One case per function and operand type: each function is typed by its own
# rule, and a misclassified one must fail on its own.
@pytest.mark.parametrize(
    ("call", "contracted"),
    [
        # Integer operands give an integer, also passed by keyword.
        ("cute.math.max(i32, j32)", False),
        ("cute.math.min(x=i32, y=j32, propagate_nan=True)", False),
        ("cute.math.clamp(i32, j32, j32)", False),
        ("cute.math.abs(i32)", False),
        ("cute.math.absf(x=i32)", False),
        # fp32 operands give fp32.
        ("cute.math.max(f32, g32)", True),
        ("cute.math.min(x=f32, y=g32, propagate_nan=True)", True),
        ("cute.math.clamp(f32, 0.0, 1.0)", True),
        ("cute.math.absf(x=f32)", True),
        # 16-bit operands give a 16-bit value (a bf16 relu's max).
        ("cute.math.max(cutlass.BFloat16(x), cutlass.BFloat16(y))", False),
        # Integer and bool results, whatever the operands.
        ("cute.math.absi(i32)", False),
        ("cute.math.ipowi(i32, j32)", False),
        ("cute.math.isnan(f32)", False),
        ("cute.math.isfinite(f32)", False),
        # A float-only call returns its operands' width.
        ("cute.math.exp(f32)", True),
        ("cute.math.exp(cutlass.Float64(x))", False),
        ("cute.math.rsqrt(-cutlass.BFloat16(x), fastmath=True)", False),
        ("cute.math.pow(base=cutlass.Float16(x), exponent=g32)", False),
        # fpowi's base alone: its exponent is an integer.
        ("cute.math.fpowi(f32, i32)", True),
        ("cute.math.fpowi(base=cutlass.Float64(x), exponent=i32)", False),
    ],
)
def test_math_call_results_contract_by_their_type(call, contracted):
    """``max``, ``min``, ``clamp``, ``abs`` and ``absf`` return their
    operands' type, ``absi``, ``ipowi`` and the ``isnan``-style tests an
    integer or a bool, and a float-only call such as ``exp`` a Float64 or a
    16-bit value for such an operand: ``math.fma`` takes one type, so only
    fp32 results contract with the fp32 ``e``."""
    source = _MATH_RESULT.format(call=call)
    assert _count_fmas(fuse_fma(ast.parse(source).body)) == int(contracted)


@pytest.mark.parametrize(
    "source",
    [
        """
v_0 = x_load[idx]
v_1 = cutlass.Float32(y)
d = cute.math.div(v_0, v_1, approx=True, ftz=True)
e = cutlass.Float32(z)
t = d * e
out = t + e
""",
        """
v_1 = cutlass.Float32(x)
e = cutlass.Float32(y)
m = cute.math.max(v_1 * e, 0.0)
t = m * e
out = t + e
""",
        """
p = cutlass.Float32(x)
q = cutlass.Float32(y)
a = cute.math.max(p, -1.0)
t = a * q
out = t + p
""",
    ],
    ids=["float_only_div", "arithmetic_operand", "negated_operand"],
)
def test_fp32_math_calls_still_contract(source):
    """A float-only call such as ``cute.math.div`` stays fp32 unless an
    operand is proven not to be (``v_0``, a load, is not), and an operand of
    a polymorphic call is fp32 as an expression is: arithmetic of fp32
    values, or one negated, such as ``-1.0``."""
    assert _count_fmas(fuse_fma(ast.parse(source).body)) == 1


@pytest.mark.parametrize(
    "source",
    [
        """
v_0 = cutlass.Float64(x)
a = cute.math.tanh(v_0)
e = cutlass.Float32(y)
t = a * e
out = t + e
""",
        """
v_0 = cutlass.Float64(x)
v_2 = cute.math.max(cutlass.Float64(v_0), cutlass.Float64(0.0), propagate_nan=True)
a = cute.math.sqrt(v_2)
e = cutlass.Float32(y)
t = a * e
out = t + e
""",
        """
v = cutlass.Float64(x) if c else cutlass.Float64(y)
a = cute.math.tanh(v)
e = cutlass.Float32(z)
t = a * e
out = t + e
""",
    ],
    ids=["cast_in_a_temporary", "declined_call_in_a_temporary", "selected_casts"],
)
def test_non_fp32_values_stay_so_through_names(source):
    """Codegen casts into a temporary and reads it by name, a call the pass
    proves no fp32 value feeds the next one by name, and ``torch.where``
    selects between two casts: no ``a`` is fp32, so no product contracts
    with the fp32 ``e``."""
    assert _count_fmas(fuse_fma(ast.parse(source).body)) == 0


def test_a_subscript_target_names_nothing_non_fp32():
    """A store of a float64 value into ``buf[k]`` assigns no name: ``k``
    stays unproven, so the fp32 ``exp`` it feeds contracts."""
    source = """
buf[k] = cutlass.Float64(x)
f = cutlass.Float32(y)
a = cute.math.exp(f * k)
e = cutlass.Float32(z)
t = a * e
out = t + e
"""
    assert _count_fmas(fuse_fma(ast.parse(source).body)) == 1


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _float64_affine(x: torch.Tensor, relu_first: hl.constexpr) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    for tile in hl.tile(x.size(0)):
        wide = x[tile].to(torch.float64)
        if relu_first:
            value = torch.sqrt(torch.relu(wide))
        else:
            value = torch.tanh(wide)
        out[tile] = value * 2.0 + 1.0
    return out


@pytest.mark.parametrize("relu_first", [False, True], ids=["tanh", "sqrt_relu"])
def test_float64_math_results_do_not_contract_with_fp32_constants(relu_first):
    """``tanh`` and ``sqrt`` of a float64 cast return float64 values, held in
    temporaries: ``math.fma`` takes one type, so neither contracts with the
    fp32 constants (the DSL rejects ``fma(Float64, Float32, Float32)``)."""
    with cpu_render_target():
        bound = cpu_bind(_float64_affine, (torch.empty(1024), relu_first))
        code = bound.to_code(bound.config_spec.default_config())
    assert "cute.math.fma" not in code
