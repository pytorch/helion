from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.persistent_branch_vec import (
    vectorize_branch_local_persistent_fragments,
)


def fixture(*, suffix="", prefix="", load="(slots.iterator + row).load()", marker=True):
    ptr = "state.iterator + cutlass.Int32(slot) * cutlass.Int32(state.layout.stride[0]) + lane"
    scalar = f"({ptr}).load() if cutlass.Int32(slot) < 64 and lane < 128 else cutlass.BFloat16(0)"
    value = (
        f"_helion_persistent_branch_vec_load(3, 4, 'cutlass.BFloat16', '', {ptr}, {scalar})"
        if marker
        else scalar
    )
    return ast.parse(f"""
cache = cute.make_rmem_tensor(4, cutlass.Int32)
{prefix}
for synthetic_lane_3 in cutlass.range_constexpr(4):
    lane = base + cutlass.Int32(synthetic_lane_3)
    loaded_slot = {load}
    cache[synthetic_lane_3] = loaded_slot
    slot = cutlass.Int64(loaded_slot)
    value = {value}
    sink = cutlass.Float32(value) * gain
{suffix}
""").body


def transform(body, pairs=()):
    return vectorize_branch_local_persistent_fragments(
        body,
        proven_disjoint_tensor_pairs=set(map(frozenset, pairs)),
        proven_tensor_stride_values={("state", 0): 128},
    )


def test_exact_hoist_cast_mask_and_packet():
    original = fixture()
    load = ast.dump(original[-1].body[1], include_attributes=False)
    cast = ast.dump(original[-1].body[3], include_attributes=False)
    result = transform(original)
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    loop = next(s for s in result if isinstance(s, ast.For))
    assert ast.dump(result[1], include_attributes=False) == load
    assert ast.dump(result[2], include_attributes=False) == cast
    assert "ir.VectorType.get([4], cutlass.Uint16.mlir_type)" in code
    assert "cutlass.Int32(slot) < 64" in code and ">= 0" not in code
    assert "else cutlass.BFloat16(0)" in ast.unparse(loop)
    assert "slots.iterator" not in ast.unparse(loop)
    assert "cache[synthetic_lane_3] = loaded_slot" in ast.unparse(loop)


@pytest.mark.parametrize(
    "suffix,pairs,allowed",
    [
        ("    (slots.iterator + row).store(replacement)", (), False),
        ("    (other.iterator + row).store(replacement)", (), False),
        (
            "    (other.iterator + row).store(replacement)",
            (("slots", "other"), ("state", "other")),
            True,
        ),
        ("    loaded_slot = replacement", (), False),
        ("    loaded_slot.value = replacement", (), False),
        ("    unknown(loaded_slot)", (), False),
        ("    cute.arch.barrier()", (), False),
        ("    alias[0] = replacement", (), False),
    ],
)
def test_effects(suffix, pairs, allowed):
    code = ast.unparse(
        ast.Module(body=transform(fixture(suffix=suffix), pairs), type_ignores=[])
    )
    assert ("cute.arch.load(state.iterator" in code) is allowed


@pytest.mark.parametrize(
    "load",
    [
        "(slots.iterator + synthetic_lane_3).load()",
        "(slots.iterator + lane).load()",
        "(slots.iterator + row).load() if synthetic_lane_3 < 2 else cutlass.Int32(0)",
        "(slots.iterator + row).load() if flag else cutlass.Int32(0)",
    ],
)
def test_varying_or_masked_prefix_remains_scalar(load):
    result = transform(fixture(load=load))
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "cute.arch.load(state.iterator" not in code
    assert "(slots.iterator" in ast.unparse(
        next(s for s in result if isinstance(s, ast.For))
    )


@pytest.mark.parametrize("prefix", ["alias = cache", "consume(cache)", "cache = other"])
def test_register_cache_alias_rejects(prefix):
    code = ast.unparse(
        ast.Module(body=transform(fixture(prefix=prefix)), type_ignores=[])
    )
    assert "cute.arch.load(state.iterator" not in code


def test_unmarked_alignment_or_tail_decline_stays_scalar():
    before = fixture(marker=False)
    original = ast.dump(
        ast.Module(body=before, type_ignores=[]), include_attributes=False
    )
    after = transform(before)
    assert (
        ast.dump(ast.Module(body=after, type_ignores=[]), include_attributes=False)
        == original
    )


@pytest.mark.parametrize(
    "expression",
    [
        "cute.math.rsqrt(gain)",
        "cute.math.exp2(gain)",
        "cute.math.log(gain)",
        "cute.math.min(gain, other, propagate_nan=True)",
        "cute.math.rcp(gain, approx=True, ftz=True)",
        "operator.gt(gain, other)",
    ],
)
def test_known_scalar_math_does_not_block_readonly_address(expression):
    body = fixture(suffix="    math_result = " + expression)
    original_math = ast.dump(body[-1].body[-1], include_attributes=False)
    result = transform(body)
    loop = next(stmt for stmt in result if isinstance(stmt, ast.For))
    assert ast.dump(loop.body[-1], include_attributes=False) == original_math
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "cute.arch.load(state.iterator" in code
    assert "cutlass.Int32(slot) < 64" in code


@pytest.mark.parametrize(
    "expression",
    [
        "cute.math.unknown(gain)",
        "cute.arch.barrier()",
        "cute.arch.warp_reduction_sum(gain)",
        "unknown(gain)",
        "cute.math.rsqrt(unknown(gain))",
    ],
)
def test_unknown_or_synchronizing_calls_still_decline(expression):
    code = ast.unparse(
        ast.Module(
            body=transform(fixture(suffix="    ignored = " + expression)),
            type_ignores=[],
        )
    )
    assert "cute.arch.load(state.iterator" not in code


def test_math_does_not_override_alias_or_mask_proof():
    for body in (
        fixture(
            suffix="    math_value = cute.math.rsqrt(gain)\n    (slots.iterator + row).store(other)"
        ),
        fixture(
            load="(slots.iterator + row).load() if flag else cutlass.Int32(0)",
            suffix="    math_value = cute.math.rsqrt(gain)",
        ),
        fixture(
            load="(slots.iterator + synthetic_lane_3).load()",
            suffix="    math_value = cute.math.rsqrt(gain)",
        ),
    ):
        code = ast.unparse(ast.Module(body=transform(body), type_ignores=[]))
        assert "cute.arch.load(state.iterator" not in code
