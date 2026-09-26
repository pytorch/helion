"""Admission of explicit FP32 FMA in the existing packed cluster exchange."""

from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.cluster_online_pair import fuse_cluster_online_pair

_SOURCE = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
max_buffer = cute.arch.alloc_smem(cutlass.Float32, 2)
max_barrier = cute.arch.alloc_smem(cutlass.Int64, 1)
sum_buffer = cute.arch.alloc_smem(cutlass.Float32, 2)
sum_barrier = cute.arch.alloc_smem(cutlass.Int64, 1)
mi = cutlass.Float32(float('-inf'))
for tile in range(0, 8, 8):
    max_acc = cutlass.Float32(float('-inf'))
    for lane in cutlass.range_constexpr(8):
        max_acc = cute.arch.fmax(max_acc, cutlass.Float32(cache[lane]))
    local_max = _cute_grouped_reduce_cluster(max_acc, 'max', cutlass.Float32(float('-inf')), lane_id, max_buffer, max_barrier, group_span=32, cluster_n=2)
    mi = cutlass.Float32(local_max)
scaled0 = mi * 1.25
for tile in range(0, 8, 8):
    sum_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = cutlass.Float32(cache[lane])
        exponential = EXPRESSION_B
        sum_acc = sum_acc + cutlass.Float32(EXPRESSION_B)
    total = _cute_grouped_reduce_cluster(sum_acc, 'sum', cutlass.Float32(0), lane_id, sum_buffer, sum_barrier, group_span=32, cluster_n=2)
scaled1 = mi * 1.25
inverse = 1.0 / total
for tile in range(0, 8, 8):
    for lane in cutlass.range_constexpr(8):
        value = cutlass.Float32(cache[lane])
        exponential = EXPRESSION_C
        result = exponential * inverse
        out[lane] = result
"""

_FUSED = "cute.math.exp2(cute.math.fma(value, 1.25, -SCALED))"
_UNFUSED = "cute.math.exp2(value * 1.25 - SCALED)"


def _body(sum_expression: str, output_expression: str | None = None) -> list[ast.stmt]:
    output_expression = (
        sum_expression if output_expression is None else output_expression
    )
    return ast.parse(
        _SOURCE.replace(
            "EXPRESSION_B", sum_expression.replace("SCALED", "scaled0")
        ).replace("EXPRESSION_C", output_expression.replace("SCALED", "scaled1"))
    ).body


def _source(body: list[ast.stmt]) -> str:
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


@pytest.mark.parametrize("expression", [_FUSED, _UNFUSED], ids=["fma", "subtraction"])
@pytest.mark.parametrize("fast_math", [False, True])
def test_pair_preserves_the_matched_arithmetic_and_frame(
    expression: str, fast_math: bool
) -> None:
    body = _body(expression)
    before = _source(body)
    rewritten = fuse_cluster_online_pair(body, {}, fast_math=fast_math)
    after = _source(rewritten)
    assert after.count("_cute_grouped_reduce_cluster_online_pair(") == 1
    assert "_cute_grouped_reduce_cluster(" not in after
    assert "_cute_grouped_reduce_block(" in after
    assert "mi = _pair_gmax_0" in after
    assert "_pair_negative_inf_0" in after
    assert "_pair_rescale_0" in after
    assert f"fastmath={fast_math}" in after
    expected_exp = expression.replace("SCALED", "scaled0")
    assert before.count(expected_exp) == 2
    assert after.count(expected_exp) == 1
    assert expression.replace("SCALED", "scaled1") not in after
    assert after.count("cute.math.fma(") == (1 if expression == _FUSED else 0)


@pytest.mark.parametrize("dtype", ["Float16", "BFloat16"])
def test_raw_half_cache_keeps_fp32_fma_after_bitcast(dtype: str) -> None:
    body = ast.parse(
        _source(_body(_FUSED)).replace(
            "cutlass.Float32(cache[lane])",
            f"cutlass.Float32(cutlass.Uint16(cache[lane]).bitcast(cutlass.{dtype}))",
        )
    ).body
    after = _source(fuse_cluster_online_pair(body, {}))
    assert after.count("_cute_grouped_reduce_cluster_online_pair(") == 1
    assert _FUSED.replace("SCALED", "scaled0") in after


@pytest.mark.parametrize("reverse", [False, True])
def test_different_rounding_between_sweeps_keeps_two_exchanges(reverse: bool) -> None:
    pair = (_UNFUSED, _FUSED) if reverse else (_FUSED, _UNFUSED)
    body = _body(*pair)
    before = _source(body)
    assert _source(fuse_cluster_online_pair(body, {})) == before


@pytest.mark.parametrize(
    "fma",
    [
        "cute.math.fma(value, 1.25, SCALED)",
        "cute.math.fma(value, 1.25, -cutlass.Float32(SCALED))",
        "cute.math.fma(value, 1.25, -(SCALED + 1.0))",
        "cute.math.fma(value, 1.5, -SCALED)",
        "cute.math.fma(value, 1, -SCALED)",
        "cute.math.fma(value, factor, -SCALED)",
        "cute.math.fma(value, -1.25, -SCALED)",
        "cute.math.fma(value, 0.0, -SCALED)",
        "cute.math.fma(value, 1e999, -SCALED)",
        "cute.math.fma(value, 1.25, -SCALED, fastmath=True)",
        "cute.math.fma(value, 1.25, -SCALED, rounding=mode)",
        "cute.math.fma(value, 1.25, -SCALED, True)",
        "other.math.fma(value, 1.25, -SCALED)",
    ],
)
def test_unmodelled_fma_keeps_original_exchanges(fma: str) -> None:
    body = _body(f"cute.math.exp2({fma})")
    before = _source(body)
    assert _source(fuse_cluster_online_pair(body, {})) == before


@pytest.mark.parametrize(
    "value",
    [
        "cache[lane]",
        "cutlass.Float16(cache[lane])",
        "cutlass.Float32(opaque(cache[lane]))",
        "cutlass.Float32(cache[opaque(lane)])",
        "cutlass.Float32(cache[(lane := lane + 1)])",
        "cutlass.Float32(cache[lane + SCALED])",
        "cutlass.Float32(cache[lane] + SCALED)",
        "cutlass.Float32(cache[lane] + cache[lane + 1])",
    ],
)
def test_new_fma_path_requires_one_fp32_cache_read(value: str) -> None:
    body = ast.parse(
        _source(_body(_FUSED))
        .replace("value = cutlass.Float32(cache[lane])", f"value = {value}", 1)
        .replace("SCALED", "scaled0")
    ).body
    before = _source(body)
    assert _source(fuse_cluster_online_pair(body, {})) == before


def test_scaled_max_hidden_by_a_local_index_alias_keeps_original_exchanges() -> None:
    body = ast.parse(
        _source(_body(_FUSED)).replace(
            "value = cutlass.Float32(cache[lane])",
            "index = lane + scaled0\n        value = cutlass.Float32(cache[index])",
            1,
        )
    ).body
    before = _source(body)
    assert _source(fuse_cluster_online_pair(body, {})) == before
