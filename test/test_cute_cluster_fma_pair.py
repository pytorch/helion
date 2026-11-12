"""Admission of explicit FP32 FMA in the existing packed cluster exchange."""

from __future__ import annotations

import ast
from functools import partial

import pytest
import torch

from test._cute_kernels import softmax_two_pass_kernel

import helion
from helion._compiler.cute.cluster_online_pair import fuse_cluster_online_pair
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.benchmarking import _make_cudagraph_replay

pytestmark = pytest.mark.cpu_only

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
def test_cached_pair_needs_fast_math(expression: str) -> None:
    """Rescaling cached exponentials rounds each output differently, so it
    needs fast_math; the exact pair does not match a distributed scale."""
    body = _body(expression)
    before = _source(body)
    assert _source(fuse_cluster_online_pair(body, {}, fast_math=False)) == before


@pytest.mark.parametrize("expression", [_FUSED, _UNFUSED], ids=["fma", "subtraction"])
def test_pair_preserves_the_matched_arithmetic_and_frame(expression: str) -> None:
    body = _body(expression)
    before = _source(body)
    after = _source(fuse_cluster_online_pair(body, {}, fast_math=True))
    assert after.count("_cute_grouped_reduce_cluster_online_pair(") == 1
    assert "_cute_grouped_reduce_cluster(" not in after
    assert "_cute_grouped_reduce_block(" in after
    assert "mi = _pair_gmax_0" in after
    assert "_pair_negative_inf_0" in after
    assert "_pair_rescale_0" in after
    assert "fastmath=True" in after
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
    after = _source(fuse_cluster_online_pair(body, {}, fast_math=True))
    assert after.count("_cute_grouped_reduce_cluster_online_pair(") == 1
    assert _FUSED.replace("SCALED", "scaled0") in after


@pytest.mark.parametrize("reverse", [False, True])
def test_different_rounding_between_sweeps_keeps_two_exchanges(reverse: bool) -> None:
    pair = (_UNFUSED, _FUSED) if reverse else (_FUSED, _UNFUSED)
    body = _body(*pair)
    before = _source(body)
    assert _source(fuse_cluster_online_pair(body, {}, fast_math=True)) == before


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
    assert _source(fuse_cluster_online_pair(body, {}, fast_math=True)) == before


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
    assert _source(fuse_cluster_online_pair(body, {}, fast_math=True)) == before


def test_scaled_max_hidden_by_a_local_index_alias_keeps_original_exchanges() -> None:
    body = ast.parse(
        _source(_body(_FUSED)).replace(
            "value = cutlass.Float32(cache[lane])",
            "index = lane + scaled0\n        value = cutlass.Float32(cache[index])",
            1,
        )
    ).body
    before = _source(body)
    assert _source(fuse_cluster_online_pair(body, {}, fast_math=True)) == before


# Runtime composition of scaled FP32 FMA and the packed cluster exchange.


CUDA_DEVICE = "cuda"


def _fill_cluster_rows(x: torch.Tensor, cluster: int, generation: int) -> None:
    columns = x.shape[1]
    segment = columns // cluster
    x.normal_()
    x[1:5].fill_(-torch.inf)
    finite_segment = generation % cluster
    x[1, finite_segment * segment : (finite_segment + 1) * segment] = -1000.0
    x[2, :segment] = -1.0
    x[4, -1] = torch.inf
    x[5, generation % columns] = torch.nan
    x[6, columns - 1 - generation % columns] = torch.nan
    x[7].zero_()
    x[7, ::2] = -0.0


def _check_cluster_rows(actual: torch.Tensor, x: torch.Tensor) -> None:
    expected = x.float().softmax(dim=1).to(x.dtype)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3, equal_nan=True)
    # Empty CTA slices of otherwise finite rows must contribute exact zeros.
    empty = torch.isneginf(x[1:3])
    assert torch.equal(actual[1:3][empty], torch.zeros_like(actual[1:3][empty]))


@pytest.mark.cuda
@skipUnlessCuteAvailable("requires the supported CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize(
    "dtype,columns,cluster", [(torch.bfloat16, 32768, 2), (torch.float16, 65536, 4)]
)
def test_cluster_fma_frames_fresh_pointers_and_mutated_graphs(
    dtype: torch.dtype, columns: int, cluster: int, fast_math: bool
) -> None:
    torch.manual_seed(20260913)
    kernel = helion.kernel(
        softmax_two_pass_kernel.fn,
        backend="cute",
        static_shapes=True,
        fast_math=fast_math,
        autotune_effort="none",
    )
    config = helion.Config(
        block_sizes=[1, columns],
        num_threads=[0, 256],
        cute_vector_widths=[1, 8],
        cute_lane_layouts=["blocked", "strided"],
        cute_cluster_n=cluster,
    )
    retained: list[torch.Tensor] = []
    for replacement in range(2):
        x = torch.empty((8, columns), dtype=dtype, device=CUDA_DEVICE)
        if retained:
            assert x.data_ptr() != retained[0].data_ptr()
        retained.append(x)
        _fill_cluster_rows(x, cluster, replacement)
        bound = kernel.bind((x,))
        source = bound.to_code(config)
        assert source.count("_cute_grouped_reduce_cluster_online_pair(") == 1
        assert "_cute_grouped_reduce_cluster(" not in source
        assert f"fastmath={fast_math}" in source
        assert source.index("operator.eq(mi,") > source.index("mi = _pair_gmax_0")
        if fast_math:
            assert "cute.math.exp2(cute.math.fma(" in source
            assert "_pair_negative_inf_0" in source and "_pair_rescale_0" in source
        else:
            # Exact exponentials: no distributed scale, no cached and
            # rescaled values, sweep B in the zero-guarded CTA frame.
            assert "_pair_frame_0" in source
            for absent in ("_helion_scaled_", "_pair_exp_cache_", "_pair_rescale_"):
                assert absent not in source
        bound.set_config(config)

        invoke = partial(bound, x)
        _check_cluster_rows(invoke(), x)
        replay = _make_cudagraph_replay(invoke)
        output = replay()
        for mutation in range(4):
            _fill_cluster_rows(x, cluster, replacement * 4 + mutation)
            snapshot = x.view(torch.uint8).clone()
            _check_cluster_rows(invoke(), x)
            output.fill_(torch.nan)
            actual = replay()
            _check_cluster_rows(actual, x)
            assert torch.equal(x.view(torch.uint8), snapshot)
