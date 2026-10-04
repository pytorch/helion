from __future__ import annotations

import ast

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_unroll import _cached_vector_inputs
from .test_cute_chained_residency_integration import _loop_cached
from .test_cute_shared_prefill import (
    test_shared_prefill_gpu_preserves_source_and_ragged_state as _check_shared_prefill,
)
import helion
from helion import exc
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_mma_selection import validate_warp_mma_selection
from helion._compiler.cute.chained_mma_selection import warp_mma_shape
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_loop(left, right, other, feedback, initial, steps: int):
    steps = hl.specialize(steps)
    m = left.size(1)
    n = feedback.size(-1)
    history = torch.empty((max(steps, 1), m, n), device=left.device)
    output = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[32, 48]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(left.size(-1))
            pp = hl.arange(feedback.size(-2))
            a = hl.load(left, [step.id, rows, kk], extra_mask=kk[None, :] % 3 != 1)
            first = hl.dot(a, right[step.id, kk, pp], out_dtype=torch.float32)
            second = hl.dot(a, other[step.id, kk, cols], out_dtype=torch.float32)
            state = hl.dot(
                first.to(left.dtype),
                feedback[step.id, pp, cols],
                acc=state * 0.5 + second,
                out_dtype=torch.float32,
            )
            history[step.id, rows, cols] = state
        output[rows, cols] = state
    return history, output


def _mixed_inputs(device, dtype, steps):
    torch.manual_seed(947)
    count = max(steps, 1)
    return (
        *(
            torch.randn((count, m, n), device=device, dtype=dtype) * 0.125
            for m, n in ((29, 48), (48, 48), (48, 37), (48, 37))
        ),
        torch.randn((29, 37), device=device) * 0.125,
        steps,
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_group(left, other, right, initial):
    steps, _, k = left.shape
    n = hl.specialize(right.size(-1))
    history = torch.empty((steps, 16, n), device=left.device)
    output = torch.empty_like(initial)
    for rr, cols in hl.tile([32, n], block_size=[32, n]):
        state = initial[rr, cols]
        for step in hl.tile(steps, block_size=1):
            mm, kk = hl.arange(16), hl.arange(k)
            b = right[step.id, kk, cols]
            first = hl.dot(left[step.id, mm, kk], b, out_dtype=torch.float32)
            second = hl.dot(other[step.id, rr, kk], b, out_dtype=torch.float32)
            state = state * 0.5 + second
            history[step.id, mm, cols] = first
        output[rr, cols] = state
    return history, output


def _transposed_inputs(device, n):
    torch.manual_seed(948)
    return (
        *(
            torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.125
            for shape in ((3, 16, 16), (3, 32, 16), (3, 16, n))
        ),
        torch.randn((32, n), device=device) * 0.125,
    )


@pytest.mark.parametrize("n", [8, 24])
def test_transposed_group_rounds_physical_rows_to_warp_atom_cpu(n):
    with _cpu_codegen():
        bound = _transposed_group._bind_isolated(_transposed_inputs("cpu", n))
        source = bound.to_code(_config(32, cute_chained_scratch_layout="row_major"))
    rows = (n + 15) // 16 * 16
    assert f"chain_0_warp_a = cute.local_tile(chain_0_a, ({rows}, 16)" in source
    assert "chain_1_warp_mma" not in source
    assert "chain_1_c" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("n", [8, 24])
def test_transposed_group_rounding_preserves_padded_domain_gpu(n):
    args = _transposed_inputs(DEVICE, n)
    saved = tuple(arg.clone() for arg in args)
    bound = _transposed_group._bind_isolated(args)
    ordinary = bound.compile_config(_config(0, cute_chained_scratch_layout="row_major"))
    mixed = bound.compile_config(_config(32, cute_chained_scratch_layout="row_major"))
    expected, actual = ordinary(*args), mixed(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(mixed(*args), actual, rtol=0, atol=0)
    torch.testing.assert_close(args, saved, rtol=0, atol=0)


def _config(rows=32, grouped=True, **overrides):
    return helion.Config.from_dict(
        {
            "num_warps": 16,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_group_contractions": grouped,
            "cute_chained_warp_mma_rows": rows,
            "cute_chained_scratch_layout": "xor",
            "cute_chained_pointwise_unroll": 8,
            **overrides,
        }
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_group_shape_trims_only_physical_row_padding(transpose):
    first = StageGeometry((16, 32, 48), transpose)
    second = StageGeometry((16, 64, 48), transpose)
    group = ContractionGroup((0, 1), (first, second))
    assert warp_mma_shape(first) == ((32, 16, 48) if transpose else (16, 32, 48))
    assert warp_mma_shape(first, group) == ((64, 32, 48) if transpose else (16, 96, 48))


@pytest.mark.parametrize("value", [True, False, None, "32", 0.0, 1, 16, 32, 64, 128])
def test_nonzero_or_malformed_request_requires_actual_stage(value):
    with pytest.raises(exc.BackendUnsupported, match="eligible common TCgen05"):
        validate_warp_mma_selection(None, value)
    validate_warp_mma_selection(None, 0)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mixed_engine_preserves_producers_and_initialized_stage_cpu(grouped, dtype):
    with _cpu_codegen():
        bound = _mixed_loop._bind_isolated(_mixed_inputs("cpu", dtype, 3))
        ordinary = bound.to_code(_config(0, grouped))
        mixed = bound.to_code(_config(32, grouped))
        with pytest.raises(exc.BackendUnsupported, match="eligible common TCgen05"):
            bound.to_code(_config(16, grouped))
    assert "chain_0_warp_mma" in mixed
    first_mma = next(
        node.value
        for node in ast.walk(ast.parse(mixed))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "chain_0_warp_mma"
            for target in node.targets
        )
    )
    assert ast.literal_eval(first_mma.keywords[0].value) == (1, 2 if grouped else 8, 1)
    assert ("chain_1_warp_mma" in mixed) is not grouped
    assert "chain_2_warp_mma" not in mixed
    assert "chain_2_seed" in mixed
    assert "tcgen05.commit(chain_bars + 2)" in mixed
    for operation in ("cute.arch.sync_threads", "cute.arch.alloc_smem"):

        def calls(source, operation=operation):
            return [
                ast.dump(node)
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call) and ast.unparse(node.func) == operation
            ]

        assert calls(ordinary) == calls(mixed)

    def producers(source):
        return [
            ast.dump(node)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id.endswith("_step")
            and node.target.id.startswith("chain_")
            and ("_a_" in node.target.id or "_b_" in node.target.id)
        ]

    assert producers(ordinary) == producers(mixed)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mixed_engine_grouped_ragged_gpu(grouped, dtype):
    args = _mixed_inputs(DEVICE, dtype, 3)
    saved = tuple(arg.clone() for arg in args[:-1])
    bound = _mixed_loop._bind_isolated(args)
    ordinary = bound.compile_config(_config(0, grouped))
    mixed = bound.compile_config(_config(32, grouped))
    expected, actual = ordinary(*args), mixed(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    left, right, other, feedback, state, steps = (*saved, 3)
    state = state.clone()
    mask = torch.arange(48, device=DEVICE) % 3 != 1
    for step in range(steps):
        # Retain native operand precision with FP32 accumulation. Replacing
        # the first dot by an IEEE FP32 GEMM changes its reduction rounding
        # just before the explicit FP16 cast at halfway values.
        a = torch.where(mask, left[step], 0)
        first = torch.mm(a, right[step], out_dtype=torch.float32).to(dtype)
        second = torch.mm(a, other[step], out_dtype=torch.float32)
        state = (
            state * 0.5
            + second
            + torch.mm(first, feedback[step], out_dtype=torch.float32)
        )
        torch.testing.assert_close(actual[0][step], state, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(mixed(*args), actual, rtol=0, atol=0)
    torch.testing.assert_close(args[:-1], saved, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("steps", [0, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mixed_engine_with_scan_cache_vector_and_zero_trip_gpu(steps, dtype):
    args = _cached_vector_inputs(DEVICE, dtype, steps)
    saved = tuple(arg.clone() for arg in args[:-1])
    bound = _loop_cached._bind_isolated(args)
    config = _config(
        0,
        False,
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
    )
    ordinary = bound.compile_config(config)
    mixed_config = helion.Config.from_dict(
        config.config | {"cute_chained_warp_mma_rows": 32}
    )
    mixed = bound.compile_config(mixed_config)
    expected, actual = ordinary(*args), mixed(*args)
    torch.testing.assert_close(
        actual if steps else actual[1],
        expected if steps else expected[1],
        atol=0,
        rtol=0,
    )
    replay = mixed(*args)
    torch.testing.assert_close(
        replay if steps else replay[1], actual if steps else actual[1], atol=0, rtol=0
    )
    torch.testing.assert_close(args[:-1], saved, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
@pytest.mark.parametrize("value_tile", [64, 128])
def test_mixed_full_prefill_source_reference_gpu(bt32, value_tile):
    _check_shared_prefill(bt32, value_tile, 4096, warp_rows=32)
