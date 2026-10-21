"""GPU numerics of the register-tile schedule knobs (see
``test_cute_register_tile_schedule.py`` for the codegen shape)."""

from __future__ import annotations

from typing import Any
from typing import Callable

from examples.linear.linear_attention_engine import recurrent_step_correction_fused
from examples.linear.linear_attention_engine import recurrent_step_fused
import pytest
import torch

from test._cute_register_tile_kernels import col_scale
from test._cute_register_tile_kernels import col_stats
from test._cute_register_tile_kernels import col_sum
from test._cute_register_tile_kernels import column_config

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

BH, D, DV = 256, 128, 128
COLUMN_REDUCE = "_cute_grouped_reduce_shared_columns("
WARP_FOLD = "fold='warp'"
STATE_STORE = "_cute_store_u32_vec(state.iterator"
KNOBS: dict[str, Any] = {
    "cute_register_tile_warp_fold": True,
    "cute_register_tile_store_first": True,
}


def _config(
    bound: object,
    *,
    dv_block: int,
    dv_threads: int,
    reduction_threads: int,
    vec: int = 4,
    **extra: Any,
) -> helion.Config:
    spec = bound.config_spec  # type: ignore[attr-defined]
    env = bound.env  # type: ignore[attr-defined]
    (reduction,) = [block.block_id for block in env.block_sizes if block.reduction]
    (dv,) = spec.block_sizes.valid_block_ids()
    threads = {dv: dv_threads, reduction: reduction_threads}
    return spec.normalized_config(
        helion.Config(
            block_sizes=[dv_block],
            num_threads=[
                threads.get(block_id, 0)
                for block_id in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[None],
            cute_vector_widths=[
                vec if block_id == dv else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            **extra,
        )
    )


def _check_recurrent_step(
    *,
    dtype: torch.dtype,
    dv_block: int,
    dv_threads: int,
    reduction_threads: int,
    vec: int = 4,
    **extra: Any,
) -> str:
    """Run the fused recurrent step on two input sets (a stale output cannot
    pass twice) and return the generated code."""
    kernel = helion.kernel(
        recurrent_step_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    q = torch.randn(BH, D, device=DEVICE).to(dtype)
    k = torch.randn(BH, D, device=DEVICE).to(dtype)
    v = torch.randn(BH, DV, device=DEVICE).to(dtype)
    alpha = torch.rand(BH, device=DEVICE).to(dtype)
    initial = torch.randn(BH, D, DV, device=DEVICE).to(dtype)
    state = initial.clone()
    bound = kernel.bind((q, k, v, state, alpha))
    config = _config(
        bound,
        dv_block=dv_block,
        dv_threads=dv_threads,
        reduction_threads=reduction_threads,
        vec=vec,
        **extra,
    )
    code = bound.to_code(config)
    assert "_lane_stash" in code
    compiled = bound.compile_config(config)
    rtol, atol = (1e-3, 1e-3) if dtype is torch.float32 else (2e-2, 2e-1)
    for _ in range(2):
        out = compiled(q, k, v, state, alpha)
        torch.cuda.synchronize()
        expected_state = (
            alpha.float()[:, None, None] * initial.float()
            + k.float()[:, :, None] * v.float()[:, None, :]
        )
        expected_out = (q.float()[:, :, None] * expected_state).sum(1)
        torch.testing.assert_close(state.float(), expected_state, rtol=rtol, atol=atol)
        torch.testing.assert_close(out.float(), expected_out, rtol=rtol, atol=atol)
        q.copy_(torch.rand_like(q, dtype=torch.float32).mul_(2).sub_(1).to(dtype))
        state.copy_(initial)
    return code


@pytest.mark.parametrize(
    ("dv_block", "dv_threads", "reduction_threads", "extra"),
    (
        pytest.param(32, 8, 8, KNOBS, id="two-warps-both-knobs"),
        pytest.param(32, 8, 16, KNOBS, id="four-warps-both-knobs"),
        pytest.param(32, 8, 32, KNOBS, id="eight-warps-both-knobs"),
        pytest.param(64, 16, 8, KNOBS, id="wide-tile-both-knobs"),
        pytest.param(
            32,
            8,
            16,
            {"cute_register_tile_warp_fold": True},
            id="four-warps-warp-fold",
        ),
        pytest.param(
            32,
            8,
            16,
            {"cute_register_tile_store_first": True},
            id="four-warps-store-first",
        ),
        pytest.param(
            32,
            8,
            16,
            {**KNOBS, "pid_type": "persistent_blocked"},
            id="four-warps-both-knobs-persistent",
        ),
    ),
)
def test_schedule_knobs_match_reference(
    dv_block: int, dv_threads: int, reduction_threads: int, extra: dict[str, Any]
) -> None:
    code = _check_recurrent_step(
        dtype=torch.float32,
        dv_block=dv_block,
        dv_threads=dv_threads,
        reduction_threads=reduction_threads,
        **extra,
    )
    assert code.count(COLUMN_REDUCE) == 1
    assert (WARP_FOLD in code) == bool(extra.get("cute_register_tile_warp_fold"))
    store_first = bool(extra.get("cute_register_tile_store_first"))
    assert (code.index(STATE_STORE) < code.index(COLUMN_REDUCE)) == store_first


def test_bf16_register_tile_with_both_knobs() -> None:
    code = _check_recurrent_step(
        dtype=torch.bfloat16,
        dv_block=64,
        dv_threads=8,
        reduction_threads=16,
        vec=8,
        **KNOBS,
    )
    assert WARP_FOLD in code
    assert code.count("_cute_store_u16_vec(state.iterator") == 1


def _col_sum_reference(x: torch.Tensor) -> torch.Tensor:
    return x.sum(0)


def _col_scale_reference(x: torch.Tensor) -> torch.Tensor:
    return x / ((x * x).sum(0) + 1.0)[None, :]


def _col_stats_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x.sum(0), (x * x).sum(0)])


@pytest.mark.parametrize(
    ("kernel", "reference", "reduction_threads"),
    (
        pytest.param(col_sum, _col_sum_reference, 16, id="col-sum-four-warps"),
        pytest.param(col_sum, _col_sum_reference, 32, id="col-sum-eight-warps"),
        pytest.param(col_stats, _col_stats_reference, 16, id="col-stats-four-warps"),
        # The scale's store needs the sum: store_first is ignored, the fold is not.
        pytest.param(col_scale, _col_scale_reference, 16, id="col-scale-four-warps"),
    ),
)
def test_column_kernels_with_both_knobs_match_reference(
    kernel: helion.Kernel,
    reference: Callable[[torch.Tensor], torch.Tensor],
    reduction_threads: int,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 16384, device=DEVICE)
    bound = kernel.bind((x,))
    config = column_config(
        bound,
        block=32,
        tile_threads=8,
        reduction_threads=reduction_threads,
        **KNOBS,
    )
    code = bound.to_code(config)
    assert WARP_FOLD in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference(x), rtol=1e-4, atol=1e-3)
        x.uniform_(-1, 1)


def test_diagonal_decay_with_both_knobs_matches_reference() -> None:
    # ``alpha[idx, :]`` (a [D] row) scales the rows of the state tile: the
    # extra per-lane load joins the accumulate nest like ``k`` and ``q``.
    kernel = helion.kernel(
        recurrent_step_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    q = torch.randn(BH, D, device=DEVICE)
    k = torch.randn(BH, D, device=DEVICE)
    v = torch.randn(BH, DV, device=DEVICE)
    alpha = torch.rand(BH, D, device=DEVICE)
    initial = torch.randn(BH, D, DV, device=DEVICE)
    state = initial.clone()
    bound = kernel.bind((q, k, v, state, alpha))
    config = _config(bound, dv_block=32, dv_threads=8, reduction_threads=16, **KNOBS)
    code = bound.to_code(config)
    assert WARP_FOLD in code
    assert code.index(STATE_STORE) < code.index(COLUMN_REDUCE)
    compiled = bound.compile_config(config)
    out = compiled(q, k, v, state, alpha)
    torch.cuda.synchronize()
    expected_state = alpha[:, :, None] * initial + k[:, :, None] * v[:, None, :]
    expected_out = (q[:, :, None] * expected_state).sum(1)
    torch.testing.assert_close(state, expected_state, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out, expected_out, rtol=1e-3, atol=1e-3)


OWNER_LOOP = "for synthetic_lane_2 in cutlass.range_constexpr("


@pytest.mark.parametrize(
    ("diagonal", "extra"),
    (
        pytest.param(False, {}, id="scalar-decay-knob-free"),
        pytest.param(False, KNOBS, id="scalar-decay-both-knobs"),
        pytest.param(True, {}, id="diagonal-decay-knob-free"),
        pytest.param(True, KNOBS, id="diagonal-decay-both-knobs"),
    ),
)
def test_correction_step_chains_two_register_tile_levels(
    diagonal: bool, extra: dict[str, Any]
) -> None:
    # The delta-rule step feeds ``kts = (k * s).sum(0)`` into the state it
    # then reduces again (``o = (q * s).sum(0)``): the tile is loaded once,
    # ``kts`` is combined, a second accumulate nest applies the correction
    # from the stashed tile and accumulates ``o``.  With ``store_first`` the
    # state store joins that nest (ahead of the ``o`` combine); without it the
    # corrected tile is stashed and stored behind the last combine.
    kernel = helion.kernel(
        recurrent_step_correction_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    q = torch.randn(BH, D, device=DEVICE)
    k = torch.randn(BH, D, device=DEVICE)
    v = torch.randn(BH, DV, device=DEVICE)
    alpha = torch.rand((BH, D) if diagonal else (BH,), device=DEVICE)
    beta = torch.rand(BH, device=DEVICE)
    initial = torch.randn(BH, D, DV, device=DEVICE)
    state = initial.clone()
    bound = kernel.bind((q, k, v, state, alpha, beta))
    spec = bound.config_spec
    assert spec.cute_register_tile_reduction_blocks
    config = _config(bound, dv_block=32, dv_threads=8, reduction_threads=16, **extra)
    for key, value in extra.items():
        assert config.config[key] is value
    code = bound.to_code(config)
    assert code.count("cute.arch.load(state.iterator") == 1
    assert code.count(COLUMN_REDUCE) == 2
    assert (WARP_FOLD in code) == bool(extra.get("cute_register_tile_warp_fold"))
    first_combine = code.index(COLUMN_REDUCE)
    second_combine = code.index(COLUMN_REDUCE, first_combine + 1)
    assert first_combine < code.index(STATE_STORE)
    store_first = bool(extra.get("cute_register_tile_store_first"))
    assert (code.index(STATE_STORE) < second_combine) == store_first
    assert code.count(OWNER_LOOP) == (2 if store_first else 3)
    compiled = bound.compile_config(config)
    a = alpha[:, :, None] if diagonal else alpha[:, None, None]
    for _ in range(2):
        out = compiled(q, k, v, state, alpha, beta)
        torch.cuda.synchronize()
        decayed = a * initial
        kts = (k[:, :, None] * decayed).sum(1)
        expected_state = (
            decayed
            - beta[:, None, None] * k[:, :, None] * kts[:, None, :]
            + beta[:, None, None] * k[:, :, None] * v[:, None, :]
        )
        expected_out = (q[:, :, None] * expected_state).sum(1)
        torch.testing.assert_close(state, expected_state, rtol=1e-3, atol=1e-2)
        torch.testing.assert_close(out, expected_out, rtol=1e-3, atol=1e-2)
        q.copy_(torch.rand_like(q).mul_(2).sub_(1))
        state.copy_(initial)
