"""GPU numerics for register-tile lane reductions (see
``test_cute_register_tile_reductions.py`` for the codegen shape)."""

from __future__ import annotations

from typing import Any
from typing import Callable

from examples.linear.linear_attention_engine import recurrent_step_fused
import pytest
import torch

from test._cute_register_tile_kernels import col_scale
from test._cute_register_tile_kernels import col_scale_into
from test._cute_register_tile_kernels import col_scale_three
from test._cute_register_tile_kernels import col_softmax
from test._cute_register_tile_kernels import col_stats
from test._cute_register_tile_kernels import col_sum
from test._cute_register_tile_kernels import col_sum_dynamic
from test._cute_register_tile_kernels import col_sum_f32_out
from test._cute_register_tile_kernels import col_sum_guarded
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
    """Run the fused recurrent step with a register-tile config on two input
    sets (a stale output cannot pass twice) and return the generated code."""
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
    # The kernel computes in fp32 and rounds the state and the output once.
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
    ("dv_block", "dv_threads", "reduction_threads"),
    (
        pytest.param(32, 8, 8, id="one-warp-columns-64-elements"),
        pytest.param(32, 8, 16, id="four-warp-columns"),
        pytest.param(64, 16, 8, id="two-warp-wide-tile"),
    ),
)
def test_register_tile_matches_reference(
    dv_block: int, dv_threads: int, reduction_threads: int
) -> None:
    _check_recurrent_step(
        dtype=torch.float32,
        dv_block=dv_block,
        dv_threads=dv_threads,
        reduction_threads=reduction_threads,
    )


@pytest.mark.parametrize("pid_type", ("persistent_blocked", "persistent_interleaved"))
def test_persistent_pid_reuses_the_column_reduce_safely(pid_type: str) -> None:
    # A persistent CTA runs the column reduce once per tile with no other
    # barrier in between; the helper's trailing barrier keeps a fast warp's
    # next-tile partials from overwriting slots a slower warp still reads.
    # The race is timing dependent, so a pass here does not prove the barrier
    # is present: ``test_column_reduce_returns_behind_a_barrier`` pins it in
    # the helper's source; this test checks the persistent register tile runs
    # and its numerics.
    code = _check_recurrent_step(
        dtype=torch.float32,
        dv_block=32,
        dv_threads=8,
        reduction_threads=16,
        pid_type=pid_type,
    )
    assert "for virtual_pid in range(" in code
    assert code.count(COLUMN_REDUCE) == 1


def test_bf16_register_tile_flushes_u16_packets() -> None:
    # bf16 state with V=8: one 16-byte packet per lane and a u16 flush.
    code = _check_recurrent_step(
        dtype=torch.bfloat16,
        dv_block=64,
        dv_threads=8,
        reduction_threads=16,
        vec=8,
    )
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in code
    assert code.count("_cute_store_u16_vec(state.iterator") == 1


def _col_sum_reference(x: torch.Tensor) -> torch.Tensor:
    return x.sum(0)


def _col_scale_reference(x: torch.Tensor) -> torch.Tensor:
    return x / ((x * x).sum(0) + 1.0)[None, :]


def _col_stats_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x.sum(0), (x * x).sum(0)])


@pytest.mark.parametrize(
    ("kernel", "reference", "block", "tile_threads", "reduction_threads", "extra"),
    (
        pytest.param(col_sum, _col_sum_reference, 32, 8, 8, {}, id="col-sum"),
        pytest.param(
            col_sum,
            _col_sum_reference,
            4,
            1,
            64,
            {},
            id="col-sum-warp-columns",
        ),
        pytest.param(
            col_sum,
            _col_sum_reference,
            4,
            1,
            64,
            {"pid_type": "persistent_blocked"},
            id="col-sum-warp-columns-persistent",
        ),
        pytest.param(
            col_sum,
            _col_sum_reference,
            32,
            8,
            8,
            {"load_eviction_policies": ["l2_last"]},
            id="col-sum-l2-last",
        ),
        pytest.param(
            col_sum,
            _col_sum_reference,
            32,
            8,
            8,
            {"load_eviction_policies": ["l1_l2_first"]},
            id="col-sum-l1-l2-first",
        ),
        pytest.param(col_scale, _col_scale_reference, 32, 8, 8, {}, id="col-scale"),
        pytest.param(
            col_scale,
            _col_scale_reference,
            32,
            8,
            8,
            {"pid_type": "persistent_interleaved"},
            id="col-scale-persistent",
        ),
        pytest.param(col_stats, _col_stats_reference, 32, 8, 8, {}, id="col-stats"),
    ),
)
def test_column_register_tiles_match_reference(
    kernel: helion.Kernel,
    reference: Callable[[torch.Tensor], torch.Tensor],
    block: int,
    tile_threads: int,
    reduction_threads: int,
    extra: dict[str, Any],
) -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 16384, device=DEVICE)
    bound = kernel.bind((x,))
    config = column_config(
        bound,
        block=block,
        tile_threads=tile_threads,
        reduction_threads=reduction_threads,
        **extra,
    )
    code = bound.to_code(config)
    assert COLUMN_REDUCE in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference(x), rtol=1e-4, atol=1e-3)
        x.uniform_(-1, 1)


def _col_sum_f32_reference(x: torch.Tensor) -> torch.Tensor:
    return x.float().sum(0)


def _col_scale_three_reference(
    x: torch.Tensor, u: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    r = 1.0 / ((x * x).sum(0) + 1.0)[None, :]
    return x * r + u * r + w


def _randn_like_inputs(inputs: tuple[torch.Tensor, ...]) -> None:
    for tensor in inputs:
        tensor.copy_(
            torch.rand_like(tensor, dtype=torch.float32)
            .mul_(2)
            .sub_(1)
            .to(tensor.dtype)
        )


@pytest.mark.parametrize(
    ("kernel", "reference", "dtype", "inputs", "block", "reduction_threads", "vec"),
    (
        pytest.param(
            col_sum_f32_out,
            _col_sum_f32_reference,
            torch.bfloat16,
            1,
            64,
            16,
            8,
            id="col-sum-bf16-to-fp32",
        ),
        pytest.param(
            col_scale_three,
            _col_scale_three_reference,
            torch.float32,
            3,
            32,
            8,
            4,
            id="col-scale-three-tiles",
        ),
    ),
)
def test_rolled_fallback_matches_reference(
    kernel: helion.Kernel,
    reference: Callable[..., torch.Tensor],
    dtype: torch.dtype,
    inputs: int,
    block: int,
    reduction_threads: int,
    vec: int,
) -> None:
    # A register-tile config over a body the two-pass schedule cannot place
    # (a per-element fp32 store of a bf16 tile, a live-value budget overrun)
    # compiles with the rolled lane nesting instead of raising.
    torch.manual_seed(0)
    tensors = tuple(
        torch.randn(128, 16384, device=DEVICE).to(dtype) for _ in range(inputs)
    )
    bound = kernel.bind(tensors)
    config = column_config(
        bound, block=block, tile_threads=8, reduction_threads=reduction_threads, vec=vec
    )
    assert config.reduction_loops == [None]
    code = bound.to_code(config)
    assert "for synthetic_lane_1 in range(" in code
    assert "_lane_stash" not in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(*tensors)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(), reference(*tensors).float(), rtol=1e-4, atol=1e-4
        )
        _randn_like_inputs(tensors)


def _col_sum_guarded_reference(x: torch.Tensor, flag: int) -> torch.Tensor:
    return x.sum(0) * (2.0 if flag > 0 else 1.0)


@pytest.mark.parametrize(
    ("kernel", "reference", "inputs", "extra"),
    (
        pytest.param(
            col_sum_guarded,
            _col_sum_guarded_reference,
            1,
            (1,),
            id="guard-inside-the-element-loops",
        ),
        pytest.param(
            col_scale_three,
            _col_scale_three_reference,
            3,
            (),
            id="three-stashed-tiles",
        ),
    ),
)
def test_looped_retry_matches_reference(
    kernel: helion.Kernel,
    reference: Callable[..., torch.Tensor],
    inputs: int,
    extra: tuple[int, ...],
) -> None:
    # Over 1024 rows with threads [8, 64] the register tile is the only reason
    # the config keeps the reduction persistent.  When the split rejects the
    # body, the retry compiles the looped reduction (reduction_loops=[128],
    # 64 reduction threads) the geometry meant before register tiles existed.
    torch.manual_seed(0)
    tensors = tuple(torch.randn(1024, 4096, device=DEVICE) for _ in range(inputs))
    bound = kernel.bind((*tensors, *extra))
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=64)
    assert config.reduction_loops == [None]
    code = bound.to_code(config)
    assert "synthetic_lane_1" not in code
    assert "_REDUCTION_BLOCK_1" in code
    assert "block=(64, 8, 1)" in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(*tensors, *extra)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, reference(*tensors, *extra), rtol=1e-4, atol=1e-4
        )
        _randn_like_inputs(tensors)


def test_column_softmax_keeps_the_rolled_lane() -> None:
    # One reduction feeding another is kept out of the register tile by the
    # device IR; the requested geometry compiles as the rolled persistent lane
    # it always was.
    torch.manual_seed(0)
    x = torch.randn(128, 16384, device=DEVICE)
    bound = col_softmax.bind((x,))
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    assert config.reduction_loops == [None]
    code = bound.to_code(config)
    assert "for synthetic_lane_1 in range(" in code
    assert "_lane_stash" not in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.softmax(x, dim=0), rtol=1e-4, atol=1e-5)
        x.uniform_(-1, 1)


def test_eight_byte_policy_register_tile_matches_reference() -> None:
    # fp32 V=2 under ``l1_l2_first``: the 8-byte policy helper is a load the
    # register tile stashes for its consume nest.
    torch.manual_seed(0)
    x = torch.randn(128, 16384, device=DEVICE)
    y = torch.empty_like(x)
    bound = col_scale_into.bind((x, y))
    config = column_config(
        bound,
        block=16,
        tile_threads=8,
        reduction_threads=8,
        vec=2,
        load_eviction_policies=["l1_l2_first"],
    )
    code = bound.to_code(config)
    assert code.count("_cute_load_l1_l2_evict_first_8b(") == 1
    assert "_lane_stash" in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        compiled(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, _col_scale_reference(x), rtol=1e-4, atol=1e-3)
        x.uniform_(-1, 1)
        y.zero_()


def test_dynamic_extent_binding_covers_larger_extents() -> None:
    # ``static_shapes=False``: the shrunk reduction thread count is forced
    # looped, so a binding made at 1024 rows reads the extent at runtime and
    # covers 2048 rows too instead of baking a 32-lane loop from the hint.
    torch.manual_seed(0)
    small = torch.randn(1024, 4096, device=DEVICE)
    bound = col_sum_dynamic.bind((small,))
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=64)
    assert config.reduction_loops == [128]
    compiled = bound.compile_config(config)
    for x in (small, torch.randn(2048, 4096, device=DEVICE)):
        out = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x.sum(0), rtol=1e-4, atol=1e-3)
