"""Runtime coverage of the ``cute_mma_dag`` tcgen05 compute mode: the
linear-attention engine's serial state recurrences (``chunk_fwd_h_diag_fused``
/ ``chunk_bwd_dh_diag_fused``) in every decay mode against the Triton
backend, repeat-launch determinism, and the stage / store / warp knobs being
pure data movement.  sm100 / sm103 only."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from test.test_cute_mma_dag_gpu import _KDA_INEXACT
from test.test_cute_mma_dag_gpu import _KDA_SCALED
from test.test_cute_mma_dag_gpu import _M4_ARGS
from test.test_cute_mma_dag_gpu import _assert_close
from test.test_cute_mma_dag_gpu import _bwd_args
from test.test_cute_mma_dag_gpu import _dqk_args
from test.test_cute_mma_dag_gpu import _dstate_args
from test.test_cute_mma_dag_gpu import _dv_args
from test.test_cute_mma_dag_gpu import _fwd_args
from test.test_cute_mma_dag_gpu import _fwd_h_delta_args
from test.test_cute_mma_dag_gpu import _fwd_o_args
from test.test_cute_mma_dag_gpu import _head_sum_args
from test.test_cute_mma_dag_gpu import _inputs
from test.test_cute_mma_dag_gpu import _inputs3
from test.test_cute_mma_dag_gpu import _inputs4
from test.test_cute_mma_dag_gpu import _inputs_kda
from test.test_cute_mma_dag_gpu import _inputs_m4
from test.test_cute_mma_dag_gpu import _kda_args
from test.test_cute_mma_dag_gpu import _kernel_fn
from test.test_cute_mma_dag_gpu import _probe_kernels
from test.test_cute_mma_dag_gpu import _randn
from test.test_cute_mma_dag_gpu import _run_triton
from test.test_cute_mma_dag_gpu import _tensors
from test.test_cute_mma_dag_tcgen05 import _bind_fwd_o_anch_f32  # noqa: F401
from test.test_cute_mma_dag_tcgen05 import _probe_carry2
from test.test_cute_mma_dag_tcgen05 import _probe_dotless_store
from test.test_cute_mma_dag_tcgen05 import _probe_drainrace
from test.test_cute_mma_dag_tcgen05 import _probe_h1
from test.test_cute_mma_dag_tcgen05 import _probe_h3
from test.test_cute_mma_dag_tcgen05 import _probe_h4
from test.test_cute_mma_dag_tcgen05 import _probe_hoist
from test.test_cute_mma_dag_tcgen05 import _probe_hoist_loop
from test.test_cute_mma_dag_tcgen05 import _probe_kk
from test.test_cute_mma_dag_tcgen05 import _probe_rawlate
from test.test_cute_mma_dag_tcgen05 import _probe_recompute
from test.test_cute_mma_dag_tcgen05 import _probe_register_carry
from test.test_cute_mma_dag_tcgen05 import _probe_register_carry_budget
from test.test_cute_mma_dag_tcgen05 import _rmw_scalar_first
from test.test_cute_mma_dag_tcgen05 import _snap_reread
from test.test_cute_mma_dag_tcgen05 import _snap_reread2
from test.test_cute_mma_dag_tcgen05 import _snap_reread_view
from test.test_cute_mma_dag_tcgen05 import _snap_same_dst
from test.test_cute_mma_dag_tcgen05 import _store_after
from test.test_cute_mma_dag_tcgen05 import _store_first

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        os.environ.get("HELION_INTERPRET") == "1",
        reason="the tests pin configs and inspect generated code",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="tcgen05 needs sm100 / sm103",
    ),
]

RT = "_helion_mma_dag_tc"
LAUNCHES = 8


def _config(
    block_sizes: list[int], warps: int, stages: int, **extra: object
) -> helion.Config:
    return helion.Config(
        block_sizes=block_sizes,
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=warps,
        cute_mma_dag_stages=stages,
        **extra,
    )


def _run_cute(
    fn: Callable[..., object], args: tuple[object, ...], config: helion.Config
) -> tuple[str, tuple[torch.Tensor, ...]]:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    assert normalized.config["cute_mma_dag"] == "tcgen05", normalized
    source = bound.to_code(normalized)
    assert f"{RT}.prims.tcgen05_mma(" in source, source
    assert "_cute_grouped_reduce_shared_two_stage" not in source
    assert "_helion_mma_dag.mma_" not in source
    compiled = bound.compile_config(normalized)
    outputs = [_tensors(compiled(*args)) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    for other in outputs[1:]:
        for lhs, rhs in zip(outputs[0], other, strict=True):
            assert torch.equal(lhs, rhs)
    return source, outputs[0]


# kernel, mode, has_h0, (n, d, dv), dtype, bm, bn, warps, stages
_CASES = (
    ("fwd", "vanilla", False, (8, 64, 64), torch.bfloat16, 64, 64, 8, 4),
    ("fwd", "vanilla", False, (8, 64, 64), torch.bfloat16, 64, 64, 4, 2),
    ("fwd", "vanilla", False, (64, 64, 64), torch.bfloat16, 64, 64, 8, 8),
    ("fwd", "vanilla", False, (8, 128, 128), torch.bfloat16, 128, 128, 8, 2),
    ("fwd", "vanilla", False, (8, 128, 128), torch.bfloat16, 64, 128, 8, 3),
    ("fwd", "vanilla", False, (8, 128, 128), torch.bfloat16, 128, 64, 4, 4),
    ("fwd", "vanilla", False, (8, 64, 64), torch.float16, 64, 64, 8, 4),
    ("fwd", "vanilla", True, (8, 64, 64), torch.bfloat16, 64, 64, 8, 2),
    ("fwd", "diag", True, (8, 64, 64), torch.bfloat16, 64, 64, 8, 4),
    ("fwd", "diag", False, (8, 128, 128), torch.bfloat16, 128, 128, 8, 2),
    ("fwd", "diag", True, (64, 128, 128), torch.bfloat16, 64, 64, 8, 8),
    ("fwd", "scalar", True, (8, 64, 64), torch.bfloat16, 64, 64, 8, 4),
    ("fwd", "scalar", True, (8, 128, 128), torch.bfloat16, 128, 64, 8, 4),
    ("fwd", "scalar", False, (8, 64, 64), torch.float16, 64, 32, 4, 2),
    ("fwd", "vanilla", False, (8, 64, 64), torch.bfloat16, 64, 16, 8, 2),
    ("bwd", "vanilla", False, (8, 64, 64), torch.bfloat16, 64, 64, 8, 4),
    ("bwd", "vanilla", False, (64, 128, 128), torch.bfloat16, 64, 64, 8, 8),
    ("bwd", "vanilla", False, (8, 128, 128), torch.bfloat16, 128, 128, 4, 2),
    ("bwd", "diag", False, (8, 64, 64), torch.bfloat16, 64, 64, 8, 1),
    ("bwd", "diag", False, (8, 128, 128), torch.float16, 128, 128, 8, 2),
    ("bwd", "scalar", False, (8, 64, 64), torch.bfloat16, 64, 64, 8, 2),
    ("bwd", "scalar", False, (8, 128, 128), torch.bfloat16, 64, 128, 4, 4),
)


@pytest.mark.parametrize(
    ("kernel", "mode", "has_h0", "shape", "dtype", "bm", "bn", "warps", "stages"),
    _CASES,
    ids=[
        f"{k}-{m}{'-h0' if h else ''}-n{s[0]}d{s[1]}dv{s[2]}-{str(dt).split('.')[-1]}-{bm}x{bn}w{w}s{st}"
        for k, m, h, s, dt, bm, bn, w, st in _CASES
    ],
)
def test_state_recurrences_match_the_triton_backend(
    kernel: str,
    mode: str,
    has_h0: bool,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    bm: int,
    bn: int,
    warps: int,
    stages: int,
) -> None:
    n, d, dv = shape
    t = _inputs(bh=2, n=n, d=d, dv=dv, dtype=dtype)
    args = _fwd_args(t, mode, has_h0) if kernel == "fwd" else _bwd_args(t, mode)
    fn = _kernel_fn(kernel)
    source, out = _run_cute(fn, args, _config([bm, bn], warps, stages))
    assert f"block=({32 * warps}, 1, 1)" in source
    _assert_close(out, _run_triton(fn, args, [bm, bn]))


@pytest.mark.parametrize("kernel", ("fwd", "bwd"))
def test_stage_depth_warps_and_store_path_are_pure_data_movement(
    kernel: str,
) -> None:
    t = _inputs(bh=2, n=16, d=64, dv=64, dtype=torch.bfloat16, seed=3)
    args = _fwd_args(t, "diag", True) if kernel == "fwd" else _bwd_args(t, "scalar")
    fn = _kernel_fn(kernel)
    baseline = _run_cute(fn, args, _config([64, 64], 8, 1))[1]
    for stages in (2, 4, 8):
        assert torch.equal(
            _run_cute(fn, args, _config([64, 64], 8, stages))[1][0], baseline[0]
        )
    assert torch.equal(_run_cute(fn, args, _config([64, 64], 4, 4))[1][0], baseline[0])
    source, direct = _run_cute(
        fn, args, _config([64, 64], 8, 4, cute_mma_dag_store="direct")
    )
    assert "_helion_mma_dag.stg_v4_b32(" not in source
    assert torch.equal(direct[0], baseline[0])
    staged = _run_cute(fn, args, _config([64, 64], 8, 4, cute_mma_dag_store="staged"))[
        0
    ]
    assert "_helion_mma_dag.stg_v4_b32(" in staged


@pytest.mark.parametrize(
    ("kernel", "mode", "has_h0"),
    (("fwd", "vanilla", False), ("fwd", "scalar", True), ("bwd", "vanilla", False)),
)
def test_the_warp_and_tcgen05_modes_agree(kernel: str, mode: str, has_h0: bool) -> None:
    # The operand passes round once per element either way (the warp mode
    # through re-typed leaves, this mode through the final pack), the bf16
    # products are exact and both accumulate the K steps in order in fp32,
    # so the modes agree to the bit.
    t = _inputs(bh=2, n=16, d=64, dv=64, dtype=torch.bfloat16, seed=7)
    args = _fwd_args(t, mode, has_h0) if kernel == "fwd" else _bwd_args(t, mode)
    fn = _kernel_fn(kernel)
    tc = _run_cute(fn, args, _config([64, 64], 8, 4))[1][0]
    kernel_obj = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel_obj.bind(args)
    warp = _tensors(
        bound.compile_config(
            helion.Config(
                block_sizes=[64, 64],
                cute_mma_dag="warp",
                cute_mma_dag_warps=8,
                cute_mma_dag_stages=4,
            )
        )(*args)
    )[0]
    torch.cuda.synchronize()
    assert torch.equal(tc, warp)


def test_the_promoted_default_runs_on_tensor_memory() -> None:
    t = _inputs(bh=2, n=8, d=128, dv=128, dtype=torch.bfloat16)
    for kernel, args in (
        ("fwd", _fwd_args(t, "vanilla", False)),
        ("bwd", _bwd_args(t, "diag")),
    ):
        fn = _kernel_fn(kernel)
        bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
        config = bound.config_spec.default_config()
        assert config.config["cute_mma_dag"] == "tcgen05", kernel
        assert config.config["cute_mma_dag_warps"] == 4
        # Eight stages, or four where eight 128-wide stages exceed shared memory.
        assert config.config["cute_mma_dag_stages"] in (8, 4)
        source = bound.to_code(config)
        assert f"{RT}.prims.tcgen05_mma(" in source
        out = _tensors(bound.compile_config(config)(*args))
        _assert_close(out, _run_triton(fn, args, config.config["block_sizes"]))


# kernel, mode, (d, dv), block_sizes, warps, stages
_K_LOOP_CASES = (
    ("fwd_o", "vanilla", (64, 64), [1, 64, 64], 4, 2),
    ("fwd_o", "vanilla", (128, 128), [1, 128, 128], 4, 2),
    ("fwd_o", "vanilla", (128, 128), [1, 128, 64], 8, 4),
    ("fwd_o", "vanilla", (128, 128), [1, 64, 32], 4, 8),
    ("fwd_o", "vanilla", (64, 128), [1, 16, 16], 4, 2),
    ("fwd_o", "gated", (128, 128), [1, 128, 64], 4, 4),
    ("fwd_o", "gated", (64, 64), [1, 64, 64], 8, 2),
    ("dqk", "vanilla", (64, 64), [1, 64, 64], 4, 2),
    ("dqk", "vanilla", (128, 128), [1, 128, 32], 4, 2),
    ("dqk", "vanilla", (128, 128), [1, 64, 64], 8, 2),
    ("dqk", "gated", (64, 64), [1, 64, 64], 4, 2),
    ("dqk", "gated", (128, 64), [1, 64, 64], 8, 2),
    ("dv", "vanilla", (64, 64), [1, 64, 64], 4, 2),
    ("dv", "vanilla", (128, 128), [1, 128, 64], 4, 2),
    ("dv", "diag", (128, 128), [1, 64, 128], 8, 2),
    ("dv", "scalar", (64, 64), [1, 64, 64], 4, 4),
    ("dv", "scalar", (128, 128), [1, 128, 32], 4, 2),
)


@pytest.mark.parametrize(
    ("kernel", "mode", "shape", "block_sizes", "warps", "stages"),
    _K_LOOP_CASES,
    ids=[
        f"{k}-{m}-d{s[0]}dv{s[1]}-{'x'.join(map(str, b))}w{w}s{st}"
        for k, m, s, b, w, st in _K_LOOP_CASES
    ],
)
def test_k_loop_kernels_match_the_triton_backend(
    kernel: str,
    mode: str,
    shape: tuple[int, int],
    block_sizes: list[int],
    warps: int,
    stages: int,
) -> None:
    d, dv = shape
    t = _inputs3(bhn=6, d=d, dv=dv, dtype=torch.bfloat16)
    args = {"fwd_o": _fwd_o_args, "dqk": _dqk_args, "dv": _dv_args}[kernel](t, mode)
    fn = _kernel_fn(kernel)
    source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    # The fragment operands of the epilogue dots go through an exchange tile
    # read by the tensor core (no mma.sync repack).
    assert f"{RT}.prims.tcgen05_commit(md_eacc" in source
    _assert_close(out, _run_triton(fn, args, block_sizes))


@pytest.mark.parametrize(
    ("kernel", "mode"),
    (("fwd_o", "vanilla"), ("fwd_o", "gated"), ("dqk", "vanilla"), ("dv", "scalar")),
)
def test_k_loop_kernels_agree_with_the_warp_mode(kernel: str, mode: str) -> None:
    # Same K order per dot, same pointwise lowering of the masks / scales /
    # casts (one rounding at the pack) and fp32 accumulation in both modes.
    t = _inputs3(bhn=4, d=64, dv=64, dtype=torch.bfloat16, seed=11)
    args = {"fwd_o": _fwd_o_args, "dqk": _dqk_args, "dv": _dv_args}[kernel](t, mode)
    fn = _kernel_fn(kernel)
    tc = _run_cute(fn, args, _config([1, 64, 64], 4, 2))[1]
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    warp = _tensors(
        bound.compile_config(
            helion.Config(
                block_sizes=[1, 64, 64],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )(*args)
    )
    torch.cuda.synchronize()
    for lhs, rhs in zip(tc, warp, strict=True):
        assert torch.equal(lhs, rhs)


def test_gated_k_loop_epilogues_prefetch_their_decay_rows() -> None:
    # The decay row's per-thread values go out before the accumulator
    # drain (gated fwd_o / dqk, the scalar-decay dv) and the epilogue
    # issues no global load; the hoisted loads move no arithmetic: the
    # outputs stay within the Triton tolerance, bit-equal to the warp
    # mode, and deterministic over eight launches.
    t = _inputs3(bhn=6, d=128, dv=128, dtype=torch.bfloat16, seed=3)
    for kernel, mode, block_sizes, warps in (
        ("fwd_o", "gated", [1, 128, 128], 4),
        ("fwd_o", "gated", [1, 64, 64], 8),
        ("dqk", "gated", [1, 64, 64], 4),
        ("dv", "scalar", [1, 128, 64], 4),
    ):
        args = {"fwd_o": _fwd_o_args, "dqk": _dqk_args, "dv": _dv_args}[kernel](t, mode)
        fn = _kernel_fn(kernel)
        source, out = _run_cute(fn, args, _config(block_sizes, warps, 2))
        epilogue = source.rsplit(f"{RT}.mbar_wait(md_accb, ", 1)[1]
        assert "md_ex0 = _helion_mma_dag.ldg_f32(" in source, kernel
        assert "ldg_" not in epilogue, kernel
        _assert_close(out, _run_triton(fn, args, block_sizes))
    bound = helion.kernel(_kernel_fn("fwd_o"), backend="cute", static_shapes=True).bind(
        _fwd_o_args(t, "gated")
    )
    warp = _tensors(
        bound.compile_config(
            helion.Config(
                block_sizes=[1, 128, 128],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )(*_fwd_o_args(t, "gated"))
    )
    tc = _run_cute(
        _kernel_fn("fwd_o"), _fwd_o_args(t, "gated"), _config([1, 128, 128], 4, 2)
    )[1]
    torch.cuda.synchronize()
    for lhs, rhs in zip(tc, warp, strict=True):
        assert torch.equal(lhs, rhs)


def test_k_loop_stage_depth_warps_and_store_path_are_pure_data_movement() -> None:
    t = _inputs3(bhn=4, d=128, dv=64, dtype=torch.bfloat16, seed=5)
    args = _dqk_args(t, "gated")
    fn = _kernel_fn("dqk")
    baseline = _run_cute(fn, args, _config([1, 64, 64], 4, 1))[1]
    # Two dA exchange tiles (straight and transposed) leave room for three
    # 48 KB stages of the gated dqk at [1, 64, 64].
    for warps, stages in ((4, 2), (8, 2), (4, 3)):
        out = _run_cute(fn, args, _config([1, 64, 64], warps, stages))[1]
        for lhs, rhs in zip(out, baseline, strict=True):
            assert torch.equal(lhs, rhs)
    # Without loop snapshots the store knob has no subject: normalize pins
    # it, the epilogue stores stay staged.
    kernel_obj = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel_obj.bind(args)
    with bound.env:
        pinned = bound.config_spec.normalized_config(
            _config([1, 64, 64], 4, 2, cute_mma_dag_store="direct")
        )
    assert pinned.config["cute_mma_dag_store"] == "staged"
    assert "_helion_mma_dag.stg_v4_b32(" in bound.to_code(pinned)


@pytest.mark.parametrize(
    "config",
    (
        None,
        _config([64, 128, 128], warps=4, stages=2),
        _config([64, 64, 64], warps=4, stages=1),
        _config([64, 128, 64], warps=8, stages=4),
    ),
    ids=["default", "64x128x128w4s2", "64x64x64w4s1", "64x128x64w8s4"],
)
def test_the_chunk_state_example_matches_the_reference(
    config: helion.Config | None,
) -> None:
    # examples/mamba2_chunk_state.py: the chunk offset of the x / B tiles and
    # the leading tiles' begins as coordinates (4 heads per group).
    from examples.mamba2_chunk_state import helion_mamba2_chunk_state_kernel
    from examples.mamba2_chunk_state import ref_chunk_state

    torch.manual_seed(0)
    batch, seqlen, nheads, ngroups, chunk = 2, 1024, 8, 2, 256
    nchunks = seqlen // chunk
    x = torch.randn(batch, seqlen, nheads, 64, device=DEVICE, dtype=torch.bfloat16)
    B = torch.randn(batch, seqlen, ngroups, 128, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.rand(batch, nheads, nchunks, chunk, device=DEVICE, dtype=torch.bfloat16)
    dA = -torch.rand_like(dt).cumsum(-1)
    args = (B, x, dt, dA)
    fn = helion_mamba2_chunk_state_kernel.fn
    if config is None:
        bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
        config = bound.config_spec.default_config()
        assert config.config["cute_mma_dag"] == "tcgen05"
        out = bound.compile_config(config)(*args)
    else:
        _source, (out,) = _run_cute(fn, args, config)
    ref = ref_chunk_state(B.float(), x.float(), dt.float(), dA.float())
    torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-2)


def test_the_k_loop_defaults_run_on_tensor_memory() -> None:
    t = _inputs3(bhn=4, d=128, dv=128, dtype=torch.bfloat16)
    for kernel, args in (
        ("fwd_o", _fwd_o_args(t, "vanilla")),
        ("dqk", _dqk_args(t, "vanilla")),
        ("dv", _dv_args(t, "vanilla")),
    ):
        fn = _kernel_fn(kernel)
        bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
        config = bound.config_spec.default_config()
        assert config.config["cute_mma_dag"] == "tcgen05", kernel
        source = bound.to_code(config)
        assert f"{RT}.prims.tcgen05_mma(" in source
        out = _tensors(bound.compile_config(config)(*args))
        _assert_close(out, _run_triton(fn, args, config.config["block_sizes"]))


# kernel, gated, (n, d, dv), bdv, warps, stages
_DELTA_CASES = (
    ("fwd_h_delta", False, (8, 128, 32), 32, 4, 2),
    ("fwd_h_delta", False, (8, 128, 64), 32, 8, 2),
    ("fwd_h_delta", False, (64, 128, 128), 32, 4, 4),
    ("fwd_h_delta", False, (8, 128, 128), 64, 4, 2),
    ("fwd_h_delta", False, (8, 128, 64), 16, 4, 1),
    ("fwd_h_delta", True, (8, 128, 64), 32, 4, 2),
    ("fwd_h_delta", True, (64, 128, 128), 64, 8, 4),
    ("fwd_h_delta", True, (8, 128, 32), 16, 4, 1),
    ("fwd_h_delta", True, (8, 64, 64), 32, 4, 2),
    ("dstate", False, (8, 128, 64), 32, 4, 2),
    ("dstate", False, (64, 128, 128), 32, 8, 3),
    ("dstate", False, (8, 128, 32), 16, 4, 2),
    ("dstate", False, (8, 128, 64), 64, 4, 2),
    ("dstate", False, (8, 128, 64), 32, 4, 1),
    ("dstate", True, (8, 128, 64), 32, 4, 2),
    ("dstate", True, (64, 128, 128), 32, 8, 3),
    ("dstate", True, (8, 128, 128), 64, 4, 1),
    ("dstate", False, (8, 64, 64), 32, 4, 2),
)


@pytest.mark.parametrize(
    ("kernel", "gated", "shape", "bdv", "warps", "stages"),
    _DELTA_CASES,
    ids=[
        f"{k}{'-gated' if g else ''}-n{s[0]}d{s[1]}dv{s[2]}-{bdv}w{w}s{st}"
        for k, g, s, bdv, w, st in _DELTA_CASES
    ],
)
def test_delta_state_recurrences_match_the_triton_backend(
    kernel: str,
    gated: bool,
    shape: tuple[int, int, int],
    bdv: int,
    warps: int,
    stages: int,
) -> None:
    n, d, dv = shape
    t = _inputs4(bh=2, n=n, d=d, dv=dv, dtype=torch.bfloat16)
    args = (_fwd_h_delta_args if kernel == "fwd_h_delta" else _dstate_args)(t, gated)
    fn = _kernel_fn(kernel)
    source, out = _run_cute(fn, args, _config([bdv], warps, stages))
    assert f"block=({32 * warps}, 1, 1)" in source
    # The carry feeds a dot of the same iteration through an exchange tile,
    # the derived values round trip through tensor memory: several commits
    # per iteration.
    assert f"{RT}.prims.tcgen05_commit(md_stepb" in source
    # Triton's dot is tcgen05 on sm100 too, over the same K order, and the
    # pointwise steps share the Inductor lowering: bit-equal, except that
    # Triton contracts the gated dstate's ``dv_i + dv_kh * decay`` into an
    # FMA (one bf16 ulp on a few elements; the warp mode shows the same).
    ref = _run_triton(fn, args, [bdv])
    _assert_close(out, ref)
    if not (gated and kernel == "dstate"):
        for lhs, rhs in zip(out, ref, strict=True):
            assert torch.equal(lhs, rhs)
    # Same program order, same pointwise lowering, fp32 accumulation over
    # the same K steps as the warp mode at the same tile: bit-equal (the
    # warp mode's register budget may ask for more warps at this tile).
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    warp_config = dict(
        _config([bdv], warps, min(stages, 2)).config, cute_mma_dag="warp"
    )
    with bound.env:
        bound.config_spec.normalize(warp_config, _fix_invalid=True)
    assert warp_config["cute_mma_dag"] == "warp", warp_config
    assert warp_config["block_sizes"] == [bdv], warp_config
    warp = _tensors(bound.compile_config(helion.Config(**warp_config))(*args))
    torch.cuda.synchronize()
    for lhs, rhs in zip(out, warp, strict=True):
        assert torch.equal(lhs, rhs)


@pytest.mark.parametrize(("warps", "stages"), ((4, 2), (8, 4), (4, 8)))
def test_a_body_store_in_the_first_block_matches_the_triton_backend(
    warps: int, stages: int
) -> None:
    # ``d = h - u`` stored before the iteration's only dot; its staging buffer
    # alternates by parity so a warp still draining it cannot be overrun by the
    # next iteration's stage-in.
    t = _inputs(bh=2, n=64, d=64, dv=64, dtype=torch.bfloat16, seed=3)
    g = torch.Generator(device=DEVICE).manual_seed(13)
    args = (t["k"], t["v"], _randn(g, 2, 64, 64, 64, dtype=torch.bfloat16))
    source, out = _run_cute(_store_first, args, _config([64, 64], warps, stages))
    assert "md_parity * 8192 + md_row64 * 128" in source
    _assert_close(out, _run_triton(_store_first, args, [64, 64]))


@pytest.mark.parametrize(
    "name",
    (
        "_acc_frag_stored",
        "_acc_dag_stored",
        "_epi_chain",
        "_cc_store",
        "_chain_collide",
        "_cc_load_operand",
        "_two_dots_same_carry",
        "_loop_chain",
        "_loop_chain_store",
    ),
)
@pytest.mark.parametrize(("warps", "stages"), ((4, 2), (8, 1)))
def test_probe_kernels_match_the_triton_backend(
    name: str, warps: int, stages: int
) -> None:
    # The K-loop probe shapes on tensor memory: stored accumulators (fresh
    # slots), chained epilogue dots, [C, C] stores and loads, colliding
    # sharing keys, and the loop chains (two batches per iteration, with
    # and without a body store of the first batch's result).
    import importlib

    fn = getattr(importlib.import_module("test.test_cute_mma_dag"), name)
    if name in ("_two_dots_same_carry", "_loop_chain", "_loop_chain_store"):
        t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=7)
        args: tuple[object, ...] = (t["k"], t["v"])
        block_sizes = [64, 64]
    else:
        t3 = _inputs3(bhn=4, d=64, dv=64, dtype=torch.bfloat16, seed=7)
        g = torch.Generator(device=DEVICE).manual_seed(11)
        args = (t3["q"], t3["k"], t3["v"], t3["h"])
        if name == "_epi_chain":
            args = (
                *args,
                _randn(g, 4, 64, 32, dtype=torch.bfloat16),
                _randn(g, 4, 64, 32, dtype=torch.bfloat16),
            )
        elif name == "_chain_collide":
            args = (*args, _randn(g, 4, 64, scale=0.1, dtype=torch.float32))
        elif name == "_cc_load_operand":
            args = (*args, _randn(g, 4, 64, 64, dtype=torch.bfloat16))
        block_sizes = [1, 64, 64]
    _source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    _assert_close(out, _run_triton(fn, args, block_sizes))


def test_unaligned_operand_sources_take_the_synchronous_pass() -> None:
    # A source whose base is 8-byte aligned has no tensor map: the operand
    # goes through the cooperative pass into the swizzled slot while the
    # other operand still arrives by TMA.
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=5)
    k_wide = torch.randn((2, 8, 64, 80), device=DEVICE, dtype=torch.bfloat16)
    k = k_wide[..., 4:68]
    assert k.data_ptr() % 16 == 8 and k.stride(2) == 80
    fn = _kernel_fn("fwd")
    args = (k, t["v"], None, None, None, False, False, False, False)
    source, out = _run_cute(fn, args, _config([64, 64], 8, 2))
    assert "_helion_mma_dag.ldg_b32(" in source and "_helion_mma_dag.sts_b32(" in source
    assert "'tma_count': 1" in source
    _assert_close(out, _run_triton(fn, args, [64, 64]))


def test_the_program_order_loop_prefetches_the_epilogue_row_after_the_loop() -> None:
    # ``_gen_epi``: the carry is the B operand (a program-order loop), so
    # the decay row goes out after the loop, ahead of the drain, and the
    # epilogue issues no global load.  Within the Triton tolerance,
    # bit-equal to the warp mode, deterministic over the eight launches of
    # ``_run_cute``.
    fn = _probe_kernels()._gen_epi
    g = torch.Generator(device=DEVICE).manual_seed(19)
    args = (
        _randn(g, 6, 4, 64, 64, scale=0.15, dtype=torch.bfloat16),
        _randn(g, 6, 64, 64, dtype=torch.bfloat16),
        _randn(g, 6, 64, scale=0.1, dtype=torch.float32),
    )
    source, out = _run_cute(fn, args, _config([], 4, 2))
    loop, epilogue = source.rsplit(f"{RT}.mbar_wait(md_accb, 4 - 0 - 1 & 1)", 1)
    last = loop.rstrip().rsplit("\n", 1)[1]
    assert last.startswith("    md_ex0 = _helion_mma_dag.ldg_f32("), last
    assert "ldg_" not in epilogue
    _assert_close(out, _run_triton(fn, args, []))
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    warp = _tensors(
        bound.compile_config(
            helion.Config(
                block_sizes=[],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )(*args)
    )
    torch.cuda.synchronize()
    for lhs, rhs in zip(out, warp, strict=True):
        assert torch.equal(lhs, rhs)


def test_a_wide_epilogue_fragment_prefetches_only_its_row_site() -> None:
    # Gated fwd_o at c = 128: one thread per row of the 128-row attention
    # exchange holds all 128 columns, so only the row site goes out ahead
    # of the drain and the 128 column loads stay in the epilogue.
    g = torch.Generator(device=DEVICE).manual_seed(5)
    bhn, c, d, dv = 4, 128, 128, 128
    args = (
        _randn(g, bhn, c, d, dtype=torch.bfloat16),
        _randn(g, bhn, c, d, dtype=torch.bfloat16),
        _randn(g, bhn, c, dv, dtype=torch.bfloat16),
        _randn(g, bhn, c, scale=0.1, dtype=torch.float32),
        _randn(g, bhn, d, dv, dtype=torch.bfloat16),
        True,
        0.125,
    )
    fn = _kernel_fn("fwd_o")
    source, out = _run_cute(fn, args, _config([1, 64, 64], 4, 2))
    epilogue = source.rsplit(f"{RT}.mbar_wait(md_accb, ", 1)[1]
    assert source.count("md_ex0 = _helion_mma_dag.ldg_f32(") == 1
    assert "md_ex1" not in source
    assert epilogue.count("_helion_mma_dag.ldg_f32(") == 128
    _assert_close(out, _run_triton(fn, args, [1, 64, 64]))


# kernel, gated, block_sizes, warps, stages
_M4_CASES = (
    ("dk", False, [64], 4, 2),
    ("dk", False, [32], 8, 4),
    ("dk", False, [128], 4, 1),
    ("dk", False, [16], 4, 8),
    ("dqkw", False, [64, 64, 64], 4, 2),
    ("dqkw", False, [32, 64, 32], 8, 1),
    ("dqkw", False, [64, 32, 16], 4, 4),
    ("dqkw", True, [64, 64, 64, 64, 64], 4, 2),
    ("dqkw", True, [32, 32, 32, 64, 64], 8, 2),
    ("dqkw", True, [64, 32, 64, 128, 32], 4, 1),
    ("wy_dL", False, [64, 64, 64], 4, 2),
    ("wy_dL", False, [32, 64, 32], 8, 1),
    ("wy_dL", False, [64, 32, 128], 4, 4),
    ("wy_dL", True, [64, 64, 64], 4, 2),
    ("wy_dL", True, [32, 32, 32], 4, 4),
    ("wy_dL", True, [64, 32, 64], 8, 2),
    ("wy", False, [64, 64, 64], 4, 2),
    ("wy", False, [32, 64, 64], 4, 4),
    ("wy", False, [64, 128, 128], 4, 1),
    ("wy", False, [64, 64, 64], 8, 2),
    ("wy", True, [64, 64, 64], 4, 2),
    ("wy", True, [64, 32, 128], 8, 1),
    # The kda / anchored region programs (a tile-only derived operand in
    # tensor memory, a dots-only batch behind a loop, the outer loop's staging
    # behind a sibling loop's last batch).
    ("bwd_o_kda", False, [64, 64, 16], 4, 4),
    ("bwd_o_kda", False, [64, 64, 64], 4, 2),
    ("bwd_o_kda", False, [32, 64, 32], 8, 1),
    ("state_du_kda", False, [16], 4, 8),
    ("state_du_kda", False, [64], 4, 2),
    ("state_du_kda", False, [32], 8, 4),
    ("fwd_o_anch", False, [1, 16, 16], 4, 4),
    ("fwd_o_anch", False, [1, 64, 64], 4, 2),
    ("fwd_o_anch", False, [1, 32, 128], 8, 1),
)


_KDA_KERNELS = ("bwd_o_kda", "state_du_kda", "fwd_o_anch")


def _assert_region_outputs(
    kernel: str, out: tuple[torch.Tensor, ...], ref: tuple[torch.Tensor, ...]
) -> None:
    """Fragment outputs bit-equal, vectors by tolerance; state_du_kda's
    serially carried fp32 state as in the warp mode's tests (bf16 operand
    roundings flip where the fp32 op order differs: a few bf16 ulps of the
    large terms)."""
    inexact = _KDA_INEXACT.get(kernel, set())
    for index, (lhs, rhs) in enumerate(zip(out, ref, strict=True)):
        if index in inexact and kernel in _KDA_SCALED:
            atol = 2e-2 * max(1.0, rhs.abs().max().item())
            torch.testing.assert_close(lhs, rhs, atol=atol, rtol=2e-2)
        elif lhs.dim() >= 3 and index not in inexact:
            assert torch.equal(lhs, rhs), index
        else:
            _assert_close((lhs,), (rhs,))


def _m4_args(kernel: str, gated: bool, seed: int = 3) -> tuple[object, ...]:
    if kernel in _KDA_KERNELS:
        return _kda_args(kernel, _inputs_kda(bh=2, n=2, d=128, dv=128, seed=seed), 2)
    t = _inputs_m4(bhn=4, d=128, dv=128, dtype=torch.bfloat16, seed=seed)
    return _M4_ARGS[kernel](t, gated)


@pytest.mark.parametrize(
    ("kernel", "gated", "block_sizes", "warps", "stages"),
    _M4_CASES,
    ids=[
        f"{k}{'-gated' if g else ''}-{'x'.join(map(str, b))}w{w}s{st}"
        for k, g, b, w, st in _M4_CASES
    ],
)
def test_region_programs_match_the_triton_backend(
    kernel: str, gated: bool, block_sizes: list[int], warps: int, stages: int
) -> None:
    # The region programs on tensor memory (the region renderer): nested and
    # sequential loops, hoisted tiles, seeded fresh accumulators, carries
    # continued across loops, vector carries with row / column / scalar
    # reductions, and wy's tf32 Neumann chain (fp32 exchange tiles, carries
    # seeded from a pointwise value and copied from the dot's slot, the
    # widen pass into transposed fp32 tiles).  Fragment outputs (3-D) are
    # bit-equal to Triton (same K order per dot, same pointwise lowering,
    # the tensor core's tf32 truncation) and, for the 16-bit kernels, to
    # the warp mode; the vector outputs differ only in summation order.
    args = _m4_args(kernel, gated)
    fn = _kernel_fn(kernel)
    source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    assert "md_cb = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)" in source
    assert "md_cn = md_cn + 1" in source
    assert f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)" in source
    if kernel == "wy":
        assert f"{RT}.prims.Tcgen05MMAKind.TF32" in source
        assert "a_dtype=cutlass.Float32, b_dtype=cutlass.Float32" in source
    ref = _run_triton(fn, args, block_sizes)
    _assert_region_outputs(kernel, out, ref)
    if kernel == "wy":
        # mma.sync's tf32 m16n8k8 and the UMMA accumulate in different
        # orders inside a K step: no bit-equality with the warp mode.
        return
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    warp_config = dict(
        _config(block_sizes, warps, min(stages, 2)).config, cute_mma_dag="warp"
    )
    with bound.env:
        bound.config_spec.normalize(warp_config, _fix_invalid=True)
    assert warp_config["cute_mma_dag"] == "warp", warp_config
    warp = _tensors(bound.compile_config(helion.Config(**warp_config))(*args))
    torch.cuda.synchronize()
    _assert_region_outputs(kernel, out, warp)


@pytest.mark.parametrize(
    ("kernel", "gated", "block_sizes", "warps", "stages"),
    (
        ("dk", False, [64], 4, 2),
        ("dk", False, [32], 4, 1),
        ("wy_dL", False, [32, 32, 32], 4, 2),
        ("wy_dL", True, [64, 32, 16], 4, 2),
        ("dqkw", True, [64, 32, 16, 64, 16], 4, 2),
    ),
    ids=["dk-64w4s2", "dk-32w4s1", "wy_dL-32x32x32w4s2", "wy_dL-gated", "dqkw-gated"],
)
def test_the_loads_knob_matches_the_triton_backend(
    kernel: str, gated: bool, block_sizes: list[int], warps: int, stages: int
) -> None:
    # The derived values' tile inputs through the ring (``tma``: dk's seed
    # tiles, wy_dL's reduction operands, gated dqkw's load-only ``hd`` loop)
    # and from global memory per lane (``lane``) are the same numbers:
    # fragment outputs bit-equal to Triton, vectors by tolerance, eight
    # launches each.
    args = _m4_args(kernel, gated)
    fn = _kernel_fn(kernel)
    ref = _run_triton(fn, args, block_sizes)
    outs = {}
    for loads in ("tma", "lane"):
        source, out = _run_cute(
            fn, args, _config(block_sizes, warps, stages, cute_mma_dag_loads=loads)
        )
        if kernel == "dk":
            assert (f"{RT}.lds_v4_f32(md_s0 + 0 + md_st0" in source) == (loads == "tma")
        _assert_region_outputs(kernel, out, ref)
        outs[loads] = out
    for lhs, rhs in zip(outs["tma"], outs["lane"], strict=True):
        assert torch.equal(lhs, rhs)


def test_root_items_overlaying_the_loop_frames_match_the_triton_backend() -> None:
    # The root items placed over the loop frames (a store's staging buffer
    # between two loops drained behind a barrier before the next prologue;
    # a hoisted fp32 operand's raw tile over a narrow ring, converted ahead
    # of the ring prologue) leave the numbers bit-equal to Triton over
    # eight launches.
    from test.test_cute_mma_dag_tcgen05 import _hoisted_raw_over_a_narrow_ring
    from test.test_cute_mma_dag_tcgen05 import _two_loops_store_between

    g = torch.Generator(device=DEVICE).manual_seed(5)
    bf16 = torch.bfloat16
    a = torch.randn((48, 64, 128), device=DEVICE, dtype=torch.float32, generator=g).to(
        bf16
    )
    b = torch.randn((48, 128, 64), device=DEVICE, dtype=torch.float32, generator=g).to(
        bf16
    )
    c = torch.randn((48, 64, 128), device=DEVICE, dtype=torch.float32, generator=g).to(
        bf16
    )
    d = torch.randn((48, 128, 64), device=DEVICE, dtype=torch.float32, generator=g).to(
        bf16
    )
    fn = _two_loops_store_between.fn
    for warps, stages in ((4, 2), (8, 1)):
        source, out = _run_cute(fn, (a, b, c, d), _config([64, 64], warps, stages))
        assert (
            "alloc_smem(cutlass.Uint8, 32768, alignment=1024)" in source or stages == 1
        )
        ref = _run_triton(fn, (a, b, c, d), [64, 64])
        for lhs, rhs in zip(out, ref, strict=True):
            assert torch.equal(lhs, rhs)
    a32 = torch.randn((48, 64, 64), device=DEVICE, dtype=torch.float32, generator=g)
    b2 = torch.randn((48, 64, 128), device=DEVICE, dtype=torch.float32, generator=g).to(
        bf16
    )
    fn = _hoisted_raw_over_a_narrow_ring.fn
    for sizes, warps, stages in (([16], 4, 2), ([32], 4, 1), ([16], 8, 4)):
        source, out = _run_cute(fn, (a32, b2), _config(sizes, warps, stages))
        if sizes == [16] and stages == 2:
            assert source.index(f"{RT}.mbar_wait(md_sfr, 0)") < source.index(
                f"{RT}.prims.mbarrier_arrive_expect_tx(md_full0"
            )
        ref = _run_triton(fn, (a32, b2), sizes)
        for lhs, rhs in zip(out, ref, strict=True):
            assert torch.equal(lhs, rhs)


@pytest.mark.parametrize(
    ("gated", "block_sizes", "warps", "stages"),
    (
        (False, [64, 32, 128], 4, 2),
        (False, [32, 32, 64], 4, 2),
        (True, [64, 32, 128], 4, 3),
        (True, [32, 32, 64], 8, 2),
    ),
    ids=["64x32x128w4s2", "32x32x64w4s2", "gated-64x32x128w4s3", "gated-32x32x64w8s2"],
)
def test_shared_tmem_columns_match_the_triton_backend(
    gated: bool, block_sizes: list[int], warps: int, stages: int
) -> None:
    # The batch-aware TMEM layout shares columns across wy_dL's root chain
    # (dAinv, dot(T, dAinv), dA and kk: 128 columns instead of 256; gated
    # 128 instead of 512 once dLL stays in registers) with every loop
    # running at least two trips: the
    # sharing happens in the render (two slots at one column) and the
    # numbers stay Triton's over eight launches.
    args = _m4_args("wy_dL", gated)
    fn = _kernel_fn("wy_dL")
    source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    assert f"{RT}.prims.tcgen05_alloc(md_tslot, 128," in source
    columns = re.findall(r"md_acc(\d+) = cutlass\.inttoptr\(md_tmem \+ (\d+),", source)
    assert len({column for _slot, column in columns}) < len(columns)
    _assert_region_outputs("wy_dL", out, _run_triton(fn, args, block_sizes))


@pytest.mark.parametrize(
    ("kernel", "gated"),
    (("dk", False), ("wy_dL", False), ("wy_dL", True), ("dqkw", False), ("dqkw", True)),
    ids=["dk", "wy_dL", "wy_dL-gated", "dqkw", "dqkw-gated"],
)
def test_flat_streaming_region_programs_run_on_tensor_memory(
    kernel: str, gated: bool
) -> None:
    # The promoted default of the streaming region programs (loops, loop
    # dots over loaded tiles or hoisted root exchanges, no root tile) is a
    # tensor-memory seed; it runs and matches Triton.
    args = _m4_args(kernel, gated)
    fn = _kernel_fn(kernel)
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    spec = bound.config_spec
    default = spec.default_config()
    assert default.config["cute_mma_dag"] == "tcgen05"
    assert default.config["cute_mma_dag_loads"] == "tma"
    with bound.env:
        default = spec.normalized_config(default)
    out = _tensors(bound.compile_config(default)(*args))
    _assert_region_outputs(
        kernel, out, _run_triton(fn, args, default.config["block_sizes"])
    )


@pytest.mark.parametrize("kernel", ("bwd_o_kda", "state_du_kda", "fwd_o_anch"))
def test_region_programs_keep_the_warp_default(kernel: str) -> None:
    # The region programs outside the streaming class (a fragment operand
    # exchanged every iteration, a root tile) seed the warp mode first; the
    # tensor-memory seeds follow and compile.
    args = _m4_args(kernel, False)
    fn = _kernel_fn(kernel)
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    spec = bound.config_spec
    assert spec.default_config().config["cute_mma_dag"] == "warp"
    modes = [c.config["cute_mma_dag"] for c in spec.compiler_seed_configs]
    assert "tcgen05" in modes
    seed = next(
        c for c in spec.compiler_seed_configs if c.config["cute_mma_dag"] == "tcgen05"
    )
    with bound.env:
        seed = spec.normalized_config(seed)
    out = _tensors(bound.compile_config(seed)(*args))
    _assert_region_outputs(
        kernel, out, _run_triton(fn, args, seed.config["block_sizes"])
    )


def _probe_inputs(seed: int = 5) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    bhn, c, d, dv = 4, 64, 128, 64
    bf16, f32 = torch.bfloat16, torch.float32
    return {
        "q": _randn(g, bhn, c, d, dtype=bf16),
        "k": _randn(g, bhn, c, d, dtype=bf16),
        "v": _randn(g, bhn, c, dv, dtype=bf16),
        "do": _randn(g, bhn, c, dv, dtype=bf16),
        "h": _randn(g, bhn, d, dv, dtype=bf16),
        "g": _randn(g, bhn, d, dv, dtype=f32),
        "a32": _randn(g, bhn, c, dv, dtype=f32),
        "b32": _randn(g, bhn, c, dv, dtype=f32),
    }


# Chain probes: (fn, args, block sizes, warps, stages, inexact outputs)
_CHAIN_PROBES = (
    ("h1", _probe_h1, ("q", "k", "v", "do"), [64, 64, 64], 4, 2, set()),
    ("h1", _probe_h1, ("q", "k", "v", "do"), [32, 32, 64], 8, 1, set()),
    ("h3", _probe_h3, ("do", "h", "g"), [64, 64], 4, 2, {1}),
    ("h3", _probe_h3, ("do", "h", "g"), [64, 32], 4, 1, {1}),
    ("h4", _probe_h4, ("a32", "b32", "q", "k"), [64, 64], 4, 2, set()),
    ("h4", _probe_h4, ("a32", "b32", "q", "k"), [32, 32], 8, 1, set()),
)


@pytest.mark.parametrize(
    ("name", "fn", "keys", "block_sizes", "warps", "stages", "inexact"),
    _CHAIN_PROBES,
    ids=[f"{p[0]}-{'x'.join(map(str, p[3]))}w{p[4]}s{p[5]}" for p in _CHAIN_PROBES],
)
def test_chain_probes_match_the_triton_backend(
    name: str,
    fn: Callable[..., object],
    keys: tuple[str, ...],
    block_sizes: list[int],
    warps: int,
    stages: int,
    inexact: set[int],
) -> None:
    # h1 -- a root value read by an inner loop's dot must survive every
    # iteration of the enclosing loop (D / bd = 2 here: the second iteration
    # read the fresh dot that shared its columns); h3 -- a tile-only derived
    # value reduced behind a nested loop takes tensor memory (the render
    # crashed); h4 -- a bf16 tile rewritten in place behind a tf32 dot is
    # unpacked as bf16 (it was unpacked as f16).
    t = _probe_inputs()
    args = tuple(t[key] for key in keys)
    source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    if name == "h4":
        assert f"{RT}.prims.Tcgen05MMAKind.TF32" in source
        assert "unpack_bf16x2(" in source and "unpack_f16x2(" not in source
    ref = _run_triton(fn, args, block_sizes)
    _assert_close(out, ref)
    for index, (lhs, rhs) in enumerate(zip(out, ref, strict=True)):
        if index not in inexact:
            assert torch.equal(lhs, rhs), index


def _launches(
    fn: Callable[..., object],
    args: tuple[object, ...],
    config: helion.Config,
    count: int,
) -> tuple[str, list[tuple[torch.Tensor, ...]]]:
    """``count`` launches of the pinned tcgen05 config (every output
    tuple) and the source."""
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    assert normalized.config["cute_mma_dag"] == "tcgen05", normalized
    compiled = bound.compile_config(normalized)
    outputs = [_tensors(compiled(*args)) for _ in range(count)]
    torch.cuda.synchronize()
    return bound.to_code(normalized), outputs


def _probe_kv(bh: int, n: int, d: int, dv: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(0)
    return (
        _randn(g, bh, n, 64, d, scale=0.25, dtype=torch.bfloat16),
        _randn(g, bh, n, 64, dv, scale=0.25, dtype=torch.bfloat16),
    )


def test_tensors_accessed_after_their_snapshot_see_the_stored_values() -> None:
    # By TMA the snapshot's global writes completed at the kernel end only.
    # The epilogue re-reading the last snapshot (snap_reread) now declines the
    # lowering altogether (a load sharing storage with a store: no dag knob;
    # the cute backend's tile lowering declines the program as well, so nothing
    # silently falls back); the epilogue's own store of the snapshot tile
    # (snap_same_dst, vs Triton) keeps the packet drain.
    args = _probe_kv(128, 8, 64, 64)
    config = _config([64, 16], 4, 2)
    bound = helion.kernel(_snap_reread, backend="cute", static_shapes=True).bind(args)
    assert bound.config_spec.cute_mma_dag is None
    assert bound.config_spec.cute_mma_dag_facts is None
    source, outputs = _launches(_snap_same_dst, args, config, 20)
    assert f"{RT}.prims.cp_async_bulk_tensor_global_shared_cta(" not in source
    (ref,) = _run_triton(_snap_same_dst, args, [64, 16])
    for (out,) in outputs:
        assert torch.equal(out, ref)


@pytest.mark.parametrize("stages", (1, 2))
def test_trailing_tma_stores_match_the_triton_backend(stages: int) -> None:
    # The trailing block's bulk store now waits for the previous iteration's
    # group in the loop (ordered before the stage-in that reuses its buffer).
    args = _probe_kv(128, 8, 64, 64)
    for cols in (16, 64):
        source, outputs = _launches(
            _store_after, args, _config([64, cols], 4, stages), 20
        )
        assert source.count(f"{RT}.prims.cp_async_bulk_wait_group(1, read=True)") == 1
        (ref,) = _run_triton(_store_after, args, [64, cols])
        for (out,) in outputs:
            assert torch.equal(out, ref)


@pytest.mark.parametrize(
    ("block_sizes", "warps", "stages"),
    (([1, 64, 64], 4, 2), ([1, 64, 32], 4, 1), ([1, 32, 64], 8, 2)),
)
def test_fp32_gla_output_reads_h_mn_major_by_tma(
    block_sizes: list[int], warps: int, stages: int
) -> None:
    # The GLA example over fp32 inputs: ``h[bhn, tile_d, tile_dv]`` is an
    # fp32 source read with K along its rows -- an MN-major tf32 B tile
    # fetched by TMA under the 32-byte-unit swizzle (box of 32 columns per
    # slab) and read with SBO 512.  Bit-equal to Triton (tcgen05 tf32 over
    # the same K order), deterministic over the launches.
    g = torch.Generator(device=DEVICE).manual_seed(11)
    bhn, c, d, dv = 8, 64, 128, 128
    f32 = torch.float32
    gc = -_randn(g, bhn, c, d, scale=0.1, dtype=f32).abs().cumsum(1)
    args = (
        _randn(g, bhn, c, d, dtype=f32),
        _randn(g, bhn, c, dv, dtype=f32),
        gc,
        _randn(g, bhn, d, dv, dtype=f32),
        _randn(g, bhn, c, c, scale=0.3, dtype=f32),
        0.125,
    )
    fn = _kernel_fn("fwd_o_anch")
    source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    assert "'s128b_atom_32b'" in source
    assert f"{RT}.prims.Tcgen05SmemSwizzle.SWIZZLE_128B_ATOM_32B" in source
    ref = _run_triton(fn, args, block_sizes)
    for lhs, rhs in zip(out, ref, strict=True):
        assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("cols", (16, 32))
def test_snapshot_rereads_through_views_see_the_stores(cols: int) -> None:
    # A host-side view of the snapshot tensor read after the loop, and an
    # epilogue derived value over the snapshot tile: both load a tensor the
    # loop stores, so the matcher declines the lowering in both modes -- no dag
    # knob, no facts.
    args = _probe_kv(128, 8, 64, 64)
    for fn in (_snap_reread_view, _snap_reread2):
        bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
        assert bound.config_spec.cute_mma_dag is None, fn.__name__
        assert bound.config_spec.cute_mma_dag_facts is None, fn.__name__
        with bound.env, pytest.raises(InvalidConfig):
            bound.config_spec.normalize(
                dict(_config([64, cols], 4, 2).config), _fix_invalid=False
            )


_PASS_WARP_CASES = (
    ("fwd_h_delta", True, (8, 128, 64), 32, 4),
    ("fwd_h_delta", True, (8, 128, 64), 32, 2),
    ("fwd_h_delta", True, (1, 128, 64), 32, 4),
    ("fwd_h_delta", True, (2, 128, 128), 64, 3),
    ("dstate", True, (8, 128, 64), 32, 2),
    ("dstate", True, (2, 128, 64), 32, 3),
    ("dstate", True, (1, 128, 128), 64, 2),
    ("bwd", "scalar", (8, 64, 64), 64, 4),
    ("bwd", "scalar", (1, 64, 64), 64, 2),
    ("bwd", "scalar", (64, 128, 128), 64, 8),
    # The K-loop kernels' fp32 dh convert pass (a raw tile beside the operand).
    ("dqk", "vanilla", (6, 128, 128), 64, 3),
    ("dqk", "gated", (6, 128, 128), 64, 2),
    ("dv", "vanilla", (6, 128, 128), 64, 2),
)


@pytest.mark.parametrize(
    ("kernel", "mode", "shape", "bn", "stages"),
    _PASS_WARP_CASES,
    ids=[
        f"{k}-{m}-n{s[0]}d{s[1]}dv{s[2]}-{bn}s{st}"
        for k, m, s, bn, st in _PASS_WARP_CASES
    ],
)
def test_pass_warps_match_the_chain_passes(
    kernel: str, mode: object, shape: tuple[int, int, int], bn: int, stages: int
) -> None:
    # ``cute_mma_dag_pass_warps='on'``: warps 4-7 rewrite the operand tiles
    # one tile ahead and hand them over through the pass barriers.  Same
    # bytes written in the same order per tile as the chain's own passes:
    # bit-equal to the eight-warp render without pass warps and to the
    # four-warp one, deterministic over the launches, within the Triton
    # tolerance (bit-equal where the chain is); trips 1, 2 and N, stages
    # 2, 3, 4 and 8.
    n, d, dv = shape
    if kernel == "bwd":
        t = _inputs(bh=2, n=n, d=d, dv=dv, dtype=torch.bfloat16)
        args = _bwd_args(t, cast("str", mode))
        block_sizes = [bn, bn]
    elif kernel in ("dqk", "dv"):
        t = _inputs3(bhn=n, d=d, dv=dv, dtype=torch.bfloat16)
        args = (_dqk_args if kernel == "dqk" else _dv_args)(t, cast("str", mode))
        block_sizes = [1, bn, 128 if kernel == "dv" else bn]
        if kernel == "dv":
            block_sizes = [1, 128, bn]
    else:
        t = _inputs4(bh=2, n=n, d=d, dv=dv, dtype=torch.bfloat16)
        args = (_fwd_h_delta_args if kernel == "fwd_h_delta" else _dstate_args)(
            t, cast("bool", mode)
        )
        block_sizes = [bn]
    fn = _kernel_fn(kernel)
    source, out = _run_cute(
        fn, args, _config(block_sizes, 8, stages, cute_mma_dag_pass_warps="on")
    )
    assert "for md_j in cutlass.range(" in source
    assert f"{RT}.prims.mbarrier_arrive(md_pass + md_stage)" in source
    assert "barrier_id=1, number_of_threads=128" in source
    for warps in (8, 4):
        _source, other = _run_cute(fn, args, _config(block_sizes, warps, stages))
        for lhs, rhs in zip(out, other, strict=True):
            assert torch.equal(lhs, rhs)
    _assert_close(out, _run_triton(fn, args, block_sizes))


def test_in_loop_vector_stores_run_on_the_warp_mode() -> None:
    # The single-loop tcgen05 emitter dropped an in-loop vector store with its
    # reduction and load (s_out stayed zero); such programs decline the mode
    # now and the warp mode renders them, matching Triton.
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16)
    g = torch.Generator(device=DEVICE).manual_seed(5)
    s_in = _randn(g, 2, 64, dtype=torch.float32)
    args = (
        0.5,
        t["k"],
        t["v"],
        s_in,
        torch.zeros(2, 64, dtype=torch.float32, device=DEVICE),
    )
    bound = helion.kernel(_rmw_scalar_first, backend="cute", static_shapes=True).bind(
        args
    )
    assert bound.config_spec.cute_mma_dag is not None
    assert bound.config_spec.cute_mma_dag.choices == ("off", "warp")
    config = helion.Config(
        block_sizes=[64, 64],
        cute_mma_dag="warp",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
    )
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    assert normalized.config["cute_mma_dag"] == "warp", normalized
    compiled = bound.compile_config(normalized)
    outs = []
    for _ in range(LAUNCHES):
        args[4].zero_()
        h_out = compiled(*args)
        torch.cuda.synchronize()
        outs.append((h_out.clone(), args[4].clone()))
    for h_out, s_out in outs[1:]:
        assert torch.equal(h_out, outs[0][0]) and torch.equal(s_out, outs[0][1])
    ref_args = (
        0.5,
        t["k"],
        t["v"],
        s_in,
        torch.zeros(2, 64, dtype=torch.float32, device=DEVICE),
    )
    tb = helion.kernel(_rmw_scalar_first, backend="triton", static_shapes=True).bind(
        ref_args
    )
    h_ref = tb.compile_config(helion.Config(block_sizes=[64, 64]))(*ref_args)
    torch.cuda.synchronize()
    assert torch.equal(outs[0][0], h_ref)
    # The row sum's fp32 products are fused into FMAs by the cute lowering and
    # rounded first by Triton >= 3.9; over 64 cancelling terms of magnitude
    # ~30 that is a few 1e-6 of absolute difference.
    torch.testing.assert_close(outs[0][1], ref_args[4], atol=1e-4, rtol=1e-4)
    assert outs[0][1].abs().sum() > 0


@pytest.mark.parametrize(("n", "stages"), ((64, 2), (256, 3)))
def test_sync_route_pass_tiles_are_paced_by_the_free_barrier(
    n: int, stages: int
) -> None:
    # With k and v sliced out of 80-wide rows the scalar forward's passes copy
    # from global memory and never arm the ring barrier; the pass warps ran
    # unpaced (0/10 launches right at trips 64 / stages 2, 6/10 at 256 / 3).
    # Paced by the slot's free barrier they match Triton bit for bit over ten
    # launches.
    t = _inputs(bh=4, n=n, d=64, dv=64, dtype=torch.bfloat16, seed=1)

    def unalign(x: torch.Tensor) -> torch.Tensor:
        wide = torch.zeros((*x.shape[:-1], 80), dtype=x.dtype, device=x.device)
        wide[..., 4:68] = x
        return wide[..., 4:68]

    t["k"], t["v"] = unalign(t["k"]), unalign(t["v"])
    args = _fwd_args(t, "scalar", True)
    fn = _kernel_fn("fwd")
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    with bound.env:
        normalized = bound.config_spec.normalized_config(
            _config([64, 64], 8, stages, cute_mma_dag_pass_warps="on")
        )
    assert normalized.config["cute_mma_dag"] == "tcgen05"
    assert normalized.config["cute_mma_dag_pass_warps"] == "on"
    source = bound.to_code(normalized)
    pass_loop = source.split("if md_warp >= 4:", 1)[1].split("\n    else:\n", 1)[0]
    assert f"{RT}.mbar_wait_sleep(md_free + md_stage, md_round)" in pass_loop
    assert "md_full" not in pass_loop and f"{RT}.ldg_" not in pass_loop
    compiled = bound.compile_config(normalized)
    outputs = [_tensors(compiled(*args)) for _ in range(10)]
    torch.cuda.synchronize()
    ref = _run_triton(fn, args, [64, 64])
    for out in outputs:
        for lhs, rhs in zip(out, ref, strict=True):
            assert torch.equal(lhs, rhs)


def _regions_probe_inputs(seed: int = 6) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    bhn, c, d, dv = 16, 64, 128, 64
    bf16, f32 = torch.bfloat16, torch.float32
    return {
        "q": _randn(g, bhn, c, d, scale=0.25, dtype=bf16),
        "k": _randn(g, bhn, c, d, scale=0.25, dtype=bf16),
        "q2": _randn(g, bhn, c, d, scale=0.25, dtype=bf16),
        "s": _randn(g, bhn, c, dv, scale=0.25, dtype=bf16),
        "t": _randn(g, bhn, c, dv, scale=0.25, dtype=bf16),
        "u": _randn(g, bhn, d, dv, scale=0.25, dtype=bf16),
        "w": _randn(g, bhn, d, dv, scale=0.25, dtype=bf16),
        "x64": _randn(g, bhn, c, dv, dtype=f32),
        "x64b": _randn(g, bhn, c, dv, dtype=f32),
        "x64c": _randn(g, bhn, c, dv, dtype=f32),
        "x128": _randn(g, bhn, c, d, dtype=f32),
        "x512": _randn(g, bhn, c, 512, dtype=f32),
        "y512": _randn(g, bhn, c, 512, dtype=f32),
    }


# Region probes: (name, fn, arg keys, block sizes, warps, stages)
_REGION_PROBES = (
    ("hoist", _probe_hoist, ("q", "k", "s", "x64"), [64, 128], 4, 2),
    ("hoist", _probe_hoist, ("q", "k", "s", "x64"), [64, 64], 4, 2),
    ("hoist_loop", _probe_hoist_loop, ("s", "u", "w", "x128"), [32, 64], 4, 2),
    ("hoist_loop", _probe_hoist_loop, ("s", "u", "w", "x128"), [64, 64], 4, 2),
    ("rawlate", _probe_rawlate, ("q", "k", "x64", "q2"), [64, 64], 4, 2),
    ("rawlate", _probe_rawlate, ("q", "k", "x64", "q2"), [32, 64], 4, 2),
    ("rawlate", _probe_rawlate, ("q", "k", "x64", "q2"), [64, 64], 4, 1),
    ("carry2", _probe_carry2, ("s", "t", "q", "x64"), [64, 64], 4, 2),
    ("carry2", _probe_carry2, ("s", "t", "q", "x64"), [32, 64], 4, 2),
    ("kk", _probe_kk, ("q", "k"), [64, 128], 4, 2),
    (
        "recompute",
        _probe_recompute,
        ("q", "k", "s", "t", "x64", "x64b", "x64c", "s"),
        [64, 128],
        4,
        2,
    ),
    (
        "recompute",
        _probe_recompute,
        ("q", "k", "s", "t", "x64", "x64b", "x64c", "s"),
        [64, 64],
        4,
        1,
    ),
    ("drainrace", _probe_drainrace, ("q", "k", "s", "t"), [64, 64], 4, 2),
    ("drainrace", _probe_drainrace, ("q", "k", "s", "t"), [32, 128], 4, 3),
    ("dotless_store", _probe_dotless_store, ("q", "k", "x512", "y512"), [64, 64], 4, 2),
    ("dotless_store", _probe_dotless_store, ("q", "k", "x512", "y512"), [64, 32], 4, 2),
    ("register_carry", _probe_register_carry, ("q", "k", "s", "x64"), [64, 64], 4, 2),
    ("register_carry", _probe_register_carry, ("q", "k", "s", "x64"), [32, 64], 8, 2),
    ("register_carry", _probe_register_carry, ("q", "k", "s", "x64"), [64, 32], 4, 1),
    ("register_carry_budget", _probe_register_carry_budget, ("k",), [64], 4, 2),
    ("register_carry_budget", _probe_register_carry_budget, ("k",), [32], 8, 4),
)


@pytest.mark.parametrize(
    ("name", "fn", "keys", "block_sizes", "warps", "stages"),
    _REGION_PROBES,
    ids=[f"{p[0]}-{'x'.join(map(str, p[3]))}w{p[4]}s{p[5]}" for p in _REGION_PROBES],
)
def test_region_probes_match_the_triton_backend(
    name: str,
    fn: Callable[..., object],
    keys: tuple[str, ...],
    block_sizes: list[int],
    warps: int,
    stages: int,
) -> None:
    # hoist / hoist_loop -- a derived value independent of the pending batch is
    # stored to tensor memory ahead of the batch's operand reads, so it must
    # not share the columns of a slot dying at that batch (a was fully wrong);
    # rawlate -- a root raw tile overlaying the loop frames converts before the
    # first loop's body refills its bytes (r was fully wrong); carry2 / kk -- a
    # 128-column tile sharing a 64-column tile's tensor map loads both slabs (f
    # was wrong, kk NaN); recompute -- a derived value whose store is deferred
    # behind a batch takes tensor memory instead of a recomputation from an
    # overwritten slot (d was (s0 * z) * x); drainrace / dotless_store -- the
    # TMA store groups finish reading their staging buffers before a loop's
    # prologue lands on them and before a dot-less loop reuses them (latent
    # races).  Every output is bit-equal to Triton over eight launches, except
    # the register carry's fp32 product, which the cute lowering fuses into
    # an FMA and Triton >= 3.9 rounds first (one fp32 rounding apart).
    t = _regions_probe_inputs()
    args = tuple(t[key] for key in keys)
    _source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    ref = _run_triton(fn, args, block_sizes)
    for index, (lhs, rhs) in enumerate(zip(out, ref, strict=True)):
        if name == "register_carry_budget" and index == 0:
            atol = 1e-6 * rhs.abs().max().item()
            torch.testing.assert_close(lhs, rhs, atol=atol, rtol=1e-5)
        else:
            assert torch.equal(lhs, rhs), (name, index)


@pytest.mark.parametrize(
    ("kernel", "gated", "block_sizes", "warps", "stages", "extra"),
    (
        ("wy_dL", False, [32, 64, 128], 4, 2, {}),
        ("wy_dL", False, [64, 64, 128], 4, 2, {}),
        ("wy_dL", True, [32, 64, 128], 4, 2, {}),
        ("wy_dL", True, [64, 64, 128], 4, 4, {}),
        ("dqkw", True, [128, 64, 64, 64, 128], 4, 3, {"cute_mma_dag_exchange": "smem"}),
    ),
    ids=[
        "wy_dL-32x64x128w4s2",
        "wy_dL-64x64x128w4s2",
        "wy_dL-gated-32x64x128w4s2",
        "wy_dL-gated-64x64x128w4s4",
        "dqkw-gated-128x64x64x64x128w4s3",
    ],
)
def test_wide_tiles_sharing_a_tensor_map_match_the_triton_backend(
    kernel: str,
    gated: bool,
    block_sizes: list[int],
    warps: int,
    stages: int,
    extra: dict[str, object],
) -> None:
    # A 128-column tile of a tensor another loop loads in 64-column tiles
    # shares their tensor map (the box is one swizzle atom) and loaded one
    # slab: wy_dL at [.., 64, 128] failed the repeat-launch check (dvni half
    # loaded), and a cold autotune rejected these configs (dqkw's dk_as 3.8 M
    # elements wrong, gated wy_dL's dbeta / dg_wy all NaN).  Fragment outputs
    # bit-equal to Triton, the vectors by tolerance, eight launches each.
    args = _m4_args(kernel, gated)
    fn = _kernel_fn(kernel)
    _source, out = _run_cute(fn, args, _config(block_sizes, warps, stages, **extra))
    _assert_region_outputs(kernel, out, _run_triton(fn, args, block_sizes))


@pytest.mark.parametrize(
    ("gated", "block_sizes", "warps", "stages", "extra"),
    (
        (False, [64, 64, 64], 4, 2, {}),
        (False, [64, 64, 64], 4, 1, {"cute_mma_dag_exchange": "smem"}),
        (False, [128, 32, 32], 4, 3, {}),
        (False, [128, 32, 32], 4, 2, {}),
        (True, [64, 64, 64], 4, 2, {}),
        (True, [64, 32, 128], 8, 1, {"cute_mma_dag_exchange": "smem"}),
        (True, [128, 32, 32], 4, 3, {}),
        (True, [128, 32, 32], 4, 2, {}),
    ),
    ids=[
        "64x64x64w4s2",
        "64x64x64w4s1-smem",
        "128x32x32w4s3",
        "128x32x32w4s2",
        "gated-64x64x64w4s2",
        "gated-64x32x128w8s1-smem",
        "gated-128x32x32w4s3",
        "gated-128x32x32w4s2",
    ],
)
def test_the_chains_squared_carry_takes_its_own_columns(
    gated: bool,
    block_sizes: list[int],
    warps: int,
    stages: int,
    extra: dict[str, object],
) -> None:
    # wy's Apow = dot(Apow, Apow) writes the carry's own columns (no copy
    # at the iteration's end) and the seed kk shares them: 128 TMEM columns
    # instead of 256, bit-equal to Triton over eight launches in both decay
    # modes and both exchange modes; [128, 32, 32] w4 s2 is the promoted
    # default (four CTAs per SM).
    args = _m4_args("wy", gated)
    fn = _kernel_fn("wy")
    source, out = _run_cute(fn, args, _config(block_sizes, warps, stages, **extra))
    assert f"{RT}.prims.tcgen05_alloc(md_tslot, 128," in source
    _assert_region_outputs("wy", out, _run_triton(fn, args, block_sizes))


@pytest.mark.parametrize(
    ("block_sizes", "warps", "stages"), (([128, 64], 4, 4), ([128, 32], 8, 2))
)
def test_a_16_bit_matmul_with_a_tile_index_mask_matches_the_triton_backend(
    block_sizes: list[int], warps: int, stages: int
) -> None:
    # The sparse-attention indexer's logits in tensor memory: each head's
    # bf16 score read rounded from TMEM, masked by ``tile_n.index``.
    fn = _probe_kernels()._head_sum_logits
    args = _head_sum_args()
    _source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    (ref,) = _run_triton(fn, args, block_sizes)
    assert torch.equal(out[0], ref)
