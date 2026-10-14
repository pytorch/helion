"""Runtime coverage of the ``cute_mma_dag`` tcgen05 compute mode: the
linear-attention engine's serial state recurrences (``chunk_fwd_h_diag_fused``
/ ``chunk_bwd_dh_diag_fused``) in every decay mode against the Triton
backend, repeat-launch determinism, and the stage / store / warp knobs being
pure data movement.  sm100 / sm103 only."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

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
from test.test_cute_mma_dag_tcgen05 import _probe_h1
from test.test_cute_mma_dag_tcgen05 import _probe_h3
from test.test_cute_mma_dag_tcgen05 import _probe_h4
from test.test_cute_mma_dag_tcgen05 import _store_first

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

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
    "kernel", ("dk", "dqkw", "wy_dL", "bwd_o_kda", "state_du_kda", "fwd_o_anch")
)
def test_region_programs_keep_the_warp_default(kernel: str) -> None:
    # The region programs seed the warp mode first (their tensor-memory
    # chain of commit waits runs behind it on these small CTAs); the
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
