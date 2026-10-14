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

from test.test_cute_mma_dag_gpu import _assert_close
from test.test_cute_mma_dag_gpu import _bwd_args
from test.test_cute_mma_dag_gpu import _dqk_args
from test.test_cute_mma_dag_gpu import _dv_args
from test.test_cute_mma_dag_gpu import _fwd_args
from test.test_cute_mma_dag_gpu import _fwd_o_args
from test.test_cute_mma_dag_gpu import _inputs
from test.test_cute_mma_dag_gpu import _inputs3
from test.test_cute_mma_dag_gpu import _kernel_fn
from test.test_cute_mma_dag_gpu import _run_triton
from test.test_cute_mma_dag_gpu import _tensors

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
