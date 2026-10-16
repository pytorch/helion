"""Runtime coverage of the ``cute_mma_dag`` warp mode: the linear-attention
engine's serial state recurrences (``chunk_fwd_h_diag_fused`` /
``chunk_bwd_dh_diag_fused``) and K-loop + epilogue kernels
(``chunk_fwd_o_helion`` / ``chunk_bwd_dqk_helion`` / ``chunk_bwd_dv_helion``)
in every decay mode against the Triton backend, repeat-launch determinism,
and the stage / store / exchange knobs being pure data movement."""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

import pytest
import torch

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
]

RT = "_helion_mma_dag"
LAUNCHES = 8


def _engine() -> object:
    return importlib.import_module("examples.linear.linear_attention_engine")


def _randn(
    generator: torch.Generator, *shape: int, scale: float = 1.0, dtype: torch.dtype
):
    return (
        torch.randn(shape, device=DEVICE, dtype=torch.float32, generator=generator)
        * scale
    ).to(dtype)


def _inputs(
    *, bh: int, n: int, d: int, dv: int, dtype: torch.dtype, seed: int = 0
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = 64
    f32 = torch.float32
    return {
        "k": _randn(g, bh, n, c, d, dtype=dtype),
        "v": _randn(g, bh, n, c, dv, dtype=dtype),
        "g_last": _randn(g, bh, n, d, scale=0.1, dtype=f32),
        "g_last_s": _randn(g, bh, n, scale=0.1, dtype=f32),
        "gc": _randn(g, bh, n, c, scale=0.1, dtype=f32),
        "h0": _randn(g, bh, d, dv, dtype=dtype),
        "dh_init": _randn(g, bh, d, dv, dtype=f32),
    }


def _inputs3(
    *, bhn: int, d: int, dv: int, dtype: torch.dtype, seed: int = 0
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = 64
    f32 = torch.float32
    return {
        "q": _randn(g, bhn, c, d, dtype=dtype),
        "k": _randn(g, bhn, c, d, dtype=dtype),
        "v": _randn(g, bhn, c, dv, dtype=dtype),
        "do": _randn(g, bhn, c, dv, dtype=dtype),
        "h": _randn(g, bhn, d, dv, dtype=dtype),
        "dh": _randn(g, bhn, d, dv, dtype=f32),
        "gcs": _randn(g, bhn, c, scale=0.1, dtype=f32),
        "gl": _randn(g, bhn, scale=0.1, dtype=f32),
    }


def _fwd_args(
    t: dict[str, torch.Tensor], mode: str, has_h0: bool
) -> tuple[object, ...]:
    h0 = t["h0"] if has_h0 else None
    if mode == "vanilla":
        return (t["k"], t["v"], None, h0, None, False, False, False, has_h0)
    if mode == "diag":
        return (t["k"], t["v"], t["g_last"], h0, None, True, False, False, has_h0)
    return (t["k"], t["v"], t["g_last_s"], h0, t["gc"], True, True, False, has_h0)


def _bwd_args(t: dict[str, torch.Tensor], mode: str) -> tuple[object, ...]:
    if mode == "vanilla":
        return (t["k"], t["v"], None, t["dh_init"], None, False, False, False, 0.125)
    if mode == "diag":
        return (
            t["k"],
            t["v"],
            t["g_last"],
            t["dh_init"],
            None,
            True,
            False,
            False,
            1.0,
        )
    return (
        t["k"],
        t["v"],
        t["g_last_s"],
        t["dh_init"],
        t["gc"],
        True,
        True,
        False,
        0.125,
    )


def _fwd_o_args(t: dict[str, torch.Tensor], mode: str) -> tuple[object, ...]:
    gated = mode != "vanilla"
    return (t["q"], t["k"], t["v"], t["gcs"] if gated else None, t["h"], gated, 0.125)


def _dqk_args(t: dict[str, torch.Tensor], mode: str) -> tuple[object, ...]:
    gated = mode != "vanilla"
    return (
        t["q"],
        t["k"],
        t["v"],
        t["gcs"] if gated else None,
        t["gl"] if gated else None,
        t["h"],
        t["do"],
        t["dh"],
        gated,
        0.125,
    )


def _dv_args(t: dict[str, torch.Tensor], mode: str) -> tuple[object, ...]:
    if mode == "vanilla":
        return (
            t["q"],
            t["k"],
            t["k"],
            None,
            t["do"],
            t["dh"],
            None,
            None,
            False,
            False,
            False,
            0.125,
        )
    if mode == "diag":
        return (
            t["q"],
            t["k"],
            t["k"],
            t["gcs"],
            t["do"],
            t["dh"],
            None,
            None,
            True,
            False,
            False,
            0.125,
        )
    return (
        t["q"],
        t["k"],
        t["k"],
        t["gcs"],
        t["do"],
        t["dh"],
        t["gl"],
        None,
        True,
        True,
        False,
        0.125,
    )


def _inputs4(
    *, bh: int, n: int, d: int, dv: int, dtype: torch.dtype, seed: int = 0
) -> dict[str, torch.Tensor]:
    """The delta-rule state kernels' inputs, scaled so the recurrences stay
    bounded over 64 chunks (the raw delta update grows geometrically) and
    with contracting log-decays as the engine produces them."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = 64
    f32 = torch.float32
    return {
        "q": _randn(g, bh, n, c, d, scale=0.05, dtype=dtype),
        "k": _randn(g, bh, n, c, d, scale=0.05, dtype=dtype),
        "w": _randn(g, bh, n, c, d, scale=0.05, dtype=dtype),
        "u": _randn(g, bh, n, c, dv, dtype=dtype),
        "do": _randn(g, bh, n, c, dv, dtype=dtype),
        "h0": _randn(g, bh, d, dv, scale=0.5, dtype=dtype),
        "g_cs": -_randn(g, bh, n, c, scale=0.1, dtype=f32).abs(),
        "dl": -_randn(g, bh, n, scale=0.2, dtype=f32).abs(),
    }


def _fwd_h_delta_args(t: dict[str, torch.Tensor], gated: bool) -> tuple[object, ...]:
    g_cs, dl = (t["g_cs"], t["dl"]) if gated else (None, None)
    return (t["k"], t["w"], t["u"], t["h0"], g_cs, dl, gated, False, False)


def _dstate_args(t: dict[str, torch.Tensor], gated: bool) -> tuple[object, ...]:
    g_cs, dl = (t["g_cs"], t["dl"]) if gated else (None, None)
    return (t["q"], t["k"], t["w"], t["do"], t["h0"], g_cs, dl, gated, 0.125)


_KERNELS = {
    "fwd": "chunk_fwd_h_diag_fused",
    "bwd": "chunk_bwd_dh_diag_fused",
    "fwd_o": "chunk_fwd_o_helion",
    "dqk": "chunk_bwd_dqk_helion",
    "dv": "chunk_bwd_dv_helion",
    "fwd_h_delta": "chunk_fwd_h_delta_helion",
    "dstate": "chunk_bwd_dstate_delta_helion",
}


def _kernel_fn(kernel: str) -> Callable[..., object]:
    return getattr(_engine(), _KERNELS[kernel]).fn


def _config(
    block_sizes: list[int], warps: int, stages: int, **extra: object
) -> helion.Config:
    return helion.Config(
        block_sizes=block_sizes,
        cute_mma_dag="warp",
        cute_mma_dag_warps=warps,
        cute_mma_dag_stages=stages,
        **extra,
    )


def _tensors(result: object) -> tuple[torch.Tensor, ...]:
    return result if isinstance(result, tuple) else (cast_tensor(result),)


def cast_tensor(value: object) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    return value


def _run_cute(
    fn: Callable[..., object], args: tuple[object, ...], config: helion.Config
) -> tuple[str, tuple[torch.Tensor, ...]]:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert f"{RT}.mma_" in source, source
    assert "_cute_grouped_reduce_shared_two_stage" not in source
    compiled = bound.compile_config(normalized)
    outputs = [_tensors(compiled(*args)) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    for other in outputs[1:]:
        for lhs, rhs in zip(outputs[0], other, strict=True):
            assert torch.equal(lhs, rhs)
    return source, outputs[0]


def _run_triton(
    fn: Callable[..., object], args: tuple[object, ...], block_sizes: list[int]
) -> tuple[torch.Tensor, ...]:
    kernel = helion.kernel(fn, backend="triton", static_shapes=True)
    bound = kernel.bind(args)
    out = _tensors(bound.compile_config(helion.Config(block_sizes=block_sizes))(*args))
    torch.cuda.synchronize()
    return out


def _assert_close(
    outs: tuple[torch.Tensor, ...],
    refs: tuple[torch.Tensor, ...],
    *,
    scaled: bool = False,
) -> None:
    """``scaled``: the absolute tolerance follows the tensor's magnitude (a
    recurrence of cancelling fp32 sums rounded to bf16 leaves single
    elements a few bf16 ulps of the large terms apart between backends; the
    SIMT cute path shows the same elements against Triton)."""
    for out, ref in zip(outs, refs, strict=True):
        assert out.dtype == ref.dtype and out.shape == ref.shape
        if out.dtype == torch.float32:
            torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
        else:
            atol = 2e-2 * max(1.0, ref.float().abs().max().item()) if scaled else 2e-2
            torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=2e-2)


# kernel, mode, has_h0, (n, d, dv), dtype, bm, bn, warps, stages
_STATE_CASES = (
    ("fwd", "vanilla", False, (8, 64, 64), torch.bfloat16, 16, 8, 1, 1),
    ("fwd", "vanilla", False, (8, 64, 64), torch.bfloat16, 64, 64, 4, 2),
    ("fwd", "vanilla", False, (64, 64, 64), torch.bfloat16, 64, 64, 8, 4),
    ("fwd", "vanilla", False, (8, 128, 128), torch.bfloat16, 128, 128, 4, 2),
    ("fwd", "vanilla", False, (8, 128, 128), torch.bfloat16, 64, 128, 8, 3),
    ("fwd", "vanilla", False, (8, 64, 64), torch.float16, 64, 64, 4, 2),
    ("fwd", "vanilla", True, (8, 64, 64), torch.bfloat16, 32, 32, 2, 2),
    ("fwd", "diag", True, (8, 64, 64), torch.bfloat16, 64, 64, 4, 2),
    ("fwd", "diag", False, (8, 64, 64), torch.bfloat16, 32, 16, 2, 1),
    ("fwd", "diag", True, (64, 128, 128), torch.bfloat16, 64, 64, 8, 4),
    ("fwd", "scalar", True, (8, 64, 64), torch.bfloat16, 64, 64, 4, 2),
    ("fwd", "scalar", True, (8, 128, 128), torch.bfloat16, 128, 64, 8, 4),
    ("fwd", "scalar", False, (8, 64, 64), torch.float16, 32, 64, 4, 1),
    ("bwd", "vanilla", False, (8, 64, 64), torch.bfloat16, 64, 64, 4, 2),
    ("bwd", "vanilla", False, (8, 64, 64), torch.bfloat16, 16, 8, 1, 1),
    ("bwd", "vanilla", False, (64, 128, 128), torch.bfloat16, 64, 64, 8, 4),
    ("bwd", "diag", False, (8, 64, 64), torch.bfloat16, 64, 64, 4, 4),
    ("bwd", "diag", False, (8, 128, 128), torch.float16, 128, 128, 8, 2),
    ("bwd", "scalar", False, (8, 64, 64), torch.bfloat16, 64, 64, 8, 2),
    ("bwd", "scalar", False, (8, 128, 128), torch.bfloat16, 64, 128, 4, 1),
)


@pytest.mark.parametrize(
    ("kernel", "mode", "has_h0", "shape", "dtype", "bm", "bn", "warps", "stages"),
    _STATE_CASES,
    ids=[
        f"{k}-{m}{'-h0' if h else ''}-n{s[0]}d{s[1]}dv{s[2]}-{str(dt).split('.')[-1]}-{bm}x{bn}w{w}s{st}"
        for k, m, h, s, dt, bm, bn, w, st in _STATE_CASES
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


# kernel, mode, (d, dv), block_sizes (lead, root, loop), warps, stages, extra
_DAG_CASES = (
    ("fwd_o", "vanilla", (64, 64), [1, 64, 64], 4, 2, {}),
    ("fwd_o", "vanilla", (128, 128), [1, 128, 128], 8, 2, {}),
    ("fwd_o", "vanilla", (128, 128), [1, 128, 64], 8, 4, {}),
    ("fwd_o", "vanilla", (64, 128), [1, 32, 16], 8, 1, {}),
    ("fwd_o", "gated", (128, 128), [1, 128, 64], 8, 4, {}),
    ("fwd_o", "gated", (64, 64), [1, 64, 64], 4, 2, {"cute_mma_dag_exchange": "smem"}),
    ("dqk", "vanilla", (64, 64), [1, 64, 64], 4, 2, {}),
    ("dqk", "vanilla", (128, 128), [1, 128, 32], 8, 2, {}),
    ("dqk", "gated", (64, 64), [1, 64, 64], 8, 2, {}),
    ("dqk", "gated", (128, 64), [1, 64, 64], 4, 4, {"cute_mma_dag_exchange": "smem"}),
    ("dv", "vanilla", (64, 64), [1, 64, 64], 4, 2, {}),
    ("dv", "vanilla", (128, 128), [1, 128, 32], 8, 4, {}),
    ("dv", "diag", (128, 128), [1, 64, 128], 8, 2, {}),
    ("dv", "scalar", (64, 64), [1, 64, 64], 8, 2, {}),
    ("dv", "scalar", (64, 128), [1, 32, 64], 2, 1, {}),
)


@pytest.mark.parametrize(
    ("kernel", "mode", "shape", "block_sizes", "warps", "stages", "extra"),
    _DAG_CASES,
    ids=[
        f"{k}-{m}-d{s[0]}dv{s[1]}-{'x'.join(map(str, bs[1:]))}w{w}s{st}{'-' + '-'.join(e.values()) if e else ''}"
        for k, m, s, bs, w, st, e in _DAG_CASES
    ],
)
def test_k_loop_kernels_match_the_triton_backend(
    kernel: str,
    mode: str,
    shape: tuple[int, int],
    block_sizes: list[int],
    warps: int,
    stages: int,
    extra: dict[str, object],
) -> None:
    d, dv = shape
    t = _inputs3(bhn=6, d=d, dv=dv, dtype=torch.bfloat16)
    args = {"fwd_o": _fwd_o_args, "dqk": _dqk_args, "dv": _dv_args}[kernel](t, mode)
    fn = _kernel_fn(kernel)
    source, out = _run_cute(fn, args, _config(block_sizes, warps, stages, **extra))
    assert f"block=({32 * warps}, 1, 1)" in source
    _assert_close(out, _run_triton(fn, args, block_sizes))


# kernel, gated, (n, d, dv), bdv, warps, stages, extra
_DELTA_CASES = (
    ("fwd_h_delta", False, (8, 128, 32), 32, 4, 2, {}),
    ("fwd_h_delta", False, (8, 128, 64), 32, 8, 2, {}),
    ("fwd_h_delta", False, (64, 128, 128), 32, 4, 4, {}),
    ("fwd_h_delta", False, (8, 128, 128), 128, 8, 2, {}),
    ("fwd_h_delta", False, (8, 128, 64), 16, 2, 1, {}),
    ("fwd_h_delta", False, (8, 128, 64), 64, 8, 3, {}),
    ("fwd_h_delta", True, (8, 128, 64), 32, 8, 2, {}),
    ("fwd_h_delta", True, (64, 128, 128), 64, 8, 4, {}),
    ("fwd_h_delta", True, (8, 128, 32), 16, 4, 1, {}),
    ("fwd_h_delta", True, (8, 128, 128), 128, 8, 2, {}),
    ("dstate", False, (8, 128, 64), 32, 8, 2, {}),
    ("dstate", False, (64, 128, 128), 32, 8, 3, {}),
    ("dstate", False, (8, 128, 32), 16, 4, 2, {}),
    ("dstate", False, (8, 128, 64), 64, 8, 2, {}),
    ("dstate", False, (8, 128, 128), 64, 8, 1, {}),
    ("dstate", False, (8, 128, 64), 32, 4, 1, {}),
    ("dstate", True, (8, 128, 64), 32, 4, 2, {}),
    ("dstate", True, (8, 128, 64), 32, 8, 2, {}),
    ("dstate", True, (64, 128, 128), 32, 8, 3, {}),
    ("dstate", True, (8, 128, 128), 64, 8, 1, {}),
    # D == C == 64 (s0): the two ``:`` dims of q / k / w share one axis.
    ("fwd_h_delta", True, (8, 64, 64), 32, 4, 2, {}),
    ("dstate", False, (8, 64, 64), 32, 8, 2, {}),
)


@pytest.mark.parametrize(
    ("kernel", "gated", "shape", "bdv", "warps", "stages", "extra"),
    _DELTA_CASES,
    ids=[
        f"{k}{'-gated' if g else ''}-n{s[0]}d{s[1]}dv{s[2]}-{bdv}w{w}s{st}"
        for k, g, s, bdv, w, st, _e in _DELTA_CASES
    ],
)
def test_delta_state_recurrences_match_the_triton_backend(
    kernel: str,
    gated: bool,
    shape: tuple[int, int, int],
    bdv: int,
    warps: int,
    stages: int,
    extra: dict[str, object],
) -> None:
    n, d, dv = shape
    t = _inputs4(bh=2, n=n, d=d, dv=dv, dtype=torch.bfloat16)
    args = (_fwd_h_delta_args if kernel == "fwd_h_delta" else _dstate_args)(t, gated)
    fn = _kernel_fn(kernel)
    source, out = _run_cute(fn, args, _config([bdv], warps, stages, **extra))
    assert f"block=({32 * warps}, 1, 1)" in source
    _assert_close(out, _run_triton(fn, args, [bdv]), scaled=True)


@pytest.mark.parametrize("kernel", ("fwd_h_delta", "dstate"))
def test_delta_register_routes_are_pure_data_movement(kernel: str) -> None:
    # Four warps on 32-wide fragments: ``auto`` transposes the carry and the
    # derived value into B words in registers (movmatrix); ``smem`` exchanges
    # them through shared tiles.  Same bits, at every stage depth.
    t = _inputs4(bh=2, n=16, d=128, dv=64, dtype=torch.bfloat16, seed=5)
    args = (_fwd_h_delta_args if kernel == "fwd_h_delta" else _dstate_args)(t, True)
    fn = _kernel_fn(kernel)
    auto_source, auto = _run_cute(fn, args, _config([32], 4, 2))
    smem_source, smem = _run_cute(
        fn, args, _config([32], 4, 2, cute_mma_dag_exchange="smem")
    )
    assert f"{RT}.movmatrix_trans(" in auto_source
    assert f"{RT}.movmatrix_trans(" not in smem_source
    for lhs, rhs in zip(auto, smem, strict=True):
        assert torch.equal(lhs, rhs)
    for stages in (1, 3):
        for lhs, rhs in zip(
            _run_cute(fn, args, _config([32], 4, stages))[1], auto, strict=True
        ):
            assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("kernel", ("fwd", "bwd"))
def test_stage_depth_and_store_path_are_pure_data_movement(kernel: str) -> None:
    t = _inputs(bh=2, n=16, d=64, dv=64, dtype=torch.bfloat16, seed=3)
    args = _fwd_args(t, "diag", True) if kernel == "fwd" else _bwd_args(t, "scalar")
    fn = _kernel_fn(kernel)
    baseline = _run_cute(fn, args, _config([64, 64], 4, 1))[1]
    for stages in (2, 4):
        assert torch.equal(
            _run_cute(fn, args, _config([64, 64], 4, stages))[1][0], baseline[0]
        )
    source, direct = _run_cute(
        fn, args, _config([64, 64], 4, 2, cute_mma_dag_store="direct")
    )
    assert f"{RT}.stg_v4_b32(" not in source
    assert torch.equal(direct[0], baseline[0])
    staged = _run_cute(fn, args, _config([64, 64], 4, 2, cute_mma_dag_store="staged"))[
        0
    ]
    assert f"{RT}.stg_v4_b32(" in staged


@pytest.mark.parametrize("kernel", ("fwd_o", "dqk"))
def test_exchange_routes_are_pure_data_movement(kernel: str) -> None:
    # 4 warps on 64-wide fragments give every warp whole rows: ``auto``
    # repacks the attention / dA fragment into A words from registers,
    # ``smem`` exchanges through the shared bf16 tile.  Same bits.
    t = _inputs3(bhn=4, d=64, dv=64, dtype=torch.bfloat16, seed=5)
    args = {"fwd_o": _fwd_o_args, "dqk": _dqk_args}[kernel](t, "vanilla")
    fn = _kernel_fn(kernel)
    auto_source, auto = _run_cute(fn, args, _config([1, 64, 64], 4, 2))
    smem_source, smem = _run_cute(
        fn, args, _config([1, 64, 64], 4, 2, cute_mma_dag_exchange="smem")
    )
    assert "md_aw" in auto_source and "md_aw" not in smem_source
    for lhs, rhs in zip(auto, smem, strict=True):
        assert torch.equal(lhs, rhs)
    for stages in (1, 4):
        for lhs, rhs in zip(
            _run_cute(fn, args, _config([1, 64, 64], 4, stages))[1], auto, strict=True
        ):
            assert torch.equal(lhs, rhs)


def test_the_promoted_defaults_run_on_tensor_cores() -> None:
    capability = torch.cuda.get_device_capability(DEVICE)
    if capability[0] != 10:
        pytest.skip("the dag seed is promoted to the default on sm100 / sm103 only")
    t = _inputs(bh=2, n=8, d=128, dv=128, dtype=torch.bfloat16)
    t3 = _inputs3(bhn=6, d=128, dv=128, dtype=torch.bfloat16)
    t4 = _inputs4(bh=2, n=8, d=128, dv=128, dtype=torch.bfloat16)
    for kernel, args in (
        ("fwd", _fwd_args(t, "vanilla", False)),
        ("fwd_o", _fwd_o_args(t3, "vanilla")),
        ("dqk", _dqk_args(t3, "vanilla")),
        ("dv", _dv_args(t3, "vanilla")),
        ("fwd_h_delta", _fwd_h_delta_args(t4, False)),
        ("fwd_h_delta", _fwd_h_delta_args(t4, True)),
        ("dstate", _dstate_args(t4, False)),
        ("dstate", _dstate_args(t4, True)),
    ):
        fn = _kernel_fn(kernel)
        bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
        config = bound.config_spec.default_config()
        assert config.config["cute_mma_dag"] == "warp", kernel
        source = bound.to_code(config)
        assert f"{RT}.mma_bf16(" in source
        out = _tensors(bound.compile_config(config)(*args))
        _assert_close(
            out,
            _run_triton(fn, args, config.config["block_sizes"]),
            scaled=kernel in ("fwd_h_delta", "dstate"),
        )


def test_unaligned_operand_sources_take_the_synchronous_pass() -> None:
    # A source whose base is 8-byte aligned cannot be staged with 16-byte
    # cp.async packets: the operand goes through the cooperative pass of
    # pair loads (same numerics, no tensor-core change).
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=5)
    k_wide = torch.randn((2, 8, 64, 80), device=DEVICE, dtype=torch.bfloat16)
    k = k_wide[..., 4:68]
    assert k.data_ptr() % 16 == 8 and k.stride(2) == 80
    fn = _kernel_fn("fwd")
    args = (k, t["v"], None, None, None, False, False, False, False)
    source, out = _run_cute(fn, args, _config([64, 64], 4, 2))
    assert f"{RT}.ldg_b32(" in source and f"{RT}.sts_b32(" in source
    assert source.count(f"{RT}.cp_async_16(") == 8
    _assert_close(out, _run_triton(fn, args, [64, 64]))


# --------------------------------------------------------------------------
# Probes: fragments with several readers, chained epilogue dots, [C, C]
# stores, colliding sharing keys, exchange routes on more kernels / warp
# grids, fp16 and the synchronous route on a K-loop kernel
# --------------------------------------------------------------------------


def _probe_kernels() -> object:
    return importlib.import_module("test.test_cute_mma_dag")


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
    ),
)
@pytest.mark.parametrize("exchange", ("auto", "smem"))
def test_probe_kernels_match_the_triton_backend(name: str, exchange: str) -> None:
    fn = getattr(_probe_kernels(), name)
    if name == "_two_dots_same_carry":
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
    source, out = _run_cute(
        fn, args, _config(block_sizes, 4, 2, cute_mma_dag_exchange=exchange)
    )
    if name == "_epi_chain" and exchange == "smem":
        assert source.count(f"{RT}.sts_b32(md_x") == 16
    _assert_close(out, _run_triton(fn, args, block_sizes))


@pytest.mark.parametrize(
    ("kernel", "mode", "block_sizes", "warps"),
    (
        ("dv", "vanilla", [1, 64, 64], 4),
        ("dv", "scalar", [1, 64, 64], 8),
        ("fwd_o", "gated", [1, 64, 64], 4),
        ("dqk", "gated", [1, 64, 64], 8),
        # Eight warps on a 64 x 128 grid give two warp columns: ``auto``
        # has to exchange as well; same bits either way.
        ("fwd_o", "vanilla", [1, 128, 64], 8),
        ("dv", "vanilla", [1, 128, 64], 8),
        # Eight warps on 64 x 64: dA is read straight and transposed from
        # one shared tile.
        ("dqk", "vanilla", [1, 64, 64], 8),
    ),
)
def test_exchange_modes_are_bit_equal_on_every_m2_kernel(
    kernel: str, mode: str, block_sizes: list[int], warps: int
) -> None:
    d, dv = (64, 64) if block_sizes[1] == 64 else (128, 128)
    t = _inputs3(bhn=4, d=d, dv=dv, dtype=torch.bfloat16, seed=9)
    args = {"fwd_o": _fwd_o_args, "dqk": _dqk_args, "dv": _dv_args}[kernel](t, mode)
    fn = _kernel_fn(kernel)
    _auto_source, auto = _run_cute(fn, args, _config(block_sizes, warps, 2))
    _smem_source, smem = _run_cute(
        fn, args, _config(block_sizes, warps, 2, cute_mma_dag_exchange="smem")
    )
    for lhs, rhs in zip(auto, smem, strict=True):
        assert torch.equal(lhs, rhs)
    _assert_close(auto, _run_triton(fn, args, block_sizes))


def test_fp16_k_loop_kernel_matches_the_triton_backend() -> None:
    t = _inputs3(bhn=4, d=64, dv=64, dtype=torch.float16, seed=13)
    args = _fwd_o_args(t, "vanilla")
    fn = _kernel_fn("fwd_o")
    source, out = _run_cute(fn, args, _config([1, 64, 64], 4, 2))
    assert f"{RT}.mma_f16(" in source
    _assert_close(out, _run_triton(fn, args, [1, 64, 64]))


def test_unaligned_fp32_source_takes_the_synchronous_pass_on_dv() -> None:
    # An fp32 ``dh`` whose base is 8-byte aligned cannot be cp.async'd: the
    # operand goes through the pair-load pass instead of the convert route.
    t = _inputs3(bhn=4, d=64, dv=64, dtype=torch.bfloat16, seed=15)
    g = torch.Generator(device=DEVICE).manual_seed(16)
    dh_wide = _randn(g, 4, 64, 66, dtype=torch.float32)
    dh = dh_wide[..., 2:66]
    assert dh.data_ptr() % 16 == 8 and dh.stride(1) == 66
    t["dh"] = dh
    args = _dv_args(t, "vanilla")
    fn = _kernel_fn("dv")
    source, out = _run_cute(fn, args, _config([1, 64, 64], 4, 2))
    assert f"{RT}.ldg_v2_f32(" in source and f"{RT}.lds_v4_f32(" not in source
    _assert_close(out, _run_triton(fn, args, [1, 64, 64]))


@pytest.mark.parametrize("name", ("_loop_chain", "_loop_chain_store"))
@pytest.mark.parametrize(("warps", "stages"), ((4, 2), (8, 4), (2, 1)))
def test_chained_loop_dots_match_the_triton_backend(
    name: str, warps: int, stages: int
) -> None:
    # The second loop dot accumulates the first one's result (rescaled),
    # with and without a store of that result in between.
    fn = getattr(_probe_kernels(), name)
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=21)
    args = (t["k"], t["v"])
    _source, out = _run_cute(fn, args, _config([64, 64], warps, stages))
    _assert_close(out, _run_triton(fn, args, [64, 64]))
