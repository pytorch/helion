"""Runtime coverage of the ``cute_mma_dag`` warp mode: the linear-attention
engine's serial state recurrences (``chunk_fwd_h_diag_fused`` /
``chunk_bwd_dh_diag_fused``) and K-loop + epilogue kernels
(``chunk_fwd_o_helion`` / ``chunk_bwd_dqk_helion`` / ``chunk_bwd_dv_helion``)
in every decay mode against the Triton backend, repeat-launch determinism,
and the stage / store / exchange knobs being pure data movement."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import itertools
import os
import re
import sys
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute import sass_check

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

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
        # The anchored (full_gla) mode's per-channel log-decay cumsum: a
        # non-positive running sum down the chunk.
        "gc4": (-_randn(g, bh, n, c, d, scale=0.05, dtype=f32).abs())
        .cumsum(2)
        .contiguous(),
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
    if mode == "anchored":
        return (t["k"], t["v"], t["g_last"], h0, t["gc4"], True, False, True, has_h0)
    return (t["k"], t["v"], t["g_last_s"], h0, t["gc"], True, True, False, has_h0)


def _bwd_args(t: dict[str, torch.Tensor], mode: str) -> tuple[object, ...]:
    if mode == "vanilla":
        return (t["k"], t["v"], None, t["dh_init"], None, False, False, False, 0.125)
    if mode == "anchored":
        return (
            t["k"],
            t["v"],
            t["g_last"],
            t["dh_init"],
            t["gc4"],
            True,
            False,
            True,
            0.125,
        )
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


def _inputs_varlen4(
    *,
    h: int,
    chunks: tuple[int, ...],
    d: int,
    dv: int,
    dtype: torch.dtype,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """The varlen state pass's inputs: chunk-major rows ``h * NT + j``,
    sequence n owning chunks ``chunk_offsets[n] : chunk_offsets[n + 1]``
    (``chunks[n]`` of them), scaled like ``_inputs4``."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = 64
    f32 = torch.float32
    nt = sum(chunks)
    n = len(chunks)
    offsets = [0, *itertools.accumulate(chunks)]
    return {
        "k": _randn(g, h * nt, c, d, scale=0.05, dtype=dtype),
        "w": _randn(g, h * nt, c, d, scale=0.05, dtype=dtype),
        "u": _randn(g, h * nt, c, dv, dtype=dtype),
        "h0": _randn(g, h * n, d, dv, scale=0.5, dtype=f32),
        "decay_last": -_randn(g, h * nt, d, scale=0.2, dtype=f32).abs(),
        "chunk_offsets": torch.tensor(offsets, dtype=torch.int64, device=DEVICE),
    }


def _fwd_h_delta_varlen_args(
    t: dict[str, torch.Tensor], chunks: tuple[int, ...], h: int
) -> tuple[object, ...]:
    return (
        t["k"],
        t["w"],
        t["u"],
        t["h0"],
        t["decay_last"],
        t["chunk_offsets"],
        sum(chunks),
        h,
    )


def _inputs_m4(
    *, bhn: int, d: int, dv: int, dtype: torch.dtype, seed: int = 0
) -> dict[str, torch.Tensor]:
    """The per-chunk delta-rule kernels' inputs (dqkw / wy_dL / dk / wy)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = 64
    f32 = torch.float32
    return {
        "q": _randn(g, bhn, c, d, dtype=dtype),
        "k": _randn(g, bhn, c, d, dtype=dtype),
        "v": _randn(g, bhn, c, dv, dtype=dtype),
        # A is strictly lower triangular, so the WY chain is exact; this scale
        # keeps every step's term visible in w / u without overflowing bf16.
        "k_small": _randn(g, bhn, c, d, scale=0.3, dtype=dtype),
        "v_new": _randn(g, bhn, c, dv, dtype=dtype),
        "do": _randn(g, bhn, c, dv, dtype=dtype),
        "dvni": _randn(g, bhn, c, dv, dtype=dtype),
        "h": _randn(g, bhn, d, dv, dtype=dtype),
        "dS": _randn(g, bhn, d, dv, dtype=dtype),
        # The pipeline hands the per-chunk kernels beta in the model dtype.
        "beta": _randn(g, bhn, c, scale=0.5, dtype=dtype),
        "beta32": _randn(g, bhn, c, scale=0.5, dtype=f32),
        "A_inv": _randn(g, bhn, c, c, scale=0.3, dtype=f32),
        "dL": _randn(g, bhn, c, c, dtype=dtype),
        "dL32": _randn(g, bhn, c, c, dtype=f32),
        "dW": _randn(g, bhn, c, d, dtype=f32),
        "d_ba": _randn(g, bhn, c, d, dtype=f32),
        "dk_as": _randn(g, bhn, c, d, dtype=f32),
        "dvni32": _randn(g, bhn, c, dv, dtype=f32),
        "gcs": -_randn(g, bhn, c, scale=0.1, dtype=f32).abs(),
        "gl": -_randn(g, bhn, scale=0.2, dtype=f32).abs(),
    }


def _dk_args(t: dict[str, torch.Tensor], gated: bool) -> tuple[object, ...]:
    return (t["k"], t["dL"], t["d_ba"], t["beta"], t["dk_as"])


def _dk_args_f32(t: dict[str, torch.Tensor], gated: bool) -> tuple[object, ...]:
    return (t["k"], t["dL32"], t["d_ba"], t["beta32"], t["dk_as"])


def _dqkw_args(t: dict[str, torch.Tensor], gated: bool) -> tuple[object, ...]:
    gcs, gl = (t["gcs"], t["gl"]) if gated else (None, None)
    return (
        t["q"],
        t["k"],
        t["h"],
        t["v_new"],
        t["do"],
        t["dvni"],
        t["dS"],
        gcs,
        gl,
        gated,
        0.125,
    )


def _wy_dl_args(t: dict[str, torch.Tensor], gated: bool) -> tuple[object, ...]:
    return (
        t["k"],
        t["v"],
        t["beta"],
        t["A_inv"],
        t["dW"],
        t["dvni"],
        t["gcs"] if gated else None,
        gated,
    )


def _wy_dl_args_f32(t: dict[str, torch.Tensor], gated: bool) -> tuple[object, ...]:
    args = _wy_dl_args(t, gated)
    return (*args[:2], t["beta32"], *args[3:5], t["dvni32"], *args[6:])


def _wy_args(t: dict[str, torch.Tensor], gated: bool) -> tuple[object, ...]:
    gcs = t["gcs"] if gated else None
    return (t["k_small"], t["v"], t["beta"], gcs, None, None, gated, False)


# The pipeline's dtypes (dL and dvni are bf16 there); the ``_f32`` variants
# feed the fp32 [C, C] / [C, DV] tiles that are narrowed on the way in.
_M4_ARGS = {
    "dk": _dk_args,
    "dk_f32": _dk_args_f32,
    "dqkw": _dqkw_args,
    "wy_dL": _wy_dl_args,
    "wy_dL_f32": _wy_dl_args_f32,
    "wy": _wy_args,
}
# Outputs summed over a fragment dim (fp32 vectors / scalars): the summation
# order differs from Triton's, so they are compared with a tolerance.  The
# fp32 dk epilogue fuses its product into an FMA (as the cute scalar lowering
# does) where Triton >= 3.9 rounds the product first: isolated elements land
# one bf16 ulp apart.
_M4_INEXACT = {
    ("wy_dL", False): {2},
    ("wy_dL", True): {2, 4},
    ("wy_dL_f32", False): {2},
    ("wy_dL_f32", True): {2, 4},
    ("dqkw", True): {3, 4},
    ("dk_f32", False): {0},
}

_KERNELS = {
    "fwd": "chunk_fwd_h_diag_fused",
    "bwd": "chunk_bwd_dh_diag_fused",
    "fwd_o": "chunk_fwd_o_helion",
    "dqk": "chunk_bwd_dqk_helion",
    "dv": "chunk_bwd_dv_helion",
    "fwd_h_delta": "chunk_fwd_h_delta_helion",
    "fwd_h_delta_varlen": "chunk_fwd_h_delta_varlen_helion",
    "dstate": "chunk_bwd_dstate_delta_helion",
    "dk": "chunk_bwd_dk_delta_helion",
    "dk_f32": "chunk_bwd_dk_delta_helion",
    "dqkw": "chunk_bwd_dqkw_delta_helion",
    "wy_dL": "chunk_bwd_wy_dL_delta_helion",
    "wy_dL_f32": "chunk_bwd_wy_dL_delta_helion",
    "wy": "chunk_fwd_wy_delta_helion",
    "bwd_o_kda": "chunk_bwd_o_kda_helion",
    "state_du_kda": "chunk_bwd_state_du_kda_helion",
    "wu_kda": "chunk_bwd_wu_kda_helion",
    "fwd_o_anch": "chunk_fwd_o_diag_anchored_helion",
    "state_dwk_kda": "chunk_bwd_state_dwk_kda_helion",
    "wy_anch": "chunk_fwd_wy_delta_helion",
    "dqkg": "chunk_bwd_dqkg_scalar_helion",
}


def _kernel_fn(kernel: str) -> Callable[..., object]:
    return getattr(_engine(), _KERNELS[kernel]).fn


def _config(
    block_sizes: list[int], warps: int, stages: int, **extra: object
) -> helion.Config:
    # The warp mode's ring tiles ride per-thread packets unless a test asks
    # for the TMA ring (``cute_mma_dag_loads="tma"``).
    if extra.get("cute_mma_dag", "warp") == "warp":
        extra.setdefault("cute_mma_dag_loads", "lane")
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


@pytest.mark.parametrize(
    ("bdv", "warps", "stages"), ((32, 4, 2), (64, 8, 4), (16, 2, 1), (64, 4, 1))
)
def test_varlen_state_pass_matches_the_triton_backend(
    bdv: int, warps: int, stages: int
) -> None:
    # chunk_fwd_h_delta_varlen_helion walks ``hl.tile(chunk_offsets[n],
    # chunk_offsets[n + 1], block_size=1)``: the trip count is a tensor
    # element read ahead of the loop (5, 1 and 3 chunks per sequence here,
    # the one-chunk sequence exercising a prologue past the loop's end), and
    # the final state is stored after the loop.
    chunks = (5, 1, 3)
    t = _inputs_varlen4(h=2, chunks=chunks, d=128, dv=64, dtype=torch.bfloat16)
    args = _fwd_h_delta_varlen_args(t, chunks, 2)
    fn = _kernel_fn("fwd_h_delta_varlen")
    source, out = _run_cute(fn, args, _config([bdv], warps, stages))
    assert source.count(f"{RT}.ldg_s64(") == 2
    assert "for md_i in cutlass.range(md_scalar, md_scalar_1):" in source
    _assert_close(out, _run_triton(fn, args, [bdv]), scaled=True)


def _inputs_varlen_tokens(
    *, h: int, lens: tuple[int, ...], d: int, dv: int, dtype: torch.dtype, seed: int = 0
) -> dict[str, object]:
    """The token-major varlen kernels' inputs: sequences of ``lens`` tokens
    under one ``cu_seqlens`` (the engine's chunk tables place the chunks;
    a length that is not a multiple of 64 leaves a ragged last chunk), the
    per-chunk operands chunk-major, row ``h * NT + j``."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = 64
    f32 = torch.float32
    total = sum(lens)
    cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(lens)], dtype=torch.int32, device=DEVICE
    )
    _offsets, token_base, valid_len, nt = _engine()._kda_varlen_chunk_tables(
        cu_seqlens, c, total
    )
    hnt = h * nt
    rows = torch.arange(c, device=DEVICE)
    strict = (rows[:, None] > rows[None, :]).to(dtype)
    causal = (rows[:, None] >= rows[None, :]).to(dtype)
    return {
        "q": _randn(g, total, h, d, dtype=dtype),
        "k": _randn(g, total, h, d, dtype=dtype),
        "v": _randn(g, total, h, dv, dtype=dtype),
        "beta": torch.sigmoid(_randn(g, total, h, dtype=f32)).to(dtype),
        "g_cs": -_randn(g, hnt, c, d, scale=0.1, dtype=f32).abs().cumsum(1),
        "Akk": _randn(g, hnt, c, c, scale=0.3, dtype=dtype) * strict,
        "v_new": _randn(g, hnt, c, dv, dtype=dtype),
        "h": _randn(g, hnt, d, dv, scale=0.3, dtype=dtype),
        "A": _randn(g, hnt, c, c, scale=0.3, dtype=dtype) * causal,
        "token_base": token_base,
        "valid_len": valid_len,
        "hnt": hnt,
        "total": total,
    }


def _run_backend(
    fn: Callable[..., object],
    backend: str,
    args: tuple[object, ...],
    outputs: tuple[int, ...],
    config: helion.Config,
) -> tuple[str, tuple[torch.Tensor, ...]]:
    """Run ``fn`` under ``backend`` at ``config``; the kernel writes its
    outputs into the arguments at ``outputs`` (returned as copies)."""
    kernel = helion.kernel(fn, backend=backend, static_shapes=True)
    bound = kernel.bind(args)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    compiled = bound.compile_config(normalized)
    for position in outputs:
        cast_tensor(args[position]).zero_()
    compiled(*args)
    torch.cuda.synchronize()
    return source, tuple(cast_tensor(args[position]).clone() for position in outputs)


@pytest.mark.parametrize(
    ("kernel", "d", "warps", "stages", "loads"),
    (
        ("wy", 64, 4, 2, "lane"),
        ("wy", 128, 8, 4, "lane"),
        ("wy", 64, 4, 1, "lane"),
        ("wy", 32, 4, 2, "tma"),
        ("wy", 32, 8, 4, "tma"),
        ("wy", 64, 4, 2, "tma"),
        ("o_diag", 64, 4, 2, "lane"),
        ("o_diag", 128, 4, 4, "lane"),
        ("o_diag", 128, 8, 2, "lane"),
    ),
)
def test_varlen_window_kernels_match_the_triton_backend(
    kernel: str, d: int, warps: int, stages: int, loads: str
) -> None:
    # The token-major kernels of the varlen KDA forward read their q / k / v
    # / beta rows through ``token_base[j] + hl.arange(C)`` under the
    # ``hl.arange(C) < valid_len[j]`` mask (a window: rows at a runtime
    # offset, live below a runtime limit) and o_diag stores its output the
    # same way.  Sequences of 200, 64 and 150 tokens: ragged last chunks
    # (8, 64 and 22 live rows), and a last chunk that reaches past the
    # tensor's end only through its masked rows.  Under the TMA ring
    # (``loads="tma"``) the windowed k tile keeps its zero-filled packets --
    # a bulk tensor copy lands its box whole, at no runtime offset and with
    # no runtime limit -- while the derived g_cs tile (a [64, 32] fp32 box
    # of 128-byte rows, no window) rides the one tensor map beside them.
    lens = (200, 64, 150)
    h = 2
    t = _inputs_varlen_tokens(h=h, lens=lens, d=d, dv=64, dtype=torch.bfloat16)
    hnt = cast("int", t["hnt"])
    e = functools.partial(torch.empty, dtype=torch.bfloat16, device=DEVICE)
    if kernel == "wy":
        fn = _engine().chunk_fwd_wy_delta_varlen_helion.fn
        args = (
            t["k"],
            t["v"],
            t["beta"],
            t["g_cs"],
            t["Akk"],
            t["token_base"],
            t["valid_len"],
            e((hnt, 64, d)),
            e((hnt, 64, 64)),
            e((hnt, 64, d)),
        )
        outputs = (7, 8, 9)
        # (D == DV rides the value loop on the key loop; else two loops.)
        block_sizes = [32] if d == 64 else [32, 32]
    else:
        fn = _engine().chunk_fwd_o_diag_anchored_varlen_helion.fn
        total = cast("int", t["total"])
        args = (
            t["q"],
            t["v_new"],
            t["g_cs"],
            t["h"],
            t["A"],
            t["token_base"],
            t["valid_len"],
            e((total, h, 64)),
            0.125,
            True,
        )
        outputs = (7,)
        block_sizes = []
    source, outs = _run_backend(
        fn,
        "cute",
        args,
        outputs,
        _config(block_sizes, warps, stages, cute_mma_dag_loads=loads),
    )
    assert (
        f"{RT}.mma_" in source and "_cute_grouped_reduce_shared_two_stage" not in source
    )
    assert ("cp_async_bulk_tensor_shared_cta_global" in source) == (loads == "tma")
    if loads == "tma":
        assert "'tma_count': 1" in source and "'tma0_elem': 4" in source
        assert f"{RT}.cp_async_16_zfill(" in source
    assert source.count(f"{RT}.ldg_s64(") == 2
    assert (
        "md_scalar_1 + " in source
    )  # the window's rows are bounded by the tensor's end
    _source, refs = _run_backend(
        fn, "triton", args, outputs, helion.Config(block_sizes=block_sizes)
    )
    _assert_close(outs, refs, scaled=True)
    if kernel == "o_diag":
        # Rows past a sequence's end are never written: the next sequence's
        # first tokens keep their values (zero here) under both backends.
        assert torch.equal(outs[0].eq(0).all(dim=(1, 2)), refs[0].eq(0).all(dim=(1, 2)))


def _vector_window_kernel(
    x: torch.Tensor,
    beta: torch.Tensor,
    hmat: torch.Tensor,
    base_t: torch.Tensor,
    lim_t: torch.Tensor,
    out: torch.Tensor,
    sums: torch.Tensor,
) -> None:
    # x [T, H, D]: a row window (the token rows); beta [H, T] and sums [H, T]:
    # vector windows along the contiguous dim.
    nt = base_t.size(0)
    c = hl.specialize(hmat.size(0))
    for tile_r in hl.tile(out.size(0), block_size=1):
        j = tile_r.begin % nt
        head = tile_r.begin // nt
        idx = hl.arange(c)
        valid = idx < lim_t[j]
        xt = torch.where(
            valid[:, None],
            hl.load(x, [base_t[j] + idx, head, slice(None)], extra_mask=valid[:, None]),
            0,
        )
        b = torch.where(
            valid, hl.load(beta, [head, base_t[j] + idx], extra_mask=valid), 0
        )
        o = hl.dot((xt.float() * b.float()[:, None]).to(x.dtype), hmat[:, :])
        out[tile_r.begin, :, :] = o.to(out.dtype)
        hl.store(sums, [head, base_t[j] + idx], o.sum(dim=-1), extra_mask=valid)


@pytest.mark.parametrize(("warps", "stages"), ((4, 2), (8, 1)))
def test_vector_windows_at_any_stride_match_the_triton_backend(
    warps: int, stages: int
) -> None:
    # A 2-D tile's window must be its rows at a 16-byte multiple of a row
    # stride (a column window is not admitted: test_cute_mma_dag.py); a
    # vector's window may sit on the contiguous dim at any offset, its
    # elements loaded and stored one by one under per-element predicates.
    # Sequences of 37, 64, 91 and 50 tokens: odd token offsets for the row
    # window of x (whole 256-byte rows) and for the contiguous beta / sums
    # windows (2- and 4-byte elements), with ragged chunks of 37, 27 and 50
    # live rows.
    lens = (37, 64, 91, 50)
    h, d, dv, c = 2, 64, 64, 64
    bf16, f32 = torch.bfloat16, torch.float32
    g = torch.Generator(device=DEVICE).manual_seed(5)
    total = sum(lens)
    cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(lens)], dtype=torch.int32, device=DEVICE
    )
    _offsets, token_base, valid_len, nt = _engine()._kda_varlen_chunk_tables(
        cu_seqlens, c, total
    )
    assert nt == 5 and token_base.tolist() == [0, 37, 101, 165, 192]
    args = (
        _randn(g, total, h, d, dtype=bf16),
        torch.sigmoid(_randn(g, h, total, dtype=f32)).to(bf16),
        _randn(g, d, dv, scale=0.3, dtype=bf16),
        token_base,
        valid_len,
        torch.empty(h * nt, c, dv, dtype=bf16, device=DEVICE),
        torch.empty(h, total, dtype=f32, device=DEVICE),
    )
    source, outs = _run_backend(
        _vector_window_kernel, "cute", args, (5, 6), _config([], warps, stages)
    )
    assert (
        f"{RT}.mma_" in source and "_cute_grouped_reduce_shared_two_stage" not in source
    )
    assert f"{RT}.cp_async_16_zfill(" in source  # x: whole rows, zero-filled
    assert f"{RT}.ldg_bf16_as_f32_pred(" in source  # beta: per element
    _source, refs = _run_backend(
        _vector_window_kernel, "triton", args, (5, 6), helion.Config()
    )
    _assert_close(outs, refs, scaled=True)


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
    # Every admitted program lands on tensor memory (the warp mode stays
    # one knob value away); the delta kernels keep the warp mode.
    for kernel, mode, args in (
        ("fwd", "tcgen05", _fwd_args(t, "vanilla", False)),
        ("fwd_o", "tcgen05", _fwd_o_args(t3, "vanilla")),
        ("dqk", "tcgen05", _dqk_args(t3, "vanilla")),
        ("dv", "tcgen05", _dv_args(t3, "vanilla")),
        ("fwd_h_delta", "tcgen05", _fwd_h_delta_args(t4, False)),
        ("fwd_h_delta", "tcgen05", _fwd_h_delta_args(t4, True)),
        ("dstate", "tcgen05", _dstate_args(t4, False)),
        ("dstate", "tcgen05", _dstate_args(t4, True)),
    ):
        fn = _kernel_fn(kernel)
        bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
        config = bound.config_spec.default_config()
        assert config.config["cute_mma_dag"] == mode, kernel
        source = bound.to_code(config)
        if mode == "warp":
            assert f"{RT}.mma_bf16(" in source
        else:
            assert "_helion_mma_dag_tc.prims.tcgen05_mma(" in source
        out = _tensors(bound.compile_config(config)(*args))
        _assert_close(
            out,
            _run_triton(fn, args, config.config["block_sizes"]),
            scaled=kernel in ("fwd_h_delta", "dstate"),
        )
        if mode == "warp":
            continue
        # (its tiles may need more warps there: normalize repairs them).
        warp = dict(config.config, cute_mma_dag="warp")
        with bound.env:
            bound.config_spec.normalize(warp, _fix_invalid=True)
        assert warp["cute_mma_dag"] == "warp"
        assert f"{RT}.mma_bf16(" in bound.to_code(helion.Config(**warp))


# Region-tree probes: a loop reading an enclosing region's fragment / vector, a
# view read after a takeover, a ringed loop nested in a bare outer loop.
_REGION_TREE_INEXACT = {"_vec_from_outer": {0}, "_reduced_view_after_takeover": {1}}
# kernel, inputs, block sizes, warps, stages
_REGION_TREE_CASES = (
    ("_acc_from_outer", "outer", [], 4, 2),
    ("_acc_from_outer", "outer", [], 8, 1),
    ("_derived_from_outer", "outer", [], 4, 2),
    ("_derived_from_outer", "outer", [], 8, 1),
    ("_vec_from_outer", "vec", [], 4, 2),
    ("_vec_from_outer", "vec", [], 8, 1),
    ("_view_read_after_takeover", "qk", [32], 4, 2),
    ("_view_read_after_takeover", "qk", [64], 8, 1),
    ("_reduced_view_after_takeover", "qk", [32], 4, 2),
    ("_reduced_view_after_takeover", "qk", [64], 8, 1),
    ("_nested_ring", "nest", [32, 128], 4, 2),
    ("_nested_ring", "nest", [64, 32], 8, 3),
    ("_nested_ring", "nest", [32, 32], 4, 1),
)


@pytest.mark.parametrize(
    ("name", "inputs", "block_sizes", "warps", "stages"),
    _REGION_TREE_CASES,
    ids=[
        f"{n}-{'x'.join(map(str, bs)) or 'grid'}w{w}s{s}"
        for n, _i, bs, w, s in _REGION_TREE_CASES
    ],
)
def test_region_tree_kernels_match_the_triton_backend(
    name: str, inputs: str, block_sizes: list[int], warps: int, stages: int
) -> None:
    fn = getattr(_probe_kernels(), name)
    g = torch.Generator(device=DEVICE).manual_seed(9)
    if inputs == "outer":
        t = _inputs(bh=4, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=9)
        a = _randn(g, 4, 64, 64, dtype=torch.float32)
        b = _randn(g, 4, 64, 64, dtype=torch.float32)
        args: tuple[object, ...] = (a, b, t["k"], t["v"])
    elif inputs == "vec":
        t = _inputs(bh=4, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=9)
        args = (t["k"], t["v"])
    elif inputs == "qk":
        t3 = _inputs3(bhn=6, d=128, dv=64, dtype=torch.bfloat16, seed=9)
        args = (t3["q"], t3["k"])
    else:
        args = (
            _randn(g, 16, 64, 128, dtype=torch.bfloat16),
            _randn(g, 16, 128, 128, dtype=torch.bfloat16),
        )
    _source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    ref = _run_triton(fn, args, block_sizes)
    inexact = _REGION_TREE_INEXACT.get(name, set())
    for index, (lhs, rhs) in enumerate(zip(out, ref, strict=True)):
        if index in inexact:
            _assert_close((lhs,), (rhs,))
        else:
            assert torch.equal(lhs, rhs), index


@pytest.mark.parametrize("warps,stages", [(4, 2), (8, 1)])
def test_side_loads_varying_with_a_nested_loop_match_the_triton_backend(
    warps: int, stages: int
) -> None:
    # Probes: a vector varying with the inner loop of a nest (never hoisted
    # ahead of the outer one) and one varying with the outer loop (hoisted
    # ahead of the inner one only).  (The read-modify-write probes decline the
    # lowering now: a program that loads and stores one storage is not mine.)
    kernels = _probe_kernels()
    g = torch.Generator(device=DEVICE).manual_seed(11)
    do = _randn(g, 4, 64, 128, dtype=torch.bfloat16)
    h = _randn(g, 4, 128, 128, dtype=torch.bfloat16)
    gate = _randn(g, 4, 128, dtype=torch.float32)
    for fn in (kernels._inner_varying_gate, kernels._inner_varying_row_gate):
        ref = _run_triton(fn, (do, h, gate), [64, 64])
        for aux in ("register", "ring"):
            for loads in ("lane", "tma"):
                config = _config(
                    [64, 64],
                    warps,
                    stages,
                    cute_mma_dag_aux=aux,
                    cute_mma_dag_loads=loads,
                )
                source, out = _run_cute(fn, (do, h, gate), config)
                # Only the inner-varying vector has the aux ring as a subject;
                # the TMA ring re-entered by the outer loop keeps its phases.
                ringed = aux == "ring" and fn is kernels._inner_varying_gate
                assert (f"{RT}.cp_async_16(md_aux1" in source) == ringed, fn.__name__
                assert ("cp_async_bulk_tensor" in source) == (loads == "tma")
                assert torch.equal(out[0], ref[0]), (fn.__name__, aux, loads)


@pytest.mark.parametrize("mode", range(5))
def test_packed_pairs_match_the_triton_backend(mode: int) -> None:
    # The packed f32x2 pairs round each lane as the scalar op does: the three
    # packable epilogue shapes of the probe are bit-identical to Triton.  The
    # FMA refusal (through views) keeps every fusable product scalar, where
    # the cute lowering fuses the product and Triton >= 3.9 rounds it first:
    # those two shapes agree to one fp32 rounding of the product.
    fn = _probe_kernels()._pack_probe
    g = torch.Generator(device=DEVICE).manual_seed(13)
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=13)
    sr = _randn(g, 2, 64, dtype=torch.float32)
    sc = _randn(g, 2, 64, dtype=torch.float32)
    s2 = _randn(g, 2, 64, dtype=torch.float32)
    args = (t["k"], t["v"], sr, sc, s2, mode)
    _source, out = _run_cute(fn, args, _config([64, 64], 4, 2))
    ref = _run_triton(fn, args, [64, 64])
    if mode in (2, 4):
        torch.testing.assert_close(out[0], ref[0], atol=1e-4, rtol=1e-5)
    else:
        assert torch.equal(out[0], ref[0])


def test_anchored_fwd_h_reads_the_decay_row_from_the_ring_slot() -> None:
    # full_gla's forward state kernel (``diag_anchored``): the chunk's last
    # decay row ``gc[idx, i_t, C - 1, tile_d]`` is read from the staged
    # ``gc`` tile's ring slot (``lds_f32``; no global row load on the chain)
    # and the key's ``.to(bf16)`` under its transpose is folded into the
    # exchange's pack -- the same bytes: the exchange and pass routes are
    # bit-equal to each other and track the Triton backend.
    t = _inputs(bh=2, n=8, d=128, dv=128, dtype=torch.bfloat16, seed=7)
    args = _fwd_args(t, "anchored", True)
    fn = _kernel_fn("fwd")
    outs = []
    for derived in ("exchange", "pass"):
        source, out = _run_cute(
            fn, args, _config([32, 128], 8, 4, cute_mma_dag_derived=derived)
        )
        loop = source[source.index("for md_i in cutlass.range(") :]
        assert f"{RT}.lds_f32(" in loop and f"{RT}.ldg_f32(" not in loop
        # (No bf16 cast of a computed value; the loaded bf16 halves are
        # re-typed exactly and their fp32 twins feed the chain.)
        assert not re.findall(
            r"cutlass\.BFloat16\((?!md_(?:lo|hi)(?:_\d+)?\))[^)]*\)", loop
        )
        outs.append(out[0])
    assert torch.equal(outs[0], outs[1])
    _assert_close((outs[0],), _run_triton(fn, args, [32, 128]))


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


def test_wy_with_an_unaligned_key_matches_the_triton_backend() -> None:
    # The key's rows are 8-byte aligned: its widened tf32 B tile takes the
    # synchronous pass into the natural tile and the per-lane B reads (the
    # aligned v tile keeps the transposed one).  Same bits as Triton.
    t = _inputs_m4(bhn=6, d=128, dv=128, dtype=torch.bfloat16, seed=3)
    g = torch.Generator(device=DEVICE).manual_seed(4)
    wide = _randn(g, 6, 64, 136, scale=0.1, dtype=torch.bfloat16)
    k = wide[:, :, 4:132]
    assert k.data_ptr() % 16 == 8 and k.stride(1) == 136
    fn = _kernel_fn("wy")
    for gated in (False, True):
        args = (
            k,
            t["v"],
            t["beta"],
            t["gcs"] if gated else None,
            None,
            None,
            gated,
            False,
        )
        for block_sizes, warps, stages in (([64, 64, 64], 4, 2), ([128, 32, 32], 4, 2)):
            source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
            assert f"{RT}.mma_tf32(" in source and f"{RT}.lds_b32(" in source
            ref = _run_triton(fn, args, block_sizes)
            for lhs, rhs in zip(out, ref, strict=True):
                assert torch.equal(lhs, rhs)


# --------------------------------------------------------------------------
# Probes: fragments with several readers, chained epilogue dots, [C, C]
# stores, colliding sharing keys, exchange routes on more kernels / warp
# grids, fp16 and the synchronous route on a K-loop kernel
# --------------------------------------------------------------------------


def _probe_kernels() -> ModuleType:
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
        "_two_dots_carry_read_after",
        "_stale_carry",
    ),
)
@pytest.mark.parametrize("exchange", ("auto", "smem"))
def test_probe_kernels_match_the_triton_backend(name: str, exchange: str) -> None:
    fn = getattr(_probe_kernels(), name)
    if name in ("_two_dots_same_carry", "_two_dots_carry_read_after", "_stale_carry"):
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


# kernel, gated, (d, dv), block sizes, warps, stages, extra knobs
_M4_CASES = (
    ("dk", False, (128, 64), [32], 4, 2, {}),
    ("dk", False, (128, 64), [64], 8, 4, {}),
    ("dk", False, (128, 64), [128], 8, 2, {}),
    ("dk", False, (64, 64), [16], 2, 1, {}),
    ("dk", False, (128, 128), [32], 4, 3, {}),
    ("dqkw", False, (128, 64), [32, 32, 32], 4, 2, {}),
    ("dqkw", False, (128, 64), [64, 32, 64], 8, 2, {}),
    ("dqkw", False, (128, 64), [32, 64, 32], 8, 1, {}),
    ("dqkw", False, (64, 64), [64, 64, 64], 8, 4, {}),
    (
        "dqkw",
        False,
        (128, 128),
        [128, 32, 128],
        4,
        2,
        {"cute_mma_dag_exchange": "smem"},
    ),
    ("wy_dL", False, (128, 64), [32, 32, 128], 4, 2, {}),
    ("wy_dL", False, (128, 64), [32, 32, 64], 8, 2, {}),
    ("wy_dL", False, (128, 64), [64, 64, 64], 4, 1, {}),
    ("wy_dL", False, (128, 64), [16, 16, 16], 8, 3, {}),
    (
        "wy_dL",
        False,
        (128, 128),
        [32, 32, 128],
        4,
        4,
        {"cute_mma_dag_exchange": "smem"},
    ),
    ("wy_dL", False, (64, 64), [64, 32, 64], 4, 2, {}),
    ("dqkw", True, (128, 64), [32, 32, 32, 32, 32], 4, 2, {}),
    ("dqkw", True, (128, 64), [32, 32, 32, 64, 64], 8, 2, {}),
    ("dqkw", True, (128, 128), [64, 32, 64, 32, 128], 8, 1, {}),
    (
        "dqkw",
        True,
        (128, 64),
        [32, 32, 32, 32, 16],
        4,
        3,
        {"cute_mma_dag_exchange": "smem"},
    ),
    ("wy_dL", True, (128, 64), [32, 32, 128], 4, 2, {}),
    ("wy_dL", True, (128, 64), [32, 32, 64], 8, 2, {}),
    ("wy_dL", True, (64, 64), [64, 64, 64], 4, 1, {"cute_mma_dag_exchange": "smem"}),
    ("wy", False, (128, 64), [128, 32, 32], 4, 2, {}),
    ("wy", True, (128, 64), [128, 32, 32], 4, 2, {}),
    ("wy", False, (128, 128), [64, 64, 64], 8, 2, {}),
    ("wy", True, (64, 64), [32, 32, 32], 4, 1, {}),
    ("wy", False, (128, 64), [32, 128, 64], 8, 4, {}),
    ("wy", False, (128, 64), [128, 32, 32], 4, 2, {"cute_mma_dag_exchange": "smem"}),
    ("wy", True, (128, 128), [64, 64, 64], 8, 2, {"cute_mma_dag_exchange": "smem"}),
    ("dk_f32", False, (128, 64), [32], 4, 2, {}),
    ("dk_f32", False, (64, 64), [64], 8, 2, {}),
    ("wy_dL_f32", False, (128, 64), [32, 32, 128], 4, 2, {}),
    ("wy_dL_f32", True, (128, 64), [32, 32, 64], 8, 2, {}),
)


@pytest.mark.parametrize(
    ("kernel", "gated", "shape", "block_sizes", "warps", "stages", "extra"),
    _M4_CASES,
    ids=[
        f"{k}{'-gated' if g else ''}-d{s[0]}dv{s[1]}-{'x'.join(map(str, bs))}w{w}s{st}"
        f"{'-' + str(e['cute_mma_dag_exchange']) if e.get('cute_mma_dag_exchange') else ''}"
        for k, g, s, bs, w, st, e in _M4_CASES
    ],
)
def test_m4_kernels_match_the_triton_backend(
    kernel: str,
    gated: bool,
    shape: tuple[int, int],
    block_sizes: list[int],
    warps: int,
    stages: int,
    extra: dict[str, object],
) -> None:
    d, dv = shape
    t = _inputs_m4(bhn=6, d=d, dv=dv, dtype=torch.bfloat16)
    args = _M4_ARGS[kernel](t, gated)
    fn = _kernel_fn(kernel)
    source, out = _run_cute(fn, args, _config(block_sizes, warps, stages, **extra))
    assert f"block=({32 * warps}, 1, 1)" in source
    ref = _run_triton(fn, args, block_sizes)
    # bf16 products accumulated in fp32 in the same order: the delta modes
    # are bit-identical to Triton, except the vectors summed over a
    # fragment dim.
    inexact = _M4_INEXACT.get((kernel, gated), set())
    for index, (lhs, rhs) in enumerate(zip(out, ref, strict=True)):
        if index in inexact:
            _assert_close((lhs,), (rhs,))
        else:
            assert torch.equal(lhs, rhs), index


def _inputs_kda(
    *, bh: int, n: int, d: int, dv: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    """The kda / anchored per-chunk kernels' inputs in the pipeline's dtypes
    (per-channel cumulative log-decay ``gc`` [BHN, C, D] in fp32)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = 64
    bhn = bh * n
    bf16, f32 = torch.bfloat16, torch.float32
    gc = -_randn(g, bhn, c, d, scale=0.1, dtype=f32).abs().cumsum(1)
    return {
        "q": _randn(g, bhn, c, d, dtype=bf16),
        "k": _randn(g, bhn, c, d, dtype=bf16),
        "w": _randn(g, bhn, c, d, dtype=bf16),
        "v": _randn(g, bhn, c, dv, dtype=bf16),
        "do": _randn(g, bhn, c, dv, dtype=bf16),
        "h": _randn(g, bhn, d, dv, dtype=bf16),
        "A": _randn(g, bhn, c, c, dtype=bf16),
        "gc": gc,
        "Tinv": _randn(g, bhn, c, c, scale=0.3, dtype=f32),
        "beta": _randn(g, bhn, c, scale=0.5, dtype=f32),
        "dw": _randn(g, bhn, c, d, dtype=f32),
        "du": _randn(g, bhn, c, dv, dtype=f32),
        "dv_new": _randn(g, bhn, c, dv, dtype=f32),
        "dh": _randn(g, bhn, d, dv, dtype=f32),
        "dS0": _randn(g, bh, d, dv, dtype=f32),
    }


def _anchored_gram(t: dict[str, torch.Tensor]) -> torch.Tensor:
    """The anchored k-gram the WY transform inverts: strictly lower
    triangular and small, so the Neumann doubling converges (the engine's
    gram carries the decay and the strict mask)."""
    return (torch.tril(t["A"].float(), -1) * 0.05).to(torch.bfloat16)


def _kda_args(kernel: str, t: dict[str, torch.Tensor], bh: int) -> tuple[object, ...]:
    if kernel == "bwd_o_kda":
        return (t["q"], t["v"], t["h"], t["A"], t["gc"], t["do"], 0.125)
    if kernel == "state_du_kda":
        n = t["k"].shape[0] // bh

        def chunks(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(bh, n, *x.shape[1:])

        return (
            chunks(t["k"]),
            chunks(t["w"]),
            chunks(t["gc"]),
            chunks(t["dv_new"]),
            chunks(t["dh"]),
            t["dS0"],
        )
    if kernel == "wu_kda":
        return (
            t["Tinv"],
            t["k"],
            t["v"],
            t["beta"],
            t["gc"],
            t["A"],
            t["dw"],
            t["du"],
        )
    if kernel == "state_dwk_kda":
        n = t["k"].shape[0] // bh

        def chunks(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(bh, n, *x.shape[1:])

        return (
            chunks(t["k"]),
            chunks(t["gc"]),
            chunks(t["h"]),
            chunks(t["v"]),
            chunks(t["du"]),
            chunks(t["dh"]),
        )
    if kernel == "wy_anch":
        # (The state key ``k_state_out`` is written in place: a fresh buffer
        # per call, compared by ``test_anchored_wy_state_key_...``.)
        return (
            t["k"],
            t["v"],
            t["beta"],
            t["gc"],
            _anchored_gram(t),
            torch.empty_like(t["k"]),
            False,
            True,
        )
    assert kernel == "fwd_o_anch"
    return (t["q"], t["v"], t["gc"], t["h"], t["A"], 0.125)


# Outputs beyond bit equality with Triton: an fp32 vector summed over a
# fragment dim (summation order), and the serially carried fp32 state whose
# bf16 operand roundings flip where the backends' fp32 op order differs (a
# few bf16 ulps of the large terms, so the tolerance follows the magnitude).
# dwk's ``dgc_s`` adds the column sum of a fragment and a loop-carried row
# reduction (summation order).
_KDA_INEXACT = {"state_du_kda": {0, 1}, "wu_kda": {3}, "state_dwk_kda": {2}}
_KDA_SCALED = {"state_du_kda"}
# kernel, (d, dv), block sizes, warps, stages
_KDA_CASES = (
    ("bwd_o_kda", (64, 64), [32, 32, 32], 4, 2),
    ("bwd_o_kda", (128, 64), [64, 32, 64], 8, 1),
    ("bwd_o_kda", (64, 128), [16, 64, 16], 4, 2),
    ("state_du_kda", (64, 64), [32], 4, 2),
    ("state_du_kda", (128, 128), [64], 8, 1),
    ("state_du_kda", (64, 128), [16], 2, 3),
    ("wu_kda", (64, 64), [32, 32], 4, 2),
    ("wu_kda", (128, 64), [64, 64], 8, 1),
    ("wu_kda", (64, 128), [32, 16], 4, 2),
    ("fwd_o_anch", (64, 64), [1, 32, 32], 4, 2),
    ("fwd_o_anch", (128, 128), [1, 64, 64], 8, 1),
    ("fwd_o_anch", (64, 64), [1, 16, 32], 2, 3),
    # The epilogue reads the column sum of its own fragment (``dgc_last``)
    # beside the fragment under the ``is_last`` row mask: the vector is
    # relaid through the scratch into the fragment's register positions.
    ("state_dwk_kda", (64, 64), [1, 64, 32], 8, 2),
    ("state_dwk_kda", (128, 64), [1, 32, 32], 4, 2),
    ("state_dwk_kda", (64, 128), [1, 64, 64], 8, 1),
    # The Neumann chain's carries are seeded with pointwise values over the
    # loaded gram (``A = -(beta * Akk[b])``, ``T = eye + A``).
    ("wy_anch", (64, 64), [32, 32], 4, 2),
    ("wy_anch", (128, 128), [64, 64], 8, 1),
    ("wy_anch", (128, 64), [32, 64], 4, 2),
)


@pytest.mark.parametrize(
    ("kernel", "shape", "block_sizes", "warps", "stages"),
    _KDA_CASES,
    ids=[
        f"{k}-d{s[0]}dv{s[1]}-{'x'.join(map(str, bs))}w{w}s{st}"
        for k, s, bs, w, st in _KDA_CASES
    ],
)
def test_kda_kernels_match_the_triton_backend(
    kernel: str,
    shape: tuple[int, int],
    block_sizes: list[int],
    warps: int,
    stages: int,
) -> None:
    d, dv = shape
    t = _inputs_kda(bh=2, n=3, d=d, dv=dv)
    args = _kda_args(kernel, t, 2)
    fn = _kernel_fn(kernel)
    _source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    ref = _run_triton(fn, args, block_sizes)
    inexact = _KDA_INEXACT.get(kernel, set())
    for index, (lhs, rhs) in enumerate(zip(out, ref, strict=True)):
        if index not in inexact:
            assert torch.equal(lhs, rhs), index
        elif kernel in _KDA_SCALED:
            atol = 2e-2 * max(1.0, rhs.abs().max().item())
            torch.testing.assert_close(lhs, rhs, atol=atol, rtol=2e-2)
        else:
            _assert_close((lhs,), (rhs,))


@pytest.mark.parametrize("gated", (False, True))
def test_wy_dl_at_the_s1_pipeline_shape_matches_the_triton_backend(gated: bool) -> None:
    # The cold autotune of delta_rule at s1 (4, 16, 2048, 128, 128) once left
    # wy_dL on SIMT: every dag config failed the accuracy check against the
    # SIMT reference (not a dag fault -- the dag matches Triton bit for bit,
    # the two fp32 accumulation orders differ by a few bf16 ulps of dL where
    # the bf16-rounded intermediate of dA amplifies them).  The promoted
    # default at the real shape and dtypes (bf16 beta and dvni) -- a
    # tensor-memory seed, wy_dL being a flat streaming region program --
    # runs and equals Triton.
    t = _inputs_m4(bhn=2048, d=128, dv=128, dtype=torch.bfloat16, seed=1)
    args = _wy_dl_args(t, gated)
    fn = _kernel_fn("wy_dL")
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    with bound.env:
        default = bound.config_spec.normalized_config(
            bound.config_spec.default_config()
        )
    assert default.config["cute_mma_dag"] == "tcgen05"
    out = _tensors(bound.compile_config(default)(*args))
    torch.cuda.synchronize()
    ref = _run_triton(fn, args, default.config["block_sizes"])
    inexact = _M4_INEXACT[("wy_dL", gated)]
    for index, (lhs, rhs) in enumerate(zip(out, ref, strict=True)):
        if index in inexact:
            _assert_close((lhs,), (rhs,))
        else:
            assert torch.equal(lhs, rhs), index


@pytest.mark.parametrize("name", ("_acc_scaled_b", "_acc_scaled_a", "_acc_scaled_b16"))
@pytest.mark.parametrize("warps", (4, 8))
@pytest.mark.parametrize("exchange", ("auto", "smem"))
def test_a_rescaled_acc_read_by_an_operand_matches_the_triton_backend(
    name: str, warps: int, exchange: str
) -> None:
    # ``dot(Apow, T, acc=T * 0.5)`` accumulated in place and the operand read
    # the rescaled T (T * 0.5 + Apow @ (T * 0.5)).
    fn = getattr(_probe_kernels(), name)
    g = torch.Generator(device=DEVICE).manual_seed(13)
    k = _randn(g, 6, 64, 128, scale=0.15, dtype=torch.bfloat16)
    _source, out = _run_cute(
        fn, (k,), _config([64], warps, 2, cute_mma_dag_exchange=exchange)
    )
    ref = _run_triton(fn, (k,), [64])
    for lhs, rhs in zip(out, ref, strict=True):
        assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("name", ("_hoisted_a_chain", "_nk_b_shuffle"))
@pytest.mark.parametrize("warps", (4, 8))
@pytest.mark.parametrize("exchange", ("auto", "smem"))
def test_shuffle_route_shapes_match_the_triton_backend(
    name: str, warps: int, exchange: str
) -> None:
    # The hoisted-A ([M, K] own accumulator) and [N, K]-B shapes of the tf32
    # shuffle route against Triton, and against the exchange route (smem).
    fn = getattr(_probe_kernels(), name)
    g = torch.Generator(device=DEVICE).manual_seed(17)
    k = _randn(g, 6, 64, 128, scale=0.15, dtype=torch.bfloat16)
    args: tuple[object, ...] = (k,)
    if name == "_nk_b_shuffle":
        args = (_randn(g, 6, 16, 64, dtype=torch.float32), k)
    source, out = _run_cute(
        fn, args, _config([64], warps, 2, cute_mma_dag_exchange=exchange)
    )
    if exchange == "auto" and warps == 4:
        marker = "md_aw" if name == "_hoisted_a_chain" else "md_qsl + 2"
        assert marker in source
    ref = _run_triton(fn, args, [64])
    for lhs, rhs in zip(out, ref, strict=True):
        assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("gated", (False, True))
def test_wy_exchange_routes_are_pure_data_movement(gated: bool) -> None:
    # Four warps shuffle the chain's A operands (``auto``), eight warps the
    # in-place B operand; ``smem`` sends both through fp32 exchange tiles.
    # Same bits.
    t = _inputs_m4(bhn=4, d=128, dv=64, dtype=torch.bfloat16, seed=6)
    args = _wy_args(t, gated)
    fn = _kernel_fn("wy")
    for block_sizes, warps in (([128, 32, 32], 4), ([64, 64, 64], 8)):
        auto_source, auto = _run_cute(fn, args, _config(block_sizes, warps, 2))
        smem_source, smem = _run_cute(
            fn, args, _config(block_sizes, warps, 2, cute_mma_dag_exchange="smem")
        )
        assert f"{RT}.shfl_pick_b32(" in auto_source
        assert f"{RT}.shfl_pick_b32(" not in smem_source
        for lhs, rhs in zip(auto, smem, strict=True):
            assert torch.equal(lhs, rhs)


def test_dqkw_exchange_routes_are_pure_data_movement() -> None:
    # Four warps on 64 x 32 fragments repack the hoisted dattn into A words
    # from registers for dq (``auto``); ``smem`` reads the exchange tile for
    # both orientations.  Same bits, at every stage depth.
    t = _inputs_m4(bhn=4, d=128, dv=64, dtype=torch.bfloat16, seed=5)
    args = _dqkw_args(t, False)
    fn = _kernel_fn("dqkw")
    auto_source, auto = _run_cute(fn, args, _config([32, 32, 32], 4, 2))
    smem_source, smem = _run_cute(
        fn, args, _config([32, 32, 32], 4, 2, cute_mma_dag_exchange="smem")
    )
    assert "md_aw" in auto_source and "md_aw" not in smem_source
    for lhs, rhs in zip(auto, smem, strict=True):
        assert torch.equal(lhs, rhs)
    for stages in (1, 4):
        for lhs, rhs in zip(
            _run_cute(fn, args, _config([32, 32, 32], 4, stages))[1], auto, strict=True
        ):
            assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("name", ("_single_ring", "_single_ring_noacc"))
@pytest.mark.parametrize("warps", (4, 8))
def test_loop_outputs_in_another_slot_match_the_triton_backend(
    name: str, warps: int
) -> None:
    # The loop output of ``dot(a_i, T, acc=T * 0.5)`` / ``dot(a_i, T)``
    # lives in a fresh slot copied back into the carry at the iteration end;
    # the tcgen05 mode, which has no such copy, declines these programs.
    # Eight launches (``_run_cute``), bit-equal to Triton.
    fn = getattr(_probe_kernels(), name)
    g = torch.Generator(device=DEVICE).manual_seed(17)
    args = (
        _randn(g, 6, 4, 64, 64, scale=0.15, dtype=torch.bfloat16),
        _randn(g, 6, 64, 64, dtype=torch.bfloat16),
    )
    spec = helion.kernel(fn, backend="cute", static_shapes=True).bind(args).config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    _source, out = _run_cute(fn, args, _config([], warps, 2))
    for lhs, rhs in zip(out, _run_triton(fn, args, []), strict=True):
        assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("warps", (4, 8))
def test_a_root_dot_rescaling_its_loops_carry_matches_the_triton_backend(
    warps: int,
) -> None:
    # ``dot(kk, kk, acc=kk * 0.5)`` after the K loop takes a fresh slot.
    fn = _probe_kernels()._root_rescaled_carry
    g = torch.Generator(device=DEVICE).manual_seed(17)
    args = (_randn(g, 6, 64, 128, scale=0.15, dtype=torch.bfloat16),)
    _source, out = _run_cute(fn, args, _config([64], warps, 2))
    for lhs, rhs in zip(out, _run_triton(fn, args, [64]), strict=True):
        assert torch.equal(lhs, rhs)


def test_aliased_arguments_are_refused_at_launch() -> None:
    # ``_rmw_two_names`` reads s_in and writes s_out; compiled against
    # distinct fakes, the s_in reads are hoisted ahead of the loop.  Passing
    # one tensor under both names is refused by the launcher's alias guard
    # on the first (marshalling) launch and on the metadata-guarded
    # relaunches alike; distinct tensors run and match Triton.
    fn = _probe_kernels()._rmw_two_names
    g = torch.Generator(device=DEVICE).manual_seed(17)
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=17)
    s_in = _randn(g, 2, 64, dtype=torch.float32)
    s_out = torch.empty_like(s_in)
    args = (t["k"], t["v"], s_in, s_out)
    source, out = _run_cute(fn, args, _config([64, 64], 4, 2))
    assert "'kind': 'helion_mma_dag_alias'" in source
    ref_out = torch.empty_like(s_in)
    ref = _run_triton(fn, (t["k"], t["v"], s_in, ref_out), [64, 64])
    assert torch.equal(out[0], ref[0])
    # s_out carries a row sum of the fragment (another reduction order).
    torch.testing.assert_close(s_out, ref_out, atol=1e-3, rtol=1e-4)
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    with bound.env:
        normalized = bound.config_spec.normalized_config(_config([64, 64], 4, 2))
    compiled = bound.compile_config(normalized)
    s = s_in.clone()
    with pytest.raises(helion.exc.BackendUnsupported, match="'s_in' and 's_out' alias"):
        compiled(t["k"], t["v"], s, s)
    compiled(*args)
    compiled(*args)
    with pytest.raises(helion.exc.BackendUnsupported, match="'s_in' and 's_out' alias"):
        compiled(t["k"], t["v"], s, s)
    with pytest.raises(helion.exc.BackendUnsupported, match="'s_in' and 's_out' alias"):
        compiled(t["k"], t["v"], s[:, 32:], s)
    torch.cuda.synchronize()


@pytest.mark.parametrize("n", (0, 3))
def test_a_zero_trip_outer_loop_matches_the_triton_backend(n: int) -> None:
    # The nested ring's first prologue, peeled ahead of the outer ``for``, read
    # iteration 0's tiles of empty tensors when the outer grid loop had no
    # iterations (an illegal address).  A static zero trip count issues
    # nothing; a dynamic one (an int argument) guards the peel.
    kernels = _probe_kernels()
    g = torch.Generator(device=DEVICE).manual_seed(5)
    a = _randn(g, 2, n, 64, 128, dtype=torch.bfloat16)
    b = _randn(g, 2, n, 128, 64, dtype=torch.bfloat16)
    source, out = _run_cute(kernels._grid_nest, (a, b), _config([64, 32], 4, 3))
    assert f"cutlass.range(0, {n})" in source
    ref = _run_triton(kernels._grid_nest, (a, b), [64, 32])
    _assert_close(out, ref)
    if n:
        eager = 2.0 * (a.float() @ b.float())
        torch.testing.assert_close(out[0], eager, atol=1e-2, rtol=1e-2)
    a3 = _randn(g, 2, 3, 64, 128, dtype=torch.bfloat16)
    b3 = _randn(g, 2, 3, 128, 64, dtype=torch.bfloat16)
    args = (a3, b3, n)
    source, out = _run_cute(kernels._grid_nest_steps, args, _config([64, 32], 4, 3))
    assert "if 0 < n:" in source
    ref = _run_triton(kernels._grid_nest_steps, args, [64, 32])
    _assert_close(out, ref)


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_the_standalone_export_keeps_the_alias_guard(tmp_path: Path) -> None:
    # The standalone export allowlisted the alias plan as launcher-only
    # metadata, so the exported wrapper ran aliased arguments on the stale
    # hoisted loads.  The launch table carries the plan's positions and store
    # flags and the exported launcher refuses what the in-process guard
    # refuses; distinct arguments run and match Triton.
    fn = _probe_kernels()._rmw_two_names
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=19)
    g = torch.Generator(device=DEVICE).manual_seed(19)
    s_in = _randn(g, 2, 64, dtype=torch.float32)
    s_out = torch.empty_like(s_in)
    args = (t["k"], t["v"], s_in, s_out)
    bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
    with bound.env:
        normalized = bound.config_spec.normalized_config(_config([64, 64], 4, 2))
    code = bound.to_code(
        normalized, options=helion.OutputCodeOptions(allow_helion_deps=False)
    )
    assert "((0, 1, 2, 3, 4), (False, False, True, False, True))" in code
    path = tmp_path / "native_rmw_two_names.py"
    path.write_text(code)
    exported = _load_module(path)
    h_out = exported._rmw_two_names(*args)
    torch.cuda.synchronize()
    ref_out = torch.empty_like(s_in)
    ref = _run_triton(fn, (t["k"], t["v"], s_in, ref_out), [64, 64])
    _assert_close((h_out,), ref)
    torch.testing.assert_close(s_out, ref_out, atol=1e-3, rtol=1e-4)
    # Arguments aliasing differently from the export's trip the exporter's
    # input-alias facts guard first; the span check is reached by an export
    # whose own arguments alias: two same-layout windows of one buffer
    # sharing a row (the kernel compiles against distinct storages still).
    s = s_in.clone()
    with pytest.raises(ValueError, match="aliasing does not match"):
        exported._rmw_two_names(t["k"], t["v"], s, s)
    wide = _randn(g, 3, 64, dtype=torch.float32)
    windows = (t["k"], t["v"], wide[:2], wide[1:])
    aliased = helion.kernel(fn, backend="cute", static_shapes=True).bind(windows)
    with aliased.env:
        normalized = aliased.config_spec.normalized_config(_config([64, 64], 4, 2))
    path = tmp_path / "native_rmw_two_names_aliased.py"
    path.write_text(
        aliased.to_code(
            normalized, options=helion.OutputCodeOptions(allow_helion_deps=False)
        )
    )
    exported_aliased = _load_module(path)
    with pytest.raises(ValueError, match="Tensor arguments alias"):
        exported_aliased._rmw_two_names(*windows)
    with pytest.raises(ValueError, match="Tensor arguments alias"):
        exported_aliased._rmw_two_names(t["k"], t["v"], wide[1:], wide[1:])
    exported._rmw_two_names(*args)
    torch.cuda.synchronize()


def test_the_fast_relaunch_guards_decide_on_real_tensors() -> None:
    # The C++ tensor guard stands in for the Python metadata guards when it
    # passes; when it misses for a property the launch does not care about
    # (requires_grad) the Python guards keep the fast path -- no marshalling
    # launch -- and the result is the same.  (A different layout is not a
    # launcher matter: a static-shapes kernel is specialized on it and the
    # Helion kernel object rebinds.)
    import helion.runtime.cute.launcher as launcher_module

    fn = _probe_kernels()._rmw_two_names
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=23)
    g = torch.Generator(device=DEVICE).manual_seed(23)
    s_in = _randn(g, 2, 64, dtype=torch.float32)
    s_out = torch.empty_like(s_in)
    args = (t["k"], t["v"], s_in, s_out)
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    with bound.env:
        normalized = bound.config_spec.normalized_config(_config([64, 64], 4, 2))
    compiled = bound.compile_config(normalized)
    expected = compiled(*args).clone()
    compiled(*args)
    cute_kernel = compiled.__globals__["_helion__rmw_two_names"]
    fastpath = cute_kernel._helion_cute_fastpath
    assert isinstance(fastpath, launcher_module._CuteFastRelaunch)
    assert fastpath.cxx_guards is not None
    assert fastpath.raw_grid == (2,) and fastpath.raw_kwargs == {"block": (128, 1, 1)}
    marshalled: list[str] = []
    original = launcher_module._launch_cute_marshalled

    def counting(kernel_object: object, *rest: object) -> object:
        marshalled.append(kernel_object.__name__)
        return original(kernel_object, *rest)

    cxx_misses: list[bool] = []
    cxx_check = fastpath.cxx_check
    assert cxx_check is not None

    def spying_check(*tensors: object) -> bool:
        hit = cxx_check(*tensors)
        cxx_misses.append(not hit)
        return hit

    fastpath.cxx_check = spying_check
    with patch.object(launcher_module, "_launch_cute_marshalled", counting):
        out = compiled(*args)
        torch.cuda.synchronize()
        assert torch.equal(out, expected) and cxx_misses == [False]
        k_grad = t["k"].clone().requires_grad_(True)
        out = compiled(k_grad, t["v"], s_in, torch.empty_like(s_in))
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        assert cxx_misses == [False, True] and marshalled == []


def test_a_negative_zero_scalar_does_not_relaunch_the_positive_zero() -> None:
    # -0.0 == 0.0 let the relaunch reuse the pinned +0.0 and divide by the
    # wrong sign of zero.  Launched with 0.0, 0.0, -0.0, -0.0, 2.0, -0.0 the
    # kernel matches eager, infinities' signs included.
    fn = _probe_kernels()._div_scalar
    g = torch.Generator(device=DEVICE).manual_seed(29)
    k = (_randn(g, 2, 8, 64, 64, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    v = (_randn(g, 2, 8, 64, 64, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    rowsum = torch.einsum("bncd,bnce->bde", k.float(), v.float()).sum(-1)
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel.bind((0.0, k, v, torch.empty((2, 64), device=DEVICE)))
    with bound.env:
        normalized = bound.config_spec.normalized_config(_config([64, 64], 4, 2))
    compiled = bound.compile_config(normalized)
    for scale in (0.0, 0.0, -0.0, -0.0, 2.0, -0.0):
        out = torch.empty((2, 64), device=DEVICE)
        compiled(scale, k, v, out)
        torch.cuda.synchronize()
        ref = rowsum / scale
        assert torch.equal(out.isinf(), ref.isinf()), scale
        assert torch.equal(torch.signbit(out), torch.signbit(ref)), scale
        finite = ~ref.isinf()
        torch.testing.assert_close(out[finite], ref[finite], atol=1e-3, rtol=1e-4)


def test_the_raw_launch_record_follows_the_fastpaths_grid() -> None:
    # With the record cleared (the state a concurrent first launch could
    # leave), a launch with another grid must not record its spelling on a
    # fastpath pinned to the host's grid, so the raw relaunch never launches
    # the pinned grid for it.
    import helion.runtime.cute.launcher as launcher_module

    fn = _probe_kernels()._rmw_two_names
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=31)
    g = torch.Generator(device=DEVICE).manual_seed(31)
    s_in = _randn(g, 2, 64, dtype=torch.float32)
    state: dict[str, object] = {"grid": None, "kernel": None}

    def launcher(
        cute_kernel: object, grid: tuple[int, ...], *args: object, **kwargs: object
    ) -> object:
        state["kernel"] = cute_kernel
        forced = state["grid"]
        return launcher_module.default_cute_launcher(
            cute_kernel, forced if forced is not None else grid, *args, **kwargs
        )

    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel.bind((t["k"], t["v"], s_in, torch.empty_like(s_in)))
    with bound.env:
        normalized = bound.config_spec.normalized_config(_config([64, 64], 4, 2))
    compiled = bound.compile_config(normalized)

    def rows_written(forced: tuple[int, ...] | None) -> list[bool]:
        state["grid"] = forced
        out = torch.full_like(s_in, -7.0)
        compiled(t["k"], t["v"], s_in, out, _launcher=launcher)
        torch.cuda.synchronize()
        return [bool((out[row] != -7.0).any()) for row in range(2)]

    assert rows_written(None) == [True, True]
    assert rows_written(None) == [True, True]
    fastpath = state["kernel"]._helion_cute_fastpath  # type: ignore[attr-defined]
    assert fastpath.grid == (2, 1, 1) and fastpath.raw_grid == (2,)
    fastpath.raw_grid = None
    fastpath.raw_kwargs = None
    # One CTA only, twice: no record is taken from it.
    assert rows_written((1,)) == [True, False]
    assert fastpath.raw_grid is None
    assert rows_written((1,)) == [True, False]
    # The host's own grid records and relaunches fast.
    assert rows_written(None) == [True, True]
    assert fastpath.raw_grid == (2,)
    assert rows_written(None) == [True, True]


def _head_sum_args(
    *, m: int = 256, h: int = 4, n: int = 512, d: int = 128, seed: int = 11
) -> tuple[torch.Tensor, ...]:
    """Integer bf16 queries / keys (every fp32 partial sum exact, scores up
    to ~1500 so their bf16 rounding is exercised) and quarter-step weights:
    any accumulation order and contraction gives the same bits."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def ints(*shape: int, low: int, high: int) -> torch.Tensor:
        return torch.randint(low, high, shape, device=DEVICE, generator=g)

    ks = ints(m, low=0, high=n // 2).to(torch.int32)
    return (
        ints(m, h, d, low=-8, high=9).to(torch.bfloat16),
        ints(n, d, low=-8, high=9).to(torch.bfloat16),
        ints(m, h, low=-8, high=9).to(torch.float32) / 4,
        ks,
        ks + ints(m, low=0, high=n // 2).to(torch.int32),
    )


@pytest.mark.parametrize(
    ("block_sizes", "warps", "stages"), (([64, 128], 4, 2), ([32, 64], 2, 3))
)
def test_a_16_bit_matmul_with_a_tile_index_mask_matches_the_triton_backend(
    block_sizes: list[int], warps: int, stages: int
) -> None:
    # The sparse-attention indexer's logits: a bf16 ``torch.matmul`` per
    # head read rounded, and a window mask of ``tile_n.index`` against
    # int32 bounds.  Bit-equal to Triton over eight launches.
    fn = _probe_kernels()._head_sum_logits
    args = _head_sum_args()
    source, out = _run_cute(fn, args, _config(block_sizes, warps, stages))
    assert f"{RT}.ldg_s32(" in source
    (ref,) = _run_triton(fn, args, block_sizes)
    assert torch.isinf(ref).any() and torch.isfinite(ref).any()
    assert torch.equal(out[0], ref)


# --------------------------------------------------------------------------
# Tile loops whose bound varies with the CTA's root tile (mamba2 chunk scan)
# --------------------------------------------------------------------------


def _causal_scaled(a: torch.Tensor, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A causal K loop whose bound is the row tile's end: the pass over
    ``a`` (its scale adds one, so a zero-filled element would not vanish)
    must zero the K positions past the bound of a partial last trip."""
    m, k = a.size()
    _, n = b.size()
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(64, 64)
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[block_m, block_n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile((tile_m.id + 1) * block_m, block_size=block_k):
            a_local = (
                a[tile_m, tile_k].to(torch.float32)
                * (s[tile_k].to(torch.float32)[None, :] + 1.0)
            ).to(a.dtype)
            acc = hl.dot(a_local, b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _causal_scaled_reference(
    a: torch.Tensor, s: torch.Tensor, b: torch.Tensor, block_m: int
) -> torch.Tensor:
    m, k = a.size()
    scaled = (a.float() * (s.float()[None, :] + 1.0)).to(a.dtype).float()
    end = (torch.arange(m, device=a.device) // block_m + 1) * block_m
    keep = (torch.arange(k, device=a.device)[None, :] < end[:, None]).float()
    return ((scaled * keep) @ b.float()).to(a.dtype)


@pytest.mark.parametrize(
    ("block_m", "warps", "stages"), [(16, 4, 2), (32, 2, 2), (64, 4, 2), (128, 8, 2)]
)
def test_varying_bound_loops_mask_the_partial_trip(
    block_m: int, warps: int, stages: int
) -> None:
    # 256 rows: the row tiles walk 1..256/64 trips, so the trip count is per CTA.
    g = torch.Generator(device=DEVICE).manual_seed(0)
    a = _randn(g, 256, 256, dtype=torch.bfloat16)
    s = torch.rand(256, device=DEVICE, generator=g).to(torch.bfloat16)
    b = _randn(g, 256, 64, dtype=torch.bfloat16)
    source, (out,) = _run_cute(
        _causal_scaled, (a, s, b), _config([block_m, 64, 64], warps, stages)
    )
    assert "for md_i in cutlass.range(0, md_trips0):" in source
    assert ("< md_end0 else cutlass.Float32(0.0)" in source) == (block_m < 64)
    expected = _causal_scaled_reference(a, s, b, block_m)
    _assert_close((out,), (expected,), scaled=True)
    # Without the mask every CTA would walk whole 64-wide trips.
    if block_m < 64:
        unmasked = _causal_scaled_reference(a, s, b, 64)
        assert not torch.allclose(unmasked.float(), expected.float(), atol=1.0)


def test_mamba2_chunk_scan_matches_the_triton_backend() -> None:
    module = importlib.import_module("examples.mamba2_chunk_scan")
    fn = module.helion_mamba2_chunk_scan_kernel.fn
    g = torch.Generator(device=DEVICE).manual_seed(0)
    batch, heads, groups, seq, chunk, d, ds = 2, 8, 2, 512, 64, 64, 64
    chunks = seq // chunk
    bf16 = torch.bfloat16
    dt = torch.rand(batch, heads, chunks, chunk, device=DEVICE, generator=g).to(bf16)
    da = (
        -torch.rand(batch, heads, chunks, chunk, device=DEVICE, generator=g)
        .cumsum(-1)
        .to(bf16)
    )
    args = (
        _randn(g, batch, chunks, groups, chunk, chunk, dtype=bf16),
        _randn(g, batch, seq, heads, d, dtype=bf16),
        dt,
        da,
        _randn(g, batch, seq, groups, ds, dtype=bf16),
        _randn(g, batch, chunks, heads, d, ds, dtype=bf16),
        _randn(g, heads, dtype=bf16),
    )
    refs = _run_triton(fn, args, [32, 64, 64])
    for block_m, warps, stages in ((16, 2, 2), (32, 4, 2), (64, 8, 4)):
        source, outs = _run_cute(fn, args, _config([block_m, 64, 64], warps, stages))
        assert ("md_end0 else" in source) == (block_m < 64)
        _assert_close(outs, refs, scaled=True)
        # The aux ring stages dt / dA_cumsum through shared memory: the same
        # fp32 values reach the same arithmetic.
        source, ringed = _run_cute(
            fn, args, _config([block_m, 64, 64], warps, stages, cute_mma_dag_aux="ring")
        )
        assert f"{RT}.lds_v4_b32(md_aux0" in source
        for lhs, rhs in zip(ringed, outs, strict=True):
            assert torch.equal(lhs, rhs)
        # The TMA ring (one bulk tensor copy per ring tile) reaches the same
        # values, with either side-load form.
        for aux in ("register", "ring"):
            source, bulk = _run_cute(
                fn,
                args,
                _config(
                    [block_m, 64, 64],
                    warps,
                    stages,
                    cute_mma_dag_loads="tma",
                    cute_mma_dag_aux=aux,
                ),
            )
            assert "cp_async_bulk_tensor_shared_cta_global" in source
            assert f"{RT}.cp_async_16(md_t2" not in source
            # The epilogue's x tile is staged ahead of the loops and read
            # from shared memory (no per-lane global pair after the loop).
            epilogue = source[source.rindex("for md_i in cutlass.range(") :]
            assert f"{RT}.lds_b32(md_d0 + " in epilogue
            assert f"{RT}.ldg_b32(" not in epilogue
            for lhs, rhs in zip(bulk, outs, strict=True):
                assert torch.equal(lhs, rhs)


def test_the_aux_ring_survives_the_early_nested_prologue() -> None:
    # An outer tile loop whose innermost ring loop scales the A operand by a
    # per-row gate and stores the accumulator after it: the outer loop's
    # epilogue issues the nested ring's prologue for its next iteration, and
    # that prologue copies the gate words of the aux ring too (it used to
    # leave the previous iteration's words in the slot).  Ring and register
    # forms are bit-equal and match Triton.
    from test.test_cute_mma_dag import _nested_gated_probe

    g = torch.Generator(device=DEVICE).manual_seed(0)
    c, d, dv = 512, 128, 128
    args = (
        _randn(g, c, d, dtype=torch.bfloat16),
        _randn(g, c, dv, dtype=torch.bfloat16),
        _randn(g, c, dtype=torch.bfloat16),
    )
    refs = _run_triton(_nested_gated_probe, args, [64, 64, 64])
    for stages in (2, 4):
        source, register = _run_cute(
            _nested_gated_probe, args, _config([64, 64, 64], 4, stages)
        )
        assert "if md_i + 1 < 2:" in source
        source, ring = _run_cute(
            _nested_gated_probe,
            args,
            _config([64, 64, 64], 4, stages, cute_mma_dag_aux="ring"),
        )
        early = source[source.index("if md_i + 1 < 2:") :]
        assert f"{RT}.cp_async_4(md_aux1 + " in early
        for lhs, rhs in zip(ring, register, strict=True):
            assert torch.equal(lhs, rhs)
        _assert_close(ring, refs)


def _tensor_with_address_bit_31(
    shape: tuple[int, ...], dtype: torch.dtype, g: torch.Generator
) -> torch.Tensor | None:
    """A random tensor whose data pointer has bit 31 set (the caching
    allocator walks upward, so one of a few 256 MB allocations does), or
    None when none turns up within 3 GB."""
    keep = []
    for _ in range(12):
        block = torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE)
        keep.append(block)
        ptr = block.data_ptr()
        if (ptr >> 31) & 1:
            numel = 1
            for size in shape:
                numel *= size
            flat = block[: numel * dtype.itemsize].view(dtype)
            flat.copy_(torch.randn(numel, device=DEVICE, generator=g).to(dtype))
            return flat.view(shape)
    return None


def test_the_aux_ring_word_alignment_survives_address_bit_31() -> None:
    # The aux ring copies a 16-bit side load as the aligned 4-byte word
    # holding it and reads the element at the address's low two bits.  Both
    # took ``cutlass.Int32(address) % 4``: a signed remainder, -2 for an odd
    # element whenever bit 31 of the address is set (half of all allocations),
    # which copied the next word and read ahead of the slot.  The low bits
    # are masked now; a gate tensor at such an address matches the register
    # form bit for bit.
    from test.test_cute_mma_dag import _nested_gated_probe

    g = torch.Generator(device=DEVICE).manual_seed(5)
    c, d, dv = 512, 128, 128
    gate = _tensor_with_address_bit_31((c,), torch.bfloat16, g)
    if gate is None:
        pytest.skip("no allocation with address bit 31 set within 3 GB")
    args = (
        _randn(g, c, d, dtype=torch.bfloat16),
        _randn(g, c, dv, dtype=torch.bfloat16),
        gate,
    )
    source, register = _run_cute(_nested_gated_probe, args, _config([64, 64, 64], 4, 2))
    source, ring = _run_cute(
        _nested_gated_probe, args, _config([64, 64, 64], 4, 2, cute_mma_dag_aux="ring")
    )
    assert (
        "- (cutlass.Int32(md_aw" in source
        and "% 4" not in source.split("md_aux1")[1][:400]
    )
    for lhs, rhs in zip(ring, register, strict=True):
        assert torch.equal(lhs, rhs)
    _assert_close(ring, _run_triton(_nested_gated_probe, args, [64, 64, 64]))


def test_a_partial_trip_does_not_read_the_other_operand_past_the_bound() -> None:
    # chunk_scan at 32-row tiles: the first row tile of a chunk runs its
    # causal K loop to 32 with a 64-deep K block, so x's rows 32-63 of the
    # chunk sit in the ring slot past the bound.  cb's pass zeroes its side;
    # x's rows are now zeroed in shared memory too, so an Inf planted in
    # the chunk's last row (read legitimately only by the second row tile)
    # leaves the first row tile's output finite, as in the reference.
    module = importlib.import_module("examples.mamba2_chunk_scan")
    fn = module.helion_mamba2_chunk_scan_kernel.fn
    g = torch.Generator(device=DEVICE).manual_seed(1)
    batch, heads, groups, seq, chunk, d, ds = 1, 4, 1, 256, 64, 64, 64
    chunks = seq // chunk
    bf16 = torch.bfloat16
    dt = torch.rand(batch, heads, chunks, chunk, device=DEVICE, generator=g).to(bf16)
    da = (
        -torch.rand(batch, heads, chunks, chunk, device=DEVICE, generator=g)
        .cumsum(-1)
        .to(bf16)
    )
    x = _randn(g, batch, seq, heads, d, dtype=bf16)
    x[0, chunk - 1, :, 0] = float("inf")
    args = (
        _randn(g, batch, chunks, groups, chunk, chunk, dtype=bf16),
        x,
        dt,
        da,
        _randn(g, batch, seq, groups, ds, dtype=bf16),
        _randn(g, batch, chunks, heads, d, ds, dtype=bf16),
        _randn(g, heads, dtype=bf16),
    )
    (ref,) = _run_triton(fn, args, [32, 64, 64])
    assert torch.isfinite(ref[0, :32]).all()
    for loads in ("lane", "tma"):
        # (Not ``_run_cute``: its launch-to-launch ``torch.equal`` cannot
        # hold with the NaNs the second row tile legitimately produces.)
        kernel = helion.kernel(fn, backend="cute", static_shapes=True)
        bound = kernel.bind(args)
        config = _config([32, 64, 64], 4, 2, cute_mma_dag_loads=loads)
        with bound.env:
            normalized = bound.config_spec.normalized_config(config)
        source = bound.to_code(normalized)
        assert f"{RT}.mma_" in source
        assert source.count(">= md_end0:") == 4
        out = bound.compile_config(normalized)(*args)
        torch.cuda.synchronize()
        assert torch.isfinite(out[0, :32]).all()
        assert torch.equal(torch.isfinite(out), torch.isfinite(ref))
        finite = torch.isfinite(ref)
        torch.testing.assert_close(
            out[finite].float(), ref[finite].float(), atol=2e-2, rtol=2e-2
        )


@pytest.mark.parametrize(
    ("has_h0", "bm", "bn", "warps", "stages"),
    [(False, 16, 16, 2, 8), (True, 32, 16, 4, 4), (False, 64, 64, 4, 2)],
)
def test_the_aux_ring_matches_the_register_form_on_the_gated_chain(
    has_h0: bool, bm: int, bn: int, warps: int, stages: int
) -> None:
    # fwd_h's per-chunk decay scalar and gc rows ride the ring as per-thread
    # word slots (the mamba2 ssd pipeline's configs).
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16)
    args = _fwd_args(t, "gated", has_h0)
    fn = _kernel_fn("fwd")
    source, ring = _run_cute(
        fn, args, _config([bm, bn], warps, stages, cute_mma_dag_aux="ring")
    )
    assert f"{RT}.cp_async_4(md_aux0" in source and f"{RT}.lds_f32(md_aux0" in source
    _source, register = _run_cute(fn, args, _config([bm, bn], warps, stages))
    for lhs, rhs in zip(ring, register, strict=True):
        assert torch.equal(lhs, rhs)
    _assert_close(ring, _run_triton(fn, args, [bm, bn]))
    # The TMA ring with each side-load form (the aux copies sit behind the
    # issuing thread's bulk copies in the same prologue stage).
    for aux in ("register", "ring"):
        source, bulk = _run_cute(
            fn,
            args,
            _config(
                [bm, bn], warps, stages, cute_mma_dag_loads="tma", cute_mma_dag_aux=aux
            ),
        )
        assert "cp_async_bulk_tensor_shared_cta_global" in source
        for lhs, rhs in zip(bulk, register, strict=True):
            assert torch.equal(lhs, rhs)


@pytest.mark.parametrize(
    ("block_sizes", "warps"), (([1, 128], 8), ([1, 64], 4), ([1, 256], 8))
)
def test_a_loop_less_batched_matmul_matches_the_triton_backend(
    block_sizes: list[int], warps: int
) -> None:
    # The indexer's decode form: no device loop, one ``aten.bmm`` over the
    # 32 heads staged once per CTA, the head sum a column reduction, the
    # masked store a vector store.  Bit-equal to Triton over eight launches.
    fn = _probe_kernels()._head_sum_logits_batched
    args = _head_sum_args(h=32)
    source, out = _run_cute(fn, args, _config(block_sizes, warps, 4))
    assert "for md_i" not in source and f"{RT}.octet_sum(" in source
    (ref,) = _run_triton(fn, args, block_sizes)
    assert torch.isinf(ref).any() and torch.isfinite(ref).any()
    assert torch.equal(out[0], ref)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(("block_size", "warps"), ((64, 4), (256, 8), (32, 1)))
def test_a_cumsum_of_a_fragment_matches_torch(
    reverse: bool, block_size: int, warps: int
) -> None:
    # Integer bf16 operands: the fp32 products and their prefix sums are
    # exact in any order, so the scan (octet shuffles, chained halves and
    # atoms, the warps along M through the scratch tile) must be bit-equal
    # to torch's cumsum of the product.
    fn = _probe_kernels()._loop_less_dot_scanned
    g = torch.Generator(device=DEVICE).manual_seed(5)
    a = torch.randint(-4, 5, (6, 32, 128), device=DEVICE, generator=g).to(
        torch.bfloat16
    )
    b = torch.randint(-4, 5, (6, 128, 512), device=DEVICE, generator=g).to(
        torch.bfloat16
    )
    source, out = _run_cute(fn, (a, b, reverse), _config([block_size], warps, 4))
    assert f"{RT}.{'shfl_down' if reverse else 'shfl_up'}_add_f32(" in source
    product = torch.bmm(a.float(), b.float()) * 0.5
    ref = product.flip(1).cumsum(1).flip(1) if reverse else product.cumsum(1)
    assert torch.equal(out[0], ref)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    ("block_sizes", "warps", "stages"), (([64, 64], 4, 2), ([128, 32], 8, 3))
)
def test_a_cumsum_inside_a_k_loop_matches_torch(
    reverse: bool, block_sizes: list[int], warps: int, stages: int
) -> None:
    # The scan of the K-loop carry runs inside the loop (a slot of its
    # own, the carry stays live; the cross-warp exchange behind the
    # iteration's barriers).  Integer operands: the last iteration's
    # prefix sums down each M tile are exact, bit-equal to torch's cumsum
    # of the product per tile.
    fn = _probe_kernels()._k_loop_dot_scanned
    g = torch.Generator(device=DEVICE).manual_seed(7)
    a = torch.randint(-4, 5, (256, 512), device=DEVICE, generator=g).to(torch.bfloat16)
    b = torch.randint(-4, 5, (512, 64), device=DEVICE, generator=g).to(torch.bfloat16)
    source, out = _run_cute(fn, (a, b, reverse), _config(block_sizes, warps, stages))
    assert "for md_i" in source
    assert f"{RT}.{'shfl_down' if reverse else 'shfl_up'}_add_f32(" in source
    product = (a.float() @ b.float()).view(-1, block_sizes[0], 64)
    ref = product.flip(1).cumsum(1).flip(1) if reverse else product.cumsum(1)
    assert torch.equal(out[0], ref.view(256, 64))


def test_a_cumsum_inside_an_inner_tile_loop_matches_torch() -> None:
    fn = _probe_kernels()._inner_loop_dot_scanned
    g = torch.Generator(device=DEVICE).manual_seed(9)
    a = torch.randint(-4, 5, (6, 32, 128), device=DEVICE, generator=g).to(
        torch.bfloat16
    )
    b = torch.randint(-4, 5, (6, 128, 512), device=DEVICE, generator=g).to(
        torch.bfloat16
    )
    source, out = _run_cute(fn, (a, b), _config([64], 4, 2))
    assert "for md_i" in source and f"{RT}.shfl_up_add_f32(" in source
    assert torch.equal(out[0], (torch.bmm(a.float(), b.float()) * 0.5).cumsum(1))


def test_a_cumsum_of_a_live_fragment_keeps_its_input() -> None:
    # The scanned value is stored too, so the scan writes a fresh slot and
    # the dot's registers reach their store unchanged.
    fn = _probe_kernels()._loop_less_dot_scanned_and_stored
    g = torch.Generator(device=DEVICE).manual_seed(3)
    a = torch.randint(-4, 5, (6, 32, 128), device=DEVICE, generator=g).to(
        torch.bfloat16
    )
    b = torch.randint(-4, 5, (6, 128, 512), device=DEVICE, generator=g).to(
        torch.bfloat16
    )
    _source, (scanned, plain) = _run_cute(fn, (a, b), _config([64], 4, 4))
    product = torch.bmm(a.float(), b.float()) * 0.5
    assert torch.equal(plain, product)
    assert torch.equal(scanned, product.cumsum(1))


@pytest.mark.parametrize(("block_size", "warps"), ((64, 4), (128, 8), (32, 2)))
def test_a_cumsum_whose_value_reads_a_vector_matches_torch(
    block_size: int, warps: int
) -> None:
    # dqkg's tail in miniature: the fragment's column sums folded into its
    # last row under an iota mask, then the reverse prefix sum.  Integer
    # operands keep the products, the column sums and the prefix sums exact,
    # so the scan over a value that reads the reduction's vector must be
    # bit-equal to torch.
    fn = _probe_kernels()._loop_less_dot_scanned_with_a_vector
    g = torch.Generator(device=DEVICE).manual_seed(17)
    a = torch.randint(-4, 5, (6, 32, 128), device=DEVICE, generator=g).to(
        torch.bfloat16
    )
    b = torch.randint(-4, 5, (6, 128, 512), device=DEVICE, generator=g).to(
        torch.bfloat16
    )
    source, out = _run_cute(fn, (a, b), _config([block_size], warps, 4))
    assert f"{RT}.shfl_down_add_f32(" in source and f"{RT}.octet_sum(" in source
    product = torch.bmm(a.float(), b.float())
    value = product.clone()
    value[:, -1, :] += product.sum(1)
    assert torch.equal(out[0], value.flip(1).cumsum(1).flip(1))


def _dqkg_args(
    t: dict[str, torch.Tensor], diag: bool, compute_dg: bool = True
) -> tuple[object, ...]:
    if diag:
        g = torch.Generator(device=DEVICE).manual_seed(11)
        bhn, c, d = t["q"].shape
        gcs = _randn(g, bhn, c, d, scale=0.1, dtype=torch.float32)
        gl = None
    else:
        gcs, gl = t["gcs"], t["gl"]
    return (
        t["q"],
        t["k"],
        t["v"],
        gcs,
        t["h"],
        t["do"],
        t["dh"],
        gl,
        diag,
        compute_dg,
        0.125,
    )


# diag_anchored, (d, dv), block sizes (lead, tile_d, tile_dv), warps, stages
_DQKG_CASES = (
    (False, (128, 128), [1, 128, 32], 8, 2),
    (False, (64, 64), [1, 32, 64], 4, 2),
    (False, (128, 64), [1, 64, 64], 8, 1),
    (True, (128, 128), [1, 64, 64], 8, 2),
    (True, (64, 128), [1, 32, 64], 4, 2),
)


@pytest.mark.parametrize(
    ("diag", "shape", "block_sizes", "warps", "stages"),
    _DQKG_CASES,
    ids=[
        f"{'anchored' if dg else 'scalar'}-d{s[0]}dv{s[1]}-{'x'.join(map(str, bs))}w{w}s{st}"
        for dg, s, bs, w, st in _DQKG_CASES
    ],
)
def test_dqkg_with_the_gate_gradient_matches_the_triton_backend(
    diag: bool,
    shape: tuple[int, int],
    block_sizes: list[int],
    warps: int,
    stages: int,
) -> None:
    # ``chunk_bwd_dqkg_scalar_helion(compute_dg=True)``: the training
    # backward of simple_gla / full_gla (the gate's gradient).  dq is
    # bit-equal; dk differs from Triton by one bf16 ulp on a few elements
    # (``dot * scale + dk_state`` is one FMA there, a mul and an add here --
    # the same elements as the ``compute_dg=False`` path); dg_by_d (fp32,
    # the reverse prefix sum of ``dg_raw`` with the state-carry term folded
    # into the last row) differs by fp32 summation order only.
    d, dv = shape
    t = _inputs3(bhn=6, d=d, dv=dv, dtype=torch.bfloat16)
    args = _dqkg_args(t, diag)
    fn = _kernel_fn("dqkg")
    source, (dq, dk, dg) = _run_cute(fn, args, _config(block_sizes, warps, stages))
    assert f"{RT}.shfl_down_add_f32(" in source
    rq, rk, rg = _run_triton(fn, args, block_sizes)
    assert torch.equal(dq, rq)
    _assert_close((dk,), (rk,), scaled=True)
    _assert_close((dg,), (rg,))


def test_anchored_wy_state_key_matches_the_triton_backend() -> None:
    # The anchored WY transform's side output ``k_state_out = k *
    # exp2(gc[C-1] - gc)`` (a derived value over two loaded tiles, stored
    # from the loop body) is written in place: separate buffers per backend.
    t = _inputs_kda(bh=2, n=3, d=64, dv=64)
    fn = _kernel_fn("wy_anch")
    outs = []
    akk = _anchored_gram(t)
    for backend in ("cute", "triton"):
        kso = torch.empty_like(t["k"])
        args = (t["k"], t["v"], t["beta"], t["gc"], akk, kso, False, True)
        if backend == "cute":
            _source, out = _run_cute(fn, args, _config([32, 32], 4, 2))
        else:
            out = _run_triton(fn, args, [32, 32])
        outs.append((*out, kso))
    for lhs, rhs in zip(*outs, strict=True):
        assert torch.equal(lhs, rhs)


# The config whose kernel died with CUDA_ERROR_ILLEGAL_INSTRUCTION in the kda
# forward's cold autotune at (8, 8, 1024, 64, 64): a warp mma_dag whose
# ``load_eviction_policies`` names "l2_last" at the staged k and v slots.  The
# emitter once hinted those cp.async packets ``.L2::cache_hint``, which the
# CuTe DSL's nvJitLink (13.4.46rc1) miscompiles (``emit_issue_tile``).
_WY_FAULT_CONFIG: dict[str, object] = {
    "block_sizes": [8, 8],
    "cute_collective_compute": "warp",
    "cute_collective_copy": "scalar",
    "cute_collective_mma": False,
    "cute_collective_native_seeded": False,
    "cute_collective_tmem_a": False,
    "cute_collective_tmem_seed": False,
    "cute_lane_layouts": ["blocked"] * 5,
    "cute_min_blocks_per_mp": 0,
    "cute_mma_dag": "warp",
    "cute_mma_dag_derived": "exchange",
    "cute_mma_dag_epilogue": "row",
    "cute_mma_dag_exchange": "auto",
    "cute_mma_dag_loads": "tma",
    "cute_mma_dag_lookahead": "off",
    "cute_mma_dag_pass_warps": "off",
    "cute_mma_dag_persistent": 0,
    "cute_mma_dag_stages": 8,
    "cute_mma_dag_store": "staged",
    "cute_mma_dag_warps": 4,
    "cute_pdl": True,
    "cute_reduction_reloads": ["auto"],
    "cute_register_chain": False,
    "cute_store_policy": "default",
    "cute_vector_widths": [1] * 5,
    "load_eviction_policies": ["", "", "l2_last", "", "", "l2_last"],
    "num_threads": [0, 0, 0],
}


def test_the_anchored_wy_autotune_fault_config_is_verified_before_it_runs() -> None:
    # The candidate that died in the kda autotune renders hinted packets
    # again, behind the launcher's SASS check: on the DSL's current
    # assembler the check refuses the kernel before any launch (its split
    # packets read UR0 / UR1, registers no instruction of the kernel
    # writes: the dropped operand fields that faulted); on an assembler
    # that encodes it correctly the kernel runs bit-equal to the plain
    # render and to Triton.  Either way the plain config is untouched.
    t = _inputs_kda(bh=64, n=16, d=64, dv=64)
    fn = _kernel_fn("wy_anch")
    akk = _anchored_gram(t)
    kso = torch.empty_like(t["k"])
    args = (t["k"], t["v"], t["beta"], t["gc"], akk, kso, False, True)
    plain_source, plain = _run_cute(
        fn,
        args,
        helion.Config.from_dict(
            {**_WY_FAULT_CONFIG, "load_eviction_policies": [""] * 6}
        ),
    )
    assert "cp_async_16_l2" not in plain_source
    assert "_helion_cute_sass_checks" not in plain_source
    plain = (*plain, kso.clone())
    kso = torch.empty_like(t["k"])
    args = (t["k"], t["v"], t["beta"], t["gc"], akk, kso, False, True)
    try:
        hinted_source, hinted = _run_cute(
            fn, args, helion.Config.from_dict(_WY_FAULT_CONFIG)
        )
    except exc.CuteSassCheckFailed as error:
        assert "hinted_cp_async" in str(error)
        assert re.search(r"LDGSTS\S* \[R\d+\+UR\d+", str(error)), str(error)
        assert "which no instruction of the kernel writes" in str(error)
        assert "assembled by Cuda compilation tools" in str(error)
        torch.cuda.synchronize()  # the context is intact: nothing launched
    else:
        # (k's and v's packets hinted; the derived ring load's tile, "" in
        # the config, rides a tensor map: the CPU pins hold the counts.)
        assert hinted_source.count(f"{RT}.cp_async_16_l2(") > 0
        assert "_helion_cute_sass_checks = ('hinted_cp_async',)" in hinted_source
        for lhs, rhs in zip((*hinted, kso), plain, strict=True):
            assert torch.equal(lhs, rhs)
    kso = torch.empty_like(t["k"])
    args = (t["k"], t["v"], t["beta"], t["gc"], akk, kso, False, True)
    ref = (*_run_triton(fn, args, [8, 8]), kso)
    for lhs, rhs in zip(plain, ref, strict=True):
        assert torch.equal(lhs, rhs)


@pytest.mark.parametrize(
    ("policies", "words"),
    (
        (("l2_last", "l2_last"), ("evict_last",)),
        (("streaming", "l1_l2_last"), ("evict_first", "evict_last")),
    ),
)
def test_l2_hinted_operand_packets_are_a_pure_cache_hint(
    policies: tuple[str, str], words: tuple[str, ...]
) -> None:
    # ``load_eviction_policies`` on the staged operands' slots (k: 0, q: 1)
    # puts an L2 cache policy on their cp.async packets: the same bits as
    # the plain kernel and as Triton over eight launches.  The launcher's
    # SASS check passed for this kernel (it ran): the indexer decode's
    # hinted packets are the campaign's winning form.
    fn = _probe_kernels()._head_sum_logits_batched
    args = _head_sum_args(m=4, h=32)
    hinted = [*policies, "", "", "", "", ""]
    source, out = _run_cute(
        fn, args, _config([1, 64], 4, 4, load_eviction_policies=hinted)
    )
    assert source.count(f"{RT}.cp_async_16_l2(") == 12
    assert f"{RT}.cp_async_16(" not in source
    assert "_helion_cute_sass_checks = ('hinted_cp_async',)" in source
    for word in words:
        assert f"{RT}.l2_policy_{word}()" in source
    _, plain = _run_cute(fn, args, _config([1, 64], 4, 4))
    assert torch.equal(out[0], plain[0])
    (ref,) = _run_triton(fn, args, [1, 64])
    assert torch.equal(out[0], ref)


def test_l2_hinted_zero_filled_packets_match_torch() -> None:
    # The ragged root tile's zero-filled packets carry the policy behind
    # their byte count; integer inputs: exact.
    fn = _probe_kernels()._ragged_rows_summed
    a, b = _ragged_sum_args(8)
    source, (out,) = _run_cute(
        fn,
        (a, b),
        _config([16, 128], 4, 2, load_eviction_policies=["l2_last", "l2_last"]),
    )
    assert f"{RT}.cp_async_16_l2_zfill(" in source
    assert f"{RT}.cp_async_16_zfill(" not in source
    ref = (a.float() @ b.float() + 1.0).sum(0)
    assert torch.equal(out, ref)


def test_the_hinted_decode_kernels_sass_has_no_collision_under_several_configs() -> (
    None
):
    # The SASS check read against the assembler's actual output for the
    # indexer decode program at the tiles the campaign reaches (one, four
    # and eight warps; 16 / 32 / 64 keys): every uniform register an LDGSTS
    # names is written by the kernel (the defect's signature is absent),
    # every LDGSTS keeps its shared address in a vector register (the
    # conservative form condition), and the hinted kernels run bit-equal
    # to the plain ones.
    fn = _probe_kernels()._head_sum_logits_batched
    args = _head_sum_args(m=1, h=32, n=4096)
    for block_n, warps in ((16, 1), (16, 4), (32, 4), (32, 8), (64, 4)):
        hinted = _config(
            [1, block_n], warps, 4, load_eviction_policies=["l2_last"] * 2 + [""] * 5
        )
        kernel = helion.kernel(fn, backend="cute", static_shapes=True)
        bound = kernel.bind(args)
        with bound.env:
            normalized = bound.config_spec.normalized_config(hinted)
        compiled = bound.compile_config(normalized)
        (out,) = _tensors(compiled(*args))
        torch.cuda.synchronize()
        module = sys.modules[compiled.__module__]
        cute_kernel = next(
            value
            for value in vars(module).values()
            if getattr(value, "_helion_cute_sass_checks", ())
        )
        (launcher,) = cute_kernel._helion_cute_compiled_launchers.values()
        sass = "\n".join(
            sass_check.disassemble(blob)
            for blob in sass_check._cubin_blobs(launcher._compiled.ir_module)
        )
        (texts,) = sass_check.functions(sass)
        ldgsts = [text for text in texts if text.startswith("LDGSTS")]
        assert ldgsts, (block_n, warps)
        assert sass_check.unwritten_uniform_register_read(sass) is None, (
            block_n,
            warps,
        )
        assert sass_check.uniform_shared_address_beside_descriptor(sass) is None, (
            block_n,
            warps,
        )
        written = sass_check.written_uniform_registers(texts)
        assert all(
            ur in written for text in ldgsts for ur in re.findall(r"\bUR\d+\b", text)
        ), (ldgsts, sorted(written))
        assert (
            sass_check.assembler(
                next(iter(sass_check._cubin_blobs(launcher._compiled.ir_module)))
            )
            is not None
        )
        _, (plain,) = _run_cute(fn, args, _config([1, block_n], warps, 4))
        assert torch.equal(out, plain), (block_n, warps)


def _se_block(kernel: str) -> Callable[..., torch.Tensor]:
    """The tf32 form of examples/se_block.py's backward (``TF32_SE_BLOCK`` in
    test_cute_mma_dag): the example itself now runs half-operand matmuls."""
    return _probe_kernels().TF32_SE_BLOCK[kernel]


def _se_block_inputs(m: int = 512, n: int = 256) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(0)
    x = _randn(g, m, n, dtype=torch.float16)
    w = _randn(g, n, n, scale=0.05, dtype=torch.float16)
    grad = _randn(g, m, n, dtype=torch.float16)
    return {"x": x, "w": w, "s": torch.sigmoid(x @ w), "grad": grad}


@pytest.mark.parametrize("derived", ("exchange", "pass"))
@pytest.mark.parametrize(
    ("kernel", "block_sizes", "warps"),
    (
        ("se_block_bwd_dw", [64, 64, 64], 4),
        ("se_block_bwd_dx", [64, 64, 64], 8),
        ("se_block_bwd_dx", [64, 64, 64], 4),
    ),
)
def test_se_block_backward_matches_the_triton_backend(
    kernel: str, block_sizes: list[int], warps: int, derived: str
) -> None:
    # fp32 ``acc += a @ b`` (``aten.addmm``) over pointwise-derived fp32
    # operands: tf32 MMAs under the default dot precision, as Triton's
    # ``tl.dot(..., input_precision='tf32')`` runs the same source.  dx's
    # pre-seed ``acc += 2.0 * grad_out * s`` is a half-typed derived value
    # (torch promotion) widened into the fp32 registers.  The derived
    # operand ``2 * g * x * s * (1 - s)`` (dx's A, dw's B) either goes
    # through the register-direct fragment (``exchange``) or is written by
    # the derived pass from the staged source tiles (``pass``).  Bit-equal
    # either way.
    t = _se_block_inputs()
    fn = _se_block(kernel)
    if kernel == "se_block_bwd_dw":
        args: tuple[object, ...] = (t["grad"], t["x"], t["s"])
    else:
        args = (t["grad"], t["x"], t["w"], t["s"])
    source, out = _run_cute(
        fn, args, _config(block_sizes, warps, 2, cute_mma_dag_derived=derived)
    )
    assert f"{RT}.mma_tf32(" in source
    if derived == "pass":
        assert "md_x0" not in source
        assert f"{RT}.lds_v4_b32(md_d" in source
    ref = _run_triton(fn, args, block_sizes)
    for lhs, rhs in zip(out, ref, strict=True):
        assert torch.equal(lhs, rhs)


def _half_chain_args() -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device=DEVICE).manual_seed(3)
    m, k, n = 512, 256, 256
    grad = _randn(g, m, k, dtype=torch.bfloat16)
    x = _randn(g, m, k, dtype=torch.bfloat16)
    s = torch.sigmoid(_randn(g, m, k, dtype=torch.bfloat16))
    w = _randn(g, k, n, scale=0.1, dtype=torch.bfloat16)
    return grad, x, s, w


@pytest.mark.parametrize(
    ("derived", "warps"), (("exchange", 4), ("pass", 4), ("pass", 8))
)
def test_a_half_typed_pass_chain_matches_the_triton_backend(
    derived: str, warps: int
) -> None:
    # A bf16-typed chain ``2 * g * x * (1 - s)`` (torch promotion) feeding a
    # bf16 dot.  The pass evaluates the chain as the SIMT lowering does and
    # packs each eight-element group once into the 16-bit A tile; the
    # exchange route rounds the same values into the fragment registers.
    # Bit-equal to Triton either way.
    fn = _probe_kernels()._half_chain_operand
    args = _half_chain_args()
    source, (out,) = _run_cute(
        fn, args, _config([64, 64, 64], warps, 2, cute_mma_dag_derived=derived)
    )
    assert f"{RT}.mma_bf16(" in source
    if derived == "pass":
        assert f"{RT}.sts_v4_b32(md_t0 + " in source
        assert f"{RT}.pack_bf16x2(" in source
    (ref,) = _run_triton(fn, args, [64, 64, 64])
    assert torch.equal(out, ref)


def _masked_neg_inf_args() -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device=DEVICE).manual_seed(5)
    a = _randn(g, 4, 64, 128, dtype=torch.bfloat16)
    w = _randn(g, 4, 128, 64, dtype=torch.bfloat16)
    out = torch.empty((4, 64, 64), dtype=torch.float32, device=DEVICE)
    return a, w, out


@pytest.mark.parametrize("warps", (4, 8))
def test_a_non_finite_fill_beside_a_fragment_matches_the_triton_backend(
    warps: int,
) -> None:
    # ``where(mask, acc, hl.full(..., float("-inf")))``: the fill renders
    # as a ``float('-inf')`` call (a bare ``-inf`` is a NameError at the
    # DSL compile); the masked elements are ``-inf``, the rest bit-equal.
    fn = _probe_kernels()._masked_neg_inf_stored
    args = _masked_neg_inf_args()
    source, (out,) = _run_cute(fn, args, _config([64], warps, 2))
    assert "cutlass.Float32(float('-inf'))" in source
    (ref,) = _run_triton(fn, _masked_neg_inf_args(), [64])
    assert torch.isinf(ref).any() and torch.isfinite(ref).any()
    assert torch.equal(out, ref)


# m, block_sizes, warps: a 16-row tile over 8 rows; blocks of 8 computed at
# the atom (one CTA owning the rows it stores); one row (the frontend pins
# the block to 1); two CTAs with a ragged tail (24 under 16); two CTAs of 8
# rows each computed at the atom (16 under 8).
_RAGGED_CASES = (
    (8, [16, 128], 4),
    (8, [8, 128], 4),
    (1, [16, 64], 2),
    (24, [16, 128], 4),
    (16, [8, 64], 2),
)


@pytest.mark.parametrize(("m", "block_sizes", "warps"), _RAGGED_CASES)
def test_ragged_root_tiles_match_the_triton_backend(
    m: int, block_sizes: list[int], warps: int
) -> None:
    # The indexer's logits with M below (or not a multiple of) the 16-row
    # MMA atom: the rows past the extent are zero-filled operand packets,
    # zero side loads and dropped stores.  Integer inputs: bit-equal to
    # Triton's masked tile over eight launches.
    fn = _probe_kernels()._head_sum_logits
    args = _head_sum_args(m=m)
    source, out = _run_cute(fn, args, _config(block_sizes, warps, 2))
    assert f"{RT}.cp_async_16_zfill(" in source
    assert f"{RT}.ldg_s32_pred(" in source and f"{RT}.ldg_f32_pred(" in source
    (ref,) = _run_triton(fn, args, block_sizes)
    assert torch.isinf(ref).any() and torch.isfinite(ref).any()
    assert torch.equal(out[0], ref)


def _ragged_sum_args(m: int, k: int = 64, n: int = 256) -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device=DEVICE).manual_seed(m)
    a = torch.randint(-4, 5, (m, k), device=DEVICE, generator=g).to(torch.bfloat16)
    b = torch.randint(-4, 5, (k, n), device=DEVICE, generator=g).to(torch.bfloat16)
    return a, b


@pytest.mark.parametrize(
    ("m", "block_sizes", "warps"),
    ((8, [16, 128], 4), (1, [16, 64], 2), (24, [32, 128], 4)),
)
def test_a_reduction_over_ragged_rows_matches_torch(
    m: int, block_sizes: list[int], warps: int
) -> None:
    # ``(a @ b + 1).sum(0)`` with the rows in one tile: the frontend's
    # ``_mask_to`` zeroes the rows past the extent ahead of the sum (the
    # add's masked value is one), rendered as a select per register.
    # Integer inputs: exact.
    fn = _probe_kernels()._ragged_rows_summed
    a, b = _ragged_sum_args(m)
    source, (out,) = _run_cute(fn, (a, b), _config(block_sizes, warps, 2))
    assert " else cutlass.Float32(0.0)" in source
    ref = (a.float() @ b.float() + 1.0).sum(0)
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    ("m", "block_size", "warps"), ((8, 16, 4), (24, 16, 4), (24, 32, 4), (1, 16, 2))
)
def test_a_vector_along_ragged_rows_matches_torch(
    m: int, block_size: int, warps: int
) -> None:
    # ``(a @ b + 1).sum(1)`` over the whole ``b``: a vector along the ragged
    # rows stored under the row guard (two CTAs at 24 under 16, the second
    # with eight rows past the extent).
    fn = _probe_kernels()._ragged_cols_summed
    a, b = _ragged_sum_args(m, n=128)
    kernel = helion.kernel(
        fn,  # type: ignore[arg-type]
        backend="cute",
        static_shapes=True,
        cute_full_slice_matmul_tiling=False,
    )
    bound = kernel.bind((a, b))
    config = _config([block_size], warps, 2)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert f"{RT}.mma_bf16(" in source and f"{RT}.cp_async_16_zfill(" in source
    compiled = bound.compile_config(normalized)
    outputs = [cast_tensor(compiled(a, b)).clone() for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    ref = (a.float() @ b.float() + 1.0).sum(1)
    for out in outputs:
        assert torch.equal(out, ref)


def _half_addmm_db(
    grad_out: torch.Tensor, x: torch.Tensor, d: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """the pre-split squeeze_and_excitation_net ``bwd_db`` body (grad_d recomputed per K tile): ``torch.addmm(acc, c[tile_m,
    tile_k].T, grad_d)`` with an fp32 accumulator over half operands -- the
    B operand a half pass chain, the A operand a transposed half tile."""
    m, n = grad_out.size()
    k = c.size(1)
    grad_b = torch.empty([k, n], dtype=grad_out.dtype, device=grad_out.device)
    for tile_k, tile_n in hl.tile([k, n]):
        acc = hl.zeros([tile_k, tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            grad_d = (
                grad_out[tile_m, tile_n]
                * x[tile_m, tile_n]
                * d[tile_m, tile_n]
                * (1.0 - d[tile_m, tile_n])
            )
            acc = torch.addmm(acc, c[tile_m, tile_k].T, grad_d)
        grad_b[tile_k, tile_n] = acc
    return grad_b


_HALF_ADDMM_CASES = (
    ("warp", [32, 32, 64], 4, 2, "exchange"),
    ("warp", [64, 64, 64], 8, 2, "pass"),
    ("tcgen05", [64, 16, 128], 4, 2, "exchange"),
    ("tcgen05", [64, 32, 128], 4, 2, "pass"),
)


@pytest.mark.parametrize(
    ("mode", "block_sizes", "warps", "stages", "derived"),
    _HALF_ADDMM_CASES,
    ids=[
        f"{m}-{'x'.join(map(str, b))}-w{w}s{s}-{d}"
        for m, b, w, s, d in _HALF_ADDMM_CASES
    ],
)
def test_a_half_addmm_into_an_fp32_accumulator_matches_the_references(
    mode: str, block_sizes: list[int], warps: int, stages: int, derived: str
) -> None:
    # ``acc = torch.addmm(acc, a16, b16)`` is a dot of this owner (the
    # ``hl.dot(a16, b16, acc=acc)`` contract): fp32 products into the fp32
    # accumulator, no rounding, on both emitters and both derived routes.
    g = torch.Generator(device=DEVICE).manual_seed(29)
    m, k, n = 1024, 256, 512
    grad_out = _randn(g, m, n, dtype=torch.float16)
    x = _randn(g, m, n, dtype=torch.float16)
    d = torch.sigmoid(_randn(g, m, n, dtype=torch.float16))
    c = torch.relu(_randn(g, m, k, dtype=torch.float16))
    args = (grad_out, x, d, c)
    config = helion.Config(
        block_sizes=block_sizes,
        cute_mma_dag=mode,
        cute_mma_dag_warps=warps,
        cute_mma_dag_stages=stages,
        cute_mma_dag_loads="tma",
        cute_mma_dag_derived=derived,
    )
    # At 256 output rows the operand-materialization rewrite takes the
    # binding (the B recipe is derived once per 128-row tile, twice, from
    # three same-width sources), which has no fused owner: this test pins
    # the fused emitters, so the rewrite is off; the materialized route's
    # numerics are pinned below.
    kernel = helion.kernel(
        _half_addmm_db,  # type: ignore[arg-type]
        backend="cute",
        static_shapes=True,
        cute_materialize_transformed_operands=False,
    )
    bound = kernel.bind(args)
    assert bound.env.cute_fission_plan is None
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert "_cute_grouped_reduce_shared_two_stage" not in source
    if mode == "warp":
        assert f"{RT}.mma_f16(" in source
    else:
        assert "_helion_mma_dag_tc" in source
    compiled = bound.compile_config(normalized)
    outputs = [cast_tensor(compiled(*args)) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    for other in outputs[1:]:
        assert torch.equal(outputs[0], other)
    out = outputs[0]
    grad_d = grad_out.float() * x.float() * d.float() * (1.0 - d.float())
    ref = (c.float().T @ grad_d).to(torch.float16)
    rel_l2 = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel_l2 < 1e-3, rel_l2
    (triton,) = _run_triton(_half_addmm_db, args, block_sizes)
    _assert_close((out,), (triton,))


def test_a_half_addmm_whose_recipe_is_reused_materializes_and_matches() -> None:
    # The same program with the operand materialization rewrite on (the
    # autotuning efforts' structural policy switches it on; it is asked for
    # here since ``autotune_effort="none"`` leaves every structural setting
    # at its False default): the rewrite admits the transposed direct
    # operand and the three-source B recipe reused across two 128-row output
    # tiles, so the binding runs as a producer launch plus a plain-load
    # tcgen05 consumer and matches the fp32 reference as the fused emitters
    # do.
    g = torch.Generator(device=DEVICE).manual_seed(29)
    m, k, n = 1024, 256, 512
    grad_out = _randn(g, m, n, dtype=torch.float16)
    x = _randn(g, m, n, dtype=torch.float16)
    d = torch.sigmoid(_randn(g, m, n, dtype=torch.float16))
    c = torch.relu(_randn(g, m, k, dtype=torch.float16))
    args = (grad_out, x, d, c)
    kernel = helion.kernel(
        _half_addmm_db,  # type: ignore[arg-type]
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )
    bound = kernel.bind(args)
    assert bound.env.cute_fission_plan is not None
    assert bound.config_spec.cute_mma_dag is None
    out = cast_tensor(bound(*args))
    torch.cuda.synchronize()
    grad_d = grad_out.float() * x.float() * d.float() * (1.0 - d.float())
    ref = (c.float().T @ grad_d).to(torch.float16)
    rel_l2 = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel_l2 < 1e-3, rel_l2


# --------------------------------------------------------------------------
# Side loads riding the ring (``cute_mma_dag_aux='ring'``)
# --------------------------------------------------------------------------

# kernel, decay mode, (n, d, dv), block sizes, warps, stages
_SIDE_RING_CASES = (
    ("bwd", "scalar", (16, 64, 64), [32, 64], 4, 4),
    ("bwd", "scalar", (16, 64, 64), [64, 64], 8, 2),
    ("bwd", "scalar", (8, 64, 64), [64, 64], 4, 1),
    ("bwd", "scalar", (5, 64, 64), [32, 32], 2, 8),
    ("fwd", "scalar", (16, 64, 64), [32, 64], 2, 3),
    ("fwd", "scalar", (8, 128, 128), [64, 128], 8, 4),
    # The per-channel decay (``diag``): a vector along a ROOT axis.
    ("bwd", "diag", (16, 64, 64), [32, 64], 4, 4),
    ("bwd", "diag", (7, 128, 64), [64, 64], 8, 2),
    ("fwd", "diag", (8, 64, 128), [64, 64], 4, 2),
    ("fwd", "diag", (9, 64, 64), [32, 32], 4, 1),
    # full_gla's anchored mode: the chunk's last decay row is a row of the
    # staged gc tile -- the ring slot serves it under ``lane``
    # (``_ring_aliases``), the side tile under ``ring``.
    ("fwd", "anchored", (8, 128, 128), [32, 128], 8, 4),
    ("bwd", "anchored", (8, 128, 128), [32, 64], 8, 3),
    ("bwd", "anchored", (5, 64, 64), [64, 64], 4, 1),
)


@pytest.mark.parametrize(
    ("kernel", "mode", "shape", "block_sizes", "warps", "stages"),
    _SIDE_RING_CASES,
    ids=[
        f"{k}-{m}-n{s[0]}d{s[1]}dv{s[2]}-{bs[0]}x{bs[1]}w{w}s{st}"
        for k, m, s, bs, w, st in _SIDE_RING_CASES
    ],
)
def test_side_loads_riding_the_ring_match_the_lane_loads_bitwise(
    kernel: str,
    mode: str,
    shape: tuple[int, int, int],
    block_sizes: list[int],
    warps: int,
    stages: int,
) -> None:
    # The decay recurrences read a gate row (the operand pass) and a decay
    # scalar (the carry rescale) -- or a per-channel decay vector along a
    # ROOT axis (``diag``), or a row of the staged gc tile (``anchored``)
    # -- per chunk: staged through the aux ring they are the same bytes, so
    # the outputs are bit-identical to the register prefetch's at every ring
    # depth (one stage included) and trip count.
    n, d, dv = shape
    t = _inputs(bh=2, n=n, d=d, dv=dv, dtype=torch.bfloat16, seed=3)
    args = _fwd_args(t, mode, True) if kernel == "fwd" else _bwd_args(t, mode)
    fn = _kernel_fn(kernel)
    register_source, register = _run_cute(
        fn, args, _config(block_sizes, warps, stages, cute_mma_dag_aux="register")
    )
    ring_source, ring = _run_cute(
        fn, args, _config(block_sizes, warps, stages, cute_mma_dag_aux="ring")
    )
    assert f"{RT}.cp_async_4(" not in register_source
    assert f"{RT}.cp_async_4(md_aux0" in ring_source
    assert f"{RT}.lds_f32(md_aux0" in ring_source
    assert f"{RT}.ldg_f32(" not in ring_source[ring_source.index("for md_i in") :]
    for lhs, rhs in zip(register, ring, strict=True):
        assert torch.equal(lhs, rhs)
    _assert_close(ring, _run_triton(fn, args, block_sizes))


@pytest.mark.parametrize(("warps", "stages"), ((4, 2), (4, 4), (8, 1)))
def test_side_loads_riding_the_ring_with_hoisted_loads_match_the_lane_loads(
    warps: int, stages: int
) -> None:
    # The scalar-decay dh with a loop-invariant per-channel row in the carry
    # rescale: the row's loads are hoisted ahead of the loop while the body
    # renders, after which the pre-loop wait and barrier for the side tile
    # (read at the top of every iteration, ahead of the operand wait) are
    # inserted.  Misplaced inside the prologue they wait for nothing and
    # iteration 0 reads an unlanded tile: the ring form must stay
    # bit-identical to the register prefetch.
    t = _inputs(bh=2, n=16, d=64, dv=64, dtype=torch.bfloat16, seed=5)
    g = torch.Generator(device=DEVICE).manual_seed(6)
    w = _randn(g, 2, 64, scale=0.5, dtype=torch.float32)
    args = (t["k"], t["v"], t["g_last_s"], t["dh_init"], t["gc"], w, 0.125)
    fn = _probe_kernels()._dh_scalar_decay_channel_scale
    _, register = _run_cute(
        fn, args, _config([32, 64], warps, stages, cute_mma_dag_aux="register")
    )
    ring_source, ring = _run_cute(
        fn, args, _config([32, 64], warps, stages, cute_mma_dag_aux="ring")
    )
    prologue = ring_source[: ring_source.index("for md_i in cutlass.range(")]
    assert f"{RT}.ldg_f32(" in prologue
    assert prologue.rstrip().endswith(
        "cute.arch.cp_async_commit_group()\n"
        f"    cute.arch.cp_async_wait_group({max(stages - 2, 0)})\n"
        "    cute.arch.barrier()"
    )
    assert f"{RT}.ldg_f32(" not in ring_source[ring_source.index("for md_i in") :]
    for lhs, rhs in zip(register, ring, strict=True):
        assert torch.equal(lhs, rhs)
    _assert_close(ring, _run_triton(fn, args, [32, 64]))


def test_side_loads_of_a_loop_whose_ring_stages_nothing_take_the_registers() -> None:
    # k and v sliced [..., 4:68] out of 80-wide rows: 8-byte-aligned bases
    # the packets cannot stage, so both operand tiles take the synchronous
    # pass and the chunk loop has no cp.async ring for an aux ring to ride:
    # ``ring`` is pinned to the register prefetch (the same code).  With k
    # alone under-aligned, v's tile rides the ring and the aux copies with
    # it, beside k's synchronous pass -- bitwise the register form's result.
    t = _inputs(bh=2, n=8, d=64, dv=64, dtype=torch.bfloat16, seed=9)
    g = torch.Generator(device=DEVICE).manual_seed(9)
    k = _randn(g, 2, 8, 64, 80, dtype=torch.bfloat16)[..., 4:68]
    v = _randn(g, 2, 8, 64, 80, dtype=torch.bfloat16)[..., 4:68]
    assert k.data_ptr() % 16 == 8 and v.data_ptr() % 16 == 8
    fn = _kernel_fn("fwd")
    for (kk, vv), rides in (((k, v), False), ((k, t["v"]), True)):
        args = (kk, vv, t["g_last_s"], t["h0"], t["gc"], True, True, False, True)
        for block_sizes, warps, stages in (([64, 64], 4, 2), ([32, 64], 4, 1)):
            register_source, register = _run_cute(
                fn,
                args,
                _config(block_sizes, warps, stages, cute_mma_dag_aux="register"),
            )
            ring_source, ring = _run_cute(
                fn,
                args,
                _config(block_sizes, warps, stages, cute_mma_dag_aux="ring"),
            )
            assert f"{RT}.ldg_b32(" in ring_source
            assert (f"{RT}.cp_async_4(md_aux0" in ring_source) == rides
            assert (ring_source == register_source) == (not rides)
            for lhs, rhs in zip(register, ring, strict=True):
                assert torch.equal(lhs, rhs)
            _assert_close(ring, _run_triton(fn, args, block_sizes))


# kernel, (n, d, dv) or the varlen chunks, block size, warps, stages
_SIDE_RING_DELTA_CASES = (
    ("dstate", (8, 64, 64), [32], 4, 2),
    ("dstate", (8, 64, 64), [64], 8, 4),
    ("dstate", (3, 128, 64), [64], 8, 1),
    ("fwd_h_delta", (8, 64, 64), [64], 4, 2),
    ("fwd_h_delta", (8, 64, 64), [32], 8, 3),
    ("fwd_h_delta", (3, 128, 64), [64], 4, 2),
    ("fwd_h_delta_varlen", (5, 1, 3), [64], 4, 2),
    ("fwd_h_delta_varlen", (2, 7, 1, 4), [32], 8, 4),
    ("fwd_h_delta_varlen", (1,), [16], 4, 1),
)


@pytest.mark.parametrize(
    ("kernel", "shape", "block_sizes", "warps", "stages"),
    _SIDE_RING_DELTA_CASES,
    ids=[
        f"{k}-{'x'.join(map(str, s))}-{bs[0]}w{w}s{st}"
        for k, s, bs, w, st in _SIDE_RING_DELTA_CASES
    ],
)
def test_side_loads_of_the_delta_rule_recurrences_ride_the_ring_bitwise(
    kernel: str,
    shape: tuple[int, ...],
    block_sizes: list[int],
    warps: int,
    stages: int,
) -> None:
    # The gated delta-rule state passes read a decay scalar and a gate row
    # per chunk (``dstate`` backward, ``fwd_h_delta`` forward); the varlen
    # pass a per-channel decay vector under a dynamic trip count (every
    # ahead-issue of the side tiles guarded against the loop's end, a
    # one-chunk sequence included).  The ring and the register prefetch
    # agree bitwise.
    fn = _kernel_fn(kernel)
    if kernel == "fwd_h_delta_varlen":
        t = _inputs_varlen4(
            h=2, chunks=shape, d=128, dv=64, dtype=torch.bfloat16, seed=5
        )
        args = _fwd_h_delta_varlen_args(t, shape, 2)
    else:
        n, d, dv = shape
        t = _inputs4(bh=2, n=n, d=d, dv=dv, dtype=torch.bfloat16, seed=3)
        args = (_dstate_args if kernel == "dstate" else _fwd_h_delta_args)(t, True)
    register_source, register = _run_cute(fn, args, _config(block_sizes, warps, stages))
    ring_source, ring = _run_cute(
        fn, args, _config(block_sizes, warps, stages, cute_mma_dag_aux="ring")
    )
    assert f"{RT}.cp_async_4(" not in register_source
    assert f"{RT}.cp_async_4(md_aux0" in ring_source
    for lhs, rhs in zip(register, ring, strict=True):
        assert torch.equal(lhs, rhs)
    _assert_close(ring, _run_triton(fn, args, block_sizes), scaled=True)


def test_a_side_load_of_a_nested_ring_loop_rides_its_ring() -> None:
    # The DV tile of the gate vector varies with the inner loop of a nest:
    # staged in the inner loop's frame, re-primed per outer iteration.
    fn = _probe_kernels()._inner_varying_gate
    g = torch.Generator(device=DEVICE).manual_seed(11)
    do = _randn(g, 4, 64, 128, dtype=torch.bfloat16)
    h = _randn(g, 4, 128, 128, dtype=torch.bfloat16)
    gate = _randn(g, 4, 128, dtype=torch.float32)
    for warps, stages in ((4, 2), (8, 3)):
        _register_source, register = _run_cute(
            fn, (do, h, gate), _config([64, 64], warps, stages)
        )
        ring_source, ring = _run_cute(
            fn,
            (do, h, gate),
            _config([64, 64], warps, stages, cute_mma_dag_aux="ring"),
        )
        assert f"{RT}.cp_async_16(md_aux1" in ring_source
        assert torch.equal(register[0], ring[0])
        assert torch.equal(ring[0], _run_triton(fn, (do, h, gate), [64, 64])[0])
