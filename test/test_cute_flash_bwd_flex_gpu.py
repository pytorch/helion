"""GPU tests of the FlexAttention BACKWARD envelope on the CuTe fused backward.

``examples/flex_attention.py::helion_flex_attention_backward_kernel`` goes
through the flex backward matcher (``flex_bwd_match``), the hook renderer in
the transposed frame and the list-walk emitter (``cute_flash_bwd_flex``).
Expectations against an fp32 autograd reference (per (batch, head) slice,
FlexAttention semantics) within bf16 tolerance:

* a dense BlockMask with the noop mask (every block in the full list);
* a causal mask_mod (full and partial lists) and a sliding-window mask whose
  KV tiles are unattended by some query blocks;
* grouped-query heads (the q-head loop inside the CTA) at head_dim 128;
* ragged lengths (``M % 128 == 64`` and ``N % 128 == 64``, ``M != N``);
* a score_mod with an additive table gather (identity ``score_mod_bwd``) and
  a softcap score_mod with a non-identity ``score_mod_bwd`` (the Jacobian
  hook);
* a mask leaving whole KV tiles unattended (empty step tables store zeros);
* the non-default knobs: the CLC persistent walk, statistics through the MMAs,
  dS^T through TMEM, the polynomial exp2 share, the per-element mask form, hook
  group widths, register splits, the KV-fastest block order, the gathered aux
  table staged in shared memory;
* repeated launches give stable gradients.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import noop_mask

from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA

if TYPE_CHECKING:
    from collections.abc import Callable

flex_example = import_path(EXAMPLES_DIR / "flex_attention.py")


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.skipif(
    not _is_sm100(), reason="fused flash backward needs SM100"
)


def _gpu_test(fn):
    return skipIfNotCUDA()(onlyBackends(["cute"])(fn))


def _identity_score(score, b, h, q_idx, kv_idx):
    return score


def _identity_grad(grad, score, b, h, q_idx, kv_idx):
    return grad


def _reference_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    score_mod: Callable[..., torch.Tensor],
    mask_mod: Callable[..., torch.Tensor] | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 autograd per (b, h): (dq, dk, dv, out, lse) with FlexAttention
    semantics (score_mod, then -inf where mask_mod is False, softmax)."""
    b_sz, hq, m, d = q.shape
    hk, n = k.shape[1], k.shape[2]
    groups = hq // hk
    dq = torch.zeros((b_sz, hq, m, d), dtype=torch.float32, device=q.device)
    dk = torch.zeros((b_sz, hk, n, d), dtype=torch.float32, device=q.device)
    dv = torch.zeros((b_sz, hk, n, v.shape[-1]), dtype=torch.float32, device=q.device)
    out = torch.zeros((b_sz, hq, m, v.shape[-1]), dtype=torch.float32, device=q.device)
    lse = torch.zeros((b_sz, hq, m), dtype=torch.float32, device=q.device)
    q_idx = torch.arange(m, device=q.device)[:, None]
    kv_idx = torch.arange(n, device=q.device)[None, :]
    for b in range(b_sz):
        b_t = torch.tensor(b, device=q.device)
        for h in range(hq):
            h_t = torch.tensor(h, device=q.device)
            with torch.enable_grad():
                qf = q[b, h].detach().float().requires_grad_(True)
                kf = k[b, h // groups].detach().float().requires_grad_(True)
                vf = v[b, h // groups].detach().float().requires_grad_(True)
                s = (qf @ kf.T) * scale
                s = score_mod(s, b_t, h_t, q_idx, kv_idx)
                if mask_mod is not None:
                    s = torch.where(mask_mod(b_t, h_t, q_idx, kv_idx), s, float("-inf"))
                p = torch.softmax(s, dim=-1)
                p = torch.nan_to_num(p, nan=0.0)
                o = p @ vf
                gq, gk, gv = torch.autograd.grad(o, (qf, kf, vf), do[b, h].float())
            dq[b, h] = gq
            dk[b, h // groups] += gk
            dv[b, h // groups] += gv
            out[b, h] = o.detach()
            lse[b, h] = torch.logsumexp(s.detach(), dim=-1)
    return dq, dk, dv, out, lse


def _run(
    *,
    b: int,
    hq: int,
    hkv: int,
    m: int,
    n: int,
    d: int,
    dv: int | None = None,
    score_mod: Callable[..., torch.Tensor] = _identity_score,
    score_mod_bwd: Callable[..., torch.Tensor] = _identity_grad,
    mask_mod: Callable[..., torch.Tensor] | None = None,
    ref_score_mod: Callable[..., torch.Tensor] | None = None,
    repeats: int = 1,
    seed: int = 0,
    config_overrides: dict[str, object] | None = None,
    layout: str = "bhsd",
    dtype: torch.dtype = torch.bfloat16,
    table_steps: int = 0,
    block_size: int | tuple[int, int] = 128,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], str]:
    """Run the example backward on the fused CuTe path with the default config
    (``config_overrides``: knob values on top of it; ``table_steps``: a
    step-table cap, forced even when the shape is too small to offer it) and
    return the gradients, the fp32 reference gradients and the generated code.

    ``layout``: ``bhsd`` (contiguous), ``bshd`` (``[B, S, H, D]`` allocations
    viewed ``[B, H, S, D]``, the forward output O included) or ``packed``
    (Q/K/V head-dim slices of one ``[B, S, H, 3 * D]`` buffer, O and dO
    transposed); the values are the same for every layout. ``dv`` is the
    value head dim (V / dO / dV columns; default ``d``)."""
    dv = d if dv is None else dv
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn((b, hq, m, d), device=DEVICE, dtype=dtype, generator=gen)
    k = torch.randn((b, hkv, n, d), device=DEVICE, dtype=dtype, generator=gen)
    v = torch.randn((b, hkv, n, dv), device=DEVICE, dtype=dtype, generator=gen)
    do = torch.randn((b, hq, m, dv), device=DEVICE, dtype=dtype, generator=gen)
    if layout == "bshd":
        q, k, v, do = (_as_bshd(t) for t in (q, k, v, do))
    elif layout == "packed":
        assert hq == hkv and m == n
        qkv = torch.empty((b, m, hq, 2 * d + dv), device=DEVICE, dtype=dtype)
        q = qkv[..., :d].transpose(1, 2).copy_(q)
        k = qkv[..., d : 2 * d].transpose(1, 2).copy_(k)
        v = qkv[..., 2 * d :].transpose(1, 2).copy_(v)
        do = _as_bshd(do)
    else:
        assert layout == "bhsd"
    scale = 1.0 / math.sqrt(d)
    ref_dq, ref_dk, ref_dv, out, lse = _reference_grads(
        q, k, v, do, ref_score_mod or score_mod, mask_mod, scale
    )
    block_mask = create_block_mask(
        mask_mod if mask_mod is not None else noop_mask,
        None,
        None,
        m,
        n,
        device=str(DEVICE),
        BLOCK_SIZE=block_size,
    )
    kernel = flex_example.helion_flex_attention_backward_kernel
    args = flex_example.flex_attention_backward_args(
        q,
        k,
        v,
        do,
        lse,
        out,
        score_mod,
        score_mod_bwd,
        block_mask,
        scale,
        hq != hkv,
    )
    bound = kernel.bind(args)
    config = bound.config_spec.default_config()
    if config_overrides:
        import helion

        fields = bound.config_spec._flat_fields()
        for key in config_overrides:
            assert key in fields, f"{key} is not a knob of this kernel"
        config = helion.Config(**{**config.config, **config_overrides})
    if table_steps:
        import helion

        config = helion.Config(
            **{**config.config, "cute_flash_bwd_table_steps": table_steps}
        )
    code = bound.to_code(config)
    assert "fbwd_sSteps" in code, "the fused flex backward did not fire"
    bound.set_config(config)
    out = out.to(dtype) if layout == "bhsd" else _as_bshd(out.to(dtype))
    # The row-sum preprocess kernel would otherwise autotune per test shape.
    pre = flex_example._flex_attention_bwd_preprocess
    pre_bound = pre.bind((out, do, lse))
    pre_bound.set_config(pre_bound.config_spec.default_config())
    got = None
    for _ in range(repeats):
        got = flex_example.helion_flex_attention_backward(
            q,
            k,
            v,
            out,
            lse,
            do,
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=hq != hkv,
        )
    assert got is not None
    torch.cuda.synchronize()
    return got, (ref_dq, ref_dk, ref_dv), code


def _as_bshd(t: torch.Tensor) -> torch.Tensor:
    """``t``'s values in a ``[B, S, H, D]`` allocation viewed ``[B, H, S, D]``."""
    batch, heads, seq, head_dim = t.shape
    return (
        torch.empty((batch, seq, heads, head_dim), device=t.device, dtype=t.dtype)
        .transpose(1, 2)
        .copy_(t)
    )


def _assert_close(
    got: tuple[torch.Tensor, ...], ref: tuple[torch.Tensor, ...], tol: float = 2e-2
) -> None:
    for name, g, r in zip(("dq", "dk", "dv"), got, ref, strict=True):
        assert not torch.isnan(g).any(), f"{name} has NaN"
        peak = max(r.abs().max().item(), 1e-6)
        rel = (g.float() - r).abs().max().item() / peak
        assert rel < tol, f"{name}: relative-to-peak error {rel:.4f}"


def _causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _window64(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= 64)


def _prefix_only(b, h, q_idx, kv_idx):
    # Only the first KV block is ever attended: every other KV tile is empty.
    return kv_idx < 128


@_gpu_test
def test_dense_noop_mask() -> None:
    got, ref, _ = _run(b=1, hq=2, hkv=2, m=256, n=256, d=64)
    _assert_close(got, ref)


@_gpu_test
def test_causal_mask_mod() -> None:
    got, ref, code = _run(b=2, hq=2, hkv=2, m=512, n=512, d=64, mask_mod=_causal)
    assert "fbwd_masked != 0" in code
    _assert_close(got, ref)


@_gpu_test
def test_sliding_window_mask() -> None:
    got, ref, _ = _run(b=1, hq=2, hkv=2, m=512, n=512, d=64, mask_mod=_window64)
    _assert_close(got, ref)


@_gpu_test
def test_grouped_query_heads_head_dim_128() -> None:
    got, ref, _ = _run(b=1, hq=4, hkv=2, m=256, n=256, d=128, mask_mod=_causal)
    _assert_close(got, ref)


@_gpu_test
@pytest.mark.parametrize(("d", "dv"), [(64, 128), (128, 64)])
def test_mixed_head_dims(d: int, dv: int) -> None:
    """D != Dv: the dP MMA's K extent and the dV width follow Dv, the dK / dQ
    widths D (grouped heads, causal, ragged N)."""
    got, ref, code = _run(b=1, hq=4, hkv=2, m=256, n=320, d=d, dv=dv, mask_mod=_causal)
    assert f"'head_dim_v': {dv}" in code
    assert f"partition_shape_C((128, {dv}))" in code
    _assert_close(got, ref)


@_gpu_test
@pytest.mark.parametrize(("d", "dv"), [(96, 96), (80, 80), (32, 32), (48, 112)])
def test_padded_head_dims(d: int, dv: int) -> None:
    """Head dims that are not multiples of 64: the smem tiles pad to whole
    swizzle atoms while the MMAs, TMEM accumulators and stores run the exact
    extents (grouped heads, causal, ragged M and N)."""
    got, ref, code = _run(b=1, hq=4, hkv=2, m=320, n=192, d=d, dv=dv, mask_mod=_causal)
    assert f"flash_bwd_shared_storage({-(-d // 64) * 64}, " in code
    _assert_close(got, ref)


@_gpu_test
def test_mla_head_dims_192_128() -> None:
    """The MLA pair (D=192, Dv=128): dQ aliases S / dP in TMEM (``alias_s``),
    dP's MMA waits on the s_read barrier, one Q / dO stage, the dK epilogue
    reuses its two staging slots for three boxes."""
    got, ref, code = _run(
        b=1, hq=4, hkv=2, m=320, n=192, d=192, dv=128, mask_mod=_causal
    )
    assert "fbwd_sread_ptr" in code
    # Six steps per tile: the eight-box staging still fits beside the table.
    assert "flash_bwd_shared_storage(192, 1, 1, 1, cutlass.BFloat16, 6, 8, " in code
    _assert_close(got, ref)


@_gpu_test
@pytest.mark.parametrize(
    ("d", "dv", "overrides"),
    [
        # The MLA pair on the persistent walk with the statistics MMAs (one Q
        # stage, the alias_s drain / s_read waits across tiles).
        (192, 128, {"cute_flash_bwd_persistent": 1, "cute_flash_bwd_stats_mma": 1}),
        # A padded width (dQ aliases dP: the dS^T-through-TMEM knob is not
        # offered) on the persistent walk with the statistics MMAs.
        (96, 96, {"cute_flash_bwd_persistent": 1, "cute_flash_bwd_stats_mma": 1}),
        # A small width keeps dQ's own TMEM columns: dS^T through TMEM applies;
        # three dQ boxes per step alternate the staging slots across steps.
        (96, 96, {"cute_flash_bwd_tile_group": 0}),
        (32, 32, {"cute_flash_bwd_ds_tmem": 1, "cute_flash_bwd_persistent": 1}),
        # Mixed widths with the dO ring at one stage and the KV-fastest order.
        (64, 128, {"cute_flash_bwd_persistent": 1, "cute_flash_bwd_tile_group": 0}),
    ],
)
def test_head_dims_non_default_knobs(
    d: int, dv: int, overrides: dict[str, object]
) -> None:
    got, ref, code = _run(
        b=1,
        hq=4,
        hkv=2,
        m=640,
        n=512,
        d=d,
        dv=dv,
        mask_mod=_causal,
        config_overrides=overrides,
    )
    if overrides.get("cute_flash_bwd_persistent"):
        assert "fbwd_tfull_ptr" in code
    if (d, dv) == (192, 128) and overrides.get("cute_flash_bwd_stats_mma"):
        # The statistics slices leave room for four staging boxes: the chunked
        # dQ drain with one box per warp (each store waits for its slot).
        assert ", 4, 1, False, False, 0, head_dim_v=128, flex_mbars=9)" in code
        assert "fbwd_sdq_box = fbwd_sdq[None, None, warp_idx]" in code
    _assert_close(got, ref)


@_gpu_test
def test_ragged_lengths() -> None:
    got, ref, _ = _run(b=1, hq=2, hkv=2, m=192, n=320, d=64, mask_mod=_window64)
    _assert_close(got, ref)


@_gpu_test
def test_empty_kv_tiles_store_zero() -> None:
    got, ref, _ = _run(b=1, hq=2, hkv=2, m=256, n=384, d=64, mask_mod=_prefix_only)
    _assert_close(got, ref)
    assert torch.all(got[1][:, :, 128:] == 0) and torch.all(got[2][:, :, 128:] == 0)


@_gpu_test
def test_score_mod_table_gather() -> None:
    bias = torch.randn((2, 1023), device=DEVICE, dtype=torch.float32)

    def score_mod(score, b, h, q_idx, kv_idx):
        import helion.language as hl

        return score + hl.load(bias, [h, q_idx - kv_idx + 511])

    def ref_score_mod(score, b, h, q_idx, kv_idx):
        return score + bias[h, q_idx - kv_idx + 511]

    got, ref, code = _run(
        b=1,
        hq=2,
        hkv=2,
        m=256,
        n=256,
        d=64,
        score_mod=score_mod,
        mask_mod=_causal,
        ref_score_mod=ref_score_mod,
    )
    assert "_flash_aux0" in code
    _assert_close(got, ref)


@_gpu_test
@pytest.mark.parametrize(
    "overrides",
    [
        # The CLC persistent walk with the statistics MMAs, dS^T through TMEM,
        # the polynomial exp2 share, the per-element mask form, 8-lane hooks
        # and the KV-fastest block order.
        {
            "cute_flash_bwd_persistent": 1,
            "cute_flash_bwd_stats_mma": 1,
            "cute_flash_bwd_ds_tmem": 1,
            "cute_flash_bwd_exp2_emu": 8,
            "cute_flash_bwd_mask_form": "element",
            "cute_flash_bwd_hook_cols": 8,
            "cute_flash_bwd_tile_group": 0,
        },
        # Non-persistent with the statistics MMAs, the packed f16x2 exp2 pair
        # and the widest register split for the compute warps.
        {
            "cute_flash_bwd_stats_mma": 1,
            "cute_flash_bwd_exp2_f32": 0,
            "cute_flash_bwd_regs": "88/168/88",
            "cute_flash_bwd_hook_pin": 0,
        },
    ],
)
def test_causal_non_default_knobs(overrides: dict[str, object]) -> None:
    got, ref, code = _run(
        b=2,
        hq=2,
        hkv=2,
        m=512,
        n=512,
        d=64,
        mask_mod=_causal,
        config_overrides=overrides,
    )
    if overrides.get("cute_flash_bwd_persistent"):
        assert "fbwd_tissued_ptr" in code
    if overrides.get("cute_flash_bwd_ds_tmem"):
        assert "gemm_ptx_precomputed_pv_ts(fbwd_dk_addr" in code
    _assert_close(got, ref)


@_gpu_test
@pytest.mark.parametrize("aux_smem", [0, 4096])
def test_gathered_table_score_mod(aux_smem: int) -> None:
    # graphormer-style: a per-(b, q, kv) index table selects a row of a small
    # per-head bias table; with the budget the bias table is staged in smem.
    index = torch.randint(0, 7, (1, 256, 256), device=DEVICE, dtype=torch.int32)
    bias = torch.randn((2, 7), device=DEVICE, dtype=torch.float32)

    def score_mod(score, b, h, q_idx, kv_idx):
        import helion.language as hl

        dist = hl.load(index, [b, q_idx, kv_idx])
        return score + hl.load(bias, [h, dist])

    def ref_score_mod(score, b, h, q_idx, kv_idx):
        return score + bias[h, index[b, q_idx, kv_idx]]

    got, ref, code = _run(
        b=1,
        hq=2,
        hkv=2,
        m=256,
        n=256,
        d=64,
        score_mod=score_mod,
        mask_mod=_causal,
        ref_score_mod=ref_score_mod,
        config_overrides={
            "cute_flash_bwd_aux_smem": aux_smem,
            "cute_flash_bwd_bias_fold": 0,
            "cute_flash_bwd_hook_cols": 16,
        },
    )
    assert ("flash_flex_aux_smem" in code) == (aux_smem > 0)
    _assert_close(got, ref)


@_gpu_test
def test_softcap_score_mod_with_jacobian() -> None:
    cap = 20.0

    def score_mod(score, b, h, q_idx, kv_idx):
        return cap * torch.tanh(score / cap)

    def score_mod_bwd(grad, score, b, h, q_idx, kv_idx):
        t = torch.tanh(score / cap)
        return grad * (1.0 - t * t)

    got, ref, code = _run(
        b=1,
        hq=2,
        hkv=2,
        m=256,
        n=256,
        d=64,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
    )
    assert "fbwd_spre" in code and "fbwd_mul_pairs_packed" in code
    _assert_close(got, ref)


def _assert_same_walk(
    whole: tuple[torch.Tensor, ...], chunked: tuple[torch.Tensor, ...]
) -> None:
    """A chunked walk runs the same step sequence: dK / dV accumulate in the
    same order (bitwise equal); dQ's fp32 reduce-adds from different CTAs
    land in any order (run to run), so its bf16 result may round one ulp
    either way."""
    assert torch.equal(whole[1], chunked[1]), "dK differs"
    assert torch.equal(whole[2], chunked[2]), "dV differs"
    dq_a, dq_b = whole[0].float(), chunked[0].float()
    ulp = torch.maximum(dq_a.abs(), dq_b.abs()) * 2.0**-7
    slack = 1e-6 * dq_a.abs().max()
    assert bool(((dq_a - dq_b).abs() <= ulp + slack).all()), "dQ differs"


@_gpu_test
def test_chunked_table_matches_the_whole_walk() -> None:
    # 17 Q tiles x 2 q heads: 34 steps on KV tile 0 (28 on tile 3) in chunks
    # of 16 (16 + 16 + 2) against the whole walk in one table; the causal
    # diagonal puts masked steps in every chunk sequence.
    kw = {"b": 1, "hq": 2, "hkv": 1, "m": 2176, "n": 512, "d": 64, "mask_mod": _causal}
    whole, ref, _ = _run(**kw)
    chunked, _, code = _run(**kw, table_steps=16)
    assert "fbwd_nchunks = (fbwd_steps + 15) // 16" in code
    assert "named_barrier_wait_unaligned(7, 448)" in code
    _assert_close(chunked, ref)
    _assert_same_walk(whole, chunked)


@_gpu_test
@pytest.mark.parametrize(
    ("d", "dv", "overrides"),
    [
        (64, 64, {}),
        (128, 128, {}),
        (64, 64, {"cute_flash_bwd_persistent": 1}),
        (128, 128, {"cute_flash_bwd_persistent": 1}),
        # The statistics MMAs (their per-stage aug barriers) across chunks.
        (64, 64, {"cute_flash_bwd_stats_mma": 1}),
        (128, 128, {"cute_flash_bwd_stats_mma": 1}),
        (128, 128, {"cute_flash_bwd_persistent": 1, "cute_flash_bwd_stats_mma": 1}),
        # Every knob that touches the per-step pipelines, chunked.
        (
            64,
            64,
            {
                "cute_flash_bwd_persistent": 1,
                "cute_flash_bwd_stats_mma": 1,
                "cute_flash_bwd_ds_tmem": 1,
                "cute_flash_bwd_tile_group": 0,
            },
        ),
        # A padded width (dQ aliases dP: the chunk's first dP waits for the
        # drain) and the MLA pair (dQ aliases S: the chunk's first S waits).
        (96, 96, {}),
        (192, 128, {}),
        (192, 128, {"cute_flash_bwd_persistent": 1}),
    ],
)
def test_chunked_grouped_heads_ragged(
    d: int, dv: int, overrides: dict[str, object]
) -> None:
    # Group 4, ragged M (1000 rows: 8 Q tiles) and N (384 rows: 3 KV tiles):
    # 32 steps on KV tile 0 in chunks of 12 (12 + 12 + 8).
    kw = {
        "b": 1,
        "hq": 4,
        "hkv": 1,
        "m": 1000,
        "n": 384,
        "d": d,
        "dv": dv,
        "mask_mod": _causal,
    }
    whole, ref, _ = _run(**kw, config_overrides=overrides or None)
    chunked, _, code = _run(**kw, config_overrides=overrides or None, table_steps=12)
    assert "fbwd_nchunks = (fbwd_steps + 11) // 12" in code
    if d == 192:
        # The chunked dQ drain with the eight-box staging across chunked tables.
        assert "cute.copy(fbwd_tiled_dq_ld, tRDtDQ_B, tRDrDQ)" in code
        assert ", 8, 0, True, True, 0, head_dim_v=128, flex_mbars=9)" in code
    if overrides.get("cute_flash_bwd_persistent"):
        assert "fbwd_sSteps[24 + fbwd_tbuf]" in code
        assert "if (fbwd_i == 0) & (fbwd_chunk == 0) & (fbwd_nz_it > 0):" in code
    if overrides.get("cute_flash_bwd_stats_mma"):
        assert "fbwd_aug_full_ptr" in code
    _assert_close(chunked, ref)
    _assert_same_walk(whole, chunked)


@_gpu_test
@pytest.mark.parametrize("table_steps", [5, 7])
def test_chunked_q_block_256(table_steps: int) -> None:
    # 256-row Q blocks over 1000 rows: 4 blocks = 8 tiles (the last past M),
    # 2 q heads: 16 steps in chunks of 5 (5 + 5 + 5 + 1) or 7 (7 + 7 + 2), so
    # chunks start on the second tile of a block (odd step in the pair) too.
    kw = {"b": 1, "hq": 2, "hkv": 1, "m": 1000, "n": 256, "d": 64, "mask_mod": _causal}
    whole, ref, _ = _run(**kw, block_size=(256, 128))
    chunked, _, code = _run(**kw, block_size=(256, 128), table_steps=table_steps)
    assert f"fbwd_nchunks = (fbwd_steps + {table_steps - 1}) // {table_steps}" in code
    _assert_close(chunked, ref)
    _assert_same_walk(whole, chunked)


@_gpu_test
@pytest.mark.parametrize("d", [64, 128])
def test_chunked_persistent_two_tiles_per_cta(d: int) -> None:
    # 2 batches x 8 kv heads x 12 KV tiles = 192 tiles over 148 SMs: the
    # persistent CTAs walk a second tile (table buffer 1, the nz_it > 0 and
    # chunk terms together, the TMA warp building tile t+1's table while tile
    # t refills its chunks); group 2, causal (masked diagonal), 3-step chunks
    # of up to 24 steps per tile.
    overrides = {"cute_flash_bwd_persistent": 1}
    kw = {"b": 2, "hq": 16, "hkv": 8, "m": 1536, "n": 1536, "d": d, "mask_mod": _causal}
    whole, ref, code = _run(**kw, config_overrides=overrides)
    assert "fbwd_tissued_ptr" in code and "fbwd_nchunks" not in code
    chunked, _, code = _run(**kw, config_overrides=overrides, table_steps=3)
    assert "fbwd_nchunks = (fbwd_steps + 2) // 3" in code
    # The length words follow the two 3-step buffers.
    assert "fbwd_sSteps[6 + fbwd_tbuf]" in code
    assert torch.cuda.get_device_properties(0).multi_processor_count < 192
    _assert_close(chunked, ref)
    _assert_same_walk(whole, chunked)


@_gpu_test
@pytest.mark.parametrize("persistent", [0, 1])
def test_single_step_chunks(persistent: int) -> None:
    # A sliding window: 1-2 (partial) steps per KV tile, one step per chunk:
    # every chunk is a peeled first step with an empty pipelined loop.
    overrides = {"cute_flash_bwd_persistent": 1} if persistent else None
    kw = {
        "b": 1,
        "hq": 2,
        "hkv": 2,
        "m": 1024,
        "n": 1024,
        "d": 64,
        "mask_mod": _window64,
    }
    whole, ref, _ = _run(**kw, config_overrides=overrides)
    chunked, _, code = _run(**kw, config_overrides=overrides, table_steps=1)
    assert "fbwd_nchunks = (fbwd_steps + 0) // 1" in code
    _assert_close(chunked, ref)
    _assert_same_walk(whole, chunked)


@_gpu_test
@pytest.mark.parametrize(
    ("d", "dv"), [(64, 32), (32, 48), (64, 128), (96, 64), (48, 32)]
)
def test_ds_tmem_mixed_head_dims(d: int, dv: int) -> None:
    """dS^T through TMEM with D != Dv (both orders, padded D): the dK MMA
    runs with N = D, not the dV MMA's N = Dv."""
    got, ref, code = _run(
        b=1,
        hq=4,
        hkv=2,
        m=320,
        n=256,
        d=d,
        dv=dv,
        mask_mod=_causal,
        config_overrides={"cute_flash_bwd_ds_tmem": 1},
    )
    assert "helion_fbwd_dskts_idesc" in code
    _assert_close(got, ref)


@_gpu_test
@pytest.mark.parametrize(("d", "dv"), [(96, 64), (80, 64), (96, 48)])
def test_dq_partition_past_tmem_end(d: int, dv: int) -> None:
    """Plain plans whose padded dQ partition would reach past TMEM column
    512: only the stored chunks are loaded (dq checked)."""
    got, ref, code = _run(b=1, hq=2, hkv=2, m=384, n=256, d=d, dv=dv, mask_mod=_causal)
    assert "tRDtDQ[None, fbwd_c, None, None]" in code
    _assert_close(got, ref)


@_gpu_test
@pytest.mark.parametrize(
    ("d", "dv", "dtype"),
    [
        (80, 80, torch.float16),
        (64, 128, torch.float16),
        (96, 64, torch.bfloat16),
        (192, 128, torch.bfloat16),
    ],
)
def test_softcap_jacobian_head_dims(d: int, dv: int, dtype: torch.dtype) -> None:
    """The Jacobian hook (a non-identity ``score_mod_bwd``) with the causal
    mask on padded / mixed / MLA head dims, fp16 included."""
    cap = 20.0

    def score_mod(score, b, h, q_idx, kv_idx):
        return cap * torch.tanh(score / cap)

    def score_mod_bwd(grad, score, b, h, q_idx, kv_idx):
        t = torch.tanh(score / cap)
        return grad * (1.0 - t * t)

    got, ref, code = _run(
        b=1,
        hq=2,
        hkv=2,
        m=384,
        n=256,
        d=d,
        dv=dv,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        mask_mod=_causal,
        dtype=dtype,
    )
    assert "fbwd_spre" in code
    _assert_close(got, ref)


@_gpu_test
def test_repeated_launches_are_stable() -> None:
    got, ref, _ = _run(
        b=1, hq=2, hkv=2, m=384, n=384, d=64, mask_mod=_causal, repeats=50
    )
    _assert_close(got, ref)


# -- strided operands ---------------------------------------------------------


def _assert_rank4_maps(code: str) -> None:
    assert "'flex_strides': (" in code
    assert "tQgQ = tQgQ_qdl[None, None, 0, fbwd_hq, fbwd_b]" in code
    assert "tKgK_kdl[None, None, 0, fbwd_hk, fbwd_b]" in code
    assert "_fbwd_mdKt[None, None, fbwd_hk, fbwd_b]" in code
    assert "tQgQ_qdl[None, None, 0, fbwd_bhq]" not in code


def _assert_same_gradients(
    got: tuple[torch.Tensor, ...], got_c: tuple[torch.Tensor, ...]
) -> None:
    """dK / dV are accumulated in TMEM in the step-table order (bitwise across
    layouts); dQ goes through fp32 reduce-adds whose arrival order varies."""
    dq, dk, dv = got
    dq_c, dk_c, dv_c = got_c
    assert torch.equal(dk, dk_c)
    assert torch.equal(dv, dv_c)
    peak = max(dq_c.float().abs().max().item(), 1e-6)
    assert (dq.float() - dq_c.float()).abs().max().item() / peak < 1e-3


@_gpu_test
@pytest.mark.parametrize(
    ("hq", "hkv", "m", "n", "d", "dtype", "mask_mod"),
    [
        (2, 2, 512, 512, 64, torch.bfloat16, _causal),
        (4, 2, 256, 256, 128, torch.float16, _causal),
        (2, 2, 192, 320, 64, torch.bfloat16, _window64),
        (2, 2, 256, 384, 64, torch.bfloat16, _prefix_only),
    ],
    ids=["causal_d64", "gqa_d128_fp16", "ragged_unequal", "empty_kv_tiles"],
)
def test_transposed_views_match_the_contiguous_path(
    hq: int, hkv: int, m: int, n: int, d: int, dtype: torch.dtype, mask_mod
) -> None:
    """``[B, S, H, D].transpose(1, 2)`` Q/K/V/dO (dK/dV ``empty_like`` keep the
    layout) run the fused backward through rank-4 TMA maps: dK/dV bitwise the
    contiguous run, dQ within reduce-add reordering, all within tolerance of
    the fp32 reference; the empty KV tiles' zero rows land in the strided
    dK/dV too."""
    got_c, _, code_c = _run(
        b=2, hq=hq, hkv=hkv, m=m, n=n, d=d, mask_mod=mask_mod, dtype=dtype
    )
    assert "'flex_strides'" not in code_c
    got, ref, code = _run(
        b=2,
        hq=hq,
        hkv=hkv,
        m=m,
        n=n,
        d=d,
        mask_mod=mask_mod,
        dtype=dtype,
        layout="bshd",
    )
    _assert_rank4_maps(code)
    assert got[1].stride() == (hkv * n * d, d, hkv * d, 1)
    assert got[2].stride() == (hkv * n * d, d, hkv * d, 1)
    _assert_same_gradients(got, got_c)
    _assert_close(got, ref)
    if mask_mod is _prefix_only:
        assert torch.all(got[1][:, :, 128:] == 0) and torch.all(got[2][:, :, 128:] == 0)


@_gpu_test
def test_packed_qkv_slices_backward() -> None:
    got_c, _, _ = _run(b=2, hq=2, hkv=2, m=512, n=512, d=64, mask_mod=_causal)
    got, ref, code = _run(
        b=2, hq=2, hkv=2, m=512, n=512, d=64, mask_mod=_causal, layout="packed"
    )
    _assert_rank4_maps(code)
    _assert_same_gradients(got, got_c)
    _assert_close(got, ref)


@_gpu_test
def test_transposed_views_persistent_walk() -> None:
    """The CLC persistent walk decodes the head / batch per tile too."""
    overrides = {"cute_flash_bwd_persistent": 1}
    got_c, _, _ = _run(
        b=2,
        hq=2,
        hkv=2,
        m=512,
        n=512,
        d=64,
        mask_mod=_causal,
        config_overrides=overrides,
    )
    got, ref, code = _run(
        b=2,
        hq=2,
        hkv=2,
        m=512,
        n=512,
        d=64,
        mask_mod=_causal,
        config_overrides=overrides,
        layout="bshd",
    )
    assert "fbwd_tissued_ptr" in code
    _assert_rank4_maps(code)
    _assert_same_gradients(got, got_c)
    _assert_close(got, ref)


@_gpu_test
def test_transposed_o_and_do_preprocess_is_bitwise() -> None:
    """The preprocess reads O / dO through their 4-D views (no host copy):
    transposed inputs give bitwise the contiguous ``delta`` / ``lse2``."""
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    o = torch.randn((2, 4, 512, 64), device=DEVICE, dtype=torch.bfloat16, generator=gen)
    do = torch.randn(
        (2, 4, 512, 64), device=DEVICE, dtype=torch.bfloat16, generator=gen
    )
    lse = torch.randn((2, 4, 512), device=DEVICE, dtype=torch.float32, generator=gen)
    lse[0, 0, :7] = float("-inf")
    pre = flex_example._flex_attention_bwd_preprocess
    for args in ((o, do, lse), (_as_bshd(o), _as_bshd(do), lse)):
        pre_bound = pre.bind(args)
        pre_bound.set_config(pre_bound.config_spec.default_config())
    delta_c, lse2_c = pre(o, do, lse)
    delta, lse2 = pre(_as_bshd(o), _as_bshd(do), lse)
    torch.cuda.synchronize()
    assert torch.equal(delta, delta_c)
    assert torch.equal(lse2, lse2_c)
    expected = (o.float() * do.float()).sum(-1)
    torch.testing.assert_close(delta_c, expected, atol=1e-2, rtol=1e-3)
    assert torch.all(lse2_c[0, 0, :7] == 0)
