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
    score_mod: Callable[..., torch.Tensor] = _identity_score,
    score_mod_bwd: Callable[..., torch.Tensor] = _identity_grad,
    mask_mod: Callable[..., torch.Tensor] | None = None,
    ref_score_mod: Callable[..., torch.Tensor] | None = None,
    repeats: int = 1,
    seed: int = 0,
    config_overrides: dict[str, object] | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], str]:
    """Run the example backward on the fused CuTe path with the default config
    (``config_overrides``: knob values on top of it) and return the gradients,
    the fp32 reference gradients and the generated code."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    dtype = torch.bfloat16
    q = torch.randn((b, hq, m, d), device=DEVICE, dtype=dtype, generator=gen)
    k = torch.randn((b, hkv, n, d), device=DEVICE, dtype=dtype, generator=gen)
    v = torch.randn((b, hkv, n, d), device=DEVICE, dtype=dtype, generator=gen)
    do = torch.randn((b, hq, m, d), device=DEVICE, dtype=dtype, generator=gen)
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
        BLOCK_SIZE=128,
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
    code = bound.to_code(config)
    assert "fbwd_sSteps" in code, "the fused flex backward did not fire"
    bound.set_config(config)
    # The row-sum preprocess kernel would otherwise autotune per test shape.
    pre = flex_example._flex_attention_bwd_preprocess
    pre_bound = pre.bind((out.to(dtype), do, lse))
    pre_bound.set_config(pre_bound.config_spec.default_config())
    got = None
    for _ in range(repeats):
        got = flex_example.helion_flex_attention_backward(
            q,
            k,
            v,
            out.to(dtype),
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


@_gpu_test
def test_repeated_launches_are_stable() -> None:
    got, ref, _ = _run(
        b=1, hq=2, hkv=2, m=384, n=384, d=64, mask_mod=_causal, repeats=50
    )
    _assert_close(got, ref)
