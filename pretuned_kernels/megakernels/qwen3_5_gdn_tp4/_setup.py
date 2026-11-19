# ruff: noqa: ANN001, ANN002, ANN202, C408
"""Synthetic per-rank Qwen3.5 TP4 GDN mixer weights / state / inputs in SGLang's
served layouts, and the GDN mixer megakernel's argument list."""

from __future__ import annotations

import torch

T = 7  # verify window (speculative-num-draft-tokens)
H, HV, K, V = 4, 16, 128, 128  # per rank (16 / 64 heads over TP4)
L = 16  # ReplaySSM ring length
HIDDEN = 4096
KEY_DIM, VALUE_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 3072
QKVZ = CONV_DIM + VALUE_DIM  # 5120
WIDTH = 4
SLOTS, RSLOTS, WSLOTS = 3, 3, 4  # mamba slots, ring slots, window rows (slots + 1)


def _rn(g, *s, scale=1.0, dtype=torch.bfloat16, device="cuda"):
    return (torch.randn(*s, generator=g, device=device) * scale).to(dtype)


def make_layer(seed: int, device="cuda") -> dict:
    """One layer's weights. ``out_proj_stored`` is SGLang's ``layer.weight`` after
    ``process_weights_after_loading`` (``quantized_weight.t()``: [2048, 4096] with
    strides (1, 2048)); the kernel reads its physical [4096, 2048] layout."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    w_real = torch.randn(HIDDEN, VALUE_DIM, generator=g, device=device) * 0.02
    weight_scale = (w_real.abs().amax() / 448.0).reshape(())
    w_fp8 = (w_real / weight_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    input_scale = torch.tensor(0.02, device=device)
    # in_proj_qkvz (ModelOpt per-tensor FP8, stored transposed like out_proj) and the
    # BF16 in_proj_ba.
    wi_real = torch.randn(QKVZ, HIDDEN, generator=g, device=device) * 0.02
    in_weight_scale = (wi_real.abs().amax() / 448.0).reshape(())
    wi_fp8 = (wi_real / in_weight_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    in_input_scale = torch.tensor(0.01, device=device)
    return dict(
        inproj_stored=wi_fp8.t(),
        inproj_weight_scale=in_weight_scale,
        inproj_input_scale=in_input_scale,
        inproj_scales=torch.stack(
            (in_input_scale, in_input_scale * in_weight_scale)
        ).float(),
        ba_weight=_rn(g, 2 * HV, HIDDEN, scale=0.02, device=device),
        conv_weight=_rn(g, CONV_DIM, WIDTH, scale=0.3, device=device),
        A_log=(torch.rand(HV, generator=g, device=device) * 2.0).float(),
        # nn.Parameter(torch.ones(n)) under the BF16 default dtype.
        dt_bias=_rn(g, HV, scale=0.5, device=device),
        gate_norm_weight=(1 + 0.1 * torch.randn(V, generator=g, device=device)).to(
            torch.bfloat16
        ),
        out_proj_stored=w_fp8.t(),
        weight_scale=weight_scale,
        input_scale=input_scale,
        out_proj_scales=torch.stack((input_scale, input_scale * weight_scale)).float(),
        # GemmaRMSNorm.gemma_weight: BF16 (w + 1).
        norm_weight=(1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).to(
            torch.bfloat16
        ),
    )


def make_state(seed: int, wp: int = 2, base: int = 5, device="cuda") -> dict:
    """The request's conv state, conv window scratch (garbage before the run),
    ReplaySSM checkpoint, rings and cursors."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return dict(
        conv_state=_rn(g, SLOTS, CONV_DIM, WIDTH - 1, device=device),
        conv_window=_rn(g, WSLOTS, CONV_DIM, T + WIDTH - 2, device=device),
        h0=_rn(g, SLOTS, HV, V, K, scale=0.05, device=device),
        d_cache=_rn(g, RSLOTS, HV, L, V, scale=0.1, device=device),
        k_cache=_rn(g, RSLOTS, H, L, K, scale=0.09, device=device),
        g_cache=-torch.rand(RSLOTS, HV, L, generator=g, device=device) * 0.3,
        dlow_cache=_rn(g, RSLOTS, HV, L, V, scale=1e-4, device=device),
        klow_cache=_rn(g, RSLOTS, H, L, K, scale=1e-4, device=device),
        state_index=torch.tensor([1], dtype=torch.int32, device=device),
        window_index=torch.tensor([2], dtype=torch.int32, device=device),
        replay_index=torch.tensor([2], dtype=torch.int32, device=device),
        write_pos=torch.tensor([0, 0, wp], dtype=torch.int32, device=device),
        cache_base=torch.tensor([0, 0, base], dtype=torch.int32, device=device),
    )


def make_inputs(seed: int, device="cuda") -> dict:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    # Random in_proj outputs (qkvz, ba) come first in the draws; the kernel computes
    # its own from ``hidden``.
    return dict(
        qkvz=_rn(g, T, QKVZ, device=device),
        ba=_rn(g, T, 2 * HV, device=device),
        residual=_rn(g, T, HIDDEN, device=device),
        # The layer's input RMSNorm output (the in_proj input).
        hidden=_rn(g, T, HIDDEN, device=device),
    )


def empty_layer(device) -> dict:
    return dict(
        inproj_stored=torch.empty(
            QKVZ, HIDDEN, dtype=torch.float8_e4m3fn, device=device
        ).t(),
        inproj_scales=torch.empty(2, dtype=torch.float32, device=device),
        ba_weight=torch.empty(2 * HV, HIDDEN, dtype=torch.bfloat16, device=device),
        conv_weight=torch.empty(CONV_DIM, WIDTH, dtype=torch.bfloat16, device=device),
        A_log=torch.empty(HV, dtype=torch.float32, device=device),
        dt_bias=torch.empty(HV, dtype=torch.bfloat16, device=device),
        gate_norm_weight=torch.empty(V, dtype=torch.bfloat16, device=device),
        out_proj_stored=torch.empty(
            HIDDEN, VALUE_DIM, dtype=torch.float8_e4m3fn, device=device
        ).t(),
        out_proj_scales=torch.empty(2, dtype=torch.float32, device=device),
        norm_weight=torch.empty(HIDDEN, dtype=torch.bfloat16, device=device),
    )


def empty_state(device) -> dict:
    bf = torch.bfloat16
    i32 = torch.int32
    return dict(
        conv_state=torch.empty(SLOTS, CONV_DIM, WIDTH - 1, dtype=bf, device=device),
        conv_window=torch.empty(
            WSLOTS, CONV_DIM, T + WIDTH - 2, dtype=bf, device=device
        ),
        h0=torch.empty(SLOTS, HV, V, K, dtype=bf, device=device),
        d_cache=torch.empty(RSLOTS, HV, L, V, dtype=bf, device=device),
        k_cache=torch.empty(RSLOTS, H, L, K, dtype=bf, device=device),
        g_cache=torch.empty(RSLOTS, HV, L, dtype=torch.float32, device=device),
        dlow_cache=torch.empty(RSLOTS, HV, L, V, dtype=bf, device=device),
        klow_cache=torch.empty(RSLOTS, H, L, K, dtype=bf, device=device),
        state_index=torch.empty(1, dtype=i32, device=device),
        window_index=torch.empty(1, dtype=i32, device=device),
        replay_index=torch.empty(1, dtype=i32, device=device),
        write_pos=torch.empty(RSLOTS, dtype=i32, device=device),
        cache_base=torch.empty(RSLOTS, dtype=i32, device=device),
    )


def empty_inputs(device) -> dict:
    bf = torch.bfloat16
    return dict(
        residual=torch.empty(T, HIDDEN, dtype=bf, device=device),
        hidden=torch.empty(T, HIDDEN, dtype=bf, device=device),
    )


def kernel_args(layer: dict, state: dict, x: dict, partial, group_name) -> tuple:
    """qwen3_5_gdn_mixer_tp4's arguments."""
    return (
        x["hidden"],
        layer["inproj_stored"].t(),  # the [5120, 4096] physical layout
        layer["inproj_scales"],
        layer["ba_weight"],
        layer["conv_weight"],
        state["conv_state"],
        state["conv_window"],
        layer["A_log"],
        layer["dt_bias"],
        state["h0"],
        state["d_cache"],
        state["k_cache"],
        state["g_cache"],
        state["dlow_cache"],
        state["klow_cache"],
        state["state_index"],
        state["window_index"],
        state["replay_index"],
        state["write_pos"],
        state["cache_base"],
        layer["gate_norm_weight"],
        # The stored [2048, 4096] (1, 2048)-strided view's memory, read as rows.
        layer["out_proj_stored"].t(),
        layer["out_proj_scales"],
        x["residual"],
        layer["norm_weight"],
        partial,
        group_name,
    )
