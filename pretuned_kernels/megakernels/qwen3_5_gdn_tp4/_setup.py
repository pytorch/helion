# ruff: noqa: ANN001, ANN202, C408
"""Synthetic per-rank Qwen3.5 TP4 GDN mixer weights / state / inputs in SGLang's
served layouts, the GDN mixer megakernel's argument list, and the per-stage
references: SGLang's own kernels for its target-verify path (imported when
called) and torch for the FP8 GEMMs."""

from __future__ import annotations

from pretuned_kernels.megakernels._qwen3_5_tp4 import fp8_linear
from pretuned_kernels.megakernels._qwen3_5_tp4 import fp8_weight
from pretuned_kernels.megakernels._qwen3_5_tp4 import randn
from pretuned_kernels.megakernels._qwen3_5_tp4 import sglang_quant
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


def make_layer(seed: int, device="cuda") -> dict:
    """One layer's weights. ``out_proj_stored`` is SGLang's ``layer.weight`` after
    ``process_weights_after_loading`` (``quantized_weight.t()``: [2048, 4096] with
    strides (1, 2048)); the kernel reads its physical [4096, 2048] layout."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    w_fp8, weight_scale = fp8_weight(g, HIDDEN, VALUE_DIM, device)
    input_scale = torch.tensor(0.02, device=device)
    # in_proj_qkvz (ModelOpt per-tensor FP8, stored transposed like out_proj) and the
    # BF16 in_proj_ba.
    wi_fp8, in_weight_scale = fp8_weight(g, QKVZ, HIDDEN, device)
    in_input_scale = torch.tensor(0.01, device=device)
    return dict(
        inproj_stored=wi_fp8.t(),
        inproj_weight_scale=in_weight_scale,
        inproj_input_scale=in_input_scale,
        inproj_scales=torch.stack(
            (in_input_scale, in_input_scale * in_weight_scale)
        ).float(),
        ba_weight=randn(g, 2 * HV, HIDDEN, scale=0.02, device=device),
        conv_weight=randn(g, CONV_DIM, WIDTH, scale=0.3, device=device),
        A_log=(torch.rand(HV, generator=g, device=device) * 2.0).float(),
        # nn.Parameter(torch.ones(n)) under the BF16 default dtype.
        dt_bias=randn(g, HV, scale=0.5, device=device),
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
        conv_state=randn(g, SLOTS, CONV_DIM, WIDTH - 1, device=device),
        conv_window=randn(g, WSLOTS, CONV_DIM, T + WIDTH - 2, device=device),
        h0=randn(g, SLOTS, HV, V, K, scale=0.05, device=device),
        d_cache=randn(g, RSLOTS, HV, L, V, scale=0.1, device=device),
        k_cache=randn(g, RSLOTS, H, L, K, scale=0.09, device=device),
        g_cache=-torch.rand(RSLOTS, HV, L, generator=g, device=device) * 0.3,
        dlow_cache=randn(g, RSLOTS, HV, L, V, scale=1e-4, device=device),
        klow_cache=randn(g, RSLOTS, H, L, K, scale=1e-4, device=device),
        state_index=torch.tensor([1], dtype=torch.int32, device=device),
        window_index=torch.tensor([2], dtype=torch.int32, device=device),
        replay_index=torch.tensor([2], dtype=torch.int32, device=device),
        write_pos=torch.tensor([0, 0, wp], dtype=torch.int32, device=device),
        cache_base=torch.tensor([0, 0, base], dtype=torch.int32, device=device),
    )


def make_inputs(seed: int, device="cuda") -> dict:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return dict(
        residual=randn(g, T, HIDDEN, device=device),
        # The layer's input RMSNorm output (the in_proj input).
        hidden=randn(g, T, HIDDEN, device=device),
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


def window_view(phys: torch.Tensor) -> torch.Tensor:
    """MambaPool's deduplicated conv window view of one layer's physical
    [slots + 1, dim, T + 2] buffer: [slots + 1, T, dim, 3], step t = columns t..t+2."""
    s = phys.stride()
    return phys.as_strided(
        (phys.size(0), T, phys.size(1), WIDTH - 1), (s[0], s[2], s[1], s[2])
    )


def sglang_conv(qkvz, layer, state):
    """causal_conv1d_update as GDNAttnBackend.forward_extend calls it in target verify
    (mutates state's conv_state / conv_window). Returns mixed q|k|v [T, 3072]."""
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    mixed_in = qkvz[:, :CONV_DIM].contiguous()
    x = mixed_in.view(1, T, -1).transpose(1, 2)
    out = causal_conv1d_update(
        x,
        state["conv_state"],
        layer["conv_weight"],
        None,
        "silu",
        conv_state_indices=state["state_index"],
        intermediate_conv_window=window_view(state["conv_window"]),
        intermediate_state_indices=state["window_index"],
    )
    return out.transpose(1, 2).reshape(T, CONV_DIM)


def sglang_core(mixed, ba, layer, state):
    """gdn_replayssm_spec_decode(launch_mode="verify") as _replayssm_target_verify
    calls it (mutates the rings). Returns [T, 2048] bf16."""
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        gdn_replayssm_spec_decode,
    )

    q = mixed[:, :KEY_DIM].contiguous().view(T, H, K)
    k = mixed[:, KEY_DIM : 2 * KEY_DIM].contiguous().view(T, H, K)
    v = mixed[:, 2 * KEY_DIM :].contiguous().view(T, HV, V)
    out = torch.empty(T, HV, V, dtype=torch.bfloat16, device=mixed.device)
    gdn_replayssm_spec_decode(
        q=q,
        k=k,
        v=v,
        a=ba[:, HV:].contiguous(),
        b=ba[:, :HV].contiguous(),
        A_log=layer["A_log"],
        dt_bias=layer["dt_bias"],
        checkpoint_state=state["h0"],
        d_cache=state["d_cache"],
        k_cache=state["k_cache"],
        g_cache=state["g_cache"],
        rawv_cache=state["dlow_cache"],
        rawk_cache=state["klow_cache"],
        beta_cache=None,
        out=out,
        query_start_loc=torch.tensor([0, T], dtype=torch.int32, device=mixed.device),
        ssm_state_indices=state["state_index"],
        replay_indices=state["replay_index"],
        write_pos=state["write_pos"],
        cache_base=state["cache_base"],
        is_flush=torch.zeros(RSLOTS, dtype=torch.int8, device=mixed.device),
        max_cache_len=L,
        max_spec_len=T,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
        launch_mode="verify",
    )
    return out.view(T, VALUE_DIM)


def sglang_gated_norm(core, qkvz, layer):
    """rms_norm_gated (RMSNormGated's Triton path) of the core output with z."""
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

    z = qkvz[:, CONV_DIM:].contiguous().view(-1, V)
    y = rms_norm_gated(
        x=core.reshape(-1, V),
        weight=layer["gate_norm_weight"],
        bias=None,
        z=z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="swish",
    )
    return y.view(T, VALUE_DIM)


def torch_in_proj(hidden, layer):
    """in_proj_qkvz (static FP8 input quant, then ``fp8_linear``) and the BF16
    in_proj_ba."""
    in_scale = layer["inproj_input_scale"]
    q8 = sglang_quant(hidden, in_scale)
    qkvz = fp8_linear(
        q8, layer["inproj_stored"], in_scale, layer["inproj_weight_scale"]
    )
    ba = (hidden.float() @ layer["ba_weight"].float().T).to(torch.bfloat16)
    return qkvz, ba


def torch_out_proj(q8, layer):
    """out_proj (``fp8_linear``) on its FP8 input."""
    return fp8_linear(
        q8, layer["out_proj_stored"], layer["input_scale"], layer["weight_scale"]
    )
