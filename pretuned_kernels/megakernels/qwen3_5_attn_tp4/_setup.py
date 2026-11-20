# ruff: noqa: ANN001, ANN002, ANN202, C408
"""Synthetic per-rank Qwen3.5-397B TP4 full-attention mixer weights / KV state /
inputs in SGLang's served layouts (dims from the checkpoint config), and the
attention mixer megakernel's argument list.

Per rank (TP4): 8 of 32 query heads (q and the output gate interleaved per head,
256 each), 1 of 2 KV heads (replicated over rank pairs), head_dim 256, partial
NeoX rope over the first 64 dims (interleaved mrope sections [11, 11, 10], theta
1e7), FP8 E4M3 paged KV cache (page 64, K and V halves of one allocation),
ModelOpt per-tensor FP8 qkv_proj / o_proj (scales of checkpoint layer 3)."""

from __future__ import annotations

import torch

T = 7  # verify window (speculative-num-draft-tokens)
HIDDEN = 4096
HEADS, KV_HEADS, HEAD_DIM = 8, 1, 256  # per rank
Q_SIZE = HEADS * HEAD_DIM  # 2048
QG_SIZE = 2 * Q_SIZE  # q | gate interleaved per head
KV_SIZE = KV_HEADS * HEAD_DIM  # 256
QKV_ROWS = QG_SIZE + 2 * KV_SIZE  # 4608
ROTARY_DIM = 64
MROPE_SECTION = [11, 11, 10]
ROPE_THETA = 10_000_000
MAX_POSITION = 262_144  # max_position_embeddings (cos_sin_cache rows)
PAGE = 64
CONTEXT_LEN = 32_768  # --context-length
MAX_PAGES = CONTEXT_LEN // PAGE  # trtllm_mha max_num_pages (page-table width)
POOL_PAGES = 600  # KV pool pages per layer (+ the padding page 0)
EPS = 1e-6
# Checkpoint layer 3 (model.language_model.layers.3.self_attn.*).
QKV_INPUT_SCALE = 0.113839291036129
O_INPUT_SCALE = 0.0035226005129516125
K_SCALE = 0.0193917416036129
V_SCALE = 0.0159737728536129


def _rn(g, *s, scale=1.0, dtype=torch.bfloat16, device="cuda"):
    return (torch.randn(*s, generator=g, device=device) * scale).to(dtype)


def _fp8_weight(g, rows, cols, device):
    w = torch.randn(rows, cols, generator=g, device=device) * 0.02
    s = (w.abs().amax() / 448.0).reshape(())
    return (w / s).clamp(-448, 448).to(torch.float8_e4m3fn), s


def rope_tables(device="cuda"):
    """SGLang's MRotaryEmbedding (get_rope with the checkpoint's rope parameters):
    the FP32 [MAX_POSITION, 64] cos | sin cache and the [32] lane -> axis map."""
    from sglang.srt.layers.rotary_embedding import get_rope

    rope = get_rope(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position=MAX_POSITION,
        base=ROPE_THETA,
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": MROPE_SECTION,
            "rope_type": "default",
        },
        partial_rotary_factor=ROTARY_DIM / HEAD_DIM,
        is_neox_style=True,
        dtype=torch.bfloat16,
    )
    assert rope.rotary_dim == ROTARY_DIM
    return rope.cos_sin_cache.to(device), rope.axis_map.to(device)


def make_layer(seed: int, device="cuda") -> dict:
    """One layer's weights. ``qkv_stored`` / ``o_proj_stored`` are SGLang's
    ``layer.weight`` after ModelOpt's ``process_weights_after_loading``: the q|k|v
    shards requantized to the max shard scale (requantize_with_max_scale), then
    ``.t()`` ([K, N] with strides (1, K)); the kernel reads the physical [N, K]."""
    from sglang.srt.layers.quantization.utils import requantize_with_max_scale

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    shards = [_fp8_weight(g, n, HIDDEN, device) for n in (QG_SIZE, KV_SIZE, KV_SIZE)]
    w = torch.cat([s[0] for s in shards])
    w_scale = torch.stack([s[1] for s in shards]).float()
    qkv_scale, w = requantize_with_max_scale(w, w_scale, [QG_SIZE, KV_SIZE, KV_SIZE])
    qkv_scale = qkv_scale.reshape(())
    o_fp8, o_scale = _fp8_weight(g, HIDDEN, Q_SIZE, device)
    qkv_in = torch.tensor(QKV_INPUT_SCALE, device=device)
    o_in = torch.tensor(O_INPUT_SCALE, device=device)
    k_scale = torch.tensor(K_SCALE, device=device)
    v_scale = torch.tensor(V_SCALE, device=device)
    return dict(
        qkv_stored=w.t(),
        qkv_weight_scale=qkv_scale,
        qkv_input_scale=qkv_in,
        qkv_scales=torch.stack((qkv_in, qkv_in * qkv_scale)).float(),
        # GemmaRMSNorm.weight (raw: the norm multiplies by 1 + w).
        q_norm_weight=_rn(g, HEAD_DIM, scale=0.15, device=device),
        k_norm_weight=_rn(g, HEAD_DIM, scale=0.15, device=device),
        o_proj_stored=o_fp8.t(),
        o_weight_scale=o_scale,
        o_input_scale=o_in,
        o_scales=torch.stack((o_in, o_in * o_scale)).float(),
        k_scale=k_scale,
        v_scale=v_scale,
        # [1 / k_scale, 1 / v_scale (IEEE, as the CUDA cache write divides),
        #  bmm1 = k_scale * head_dim^-0.5 * log2(e), bmm2 = v_scale]
        kv_scales=torch.tensor(
            [
                1.0 / K_SCALE,
                1.0 / V_SCALE,
                K_SCALE * HEAD_DIM**-0.5 * 1.4426950408889634,
                V_SCALE,
            ],
            dtype=torch.float32,
            device=device,
        ),
        # The next norm's GemmaRMSNorm.gemma_weight: BF16 (w + 1).
        norm_weight=(1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).to(
            torch.bfloat16
        ),
    )


def kv_buffers(device="cuda"):
    """One layer's K / V pool buffers as MHATokenToKVPool allocates them: the two
    block-aligned halves of one [2, rows, 1, 256] FP8 allocation (the kernel takes
    that allocation, ``k_cache._base``, as one [2, rows, 256] tensor)."""
    rows = (POOL_PAGES + 1) * PAGE
    fused = torch.zeros(2, rows, KV_HEADS, HEAD_DIM, dtype=torch.uint8, device=device)
    fused = fused.view(torch.float8_e4m3fn)
    return fused[0], fused[1]


def kv_fused(state) -> torch.Tensor:
    """The [2, rows, 256] K | V allocation behind state's two halves."""
    k = state["k_cache"]
    rows = k.size(0)
    fused = k.as_strided((2, rows, HEAD_DIM), (rows * HEAD_DIM, HEAD_DIM, 1))
    assert fused[1].data_ptr() == state["v_cache"].data_ptr()
    return fused


def make_state(seed: int, prefix: int, device="cuda") -> dict:
    """The request's KV state for one layer: ``prefix`` committed tokens in random
    pages (FP8 values as the cache write would store them), the page table, the
    verify tokens' slots and (m)rope positions, and cache_seqlens = prefix + T."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    k_cache, v_cache = kv_buffers(device)
    total = prefix + T
    pages = (total + PAGE - 1) // PAGE
    assert pages <= POOL_PAGES
    # Page 0 holds slot 0 (the padding sentinel): requests never own it.
    perm = torch.randperm(POOL_PAGES, generator=g, device=device)[:pages] + 1
    page_table = torch.zeros(1, MAX_PAGES, dtype=torch.int32, device=device)
    page_table[0, :pages] = perm.to(torch.int32)
    pos = torch.arange(total, device=device)
    slots = page_table[0, pos // PAGE].long() * PAGE + pos % PAGE
    # Committed K: unit-RMS rows (post norm/rope), V: projection-sized rows.
    k = torch.randn(prefix, HEAD_DIM, generator=g, device=device) * 1.2
    v = torch.randn(prefix, HEAD_DIM, generator=g, device=device) * 1.3
    k_cache.view(-1, HEAD_DIM)[slots[:prefix]] = (
        (k / K_SCALE).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    v_cache.view(-1, HEAD_DIM)[slots[:prefix]] = (
        (v / V_SCALE).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    # Garbage where this window's rows go (the kernel must write them).
    junk = torch.randn(T, HEAD_DIM, generator=g, device=device).to(torch.float8_e4m3fn)
    k_cache.view(-1, HEAD_DIM)[slots[prefix:]] = junk
    v_cache.view(-1, HEAD_DIM)[slots[prefix:]] = junk
    return dict(
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        seq_lens=torch.tensor([total], dtype=torch.int32, device=device),
        cache_loc=slots[prefix:].clone(),
        positions=pos[prefix:].expand(3, T).contiguous(),
    )


def make_inputs(seed: int, device="cuda") -> dict:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return dict(
        # The layer's input RMSNorm output (qkv_proj's input) and the residual.
        hidden=_rn(g, T, HIDDEN, device=device),
        residual=_rn(g, T, HIDDEN, device=device),
    )


def empty_layer(device) -> dict:
    f32 = torch.float32
    bf = torch.bfloat16
    return dict(
        qkv_stored=torch.empty(
            QKV_ROWS, HIDDEN, dtype=torch.float8_e4m3fn, device=device
        ).t(),
        qkv_scales=torch.empty(2, dtype=f32, device=device),
        q_norm_weight=torch.empty(HEAD_DIM, dtype=bf, device=device),
        k_norm_weight=torch.empty(HEAD_DIM, dtype=bf, device=device),
        o_proj_stored=torch.empty(
            HIDDEN, Q_SIZE, dtype=torch.float8_e4m3fn, device=device
        ).t(),
        o_scales=torch.empty(2, dtype=f32, device=device),
        kv_scales=torch.empty(4, dtype=f32, device=device),
        norm_weight=torch.empty(HIDDEN, dtype=bf, device=device),
    )


def empty_state(device) -> dict:
    k_cache, v_cache = kv_buffers(device)
    return dict(
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=torch.empty(1, MAX_PAGES, dtype=torch.int32, device=device),
        seq_lens=torch.empty(1, dtype=torch.int32, device=device),
        cache_loc=torch.empty(T, dtype=torch.int64, device=device),
        positions=torch.empty(3, T, dtype=torch.int64, device=device),
    )


def empty_inputs(device) -> dict:
    bf = torch.bfloat16
    return dict(
        hidden=torch.empty(T, HIDDEN, dtype=bf, device=device),
        residual=torch.empty(T, HIDDEN, dtype=bf, device=device),
    )


def kernel_args(layer, state, x, rope, partial, group_name) -> tuple:
    """qwen3_5_attn_mixer_tp4's arguments (rope = rope_tables())."""
    cos_sin, axis_map = rope
    return (
        x["hidden"],
        layer["qkv_stored"].t(),  # the [4608, 4096] physical layout
        layer["qkv_scales"],
        layer["q_norm_weight"],
        layer["k_norm_weight"],
        cos_sin,
        state["positions"],
        axis_map,
        kv_fused(state),
        layer["kv_scales"],
        state["cache_loc"],
        state["page_table"],
        state["seq_lens"],
        # The stored [2048, 4096] (1, 2048)-strided view's memory, read as rows.
        layer["o_proj_stored"].t(),
        layer["o_scales"],
        x["residual"],
        layer["norm_weight"],
        partial,
        group_name,
    )
