# ruff: noqa: ANN001, ANN202, C408
"""Synthetic per-rank Qwen3.5-397B TP4 full-attention mixer weights / KV state /
inputs in SGLang's served layouts (dims from the checkpoint config), and the
attention mixer megakernel's argument list.

Per rank (TP4): 8 of 32 query heads (q and the output gate interleaved per head,
256 each), 1 of 2 KV heads (replicated over rank pairs), head_dim 256, partial
NeoX rope over the first 64 dims (interleaved mrope sections [11, 11, 10], theta
1e7), FP8 E4M3 paged KV cache (page 64, K and V halves of one allocation),
ModelOpt per-tensor FP8 qkv_proj / o_proj (scales of checkpoint layer 3).

Also the per-stage references: SGLang's own kernels for its target-verify path
(imported when called), FlashInfer's TRT-LLM decode kernel, torch for the FP8
GEMMs and an exact FP32 attention."""

from __future__ import annotations

from pretuned_kernels.megakernels._qwen3_5_tp4 import NORM_EPS
from pretuned_kernels.megakernels._qwen3_5_tp4 import fp8_linear
from pretuned_kernels.megakernels._qwen3_5_tp4 import fp8_weight
from pretuned_kernels.megakernels._qwen3_5_tp4 import randn
from pretuned_kernels.megakernels._qwen3_5_tp4 import sglang_quant
import torch

T = 7  # verify window (speculative-num-draft-tokens)
HIDDEN = 4096
HEADS, KV_HEADS, HEAD_DIM = 8, 1, 256  # per rank
Q_SIZE = HEADS * HEAD_DIM  # 2048
QG_SIZE = 2 * Q_SIZE  # q | gate interleaved per head
KV_SIZE = KV_HEADS * HEAD_DIM  # 256
ROTARY_DIM = 64
MROPE_SECTION = [11, 11, 10]
ROPE_THETA = 10_000_000
MAX_POSITION = 262_144  # max_position_embeddings (cos_sin_cache rows)
PAGE = 64
CONTEXT_LEN = 32_768  # --context-length
MAX_PAGES = CONTEXT_LEN // PAGE  # trtllm_mha max_num_pages (page-table width)
POOL_PAGES = 600  # KV pool pages per layer (+ the padding page 0)
# Checkpoint layer 3 (model.language_model.layers.3.self_attn.*).
QKV_INPUT_SCALE = 0.113839291036129
O_INPUT_SCALE = 0.0035226005129516125
K_SCALE = 0.0193917416036129
V_SCALE = 0.0159737728536129


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
    shards = [fp8_weight(g, n, HIDDEN, device) for n in (QG_SIZE, KV_SIZE, KV_SIZE)]
    w = torch.cat([s[0] for s in shards])
    w_scale = torch.stack([s[1] for s in shards]).float()
    qkv_scale, w = requantize_with_max_scale(w, w_scale, [QG_SIZE, KV_SIZE, KV_SIZE])
    qkv_scale = qkv_scale.reshape(())
    o_fp8, o_scale = fp8_weight(g, HIDDEN, Q_SIZE, device)
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
        q_norm_weight=randn(g, HEAD_DIM, scale=0.15, device=device),
        k_norm_weight=randn(g, HEAD_DIM, scale=0.15, device=device),
        o_proj_stored=o_fp8.t(),
        o_weight_scale=o_scale,
        o_input_scale=o_in,
        o_scales=torch.stack((o_in, o_in * o_scale)).float(),
        k_scale=k_scale,
        v_scale=v_scale,
        # [1 / k_scale, 1 / v_scale (FP32 IEEE, as the CUDA cache write divides),
        #  bmm1 = k_scale * head_dim^-0.5 * log2(e), bmm2 = v_scale]
        kv_scales=torch.stack(
            (
                1 / k_scale,
                1 / v_scale,
                torch.tensor(
                    K_SCALE * HEAD_DIM**-0.5 * 1.4426950408889634, device=device
                ),
                v_scale,
            )
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
        hidden=randn(g, T, HIDDEN, device=device),
        residual=randn(g, T, HIDDEN, device=device),
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


def clone_state(state: dict) -> dict:
    """A copy of ``state`` with K / V halves of one new allocation (as
    ``kv_buffers`` lays them out)."""
    out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
    out["k_cache"], out["v_cache"] = kv_buffers(state["k_cache"].device)
    out["k_cache"].copy_(state["k_cache"])
    out["v_cache"].copy_(state["v_cache"])
    return out


_WORKSPACE = {}


def _workspace(device):
    if device not in _WORKSPACE:
        _WORKSPACE[device] = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
    return _WORKSPACE[device]


def _paged(cache):
    """trtllm_mha's _reshape_paged_kv_cache view (HND; 1 KV head)."""
    return cache.view(-1, PAGE, KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)


def exact_attention(q, state):
    """FP32 attention over the FP8 values the cache holds (after the window's write):
    q as FP8 (scale 1), K * k_scale, V * v_scale, exact softmax, causal window."""
    total = int(state["seq_lens"][0])
    pos = torch.arange(total, device=q.device)
    slots = state["page_table"][0, pos // PAGE].long() * PAGE + pos % PAGE
    k = state["k_cache"].view(-1, HEAD_DIM)[slots].float() * K_SCALE
    v = state["v_cache"].view(-1, HEAD_DIM)[slots].float() * V_SCALE
    q8 = q.to(torch.float8_e4m3fn).float().view(T, HEADS, HEAD_DIM)
    scores = torch.einsum("thd,kd->htk", q8, k) * HEAD_DIM**-0.5
    mask = pos[None, :] > (total - T + torch.arange(T, device=q.device))[:, None]
    scores = scores.masked_fill(mask[None], float("-inf"))
    out = torch.einsum("htk,kd->thd", scores.softmax(-1), v)
    return out.reshape(T, Q_SIZE)


def torch_qkv(hidden, layer):
    """qkv_proj: static FP8 input quant, then ``fp8_linear`` ([T, 4608])."""
    in_scale = layer["qkv_input_scale"]
    q8 = sglang_quant(hidden, in_scale)
    return fp8_linear(q8, layer["qkv_stored"], in_scale, layer["qkv_weight_scale"])


def sglang_qk_prep(qkv, layer, rope, state):
    """forward_prepare_cuda_fused: q, k (norm + rope) and the de-interleaved gate."""
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    cos_sin, axis_map = rope
    q_gate, k, v = qkv.split([QG_SIZE, KV_SIZE, KV_SIZE], dim=-1)
    q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        layer["q_norm_weight"],
        layer["k_norm_weight"],
        cos_sin,
        state["positions"],
        NORM_EPS,
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
        has_gate=True,
        mrope_axis_map=axis_map,
    )
    return q.view(T, -1), k.view(T, -1), v, gate


def sglang_attention(q, k, v, layer, state):
    """fused_fp8_qkv_kv_cache + trtllm_batch_decode_with_kv_cache as trtllm_mha's
    forward_extend runs them in target verify (mutates state's caches)."""
    import flashinfer
    from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import fused_fp8_qkv_kv_cache

    q8 = fused_fp8_qkv_kv_cache(
        q=q,
        k=k,
        v=v,
        k_cache=state["k_cache"],
        v_cache=state["v_cache"],
        cache_loc=state["cache_loc"],
        k_scale=layer["k_scale"],
        v_scale=layer["v_scale"],
    )
    out = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q8.view(-1, HEADS, HEAD_DIM),
        kv_cache=(_paged(state["k_cache"]), _paged(state["v_cache"])),
        workspace_buffer=_workspace(q.device),
        block_tables=state["page_table"],
        seq_lens=state["seq_lens"],
        max_seq_len=CONTEXT_LEN,
        bmm1_scale=K_SCALE * HEAD_DIM**-0.5,
        bmm2_scale=V_SCALE,
        window_left=-1,
        out_dtype=torch.bfloat16,
        q_len_per_req=T,
    )
    return out.view(T, Q_SIZE)


def sglang_gate(attn, gate):
    """fused_sigmoid_mul: the output gate ``bf16(attn * sigmoid(gate))``."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    return fused_sigmoid_mul(attn.clone(), gate, inplace=True)


def torch_o_proj(q8, layer):
    """o_proj (``fp8_linear``) on its FP8 input."""
    return fp8_linear(
        q8, layer["o_proj_stored"], layer["o_input_scale"], layer["o_weight_scale"]
    )
