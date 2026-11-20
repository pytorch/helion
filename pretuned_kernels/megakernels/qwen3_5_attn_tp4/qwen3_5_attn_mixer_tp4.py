# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""Qwen3.5 full-attention mixer for a TP4 speculative-verify window (one request,
T <= 8 tokens), from the layer's input RMSNorm output to the next RMSNorm, as one
persistent Gluon megakernel.

Per rank: 8 query heads (q and the output gate interleaved per head) on 1 KV head,
head_dim 256, FP8 E4M3 paged KV cache (page 64). The roots reproduce SGLang's
target-verify path (trtllm_mha backend, SM100):

- q / kv / gate projection: qkv_proj (ModelOpt per-tensor FP8 [4608, 4096], the
  physical layout of SGLang's transposed ``weight``) as FP32 partials of 4 K chunks:
  the q rows in 64-row tasks (128 tasks), the k | v rows (16 tasks, each also writing
  the window lanes' rope cos / sin table, gathered while its weights stream) and, in
  a root of their own that only the merge waits for, the gate rows (64 tasks). Each
  K step quantizes its slice of the BF16 input in registers with qkv_proj's static
  FP8 input quant (``clamp(x * (1 / input_scale))`` -> E4M3): the input rides the
  ring with the weights, so no task waits for a quant step. Partials sum in chunk
  order, * input_scale * weight_scale, and round to BF16 as SGLang's GEMM output does.
- q / kv prep: ``fused_qk_gemma_rmsnorm_rope_gate`` per head (GemmaRMSNorm with
  ``1 + w``, BF16; NeoX rope on the first 64 dims with mrope positions per lane,
  BF16), then ``fused_fp8_qkv_kv_cache``'s FP8: q with scale 1, K / V * (1 /
  scale). The kv prep reads each K chunk's k | v partial rows as one [token, 2, 256]
  load and the rope table (no dependent gathers behind its root barrier).
- cache splits / merge: the attention core, query rows token * 8 + head (64-row FP8
  tile). Up to 64 splits, each attending at least 2 of the committed 64-key pages,
  with an online softmax (TMA ring, page coordinates from the page table; the ring's
  KV producer orders on nothing the splits wait for, so the pages stream while q is
  prepared); BF16 partials + log2-sum-exp2 per row. The merge per (head, 64
  columns): first the window's own keys (causal) for the head's 8 query rows, which
  wait only for the q / kv prep (so they run while the splits finish), then every
  group's lse loaded up front (their max), a loop over groups of 16 splits whose
  partials are one [16 split, 8 token, 64] box of the workspace (ring), the output
  gate (``fused_sigmoid_mul``: ``bf16(attn * sigmoid(gate))``) and o_proj's static
  FP8 input quant.
- o_proj: ModelOpt per-tensor FP8 ``[4096, 2048]`` as K-split FP32 partials, one
  publish task per (token, 512 columns): ``bf16(sum * input_scale * weight_scale)``
  into the symmetric partial.
- the tensor-parallel all-reduce (rank order, FP32) + residual and the next
  GemmaRMSNorm as SGLang's CuTe DSL ``kARResidualRMSNorm`` fusion computes them.
- cache write: the window's FP8 K / V into their slots, after every cache reader
  (the attention never reads the window's slots from the cache).

Latency rules the layout follows (GB300 timelines): every independent weight /
scale load sits first in its task; loads right behind a dependency wait that only
feed registers are pointer loads (the ring's register channel holds two slots, a
ring load there costs a round trip each), and small outputs are pointer stores (no
bulk-store drain ahead of the root's release); see ``_POINTER_OPS``.

Besides the next layer's input and residual, it returns the intermediate
activations SGLang materializes (qkv, q / k after norm and rope, attention output,
FP8 o_proj input) for checks.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

FP8_MAX = 448.0
NORM_EPS = 1e-6


def _online_step(q_tile, key, value, valid, run_max, run_sum, acc, bmm1_log2):
    """One online-softmax update of the [64] row state with FP8 keys / values."""
    scores = hl.dot(q_tile, key.T, out_dtype=torch.float32)
    scores = torch.where(valid, scores * bmm1_log2, float("-inf"))
    new_max = torch.maximum(run_max, torch.amax(scores, dim=1))
    correction = torch.exp2(run_max - new_max)
    prob = torch.exp2(scores - new_max[:, None])
    acc = hl.dot(
        (prob * 256.0).to(torch.float8_e4m3fn),
        value,
        acc=acc * correction[:, None],
        out_dtype=torch.float32,
    )
    run_sum = run_sum * correction + torch.sum(prob, dim=1)
    return new_max, run_sum, acc


def _pair_online(
    q_tile,
    key_a,
    key_b,
    value_a,
    value_b,
    valid_a,
    valid_b,
    run_max,
    run_sum,
    acc,
    bmm1_log2,
):
    """Two pages as one online-softmax update: both QKs, the exact max over both,
    one rescale, then both PVs (the second accumulates without a rescale)."""
    s_a = hl.dot(q_tile, key_a.T, out_dtype=torch.float32)
    s_b = hl.dot(q_tile, key_b.T, out_dtype=torch.float32)
    s_a = torch.where(valid_a, s_a * bmm1_log2, float("-inf"))
    s_b = torch.where(valid_b, s_b * bmm1_log2, float("-inf"))
    new_max = torch.maximum(
        run_max, torch.maximum(torch.amax(s_a, dim=1), torch.amax(s_b, dim=1))
    )
    correction = torch.exp2(run_max - new_max)
    p_a = torch.exp2(s_a - new_max[:, None])
    p_b = torch.exp2(s_b - new_max[:, None])
    acc = hl.dot(
        (p_a * 256.0).to(torch.float8_e4m3fn),
        value_a,
        acc=acc * correction[:, None],
        out_dtype=torch.float32,
    )
    acc = hl.dot(
        (p_b * 256.0).to(torch.float8_e4m3fn), value_b, acc=acc, out_dtype=torch.float32
    )
    run_sum = run_sum * correction + torch.sum(p_a, dim=1) + torch.sum(p_b, dim=1)
    return new_max, run_sum, acc


def _proj(parts, chunks, scale, t_tile, head, cols):
    """Projection rows of one head over a column slice: BF16(sum over the K chunks'
    FP32 partials in chunk order * scale), as FP32 holding the BF16 values."""
    acc = parts[0, t_tile, head, cols]
    for c in hl.static_range(1, chunks):
        acc = acc + parts[c, t_tile, head, cols]
    return (acc * scale).to(torch.bfloat16).to(torch.float32)


def _proj_kv(parts, chunks, scale, t_tile):
    """k and v rows of the k | v projection over all columns from [token, 2, 256]
    boxes of each K chunk: BF16(sum over the chunks' FP32 partials in chunk order *
    scale), as FP32 holding the BF16 values."""
    acc = parts[0, t_tile, :, :]
    for c in hl.static_range(1, chunks):
        acc = acc + parts[c, t_tile, :, :]
    k, v = hl.split(acc.permute(0, 2, 1))
    return (k * scale).to(torch.bfloat16).to(torch.float32), (v * scale).to(
        torch.bfloat16
    ).to(torch.float32)


def _norm_rope_x(x, w1, cos, sin, head_dim):
    """fused_qk_gemma_rmsnorm_rope_gate for one head's projection rows x (BF16 values
    as FP32): GemmaRMSNorm over all 256 (BF16), NeoX rope on dims [0, 64) (x1 = [0,
    32), x2 = [32, 64); first = x1 cos - x2 sin, second = x2 cos + x1 sin, BF16), dims
    [64, 256) passed through. Returns (x, out)."""
    inv_rms = torch.rsqrt(torch.sum(x * x, dim=1) / head_dim + 1e-6)
    xn = _gemma_norm(x, inv_rms, w1)
    xp = _pair_swap32(xn)
    cols = hl.arange(head_dim)[None, :]
    rot = torch.where(cols < 32, xn * cos - xp * sin, xn * cos + xp * sin)
    out = torch.where(cols < 64, rot.to(torch.bfloat16).to(torch.float32), xn)
    return x, out


def _gemma_norm(x, inv_rms, w1):
    """GemmaRMSNorm's BF16 output (``x * inv_rms * w1``, w1 = 1 + w in FP32) as FP32."""
    return (x * inv_rms[:, None] * w1[None, :]).to(torch.bfloat16).to(torch.float32)


def _rope_tables(cos_sin, positions, axis_map, t_tile, tokens, head_dim):
    """cos / sin [lanes, head_dim] of the window tokens, column d using rotary lane
    d % 32, which reads position positions[axis_map[lane], t] (mrope)."""
    rot = hl.arange(head_dim) % 32
    live = t_tile.index < tokens
    lane = torch.where(live, t_tile.index, 0)
    pos = hl.load(positions, [hl.load(axis_map, [rot])[None, :], lane[:, None]])
    cos = hl.load(cos_sin, [pos, rot[None, :]])
    sin = hl.load(cos_sin, [pos, (rot + 32)[None, :]])
    return cos, sin


def _pair_swap32(x):
    """x [n, 256] with its 32-column blocks swapped pairwise (0 <-> 1, 2 <-> 3, ...):
    column d's NeoX rotation partner for d < 64."""
    n = x.size(0)
    lo, hi = hl.split(x.reshape(n, 4, 2, 32).permute(0, 1, 3, 2))
    return hl.join(hi, lo).permute(0, 1, 3, 2).reshape(n, 256)


def _norm_rope_head(parts, chunks, scale, t_tile, head, w1, cos, sin, head_dim):
    """_norm_rope_x of one head's projection rows (from the K-chunk partials)."""
    return _norm_rope_x(
        _proj(parts, chunks, scale, t_tile, head, slice(None)), w1, cos, sin, head_dim
    )


# Config.indexing slots (one per load / store, in the order Helion visits them: a
# loop's ops at the loop's place in its root) lowered as pointer accesses; every
# other memory op is a tensor descriptor (TMA ring / bulk store). The schedule's
# readiness keys come from the tile subscripts, so they stay as they are.
_MEMORY_OPS = 101
_POINTER_OPS = {
    3: "q projection partials",
    13: "k | v projection partials",
    17: "gate projection partials",
    24: "q prep: q partials, K chunk 0",
    25: "q prep: q partials, K chunk 1",
    26: "q prep: q partials, K chunk 2",
    27: "q prep: q partials, K chunk 3",
    28: "q prep: q_x",
    29: "q prep: q_rope",
    41: "kv prep: kv_x k",
    42: "kv prep: kv_x v",
    43: "kv prep: k_rope",
    49: "cache splits: q8",
    71: "merge: window q rows",
    72: "merge: window keys",
    73: "merge: window V",
    85: "o_proj partials",
    88: "publish: symmetric partial",
    89: "all-reduce: residual",
    94: "all-reduce: residual output",
    98: "norm: row partial sums",
    99: "norm: input",
    100: "norm: output",
}


def _indexing():
    return [
        "pointer" if op in _POINTER_OPS else "tensor_descriptor"
        for op in range(_MEMORY_OPS)
    ]


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    config=helion.Config(
        block_sizes=[16, 256, 16, 128, 16, 128, 16, 128],
        num_warps=4,
        pid_type="persistent_blocked",
        cross_loop_pipeline="dynamic",
        num_sm_multiplier=1,
        num_stages=4,
        range_warp_specializes=[True] * 18,
        indexing=_indexing(),
    ),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def qwen3_5_attn_mixer_tp4(
    hidden_states,  # [T, 4096] bf16: the layer's input RMSNorm output
    qkv_weight,  # [4608, 4096] e4m3: q|gate per head (8 x 512), k (256), v (256)
    qkv_scales,  # [2] f32: input_scale, input_scale * weight_scale
    q_norm_weight,  # [256] bf16 (GemmaRMSNorm.weight; the norm adds 1)
    k_norm_weight,  # [256] bf16
    cos_sin_cache,  # [max_position, 64] f32: cos | sin
    positions,  # [3, T] int64: mrope positions (temporal, height, width)
    axis_map,  # [32] int64: the mrope axis of each rotary lane
    kv_cache,  # [2, slots, 256] e4m3: one layer's K | V pool allocation (1 KV head)
    kv_scales,  # [4] f32: 1 / k_scale, 1 / v_scale, k_scale * d^-0.5 * log2e, v_scale
    cache_loc,  # [T] int64: the window's slots
    page_table,  # [1, max_pages] int32
    seq_lens,  # [1] int32: committed + T
    o_proj_weight,  # [4096, 2048] e4m3 (row = output, K contiguous)
    o_scales,  # [2] f32: input_scale, input_scale * weight_scale
    residual,  # [T, 4096] bf16
    norm_weight,  # [4096] bf16, the next norm's (1 + w)
    symmetric_partial,  # [T, 4096] bf16 symmetric memory
    group_name: hl.ProcessGroupName,
):
    hl.specialize(hidden_states.size())
    hl.specialize(hidden_states.stride())
    hl.specialize(qkv_weight.size())
    hl.specialize(qkv_weight.stride())
    hl.specialize(qkv_scales.size())
    hl.specialize(qkv_scales.stride())
    hl.specialize(q_norm_weight.size())
    hl.specialize(q_norm_weight.stride())
    hl.specialize(k_norm_weight.size())
    hl.specialize(k_norm_weight.stride())
    hl.specialize(cos_sin_cache.size())
    hl.specialize(cos_sin_cache.stride())
    hl.specialize(positions.size())
    hl.specialize(positions.stride())
    hl.specialize(axis_map.size())
    hl.specialize(axis_map.stride())
    hl.specialize(kv_cache.size())
    hl.specialize(kv_cache.stride())
    hl.specialize(kv_scales.size())
    hl.specialize(kv_scales.stride())
    hl.specialize(cache_loc.size())
    hl.specialize(cache_loc.stride())
    hl.specialize(page_table.size())
    hl.specialize(page_table.stride())
    hl.specialize(seq_lens.size())
    hl.specialize(seq_lens.stride())
    hl.specialize(o_proj_weight.size())
    hl.specialize(o_proj_weight.stride())
    hl.specialize(o_scales.size())
    hl.specialize(o_scales.stride())
    hl.specialize(residual.size())
    hl.specialize(residual.stride())
    hl.specialize(norm_weight.size())
    hl.specialize(norm_weight.stride())
    hl.specialize(symmetric_partial.size())
    hl.specialize(symmetric_partial.stride())

    tokens, hidden_dim = hidden_states.shape
    head_dim = q_norm_weight.size(0)
    q_width = o_proj_weight.size(1)
    heads = q_width // head_dim
    hidden = residual.size(1)
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    lanes_n = 8  # window token lanes (T <= 8)
    rows_n = 64  # heads * lanes: the attention's query rows
    page = 64
    merge_n = 16  # splits per merge load
    splits = 64
    ip_chunks = 4
    assert tokens <= lanes_n and heads * lanes_n == rows_n
    assert qkv_weight.shape == (2 * q_width + 2 * head_dim, hidden_dim)
    assert o_proj_weight.shape == (hidden, q_width)
    assert symmetric_partial.size() == (tokens, hidden)
    dev = residual.device

    # qkv_proj: the BF16 input in K chunks (each K step quantizes its slice); FP32
    # partials per K chunk.
    chunk_k = hidden_dim // ip_chunks
    hidden_chunked = hidden_states.view(tokens, ip_chunks, chunk_k)
    # [head, q | gate, row, K chunk, k] and [k | v, row, K chunk, k]
    qg_weight = qkv_weight[: 2 * q_width].view(heads, 2, head_dim, ip_chunks, chunk_k)
    kv_weight = qkv_weight[2 * q_width :].view(2, head_dim, ip_chunks, chunk_k)
    q_parts = torch.empty(
        (ip_chunks, tokens, heads, head_dim), dtype=torch.float32, device=dev
    )
    g_parts = torch.empty(
        (ip_chunks, tokens, heads, head_dim), dtype=torch.float32, device=dev
    )
    kv_parts = torch.empty(
        (ip_chunks, tokens, 2, head_dim), dtype=torch.float32, device=dev
    )
    # The window lanes' rope cos | sin per column (written by every k | v GEMM task).
    rope_cs = torch.empty((2, tokens, head_dim), dtype=torch.float32, device=dev)

    # The attention's inputs and partials. q8 rows: token * 8 + head (token lanes
    # past the window are never written: their rows only reach their own rows), so a
    # partial's rows are [token, 2048] = [token, o_proj K chunk, 512] (the merge's
    # tiles are boxes); lse is stored [split, head, token].
    q8 = torch.empty((lanes_n, heads, head_dim), dtype=torch.float8_e4m3fn, device=dev)
    # The window's FP8 K / V rows (the merge reads them; then the cache write).
    k_win = torch.empty((lanes_n, head_dim), dtype=torch.float8_e4m3fn, device=dev)
    v_win = torch.empty((lanes_n, head_dim), dtype=torch.float8_e4m3fn, device=dev)
    att_ws = torch.empty((splits, rows_n, head_dim), dtype=torch.bfloat16, device=dev)
    lse_ws = torch.empty((splits, rows_n), dtype=torch.float32, device=dev)
    merge_groups = splits // merge_n
    att_grp = att_ws.view(merge_groups, merge_n, lanes_n, heads, head_dim)
    kv_pages = kv_cache.view(2, kv_cache.size(1) // page, page, head_dim)

    # Checks: what SGLang materializes (one tensor per writing root: q, gate and
    # k | v columns of qkv_proj's output, q / k after norm and rope, attention).
    q_x = torch.empty((tokens, heads, head_dim), dtype=torch.bfloat16, device=dev)
    g_x = torch.empty((tokens, heads, head_dim), dtype=torch.bfloat16, device=dev)
    kv_x = torch.empty((tokens, 2, head_dim), dtype=torch.bfloat16, device=dev)
    q_rope = torch.empty((tokens, heads, head_dim), dtype=torch.bfloat16, device=dev)
    k_rope = torch.empty((tokens, head_dim), dtype=torch.bfloat16, device=dev)
    attn = torch.empty((tokens, heads, head_dim), dtype=torch.bfloat16, device=dev)

    # o_proj: K chunks of 2 heads.
    op_chunks = 4
    op_chunk_k = q_width // op_chunks
    attn_fp8 = torch.empty((tokens, q_width), dtype=torch.float8_e4m3fn, device=dev)
    attn_fp8_chunked = attn_fp8.view(tokens, op_chunks, op_chunk_k)
    op_weight_chunked = o_proj_weight.view(hidden, op_chunks, op_chunk_k)
    op_partials = torch.empty(
        (op_chunks, tokens, hidden), dtype=torch.float32, device=dev
    )

    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )
    output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=dev)
    residual_output = torch.empty_like(output)
    communication_n = 512
    norm_tasks = hidden // communication_n
    norm_partial = torch.empty((tokens, norm_tasks), dtype=torch.float32, device=dev)
    norm_input = torch.empty((tokens, hidden), dtype=torch.float32, device=dev)

    # Programmatic dependent launch: wait for the previous kernel at entry.
    hl.pdl_wait()
    # q rows of each head: FP8 K chunks of 128 rows.
    for iq_h, iq_n, iq_c, iq_t in hl.tile(
        [heads, head_dim, ip_chunks, tokens], block_size=[1, 64, 1, None]
    ):
        iq_acc = hl.zeros([iq_n, iq_t], dtype=torch.float32)
        for iq_k in hl.tile(chunk_k):
            iq_acc = hl.dot(
                qg_weight[iq_h.begin, 0, iq_n, iq_c.begin, iq_k],
                torch.clamp(
                    hidden_chunked[iq_t, iq_c.begin, iq_k].to(torch.float32)
                    * (1.0 / qkv_scales[0]),
                    -FP8_MAX,
                    FP8_MAX,
                )
                .to(torch.float8_e4m3fn)
                .T,
                acc=iq_acc,
            )
        q_parts[iq_c.begin, iq_t, iq_h.begin, iq_n] = iq_acc.T

    # k | v rows.
    for ik_s, ik_n, ik_c, ik_t in hl.tile(
        [2, head_dim, ip_chunks, tokens], block_size=[1, 128, 1, None]
    ):
        ik_lanes = hl.arange(lanes_n)
        ik_live = ik_lanes < tokens
        ik_lane = torch.where(ik_live, ik_lanes, 0)
        ik_rot = hl.arange(head_dim) % 32
        ik_pos = hl.load(
            positions, [hl.load(axis_map, [ik_rot])[None, :], ik_lane[:, None]]
        )
        hl.store(
            rope_cs,
            [0, ik_lane[:, None], hl.arange(head_dim)[None, :]],
            hl.load(cos_sin_cache, [ik_pos, ik_rot[None, :]]),
            extra_mask=ik_live[:, None],
        )
        hl.store(
            rope_cs,
            [1, ik_lane[:, None], hl.arange(head_dim)[None, :]],
            hl.load(cos_sin_cache, [ik_pos, (ik_rot + 32)[None, :]]),
            extra_mask=ik_live[:, None],
        )
        ik_acc = hl.zeros([ik_n, ik_t], dtype=torch.float32)
        for ik_k in hl.tile(chunk_k):
            ik_acc = hl.dot(
                kv_weight[ik_s.begin, ik_n, ik_c.begin, ik_k],
                torch.clamp(
                    hidden_chunked[ik_t, ik_c.begin, ik_k].to(torch.float32)
                    * (1.0 / qkv_scales[0]),
                    -FP8_MAX,
                    FP8_MAX,
                )
                .to(torch.float8_e4m3fn)
                .T,
                acc=ik_acc,
            )
        kv_parts[ik_c.begin, ik_t, ik_s.begin, ik_n] = ik_acc.T

    # gate rows of each head (read by the merge only).
    for ig_h, ig_n, ig_c, ig_t in hl.tile(
        [heads, head_dim, ip_chunks, tokens], block_size=[1, 128, 1, None]
    ):
        ig_acc = hl.zeros([ig_n, ig_t], dtype=torch.float32)
        for ig_k in hl.tile(chunk_k):
            ig_acc = hl.dot(
                qg_weight[ig_h.begin, 1, ig_n, ig_c.begin, ig_k],
                torch.clamp(
                    hidden_chunked[ig_t, ig_c.begin, ig_k].to(torch.float32)
                    * (1.0 / qkv_scales[0]),
                    -FP8_MAX,
                    FP8_MAX,
                )
                .to(torch.float8_e4m3fn)
                .T,
                acc=ig_acc,
            )
        g_parts[ig_c.begin, ig_t, ig_h.begin, ig_n] = ig_acc.T

    # q prep per head: norm + rope (BF16), FP8 with scale 1.
    for qp_h, qp_t in hl.tile([heads, tokens], block_size=[1, lanes_n]):
        qp_head = qp_h.begin
        qp_w1 = hl.load(q_norm_weight, [hl.arange(head_dim)]).to(torch.float32) + 1.0
        qp_scale = qkv_scales[1]
        qp_cos, qp_sin = _rope_tables(
            cos_sin_cache, positions, axis_map, qp_t, tokens, head_dim
        )
        qp_x, qp_out = _norm_rope_head(
            q_parts, ip_chunks, qp_scale, qp_t, qp_head, qp_w1, qp_cos, qp_sin, head_dim
        )
        q_x[qp_t, qp_head, :] = qp_x.to(torch.bfloat16)
        q_rope[qp_t, qp_head, :] = qp_out.to(torch.bfloat16)
        q8[qp_t, qp_head, :] = qp_out.to(torch.float8_e4m3fn)

    # k prep (norm + rope, BF16) and v (BF16): the window's FP8 K / V rows
    # (* (1 / scale)).
    for kp_t in hl.tile(tokens, block_size=2):
        kp_w1 = hl.load(k_norm_weight, [hl.arange(head_dim)]).to(torch.float32) + 1.0
        kp_scale = qkv_scales[1]
        kp_kscale = kv_scales[0]
        kp_vscale = kv_scales[1]
        kp_rlane = torch.where(kp_t.index < tokens, kp_t.index, 0)
        kp_rcols = hl.arange(head_dim)
        kp_cos = hl.load(rope_cs, [0, kp_rlane[:, None], kp_rcols[None, :]])
        kp_sin = hl.load(rope_cs, [1, kp_rlane[:, None], kp_rcols[None, :]])
        kp_k, kp_v = _proj_kv(kv_parts, ip_chunks, kp_scale, kp_t)
        kp_x, kp_out = _norm_rope_x(kp_k, kp_w1, kp_cos, kp_sin, head_dim)
        kv_x[kp_t, 0, :] = kp_x.to(torch.bfloat16)
        kv_x[kp_t, 1, :] = kp_v.to(torch.bfloat16)
        k_rope[kp_t, :] = kp_out.to(torch.bfloat16)
        k_win[kp_t, :] = (kp_out * kp_kscale).to(torch.float8_e4m3fn)
        v_win[kp_t, :] = (kp_v * kp_vscale).to(torch.float8_e4m3fn)

    # Cache splits: split s attends pages [lo, hi) of the committed keys, at least 2
    # pages a split (fewer partials to merge at short prefixes). The first page is
    # loaded ahead of q, so the ring streams it before q is ready.
    for cs_s in hl.tile(splits, block_size=1):
        split = cs_s.begin
        cache_end = seq_lens[0] - tokens
        bmm1_log2 = kv_scales[2]
        out_scale = kv_scales[3] / 256.0
        q_tile = q8[:, :, :].reshape(rows_n, head_dim)
        blocks = (cache_end + page - 1) // page
        active = torch.maximum(
            torch.minimum(cache_end * 0 + splits, (blocks + 1) // 2), cache_end * 0 + 1
        )
        base = blocks // active
        extra = blocks - base * active
        lo = torch.where(
            split < active,
            split * base + torch.minimum(cache_end * 0 + split, extra),
            blocks,
        )
        hi = torch.where(
            split < active, lo + base + (split < extra).to(torch.int32), blocks
        )
        keys = hl.arange(page)
        run_max = hl.full([rows_n], float("-inf"), dtype=torch.float32)
        run_sum = hl.zeros([rows_n], dtype=torch.float32)
        acc = hl.zeros([rows_n, head_dim], dtype=torch.float32)
        for blk in hl.grid(lo, hi, 2):
            pg_a = page_table[0, blk]
            pg_b = page_table[0, torch.minimum(hi * 0 + blk + 1, hi - 1)]
            key_a = kv_pages[0, pg_a, :, :]
            key_b = kv_pages[0, pg_b, :, :]
            value_a = kv_pages[1, pg_a, :, :]
            value_b = kv_pages[1, pg_b, :, :]
            valid_a = (blk * page + keys < cache_end)[None, :]
            valid_b = ((blk * page + page + keys < cache_end) & (blk + 1 < hi))[None, :]
            run_max, run_sum, acc = _pair_online(
                q_tile,
                key_a,
                key_b,
                value_a,
                value_b,
                valid_a,
                valid_b,
                run_max,
                run_sum,
                acc,
                bmm1_log2,
            )
        seen = run_sum > 0.0
        att_ws[split, :, :] = (
            acc * (out_scale / torch.where(seen, run_sum, 1.0))[:, None]
        ).to(torch.bfloat16)
        lse_rows = hl.arange(rows_n)
        hl.store(
            lse_ws,
            [split, (lse_rows % heads) * lanes_n + lse_rows // heads],
            torch.where(
                seen,
                run_max + torch.log2(torch.where(seen, run_sum, 1.0)),
                float("-inf"),
            ),
        )

    # The window's K / V into their cache slots (after every cache reader: the splits; the merge's window reads k_win / v_win).
    for _wr_t in hl.tile(1, block_size=1):
        wr_lanes = hl.arange(lanes_n)
        wr_live = wr_lanes < tokens
        wr_lane = torch.where(wr_live, wr_lanes, 0)
        wr_slot = hl.load(cache_loc, [wr_lane], extra_mask=wr_live)
        wr_cols = hl.arange(head_dim)
        hl.store(
            kv_cache,
            [0, wr_slot[:, None], wr_cols[None, :]],
            hl.load(k_win, [wr_lanes[:, None], wr_cols[None, :]]),
            extra_mask=wr_live[:, None],
        )
        hl.store(
            kv_cache,
            [1, wr_slot[:, None], wr_cols[None, :]],
            hl.load(v_win, [wr_lanes[:, None], wr_cols[None, :]]),
            extra_mask=wr_live[:, None],
        )

    # Merge, per (head, 64 columns): the window's partial, then the active splits'
    # (those past the committed pages hold nothing) in split order; the output
    # gate; o_proj's static FP8 input.
    for mg_j, mg_h, mg_cb, mg_t in hl.tile(
        [op_chunks, 2, head_dim, lanes_n], block_size=[1, 1, 32, lanes_n]
    ):
        mg_head = mg_j.begin * 2 + mg_h.begin
        mg_cols = mg_cb.begin + hl.arange(32)
        mg_lanes = mg_t.index
        mg_live = mg_lanes < tokens
        mg_lane = torch.where(mg_live, mg_lanes, 0)
        mg_gscale = qkv_scales[1]
        mg_oscale = 1.0 / o_scales[0]
        # The gate (its GEMM long done): loaded ahead of the partials' wait.
        mg_gate = hl.zeros([lanes_n, 32], dtype=torch.float32)
        for mg_c in hl.static_range(ip_chunks):
            mg_gate = mg_gate + hl.load(
                g_parts,
                [
                    mg_c,
                    mg_lane[:, None],
                    (mg_head + mg_lane * 0)[:, None],
                    mg_cols[None, :],
                ],
                extra_mask=mg_live[:, None],
            )
        mg_gate16 = (mg_gate * mg_gscale).to(torch.bfloat16)
        # The window's own keys (causal: token i sees keys j <= i) for this head's
        # query rows, ahead of the splits' wait: scores, then P V on this task's 64
        # columns over the 8 key lanes (lanes past the window selected away), the
        # BF16 partial and lse a window split would store.
        wm_bmm1 = kv_scales[2]
        wm_oscale = kv_scales[3] / 256.0
        wm_keys = hl.arange(lanes_n)
        wm_valid = (wm_keys[None, :] <= mg_lane[:, None]) & (wm_keys[None, :] < tokens)
        wm_scores = hl.dot(q8[mg_t, mg_head, :], k_win[:, :].T, out_dtype=torch.float32)
        wm_scores = torch.where(wm_valid, wm_scores * wm_bmm1, float("-inf"))
        wm_max = torch.amax(wm_scores, dim=1)
        wm_prob = torch.exp2(wm_scores - wm_max[:, None])
        wm_p8 = (wm_prob * 256.0).to(torch.float8_e4m3fn).to(torch.float32)
        wm_acc = torch.sum(
            torch.where(
                wm_valid[:, :, None],
                wm_p8[:, :, None] * v_win[:, mg_cb].to(torch.float32)[None, :, :],
                0.0,
            ),
            dim=1,
        )
        wm_sum = torch.sum(wm_prob, dim=1)
        mg_new = (
            (wm_acc * (wm_oscale / wm_sum)[:, None])
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        mg_lse_new = wm_max + torch.log2(wm_sum)
        mg_cache_end = seq_lens[0] - tokens
        # The splits' active count (0 with no committed keys).
        mg_active = torch.minimum(
            splits + mg_cache_end * 0, ((mg_cache_end + page - 1) // page + 1) // 2
        )
        # Every group's lse up front (loads issued together), the window's: their max.
        mg_ids = hl.arange(merge_n)
        mg_ok0 = 0 * merge_n + mg_ids < mg_active
        mg_l0 = torch.where(
            mg_ok0[:, None],
            hl.load(
                lse_ws,
                [
                    (0 * merge_n + mg_ids)[:, None],
                    (mg_head * lanes_n + mg_lanes)[None, :],
                ],
                extra_mask=mg_ok0[:, None],
            ),
            float("-inf"),
        )
        mg_ok1 = 1 * merge_n + mg_ids < mg_active
        mg_l1 = torch.where(
            mg_ok1[:, None],
            hl.load(
                lse_ws,
                [
                    (1 * merge_n + mg_ids)[:, None],
                    (mg_head * lanes_n + mg_lanes)[None, :],
                ],
                extra_mask=mg_ok1[:, None],
            ),
            float("-inf"),
        )
        mg_ok2 = 2 * merge_n + mg_ids < mg_active
        mg_l2 = torch.where(
            mg_ok2[:, None],
            hl.load(
                lse_ws,
                [
                    (2 * merge_n + mg_ids)[:, None],
                    (mg_head * lanes_n + mg_lanes)[None, :],
                ],
                extra_mask=mg_ok2[:, None],
            ),
            float("-inf"),
        )
        mg_ok3 = 3 * merge_n + mg_ids < mg_active
        mg_l3 = torch.where(
            mg_ok3[:, None],
            hl.load(
                lse_ws,
                [
                    (3 * merge_n + mg_ids)[:, None],
                    (mg_head * lanes_n + mg_lanes)[None, :],
                ],
                extra_mask=mg_ok3[:, None],
            ),
            float("-inf"),
        )
        mg_m = mg_l0
        mg_m = torch.maximum(mg_m, mg_l1)
        mg_m = torch.maximum(mg_m, mg_l2)
        mg_m = torch.maximum(mg_m, mg_l3)
        mg_max = torch.maximum(mg_lse_new, torch.amax(mg_m, dim=0))
        mg_den = torch.exp2(mg_lse_new - mg_max)
        mg_num = mg_den[:, None] * mg_new
        # Groups of 16 splits: [split, token, cols] boxes. Splits past the active ones
        # hold anything (NaN included): selected away, not multiplied by 0.
        mg_den3 = hl.zeros([merge_n, lanes_n], dtype=torch.float32)
        mg_num3 = hl.zeros([merge_n, lanes_n, 32], dtype=torch.float32)
        for mg_g in hl.grid((mg_active + merge_n - 1) // merge_n):
            mg_use = mg_g * merge_n + mg_ids < mg_active
            mg_sel = (mg_ids * 0 + mg_g)[:, None]
            mg_lse = mg_l0
            mg_lse = torch.where(mg_sel == 1, mg_l1, mg_lse)
            mg_lse = torch.where(mg_sel == 2, mg_l2, mg_lse)
            mg_lse = torch.where(mg_sel == 3, mg_l3, mg_lse)
            mg_w = torch.where(
                mg_use[:, None], torch.exp2(mg_lse - mg_max[None, :]), 0.0
            )
            mg_part = att_grp[mg_g, :, :, mg_head, mg_cb].to(torch.float32)
            mg_den3 = mg_den3 + mg_w
            mg_num3 = mg_num3 + torch.where(
                mg_use[:, None, None], mg_w[:, :, None] * mg_part, 0.0
            )
        mg_den = mg_den + torch.sum(mg_den3, dim=0)
        mg_num = mg_num + torch.sum(mg_num3, dim=0)
        mg_attn = (mg_num * (1.0 / mg_den)[:, None]).to(torch.bfloat16)
        hl.store(
            attn,
            [mg_lane[:, None], (mg_head + mg_lane * 0)[:, None], mg_cols[None, :]],
            mg_attn,
            extra_mask=mg_live[:, None],
        )
        mg_gated = (
            mg_attn.to(torch.float32) * torch.sigmoid(mg_gate16.to(torch.float32))
        ).to(torch.bfloat16)
        hl.store(
            attn_fp8_chunked,
            [
                mg_lane[:, None],
                (mg_j.begin + mg_lane * 0)[:, None],
                (mg_h.begin * head_dim + mg_cols)[None, :],
            ],
            torch.clamp(mg_gated.to(torch.float32) * mg_oscale, -FP8_MAX, FP8_MAX).to(
                torch.float8_e4m3fn
            ),
            extra_mask=mg_live[:, None],
        )
        hl.store(
            g_x,
            [mg_lane[:, None], (mg_head + mg_lane * 0)[:, None], mg_cols[None, :]],
            mg_gate16,
            extra_mask=mg_live[:, None],
        )

    # o_proj: FP8 K chunks (2 heads) of 128 output rows.
    for op_tile_n, op_tile_c, op_tile_t in hl.tile(
        [hidden, op_chunks, tokens], block_size=[128, 1, None]
    ):
        op_acc = hl.zeros([op_tile_n, op_tile_t], dtype=torch.float32)
        for op_tile_k in hl.tile(op_chunk_k):
            op_acc = hl.dot(
                op_weight_chunked[op_tile_n, op_tile_c.begin, op_tile_k],
                attn_fp8_chunked[op_tile_t, op_tile_c.begin, op_tile_k].T,
                acc=op_acc,
            )
        op_partials[op_tile_c.begin, op_tile_t, op_tile_n] = op_acc.T

    # This rank's o_proj output, the GEMM's BF16 result, into the symmetric partial.
    for pb_tile_t, pb_tile_n in hl.tile(
        [tokens, hidden], block_size=[1, communication_n]
    ):
        pb_token = pb_tile_t.begin
        pb_scale = o_scales[1]
        pb_sum = torch.sum(op_partials[:, pb_token, pb_tile_n], dim=0)
        symmetric_partial[pb_token, pb_tile_n] = (pb_sum * pb_scale).to(torch.bfloat16)

    # The next kernel may launch once this one reaches its all-reduce.
    hl.pdl_launch_dependents()

    # Each token's sum over ranks (in rank order), then the residual.
    for comm_tile_t, comm_tile_n in hl.tile(
        [tokens, hidden], block_size=[1, communication_n]
    ):
        comm_token = comm_tile_t.begin
        comm_residual = residual[comm_token, comm_tile_n].to(torch.float32)
        comm_total = hl.zeros([comm_tile_n], dtype=torch.float32)
        for remote_partial in remote_partials:
            comm_total = comm_total + remote_partial[comm_token, comm_tile_n].to(
                torch.float32
            )
        comm_sum = comm_total + comm_residual
        residual_output[comm_token, comm_tile_n] = comm_sum.to(torch.bfloat16)
        comm_sum = comm_sum.to(torch.bfloat16).to(torch.float32)
        norm_partial[comm_token, comm_tile_n.begin // communication_n] = torch.sum(
            comm_sum * comm_sum
        )
        norm_input[comm_token, comm_tile_n] = comm_sum
    for norm_tile_t, norm_tile_n in hl.tile(
        [tokens, hidden], block_size=[1, communication_n]
    ):
        norm_token = norm_tile_t.begin
        norm_w = norm_weight[norm_tile_n].to(torch.float32)
        norm_scale = torch.rsqrt(
            torch.sum(norm_partial[norm_token, :]) / hidden + NORM_EPS
        )
        output[norm_token, norm_tile_n] = (
            norm_input[norm_token, norm_tile_n] * norm_scale * norm_w
        ).to(torch.bfloat16)

    return output, residual_output, q_x, g_x, kv_x, q_rope, k_rope, attn, attn_fp8
