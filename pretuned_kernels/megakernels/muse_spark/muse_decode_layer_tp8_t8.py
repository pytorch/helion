"""The Muse Spark verify decoder layer as one Helion megakernel, TP8 design (any tensor-parallel world, served at TP8
with T = 8): attention, the O-projection and its all-reduce, the attention boundary, the expert input, the expert-TP
NVFP4 MoE, the unpooler shard and its all-gather, the FFN boundary and the next layer's QKV, one launch. Its device
helpers and host-side shapes come from _decode_layer (attention .. expert input, where this design's attention
splits use ``_cache_tiles_nr0``: the last split owns the new rows; and the MoE .. next QKV), as the TP4 design's
do, and the body reads their names unqualified. Every per-rank shape comes from the operands and ``world``.

The PF roots (EPF2) prefetch into L2 the bank of the first ``pf_hot_experts(world)`` = 7 * world / 4 experts of
pf_list (14 at TP8: 47.25 MiB per rank), in three waves of up to PF_SLOTS 32 KiB boxes per worker. pf_list is the
layer's hot-expert list: the host owns it per layer (all -1 = nothing to prefetch), and FIN rewrites it with the last
token's 8 experts for the layer's next launch.

``expert_bf16`` and ``final_norm`` select the images of the checkpoint's edge layers (whose experts stay bf16; the
model's last layer ends at the final norm), and ``entry`` the model's prelude before layer 0, as in
muse_decode_layer_tp4_t8.py; the bf16 images issue no expert prefetch.

Formatting is off in this file: Helion's generated code quotes the kernel's source lines with their line numbers, so
reflowing the body would change every generated module.
"""

# The body reads the helper modules' names unqualified (F821) and keeps the served source as written.
# ruff: noqa: ANN001, ANN201, ANN202, B905, F821, F841, I001, Q000
# fmt: off
from __future__ import annotations

import helion.language as hl
import torch
import torch.distributed._symmetric_memory as symm_mem  # noqa: F401

# The layer helpers (attention .. expert input, and the MoE .. next QKV).
from pretuned_kernels.megakernels.muse_spark import _decode_layer as _A

# The body reads the helpers' names unqualified.
globals().update({k: v for k, v in vars(_A).items() if not k.startswith("__")})
PF_NW, PF_SLOTS = 144, 4  # the PF roots' workers and boxes per worker and wave


def _layer_megakernel(qkv: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, cache_loc: torch.Tensor, block_tables: torch.Tensor, seq_lens: torch.Tensor, att_ws: torch.Tensor, lse_ws: torch.Tensor, out: torch.Tensor, wo: torch.Tensor, o_loc: torch.Tensor, opart: torch.Tensor, stream: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, pre_w: torch.Tensor, branch: torch.Tensor, moments: torch.Tensor, rstd: torch.Tensor, new_stream: torch.Tensor, act: torch.Tensor, wp: torch.Tensor, wr_hi: torch.Tensor, wr_lo: torch.Tensor, q_pre_w: torch.Tensor, router_bias: torch.Tensor, part_p: torch.Tensor, part_r: torch.Tensor, pooled: torch.Tensor, logit_q: torch.Tensor, x_q: torch.Tensor, x_sf: torch.Tensor, x_ts: torch.Tensor, packed: torch.Tensor, topk_ids: torch.Tensor, weights_g: torch.Tensor, weights2: torch.Tensor, scales: torch.Tensor, up_alpha: torch.Tensor, gate_alpha: torch.Tensor, out_alpha: torch.Tensor, z: torch.Tensor, z_scale: torch.Tensor, gemm2_ws: torch.Tensor, sumsq_ws: torch.Tensor, norm_weight: torch.Tensor, y16: torch.Tensor, wu4: torch.Tensor, u_part: torch.Tensor, talpha: torch.Tensor, tbeta: torch.Tensor, tpost_w: torch.Tensor, tpre_w: torch.Tensor, scaled_rows: torch.Tensor, tmoments: torch.Tensor, trstd: torch.Tensor, qact: torch.Tensor, wq4: torch.Tensor, q_part: torch.Tensor, qkv_g: torch.Tensor, new_stream_g: torch.Tensor, page_size: hl.constexpr, k_page_stride: hl.constexpr, v_page_stride: hl.constexpr, scaled_log2: hl.constexpr, eps: hl.constexpr, use_rope: hl.constexpr, window_left: hl.constexpr, key_tile: hl.constexpr, group_name: hl.ProcessGroupName, rank: hl.constexpr, world: hl.constexpr, post_eps: hl.constexpr, pre_eps: hl.constexpr, quant_eps: hl.constexpr, global_scale_inv: hl.constexpr, router_multiplier: hl.constexpr, route_scale: hl.constexpr, route_eps: hl.constexpr, norm_eps: hl.constexpr, pf_w13: torch.Tensor, pf_w2: torch.Tensor, pf_sf: torch.Tensor, pf_list: torch.Tensor, expert_bf16: hl.constexpr, final_norm: hl.constexpr, entry: hl.constexpr):
    hl.specialize(qkv.size())
    hl.specialize(qkv.stride())
    hl.specialize(positions.size())
    hl.specialize(cos_sin_cache.size())
    hl.specialize(cos_sin_cache.stride())
    hl.specialize(k_cache.size())
    hl.specialize(k_cache.stride())
    hl.specialize(v_cache.size())
    hl.specialize(v_cache.stride())
    hl.specialize(cache_loc.size())
    hl.specialize(block_tables.size())
    hl.specialize(block_tables.stride())
    hl.specialize(seq_lens.size())
    hl.specialize(att_ws.size())
    hl.specialize(att_ws.stride())
    hl.specialize(lse_ws.size())
    hl.specialize(lse_ws.stride())
    hl.specialize(out.size())
    hl.specialize(out.stride())
    hl.specialize(wo.size())
    hl.specialize(wo.stride())
    hl.specialize(o_loc.size())
    hl.specialize(o_loc.stride())
    hl.specialize(opart.size())
    hl.specialize(opart.stride())
    hl.specialize(stream.size())
    hl.specialize(stream.stride())
    hl.specialize(alpha.size())
    hl.specialize(alpha.stride())
    hl.specialize(beta.size())
    hl.specialize(beta.stride())
    hl.specialize(pre_w.size())
    hl.specialize(pre_w.stride())
    hl.specialize(branch.size())
    hl.specialize(branch.stride())
    hl.specialize(moments.size())
    hl.specialize(moments.stride())
    hl.specialize(rstd.size())
    hl.specialize(rstd.stride())
    hl.specialize(new_stream.size())
    hl.specialize(new_stream.stride())
    hl.specialize(act.size())
    hl.specialize(act.stride())
    hl.specialize(wp.size())
    hl.specialize(wp.stride())
    hl.specialize(wr_hi.size())
    hl.specialize(wr_hi.stride())
    hl.specialize(wr_lo.size())
    hl.specialize(wr_lo.stride())
    hl.specialize(q_pre_w.size())
    hl.specialize(q_pre_w.stride())
    hl.specialize(router_bias.size())
    hl.specialize(router_bias.stride())
    hl.specialize(part_p.size())
    hl.specialize(part_p.stride())
    hl.specialize(part_r.size())
    hl.specialize(part_r.stride())
    hl.specialize(pooled.size())
    hl.specialize(pooled.stride())
    hl.specialize(logit_q.size())
    hl.specialize(logit_q.stride())
    hl.specialize(x_q.size())
    hl.specialize(x_q.stride())
    hl.specialize(x_sf.size())
    hl.specialize(x_sf.stride())
    hl.specialize(x_ts.size())
    hl.specialize(x_ts.stride())
    hl.specialize(packed.size())
    hl.specialize(packed.stride())
    hl.specialize(topk_ids.size())
    hl.specialize(topk_ids.stride())
    num_kv_heads, ws_splits, tokens, heads_per_kv, head_dim = att_ws.size()
    num_splits = ws_splits // 2
    att_h = heads_per_kv // 2
    att_rows = tokens * att_h
    att_v = att_ws.view(num_kv_heads, 2, num_splits, tokens, heads_per_kv, head_dim)
    lse_v = lse_ws.view(num_kv_heads, 2, num_splits, tokens, heads_per_kv)
    qkv_width = qkv.size(1)
    kv_width = num_kv_heads * head_dim
    q_width = num_kv_heads * heads_per_kv * head_dim
    page_shift = page_size.bit_length() - 1
    qkv_flat = qkv.view(-1)
    table_flat = block_tables.view(-1)
    cos_sin_flat = cos_sin_cache.view(-1)
    out_flat = out.view(-1)
    group_width = heads_per_kv * head_dim
    mma_rows = out.size(0)
    out3 = out.view(mma_rows, num_kv_heads, group_width)
    hidden = o_loc.size(1)
    o_rows = o_loc
    o_split = o_loc.view(o_loc.size(0), opart.size(1), opart.size(2))
    opart_f = opart.view(opart.size(0), opart.size(1) * opart.size(2))
    bm, bk = (wo.size(2), wo.size(3))
    o_tasks, o_steps = (wo.size(0), wo.size(1))
    out4 = out.view(mma_rows, num_kv_heads, group_width // bk, bk)
    out_heads = out.view(mma_rows, num_kv_heads, heads_per_kv, head_dim)
    full_hidden = branch.size(1) * branch.size(2)
    remote_opart = torch.ops.symm_mem.get_remote_tensors(opart, group_name)
    remote_pooled = torch.ops.symm_mem.get_remote_tensors(pooled, group_name)
    remote_logit = torch.ops.symm_mem.get_remote_tensors(logit_q, group_name)
    s_p = wp.size(1)
    k_p = wp.size(3)
    num_experts = wr_lo.size(1)
    topk = packed.size(1)
    p_rows = wp.size(2)
    n_local = wp.size(0) * p_rows
    quant_lanes = q_pre_w.size(0) // 16
    q_local = n_local // 16
    x = act
    act2 = act.view(2, tokens, s_p, k_p)
    part_p5 = part_p
    part_p4 = part_p.view(s_p, n_local, 2, tokens)
    part_r4 = part_r.view(part_r.size(0), 2, tokens, part_r.size(2))
    s_own = s_p // world
    kp_blk = hl.register_block_size(k_p)
    c_blk = hl.register_block_size(n_local)
    r1_kblk = hl.register_block_size(k_p)
    hl.specialize(weights_g.size())
    hl.specialize(weights_g.stride())
    hl.specialize(weights2.size())
    hl.specialize(weights2.stride())
    hl.specialize(scales.size())
    hl.specialize(scales.stride())
    hl.specialize(up_alpha.size())
    hl.specialize(up_alpha.stride())
    hl.specialize(gate_alpha.size())
    hl.specialize(gate_alpha.stride())
    hl.specialize(out_alpha.size())
    hl.specialize(out_alpha.stride())
    hl.specialize(z.size())
    hl.specialize(z.stride())
    hl.specialize(z_scale.size())
    hl.specialize(z_scale.stride())
    hl.specialize(gemm2_ws.size())
    hl.specialize(gemm2_ws.stride())
    hl.specialize(sumsq_ws.size())
    hl.specialize(sumsq_ws.stride())
    hl.specialize(norm_weight.size())
    hl.specialize(norm_weight.stride())
    hl.specialize(y16.size())
    hl.specialize(y16.stride())
    hl.specialize(wu4.size())
    hl.specialize(wu4.stride())
    hl.specialize(u_part.size())
    hl.specialize(u_part.stride())
    hl.specialize(talpha.size())
    hl.specialize(talpha.stride())
    hl.specialize(tbeta.size())
    hl.specialize(tbeta.stride())
    hl.specialize(tpost_w.size())
    hl.specialize(tpost_w.stride())
    hl.specialize(tpre_w.size())
    hl.specialize(tpre_w.stride())
    hl.specialize(scaled_rows.size())
    hl.specialize(scaled_rows.stride())
    hl.specialize(tmoments.size())
    hl.specialize(tmoments.stride())
    hl.specialize(trstd.size())
    hl.specialize(trstd.stride())
    hl.specialize(qact.size())
    hl.specialize(qact.stride())
    hl.specialize(wq4.size())
    hl.specialize(wq4.stride())
    hl.specialize(q_part.size())
    hl.specialize(q_part.stride())
    hl.specialize(qkv_g.size())
    hl.specialize(qkv_g.stride())
    hl.specialize(new_stream_g.size())
    hl.specialize(new_stream_g.stride())
    routes = packed.view(-1)
    tstream = new_stream.view(new_stream.size(0), world, scaled_rows.size(1), scaled_rows.size(2))
    hidden_g = y16.size(1)
    local = gemm2_ws.size(1)
    topk_g = packed.size(1)
    slots = routes.size(0)
    tokens_g = packed.size(0)
    rows_g = x_q.size(0)
    # bf16 experts: weights_g is FI's BlockMajorK W13 as [E, H / 64, 2I / W / 128, 128, 64]; a K step is two blocks
    k_steps = weights_g.size(1) // 2 if expert_bf16 else weights_g.size(1)
    q_tiles = weights_g.size(2) // 2 if expert_bf16 else z.size(2) * 4 // world // 256
    h2_tiles = sumsq_ws.size(1)
    h_tiles = h2_tiles // 2
    k_pairs = weights2.size(1)
    w13_tiles = up_alpha.size(0) * q_tiles
    remote_z = torch.ops.symm_mem.get_remote_tensors(z, group_name)
    remote_z_scale = torch.ops.symm_mem.get_remote_tensors(z_scale, group_name)
    remote_g2 = torch.ops.symm_mem.get_remote_tensors(gemm2_ws, group_name)
    remote_sumsq = torch.ops.symm_mem.get_remote_tensors(sumsq_ws, group_name)
    y3 = y16.view(y16.size(0), world, local)
    x3 = x_q.view(1, rows_g, x_q.size(1))
    weights5 = weights_g
    xs3 = x_sf.view(1, rows_g, x_sf.size(1))
    e4m3 = torch.float8_e4m3fn
    tokens_g = qkv_g.size(0)
    n_blocks = scaled_rows.size(1)
    c_block = scaled_rows.size(2)
    full_hidden_g = world * n_blocks * c_block
    u_rows, u_kb = (wu4.size(2), wu4.size(3))
    q_rows, q_kb = (wq4.size(2), wq4.size(3))
    u_len = wu4.size(1) * u_kb // u_part.size(0)
    q_len = wq4.size(1) * q_kb // q_part.size(0)
    yu3 = y16.view(y16.size(0), y16.size(1) // u_kb, u_kb)
    act3 = qact.view(qact.size(0), full_hidden_g // q_kb, q_kb)
    u_part4 = u_part.view(u_part.size(0), u_part.size(1), wu4.size(0), u_rows)
    u_part_c = u_part.view(u_part.size(0), u_part.size(1), n_blocks, c_block)
    q_part4 = q_part.view(q_part.size(0), q_part.size(1), wq4.size(0), q_rows)
    q_part5 = q_part.view(q_part.size(0), 2, tokens_g, wq4.size(0), q_rows)
    qkv3 = qkv_g.view(tokens_g, wq4.size(0), q_rows)
    remote_rows = torch.ops.symm_mem.get_remote_tensors(scaled_rows, group_name)
    remote_moments = torch.ops.symm_mem.get_remote_tensors(tmoments, group_name)
    nblk = 256 // c_block
    stream_f = stream.view(stream.size(0), stream.size(1) * stream.size(2))  # entry: this rank's masked embedding rows
    branch_e = branch.view(branch.size(0), world, n_blocks, c_block)  # entry: the all-reduced embedding rows
    wq_pq = wq4.permute(0, 2, 1, 3).reshape(wq4.size(0), q_rows, q_part.size(0), 4, q_len // 4)
    pq_tasks = wq4.size(0) * q_part.size(0)
    hl.specialize(pf_list.size())
    hl.specialize(pf_w13.size())
    hl.specialize(pf_w13.stride())
    hl.specialize(pf_w2.size())
    hl.specialize(pf_w2.stride())
    hl.specialize(pf_sf.size())
    hl.specialize(pf_sf.stride())
    pf_bw13 = pf_w13.size(1)
    pf_bw2 = pf_w2.size(1)
    pf_bs13 = pf_bw13 // 8
    pf_bs2 = pf_bw2 // 8
    pf_s2_base = pf_w13.size(0) * pf_bs13
    pf_ha = pf_bw13 // 2
    pf_hb = pf_bw13 - pf_ha + pf_bs13
    pf_hc = pf_bw2 + pf_bs2
    hl.prefetch_code()
    hl.pdl_wait()
    for a_g, a_s, a_h in hl.tile([num_kv_heads, num_splits, heads_per_kv], block_size=[1, 1, att_h]):
        if not entry:
            g = a_g.begin
            split = a_s.begin
            hh = a_h.begin
            seq_len = seq_lens[0]
            cache_end = seq_len - tokens
            first, lo, hi = _cache_tiles_nr0(cache_end, split, key_tile, num_splits, window_left)
            rows = hl.arange(att_rows)
            cols = hl.arange(head_dim)
            keys = hl.arange(key_tile)
            q_tok = rows // att_h
            q_head = g * heads_per_kv + hh + rows % att_h
            q_idx = (q_tok * qkv_width + q_head * head_dim)[:, None] + cols[None, :]
            q_raw = qkv_flat[q_idx]
            if use_rope:
                half = hl.arange(head_dim // 2)[None, :]
                q_pos = positions[q_tok][:, None] * head_dim
            tile0 = first + lo * key_tile
            page0 = hl.load(table_flat, [(tile0 >> page_shift) + hl.arange(1)], extra_mask=lo + hl.arange(1) < hi)
            if use_rope:
                q_cos = cos_sin_flat[q_pos + half]
                q_sin = cos_sin_flat[q_pos + head_dim // 2 + half]
            else:
                q_cos = hl.zeros([att_rows, head_dim // 2], dtype=torch.float32)
                q_sin = hl.zeros([att_rows, head_dim // 2], dtype=torch.float32)
            token0 = tile0 + keys
            in_page0 = (((tile0 & page_size - 1) + keys) * kv_width + g * head_dim)[:, None] + cols[None, :]
            in_cache0 = (token0 < cache_end) & (lo + keys * 0 < hi)
            key0 = hl.load(k_cache, [page0.to(torch.int64)[:, None] * k_page_stride + in_page0], extra_mask=in_cache0[:, None])
            value0 = hl.load(v_cache, [page0.to(torch.int64)[:, None] * v_page_stride + in_page0], extra_mask=in_cache0[:, None])
            tile1 = tile0 + key_tile
            page1 = hl.load(table_flat, [(tile1 >> page_shift) + hl.arange(1)], extra_mask=lo + 1 + hl.arange(1) < hi)
            token1 = tile1 + keys
            in_page1 = (((tile1 & page_size - 1) + keys) * kv_width + g * head_dim)[:, None] + cols[None, :]
            in_cache1 = (token1 < cache_end) & (lo + 1 + keys * 0 < hi)
            key1 = hl.load(k_cache, [page1.to(torch.int64)[:, None] * k_page_stride + in_page1], extra_mask=in_cache1[:, None])
            value1 = hl.load(v_cache, [page1.to(torch.int64)[:, None] * v_page_stride + in_page1], extra_mask=in_cache1[:, None])
            q_tile = _qk_prep(q_raw, q_cos, q_sin, att_rows, head_dim, eps, use_rope).to(torch.float8_e4m3fn)
            new_tok = hl.arange(tokens)
            last_t = new_tok * 0 + split == num_splits - 1
            k_idx = (new_tok * qkv_width + q_width + g * head_dim)[:, None] + cols[None, :]
            if use_rope:
                k_half = hl.arange(head_dim // 2)[None, :]
                k_pos = hl.load(positions, [new_tok], extra_mask=last_t)[:, None] * head_dim
                k_cos = hl.load(cos_sin_flat, [k_pos + k_half], extra_mask=last_t[:, None])
                k_sin = hl.load(cos_sin_flat, [k_pos + head_dim // 2 + k_half], extra_mask=last_t[:, None])
            else:
                k_cos = hl.zeros([tokens, head_dim // 2], dtype=torch.float32)
                k_sin = hl.zeros([tokens, head_dim // 2], dtype=torch.float32)
            k_new = _qk_prep(hl.load(qkv_flat, [k_idx], extra_mask=last_t[:, None]), k_cos, k_sin, tokens, head_dim, eps, use_rope).to(torch.float8_e4m3fn)
            v_new = _fp8_clamp(hl.load(qkv_flat, [k_idx + kv_width], extra_mask=last_t[:, None]).to(torch.float32)).to(torch.float8_e4m3fn)
            loc = hl.load(cache_loc, [new_tok], extra_mask=last_t)
            live = loc != 0
            dead = (loc == 0) & last_t
            store_new = live & (hh + new_tok * 0 == 0) & last_t
            new_pos = cache_end + new_tok
            new_page = hl.load(table_flat, [new_pos >> page_shift], extra_mask=dead).to(torch.int64)
            new_in_page = ((new_pos & page_size - 1) * kv_width + g * head_dim)[:, None] + cols[None, :]
            old_k = hl.load(k_cache, [new_page[:, None] * k_page_stride + new_in_page], extra_mask=dead[:, None])
            old_v = hl.load(v_cache, [new_page[:, None] * v_page_stride + new_in_page], extra_mask=dead[:, None])
            loc_in_page = (loc % page_size * kv_width + g * head_dim)[:, None] + cols[None, :]
            hl.store(k_cache, [(loc // page_size)[:, None] * k_page_stride + loc_in_page], k_new, extra_mask=store_new[:, None])
            hl.store(v_cache, [(loc // page_size)[:, None] * v_page_stride + loc_in_page], v_new, extra_mask=store_new[:, None])
            new_key = torch.where(live[:, None], k_new.to(torch.float32), old_k.to(torch.float32))
            new_value = torch.where(live[:, None], v_new.to(torch.float16), old_v.to(torch.float16))
            run_max = hl.full([att_rows], -1e+30, dtype=torch.float32)
            run_sum = hl.zeros([att_rows], dtype=torch.float32)
            acc = hl.zeros([att_rows, head_dim], dtype=torch.float32)
            q_last = cache_end + q_tok
            valid0 = in_cache0[None, :] & (q_last[:, None] >= token0[None, :])
            valid1 = in_cache1[None, :] & (q_last[:, None] >= token1[None, :])
            if window_left >= 0:
                valid0 = valid0 & (q_last[:, None] - token0[None, :] <= window_left)
                valid1 = valid1 & (q_last[:, None] - token1[None, :] <= window_left)
            run_max, run_sum, acc = _softmax_step(q_tile, key0, _e4m3_to_f16(value0), valid0, run_max, run_sum, acc, scaled_log2)
            if lo + 1 < hi:
                run_max, run_sum, acc = _softmax_step(q_tile, key1, _e4m3_to_f16(value1), valid1, run_max, run_sum, acc, scaled_log2)
            for tile in hl.grid(lo + 2, hi):
                tile_begin = first + tile * key_tile
                token = tile_begin + keys
                page = hl.load(table_flat, [(tile_begin >> page_shift) + hl.arange(1)])
                in_page = (((tile_begin & page_size - 1) + keys) * kv_width + g * head_dim)[:, None] + cols[None, :]
                in_cache = token < cache_end
                key = hl.load(k_cache, [page.to(torch.int64)[:, None] * k_page_stride + in_page], extra_mask=in_cache[:, None])
                value = hl.load(v_cache, [page.to(torch.int64)[:, None] * v_page_stride + in_page], extra_mask=in_cache[:, None])
                valid = in_cache[None, :] & (q_last[:, None] >= token[None, :])
                if window_left >= 0:
                    valid = valid & (q_last[:, None] - token[None, :] <= window_left)
                run_max, run_sum, acc = _softmax_step(q_tile, key, _e4m3_to_f16(value), valid, run_max, run_sum, acc, scaled_log2)
            if split == num_splits - 1:
                run_max, run_sum, acc = _softmax_step(q_tile, new_key.to(torch.float8_e4m3fn), new_value, q_tok[:, None] >= new_tok[None, :], run_max, run_sum, acc, scaled_log2)
            seen = run_sum > 0.0
            att_v[a_g, 0, a_s, :, a_h, :] = (acc / torch.where(seen, run_sum, 1.0)[:, None]).reshape(1, 1, tokens, att_h, head_dim)
            lse_v[a_g, 0, a_s, :, a_h] = torch.where(seen, run_max + torch.log2(run_sum), float('-inf')).reshape(1, 1, tokens, att_h)
    for m_g, m_t, m_h in hl.tile([num_kv_heads, tokens, heads_per_kv], block_size=[1, 1, 4]):
        if not entry:
            mg = m_g.begin
            mt = m_t.begin
            m_cols = hl.arange(head_dim)
            heads = mg * heads_per_kv + m_h.index
            head_idx = (heads * head_dim)[:, None] + m_cols[None, :]
            gate = qkv_flat[mt * qkv_width + q_width + 2 * kv_width + head_idx]
            gate_factor = rcp_approx_ftz_f32(1.0 + exp2_approx_ftz_f32(gate.to(torch.float32) * -LOG2E))
            m_att = att_v[mg, 0, :, mt, m_h, :]
            m_lse = lse_v[mg, 0, :, mt, m_h]
            weight = exp2_approx_ftz_f32(m_lse - torch.amax(m_lse, dim=0)[None, :])
            merged = torch.sum(m_att * weight[:, :, None], dim=0)
            attn = _cvt_rn_satfinite_bf16(merged / torch.sum(weight, dim=0)[:, None]).to(torch.float32)
            variance = torch.sum(attn * attn, dim=-1) / head_dim
            normed = attn * rsqrt_approx_ftz_f32(variance + eps)[:, None]
            out_heads[m_t, m_g, m_h, :] = (normed * gate_factor).to(torch.bfloat16)[None, None, :, :]
    for o_t, o_n in hl.tile([tokens, hidden], block_size=[tokens, bm]):
        if entry:  # the embedding all-reduce's exchange: this rank's masked rows to its peers (as O's partials)
            opart_f[o_t, o_n] = stream_f[o_t, o_n]
        else:
            o_acc = hl.zeros([o_n, mma_rows], dtype=torch.float32)
            for o_k in hl.tile(q_width, block_size=bk):
                w_tile = hl.load(wo, [o_n.id + hl.zeros([], dtype=torch.int32), o_k.id + hl.zeros([], dtype=torch.int32), slice(None), slice(None)])
                o_grp = o_k.begin // group_width
                o_half = o_k.begin % group_width // bk
                o_x = out4[:, o_grp, o_half, :]
                o_acc = hl.dot(w_tile, o_x.permute(1, 0), acc=o_acc)
            o_real, o_pad = hl.split(o_acc.reshape(bm, mma_rows // tokens, tokens).permute(0, 2, 1))
            opart_f[o_t, o_n] = o_real.permute(1, 0).to(torch.bfloat16)
    for ar_t, ar_s in hl.tile([tokens, s_p], block_size=[2, 1]):
        if entry:  # the embedding all-reduce (a token's row is one rank's, so the sum is exact)
            total_e = hl.zeros([ar_t, k_p], dtype=torch.float32)
            for peer_part in remote_opart:
                total_e = total_e + peer_part[ar_t, ar_s.begin, :].to(torch.float32)
            branch[ar_t, ar_s.begin, :] = total_e.to(torch.bfloat16)
        else:
            gated_a = alpha[ar_s.begin, :][None, :] * stream[ar_t, ar_s.begin, :]
            beta_a = beta[ar_s.begin, :][None, :]
            pre = pre_w[ar_s.begin, :].to(torch.float32)[None, :]
            total = hl.zeros([ar_t, k_p], dtype=torch.float32)
            for peer_part in remote_opart:
                total = total + peer_part[ar_t, ar_s.begin, :].to(torch.float32)
            br_a = total.to(torch.bfloat16)
            branch[ar_t, ar_s.begin, :] = br_a
            brf = br_a.to(torch.float32)
            gb = beta_a * brf
            act2[0, ar_t, ar_s.begin, :] = (gated_a * pre).to(torch.bfloat16)
            act2[1, ar_t, ar_s.begin, :] = (gb * pre).to(torch.bfloat16)
            m4 = hl.arange(4)[None, :]
            moments[ar_s.begin, ar_t, :] = torch.where(m4 == 0, torch.sum(brf * brf, dim=1)[:, None], 0.0) + torch.where(m4 == 1, torch.sum(gated_a * gated_a, dim=1)[:, None], 0.0) + torch.where(m4 == 2, torch.sum(gated_a * gb, dim=1)[:, None], 0.0) + torch.where(m4 == 3, torch.sum(gb * gb, dim=1)[:, None], 0.0)
    for mm_t in hl.tile(tokens, block_size=1):
        if not entry:
            total_m = torch.sum(moments[:, mm_t.begin, :], dim=0)
            mm = hl.arange(4)
            post_sq = torch.sum(torch.where(mm == 0, total_m, 0.0))
            stream_sq = torch.sum(torch.where(mm == 1, total_m, 0.0))
            cross = torch.sum(torch.where(mm == 2, total_m, 0.0))
            branch_sq = torch.sum(torch.where(mm == 3, total_m, 0.0))
            post_rstd = torch.rsqrt(post_sq / full_hidden + post_eps)
            pre_var = (stream_sq + 2.0 * post_rstd * cross + post_rstd * post_rstd * branch_sq) / full_hidden
            pre_rstd = torch.rsqrt(pre_var + pre_eps)
            rstd[mm_t, :] = torch.where(hl.arange(2) == 0, post_rstd, pre_rstd)[None, :]
    for r1_a, r1_b in hl.tile([s_own, num_experts // 64], block_size=[1, 1]):
        if not entry:
            acc_r = hl.zeros([128, 2 * tokens], dtype=torch.float32)
            for r1_k in hl.tile(k_p, block_size=r1_kblk):
                x_r = act[:, rank * s_own + r1_a.begin, r1_k].permute(1, 0)
                acc_r = hl.dot(wr_hi[r1_a.begin, r1_b.begin, :, r1_k], x_r, acc=acc_r)
            r_hi, r_lo = hl.split(acc_r.reshape(2, 64, 2 * tokens).permute(1, 2, 0))
            part_r[r1_a.begin, :, r1_b.begin * 64 + hl.arange(64)] = (r_hi + r_lo).permute(1, 0)
    for p1_b, p1_s in hl.tile([wp.size(0), s_p], block_size=[1, 1]):
        if not entry:
            acc = hl.zeros([p_rows, 2 * tokens], dtype=torch.float32)
            for p1_k in hl.tile(k_p, block_size=kp_blk):
                acc = hl.dot(wp[p1_b.begin, p1_s.begin, :, p1_k], x[:, p1_s.begin, p1_k].permute(1, 0), acc=acc)
            part_p5[p1_s.begin, p1_b.begin, :, :, :] = acc.reshape(p_rows, 2, tokens)
    for p2_c in hl.tile(n_local, block_size=c_blk):
        if not entry:
            post_r = rstd[:, 0]
            pre_r = rstd[:, 1]
            tot_a, tot_b = hl.split(torch.sum(part_p4[:, p2_c, :, :], dim=0).permute(0, 2, 1))
            pooled[:, p2_c] = (pre_r[:, None] * (tot_a.permute(1, 0) + post_r[:, None] * tot_b.permute(1, 0))).to(torch.bfloat16)
    for r2_e in hl.tile(num_experts, block_size=64):
        if not entry:
            pre_e = rstd[:, 1]
            post_e = rstd[:, 0]
            la = torch.sum(part_r4[:, 0, :, r2_e], dim=0)
            lb = torch.sum(part_r4[:, 1, :, r2_e], dim=0)
            logit_q[:, r2_e] = pre_e[:, None] * (la + post_e[:, None] * lb)
    for r3_t in hl.tile(tokens, block_size=1):
        if not entry:
            experts = hl.arange(num_experts)
            bias = router_bias[experts]
            logits = hl.zeros([num_experts], dtype=torch.float32)
            for peer_logit in remote_logit:
                logits = logits + peer_logit[r3_t.begin, experts]
            weights, sel_ids = _route(logits, bias, experts, num_experts, router_multiplier, route_scale, route_eps)
            bits = weights.to(torch.bfloat16).view(dtype=torch.int16).to(torch.int32)
            packed[r3_t.begin, hl.arange(topk)] = sel_ids << 16 | bits & 65535
            topk_ids[r3_t.begin, hl.arange(topk)] = sel_ids
    for q_t in hl.tile(tokens, block_size=1):
        if not entry:
            lanes = hl.arange(quant_lanes)
            elem = lanes[:, None] * 16 + hl.arange(16)[None, :]
            norm_w = q_pre_w[elem].to(torch.float32)
            quarters = hl.zeros([world, q_local, 16], dtype=torch.float32)
            owner_q = hl.arange(world)[:, None, None]
            for peer, peer_pooled in enumerate(remote_pooled):
                row = peer_pooled[q_t.begin, hl.arange(n_local)].to(torch.float32)
                quarters = torch.where(owner_q == peer, row.reshape(1, q_local, 16), quarters)
            xv = quarters.reshape(quant_lanes, 16)
            sum_lane = _chain_sq(_columns(xv, 16))
            warp_sums = _butterfly(sum_lane.reshape(quant_lanes // 32, 32), 32)
            sum_sq = _butterfly(warp_sums, quant_lanes // 32)
            rs = rsqrt_approx_ftz_f32(sum_sq * (1.0 / q_pre_w.size(0)) + quant_eps)
            normed_q = (xv * rs * norm_w).to(torch.bfloat16).to(torch.float32)
            if expert_bf16:
                x_q[q_t.begin, elem] = normed_q.to(torch.bfloat16)
            else:
                block_amax = hl.reduce(_max, torch.abs(normed_q), dim=1)
                token_scale = hl.reduce(_max, block_amax, dim=0) * global_scale_inv
                fp8_scale, out_scale = encode_block_scale(token_scale + hl.zeros([quant_lanes], dtype=torch.float32), block_amax)
                low, high = hl.split((normed_q * out_scale[:, None]).reshape(quant_lanes, 8, 2))
                fp4_bytes = pack_e2m1x2(low, high).to(torch.uint8)
                x_q[q_t.begin, lanes[:, None] * 8 + hl.arange(8)[None, :]] = fp4_bytes
                x_sf[q_t.begin, lanes] = fp8_scale.to(torch.uint8)
                x_ts[q_t.begin + hl.arange(1)] = token_scale + hl.zeros([1], dtype=torch.float32)
    for r_t, r_s in hl.tile([tokens, s_p], block_size=[2, 1]):
        if not entry:
            gated_r = alpha[r_s.begin, :][None, :] * stream[r_t, r_s.begin, :]
            beta_r = beta[r_s.begin, :][None, :]
            br_r = branch[r_t, r_s.begin, :].to(torch.float32)
            post_rs = rstd[r_t, 0]
            normed = (br_r * post_rs[:, None]).to(torch.bfloat16).to(torch.float32)
            new_stream[r_t, r_s.begin, :] = gated_r + beta_r * normed
    for pfa_t in hl.tile(144, block_size=1):
        if not entry:
            if not expert_bf16:
                pfa_hot = pf_list[:]
                for pfa_i in hl.static_range(4):
                    pfa_b = pfa_t.begin + pfa_i * 144
                    if pfa_b < 7 * world // 4 * pf_ha:
                        pfa_e = torch.sum(torch.where(hl.arange(16) == pfa_b // pf_ha, pfa_hot, 0))
                        if pfa_e >= 0:
                            hl.prefetch(pf_w13, [pfa_e, pfa_b % pf_ha, slice(None), slice(None)])
    for pfb_t in hl.tile(144, block_size=1):
        if not entry:
            if not expert_bf16:
                pfb_hot = pf_list[:]
                for pfb_i in hl.static_range(4):
                    pfb_b = pfb_t.begin + pfb_i * 144
                    if pfb_b < 7 * world // 4 * pf_hb:
                        pfb_e = torch.sum(torch.where(hl.arange(16) == pfb_b // pf_hb, pfb_hot, 0))
                        pfb_c = pfb_b % pf_hb
                        if pfb_e >= 0:
                            if pfb_c < pf_bw13 - pf_ha:
                                hl.prefetch(pf_w13, [pfb_e, pf_ha + pfb_c, slice(None), slice(None)])
                            else:
                                hl.prefetch(pf_sf, [pfb_e * pf_bs13 + (pfb_c - (pf_bw13 - pf_ha)), slice(None), slice(None)])
    for pfc_t in hl.tile(144, block_size=1):
        if not entry:
            if not expert_bf16:
                pfc_hot = pf_list[:]
                for pfc_i in hl.static_range(4):
                    pfc_b = pfc_t.begin + pfc_i * 144
                    if pfc_b < 7 * world // 4 * pf_hc:
                        pfc_e = torch.sum(torch.where(hl.arange(16) == pfc_b // pf_hc, pfc_hot, 0))
                        pfc_c = pfc_b % pf_hc
                        if pfc_e >= 0:
                            if pfc_c < pf_bw2:
                                hl.prefetch(pf_w2, [pfc_e, pfc_c, slice(None), slice(None)])
                            else:
                                hl.prefetch(pf_sf, [pf_s2_base + pfc_e * pf_bs2 + (pfc_c - pf_bw2), slice(None), slice(None)])
    for w13_tile_q, w13_tile_j in hl.tile([2 * q_tiles, slots], block_size=[1, 1]):
        if not entry:
            words, counts, before, distinct = _distinct_table(routes[:] >> 16)
            if w13_tile_j.begin < distinct:
                expert = _expert_of(words, counts, before, w13_tile_j.begin)
                routed = _routed_rows(routes, expert, rows_g, topk_g, slots)
                q2 = w13_tile_q.begin // 2
                fi = w13_tile_q.begin % 2
                wt = expert * q_tiles + q2
                acc_g = hl.zeros([128, rows_g], dtype=torch.float32)
                row_scale = x_ts[hl.arange(rows_g)]
                ua = up_alpha[expert]
                ga = gate_alpha[expert]
                rot = 3 * (w13_tile_j.begin * q_tiles + q2)
                if expert_bf16:
                    for w13_tile_k in hl.tile(k_steps, block_size=1):
                        kk = (w13_tile_k.begin + rot) % k_steps
                        lhs = hl.load(weights5, [expert, kk * 2 + hl.arange(2), w13_tile_q.begin, slice(None), slice(None)])
                        lhs = lhs.permute(1, 0, 2).reshape(128, 2 * weights5.size(4))
                        rhs = x3[0, :, kk * 128 + hl.arange(128)]
                        acc_g = hl.dot(lhs, rhs.permute(1, 0), acc=acc_g)
                else:
                    for w13_tile_k in hl.tile(k_steps, block_size=1):
                        kk = (w13_tile_k.begin + rot) % k_steps
                        lhs = hl.load(weights5, [wt * 2 + fi, kk, slice(None), slice(None), slice(None)])
                        lhs = lhs.permute(1, 0, 2).reshape(128, 2 * weights5.size(4))
                        lhs_scale = _scale_tile128(hl.load(scales, [wt, fi, kk * 16 + hl.arange(16), slice(None)]))
                        rhs = x3[0, :, kk * 256 + hl.arange(256)]
                        rhs_scale = xs3[0, :, kk * 32 + hl.arange(32)].view(e4m3)
                        acc_g = hl.dot_scaled(lhs, lhs_scale, 'e2m1', rhs.permute(1, 0), rhs_scale, 'e2m1', acc=acc_g, out_dtype=torch.float32)
                act_g = _swiglu128(acc_g, row_scale, ua, ga)
                if expert_bf16:
                    col = (rank * 2 * q_tiles + w13_tile_q.begin) * 64
                    for peer_z in remote_z:
                        hl.store(peer_z, [w13_tile_j, slice(None), col + hl.arange(64)], act_g.to(torch.bfloat16)[None, :, :], extra_mask=routed[None, :, None])
                else:
                    zq, zsc = _quant_z(act_g)
                    col = (rank * 2 * q_tiles + w13_tile_q.begin) * 32
                    scol = (rank * 2 * q_tiles + w13_tile_q.begin) * 4
                    for peer_z in remote_z:
                        hl.store(peer_z, [w13_tile_j, slice(None), col + hl.arange(32)], zq[None, :, :], extra_mask=routed[None, :, None])
                    for peer_zs in remote_z_scale:
                        hl.store(peer_zs, [w13_tile_j, slice(None), scol + hl.arange(4)], zsc[None, :, :], extra_mask=routed[None, :, None])
    for w2_tile_h, w2_tile_f in hl.tile([h2_tiles, slots], block_size=[1, 1]):
        if not entry:
            words2, counts2, before2, distinct2 = _distinct_table(routes[:] >> 16)
            if w2_tile_f.begin < distinct2:
                expert2 = _expert_of(words2, counts2, before2, w2_tile_f.begin)
                wt2 = expert2 * h2_tiles + w2_tile_h.begin
                st2 = w13_tiles + expert2 * h_tiles + w2_tile_h.begin // 2
                half2 = w2_tile_h.begin % 2
                acc2 = hl.zeros([128, rows_g], dtype=torch.float32)
                oa = out_alpha[expert2]
                slot_id = hl.arange(rows_g)[:, None] * topk_g + hl.arange(topk_g)[None, :]
                live_g = slot_id < slots
                ids = hl.load(routes, [slot_id], extra_mask=live_g) >> 16
                match = ((ids == expert2) & live_g).to(torch.int32)
                route_row = torch.sum(match * slot_id, dim=-1)
                routed = torch.sum(match, dim=-1) > 0
                cols_g = w2_tile_h.begin * 128 + hl.arange(128)
                nw = norm_weight[rank * local + cols_g].to(torch.float32)
                rot2 = 3 * (w2_tile_f.begin * h2_tiles + w2_tile_h.begin)
                if expert_bf16:
                    for w2_tile_k in hl.tile(2 * k_pairs, block_size=1):
                        jj = (w2_tile_k.begin // 2 + rot2) % k_pairs
                        jh = w2_tile_k.begin % 2
                        lhs2 = hl.load(weights2, [wt2, jj, jh, slice(None), slice(None)])
                        rhs2 = z[w2_tile_f.begin, :, jj * 256 + jh * 128 + hl.arange(128)]
                        acc2 = hl.dot(lhs2, rhs2.permute(1, 0), acc=acc2)
                else:
                    for w2_tile_k in hl.tile(2 * k_pairs, block_size=1):
                        jj = (w2_tile_k.begin // 2 + rot2) % k_pairs
                        jh = w2_tile_k.begin % 2
                        lhs2 = hl.load(weights2, [wt2, jj, jh, slice(None), slice(None)])
                        lhs2_scale = _scale_tile128(hl.load(scales, [st2, half2, jj * 32 + jh * 16 + hl.arange(16), slice(None)]))
                        rhs2 = z[w2_tile_f.begin, :, jj * 512 + jh * 256 + hl.arange(256)]
                        rhs2_scale = z_scale[w2_tile_f.begin, :, jj * 64 + jh * 32 + hl.arange(32)].view(e4m3)
                        acc2 = hl.dot_scaled(lhs2, lhs2_scale, 'e2m1', rhs2.permute(1, 0), rhs2_scale, 'e2m1', acc=acc2, out_dtype=torch.float32)
                val = (_logical_rows128(acc2) * oa).permute(1, 0).to(torch.bfloat16)
                hl.store(gemm2_ws, [route_row, cols_g], val, extra_mask=routed[:, None])
                gm = val.to(torch.float32) * nw[None, :]
                hl.store(sumsq_ws, [route_row, w2_tile_h.begin], torch.sum(gm * gm, dim=-1), extra_mask=routed)
    # A FIN task reads every rank's partial of its columns: 512 / world columns keep that load at W4's (and W8's
    # 128-column tiles spilled 72 bytes of stack).
    for fin_tile_s, fin_tile_c in hl.tile([slots, local], block_size=[topk_g, 512 // world]):
        if not entry:
            if not expert_bf16:
                if fin_tile_c.begin == 0:  # history for the layer's next launch: the last token's 8 experts (its own tile)
                    if fin_tile_s.begin == slots - 8:
                        pf_list[hl.arange(8)] = routes[fin_tile_s] >> 16
            bits_g = (routes[fin_tile_s] & 65535).to(torch.int16)
            weight_g = bits_g.view(torch.bfloat16).to(torch.float32)
            nws = [norm_weight[r * local + fin_tile_c.index] for r in hl.static_range(world)]
            ss = hl.zeros([fin_tile_s], dtype=torch.float32)
            for peer_sumsq in remote_sumsq:
                ss = ss + torch.sum(peer_sumsq[fin_tile_s, :], dim=1)
            scale = torch.rsqrt(ss / hidden_g + norm_eps) * weight_g
            parts = [peer_g2[fin_tile_s, fin_tile_c] for peer_g2 in remote_g2]
            for r in hl.static_range(world):
                nw_r = nws[r].to(torch.float32)
                y3[fin_tile_s.id, r, fin_tile_c] = torch.sum(parts[r].to(torch.float32) * nw_r[None, :] * scale[:, None], dim=0).to(torch.bfloat16)
    for pq_t in hl.tile(152, block_size=1):
        if not final_norm:
            if pq_t.begin < pq_tasks:
                hl.prefetch(wq_pq, [pq_t.begin // q_part.size(0), slice(None), pq_t.begin % q_part.size(0), 1, slice(None)])
                hl.prefetch(wq_pq, [pq_t.begin // q_part.size(0), slice(None), pq_t.begin % q_part.size(0), 2, slice(None)])
    for u_b, u_s in hl.tile([wu4.size(0), u_part.size(0)], block_size=[1, 1]):
        if not entry:
            acc_u = hl.zeros([u_rows, y16.size(0)], dtype=torch.float32)
            for u_k in hl.tile(u_len, block_size=u_kb):
                uk = (u_s.begin * u_len + u_k.begin) // u_kb
                w_u = hl.load(wu4, [u_b.id + hl.zeros([], dtype=torch.int32), uk + hl.zeros([], dtype=torch.int32), slice(None), slice(None)])
                acc_u = hl.dot(w_u, yu3[:, uk, :].permute(1, 0), acc=acc_u)
            u_part4[u_s.begin, :, u_b.begin, :] = acc_u.permute(1, 0)
    for uc_t, uc_b in hl.tile([tokens_g, n_blocks], block_size=[tokens_g, 1]):
        if not entry:
            post = tpost_w[rank, uc_b.begin, :].to(torch.float32)
            gated_stream = talpha[rank, uc_b.begin, :][None, :] * tstream[uc_t, rank, uc_b.begin, :]
            beta_c = tbeta[rank, uc_b.begin, :]
            unpooled = torch.sum(u_part_c[:, uc_t, uc_b.begin, :], dim=0).to(torch.bfloat16)
            scaled = (unpooled.to(torch.float32) * post[None, :]).to(torch.bfloat16)
            scaled_rows[uc_t, uc_b.begin, :] = scaled
            scaled_f = scaled.to(torch.float32)
            gated_branch = beta_c[None, :] * scaled_f
            m4_g = hl.arange(4)[None, :]
            tmoments[uc_b.begin, uc_t, :] = torch.where(m4_g == 0, torch.sum(scaled_f * scaled_f, dim=-1)[:, None], 0.0) + torch.where(m4_g == 1, torch.sum(gated_stream * gated_stream, dim=-1)[:, None], 0.0) + torch.where(m4_g == 2, torch.sum(gated_stream * gated_branch, dim=-1)[:, None], 0.0) + torch.where(m4_g == 3, torch.sum(gated_branch * gated_branch, dim=-1)[:, None], 0.0)
    for m_t_g in hl.tile(tokens_g, block_size=1):
        if not entry:
            total_g = hl.zeros([4], dtype=torch.float32)
            for peer_moments in remote_moments:
                total_g = total_g + torch.sum(peer_moments[:, m_t_g.begin, :], dim=0)
            mm_g = hl.arange(4)
            post_sq_g = torch.sum(torch.where(mm_g == 0, total_g, 0.0))
            stream_sq_g = torch.sum(torch.where(mm_g == 1, total_g, 0.0))
            cross_g = torch.sum(torch.where(mm_g == 2, total_g, 0.0))
            branch_sq_g = torch.sum(torch.where(mm_g == 3, total_g, 0.0))
            post_rstd_g = torch.rsqrt(post_sq_g / full_hidden_g + post_eps)
            pre_var_g = (stream_sq_g + 2.0 * post_rstd_g * cross_g + post_rstd_g * post_rstd_g * branch_sq_g) / full_hidden_g
            pre_rstd_g = torch.rsqrt(torch.clamp_min(pre_var_g, 0.0) + pre_eps)
            trstd[m_t_g, :] = torch.where(hl.arange(2) == 0, post_rstd_g, pre_rstd_g)[None, :]
    for n_t, n_b in hl.tile([tokens_g, n_blocks], block_size=[1, nblk]):
        if entry:
            # The embedding's weightless RMSNorm (the stock embed_norm, rounded to bf16) is the stream; each task
            # recomputes its token's row rstd and the input norm's (identical in every task of the token). No branch:
            # QC's post_rstd slot is 0, so the unwritten qact[T:] drops out.
            x_e = branch_e[n_t.begin, :, :, :].to(torch.float32)
            r_e = torch.rsqrt(torch.sum(torch.sum(torch.sum(x_e * x_e, dim=2), dim=1), dim=0) / full_hidden_g + pre_eps)
            s_e = (x_e * r_e).to(torch.bfloat16).to(torch.float32)
            p_e = torch.rsqrt(torch.sum(torch.sum(torch.sum(s_e * s_e, dim=2), dim=1), dim=0) / full_hidden_g + pre_eps)
            trstd[n_t, :] = torch.where(hl.arange(2) == 0, 0.0, p_e)[None, :]
            s_n = (branch_e[n_t.begin, :, n_b, :].to(torch.float32) * r_e).to(torch.bfloat16).to(torch.float32)
            new_stream_g[n_t.begin, :, n_b, :] = s_n
            qact[n_t.begin, :, n_b, :] = (s_n * tpre_w[:, n_b, :].to(torch.float32)).to(torch.bfloat16)
        elif not final_norm:
            owner = hl.arange(world)[:, None, None]
            pre_g = tpre_w[:, n_b, :].to(torch.float32)
            gated_n = talpha[:, n_b, :] * tstream[n_t.begin, :, n_b, :]
            beta_n = tbeta[:, n_b, :]
            branch_n = hl.zeros([world, n_b, c_block], dtype=torch.float32)
            for peer_g, peer_rows in enumerate(remote_rows):
                branch_n = torch.where(owner == peer_g, peer_rows[n_t.begin, n_b, :].to(torch.float32)[None, :, :], branch_n)
            qact[n_t.begin, :, n_b, :] = (gated_n * pre_g).to(torch.bfloat16)
            qact[n_t.begin + tokens_g, :, n_b, :] = (beta_n * branch_n * pre_g).to(torch.bfloat16)
    for r_t_g, r_b in hl.tile([tokens_g, n_blocks], block_size=[1, nblk]):
        if not entry:
            owner_r = hl.arange(world)[:, None, None]
            gated_r_g = talpha[:, r_b, :] * tstream[r_t_g.begin, :, r_b, :]
            beta_r_g = tbeta[:, r_b, :]
            branch_r = hl.zeros([world, r_b, c_block], dtype=torch.float32)
            for peer_g, peer_rows in enumerate(remote_rows):
                branch_r = torch.where(owner_r == peer_g, peer_rows[r_t_g.begin, r_b, :].to(torch.float32)[None, :, :], branch_r)
            post_rs_g = trstd[r_t_g.begin, 0]
            normed_g = (branch_r * post_rs_g).to(torch.bfloat16).to(torch.float32)
            new_stream_g[r_t_g.begin, :, r_b, :] = gated_r_g + beta_r_g * normed_g
            if final_norm:  # the model's final norm (the boundary's pre-norm) of the bf16 new stream
                final_w = tpre_w[:, r_b, :].to(torch.float32)
                final_in = (gated_r_g + beta_r_g * normed_g).to(torch.bfloat16).to(torch.float32)
                qact[r_t_g.begin, :, r_b, :] = (final_in * trstd[r_t_g.begin, 1] * final_w).to(torch.bfloat16)
    # The next layer's launch overlaps this one's QKV / QC tail (its grid wait still follows this launch).
    hl.pdl_launch_dependents()
    for q_b, q_s in hl.tile([wq4.size(0), q_part.size(0)], block_size=[1, 1]):
        if not final_norm:
            acc_q = hl.zeros([q_rows, qact.size(0)], dtype=torch.float32)
            for q_k in hl.tile(q_len, block_size=q_kb):
                qk = (q_s.begin * q_len + q_k.begin) // q_kb
                w_q = hl.load(wq4, [q_b.id + hl.zeros([], dtype=torch.int32), qk + hl.zeros([], dtype=torch.int32), slice(None), slice(None)])
                acc_q = hl.dot(w_q, act3[:, qk, :].permute(1, 0), acc=acc_q)
            q_part4[q_s.begin, :, q_b.begin, :] = acc_q.permute(1, 0)
    for qc_b, qc_r in hl.tile([wq4.size(0), q_rows], block_size=[1, 32]):
        if not final_norm:
            post_r_g = trstd[:, 0]
            pre_r_g = trstd[:, 1]
            tot_a_g, tot_b_g = hl.split(torch.sum(q_part5[:, :, :, qc_b.begin, qc_r], dim=0).permute(1, 2, 0))
            qkv3[:, qc_b.begin, qc_r] = (pre_r_g[:, None] * (tot_a_g + post_r_g[:, None] * tot_b_g)).to(torch.bfloat16)
    return (packed, qkv_g)


KERNEL = _layer_megakernel


def pf_hot_experts(world: int) -> int:
    """How many experts of pf_list the PF roots prefetch (their bound reads ``7 * world // 4``)."""
    return 7 * world // 4


def _dense_rows(t, rows):
    """t's storage span as [rows, -1, 2048] (t: a permuted view of a dense region)."""
    span = 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))
    assert span == t.numel(), (t.shape, t.stride())
    return torch.as_strided(t, (t.numel(),), (1,)).view(rows, -1, 2048)


def _dense_boxes(t, rows):
    """t's storage span as [rows, -1, 128, 256]: one 32 KiB TMA box (128 x 256 B) per [e, c]."""
    flat = _dense_rows(t, rows)
    assert flat.size(1) % 16 == 0, flat.shape
    return flat.view(rows, -1, 128, 256)


def prefetch_views(weights_g, weights2, scales, experts, world):
    """The PF roots' operands (pf_w13, pf_w2, pf_sf): the bank's W13 / W2 regions as one 32 KiB box per
    [expert, box], and its block scales (W13 then W2, expert-major) as boxes."""
    w13 = _dense_boxes(weights_g, experts)
    w2 = _dense_boxes(weights2, experts)
    sf = _dense_rows(scales, 1)[0].reshape(-1, 128, 256)
    b13, b2 = w13.size(1), w2.size(1)
    assert sf.size(0) == experts * (b13 + b2) // 8, "scales: W13 then W2 block scales, expert-major"
    per_wave = max(b13 // 2, b13 - b13 // 2 + b13 // 8, b2 + b2 // 8)
    assert pf_hot_experts(world) * per_wave <= PF_SLOTS * PF_NW, "a wave exceeds PF_SLOTS boxes per task"
    return w13, w2, sf
