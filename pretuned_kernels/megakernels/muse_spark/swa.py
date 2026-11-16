"""Muse Spark spec-decode verify attention as one persistent Helion megakernel.

TP4 rank-local shapes: 32 Q heads, 4 KV heads, head_dim 64, FP8 E4M3 paged KV
with page size 128, one request of T = 8 or 16 query tokens. The megakernel
replaces the three launches sglang runs after the QKVG GEMM:

    FusedQKNorm (Triton)            per-head RMSNorm (+ interleaved RoPE on
                                    sliding-window layers), FP8 Q, FP8 K/V store
    FlashInfer trtllm decode        FP8 paged causal decode, q_len_per_req=T
    rms_norm_gate (CUDA)            per-head RMSNorm + sigmoid gate

Roots, in source order:

    ATTEND  [4 KV heads, splits]   each task recomputes its T x 8 Q rows,
                                   attends its 64-key cache tiles; the last
                                   split also attends and stores the T new
                                   K/V rows from registers
    MERGE   [4 KV heads, T]        merges every split of one (KV head, token)
                                   and applies norm + gate

MERGE reads every split of its KV head, so Helion's tile-dependency scheduler
orders it after them: no counters or atomics. Only the KV length, positions,
cache slots and page-table contents are dynamic; every shape is specialized.

``main()`` benchmarks the sliding-window layer (the served use) against the sglang
defaults and needs sglang (``muse-spark`` branch) and FlashInfer::

    PYTHONPATH=. python pretuned_kernels/megakernels/muse_spark/swa.py
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

from pretuned_kernels.megakernels.muse_spark._common import VERIFY_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import exp2_approx_ftz_f32
from pretuned_kernels.megakernels.muse_spark._common import rcp_approx_ftz_f32
from pretuned_kernels.megakernels.muse_spark._common import relative_rmse_and_cosine
from pretuned_kernels.megakernels.muse_spark._common import rsqrt_approx_ftz_f32
import torch

import helion
import helion.language as hl

NUM_Q_HEADS = 32

NUM_KV_HEADS = 4

HEAD_DIM = 64

HEADS_PER_KV = 8

PAGE_SIZE = 128

Q_WIDTH = NUM_Q_HEADS * HEAD_DIM

KV_WIDTH = NUM_KV_HEADS * HEAD_DIM

QKV_WIDTH = 2 * Q_WIDTH + 2 * KV_WIDTH

# Muse Spark defaults; the served model passes its own scale and eps.
BMM1_SCALE = 43.7840518911 / HEAD_DIM

EPS = 1e-5

KEY_TILE = 64
LOG2E = 1.4426950408889634


def _fp8_clamp(value: torch.Tensor) -> torch.Tensor:
    """Production Triton's ordered clamp tl.minimum(tl.maximum(v, -448), 448):
    NaN -> -448, which torch.clamp would propagate."""

    value = torch.where(value > -448.0, value, -448.0)
    return torch.where(value < 448.0, value, 448.0)


def _cvt_rn_satfinite_bf16(value: torch.Tensor) -> torch.Tensor:
    """TRTLLM-GEN's saturating FP32 -> BF16 epilogue conversion."""

    return hl.inline_asm_elementwise(
        "cvt.rn.satfinite.bf16.f32 $0, $1;",
        "=h,f",
        [value],
        dtype=torch.bfloat16,
        is_pure=True,
        pack=1,
    )


def _e4m3_to_f16(value: torch.Tensor) -> torch.Tensor:
    """Exact FP8 -> FP16."""
    return hl.inline_asm_elementwise(
        """
        {
            .reg .b16 lo, hi;
            mov.b32 { lo, hi }, $2;
            cvt.rn.f16x2.e4m3x2 $0, lo;
            cvt.rn.f16x2.e4m3x2 $1, hi;
        }
        """,
        "=r,=r,r",
        [value],
        dtype=torch.float16,
        is_pure=True,
        pack=4,
    )


def _qk_prep(
    raw: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rows: int,
    head_dim: int,
    eps: float,
    use_rope: bool,
) -> torch.Tensor:
    """FusedQKNorm FP8-path value for Q/K rows, before the FP8 cast."""

    raw_fp32 = raw.to(torch.float32)
    variance = torch.sum(raw_fp32 * raw_fp32, dim=-1, keepdim=True) / head_dim
    normalized = (raw_fp32 * torch.rsqrt(variance + eps)).to(torch.bfloat16)
    if use_rope:
        even, odd = hl.split(
            normalized.to(torch.float32).reshape(rows, head_dim // 2, 2)
        )
        values = hl.join(
            even * cos - odd * sin,
            even * sin + odd * cos,
        ).reshape(rows, head_dim)
        values = values.to(torch.bfloat16).to(torch.float32)
    else:
        values = normalized.to(torch.float32)
    return _fp8_clamp(values)


def _cache_tiles(
    cache_end: torch.Tensor,
    split: int,
    key_tile: int,
    num_splits: int,
    window_left: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(first, lo, hi): split's 64-key cache tiles, counted from the tile holding
    the first cached key the oldest query sees. Low splits take the extras."""

    if window_left < 0:
        first_key = cache_end * 0
    else:
        first_key = torch.clamp_min(cache_end - window_left, 0)
    first = first_key & -key_tile
    num_tiles = (cache_end - first + key_tile - 1) >> (key_tile.bit_length() - 1)
    base = num_tiles // num_splits
    extra = num_tiles - base * num_splits
    split = cache_end * 0 + split
    lo = split * base + torch.minimum(split, extra)
    return first, lo, lo + base + (split < extra).to(torch.int32)


def _softmax_step(
    q_tile: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor,
    run_max: torch.Tensor,
    run_sum: torch.Tensor,
    acc: torch.Tensor,
    scaled_log2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax update of [rows] state with FP8 keys and FP16 values."""

    scores = hl.dot(q_tile, key.transpose(-2, -1), out_dtype=torch.float32)
    scores = torch.where(valid, scores * scaled_log2, float("-inf"))
    tile_max = torch.amax(scores, dim=-1)
    new_max = torch.maximum(run_max, tile_max)
    correction = exp2_approx_ftz_f32(run_max - new_max)
    prob = exp2_approx_ftz_f32(scores - new_max[:, None])
    acc = hl.dot(
        prob.to(torch.float16),
        value,
        acc=acc * correction[:, None],
        out_dtype=torch.float32,
    )
    run_sum = run_sum * correction + torch.sum(prob, dim=-1)
    # A fresh node: returning new_max would clobber the carried name early.
    return torch.maximum(tile_max, run_max), run_sum, acc


def _muse_spark_swa_megakernel(
    qkv: torch.Tensor,  # [T, QKV_WIDTH] bf16, [q | k | v | gate]
    positions: torch.Tensor,  # [T] int64
    cos_sin_cache: torch.Tensor,  # [P, D] fp32, cos then sin halves
    k_cache: torch.Tensor,  # flat FP8 page storage
    v_cache: torch.Tensor,  # flat FP8 page storage
    cache_loc: torch.Tensor,  # [T] int64, 0 = padding
    block_tables: torch.Tensor,  # [B, pages] int32, row 0 is the request
    seq_lens: torch.Tensor,  # [B] int32, including the T new tokens
    att_ws: torch.Tensor,  # [Hkv, S, T, Hq / Hkv, D] fp32
    lse_ws: torch.Tensor,  # [Hkv, S, T, Hq / Hkv] fp32, log2 units
    out: torch.Tensor,  # [T, Hq * D] bf16
    page_size: hl.constexpr,
    k_page_stride: hl.constexpr,
    v_page_stride: hl.constexpr,
    scaled_log2: hl.constexpr,
    eps: hl.constexpr,
    use_rope: hl.constexpr,
    window_left: hl.constexpr,
    key_tile: hl.constexpr,
) -> torch.Tensor:
    """ATTEND: (kv_head, split) tasks store a partial per (token, head) row;
    MERGE: (kv_head, token) tasks merge the splits and apply norm + gate."""

    hl.specialize(qkv.size())
    hl.specialize(qkv.stride())
    hl.specialize(positions.size())
    hl.specialize(positions.stride())
    hl.specialize(cos_sin_cache.size())
    hl.specialize(cos_sin_cache.stride())
    hl.specialize(k_cache.size())
    hl.specialize(k_cache.stride())
    hl.specialize(v_cache.size())
    hl.specialize(v_cache.stride())
    hl.specialize(cache_loc.size())
    hl.specialize(cache_loc.stride())
    hl.specialize(block_tables.size())
    hl.specialize(block_tables.stride())
    hl.specialize(seq_lens.size())
    hl.specialize(seq_lens.stride())
    hl.specialize(att_ws.size())
    hl.specialize(att_ws.stride())
    hl.specialize(lse_ws.size())
    hl.specialize(lse_ws.stride())
    hl.specialize(out.size())
    hl.specialize(out.stride())

    num_kv_heads, num_splits, tokens, heads_per_kv, head_dim = att_ws.size()
    att_rows = tokens * heads_per_kv
    qkv_width = qkv.size(1)
    kv_width = num_kv_heads * head_dim
    q_width = num_kv_heads * heads_per_kv * head_dim
    page_shift = page_size.bit_length() - 1
    qkv_flat = qkv.view(-1)
    table_flat = block_tables.view(-1)
    cos_sin_flat = cos_sin_cache.view(-1)
    out_flat = out.view(-1)
    # Rows r = token * heads_per_kv + head.
    att_rows_ws = att_ws.view(num_kv_heads, num_splits, att_rows, head_dim)
    lse_rows_ws = lse_ws.view(num_kv_heads, num_splits, att_rows)

    hl.pdl_wait()
    # Trigger at entry: the next kernel's CTAs land on SMs this grid frees early,
    # and its own wait still orders it after this grid.
    hl.pdl_launch_dependents()
    # ATTEND. Stock: FusedQKNorm (Q/K norm, RoPE, FP8 Q, FP8 K/V store) and the
    # KV-split part of FI trtllm_batch_decode.
    for a_g, a_s in hl.tile([num_kv_heads, num_splits], block_size=[1, 1]):
        g = a_g.begin
        split = a_s.begin
        seq_len = seq_lens[0]
        # Cached keys end where the new tokens begin; those come from registers.
        cache_end = seq_len - tokens
        first, lo, hi = _cache_tiles(
            cache_end, split, key_tile, num_splits, window_left
        )
        rows = hl.arange(att_rows)
        cols = hl.arange(head_dim)
        q_tok = rows // heads_per_kv
        q_head = g * heads_per_kv + rows % heads_per_kv
        q_idx = (q_tok * qkv_width + q_head * head_dim)[:, None] + cols[None, :]
        if use_rope:
            half = hl.arange(head_dim // 2)[None, :]
            q_pos = positions[q_tok][:, None] * head_dim
            q_cos = cos_sin_flat[q_pos + half]
            q_sin = cos_sin_flat[q_pos + head_dim // 2 + half]
        else:
            q_cos = hl.zeros([att_rows, head_dim // 2], dtype=torch.float32)
            q_sin = hl.zeros([att_rows, head_dim // 2], dtype=torch.float32)
        q_tile = _qk_prep(
            qkv_flat[q_idx], q_cos, q_sin, att_rows, head_dim, eps, use_rope
        ).to(torch.float8_e4m3fn)

        run_max = hl.full([att_rows], -1e30, dtype=torch.float32)
        run_sum = hl.zeros([att_rows], dtype=torch.float32)
        acc = hl.zeros([att_rows, head_dim], dtype=torch.float32)
        q_last = cache_end + q_tok
        keys = hl.arange(key_tile)
        for tile in hl.grid(lo, hi):
            tile_begin = first + tile * key_tile
            token = tile_begin + keys
            page = hl.load(table_flat, [(tile_begin >> page_shift) + hl.arange(1)])
            in_page = (
                ((tile_begin & (page_size - 1)) + keys) * kv_width + g * head_dim
            )[:, None] + cols[None, :]
            # Rows past cache_end may be this launch's new rows: never read them.
            in_cache = token < cache_end
            key = hl.load(
                k_cache,
                [page.to(torch.int64)[:, None] * k_page_stride + in_page],
                extra_mask=in_cache[:, None],
            )
            value = hl.load(
                v_cache,
                [page.to(torch.int64)[:, None] * v_page_stride + in_page],
                extra_mask=in_cache[:, None],
            )
            valid = in_cache[None, :] & (q_last[:, None] >= token[None, :])
            if window_left >= 0:
                valid = valid & (q_last[:, None] - token[None, :] <= window_left)
            run_max, run_sum, acc = _softmax_step(
                q_tile,
                key,
                _e4m3_to_f16(value),
                valid,
                run_max,
                run_sum,
                acc,
                scaled_log2,
            )

        # The last split has the fewest cache tiles; it owns the new rows.
        if split == num_splits - 1:
            new_tok = hl.arange(tokens)
            k_idx = (new_tok * qkv_width + q_width + g * head_dim)[:, None] + cols[
                None, :
            ]
            if use_rope:
                k_half = hl.arange(head_dim // 2)[None, :]
                k_pos = positions[new_tok][:, None] * head_dim
                k_cos = cos_sin_flat[k_pos + k_half]
                k_sin = cos_sin_flat[k_pos + head_dim // 2 + k_half]
            else:
                k_cos = hl.zeros([tokens, head_dim // 2], dtype=torch.float32)
                k_sin = hl.zeros([tokens, head_dim // 2], dtype=torch.float32)
            k_new = _qk_prep(
                qkv_flat[k_idx], k_cos, k_sin, tokens, head_dim, eps, use_rope
            ).to(torch.float8_e4m3fn)
            v_new = _fp8_clamp(qkv_flat[k_idx + kv_width].to(torch.float32)).to(
                torch.float8_e4m3fn
            )
            loc = cache_loc[new_tok]
            live = loc != 0
            dead = loc == 0
            # Slot 0 is the padding sentinel: nothing is written, so a padded
            # token attends the cache row its position already maps to.
            new_pos = cache_end + new_tok
            new_page = hl.load(table_flat, [new_pos >> page_shift], extra_mask=dead).to(
                torch.int64
            )
            new_in_page = ((new_pos & (page_size - 1)) * kv_width + g * head_dim)[
                :, None
            ] + cols[None, :]
            old_k = hl.load(
                k_cache,
                [new_page[:, None] * k_page_stride + new_in_page],
                extra_mask=dead[:, None],
            )
            old_v = hl.load(
                v_cache,
                [new_page[:, None] * v_page_stride + new_in_page],
                extra_mask=dead[:, None],
            )
            loc_in_page = ((loc % page_size) * kv_width + g * head_dim)[:, None] + cols[
                None, :
            ]
            hl.store(
                k_cache,
                [(loc // page_size)[:, None] * k_page_stride + loc_in_page],
                k_new,
                extra_mask=live[:, None],
            )
            hl.store(
                v_cache,
                [(loc // page_size)[:, None] * v_page_stride + loc_in_page],
                v_new,
                extra_mask=live[:, None],
            )
            # Assumes cache_loc[t] is the slot of position cache_end + t.
            new_key = torch.where(
                live[:, None], k_new.to(torch.float32), old_k.to(torch.float32)
            )
            new_value = torch.where(
                live[:, None], v_new.to(torch.float16), old_v.to(torch.float16)
            )
            run_max, run_sum, acc = _softmax_step(
                q_tile,
                new_key.to(torch.float8_e4m3fn),
                new_value,
                q_tok[:, None] >= new_tok[None, :],
                run_max,
                run_sum,
                acc,
                scaled_log2,
            )

        # A split with no visible key publishes weight 0: ex2(-inf) is +0.
        seen = run_sum > 0.0
        att_rows_ws[g, split, :, :] = acc / torch.where(seen, run_sum, 1.0)[:, None]
        lse_rows_ws[g, split, :] = torch.where(
            seen, run_max + torch.log2(run_sum), float("-inf")
        )

    # MERGE reads every split of its KV head, so it runs after all of them.
    # MERGE. Stock: trtllm_batch_decode's split merge, then rms_norm_gate.
    for m_g, m_t in hl.tile([num_kv_heads, tokens], block_size=[1, 1]):
        mg = m_g.begin
        mt = m_t.begin
        m_cols = hl.arange(head_dim)
        heads = mg * heads_per_kv + hl.arange(heads_per_kv)
        head_idx = (heads * head_dim)[:, None] + m_cols[None, :]
        gate = qkv_flat[mt * qkv_width + q_width + 2 * kv_width + head_idx]
        gate_factor = rcp_approx_ftz_f32(
            1.0 + exp2_approx_ftz_f32(gate.to(torch.float32) * -LOG2E)
        )
        m_lse = lse_ws[mg, :, mt, :]
        weight = exp2_approx_ftz_f32(m_lse - torch.amax(m_lse, dim=0)[None, :])
        merged = torch.sum(att_ws[mg, :, mt, :, :] * weight[:, :, None], dim=0)
        attn = _cvt_rn_satfinite_bf16(merged / torch.sum(weight, dim=0)[:, None]).to(
            torch.float32
        )
        variance = torch.sum(attn * attn, dim=-1) / head_dim
        normed = attn * rsqrt_approx_ftz_f32(variance + eps)[:, None]
        out_flat[mt * q_width + head_idx] = (normed * gate_factor).to(torch.bfloat16)

    return out


muse_spark_swa = helion.kernel(
    _muse_spark_swa_megakernel,
    # Tile dependencies need persistent_blocked; no root has a tunable block.
    config=helion.Config(
        block_sizes=[],
        cross_loop_pipeline="static",
        indexing="pointer",
        num_stages=1,
        num_warps=8,
        pid_type="persistent_blocked",
    ),
    static_shapes=True,
    backend="triton",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)

MAX_CONTEXT = 1 << 20
ROPE_CAPACITY = 1 << 17
WINDOW_LEFT = 2047


@dataclasses.dataclass(frozen=True)
class AttentionCase:
    """One verify step at KV length ``seq_len``; the T new tokens are the last."""

    sliding_window: bool
    seq_len: int
    qkv: torch.Tensor  # [T, QKV_WIDTH] bf16, [q | k | v | gate]
    positions: torch.Tensor  # [T] int64
    cos_sin_cache: torch.Tensor  # [P, D] fp32, cos then sin halves
    k_cache: torch.Tensor  # [pages, PAGE_SIZE * KV_WIDTH] fp8
    v_cache: torch.Tensor  # [pages, PAGE_SIZE * KV_WIDTH] fp8
    cache_loc: torch.Tensor  # [T] int64, 0 = padding
    block_tables: torch.Tensor  # [1, MAX_CONTEXT / PAGE_SIZE] int32
    seq_lens: torch.Tensor  # [1] int32, including the T new tokens

    def clone(self) -> AttentionCase:
        return dataclasses.replace(
            self,
            **{
                field.name: getattr(self, field.name).clone()
                for field in dataclasses.fields(self)
                if isinstance(getattr(self, field.name), torch.Tensor)
            },
        )


@dataclasses.dataclass(frozen=True)
class Workspaces:
    att_ws: torch.Tensor  # [Hkv, S, T, Hq / Hkv, D] fp32
    lse_ws: torch.Tensor  # [Hkv, S, T, Hq / Hkv] fp32, log2 units
    no_positions: torch.Tensor  # [T] int64, for layers without RoPE
    no_cos_sin: torch.Tensor  # [1, D] fp32, for layers without RoPE


@functools.cache
def _rope_cache(device: torch.device) -> torch.Tensor:
    exponent = torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM
    freqs = torch.outer(
        torch.arange(ROPE_CAPACITY, dtype=torch.float64), 1.0 / (10000.0**exponent)
    )
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).float().to(device)


def make_case(
    tokens: int,
    *,
    sliding_window: bool,
    seq_len: int,
    seed: int,
    padded: tuple[int, ...] = (),
) -> AttentionCase:
    """Random QKVG rows and a paged FP8 cache; tokens in ``padded`` get slot 0,
    the CUDA-graph padding sentinel, and store nothing."""
    # RoPE (sliding-window layers) gathers cos/sin rows unmasked.
    assert not sliding_window or seq_len <= ROPE_CAPACITY
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    pool_pages = pages + 16
    k_raw = torch.randn(pool_pages * PAGE_SIZE, KV_WIDTH, generator=g, device=device)
    rows = k_raw.view(-1, HEAD_DIM)
    k_norm = rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + 1e-5)
    v_raw = torch.randn(pool_pages * PAGE_SIZE, KV_WIDTH, generator=g, device=device)
    block_tables = torch.zeros(
        1, MAX_CONTEXT // PAGE_SIZE, dtype=torch.int32, device=device
    )
    # Page 0 holds slot 0, the padding sentinel: requests never own it.
    perm = torch.randperm(pool_pages - 1, generator=g, device=device)[:pages] + 1
    block_tables[0, :pages] = perm.to(torch.int32)
    positions = torch.arange(seq_len - tokens, seq_len, device=device)
    cache_loc = block_tables[0, positions // PAGE_SIZE].long() * PAGE_SIZE
    cache_loc = cache_loc + positions % PAGE_SIZE
    cache_loc[list(padded)] = 0
    return AttentionCase(
        sliding_window=sliding_window,
        seq_len=seq_len,
        # Unit-scale QKVG projections: the gate spans both sigmoid tails.
        qkv=torch.randn(tokens, QKV_WIDTH, generator=g, device=device).bfloat16(),
        positions=positions,
        cos_sin_cache=_rope_cache(device),
        k_cache=k_norm.bfloat16()
        .to(torch.float8_e4m3fn)
        .view(pool_pages, PAGE_SIZE * KV_WIDTH),
        v_cache=(v_raw * 0.25)
        .bfloat16()
        .to(torch.float8_e4m3fn)
        .view(pool_pages, PAGE_SIZE * KV_WIDTH),
        cache_loc=cache_loc,
        block_tables=block_tables,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
    )


def make_workspaces(tokens: int, device: torch.device) -> Workspaces:
    # One ATTEND task per SM: a 2047-token window spans at most 33 cache tiles
    # of 64 keys, so 37 splits on B200 leave none idle for long.
    num_splits = torch.cuda.get_device_properties(device).multi_processor_count // (
        NUM_KV_HEADS
    )
    rows = (NUM_KV_HEADS, num_splits, tokens, HEADS_PER_KV)
    return Workspaces(
        att_ws=torch.empty(*rows, HEAD_DIM, dtype=torch.float32, device=device),
        lse_ws=torch.empty(*rows, dtype=torch.float32, device=device),
        no_positions=torch.zeros(tokens, dtype=torch.int64, device=device),
        no_cos_sin=torch.zeros(1, HEAD_DIM, dtype=torch.float32, device=device),
    )


def launch(
    case: AttentionCase,
    workspaces: Workspaces,
    *,
    bmm1_scale: float = BMM1_SCALE,
    eps: float = EPS,
) -> torch.Tensor:
    """Gated [T, Q_WIDTH] bf16 output; also stores the T new K/V rows."""
    tokens = case.qkv.size(0)
    k_cache, v_cache = case.k_cache, case.v_cache
    k_page_stride, v_page_stride = k_cache.stride(0), v_cache.stride(0)
    k_storage = torch.as_strided(
        k_cache, ((k_cache.size(0) - 1) * k_page_stride + k_cache.size(1),), (1,)
    )
    v_storage = torch.as_strided(
        v_cache, ((v_cache.size(0) - 1) * v_page_stride + v_cache.size(1),), (1,)
    )
    positions, cos_sin_cache = case.positions, case.cos_sin_cache
    if not case.sliding_window:
        positions, cos_sin_cache = workspaces.no_positions, workspaces.no_cos_sin
    out = torch.empty(tokens, Q_WIDTH, dtype=torch.bfloat16, device=case.qkv.device)
    return muse_spark_swa(
        case.qkv,
        positions,
        cos_sin_cache,
        k_storage,
        v_storage,
        case.cache_loc,
        case.block_tables,
        case.seq_lens,
        workspaces.att_ws,
        workspaces.lse_ws,
        out,
        PAGE_SIZE,
        k_page_stride,
        v_page_stride,
        bmm1_scale * LOG2E,
        eps,
        case.sliding_window,
        WINDOW_LEFT if case.sliding_window else -1,
        KEY_TILE,
    )


def _qk_prep_reference(
    raw: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float, rope: bool
) -> torch.Tensor:
    """FusedQKNorm's FP8-path rows: RMSNorm to BF16, interleaved RoPE to BF16."""
    x = raw.float()
    normalized = (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)).bfloat16()
    if not rope:
        values = normalized.float()
    else:
        even, odd = normalized.float()[..., 0::2], normalized.float()[..., 1::2]
        values = (
            torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
            .flatten(-2)
            .bfloat16()
            .float()
        )
    return values.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def reference(
    case: AttentionCase, *, bmm1_scale: float = BMM1_SCALE, eps: float = EPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(FP64 gated output, post-store K cache, post-store V cache).

    Q/K/V rows are prepared and stored as the served FP8 path does; attention,
    the merge and the norm + gate run in FP64 over the post-store cache.
    """
    tokens, seq_len = case.qkv.size(0), case.seq_len
    qkv = case.qkv
    rope = case.sliding_window
    cos = case.cos_sin_cache[case.positions, : HEAD_DIM // 2][:, None, :]
    sin = case.cos_sin_cache[case.positions, HEAD_DIM // 2 :][:, None, :]
    q = qkv[:, :Q_WIDTH].view(tokens, NUM_Q_HEADS, HEAD_DIM)
    k = qkv[:, Q_WIDTH : Q_WIDTH + KV_WIDTH].view(tokens, NUM_KV_HEADS, HEAD_DIM)
    v = qkv[:, Q_WIDTH + KV_WIDTH : Q_WIDTH + 2 * KV_WIDTH]
    q_fp8 = _qk_prep_reference(q, cos, sin, eps, rope)
    k_fp8 = _qk_prep_reference(k, cos, sin, eps, rope).view(tokens, KV_WIDTH)
    v_fp8 = v.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    k_cache, v_cache = case.k_cache.clone(), case.v_cache.clone()
    live = case.cache_loc != 0
    k_cache.view(-1, KV_WIDTH)[case.cache_loc[live]] = k_fp8[live]
    v_cache.view(-1, KV_WIDTH)[case.cache_loc[live]] = v_fp8[live]

    pages = case.block_tables[0, : (seq_len + PAGE_SIZE - 1) // PAGE_SIZE].long()

    def logical(cache: torch.Tensor) -> torch.Tensor:
        c = cache.view(-1, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM).index_select(0, pages)
        c = c.permute(2, 0, 1, 3).reshape(NUM_KV_HEADS, -1, HEAD_DIM)
        return c[:, :seq_len].double()

    key, value = logical(k_cache), logical(v_cache)
    kv_head = torch.arange(NUM_Q_HEADS, device=qkv.device) // HEADS_PER_KV
    scores = torch.einsum("thd,hkd->thk", q_fp8.double(), key[kv_head]) * bmm1_scale
    back = (
        torch.arange(seq_len - tokens, seq_len, device=qkv.device)[:, None]
        - torch.arange(seq_len, device=qkv.device)[None, :]
    )
    visible = back >= 0
    if rope:
        visible &= back <= WINDOW_LEFT
    scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
    attn = torch.einsum("thk,hkd->thd", torch.softmax(scores, dim=-1), value[kv_head])
    attn = attn * torch.rsqrt(attn.pow(2).mean(-1, keepdim=True) + eps)
    gate = qkv[:, -Q_WIDTH:].double().view(tokens, NUM_Q_HEADS, HEAD_DIM)
    return (attn * torch.sigmoid(gate)).reshape(tokens, Q_WIDTH), k_cache, v_cache


def _check_against_reference(
    actual: torch.Tensor,
    case: AttentionCase,
    before: AttentionCase,
    *,
    bmm1_scale: float = BMM1_SCALE,
    eps: float = EPS,
) -> None:
    expected, k_cache, v_cache = reference(before, bmm1_scale=bmm1_scale, eps=eps)
    for name, got, want in (("K", case.k_cache, k_cache), ("V", case.v_cache, v_cache)):
        if not torch.equal(got.view(torch.uint8), want.view(torch.uint8)):
            raise AssertionError(f"stored {name} rows differ from the FP8 reference")
    if not torch.isfinite(actual.float()).all():
        raise AssertionError("non-finite attention output")
    # The megakernel lands at ~2.3e-3 (the sglang path at 5e-3 to 8e-3).
    rrmse, cosine = relative_rmse_and_cosine(actual, expected)
    if rrmse > 5e-3 or cosine < 0.99999:
        raise AssertionError(
            f"attention vs FP64 reference: rrmse {rrmse:.3e}, cos {cosine:.7f}"
        )


# (tokens, sliding window, KV length, padded tokens): short and long, padded
# CUDA-graph slots, and the global-layer form the kernel also supports.
CHECK_CASES = (
    (8, True, 300, ()),
    (8, True, 2500, (0, -1)),
    (16, True, 2100, ()),
    (16, True, 2500, (0, -1)),
    (8, True, 79872, ()),
    (8, False, 2500, ()),
    (16, False, 16, ()),
)


def correctness_check() -> None:
    """Check outputs and stored K/V rows against the FP64 reference."""
    for tokens, sliding_window, seq_len, padded in CHECK_CASES:
        before = make_case(
            tokens,
            sliding_window=sliding_window,
            seq_len=seq_len,
            seed=seq_len,
            padded=padded,
        )
        case = before.clone()
        actual = launch(case, make_workspaces(tokens, case.qkv.device))
        _check_against_reference(actual, case, before)


def use_cudagraph() -> bool:
    return True


def _sglang_default(
    case: AttentionCase, *, bmm1_scale: float = BMM1_SCALE, eps: float = EPS
) -> tuple[Any, str]:
    """The served sglang path for one verify step, every kernel launched with
    PDL as in serving: FusedQKNorm (Triton), FlashInfer's TRT-LLM paged decode
    and the CUDA ``rms_norm_gate``."""
    import flashinfer
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

    # FusedQKNorm and rms_norm_gate launch with PDL exactly when this holds.
    assert is_arch_support_pdl()

    tokens = case.qkv.size(0)
    qk_norm = FusedQKNorm(
        head_dim=HEAD_DIM, num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS, eps=eps
    )
    sms = torch.cuda.get_device_properties(case.qkv.device).multi_processor_count
    counters = torch.zeros(
        flashinfer.utils.get_trtllm_gen_multi_ctas_kv_counter_bytes(
            tokens, NUM_Q_HEADS, sms
        ),
        dtype=torch.uint8,
        device=case.qkv.device,
    )
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=case.qkv.device)
    rope = case.sliding_window

    def paged_hnd(cache: torch.Tensor) -> torch.Tensor:
        return cache.view(-1, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    def call() -> torch.Tensor:
        q_fp8, _ = qk_norm.forward_triton(
            case.qkv,
            case.positions if rope else None,
            case.cos_sin_cache if rope else None,
            (
                case.k_cache.view(-1, KV_WIDTH),
                case.v_cache.view(-1, KV_WIDTH),
                case.cache_loc,
            ),
        )
        attn = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q_fp8.view(tokens, NUM_Q_HEADS, HEAD_DIM),
            kv_cache=(paged_hnd(case.k_cache), paged_hnd(case.v_cache)),
            workspace_buffer=workspace,
            block_tables=case.block_tables,
            seq_lens=case.seq_lens,
            # The served value: the model's context length (1M) by default.
            max_seq_len=MAX_CONTEXT,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            window_left=WINDOW_LEFT if rope else -1,
            out_dtype=torch.bfloat16,
            backend="trtllm-gen",
            q_len_per_req=tokens,
            multi_ctas_kv_counter_buffer=counters,
            enable_pdl=True,
        )
        return rms_norm_gate(
            attn.view(tokens, Q_WIDTH),
            case.qkv[:, -Q_WIDTH:],
            None,
            head_dim=HEAD_DIM,
            eps=eps,
        )

    return call, "sglang_default"


# (tokens, KV length): a short request and an OpenHands-scale context; the
# sliding window bounds the work, so the long context costs only page lookups.
BENCHMARK_CASES = tuple(
    (tokens, seq_len) for tokens in VERIFY_TOKENS for seq_len in (2048, 79872)
)


@torch.inference_mode()
def main(verbose: bool = True) -> dict[str, Any]:
    """Check against the FP64 reference and the sglang path, then compare
    cold-L2 CUDA-graph latencies on the sliding-window layer."""
    from pretuned_kernels._bench import capture_cuda_graph
    from pretuned_kernels._bench import run_sweep

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("muse_spark_swa is pretuned for SM100")
    correctness_check()

    def make_calls(shape: tuple[int, int]) -> tuple:
        tokens, seq_len = shape
        before = make_case(tokens, sliding_window=True, seq_len=seq_len, seed=seq_len)
        helion_case, sglang_case = before.clone(), before.clone()
        workspaces = make_workspaces(tokens, before.qkv.device)
        sglang_call, sglang_name = _sglang_default(sglang_case)
        expected = sglang_call()
        actual = launch(helion_case, workspaces)
        _check_against_reference(actual, helion_case, before)
        rrmse, cosine = relative_rmse_and_cosine(actual, expected)
        if rrmse > 1e-2 or cosine < 0.9999:
            raise AssertionError(f"vs sglang: rrmse {rrmse:.3e}, cos {cosine:.6f}")
        # Replays store the same new K/V rows again: no reset is needed.
        helion_graph, _ = capture_cuda_graph(lambda: launch(helion_case, workspaces))
        sglang_graph, _ = capture_cuda_graph(sglang_call)
        return (
            helion_graph.replay,
            [(sglang_name, sglang_graph.replay)],
            f"{tokens:>6d}  {seq_len:>7d}",
        )

    return run_sweep(
        BENCHMARK_CASES,
        make_calls,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        thermal_warmup_ms=5_000,
        verbose=verbose,
        shape_header=f"{'tokens':>6s}  {'kv_len':>7s}",
    )


if __name__ == "__main__":
    main()
