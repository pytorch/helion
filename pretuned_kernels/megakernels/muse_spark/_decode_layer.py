"""Device helpers, host-side shapes and MoE buffers of the layer megakernels (muse_decode_layer_tp4_t8.py and
muse_decode_layer_tp8_t8.py), which read these names unqualified (their module namespaces import this one's). The
first half (attention .. expert input): QK norm / RoPE prep, the cache tile walk and the online-softmax step of the
paged attention, and the pooler / router split shapes. The second half (the expert-TP MoE .. the next layer's QKV):
NVFP4 scale tiles, SwiGLU and the z quant (the routed-row tables are _moe's), the unpooler weight tile and the K
splits, and the per-rank symmetric MoE state."""

from __future__ import annotations

from pretuned_kernels.megakernels.muse_spark._common import EXPERT_HIDDEN
from pretuned_kernels.megakernels.muse_spark._common import LOG2E
from pretuned_kernels.megakernels.muse_spark._common import MOE_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import TOPK
from pretuned_kernels.megakernels.muse_spark._common import _butterfly
from pretuned_kernels.megakernels.muse_spark._common import _chain_sq
from pretuned_kernels.megakernels.muse_spark._common import _columns
from pretuned_kernels.megakernels.muse_spark._common import _max
from pretuned_kernels.megakernels.muse_spark._common import _route
from pretuned_kernels.megakernels.muse_spark._common import encode_block_scale
from pretuned_kernels.megakernels.muse_spark._common import exp2_approx_ftz_f32
from pretuned_kernels.megakernels.muse_spark._common import pack_e2m1x2
from pretuned_kernels.megakernels.muse_spark._common import rcp_approx_ftz_f32
from pretuned_kernels.megakernels.muse_spark._common import rsqrt_approx_ftz_f32
from pretuned_kernels.megakernels.muse_spark._moe import INTERMEDIATE
from pretuned_kernels.megakernels.muse_spark._moe import MMA_M
from pretuned_kernels.megakernels.muse_spark._moe import (
    distinct_table as _distinct_table,
)
from pretuned_kernels.megakernels.muse_spark._moe import expert_of as _expert_of
from pretuned_kernels.megakernels.muse_spark._moe import (
    logical_rows128 as _logical_rows128,
)
from pretuned_kernels.megakernels.muse_spark._moe import quant_z as _quant_z
from pretuned_kernels.megakernels.muse_spark._moe import routed_rows as _routed_rows
from pretuned_kernels.megakernels.muse_spark._moe import scale_tile
from pretuned_kernels.megakernels.muse_spark._moe import swiglu
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion.language as hl


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
    """Exact FP8 -> FP16 (pure: Triton sinks the cast into the dot operand
    layout)."""
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


def _cache_tiles_nr0(
    cache_end: torch.Tensor,
    split: int,
    key_tile: int,
    num_splits: int,
    window_left: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_cache_tiles`` when the last split owns the new rows and no cache
    tiles (the TP8 kernel's attention)."""

    if window_left < 0:
        first_key = cache_end * 0
    else:
        first_key = torch.clamp_min(cache_end - window_left, 0)
    first = first_key & -key_tile
    num_tiles = (cache_end - first + key_tile - 1) >> (key_tile.bit_length() - 1)
    # nr0: the last split owns the new rows and no cache tiles.
    cache_splits = num_splits - 1
    base = num_tiles // cache_splits
    extra = num_tiles - base * cache_splits
    split = cache_end * 0 + split
    lo = split * base + torch.minimum(split, extra)
    hi = lo + base + (split < extra).to(torch.int32)
    return first, lo, torch.where(split < cache_splits, hi, lo)


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


SPLIT_P = 16
P_ROWS = 128
PAD_ROWS = 16


def router_parts(world: int) -> int:
    """R1's router partials: one per pooler K split this rank owns."""
    return SPLIT_P // world


# The second half: the expert-TP MoE .. the next layer's QKV.


def _scale_tile128(chunks: torch.Tensor) -> torch.Tensor:
    """[16, 256] SWIZZLE_32_4_4 bytes of one 128-row tile over one K step -> [128 rows, 32]."""

    return scale_tile(chunks, tiles=1)


def _swiglu128(
    acc: torch.Tensor,
    row_scale: torch.Tensor,
    up_alpha: torch.Tensor,
    gate_alpha: torch.Tensor,
) -> torch.Tensor:
    """Interleaved (up, gate) rows of one FI tile [128, t] -> SwiGLU [t, 64]."""

    return swiglu(acc, row_scale, up_alpha, gate_alpha, order=_logical_rows128)


U_BM, U_BK = 64, 256  # U's weight tile (one 32 KB ring stage)


def parts_for(row_blocks: int, k_steps: int, ctas: int) -> int:
    """K parts per row block so the tasks fill one wave of CTAs (a power of two dividing the K steps)."""
    parts = 1
    while row_blocks * parts * 2 <= ctas and k_steps % (parts * 2) == 0:
        parts *= 2
    return parts


def fi_w13_blocks(w13: torch.Tensor) -> torch.Tensor:
    """The bank's W13 [tiles, K steps, 256 rows, 256 B] (strided over FI's row-major rows) as the kernel's
    (128-row block, K step, K half, row, byte) view; no copy for FI's layout."""
    t, k, r, b = w13.shape
    return (
        w13.unflatten(2, (2, r // 2))
        .unflatten(4, (2, b // 2))
        .permute(0, 2, 1, 4, 3, 5)
        .reshape(t * 2, k, 2, r // 2, b // 2)
    )


class State:
    """Per-rank symmetric MoE buffers and the MoE output rows (padded to 16)."""

    def __init__(
        self,
        device: torch.device,
        group: dist.ProcessGroup,
        *,
        tokens: int,
        intermediate: int = INTERMEDIATE,
        hidden: int = EXPERT_HIDDEN,
        expert_bf16: bool = False,
    ) -> None:
        world = group.size()
        local = hidden // world
        slots = tokens * TOPK
        # The bf16-expert images' z: bf16 SwiGLU rows (z_scale stays unread).
        if expert_bf16:
            self.z = symm_mem.empty(
                slots, MOE_TOKENS, intermediate, dtype=torch.bfloat16, device=device
            )
        else:
            self.z = symm_mem.empty(
                slots, MOE_TOKENS, intermediate // 2, dtype=torch.uint8, device=device
            )
        self.z_scale = symm_mem.empty(
            slots, MOE_TOKENS, intermediate // 16, dtype=torch.uint8, device=device
        )
        self.gemm2_ws = symm_mem.empty(
            slots, local, dtype=torch.bfloat16, device=device
        )
        self.sumsq_ws = symm_mem.empty(
            slots, local // MMA_M, dtype=torch.float32, device=device
        )
        for buffer in (self.z, self.z_scale, self.gemm2_ws, self.sumsq_ws):
            symm_mem.rendezvous(buffer, group.group_name)
        self.y16 = torch.zeros(MOE_TOKENS, hidden, dtype=torch.bfloat16, device=device)


# The names the host and the layer kernels read (the kernels copy this module's whole namespace).
__all__ = [
    "LOG2E",
    "MMA_M",
    "MOE_TOKENS",
    "PAD_ROWS",
    "P_ROWS",
    "SPLIT_P",
    "TOPK",
    "U_BK",
    "U_BM",
    "State",
    "_butterfly",
    "_cache_tiles",
    "_cache_tiles_nr0",
    "_chain_sq",
    "_columns",
    "_cvt_rn_satfinite_bf16",
    "_distinct_table",
    "_e4m3_to_f16",
    "_expert_of",
    "_fp8_clamp",
    "_logical_rows128",
    "_max",
    "_qk_prep",
    "_quant_z",
    "_route",
    "_routed_rows",
    "_scale_tile128",
    "_softmax_step",
    "_swiglu128",
    "encode_block_scale",
    "exp2_approx_ftz_f32",
    "fi_w13_blocks",
    "pack_e2m1x2",
    "parts_for",
    "rcp_approx_ftz_f32",
    "router_parts",
    "rsqrt_approx_ftz_f32",
]
