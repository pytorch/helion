"""Registry of attention-gym variants for the attngym benchmark suite.

Defines 16 attention flavors selected from meta-pytorch/attention-gym
(https://github.com/meta-pytorch/attention-gym) that are NOT covered by the
existing dense/causal/biased benchmarks in compare_attention_backends.py.
The score_mod / mask_mod definitions below are vendored (lightly adapted for
closure-tensor tracing) from attention-gym, which is BSD-3 licensed.

Each variant is expressed in FlexAttention terms: an optional ``score_mod``,
an optional ``mask_mod`` (compiled into a BlockMask), tensor shapes, dtype,
and optional GQA grouping. All masks and mods here are batch/head-broadcast
(no mask depends on ``b``; only score mods use ``h``), so BlockMasks are
built once with B=1, H=1 and expanded where a per-(b, h) layout is needed.

The Helion implementation vehicle is an output-only copy of
``examples/flex_attention.py::helion_flex_attention_kernel`` (LSE elided so
baselines that skip LSE, like SDPA-style kernels, are comparable).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import functools
import math
from typing import Any
from typing import Callable

import torch

MaskMod = Callable[..., torch.Tensor]
ScoreMod = Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# Vendored mask mods (attn_gym.masks)
# ---------------------------------------------------------------------------


def causal_mask(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
) -> torch.Tensor:
    return q_idx >= kv_idx


def make_sliding_window_causal(window_size: int) -> MaskMod:
    def sliding_window_causal(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return (q_idx - kv_idx <= window_size) & (q_idx >= kv_idx)

    return sliding_window_causal


def make_dilated_sliding_window(window_size: int, dilation: int) -> MaskMod:
    def dilated_sliding_window(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        diff = torch.abs(q_idx - kv_idx)
        return (diff <= window_size) & ((diff % dilation) == 0)

    return dilated_sliding_window


def make_prefix_lm(prefix_length: int) -> MaskMod:
    def prefix_lm(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return (kv_idx < prefix_length) | (q_idx >= kv_idx)

    return prefix_lm


def make_global_sliding_window(window_size: int, is_global: torch.Tensor) -> MaskMod:
    def global_sliding_window(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        in_window = torch.abs(q_idx - kv_idx) <= window_size
        return in_window | is_global[q_idx] | is_global[kv_idx]

    return global_sliding_window


def make_document_causal(document_id: torch.Tensor, offsets: torch.Tensor) -> MaskMod:
    def document_causal(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        q_doc = document_id[q_idx]
        same_doc = q_doc == document_id[kv_idx]
        return same_doc & (q_idx >= kv_idx)

    return document_causal


def make_natten2d(
    canvas_w: int, canvas_h: int, kernel_w: int, kernel_h: int
) -> MaskMod:
    def natten2d(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        q_x = q_idx // canvas_w
        q_y = q_idx % canvas_w
        kv_x = kv_idx // canvas_w
        kv_y = kv_idx % canvas_w
        center_x = q_x.clamp(kernel_w // 2, (canvas_w - 1) - kernel_w // 2)
        center_y = q_y.clamp(kernel_h // 2, (canvas_h - 1) - kernel_h // 2)
        hori = (center_x - kernel_w // 2 <= kv_x) & (kv_x <= center_x + kernel_w // 2)
        vert = (center_y - kernel_h // 2 <= kv_y) & (kv_y <= center_y + kernel_h // 2)
        return hori & vert

    return natten2d


def make_sta2d(
    canvas_hw: tuple[int, int],
    kernel_hw: tuple[int, int],
    tile_hw: tuple[int, int],
) -> MaskMod:
    canvas_h, canvas_w = canvas_hw
    kernel_h, kernel_w = kernel_hw
    tile_h, tile_w = tile_hw
    tile_numel = tile_h * tile_w
    canvas_tile_w = canvas_w // tile_w
    canvas_tile_h = canvas_h // tile_h
    kernel_tile_h = kernel_h // tile_h
    kernel_tile_w = kernel_w // tile_w

    def sta2d(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        q_tile = q_idx // tile_numel
        kv_tile = kv_idx // tile_numel
        q_tile_h = q_tile // canvas_tile_w
        q_tile_w = q_tile % canvas_tile_w
        kv_tile_h = kv_tile // canvas_tile_w
        kv_tile_w = kv_tile % canvas_tile_w
        left_h = kernel_tile_h // 2
        right_h = kernel_tile_h // 2 + (kernel_tile_h % 2 - 1)
        left_w = kernel_tile_w // 2
        right_w = kernel_tile_w // 2 + (kernel_tile_w % 2 - 1)
        center_h = q_tile_h.clamp(left_h, (canvas_tile_h - 1) - right_h)
        center_w = q_tile_w.clamp(left_w, (canvas_tile_w - 1) - right_w)
        h_mask = (kv_tile_h >= center_h - left_h) & (kv_tile_h <= center_h + right_h)
        w_mask = (kv_tile_w >= center_w - left_w) & (kv_tile_w <= center_w + right_w)
        return h_mask & w_mask

    return sta2d


def make_block_diffusion(seq_len: int, block_size: int) -> MaskMod:
    def block_diffusion(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        q_noised = q_idx < seq_len
        kv_noised = kv_idx < seq_len
        q_block = (q_idx % seq_len) // block_size
        kv_block = (kv_idx % seq_len) // block_size
        block_diagonal = (q_block == kv_block) & (q_noised == kv_noised)
        offset_block_causal = (q_block > kv_block) & q_noised & ~kv_noised
        block_causal = (q_block >= kv_block) & ~q_noised & ~kv_noised
        return block_diagonal | offset_block_causal | block_causal

    return block_diffusion


def make_shared_prefix(
    document_id: torch.Tensor,
    offsets: torch.Tensor,
    prefix_document_id: torch.Tensor,
) -> MaskMod:
    def shared_prefix(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        q_document = document_id[q_idx]
        q_document_start = offsets[q_document]
        q_prefix_document = prefix_document_id[q_document]
        q_prefix_start = offsets[q_prefix_document]
        q_prefix_end = offsets[q_prefix_document + 1]
        same_document_causal = (kv_idx >= q_document_start) & (q_idx >= kv_idx)
        prefix_document = (
            (q_document != q_prefix_document)
            & (kv_idx >= q_prefix_start)
            & (kv_idx < q_prefix_end)
        )
        return same_document_causal | prefix_document

    return shared_prefix


def make_flamingo(
    interval_start: torch.Tensor,
    interval_end: torch.Tensor,
    image_boundaries: torch.Tensor,
) -> MaskMod:
    def flamingo_xattn(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        image_idx = image_boundaries[kv_idx]
        return (q_idx >= interval_start[image_idx]) & (q_idx < interval_end[image_idx])

    return flamingo_xattn


# ---------------------------------------------------------------------------
# Vendored score mods (attn_gym.mods)
# ---------------------------------------------------------------------------


def make_alibi(num_heads: int) -> ScoreMod:
    def alibi(
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.exp2(-((h + 1) * 8.0 / num_heads))
        return score + (kv_idx - q_idx) * scale

    return alibi


def make_tanh_softcap(soft_cap: float) -> ScoreMod:
    def tanh_softcap(
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        return soft_cap * torch.tanh(score / soft_cap)

    return tanh_softcap


def make_sandwich(
    rel_bias: torch.Tensor, head_scale: torch.Tensor, offset: int
) -> ScoreMod:
    def sandwich(
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        return score + rel_bias[q_idx - kv_idx + offset] * head_scale[h]

    return sandwich


def make_sigmoid_act() -> ScoreMod:
    def sigmoid_act(
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        return torch.log(torch.sigmoid(score) + 1.0)

    return sigmoid_act


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    """One attention flavor: shape + score_mod/mask_mod builders.

    ``make_mods(device)`` returns ``(score_mod, mask_mod)`` (either may be
    None); closure tensors are created on ``device`` deterministically.
    """

    name: str
    b: int
    hq: int
    hkv: int
    m: int
    n: int
    d: int
    dtype: torch.dtype
    make_mods: Callable[[torch.device], tuple[ScoreMod | None, MaskMod | None]]
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def enable_gqa(self) -> bool:
        return self.hkv != self.hq


def _sandwich_tables(
    device: torch.device, num_heads: int, max_seq_len: int, d_bar: int = 128
) -> tuple[torch.Tensor, torch.Tensor, int]:
    freqs = 1.0 / 10000 ** (
        2 * torch.arange(d_bar // 2, device=device, dtype=torch.float32) / d_bar
    )
    padded_len = -(-(2 * max_seq_len - 1) // 8) * 8
    distances = torch.arange(padded_len, device=device, dtype=torch.float32) - (
        max_seq_len - 1
    )
    rel_bias = torch.cos(distances.abs()[:, None] * freqs).sum(dim=-1)
    head_scale = num_heads / (
        8.0 * torch.arange(1, num_heads + 1, device=device, dtype=torch.float32)
    )
    return rel_bias, head_scale, max_seq_len - 1


def _document_layout(
    device: torch.device, total_len: int, num_docs: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random doc lengths summing to total_len (deterministic)."""
    gen = torch.Generator().manual_seed(seed)
    cuts = torch.randperm(total_len - num_docs, generator=gen)[: num_docs - 1]
    cuts, _ = torch.sort(cuts)
    lengths = torch.diff(
        torch.cat([torch.tensor([0]), cuts + 1, torch.tensor([total_len - num_docs])])
    )
    lengths = (lengths + 1).tolist()
    assert sum(lengths) == total_len
    offsets = torch.zeros(num_docs + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(
        torch.tensor(lengths, dtype=torch.int32, device=device), dim=0
    )
    counts = offsets[1:] - offsets[:-1]
    document_id = torch.repeat_interleave(
        torch.arange(num_docs, device=device, dtype=torch.int32), counts.long()
    )
    return document_id, offsets


def _mods_alibi(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    return make_alibi(16), causal_mask


def _mods_softcap(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    return make_tanh_softcap(30.0), causal_mask


def _mods_sandwich(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    rel_bias, head_scale, offset = _sandwich_tables(device, 16, 8192)
    return make_sandwich(rel_bias, head_scale, offset), causal_mask


def _mods_sigmoid_act(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    return make_sigmoid_act(), None


def _mods_sliding_window(
    device: torch.device,
) -> tuple[ScoreMod | None, MaskMod | None]:
    return None, make_sliding_window_causal(1024)


def _mods_dilated_sw(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    return None, make_dilated_sliding_window(512, 2)


def _mods_prefix_lm(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    return None, make_prefix_lm(1024)


def _mods_global_sw(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    is_global = torch.zeros(8192, dtype=torch.bool, device=device)
    is_global[:64] = True
    is_global[::1024] = True
    return None, make_global_sliding_window(512, is_global)


def _mods_document(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    document_id, offsets = _document_layout(device, 32768, 12, seed=0)
    return None, make_document_causal(document_id, offsets)


def _mods_natten2d(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    return None, make_natten2d(128, 128, 13, 13)


def _mods_sta2d(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    return None, make_sta2d((128, 128), (64, 64), (16, 16))


def _mods_block_diffusion(
    device: torch.device,
) -> tuple[ScoreMod | None, MaskMod | None]:
    return None, make_block_diffusion(4096, 128)


def _mods_gqa_causal(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    return None, causal_mask


def _mods_gemma2(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    return make_tanh_softcap(50.0), make_sliding_window_causal(1024)


def _mods_shared_prefix(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    # 2 groups of (1 shared prefix + 3 continuations), 16384 tokens total.
    lengths = [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]
    offsets = torch.zeros(9, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(
        torch.tensor(lengths, dtype=torch.int32, device=device), dim=0
    )
    counts = offsets[1:] - offsets[:-1]
    document_id = torch.repeat_interleave(
        torch.arange(8, device=device, dtype=torch.int32), counts.long()
    )
    prefix_document_id = torch.tensor(
        [0, 0, 0, 0, 4, 4, 4, 4], dtype=torch.int32, device=device
    )
    return None, make_shared_prefix(document_id, offsets, prefix_document_id)


def _mods_flamingo(device: torch.device) -> tuple[ScoreMod | None, MaskMod | None]:
    # 16 images x 256 tokens (N=4096); image i sits at text position i*512 and
    # stays visible for the next 2048 text tokens ("last 4 images" policy).
    num_images = 16
    image_tokens = 256
    starts = torch.arange(num_images, dtype=torch.int32, device=device) * 512
    ends = torch.clamp(starts + 2048, max=8192)
    image_boundaries = torch.repeat_interleave(
        torch.arange(num_images, device=device, dtype=torch.int32), image_tokens
    )
    return None, make_flamingo(starts, ends, image_boundaries)


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        Variant(
            "alibi",
            16,
            16,
            16,
            8192,
            8192,
            64,
            torch.float16,
            _mods_alibi,
            "causal + ALiBi head-dependent linear bias",
        ),
        Variant(
            "softcap",
            16,
            16,
            16,
            8192,
            8192,
            64,
            torch.float16,
            _mods_softcap,
            "causal + 30*tanh(score/30) (Gemma2/Grok softcap)",
        ),
        Variant(
            "sandwich",
            8,
            16,
            16,
            8192,
            8192,
            64,
            torch.float16,
            _mods_sandwich,
            "causal + cosine relative-position table bias * head_scale[h]",
        ),
        Variant(
            "sigmoid-act",
            8,
            16,
            16,
            4096,
            4096,
            128,
            torch.bfloat16,
            _mods_sigmoid_act,
            "dense, log(sigmoid(score)+1) activation mod",
        ),
        Variant(
            "sliding-window",
            16,
            16,
            16,
            8192,
            8192,
            64,
            torch.float16,
            _mods_sliding_window,
            "causal AND q-kv<=1024",
        ),
        Variant(
            "dilated-sw",
            8,
            16,
            16,
            8192,
            8192,
            64,
            torch.float16,
            _mods_dilated_sw,
            "abs(d)<=512 AND d%2==0 (no full blocks)",
        ),
        Variant(
            "prefix-lm",
            16,
            16,
            16,
            8192,
            8192,
            64,
            torch.float16,
            _mods_prefix_lm,
            "causal OR kv<1024",
        ),
        Variant(
            "global-sw",
            8,
            16,
            16,
            8192,
            8192,
            64,
            torch.float16,
            _mods_global_sw,
            "abs(d)<=512 OR is_global[q] OR is_global[kv]",
        ),
        Variant(
            "document",
            1,
            16,
            16,
            32768,
            32768,
            64,
            torch.float16,
            _mods_document,
            "12 jagged docs packed in 32768, causal within doc",
        ),
        Variant(
            "natten2d",
            4,
            16,
            16,
            16384,
            16384,
            64,
            torch.float16,
            _mods_natten2d,
            "128x128 canvas, 13x13 neighborhood",
        ),
        Variant(
            "sta2d",
            4,
            16,
            16,
            16384,
            16384,
            64,
            torch.float16,
            _mods_sta2d,
            "sliding tile attention: 128x128 canvas, 16x16 tiles, 64x64 kernel",
        ),
        Variant(
            "block-diffusion",
            4,
            16,
            16,
            8192,
            8192,
            64,
            torch.float16,
            _mods_block_diffusion,
            "noised+clean halves, 128-blocks, S=4096",
        ),
        Variant(
            "gqa-causal",
            4,
            32,
            8,
            8192,
            8192,
            128,
            torch.bfloat16,
            _mods_gqa_causal,
            "causal with 32 q heads / 8 kv heads (Llama3-8B)",
        ),
        Variant(
            "gemma2",
            4,
            16,
            8,
            8192,
            8192,
            128,
            torch.bfloat16,
            _mods_gemma2,
            "softcap 50 + sliding window 1024 + GQA 16/8",
        ),
        Variant(
            "shared-prefix",
            2,
            16,
            16,
            16384,
            16384,
            64,
            torch.float16,
            _mods_shared_prefix,
            "2 groups: shared 2048-prefix + 3 continuations",
        ),
        Variant(
            "flamingo",
            8,
            16,
            16,
            8192,
            4096,
            64,
            torch.float16,
            _mods_flamingo,
            "rectangular text->image cross attention (M!=N)",
        ),
    ]
}


def make_qkv(
    variant: Variant, device: torch.device, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(b: int, h: int, s: int) -> torch.Tensor:
        return torch.randn(
            (b, h, s, variant.d), device=device, dtype=variant.dtype, generator=gen
        )

    q = rand(variant.b, variant.hq, variant.m)
    k = rand(variant.b, variant.hkv, variant.n)
    v = rand(variant.b, variant.hkv, variant.n)
    return q, k, v


def build_block_mask(
    variant: Variant,
    mask_mod: MaskMod | None,
    device: torch.device,
    block_size: int | tuple[int, int] = 128,
) -> object:
    """BlockMask for the variant (broadcast B=1, H=1); noop mask when None."""
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import noop_mask

    compiled = torch.compile(create_block_mask)
    return compiled(
        mask_mod if mask_mod is not None else noop_mask,
        None,
        None,
        variant.m,
        variant.n,
        device=str(device),
        BLOCK_SIZE=block_size,
    )


@functools.cache
def _cached_pair_count(name: str, device_str: str) -> int:
    variant = VARIANTS[name]
    device = torch.device(device_str)
    _, mask_mod = variant.make_mods(device)
    if mask_mod is None:
        return variant.b * variant.hq * variant.m * variant.n
    total = 0
    b = torch.zeros((), dtype=torch.int32, device=device)
    h = torch.zeros((), dtype=torch.int32, device=device)
    kv = torch.arange(variant.n, device=device)[None, :]
    chunk = max(1, (1 << 24) // variant.n)
    for q0 in range(0, variant.m, chunk):
        qi = torch.arange(q0, min(q0 + chunk, variant.m), device=device)[:, None]
        total += int(mask_mod(b, h, qi, kv).sum().item())
    return total * variant.b * variant.hq


def variant_flops(variant: Variant, device: torch.device) -> float:
    """4*D flops per unmasked (q, kv) pair -- identical for every impl."""
    pairs = _cached_pair_count(variant.name, str(device))
    return 4.0 * variant.d * pairs


def reference_output(
    variant: Variant,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    score_mod: ScoreMod | None,
    mask_mod: MaskMod | None,
    q_chunk: int = 1024,
) -> torch.Tensor:
    """Chunked fp32 reference implementing FlexAttention semantics."""
    b_sz, hq, m, d = q.shape
    n = k.shape[2]
    groups = hq // k.shape[1]
    scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q, dtype=torch.float32)
    kv_idx = torch.arange(n, device=q.device)[None, :]
    for b in range(b_sz):
        b_t = torch.tensor(b, device=q.device)
        for h in range(hq):
            h_t = torch.tensor(h, device=q.device)
            k_bh = k[b, h // groups].float()
            v_bh = v[b, h // groups].float()
            for q0 in range(0, m, q_chunk):
                q1 = min(q0 + q_chunk, m)
                s = (q[b, h, q0:q1].float() @ k_bh.T) * scale
                q_idx = torch.arange(q0, q1, device=q.device)[:, None]
                if score_mod is not None:
                    s = score_mod(s, b_t, h_t, q_idx, kv_idx)
                if mask_mod is not None:
                    s = torch.where(mask_mod(b_t, h_t, q_idx, kv_idx), s, float("-inf"))
                p = torch.softmax(s, dim=-1)
                p = torch.nan_to_num(p, nan=0.0)  # rows with no unmasked kv
                out[b, h, q0:q1] = p @ v_bh
    return out.to(q.dtype)
