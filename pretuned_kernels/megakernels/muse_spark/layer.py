"""Muse Spark NVFP4 spec-decode verify layer (B=1, T draft tokens) as ONE Helion
megakernel per layer (native Gluon backend), at tensor-parallel world 4 or 8:
the layer megakernel serving runs.

One launch runs: QK norm/RoPE/KV store + paged sliding-window attention +
norm/gate, O-projection + all-reduce + the post-attention / pre-FFN norms,
pooler + router top-8 + NVFP4 expert input, the expert-TP NVFP4 MoE +
post-expert norm, the unpooler shard + all-gather + the post-FFN boundary,
and the NEXT layer's QKV GEMM. A layer takes its own QKV (from the previous
layer's launch, or the stock GEMM after a stock layer) and returns the next's.

The kernel family is picked per (world, T) by ``layer_family``:

- TP4 (world 4, T=8): muse_decode_layer_tp4_t8.py.
- TP8 (world 8, T=8): muse_decode_layer_tp8_t8.py (world-generic), with the
  EPF2 expert L2 prefetch of a per-layer hot-expert list.

Each of a family's five images has its own measured config next to the kernel
(``LayerFamily.config_name``): ``<kernel>.json`` (sliding window),
``<kernel>_global.json``, ``<kernel>_bf16.json`` (bf16 experts),
``<kernel>_global_bf16_tail.json`` (final norm) and ``<kernel>_entry.json``
(the prelude).

Both kernels read their device helpers and host-side shapes from
_decode_layer.py.

Every per-rank shape comes from ``LayerDims``. A global layer is the same
program built without RoPE and without a window (its KV splits cover the whole
context, so its attention cost grows with the context).

The checkpoint's first and last layers keep bf16 experts; they run images of the
same program too (``expert_bf16``: bf16 expert dots on a bf16 bank, no NVFP4
activation quant), and the last layer's image ends at the model's final norm
(``final_norm``: no next-layer QKV; the launch returns the LM head's input).
The model's prelude before layer 0 is the ``entry`` image: the embedding
all-reduce and norm, layer 0's input norm and its QKV.

The weights use FlashInfer's TRT-LLM layouts (the MoE banks of ``_moe.py``).
Calls and graph replays must occur in identical collective order across ranks
on one ordered CUDA stream. ``main()`` launches the sliding-window, global,
bf16-expert and final-norm images on random weights and checks every stage
against FP64 torch references computed from the kernel's own inputs to that
stage, then repeat and CUDA-graph replays bitwise, then reports each image's
cold-L2 graph replay time; ``--check`` skips the timing. It needs FlashInfer,
which lays the random checkpoint-order experts out as serving loads them::

    PYTHONPATH=. python -m torch.distributed.run --standalone --nproc-per-node=4 \
      pretuned_kernels/megakernels/muse_spark/layer.py [--check]
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import math
import os
import sys
from typing import TYPE_CHECKING
from typing import Any

from pretuned_kernels.megakernels._distributed import _capture
from pretuned_kernels.megakernels._distributed import _cold_replay_us
from pretuned_kernels.megakernels._distributed import initialize_group
from pretuned_kernels.megakernels.muse_spark import _decode_layer as helpers
from pretuned_kernels.megakernels.muse_spark._common import EXPERT_HIDDEN
from pretuned_kernels.megakernels.muse_spark._common import HIDDEN
from pretuned_kernels.megakernels.muse_spark._common import LOG2E
from pretuned_kernels.megakernels.muse_spark._common import MOE_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import NUM_EXPERTS as ROUTER_E
from pretuned_kernels.megakernels.muse_spark._common import TOPK
from pretuned_kernels.megakernels.muse_spark._common import dequantize_nvfp4
from pretuned_kernels.megakernels.muse_spark._common import e2m1_values
from pretuned_kernels.megakernels.muse_spark._moe import GLOBAL_SCALE_INV
from pretuned_kernels.megakernels.muse_spark._moe import (
    INTERMEDIATE as EXPERT_INTERMEDIATE,
)
from pretuned_kernels.megakernels.muse_spark._moe import Bf16VerifyBank
from pretuned_kernels.megakernels.muse_spark._moe import Nvfp4VerifyBank
from pretuned_kernels.megakernels.muse_spark._moe import build_bf16_verify_bank
from pretuned_kernels.megakernels.muse_spark._moe import build_nvfp4_verify_bank
from pretuned_kernels.megakernels.muse_spark._moe import flashinfer_layout
from pretuned_kernels.megakernels.muse_spark._moe import nvfp4_quantize
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion

if TYPE_CHECKING:
    from types import ModuleType

Q_HEADS = 128
KV_HEADS = 16
HEAD_DIM = 64
# The paged KV cache's page (tokens) and the attention's key tile.
PAGE_SIZE = 128
KEY_TILE = 64
SWA_WINDOW_LEFT = 2047
# O-projection weight tile (one 32 KB ring stage).
O_BM, O_BK = 64, 256
# Persistent workers, one wave (GB300 measured: 144 of 152 SMs beats all 152).
WORKERS = 144
# TP4: the QKV root's K splits (256-column K tiles: 72 tasks x 2 splits, the reference's shape).
QKV_SPLITS = 2
# The tail's column block: U's 64 rows (U -> UC is then a per-task fan-in; measured -0.6 us CW, -1.65 us GB300).
TAIL_COLUMNS = 64
# The EPF2 hot-expert list: 16 int32 slots, -1 = empty.
PF_LIST_SLOTS = 16
_PACKAGE = "pretuned_kernels.megakernels.muse_spark"


@dataclasses.dataclass(frozen=True)
class LayerDims:
    """One rank's shapes of the Muse Spark layer at tensor-parallel world ``world``."""

    world: int
    q_heads: int
    kv_heads: int
    heads_per_kv: int
    q_width: int  # q heads x head_dim, also the o_proj K
    kv_width: int
    qkv_width: int  # [q | k | v | gate]
    n_local: int  # pooler rows: expert hidden / W
    local: int  # unpooler rows and AG-norm columns: hidden / W
    intermediate: int  # expert intermediate / W

    @classmethod
    def of(cls, world: int) -> LayerDims:
        if KV_HEADS % world or EXPERT_INTERMEDIATE % world:
            raise ValueError(
                f"the Muse Spark verify layer does not shard over {world} ranks"
            )
        q_heads, kv_heads = Q_HEADS // world, KV_HEADS // world
        return cls(
            world=world,
            q_heads=q_heads,
            kv_heads=kv_heads,
            heads_per_kv=Q_HEADS // KV_HEADS,
            q_width=q_heads * HEAD_DIM,
            kv_width=kv_heads * HEAD_DIM,
            qkv_width=2 * (q_heads + kv_heads) * HEAD_DIM,
            n_local=EXPERT_HIDDEN // world,
            local=HIDDEN // world,
            intermediate=EXPERT_INTERMEDIATE // world,
        )


@dataclasses.dataclass(frozen=True)
class LayerFamily:
    """A layer megakernel build: its kernel module, its measured config, and the host shapes it was measured with."""

    kernel: str  # the kernel's module in this package
    # The measured configs next to this module, in the Gluon schema (num_stages = the weight ring's depth):
    config: str  # the sliding-window (RoPE) build's config,
    global_config: str  # and the global (NoPE, whole-context) build's (its memory ops differ, so its indexing does)
    attention_splits: int  # att_ws split dimension
    qkv_splits: int  # the QKV root's K splits; 0 = helpers.parts_for over the SMs
    q_tile: tuple[int, int]  # the next layer's QKV weight tile (rows, K)
    # Router planes stacked per 64-expert chunk (hi rows, then lo rows).
    stacked_router: bool
    expert_prefetch: bool  # pf_* operands: the bank as prefetch boxes and the layer's hot-expert list
    # The NVFP4 global image's MoE MMA token rows (its expert dots' N); every other
    # image keeps MOE_TOKENS (bf16 dots need 16).
    global_moe_rows: int = MOE_TOKENS

    def moe_rows(self, *, global_attn: bool, expert_bf16: bool = False) -> int:
        """An image's MoE MMA token rows: its x / z slots' rows and its expert dots' N."""
        return self.global_moe_rows if global_attn and not expert_bf16 else MOE_TOKENS

    def config_name(
        self,
        *,
        global_attn: bool,
        expert_bf16: bool = False,
        final_norm: bool = False,
        entry: bool = False,
    ) -> str:
        """The config JSON of an image: the edge images (bf16 experts, the final-norm tail) and the entry image have
        their own, since their memory ops differ (``<base>_bf16.json`` / ``<base>_bf16_tail.json`` /
        ``<base>_entry.json``)."""
        if final_norm and not expert_bf16:
            raise ValueError(
                "the final-norm image is built only with bf16 experts (the checkpoint's last layer)"
            )
        if expert_bf16 and global_attn != final_norm:
            raise ValueError(
                "the bf16-expert images are the sliding-window first layer and the global last layer (final norm)"
            )
        base = self.global_config if global_attn else self.config
        if entry:
            if global_attn or expert_bf16 or final_norm:
                raise ValueError(
                    "the entry image is one build (its attention / MoE / tail roots are dead)"
                )
            return base.removesuffix(".json") + "_entry.json"
        if not expert_bf16:
            return base
        return base.removesuffix(".json") + (
            "_bf16_tail.json" if final_norm else "_bf16.json"
        )


TP4_FAMILY = LayerFamily(
    kernel="muse_decode_layer_tp4_t8",
    config="muse_decode_layer_tp4_t8.json",
    global_config="muse_decode_layer_tp4_t8_global.json",
    # 36 ATTEND splits x 4 KV heads, one wave.
    attention_splits=WORKERS // 4,
    qkv_splits=QKV_SPLITS,
    # 64-row tiles, 72 blocks x 2 K splits (the reference's shape).
    q_tile=(64, 256),
    stacked_router=False,
    expert_prefetch=False,
    # 8-column block-scaled MMAs: faster in the global image, slower in the sliding-window one.
    global_moe_rows=8,
)


TP8_FAMILY = LayerFamily(
    kernel="muse_decode_layer_tp8_t8",
    config="muse_decode_layer_tp8_t8.json",
    global_config="muse_decode_layer_tp8_t8_global.json",
    # (KV head, split, head half) ATTEND tasks over half the splits: 144 at W8, one wave.
    attention_splits=72,
    qkv_splits=0,
    # 128-row tiles (TMEM-stashable), K split by parts_for (8 at W8).
    q_tile=(128, 128),
    stacked_router=True,
    expert_prefetch=True,
)


# The layer megakernel serving runs per (world, T).
_SERVED = {(4, 8): TP4_FAMILY, (8, 8): TP8_FAMILY}


def layer_family(world: int, tokens: int) -> LayerFamily:
    family = _SERVED.get((world, tokens))
    if family is None:
        raise ValueError(
            f"no Helion verify layer megakernel for TP{world} at T={tokens}"
        )
    return family


def _module(name: str) -> ModuleType:
    return importlib.import_module(f"{_PACKAGE}.{name}")


def load_config(
    family: LayerFamily,
    global_attn: bool,
    expert_bf16: bool = False,
    final_norm: bool = False,
    entry: bool = False,
) -> dict:
    name = family.config_name(
        global_attn=global_attn,
        expert_bf16=expert_bf16,
        final_norm=final_norm,
        entry=entry,
    )
    with open(os.path.join(os.path.dirname(__file__), name)) as f:
        return json.load(f)


@dataclasses.dataclass(frozen=True)
class AttentionInputs:
    """Live cache metadata; K/V are ``[pages, page_size * kv_width]`` views.

    A row may have a larger physical stride when the cache embeds quantization
    metadata after each page, so the layer megakernel must retain the page
    dimension.
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    cache_loc: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor


@dataclasses.dataclass(frozen=True)
class BoundaryWeights:
    """One layer's expert-input weights, as views (the router's K rows are a transposed copy)."""

    pooler: torch.Tensor  # [Ne / W / P_ROWS, S_p, P_ROWS, N / S_p] bf16
    router: torch.Tensor  # [N / W, E] fp32, this rank's router K rows
    pre_expert_norm: torch.Tensor  # [Ne] bf16
    router_bias: torch.Tensor  # [E] fp32


def prepare_boundary_weights(
    *,
    pre_expert: torch.Tensor,
    router: torch.Tensor,
    pre_expert_norm: torch.Tensor,
    router_bias: torch.Tensor,
    rank: int,
    world: int,
) -> BoundaryWeights:
    """One-time per layer: ``router`` is the full HF ``[E, N]`` FP32 weight,
    ``pre_expert`` the full ``[Ne, N]`` pooler."""

    n_local = pre_expert.size(0) // world
    k_local = router.size(1) // world
    return BoundaryWeights(
        # [N/W/P_ROWS, SPLIT_P, P_ROWS, K/SPLIT_P]: each P1 task's tile is dense.
        pooler=pre_expert[rank * n_local : (rank + 1) * n_local]
        .reshape(n_local // helpers.P_ROWS, helpers.P_ROWS, helpers.SPLIT_P, -1)
        .permute(0, 2, 1, 3)
        .contiguous(),
        router=router[:, rank * k_local : (rank + 1) * k_local]
        .t()
        .float()
        .contiguous(),
        pre_expert_norm=pre_expert_norm.reshape(-1).bfloat16().contiguous(),
        router_bias=router_bias.reshape(-1).float().contiguous(),
    )


@dataclasses.dataclass(frozen=True)
class LayerMkScalars:
    """Checkpoint scalars compiled in as constants (one kernel per distinct set)."""

    attn_scale: float
    attn_norm_eps: float
    boundary_post_eps: float  # both residual boundaries share one post / pre eps pair
    boundary_pre_eps: float
    pre_expert_eps: float
    router_multiplier: float
    route_scale: float
    route_eps: float
    post_expert_eps: float


@dataclasses.dataclass(frozen=True)
class LayerMkWeights:
    """One layer's operands: views of the model's weights, the router planes, and (TP8 family) the layer's
    expert prefetch state."""

    wo: torch.Tensor  # [H / O_BM, KH / O_BK, O_BM, O_BK] view of the rank's o_proj
    alpha: torch.Tensor  # attention boundary vectors as [S_P, H / S_P] views
    beta: torch.Tensor
    pre_w: torch.Tensor
    wp: torch.Tensor  # prepare_boundary_weights' pooler
    # The router's bf16 hi / lo planes (_router_planes); TP8 family: wr_hi stacks
    # both (_stacked_router_planes) and the kernel reads only wr_lo's shape
    wr_hi: torch.Tensor
    wr_lo: torch.Tensor
    q_pre_w: torch.Tensor  # pre-expert norm [He] bf16
    router_bias: torch.Tensor  # [E] fp32
    # W13: (128-row block, K step, K half, row, byte) view of FI's NVFP4 W13; bf16 experts: FI's BlockMajorK
    # [E, H / 64, 2I / W, 64] as [E, H / 64, 2I / W / 128, 128, 64]
    weights_g: torch.Tensor
    # W2: (cell, K pair, K step, row, byte or bf16 column) view of the bank's cells
    weights2: torch.Tensor
    scales: torch.Tensor  # NVFP4 block scales (bf16 experts: an unread placeholder)
    up_alpha: torch.Tensor
    gate_alpha: torch.Tensor
    out_alpha: torch.Tensor
    norm_weight: torch.Tensor  # post-expert norm [He] bf16
    # [L / U_BM, He / U_BK, U_BM, U_BK] view of the rank's unpooler rows
    wu4: torch.Tensor
    talpha: torch.Tensor  # FFN boundary vectors as [W, L / C, C] views
    tbeta: torch.Tensor
    tpost_w: torch.Tensor
    tpre_w: torch.Tensor
    # The NEXT layer's QKV as a [NQ / Q_BM, H / Q_BK, Q_BM, Q_BK] view (last layer: a placeholder)
    wq4: torch.Tensor
    # TP8 family: the bank as 32 KiB prefetch boxes (W13, W2, block scales) and the layer's hot-expert list,
    # which every launch of the layer reads and rewrites on device (allocated outside graph capture).
    pf_w13: torch.Tensor | None = None
    pf_w2: torch.Tensor | None = None
    pf_sf: torch.Tensor | None = None
    pf_list: torch.Tensor | None = None


def _router_planes(
    router: torch.Tensor, world: int, split_p: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rank's K-major fp32 router shard [H / W, E] as bf16 hi / lo planes [S_P / W, E, H / S_P]."""
    s_own = split_p // world
    k_p = router.size(0) * world // split_p
    w = router.t().reshape(router.size(1), s_own, k_p).permute(1, 0, 2).contiguous()
    hi = w.bfloat16()
    return hi, (w - hi.float()).bfloat16()


def _stacked_router_planes(
    router: torch.Tensor, world: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_router_planes`` stacked per 64-expert chunk as [S_P / W, E / 64, 128,
    H / S_P] (64 hi rows, then 64 lo rows), and the lo plane (unread by the
    kernel)."""
    hi, lo = _router_planes(router, world, helpers.SPLIT_P)
    s_own, experts, k_p = hi.shape
    stacked = torch.stack(
        [
            hi.view(s_own, experts // 64, 64, k_p),
            lo.view(s_own, experts // 64, 64, k_p),
        ],
        dim=2,
    )
    return stacked.reshape(s_own, experts // 64, 128, k_p).contiguous(), lo


def prepare_layer_weights(
    *,
    family: LayerFamily,
    o_proj: torch.Tensor,  # [H, KH] bf16, this rank's input columns
    attn_boundary: dict,  # jit_kernel_kwargs() of the attention boundary
    ffn_boundary: dict,  # jit_kernel_kwargs() of the FFN boundary (its pre-norm is the next layer's)
    prepared_boundary: BoundaryWeights,  # prepare_boundary_weights(...)
    bank: Nvfp4VerifyBank | Bf16VerifyBank,
    up_alpha: torch.Tensor,  # bf16 experts: ones
    gate_alpha: torch.Tensor,
    out_alpha: torch.Tensor,
    post_expert_norm: torch.Tensor,
    unpooler: torch.Tensor,  # [H, He] bf16, replicated
    # [NQ, H] bf16, the next layer's; None on the model's last layer (final norm)
    next_qkv: torch.Tensor | None,
    rank: int,
    world: int,
    expert_bf16: bool = False,
) -> LayerMkWeights:
    if expert_bf16 != isinstance(bank, Bf16VerifyBank):
        raise TypeError(f"expert_bf16={expert_bf16} with a {type(bank).__name__}")
    q_bm, q_bk = family.q_tile
    dims = LayerDims.of(world)
    local = dims.local
    n_blocks = local // TAIL_COLUMNS

    def v3(t: torch.Tensor) -> torch.Tensor:
        return t.view(helpers.SPLIT_P, HIDDEN // helpers.SPLIT_P)

    def cols(t: torch.Tensor) -> torch.Tensor:
        return t.view(world, n_blocks, TAIL_COLUMNS)

    w2 = bank.weights2
    if family.stacked_router:
        wr_hi, wr_lo = _stacked_router_planes(prepared_boundary.router, world)
    else:
        wr_hi, wr_lo = _router_planes(prepared_boundary.router, world, helpers.SPLIT_P)
    weights_g = (
        bank.weights.unflatten(2, (-1, 128))
        if expert_bf16
        else helpers.fi_w13_blocks(bank.weights)
    )
    weights2 = w2.unflatten(3, (2, w2.size(3) // 2)).permute(0, 1, 3, 2, 4)
    device = o_proj.device
    scales = (
        torch.zeros(1, dtype=torch.uint8, device=device) if expert_bf16 else bank.scales
    )
    if next_qkv is None:  # the final-norm image never reads the next QKV weight
        wq4 = torch.zeros(1, 1, q_bm, q_bk, dtype=torch.bfloat16, device=device)
    else:
        wq4 = next_qkv.view(dims.qkv_width // q_bm, q_bm, HIDDEN // q_bk, q_bk).permute(
            0, 2, 1, 3
        )
    prefetch = {}
    if family.expert_prefetch:
        if expert_bf16:
            # The bf16 images issue no expert prefetch (their PF roots are dead): unread placeholders.
            box = torch.zeros(1, 1, 128, 256, dtype=torch.uint8, device=device)
            pf_w13, pf_w2, pf_sf = box, box, box[0]
        else:
            pf_w13, pf_w2, pf_sf = _module(family.kernel).prefetch_views(
                weights_g, weights2, bank.scales, up_alpha.numel(), world
            )
        prefetch = {
            "pf_w13": pf_w13,
            "pf_w2": pf_w2,
            "pf_sf": pf_sf,
            "pf_list": torch.full(
                (PF_LIST_SLOTS,), -1, dtype=torch.int32, device=device
            ),
        }
    return LayerMkWeights(
        wo=o_proj.view(HIDDEN // O_BM, O_BM, o_proj.size(1) // O_BK, O_BK).permute(
            0, 2, 1, 3
        ),
        alpha=v3(attn_boundary["alpha"]),
        beta=v3(attn_boundary["beta"]),
        pre_w=v3(attn_boundary["pre_weight"]),
        wp=prepared_boundary.pooler,
        wr_hi=wr_hi,
        wr_lo=wr_lo,
        q_pre_w=prepared_boundary.pre_expert_norm,
        router_bias=prepared_boundary.router_bias,
        weights_g=weights_g,
        weights2=weights2,
        scales=scales,
        up_alpha=up_alpha,
        gate_alpha=gate_alpha,
        out_alpha=out_alpha,
        norm_weight=post_expert_norm,
        wu4=unpooler[rank * local : (rank + 1) * local]
        .view(
            local // helpers.U_BM,
            helpers.U_BM,
            EXPERT_HIDDEN // helpers.U_BK,
            helpers.U_BK,
        )
        .permute(0, 2, 1, 3),
        talpha=cols(ffn_boundary["alpha"]),
        tbeta=cols(ffn_boundary["beta"]),
        tpost_w=cols(ffn_boundary["post_weight"]),
        tpre_w=cols(ffn_boundary["pre_weight"]),
        wq4=wq4,
        **prefetch,
    )


def entry_weights(
    base: LayerMkWeights,
    *,
    family: LayerFamily,
    qkv: torch.Tensor,  # [NQ, H] bf16: layer 0's qkv_proj (this rank's output rows)
    input_norm: torch.Tensor,  # [H]: layer 0's input RMSNorm weight (effective)
    world: int,
) -> LayerMkWeights:
    """The entry image's operands: a layer's prepared weights (views) for the dead roots, with the next-QKV roots
    pointed at layer 0's QKV and input norm weight."""
    dims = LayerDims.of(world)
    q_bm, q_bk = family.q_tile
    wq4 = qkv.view(dims.qkv_width // q_bm, q_bm, HIDDEN // q_bk, q_bk).permute(
        0, 2, 1, 3
    )
    tpre_w = input_norm.to(base.tpre_w.dtype).contiguous().view(base.tpre_w.shape)
    return dataclasses.replace(base, tpre_w=tpre_w, wq4=wq4)


def _entry_placeholders(
    device: torch.device, dims: LayerDims, tokens: int
) -> tuple[torch.Tensor, AttentionInputs]:
    """The entry image's unread QKV and attention inputs (one page of KV cache): its dead roots' operands."""
    page = torch.zeros(
        1, PAGE_SIZE * dims.kv_width, dtype=torch.float8_e4m3fn, device=device
    )
    attention = AttentionInputs(
        k_cache=page,
        v_cache=page.clone(),
        cache_loc=torch.zeros(tokens, dtype=torch.int64, device=device),
        block_tables=torch.zeros(1, 1, dtype=torch.int32, device=device),
        seq_lens=torch.full((1,), tokens, dtype=torch.int32, device=device),
        positions=torch.arange(tokens, device=device),
        cos_sin_cache=torch.zeros(1, 2 * 32, dtype=torch.float32, device=device),
    )
    return torch.zeros(
        tokens, dims.qkv_width, dtype=torch.bfloat16, device=device
    ), attention


def _workspaces(
    *,
    family: LayerFamily,
    dims: LayerDims,
    tokens: int,
    group: dist.ProcessGroup,
    device: torch.device,
    sms: int,
    global_attn: bool,
    expert_bf16: bool,
    final_norm: bool,
) -> tuple[dict[str, torch.Tensor], helpers.State]:
    """Everything the kernel reads or writes besides the per-layer weights and per-call tensors, and the MoE
    state."""
    group_name, world, T = group.group_name, dims.world, tokens
    q_bm, q_bk = family.q_tile
    split_p = helpers.SPLIT_P
    k_p, local, n_local = HIDDEN // split_p, dims.local, dims.n_local
    n_blocks = local // TAIL_COLUMNS

    def symm(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        t = symm_mem.empty(*shape, dtype=dtype, device=device)
        symm_mem.rendezvous(t, group_name)
        t.zero_()
        return t

    def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(*shape, dtype=dtype, device=device)

    rows = (dims.kv_heads, family.attention_splits, T, dims.heads_per_kv)
    x_rows = family.moe_rows(global_attn=global_attn, expert_bf16=expert_bf16)
    assert x_rows >= T, f"a {T}-token window in {x_rows} MoE token rows"
    moe = helpers.State(device, group, tokens=T, expert_bf16=expert_bf16, rows=x_rows)
    s_u = helpers.parts_for(local // helpers.U_BM, EXPERT_HIDDEN // helpers.U_BK, sms)
    s_q = family.qkv_splits or helpers.parts_for(
        dims.qkv_width // q_bm, HIDDEN // q_bk, sms
    )
    # The final-norm image's next-QKV operands: placeholders matching prepare_layer_weights' one-block wq4.
    q_part = zeros(1, 2 * T, q_bm) if final_norm else zeros(s_q, 2 * T, dims.qkv_width)
    # The bf16 images' expert input: the normed bf16 rows, unit row scales (NVFP4: e2m1 bytes and the token scales).
    if expert_bf16:
        x_q, x_ts = (
            zeros(x_rows, EXPERT_HIDDEN, dtype=torch.bfloat16),
            torch.ones(x_rows, device=device),
        )
    else:
        x_q, x_ts = (
            zeros(x_rows, EXPERT_HIDDEN // 2, dtype=torch.uint8),
            zeros(x_rows),
        )
    buffers = {
        "out": zeros(helpers.PAD_ROWS, dims.q_width, dtype=torch.bfloat16),
        "o_loc": zeros(helpers.PAD_ROWS, HIDDEN, dtype=torch.bfloat16),
        "opart": symm(T, split_p, k_p, dtype=torch.bfloat16),
        "branch": zeros(T, split_p, k_p, dtype=torch.bfloat16),
        "moments": zeros(split_p, T, 4),
        "rstd": zeros(T, 2),
        "new_stream": zeros(T, split_p, k_p),
        "act": zeros(2 * T, split_p, k_p, dtype=torch.bfloat16),
        "part_p": zeros(split_p, n_local // helpers.P_ROWS, helpers.P_ROWS, 2, T),
        "part_r": zeros(helpers.router_parts(world), 2 * T, ROUTER_E),
        "pooled": symm(T, n_local, dtype=torch.bfloat16),
        "logit_q": symm(T, ROUTER_E, dtype=torch.float32),
        "x_q": x_q,
        "x_sf": zeros(x_rows, EXPERT_HIDDEN // 16, dtype=torch.uint8),
        "x_ts": x_ts,
        "packed": zeros(T, TOPK, dtype=torch.int32),
        "topk_ids": zeros(T, TOPK, dtype=torch.int32),
        "z": moe.z,
        "z_scale": moe.z_scale,
        "gemm2_ws": moe.gemm2_ws,
        "sumsq_ws": moe.sumsq_ws,
        "y16": moe.y16,
        "u_part": zeros(s_u, MOE_TOKENS, local),
        "scaled_rows": symm(T, n_blocks, TAIL_COLUMNS, dtype=torch.bfloat16),
        "tmoments": symm(n_blocks, T, 4, dtype=torch.float32),
        "trstd": zeros(T, 2),
        "qact": zeros(2 * T, world, n_blocks, TAIL_COLUMNS, dtype=torch.bfloat16),
        "q_part": q_part,
        "att_ws": zeros(*rows, HEAD_DIM),
        "lse_ws": zeros(*rows),
    }
    return buffers, moe


class LayerVerifyMegakernel:
    """One rank's compiled layer megakernel and the workspaces every layer shares:
    layers run in stream order, and the LL epochs advance on device."""

    def __init__(
        self,
        *,
        device: torch.device,
        group: dist.ProcessGroup,
        scalars: LayerMkScalars,
        tokens: int,
        global_attn: bool = False,
        expert_bf16: bool = False,  # the bf16-expert image (the checkpoint's first and last layers)
        final_norm: bool = False,  # the model's last layer: ends at the final norm, no next QKV
        entry: bool = False,  # the model's prelude: embedding norm + layer 0's input norm and QKV
        family: LayerFamily | None = None,  # default: the served build for (world, T)
    ) -> None:
        world = group.size()
        self._group_name, self._rank, self._world = (
            group.group_name,
            group.rank(),
            world,
        )
        self._tokens, self._scalars, self._global = tokens, scalars, global_attn
        self._expert_bf16, self._final_norm, self._entry = (
            expert_bf16,
            final_norm,
            entry,
        )
        self._dims = dims = LayerDims.of(world)
        self._family = family = family or layer_family(world, tokens)
        self._split_p = helpers.SPLIT_P
        self._n_blocks = dims.local // TAIL_COLUMNS
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        if sms < WORKERS:
            raise RuntimeError(
                f"the layer megakernel needs {WORKERS} SMs for its workers, the device has {sms}"
            )
        self._buffers, self._moe_state = _workspaces(
            family=family,
            dims=dims,
            tokens=tokens,
            group=group,
            device=device,
            sms=sms,
            global_attn=global_attn,
            expert_bf16=expert_bf16,
            final_norm=final_norm,
        )
        self.topk_ids = self._buffers["topk_ids"]
        module = _module(family.kernel)
        # The launch's arguments, in the kernel's parameter order.
        self._order = tuple(inspect.signature(module.KERNEL).parameters)
        # The entry image reads no QKV and no attention inputs: placeholders its build specializes on.
        self._entry_inputs = (
            _entry_placeholders(device, dims, tokens) if entry else None
        )
        config = load_config(family, global_attn, expert_bf16, final_norm, entry)
        self._kernel = helion.kernel(
            module.KERNEL,
            config=helion.Config(**config),
            static_shapes=False,
            backend="gluon",
            dot_precision="tf32x3",
            persistent_reserved_sms=sms - WORKERS,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )

    def _outputs(self, qkv: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each launch's own outputs: the next layer's QKV and the new stream; the final-norm image returns the LM
        head's input in qact's first T rows instead (its qkv_g is an unwritten placeholder)."""
        T, device = self._tokens, qkv.device
        outputs = {
            "stream_next": torch.empty(T, HIDDEN, dtype=torch.float32, device=device)
        }
        if self._final_norm:
            outputs.update(
                qkv_next=torch.empty(
                    T,
                    self._family.q_tile[0],
                    dtype=torch.bfloat16,
                    device=device,
                ),
                qact=torch.empty(
                    2 * T,
                    self._world,
                    self._n_blocks,
                    TAIL_COLUMNS,
                    dtype=torch.bfloat16,
                    device=device,
                ),
            )
        else:
            outputs.update(
                qkv_next=torch.empty(
                    T, self._dims.qkv_width, dtype=torch.bfloat16, device=device
                )
            )
        return outputs

    def _args(
        self,
        *,
        qkv: torch.Tensor | None,
        stream: torch.Tensor,
        weights: LayerMkWeights,
        attention: AttentionInputs | None,
    ) -> tuple[list[object], dict[str, torch.Tensor]]:
        """One call's kernel arguments and its outputs (the entry image's QKV and attention inputs are its own
        placeholders)."""
        if self._entry_inputs is not None:
            qkv, attention = self._entry_inputs
        assert qkv is not None and attention is not None
        outputs = self._outputs(qkv)
        T, sc, world = self._tokens, self._scalars, self._world
        k_cache, v_cache = attention.k_cache, attention.v_cache
        attn_args = {
            "positions": attention.positions,
            "cos_sin_cache": attention.cos_sin_cache,
            # Flat views of the paged caches; the page strides are compile-time constants.
            "k_cache": torch.as_strided(
                k_cache,
                ((k_cache.size(0) - 1) * k_cache.stride(0) + k_cache.size(1),),
                (1,),
            ),
            "v_cache": torch.as_strided(
                v_cache,
                ((v_cache.size(0) - 1) * v_cache.stride(0) + v_cache.size(1),),
                (1,),
            ),
            "cache_loc": attention.cache_loc,
            "block_tables": attention.block_tables,
            "seq_lens": attention.seq_lens,
            "page_size": PAGE_SIZE,
            "k_page_stride": k_cache.stride(0),
            "v_page_stride": v_cache.stride(0),
            "scaled_log2": sc.attn_scale * LOG2E,
            # Global layers are NoPE and attend to the whole context (window_left < 0).
            "use_rope": not self._global,
            "window_left": -1 if self._global else SWA_WINDOW_LEFT,
            "key_tile": KEY_TILE,
        }
        args = dict(
            self._buffers,
            **{f.name: getattr(weights, f.name) for f in dataclasses.fields(weights)},
            **attn_args,
            qkv=qkv,
            stream=stream.view(T, self._split_p, HIDDEN // self._split_p),
            qkv_g=outputs["qkv_next"],
            new_stream_g=outputs["stream_next"].view(
                T, world, self._n_blocks, TAIL_COLUMNS
            ),
            eps=sc.attn_norm_eps,
            group_name=self._group_name,
            rank=self._rank,
            world=world,
            post_eps=sc.boundary_post_eps,
            pre_eps=sc.boundary_pre_eps,
            quant_eps=sc.pre_expert_eps,
            global_scale_inv=GLOBAL_SCALE_INV,
            router_multiplier=sc.router_multiplier,
            route_scale=sc.route_scale,
            route_eps=sc.route_eps,
            norm_eps=sc.post_expert_eps,
            expert_bf16=self._expert_bf16,
            final_norm=self._final_norm,
            entry=self._entry,
        )
        if "qact" in outputs:
            args["qact"] = outputs["qact"]
        return [args[n] for n in self._order], outputs

    def __call__(
        self,
        *,
        qkv: torch.Tensor | None,  # [T, NQ] bf16, this layer's (entry: None)
        stream: torch.Tensor,  # [T, H] fp32 (entry: this rank's masked bf16 embedding rows, others zero)
        weights: LayerMkWeights,
        attention: AttentionInputs | None,  # (entry: None)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (the next layer's QKV [T, NQ] bf16, the new stream [T, H] fp32); the final-norm image returns
        (the final-normed hidden state [T, H] bf16, the new stream); the entry image (layer 0's QKV, layer 0's input
        stream)."""
        args, outputs = self._args(
            qkv=qkv, stream=stream, weights=weights, attention=attention
        )
        self._kernel(*args)
        if self._final_norm:
            return outputs["qact"][: self._tokens].view(self._tokens, HIDDEN), outputs[
                "stream_next"
            ]
        return outputs["qkv_next"], outputs["stream_next"]


# ---- main(): random weights, FP64 stage references, determinism, timing ----

# The verify window serving runs the layer megakernel at.
TOKENS = 8
# Checkpoint scalars (the served model's).
SCALARS = LayerMkScalars(
    attn_scale=43.7840518911 / 64,
    attn_norm_eps=1e-5,
    boundary_post_eps=1e-6,
    boundary_pre_eps=1e-6,
    pre_expert_eps=1e-5,
    router_multiplier=1.0 / math.sqrt(32.0),
    route_scale=1.0,
    route_eps=1e-15,
    post_expert_eps=1e-8,
)
# The checkpoint's layer kinds: (global_attn, expert_bf16, final_norm).
IMAGES = {
    "sliding window": (False, False, False),
    "global": (True, False, False),
    "bf16 experts": (False, True, False),
    "final norm": (True, True, True),
}
# Context lengths of the checks: one inside the sliding window, one past it.
CHECK_CONTEXTS = (640, 3000)
# Routes concentrate on a pool of experts, as real routing does.
EXPERT_POOL = 40
F64 = torch.float64


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(F64)


def _f(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(F64)


def _gather(t: torch.Tensor) -> list[torch.Tensor]:
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, t.contiguous())
    return parts


def _rank_max(t: torch.Tensor) -> torch.Tensor:
    """``t``'s elementwise max over the ranks, on every rank."""
    return torch.stack(_gather(t.cuda())).amax(0)


class _Report:
    """Per stage, the max |error| relative to the reference's max |value| and
    the share of elements off (max over ranks): beyond ``tol`` of that max, or
    of their own |value| plus ``floor``, or (``exact``) unequal; an empty stage
    is all off. A stage fails when more are off than it allows (none unless
    given)."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, float, float, str, float]] = []

    def add(
        self,
        name: str,
        got: torch.Tensor,
        ref: torch.Tensor,
        tol: float,
        floor: float | None = None,
    ) -> None:
        bound = f"tol {tol:.0e}" + ("" if floor is None else f" |ref| + {floor:.0e}")
        if ref.numel() == 0:  # e.g. no rows where the kernel's routing should have some
            self.rows.append((name, float("nan"), 1.0, bound, 0.0))
            return
        got, ref = got.to(F64), ref.to(F64)
        scale = ref.abs().max().clamp_min(1e-30)
        err = (got - ref).abs()
        limit = tol * scale if floor is None else tol * ref.abs() + floor
        # A NaN on either side counts as off.
        off = float((~(err <= limit)).double().mean())
        self.rows.append((name, float(err.max() / scale), off, bound, 0.0))

    def exact(
        self, name: str, got: torch.Tensor, ref: torch.Tensor, allowed: float = 0.0
    ) -> None:
        off = float((got != ref).double().mean()) if ref.numel() else 1.0
        bound = "exact" + (f", {allowed:.1%} may differ" if allowed else "")
        self.rows.append((name, off, off, bound, allowed))

    def finish(self, title: str, verbose: bool) -> None:
        """Prints the stage table on rank 0 if ``verbose`` or a stage failed."""
        values = torch.tensor([[r[1], r[2]] for r in self.rows], dtype=F64)
        worst = _rank_max(values).tolist()
        rows = list(zip(self.rows, worst, strict=True))
        failed = [r[0] for r, (_, off) in rows if off > r[4]]
        if dist.get_rank() == 0 and (verbose or failed):
            print(f"{title}: rel = max |err| / max |ref|, off = elements beyond tol")
            for (name, _, _, bound, _), (rel, off) in rows:
                print(f"  {name:24s} rel {rel:9.2e}  off {off:7.2%}  ({bound})")
        if failed:
            raise AssertionError(f"{title}: stages off their FP64 references: {failed}")


def _rope_cache(device: torch.device) -> torch.Tensor:
    exponent = torch.arange(0, HEAD_DIM, 2, dtype=F64) / HEAD_DIM
    freqs = torch.outer(torch.arange(1 << 17, dtype=F64), 1.0 / (10000.0**exponent))
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).float().to(device)


def _attention_inputs(
    dims: LayerDims, seq_len: int, seed: int, cos_sin: torch.Tensor
) -> tuple[AttentionInputs, torch.Tensor]:
    """A paged FP8 KV cache of ``seq_len - T`` tokens (RMS-normed keys, pages
    in random order) and this layer's QKV rows. The pool holds the longest
    check context, so every context binds one kernel."""
    device = cos_sin.device
    g = torch.Generator(device=device).manual_seed(seed)
    pages = -(-seq_len // PAGE_SIZE)
    pool_pages = -(-max(CHECK_CONTEXTS) // PAGE_SIZE) + 16
    keys = torch.randn(
        pool_pages * PAGE_SIZE, dims.kv_width, generator=g, device=device
    )
    rows = keys.view(-1, HEAD_DIM)
    keys = rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + 1e-5)
    values = torch.randn(
        pool_pages * PAGE_SIZE, dims.kv_width, generator=g, device=device
    )
    block_tables = torch.zeros(
        1, (1 << 17) // PAGE_SIZE, dtype=torch.int32, device=device
    )
    perm = torch.randperm(pool_pages - 1, generator=g, device=device)[:pages] + 1
    block_tables[0, :pages] = perm.to(torch.int32)
    positions = torch.arange(seq_len - TOKENS, seq_len, device=device)
    cache_loc = (
        block_tables[0, positions // PAGE_SIZE].long() * PAGE_SIZE
        + positions % PAGE_SIZE
    )
    fp8 = torch.float8_e4m3fn
    attention = AttentionInputs(
        k_cache=keys.bfloat16().to(fp8).view(pool_pages, -1),
        v_cache=(values * 0.25).bfloat16().to(fp8).view(pool_pages, -1),
        cache_loc=cache_loc,
        block_tables=block_tables,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        positions=positions,
        cos_sin_cache=cos_sin,
    )
    qkv = torch.randn(TOKENS, dims.qkv_width, generator=g, device=device)
    return attention, qkv.bfloat16()


def _attention_reference(
    qkv: torch.Tensor, attention: AttentionInputs, dims: LayerDims, global_attn: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP64 attention + output norm and gate [T, q heads x 64] (bf16 values),
    with the kernel's rounding points: the QK norm and RoPE round to bf16, then
    the clamped FP8 the cache holds; probabilities round to fp16. Also the new
    tokens' K and V rows the cache should hold [T, KV width] (FP8 values)."""
    hd, kvw, qw, T = HEAD_DIM, dims.kv_width, dims.q_width, TOKENS
    hpk, eps = dims.heads_per_kv, SCALARS.attn_norm_eps
    window = -1 if global_attn else SWA_WINDOW_LEFT
    pos, cs = attention.positions, attention.cos_sin_cache
    seq = int(attention.seq_lens[0])
    cache_end = seq - T
    first = max(cache_end - window, 0) if window >= 0 else 0
    kpos = torch.arange(first, cache_end, device=qkv.device)
    pages = attention.block_tables[0, kpos // PAGE_SIZE].long()
    k_cache = attention.k_cache.view(attention.k_cache.size(0), PAGE_SIZE, kvw)
    v_cache = attention.v_cache.view(attention.v_cache.size(0), PAGE_SIZE, kvw)
    x = qkv.float()

    def prep(raw: torch.Tensor) -> torch.Tensor:
        var = (raw * raw).sum(-1, keepdim=True) / hd
        nrm = (raw * torch.rsqrt(var + eps)).to(torch.bfloat16).float()
        if not global_attn:
            cos, sin = cs[pos, : hd // 2][:, None, :], cs[pos, hd // 2 :][:, None, :]
            ev, od = nrm[..., 0::2], nrm[..., 1::2]
            rot = torch.stack((ev * cos - od * sin, ev * sin + od * cos), -1)
            nrm = rot.flatten(-2).to(torch.bfloat16).float()
        return nrm.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(F64)

    out = torch.empty(T, dims.q_heads, hd, dtype=F64, device=qkv.device)
    k_rows = torch.empty(T, kvw, dtype=F64, device=qkv.device)
    v_rows = torch.empty(T, kvw, dtype=F64, device=qkv.device)
    qpos = cache_end + torch.arange(T, device=qkv.device)
    kp = torch.cat((kpos, qpos))
    valid = kp[None, :] <= qpos[:, None]
    if window >= 0:
        valid &= qpos[:, None] - kp[None, :] <= window
    for g in range(dims.kv_heads):
        q = prep(x[:, g * hpk * hd : (g + 1) * hpk * hd].view(T, hpk, hd))
        k_new = prep(x[:, qw + g * hd : qw + (g + 1) * hd].view(T, 1, hd))[:, 0]
        v_new = x[:, qw + kvw + g * hd : qw + kvw + (g + 1) * hd]
        v_new = v_new.clamp(-448, 448).to(torch.float8_e4m3fn).to(F64)
        cached = slice(g * hd, (g + 1) * hd)
        k_rows[:, cached], v_rows[:, cached] = k_new, v_new
        keys = torch.cat((k_cache[pages, kpos % PAGE_SIZE, cached].to(F64), k_new))
        vals = torch.cat((v_cache[pages, kpos % PAGE_SIZE, cached].to(F64), v_new))
        s = torch.einsum("thd,kd->thk", q, keys) * SCALARS.attn_scale * LOG2E
        s = torch.where(valid[:, None, :], s, float("-inf"))
        p = torch.exp2(s - s.amax(-1, keepdim=True))
        o = torch.einsum("thk,kd->thd", p.to(torch.float16).to(F64), vals)
        attn = _bf(o / p.sum(-1, keepdim=True))
        var = (attn * attn).sum(-1, keepdim=True) / hd
        gate = x[:, qw + 2 * kvw + g * hpk * hd : qw + 2 * kvw + (g + 1) * hpk * hd]
        gate = gate.view(T, hpk, hd).to(F64)
        out[:, g * hpk : (g + 1) * hpk] = _bf(
            attn * torch.rsqrt(var + eps) / (1.0 + torch.exp(-gate))
        )
    return out.view(T, -1), k_rows, v_rows


def _fi_shuffle(
    gate_up: torch.Tensor, down: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint-order bf16 expert slices [E, 2I/W, H] ([gate; up]) and
    [E, H, I/W] in FlashInfer's shuffled BlockMajorK layout, as serving loads
    them."""
    from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices
    from flashinfer.fused_moe.core import convert_to_block_layout
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    experts, two_local, hidden = gate_up.shape
    local = down.size(2)
    cache: dict[Any, Any] = {}
    bf16 = torch.bfloat16
    w13 = torch.empty(
        experts, hidden // 64, two_local, 64, dtype=bf16, device=gate_up.device
    )
    w2 = torch.empty(
        experts, local // 64, hidden, 64, dtype=bf16, device=gate_up.device
    )
    for i in range(experts):
        gate, up = gate_up[i].split(two_local // 2, dim=0)
        rows = torch.cat([up, gate], dim=0).view(torch.uint8)
        idx = _maybe_get_cached_w3_w1_permute_indices(cache, rows, 128)
        w13[i] = convert_to_block_layout(
            rows[idx.to(rows.device)].contiguous(), 128
        ).view(bf16)
        rows = down[i].view(torch.uint8)
        idx = get_w2_permute_indices_with_cache(cache, rows, 128)
        w2[i] = convert_to_block_layout(
            rows[idx.to(rows.device)].contiguous(), 128
        ).view(bf16)
    return w13, w2


@dataclasses.dataclass
class _Case:
    """One image on random weights: the host kernel, its prepared operands, and
    the full tensors the references read."""

    global_attn: bool
    expert_bf16: bool
    final_norm: bool
    host: LayerVerifyMegakernel
    weights: LayerMkWeights
    boundary: BoundaryWeights
    tensors: dict[str, torch.Tensor]


def _make_case(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    global_attn: bool = False,
    expert_bf16: bool = False,
    final_norm: bool = False,
    entry: bool = False,
    init: bool = True,
) -> _Case:
    """An image on random weights, its experts laid out by FlashInfer
    (``init=False``: uninitialized ones, not laid out, enough to bind and
    generate code)."""
    rank, world = group.rank(), group.size()
    family, dims = layer_family(world, TOKENS), LayerDims.of(world)

    def replicated(
        *shape: int,
        seed: int,
        scale: float = 1.0,
        shift: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if not init:
            return torch.empty(shape, dtype=dtype, device=device)
        g = torch.Generator().manual_seed(seed)
        value = torch.randn(*shape, generator=g) * scale + shift
        return value.to(dtype).to(device).contiguous()

    def own(
        *shape: int, seed: int, scale: float, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        if not init:
            return torch.empty(shape, dtype=dtype, device=device)
        g = torch.Generator(device=device).manual_seed(seed * 7 + rank)
        value = torch.randn(*shape, generator=g, device=device) * scale
        return value.to(dtype).contiguous()

    def gates(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        gate = replicated(HIDDEN, seed=seed)
        beta = torch.sigmoid(gate)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-gate) * (1.0 + beta), min=1e-3))
        return alpha.contiguous(), beta.contiguous()

    bf16 = torch.bfloat16
    # Each boundary its own vectors, so a kernel reading the other's fails.
    alpha, beta = gates(3)
    alpha_f, beta_f = gates(6)
    t = {
        "alpha": alpha,
        "beta": beta,
        "alpha_f": alpha_f,
        "beta_f": beta_f,
        "post_w": replicated(HIDDEN, seed=4, scale=0.1, shift=1.0, dtype=bf16),
        "pre_w": replicated(HIDDEN, seed=5, scale=0.1, shift=1.0, dtype=bf16),
        "pre_f": replicated(HIDDEN, seed=7, scale=0.1, shift=1.0, dtype=bf16),
        "wo": own(HIDDEN, dims.q_width, seed=100, scale=(dims.q_width * world) ** -0.5),
        "wu": replicated(
            HIDDEN, EXPERT_HIDDEN, seed=150, scale=EXPERT_HIDDEN**-0.5, dtype=bf16
        ),
        "wq": own(dims.qkv_width, HIDDEN, seed=170, scale=HIDDEN**-0.5),
        "pre_expert": replicated(
            EXPERT_HIDDEN, HIDDEN, seed=200, scale=HIDDEN**-0.5, dtype=bf16
        ),
        "router": replicated(ROUTER_E, HIDDEN, seed=300, scale=0.01),
        "norm_w": (1.0 + replicated(EXPERT_HIDDEN, seed=410, scale=0.02)).bfloat16(),
    }
    pool = torch.randperm(ROUTER_E, generator=torch.Generator().manual_seed(1040))
    bias = torch.full((ROUTER_E,), -1.0e4)
    bias[pool[:EXPERT_POOL]] = 4.0
    bias += 0.01 * torch.randn(ROUTER_E, generator=torch.Generator().manual_seed(7))
    boundary = prepare_boundary_weights(
        pre_expert=t["pre_expert"],
        router=t["router"],
        pre_expert_norm=(1.0 + replicated(EXPERT_HIDDEN, seed=400, scale=0.02)),
        router_bias=bias.to(device),
        rank=rank,
        world=world,
    )
    il = dims.intermediate
    if expert_bf16:
        # The rank's expert-TP slices in checkpoint order, shuffled as served.
        t["gate_up"] = own(
            ROUTER_E, 2 * il, EXPERT_HIDDEN, seed=11, scale=EXPERT_HIDDEN**-0.5
        )
        t["down"] = own(
            ROUTER_E, EXPERT_HIDDEN, il, seed=12, scale=EXPERT_INTERMEDIATE**-0.5
        )
        if init:
            w13, w2 = _fi_shuffle(t["gate_up"], t["down"])
        else:  # FlashInfer's BlockMajorK shapes
            w13 = t["gate_up"].view(ROUTER_E, EXPERT_HIDDEN // 64, 2 * il, 64)
            w2 = t["down"].view(ROUTER_E, il // 64, EXPERT_HIDDEN, 64)
        bank: Nvfp4VerifyBank | Bf16VerifyBank = build_bf16_verify_bank(
            w13=w13, w2=w2, group=group
        )
        for name in ("up_alpha", "gate_alpha", "out_alpha"):
            t[name] = torch.ones(ROUTER_E, device=device)
    else:
        # The rank's expert-TP slices in checkpoint order ([gate; up] W13 rows),
        # laid out as served: random e2m1 codes, an E4M3 scale of 0.5, 1 or 2
        # per block and per-expert alphas, so a kernel reading another block's
        # scale or another expert's alphas, or swapping the three alphas, fails.
        u8 = torch.uint8
        g = torch.Generator(device=device).manual_seed(11 + rank)
        g_e = torch.Generator().manual_seed(13)

        def codes(*shape: int, values: int = 256) -> torch.Tensor:
            if not init:
                return torch.empty(shape, dtype=u8, device=device)
            return torch.randint(values, shape, dtype=u8, device=device, generator=g)

        def scales(*shape: int) -> torch.Tensor:  # E4M3 0x30, 0x38, 0x40
            return 0x30 + 8 * codes(*shape, values=3) if init else codes(*shape)

        t["w13"] = codes(ROUTER_E, 2 * il, EXPERT_HIDDEN // 2)
        t["w13_scale"] = scales(ROUTER_E, 2 * il, EXPERT_HIDDEN // 16)
        t["w2"] = codes(ROUTER_E, EXPERT_HIDDEN, il // 2)
        t["w2_scale"] = scales(ROUTER_E, EXPERT_HIDDEN, il // 16)
        laid_out = (t["w13"], t["w13_scale"], t["w2"], t["w2_scale"])
        if init:  # FlashInfer's gated GEMM takes [w3 (up); w1 (gate)]
            up_gate = [torch.cat((w[:, il:], w[:, :il]), 1) for w in laid_out[:2]]
            laid_out = flashinfer_layout(*up_gate, t["w2"], t["w2_scale"])
        bank = build_nvfp4_verify_bank(
            w13=laid_out[0],
            w13_scale=laid_out[1],
            w2=laid_out[2],
            w2_scale=laid_out[3],
            group=group,
        )
        for name in ("up_alpha", "gate_alpha", "out_alpha"):
            alphas = 0.005 + 0.01 * torch.rand(ROUTER_E, generator=g_e)
            t[name] = alphas.to(device)
    weights = prepare_layer_weights(
        family=family,
        o_proj=t["wo"],
        attn_boundary={
            "alpha": t["alpha"],
            "beta": t["beta"],
            "pre_weight": t["pre_w"],
        },
        ffn_boundary={
            "alpha": t["alpha_f"],
            "beta": t["beta_f"],
            "pre_weight": t["pre_f"],
            "post_weight": t["post_w"],
        },
        prepared_boundary=boundary,
        bank=bank,
        up_alpha=t["up_alpha"],
        gate_alpha=t["gate_alpha"],
        out_alpha=t["out_alpha"],
        post_expert_norm=t["norm_w"],
        unpooler=t["wu"],
        next_qkv=None if final_norm else t["wq"],
        rank=rank,
        world=world,
        expert_bf16=expert_bf16,
    )
    if entry:  # the next-QKV roots project layer 0's QKV
        weights = entry_weights(
            weights, family=family, qkv=t["wq"], input_norm=t["pre_w"], world=world
        )
    host = LayerVerifyMegakernel(
        device=device,
        group=group,
        scalars=SCALARS,
        tokens=TOKENS,
        global_attn=global_attn,
        expert_bf16=expert_bf16,
        final_norm=final_norm,
        entry=entry,
    )
    return _Case(global_attn, expert_bf16, final_norm, host, weights, boundary, t)


def _check_launch(
    case: _Case,
    attention: AttentionInputs,
    qkv: torch.Tensor,
    stream: torch.Tensor,
    out: torch.Tensor,
    stream_next: torch.Tensor,
    title: str,
    verbose: bool,
) -> None:
    """Every stage of one launch against FP64 references computed from the
    kernel's own inputs to that stage, so each stage is held to its own
    rounding."""
    rank, world = dist.get_rank(), dist.get_world_size()
    dims, T, H, NE = LayerDims.of(world), TOKENS, HIDDEN, EXPERT_HIDDEN
    hb, t = case.host._buffers, case.tensors
    state = case.host._moe_state
    rep = _Report()
    # Attention, the O-projection partial and its all-reduce.
    att, k_rows, v_rows = _attention_reference(qkv, attention, dims, case.global_attn)
    rep.add("attention out", hb["out"][:T], att, 2**-7)
    # The new tokens' cache rows. K (normed and rotated, then FP8) within one
    # FP8 rounding flip of each element: an E4M3 step is at most 2**-3 of its
    # value, 2**-9 below the normals. V, the clamped FP8 of the input rows,
    # exactly.
    stored = {
        name: cache.view(-1, dims.kv_width)[attention.cache_loc].to(F64)
        for name, cache in (("K", attention.k_cache), ("V", attention.v_cache))
    }
    rep.add("KV store (K)", stored["K"], k_rows, 2**-3, floor=2**-9)
    rep.exact("KV store (V)", stored["V"], v_rows)
    o_part = hb["opart"].reshape(T, H)
    rep.add(
        "o_proj partial", o_part, _bf(_f(hb["out"][:T]) @ t["wo"].to(F64).t()), 2**-7
    )
    branch = _f(hb["branch"].reshape(T, H))
    rep.add("AR branch", branch, _bf(sum(_f(p) for p in _gather(o_part))), 2**-8)
    # The attention boundary (post norm of the branch, pre-FFN norm).
    alpha, beta = _f(t["alpha"]), _f(t["beta"])
    post_w, pre_w = _f(t["post_w"]), _f(t["pre_w"])
    gated, gb = alpha * _f(stream), beta * branch
    post_r = torch.rsqrt((branch * branch).sum(-1) / H + SCALARS.boundary_post_eps)
    pre_var = (gated * gated).sum(-1) + 2 * post_r * (gated * gb).sum(-1)
    pre_var = (pre_var + post_r**2 * (gb * gb).sum(-1)) / H
    pre_r = torch.rsqrt(pre_var + SCALARS.boundary_pre_eps)
    act = hb["act"].reshape(2, T, H)
    rep.add("x_A / x_B", act, torch.stack((_bf(gated * pre_w), _bf(gb * pre_w))), 2**-7)
    rep.add(
        "boundary rstd (attention)", hb["rstd"], torch.stack((post_r, pre_r), 1), 1e-4
    )
    # The stream adds the branch normed by the kernel's own FP32 rstd and
    # rounded to bf16 (an FP64 rstd would round some elements the other way).
    normed_b = (hb["branch"].reshape(T, H).float() * hb["rstd"][:, :1]).bfloat16()
    stream_b = gated + beta * _f(normed_b)
    rep.add("new stream (attention)", hb["new_stream"].reshape(T, H), stream_b, 1e-5)
    # Router logits (this rank's K shard), top-8 and the pooler, from the
    # kernel's x_A / x_B.
    a_f, b_f = _f(act[0]), _f(act[1])
    rstd = torch.stack((post_r, pre_r), 1)
    # The checkpoint router (the kernel's bf16 hi + lo planes are within 2**-16).
    own_k = slice(rank * H // world, (rank + 1) * H // world)
    w_r = t["router"][:, own_k].to(F64)
    logit = rstd[:, 1:2] * (
        a_f[:, own_k] @ w_r.t() + rstd[:, 0:1] * (b_f[:, own_k] @ w_r.t())
    )
    rep.add("router logit partial", hb["logit_q"], logit, 1e-4)
    logits = sum(_f(p) for p in _gather(hb["logit_q"]))
    score = 1.0 / (1.0 + torch.exp(-logits * SCALARS.router_multiplier))
    bias = case.boundary.router_bias.to(F64)
    top = torch.topk(score + bias, TOPK, dim=-1).indices
    topk_ids = hb["topk_ids"].long()
    rep.exact("top-8 ids", topk_ids.sort(-1).values, top.sort(-1).values)
    packed = hb["packed"]
    # The MoE (and its reference below) reads the experts from packed, and the
    # weights checked next are packed's, in topk_ids' order.
    rep.exact("packed ids", (packed >> 16).long(), topk_ids)
    selected = torch.gather(score, 1, topk_ids)
    route_w = (
        selected
        * SCALARS.route_scale
        / (selected.sum(-1, keepdim=True) + SCALARS.route_eps)
    )
    weights_got = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16).to(F64)
    rep.add("top-8 weights", weights_got, _bf(route_w), 2**-7)
    wp = t["pre_expert"][rank * dims.n_local : (rank + 1) * dims.n_local].to(F64)
    pooled = _bf(rstd[:, 1:2] * (a_f @ wp.t() + rstd[:, 0:1] * (b_f @ wp.t())))
    rep.add("pooled (rank rows)", hb["pooled"], pooled, 2**-7)
    # The expert input: the gathered pooled row's pre-expert norm (NVFP4: quantized).
    row = torch.cat([_f(p) for p in _gather(hb["pooled"])], 1)
    q_w = _f(case.boundary.pre_expert_norm)
    normed = _bf(
        row
        * torch.rsqrt((row * row).sum(-1, keepdim=True) / NE + SCALARS.pre_expert_eps)
        * q_w
    )
    ids = (packed >> 16).long()
    # z slots: the routed experts, ascending; from the reference routing, the
    # same on every rank, so the collectives below match even if the kernel's
    # routing (checked above) is off.
    distinct = torch.unique(top)
    routed = (ids[None, :, :] == distinct[:, None, None]).any(-1)
    route_weight = _f((packed & 0xFFFF).to(torch.int16).view(torch.bfloat16))
    nw = _f(t["norm_w"])
    local = dims.n_local  # the rank's MoE output columns
    g2 = torch.zeros(T, TOPK, local, dtype=F64, device=row.device)
    own_rows = slice(rank * local, (rank + 1) * local)  # of W2
    if case.expert_bf16:
        rep.add("expert input (bf16)", hb["x_q"][:T], normed, 2**-7)
        x, il = _f(hb["x_q"][:T]), dims.intermediate
        z_got, z_ref = [], []
        for j, e in enumerate(distinct.tolist()):
            gate_up = torch.stack(_gather(t["gate_up"][e]))  # [W, 2 I / W, NE]
            gate_w = gate_up[:, :il].reshape(-1, NE).to(F64)
            up_w = gate_up[:, il:].reshape(-1, NE).to(F64)
            down_w = torch.cat(_gather(t["down"][e]), 1).to(F64)  # [NE, I]
            gate, up = x @ gate_w.t(), x @ up_w.t()
            h = _bf(gate / (1.0 + torch.exp(-gate)) * up)
            z_got.append(state.z[j, :T][routed[j]].to(F64))
            z_ref.append(h[routed[j]])
            rows_out = _bf(_f(state.z[j, :T]) @ down_w[own_rows].t())
            g2[ids == e] = rows_out[(ids == e).nonzero()[:, 0]]
        rep.add("z (gathered, bf16)", torch.cat(z_got), torch.cat(z_ref), 2**-7)
    else:
        x_q, x_sf, x_ts = hb["x_q"][:T], hb["x_sf"][:T], hb["x_ts"][:T].to(F64)
        # The quant against FlashInfer's recipe on the reference norm (the
        # kernel's FP32 norm rounds a rare element to bf16 the other way).
        x_codes, x_scales, token_scale = nvfp4_quantize(normed, GLOBAL_SCALE_INV)
        rep.add("token scale", x_ts, token_scale[:, 0], 1e-3)
        rep.exact("expert input codes", e2m1_values(x_q), x_codes, allowed=1e-3)
        rep.exact("expert input scales", x_sf, x_scales, allowed=1e-3)
        # W13 + SwiGLU from the kernel's expert input, the NVFP4 requant of z,
        # then W2 on the kernel's own z, all from the checkpoint-order experts.
        x, il, slots = dequantize_nvfp4(x_q, x_sf), dims.intermediate, len(distinct)
        ua, ga, oa = (t[n].to(F64) for n in ("up_alpha", "gate_alpha", "out_alpha"))
        z = torch.zeros(slots, T, il, dtype=F64, device=row.device)
        for j, e in enumerate(distinct.tolist()):
            w13 = dequantize_nvfp4(t["w13"][e], t["w13_scale"][e])
            acc = x @ w13.t() * x_ts[:, None]
            gate, up = acc[:, :il] * ga[e], acc[:, il:] * ua[e]
            z[j] = gate / (1.0 + torch.exp(-gate)) * up
        z_codes, z_scales = (torch.cat(_gather(r), -1) for r in nvfp4_quantize(z)[:2])
        z_q, z_sf = state.z[:slots, :T], state.z_scale[:slots, :T]
        z_got = e2m1_values(z_q)[routed]
        rep.exact("z codes (gathered)", z_got, z_codes[routed], allowed=1e-3)
        rep.exact("z scales (gathered)", z_sf[routed], z_scales[routed], allowed=1e-3)
        z_k = dequantize_nvfp4(z_q, z_sf)
        for j, e in enumerate(distinct.tolist()):
            # This rank's W2 rows over every rank's K slice.
            parts = zip(_gather(t["w2"][e]), _gather(t["w2_scale"][e]), strict=True)
            w2 = torch.cat(
                [dequantize_nvfp4(c[own_rows], s[own_rows]) for c, s in parts], 1
            )
            rows_out = _bf(z_k[j] @ w2.t() * oa[e])
            g2[ids == e] = rows_out[(ids == e).nonzero()[:, 0]]
    # The post-expert norm and the weighted route sum, over every rank's columns.
    g2_parts = _gather(g2)
    sumsq = sum(
        ((part * nw[r * local : (r + 1) * local]) ** 2).sum(-1)
        for r, part in enumerate(g2_parts)
    )
    scale = torch.rsqrt(sumsq / NE + SCALARS.post_expert_eps) * route_weight
    y_ref = _bf((torch.cat(g2_parts, -1) * nw * scale[..., None]).sum(1))
    rep.add("MoE y", state.y16[:T], y_ref, 2**-7)
    # The tail from the kernel's y: the unpooler shard, the FFN boundary and
    # the next layer's QKV (or the final norm).
    y = _f(state.y16[:T])
    shard = slice(rank * dims.local, (rank + 1) * dims.local)
    scaled = _bf(_bf(y @ t["wu"][shard].to(F64).t()) * post_w[shard])
    rep.add("tail scaled rows", hb["scaled_rows"].reshape(T, -1), scaled, 2**-7)
    rows_bf = torch.cat([r.reshape(T, -1) for r in _gather(hb["scaled_rows"])], 1)
    rows_all = _f(rows_bf)
    alpha_f, beta_f, pre_f = _f(t["alpha_f"]), _f(t["beta_f"]), _f(t["pre_f"])
    g_t, gb_t = alpha_f * _f(hb["new_stream"].reshape(T, H)), beta_f * rows_all
    post_t = torch.rsqrt((rows_all * rows_all).sum(-1) / H + SCALARS.boundary_post_eps)
    pv = (g_t * g_t).sum(-1) + 2 * post_t * (g_t * gb_t).sum(-1)
    pv = (pv + post_t**2 * (gb_t * gb_t).sum(-1)) / H
    pre_t = torch.rsqrt(pv.clamp_min(0) + SCALARS.boundary_pre_eps)
    rep.add("boundary rstd (FFN)", hb["trstd"], torch.stack((post_t, pre_t), 1), 1e-4)
    normed_t = (rows_bf.float() * hb["trstd"][:, :1]).bfloat16()
    new_stream = g_t + beta_f * _f(normed_t)
    rep.add("new stream (FFN)", stream_next, new_stream, 1e-5)
    if case.final_norm:
        # The model's final norm (here with the FFN boundary's pre-norm weight).
        x_f = _bf(new_stream)
        x_f = _bf(
            x_f
            * torch.rsqrt(
                (x_f * x_f).sum(-1, keepdim=True) / H + SCALARS.boundary_pre_eps
            )
            * pre_f
        )
        rep.add("final-normed hidden", out, x_f, 2**-7)
    else:
        qa, qb, wq = _bf(g_t * pre_f), _bf(gb_t * pre_f), t["wq"].to(F64)
        qkv_ref = _bf(pre_t[:, None] * (qa @ wq.t() + post_t[:, None] * (qb @ wq.t())))
        rep.add("next qkv", out, qkv_ref, 2**-7)
    rep.finish(title, verbose)


def _check_determinism(
    case: _Case, attention: AttentionInputs, qkv: torch.Tensor, stream: torch.Tensor
) -> torch.cuda.CUDAGraph:
    """Repeated calls and a CUDA graph replay reproduce the first call bitwise;
    returns the graph."""
    hb, state = case.host._buffers, case.host._moe_state

    def call() -> dict[str, torch.Tensor]:
        out, stream_next = case.host(
            qkv=qkv, stream=stream, weights=case.weights, attention=attention
        )
        return {
            "out": out,
            "stream": stream_next,
            "y": state.y16,
            "x": hb["x_q"],
            "topk": hb["topk_ids"],
        }

    first = {k: v.clone() for k, v in call().items()}

    def compare(tag: str, got: dict[str, torch.Tensor]) -> None:
        # Every rank raises together: one rank leaving would hang the others.
        torch.cuda.synchronize()
        differs = [not torch.equal(got[k], v) for k, v in first.items()]
        on_any = _rank_max(torch.tensor(differs, dtype=F64)).tolist()
        differ = [k for k, d in zip(first, on_any, strict=True) if d]
        if differ:
            raise AssertionError(f"{tag} differs in {differ} on some rank")

    for _ in range(2):
        compare("a repeated call", call())
    graph, replayed = _capture(call)
    for _ in range(3):
        graph.replay()
    compare("a graph replay", replayed)
    return graph


def _cold_l2_us(graph: torch.cuda.CUDAGraph, replays: int = 21) -> float:
    """Median over replays of the slowest rank's graph replay after a 512 MB L2
    scrub."""
    scrub = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    return float(
        torch.tensor([_cold_replay_us(graph, scrub) for _ in range(replays)]).median()
    )


def _run(
    group: dist.ProcessGroup, *, verbose: bool, check_only: bool
) -> dict[str, Any]:
    rank, world = group.rank(), group.size()
    device = torch.device("cuda", torch.cuda.current_device())
    cos_sin = _rope_cache(device)
    results: dict[str, Any] = {}
    # Each image's symmetric buffers stay allocated: torch finds a symmetric
    # handle by address, so a later image's buffer at a freed one's address
    # would get its stale handle.
    hosts: list[LayerVerifyMegakernel] = []
    for name, (global_attn, expert_bf16, final_norm) in IMAGES.items():
        case = _make_case(
            group,
            device,
            global_attn=global_attn,
            expert_bf16=expert_bf16,
            final_norm=final_norm,
        )
        stream = torch.randn(
            TOKENS, HIDDEN, generator=torch.Generator().manual_seed(2)
        ).to(device)
        for seq_len in CHECK_CONTEXTS:
            attention, qkv = _attention_inputs(
                LayerDims.of(world), seq_len, 1000 + 17 * rank, cos_sin
            )
            out, stream_next = case.host(
                qkv=qkv, stream=stream, weights=case.weights, attention=attention
            )
            torch.cuda.synchronize()
            _check_launch(
                case,
                attention,
                qkv,
                stream,
                out,
                stream_next,
                f"TP{world} {name} image, context {seq_len}",
                verbose,
            )
        graph = _check_determinism(case, attention, qkv, stream)
        if not check_only:
            results[name] = _cold_l2_us(graph)
            if verbose and rank == 0:
                print(
                    f"TP{world} {name} image: {results[name]:7.1f} us cold-L2 graph replay"
                )
        hosts.append(case.host)
        del case, graph
        torch.cuda.empty_cache()
        dist.barrier()
    return results


@torch.inference_mode()
def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """Check every image at this world (4 or 8 ranks), then time it."""
    # Not initialize(): TP8 spans two nodes, and the layer's symmetric buffers
    # keep torch's default backend (initialize() selects NVSHMEM).
    _, _, group = initialize_group("the Muse Spark layer megakernel")
    try:
        layer_family(group.size(), TOKENS)  # a world serving runs
        return _run(group, verbose=verbose, check_only=check_only)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
