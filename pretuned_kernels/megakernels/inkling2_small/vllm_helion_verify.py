# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN001, ANN201, ANN202, ANN204
"""Experimental: the batch-1 8-row target-verify forward of Inkling2 (TP4 = EP4) as Helion megakernels.

The vLLM-side driver: vllm/models/inkling2/helion_verify.py on vLLM's Inkling 2 branch, hooked into
vllm/models/inkling2/model.py by vllm_model.patch. It loads the kernels of this directory (Helion checkout on
PYTHONPATH).

INK_HELION=1 runs every 8-token single-request forward as one or two kernels a layer:
    one launch: layer_local_nvfp4 (local NVFP4 layers), layer_block_nvfp4 (block-boundary NVFP4 layers: also the
                new attn_res block's stats), layer_local_mxfp8 (local MXFP8 layers);
    attention:  attn_global (global layers), attn_block (block-boundary MXFP8 layers), with INK_HELION_LAYERS=0
                also attn_block (block-boundary NVFP4 layers) and attn_local (local layers);
    MoE:        moe_nvfp4 (NVFP4 layers), moe_mxfp8 (MXFP8 layers), after an attention kernel.
The attn_res mixes + RMSNorms between the sublayers fold into the kernel producing their input: the attention kernel
writes its MoE kernel's input rows, the MoE kernel the next layer's attention input rows (for the last layer, the
final-norm output). Configs (a _config.CONFIGS name, the default, or a JSON file of scalar keys: each variable
reaches kernels of different loop structures): INK_HELION_ATTN (attention kernels), INK_HELION_MOE /
INK_HELION_MOE_MX (MoE and layer kernels).
"""

from __future__ import annotations

import importlib
import inspect
import os

import torch
from vllm.forward_context import get_forward_context

ENABLED = os.environ.get("INK_HELION", "") == "1"
_PKG = "pretuned_kernels.megakernels.inkling2_small"
CFG_ATTN = os.environ.get("INK_HELION_ATTN", "attn")
CFG_MOE = os.environ.get("INK_HELION_MOE", "moe_nvfp4")
CFG_MOE_MX = os.environ.get("INK_HELION_MOE_MX", "moe_mxfp8")
# The one-launch layer kernels (attention + MoE of a layer) that local NVFP4 / MXFP8 and block-boundary NVFP4 layers
# run instead of the attention and MoE pair; INK_HELION_LAYERS=0 runs the pair on every layer.
_LAYERS = os.environ.get("INK_HELION_LAYERS", "1") == "1"
LAYER_MOD = f"{_PKG}.layer_local_nvfp4" if _LAYERS else ""
LAYER_MX_MOD = f"{_PKG}.layer_local_mxfp8" if _LAYERS else ""
LAYER_BLOCK_MOD = f"{_PKG}.layer_block_nvfp4" if _LAYERS else ""

VERIFY_ROWS = 8
MMA_ROWS = 16  # the GEMMs pad the 8 verify rows to 16
K_SPLITS = 4  # QKVR's K split (kernel partials)
AR_COLS = 256  # attention all-reduce + attn_sconv columns a task
BLOCKS = 8  # attn_res blocks: the partials before layers 0, 6, .., 42
COLS = 512  # the fold's column split (the kernels' res_* views: [.., H / COLS, COLS])
QUERY_PAD = 128
SLOTS = (
    4  # query slots of a MoE kernel's F: the next MLP input + up to 3 attention inputs
)
MOE_SLOTS = 64  # one per (token, expert) pair: the most distinct local experts a window can route to
PAGE_SIZE = 128  # the KV cache block size the attention workspaces are sized for
# The tensors the kernels access through tensor descriptors (TMA_ATTN: the global attention kernel's; TMA_LOCAL:
# both the local and block attention kernels'; TMA_LAYER: both the local and block layer kernels').
TMA_ATTN = "w_qkvr,wo"
TMA_LOCAL = "w_qkvr,wo,x3,o_in"
TMA_MOE = "w13,w13_s,xq,xs,gate_w,w2f,w2sf,zqf,zsf"
TMA_LAYER = TMA_MOE + ",a_w_qkvr,a_w_qr,a_w_g,a_w_kv,a_wo,a_x3,a_x8,a_x6,a_o_in,x"
TMA_MOE_MX = "w13,w13_s,w2,w2_s,xq,zq,gate_w,xsf,zsf"
TMA_LAYER_MX = TMA_MOE_MX + ",a_w_qr,a_w_g,a_w_kv,a_wo,a_x8,a_x6,a_o_in,x"


class _Fold:
    """The attn_res state shared by the kernels: the query table (every attn_res query, vLLM's pre-scaled fp32, as
    bf16 high + low parts), per block its rows (hist), rstd and dots with every query, and the MoE kernels' F
    (softmax-weighted block sums for the next layer's queries) by layer parity."""

    def __init__(self, model, device):
        res = model.attn_res
        hidden = model.config.hidden_size
        self.hidden = hidden
        self.n_layers = len(model.layers)
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(  # noqa: E731
            shape, dtype=dtype, device=device
        )
        attn = [q.detach().float().reshape(-1, hidden) for q in res.attn_queries]
        mlp = [q.detach().float().reshape(-1, hidden) for q in res.mlp_queries]
        queries = [
            q[0]
            for q in (
                *attn,
                *mlp,
                res.output_query.detach().float().reshape(-1, hidden),
            )
        ]
        self.output_index = len(queries) - 1
        self.attn_ids = {}  # layer -> its attention-input queries' indices (global layers: q, k, v)
        for i, q in enumerate(attn):
            ids = [i]
            for row in q[1:]:
                ids.append(len(queries))
                queries.append(row)
            self.attn_ids[i + 1] = ids
        self.queries = torch.stack(queries).to(device)
        n_queries = self.queries.shape[0]
        assert n_queries <= QUERY_PAD
        high = self.queries.bfloat16()
        low = (self.queries - high.float()).bfloat16()
        table = zeros(2 * QUERY_PAD, hidden, dtype=torch.bfloat16)
        table[:n_queries], table[QUERY_PAD : QUERY_PAD + n_queries] = high, low
        self.query_table = table.view(2 * QUERY_PAD, hidden // COLS, COLS)
        self.dpart = zeros(hidden // COLS, 3, QUERY_PAD, MMA_ROWS)
        self.hist = zeros(
            BLOCKS, VERIFY_ROWS, hidden // COLS, COLS, dtype=torch.bfloat16
        )
        self.rstd = zeros(BLOCKS, VERIFY_ROWS)
        self.dots = zeros(BLOCKS, n_queries, VERIFY_ROWS)
        self.acc2 = zeros(2, SLOTS, VERIFY_ROWS, hidden // COLS, COLS)
        self.stat2 = zeros(2, SLOTS, VERIFY_ROWS, hidden // COLS, 4)
        self.acc_empty = zeros(VERIFY_ROWS, hidden // COLS, COLS)
        self.stat_empty = zeros(VERIFY_ROWS, hidden // COLS, 4)
        self.stat_empty[..., 0] = -1e30
        self.block_slot = {
            lid: torch.tensor([lid // 6], dtype=torch.int32, device=device)
            for lid in range(0, self.n_layers, 6)
        }
        self.moe_slots = {
            lid: self._moe_slots(model, lid, device) for lid in range(self.n_layers)
        }

    def mlp_index(self, lid):
        return self.n_layers - 1 + lid

    def query(self, index):
        return self.queries[index].view(self.hidden // COLS, COLS)

    def acc_in(self, lid):
        return self.acc2[lid % 2], self.stat2[lid % 2]

    def _moe_slots(self, model, lid, device):
        """Layer lid's MoE kernel's fold inputs for layer lid + 1 (the last layer: the final norm's input): qidx [8]
        (live blocks, A, slot query ids, 0 ..), the slot queries and norm weights [QP, H/V, V]."""
        if lid + 1 < self.n_layers:
            nxt = model.layers[lid + 1]
            ids = [self.mlp_index(lid + 1), *self.attn_ids[lid + 1]]
            weights_in = [nxt.attn_norm.weight[j] for j in range(nxt.num_attn_inputs)]
        else:
            ids = [self.output_index, self.output_index]
            weights_in = [model.norm.weight[0]]
        n_slots = 1 << (len(ids) - 1).bit_length()
        qidx = torch.zeros(8, dtype=torch.int32, device=device)
        qidx[0], qidx[1] = lid // 6 + 1, len(weights_in)
        qidx[2 : 2 + len(ids)] = torch.tensor(ids, dtype=torch.int32)
        cols = self.hidden // COLS
        queries = torch.zeros(n_slots, cols, COLS, device=device)
        weights = torch.zeros(
            n_slots, cols, COLS, dtype=weights_in[0].dtype, device=device
        )
        for s, index in enumerate(ids):
            queries[s] = self.query(index)
        for a, w in enumerate(weights_in):
            weights[1 + a] = w.view(cols, COLS)
        return qidx, queries, weights


def _variant(fn, args, config, tma):
    from pretuned_kernels.megakernels.inkling2_small._config import variant_kernel

    return variant_kernel(fn, args, config, tma)


class _Runner:
    def __init__(self, model, device):
        import torch.distributed._symmetric_memory as symm_mem

        attn_global = importlib.import_module(f"{_PKG}.attn_global")
        attn_local = importlib.import_module(f"{_PKG}.attn_local")
        attn_block = importlib.import_module(f"{_PKG}.attn_block")
        moe_nvfp4 = importlib.import_module(f"{_PKG}.moe_nvfp4")
        moe_mxfp8 = importlib.import_module(f"{_PKG}.moe_mxfp8")
        views = importlib.import_module(f"{_PKG}._common")  # the weight views

        from vllm.distributed import get_tp_group

        group = get_tp_group()
        self.group_name = group.device_group.group_name
        self.rank = group.rank_in_group
        self.model = model
        config = model.config
        hidden = config.hidden_size
        self.hidden = hidden
        self.fold = _Fold(model, device)
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(  # noqa: E731
            shape, dtype=dtype, device=device
        )
        self.fns = {
            "local": attn_local._ink_attn_local,
            "block": attn_block._ink_attn_block,
            "global": attn_global._ink_attn_global,
            "moe": moe_nvfp4._ink_moe_nvfp4,
            "moe_mx": moe_mxfp8._ink_moe_mxfp8,
        }
        if LAYER_MOD:
            module = importlib.import_module(LAYER_MOD)
            self.fns["layer"] = module._ink_layer_local_nvfp4
        if LAYER_MX_MOD:
            module = importlib.import_module(LAYER_MX_MOD)
            self.fns["layer_mx"] = module._ink_layer_local_mxfp8
        if LAYER_BLOCK_MOD:
            module = importlib.import_module(LAYER_BLOCK_MOD)
            self.fns["layer_block"] = module._ink_layer_block_nvfp4
        # launch key -> helion.Kernel: the bare name for the attention kernels, (name, rows of the next layer's
        # attention input, 1 or 3) for the layer and MoE kernels
        self.kernels = {}
        self.params = {}  # kernel name -> its parameter names

        # Attention workspaces (shared by the layers).
        attn = model.layers[0].attn
        heads, kv_heads, head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        n = attn.qkvr.weight.shape[0]
        rows = 8 * VERIFY_ROWS  # (token, padded head) rows of a KV group
        # The local kernels mask to window_left = rel_proj.size(1) - 1 keys behind each query; bias_ws and
        # att_ws below are sized from the config's window, so the two must agree.
        assert attn.rel_logits_proj.shape[1] == config.sliding_window_size, (
            attn.rel_logits_proj.shape,
            config.sliding_window_size,
        )
        # (made before the KV cache exists: applies() checks its block size)
        live_pages = -(-(config.sliding_window_size - 1) // PAGE_SIZE) + 1
        pages = 1 << (live_pages - 1).bit_length()
        self.x = zeros(MMA_ROWS, hidden, dtype=torch.bfloat16)
        self.xs = zeros(3, MMA_ROWS, hidden, dtype=torch.bfloat16)
        self.part = zeros(K_SPLITS, MMA_ROWS, n)
        # The local layer kernels' qkvr partials: q | r by eighth of K, sinks | k | v and the gate rows by sixth.
        sink_col = 2 * heads * head_dim + heads * attn.r_scale.shape[-1]
        self.part_qr8 = zeros(8, MMA_ROWS, sink_col - heads * head_dim)
        self.part_kv6 = zeros(6, MMA_ROWS, n - sink_col)
        self.part_g6 = zeros(6, MMA_ROWS, heads * head_dim)
        self.q_ws = zeros(VERIFY_ROWS, kv_heads, 8, head_dim, dtype=torch.bfloat16)
        self.rel_ws = zeros(VERIFY_ROWS, heads, attn.rel_extent, dtype=torch.bfloat16)
        # The layer kernels' A / AN relative-position biases by (KV head, row, key), rewritten by PQR every launch;
        # the padded-head rows keep this -inf fill.
        self.bias_ws = torch.full(
            (kv_heads, rows, live_pages * PAGE_SIZE),
            float("-inf"),
            dtype=torch.bfloat16,
            device=device,
        )
        self.nbias_ws = torch.full(
            (kv_heads, rows, 16), float("-inf"), dtype=torch.bfloat16, device=device
        )
        self.att_ws = zeros(kv_heads, pages, rows, head_dim)
        self.lse_ws = zeros(kv_heads, pages, rows)
        self.kv_new = zeros(2, kv_heads, 16, head_dim, dtype=torch.bfloat16)
        self.att_new = zeros(kv_heads, rows, head_dim)
        self.lse_new = zeros(kv_heads, rows)
        o_parts = 2 * kv_heads
        self.o_in = zeros(
            MMA_ROWS, o_parts, heads * head_dim // o_parts, dtype=torch.bfloat16
        )
        self.part_o = zeros(o_parts, MMA_ROWS, hidden)
        self.res_part = zeros(VERIFY_ROWS, hidden // AR_COLS, 4)
        self.res_part8 = zeros(VERIFY_ROWS, hidden // AR_COLS, 8)
        self.opart = symm_mem.empty(
            VERIFY_ROWS, hidden // AR_COLS, AR_COLS, dtype=torch.bfloat16, device=device
        )
        self.opart.zero_()
        symm_mem.rendezvous(self.opart, self.group_name)
        splits = attn_global.SPLITS
        self.g_att_ws = zeros(kv_heads, splits, rows, head_dim)
        self.g_lse_ws = zeros(kv_heads, splits, rows)
        self.g_dyn = zeros(2, attn_global.DYN_PARTS, VERIFY_ROWS, kv_heads * 4)
        self.g_src_tab = attn_global.source_table(
            n, heads * head_dim, kv_heads * head_dim, device
        )

        # Per-layer weight views; global layers: their qkvr weight in SGLang's row order (q | k | v | r | gate |
        # sinks | pad: P's 128-column tiles each project one input) and stacked dynamic-conv projections.
        self.layer_args = []
        for layer in model.layers:
            a = layer.attn
            w = a.qkvr.weight
            if not layer.is_local:
                q, r, gate, sinks, k, v, pad = w.split(a.split_sizes)
                w = torch.cat([q, k, v, r, gate, sinks, pad]).contiguous()
            wq3, wo3 = views.attn_views(w, a.wo_ud.weight, K_SPLITS)
            e = layer.mlp.experts
            if e.nvfp4:
                w13, w13_s, w2, w2_s = views.moe_views(
                    e.w13_weight,
                    e.w13_weight_scale.view(torch.uint8),
                    e.w2_weight,
                    e.w2_weight_scale.view(torch.uint8),
                )
            else:
                w13, w13_s, w2, w2_s = views.mx_views(
                    e.w13_weight,
                    e.w13_weight_scale.view(torch.uint8),
                    e.w2_weight,
                    e.w2_weight_scale.view(torch.uint8),
                )
            weights = (
                (
                    w13,
                    w13_s,
                    e.w13_weight_amax,
                    e.w13_weight_outer_scale_denominator,
                    w2,
                    w2_s,
                    e.w2_weight_amax,
                    e.w2_weight_outer_scale_denominator,
                )
                if e.nvfp4
                else (w13, w13_s, w2, w2_s)
            )
            entry = {
                "wq3": wq3,
                "wo3": wo3,
                "moe_weights": weights,
                "nvfp4": e.nvfp4,
                # rel_logits_proj widened to fp32 once (the layer kernels' PQR skips the per-task converts)
                "rel_proj32": a.rel_logits_proj.float(),
            }
            if not layer.is_local:
                entry["dyn_proj"] = (
                    torch.stack([a.k_sconv.weight_proj, a.v_sconv.weight_proj])
                    .to(torch.bfloat16)
                    .contiguous()
                )
            self.layer_args.append(entry)

        # MoE workspaces, one set per expert format.
        self.moe_ws = {}
        e0 = model.layers[0].mlp
        router_n = e0.gate.weight.shape[0]
        parts = 1 << (router_n // 128 - 1).bit_length()
        topk = e0.gate.topk
        # The kernels select a top-8 (_common._select / _renorm), keep a rank's local experts in 128-bit bitmaps
        # and give each distinct local expert a slot, at most rows x top-k = MOE_SLOTS of them.
        assert topk == 8 and VERIFY_ROWS * topk == MOE_SLOTS, topk
        assert e0.experts.w13_weight.shape[0] <= 128, e0.experts.w13_weight.shape
        fp8 = torch.float8_e4m3fn
        inter = config.intermediate_size
        for nvfp4 in sorted({layer.mlp.experts.nvfp4 for layer in model.layers}):
            ws = {
                "x": zeros(MMA_ROWS, hidden, dtype=torch.bfloat16),
                "pre": (
                    zeros(hidden // 256, parts, MMA_ROWS, 128),
                    zeros(VERIFY_ROWS * topk, dtype=torch.int32),
                    zeros(VERIFY_ROWS * topk),
                ),
                "post": (
                    zeros(MMA_ROWS, hidden // 512, 256, dtype=torch.uint8),
                    zeros(MMA_ROWS, hidden // 512, 32, dtype=torch.uint8),
                    zeros(MMA_ROWS),
                    zeros(
                        MOE_SLOTS,
                        2 * inter // 128,
                        VERIFY_ROWS,
                        64,
                        dtype=torch.bfloat16,
                    ),
                    zeros(MOE_SLOTS, 2 * inter // 128, VERIFY_ROWS),
                    zeros(MOE_SLOTS, MMA_ROWS, inter // 256, 128, dtype=torch.uint8),
                    zeros(MOE_SLOTS, MMA_ROWS, inter // 256, 16, dtype=torch.uint8),
                    zeros(MOE_SLOTS, MMA_ROWS),
                )
                if nvfp4
                else (
                    zeros(MMA_ROWS, hidden // 256, 256, dtype=fp8),
                    zeros(MMA_ROWS, hidden // 256, 8, dtype=torch.uint8),
                    zeros(MOE_SLOTS, MMA_ROWS, inter // 256, 256, dtype=fp8),
                    zeros(MOE_SLOTS, MMA_ROWS, inter // 256, 8, dtype=torch.uint8),
                ),
                "rows": zeros(VERIFY_ROWS * topk, hidden, dtype=torch.bfloat16),
                "no_partial": zeros(VERIFY_ROWS, hidden, dtype=torch.bfloat16),
                "res_part": zeros(SLOTS, VERIFY_ROWS, hidden // 256, 4),
            }
            ws["ppart"] = symm_mem.empty(
                VERIFY_ROWS, hidden // 256, 256, dtype=torch.float32, device=device
            )
            ws["ppart"].zero_()
            symm_mem.rendezvous(ws["ppart"], self.group_name)
            self.moe_ws[nvfp4] = ws

    def _launch(self, name, named, config, tma, key):
        fn = self.fns[name]
        params = self.params.get(name)
        if params is None:  # inspect.signature per launch costs ~60 us
            params = self.params[name] = tuple(inspect.signature(fn).parameters)
        args = tuple(named[p] for p in params)
        kernel = self.kernels.get(key)
        if kernel is None:
            kernel = self.kernels[key] = _variant(fn, args, config, tma)
        kernel(*args)

    def forward(self, inputs_embeds):
        model = self.model
        attn_md = get_forward_context().attn_metadata
        partial = inputs_embeds
        layer0 = model.layers[0]
        self.x[:VERIFY_ROWS].copy_(layer0.attn_norm(partial, 0))
        aux = []
        out = None
        for lid, layer in enumerate(model.layers):
            if lid in model.aux_hidden_state_layers:
                aux.append(partial)
            nvfp4 = layer.mlp.experts.nvfp4
            block = lid % 6 == 0
            one = (
                (LAYER_BLOCK_MOD if nvfp4 else "")
                if block
                else (LAYER_MOD if nvfp4 else LAYER_MX_MOD)
            )
            if one and layer.is_local:
                an, attention = self._attention(
                    lid, layer, partial, attn_md, launch=False
                )
                named, partial, out = self._moe(
                    lid, layer, attention, partial, attn_md, launch=False
                )
                rows = named["x_next"].shape[0]
                named.update({f"a_{k}": v for k, v in an.items()})
                name, config, tma = (
                    ("layer_block" if block else "layer", CFG_MOE, TMA_LAYER)
                    if nvfp4
                    else ("layer_mx", CFG_MOE_MX, TMA_LAYER_MX)
                )
                self._launch(name, named, config, tma, (name, rows))
            else:
                attention = self._attention(lid, layer, partial, attn_md)
                partial, out = self._moe(lid, layer, attention, partial, attn_md)
        if model.aux_hidden_state_layers:
            return out, [*aux, out]
        return out

    def _attention(self, lid, layer, partial, attn_md, launch=True):
        fold, hidden, cols = self.fold, self.hidden, COLS
        attn, cs = layer.attn, layer.conv_state
        md = attn_md[attn.prefix]
        cmd = attn_md[cs.prefix]
        la = self.layer_args[lid]
        head_dim = attn.head_dim
        kv, cc = _slots(attn.kv_cache), _slots(cs.kv_cache)
        (k_off, k_w), (v_off, v_w), (a_off, a_w), _ = cs.stream_ranges
        moe_x = self.moe_ws[layer.mlp.experts.nvfp4]["x"]
        out = torch.empty(
            VERIFY_ROWS, hidden, dtype=torch.bfloat16, device=self.x.device
        )
        named = {
            "x": self.x,
            "xs": self.xs,
            "src_tab": self.g_src_tab,
            "w_qkvr": la["wq3"],
            "part": self.part,
            "part_qr8": self.part_qr8,
            "part_kv6": self.part_kv6,
            "part_g6": self.part_g6,
            "dyn_proj": la.get("dyn_proj"),
            "dyn": self.g_dyn,
            "k_conv_w": attn.k_sconv.conv_weight,
            "v_conv_w": attn.v_sconv.conv_weight,
            "k_conv_cache": cc[..., k_off : k_off + k_w].flatten(0, 1),
            "v_conv_cache": cc[..., v_off : v_off + v_w].flatten(0, 1),
            "conv_pages": cmd.block_table[0],
            "conv_loc": cmd.slot_mapping[:VERIFY_ROWS],
            "conv_block": cs.kv_cache.shape[2],
            "q_scale": attn.q_scale,
            "r_scale": attn.r_scale,
            "rel_proj": attn.rel_logits_proj,
            "rel_proj32": la["rel_proj32"],
            "k_buf": kv[..., :head_dim],
            "v_buf": kv[..., head_dim:],
            "pages": md.block_table[0],
            "seq_len": md.seq_lens[:1],
            "kv_loc": md.slot_mapping[:VERIFY_ROWS],
            "page_size": attn.kv_cache.shape[2],
            "q_ws": self.q_ws,
            "rel_ws": self.rel_ws,
            "bias_ws": self.bias_ws,
            "nbias_ws": self.nbias_ws,
            "att_ws": self.att_ws if layer.is_local else self.g_att_ws,
            "lse_ws": self.lse_ws if layer.is_local else self.g_lse_ws,
            "kv_new": self.kv_new,
            "att_new": self.att_new,
            "lse_new": self.lse_new,
            "o_in": self.o_in,
            "wo": la["wo3"],
            "part_o": self.part_o,
            "opart": self.opart,
            "a_conv_w": layer.attn_sconv.conv_weight,
            "a_conv_cache": cc[..., a_off : a_off + a_w].flatten(0, 1),
            "out": out.view(VERIFY_ROWS, hidden // cols, cols),
            "res_q": fold.query(fold.mlp_index(lid)),
            "res_w": layer.mlp_norm.weight[0].view(hidden // cols, cols),
            "x_next": moe_x.view(MMA_ROWS, hidden // cols, cols),
            "group_name": self.group_name,
        }
        if lid % 6 == 0:  # a block-boundary (local) layer: its partial is a new block
            assert layer.is_local
            name = "block"
            named.update(
                res_block=partial,
                res_queries=fold.query_table,
                res_dpart=fold.dpart,
                res_hist=fold.hist,
                res_rstd=fold.rstd,
                res_dots=fold.dots,
                res_slot=fold.block_slot[lid],
                res_acc=fold.acc_in(lid)[0][0] if lid else fold.acc_empty,
                res_stat=fold.acc_in(lid)[1][0] if lid else fold.stat_empty,
                res_part=self.res_part8,
            )
        else:
            name = "local" if layer.is_local else "global"
            named.update(
                res_partial=partial.view(VERIFY_ROWS, hidden // cols, cols),
                res_acc=fold.acc_in(lid)[0][0],
                res_stat=fold.acc_in(lid)[1][0],
                res_part=self.res_part,
            )
        if not launch:
            return named, out
        tma = TMA_LOCAL if name in ("local", "block") else TMA_ATTN
        self._launch(name, named, CFG_ATTN, tma, name)
        return out

    def _moe(self, lid, layer, attention, partial, attn_md, launch=True):
        """Layer lid's MoE half: returns the next partial and (the last layer) the final-norm output."""
        fold, hidden, cols = self.fold, self.hidden, COLS
        mlp, cs = layer.mlp, layer.conv_state
        experts, gate = mlp.experts, mlp.gate
        cmd = attn_md[cs.prefix]
        md = attn_md[layer.attn.prefix]
        la = self.layer_args[lid]
        ws = self.moe_ws[experts.nvfp4]
        _, _, _, (m_off, m_w) = cs.stream_ranges
        last = lid + 1 == len(self.model.layers)
        out = torch.empty(
            VERIFY_ROWS, hidden, dtype=torch.bfloat16, device=attention.device
        )
        qidx, queries, weights = fold.moe_slots[lid]
        n_slots = queries.shape[0]
        if last:
            final = torch.empty(
                1, MMA_ROWS, hidden, dtype=torch.bfloat16, device=attention.device
            )
            x_next = final
        elif self.model.layers[lid + 1].is_local:
            x_next = self.x[None]
        else:
            x_next = self.xs
        acc_out, stat_out = fold.acc_in(lid + 1)
        named = {
            "x": ws["x"],
            "gate_w": gate.weight.view(-1, 128, hidden // 128, 128),
            "gate_b": gate.bias,
            "logit_part": ws["pre"][0],
            "ids": ws["pre"][1],
            "gammas": ws["pre"][2],
            "rows_out": ws["rows"],
            "ppart": ws["ppart"],
            "attention": attention,
            "partial": ws["no_partial"] if lid % 6 == 0 else partial,
            "conv_w": layer.mlp_sconv.conv_weight,
            "conv_cache": _slots(cs.kv_cache)[..., m_off : m_off + m_w].flatten(0, 1),
            "conv_pages": cmd.block_table[0],
            "conv_loc": cmd.slot_mapping[:VERIFY_ROWS],
            "seq_len": md.seq_lens[:1],
            "conv_block": cs.kv_cache.shape[2],
            "out": out.view(VERIFY_ROWS, hidden // cols, cols),
            "res_hist": fold.hist,
            "res_rstd": fold.rstd,
            "res_dots": fold.dots,
            "res_qidx": qidx,
            "res_q": queries,
            "res_w": weights,
            "res_acc": acc_out[:n_slots],
            "res_stat": stat_out[:n_slots],
            "res_part": ws["res_part"][:n_slots],
            "x_next": x_next.view(x_next.shape[0], MMA_ROWS, hidden // cols, cols),
            "group_name": self.group_name,
            "rank": self.rank,
            "gate_cap": experts.gate_cap,
            "up_cap": experts.up_cap,
            "route_scale": gate.route_scale,
        }
        if experts.nvfp4:
            (
                named["w13"],
                named["w13_s"],
                named["w13_amax"],
                named["w13_den"],
                named["w2"],
                named["w2_s"],
                named["w2_amax"],
                named["w2_den"],
            ) = la["moe_weights"]
            (
                named["xq"],
                named["xs"],
                named["x_amax"],
                named["h"],
                named["h_max"],
                named["zq"],
                named["zs"],
                named["z_amax"],
            ) = ws["post"]
            name, config, tma = "moe", CFG_MOE, TMA_MOE
        else:
            named["w13"], named["w13_s"], named["w2"], named["w2_s"] = la["moe_weights"]
            named["xq"], named["xs"], named["zq"], named["zs"] = ws["post"]
            name, config, tma = "moe_mx", CFG_MOE_MX, TMA_MOE_MX
        if not launch:
            return named, out, final[0, :VERIFY_ROWS] if last else None
        self._launch(name, named, config, tma, (name, x_next.shape[0]))
        return out, final[0, :VERIFY_ROWS] if last else None


def _slots(cache):
    """A paged cache [blocks, heads, block size, width] whose blocks are slot-major ([block size][heads][width], vLLM's
    layout) as the kernels' view by slot: [blocks * block size, heads, width]."""
    blocks, heads, size, width = cache.shape
    return cache.transpose(1, 2).view(blocks * size, heads, width)


_RUNNER = None


def prepare(model, device) -> None:
    """Builds the kernels' state (workspaces, symmetric buffers: collective) at the first forward outside CUDA graph
    capture (the memory profiling run)."""
    global _RUNNER
    if _RUNNER is None and not torch.cuda.is_current_stream_capturing():
        from vllm.lora.layers import BaseLayerWithLoRA

        # The kernels apply no LoRA adapter: a model with LoRA-wrapped layers keeps the stock path (no runner).
        lora = any(isinstance(m, BaseLayerWithLoRA) for m in model.modules())
        _RUNNER = False if lora else _Runner(model, device)


def applies(model, positions) -> bool:
    # (the runner serves the model it was built for, not a second one, e.g. a drafter)
    if not _RUNNER or _RUNNER.model is not model or positions.shape[0] != VERIFY_ROWS:
        return False
    attn_md = get_forward_context().attn_metadata
    if not isinstance(attn_md, dict):
        return False
    layer0 = model.layers[0]
    md = attn_md.get(layer0.attn.prefix)
    # One request of VERIFY_ROWS query tokens (the kernels place row r at seq_len - VERIFY_ROWS + r), not a shorter
    # one padded to VERIFY_ROWS positions.
    if (
        md is None
        or md.seq_lens.shape[0] != 1
        or md.max_query_len != VERIFY_ROWS
        or attn_md.get(layer0.conv_state.prefix) is None
    ):
        return False
    assert layer0.attn.kv_cache.shape[2] == PAGE_SIZE, layer0.attn.kv_cache.shape
    # The kernels index the caches (every layer's of these shapes) in int32: their K / V and conv-stream views are
    # strided slices, whose numel understates their extent. --num-gpu-blocks-override 32000 keeps each below 2**31.
    for cache in (layer0.attn.kv_cache, layer0.conv_state.kv_cache):
        assert cache.numel() < 2**31, cache.shape
    return True


def forward(model, positions, inputs_embeds):
    # (positions: the model forward's, which applies() checked; the kernels place row r at seq_len - VERIFY_ROWS + r)
    return _RUNNER.forward(inputs_embeds)
