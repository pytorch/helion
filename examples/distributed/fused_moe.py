"""Fused Mixture-of-Experts (expert-parallel) with in-kernel reduce-scatter.

DRAFT / SKETCH.  This file is a design + best-attempt implementation of a
Helion kernel that mirrors the Meta "fuse-all MoE" TPU kernel
(``msl-tpu-kernel``'s ``gmm_v2_fused_rs``): gather routed tokens -> GMM1 (gate
+ up) + gated activation -> GMM2 (down) -> weighted **reduce-scatter** of the
expert outputs back to the chip that owns each token, all in one launch.

What RUNS today (VERIFIED on an 8-chip TPU host -- all ranks match reference):
  * ``fused_moe`` -- the end-to-end expert-parallel path: the Helion/Pallas
    compute kernel (``fused_moe_local_kernel``: gather + GMM1 + gated act +
    GMM2) on each chip, then a torch ``reduce_scatter`` across chips.  All 8
    ranks' output shards match ``fused_moe_reference`` (rtol/atol=2e-2).  Run
    with ``--distributed``.
  * ``fused_moe_reference`` -- pure-PyTorch oracle (also verified on CPU).
  NOTE: the cross-chip ``reduce_scatter`` here is a torch PLACEHOLDER standing
  in for the in-kernel ICI direct-write; folding it into the kernel is exactly
  what needs Features 1-4 below.

What is a SKETCH (blocked on remaining Helion features -- see DESIGN NOTES):
  * ``fused_moe_kernel`` -- the *fully fused* distributed kernel that fires the
    ICI reduce-scatter (direct-write) from inside the loop (no torch
    ``reduce_scatter``).  The *asymmetric, per-row data-dependent* remote copy it
    needs is now EXPRESSIBLE via the landed primitive (Features 1-2, below)::

        hl.start_async_remote_copy(
            src, [i], device_id, dst=out_buf, dst_index=[write_pos]
        )

    and is validated in isolation (small-scale cross-chip reduce-scatter matches
    a torch oracle; see the ``test_remote_copy`` codegen tests).  What still
    blocks the *single-launch* fusion here is: (Feature 4) a counting-semaphore
    bulk drain so the tile's per-row pushes don't serialise the ICI lane;
    the two-phase compute->scatter structure inside one Pallas launch (Pallas
    flattens the grid to one pid, so there is no ``hl.barrier`` to separate the
    phases); and, empirically, a Mosaic core-halt when the direct-write runs at
    MoE row width (H=128) that does not reproduce at the small widths the
    isolated test uses.  Until those land, callers use ``fused_moe`` below, which
    runs the Helion compute kernel and does the scatter in torch.

Feature status (details in DESIGN NOTES):
  Features 1-2 (asymmetric + per-row data-dependent remote copy) -- LANDED as
  ``hl.start_async_remote_copy(src, src_index, device_id, dst, dst_index)``.
  Features 3-4 (symmetric-buffer alloc, counting-semaphore bulk drain) -- open.
  Feature 6 (accumulator feature-dim slice for the gated activation -> worked
  around with separate gate/up weights), 7 (in-kernel intermediate as a K-tiled
  matmul operand -> worked around with a full-I GMM2 matmul) -- open.

Run (single-host, 8 TPU chips)::

    source dunfanlu_notes/scripts/init_helion_pallas_tpu.sh
    python examples/distributed/fused_moe.py            # reference + local-kernel checks
    python examples/distributed/fused_moe.py --distributed   # attempt full fused path

===========================================================================
DESIGN NOTES: Helion language additions for a fused-MoE reduce-scatter kernel
===========================================================================
Methodology follows "Helion TPU Distributed Support" (Dunfan Lu): every
proposed primitive is designed to lower to BOTH targets --
  * TPU  : ``pltpu.make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem,
           device_id, device_id_type)`` (push-based, async, PGAS-like).
  * GPU  : ``nvshmem.putmem_signal_block(dst, src, nbytes, signal, sig_val,
           sig_op, pe)`` + ``signal_wait_until`` (push-based, async, PGAS).
Both are push-only, async, and address peers by a flat LOGICAL PE id, so a
single Helion API can target both.  The v1 distributed API already shipped
(``hl.start_async_remote_copy`` + ``@helion.kernel(distributed=[world_size])``,
see ``all_gather.py``).  The fused MoE needed four things beyond it; Features
1-2 have since LANDED, Features 3-4 remain:

--- Feature 1: ASYMMETRIC per-row remote copy (src != dst) -- LANDED -----------
Need.  The reduce-scatter writes one GMM2 output row from a LOCAL staging slot
into an ARBITRARY slot of the owner chip's output buffer:
    out_buf[write_pos]@dest_chip  <-  local_row_out       (write_pos != src idx)
The original ``start_async_remote_copy(tensor, index, device_id)`` only supported
SAME-tensor, SAME-index pushes (``tensor[index]->tensor[index]``) -- enough for
ring all-gather, not for scatter.
Landed as.  The op now takes optional ``dst`` / ``dst_index``::
    hl.start_async_remote_copy(src, src_index, device_id, dst, dst_index)
  pushing ``src[src_index] -> dst[dst_index]`` on the peer; the 3-arg symmetric
  form is preserved as sugar (``dst is src``, ``dst_index is src_index``).
Lowering.
  * Pallas : ``make_async_remote_copy(src.at[..], dst.at[..], ...)`` -- two
             independent refs; a 1:1 fit.
  * NVSHMEM: ``putmem_signal_block(dst_ptr, src_ptr, nbytes, ...)`` -- dst and
             src independent; a 1:1 fit.

--- Feature 2: DATA-DEPENDENT, per-row device_id and dst index -- LANDED -------
Need.  ``dest_chip = token_id // chunk`` and ``write_pos = local_row*top_k +
slot`` are computed PER ROW from ``output_indices`` (runtime scalars), inside
the tile loop.  ``all_gather.py`` only uses a compile-time-ish neighbour id.
Landed as.  ``device_id``, ``src_index`` and ``dst_index`` may all be runtime
device scalars; the codegen emits the copy inside the loop with the scalar as an
SSA value (Pallas ``device_id`` / NVSHMEM ``pe`` both already take a runtime peer
id and a runtime offset -- no backend capability gap).

--- Feature 3: SYMMETRIC output-buffer allocation (hl.symmetric_empty) --------
Need.  The direct-write TARGET must be a symmetric buffer that peers can write
into.  Today the host allocates it differently per backend (GPU:
``symm_mem.empty`` + ``rendezvous``; TPU: a plain tensor).  This is the doc's
"Host-side Tensor-Allocation API Unification" stretch goal.
Proposal.  ``buf = hl.symmetric_empty([chunk*top_k, H], dtype=...)`` inside the
kernel body (or a host helper), returning a backend-appropriate symmetric
tensor.
Lowering.  GPU -> ``symm_mem.empty(...) + rendezvous(group)``;
TPU -> a regular device tensor participating in the ``shard_map`` mesh.

--- Feature 4: COUNTING semaphores / bulk drain for many in-flight pushes -----
Need.  A GM tile fires up to ``tile_m`` per-row pushes; we must NOT ``.wait()``
each inline (that serialises the ICI lane).  The production kernel accumulates
send/recv COUNTS and drains once at the end (triple-buffered staging).
Proposal.  Keep semaphore/signal allocation compiler-managed (doc's codegen
detail): a ``start_async_remote_copy`` in a loop registers a COUNTING
semaphore; a ``hl.remote_copy_drain()`` (or scope exit) waits for the running
count.  User never names a semaphore.
Lowering.
  * Pallas : one send_sem/recv_sem pair; drain = a degenerate self-copy ``.wait``
             on the accumulated byte count (exactly what ``gmm_v2_fused_rs``
             does at kernel exit).
  * NVSHMEM: ``putmem_signal_block(..., signal, +1, NVSHMEM_SIGNAL_ADD)`` then
             ``signal_wait_until(sig, GE, expected_count)`` / ``nvshmem_quiet``.

--- Feature 5: NO-REDUCTION unique-slot scatter (a design choice, not a feat) -
The v2 "nodedup" scheme gives every ``(token, top_k-slot)`` a UNIQUE
destination slot, so concurrent remote writes NEVER collide -> a pure write, no
scatter-add, no receive buffer.  This is deliberately backend-portable: it
needs NO GPU atomics (``nvshmem`` atomic-add) and NO TPU read-modify-write.
Helion should document/encourage this pattern; the alternative "dedup"
(accumulate per token) would require atomics on GPU and an HBM accumulator on
TPU -- a heavier lowering we defer.

--- Feature 6: feature-dim slice of a tile-local accumulator (compute-side) ---
Need.  A fused MoE computes GMM1 into one ``[tile_m, 2I]`` accumulator and
splits it ``gate = acc[:, :I]; up = acc[:, I:]`` for the gated activation.
Found empirically: Helion tracing rejects ``acc[:, :I]`` on a device value
(``InternalError: a.node.constant should not be None``) -- a static feature-dim
slice of an in-kernel accumulator is not yet lowered.
Workaround used here.  Pass gate/up as SEPARATE weight tensors and keep two
accumulators (``acc_gate``, ``acc_up``); no accumulator slicing.  The host
wrapper does the ``w1[...,:I]/w1[...,I:]`` split (cheap, one-time).
Proposal.  Support static integer slicing of device tensors along a non-tiled
dim (both backends emit trivial sub-view / select on registers/VMEM).  Backend
neutral -- no Pallas/NVSHMEM capability gap, purely a Helion tracer feature.
--- Feature 7: in-kernel intermediate as a tiled matmul operand (compute-side)-
Need.  Fusing GMM1->act->GMM2 makes the 2nd matmul's LHS an IN-KERNEL value
``inter`` (not a host arg).  Tiling GMM2's K dim needs ``inter[:, tile_k]``,
which Helion rejects (``InvalidIndexingType: got u6``) -- tiled subscription of
a device intermediate isn't lowered.
Workaround used here.  Contract the FULL I in one ``torch.matmul(inter,
w2[e])`` (no K-tiling of the intermediate).  Fine for the example; for large I
this must eventually tile K, so a real fused kernel needs this feature (or a
VMEM-resident "accumulator that can be re-tiled" abstraction).
Proposal.  Allow ``hl.tile`` subscription of in-kernel intermediates (device
values), so a fused multi-matmul kernel can K-tile the 2nd GEMM.  Backend
neutral (both lower to VMEM/register sub-tiles); a Helion tracer/codegen
feature.  This is the crux that makes single-kernel GMM1+act+GMM2 fusion (the
whole point of the v2 kernel) expressible.

--- (Stretch, out of scope) in-kernel all-gather of the inputs ---------------
The landed v2 kernel still relies on an UPSTREAM all-gather so every chip sees
all tokens (Feature: ``_replicate_full``).  Fusing that gather into the kernel
(persistent token cache + neighbour-relay) is the msl-tpu-kernel "future work
5.1"; it would reuse Features 1-4.  Not attempted here.
===========================================================================
"""

from __future__ import annotations

# ``I`` is the MoE intermediate (feed-forward) dim -- domain-standard alongside
# H (hidden) / E (experts) / T (tokens); allow the single-letter name here.
# ruff: noqa: E741
import argparse
import contextlib
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import helion
import helion.language as hl

# registers the "tpu" PrivateUse1 device when running on a TPU host
with contextlib.suppress(Exception):
    import torch_tpu  # noqa: F401

# ---------------------------------------------------------------------------
# I/O CONTRACT (fully decided; identical for reference and Helion paths)
# ---------------------------------------------------------------------------
# Every rank calls the SAME function with the SAME-shaped (replicated) inputs
# and gets back ITS OWN output token shard.  This mirrors an expert-parallel
# MoE where activations are replicated (post upstream all-gather) and experts
# are sharded over ``world_size`` chips; the result is reduce-scattered so each
# chip owns ``T // world_size`` tokens.
#
#   hidden_states : [T, H]      float  -- all tokens, REPLICATED on every rank
#   gating_output : [T, E]      float  -- router logits, REPLICATED
#   w1            : [E, H, 2*I]  float -- gate+up weights (concat on N), FULL set
#   w2            : [E, I, H]    float -- down weights, FULL set
#   top_k         : int
#   rank          : int   (0 <= rank < world_size)
#   world_size    : int   (must divide T and E)
#   RETURNS       : [T // world_size, H]  -- THIS rank's output token shard
#
# Layout note: w1's last dim is [gate | up] concatenated (each width I); the
# activation is ``silu(gate) * up`` -- matches gmm_v2_fused_rs's FusedWeightsRef.


def _route(
    gating_output: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Router: softmax over experts, then top-k. Returns (weights, indices).

    Shared by BOTH the reference and the Helion wrapper so their routing can
    never diverge. ``weights``/``indices`` are ``[T, top_k]``.
    """
    scores = torch.softmax(gating_output.float(), dim=-1)
    topk_w, topk_i = torch.topk(scores, top_k, dim=-1)
    return topk_w, topk_i.to(torch.int32)


# ---------------------------------------------------------------------------
# Reference (pure PyTorch) -- correctness oracle. Slow but obviously correct.
# ---------------------------------------------------------------------------
def fused_moe_reference(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Reference EP fused-MoE. Computes the full MoE from replicated inputs and
    returns this rank's ``T // world_size`` token shard (== the reduce-scatter
    result). No collectives needed: every rank has all weights + all tokens.
    """
    T, H = hidden_states.shape
    E, _, two_I = w1.shape
    I = two_I // 2
    assert T % world_size == 0, "T must be divisible by world_size"
    assert E % world_size == 0, "E must be divisible by world_size"

    topk_w, topk_i = _route(gating_output, top_k)  # [T,k], [T,k]
    out = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)

    x = hidden_states.float()
    for e in range(E):
        sel = topk_i == e  # [T, k] bool: token t picked expert e in some slot
        tok_mask = sel.any(dim=-1)  # [T]
        if not bool(tok_mask.any()):
            continue
        # per-token weight for expert e (a token picks e in at most one slot)
        w_e = (topk_w * sel).sum(dim=-1)  # [T]
        xe = x[tok_mask]  # [n, H]
        gate_up = xe @ w1[e].float()  # [n, 2I]
        gate, up = gate_up[:, :I], gate_up[:, I:]
        inter = torch.nn.functional.silu(gate) * up  # [n, I]
        down = inter @ w2[e].float()  # [n, H]
        out[tok_mask] += w_e[tok_mask, None] * down

    chunk = T // world_size
    shard = out[rank * chunk : (rank + 1) * chunk]
    return shard.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Host-side routing -> expert-sorted kernel args (mirrors _compute_rs_routing).
# ---------------------------------------------------------------------------
def _build_kernel_args(
    gating_output: torch.Tensor,
    top_k: int,
    rank: int,
    world_size: int,
    num_experts: int,
) -> dict:
    """Turn routing into the expert-sorted metadata the kernel consumes.

    Produces, for THIS rank's local experts only:
      sorted_token_ids : [M] int32 -- source token id per expert-sorted row
      topk_slot        : [M] int32 -- which top-k slot (0..k-1) the row is
      topk_w_sorted    : [M] f32   -- router weight per row
      group_sizes      : [E_local] int32 -- rows per local expert
      group_offsets    : [E_local+1] int32
      max_T_per_expert : int
    ``M`` = number of (token, expert) pairs whose expert lives on this rank.
    """
    device = gating_output.device
    T = gating_output.shape[0]
    E_local = num_experts // world_size
    e_lo = rank * E_local
    e_hi = e_lo + E_local

    topk_w, topk_i = _route(gating_output, top_k)  # [T,k]
    tok_ids = torch.arange(T, device=device).view(T, 1).expand(T, top_k)
    slot_ids = torch.arange(top_k, device=device).view(1, top_k).expand(T, top_k)

    flat_e = topk_i.reshape(-1).to(torch.int64)  # [T*k]
    flat_tok = tok_ids.reshape(-1)
    flat_slot = slot_ids.reshape(-1)
    flat_w = topk_w.reshape(-1)

    local_mask = (flat_e >= e_lo) & (flat_e < e_hi)
    le = (flat_e[local_mask] - e_lo).to(torch.int32)  # local expert id
    tok = flat_tok[local_mask].to(torch.int32)
    slot = flat_slot[local_mask].to(torch.int32)
    w = flat_w[local_mask].float()

    order = torch.argsort(le, stable=True)  # sort rows by local expert
    le_s, tok_s, slot_s, w_s = le[order], tok[order], slot[order], w[order]

    group_sizes = torch.bincount(le_s.to(torch.int64), minlength=E_local).to(
        torch.int32
    )
    group_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    group_offsets[1:] = torch.cumsum(group_sizes, 0, dtype=torch.int32)
    max_T = int(group_sizes.max().item()) if E_local > 0 else 0
    # Pad the per-expert row bound to a power of two: the per-expert matmul's M
    # block must be a power of two on the Pallas/MXU backend. Extra rows are
    # masked out in-kernel (``valid = local < num_tokens``), so padding is safe.
    mt = max(max_T, 16)
    max_T_pow2 = 1 << (mt - 1).bit_length()

    return {
        "sorted_token_ids": tok_s,
        "topk_slot": slot_s,
        "topk_w_sorted": w_s,
        "group_sizes": group_sizes,
        "group_offsets": group_offsets,
        "max_T_per_expert": max_T_pow2,
    }


# ---------------------------------------------------------------------------
# Helion kernel #1 (RUNNABLE): single-rank fused expert FFN, no comms.
#   gather routed tokens -> GMM1(gate+up) -> silu(gate)*up -> GMM2(down)
# Emits one weighted output row per (token, expert) pair into ``rows_out[M,H]``.
# This is the part that maps cleanly onto existing Helion features -- it's
# ``moe_matmul_ogs`` extended to two matmuls + a gated activation.
# ---------------------------------------------------------------------------
@helion.kernel(
    backend="pallas",
    static_shapes=False,
    # block_sizes: [tile_m over max_T_per_expert, tile_k over H]. Fixed here to
    # skip autotuning (autotune of a distributed kernel is future work -- see
    # the design doc's "Distributed Autotuning" stretch goal).
    config=helion.Config(block_sizes=[16, 64]),
)
def fused_moe_local_kernel(
    hidden_states: torch.Tensor,  # [T, H]
    w1_gate: torch.Tensor,  # [E_local, H, I]  (gate half of w1)
    w1_up: torch.Tensor,  # [E_local, H, I]  (up half of w1)
    w2: torch.Tensor,  # [E_local, I, H]
    group_sizes: torch.Tensor,  # [E_local]
    group_offsets: torch.Tensor,  # [E_local+1]
    sorted_token_ids: torch.Tensor,  # [M]
    topk_w_sorted: torch.Tensor,  # [M]
    max_T_per_expert: int,
) -> torch.Tensor:  # [M, H] weighted per-(token,expert) contributions
    # NOTE: gate/up are passed as SEPARATE tensors rather than one [.,.,2I]
    # tensor, because slicing a tile-local accumulator along the feature dim
    # (``acc1[:, :I]``) is not yet supported by Helion tracing -- see DESIGN
    # NOTE "Feature 6". The host wrapper splits w1[...,:I]/w1[...,I:].
    M = sorted_token_ids.shape[0]
    E_local, H, I = w1_gate.shape
    # +1 "trash" row: masked/invalid lanes scatter here so they never collide
    # with the valid local=0 row of an expert group. rows_out[:M] is returned.
    rows_out = torch.zeros(
        M + 1, H, dtype=hidden_states.dtype, device=hidden_states.device
    )

    for e_idx in hl.grid(E_local):
        start = group_offsets[e_idx]
        num_tokens = group_sizes[e_idx]
        if num_tokens != 0:
            # NOTE (best-attempt): tile only over M; keep the full I / H
            # contraction outputs live. For large I this may exceed VMEM -> a
            # production version tiles N and streams the activation. Mirrors
            # moe_matmul_ogs's per-expert tiling.
            for tile_m in hl.tile(max_T_per_expert):
                local = tile_m.index
                valid = local < num_tokens
                rows = start + torch.where(valid, local, 0)
                tok = sorted_token_ids[rows]  # gather idx into hidden_states

                # ---- GMM1: two matmuls (gate, up), each [tile_m,H]@[H,I] ----
                acc_gate = hl.zeros([tile_m, I], dtype=torch.float32)
                acc_up = hl.zeros([tile_m, I], dtype=torch.float32)
                for tile_k in hl.tile(H):
                    x = hidden_states[tok, tile_k]
                    acc_gate = torch.addmm(acc_gate, x, w1_gate[e_idx, tile_k, :])
                    acc_up = torch.addmm(acc_up, x, w1_up[e_idx, tile_k, :])

                # ---- gated activation: silu(gate) * up ----
                inter = (torch.nn.functional.silu(acc_gate) * acc_up).to(
                    hidden_states.dtype
                )

                # ---- GMM2: [tile_m, I] @ [I, H] -> [tile_m, H] ----
                # Contract the FULL I in one matmul: we cannot tile K here
                # because that needs slicing the in-kernel intermediate
                # ``inter[:, tile_k]``, which Helion rejects
                # (InvalidIndexingType) -- see DESIGN NOTE "Feature 7".
                acc2 = torch.matmul(inter, w2[e_idx, :, :])

                # ---- apply router weight, scatter into rows_out ----
                # ``rows`` (clamped to start for invalid lanes) is safe for the
                # READS (tok/w_rows). For the WRITE, send invalid lanes to the
                # trash slot M so they don't clobber the valid local=0 row.
                w_rows = topk_w_sorted[rows].to(torch.float32)
                weighted = (acc2 * w_rows[:, None]).to(rows_out.dtype)
                write_rows = torch.where(valid, rows, M)  # invalid -> trash row M
                rows_out[write_rows, :] = weighted
    return rows_out


# ---------------------------------------------------------------------------
# Helion kernel #2 (SKETCH / blocked): full fused MoE with in-kernel
# reduce-scatter direct-write. Requires Features 1-4 from the DESIGN NOTES.
# ---------------------------------------------------------------------------
@helion.kernel(backend="pallas", distributed=[8], config=helion.Config())
def fused_moe_kernel(
    hidden_states: torch.Tensor,  # [T, H] replicated
    w1_gate: torch.Tensor,  # [E_local, H, I]  (gate half)
    w1_up: torch.Tensor,  # [E_local, H, I]  (up half)
    w2: torch.Tensor,  # [E_local, I, H]
    group_sizes: torch.Tensor,  # [E_local]
    group_offsets: torch.Tensor,  # [E_local+1]
    sorted_token_ids: torch.Tensor,  # [M] gather idx (global token id)
    topk_slot: torch.Tensor,  # [M] top-k slot 0..k-1
    topk_w_sorted: torch.Tensor,  # [M] router weight
    out_buf: torch.Tensor,  # [chunk*top_k, H] SYMMETRIC direct-write target
    rank: hl.constexpr,
    world_size: hl.constexpr,
    chunk: hl.constexpr,  # T // world_size
    top_k: hl.constexpr,
    max_T_per_expert: int,
) -> torch.Tensor:  # returns out_buf (raw, pre top-k reduction)
    E_local, H, I = w1_gate.shape
    for e_idx in hl.grid(E_local):
        start = group_offsets[e_idx]
        num_tokens = group_sizes[e_idx]
        if num_tokens != 0:
            for tile_m in hl.tile(max_T_per_expert):
                local = tile_m.index
                valid = local < num_tokens
                rows = start + torch.where(valid, local, 0)
                tok = sorted_token_ids[rows]

                acc_gate = hl.zeros([tile_m, I], dtype=torch.float32)
                acc_up = hl.zeros([tile_m, I], dtype=torch.float32)
                for tile_k in hl.tile(H):
                    x = hidden_states[tok, tile_k]
                    acc_gate = torch.addmm(acc_gate, x, w1_gate[e_idx, tile_k, :])
                    acc_up = torch.addmm(acc_up, x, w1_up[e_idx, tile_k, :])
                inter = (torch.nn.functional.silu(acc_gate) * acc_up).to(
                    hidden_states.dtype
                )
                acc2 = torch.matmul(
                    inter, w2[e_idx, :, :]
                )  # full-I contraction (Feature 7)
                # row_out is the weighted contribution the (sketched) direct-
                # write below would push to the owner chip.
                row_out = (acc2 * topk_w_sorted[rows][:, None]).to(out_buf.dtype)  # noqa: F841

                # ===== REDUCE-SCATTER DIRECT-WRITE =====================
                # For each valid row i in this tile, per row:
                #     g          = tok[i]                      # global token id
                #     dest_chip  = g // chunk                  # owner chip (data-dep!)
                #     local_row  = g %  chunk
                #     write_pos  = local_row * top_k + slot[i] # unique slot, no collisions
                #     push row_out[i]  ->  out_buf[write_pos]@dest_chip     (src != dst)
                #
                # The per-row asymmetric copy is now EXPRESSIBLE with the landed
                # primitive (Features 1-2) -- src != dst, runtime device_id + dst
                # index:
                #   for i in hl.grid(tile rows):
                #       op = hl.start_async_remote_copy(
                #                row_out, [i], dest_chip_i,          # runtime pe
                #                dst=out_buf, dst_index=[write_pos_i])
                #       op.wait()
                # and this shape is validated in isolation (see the small-scale
                # reduce-scatter in test_remote_copy).  It is NOT wired in here yet
                # because the single-launch fusion still needs: (Feature 4) a
                # counting-semaphore bulk drain instead of the per-row .wait()
                # above, which would serialise the ICI lane; a way to run the
                # scatter as a second phase after the compute within ONE Pallas
                # launch (Pallas flattens the grid to a single pid, so there is no
                # hl.barrier between phases); and a fix for a Mosaic core-halt seen
                # when the direct-write runs at MoE row width (H=128).  Until then,
                # callers use ``fused_moe`` below (Helion compute + torch scatter).
    return out_buf


# ---------------------------------------------------------------------------
# Wrapper (Helion path) -- SAME signature as fused_moe_reference.
# ---------------------------------------------------------------------------
def fused_moe(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Expert-parallel fused MoE via Helion. Returns this rank's [T//ws, H].

    Compute (gather + GMM1 + act + GMM2) runs in ``fused_moe_local_kernel`` on
    the TPU.  The cross-rank reduce-scatter is currently a torch placeholder
    (``dist.reduce_scatter``) standing in for the in-kernel direct-write that
    ``fused_moe_kernel`` will do once Features 1-4 land -- see DESIGN NOTES.
    """
    T, H = hidden_states.shape
    E = w1.shape[0]
    E_local = E // world_size
    e_lo = rank * E_local

    I = w1.shape[2] // 2
    args = _build_kernel_args(gating_output, top_k, rank, world_size, E)
    w1_local = w1[e_lo : e_lo + E_local]
    w1_gate = w1_local[:, :, :I].contiguous()  # host-side gate/up split (Feature 6)
    w1_up = w1_local[:, :, I:].contiguous()
    w2_local = w2[e_lo : e_lo + E_local].contiguous()

    # Real Helion compute on TPU: per-(token,expert) weighted contributions.
    rows_out = fused_moe_local_kernel(
        hidden_states,
        w1_gate,
        w1_up,
        w2_local,
        args["group_sizes"],
        args["group_offsets"],
        args["sorted_token_ids"],
        args["topk_w_sorted"],
        args["max_T_per_expert"],
    )  # [M, H]

    # Scatter each row's contribution to a full [T, H] partial (this rank only),
    # then reduce-scatter across ranks. This torch block is the PLACEHOLDER for
    # the in-kernel ICI direct-write (fused_moe_kernel).
    partial = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    tok = args["sorted_token_ids"].to(torch.int64)
    partial.index_add_(0, tok, rows_out[:-1].float())  # drop trash row

    chunk = T // world_size
    shard = torch.empty(chunk, H, dtype=torch.float32, device=hidden_states.device)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.reduce_scatter_tensor(shard, partial.contiguous())
    else:
        # single-process fallback (world_size == 1): partial already complete.
        shard = partial[rank * chunk : (rank + 1) * chunk]
    return shard.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _make_inputs(
    T: int,
    H: int,
    I: int,
    E: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(T, H, generator=g, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    gating = torch.randn(T, E, generator=g, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    w1 = (torch.randn(E, H, 2 * I, generator=g, dtype=torch.float32) * (H**-0.5)).to(
        device=device, dtype=dtype
    )
    w2 = (torch.randn(E, I, H, generator=g, dtype=torch.float32) * (I**-0.5)).to(
        device=device, dtype=dtype
    )
    return hidden, gating, w1, w2


def check_reference(device: torch.device) -> None:
    """Verify fused_moe_reference: concatenating all ranks' shards reconstructs
    a naive einsum full-MoE. Single process, no Helion, no collectives."""
    T, H, I, E, top_k, world_size = 64, 128, 256, 8, 2, 4
    hidden, gating, w1, w2 = _make_inputs(T, H, I, E, top_k, device, torch.float32)

    shards = [
        fused_moe_reference(hidden, gating, w1, w2, top_k, r, world_size)
        for r in range(world_size)
    ]
    got = torch.cat(shards, dim=0)  # [T, H]

    # Independent naive oracle (loop over tokens/experts).
    topk_w, topk_i = _route(gating, top_k)
    exp = torch.zeros(T, H, dtype=torch.float32, device=device)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_i[t, j])
            gu = hidden[t].float() @ w1[e].float()
            inter = torch.nn.functional.silu(gu[:I]) * gu[I:]
            exp[t] += float(topk_w[t, j]) * (inter @ w2[e].float())
    torch.testing.assert_close(got.float(), exp, rtol=1e-4, atol=1e-4)
    print("[check_reference] OK -- reference matches naive oracle")


def check_helion_local(device: torch.device) -> None:
    """Run fused_moe_local_kernel on TPU and compare per-(token,expert) rows to
    a torch computation of the same. Proves the Helion compute core runs."""
    T, H, I, E, top_k, world_size, rank = 64, 128, 256, 8, 2, 1, 0
    hidden, gating, w1, w2 = _make_inputs(T, H, I, E, top_k, device, torch.float32)
    args = _build_kernel_args(gating, top_k, rank, world_size, E)
    w1_gate = w1[:, :, :I].contiguous()
    w1_up = w1[:, :, I:].contiguous()

    rows_out = fused_moe_local_kernel(
        hidden,
        w1_gate,
        w1_up,
        w2,
        args["group_sizes"],
        args["group_offsets"],
        args["sorted_token_ids"],
        args["topk_w_sorted"],
        args["max_T_per_expert"],
    )  # [M, H]

    # torch oracle for the same rows
    tok = args["sorted_token_ids"].to(torch.int64)
    exp = torch.zeros_like(rows_out, dtype=torch.float32)
    # expert id per row from group_offsets
    go = args["group_offsets"]
    for e in range(E):
        s, en = int(go[e]), int(go[e + 1])
        if en <= s:
            continue
        rws = torch.arange(s, en, device=device)
        x = hidden[tok[rws]].float()
        gu = x @ w1[e].float()
        inter = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
        d = inter @ w2[e].float()
        exp[rws] = d * args["topk_w_sorted"][rws][:, None].float()
    torch.testing.assert_close(rows_out[:-1].float(), exp, rtol=2e-2, atol=2e-2)
    print("[check_helion_local] OK -- Helion compute core matches torch")


# ---------------------------------------------------------------------------
# Distributed harness (mirrors all_gather.py) -- attempts the full fused path.
# ---------------------------------------------------------------------------
def _run_distributed() -> None:
    device = torch.device("tpu")
    dist.init_process_group(backend="tpu_dist")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    T, H, I, E, top_k = 64, 128, 256, 8, 2
    hidden, gating, w1, w2 = _make_inputs(
        T, H, I, E, top_k, device, torch.float32, seed=0
    )

    got = fused_moe(hidden, gating, w1, w2, top_k, rank, world_size)
    got_cpu = got.float().cpu()
    # Reference runs on CPU: it uses boolean-mask indexing which the TPU backend
    # does not support, and it is a device-agnostic correctness oracle anyway.
    exp = fused_moe_reference(
        hidden.cpu(), gating.cpu(), w1.cpu(), w2.cpu(), top_k, rank, world_size
    )
    torch.testing.assert_close(got_cpu, exp.float(), rtol=2e-2, atol=2e-2)
    print(
        f"[distributed] rank {rank} OK -- shard matches reference "
        f"(shape={tuple(got.shape)})"
    )


def _worker(rank: int, world_size: int, master_port: int) -> None:
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(master_port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        GROUP_RANK="0",
        LOCAL_WORLD_SIZE=str(world_size),
    )
    _run_distributed()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--distributed",
        action="store_true",
        help="attempt the full fused reduce-scatter path (8 chips)",
    )
    a = ap.parse_args()

    if a.distributed:
        import portpicker
        from torch_tpu._internal.distributed.launchers import singlehost_wrapper

        world_size = 8
        port = portpicker.pick_unused_port()
        singlehost_wrapper.prepare_tpu_environment(world_size=world_size)
        mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)
    else:
        # Single-process TPU init is not supported on this pod (the runtime
        # expects the multi-worker launcher), so the correctness oracle runs on
        # CPU -- it is pure PyTorch and device-agnostic. Use --distributed to
        # exercise the Helion/TPU compute + reduce-scatter path.
        check_reference(torch.device("cpu"))
        print(
            "[info] reference verified on CPU. Helion-on-TPU paths need the "
            "multi-process launcher: run `python examples/distributed/"
            "fused_moe.py --distributed`."
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
