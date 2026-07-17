"""Expert-parallel fused MoE + ICI reduce-scatter in Helion (one process per chip).

A distributed, single-launch collective MoE for the Pallas/TPU backend: per chip,
for each of this chip's experts, run GMM1(gate|up)+SiLU -> GMM2(down) on its routed
tokens and direct-write each output row into its destination chip's output buffer
(an inter-chip all-to-all reduce-scatter), closing with a counting receive-drain.
The comms+compute fusion is the whole point, so it only makes sense on >1 chip --
run it via ``mp.spawn`` (torch_tpu cannot drive multiple chips from one process).

It uses two features that let it scale to large hidden/intermediate sizes and
expert counts:

1. GMM2 output-tiling: ``down = matmul(inter, w2[e, :, th])`` streams only a
   ``[I, th]`` slice of the down-projection weight per H-tile, so the full
   ``w2[e] = [I, H]`` is never VMEM-resident.

2. Batched (dest-blocked) reduce-scatter: the routing is *dest-sorted* so each
   expert's CAP rows form ``WORLD_SIZE`` contiguous ``[cap_pc, H]`` blocks (block
   ``d`` -> chip ``d``), and the scatter issues ONE block DMA per dest chip via
   ``hl.start_async_remote_copy(..., block_rows=cap_pc)`` -- ``WORLD_SIZE`` copies
   per expert, not one per row (which would blow up Mosaic compile time).

``out_buf`` is global-expert-major ``[E_GLOBAL*cap_pc, H]``; chip ``d`` receives,
from every global expert ``g``, that expert's ``cap_pc`` outputs for ``d``'s tokens
at ``out_buf[g*cap_pc : (g+1)*cap_pc]``. A host reindex (closed-form post_tl /
post_slot) then applies the top-k weights.

Routing is a fixed balanced pattern: chip-local assignment ``a = t_local*top_k +
slot``, ``expert(a) = a % E_GLOBAL``; expert ``g``'s ``j``-th token has
``a = g + j*E_GLOBAL``. Requires ``CAP % WORLD_SIZE == 0`` and
``(CHUNK*top_k) % E_GLOBAL == 0``. Shape/dtype are env-configurable (MSL_*).
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import helion
import helion.language as hl

with contextlib.suppress(Exception):
    import torch_tpu  # noqa: F401

WORLD_SIZE = int(os.environ.get("MSL_WORLD_SIZE", "8"))
E_GLOBAL = int(os.environ.get("MSL_E", "16"))
TOPK = int(os.environ.get("MSL_TOPK", "2"))
T = int(os.environ.get("MSL_T", "1024"))
H = int(os.environ.get("MSL_H", "128"))
I_DIM = int(os.environ.get("MSL_I", "256"))
DTYPE = torch.bfloat16 if os.environ.get("MSL_BF16", "0") == "1" else torch.float32
E_LOCAL = E_GLOBAL // WORLD_SIZE
CHUNK = T // WORLD_SIZE
CAP = (T * TOPK) // E_GLOBAL
CAP_PC = CAP // WORLD_SIZE  # rows per dest-chip block
assert CAP % WORLD_SIZE == 0, f"CAP={CAP} not divisible by WORLD_SIZE={WORLD_SIZE}"
assert (CHUNK * TOPK) % E_GLOBAL == 0, "CHUNK*TOPK must be divisible by E_GLOBAL"
_HB = min(512, H)
_KB = min(512, H)


@helion.kernel(
    backend="pallas",
    distributed=[WORLD_SIZE],
    static_shapes=True,
    config=helion.Config(block_sizes=[128, _KB, _HB]),
)
def fused_moe_rs(
    gathered: torch.Tensor,  # [E_local, CAP, H] pre-gathered routed tokens (dest-sorted)
    wg: torch.Tensor,  # [E_local, H, I] gate
    wu: torch.Tensor,  # [E_local, H, I] up
    w2: torch.Tensor,  # [E_local, I, H] down
    glob_e: torch.Tensor,  # [E_local] int32 global expert id per local expert
    scratch: torch.Tensor,  # [E_local, CAP, H] fp32 per-row FFN (dest-blocked rows)
    out_buf: torch.Tensor,  # [E_GLOBAL*cap_pc, H] fp32 symmetric RS target
    ecap: hl.constexpr,  # CAP
    idim: hl.constexpr,  # I
    hdim: hl.constexpr,  # H
    cap_pc: hl.constexpr,  # CAP // WORLD_SIZE
    wsize: hl.constexpr,  # WORLD_SIZE
) -> torch.Tensor:
    e_experts = wg.shape[0]
    for e in hl.grid(e_experts):
        for tc in hl.tile(ecap):
            acc_g = hl.zeros([tc, idim], dtype=torch.float32)
            acc_u = hl.zeros([tc, idim], dtype=torch.float32)
            for tk in hl.tile(gathered.shape[2]):
                x = gathered[e, tc, tk]
                acc_g = acc_g + torch.matmul(x, wg[e, tk, :]).to(torch.float32)
                acc_u = acc_u + torch.matmul(x, wu[e, tk, :]).to(torch.float32)
            inter = (torch.nn.functional.silu(acc_g) * acc_u).to(wg.dtype)
            for th in hl.tile(hdim):  # GMM2 H-output tiling (stream w2[e,:,th])
                down = torch.matmul(inter, w2[e, :, th]).to(scratch.dtype)
                scratch[e, tc.index, th] = down
        # Batched dest-blocked reduce-scatter: ONE block DMA per dest chip.
        # Rows [d*cap_pc:(d+1)*cap_pc] of expert e go to chip d, landing in
        # out_buf[glob_e*cap_pc : ...] (global-expert-major) via block_rows.
        for d in range(wsize):
            op = hl.start_async_remote_copy(
                scratch,
                [e, d * cap_pc],
                d,
                dst=out_buf,
                dst_index=[glob_e[e] * cap_pc],
                block_rows=cap_pc,
            )
            op.wait_send()
        if e == e_experts - 1:
            hl.wait_async_remote_recv(out_buf, out_buf.shape[0])
    return out_buf


def _post_maps() -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form (global_expert g, j) -> (t_local, slot) for the fixed routing.

    Chip-local assignment a = t_local*TOPK + slot; expert(a) = a % E_GLOBAL; so
    expert g's j-th assignment has a = g + j*E_GLOBAL.
    """
    g = torch.arange(E_GLOBAL).view(E_GLOBAL, 1)
    j = torch.arange(CAP_PC).view(1, CAP_PC)
    a = g + j * E_GLOBAL  # [E_GLOBAL, CAP_PC]
    return (a // TOPK).to(torch.int64), (a % TOPK).to(torch.int64)


def build_routing(rank: int) -> dict:
    post_tl, post_slot = _post_maps()  # [E_GLOBAL, CAP_PC]
    gathered = torch.zeros(E_LOCAL, CAP, dtype=torch.int64)
    glob_e = torch.zeros(E_LOCAL, dtype=torch.int32)
    for le in range(E_LOCAL):
        g = rank * E_LOCAL + le
        glob_e[le] = g
        for d in range(WORLD_SIZE):
            # chip d's tokens routed to global expert g, dest-block d.
            gathered[le, d * CAP_PC : (d + 1) * CAP_PC] = d * CHUNK + post_tl[g]
    return {
        "gathered": gathered,
        "glob_e": glob_e,
        "post_tl": post_tl,
        "post_slot": post_slot,
    }


def reference(
    hidden: torch.Tensor,
    gating: torch.Tensor,
    wg_all: torch.Tensor,
    wu_all: torch.Tensor,
    wd_all: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    # token gt (chip d, t_local): experts = [(t_local*TOPK+slot) % E_GLOBAL].
    x = hidden.float()
    out = torch.zeros(CHUNK, H, dtype=torch.float32)
    weights = torch.softmax(gating.float(), dim=-1)
    for t_local in range(CHUNK):
        gt = rank * CHUNK + t_local
        for slot in range(TOPK):
            e = (t_local * TOPK + slot) % E_GLOBAL
            g_ = x[gt] @ wg_all[e].float()
            u = x[gt] @ wu_all[e].float()
            inter = torch.nn.functional.silu(g_) * u
            w = float(weights[gt, e])
            out[t_local] += w * (inter @ wd_all[e].float())
    return out


def _run() -> None:
    from torch_tpu._internal.distributed import tpu_distributed

    dist.init_process_group(backend="tpu_dist")
    rank = tpu_distributed.global_device_id()
    dev = torch.device("tpu")
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(T, H, generator=g)
    gating = torch.randn(T, E_GLOBAL, generator=g)
    wg_all = torch.randn(E_GLOBAL, H, I_DIM, generator=g) * H**-0.5
    wu_all = torch.randn(E_GLOBAL, H, I_DIM, generator=g) * H**-0.5
    wd_all = torch.randn(E_GLOBAL, I_DIM, H, generator=g) * I_DIM**-0.5

    r = build_routing(rank)
    e_lo = rank * E_LOCAL
    gathered_hidden = hidden[r["gathered"].reshape(-1)].reshape(E_LOCAL, CAP, H)

    dev_args = (
        gathered_hidden.to(DTYPE).to(dev),
        wg_all[e_lo : e_lo + E_LOCAL].contiguous().to(DTYPE).to(dev),
        wu_all[e_lo : e_lo + E_LOCAL].contiguous().to(DTYPE).to(dev),
        wd_all[e_lo : e_lo + E_LOCAL].contiguous().to(DTYPE).to(dev),
        r["glob_e"].to(dev),
        torch.zeros(E_LOCAL, CAP, H, dtype=torch.float32, device=dev),
        torch.zeros(E_GLOBAL * CAP_PC, H, dtype=torch.float32, device=dev),
        hl.constexpr(CAP),
        hl.constexpr(I_DIM),
        hl.constexpr(H),
        hl.constexpr(CAP_PC),
        hl.constexpr(WORLD_SIZE),
    )
    out_buf = fused_moe_rs(*dev_args)
    dist.barrier()

    # Post: out_buf[g, j] -> token (post_tl[g,j]) with weight[.., post_slot].
    ob = out_buf.float().cpu().reshape(E_GLOBAL * CAP_PC, H)
    tl = r["post_tl"].reshape(-1)
    weights = torch.softmax(gating.float(), dim=-1)
    g_ids = (
        torch.arange(E_GLOBAL).view(E_GLOBAL, 1).expand(E_GLOBAL, CAP_PC).reshape(-1)
    )
    w_sel = weights[rank * CHUNK + tl, g_ids]  # weight[token, expert]
    contrib = ob * w_sel[:, None]
    got = torch.zeros(CHUNK, H, dtype=torch.float32).index_add_(0, tl, contrib)

    exp = reference(hidden, gating, wg_all, wu_all, wd_all, rank)
    if rank == 0:
        maxdiff = float((got - exp).abs().max())
        print(f"[rank {rank}] t5 fused MoE + RS maxdiff = {maxdiff:.5f}")
        tol = 6e-2 if torch.bfloat16 == DTYPE else 2e-2
        torch.testing.assert_close(got, exp, rtol=tol, atol=tol)
        print("OK")

    if os.environ.get("MSL_TPU_BENCH") == "1":
        import time

        def _sync() -> None:
            if torch.accelerator.is_available():
                torch.accelerator.synchronize()

        for _ in range(5):
            fused_moe_rs(*dev_args)
        _sync()
        dist.barrier()
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            fused_moe_rs(*dev_args)
        _sync()
        ms = (time.perf_counter() - t0) / iters * 1000.0
        ms_t = torch.tensor([ms], device=dev)
        dist.all_reduce(ms_t, op=dist.ReduceOp.MAX)
        if rank == 0:
            flops = 6 * H * I_DIM * (E_LOCAL * CAP) * WORLD_SIZE
            worst = float(ms_t)
            print(
                f"[helion fused-MoE t5] world={WORLD_SIZE} dtype={DTYPE} "
                f"shape(T={T},H={H},I={I_DIM},E={E_GLOBAL},tk={TOPK}) | "
                f"worst-chip {worst:.4f} ms | {flops / (worst / 1000.0) / 1e12:.1f} TFLOP/s"
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
    _run()


def main() -> None:
    import portpicker
    from torch_tpu._internal.distributed.launchers import singlehost_wrapper

    port = portpicker.pick_unused_port()
    singlehost_wrapper.prepare_tpu_environment(world_size=WORLD_SIZE)
    mp.spawn(_worker, args=(WORLD_SIZE, port), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
