# ruff: noqa: ANN201, ANN202
"""Cold-weight timing of the Qwen3.5 TP4 MoE megakernel.

torchrun --nproc-per-node=4 -m pretuned_kernels.megakernels.qwen3_5_moe_tp4.bench

LAYERS distinct weight copies (default 8, ~0.8 GB each) run back to back, REPS
(default 4) times in one CUDA graph, so each launch streams its experts from HBM
as consecutive layers do; the previous launch's all-reduce keeps the ranks in
step. Reports us per launch (graph time / launches, median and min of 10
replays) and the distinct-expert count. TOKENS (default 7) window tokens; XCORR
(default 0) correlates them (see below); SEED (default 3) draws the input.
"""

from __future__ import annotations

import os

from pretuned_kernels.megakernels.qwen3_5_moe_tp4 import _setup
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.qwen3_5_moe_tp4 import qwen3_5_moe_tp4
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev)
    group = dist.group.WORLD
    symm_mem.set_backend("NVSHMEM")
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), 32 * 1024))
    T = int(os.environ.get("TOKENS", "7"))
    L = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    shape = _setup.Shape()
    layers = [_setup.make_weights(shape, seed=1000 * i + rank) for i in range(L)]
    # TP replicates the router and the shared gate (same input -> same routing on every rank);
    # only the expert / shared-expert shards differ per rank.
    for i, lw in enumerate(layers):
        gr = torch.Generator(device=dev)
        gr.manual_seed(1000 * i + 777)
        lw["router"] = (
            torch.randn(shape.num_experts, shape.hidden, generator=gr, device=dev)
            * 0.02
        ).to(torch.bfloat16)
        lw["gate_w"] = (torch.randn(shape.hidden, generator=gr, device=dev) * 0.02).to(
            torch.bfloat16
        )
    for lw in layers:  # drop the logical copies
        for k in ("w13_log", "w13s_log", "w2_log", "w2s_log"):
            del lw[k]
    torch.cuda.empty_cache()
    g = torch.Generator(device=dev)
    g.manual_seed(int(os.environ.get("SEED", "3")))
    # XCORR = m: tokens share sqrt(m) of a common direction (verify tokens overlap in experts;
    # m = 0 -> ~63 distinct of 70 picks, 0.6 -> ~43, 0.7 -> ~36).
    m = float(os.environ.get("XCORR", "0"))
    base = torch.randn(1, shape.hidden, generator=g, device=dev)
    x = (
        m**0.5 * base
        + (1 - m) ** 0.5 * torch.randn(T, shape.hidden, generator=g, device=dev)
    ).to(torch.bfloat16)
    residual = torch.randn(T, shape.hidden, generator=g, device=dev).to(torch.bfloat16)
    norm_w = torch.ones(shape.hidden, device=dev, dtype=torch.bfloat16)
    partials = []
    for _ in range(2):
        p = symm_mem.empty((T, shape.hidden), dtype=torch.bfloat16, device=dev)
        symm_mem.rendezvous(p, group.group_name)
        partials.append(p)
    args = [_setup.kernel_weight_args(lw, shape) for lw in layers]

    def step():
        for i in range(L):
            # The same input every layer: a fixed routing.
            last = qwen3_5_moe_tp4(
                x, *args[i], residual, norm_w, partials[i % 2], group.group_name
            )
        return last

    # Two eager steps before capture: hl.prefetch tensor maps are marked written only after
    # the first call, otherwise the captured graph re-copies them (H2D memcpy nodes) every launch.
    step()
    torch.cuda.synchronize()
    out = step()
    torch.cuda.synchronize()
    ids = out[4][:T, :10]
    distinct = len(set(ids.flatten().tolist()))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(10):
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        dist.barrier()
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / (reps * L))
    times.sort()
    mb = distinct * (512 * 2048 + 512 * 256 + 4096 * 128 + 4096 * 16) / 1e6 + 4.2 + 3.1
    med = times[len(times) // 2]
    print(
        f"rank{rank} T={T} distinct={distinct} us/launch med {med:.2f} min {times[0]:.2f} "
        f"(~{mb:.0f} MB -> {mb / med:.2f} TB/s)",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
