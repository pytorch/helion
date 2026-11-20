# ruff: noqa: ANN001, ANN201
"""Check the Qwen3.5 TP4 MoE megakernel against a torch reference on the logical
(unshuffled) weights.

torchrun --nproc-per-node=4 -m pretuned_kernels.megakernels.qwen3_5_moe_tp4.check

Two routing cases at TOKENS (default 7) tokens: "spread" (independent tokens) and
"clustered" (every token the same row: one expert set). Per rank: the top-k ids
must equal the reference's (as sets); the reference reuses the kernel's NVFP4
input and activation copies (ties round by the kernel's division), and reports
how far those activations are from the exact quantization. The residual output
(the all-gathered references' sum in rank order + residual) and the normed output
must be within 2e-2 relative RMS; a second run must be bitwise equal.
"""

from __future__ import annotations

import os
import sys

from pretuned_kernels.megakernels.qwen3_5_moe_tp4 import _setup
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import NORM_EPS
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.qwen3_5_moe_tp4 import qwen3_5_moe_tp4
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def activation_by_rank(out, top_k, tokens):
    """The kernel's activation rows (top-k pairs, the expert's first pair of the
    window holding them) in distinct-expert order, as the reference keeps them."""
    act_q, act_s = out[9], out[10]
    first = {}
    for pair, expert in enumerate(out[4][:tokens, :top_k].flatten().tolist()):
        first.setdefault(expert, pair)
    order = [first[e] for e in sorted(first)]
    return act_q[order], act_s[order]


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    group = dist.group.WORLD
    symm_mem.set_backend("NVSHMEM")
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), 32 * 1024))
    T = int(os.environ.get("TOKENS", "7"))
    shape = _setup.Shape()
    w = _setup.make_weights(shape, seed=100 + rank)
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    cases = {
        "spread": torch.randn(T, shape.hidden, generator=g, device="cuda"),
        "clustered": torch.randn(1, shape.hidden, generator=g, device="cuda").repeat(
            T, 1
        ),
    }
    residual = torch.randn(T, shape.hidden, generator=g, device="cuda").to(
        torch.bfloat16
    )
    norm_w = (1 + 0.1 * torch.randn(shape.hidden, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    partial = symm_mem.empty(
        (T, shape.hidden), dtype=torch.bfloat16, device=torch.device("cuda", local)
    )
    symm_mem.rendezvous(partial, group.group_name)
    args_w = _setup.kernel_weight_args(w, shape)
    ok = True
    for name, xf in cases.items():
        x = xf.to(torch.bfloat16)
        out = qwen3_5_moe_tp4(x, *args_w, residual, norm_w, partial, group.group_name)
        torch.cuda.synchronize()
        act_q, act_s = activation_by_rank(out, shape.top_k, T)
        ref_part, ref_ids, _, _ = _setup.reference_partial(
            x, w, shape, quant=(out[7], out[8], act_q, act_s)
        )
        print(
            f"rank{rank} {name} kernel-vs-exact activation quant rel "
            f"{getattr(_setup.reference_partial, 'act_err', 0):.3g}"
        )
        ids = out[4][:T, : shape.top_k]
        if not torch.equal(ids.sort(-1).values, ref_ids.sort(-1).values):
            print(f"rank{rank} {name}: topk ids differ\n{ids}\n{ref_ids}")
            ok = False
        parts = [torch.empty_like(ref_part) for _ in range(world)]
        dist.all_gather(parts, ref_part)
        total = sum(p.float() for p in parts) + residual.float()
        tb = total.to(torch.bfloat16).float()
        normed = (
            tb
            * torch.rsqrt(tb.square().mean(-1, keepdim=True) + NORM_EPS)
            * norm_w.float()
        ).to(torch.bfloat16)
        for label, got, exp in (
            ("residual_out", out[1], total.to(torch.bfloat16)),
            ("output", out[0], normed),
        ):
            err = (got.float() - exp.float()).abs()
            rel = (err.norm() / exp.float().norm()).item()
            print(
                f"rank{rank} {name} {label}: max abs {err.max().item():.4g} "
                f"rel-rms {rel:.3g}"
            )
            if not rel <= 2e-2:  # NaN fails too
                ok = False
        out2 = qwen3_5_moe_tp4(x, *args_w, residual, norm_w, partial, group.group_name)
        if not torch.equal(out2[0], out[0]):
            print(f"rank{rank} {name}: NOT deterministic")
            ok = False
    print(f"rank{rank} {'PASS' if ok else 'FAIL'}", flush=True)
    dist.destroy_process_group()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
