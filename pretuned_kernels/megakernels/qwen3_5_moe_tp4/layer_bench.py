# ruff: noqa: ANN001, ANN201, ANN202
"""The merged Qwen3.5 TP4 layer megakernel (a layer's MoE + the next layer's GDN
mixer, one launch) vs the two standalone kernels back to back: check and timing.

torchrun --nproc-per-node=4 -m pretuned_kernels.megakernels.qwen3_5_moe_tp4.layer_bench \
    [--check] [--bench [--merged-only | --separate-only]]

LAYER (default qwen3_5_layer_tp4): the merged kernel's module; its LAYER_MOE /
LAYER_MIXER name the standalone kernels.

--check: one layer, fresh mixer state for each path; the merged kernel's MoE and
mixer outputs and the updated mixer state must equal the standalone kernels'
bitwise.

--bench: LAYERS (default 8) distinct layers (MoE weights, mixer weights and
state), REPS (default 4) steps in one CUDA graph (two eager steps before
capture); us per layer (median and min of 10 replays) for the standalone pair (2
launches) and the merged kernel (1 launch). TP-replicated weights (router, shared
gate, norm weights) are seeded by layer index only, so every rank routes the same
tokens to the same experts. ROUTER_SUPPORT=S scales the router rows outside a
layer-seeded set of S experts by 0.01 (about S distinct experts per layer);
XCORR=m correlates the window's tokens (the MoE input shares sqrt(m) of a common
direction).

MOE_RESERVED / MIXER_RESERVED / LAYER_RESERVED: each kernel's
persistent_reserved_sms (unset: HELION_PERSISTENT_RESERVED_SMS for all).
"""

from __future__ import annotations

import os
import sys

from pretuned_kernels.megakernels.qwen3_5_gdn_tp4 import _setup as ms
from pretuned_kernels.megakernels.qwen3_5_moe_tp4 import _setup
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.layer_compile import layer_module
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.layer_compile import merged_args
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.layer_compile import separate_kernels
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion


def reserve(kernel, var):
    """The kernel with persistent_reserved_sms from ``var`` (unset: unchanged)."""
    if var not in os.environ:
        return kernel
    return helion.kernel(
        kernel.fn,
        config=helion.Config(**dict(kernel.configs[0])),
        static_shapes=False,
        backend="gluon",
        triton_do_not_specialize=False,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
        persistent_reserved_sms=int(os.environ[var]),
        dot_precision=kernel.settings.dot_precision,
        fast_math=kernel.settings.fast_math,
    )


def graph_time(fn, n, reps):
    """Median and min us per layer of ``reps`` steps of ``fn(i)`` for i < n."""
    for _ in range(2):
        for i in range(n):
            fn(i)
        torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            for i in range(n):
                fn(i)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    ts = []
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
        ts.append(s.elapsed_time(e) * 1e3 / (reps * n))
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def main() -> None:
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev)
    gname = dist.group.WORLD.group_name
    symm_mem.set_backend("NVSHMEM")
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), 32 * 1024))
    lmod = layer_module()
    moe, mixer = separate_kernels(lmod)
    merged = reserve(lmod.qwen3_5_layer_tp4, "LAYER_RESERVED")
    moe, mixer = reserve(moe, "MOE_RESERVED"), reserve(mixer, "MIXER_RESERVED")
    shape = _setup.Shape()
    n = int(os.environ.get("LAYERS", "8")) if "--bench" in sys.argv else 1
    reps = int(os.environ.get("REPS", "4"))

    mix_layers, states, moe_layers = [], [], []
    for i in range(n):
        lw = ms.make_layer(1000 * i + rank, dev)
        g = torch.Generator(device=dev)
        g.manual_seed(1000 * i + 555)
        lw["norm_weight"] = (
            1 + 0.1 * torch.randn(shape.hidden, generator=g, device=dev)
        ).to(torch.bfloat16)
        mix_layers.append(lw)
        states.append(ms.make_state(2000 * i + rank, 2, 5, dev))
        mw = _setup.make_weights(shape, seed=1000 * i + rank)
        gr = torch.Generator(device=dev)
        gr.manual_seed(1000 * i + 777)
        mw["router"] = (
            torch.randn(shape.num_experts, shape.hidden, generator=gr, device=dev)
            * 0.02
        ).to(torch.bfloat16)
        support = int(os.environ.get("ROUTER_SUPPORT", "0"))
        if support:
            gs = torch.Generator(device=dev)
            gs.manual_seed(1000 * i + 999)
            off = torch.randperm(shape.num_experts, generator=gs, device=dev)[support:]
            mw["router"][off] = (mw["router"][off].float() * 0.01).to(torch.bfloat16)
        mw["gate_w"] = (torch.randn(shape.hidden, generator=gr, device=dev) * 0.02).to(
            torch.bfloat16
        )
        for k in ("w13_log", "w13s_log", "w2_log", "w2s_log"):
            mw.pop(k, None)
        moe_layers.append(mw)
    torch.cuda.empty_cache()
    x = ms.make_inputs(77 + rank, dev)
    # The MoE's input (a post-all-reduce RMSNorm output) and the residual are the
    # same on every rank, as under TP.
    shared_in = ms.make_inputs(4242, dev)
    x["residual"], x["hidden"] = shared_in["residual"], shared_in["hidden"]
    xcorr = float(os.environ.get("XCORR", "0"))
    if xcorr:
        gx = torch.Generator(device=dev)
        gx.manual_seed(4343)
        common = torch.randn(1, shape.hidden, generator=gx, device=dev)
        own = torch.randn(x["hidden"].shape[0], shape.hidden, generator=gx, device=dev)
        x["hidden"] = (xcorr**0.5 * common + (1 - xcorr) ** 0.5 * own).to(
            torch.bfloat16
        )
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    norm2 = (1 + 0.1 * torch.randn(shape.hidden, generator=g, device=dev)).to(
        torch.bfloat16
    )

    def symm():
        out = []
        for _ in range(2):
            p = symm_mem.empty((ms.T, ms.HIDDEN), dtype=torch.bfloat16, device=dev)
            symm_mem.rendezvous(p, gname)
            out.append(p)
        return out

    mix_parts, moe_parts = symm(), symm()
    names = {"moe": list(moe.signature.parameters)}
    names["mixer"] = list(mixer.signature.parameters)

    def mixer_args(i, hidden=None, residual=None):
        vals = ms.kernel_args(mix_layers[i], states[i], x, mix_parts[i % 2], gname)
        a = dict(zip(names["mixer"], vals, strict=True))
        if hidden is not None:
            a["hidden_states"], a["residual"] = hidden, residual
        return a

    def moe_args(i, hidden, residual):
        vals = (
            hidden,
            *_setup.kernel_weight_args(moe_layers[i], shape),
            residual,
            norm2,
            moe_parts[i % 2],
            gname,
        )
        return dict(zip(names["moe"], vals, strict=True))

    def call(kernel, key, a):
        return kernel(*[a[name] for name in names[key]])

    def separate(i):
        """MoE(i) on the layer input, then mixer(i) on the MoE's output."""
        out = call(moe, "moe", moe_args(i, x["hidden"], x["residual"]))
        return out, call(mixer, "mixer", mixer_args(i, out[0], out[1]))

    def fused(i):
        ma = moe_args(i, x["hidden"], x["residual"])
        return merged(*merged_args(lmod, ma, mixer_args(i)))

    def split(out_m):
        """(MoE outputs, mixer outputs) of a merged call (the mixer's come first)."""
        k = len(out_m) - lmod.LAYER_FIRST_RETURNS
        return out_m[k:], out_m[:k]

    ok = True
    if "--check" in sys.argv:
        st0 = {k: v.clone() for k, v in states[0].items()}
        out_s, mo_s = separate(0)
        torch.cuda.synchronize()
        st_s = {k: v.clone() for k, v in states[0].items()}
        for k, v in states[0].items():
            v.copy_(st0[k])
        dist.barrier()
        out_m = fused(0)
        torch.cuda.synchronize()
        moe_m, mix_m = split(out_m)
        assert len(moe_m) == len(out_s) and len(mix_m) == len(mo_s)
        checks = [
            ("moe output", moe_m[0], out_s[0]),
            ("moe residual_out", moe_m[1], out_s[1]),
            ("moe topk_ids", moe_m[4], out_s[4]),
            ("mixer output", mix_m[0], mo_s[0]),
            ("mixer residual_out", mix_m[1], mo_s[1]),
        ]
        checks += [
            (f"state {k}", states[0][k], st_s[k])
            for k in st_s
            if states[0][k].dtype != torch.int32
        ]
        for label, a, b in checks:
            # torch.equal has no FP8 kernel: compare the bytes.
            same = torch.equal(
                *(
                    t.view(torch.uint8)
                    if t.element_size() == 1 and t.is_floating_point()
                    else t
                    for t in (a, b)
                )
            )
            err = (a.float() - b.float()).abs().max().item()
            ok &= same
            print(
                f"rank{rank} {label}: {'bitwise' if same else f'DIFF max abs {err:.3g}'}",
                flush=True,
            )
        ids = out_s[4][: ms.T, : shape.top_k]
        print(
            f"rank{rank} distinct={len(set(ids.flatten().tolist()))} "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )

    if "--bench" in sys.argv:
        nan = (float("nan"),) * 2
        sep = nan if "--merged-only" in sys.argv else graph_time(separate, n, reps)
        fus = nan if "--separate-only" in sys.argv else graph_time(fused, n, reps)
        per = [
            split(fused(i))[0] if "--merged-only" in sys.argv else separate(i)[0]
            for i in range(n)
        ]
        torch.cuda.synchronize()
        distinct = "/".join(
            str(len(set(out[4][: ms.T, : shape.top_k].flatten().tolist())))
            for out in per
        )
        print(
            f"rank{rank} layers={n} distinct={distinct} us/layer: separate med "
            f"{sep[0]:.2f} min {sep[1]:.2f} | merged med {fus[0]:.2f} min {fus[1]:.2f} "
            f"({100 * (fus[0] / sep[0] - 1):+.1f}%)",
            flush=True,
        )
    dist.destroy_process_group()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
