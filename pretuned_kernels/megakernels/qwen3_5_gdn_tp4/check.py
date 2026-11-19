# ruff: noqa: ANN001, ANN201, ANN202
"""Check (and time) the Qwen3.5 TP4 GDN mixer megakernel against SGLang's verify path.

torchrun --nproc-per-node=4 -m pretuned_kernels.megakernels.qwen3_5_gdn_tp4.check \
    [--bench | --bench-only]

Per-stage references are SGLang's own kernels fed with the megakernel's previous
stage (so each stage's error is its own): torch's FP8 in_proj_qkvz (static input
quant, FP32 accumulate) and BF16 in_proj_ba first, then causal_conv1d_update
(dedup window view), gdn_replayssm_spec_decode(launch_mode="verify"), the gated RMSNorm
(rms_norm_gated, RMSNormGated's Triton path), static_quant_fp8 (apply_fp8_linear's
static input quant), then torch for the FP8 GEMM (FP32 accumulate, * input_scale *
weight_scale, BF16), the all-reduce (all_gather, FP32 sum in rank order, as the
served CuTe DSL AR fusion sums in FP32), + residual and GemmaRMSNorm (BF16
pre-norm, (1 + w) gamma). "chain" rows run the whole SGLang sequence from the
inputs. Determinism: a second run on the same inputs/state must be bitwise equal.

--bench: LAYERS (default 16) distinct layers (weights + state), each launched
once per step, REPS steps in one CUDA graph; us per layer for the megakernel
(includes the all-reduce + residual + norm) and for SGLang's stock sequence
(the FP8 in_proj_qkvz and BF16 in_proj_ba GEMMs, qkvzba split,
causal_conv1d_update, qkv split, ReplaySSM verify, gated RMSNorm, static FP8 quant +
FlashInfer bmm_fp8) WITHOUT the all-reduce / residual / norm. --bench-only skips
the check.
"""

from __future__ import annotations

from itertools import starmap
import os
import sys

from pretuned_kernels.megakernels.qwen3_5_gdn_tp4 import _setup as ms
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4.qwen3_5_gdn_mixer_tp4 import NORM_EPS
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4.qwen3_5_gdn_mixer_tp4 import (
    qwen3_5_gdn_mixer_tp4,
)
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

CASES = ((0, 0), (2, 5), (3, 14), (9, 11))  # (write_pos, cache_base)


def _clone(d: dict) -> dict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def window_view(phys: torch.Tensor) -> torch.Tensor:
    """MambaPool's deduplicated conv window view of one layer's physical
    [slots + 1, dim, T + 2] buffer: [slots + 1, T, dim, 3], step t = columns t..t+2."""
    s = phys.stride()
    return phys.as_strided(
        (phys.size(0), ms.T, phys.size(1), ms.WIDTH - 1), (s[0], s[2], s[1], s[2])
    )


def sglang_conv(qkvz, layer, state):
    """causal_conv1d_update as GDNAttnBackend.forward_extend calls it in target verify
    (mutates state's conv_state / conv_window). Returns mixed q|k|v [T, 3072]."""
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    mixed_in = qkvz[:, : ms.CONV_DIM].contiguous()
    x = mixed_in.view(1, ms.T, -1).transpose(1, 2)
    out = causal_conv1d_update(
        x,
        state["conv_state"],
        layer["conv_weight"],
        None,
        "silu",
        conv_state_indices=state["state_index"],
        intermediate_conv_window=window_view(state["conv_window"]),
        intermediate_state_indices=state["window_index"],
    )
    return out.transpose(1, 2).reshape(ms.T, ms.CONV_DIM)


def sglang_core(mixed, ba, layer, state):
    """gdn_replayssm_spec_decode(launch_mode="verify") as _replayssm_target_verify
    calls it (mutates the rings). Returns [T, 2048] bf16."""
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        gdn_replayssm_spec_decode,
    )

    q = mixed[:, : ms.KEY_DIM].contiguous().view(ms.T, ms.H, ms.K)
    k = mixed[:, ms.KEY_DIM : 2 * ms.KEY_DIM].contiguous().view(ms.T, ms.H, ms.K)
    v = mixed[:, 2 * ms.KEY_DIM :].contiguous().view(ms.T, ms.HV, ms.V)
    b = ba[:, : ms.HV].contiguous()
    a = ba[:, ms.HV :].contiguous()
    out = torch.empty(ms.T, ms.HV, ms.V, dtype=torch.bfloat16, device=mixed.device)
    gdn_replayssm_spec_decode(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=layer["A_log"],
        dt_bias=layer["dt_bias"],
        checkpoint_state=state["h0"],
        d_cache=state["d_cache"],
        k_cache=state["k_cache"],
        g_cache=state["g_cache"],
        rawv_cache=state["dlow_cache"],
        rawk_cache=state["klow_cache"],
        beta_cache=None,
        out=out,
        query_start_loc=torch.tensor([0, ms.T], dtype=torch.int32, device=mixed.device),
        ssm_state_indices=state["state_index"],
        replay_indices=state["replay_index"],
        write_pos=state["write_pos"],
        cache_base=state["cache_base"],
        is_flush=torch.zeros(ms.RSLOTS, dtype=torch.int8, device=mixed.device),
        max_cache_len=ms.L,
        max_spec_len=ms.T,
        scale=ms.K**-0.5,
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
        launch_mode="verify",
    )
    return out.view(ms.T, ms.VALUE_DIM)


def sglang_gated_norm(core, qkvz, layer):
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

    z = qkvz[:, ms.CONV_DIM :].contiguous().view(-1, ms.V)
    y = rms_norm_gated(
        x=core.reshape(-1, ms.V),
        weight=layer["gate_norm_weight"],
        bias=None,
        z=z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="swish",
    )
    return y.view(ms.T, ms.VALUE_DIM)


def sglang_quant(y, layer, key="input_scale"):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    q, _ = static_quant_fp8(y, layer[key].reshape(1), repeat_scale=False)
    return q


def torch_in_proj(hidden, layer):
    """in_proj_qkvz (static FP8 input quant, FP8 GEMM with FP32 accumulate, *
    input_scale * weight_scale, BF16) and the BF16 in_proj_ba."""
    q8 = sglang_quant(hidden, layer, "inproj_input_scale")
    w = layer["inproj_stored"].t().float()  # [5120, 4096]
    scale = layer["inproj_input_scale"] * layer["inproj_weight_scale"]
    qkvz = ((q8.float() @ w.T) * scale).to(torch.bfloat16)
    ba = (hidden.float() @ layer["ba_weight"].float().T).to(torch.bfloat16)
    return qkvz, ba


def torch_out_proj(q8, layer):
    w = layer["out_proj_stored"].t().float()  # [4096, 2048]
    acc = q8.float() @ w.T
    return (acc * (layer["input_scale"] * layer["weight_scale"])).to(torch.bfloat16)


def all_reduce_norm(partial, residual, layer, world):
    parts = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(parts, partial)
    total = torch.zeros_like(partial, dtype=torch.float32)
    for p in parts:
        total = total + p.float()
    pre = (total + residual.float()).to(torch.bfloat16)
    pf = pre.float()
    out = pf * torch.rsqrt(pf.square().mean(-1, keepdim=True) + NORM_EPS)
    out = (out * layer["norm_weight"].float()).to(torch.bfloat16)
    return out, pre


def _same(a, b):
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return torch.equal(a, b)


def _report(rows, label, got, exp, tol, bitwise=False) -> bool:
    """Append one stage's errors to ``rows``; False when it fails ``tol``."""
    mx, rel = _err(got, exp)
    exact = _same(got, exp)
    rows.append(
        f"  {label:<22} max {mx:.3g} rel-rms {rel:.3g}"
        + (" (bitwise)" if exact else "")
    )
    if (bitwise and not exact) or rel > tol:
        rows[-1] += "  <-- FAIL"
        return False
    return True


def _err(got, exp):
    got, exp = got.float(), exp.float()
    e = (got - exp).abs()
    den = exp.norm().item()
    return e.max().item(), (e.norm().item() / den) if den > 0 else e.norm().item()


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev)
    group = dist.group.WORLD
    symm_mem.set_backend("NVSHMEM")
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), 32 * 1024))
    kernel = qwen3_5_gdn_mixer_tp4
    kernel_args = ms.kernel_args
    partials = []
    for _ in range(2):
        p = symm_mem.empty((ms.T, ms.HIDDEN), dtype=torch.bfloat16, device=dev)
        symm_mem.rendezvous(p, group.group_name)
        partials.append(p)

    def run(layer, state, x, partial):
        out = kernel(*kernel_args(layer, state, x, partial, group.group_name))
        torch.cuda.synchronize()
        return out, partial.clone()

    layer = ms.make_layer(100 + rank, dev)
    ok = True
    if "--bench-only" not in sys.argv:
        for ci, (wp, base) in enumerate(CASES):
            x = ms.make_inputs(300 + 10 * ci + rank, dev)
            # The residual is replicated across TP ranks.
            x["residual"] = ms.make_inputs(999 + ci, dev)["residual"]
            state0 = ms.make_state(200 + 10 * ci + rank, wp, base, dev)
            sk = _clone(state0)
            outs, partial = run(layer, sk, x, partials[ci % 2])
            output, residual_out, core, gated, mixed, q8 = outs[:6]
            rows = []
            # in_proj (from the hidden input); later stages read the kernel's.
            qkvz, ba = outs[6], outs[7]
            qkvz_ref, ba_ref = torch_in_proj(x["hidden"], layer)
            ok &= _report(rows, "in_proj qkvz", qkvz, qkvz_ref, 1e-2)
            ok &= _report(rows, "in_proj ba", ba, ba_ref, 1e-2)

            # conv (from the in_proj output)
            sr = _clone(state0)
            mixed_ref = sglang_conv(qkvz, layer, sr)
            ok &= _report(rows, "conv mixed_qkv", mixed, mixed_ref, 1e-2)
            ok &= _report(
                rows, "conv_state", sk["conv_state"], sr["conv_state"], 0, bitwise=True
            )
            ok &= _report(
                rows,
                "conv_window",
                sk["conv_window"],
                sr["conv_window"],
                0,
                bitwise=True,
            )
            # core (on the kernel's conv output)
            sc = _clone(state0)
            core_ref = sglang_core(mixed, ba, layer, sc)
            ok &= _report(rows, "core out", core, core_ref, 1e-2)
            ok &= _report(rows, "  g_cache", sk["g_cache"], sc["g_cache"], 1e-2)
            # The rings keep each FP32 record as a BF16 pair (hi, lo = bf16(x - hi)).
            # When the FP32 value differs by an ulp (exact vs approximate exp / div)
            # and crosses a BF16 rounding boundary, one BF16 ulp moves between the
            # parts: hi is off by an ulp and lo by about minus that ulp, a large
            # relative error in lo alone while hi + lo, the value the replay uses,
            # stays within FP32 rounding. So the pair is judged on hi and on hi + lo;
            # lo alone is reported, not judged.
            for hi, lo in (("d_cache", "dlow_cache"), ("k_cache", "klow_cache")):
                ok &= _report(rows, f"  {hi}", sk[hi], sc[hi], 1e-2)
                _report(rows, f"  {lo} (info)", sk[lo], sc[lo], float("inf"))
                ok &= _report(
                    rows,
                    f"  {hi}+lo",
                    sk[hi].float() + sk[lo].float(),
                    sc[hi].float() + sc[lo].float(),
                    1e-2,
                )
            # gated norm (on the kernel's core output), FP8 input quant (on its norm)
            ok &= _report(
                rows,
                "gated norm",
                gated,
                sglang_gated_norm(core, qkvz, layer),
                1e-2,
            )
            q8_ref = sglang_quant(gated, layer)
            same = (
                (q8.view(torch.uint8) == q8_ref.view(torch.uint8)).float().mean().item()
            )
            rows.append(f"  {'fp8 input quant':<22} bytes equal {same * 100:.4f}%")
            if same < 0.999:
                ok = False
                rows[-1] += "  <-- FAIL"
            # out_proj (on the kernel's FP8 input), AR + residual + norm (on the
            # kernel's partials, gathered)
            ok &= _report(
                rows, "out_proj partial", partial, torch_out_proj(q8, layer), 1e-2
            )
            out_ref, res_ref = all_reduce_norm(partial, x["residual"], layer, world)
            ok &= _report(rows, "residual_out", residual_out, res_ref, 1e-2)
            ok &= _report(rows, "normed output", output, out_ref, 1e-2)
            # the whole SGLang chain from the inputs
            sch = _clone(state0)
            m_ch = sglang_conv(qkvz_ref, layer, sch)
            c_ch = sglang_core(m_ch, ba_ref, layer, sch)
            g_ch = sglang_gated_norm(c_ch, qkvz_ref, layer)
            p_ch = torch_out_proj(sglang_quant(g_ch, layer), layer)
            o_ch, r_ch = all_reduce_norm(p_ch, x["residual"], layer, world)
            ok &= _report(rows, "chain core out", core, c_ch, 2e-2)
            ok &= _report(rows, "chain partial", partial, p_ch, 2e-2)
            ok &= _report(rows, "chain residual_out", residual_out, r_ch, 2e-2)
            ok &= _report(rows, "chain normed output", output, o_ch, 2e-2)
            # determinism
            s2 = _clone(state0)
            out2, partial2 = run(layer, s2, x, partials[ci % 2])
            det = all(
                starmap(_same, zip((*outs, partial), (*out2, partial2), strict=True))
            ) and all(_same(sk[n], s2[n]) for n in sk)
            rows.append(
                f"  {'determinism':<22} {'bitwise' if det else 'DIFFERS  <-- FAIL'}"
            )
            ok &= det
            print(f"rank{rank} wp={wp} base={base}\n" + "\n".join(rows), flush=True)
        print(f"rank{rank} {'PASS' if ok else 'FAIL'}", flush=True)
    if "--bench" in sys.argv or "--bench-only" in sys.argv:
        bench(kernel, kernel_args, rank, dev, group, partials)
    dist.destroy_process_group()
    if not ok:
        sys.exit(1)


def bench(kernel, kernel_args, rank, dev, group, partials):
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        gdn_replayssm_spec_decode,
    )
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
    )
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_bmm_flashinfer

    n_layers = int(os.environ.get("LAYERS", "16"))
    reps = int(os.environ.get("REPS", "4"))
    layers = [ms.make_layer(1000 * i + rank, dev) for i in range(n_layers)]
    states = [ms.make_state(2000 * i + rank, 2, 5, dev) for i in range(n_layers)]
    x = ms.make_inputs(77 + rank, dev)
    x["residual"] = ms.make_inputs(4242, dev)["residual"]
    qsl = torch.tensor([0, ms.T], dtype=torch.int32, device=dev)
    flush = torch.zeros(ms.RSLOTS, dtype=torch.int8, device=dev)
    win_views = [window_view(s["conv_window"]) for s in states]

    def ours():
        for i in range(n_layers):
            kernel(
                *kernel_args(layers[i], states[i], x, partials[i % 2], group.group_name)
            )

    def stock():
        for i in range(n_layers):
            lw, st = layers[i], states[i]
            qkvz = apply_fp8_linear_bmm_flashinfer(
                x["hidden"],
                lw["inproj_stored"],
                lw["inproj_weight_scale"],
                lw["inproj_input_scale"],
                pre_quant_output_dtype=torch.bfloat16,
            )
            ba = torch.nn.functional.linear(x["hidden"], lw["ba_weight"])
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                qkvz, ba, ms.H, ms.HV, ms.K, ms.V
            )
            conv = causal_conv1d_update(
                mixed_qkv.view(1, ms.T, -1).transpose(1, 2),
                st["conv_state"],
                lw["conv_weight"],
                None,
                "silu",
                conv_state_indices=st["state_index"],
                intermediate_conv_window=win_views[i],
                intermediate_state_indices=st["window_index"],
            )
            mixed = conv.transpose(1, 2).reshape(ms.T, -1)
            q, k, v = fused_qkv_split_gdn_prefill(
                mixed, ms.H, ms.H, ms.HV, ms.K, ms.K, ms.V
            )
            core = torch.empty(ms.T, ms.HV, ms.V, dtype=torch.bfloat16, device=dev)
            gdn_replayssm_spec_decode(
                q=q.reshape(ms.T, ms.H, ms.K),
                k=k.reshape(ms.T, ms.H, ms.K),
                v=v.reshape(ms.T, ms.HV, ms.V),
                a=a,
                b=b,
                A_log=lw["A_log"],
                dt_bias=lw["dt_bias"],
                checkpoint_state=st["h0"],
                d_cache=st["d_cache"],
                k_cache=st["k_cache"],
                g_cache=st["g_cache"],
                rawv_cache=st["dlow_cache"],
                rawk_cache=st["klow_cache"],
                beta_cache=None,
                out=core,
                query_start_loc=qsl,
                ssm_state_indices=st["state_index"],
                replay_indices=st["replay_index"],
                write_pos=st["write_pos"],
                cache_base=st["cache_base"],
                is_flush=flush,
                max_cache_len=ms.L,
                max_spec_len=ms.T,
                scale=ms.K**-0.5,
                use_qk_l2norm_in_kernel=True,
                null_block_id=-1,
                launch_mode="verify",
            )
            y = rms_norm_gated(
                x=core.reshape(-1, ms.V),
                weight=lw["gate_norm_weight"],
                bias=None,
                z=z.reshape(-1, ms.V),
                eps=1e-6,
                group_size=None,
                norm_before_gate=True,
                is_rms_norm=True,
                activation="swish",
            ).reshape(ms.T, ms.VALUE_DIM)
            apply_fp8_linear_bmm_flashinfer(
                y,
                lw["out_proj_stored"],
                lw["weight_scale"],
                lw["input_scale"],
                pre_quant_output_dtype=torch.bfloat16,
            )

    def graph_time(fn, collective):
        # Two eager calls: hl.prefetch tensor maps count as written only once a later
        # eager launch sees their first copy done; until then a capture records the
        # copy again (three 128 B H2D memcpy nodes per launch, ~12 us of graph gaps).
        for _ in range(2):
            fn()
            torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                fn()
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
            if collective:
                dist.barrier()
            s.record()
            graph.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e3 / (reps * n_layers))
        times.sort()
        return times[len(times) // 2], times[0]

    t_ours = graph_time(ours, True)
    t_stock = graph_time(stock, False)
    print(
        f"rank{rank} T={ms.T} layers={n_layers} us/layer: megakernel med {t_ours[0]:.2f} "
        f"min {t_ours[1]:.2f} (incl. AR+residual+norm) | SGLang stock med {t_stock[0]:.2f} "
        f"min {t_stock[1]:.2f} (excl. AR+residual+norm)",
        flush=True,
    )


if __name__ == "__main__":
    main()
