# ruff: noqa: ANN001, ANN201, ANN202
"""Check (and time) the Qwen3.5 TP4 attention mixer megakernel against SGLang's
target-verify path (trtllm_mha backend, FP8 KV cache, SM100).

torchrun --nproc-per-node=4 -m pretuned_kernels.megakernels.qwen3_5_attn_tp4.check \
    [--bench | --bench-only]

Per-stage references are SGLang's own kernels fed with the megakernel's previous
stage (so each stage's error is its own): torch's FP8 qkv_proj (static_quant_fp8
input, FP32 accumulate, * input_scale * weight_scale, BF16), then
fused_qk_gemma_rmsnorm_rope_gate (Triton; q/k GemmaRMSNorm + partial NeoX mrope +
gate de-interleave), fused_fp8_qkv_kv_cache (FP8 q / KV-cache write) + FlashInfer's
TRT-LLM decode kernel (q_len_per_req = T), fused_sigmoid_mul + static_quant_fp8 (the
output gate and o_proj's input), torch for the FP8 o_proj, the all-reduce
(all_gather, FP32 sum in rank order) + residual and GemmaRMSNorm (BF16 pre-norm,
(1 + w) gamma). The KV cache must match bitwise. The attention output is judged
against the exact FP32 attention of the same FP8 inputs (no worse than 1.5x the
TRT-LLM kernel's own error: both round P to E4M3 their own way) and against the
TRT-LLM kernel. "chain" rows run the whole SGLang sequence from the inputs.
Determinism: a second run on the same inputs/state must be bitwise equal.

--bench: LAYERS (default 16) distinct layers (weights + KV pools), each launched
once per step, REPS steps in one CUDA graph, at each BENCH_PREFIXES committed
length (default 2048,8192); us per layer for the megakernel (includes the
all-reduce + residual + norm) and for SGLang's stock sequence (FlashInfer bmm_fp8
qkv_proj, fused_qk_gemma_rmsnorm_rope_gate, fused_fp8_qkv_kv_cache, TRT-LLM decode,
fused_sigmoid_mul, FlashInfer bmm_fp8 o_proj) WITHOUT the all-reduce / residual /
norm. STOCK=0 skips the stock arm; --bench-only skips the check. CASES (default
77,2048,8192,8100): the checked committed lengths.
"""

from __future__ import annotations

from itertools import starmap
import os
import sys

from pretuned_kernels.megakernels.qwen3_5_attn_tp4 import _setup as s
from pretuned_kernels.megakernels.qwen3_5_attn_tp4.qwen3_5_attn_mixer_tp4 import (
    NORM_EPS,
)
from pretuned_kernels.megakernels.qwen3_5_attn_tp4.qwen3_5_attn_mixer_tp4 import (
    qwen3_5_attn_mixer_tp4,
)
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Committed prefix lengths (CASES=a,b,... overrides).
CASES = tuple(int(p) for p in os.environ.get("CASES", "77,2048,8192,8100").split(","))


_WORKSPACE = {}


def workspace(dev):
    if dev not in _WORKSPACE:
        _WORKSPACE[dev] = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    return _WORKSPACE[dev]


def paged(cache):
    """trtllm_mha's _reshape_paged_kv_cache view (HND; 1 KV head)."""
    return cache.view(-1, s.PAGE, s.KV_HEADS, s.HEAD_DIM).permute(0, 2, 1, 3)


def exact_attention(q, state):
    """FP32 attention over the FP8 values the cache holds (after the window's write):
    q as FP8 (scale 1), K * k_scale, V * v_scale, exact softmax, causal window."""
    total = int(state["seq_lens"][0])
    pos = torch.arange(total, device=q.device)
    slots = state["page_table"][0, pos // s.PAGE].long() * s.PAGE + pos % s.PAGE
    k = state["k_cache"].view(-1, s.HEAD_DIM)[slots].float() * s.K_SCALE
    v = state["v_cache"].view(-1, s.HEAD_DIM)[slots].float() * s.V_SCALE
    q8 = q.to(torch.float8_e4m3fn).float().view(s.T, s.HEADS, s.HEAD_DIM)
    scores = torch.einsum("thd,kd->htk", q8, k) * s.HEAD_DIM**-0.5
    mask = pos[None, :] > (total - s.T + torch.arange(s.T, device=q.device))[:, None]
    scores = scores.masked_fill(mask[None], float("-inf"))
    out = torch.einsum("htk,kd->thd", scores.softmax(-1), v)
    return out.reshape(s.T, s.Q_SIZE)


def _clone_state(d: dict) -> dict:
    out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    out["k_cache"], out["v_cache"] = s.kv_buffers(d["k_cache"].device)
    out["k_cache"].copy_(d["k_cache"])
    out["v_cache"].copy_(d["v_cache"])
    return out


def sglang_quant(y, scale):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    q, _ = static_quant_fp8(y.contiguous(), scale.reshape(1), repeat_scale=False)
    return q


def torch_qkv(hidden, layer):
    """qkv_proj: static FP8 input quant, FP8 GEMM with FP32 accumulate, *
    input_scale * weight_scale, BF16 [T, 4608]."""
    q8 = sglang_quant(hidden, layer["qkv_input_scale"])
    w = layer["qkv_stored"].t().float()
    scale = layer["qkv_input_scale"] * layer["qkv_weight_scale"]
    return ((q8.float() @ w.T) * scale).to(torch.bfloat16)


def sglang_qk_prep(qkv, layer, rope, state):
    """forward_prepare_cuda_fused: q, k (norm + rope) and the de-interleaved gate."""
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    cos_sin, axis_map = rope
    q_gate, k, v = qkv.split([s.QG_SIZE, s.KV_SIZE, s.KV_SIZE], dim=-1)
    q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        layer["q_norm_weight"],
        layer["k_norm_weight"],
        cos_sin,
        state["positions"],
        s.EPS,
        s.HEADS,
        s.KV_HEADS,
        s.HEAD_DIM,
        s.ROTARY_DIM,
        has_gate=True,
        mrope_axis_map=axis_map,
    )
    return q.view(s.T, -1), k.view(s.T, -1), v, gate


def sglang_attention(q, k, v, layer, state):
    """fused_fp8_qkv_kv_cache + trtllm_batch_decode_with_kv_cache as trtllm_mha's
    forward_extend runs them in target verify (mutates state's caches)."""
    import flashinfer
    from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import fused_fp8_qkv_kv_cache

    q8 = fused_fp8_qkv_kv_cache(
        q=q,
        k=k,
        v=v,
        k_cache=state["k_cache"],
        v_cache=state["v_cache"],
        cache_loc=state["cache_loc"],
        k_scale=layer["k_scale"],
        v_scale=layer["v_scale"],
    )
    out = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q8.view(-1, s.HEADS, s.HEAD_DIM),
        kv_cache=(paged(state["k_cache"]), paged(state["v_cache"])),
        workspace_buffer=workspace(q.device),
        block_tables=state["page_table"],
        seq_lens=state["seq_lens"],
        max_seq_len=s.CONTEXT_LEN,
        bmm1_scale=s.K_SCALE * s.HEAD_DIM**-0.5,
        bmm2_scale=s.V_SCALE,
        window_left=-1,
        out_dtype=torch.bfloat16,
        q_len_per_req=s.T,
    )
    return out.view(s.T, s.Q_SIZE)


def sglang_gate(attn, gate):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    return fused_sigmoid_mul(attn.clone(), gate, inplace=True)


def torch_o_proj(q8, layer):
    w = layer["o_proj_stored"].t().float()  # [4096, 2048]
    acc = q8.float() @ w.T
    return (acc * (layer["o_input_scale"] * layer["o_weight_scale"])).to(torch.bfloat16)


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


def _err(got, exp):
    got, exp = got.float(), exp.float()
    e = (got - exp).abs()
    den = exp.norm().item()
    return e.max().item(), (e.norm().item() / den) if den > 0 else e.norm().item()


def _report(rows, label, got, exp, tol, bitwise=False) -> bool:
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


def _bytes_equal(rows, label, got, exp, frac=0.999) -> bool:
    same = (got.view(torch.uint8) == exp.view(torch.uint8)).float().mean().item()
    rows.append(f"  {label:<22} bytes equal {same * 100:.4f}%")
    if same < frac:
        rows[-1] += "  <-- FAIL"
        return False
    return True


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
    kernel = qwen3_5_attn_mixer_tp4
    rope = s.rope_tables(dev)
    partials = []
    for _ in range(2):
        p = symm_mem.empty((s.T, s.HIDDEN), dtype=torch.bfloat16, device=dev)
        symm_mem.rendezvous(p, group.group_name)
        partials.append(p)

    def run(layer, state, x, partial):
        out = kernel(*s.kernel_args(layer, state, x, rope, partial, group.group_name))
        torch.cuda.synchronize()
        return out, partial.clone()

    layer = s.make_layer(100 + rank, dev)
    ok = True
    if "--bench-only" not in sys.argv:
        for ci, prefix in enumerate(CASES):
            x = s.make_inputs(300 + 10 * ci + rank, dev)
            # The residual is replicated across TP ranks; so is the request's state
            # layout (each rank pair holds one KV head: values per rank here).
            x["residual"] = s.make_inputs(999 + ci, dev)["residual"]
            state0 = s.make_state(200 + 10 * ci, prefix, dev)
            sk = _clone_state(state0)
            outs, partial = run(layer, sk, x, partials[ci % 2])
            output, residual_out, q_x, g_x, kv_x, q_rope, k_rope, attn, attn8 = outs
            # qkv_proj's output as SGLang lays it out: [q_h | gate_h] x 8, k, v.
            qkv = torch.cat(
                (
                    torch.stack((q_x, g_x), dim=2).reshape(s.T, -1),
                    kv_x.reshape(s.T, -1),
                ),
                dim=1,
            )
            q_rope = q_rope.reshape(s.T, -1)
            attn = attn.reshape(s.T, -1)
            rows = []
            # qkv_proj (from the hidden input)
            ok &= _report(rows, "qkv_proj", qkv, torch_qkv(x["hidden"], layer), 1e-2)
            # q / k norm + rope, gate (on the kernel's qkv)
            q_ref, k_ref, v_in, gate = sglang_qk_prep(qkv, layer, rope, state0)
            ok &= _report(rows, "q norm+rope", q_rope, q_ref, 1e-2)
            ok &= _report(rows, "k norm+rope", k_rope, k_ref, 1e-2)
            # attention + KV-cache write (on the kernel's q / k / v)
            sr = _clone_state(state0)
            attn_ref = sglang_attention(q_rope, k_rope, v_in, layer, sr)
            exact = exact_attention(q_rope, sr)
            _, rel_x = _err(attn, exact)
            _, rel_rx = _err(attn_ref, exact)
            ok &= _report(rows, "attention vs TRT-LLM", attn, attn_ref, 5e-2)
            good = rel_x < max(1.5 * rel_rx, 1e-2)
            rows.append(
                f"  {'attention vs exact':<22} rel-rms {rel_x:.3g} (TRT-LLM {rel_rx:.3g})"
                + ("" if good else "  <-- FAIL")
            )
            ok &= good
            for name in ("k_cache", "v_cache"):
                ok &= _report(
                    rows,
                    name,
                    sk[name].view(torch.uint8),
                    sr[name].view(torch.uint8),
                    0,
                    bitwise=True,
                )
            # output gate + o_proj's FP8 input (on the kernel's attention output)
            ok &= _bytes_equal(
                rows,
                "gate + fp8 input",
                attn8,
                sglang_quant(sglang_gate(attn, gate), layer["o_input_scale"]),
            )
            # o_proj (on the kernel's FP8 input), AR + residual + norm (on its partials)
            ok &= _report(
                rows, "o_proj partial", partial, torch_o_proj(attn8, layer), 1e-2
            )
            out_ref, res_ref = all_reduce_norm(partial, x["residual"], layer, world)
            ok &= _report(rows, "residual_out", residual_out, res_ref, 1e-2)
            ok &= _report(rows, "normed output", output, out_ref, 1e-2)
            # the whole SGLang chain from the inputs
            sch = _clone_state(state0)
            qkv_ch = torch_qkv(x["hidden"], layer)
            q_ch, k_ch, v_ch, g_ch = sglang_qk_prep(qkv_ch, layer, rope, state0)
            a_ch = sglang_attention(q_ch, k_ch, v_ch, layer, sch)
            p_ch = torch_o_proj(
                sglang_quant(sglang_gate(a_ch, g_ch), layer["o_input_scale"]), layer
            )
            o_ch, r_ch = all_reduce_norm(p_ch, x["residual"], layer, world)
            ok &= _report(rows, "chain attention", attn, a_ch, 5e-2)
            ok &= _report(rows, "chain partial", partial, p_ch, 5e-2)
            ok &= _report(rows, "chain residual_out", residual_out, r_ch, 2e-2)
            ok &= _report(rows, "chain normed output", output, o_ch, 2e-2)
            # determinism
            s2 = _clone_state(state0)
            out2, partial2 = run(layer, s2, x, partials[ci % 2])
            det = all(
                starmap(_same, zip((*outs, partial), (*out2, partial2), strict=True))
            ) and all(_same(sk[n], s2[n]) for n in ("k_cache", "v_cache"))
            rows.append(
                f"  {'determinism':<22} {'bitwise' if det else 'DIFFERS  <-- FAIL'}"
            )
            ok &= det
            print(f"rank{rank} prefix={prefix}\n" + "\n".join(rows), flush=True)
        print(f"rank{rank} {'PASS' if ok else 'FAIL'}", flush=True)
    if "--bench" in sys.argv or "--bench-only" in sys.argv:
        bench(kernel, rope, rank, dev, group, partials)
    dist.destroy_process_group()
    if not ok:
        sys.exit(1)


def bench(kernel, rope, rank, dev, group, partials):
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_bmm_flashinfer

    n_layers = int(os.environ.get("LAYERS", "16"))
    reps = int(os.environ.get("REPS", "4"))
    layers = [s.make_layer(1000 * i + rank, dev) for i in range(n_layers)]
    x = s.make_inputs(77 + rank, dev)
    x["residual"] = s.make_inputs(4242, dev)["residual"]
    for prefix in [
        int(p) for p in os.environ.get("BENCH_PREFIXES", "2048,8192").split(",")
    ]:
        states = [s.make_state(2000 + i, prefix, dev) for i in range(n_layers)]

        def ours(states=states):
            for i in range(n_layers):
                kernel(
                    *s.kernel_args(
                        layers[i], states[i], x, rope, partials[i % 2], group.group_name
                    )
                )

        def stock(states=states):
            for i in range(n_layers):
                lw, st = layers[i], states[i]
                qkv = apply_fp8_linear_bmm_flashinfer(
                    x["hidden"],
                    lw["qkv_stored"],
                    lw["qkv_weight_scale"],
                    lw["qkv_input_scale"],
                    pre_quant_output_dtype=torch.bfloat16,
                )
                q, k, v, gate = sglang_qk_prep(qkv, lw, rope, st)
                attn = sglang_attention(q, k, v, lw, st)
                from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

                gated = fused_sigmoid_mul(attn, gate, inplace=True)
                apply_fp8_linear_bmm_flashinfer(
                    gated,
                    lw["o_proj_stored"],
                    lw["o_weight_scale"],
                    lw["o_input_scale"],
                    pre_quant_output_dtype=torch.bfloat16,
                )

        t_ours = graph_time(ours, n_layers, reps, True)
        # STOCK=0 skips the stock arm (FlashInfer's cuBLASLt bmm_fp8 is unsupported
        # on some local builds).
        t_stock = (
            graph_time(stock, n_layers, reps, False)
            if os.environ.get("STOCK", "1") == "1"
            else (float("nan"), float("nan"))
        )
        print(
            f"rank{rank} T={s.T} prefix={prefix} layers={n_layers} us/layer: megakernel med "
            f"{t_ours[0]:.2f} min {t_ours[1]:.2f} (incl. AR+residual+norm) | SGLang stock med "
            f"{t_stock[0]:.2f} min {t_stock[1]:.2f} (excl. AR+residual+norm)",
            flush=True,
        )


def graph_time(fn, n_layers, reps, collective):
    # Two eager calls: hl.prefetch tensor maps count as written only once a later
    # eager launch sees their first copy done (else captured H2D memcpy nodes).
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
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        if collective:
            dist.barrier()
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / (reps * n_layers))
    times.sort()
    return times[len(times) // 2], times[0]


if __name__ == "__main__":
    main()
