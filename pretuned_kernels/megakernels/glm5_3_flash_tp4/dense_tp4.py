# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""GLM-5.3 Flash NVFP4 dense MLP sublayer (layers 0-2) for a TP4 speculative-verify
window (T <= 8 tokens) as one Gluon megakernel, with the TP all-reduce and the FFN ->
next attention mHC boundary folded into its tail (``_mhc``, the same tail as
``moe_tp4``): from the FFN's input (its hc_pre output) to the next layer's attention
input, R', post' and comb'.

The model (``_setup.nvfp4_mlp`` has the PyTorch reference):
gate|up NVFP4 [2 x 3072, 4096] per rank -> alpha -> SwiGLU with SGLang's clamps (gate
<= 10, |up| <= 10) -> NVFP4 (down's input global scale) -> down NVFP4 [4096, 3072].
gate_up is rebuilt once in the trtllm-gen layout of the MoE shared expert (each 128-row
tile holds 64 intermediates' up and gate rows); down is read in place as SGLang serves
it (the ModelOpt linear's weight and 128x4-interleaved ``weight_scale_interleaved``:
one [16, 256] block per 128-row tile and 512-wide K step, ``scale_tile``).

Roots, in launch order (one persistent CTA per SM, TMA ring, dynamic dispatch; each
root's tasks on workers of their own, so every weight stream starts at launch):
  1. input quant: one task per (8 tokens, K step of 512): the NVFP4 copy of the
     input that the gate|up GEMM reads, in its K-step layout;
  2. gate|up per 128-row tile (48 tasks) over the full K, with the clamped SwiGLU and
     the activation's NVFP4 in its epilogue (``_common.clamped_swiglu_nvfp4``, the
     shared expert's; FP32 until the NVFP4 rounding, where the stock path rounds
     gate|up and the SwiGLU to BF16);
  3. down per (128-row tile, K split): BF16 partials (stock's AR dtype) straight into
     this rank's symmetric buffer [splits, T, 4096] (no combine root);
  4-6. the mHC boundary (``_mhc``): the ranks' split partials summed split-major in
     rank order, hc_post, the next layer's hc_attn_pre and input RMSNorm -> li, R',
     post', comb'.

Every split sum is in a fixed order and the all-reduce in rank order (no atomics):
deterministic run to run.

``correctness_check()`` compares every rank with the torch reference (``_setup``) on
synthetic weights; ``main()`` runs it, then times the kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/dense_tp4.py [--check]
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_buffer
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import scale_tile
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _mhc
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_flash_tp4._common import clamped_swiglu_nvfp4
from pretuned_kernels.megakernels.glm5_3_flash_tp4._common import nvfp4_pairs
from pretuned_kernels.megakernels.glm5_3_tp4._setup import trtllm_gen_w13
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._setup import _nvfp4_quant
import torch
import torch.distributed as dist

import helion
import helion.language as hl

# K splits of the down GEMM (6 K steps of 512; the symmetric buffer's shape
# carries it into the kernel)
DOWN_SPLITS = 2


@helion.kernel(
    static_shapes=True,
    backend="gluon",
    triton_do_not_specialize=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glmf_dense_tp4(
    x,  # [T, 8, 32, 16] bf16: the FFN input (its hc_pre output) by K step and FP4 group
    w13,  # [2I, 8, 256] uint8: gate_up in the trtllm-gen layout (``prepared_weights``), K steps of 512
    w13_scale,  # [2I / 128 * 8, 16, 256] e4m3: per (128-row tile, K step), 128x4-interleaved
    w2,  # [4096, I / 512, 256] uint8: down (served, rows in hidden order), K steps of 512
    w2_scale,  # [32, I / 512, 16, 256] e4m3: per (128-row tile, K step)
    scales,  # [4] fp32: gate_up input global, gate_up alpha, down input global, down alpha
    # scratch
    input_q,  # [T, 8, 256] uint8: the NVFP4 input by K step
    input_scale,  # [T, 8, 32] e4m3
    act_q,  # [T, I / 2] uint8: the NVFP4 activation
    act_scale,  # [T, I / 16] e4m3
    symmetric_partial,  # [DOWN_SPLITS, T, 4096] bf16 symmetric memory: this rank's down partials
    # the mHC boundary after the FFN (``_mhc``, as ``moe_tp4``)
    r,  # [T, 4, 4096] bf16
    post,  # [T, 4] fp32
    comb,  # [T, 4, 4] fp32
    fn_pad,  # [32, 4, 4096] bf16
    hc_scale,  # [3] fp32
    base_pre,  # [4] fp32
    base_post,  # [4] fp32
    base_comb,  # [4, 4] fp32
    norm_w,  # [4096] bf16: the next layer's input_layernorm
    mix_part,  # [S, 8, 32] fp32 scratch
    mix_t,  # [8, 32, S] fp32 scratch
    gram_part,  # [8, 4, 4, S] fp32 scratch, token-major
    sq_part,  # [8, S] fp32 scratch, token-major
    li,  # [T, 4096] bf16 out: the next attention's input
    r_new,  # [T, 4, 4096] bf16 out
    post_out,  # [T, 4] fp32 out
    comb_out,  # [T, 4, 4] fp32 out
    group_name: hl.ProcessGroupName,
    comb_in_raw: hl.constexpr,  # comb holds logits (a raw-out boundary's): Sinkhorn them here
    comb_out_raw: hl.constexpr,  # comb_out gets logits (the next megakernel takes them raw)
) -> None:
    tokens, gu_steps = x.shape[0], x.shape[1]
    w13_tiles = w13.size(0) // 128
    hidden = w2.size(0)
    down_steps = w2.size(1)
    down_splits = symmetric_partial.size(0)
    # (host code: a size read inside a device loop stays symbolic)
    down_split_steps = down_steps // down_splits
    assert tokens <= 8 and x.shape[2:] == (32, 16) and w13.shape[1:] == (gu_steps, 256)
    assert down_steps % down_splits == 0 and w13_tiles * 64 == down_steps * 512
    assert symmetric_partial.shape[1:] == (tokens, hidden)
    act_q_groups = act_q.view(tokens, w13_tiles, 4, 8)
    act_scale_groups = act_scale.view(tokens, w13_tiles, 4)
    act_q_steps = act_q.view(tokens, down_steps, 256)
    act_scale_steps = act_scale.view(tokens, down_steps, 32)
    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )

    hl.pdl_wait()
    # The code toward L2 first: the previous launch's weight stream evicted it.
    hl.prefetch_code()

    # 1. the input's NVFP4 copy per (8 tokens, K step): 32 groups of 16
    for q_s, q_t in hl.tile([gu_steps, tokens], block_size=[1, 8]):
        q_v = x[q_t, q_s.begin, :, :].to(torch.float32)  # [8, 32, 16]
        q_rows = q_v.size(0)
        q_low, q_high = hl.split(q_v.reshape(q_rows, 32, 8, 2))
        q_packed, q_bs = nvfp4_pairs(
            q_low, q_high, torch.amax(torch.abs(q_v), dim=-1), hl.load(scales, [0])
        )
        input_q[q_t, q_s.begin, :] = q_packed.reshape(q_rows, 256)
        input_scale[q_t, q_s.begin, :] = q_bs

    # 2. gate|up per 128-row tile of the trtllm-gen layout (64 intermediates' up
    # and gate rows) over the full K, then SGLang's clamped SwiGLU (after the
    # alpha) and the activation's NVFP4 (``_common.clamped_swiglu_nvfp4``, the
    # MoE shared expert's epilogue); FP32 until the NVFP4 rounding (the stock
    # path rounds gate|up and the SwiGLU to BF16)
    for g_tile, g_t in hl.tile([w13_tiles, tokens], block_size=[1, 8]):
        g_acc = hl.zeros([128, g_t], dtype=torch.float32)
        for g_s in hl.tile(gu_steps, block_size=1):
            g_acc = hl.dot_scaled(
                w13[g_tile.begin * 128 + hl.arange(128), g_s.begin, :],
                scale_tile(w13_scale[g_tile.begin * gu_steps + g_s.begin, :, :]),
                "e2m1",
                input_q[g_t, g_s.begin, :].T,
                input_scale[g_t, g_s.begin, :],
                "e2m1",
                acc=g_acc,
                out_dtype=torch.float32,
            )
        g_packed, g_bs = clamped_swiglu_nvfp4(
            g_acc, hl.load(scales, [1]), hl.load(scales, [2])
        )
        act_q_groups[g_t, g_tile.begin, :, :] = g_packed
        act_scale_groups[g_t, g_tile.begin, :] = g_bs

    # 3. down per (128-row tile, K split): BF16 partials into this rank's
    # symmetric buffer (the boundary's all-reduce sums every rank's splits: no
    # combine root)
    for w_m, w_t, w_s in hl.tile([hidden, tokens, down_splits], block_size=[128, 8, 1]):
        w_alpha = hl.load(scales, [3])
        w_acc = hl.zeros([w_m, w_t], dtype=torch.float32)
        for w_k in hl.tile(down_split_steps, block_size=1):
            w_step = w_s.begin * down_split_steps + w_k.begin
            w_acc = hl.dot_scaled(
                w2[w_m, w_step, :],
                scale_tile(w2_scale[w_m.begin // 128, w_step, :, :]),
                "e2m1",
                act_q_steps[w_t, w_step, :].T,
                act_scale_steps[w_t, w_step, :],
                "e2m1",
                acc=w_acc,
                out_dtype=torch.float32,
            )
        symmetric_partial[w_s.begin, w_t, w_m] = (w_acc * w_alpha).T.to(torch.bfloat16)

    hl.pdl_launch_dependents()
    # 4-6. the mHC boundary (``_mhc``): the all-reduce of the ranks' split
    # partials, hc_post, the next layer's hc_attn_pre -> its input, R', post',
    # comb'
    for tt, tc in hl.tile([tokens, hidden], block_size=[8, 128]):
        # a raw comb made final per chunk task (redundant across chunks, a few
        # hundred FLOPs) before its all-reduce wait: no root nor barrier of its
        # own
        cc = comb[tt, :, :]
        if comb_in_raw:
            cc = _mhc.sinkhorn_init(cc)
            for _cs_it in hl.tile(19, block_size=1):  # ITERS - 1
                cc = _mhc.sinkhorn_step(cc)
        _mhc.chunk(
            remote_partials,
            tt,
            tc,
            r,
            post,
            cc,
            fn_pad,
            r_new,
            mix_part,
            gram_part,
            sq_part,
            ksplit=down_splits,
            mix_t=mix_t,
            comb_tile=True,
        )
    for tb, tn in hl.tile([tokens, hidden], block_size=[8, 512]):
        _mhc.finalize(
            tb,
            tn,
            hidden,
            r_new,
            norm_w,
            mix_part,
            gram_part,
            sq_part,
            hc_scale,
            base_pre,
            li,
            mix_t=mix_t,
        )
    for ts in hl.tile(tokens, block_size=2):
        c = _mhc.comb_logits(
            ts,
            hidden,
            mix_part,
            sq_part,
            hc_scale,
            base_post,
            base_comb,
            post_out,
        )
        if not comb_out_raw:
            c = _mhc.sinkhorn_init(c)
            # ITERS - 1 (a literal: a module int reaches the kernel as a runtime scalar)
            for _it in hl.tile(19, block_size=1):
                c = _mhc.sinkhorn_step(c)
        comb_out[ts, :, :] = c


# The dot operands (and the weights' block scales) streamed through the TMA
# ring; every other access is a pointer. Both GEMMs' stages hold the same
# operand set (weight, its scales, activation, its scales), so they share one
# ring channel (two channels would not fit the shared memory); pointer-loaded
# weight scales cost ~0.5 us a K step of the post-wait tail.
RING_LOADS = frozenset(
    [
        "w13",
        "w13_scale",
        "input_q",
        "input_scale",
        "w2",
        "w2_scale",
        "act_q_steps",
        "act_scale_steps",
    ]
)


def mk_config(bound) -> helion.Config:
    """The dot operands of ``RING_LOADS`` through the TMA ring, every other
    access by pointer; one persistent CTA per SM claiming tasks root by root;
    the ring roots and their loops warp-specialized (``ring_config_fields``'
    ``"root_loops"``); the all-reduce in band (Lamport). Every positional list
    is derived from the bound kernel."""
    fields = ring_config_fields(bound, RING_LOADS, warp_specialize="root_loops")
    return helion.Config(
        indexing=fields["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=4,
        num_warps=4,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=fields["range_warp_specializes"],
        cross_loop_inband_protocol=["lamport"],
    )


@functools.cache
def scratch(device: torch.device, tokens: int) -> dict[str, torch.Tensor]:
    """The kernel's scratch buffers on ``device``, one set for every call."""
    u8, e4m3 = torch.uint8, torch.float8_e4m3fn
    steps, inter = _setup.HIDDEN // 512, _setup.DENSE_INTERMEDIATE
    return {
        "input_q": torch.zeros(tokens, steps, 256, device=device, dtype=u8),
        "input_scale": torch.zeros(tokens, steps, 32, device=device, dtype=e4m3),
        "act_q": torch.zeros(tokens, inter // 2, device=device, dtype=u8),
        "act_scale": torch.zeros(tokens, inter // 16, device=device, dtype=e4m3),
    }


SCRATCH_ORDER = ("input_q", "input_scale", "act_q", "act_scale")


def symmetric_partial_spec(tokens: int) -> tuple[tuple[int, ...], torch.dtype]:
    """The (shape, dtype) of the symmetric buffer (rendezvoused on the TP group) the
    kernel writes this rank's down split partials into and reads the peers' from."""
    return (DOWN_SPLITS, tokens, _setup.HIDDEN), torch.bfloat16


def prepared_weights(w: dict) -> dict:
    """The kernel's operands of one dense layer's per-rank weights (SGLang's
    processed ModelOpt linears; ``_setup.dense_weights`` names them): gate_up
    rebuilt once in the trtllm-gen layout of the MoE shared expert
    (``glm5_3_tp4._setup.trtllm_gen_w13``: each 128-row tile holds 64
    intermediates' up and gate rows, so the SwiGLU is a tile's epilogue; 13.4 MB
    a layer and rank) from its logical rows and block scales, down read in place
    (views of the weight and its 128x4-interleaved scales), the four scalars in
    one vector."""
    w13, w2 = w["w13"], w["w2"]
    two_inter, packed_hidden = w13.shape
    hidden, packed_inter = w2.shape
    steps, down_steps = packed_hidden // 256, packed_inter // 256
    t13, t13s = trtllm_gen_w13(w13[None], w["w13s_log"][None])
    scales = torch.stack(
        [
            w["in_inv1"].reshape(()),
            w["alpha1"].reshape(()),
            w["in_inv2"].reshape(()),
            w["alpha2"].reshape(()),
        ]
    )
    return {
        "w13": t13[0].view(two_inter, steps, 256),
        "w13s": t13s[0].view(two_inter // 128 * steps, 16, 256),
        "w2": w2.view(hidden, down_steps, 256),
        "w2s": w["w2s_sw"].view(hidden // 128, down_steps, 16, 256),
        "scales": scales.float().contiguous(),
    }


def kernel_args(
    x,
    pw,
    symmetric_partial,
    mhc,
    out,
    group_name,
    comb_in_raw=False,
    comb_out_raw=False,
) -> tuple:
    """``glmf_dense_tp4``'s arguments, as ``moe_tp4.kernel_args``: the input ``x``
    [T, 4096] (contiguous BF16, T <= 8), the prepared weights ``pw``
    (``prepared_weights``), the module's scratch, ``symmetric_partial``
    (``symmetric_partial_spec``), the mHC boundary's ``mhc``
    (``_mhc.boundary_operands``) and ``out`` = (li, r_new, post_out,
    comb_out)."""
    tokens, hidden = x.shape
    sc = scratch(x.device, tokens)
    return (
        x.view(tokens, hidden // 512, 32, 16),
        pw["w13"],
        pw["w13s"],
        pw["w2"],
        pw["w2s"],
        pw["scales"],
        *(sc[n] for n in SCRATCH_ORDER),
        symmetric_partial,
        *_mhc.tail_args(mhc, x.device, hidden),
        *out,
        group_name,
        bool(comb_in_raw),
        bool(comb_out_raw),
    )


def compile_kernel(args: tuple):
    """``glmf_dense_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_dense_tp4, mk_config, args)


# ---------------------------- check and timing (torch.distributed.run, 4 ranks)


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank its own weights (``_setup.dense_weights``) and input, plain, with
    30x the gate_up alpha (it drives the SwiGLU clamps) and with raw combs in and
    out: the NVFP4 input and activation against their exact quantizations, this
    rank's partial (its down splits' sum) against ``_setup.nvfp4_mlp`` (each GEMM
    on the kernel's NVFP4 copies); li, R', post', comb' against the boundary of
    the ranks' reference partials (``_setup.report_boundary``). Then every case
    again back to back (no host synchronization) and once more with NaN scratch
    and outputs: the same bits."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    T = _setup.T
    t = _setup.boundary_inputs(seed=11, device=device)
    partial = symmetric_buffer(*symmetric_partial_spec(T), group.group_name, device)
    weights = _setup.dense_weights(seed=100 + rank, device=device)
    launches, ok = [], True
    for i, (name, factor, raw) in enumerate(
        (
            ("plain", 1.0, False),
            ("gate_up alpha x30", 30.0, False),
            ("raw combs", 1.0, True),
        )
    ):
        if raw:  # the comb as logits
            t = dict(t, comb=_setup.comb_logits(seed=5, device=device))
        w = dict(weights, alpha1=weights["alpha1"] * factor)
        g = torch.Generator(device=device).manual_seed(10 * i + rank)
        x = torch.randn(T, _setup.HIDDEN, generator=g, device=device).to(torch.bfloat16)
        out = _mhc.new_outputs(t["r"])
        args = kernel_args(
            x,
            prepared_weights(w),
            partial,
            _mhc.boundary_operands(t),
            out,
            group.group_name,
            raw,
            raw,
        )
        kernel = compile_kernel(args)
        kernel(*args)
        torch.cuda.synchronize()
        sc = scratch(device, T)
        x_q = _setup.nvfp4_dequant(
            sc["input_q"].view(T, -1), sc["input_scale"].view(T, -1)
        )
        reference, act = _setup.nvfp4_mlp(x_q, w, (sc["act_q"], sc["act_scale"]))
        rows = []
        exact = _nvfp4_quant(x.float(), w["in_inv1"].item(), x_q)
        ok &= report(rows, "input (NVFP4)", x_q, exact, 2e-2)
        ok &= report(rows, "activation (NVFP4)", *act, 2e-2)
        ok &= report(rows, "partial", partial.float().sum(0), reference, 1e-2)
        ok &= _setup.report_boundary(rows, out, reference, t, group, raw)
        launches.append((functools.partial(kernel, *args), out))
        if verbose or not ok:
            say(f"rank{rank} {name}\n" + "\n".join(rows))
    poisoned = [
        *launches[-1][1],
        *scratch(device, T).values(),
        *_mhc.scratch(device, _setup.HIDDEN).values(),
    ]
    back_to_back, nan = _setup.repeat_bitwise(launches, poisoned)
    ok &= back_to_back and nan
    if verbose or not ok:
        say(
            f"rank{rank} repeats: back to back {'bitwise' if back_to_back else 'DIFFER  <-- FAIL'}, "
            f"NaN scratch {'bitwise' if nan else 'DIFFERS  <-- FAIL'}"
        )
    verdict(ok, group, "glmf_dense_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing: ``LAYERS`` distinct weight copies (default 8) back to
    back, ``REPS`` times (default 4) in one CUDA graph."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    T = _setup.T
    t = _setup.boundary_inputs(device=device)
    mhc, out = _mhc.boundary_operands(t), _mhc.new_outputs(t["r"])
    partial = symmetric_buffer(*symmetric_partial_spec(T), group.group_name, device)
    x = torch.randn(T, _setup.HIDDEN, device=device).to(torch.bfloat16)
    args = [
        kernel_args(
            x,
            prepared_weights(_setup.dense_weights(seed=1000 * i + rank, device=device)),
            partial,
            mhc,
            out,
            group.group_name,
        )
        for i in range(layers)
    ]
    kernel = compile_kernel(args[0])

    def step() -> None:
        for a in args:
            kernel(*a)

    median, minimum = graph_time(step, layers, reps)
    if verbose:
        say(f"rank{rank} T={T} us/launch med {median:.2f} min {minimum:.2f}")
    return {T: {"latency_us": median, "min_latency_us": minimum}}


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main("glmf_dense_tp4", correctness_check, benchmark, verbose, check_only)


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
