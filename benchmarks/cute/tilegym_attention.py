# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Standalone TileGym Triton attention baseline for Helion benchmarks."""

from __future__ import annotations

import inspect
import math
import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DISABLE_TUNE = os.getenv("DISABLE_TUNE", "0") == "1"


def _get_available_triton_backend() -> str:
    if os.environ.get("ENABLE_TILE") == "1":
        try:
            import triton.backends.tileir  # noqa: F401
        except ImportError:
            pass
        else:
            return "nvt"
    return "oait"


INV_LOG_2 = tl.constexpr(1.0 / math.log(2))


_VALID_LAYOUTS = ["bnsd", "nsbd"]


# Check if triton.Config supports v2_opt_level parameter
def _supports_v2_opt_level():
    try:
        sig = inspect.signature(triton.Config.__init__)
        return "v2_opt_level" in sig.parameters
    except Exception:
        return False


_TRITON_SUPPORTS_V2_OPT_LEVEL = _supports_v2_opt_level()


# Kernel helpers


@triton.jit
def _permute_by_layout(b, n, s, d, layout: tl.constexpr):
    if layout == "bnsd":
        return [b, n, s, d]
    if layout == "nsbd":
        return [n, s, b, d]
    raise ValueError(f"Unsupported layout: {layout}")


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _attn_fwd_inner(
    K_desc,
    V_desc,
    Bias_desc_or_ptr,
    Random_mask,
    acc,
    l_i,
    m_i,
    q,
    batch_idx,
    head_idx,
    off_kv_h,
    start_m,
    pid_y,
    q_len_val,
    kv_len_val,
    prefix_kvlen,
    dropout,
    seed,
    qk_scale,
    S_qo: tl.constexpr,
    S_kv: tl.constexpr,
    NEG_INF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_BLOCK_D: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    KV_LEN_MASK: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    DO_DROPOUT: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    USE_RANDOM_MASK: tl.constexpr,
    LAYOUT: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    kv_start = (
        0
        if (WINDOW_SIZE == 0)
        else (
            tl.maximum(0, prefix_kvlen + start_m * BLOCK_M - WINDOW_SIZE)
            // BLOCK_N
            * BLOCK_N
        )
    )
    kv_end = (
        kv_len_val
        if (WINDOW_SIZE == 0)
        else (
            tl.minimum(
                kv_len_val,
                prefix_kvlen + (start_m + 1) * BLOCK_M + WINDOW_SIZE,
            )
        )
    )
    # range of values handled by this stage
    if STAGE == 1 or STAGE == 2:
        stage_1_end = (prefix_kvlen + start_m * BLOCK_M) // BLOCK_N * BLOCK_N
        if STAGE == 1:
            lo, hi = kv_start, stage_1_end  # start_m=0,1,2,3
        else:
            q_seq_end_cur_block = prefix_kvlen + (start_m + 1) * BLOCK_M
            """
            Theoretically, hi should be kv_end if q_seq_end_cur_block > kv_end else q_seq_end_cur_block
            However, it will result in nearly 10us additional overhead. So now only enable this when needed.
            TODO: Investigate the cause.
            """
            if KV_LEN_MASK:
                lo, hi = (
                    stage_1_end,
                    min(q_seq_end_cur_block, kv_end),
                )
            else:
                lo, hi = stage_1_end, q_seq_end_cur_block
    # causal = False
    else:
        lo, hi = kv_start, kv_end
    lo = tl.multiple_of(lo, BLOCK_N)
    cnt = lo // BLOCK_N
    # loop over k, v and update accumulator
    # here use warp_specialize=True only for best performance in oait, for Triton-TileIR, it will ignore this argument
    for curr_n in range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        curr_n = tl.multiple_of(curr_n, BLOCK_N)
        # -- compute qk ----
        # Calculate K offsets - we load BLOCK_N x BLOCK_D and then transpose
        k_offset_b = batch_idx * 1
        k_offset_h = off_kv_h * 1
        k_offset_n = cnt * BLOCK_N
        k_offset_d = 0 * BLOCK_D
        k = K_desc.load(
            _permute_by_layout(k_offset_b, k_offset_h, k_offset_n, k_offset_d, LAYOUT)
        )
        k = tl.reshape(k, (BLOCK_N, BLOCK_D))
        # Transpose K to get (BLOCK_D, BLOCK_N) for matrix multiplication
        k = tl.trans(k)
        qk = tl.dot(q, k)

        # bias loading with TMA
        if BIAS_TYPE == "vector":
            bias = Bias_desc_or_ptr.load([batch_idx, head_idx, 0, curr_n])
            bias = tl.reshape(bias, (BLOCK_N,))
            bias = bias.to(tl.float32)
            bias = bias[None, :]
        elif BIAS_TYPE == "matrix":
            bias = Bias_desc_or_ptr.load(
                [batch_idx, head_idx, start_m * BLOCK_M, curr_n]
            )
            bias = tl.reshape(bias, (BLOCK_M, BLOCK_N))
            bias = bias.to(tl.float32)
        elif BIAS_TYPE == "alibi":
            alibi_scale = tl.load(Bias_desc_or_ptr).to(tl.float32)
            neg_dist = -tl.abs(
                (curr_n + offs_n)[None, :] - (offs_m[:, None] + prefix_kvlen)
            )
            bias = alibi_scale * neg_dist
        if BIAS_TYPE != "none" or SOFT_CAP is not None:
            qk = qk * qk_scale
            if BIAS_TYPE != "none":
                qk = qk + INV_LOG_2 * bias
            if SOFT_CAP is not None:
                qk = qk / SOFT_CAP
                qk = _tanh(qk)
                qk = qk * SOFT_CAP

        if USE_RANDOM_MASK:
            random_mask_offset_b = batch_idx * 1
            random_mask_offset_h = head_idx * 1
            random_mask_offset_m = start_m * BLOCK_M
            random_mask_offset_n = curr_n

            random_mask = Random_mask.load(
                [
                    random_mask_offset_b,
                    random_mask_offset_h,
                    random_mask_offset_m,
                    random_mask_offset_n,
                ]
            )
            random_mask = tl.reshape(random_mask, (BLOCK_M, BLOCK_N))
            qk = tl.where(random_mask, NEG_INF, qk)

        # causal mask
        if STAGE == 2:
            mask = (offs_m[:, None] + prefix_kvlen) >= (curr_n + offs_n[None, :])
            qk = tl.where(mask, qk, NEG_INF)
        # causal mask can override kv_mask, so only apply kv_mask if not causal
        elif KV_LEN_MASK or (STAGE == 3 and not EVEN_N):
            mask = (curr_n + offs_n[None, :]) < hi
            qk = tl.where(mask, qk, NEG_INF)
        # window mask
        if WINDOW_SIZE:
            # causal mask can override right side of window mask
            qk_offset = curr_n - prefix_kvlen + offs_n[None, :] - offs_m[:, None]
            qk = tl.where(
                (qk_offset >= -WINDOW_SIZE) & (qk_offset <= WINDOW_SIZE),
                qk,
                NEG_INF,
            )

        if BIAS_TYPE == "none" and SOFT_CAP is None:
            # Need to apply qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        else:
            # Already applied qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        if DO_DROPOUT:
            # Dropout remains performance-sensitive in this implementation.
            random = tl.rand(
                seed,
                pid_y.to(tl.int64) * S_qo * S_kv
                + offs_m[:, None] * S_kv
                + (curr_n + offs_n)[None, :],
            )
            x_keep = random > dropout
            p = tl.where(x_keep, p / (1.0 - dropout), 0.0)
        # update acc
        # Calculate V offsets
        v_offset_b = batch_idx * 1
        v_offset_h = off_kv_h * 1
        v_offset_n = cnt * BLOCK_N
        v_offset_d = 0 * OUT_BLOCK_D
        v = V_desc.load(
            _permute_by_layout(v_offset_b, v_offset_h, v_offset_n, v_offset_d, LAYOUT)
        )
        v = tl.reshape(v, (BLOCK_N, OUT_BLOCK_D))
        p = p.to(q.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        cnt += 1
    return acc, l_i, m_i


# Autotune configuration


def _create_tma_block_by_layout(layout, block_mn, block_d):
    block_size_map = {"b": 1, "n": 1, "s": block_mn, "d": block_d}
    return [block_size_map[dim] for dim in layout]


def _create_layout_aware_pre_hook():
    """Create layout-aware pre_hook function"""

    def layout_aware_pre_hook(nargs):
        BLOCK_M = nargs["BLOCK_M"]
        BLOCK_N = nargs["BLOCK_N"]
        BLOCK_D = nargs["BLOCK_D"]
        OUT_BLOCK_D = nargs["OUT_BLOCK_D"]
        LAYOUT = nargs["LAYOUT"]

        if not isinstance(nargs["Q"], TensorDescriptor):
            return

        # Set corresponding block_shape for different tensors
        nargs["Q"].block_shape = _create_tma_block_by_layout(LAYOUT, BLOCK_M, BLOCK_D)
        nargs["K_desc"].block_shape = _create_tma_block_by_layout(
            LAYOUT, BLOCK_N, BLOCK_D
        )
        nargs["V_desc"].block_shape = _create_tma_block_by_layout(
            LAYOUT, BLOCK_N, OUT_BLOCK_D
        )
        nargs["Out"].block_shape = _create_tma_block_by_layout(
            LAYOUT, BLOCK_M, OUT_BLOCK_D
        )
        nargs["Random_mask"].block_shape = [1, 1, BLOCK_M, BLOCK_N]
        if nargs["BIAS_TYPE"] == "vector":
            nargs["Bias_desc_or_ptr"].block_shape = [1, 1, 1, BLOCK_N]
        elif nargs["BIAS_TYPE"] == "matrix":
            nargs["Bias_desc_or_ptr"].block_shape = [1, 1, BLOCK_M, BLOCK_N]

    return layout_aware_pre_hook


def _get_default_kernel_configs():
    if _get_available_triton_backend() == "oait":
        return {"BLOCK_M": 64, "BLOCK_N": 64}
    if torch.cuda.get_device_capability() in [(12, 0), (12, 1)]:
        return {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "occupancy": 2,
        }
    return {"BLOCK_M": 256, "BLOCK_N": 128}


def _get_configs(kernel_type="prefill"):
    _hook = _create_layout_aware_pre_hook()

    if _get_available_triton_backend() == "nvt":
        return [
            triton.Config(
                dict(BLOCK_M=BM, BLOCK_N=BN, warp_specialize=False), pre_hook=_hook
            )
            for BM in [64, 128, 256]
            for BN in [64, 128]
        ]

    # warp_specialize requires SM90+ TMA hardware; force False on Ampere (sm_80)
    ws_choices = (
        [False] if torch.cuda.get_device_capability() == (8, 0) else [True, False]
    )
    # for dev and debug
    if DISABLE_TUNE:
        configs = [
            triton.Config(
                dict(BLOCK_M=64, BLOCK_N=64, warp_specialize=ws_choices[0]),
                num_stages=2,
                num_warps=4,
                pre_hook=_hook,
            )
        ]
    # full tuning space for oait
    else:
        configs = [
            triton.Config(
                dict(BLOCK_M=BM, BLOCK_N=BN, warp_specialize=ws),
                num_stages=s,
                num_warps=w,
                pre_hook=_hook,
            )
            for BM in [64, 128, 256]
            for BN in [64, 128]
            for s in [2, 3, 4]
            for w in [4, 8]
            for ws in ws_choices
        ]
    return configs


def _early_config_prune(configs, named_args, **kwargs):
    if _get_available_triton_backend() == "nvt":
        BIAS_TYPE = kwargs.get("BIAS_TYPE")
        # Avoid the known vector-bias configuration mismatch.
        if BIAS_TYPE == "vector":

            def save_config(config):
                block_m = config.kwargs.get("BLOCK_M", None)
                block_n = config.kwargs.get("BLOCK_N", None)
                return block_m != 64 or block_n != 64

            return [cfg for cfg in configs if save_config(cfg)]
        return configs
    return configs


# Device kernels


@triton.autotune(
    configs=_get_configs(),
    key=[
        "S_qo",
        "S_KV_RECOMPILE_KEY",
        "STAGE",
        "QUERY_GROUP_SIZE",
        "dtype",
        "BIAS_TYPE",
        "BLOCK_D",
        "LAYOUT",
        "DO_DROPOUT",
        "WINDOW_SIZE",
        "SOFT_CAP",
        "USE_RANDOM_MASK",
        "EVEN_M",
        "EVEN_N",
    ],
    prune_configs_by={"early_config_prune": _early_config_prune},
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["S_qo"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["S_kv"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _prefill_fmha(
    Q,
    K_desc,
    V_desc,
    Bias_desc_or_ptr,
    Random_mask,
    Q_lens,
    KV_lens,
    Out,
    L,
    sm_scale,
    H,
    S_qo: tl.constexpr,
    S_kv: tl.constexpr,
    dropout: tl.constexpr,
    seed: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_BLOCK_D: tl.constexpr,
    STAGE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_LEN_MASK: tl.constexpr,
    KV_LEN_MASK: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    DO_DROPOUT: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    NEG_INF: tl.constexpr,
    USE_RANDOM_MASK: tl.constexpr,
    LAYOUT: tl.constexpr,
    dtype: tl.constexpr,
    warp_specialize: tl.constexpr,
    S_KV_RECOMPILE_KEY: tl.constexpr,
):
    if isinstance(Q, tl.tensor_descriptor):
        dtype = Q.type.block_type.element_ty
    else:
        dtype = Q.dtype.element_ty
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    batch_idx = pid_y // H
    head_idx = pid_y % H
    if QUERY_GROUP_SIZE:
        off_kv_h = head_idx // QUERY_GROUP_SIZE
    else:
        off_kv_h = head_idx
    qk_scale = sm_scale * INV_LOG_2

    q_len_val = tl.load(Q_lens + batch_idx) if Q_LEN_MASK else S_qo
    kv_len_val = tl.load(KV_lens + batch_idx) if KV_LEN_MASK else S_kv
    prefix_kvlen = kv_len_val - q_len_val
    if pid_x * BLOCK_M < q_len_val:
        # init offset
        offs_m = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        if BIAS_TYPE == "alibi":
            Bias_desc_or_ptr = Bias_desc_or_ptr + head_idx

        # initialize pointer to m and l
        m_i = tl.full([BLOCK_M], NEG_INF * qk_scale, dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, OUT_BLOCK_D], dtype=tl.float32)

        # load q - calculate explicit offsets
        q_offset_b = batch_idx * 1
        q_offset_h = head_idx * 1
        q_offset_m = pid_x * BLOCK_M
        q_offset_d = 0 * BLOCK_D
        q = Q.load(
            _permute_by_layout(q_offset_b, q_offset_h, q_offset_m, q_offset_d, LAYOUT)
        )
        q = tl.reshape(q, (BLOCK_M, BLOCK_D))

        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(
                K_desc,
                V_desc,
                Bias_desc_or_ptr,
                Random_mask,
                acc,
                l_i,
                m_i,
                q,
                batch_idx,
                head_idx,
                off_kv_h,
                pid_x,
                pid_y,
                q_len_val,
                kv_len_val,
                prefix_kvlen,
                dropout,
                seed,
                qk_scale,
                S_qo,
                S_kv,
                NEG_INF,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
                OUT_BLOCK_D,
                4 - STAGE,
                offs_m,
                offs_n,
                KV_LEN_MASK,
                BIAS_TYPE,
                DO_DROPOUT,
                EVEN_M,
                EVEN_N,
                WINDOW_SIZE,
                SOFT_CAP,
                USE_RANDOM_MASK,
                LAYOUT,
                warp_specialize,
            )
        # stage 2: on-band
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            acc, l_i, m_i = _attn_fwd_inner(
                K_desc,
                V_desc,
                Bias_desc_or_ptr,
                Random_mask,
                acc,
                l_i,
                m_i,
                q,
                batch_idx,
                head_idx,
                off_kv_h,
                pid_x,
                pid_y,
                q_len_val,
                kv_len_val,
                prefix_kvlen,
                dropout,
                seed,
                qk_scale,
                S_qo,
                S_kv,
                NEG_INF,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
                OUT_BLOCK_D,
                2,
                offs_m,
                offs_n,
                KV_LEN_MASK,
                BIAS_TYPE,
                DO_DROPOUT,
                EVEN_M,
                EVEN_N,
                WINDOW_SIZE,
                SOFT_CAP,
                USE_RANDOM_MASK,
                LAYOUT,
                warp_specialize,
            )
        # epilogue
        acc = acc / (l_i[:, None])

        l_i = m_i + tl.math.log2(l_i)

        # write back l and o - calculate explicit offsets
        # l_offset_b = batch_idx * 1
        # l_offset_h = head_idx * 1
        # l_offset_m = pid_x * BLOCK_M
        # L.store([l_offset_b, l_offset_h, l_offset_m], l_i.reshape(1, 1, BLOCK_M))

        o_offset_b = batch_idx * 1
        o_offset_h = head_idx * 1
        o_offset_m = pid_x * BLOCK_M
        o_offset_d = 0 * OUT_BLOCK_D

        acc = acc.to(dtype)
        if LAYOUT == "bnsd":
            acc = acc.reshape(1, 1, BLOCK_M, OUT_BLOCK_D)
        elif LAYOUT == "nsbd":
            acc = acc.reshape(1, BLOCK_M, 1, OUT_BLOCK_D)
        else:
            raise ValueError(f"Unsupported layout: {LAYOUT}")
        Out.store(
            _permute_by_layout(o_offset_b, o_offset_h, o_offset_m, o_offset_d, LAYOUT),
            acc,
        )


# Host helpers


def _check_stride(tensor):
    elem_bytes = tensor.dtype.itemsize
    for s in tensor.stride()[:-1]:
        assert (s * elem_bytes) % 16 == 0, "strides must be 16-byte aligned"


def _build_tensor_descriptor(tensor):
    _check_stride(tensor)
    return TensorDescriptor(
        tensor,
        tensor.shape,
        tensor.stride(),
        [1]
        * len(tensor.shape),  # dummy block will be overridden by layout-aware pre-hook
    )


def _permute_by_layout_host(b, n, s, d, layout):
    layout_map = {"b": b, "n": n, "s": s, "d": d}
    return [layout_map[dim] for dim in layout]


# Host launchers and public API


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        is_causal,
        sm_scale,
        kernel_configs=None,
        q_lens=None,
        kv_lens=None,
        bias_type=None,
        bias=None,
        dropout=0.0,
        seed=0,
        window_size=0,
        soft_cap=None,
        random_mask=None,
        layout="bnsd",
    ):
        # Note: BLOCK_M and BLOCK_N in kernel_configs are not used because we enable autotune
        if layout not in _VALID_LAYOUTS:
            raise ValueError(
                f"Invalid layout: {layout}. Valid layouts are: {_VALID_LAYOUTS}"
            )
        assert layout[3] == "d"

        batch_dim, n_head_dim, seq_dim = (
            layout.find("b"),
            layout.find("n"),
            layout.find("s"),
        )
        B, H, S_qo, block_d = (
            q.shape[batch_dim],
            q.shape[n_head_dim],
            q.shape[seq_dim],
            q.shape[3],
        )
        out_block_d = v.size(-1)
        num_head_kv, S_kv = k.shape[n_head_dim], k.shape[seq_dim]

        BLOCK_D = triton.next_power_of_2(block_d)
        OUT_BLOCK_D = triton.next_power_of_2(out_block_d)

        o = torch.empty(
            *_permute_by_layout_host(B, H, S_qo, out_block_d, layout),
            device=q.device,
            dtype=q.dtype,
        )
        l = torch.empty((B, H, S_qo), device=q.device, dtype=torch.float32)

        stage = 3 if is_causal else 1
        if num_head_kv == H:
            query_group_size = 0
        else:
            assert H % num_head_kv == 0
            query_group_size = int(H / num_head_kv)

        # launch fmha fwd kernel
        grid = lambda args: (triton.cdiv(S_qo, args["BLOCK_M"]), B * H, 1)

        if bias_type is not None:
            assert bias.dtype in [q.dtype, torch.float]
            assert bias.is_cuda
            if bias_type == "matrix":
                assert bias.dim() == 4
                assert bias.shape[2:] == (S_qo, S_kv)
                bias = bias.expand(B, H, S_qo, S_kv)
            elif bias_type == "vector":
                assert bias.dim() == 4
                assert bias.shape[2:] == (1, S_kv)
                bias = bias.expand(B, H, S_qo, S_kv)
            elif bias_type == "alibi":
                assert bias.dim() == 1
                bias = bias.view(-1, 1, 1, 1)
        else:
            bias_type = "none"
            bias = torch.empty(0, 0, 0, 0)

        if kernel_configs is None:
            kernel_configs = _get_default_kernel_configs()
        desc_q = _build_tensor_descriptor(q)
        desc_v = _build_tensor_descriptor(v)
        desc_k = _build_tensor_descriptor(k)
        desc_o = _build_tensor_descriptor(o)

        if random_mask is not None:
            assert random_mask.shape == (B, H, S_qo, S_kv)
            desc_random_mask = _build_tensor_descriptor(random_mask)
        else:
            desc_random_mask = torch.empty(0, 0, 0, 0)

        # Create TensorDescriptor for bias (only for vector and matrix types)
        if (bias_type == "vector") or (bias_type == "matrix"):
            desc_bias = _build_tensor_descriptor(bias)
        else:
            desc_bias = bias

        def alloc_fn(size: int, alignment: int, stream: int | None):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        # accumulator will be float32 in any case
        NEG_INF = torch.finfo(torch.float32).min
        USE_RANDOM_MASK = random_mask is not None
        _prefill_fmha[grid](
            desc_q,
            desc_k,
            desc_v,
            desc_bias,
            desc_random_mask,
            q_lens,
            kv_lens,
            desc_o,
            l,
            sm_scale,
            H,
            S_qo=S_qo,
            S_kv=S_kv,
            dropout=dropout,
            seed=seed,
            BLOCK_D=BLOCK_D,
            OUT_BLOCK_D=OUT_BLOCK_D,
            STAGE=stage,
            QUERY_GROUP_SIZE=query_group_size,
            Q_LEN_MASK=q_lens is not None,
            KV_LEN_MASK=kv_lens is not None,
            BIAS_TYPE=bias_type,
            DO_DROPOUT=(dropout != 0),
            WINDOW_SIZE=window_size,
            SOFT_CAP=soft_cap,
            NEG_INF=NEG_INF,
            USE_RANDOM_MASK=USE_RANDOM_MASK,
            LAYOUT=layout,
            dtype=q.dtype,
            S_KV_RECOMPILE_KEY=triton.cdiv(S_kv, 64),
        )
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Attention backward is not implemented yet")


attention = _attention.apply


def fmha_variant_triton(
    q,
    k,
    v,
    scaling=None,
    is_causal=True,
    q_lens=None,
    kv_lens=None,
    bias_type=None,
    bias=None,
    dropout=0.0,
    seed=torch.random.initial_seed(),
    window_size=0,
    soft_cap=None,
    random_mask=None,
    layout="bnsd",
    **kwargs,
):
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))

    kernel_configs = kwargs.get("kernel_configs")
    o = attention(
        q,
        k,
        v,
        is_causal,
        scaling,
        kernel_configs,
        q_lens,
        kv_lens,
        bias_type,
        bias,
        dropout,
        seed,
        window_size,
        soft_cap,
        random_mask,
        layout,
    )
    return o


def get_fmha_variant_sdpa_paged_interface(backend=None, kernel_configs=None):
    def fmha_interface_wrapper(
        module: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        cache = kwargs.pop("cache", None)
        # transformers v5: cu_seq_lens (renamed from cumulative_seqlens_* in v4)
        cu_seqlens_q = kwargs.get("cu_seq_lens_q", kwargs.get("cumulative_seqlens_q"))
        cu_seqlens_k = kwargs.get("cu_seq_lens_k", kwargs.get("cumulative_seqlens_k"))
        # transformers v5: PagedAttentionCache.update() only accepts read_index and write_index;
        # read explicitly instead of passing **kwargs to avoid TypeError on extra keys
        # (e.g. max_seqlen_q, max_seqlen_k, logits_indices that v5 ContinuousBatchProcessor injects)
        read_index = kwargs.get("read_index")
        write_index = kwargs.get("write_index")

        if cache is not None:
            k, v = cache.update(
                k, v, module.layer_idx, read_index=read_index, write_index=write_index
            )

        # Set default values
        if scaling is None:
            scaling = 1.0 / math.sqrt(q.size(-1))
        q_lens = cu_seqlens_q
        kv_lens = cu_seqlens_k
        if q_lens is not None:
            q_lens = q_lens[1:]
        if kv_lens is not None:
            kv_lens = kv_lens[1:]

        # call fmha_interface with the given arguments
        o = fmha_variant_triton(
            q,
            k,
            v,
            is_causal=True,
            scaling=scaling,
            kernel_configs=kernel_configs,
            q_lens=q_lens,
            kv_lens=kv_lens,
        )
        return o.transpose(1, 2).contiguous(), None

    return fmha_interface_wrapper


# fmt: on
