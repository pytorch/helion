from __future__ import annotations

import math

import torch

import helion
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl


@helion.kernel(
    configs=[
        helion.Config(
            block_sizes=[256, N],
            loop_orders=[[2, 1, 0]],
            range_warp_specializes=[],
            range_multi_buffers=[],
            pid_type="persistent_interleaved",
            num_warps=4,
            num_stages=3,
            _triton_data_partition_factor=2,
        )
        for N in [64, 128]
    ],
    static_shapes=True,
    autotune_accuracy_check=False,
)
def attention_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H, M, D = q.shape
    Bk, Hk, N, Dk = k.shape
    assert Dk == D
    assert Bk == B
    assert Hk == H
    Bv, Hv, Nv, Dv = v.shape
    assert Bv == B
    assert Hv == Hk
    assert Nv == N
    D = hl.specialize(D)
    Dv = hl.specialize(Dv)
    o = q.new_empty(B, H, M, Dv)
    lse = q.new_empty(B, H, M, dtype=torch.float32)
    block_m = hl.register_block_size(M)
    block_n = hl.register_block_size(N)
    hl.register_tunable("_triton_data_partition_factor", EnumFragment(choices=(2,)))
    OUTER_LOOP = True
    SUBTILING = True
    VECT_MUL = False
    FADD2_REDUCE = False
    sm_scale = 1.0 / math.sqrt(D)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_h, tile_m in hl.tile([B, H, M], block_size=[1, 1, block_m]):
        for tile_m_lo, tile_m_hi, tile_0, tile_1 in hl.tile(
            [tile_m.begin, tile_m.begin + block_m // 2, 0, 1],
            [tile_m.begin + block_m // 2, tile_m.begin + block_m, 1, 2],
            block_size=[block_m // 2, block_m // 2, 1, 1],
        ):
            m_i0 = hl.zeros([tile_m_lo]) - float("inf")
            m_i1 = hl.zeros([tile_m_hi]) - float("inf")
            l_i0 = hl.zeros([tile_m_lo]) + 1.0
            l_i1 = hl.zeros([tile_m_hi]) + 1.0
            acc_0 = hl.zeros([tile_m_lo, Dv])
            acc_1 = hl.zeros([tile_m_hi, Dv])
            q_i0 = q[tile_b.begin, tile_h.begin, tile_m_lo, :]
            q_i1 = q[tile_b.begin, tile_h.begin, tile_m_hi, :]
            if FADD2_REDUCE:
                l_i0_1 = hl.zeros([tile_m_lo])
                l_i1_1 = hl.zeros([tile_m_hi])
            else:
                l_i0_1 = 0
                l_i1_1 = 0
            for tile_n in hl.tile(N, block_size=block_n):
                k_j = k[tile_b.begin, tile_h.begin, tile_n, :]
                v_j = v[tile_b.begin, tile_h.begin, tile_n, :]

                qk = hl.dot(q_i0, k_j.T, out_dtype=torch.float32)
                m_ij = torch.maximum(m_i0, torch.amax(qk, 1) * qk_scale)
                if VECT_MUL:
                    qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
                else:
                    qk = qk * qk_scale - m_ij[:, None]

                p = torch.exp2(qk)
                # -- compute correction factor
                alpha = torch.exp2(m_i0 - m_ij)
                if not FADD2_REDUCE:
                    l_ij = torch.sum(p, 1)

                if SUBTILING:
                    acc_0 = hl.inline_triton(
                        """
                        BM: tl.constexpr = {0}.shape[0]
                        BN: tl.constexpr = {0}.shape[1]
                        acc0, acc1 = {0}.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                        if False: #VECT_MUL:
                            acc0 = _mul_f32x2(acc0, {1}[:, None])
                            acc1 = _mul_f32x2(acc1, {1}[:, None])
                        else:
                            acc0 = acc0 * {1}[:, None]
                            acc1 = acc1 * {1}[:, None]
                        tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
                    """,
                        [acc_0, alpha],
                        acc_0,
                    )
                else:
                    acc_0 = acc_0 * alpha[:, None]

                # update m_i and l_i
                # place this at the end of the loop to reduce register pressure
                if FADD2_REDUCE:
                    p0, p1 = (
                        p.reshape([block_m // 2, 2, block_n // 2])
                        .permute(0, 2, 1)
                        .split()
                    )
                    l_ij0, l_ij1 = hl.reduce((p0, p1), axis=1, combine_fn=_reduce_fadd2)
                    l_i0 = l_i0 * alpha + l_ij0
                    l_i0_1 = l_i0_1 * alpha + l_ij1

                # We can potentially move these to be before updating l_ij, so the dot
                # is not blocked.
                # prepare p and v for the dot
                p = p.to(v.dtype)
                # note that this non transposed v for FP8 is only supported on Blackwell
                acc_0 = hl.dot(p, v_j, acc=acc_0)
                if not FADD2_REDUCE:
                    l_i0 = l_i0 * alpha + l_ij
                m_i0 = m_ij

                # and repeat again!

            if FADD2_REDUCE:
                l_i0 = l_i0 + l_i0_1
                l_i1 = l_i1 + l_i1_1

            m_i0 += torch.log2(l_i0)
            acc_0 = acc_0 / l_i0[:, None]
            lse[tile_b.begin, tile_h.begin, tile_m_lo] = m_i0
            o[tile_b.begin, tile_h.begin, tile_m_lo, :] = acc_0

            m_i1 += torch.log2(l_i1)
            acc_1 = acc_1 / l_i1[:, None]
            lse[tile_b.begin, tile_h.begin, tile_m_hi] = m_i1
            o[tile_b.begin, tile_h.begin, tile_m_hi, :] = acc_1

    return o, lse


B, H, S, D = 4, 16, 4096, 128
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
attention_kernel(q, k, v)
