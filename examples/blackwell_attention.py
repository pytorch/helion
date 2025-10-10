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
            range_warp_specializes=[True, None],
            range_multi_buffers=[None, False],
            pid_type="persistent_interleaved",
            indexing="tensor_descriptor",
            num_warps=4,
            num_stages=3,
            _triton_data_partition_factor=2,
        )
        for N in [128]
    ],
    static_shapes=True,
    autotune_accuracy_check=False,
)
def attention_kernel(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
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
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, Dv)
    MM = q.shape[0]
    NN = k.shape[0]
    assert v.shape[0] == k.shape[0]
    o = q.new_empty(MM, Dv)
    lse = q.new_empty(MM, dtype=torch.float32)
    block_m = hl.register_block_size(M)
    block_n = hl.register_block_size(N)
    assert M % block_m == 0
    assert N % block_n == 0
    hl.register_tunable("_triton_data_partition_factor", EnumFragment(choices=(2,)))
    OUTER_LOOP = True
    SUBTILING = True
    VECT_MUL = 1
    FADD2_REDUCE = False
    sm_scale = 1.0 / math.sqrt(D)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_m in hl.tile(MM, block_size=block_m):
        m_i = hl.zeros([tile_m]) - float("inf")
        l_i_0 = hl.zeros([tile_m]) + 1.0
        acc = hl.zeros([tile_m, Dv])
        q_i = q[tile_m, :]
        l_i_1 = 0

        for tile_n in hl.tile(
            tile_m.begin // M * N, tile_m.begin // M * N + N, block_size=block_n
        ):
            k_j = k[tile_n, :]
            v_j = v[tile_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            if VECT_MUL == 2 or VECT_MUL == 3:
                qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
            else:
                qk = qk * qk_scale - m_ij[:, None]

            p = torch.exp2(qk)
            # -- compute correction factor
            alpha = torch.exp2(m_i - m_ij)
            if not FADD2_REDUCE:
                l_ij = torch.sum(p, -1)

            if SUBTILING:
                if VECT_MUL == 1 or VECT_MUL == 3:
                    acc_0, acc_1 = torch.chunk(acc, dim=-1, chunks=2)
                    acc_0 = hl.inline_asm_elementwise(
                        """
                           {{
                               .reg .b64 ra, rb, rc;
                               mov.b64 ra, {{ $2, $3 }};
                               mov.b64 rb, {{ $4, $5 }};
                               mul.f32x2 rc, ra, rb;
                               mov.b64 {{ $0, $1 }}, rc;
                           }}
                           """,
                        "=r,=r,r,r,r,r",
                        [acc_0, alpha[:, None]],
                        dtype=torch.float32,
                        is_pure=True,
                        pack=2,
                    )
                    acc_1 = hl.inline_asm_elementwise(
                        """
                           {{
                               .reg .b64 ra, rb, rc;
                               mov.b64 ra, {{ $2, $3 }};
                               mov.b64 rb, {{ $4, $5 }};
                               mul.f32x2 rc, ra, rb;
                               mov.b64 {{ $0, $1 }}, rc;
                           }}
                           """,
                        "=r,=r,r,r,r,r",
                        [acc_1, alpha[:, None]],
                        dtype=torch.float32,
                        is_pure=True,
                        pack=2,
                    )
                    acc = torch.stack([acc_0, acc_1], dim=-2).reshape(
                        acc.size(0), acc.size(1)
                    )
                else:
                    acc = hl.inline_triton(
                        """
                        BM: tl.constexpr = {0}.shape[2]
                        BN: tl.constexpr = {0}.shape[3]
                        acc0, acc1 = {0}.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                        acc0 = acc0 * {1}[:, None]
                        acc1 = acc1 * {1}[:, None]
                        tl.join(acc0, acc1).permute(0, 2, 1).reshape([1, 1, BM, BN])
                    """,
                        [acc, alpha],
                        acc,
                    )
            else:
                acc = acc * alpha[:, None]

            # update m_i and l_i
            # place this at the end of the loop to reduce register pressure
            if FADD2_REDUCE:
                p0, p1 = (
                    p.reshape([block_m // 2, 2, block_n // 2]).permute(0, 2, 1).split()
                )
                l_ij0, l_ij1 = hl.reduce((p0, p1), axis=1, combine_fn=_reduce_fadd2)
                l_i0 = l_i0 * alpha + l_ij0
                l_i0_1 = l_i0_1 * alpha + l_ij1

            # We can potentially move these to be before updating l_ij, so the dot
            # is not blocked.
            # prepare p and v for the dot
            p = p.to(v.dtype)
            # note that this non transposed v for FP8 is only supported on Blackwell
            acc = hl.dot(p, v_j, acc=acc)

            if not FADD2_REDUCE:
                l_i_0 = l_i_0 * alpha + l_ij
            m_i = m_ij

        if FADD2_REDUCE:
            l_i = l_i_0 + l_i_1
        else:
            l_i = l_i_0

        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc

    return o.reshape(B, H, M, Dv), lse.reshape(B, H, M)


B, H, S = 4, 16, 4096
for D in [64, 128]:
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    attention_kernel(q, k, v)
    time = do_bench(lambda: attention_kernel(q, k, v))
    print(B, H, S, D, time, B * H * S * S * D * 4 / time)
