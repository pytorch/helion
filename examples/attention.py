from __future__ import annotations

import math

import torch
from torch.nn.attention.flex_attention import flex_attention

import helion
import helion.language as hl


@helion.kernel(
    config=helion.Config(
        # This config was autotuned on a 3090, it won't be fast for other architectures
        block_sizes=[[32], [16]],
        num_warps=1,
        num_stages=2,
        indexing="block_ptr",
    ),
    static_shapes=True,
)
def attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = q_in.size(-1)
    assert n_dim == v_in.size(-2)
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim], block_size=[1, None]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            # compute qk
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            # update m_i and l_i
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def test(
    z: int,
    h: int,
    n_ctx: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
) -> None:
    q, k, v = [
        torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=device)
        for _ in range(3)
    ]

    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v)

    # flex attention version
    # TODO(jansel): turn the above kernel into a flex attention kernel
    flex_out = flex_attention(q, k, v)
    torch.testing.assert_close(flex_out, ref_out, atol=1e-2, rtol=1e-2)

    # sdpa version
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(sdpa_out, ref_out, atol=1e-2, rtol=1e-2)

    # helion version
    hl_out = attention(q, k, v)
    torch.testing.assert_close(hl_out, ref_out, atol=1e-2, rtol=1e-2)

    # benchmark
    from triton.testing import do_bench

    spda_sec = do_bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    )
    flex_sec = do_bench(lambda: flex_attention(q, k, v))
    helion_sec = do_bench(lambda: attention(q, k, v))
    print(
        f"Helion time: {helion_sec:.4f}s, flex time: {flex_sec:.4f}, torch time: {spda_sec:.4f}"
    )


def main() -> None:
    test(2, 32, 1024, 64, torch.float16)


if __name__ == "__main__":
    main()
