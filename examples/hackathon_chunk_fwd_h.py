#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl

CHUNK_SIZE = 64

# Per-shape configs from B200 autotuning
_CFG_1 = helion.Config(block_sizes=[16], indexing=['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], l2_groupings=[16], load_eviction_policies=['', 'last', 'first', 'first', ''], loop_orders=[[1, 0, 2]], num_stages=5, num_warps=32, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[True])

_CFG_2 = helion.Config(block_sizes=[4], indexing=['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], l2_groupings=[4], load_eviction_policies=['last', 'last', 'first', 'first', 'last'], loop_orders=[[0, 2, 1]], num_stages=4, num_warps=8, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 3], range_unroll_factors=[0, 2], range_warp_specializes=[None, False], static_ranges=[False])

# Same as _CFG_2 but without static_ranges (not valid for some shapes with static_shapes=True)
_CFG_2b = helion.Config(block_sizes=[4], indexing=['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], l2_groupings=[4], load_eviction_policies=['last', 'last', 'first', 'first', 'last'], loop_orders=[[0, 2, 1]], num_stages=4, num_warps=8, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 3], range_unroll_factors=[0, 2], range_warp_specializes=[None, False])

_CFG_3 = helion.Config(block_sizes=[4], indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', '', 'last', 'first', 'last'], loop_orders=[[1, 2, 0]], num_stages=2, num_warps=2, pid_type='flat', range_flattens=[None, True], range_multi_buffers=[None, None], range_num_stages=[0, 4], range_unroll_factors=[0, 1], range_warp_specializes=[None, None], static_ranges=[False])

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes (B, T, H, K, V)
    (1, 64, 2, 64, 64): _CFG_1,
    (2, 128, 4, 64, 64): _CFG_2,
    (1, 256, 4, 64, 128): _CFG_3,
    # Benchmark shapes
    (1, 64, 1, 64, 64): _CFG_2,
    (2, 512, 3, 64, 64): _CFG_2,
    (2, 1024, 3, 64, 64): _CFG_2b,
}


def _make_kernel(config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def chunk_fwd_h_kernel(
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = u.shape[-1]
        K = hl.specialize(K)
        C = hl.specialize(chunk_size)

        NT = T // C
        dtype = k.dtype

        h_out = torch.empty(B, NT, H, K, V, dtype=dtype, device=k.device)
        v_new_out = torch.empty_like(u)

        block_v = hl.register_block_size(V)

        for tile_b, tile_h, tile_v in hl.tile(
            [B, H, V], block_size=[1, 1, block_v]
        ):
            i_b = tile_b.id
            i_h = tile_h.id

            b_h = hl.zeros([K, tile_v], dtype=torch.float32)

            for t_i in hl.tile(T, block_size=C):
                h_out[i_b, t_i.id, i_h, :, tile_v] = b_h.to(dtype)

                b_w = w[i_b, t_i, i_h, :]
                c_h = b_h.to(dtype)
                b_wh = hl.dot(b_w, c_h, out_dtype=torch.float32)
                p_u = u[i_b, t_i, i_h, tile_v].to(torch.float32)
                b_v = p_u - b_wh

                v_new_out[i_b, t_i, i_h, tile_v] = b_v.to(dtype)

                m_t = t_i.index < T
                t_i_last = min(t_i.begin + C, T) - 1
                b_g_last = g[i_b, t_i_last, i_h].to(torch.float32)
                b_g = g[i_b, t_i, i_h].to(torch.float32)
                b_v = b_v * torch.where(m_t, torch.exp(b_g_last - b_g), 0)[:, None]

                b_g_last_exp = torch.exp(b_g_last)
                b_h = b_h * b_g_last_exp

                b_v = b_v.to(dtype)
                p_k = k[i_b, t_i, i_h, :]
                b_h = hl.dot(p_k.T, b_v, acc=b_h)

        return h_out, v_new_out
    return chunk_fwd_h_kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, w, u, g, CHUNK_SIZE)
