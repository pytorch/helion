#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl

# Per-shape configs from B200 autotuning
# Config for V=64 shapes
_CFG_V64 = helion.Config(block_sizes=[], indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[64], load_eviction_policies=['first', '', 'first', 'last', 'last'], loop_orders=[[0, 1]], num_sm_multiplier=8, num_stages=2, num_warps=8, pid_type='persistent_interleaved', range_flattens=[None], range_multi_buffers=[False], range_num_stages=[2], range_unroll_factors=[1], range_warp_specializes=[None])

# Config for V=128 shapes
_CFG_V128 = helion.Config(block_sizes=[], indexing=['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', '', '', 'last', 'last'], loop_orders=[[0, 1]], maxnreg=128, num_sm_multiplier=8, num_stages=2, num_warps=16, pid_type='persistent_interleaved', range_flattens=[False], range_multi_buffers=[False], range_num_stages=[2], range_unroll_factors=[1], range_warp_specializes=[None])

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes (B, T, H, K, V)
    (1, 64, 2, 64, 64): _CFG_V64,
    (2, 128, 4, 64, 64): _CFG_V64,
    (1, 256, 4, 64, 128): _CFG_V128,
    # Benchmark shapes
    (1, 64, 1, 64, 64): _CFG_V64,
    (2, 512, 3, 64, 64): _CFG_V64,
    (2, 1024, 3, 64, 64): _CFG_V64,
}


def _make_kernel(config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def chunk_fwd_o_kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)

        BH = B * H
        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            g_vals = g[b_idx, tile_t, h_idx].to(torch.float32)
            q_tile = q[b_idx, tile_t, h_idx, :].to(torch.float32)
            k_tile = k[b_idx, tile_t, h_idx, :].to(torch.float32)
            v_tile = v[b_idx, tile_t, h_idx, :]

            qk = hl.dot(q_tile, k_tile.T, out_dtype=torch.float32)

            # Compute inter-chunk early (independent of intra-chunk) for better pipelining
            q_scaled = q_tile * torch.exp(g_vals)[:, None]
            h_blk = h[b_idx, c_idx, h_idx, :, :].to(torch.float32)
            o_inter = hl.dot(q_scaled, h_blk, out_dtype=torch.float32)

            # Intra-chunk with causal mask
            idx = hl.arange(tile_t.block_size)
            g_diff = g_vals[:, None] - g_vals[None, :]
            causal_mask = idx[:, None] >= idx[None, :]
            sim = torch.where(causal_mask, qk * torch.exp(g_diff), 0.0)
            o_intra = hl.dot(sim.to(v.dtype), v_tile, out_dtype=torch.float32)

            out[b_idx, tile_t, h_idx, :] = ((o_inter + o_intra) * scale).to(out.dtype)

        return out
    return chunk_fwd_o_kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(q, k, v_new, h, g, scale)
