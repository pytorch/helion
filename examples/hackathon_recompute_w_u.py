#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl

CHUNK_SIZE = 64

# Per-shape configs from B200 autotuning
# Test shapes
_CFG_64_64 = helion.Config(block_sizes=[], indexing=['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], l2_groupings=[32], load_eviction_policies=['', 'last', '', 'last', ''], loop_orders=[[0, 1]], num_stages=6, num_warps=8, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None])
_CFG_64_64_b = helion.Config(block_sizes=[], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[2], load_eviction_policies=['', '', 'first', 'first', ''], loop_orders=[[1, 0]], num_stages=1, num_warps=8, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None])
_CFG_64_128 = helion.Config(block_sizes=[], indexing=['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], l2_groupings=[4], load_eviction_policies=['last', '', 'first', 'first', 'first'], loop_orders=[[0, 1]], num_stages=1, num_warps=8, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None])

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): _CFG_64_64,
    (2, 128, 4, 64, 64): _CFG_64_64_b,
    (1, 256, 4, 64, 128): _CFG_64_128,
    # Benchmark shapes (reuse closest test config)
    (1, 64, 1, 64, 64): _CFG_64_64,
    (2, 512, 3, 64, 64): _CFG_64_64_b,
    (2, 1024, 3, 64, 64): _CFG_64_64_b,
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])
        K = hl.specialize(K)
        V = hl.specialize(V)

        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)

        BH = B * H
        for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H

            a = A[b_idx, rt, h_idx, :].to(torch.float32)
            beta_c = beta[b_idx, rt, h_idx].to(torch.float32)
            g_c = g[b_idx, rt, h_idx].to(torch.float32)

            v_c = v[b_idx, rt, h_idx, :].to(torch.float32)
            scaled_v = v_c * beta_c[:, None]
            u_result = hl.dot(a, scaled_v, out_dtype=torch.float32)
            u_out[b_idx, rt, h_idx, :] = u_result.to(v.dtype)

            k_c = k[b_idx, rt, h_idx, :].to(torch.float32)
            scaled_k = k_c * (beta_c * torch.exp(g_c))[:, None]
            w_result = hl.dot(a, scaled_k, out_dtype=torch.float32)
            w_out[b_idx, rt, h_idx, :] = w_result.to(k.dtype)

        return w_out, u_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, v, beta, A, g)
