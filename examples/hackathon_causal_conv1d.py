#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl

# Per-shape configs — will need re-autotuning after kernel rewrite
# Using reasonable defaults for the new kernel structure
_CFG_W4 = helion.Config(block_sizes=[128, 128], num_stages=6, num_warps=4, indexing=['pointer', 'tensor_descriptor', 'pointer', 'pointer'])
_CFG_W3 = helion.Config(block_sizes=[128, 128], num_stages=4, num_warps=4)
_CFG_W8 = helion.Config(block_sizes=[64, 64], num_stages=4, num_warps=4)

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes (B, D, S, W)
    (1, 64, 64, 4): _CFG_W4,
    (2, 128, 128, 4): _CFG_W4,
    (1, 256, 256, 3): _CFG_W3,
    (1, 128, 64, 8): _CFG_W8,
    (4, 64, 128, 4): _CFG_W4,
    # Benchmark shapes
    (1, 1536, 2048, 4): _CFG_W4,
    (1, 2560, 2048, 4): _CFG_W4,
    (1, 2560, 4096, 4): _CFG_W4,
}


def _make_kernel(config):
    @helion.kernel(static_shapes=True, config=config)
    def causal_conv1d_kernel(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))

        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)

        for rb, rd, rs in hl.tile([B, D, S], block_size=[1, None, None]):
            bi = rb.begin
            # Initialize accumulator with bias
            acc = b[rd].to(torch.float32)[:, None] + hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                coeff = w[rd, j].to(torch.float32)
                # Causal index: position (s + j - W + 1) in the original x
                # If index < 0, the value is 0 (causal padding)
                src_idx = rs.index + (j - W + 1)
                # Clamp to valid range, then zero out invalid positions
                safe_idx = torch.where(src_idx >= 0, src_idx, torch.zeros_like(src_idx))
                x_val = hl.load(x, [bi, rd, safe_idx]).to(torch.float32)
                x_val = torch.where(src_idx >= 0, x_val, torch.zeros_like(x_val))
                acc = acc + x_val * coeff[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y
    return causal_conv1d_kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    # No more external padding! Handled inside the kernel
    kernel = _KERNELS[(B, D, S, W)]
    return kernel(x, weight, bias)
