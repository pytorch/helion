import torch
from torch import Tensor
import helion
import helion.language as hl


@helion.kernel(use_default_config=True, static_shapes=True)
def matmul_bf16_int4(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    """
    A: (M, K) bf16
    B: (K, N) int4. assume b is packed with 2 `int4` elements per K. i.e., it's a
        (K//2)xNx(2xint4) matrix, represented in Triton as (K//2)xNxi8.
    C: (M, N) bf16
    """
    M, K = A.shape
    _, N = B.shape

    BLOCK_SIZE_N = hl.register_block_size(N)
    BLOCK_SIZE_K = hl.register_block_size(K)

    # Use Helion to tile the computation
    for tile_m in hl.tile(M):
        for tile_n in hl.tile(N, block_size=BLOCK_SIZE_N):
            acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)

            for tile_k in hl.tile(K, block_size=BLOCK_SIZE_K):
                # hl.load()
                b_tile = B[tile_k.begin//2:tile_k.begin//2 + tile_k.block_size//2, tile_n] # [BLOCK_SIZE_K//2, BLOCK_SIZE_N]
                _4_i8 = hl.full((1, ), 4, dtype=torch.int8)
                b_lo = (b_tile << _4_i8) >> _4_i8
                b_hi = b_tile >> _4_i8
                b_bf16 = torch.stack((b_lo.to(torch.bfloat16), b_hi.to(torch.bfloat16)), dim=2) # [BLOCK_SIZE_K//2, BLOCK_SIZE_N, 2]
                b_bf16 = b_bf16.permute(0, 2, 1) # [BLOCK_SIZE_K//2, 2, BLOCK_SIZE_N]
                b_bf16 = b_bf16.reshape([BLOCK_SIZE_K, BLOCK_SIZE_N]) # [BLOCK_SIZE_K, BLOCK_SIZE_N]
                acc = hl.dot(A[tile_m, tile_k], b_bf16, acc=acc) # [BLOCK_SIZE_M, BLOCK_SIZE_N]

            C[tile_m, tile_n] = acc


# Test the kernel
A = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
B = torch.randint(0, 16, (4096, 8192), dtype=torch.int8, device="cuda")
C = torch.randn(8192, 8192, dtype=torch.float32, device="cuda")
matmul_bf16_int4(A, B, C)
