import torch
from torch import Tensor
import helion
import helion.language as hl

def test_int4_gemm():
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
        block_size_k_packed = hl.register_block_size(K // 2)
        block_size_n = hl.register_block_size(N)
        b_bf16 = torch.empty([block_size_k_packed, 2, block_size_n], dtype=torch.bfloat16, device=A.device)

        # Use Helion to tile the computation
        for tile_m in hl.tile(M):
            for tile_n in hl.tile(N, block_size=block_size_n):
                acc = hl.zeros((tile_m, tile_n), dtype=torch.bfloat16)

                for tile_k_packed in hl.tile(K // 2, block_size=block_size_k_packed):
                    # Reshape to [BLOCK_SIZE_K, BLOCK_SIZE_N] - unpacking the int4 values
                    b_bf16_reshaped = b_bf16[tile_k_packed, :, tile_n].reshape([tile_k_packed.block_size * 2, tile_n.block_size])
                    
                    # Load corresponding tiles from A (need to load twice the packed tile size)
                    # We need to map tile_k_packed to the corresponding range in A
                    # Use arange to create indices for the second dimension
                    a_start = tile_k_packed.begin * 2
                    k_indices = hl.arange(a_start, a_start + tile_k_packed.block_size * 2)
                    a_tile = hl.load(A, [tile_m, k_indices])  # [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    
                    acc = acc + hl.dot(a_tile, b_bf16_reshaped).to(torch.bfloat16)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

                C[tile_m, tile_n] = acc

    # Test the kernel
    A = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    B = torch.randint(0, 16, (4096, 8192), dtype=torch.int8, device="cuda")
    C = torch.randn(8192, 8192, dtype=torch.float32, device="cuda")
    matmul_bf16_int4(A, B, C)

test_int4_gemm()
