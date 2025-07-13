"""
Grouped GEMM - Multiple matrix multiplications in a single kernel launch
"""

from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel(static_shapes=False)
def grouped_gemm(
    A_concat: torch.Tensor,  # Concatenated A matrices [total_M, K]
    B_concat: torch.Tensor,  # Concatenated B matrices [K, total_N]
    group_sizes_M: torch.Tensor,  # [G] - M size for each GEMM
    group_sizes_N: torch.Tensor,  # [G] - N size for each GEMM
    group_offsets_M: torch.Tensor,  # [G+1] - Starting M offset for each GEMM
    group_offsets_N: torch.Tensor,  # [G+1] - Starting N offset for each GEMM
    max_M_tensor: torch.Tensor,  # Dummy tensor of size max(M)
    max_N_tensor: torch.Tensor,  # Dummy tensor of size max(N)
) -> torch.Tensor:  # [total_M, total_N] - Concatenated output
    """Grouped GEMM kernel using concatenated tensors"""
    G = group_sizes_M.shape[0]
    total_M, K = A_concat.shape
    _, total_N = B_concat.shape
    max_M = max_M_tensor.numel()
    max_N = max_N_tensor.numel()
    
    # Allocate output tensor
    C_concat = torch.zeros(
        total_M, total_N,
        dtype=torch.promote_types(A_concat.dtype, B_concat.dtype),
        device=A_concat.device
    )
    
    # Process each GEMM
    for g_idx in hl.grid(G):
        # Get dimensions and offsets for this GEMM
        M = group_sizes_M[g_idx]
        N = group_sizes_N[g_idx]
        M_start = group_offsets_M[g_idx]
        N_start = group_offsets_N[g_idx]
        
        # Skip empty GEMMs
        valid_gemm = (M > 0) * (N > 0)  # Use multiplication instead of 'and'
        if valid_gemm:
            # Tile over output dimensions
            for tile_m, tile_n in hl.tile([max_M, max_N]):
                # Get tile indices
                m_indices = tile_m.index
                n_indices = tile_n.index
                
                # Create masks for valid elements
                m_valid = m_indices < M
                n_valid = n_indices < N
                
                # Calculate global indices
                m_indices_valid = torch.where(m_valid, m_indices, 0)
                n_indices_valid = torch.where(n_valid, n_indices, 0)
                
                # Global indices in concatenated tensors
                global_m = M_start + m_indices_valid
                global_n = N_start + n_indices_valid
                
                # Initialize accumulator
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                
                # Accumulate over K dimension
                for tile_k in hl.tile(K):
                    k_indices = tile_k.index
                    
                    # Load tiles from concatenated tensors
                    A_tile = A_concat[global_m, k_indices]
                    B_tile = B_concat[k_indices, global_n]
                    
                    # Accumulate
                    acc = torch.addmm(acc, A_tile, B_tile)
                
                # Write back to output with masking
                block_m = acc.size(0)
                block_n = acc.size(1)
                
                # Get existing values
                existing_values = C_concat[global_m, global_n]
                
                # Create 2D mask for output
                mask_2d = m_valid.view(block_m, 1).expand(block_m, block_n) & n_valid.view(1, block_n).expand(block_m, block_n)
                
                # Write results only for valid positions
                C_concat[global_m, global_n] = torch.where(
                    mask_2d, acc.to(C_concat.dtype), existing_values
                )
    
    return C_concat


def grouped_gemm_helion_kernel_args_gen(
    group_A: list[torch.Tensor],
    group_B: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate arguments for the Helion kernel by concatenating inputs"""
    device = group_A[0].device
    dtype = group_A[0].dtype
    G = len(group_A)
    
    # Check that all matrices have the same K dimension
    K = group_A[0].shape[1]
    for i in range(G):
        assert group_A[i].shape[1] == K, f"All A matrices must have same K dimension"
        assert group_B[i].shape[0] == K, f"All B matrices must have K dimension matching A"
    
    # Get sizes for each GEMM
    Ms = [A.shape[0] for A in group_A]
    Ns = [B.shape[1] for B in group_B]
    
    # Find maximum dimensions
    max_M = max(Ms)
    max_N = max(Ns)
    
    # Calculate offsets
    M_offsets = [0]
    N_offsets = [0]
    for i in range(G):
        M_offsets.append(M_offsets[-1] + Ms[i])
        N_offsets.append(N_offsets[-1] + Ns[i])
    
    # Concatenate tensors
    A_concat = torch.cat(group_A, dim=0)  # [total_M, K]
    B_concat = torch.cat(group_B, dim=1)  # [K, total_N]
    
    # Create size and offset tensors
    group_sizes_M = torch.tensor(Ms, dtype=torch.int32, device=device)
    group_sizes_N = torch.tensor(Ns, dtype=torch.int32, device=device)
    group_offsets_M = torch.tensor(M_offsets, dtype=torch.int32, device=device)
    group_offsets_N = torch.tensor(N_offsets, dtype=torch.int32, device=device)
    
    # Create dummy tensors to pass dimensions
    max_M_tensor = torch.empty(max_M, device=device)
    max_N_tensor = torch.empty(max_N, device=device)
    
    return (A_concat, B_concat, group_sizes_M, group_sizes_N, 
            group_offsets_M, group_offsets_N, max_M_tensor, max_N_tensor)


def split_output(C_concat: torch.Tensor, group_sizes_M: list[int], group_sizes_N: list[int]) -> list[torch.Tensor]:
    """Split concatenated output back into individual matrices"""
    outputs = []
    M_offset = 0
    N_offset = 0
    
    for M, N in zip(group_sizes_M, group_sizes_N):
        C = C_concat[M_offset:M_offset+M, N_offset:N_offset+N]
        outputs.append(C)
        M_offset += M
        N_offset += N
    
    return outputs


def grouped_gemm_tritonbench(group_A: list[torch.Tensor], group_B: list[torch.Tensor]) -> list[torch.Tensor]:
    """Wrapper function for tritonbench compatibility"""
    # Use the concatenated approach for better performance
    kernel_args = grouped_gemm_helion_kernel_args_gen(group_A, group_B)
    C_concat = grouped_gemm(*kernel_args)
    
    # Split output back into individual matrices
    Ms = [A.shape[0] for A in group_A]
    Ns = [B.shape[1] for B in group_B]
    return split_output(C_concat, Ms, Ns)


def grouped_gemm_pytorch(group_A: list[torch.Tensor], group_B: list[torch.Tensor]) -> list[torch.Tensor]:
    """Reference PyTorch implementation"""
    outputs = []
    for A, B in zip(group_A, group_B):
        C = torch.matmul(A, B)
        outputs.append(C)
    return outputs


def check(group_size: int = 4, base_size: int = 256) -> None:
    """Test the grouped GEMM implementation"""
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test data with varying sizes
    group_A = []
    group_B = []
    
    for i in range(group_size):
        # Vary sizes for each GEMM to test handling of different dimensions
        M = base_size + i * 64
        N = base_size + (i + 1) * 32
        K = base_size  # Keep K constant for concatenation
        
        A = torch.randn(M, K, device=device, dtype=dtype)
        B = torch.randn(K, N, device=device, dtype=dtype)
        
        group_A.append(A)
        group_B.append(B)
    
    # Test the concatenated kernel
    kernel_args = grouped_gemm_helion_kernel_args_gen(group_A, group_B)
    
    def helion_fn() -> torch.Tensor:
        return grouped_gemm(*kernel_args)
    
    def reference_fn() -> torch.Tensor:
        # Create reference output in concatenated form
        C_list = grouped_gemm_pytorch(group_A, group_B)
        
        # Concatenate in block-diagonal form
        total_M = sum(A.shape[0] for A in group_A)
        total_N = sum(B.shape[1] for B in group_B)
        C_concat = torch.zeros(total_M, total_N, device=device, dtype=torch.promote_types(group_A[0].dtype, group_B[0].dtype))
        
        M_offset = 0
        N_offset = 0
        for C in C_list:
            M, N = C.shape
            C_concat[M_offset:M_offset+M, N_offset:N_offset+N] = C
            M_offset += M
            N_offset += N
        
        return C_concat
    
    # Compare outputs
    run_example(helion_fn, reference_fn, ())


def main() -> None:
    # Test with different configurations
    print("Testing grouped GEMM with group_size=4, base_size=256")
    check(group_size=4, base_size=256)
    
    print("\nTesting grouped GEMM with group_size=8, base_size=128")
    check(group_size=8, base_size=128)
    
    print("\nTesting grouped GEMM with group_size=2, base_size=512")
    check(group_size=2, base_size=512)


if __name__ == "__main__":
    main()