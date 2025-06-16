"""
Grouped GEMM kernel for Helion

This kernel performs a group of matrix multiplications where each group can have
different matrix dimensions. It's designed to efficiently handle multiple GEMM
operations with varying sizes in a single kernel launch.

Based on the tritonbench (https://github.com/pytorch-labs/tritonbench/tree/main/tritonbench/operators/grouped_gemm) but adapted for Helion's DSL.
"""

from __future__ import annotations

import helion
import helion.language as hl

import torch


@helion.kernel(static_shapes=False)
def single_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Single matrix multiplication kernel optimized for helion.
    This will be called multiple times for grouped operations.
    """
    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, f"K dimension mismatch {K} != {K_B}"

    # Create output tensor
    C = torch.empty(
        [M, N], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )

    # Tile over M and N dimensions
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        # Reduction over K dimension
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, A[tile_m, tile_k], B[tile_k, tile_n])

        # Store result
        C[tile_m, tile_n] = acc

    return C


def grouped_gemm_v2(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> list[torch.Tensor]:
    """
    Grouped GEMM implementation that processes each group with individual kernel calls.

    This approach works within helion's type system by avoiding dynamic list indexing
    within kernels and instead using helion's optimized single GEMM kernel for each group.
    """
    assert len(group_A) == len(group_B), "group_A and group_B must have same length"
    group_size = len(group_A)

    if group_size == 0:
        return []

    group_C = []

    # Process each group with a separate kernel call
    # This allows helion to optimize each GEMM individually
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]

        # Call the optimized single GEMM kernel
        C = single_gemm(A, B)
        group_C.append(C)

    return group_C


# Alternative implementation using concatenated tensors (more advanced)
@helion.kernel(static_shapes=False)
def grouped_gemm_concatenated(
    A_concat: torch.Tensor,  # Concatenated A matrices
    B_concat: torch.Tensor,  # Concatenated B matrices
    A_offsets: torch.Tensor,  # [group_size + 1] - Start indices for A matrices
    B_offsets: torch.Tensor,  # [group_size + 1] - Start indices for B matrices
    C_offsets: torch.Tensor,  # [group_size + 1] - Start indices for C matrices
    group_shapes: torch.Tensor,  # [group_size, 3] - [M, N, K] for each group
    group_size: int,
) -> torch.Tensor:
    """
    Alternative implementation using concatenated tensors to avoid list indexing.
    This version processes all groups in a single kernel launch.
    """
    # Calculate total output size
    total_output_size = C_offsets[group_size].item()
    C_concat = torch.empty(
        [total_output_size], device=A_concat.device, dtype=A_concat.dtype
    )

    # Process each group
    for g in range(group_size):
        # Get dimensions for this group
        M = group_shapes[g, 0].item()
        N = group_shapes[g, 1].item()
        K = group_shapes[g, 2].item()

        # Skip empty groups
        if M == 0 or N == 0 or K == 0:
            continue

        # Get offsets for this group
        a_start = A_offsets[g].item()
        a_end = A_offsets[g + 1].item()
        b_start = B_offsets[g].item()
        b_end = B_offsets[g + 1].item()
        c_start = C_offsets[g].item()
        c_end = C_offsets[g + 1].item()

        # Extract matrices for this group
        A_flat = A_concat[a_start:a_end]
        B_flat = B_concat[b_start:b_end]

        # Reshape to proper matrix shapes
        A = A_flat.view(M, K)
        B = B_flat.view(K, N)

        # Compute matrix multiplication for this group
        for tile_m, tile_n in hl.tile([M, N]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

            for tile_k in hl.tile(K):
                acc = torch.addmm(acc, A[tile_m, tile_k], B[tile_k, tile_n])

            # Store result in concatenated output
            C_group = acc.view(-1)  # Flatten the result
            tile_start = c_start + tile_m.begin * N + tile_n.begin
            tile_end = tile_start + C_group.numel()
            C_concat[tile_start:tile_end] = C_group

    return C_concat


def grouped_gemm_simple(
    group_A: list[torch.Tensor],
    group_B: list[torch.Tensor],
    use_concatenated: bool = False,
) -> list[torch.Tensor]:
    """
    Main interface for grouped GEMM with option to use concatenated approach.

    Args:
        group_A: List of A matrices, each with shape [M_i, K_i]
        group_B: List of B matrices, each with shape [K_i, N_i]
        use_concatenated: Whether to use the concatenated tensor approach

    Returns:
        List of C matrices, each with shape [M_i, N_i]
    """
    if not use_concatenated:
        return grouped_gemm_v2(group_A, group_B)

    # Prepare concatenated tensors
    assert len(group_A) == len(group_B), "group_A and group_B must have same length"
    group_size = len(group_A)

    if group_size == 0:
        return []

    device = group_A[0].device

    # Flatten and concatenate all matrices
    A_flats = []
    B_flats = []
    group_shapes = []
    A_offsets = [0]
    B_offsets = [0]
    C_offsets = [0]

    for A, B in zip(group_A, group_B):
        M, K = A.shape
        K_B, N = B.shape
        assert K == K_B, f"K dimension mismatch {K} != {K_B}"

        A_flat = A.flatten()
        B_flat = B.flatten()

        A_flats.append(A_flat)
        B_flats.append(B_flat)
        group_shapes.extend([M, N, K])

        A_offsets.append(A_offsets[-1] + A_flat.numel())
        B_offsets.append(B_offsets[-1] + B_flat.numel())
        C_offsets.append(C_offsets[-1] + M * N)

    # Create concatenated tensors
    A_concat = torch.cat(A_flats)
    B_concat = torch.cat(B_flats)
    A_offsets_tensor = torch.tensor(A_offsets, device=device, dtype=torch.int32)
    B_offsets_tensor = torch.tensor(B_offsets, device=device, dtype=torch.int32)
    C_offsets_tensor = torch.tensor(C_offsets, device=device, dtype=torch.int32)
    group_shapes_tensor = torch.tensor(
        group_shapes, device=device, dtype=torch.int32
    ).view(group_size, 3)

    # Call concatenated kernel
    C_concat = grouped_gemm_concatenated(
        A_concat,
        B_concat,
        A_offsets_tensor,
        B_offsets_tensor,
        C_offsets_tensor,
        group_shapes_tensor,
        group_size,
    )

    # Split result back into individual matrices
    group_C = []
    for i in range(group_size):
        M = group_shapes_tensor[i, 0].item()
        N = group_shapes_tensor[i, 1].item()
        c_start = C_offsets[i]
        c_end = C_offsets[i + 1]
        C_flat = C_concat[c_start:c_end]
        C = C_flat.view(M, N)
        group_C.append(C)

    return group_C


def grouped_gemm_reference(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Reference implementation using standard PyTorch operations."""
    return [torch.matmul(A, B) for A, B in zip(group_A, group_B)]


def check(
    group_sizes: list[tuple[int, int, int]],
    use_v2: bool = True,
    use_concatenated: bool = False,
) -> None:
    """
    Test the grouped GEMM implementation.

    Args:
        group_sizes: List of (M, N, K) tuples for each group
        use_v2: Whether to use the v2 implementation (individual kernels)
        use_concatenated: Whether to test the concatenated approach
    """
    from triton.testing import do_bench

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # Create test data
    group_A = []
    group_B = []

    for M, N, K in group_sizes:
        A = torch.randn([M, K], device=device, dtype=dtype)
        B = torch.randn([K, N], device=device, dtype=dtype)
        group_A.append(A)
        group_B.append(B)

    # Test correctness
    if use_v2:
        result_helion = grouped_gemm_v2(group_A, group_B)
        kernel_name = "grouped_gemm_v2 (individual kernels)"
    else:
        result_helion = grouped_gemm_simple(
            group_A, group_B, use_concatenated=use_concatenated
        )
        kernel_name = f"grouped_gemm_simple (concatenated={use_concatenated})"

    result_reference = grouped_gemm_reference(group_A, group_B)

    for i, (helion_res, ref_res) in enumerate(zip(result_helion, result_reference)):
        torch.testing.assert_close(
            helion_res,
            ref_res,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Group {i} results don't match",
        )

    print(
        f"✓ Correctness test passed for {len(group_sizes)} groups using {kernel_name}"
    )

    # Benchmark performance
    if device == "cuda":
        if use_v2:
            helion_time = do_bench(lambda: grouped_gemm_v2(group_A, group_B))
        else:
            helion_time = do_bench(
                lambda: grouped_gemm_simple(
                    group_A, group_B, use_concatenated=use_concatenated
                )
            )

        reference_time = do_bench(lambda: grouped_gemm_reference(group_A, group_B))

        print(f"Helion time: {helion_time:.4f}ms")
        print(f"Reference time: {reference_time:.4f}ms")
        print(f"Speedup: {reference_time / helion_time:.2f}x")


def main() -> None:
    # Test with various group sizes
    test_cases = [
        # Small groups
        [(128, 128, 64), (256, 256, 128), (64, 64, 32)],
        # Mixed sizes
        [(512, 1024, 256), (128, 512, 128), (256, 256, 64), (64, 128, 32)],
        # Larger groups
        [(1024, 1024, 512), (512, 512, 256)],
    ]

    for i, group_sizes in enumerate(test_cases):
        print(f"\n=== Test Case {i+1}: {len(group_sizes)} groups ===")
        print(f"Group sizes: {group_sizes}")
        # Test the simpler v2 implementation (individual kernels)
        check(group_sizes, use_v2=True)


if __name__ == "__main__":
    main()
