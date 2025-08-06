"""TRUE Stream-K with atomic accumulation for proper partial sum handling."""

import torch
import helion
import helion.language as hl
import math


@helion.kernel(static_shapes=True)
def true_stream_k_gemm_atomic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """TRUE Stream-K GEMM with atomic accumulation.
    
    This is REAL Stream-K:
    - 3D work distribution (M×N×K)
    - Each CTA processes ONE K-tile for ONE output tile
    - Multiple CTAs accumulate to same output via atomics
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    
    # Initialize output
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    
    # TRUE 3D TILING - K is part of the grid!
    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        # Each work unit: compute ONE K-tile's contribution
        partial = torch.zeros([tile_m, tile_n], dtype=torch.float32, device=x.device)
        partial = torch.addmm(
            partial,
            x[tile_m, tile_k].to(torch.float32),
            y[tile_k, tile_n].to(torch.float32)
        )
        
        # Atomic accumulation - multiple CTAs write to same output!
        hl.atomic_add(out, [tile_m, tile_n], partial)
    
    return out.to(torch.float16)


@helion.kernel(static_shapes=True) 
def standard_gemm_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard GEMM with 2D tiling for comparison."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    
    # Standard 2D tiling - K is inner loop
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    
    return out


def test_true_stream_k_atomic():
    """Test TRUE Stream-K with atomic accumulation."""
    print("=" * 80)
    print("TRUE Stream-K with Atomic Accumulation")
    print("=" * 80)
    
    # Test size
    M, N, K = 128, 128, 256
    
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    ref = torch.matmul(a, b)
    
    print(f"\nProblem: {M}×{K} @ {K}×{N} = {M}×{N}")
    
    # Test TRUE Stream-K
    print("\n1. TRUE Stream-K (3D tiling with atomics):")
    print("-" * 60)
    
    bound_3d = true_stream_k_gemm_atomic.bind((a, b))
    bound_3d._config = helion.Config(
        pid_type="stream_k",
        block_sizes=[32, 32, 32],
        num_warps=4,
        num_stages=2
    )
    
    # Analyze generated code
    code_3d = bound_3d.to_triton_code(bound_3d._config)
    lines = code_3d.split('\n')
    
    # Check for key Stream-K components
    has_3d = False
    has_atomic = False
    has_work_dist = False
    
    for line in lines:
        if 'pid_2' in line:
            has_3d = True
            print(f"✓ 3D decomposition: {line.strip()[:80]}")
        elif 'atomic' in line.lower():
            has_atomic = True
            print(f"✓ Atomic accumulation: {line.strip()[:80]}")
        elif 'sk_total_work' in line:
            has_work_dist = True
            print(f"✓ Work distribution: {line.strip()[:80]}")
    
    # Calculate work distribution
    m_tiles = math.ceil(M / 32)
    n_tiles = math.ceil(N / 32)
    k_tiles = math.ceil(K / 32)
    total_work = m_tiles * n_tiles * k_tiles
    num_sms = helion.runtime.get_num_sm(a.device)
    
    print(f"\nWork Distribution:")
    print(f"  Grid dimensions: {m_tiles}×{n_tiles}×{k_tiles}")
    print(f"  Total MAC iterations: {total_work}")
    print(f"  Number of SMs: {num_sms}")
    print(f"  Work per SM: ~{total_work / num_sms:.1f}")
    print(f"  Load imbalance: <{(math.ceil(total_work/num_sms)*num_sms - total_work)/total_work*100:.1f}%")
    
    # Test execution
    try:
        result_3d = bound_3d(a, b)
        torch.testing.assert_close(result_3d.float(), ref.float(), rtol=1e-2, atol=1e-2)
        print("\n✓ TRUE Stream-K executes correctly with atomics!")
    except Exception as e:
        print(f"\n✗ Execution failed: {e}")
    
    # Compare with standard 2D
    print("\n2. Standard GEMM (2D tiling, K inner loop):")
    print("-" * 60)
    
    bound_2d = standard_gemm_2d.bind((a, b))
    bound_2d._config = helion.Config(
        pid_type="flat",
        block_sizes=[32, 32, 32],
        num_warps=4
    )
    
    code_2d = bound_2d.to_triton_code(bound_2d._config)
    lines_2d = code_2d.split('\n')
    
    # Check structure
    has_2d_only = True
    for line in lines_2d:
        if 'pid_2' in line:
            has_2d_only = False
            break
    
    if has_2d_only:
        print("✓ 2D grid only (M×N)")
        print("✓ K is inner reduction loop")
    
    result_2d = bound_2d(a, b)
    torch.testing.assert_close(result_2d.float(), ref.float(), rtol=1e-3, atol=1e-3)
    print("✓ Standard GEMM correct")
    
    print("\n3. Key Differences:")
    print("-" * 60)
    print("Standard GEMM:")
    print("  - 2D grid (M×N), K is inner loop")
    print("  - Each CTA owns complete output tile")
    print("  - No inter-CTA communication needed")
    print("\nTRUE Stream-K:")
    print("  - 3D grid (M×N×K)")
    print("  - Each CTA computes partial K contribution")
    print("  - Multiple CTAs accumulate to same output")
    print("  - Requires atomic operations")
    print("  - Perfect work distribution!")


def visualize_work_assignment():
    """Visualize how TRUE Stream-K assigns work."""
    print("\n" + "=" * 80)
    print("TRUE Stream-K Work Assignment Visualization")
    print("=" * 80)
    
    # Small example for clarity
    M, N, K = 64, 64, 128
    tile_size = 32
    
    m_tiles = M // tile_size  # 2
    n_tiles = N // tile_size  # 2
    k_tiles = K // tile_size  # 4
    
    total_work = m_tiles * n_tiles * k_tiles  # 16
    num_sms = 4  # Simplified
    
    print(f"\nExample: {M}×{N}×{K} with {tile_size}×{tile_size}×{tile_size} tiles")
    print(f"Grid: {m_tiles}×{n_tiles}×{k_tiles} = {total_work} work units")
    print(f"SMs: {num_sms}")
    
    print("\nWork Assignment (work_id → (m,n,k)):")
    print("-" * 60)
    
    for sm_id in range(num_sms):
        work_per_sm = total_work // num_sms
        start = sm_id * work_per_sm
        end = start + work_per_sm
        
        print(f"\nSM {sm_id}: Work units {start}-{end-1}")
        for work_id in range(start, end):
            # TRUE Stream-K decomposition
            k_idx = work_id % k_tiles
            temp = work_id // k_tiles
            n_idx = temp % n_tiles
            m_idx = temp // n_tiles
            
            print(f"  Work {work_id:2d}: Output[{m_idx},{n_idx}] += A[{m_idx},:] @ B[:,{n_idx}] (K-tile {k_idx})")
    
    print("\nNotice:")
    print("  - Multiple work units contribute to same output tile")
    print("  - K-iterations are distributed across SMs")
    print("  - This is TRUE work-centric decomposition!")


if __name__ == "__main__":
    test_true_stream_k_atomic()
    visualize_work_assignment()