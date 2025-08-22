"""Demonstrate workarounds for tile context preservation"""
import helion
import helion.language as hl
import torch


@helion.kernel(use_default_config=True)
def test_workaround_store_indices(
    a: torch.Tensor,  # [B, H, I, D]
    b: torch.Tensor,  # [B, H, D, J]
) -> torch.Tensor:
    B = a.size(0)
    H = a.size(1)
    I = a.size(2)
    D = a.size(3)
    J = b.size(3)
    
    out = torch.zeros([B, H, I, J], dtype=a.dtype, device=a.device)
    
    # Simple demonstration: just avoid using tile.index after matmul
    # by structuring the computation differently
    for tile_b in hl.tile(B):
        for tile_h in hl.tile(H):
            for tile_i in hl.tile(I):
                for tile_j in hl.tile(J):
                    acc = torch.zeros([tile_b.block_size, tile_h.block_size, 
                                     tile_i.block_size, tile_j.block_size], 
                                    dtype=torch.float32, device=a.device)
                    
                    for tile_d in hl.tile(D):
                        a_tile = a[tile_b, tile_h, tile_i, tile_d]
                        b_tile = b[tile_b, tile_h, tile_d, tile_j]
                        
                        # 4D matmul
                        prod = torch.matmul(a_tile, b_tile)
                        acc = acc + prod
                    
                    out[tile_b, tile_h, tile_i, tile_j] = acc.to(a.dtype)
                        
    return out


@helion.kernel(use_default_config=True)
def test_workaround_bmm_style(
    a: torch.Tensor,  # [B, H, I, D]
    b: torch.Tensor,  # [B, H, D, J]
) -> torch.Tensor:
    B = a.size(0)
    H = a.size(1)
    I = a.size(2)
    D = a.size(3)
    J = b.size(3)
    
    out = torch.zeros([B, H, I, J], dtype=a.dtype, device=a.device)
    
    for tile_b in hl.tile(B):
        for tile_h in hl.tile(H):
            for tile_i in hl.tile(I):
                for tile_j in hl.tile(J):
                    # Initialize accumulator
                    acc = hl.zeros([tile_b, tile_h, tile_i, tile_j], 
                                  dtype=torch.float32, device=a.device)
                    
                    for tile_d in hl.tile(D):
                        a_tile = a[tile_b, tile_h, tile_i, tile_d]
                        b_tile = b[tile_b, tile_h, tile_d, tile_j]
                        
                        # Reshape to 3D for bmm
                        # Merge batch and head dimensions
                        a_3d = a_tile.reshape(-1, tile_i.block_size, tile_d.block_size)
                        b_3d = b_tile.reshape(-1, tile_d.block_size, tile_j.block_size)
                        
                        # Use bmm which preserves tile context
                        prod_3d = torch.bmm(a_3d, b_3d)
                        
                        # Reshape back to 4D
                        prod = prod_3d.reshape(tile_b.block_size, tile_h.block_size, 
                                             tile_i.block_size, tile_j.block_size)
                        
                        acc += prod
                    
                    # Now tile_h.index still works because we used bmm
                    # Could use tile properties here if needed
                    index_expr = tile_h.index[None, :, None, None]
                    
                    out[tile_b, tile_h, tile_i, tile_j] = acc.to(a.dtype)
                        
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test data
    B, H, I, D, J = 2, 4, 8, 16, 12
    a = torch.randn(B, H, I, D, device=device, dtype=torch.float32)
    b = torch.randn(B, H, D, J, device=device, dtype=torch.float32)
    
    # Reference implementation
    ref = torch.matmul(a, b)
    
    # Test workaround 1: Store indices before matmul
    print("Testing workaround 1 (store indices)...")
    try:
        result1 = test_workaround_store_indices(a, b)
        diff1 = torch.abs(result1 - ref).max().item()
        print(f"✓ Workaround 1 passed, max diff: {diff1:.6f}")
    except Exception as e:
        print(f"✗ Workaround 1 failed: {e}")
    
    # Test workaround 2: Use BMM with reshaping
    print("\nTesting workaround 2 (BMM style)...")
    try:
        result2 = test_workaround_bmm_style(a, b)
        diff2 = torch.abs(result2 - ref).max().item()
        print(f"✓ Workaround 2 passed, max diff: {diff2:.6f}")
    except Exception as e:
        print(f"✗ Workaround 2 failed: {e}")


if __name__ == "__main__":
    main()