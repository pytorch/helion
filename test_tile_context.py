"""Test how tile context is preserved through operations"""
import helion
import helion.language as hl
import torch

@helion.kernel(use_default_config=True)
def test_tile_preservation_bmm(
    a: torch.Tensor,  # [N, D]
    b: torch.Tensor,  # [N, D]
) -> torch.Tensor:
    N = a.size(0)
    D = a.size(1)
    
    out = torch.zeros([N, N], dtype=a.dtype, device=a.device)
    
    # Test with bmm
    for tile_i in hl.tile(N):
        for tile_j in hl.tile(N):
            for tile_d in hl.tile(D):
                a_tile = a[tile_i, tile_d]  # [tile_i, tile_d]
                b_tile = b[tile_j, tile_d]  # [tile_j, tile_d]
                
                # Reshape for bmm: add batch dimension
                a_reshaped = a_tile.reshape(1, tile_i.block_size, tile_d.block_size)
                b_reshaped = b_tile.reshape(1, tile_d.block_size, tile_j.block_size).transpose(-2, -1)
                
                # Use bmm
                prod = torch.bmm(a_reshaped, b_reshaped).reshape(tile_i.block_size, tile_j.block_size)
                
                # This should work - tile_i should still have its block_id
                out[tile_i, tile_j] += prod
                
    return out


@helion.kernel(use_default_config=True)
def test_tile_preservation_matmul(
    a: torch.Tensor,  # [N, D]
    b: torch.Tensor,  # [N, D]
) -> torch.Tensor:
    N = a.size(0)
    D = a.size(1)
    
    out = torch.zeros([N, N], dtype=a.dtype, device=a.device)
    
    # Test with matmul
    for tile_i in hl.tile(N):
        for tile_j in hl.tile(N):
            for tile_d in hl.tile(D):
                a_tile = a[tile_i, tile_d]  # [tile_i, tile_d]
                b_tile = b[tile_j, tile_d]  # [tile_j, tile_d]
                
                # Use matmul directly
                prod = torch.matmul(a_tile, b_tile.transpose(-2, -1))
                
                # This might fail - tile_i might lose its block_id
                out[tile_i, tile_j] += prod
                
    return out


@helion.kernel(use_default_config=True)
def test_tile_index_after_matmul(
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
                    for tile_d in hl.tile(D):
                        a_tile = a[tile_b, tile_h, tile_i, tile_d]
                        b_tile = b[tile_b, tile_h, tile_d, tile_j]
                        
                        # 4D matmul
                        prod = torch.matmul(a_tile, b_tile)
                        
                        # Try to use tile_h.index after matmul - this is where the error occurs
                        index_expr = tile_h.index[None, :, None, None]
                        
                        out[tile_b, tile_h, tile_i, tile_j] += prod
                        
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test 1: BMM preserves tile context
    print("Testing BMM...")
    a = torch.randn(64, 32, device=device)
    b = torch.randn(64, 32, device=device)
    try:
        result = test_tile_preservation_bmm(a, b)
        print("✓ BMM test passed")
    except Exception as e:
        print(f"✗ BMM test failed: {e}")
    
    # Test 2: Matmul loses tile context
    print("\nTesting Matmul...")
    try:
        result = test_tile_preservation_matmul(a, b)
        print("✓ Matmul test passed")
    except Exception as e:
        print(f"✗ Matmul test failed: {e}")
    
    # Test 3: Tile index after 4D matmul
    print("\nTesting tile.index after 4D matmul...")
    a_4d = torch.randn(2, 4, 8, 16, device=device)
    b_4d = torch.randn(2, 4, 16, 12, device=device)
    try:
        result = test_tile_index_after_matmul(a_4d, b_4d)
        print("✓ Tile index after matmul test passed")
    except Exception as e:
        print(f"✗ Tile index after matmul test failed: {e}")
        

if __name__ == "__main__":
    main()