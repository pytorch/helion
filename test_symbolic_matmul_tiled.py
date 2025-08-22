import torch
import helion
import helion.language as hl

@helion.kernel(use_default_config=True)
def test_matmul_4d_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Test 4D matmul with symbolic shapes inside tiles"""
    # a: [B, H, I, D]
    # b: [B, H, D, J]
    # result: [B, H, I, J]
    B, H, I, D = a.shape
    _, _, _, J = b.shape
    
    result = torch.zeros([B, H, I, J], dtype=a.dtype, device=a.device)
    
    # Tile over batch dimension
    for tile_b in hl.tile(B):
        # Tile over head dimension
        for tile_h in hl.tile(H):
            # Tile over output dimensions I and J
            for tile_i in hl.tile(I):
                for tile_j in hl.tile(J):
                    # Initialize accumulator
                    acc = hl.zeros([tile_b, tile_h, tile_i, tile_j], dtype=torch.float32, device=a.device)
                    
                    # Tile over reduction dimension D
                    for tile_d in hl.tile(D):
                        # Load tiles from a and b
                        a_tile = hl.load(a, [tile_b.index, tile_h.index, tile_i.index, tile_d.index])
                        b_tile = hl.load(b, [tile_b.index, tile_h.index, tile_d.index, tile_j.index])
                        
                        # Perform matrix multiplication
                        # This should trigger our 4D matmul handler
                        prod = torch.matmul(a_tile, b_tile)
                        acc += prod
                    
                    # Store result
                    hl.store(result, [tile_b.index, tile_h.index, tile_i.index, tile_j.index], acc.to(a.dtype))
    
    return result

def main():
    # Test with small concrete shapes
    B, H, I, D, J = 2, 3, 4, 5, 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    a = torch.randn(B, H, I, D, device=device, dtype=torch.float32)
    b = torch.randn(B, H, D, J, device=device, dtype=torch.float32)
    
    # Reference result
    ref_result = torch.matmul(a, b)
    
    # Helion result
    try:
        helion_result = test_matmul_4d_tiled(a, b)
        
        # Compare
        max_diff = torch.max(torch.abs(helion_result - ref_result)).item()
        print(f"Max difference: {max_diff}")
        print(f"Shapes - Input A: {a.shape}, Input B: {b.shape}, Output: {helion_result.shape}")
        
        if max_diff < 1e-5:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
            print(f"Reference result sample:\n{ref_result[0, 0, :2, :2]}")
            print(f"Helion result sample:\n{helion_result[0, 0, :2, :2]}")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()