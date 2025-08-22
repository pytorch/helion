import torch
import helion
import helion.language as hl

@helion.kernel(use_default_config=True)
def test_matmul_4d_symbolic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Test 4D matmul with symbolic shapes"""
    # a: [B, H, I, D]
    # b: [B, H, D, J]
    # result: [B, H, I, J]
    return torch.matmul(a, b)

def main():
    # Test with concrete shapes
    B, H, I, D, J = 2, 4, 8, 16, 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    a = torch.randn(B, H, I, D, device=device, dtype=torch.float32)
    b = torch.randn(B, H, D, J, device=device, dtype=torch.float32)
    
    # Reference result
    ref_result = torch.matmul(a, b)
    
    # Helion result
    helion_result = test_matmul_4d_symbolic(a, b)
    
    # Compare
    max_diff = torch.max(torch.abs(helion_result - ref_result)).item()
    print(f"Max difference: {max_diff}")
    print(f"Shapes - Input A: {a.shape}, Input B: {b.shape}, Output: {helion_result.shape}")
    
    if max_diff < 1e-5:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")
        
    # Test with dynamic shapes (this will trigger symbolic shape handling)
    print("\nTesting with dynamic shapes...")
    # Create tensors with dynamic batch size
    a_dyn = torch.randn(B + 1, H, I, D, device=device, dtype=torch.float32)
    b_dyn = torch.randn(B + 1, H, D, J, device=device, dtype=torch.float32)
    
    ref_result_dyn = torch.matmul(a_dyn, b_dyn)
    helion_result_dyn = test_matmul_4d_symbolic(a_dyn, b_dyn)
    
    max_diff_dyn = torch.max(torch.abs(helion_result_dyn - ref_result_dyn)).item()
    print(f"Max difference (dynamic): {max_diff_dyn}")
    print(f"Shapes - Input A: {a_dyn.shape}, Input B: {b_dyn.shape}, Output: {helion_result_dyn.shape}")
    
    if max_diff_dyn < 1e-5:
        print("✅ Dynamic test passed!")
    else:
        print("❌ Dynamic test failed!")

if __name__ == "__main__":
    main()