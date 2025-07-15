import torch
from examples.jagged_softmax import jagged_softmax_kernel, reference_jagged_softmax_kernel_pytorch

# Simple test case
B, M = 4, 8
device = "cuda"

# Create simple test data
lengths = torch.tensor([3, 2, 4, 1], device=device)
x_offsets = torch.cat(
    [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)]
)
total_elements = int(x_offsets[-1])

# Create test values
x_values = torch.randn(total_elements, M, dtype=torch.float32, device=device)

print(f"Testing with B={B}, M={M}")
print(f"Lengths: {lengths}")
print(f"Offsets: {x_offsets}")
print(f"Values shape: {x_values.shape}")

# Test kernel
result = jagged_softmax_kernel(x_values, x_offsets)
reference = reference_jagged_softmax_kernel_pytorch(x_values, x_offsets)

print(f"\nResult shape: {result.shape}")
print(f"Reference shape: {reference.shape}")

# Check accuracy
torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)
print("\nAccuracy test PASSED!")

# Verify softmax property: sum to 1 along sequence dimension
for i in range(B):
    start = int(x_offsets[i])
    end = int(x_offsets[i + 1])
    if end > start:
        batch_result = result[start:end, :]
        sums = batch_result.sum(dim=0)
        print(f"Batch {i} softmax sums: {sums[:4]}... (should be ~1.0)")