import torch
import torch.nested

# Create a nested tensor similar to what tritonbench would create
B, M = 8, 16
device = "cuda"

# Create random sequence lengths
lengths = torch.randint(1, 32, (B,), device=device)
x_offsets = torch.cat(
    [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)]
)
total_elements = int(x_offsets[-1])

# Create random values
x_values = torch.randn(total_elements, M, dtype=torch.float32, device=device)

# Create a nested tensor with jagged layout
nested_x = torch.nested.nested_tensor(
    [x_values[x_offsets[i]:x_offsets[i+1]] for i in range(B)],
    device=device,
    dtype=torch.float32,
    layout=torch.jagged
)

try:
    print(f"Nested tensor shape: {nested_x.shape}")
except:
    print("Can't get shape of nested tensor")
    
print(f"Values shape: {nested_x.values().shape}")
print(f"Offsets: {nested_x.offsets()}")
print(f"Has values() method: {hasattr(nested_x, 'values')}")
print(f"Has offsets() method: {hasattr(nested_x, 'offsets')}")

# Test the kernel
from examples.jagged_softmax import jagged_softmax_tritonbench

try:
    result = jagged_softmax_tritonbench(nested_x, B, M, 32, 0.5)
    print(f"Result shape: {result.shape}")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()