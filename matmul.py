from __future__ import annotations

import torch

batch_size = 32
m = 64
k = 128
n = 16
device = "cuda"

# mat1: [batch_size, k] -> let's call this [m, k] for consistency
mat1 = torch.randn([m, k], device=device, dtype=torch.float16, requires_grad=True)
# mat2: [k, n]
mat2 = torch.randn([k, n], device=device, dtype=torch.float16, requires_grad=True)


def func(mat1, mat2):
    return torch.matmul(mat1, mat2)


compiled_fn = torch.compile(func)

# Forward pass
out = compiled_fn(mat1, mat2)
print(f"Matmul output shape: {out.shape}")

# Backward pass
out.sum().backward()
print(f"Mat1 gradient shape: {mat1.grad.shape}")
print(f"Mat2 gradient shape: {mat2.grad.shape}")
