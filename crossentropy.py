from __future__ import annotations

import torch

batch_size = 32
num_features = 64
device = "cuda"

# input_tensor: [batch_size, num_features] - input matrix
input_tensor = torch.randn(
    [batch_size, num_features], device=device, dtype=torch.float32, requires_grad=True
)


def func(input_tensor):
    return torch.sum(input_tensor, dim=-1)  # Sum along the last dimension


compiled_fn = torch.compile(func)

# Forward pass
out = compiled_fn(input_tensor)
print(f"Sum output shape: {out.shape}")
print(f"Sum output mean: {out.mean().item()}")

# Backward pass - sum over output to get scalar for backward
scalar_loss = out.sum()
scalar_loss.backward()
print(f"Input gradient shape: {input_tensor.grad.shape}")
print(f"Input gradient mean: {input_tensor.grad.mean().item()}")
