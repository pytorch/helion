from __future__ import annotations

import torch

batch_size = 32
m = 64
p = 16
device = "cuda"

# input: [batch_size, p]
input = torch.randn(
    [batch_size, p], device=device, dtype=torch.float16, requires_grad=True
)
# mat1: [batch_size, m]
mat1 = torch.randn(
    [batch_size, m], device=device, dtype=torch.float16, requires_grad=True
)
# mat2: [m, p]
mat2 = torch.randn([m, p], device=device, dtype=torch.float16, requires_grad=True)

beta = 1.0
alpha = 1.0


def func(input, mat1, mat2, beta=1.0, alpha=1.0):
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


compiled_fn = torch.compile(func)

out = compiled_fn(input, mat1, mat2, beta, alpha)
out.sum().backward()
