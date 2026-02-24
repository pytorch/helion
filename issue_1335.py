# Autotuning with tensors >= 2^32 errors when block pointers are used


import torch
torch.set_default_device('cuda')
from typing import Union
import torch
import helion
import helion.language as hl
from triton.testing import do_bench

@helion.kernel(static_shapes=True)
def f1(x: torch.Tensor, batch_shape: torch.Tensor) -> torch.Tensor:
    out = x.new_empty(x.shape)
    for b,  in hl.tile([batch_shape]):
        cur = x[b, :].to(torch.float32)
        cur = torch.softmax(cur, dim=-1)
        out[b, :] = cur
    return out
# for num_sm_multiplier in [1, 2, 4, 8, 16, 32, 64, 48, 96, 128, 192, 256, 512, 1024, 2048]:
for num_sm_multiplier in [1]:
    B = 2**20
    D = 2**12
    x = torch.randn(B, D)
    B_real = B // 2
    for B_real in [2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20]:
        batch_shape = torch.tensor(B_real, dtype=torch.int32)
        ms = do_bench(lambda: f1(x, batch_shape))
        ms = do_bench(lambda: f1(x, batch_shape))
        print(num_sm_multiplier, B_real)
        print(ms)
        print((1e3/ms) * B_real * D * x.dtype.itemsize * 2 / 1e9)
        print()
