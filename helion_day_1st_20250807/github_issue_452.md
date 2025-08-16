# Issue #452: reshape leads to dynamic shape issue

## Metadata
- **State**: OPEN
- **Author**: [BoyuanFeng](https://github.com/BoyuanFeng)
- **Created**: August 07, 2025 at 22:07 UTC
- **Updated**: August 12, 2025 at 18:18 UTC
- **Assignees**: [yf225](https://github.com/yf225)

## Description

I tried to write a matmul in helion `C=AxB` where
```
    A: (M, K) bf16
    B: (K, N) int4. assume b is packed with 2 `int4` elements per K. i.e., it's a
        (K//2)xNx(2xint4) matrix, represented in Triton as (K//2)xNxi8.
    C: (M, N) bf16
```

However, `b_bf16.reshape([BLOCK_SIZE_K, BLOCK_SIZE_N])` leads to a dynamic shape issue.

Triton Reference Implementation: [code](https://gist.github.com/jlebar/3435b2c00deea53258887ce37231e5e2)

Helion Repro:
```python
import torch
from torch import Tensor
import helion
import helion.language as hl


@helion.kernel(use_default_config=True, static_shapes=True)
def matmul_bf16_int4(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    """
    A: (M, K) bf16
    B: (K, N) int4. assume b is packed with 2 `int4` elements per K. i.e., it's a
        (K//2)xNx(2xint4) matrix, represented in Triton as (K//2)xNxi8.
    C: (M, N) bf16
    """
    M, K = A.shape
    _, N = B.shape

    BLOCK_SIZE_N = hl.register_block_size(N)
    BLOCK_SIZE_K = hl.register_block_size(K)

    # Use Helion to tile the computation
    for tile_m in hl.tile(M):
        for tile_n in hl.tile(N, block_size=BLOCK_SIZE_N):
            acc = hl.zeros((tile_m, tile_n), dtype=torch.bfloat16)

            for tile_k in hl.tile(K, block_size=BLOCK_SIZE_K):
                # hl.load()
                tile_k_begin = tile_k.begin
                tile_k_end = tile_k.end
                b_tile = B[tile_k_begin//2:tile_k_end//2, tile_n] # [BLOCK_SIZE_K//2, BLOCK_SIZE_N]
                _4_i8 = hl.full((1, ), 4, dtype=torch.int8)
                b_lo = (b_tile << _4_i8) >> _4_i8
                b_hi = b_tile >> _4_i8
                b_bf16 = torch.stack((b_lo.to(torch.float16), b_hi.to(torch.float16)), dim=2) # [BLOCK_SIZE_K//2, BLOCK_SIZE_N, 2]
                b_bf16 = b_bf16.permute(0, 2, 1) # [BLOCK_SIZE_K//2, 2, BLOCK_SIZE_N]
                b_bf16 = b_bf16.reshape([BLOCK_SIZE_K, BLOCK_SIZE_N]) # [BLOCK_SIZE_K, BLOCK_SIZE_N]
                acc += hl.dot(A[tile_m, tile_k], b_bf16) # [BLOCK_SIZE_M, BLOCK_SIZE_N]

            C[tile_m, tile_n] = acc


# Test the kernel
A = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
B = torch.randint(0, 16, (4096, 8192), dtype=torch.int8, device="cuda")
C = torch.randn(8192, 8192, dtype=torch.float32, device="cuda")
matmul_bf16_int4(A, B, C)
```


Error:
```
Traceback (most recent call last):
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 616, in propagate_call
    _CheckForIndexCalls.retry_call(fn, proxy_args, proxy_kwargs), origin
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/language/tile_proxy.py", line 123, in retry_call
    return fn(*proxy_args, **proxy_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[u1, u0]' is invalid for input of size 2*u0*u5

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/users/boyuan/playground/helion_example.py", line 46, in <module>
    matmul_bf16_int4(A, B, C)
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
           ^^^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/runtime/kernel.py", line 158, in bind
    bound_kernel = BoundKernel(self, args)
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/runtime/kernel.py", line 338, in __init__
    self.host_function: HostFunction = HostFunction(
                                       ^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/host_function.py", line 108, in __init__
    propagate_types(self, fake_args)
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 2251, in propagate_types
    prop.visit(stmt)
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
                ^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 2086, in visit_For
    body = self._loop_body(node.body)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 2050, in _loop_body
    self.visit(stmt)
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
                ^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 2086, in visit_For
    body = self._loop_body(node.body)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 2050, in _loop_body
    self.visit(stmt)
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
                ^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 2086, in visit_For
    body = self._loop_body(node.body)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 2050, in _loop_body
    self.visit(stmt)
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
                ^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 1974, in visit_Assign
    type_info = self.visit(node.value)
                ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
                ^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 1916, in visit_Call
    return func.propagate_call(tuple(args), kwargs, self.origin())  # pyright: ignore[reportReturnType]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/boyuan/.conda/envs/pytorch-nightly/lib/python3.12/site-packages/helion/_compiler/type_propagation.py", line 622, in propagate_call
    raise exc.TorchOpTracingError(e) from e
helion.exc.TorchOpTracingError: RuntimeError: shape '[u1, u0]' is invalid for input of size 2*u0*u5
While processing:
  File "/data/users/boyuan/playground/helion_example.py", line 36, in matmul_bf16_int4
    b_bf16 = b_bf16.reshape([BLOCK_SIZE_K, BLOCK_SIZE_N]) # [BLOCK_SIZE_K, BLOCK_SIZE_N]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```


## Comments

*No comments yet.*
