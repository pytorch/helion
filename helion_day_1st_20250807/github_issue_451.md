# Issue #451: IMA on add kernel example when bumping input sizes to (102400, 102400)

## Metadata
- **State**: CLOSED
- **Author**: [xmfan](https://github.com/xmfan)
- **Created**: August 07, 2025 at 22:06 UTC
- **Updated**: August 07, 2025 at 22:41 UTC
- **Closed**: August 07, 2025 at 22:41 UTC
- **Assignees**: [oulgen](https://github.com/oulgen)

## Description

```python
(/home/xmfan/core/a/pytorch-env) [15:01:57] ~/core/a/helion (main) > python examples/bf16_add.py 
[0s] Starting DifferentialEvolutionSearch with population=40, generations=20, crossover_rate=0.8
[61s] Timeout after 60s compiling Config(block_sizes=[512, 2048], loop_orders=[[1, 0]], flatten_loops=[True], l2_groupings=[64], range_unroll_factors=[4], range_num_stages=[4], range_multi_buffers=[False], range_flattens=[True], num_warps=32, num_stages=6, indexing='tensor_descriptor', pid_type='persistent_blocked')
[61s] Timeout after 60s compiling Config(block_sizes=[64, 1024], loop_orders=[[1, 0]], flatten_loops=[True], l2_groupings=[16], range_unroll_factors=[0], range_num_stages=[4], range_multi_buffers=[None], range_flattens=[False], num_warps=1, num_stages=5, indexing='pointer', pid_type='persistent_blocked')

[61s] Timeout after 60s compiling Config(block_sizes=[32, 1024], loop_orders=[[1, 0]], flatten_loops=[True], l2_groupings=[8], range_unroll_factors=[0], range_num_stages=[0], range_multi_buffers=[None], range_flattens=[None], num_warps=1, num_stages=3, indexing='block_ptr', pid_type='flat', range_warp_specializes=[])
Traceback (most recent call last):
  File "/home/xmfan/core/a/helion/helion/autotuner/base_search.py", line 135, in benchmark_function
    res = do_bench(
          ^^^^^^^^^
  File "/home/xmfan/core/a/pytorch-env/lib/python3.12/site-packages/triton/testing.py", line 150, in do_bench
    di.synchronize()
  File "/home/xmfan/core/a/pytorch/torch/cuda/__init__.py", line 1085, in synchronize
    return torch._C._cuda_synchronize()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/xmfan/core/a/helion/examples/bf16_add.py", line 28, in <module>
    main()
  File "/home/xmfan/core/a/helion/examples/bf16_add.py", line 25, in main
    check(102400, 102400)
  File "/home/xmfan/core/a/helion/examples/bf16_add.py", line 22, in check
    bf16_add(x, y)
  File "/home/xmfan/core/a/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/runtime/kernel.py", line 581, in __call__
    self.autotune(args)
  File "/home/xmfan/core/a/helion/helion/runtime/kernel.py", line 473, in autotune
    config = self.settings.autotuner_fn(self, args, **kwargs).autotune()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/autotuner/base_cache.py", line 165, in autotune
    config = self.autotuner.autotune()
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/autotuner/base_search.py", line 243, in autotune
    best = self._autotune()
           ^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/autotuner/differential_evolution.py", line 97, in _autotune
    self.initial_two_generations()
  File "/home/xmfan/core/a/helion/helion/autotuner/differential_evolution.py", line 59, in initial_two_generations
    self.parallel_benchmark_flat(
  File "/home/xmfan/core/a/helion/helion/autotuner/base_search.py", line 359, in parallel_benchmark_flat
    to_check, configs, self.parallel_benchmark(configs), strict=True
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/autotuner/base_search.py", line 227, in parallel_benchmark
    results.append((config, self.benchmark_function(config, fn)))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/autotuner/base_search.py", line 150, in benchmark_function
    raise exc.TritonError(f"{type(e).__qualname__}: {e}", config) from e
helion.exc.TritonError: Error running generated Triton program:
Config(block_sizes=[32, 32], loop_orders=[[0, 1]], flatten_loops=[False], l2_groupings=[1], range_unroll_factors=[0], range_num_stages=[0], range_multi_buffers=[None], range_flattens=[None], num_warps=4, num_stages=3, indexing='pointer', pid_type='flat', range_warp_specializes=[])
AcceleratorError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

Repro
```python
import torch

import helion
# from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def bf16_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y  = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]

def check(m: int, n: int):
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    bf16_add(x, y)

def main():
    check(102400, 102400)

if __name__ == "__main__":
    main()
```

## Comments

*No comments yet.*
