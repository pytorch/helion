# Issue #450: Crash if you @helion.kernel() a function without hl.tile

## Metadata
- **State**: OPEN
- **Author**: [xmfan](https://github.com/xmfan)
- **Created**: August 07, 2025 at 21:46 UTC
- **Updated**: August 07, 2025 at 21:47 UTC

## Description

Full error message
```python
(/home/xmfan/core/a/pytorch-env) [14:43:31] ~/core/a/helion (main) > python examples/bf16_add.py 
WARNING[TensorOperationInWrapper]: A tensor operation outside of the `hl.tile` or `hl.grid` loop will not be fused in the generated kernel.
Use @helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper]) to suppress this warning.
If this is not a tensor operation, please report this as a bug.
While processing:
  File "/home/xmfan/core/a/helion/examples/bf16_add.py", line 10, in bf16_add
    return x + y
           ^^^^^

[0s] Starting DifferentialEvolutionSearch with population=40, generations=20, crossover_rate=0.8
Traceback (most recent call last):
  File "/home/xmfan/core/a/helion/examples/bf16_add.py", line 22, in <module>
    main()
  File "/home/xmfan/core/a/helion/examples/bf16_add.py", line 19, in main
    check(1024, 1024)
  File "/home/xmfan/core/a/helion/examples/bf16_add.py", line 15, in check
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
  File "/home/xmfan/core/a/helion/helion/autotuner/base_search.py", line 211, in parallel_benchmark
    fns = [self.kernel.compile_config(c, allow_print=False) for c in configs]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/runtime/kernel.py", line 418, in compile_config
    module = PyCodeCache.load(triton_code)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/pytorch/torch/_inductor/codecache.py", line 3278, in load
    return cls.load_by_key_path(key, path)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/pytorch/torch/_inductor/codecache.py", line 3296, in load_by_key_path
    mod = _reload_python_module(key, path, set_sys_modules=in_toplevel)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/pytorch/torch/_inductor/runtime/compile_tasks.py", line 25, in _reload_python_module
    raise RuntimeError(
RuntimeError: Failed to import /tmp/torchinductor_xmfan/cz/cczmd5i4oivoyiolsj2nrhr7wofejf6cas2iugmvo5w6lx7a4hpx.py
IndentationError: expected an indented block after function definition on line 8 (cczmd5i4oivoyiolsj2nrhr7wofejf6cas2iugmvo5w6lx7a4hpx.py, line 10)
(/home/xmfan/core/a/pytorch-env) [14:45:45] ~/core/a/helion (main) > 
```

/tmp/torchinductor_xmfan/cz/cczmd5i4oivoyiolsj2nrhr7wofejf6cas2iugmvo5w6lx7a4hpx.py Codegen:
```python
from __future__ import annotations

import torch
import triton
from helion.runtime import default_launcher as _default_launcher

@triton.jit
def _bf16_add_kernel():

def bf16_add(x: torch.Tensor, y: torch.Tensor, *, _launcher=_default_launcher):
    return x + y
```

Repro
```python
import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def bf16_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

def check(m: int, n: int):
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    run_example(bf16_add, torch.add, (x, y))

def main():
    check(1024, 1024)

if __name__ == "__main__":
    main()
```

## Comments

*No comments yet.*
