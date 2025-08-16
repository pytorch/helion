# Issue #454: Better error message for no autotuning options found

## Metadata
- **State**: OPEN
- **Author**: [xmfan](https://github.com/xmfan)
- **Created**: August 07, 2025 at 22:28 UTC
- **Updated**: August 09, 2025 at 16:43 UTC

## Description

Hitting OOB after https://github.com/pytorch/helion/pull/453

```python
(/home/xmfan/core/a/pytorch-env) [15:17:07] ~/core/a/helion (main) > python examples/bf16_add.py 
[0s] Starting DifferentialEvolutionSearch with population=40, generations=20, crossover_rate=0.8
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
  File "/home/xmfan/core/a/helion/helion/autotuner/base_search.py", line 244, in autotune
    best = self._autotune()
           ^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/autotuner/differential_evolution.py", line 97, in _autotune
    self.initial_two_generations()
  File "/home/xmfan/core/a/helion/helion/autotuner/differential_evolution.py", line 64, in initial_two_generations
    self.log(
  File "/home/xmfan/core/a/helion/helion/autotuner/logger.py", line 49, in __call__
    self._logger.log(level, " ".join(map(_maybe_call, msg)))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/autotuner/logger.py", line 82, in _maybe_call
    return fn()
           ^^^^
  File "/home/xmfan/core/a/helion/helion/autotuner/differential_evolution.py", line 66, in <lambda>
    lambda: population_statistics(oversized_population),
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xmfan/core/a/helion/helion/autotuner/base_search.py", line 391, in population_statistics
    f"min={working[0].perf:.4f} "
           ~~~~~~~^^^
IndexError: list index out of range
```

sample that has no good options at the time of writing:
```python
import torch

import helion
# from helion._testing import run_example
import helion.language as hl


@helion.kernel(autotune_precompile=False)
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
    breakpoint()
    check(102400, 102400)

if __name__ == "__main__":
    main()
```

## Comments

### Comment 1 by [jansel](https://github.com/jansel)
*Posted on August 09, 2025 at 16:01 UTC*

cc @oulgen did your PR fix this one?

---

### Comment 2 by [oulgen](https://github.com/oulgen)
*Posted on August 09, 2025 at 16:43 UTC*

> cc [@oulgen](https://github.com/oulgen) did your PR fix this one?

My PR adds better error message but there’s a bit more to do for this example, like restricting the block sizes etc.

---
