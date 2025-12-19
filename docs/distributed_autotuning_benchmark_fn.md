# Distributed Autotuning via `autotune_benchmark_fn`

## Proposed Approach

For distributed kernel autotuning:
1. `autotune_precompile = None` (disable precompile phase)
2. Benchmark phase runs sequentially
3. Each benchmark is a spawn-based multi-GPU run with timeout monitoring
4. All of this is encapsulated in user's `autotune_benchmark_fn`

## Constraint: Signature Must Not Change

The existing `autotune_benchmark_fn` signature must be preserved:

```python
def autotune_benchmark_fn(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None
) -> list[float]
```

## Solution: Callables with Attributes

Each callable in `fns` can have `.kernel` and `.config` attributes attached:

```python
class BenchmarkCallable:
    """A callable that carries kernel and config as attributes."""
    def __init__(self, kernel, config):
        self.kernel = kernel
        self.config = config

    def __call__(self):
        # Default behavior for non-distributed (calls compiled kernel)
        # For distributed, user's benchmark_fn won't call this directly
        compiled = self.kernel.compile_config(self.config)
        return compiled(*self.kernel.args)  # or however args are accessed
```

Or simpler, just attach attributes to a function:

```python
def make_benchmark_callable(kernel, config):
    def fn():
        compiled = kernel.compile_config(config)
        # ... default benchmark behavior
    fn.kernel = kernel
    fn.config = config
    return fn
```

## User's `autotune_benchmark_fn` with Captured Input Factory

The user captures their input factory in a closure:

```python
def my_input_factory(rank, world_size):
    """Creates inputs for this rank. Called in each subprocess."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    # Distributed setup
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

    # Create inputs
    N, D = 128, 4096
    x = torch.randn(N, D, device=f"cuda:{rank}")
    symm_mem_buffer = symm_mem.empty(N, D, device=f"cuda:{rank}")
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD.group_name)
    bias = torch.randn(D, device=f"cuda:{rank}")
    weight = torch.randn(D, device=f"cuda:{rank}")

    return (
        x, symm_mem_buffer, bias, weight,
        symm_mem_hdl.signal_pad_ptrs_dev,
        1e-5,  # eps
        rank,
        world_size,
        dist.group.WORLD.group_name,
    )


def make_distributed_benchmark(input_factory, world_size=4, timeout=30):
    """
    Factory that returns an autotune_benchmark_fn with captured settings.

    The returned function has the SAME SIGNATURE as existing autotune_benchmark_fn.
    """

    def benchmark_fn(fns, *, repeat=1, desc=None):
        """
        fns: list[Callable] where each callable has .kernel and .config attributes
        Returns: list[float] timings in ms
        """
        timings = []
        for fn in fns:
            # Extract kernel and config from callable's attributes
            kernel = fn.kernel
            config = fn.config

            # Spawn multi-GPU processes with timeout
            timing = spawn_and_benchmark(
                kernel,
                config,
                input_factory,  # captured in closure
                world_size,     # captured in closure
                timeout         # captured in closure
            )
            timings.append(timing)
        return timings

    return benchmark_fn


def spawn_and_benchmark(kernel, config, input_factory, world_size, timeout):
    """
    Spawns multi-GPU processes to benchmark kernel with config.
    Returns timing in ms, or inf on timeout.
    """
    import subprocess
    import tempfile
    import pickle

    # Serialize kernel and config to temp file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump({
            'kernel_module': kernel.__module__,
            'kernel_name': kernel.__name__,
            'config': config.to_dict(),
            'input_factory_module': input_factory.__module__,
            'input_factory_name': input_factory.__name__,
        }, f)
        task_path = f.name

    # Launch with torchrun
    proc = subprocess.Popen(
        [
            "torchrun",
            "--nproc_per_node", str(world_size),
            "--master_port", "29500",
            "_benchmark_worker.py",  # helper script
            task_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        # Parse timing from stdout (e.g., "TIMING:12.345")
        for line in stdout.decode().split('\n'):
            if line.startswith('TIMING:'):
                return float(line.split(':')[1])
        return float('inf')  # No timing found
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return float('inf')
    finally:
        import os
        os.unlink(task_path)
```

## Complete Usage Example

```python
@helion.jit(
    configs=[
        helion.Config(block_sizes=[8], num_warps=8),
        helion.Config(block_sizes=[4], num_warps=4),
        helion.Config(block_sizes=[1], num_warps=8),  # This might hang!
    ],
    autotune_precompile=None,  # Disable precompile phase
    autotune_benchmark_fn=make_distributed_benchmark(
        input_factory=my_input_factory,
        world_size=4,
        timeout=30,
    ),
    static_shapes=True,
)
def one_shot_allreduce_bias_rmsnorm_kernel(
    x: torch.Tensor,
    symm_mem_buffer: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    EPS: hl.constexpr,
    RANK: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
    GROUP_NAME: hl.constexpr,
) -> torch.Tensor:
    ...
```

## What Autotuner Needs to Change

Minimal change - when creating callables for `autotune_benchmark_fn`, attach kernel and config:

```python
# In BaseSearch or wherever fns are created for autotune_benchmark_fn:

def make_benchmark_callable(self, config):
    """Create a callable with kernel and config attributes."""
    fn = self.kernel.compile_config(config)

    # Create wrapper that carries metadata
    def callable_with_attrs():
        return fn(*self.args)

    callable_with_attrs.kernel = self.kernel
    callable_with_attrs.config = config
    return callable_with_attrs
```

## Backward Compatibility

- **Existing users**: Their `autotune_benchmark_fn` just calls `fn()` → still works
- **Distributed users**: Their `autotune_benchmark_fn` accesses `fn.kernel` and `fn.config` → works with new callables
- **Signature unchanged**: `(fns: list[Callable], *, repeat: int, desc: str | None) -> list[float]`

## Summary

| What | How |
|------|-----|
| Signature | **Unchanged** - `(fns, *, repeat, desc) -> list[float]` |
| Kernel/config access | Via `fn.kernel` and `fn.config` attributes on each callable |
| Input factory | Captured in user's closure, not a kernel setting |
| World size, timeout | Captured in user's closure |
| Spawning logic | Fully in user's `autotune_benchmark_fn` |
| Autotuner change | Just attach `.kernel` and `.config` to callables |

## Benefits

1. **No new settings** on kernel decorator
2. **Signature preserved** - backward compatible
3. **User has full control** - capture anything in closure
4. **Minimal autotuner change** - just add attributes to callables
5. **Flexible** - user can customize spawn method, timeout handling, result collection
