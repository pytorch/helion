# Distributed Kernel Autotuning Design

## Problem Statement

Distributed kernels using `symm_mem_sync` can hang indefinitely with bad configs (e.g., `block_sizes=[1]`). The current autotuner approach has limitations:

1. symm_mem tensors can't be serialized via `torch.save()` - they lose their special allocation status
2. Distributed kernels require coordinated multi-process setup that the autotuner doesn't handle
3. The hacky `_detect_symm_mem_tensors` approach is fragile and doesn't generalize

## Key Difference from Current Autotuner

**Current approach:**
1. Autotuner has the compiled kernel and args
2. Autotuner directly calls `fn(*args)` or serializes/deserializes for spawn mode
3. Problem: symm_mem tensors can't be serialized, and distributed kernels need coordinated multi-process setup

**Proposed approach:**
1. User provides a callable that "wraps" the kernel call
2. Autotuner spawns subprocess(es) to run this callable with each config
3. The callable handles ALL setup internally (symm_mem, distributed init, creating inputs)
4. If subprocess hangs → kill it, try next config

## UX Brainstorm

### Option 1: `autotune_spawn_wrapper` setting

```python
def my_distributed_benchmark(kernel, config):
    """
    Called in spawned subprocess. User is responsible for:
    - Distributed setup (init_process_group, symm_mem)
    - Creating inputs
    - Running kernel with config
    - Returning timing in ms (or inf on error/hang)
    """
    # Initialize distributed (each subprocess does this)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

    # Create inputs fresh (no serialization needed!)
    x = torch.randn(N, D, device=f"cuda:{rank}")
    symm_mem_buffer = symm_mem.empty(N, D, device=f"cuda:{rank}")
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD.group_name)

    # Run kernel with specific config
    compiled = kernel.compile_config(config)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    compiled(x, symm_mem_buffer, bias, weight, ...)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000

@helion.jit(
    configs=[config1, config2, ...],
    autotune_spawn_wrapper=my_distributed_benchmark,
    autotune_spawn_launcher="torchrun --nproc_per_node=4",  # how to launch
    autotune_spawn_timeout=30,
)
def one_shot_allreduce_bias_rmsnorm_kernel(...):
    ...
```

### Option 2: Simpler callable that receives everything

```python
def distributed_benchmark_runner(config_dict):
    """
    Receives serializable config dict, returns timing.
    User handles everything.
    """
    config = helion.Config(**config_dict)

    # Full setup
    setup_distributed()
    x, symm_mem_buffer, symm_mem_hdl = create_inputs()

    # Import and run kernel
    from my_module import my_kernel
    return benchmark(my_kernel.compile_config(config), x, ...)

# Autotune API
best_config = helion.autotune_distributed(
    kernel=my_kernel,
    benchmark_fn=distributed_benchmark_runner,
    configs=[...],
    launcher="torchrun --nproc_per_node=4",
    timeout=30,
)
```

### Option 3: Script-based (most isolated)

```python
@helion.jit(
    configs=[...],
    # Script that will be launched for each config
    autotune_spawn_script="benchmark_distributed.py",
    autotune_spawn_launcher="torchrun --nproc_per_node=4",
)
def my_kernel(...):
    ...
```

Where `benchmark_distributed.py`:
```python
#!/usr/bin/env python
import os, json, helion

config = helion.Config(**json.loads(os.environ["HELION_AUTOTUNE_CONFIG"]))

# Setup everything
setup_distributed()
inputs = create_inputs()

# Import kernel and benchmark
from my_module import my_kernel
timing = benchmark(my_kernel.compile_config(config), *inputs)

# Report timing via stdout (parsed by autotuner)
print(f"HELION_TIMING:{timing}")
```

## Key Design Decisions

### 1. How does callable receive config?

| Approach | Pros | Cons |
|----------|------|------|
| As argument: `fn(kernel, config)` | Clean API, type-safe | Requires kernel to be serializable |
| Via environment: `os.environ["HELION_AUTOTUNE_CONFIG"]` | Works across process boundaries | Less ergonomic |
| Via file: config written to temp file | Most isolated | Extra I/O overhead |

### 2. How is subprocess launched?

| Approach | Pros | Cons |
|----------|------|------|
| `mp.spawn()` | Simple for single-node multi-GPU | Limited flexibility |
| `torchrun` / `torch.distributed.launch` | Industry standard, flexible | External dependency |
| User-provided launcher command | Maximum flexibility | More complex for users |

### 3. How is timing returned?

| Approach | Pros | Cons |
|----------|------|------|
| Return value (mp.spawn with Queue) | Clean API | Requires mp infrastructure |
| stdout parsing (`HELION_TIMING:12.34`) | Works with any launcher | Fragile parsing |
| File-based (write to temp file) | Reliable | Extra I/O, cleanup needed |

### 4. Timeout handling

- Parent process sets alarm/timer
- Kills all child processes on timeout
- Returns `inf` for that config
- Must handle cleanup of zombie processes

## Recommendation

**Option 1** provides the cleanest UX:

```python
@helion.jit(
    configs=[...],
    autotune_spawn_wrapper=my_distributed_benchmark,  # user's callable
    autotune_spawn_world_size=4,  # or auto-detect from launcher
    autotune_spawn_timeout=30,
)
def my_kernel(...):
    ...
```

The callable signature:
```python
def my_distributed_benchmark(
    kernel: BoundKernel,
    config: Config,
    rank: int,
    world_size: int,
) -> float:
    """Returns timing in ms, or inf on failure."""
    ...
```

The autotuner would:
1. For each config, spawn `world_size` processes
2. Each process calls `autotune_spawn_wrapper(kernel, config, rank, world_size)`
3. Collect timings from all ranks (e.g., take max or median)
4. If any process hangs past timeout, kill all and return `inf`

## Implementation Considerations

### New Settings

```python
# In helion/runtime/settings.py
autotune_spawn_wrapper: Callable[[BoundKernel, Config, int, int], float] | None = None
autotune_spawn_world_size: int | None = None  # None = auto-detect or 1
autotune_spawn_timeout: float = 30.0  # seconds
autotune_spawn_launcher: str | None = None  # e.g., "torchrun --nproc_per_node=4"
```

### Process Coordination

For distributed kernels:
1. All ranks must test the same config simultaneously
2. Timing should be collected from rank 0 (or aggregated across ranks)
3. If any rank hangs, all ranks must be killed together
4. Clean process group teardown is important to avoid resource leaks

### Fallback Behavior

- If `autotune_spawn_wrapper` is not set, use current behavior
- If set but not in a distributed context, still use the wrapper (allows single-GPU custom benchmarks too)

---

## Refinement: Reusing `autotune_benchmark_fn`

Instead of adding a new `autotune_spawn_wrapper` setting, we can extend the existing `autotune_benchmark_fn` to cover both use cases.

### Current `autotune_benchmark_fn`

```python
# Current signature (for rebenchmarking)
autotune_benchmark_fn: Callable[..., list[float]] | None = None

# Expected signature:
def my_bench(fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None) -> list[float]
```

Used in `rebenchmark()` to time multiple already-compiled kernels.

### Extended Design

We could make `autotune_benchmark_fn` polymorphic based on signature or a mode parameter:

```python
from typing import Protocol, Union, overload

class RebenchmarkFn(Protocol):
    """Current rebenchmarking signature."""
    def __call__(
        self,
        fns: list[Callable[[], object]],
        *,
        repeat: int,
        desc: str | None = None
    ) -> list[float]: ...

class SpawnBenchmarkFn(Protocol):
    """New distributed spawn signature."""
    def __call__(
        self,
        kernel: BoundKernel,
        config: Config,
        rank: int,
        world_size: int,
    ) -> float: ...

# Setting accepts either
autotune_benchmark_fn: RebenchmarkFn | SpawnBenchmarkFn | None = None
```

The autotuner would detect which mode based on:
1. **Signature inspection** - check parameter names
2. **Explicit flag** - `autotune_precompile="spawn_dist"` triggers spawn mode
3. **Context** - if distributed is initialized and world_size > 1

### Example Usage

```python
def my_distributed_benchmark(kernel, config, rank, world_size):
    """
    When autotune_precompile="spawn_dist", this is called in each spawned process.
    """
    # Setup distributed
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

    # Create inputs
    x = torch.randn(N, D, device=f"cuda:{rank}")
    symm_mem_buffer = symm_mem.empty(N, D, device=f"cuda:{rank}")
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, ...)

    # Compile and benchmark
    compiled = kernel.compile_config(config)
    torch.cuda.synchronize()
    start = time.perf_counter()
    compiled(x, symm_mem_buffer, bias, weight, ...)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000

@helion.jit(
    configs=[...],
    autotune_benchmark_fn=my_distributed_benchmark,
    autotune_precompile="spawn_dist",  # triggers spawn mode
    autotune_spawn_world_size=4,
    autotune_compile_timeout=30,
)
def one_shot_allreduce_bias_rmsnorm_kernel(...):
    ...
```

### Detection Logic

```python
def _is_spawn_benchmark_fn(fn: Callable) -> bool:
    """Detect if function has spawn benchmark signature."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    # Check for spawn signature: (kernel, config, rank, world_size)
    return "kernel" in params and "config" in params and "rank" in params
```

### Backward Compatibility

- If `autotune_benchmark_fn` has old signature → use for rebenchmarking (current behavior)
- If `autotune_benchmark_fn` has new signature + `autotune_precompile="spawn_dist"` → use for spawn benchmarking
- Both can coexist if user provides a function that handles both via `**kwargs`

### Summary of Settings

| Setting | Purpose |
|---------|---------|
| `autotune_benchmark_fn` | User-provided benchmark function (polymorphic signature) |
| `autotune_precompile="spawn_dist"` | Trigger spawn mode for distributed benchmarking |
| `autotune_spawn_world_size` | Number of processes to spawn (default: auto-detect) |
| `autotune_compile_timeout` | Timeout per config in seconds (existing setting, reused) |

### Key Insight

Using `autotune_precompile="spawn_dist"` as the trigger allows us to:
1. Reuse the existing `autotune_benchmark_fn` setting name
2. Maintain backward compatibility with existing rebenchmark usage
3. Keep the API surface minimal by extending rather than adding new settings
