# Config

The `Config` class represents kernel optimization parameters that control how Helion kernels are compiled and executed.

```{eval-rst}
.. currentmodule:: helion

.. autoclass:: Config
   :members:
   :show-inheritance:
```

## Overview

**Config** objects specify **optimization parameters** that control how Helion kernels run on the hardware.

### Key Characteristics

- **Performance-focused**: Control GPU resource allocation, memory access patterns, and execution strategies
- **Autotuned**: The autotuner searches through different Config combinations to find optimal performance
- **Kernel-specific**: Each kernel can have different optimal Config parameters based on its computation pattern
- **Hardware-dependent**: Optimal configs vary based on GPU architecture and problem size

### Config vs Settings

| Aspect | Config | Settings |
|--------|--------|----------|
| **Purpose** | Control execution performance | Control compilation behavior |
| **Autotuning** | ✅ Automatically optimized | ❌ Never autotuned |
| **Examples** | `block_sizes`, `num_warps`, `indexing` | `print_output_code`, `autotune_effort` |
| **When to use** | Performance optimization | Development, debugging, environment setup |


Configs are typically discovered automatically through autotuning, but can also be manually specified for more control.

## Configuration Parameters

### Block Sizes and Resources

```{eval-rst}
.. autoattribute:: Config.block_sizes

   List of tile sizes for ``hl.tile()`` loops. Each value controls the number of elements processed per GPU thread block for the corresponding tile dimension.

.. autoattribute:: Config.reduction_loops

   Configuration for reduction operations within loops.

.. autoattribute:: Config.num_warps

   Number of warps (groups of 32 threads) per thread block. Higher values increase parallelism but may reduce occupancy.

.. autoattribute:: Config.num_stages

   Number of pipeline stages for software pipelining. Higher values can improve memory bandwidth utilization.
```

### Loop Optimizations

```{eval-rst}
.. autoattribute:: Config.loop_orders

   Permutation of loop iteration order for each ``hl.tile()`` loop. Used to optimize memory access patterns.

.. autoattribute:: Config.flatten_loops

   Whether to flatten nested loops for each ``hl.tile()`` invocation.

.. autoattribute:: Config.range_unroll_factors

   Unroll factors for ``tl.range`` loops in generated Triton code.

.. autoattribute:: Config.range_warp_specializes

   Whether to enable warp specialization for ``tl.range`` loops.

.. autoattribute:: Config.range_num_stages

   Number of pipeline stages for ``tl.range`` loops.

.. autoattribute:: Config.range_multi_buffers

   Controls ``disallow_acc_multi_buffer`` parameter for ``tl.range`` loops.

.. autoattribute:: Config.range_flattens

   Controls ``flatten`` parameter for ``tl.range`` loops.

.. autoattribute:: Config.static_ranges

   Whether to use ``tl.static_range`` instead of ``tl.range``.
```

### Execution and Indexing

```{eval-rst}
.. autoattribute:: Config.pid_type

   Program ID layout strategy:

   - ``"flat"``: Standard linear program ID assignment
   - ``"xyz"``: 3D program ID layout
   - ``"persistent_blocked"``: Persistent kernels with blocked work distribution
   - ``"persistent_interleaved"``: Persistent kernels with interleaved distribution

.. autoattribute:: Config.l2_groupings

   Controls reordering of program IDs to improve L2 cache locality.

.. autoattribute:: Config.indexing

   Memory indexing strategy for load and store operations. Can be specified as:

   **Single strategy (applies to all loads and stores - backward compatible):**

   .. code-block:: python

      indexing="block_ptr"  # All loads and stores use block pointers

   **Per-operation strategies (list, one per load/store in execution order):**

   .. code-block:: python

      # 2 loads + 1 store = 3 indexing strategies
      indexing=["pointer", "pointer", "block_ptr"]  # loads use pointer, store uses block_ptr

   **Empty/omitted (defaults to** ``"pointer"`` **for all operations):**

   .. code-block:: python

      # indexing not specified - all loads and stores use pointer indexing

   **Valid strategies:**

   - ``"pointer"``: Pointer-based indexing (default)
   - ``"tensor_descriptor"``: Tensor descriptor indexing (requires Hopper+ GPU)
   - ``"block_ptr"``: Block pointer indexing

   .. note::
      When using a list, provide one strategy for each load and store operation in the order
      they appear in the kernel. The indexing list is ordered as:
      ``[load1, load2, ..., loadN, store1, store2, ..., storeM]``
```

### Memory and Caching

```{eval-rst}
.. autoattribute:: Config.load_eviction_policies

   Eviction policies for load operations issued from device loops. Provide one policy
   per ``hl.load`` site discovered in the kernel. Allowed values:

   - ``""``: No eviction policy (omitted)
   - ``"first"``: Maps to Triton ``eviction_policy='evict_first'``
   - ``"last"``: Maps to Triton ``eviction_policy='evict_last'``

   Notes:

   - The number of entries must match the number of load sites considered tunable by the kernel.
   - An explicit ``eviction_policy=...`` argument passed to ``hl.load`` overrides this config.
```

## Usage Examples

### Manual Config Creation

```python
import torch
import helion
import helion.language as hl

# Create a specific configuration
config = helion.Config(
    block_sizes=[64, 32],      # 64 elements per tile in dim 0, 32 in dim 1
    num_warps=8,               # Use 8 warps (256 threads) per block
    num_stages=4,              # 4-stage pipeline
    pid_type="xyz"             # Use 3D program ID layout
)

# Use with kernel
@helion.kernel(config=config)
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x)
    for i, j in hl.tile(x.shape):
        result[i, j] = x[i, j] * 2
    return result
```

### Eviction Policy Example

```python
import torch
import helion
import helion.language as hl

@helion.kernel(
    config={
        "block_size": 16,
        "load_eviction_policies": ["", "last"],  # second load uses evict_last
    }
)
def kernel_with_eviction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        a = hl.load(x, [tile])                 # No eviction policy
        b = hl.load(y, [tile])                 # Will use evict_last from config
        out[tile] = a + b
    return out

# Explicit policy on hl.load overrides config:
# hl.load(x, [tile], eviction_policy="evict_first")
```

### Per-Load Indexing Example

```python
import torch
import helion
import helion.language as hl

# Single indexing strategy for all loads and stores (backward compatible)
@helion.kernel(config={"indexing": "block_ptr"})
def kernel_uniform_indexing(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        a = hl.load(x, [tile])  # Load: uses block_ptr
        b = hl.load(y, [tile])  # Load: uses block_ptr
        out[tile] = a + b       # Store: uses block_ptr
    return out

# Per-operation indexing strategies for fine-grained control
# Indexing list is ordered: [load1, load2, ..., store1, store2, ...]
@helion.kernel(
    config={
        "block_size": 16,
        "indexing": ["pointer", "pointer", "block_ptr"],  # 2 loads + 1 store
    }
)
def kernel_mixed_indexing(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        a = hl.load(x, [tile])  # First load: pointer indexing
        b = hl.load(y, [tile])  # Second load: pointer indexing
        out[tile] = a + b       # Store: block_ptr indexing
    return out
```

### Config Serialization

```python
# Save config to file
config.save("my_config.json")

# Load config from file
loaded_config = helion.Config.load("my_config.json")

# JSON serialization
config_dict = config.to_json()
restored_config = helion.Config.from_json(config_dict)
```

### Autotuning with Restricted Configs

```python
# Restrict autotuning to specific configurations
configs = [
    helion.Config(block_sizes=[32, 32], num_warps=4),
    helion.Config(block_sizes=[64, 16], num_warps=8),
    helion.Config(block_sizes=[16, 64], num_warps=4),
]

@helion.kernel(configs=configs)
def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    k2, n = b.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out
```

## See Also

- {doc}`settings` - Compilation settings and environment variables
- {doc}`kernel` - Kernel execution and autotuning
- {doc}`autotuner` - Autotuning configuration
