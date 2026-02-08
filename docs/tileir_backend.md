# CUDA Tile IR Backend

## Overview

The CUDA Tile IR backend enables Helion to generate optimized code using the [Triton-to-tile-IR backend](https://github.com/triton-lang/Triton-to-tile-IR), a bridge tool that allows users to use the Triton DSL for developing and compiling operators targeting the [CUDA Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/index.html) backend. This integration includes performance tuning configuration adjustments tailored for the Triton-to-tile-IR bridge, generating kernels with better performance and improving the efficiency of the autotuning process.

## Requirements

To use the TileIR backend, you need:

1. **Compatible Hardware**: A GPU with compute capability 10.x or 12.x (Blackwell)
2. **Triton-to-tile-IR Backend**: The [Triton-to-tile-IR backend](https://github.com/triton-lang/Triton-to-tile-IR) must be installed
3. **Environment Variable**: Set `ENABLE_TILE=1` to enable TileIR support

```bash
export ENABLE_TILE=1
```

## CUDA Tile IR Specific Configuration Parameters

When TileIR is enabled, two new configuration parameters become available in `helion.Config`:

### num_ctas

Number of Cooperative Thread Arrays (CTAs) in one Cooperative Grid Array (CGA). This parameter is analogous to [`num_ctas`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel) in cuTile.

- **Type**: Power of 2
- **Default Tuning Range**: 1 to 2 (can be configured up to 16)
- **Default Value**: 1

```python
num_ctas=2  # Use 2 CTAs per CGA
```

### occupancy

Controls the hardware utilization/occupancy for the kernel. This parameter is analogous to [`occupancy`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel) in cuTile. Higher occupancy may improve performance by hiding memory latency and increasing parallelism.

- **Type**: Power of 2
- **Default Tuning Range**: 1 to 8 (can be configured up to 32)
- **Default Value**: 1

```python
occupancy=2  # Target occupancy of 2
```

## Tuning Knob Modifications

All the following changes only take effect for the TileIR backend and won't influence other backends.

### New Knobs Introduced

| Knob | Description | Default Tuning Range | Default |
|------|-------------|-------------|---------|
| `num_ctas` | Number of CTAs in one CGA | 1-2 (power of 2) | 1 |
| `occupancy` | Hardware utilization/occupancy | 1-8 (power of 2) | 1 |

### Unsupported Knobs (Removed from Search Space)

The following knobs are either currently unsupported by the TileIR backend or are ignored during compilation. They have been removed from the autotuning search space to prevent unnecessary tuning iterations:

- `static_ranges`
- `range_unroll_factors`
- `range_multi_buffers`
- `range_flattens`
- `range_warp_specialize`
- `load_eviction_policies`

### Modified Knobs

| Knob | Modification | Rationale |
|------|--------------|-----------|
| `num_warps` | Constrained to `4` | Acts as a placeholder. The TileIR backend is expected to introduce broader support for other values in the future. |
| `num_stages` | Changed from `IntegerFragment` to `EnumFragment` (range: 1-10) | Analogous to [`latency`](https://docs.nvidia.com/cuda/cutile-python/performance.html#load-store-performance-hints) in cuTile. Switching to EnumFragment allows for more discrete search directions, enhancing overall search effectiveness. |
| `indexing` | Removed `block_ptr` type | The `block_ptr` indexing type is currently unsupported by the TileIR backend. `pointer` and `tensor_descriptor` indexing are available. |

## Usage Examples

### Basic Example

```python
import torch
import helion
import helion.language as hl

@helion.kernel(
    autotune_effort="none",
    config=helion.Config(
        block_sizes=[128, 128],
        num_ctas=2,
        occupancy=2,
    ),
)
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        result[tile] = x[tile] + y[tile]
    return result

# Run the kernel
x = torch.randn(128, 128, device="cuda", dtype=torch.float32)
y = torch.randn(128, 128, device="cuda", dtype=torch.float32)
result = add_kernel(x, y)
```

### Autotuning with CUDA Tile IR Backend

When autotuning is enabled on TileIR-supported hardware, the autotuner will automatically search over `num_ctas` and `occupancy` values in addition to the standard tuning parameters. The search space is optimized to exclude unsupported knobs, improving autotuning efficiency.

```python
@helion.kernel()  # Full autotuning enabled by default
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out
```

### Manual Configuration for CUDA Tile IR Backend

You can also specify multiple TileIR configurations for lightweight autotuning:

```python
configs = [
    helion.Config(block_sizes=[64, 64], num_ctas=1, occupancy=1),
    helion.Config(block_sizes=[64, 64], num_ctas=2, occupancy=2),
    helion.Config(block_sizes=[128, 128], num_ctas=2, occupancy=4),
    helion.Config(block_sizes=[128, 128], num_ctas=2, occupancy=8),
]

@helion.kernel(configs=configs)
def optimized_kernel(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        result[tile] = x[tile] * 2
    return result
```

## Limitations

### Unsupported Triton Operations

There are some operations currently unsupported by the TileIR backend. Please refer to the [Triton-to-tile-IR README](https://github.com/triton-lang/Triton-to-tile-IR) for the latest list of unsupported operations.

Kernels using these operations should not be compiled with the TileIR backend.

## Summary Table

| Feature | TileIR Backend | Standard Backend |
|---------|---------------|------------------|
| **Hardware** | SM100/SM120 (compute capability 10.x, 12.x) | All CUDA GPUs |
| **Environment** | `ENABLE_TILE=1` required | No special env var |
| **num_ctas** | ✅ Supported | ❌ Not available |
| **occupancy** | ✅ Supported | ❌ Not available |
| **Indexing: pointer** | ✅ Supported | ✅ Supported |
| **Indexing: block_ptr** | ❌ Not available | ✅ Supported |
| **Indexing: tensor_descriptor** | ✅ Supported | ✅ Supported (Hopper+) |
| **Eviction policies** | ❌ Not available | ✅ Supported |
| **Warp specialization** | ❌ Not available | ✅ Supported |
| **Range config options** | ❌ Not available | ✅ Supported |
| **num_warps** | Fixed at 4 | Tunable |
| **num_stages** | EnumFragment (1-10) | IntegerFragment |

## Checking CUDA Tile IR Backend Support

You can programmatically check if TileIR is supported:

```python
from helion._compat import use_tileir_tunables

if use_tileir_tunables():
    print("TileIR backend is available")
else:
    print("TileIR backend is not available")
```

## Error Handling

If you try to use TileIR-specific parameters (`num_ctas`, `occupancy`) on hardware that doesn't support TileIR, Helion will raise an `InvalidConfig` error.

## References

- [Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR) - The bridge tool for Triton to TileIR
- [CUDA Tile IR Documentation](https://docs.nvidia.com/cuda/tile-ir/latest/index.html) - NVIDIA TileIR official documentation
- [cuTile Python API](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel) - Reference for `num_ctas` and `occupancy` parameters
- [cuTile Performance Hints](https://docs.nvidia.com/cuda/cutile-python/performance.html#load-store-performance-hints) - Reference for `latency` (analogous to `num_stages`)
