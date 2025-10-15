# Kernel

The `Kernel` class is the main entry point for executing Helion GPU kernels.

```{eval-rst}
.. currentmodule:: helion

.. autoclass:: Kernel
   :members:
   :show-inheritance:
```

## Overview

A `Kernel` object is typically created via the `@helion.kernel` decorator. It manages:

- **Compilation** of Python functions to GPU code
- **Autotuning** to find optimal configurations
- **Caching** of compiled kernels
- **Execution** with automatic argument binding

The kernel compilation process converts Python functions using `helion.language` constructs into optimized Triton GPU kernels.

## Creation and Usage

### Basic Kernel Creation

```python
import torch
import helion
import helion.language as hl

@helion.kernel
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(a)
    for i in hl.grid(a.size(0)):
        result[i] = a[i] + b[i]
    return result

# Usage
a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = vector_add(a, b)  # Automatically compiles and executes
```

### With Custom Settings

```python
@helion.kernel(
    autotune_effort="none",    # Skip autotuning
    print_output_code=True      # Debug generated code
)
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```

### With Restricted Configurations

```python
@helion.kernel(configs=[
    helion.Config(block_sizes=[32], num_warps=4),
    helion.Config(block_sizes=[64], num_warps=8)
])
def optimized_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```


## BoundKernel

When you call `kernel.bind(args)`, you get a `BoundKernel` that's specialized for specific argument types and (optionally) shapes:

```python
# Bind once, execute many times
bound = my_kernel.bind((example_tensor,))
result1 = bound(tensor1)  # Compatible tensor (same dtype/device)
result2 = bound(tensor2)  # Compatible tensor (same dtype/device)

# With static_shapes=True, tensors must have exact same shapes/strides
@helion.kernel(static_shapes=True)
def shape_specialized_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass

bound_static = shape_specialized_kernel.bind((torch.randn(100, 50),))
result = bound_static(torch.randn(100, 50))  # Must be exactly [100, 50]
```

```{warning}
Helion shape-specializes kernels by default (`static_shapes=True`) for the best performance. Bound kernels and caches require tensors with the exact same shapes and strides as the examples you compile against. Set `static_shapes=False` if you need the same compiled kernel to serve many shapes.
```

### BoundKernel Methods

The returned BoundKernel has these methods:

- `__call__(*args)` - Execute with bound arguments
- `autotune(args, **kwargs)` - Autotune this specific binding
- `set_config(config)` - Set and compile specific configuration
- `to_triton_code(config)` - Generate Triton source code
- `compile_config(config)` - Compile for specific configuration

## Advanced Usage

### Manual Autotuning

```python
# Separate autotuning from execution
kernel = my_kernel

# Find best config
config = kernel.autotune(example_args, num_iterations=100)

# Later, use the optimized config
result = kernel(actual_args)  # Uses cached config
```

### Config Management

```python
bound = kernel.bind(args)

# set a specific configuration
bound.set_config(helion.Config(block_sizes=[64], num_warps=8))

# generate Triton code for the bound kernel
triton_code = bound.to_triton_code(config)
print(triton_code)
```

## Caching and Specialization

Kernels are automatically cached based on:

- **Argument types** (dtype, device)
- **Tensor shapes** (default: `static_shapes=True`)

By default (`static_shapes=True`), Helion treats shapes and strides as compile-time constants, baking them into generated Triton code for the best performance. To reuse a single compiled kernel across size variations, set `static_shapes=False`, which instead buckets each dimension as `{0, 1, ≥2}` and allows more inputs to share the same cache entry.

```python
# These create separate cache entries
tensor_float = torch.randn(100, dtype=torch.float32, device='cuda')
tensor_int = torch.randint(0, 10, (100,), dtype=torch.int32, device='cuda')

result1 = my_kernel(tensor_float)  # Compiles for float32
result2 = my_kernel(tensor_int)    # Compiles for int32 (separate cache)
```

## Settings vs Config in Kernel Creation

When creating kernels, you'll work with two distinct types of parameters:

### Settings: Compilation Control
Settings control **how the kernel is compiled** and the development environment:

```python
@helion.kernel(
    # Settings parameters
    autotune_effort="none",      # Skip autotuning for development
    autotune_effort="quick",     # Smaller autotuning budget when search is enabled
    print_output_code=True,       # Debug: show generated Triton code
    static_shapes=True,           # Compilation optimization strategy
    autotune_log_level=logging.DEBUG  # Verbose autotuning output
)
def debug_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```

### Config: Execution Control
Config parameters control **how the kernel executes** on GPU hardware:

```python
@helion.kernel(
    # Config parameters
    config=helion.Config(
        block_sizes=[64, 128],    # GPU tile sizes
        num_warps=8,              # Thread parallelism
        num_stages=4,             # Pipeline stages
        indexing='block_ptr'      # Memory access strategy
    )
)
def production_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```

### Combined Usage
You can specify both Settings and Config together:

```python
@helion.kernel(
    # Settings: control compilation
    print_output_code=False,      # No debug output
    static_shapes=True,           # Shape specialization
    # Config: control execution
    config=helion.Config(
        block_sizes=[32, 32],     # Execution parameters
        num_warps=4
    )
)
def optimized_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```

For more details, see {doc}`settings` (compilation control) and {doc}`config` (execution control).

## See Also

- {doc}`settings` - Compilation behavior and debugging options (controls **how kernels are compiled**)
- {doc}`config` - GPU execution parameters and optimization strategies (controls **how kernels execute**)
- {doc}`exceptions` - Exception handling and error diagnostics
- {doc}`language` - Helion language constructs for kernel authoring
- {doc}`autotuner` - Autotuning configuration and search strategies
