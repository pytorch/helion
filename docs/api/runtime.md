# Runtime Module

The `helion.runtime` module handles kernel execution and configuration management.

```{eval-rst}
.. currentmodule:: helion.runtime
```

## Key Classes

The runtime module provides the core execution infrastructure. For detailed documentation of individual classes, see:

- {doc}`config` - Configuration management
- {doc}`kernel` - Kernel execution
- {doc}`settings` - Compilation settings

## Utility Functions

```{eval-rst}
.. currentmodule:: helion.runtime

.. autofunction:: triton_wait_signal

.. autofunction:: get_num_sm

.. autofunction:: set_triton_allocator
```

## Exporting Metal kernels

A Metal kernel's launches can be captured as data, for another runtime to
run: `capture_metal_launches` runs the kernel's host code with a launcher
that records each launch's MSL source, grid, threadgroup and arguments,
without compiling or dispatching it.

```python
launches = helion.runtime.capture_metal_launches(kernel, args, config)
for launch in launches:
    print(launch.name, launch.grid, launch.threadgroup)
    print([(a.name, a.binding, a.written) for a in launch.arguments])
```

`MetalLaunch.to_mlx()` arranges a launch for MLX's `mx.fast.metal_kernel`:
the buffers the kernel reads become inputs and the ones it writes outputs,
torch's c10 headers are inlined into the header, runtime scalars are baked
in, and the grid counts threads. MLX allocates fresh outputs, filled with
one `init_value` that the caller states: `None` when the kernel writes every
element of its outputs, else the value of the elements it leaves
(`"captured"` takes it from the outputs, refusing ones one value cannot
reproduce).

```python
import mlx.core as mx

k = launch.to_mlx(init_value=None)  # the kernel writes all of its outputs
fn = mx.fast.metal_kernel(
    name=k.name,
    input_names=list(k.input_names),
    output_names=list(k.output_names),
    source=k.source,
    header=k.header,
)
```

```{eval-rst}
.. currentmodule:: helion.runtime

.. autofunction:: capture_metal_launches

.. autoclass:: MetalLaunch
   :members:

.. autoclass:: MetalArgument

.. autoclass:: MLXKernel
```
