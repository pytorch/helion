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

```{eval-rst}
.. currentmodule:: helion.runtime

.. autofunction:: capture_metal_launches

.. autoclass:: MetalLaunch
   :members:

.. autoclass:: MetalArgument
```
