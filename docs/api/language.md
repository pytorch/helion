# Language Module

The `helion.language` module contains the core DSL constructs for writing GPU kernels.

## Loop Constructs

### tile()

```{eval-rst}
.. currentmodule:: helion.language

.. autofunction:: tile
```

The `tile()` function is the primary way to create parallel loops in Helion kernels. It provides several key features:

**Tiling Strategies**: The exact tiling strategy is determined by a Config object, typically created through autotuning. This allows for:
- Multidimensional tiling
- Index swizzling for cache locality
- Dimension reordering
- Flattening of iteration spaces

**Usage Patterns**:

```python
# Simple 1D tiling
for tile in hl.tile(1000):
    # tile.begin, tile.end, tile.block_size are available
    # Load entire tile (not just first element)
    data = tensor[tile]  # or hl.load(tensor, tile) for explicit loading
```

```python
# 2D tiling
for tile_i, tile_j in hl.tile([height, width]):
    # Each tile represents a portion of the 2D space
    pass
```

```python
# With explicit begin/end/block_size
for tile in hl.tile(0, 1000, block_size=64):
    pass
```

**Grid vs Loop Behavior**:
- When used at the top level of a kernel function, `tile()` becomes the grid of the kernel (parallel blocks)
- When used nested inside another loop, it becomes a sequential loop within each block

### grid()

```{eval-rst}
.. autofunction:: grid
```

The `grid()` function iterates over individual indices rather than tiles. It's equivalent to `tile(size, block_size=1)` but returns scalar indices instead of tile objects.

### static_range()

```{eval-rst}
.. autofunction:: static_range
```

`static_range()` behaves like a compile-time unrolled range for small loops. It hints the compiler to fully unroll the loop body where profitable.

## Memory Operations

### load()

```{eval-rst}
.. autofunction:: load
```

### store()

```{eval-rst}
.. autofunction:: store
```

### atomic_add()

```{eval-rst}
.. autofunction:: atomic_add
```

### atomic_and()

```{eval-rst}
.. autofunction:: atomic_and
```

### atomic_or()

```{eval-rst}
.. autofunction:: atomic_or
```

### atomic_xor()

```{eval-rst}
.. autofunction:: atomic_xor
```

### atomic_xchg()

```{eval-rst}
.. autofunction:: atomic_xchg
```

### atomic_max()

```{eval-rst}
.. autofunction:: atomic_max
```

### atomic_min()

```{eval-rst}
.. autofunction:: atomic_min
```

### atomic_cas()

```{eval-rst}
.. autofunction:: atomic_cas
```

## Inline Assembly

### inline_asm_elementwise()

```{eval-rst}
.. autofunction:: inline_asm_elementwise
```

Executes target-specific inline assembly on elements of one or more tensors with broadcasting and optional packed processing.

### inline_triton()

```{eval-rst}
.. autofunction:: inline_triton
```

Embeds small Triton code snippets directly inside a Helion kernel. Common indentation is removed automatically, placeholders are replaced using ``str.format`` with tuple or dict arguments, and the final line in the snippet becomes the return value. Provide tensors (or tuples of tensors) via ``output_like`` so Helion knows the type of the return value.

## Tensor Creation

### zeros()

```{eval-rst}
.. autofunction:: zeros
```

### full()

```{eval-rst}
.. autofunction:: full
```

### arange()

See {func}`~helion.language.arange` for details.

## Tunable Parameters

### register_block_size()

```{eval-rst}
.. autofunction:: register_block_size
```

### register_tunable()

```{eval-rst}
.. autofunction:: register_tunable
```

## Tile Operations

### Tile Class

```{eval-rst}
.. autoclass:: Tile
   :members:
   :undoc-members:
```

The `Tile` class represents a portion of an iteration space with the following key attributes:
- `begin`: Starting indices of the tile
- `end`: Ending indices of the tile
- `block_size`: Size of the tile in each dimension

## View Operations

### subscript()

```{eval-rst}
.. autofunction:: subscript
```

### split()

```{eval-rst}
.. autofunction:: split
```

### join()

```{eval-rst}
.. autofunction:: join
```

## StackTensor
### StackTensor class
```{eval-rst}
.. autoclass:: StackTensor
   :undoc-members:
```

### stacktensor_like
```{eval-rst}
.. autofunction:: stacktensor_like
```

## Reduction Operations

### reduce()

See {func}`~helion.language.reduce` for details.

## Scan Operations

### associative_scan()

See {func}`~helion.language.associative_scan` for details.

### cumsum()

See {func}`~helion.language.cumsum` for details.

### cumprod()

See {func}`~helion.language.cumprod` for details.

### tile_index()

```{eval-rst}
.. autofunction:: tile_index
```

### tile_begin()

```{eval-rst}
.. autofunction:: tile_begin
```

### tile_end()

```{eval-rst}
.. autofunction:: tile_end
```

### tile_block_size()

```{eval-rst}
.. autofunction:: tile_block_size
```

### tile_id()

```{eval-rst}
.. autofunction:: tile_id
```

## Synchronization


### signal()

```{eval-rst}
.. autofunction:: signal
```

### wait()

```{eval-rst}
.. autofunction:: wait
```

## Utilities

### device_print()

See {func}`~helion.language.device_print` for details.

## Constexpr Operations

### constexpr()

See {class}`~helion.language.constexpr` for details.

### specialize()

See {func}`~helion.language.specialize` for details.

## Matrix Operations

### dot()

See {func}`~helion.language.dot` for details.
