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

### jagged_tile()

```{eval-rst}
.. autofunction:: jagged_tile
```

The `jagged_tile()` function is the jagged counterpart to `tile()`. It iterates an
inner dimension whose extent varies per lane of an enclosing parent tile, using a
1D tensor of per-lane end positions from that parent context.

Instead of writing a dense inner loop and manually building a mask, `jagged_tile()`
lets Helion apply the masking implicitly for indices beyond each lane's true length.

### static_range()

```{eval-rst}
.. autofunction:: static_range
```

`static_range()` behaves like a compile-time unrolled range for small loops. It hints the compiler to fully unroll the loop body where profitable.

### barrier()

```{eval-rst}
.. autofunction:: barrier
```

`barrier()` inserts a grid-wide synchronization point between top-level `hl.tile` or `hl.grid` loops. It forces persistent kernel execution so that all blocks complete one phase before the next begins.

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

### triton_kernel()

```{eval-rst}
.. autofunction:: triton_kernel
```

Define (once) and call a ``@triton.jit`` function from Helion device code.

- Accepts either:
  - a source string containing a single Triton function definition,
  - a function name string referring to a ``@triton.jit`` function in the kernel’s module, or
  - a Python function object (or Triton JITFunction; unwrapped via ``.fn``).
- The function is emitted at module scope once and then invoked from the kernel body.
- Pass ``output_like`` tensors for shape/dtype checks identical to ``inline_triton``.

Example (by name):

```python
@triton.jit
def add_pairs(a, b):
    return a + b

@helion.kernel()
def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = hl.triton_kernel("add_pairs", args=(x[tile], y[tile]), output_like=x[tile])
    return out
```

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

```{eval-rst}
.. autofunction:: arange
```

### rand()

```{eval-rst}
.. autofunction:: rand
```

### randint()

```{eval-rst}
.. autofunction:: randint
```

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

```{eval-rst}
.. autofunction:: reduce
```

## Scan Operations

### associative_scan()

```{eval-rst}
.. autofunction:: associative_scan
```

### cumsum()

```{eval-rst}
.. autofunction:: cumsum
```

### cumprod()

```{eval-rst}
.. autofunction:: cumprod
```

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

## Utilities

### device_print()

```{eval-rst}
.. autofunction:: device_print
```

## Constexpr Operations

### constexpr()

```{eval-rst}
.. autoclass:: constexpr
```

### specialize()

```{eval-rst}
.. autofunction:: specialize
```

## Matrix Operations

### dot()

```{eval-rst}
.. autofunction:: dot
```

### dot_scaled()

```{eval-rst}
.. autofunction:: dot_scaled
```

`dot_scaled()` performs block-scaled matrix multiplication using low-precision formats (e.g., `e2m1`, `e4m3`, `e5m2`). Each input matrix has an associated per-block scale tensor and format string. This maps to Triton's `tl.dot_scaled` for hardware-accelerated scaled dot products on supported architectures.
