# Language Module

The `helion.language` module contains the core DSL constructs for writing GPU kernels.

Operation and dtype coverage depends on the selected backend. Inline Triton
code and target-specific assembly are not portable across all backends; see
{doc}`../backends` when choosing a target or porting a kernel.

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

`barrier()` synchronizes all blocks between top-level `hl.tile` or `hl.grid` loops, so one phase completes before the next begins. Depending on the backend and configuration, the compiler uses a persistent kernel or separate ordered kernel launches.

## Memory Operations

### load()

```{eval-rst}
.. autofunction:: load
```

### store()

```{eval-rst}
.. autofunction:: store
```

### prefetch()

```{eval-rst}
.. autofunction:: prefetch
```

### prefetch_code()

```{eval-rst}
.. autofunction:: prefetch_code
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

On CuTe, the default ``cute_rng_stream="auto"`` maps an explicit uniform
``hl.rand`` logical offset ``i`` to Philox word ``i % 4`` at counter ``i // 4``.
Use ``cute_rng_stream="word0"`` to reproduce the earlier sequence. This setting
does not change ``hl.rand4x``, ``hl.randint`` or implicit Torch random operations
under the automatic policy. Other backends retain their existing defaults.

### rand4x()

```{eval-rst}
.. autofunction:: rand4x
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

(tuple-callback-layouts)=
#### Tuple callback layouts

For a tuple of `n` input tensors, the combine function can take either two
tuples (`left, right`) or `2*n` element tensors (all left elements, then all
right elements). If only one call fits the Python signature, Helion uses it.

For two or more inputs, use a signature that accepts only the layout your
callback expects. Optional positional parameters or `*args` can make both
layouts fit; with a plain Python callback, an eager reduction can then
succeed even when the same callback fails to compile. For a pair of tensors, use keyword-only options
such as `def combine(left, right, *, i=0, j=1)` to receive two tuples.
To receive the four element tensors, declare four required positional
parameters.

For a singleton input tuple, both layouts have two operands. A callback
that accepts them receives tuples with fewer than four declared parameters
and element tensors with four or more. For example, this callback receives
two one-element tuples and returns one reduced value:

```python
def add_one(left, right):
    return (left[0] + right[0],)

(total,) = hl.reduce(add_one, (values,), dim=0)
```

Earlier eager reductions did the reverse for some singleton signatures: an
exactly two-parameter function got element tensors, and a function with
four or more parameters got tuples. Update the former to index `left[0]`
and `right[0]` as above. To keep using tuples with configurable options,
close over those options in a two-parameter function: both singleton
layouts have two positional arguments, so keyword-only options cannot
disambiguate them.

For a single tensor passed as `values` instead of `(values,)`, the callback
receives and returns tensors.


## Scan Operations

`cumsum` and `cumprod` promote integer and boolean inputs to `torch.int64`
by default, as PyTorch does. To scan in a different dtype, pass `dtype=`.
Pallas rejects unsupported 64-bit scan dtypes instead of narrowing them.
For integer `cumsum` on Pallas, supply an explicit supported dtype such as
`torch.int32`. Calling `associative_scan` directly never applies this promotion.

### associative_scan()

```{eval-rst}
.. autofunction:: associative_scan
```

For tuple inputs, see {ref}`tuple-callback-layouts`.

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

`dot_scaled()` performs block-scaled matrix multiplication using low-precision formats (e.g., `e2m1`, `e4m3`, `e5m2`). Each input matrix has an associated per-block scale tensor and format string. The lowering depends on the backend: for example, Triton uses `tl.dot_scaled`, while CuTe uses its own block-scaled matrix instructions. Supported formats and layouts depend on the backend and hardware.
