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

When a tile indexes a dimension known to have size one, Triton and CuTe
broadcast the input value across the live lanes. An explicit lane mask still
applies to each lane: with four live lanes, an all-true mask and input
`torch.ones(1)`, the sum is four. Eager reference mode slices the input and can
instead produce one. This rule describes loads.

A shifted or strided root grid counts the coordinates reachable on each axis.
A literal zero stride raises `helion.exc.InvalidAPIUsage` when the kernel is
bound. A zero host scalar stride raises `ValueError` before launch; a zero
device tensor stride signals a device error.
The persistent scheduler uses signed 64-bit counts, including products of axes
and sums of roots. A static grid with `2**63` or more programs is rejected
at compile time. A host-computed count that large raises `OverflowError`
before launch (an empty Cartesian product needs no programs). When the bounds are
device tensors, Triton and CuTe signal a device error instead of reading the
tensors back to the host. Triton emits an assertion with a diagnostic;
CuTe emits a device trap without message text. Neither requires debug mode.

Each persistent worker reads a device tensor bound before visiting its own
root tiles. Writing the bound while another worker might still read it is
an unsupported data race. Keep shared bounds unchanged until the grid finishes.

A root tile with a loaded `uint64` stop still uses signed vector
indices: a reached tile may end at `2**63`, but a stop above that boundary
signals a device error. An empty range needs no such check. Scalar `hl.grid`
retains its separate support for unsigned coordinates. On TileIR,
requesting a coordinate such as `tile.end` that may need the full
`uint64` range raises `helion.exc.InvalidConfig` at compile time.

Runtime Python bounds (including plain zero-origin tile stops and expressions such as
`n + 5`) must also fit signed 64 bits for a live range. An unrepresentable
bound raises `OverflowError` before launch, even when the trip count is small.
If a tensor supplies the other endpoint, the host still checks its own bound
without reading the device tensor back. For an empty Cartesian product, an
out-of-range Python bound can be ignored only if the generated launch no longer
passes it or an unrepresentable value derived from it to the device. Otherwise
the host raises `OverflowError` before converting that integer. A runtime
Python stride passed to the device must also fit signed 64 bits; this
applies even when a device tensor supplies an endpoint.

For a flattened multidimensional grid, the product of the axis lengths must
also fit signed 64-bit arithmetic **before** dividing by the number of elements
per tile. For example, flattening axes of lengths `2**31` and `2**32` is
unsupported even with 256 elements per tile: the grid has `2**55` programs,
but its element count is `2**63`.

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

A negative integer literal counts from the end of its dimension; for
example, `out[-1, :] = row` writes the last row.

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

### cumsum()

```{eval-rst}
.. autofunction:: cumsum
```

### cumprod()

```{eval-rst}
.. autofunction:: cumprod
```

### tile_index()

On CuTe, `tile.index` uses at least `torch.int32` when the configured index
dtype is an 8- or 16-bit integer. This lets a tile cross that dtype's limit
without wrapping its physical coordinates. It does not change the dtype of
a tensor loaded from memory.

```{eval-rst}
.. autofunction:: tile_index
```

### tile_begin()

```{eval-rst}
.. autofunction:: tile_begin
```

### tile_end()

The exclusive edge can require a wider integer type than `tile.index`.
In particular, a ragged CuTe tile can return a signed 64-bit edge even when
its input uses 32-bit indexing.

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
