# Helion TPU/Pallas Support Analysis

This document analyzes the changes needed to support running standalone Helion kernels on TPU via JAX/Pallas.

## Executive Summary

Helion can reuse PyTorch Inductor's `PallasKernelOverrides` for elementwise op mappings, minimizing code duplication. Custom work is needed for memory access patterns and kernel structure.

## Current Architecture

### Helion Compilation Flow

```
@helion.kernel                    User-defined kernel
       │
       ▼
  FakeTensor tracing             Traces kernel body
       │
       ▼
    FX Graph                     PyTorch operations
       │
       ▼
  Inductor IR                    TensorBox, Pointwise, Reduction
       │
       ▼
 GenerateASTFromInductor         Uses TritonOverrides
       │
       ▼
   @triton.jit                   Standalone Triton kernel
```

### Key Triton-Specific Code Locations

| File | Component | Purpose |
|------|-----------|---------|
| `_compiler/inductor_lowering.py:792` | `GenerateASTFromInductor` | Uses `TritonOverrides()` for op mapping |
| `_compiler/device_function.py:647` | Kernel decorator | Generates `@triton.jit` |
| `_compiler/device_function.py:672-691` | Kernel launch | `num_warps`, `num_stages` |
| `_compiler/output_header.py:14-25` | Imports | `triton`, `triton.language as tl` |
| `_compiler/reduction_strategy.py` | Reductions | `tl.reduce`, `tl.sum` |
| `_compiler/tile_strategy.py` | Memory ops | `tl.load`, `tl.store`, `tl.arange` |
| `_compiler/indexing_strategy.py` | Indexing | Triton indexing patterns |

## Reusing Inductor's Pallas Backend

### What Can Be Reused

PyTorch Inductor's `PallasKernelOverrides` (`torch/_inductor/codegen/pallas.py`) provides:

**Elementwise ops** - Both return strings, directly swappable:
```python
# TritonOverrides
def sin(x): return f"tl.sin({x})"

# PallasKernelOverrides
def sin(x): return f"jnp.sin({x})"
```

**Full list of reusable ops:**
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Exponential/Log: `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`
- Power: `pow`, `sqrt`, `rsqrt`
- Rounding: `floor`, `ceil`, `trunc`, `round`
- Comparison: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- Logical: `and_`, `or_`, `xor`, `not_`
- Arithmetic: `add`, `sub`, `mul`, `truediv`, `floordiv`, `mod`
- Special: `abs`, `neg`, `sign`, `where`, `maximum`, `minimum`
- Reductions: `sum`, `max`, `min`, `prod`
- Type conversion: `to_dtype`

### What Needs Custom Implementation

| Component | Triton | Pallas | Notes |
|-----------|--------|--------|-------|
| Memory load | `tl.load(ptr + offset, mask=mask)` | `pl.load(ref, idx)` or direct indexing | Different masking model |
| Memory store | `tl.store(ptr + offset, val, mask=mask)` | `pl.store(ref, idx, val)` or direct indexing | Different masking model |
| Kernel decorator | `@triton.jit` | Pallas kernel function | Different structure |
| Kernel launch | `kernel[grid](..., num_warps=N)` | `pl.pallas_call(..., grid=...)` | Different API |
| Block indexing | `tl.program_id(axis)` | `pl.program_id(axis)` | Similar concept |
| Range creation | `tl.arange(0, BLOCK)` | `jnp.arange(BLOCK)` | Similar |

## Memory Access & Indexing Deep Dive

### Helion's Two-Level Indexing Architecture

Helion has a **two-level indexing model** that differs from Inductor's flat iteration model:

**Level 1: High-level tile indexing** (`indexing_strategy.py`)
- Converts PyTorch subscripts (`x[tile]`) to Triton load/store patterns
- `SubscriptIndexing.create()` generates offset and mask expressions
- Handles block IDs, block sizes, and boundary masking

```python
# SubscriptIndexing.create() generates expressions like:
offset = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offset < n_elements

# Then PointerIndexingStrategy.codegen_load() emits:
tl.load(ptr + offset, mask=mask)
```

**Level 2: Low-level pointwise ops** (`inductor_lowering.py`)
- `GenerateASTFromInductor.load()` handles ops within an already-sliced tile
- The `index` parameter is effectively ignored because slicing happened at Level 1
- Returns the tile tensor directly: `self.input_name_lookup[name]`

### Inductor Pallas Indexing Model

Inductor's `PallasKernel` uses a different approach:

**Single-level with iteration variables:**
```python
# Iteration variables defined as jnp.arange arrays
x0 = jnp.arange(dim0_size)
x1 = jnp.arange(dim1_size).reshape(1, -1)  # For broadcasting

# Memory access via direct indexing
value = input_ref[...]      # Contiguous: full array
value = input_ref[::2]      # Strided: every 2nd element
value = input_ref[1::2]     # Offset + strided
```

**Key methods in `PallasKernel` (`torch/_inductor/codegen/pallas.py`):**

| Method | Line | Purpose |
|--------|------|---------|
| `_get_index_str()` | 853 | Convert sympy index → JAX slice notation |
| `_convert_to_jax_slice()` | 884 | Pattern matching for stride/offset extraction |
| `_get_index_expr()` | 1024 | Determine if flattening is needed |
| `_generate_strided_index()` | 975 | Generate index array for complex patterns |
| `load()` | 1116 | Generate load expression with proper indexing |
| `store()` | 1353 | Generate store expression |

### Reusable Inductor Components

| Component | Location | Reusability | Notes |
|-----------|----------|-------------|-------|
| `PallasKernelOverrides` | `pallas.py:68` | **Direct** | Swap for `TritonOverrides` |
| `BlockPatternMatcher` | `simd.py` | **Direct** | Generic affine pattern analysis |
| `_convert_to_jax_slice()` logic | `pallas.py:884-963` | **Partial** | Pattern matching for stride/offset |
| `pexpr()` expression printer | `pallas.py` | **Direct** | Sympy → Python/JAX string |

**`_convert_to_jax_slice()` pattern matching (reusable):**
```python
# Extract stride and offset from sympy expressions
stride = BlockPatternMatcher.match_affine_block_expr(var_expr, var)
offset = index - var_expr

# Generate JAX slice notation:
# stride=1, offset=0  →  "..."
# stride=2, offset=0  →  "::2"
# stride=2, offset=1  →  "1::2"
```

### Mapping Helion's Model to Pallas

Helion already tracks the information needed for Pallas `BlockSpec`:

| Helion Concept | Variable | Pallas Equivalent |
|----------------|----------|-------------------|
| Block ID | `block_id` | Dimension in `BlockSpec` |
| Block size | `env.block_sizes[block_id].var` | `block_shape` tuple element |
| Block offset | `offset_var` = `pid * BLOCK_SIZE` | `index_map` lambda |
| Within-block index | `index_var` = `offset + tl.arange(...)` | Implicit in `BlockSpec` |
| Boundary mask | `mask_var` | Handled by `BlockSpec` boundary |

**Conversion strategy:**
```python
# Helion currently generates (for Triton):
pid = tl.program_id(0)
offset_0 = pid * BLOCK_SIZE_0
indices_0 = offset_0 + tl.arange(0, BLOCK_SIZE_0)
mask_0 = indices_0 < n_elements
x = tl.load(x_ptr + indices_0, mask=mask_0)

# For Pallas, this maps to:
# 1. BlockSpec declaration (at pallas_call level):
in_specs = [pl.BlockSpec(block_shape=(BLOCK_SIZE_0,), index_map=lambda i: (i,))]

# 2. Kernel body (simple ref access):
x = x_ref[...]  # BlockSpec handles the tiling
```

### Recommended Implementation: PallasIndexingStrategy

Create a new indexing strategy that converts Helion's tile info to Pallas patterns:

```python
# helion/_compiler/indexing_strategy.py

class PallasIndexingStrategy(IndexingStrategy):
    """Generate Pallas ref indexing instead of Triton pointer arithmetic."""

    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
        eviction_policy: ast.AST | None,
    ) -> ast.AST:
        indexing = PallasSubscriptIndexing.create(state, fake_tensor, subscript)
        name = state.device_function.tensor_arg(fake_tensor).name
        # Generate ref[...] or ref[slice] instead of tl.load(ptr + offset, mask)
        return expr_from_string(f"{name}[{{index}}]", index=indexing.jax_slice_expr)

    def codegen_store(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        indexing = PallasSubscriptIndexing.create(state, fake_tensor, subscript)
        name = state.device_function.tensor_arg(fake_tensor).name
        return expr_from_string(f"{name}[{{index}}] = {{value}}",
                                index=indexing.jax_slice_expr, value=value)


class PallasSubscriptIndexing(NamedTuple):
    """Convert Helion subscripts to JAX slice expressions."""
    jax_slice_expr: ast.AST
    block_specs: list[tuple[int, str]]  # (block_id, index_map_expr)

    @staticmethod
    def create(state, fake_tensor, subscript) -> PallasSubscriptIndexing:
        # Reuse logic from Inductor's _convert_to_jax_slice:
        # - Contiguous access → "..."
        # - Strided access → "::stride" or "offset::stride"
        # - Complex patterns → explicit index array

        # Also collect BlockSpec info for pallas_call generation
        pass
```

### BlockSpec Generation

The kernel launcher needs to generate `BlockSpec` from Helion's tile info:

```python
# helion/_compiler/device_function.py (Pallas path)

def _generate_pallas_block_specs(self) -> tuple[list[str], list[str]]:
    """Generate in_specs and out_specs for pallas_call."""
    in_specs = []
    out_specs = []

    for tensor_arg in self.input_tensors:
        block_shape = self._get_block_shape(tensor_arg)
        index_map = self._get_index_map(tensor_arg)
        in_specs.append(f"pl.BlockSpec({block_shape}, {index_map})")

    for tensor_arg in self.output_tensors:
        block_shape = self._get_block_shape(tensor_arg)
        index_map = self._get_index_map(tensor_arg)
        out_specs.append(f"pl.BlockSpec({block_shape}, {index_map})")

    return in_specs, out_specs

def _get_block_shape(self, tensor_arg) -> str:
    """Convert Helion block sizes to BlockSpec block_shape tuple."""
    # Helion tracks: env.block_sizes[block_id].var for each tiled dimension
    # Convert to: (BLOCK_SIZE_0, BLOCK_SIZE_1, ...)
    pass

def _get_index_map(self, tensor_arg) -> str:
    """Generate index_map lambda from Helion's tile strategy."""
    # Helion's offset calculation: pid * BLOCK_SIZE
    # Becomes: lambda i, j: (i, j) for 2D tiling
    pass
```

## Proposed Implementation

### Minimal Change Approach

**Step 1: Parameterize ops handler**

```python
# helion/_compiler/inductor_lowering.py

class GenerateASTFromInductor(DefaultHandler):
    def __init__(self, cg, input_name_lookup, backend="triton"):
        super().__init__()
        if backend == "pallas":
            from torch._inductor.codegen.pallas import PallasKernelOverrides
            self.parent_handler = PallasKernelOverrides()
        else:
            from torch._inductor.codegen.triton import TritonOverrides
            self.parent_handler = TritonOverrides()
        self.backend = backend
        self.cg = cg
        self.input_name_lookup = input_name_lookup
```

**Step 2: Add Pallas-specific imports**

```python
# helion/_compiler/output_header.py

library_imports_triton = {
    "triton": "import triton",
    "tl": "import triton.language as tl",
    # ...
}

library_imports_pallas = {
    "jax": "import jax",
    "jnp": "import jax.numpy as jnp",
    "pl": "import jax.experimental.pallas as pl",
    "lax": "from jax import lax",
    # ...
}
```

**Step 3: Backend-specific kernel structure**

```python
# helion/_compiler/device_function.py

def codegen_device_function(self):
    if self.backend == "pallas":
        return self._codegen_pallas_kernel()
    return self._codegen_triton_kernel()

def _codegen_pallas_kernel(self):
    # Generate Pallas kernel structure
    # No decorator needed - Pallas kernels are regular functions
    # wrapped by pl.pallas_call()
    pass
```

**Step 4: Backend-specific memory ops**

Override `load` and `store` methods for Pallas:

```python
# In GenerateASTFromInductor or a subclass

def load(self, name: str, index: sympy.Expr) -> str:
    if self.backend == "pallas":
        # Pallas uses direct array indexing
        return f"{name}[...]"  # or appropriate slice
    # Triton path
    return self.cg.lift(self.input_name_lookup[name]).id
```

### File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `_compiler/inductor_lowering.py` | Modify | Add backend parameter, swap `TritonOverrides` → `PallasKernelOverrides` |
| `_compiler/output_header.py` | Modify | Add Pallas imports (`jax`, `jnp`, `pl`) |
| `_compiler/device_function.py` | Modify | Add `_codegen_pallas_kernel()`, `_generate_pallas_block_specs()` |
| `_compiler/tile_strategy.py` | Modify | Replace `tl.arange` → `jnp.arange`, `tl.program_id` → `pl.program_id` |
| `_compiler/indexing_strategy.py` | Modify | Add `PallasIndexingStrategy`, `PallasSubscriptIndexing` classes |
| `_compiler/reduction_strategy.py` | Modify | Replace `tl.reduce` → Pallas reduction patterns |
| `runtime/kernel.py` | Modify | Add `backend` parameter to compilation |
| `runtime/pallas_runner.py` | New | Pallas kernel execution via `pl.pallas_call()` |

## Kernel Structure Comparison

### Triton Output (Current)

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

# Launch
add_kernel[(n_elements + BLOCK - 1) // BLOCK](x, y, out, n_elements, BLOCK_SIZE=1024)
```

### Pallas Output (Target)

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, out_ref):
    x = x_ref[...]
    y = y_ref[...]
    out_ref[...] = x + y

# Launch
out = pl.pallas_call(
    add_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid=(n_blocks,),
    in_specs=[pl.BlockSpec(block_shape, lambda i: (i,))],
    out_specs=pl.BlockSpec(block_shape, lambda i: (i,)),
)(x, y)
```

## Tiling Model Comparison

Both Triton and Pallas use similar tiling concepts:

| Concept | Triton | Pallas |
|---------|--------|--------|
| Grid of blocks | `kernel[grid_size]` | `grid=(grid_size,)` |
| Block ID | `tl.program_id(axis)` | `pl.program_id(axis)` |
| Block shape | `BLOCK_SIZE: tl.constexpr` | `BlockSpec(block_shape, ...)` |
| Index mapping | Manual offset calculation | `index_map` lambda |

Helion's `hl.tile()` abstraction maps naturally to both:
- For Triton: generates `tl.program_id` + offset calculations
- For Pallas: can generate `BlockSpec` with appropriate `index_map`

## Next Steps

1. **Prototype ops handler swap** - Verify `PallasKernelOverrides` works with Helion's AST generation
2. **Implement Pallas memory access** - Handle load/store differently
3. **Generate Pallas kernel structure** - Different from `@triton.jit`
4. **Add runtime execution** - JAX-based kernel execution
5. **Test with simple kernels** - Start with elementwise add/mul

## Open Questions

1. **BlockSpec generation**: ✅ Addressed above - Helion already tracks block_id/block_size, convert to `BlockSpec(block_shape, index_map)`
2. **Masking**: ✅ Addressed - Pallas `BlockSpec` handles boundary implicitly; no explicit masks needed for aligned tiled access
3. **Reductions**: Pallas reduction API differs from Triton's `tl.reduce` - needs investigation
   - Triton: `tl.reduce(input, axis, combine_fn)`
   - Pallas: `jnp.sum(x, axis=...)` or `lax.reduce(...)`
4. **Autotuning**: Different parameters than Triton (no `num_warps`, `num_stages`)
   - TPU-specific: memory layout, pipeline depth
   - May need separate tuning infrastructure
5. **Non-contiguous access**: How to handle strided/gather patterns?
   - Inductor uses `buf[...].flatten()[index_array]` for complex patterns
   - May need similar approach in Helion's Pallas path
6. **Multi-dimensional tiling**: Helion supports N-D tiles
   - `BlockSpec` supports multi-dim via `index_map=lambda i, j: (i, j)`
   - Need to verify Helion's tile info maps correctly
