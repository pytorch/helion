# Investigation Report: Symbolic Slices with Different Pointer Types in Helion

## Executive Summary

After thorough investigation of the Helion codebase, I found that symbolic slices (created with `hl.slice()` or through automatic conversion of slices with SymInt bounds) are currently **only supported with pointer indexing**. Both `block_ptr` and `tensor_descriptor` indexing strategies fall back to pointer indexing when encountering symbolic slices.

## Key Findings

### 1. SliceProxy Implementation
- `SliceProxy` is a tensor subclass that stores symbolic slice bounds (start, stop, step)
- Bounds are stored in `CompileEnvironment.slice_bounds` to avoid FX tracing issues
- `SliceProxy` objects are created when slices contain SymInt values (e.g., from `register_block_size()`)

### 2. Indexing Strategy Support

#### Pointer Indexing (`indexing="pointer"`)
- **Full support** for symbolic slices via `SubscriptIndexing.create()`
- Handles both `slice` and `SliceProxy` objects
- Generates explicit index calculations with symbolic bounds
- Example code generation:
  ```python
  # For slice [0:block_size_m]
  indices_2 = tl.arange(0, _RDIM_SIZE_2).to(tl.int32)
  mask_2 = indices_2 < _BLOCK_SIZE_0
  ```

#### Block Pointer Indexing (`indexing="block_ptr"`)
- **No support** for symbolic slices
- `BlockedSubscriptIndexing.create()` only handles regular `slice` objects, not `SliceProxy`
- Falls back to `PointerIndexingStrategy` when `SliceProxy` is detected
- Strided slices (step != 1) are explicitly rejected with an error

#### Tensor Descriptor Indexing (`indexing="tensor_descriptor"`)
- **No support** for symbolic slices
- Uses same `BlockedSubscriptIndexing` infrastructure as block_ptr
- Falls back to `PointerIndexingStrategy` when symbolic slices are detected
- Additional restrictions: requires 16-byte aligned strides, 2-5 dimensions, etc.

### 3. Code Path Analysis

The key issue is in `BlockedSubscriptIndexing.create()` (line 812-829):
```python
elif isinstance(k, slice):
    # Only handles regular slice objects
    size = fake_value.size(len(res.offsets))
    if k.step is not None and k.step != 1:
        raise exc.InvalidIndexingType(
            f"Strided slices not supported in block_ptr mode: {k}"
        )
    # ... rest of handling
else:
    raise exc.InvalidIndexingType(k)  # SliceProxy hits this branch
```

This code doesn't have a case for `SliceProxy`, so it raises `InvalidIndexingType`.

### 4. Fallback Behavior

Both `BlockPtrIndexingStrategy` and `TensorDescriptorIndexingStrategy` check support using `BlockedSubscriptIndexing.is_supported()` and fall back gracefully:

```python
def codegen_load(self, state, fake_tensor, subscript, extra_mask):
    if not BlockedSubscriptIndexing.is_supported(state, fake_tensor, subscript, extra_mask):
        return PointerIndexingStrategy().codegen_load(state, fake_tensor, subscript, extra_mask)
    # ... block_ptr specific code
```

### 5. Test Coverage

The test suite shows:
- Extensive tests for symbolic slices with pointer indexing
- Tests like `test_range_slice_with_block_size_var` demonstrate working symbolic slices
- No tests found for symbolic slices with block_ptr or tensor_descriptor indexing

## Implications

1. **Performance**: Applications using symbolic slices cannot benefit from block_ptr or tensor_descriptor optimizations
2. **Consistency**: Different indexing strategies have different feature support
3. **User Experience**: No error messages inform users about the fallback behavior

## Recommendations

To add symbolic slice support to block_ptr and tensor_descriptor indexing:

1. Modify `BlockedSubscriptIndexing.create()` to handle `SliceProxy`:
   ```python
   elif isinstance(k, (slice, SliceProxy)):
       slice_obj = k.to_slice() if isinstance(k, SliceProxy) else k
       # ... rest of handling
   ```

2. Update `BlockedSubscriptIndexing.is_supported()` to check for `SliceProxy`

3. Add test cases for symbolic slices with all indexing strategies

4. Consider whether strided symbolic slices should be supported in block_ptr mode

## Test Results

From running test code:
- Pointer indexing: Successfully generates code with symbolic slice handling
- Block_ptr indexing: Falls back to pointer indexing (identical generated code)
- Tensor descriptor: Falls back to pointer indexing
- Strided slices: Work with pointer indexing, would error with block_ptr if not for fallback

This investigation confirms that symbolic slice support is currently limited to pointer indexing mode only.