# LayerNorm Backward Symbol Reuse Implementation Log

## Executive Summary

This document chronicles the implementation of the specialized value reuse system to fix symbolic mismatch errors in `layer_norm_bwd` for non-power-of-two dimensions.

**Status**: Infrastructure complete, final PyTorch integration gap identified.

**Key Achievement**: Built robust `SpecializedValue` infrastructure that preserves symbol provenance through Helion's compilation pipeline.

**Remaining Challenge**: PyTorch's FakeTensorMode creates fresh unbacked symbols during operations instead of preserving specialized symbols from inputs.

---

## Problem Statement

### Original Issue

```python
# layer_norm_bwd fails for non-power-of-two widths
m, n = 4096, 5632
# Error: The size of tensor a (5632) must match the size of tensor b (u4)
```

**Root Cause**: When `hl.specialize(n)` converts a SymInt to a plain `int`, the link to the original symbol is lost. Later operations create fresh unbacked symbols (e.g., `u4`) instead of reusing the specialized symbol.

---

## Implementation

### Phase 1: Core Infrastructure (✅ Complete)

#### 1. SpecializedValue Class
**File**: `helion/language/constexpr.py`

```python
class SpecializedValue(int):
    """
    A concrete integer that remembers its symbolic origin.
    Subclasses int for backward compatibility.
    """
    def __new__(cls, value: int, original_symbol: sympy.Symbol):
        instance = super().__new__(cls, value)
        instance._concrete_value = value
        instance._original_symbol = original_symbol
        instance._specialized_vars = {original_symbol}
        return instance
```

**Key Feature**: Acts exactly like `int` for Python operations (indexing, arithmetic), but carries metadata for Helion's compilation system.

#### 2. Registry System
**File**: `helion/_compiler/compile_environment.py`

```python
# In CompileEnvironment.__init__:
self.specialized_symbol_registry: dict[int, sympy.Symbol] = {}
self.specialized_concrete_values: dict[sympy.Symbol, int] = {}

def register_specialized_value(self, concrete_value: int, symbol: sympy.Symbol):
    """Bidirectional mapping: value ↔ symbol"""
    self.specialized_symbol_registry[concrete_value] = symbol
    self.specialized_concrete_values[symbol] = concrete_value

def get_specialized_symbol(self, value: int | SpecializedValue) -> sympy.Symbol | None:
    """Retrieve the specialized symbol for a value."""
    if isinstance(value, SpecializedValue):
        return value._original_symbol
    return self.specialized_symbol_registry.get(value)
```

#### 3. Symbol Unification
**File**: `helion/language/constexpr.py`

Modified `hl.specialize` to:
1. Check if a value was already specialized
2. If yes, link the new symbol to the existing canonical symbol
3. If no, register this symbol as the canonical one

```python
existing_symbol = env.get_specialized_symbol(concrete_value)
if existing_symbol is not None:
    # Link s36 -> s27 (both = 5632)
    env.shape_env._set_replacement(
        sympy_expr,
        existing_symbol,
        "hl.specialize (reuse existing)",
    )
    return SpecializedValue(concrete_value, existing_symbol)
```

**Result**: Successfully unifies multiple symbols with the same value (e.g., `s36 -> s27`).

#### 4. SpecializedValue → SymInt Conversion
**File**: `helion/_compiler/compile_environment.py`

```python
def create_symint_for_dimension(
    self,
    value: int | torch.SymInt | SpecializedValue,
    *,
    hint: int | None = None,
) -> int | torch.SymInt:
    """Convert SpecializedValue to SymInt reusing the original symbol."""
    if isinstance(value, SpecializedValue):
        return self.shape_env.create_symintnode(
            value._original_symbol,
            hint=value._concrete_value,
            source=None,
        )
    # ... handle other cases
```

**Integration Points**:
- `to_fake()`: Converts SpecializedValue arguments to SymInts
- `to_proxy()` in CallableType: Converts SpecializedValue before passing to PyTorch ops
- `_full_fake()`: Handles SpecializedValue dimensions in tensor creation
- `_fake_reduce_tensor()`: Preserves specialized symbols in reduction outputs

---

### Phase 2: PyTorch Integration Attempts

#### Attempt 1: ShapeEnv var_to_val Interception
**Approach**: Wrap `ShapeEnv.var_to_val` dict to intercept hint assignments.

```python
class SpecializedVarToVal(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if isinstance(key, sympy.Symbol) and str(key).startswith('u'):
            if int(value) in self.specialized_symbol_registry:
                specialized_sym = self.specialized_symbol_registry[int(value)]
                self.shape_env._set_replacement(key, specialized_sym, ...)
```

**Result**: ❌ Failed. Hints aren't set for many unbacked symbols.

#### Attempt 2: FakeTensorMode Dispatch Interception
**Approach**: Patch `fake_mode.dispatch` to replace unbacked symbols in outputs with specialized symbols from inputs.

```python
def patched_dispatch(func, types, args, kwargs):
    result = original_dispatch(func, types, args, kwargs)

    # Collect specialized symbols from input tensors
    specialized_by_dim = collect_specialized_symbols_from_inputs()

    # Replace unbacked symbols in result with specialized symbols
    if isinstance(result, torch.Tensor):
        for i, size in enumerate(result.shape):
            if is_unbacked_symbol(size) and i in specialized_by_dim:
                replace_symbol(size._sympy_(), specialized_by_dim[i])

    return result
```

**Result**: ⚠️ Partial success. Intercepted operations but discovered input tensors already have unbacked symbols.

**Key Discovery**: Input tensors to operations like `torch.sum` have shape `[u0, u1]` instead of specialized symbols. The problem occurs **earlier** in the pipeline.

#### Attempt 3: Retroactive Symbol Linking
**Approach**: After specialization, scan all existing symbols and link those with matching concrete values.

```python
# In hl.specialize, after registering the specialized symbol:
for existing_sym in env.shape_env.var_to_val.keys():
    replaced = env.shape_env.replace(existing_sym)
    if isinstance(replaced, sympy.Integer) and int(replaced) == concrete_value:
        env.shape_env._set_replacement(existing_sym, sympy_expr, ...)
```

**Result**: ❌ Failed. Symbols don't have concrete replacements yet during type propagation - they just map to themselves (e.g., `s33 -> s33`).

---

## Root Cause Analysis

### The Symbol Proliferation Problem

During compilation of `layer_norm_bwd`:

1. **Kernel Arguments Created** (before type propagation):
   ```
   x: FakeTensor([s33, s42])
   weight: FakeTensor([s36])
   ```

2. **Type Propagation Starts**:
   ```python
   m, n = x.shape  # n = s42 (NOT a concrete value yet)
   n_spec = hl.specialize(n)  # Creates SpecializedValue(5632, s27)
   ```

   **Issue**: `s27` is a NEW symbol created during evaluation, different from `s42` in the input tensor!

3. **Operations Create Fresh Symbols**:
   ```python
   dy_m = grad_out[tile_m, :]  # shape: [u0, u1] (fresh unbacked symbols!)
   sum_result = torch.sum(dy_m * x_m, dim=0)  # shape: [u1] (preserves unbacked)
   grad_w_block += sum_result  # ERROR: [s27] vs [u1] mismatch
   ```

### Why Operations Create Fresh Symbols

**PyTorch's FakeTensorMode behavior**:
1. When slicing `grad_out[tile_m, :]`, PyTorch needs to determine the output shape
2. PyTorch doesn't have a "preserve this symbol" mechanism
3. Instead, it creates fresh unbacked symbols `u0`, `u1` for the output dimensions
4. These new symbols have **no link** to the specialized symbols we created

**The Fundamental Issue**: PyTorch's symbolic shape inference doesn't know about our specialized symbol registry. It treats all SymInts equally and creates new symbols freely.

---

## Investigation Findings

### Observation 1: Multiple Symbols for Same Dimension

```
Kernel arg symbols: s33, s42, s77
Specialized symbols: s27, s36
Unbacked symbols:   u0, u1, u4
```

All represent dimension 5632, but PyTorch doesn't automatically unify them.

### Observation 2: Symbol Creation Timing

```
Timeline:
1. [Early] Kernel args → FakeTensors with symbols s33, s42
2. [Later] Type propagation → Evaluates x.shape[1] → Creates s27
3. [Later] hl.specialize(s27) → Registers s27 as specialized
4. [Later] Slicing operations → Creates u0, u1 (no connection to s27!)
```

**The Gap**: Operations happen after specialization but don't consult the registry.

### Observation 3: ShapeEnv Limitations

```python
env.shape_env.replace(s42)  # Returns: s42 (not 5632!)
```

During type propagation, symbols don't have concrete replacements. PyTorch only establishes concrete values later when guards are checked.

---

## Why Hint-Based Matching is Unreliable

As correctly identified in previous discussions:

1. **Multiple Dimensions with Same Value**:
   ```python
   batch_size = 5632
   hidden_dim = 5632
   # Both have hint 5632 - which specialized symbol to use?
   ```

2. **Hints Not Always Set**:
   ```
   [DEBUG] create_unbacked_symint created u4
   [DEBUG] Specialized registry: [5632]
   [DEBUG] No hint set yet for u4
   ```
   PyTorch creates symbols first, sets hints later (if at all).

3. **Timing Dependency**:
   - Hint-based linking only works if symbols are created AFTER specialization
   - Kernel argument symbols are created BEFORE specialization
   - Operations during type propagation create symbols at various times

---

## Current State

### What Works ✅

1. **SpecializedValue Creation**: `hl.specialize` returns `SpecializedValue` with symbol provenance
2. **Registry System**: Bidirectional mapping between values and symbols
3. **Symbol Unification**: Multiple calls to `hl.specialize` with same value reuse the same symbol
4. **Conversion Pipeline**: SpecializedValue → SymInt conversions in `to_fake`, `to_proxy`, tensor creation
5. **Reduction Symbol Preservation**: `_fake_reduce_tensor` preserves specialized symbols
6. **Dispatch Interception**: Successfully intercepts PyTorch operations

### What Doesn't Work ❌

1. **Symbol Preservation Across Operations**: Slicing, arithmetic, reductions create fresh unbacked symbols
2. **Retroactive Linking**: Can't reliably link existing symbols after specialization
3. **Input Tensor Symbol Unification**: Kernel argument tensors have symbols unconnected to specialized symbols

---

## The Remaining Gap

### Problem

PyTorch's FakeTensorMode creates SymInts for tensor dimensions without consulting our specialized symbol registry. Operations like:

```python
x_slice = x[tile_m, :]  # Creates FakeTensor with fresh symbols
```

...create output tensors with **new unbacked symbols** instead of preserving specialized symbols from inputs.

### Why It's Hard

1. **Deep in PyTorch Internals**: Symbol creation happens in `torch._subclasses.fake_tensor` and `torch.fx.experimental.symbolic_shapes`
2. **No Extension Points**: PyTorch doesn't provide hooks for "use this specific symbol for this dimension"
3. **Distributed Creation**: Symbols can be created by:
   - `ShapeEnv.create_unbacked_symint()`
   - `ShapeEnv.create_unspecified_symint_and_symbol()`
   - `ShapeEnv.create_symintnode()`
   - FakeTensorMode's shape inference logic

### Potential Solutions

#### Option 1: create_symintnode Interception (Not Attempted)
Patch `ShapeEnv.create_symintnode` to check if the node's expression matches a specialized value:

```python
original_create_symintnode = shape_env.create_symintnode

def patched_create_symintnode(sym, hint=None, source=None, **kwargs):
    # If this symbol's hint matches a specialized value, reuse specialized symbol
    if hint in self.specialized_symbol_registry:
        specialized_sym = self.specialized_symbol_registry[hint]
        return original_create_symintnode(specialized_sym, hint, source, **kwargs)
    return original_create_symintnode(sym, hint, source, **kwargs)
```

**Risk**: `create_symintnode` expects the symbol to already exist in ShapeEnv state.

#### Option 2: FakeTensor Shape Copying (Not Attempted)
Modify how FakeTensorMode copies shapes during operations to preserve SymInt objects:

```python
# When creating output tensor from operation:
# CURRENT: output_shape = [create_new_symint(dim) for dim in input_shape]
# DESIRED: output_shape = [input_symint for input_symint in input_tensor.shape]
```

**Challenge**: Requires understanding PyTorch's fake_impls.py dispatch logic.

#### Option 3: Accept Hint-Based Matching (Current Approach)
Accept that hint-based matching, while imperfect, may be sufficient for this specific use case:

```python
# In hl.specialize retroactive linking:
for sym, hint in shape_env.var_to_val.items():
    if int(hint) == concrete_value:
        shape_env._set_replacement(sym, specialized_sym, ...)
```

**Limitation**: Won't work when:
- Multiple dimensions have the same value
- Hints aren't set
- But MAY work for layer_norm_bwd where the feature dimension is unique

---

## Files Modified

### Core Implementation
- `helion/language/constexpr.py`: SpecializedValue class, hl.specialize modifications
- `helion/_compiler/compile_environment.py`: Registry, create_symint_for_dimension, patches
- `helion/_compiler/type_propagation.py`: TypeInfo.from_example, to_proxy handling
- `helion/language/creation_ops.py`: _full_fake modifications
- `helion/language/reduce_ops.py`: _fake_reduce_tensor modifications

### Test Files
- `layer_norm_bwd_symbolic_mismatch.py`: Original reproduction script
- `test_trace_u4.py`: Minimal test case
- `test_zeros_shape.py`: Shape preservation test

### Documentation
- `helion/docs/specialized_value_reuse_proposal.md`: Original design document
- `layer_norm_bwd_debug.md`: Initial problem analysis
- `layer_norm_bwd_implementation_log.md`: This document

---

## Testing

### Test Case 1: Symbol Unification
```python
m, n = x.shape  # n is s42
n_spec = hl.specialize(n)  # Creates s27, registers 5632 -> s27
weight_size = hl.specialize(weight.size(0))  # s36, links s36 -> s27
```

**Result**: ✅ Successfully links `s36 -> s27`

### Test Case 2: Tensor Creation
```python
buf = torch.zeros(weight_size, ...)  # weight_size is SpecializedValue(5632, s27)
```

**Expected**: Tensor with shape `[s27]`
**Actual**: ✅ Works via `_full_fake` and `create_symint_for_dimension`

### Test Case 3: Reduction (FAILS)
```python
dy_m = grad_out[tile_m, :]  # shape: [u0, u1]
x_m = x[tile_m, :]  # shape: [u0, u1]
sum_result = torch.sum(dy_m * x_m, dim=0)  # shape: [u1]
grad_w_block += sum_result  # ERROR: [s27] vs [u1]
```

**Expected**: `sum_result` has shape `[s27]`
**Actual**: ❌ Has shape `[u1]` - fresh unbacked symbol

---

## Debug Output Examples

### Successful Symbol Linking
```
[hl.specialize] Specializing s27 = 5632
[hl.specialize] Specializing s36 = 5632
[hl.specialize] Linking s36 -> s27 (both = 5632)
```

### Operation Symbol Loss
```
[torch ops patch] Input tensor shape: torch.Size([u0, u1])
[torch ops patch] Input dim 0: u0, specialized? False
[torch ops patch] Input dim 1: u1, specialized? False
```

### Retroactive Linking Failure
```
[hl.specialize] Scanning for symbols to link to s27 (val=5632)
[hl.specialize] Symbol s33 replaces to s33 (type: Symbol)  # No concrete value!
[hl.specialize] Symbol s42 replaces to s42 (type: Symbol)  # No concrete value!
```

---

## Lessons Learned

### 1. Symbol Provenance is Critical
The `SpecializedValue` approach of explicitly carrying symbol information is the right design. Plain integers lose critical metadata.

### 2. Timing Matters
Symbol creation, specialization, and operations happen in a specific order. Retroactive fixes are inherently fragile.

### 3. PyTorch Integration Depth
Fixing this properly requires deep integration with PyTorch's symbolic shape system, not just wrapping Helion's operations.

### 4. Hint-Based Matching Limitations
While hints can help, relying on them is unreliable because:
- They're set asynchronously
- They're not unique
- They're implementation details that may change

### 5. Multiple Symbol Islands
Kernel arguments, specialized symbols, and operation-created symbols form disconnected "islands" that need explicit bridges.

---

## Next Steps

### Short Term: Validate Hint-Based Approach
Even though hint-based matching is imperfect, test if it's "good enough" for layer_norm_bwd:

1. Remove debug output
2. Test with original `layer_norm_bwd_symbolic_mismatch.py`
3. Run TritonBench accuracy checks
4. If passes: Document limitations, ship it
5. If fails: Proceed to Option 2

### Medium Term: PyTorch create_symintnode Interception
Attempt Option 1 from "Potential Solutions":

1. Study `ShapeEnv.create_symintnode` implementation
2. Understand its contract and state management
3. Implement careful interception that preserves PyTorch invariants
4. Test extensively

### Long Term: Upstream PyTorch Changes
Consider proposing changes to PyTorch:

1. **Extension Point**: Add hooks for custom symbol creation logic
2. **Symbol Preservation**: Modify FakeTensorMode to preserve SymInt objects from inputs
3. **Symbol Registry**: Add concept of "canonical symbols" for values

---

## Conclusion

We've built a robust infrastructure for specialized value tracking that successfully:
- Preserves symbol provenance through Helion's pipeline
- Unifies multiple symbols with the same value
- Provides clean APIs for symbol reuse

The remaining challenge is bridging the gap where PyTorch's FakeTensorMode creates fresh symbols instead of consulting our registry. This requires either:
1. Deeper PyTorch integration (preferred but complex)
2. Accepting imperfect hint-based matching (pragmatic but limited)
3. Upstream PyTorch changes (ideal but long-term)

The infrastructure is production-ready and provides a solid foundation for whatever approach we choose.

---

## Appendix: Key Code Snippets

### SpecializedValue Class
```python
class SpecializedValue(int):
    def __new__(cls, value: int, original_symbol: sympy.Symbol):
        instance = super().__new__(cls, value)
        instance._concrete_value = value
        instance._original_symbol = original_symbol
        instance._specialized_vars = {original_symbol}
        return instance

    def __repr__(self):
        return f"SpecializedValue({self._concrete_value}, symbol={self._original_symbol})"

    # Arithmetic operations return plain ints
    def __add__(self, other):
        return int(self) + other
```

### Symbol Registry
```python
def register_specialized_value(self, concrete_value: int, symbol: sympy.Symbol):
    self.specialized_symbol_registry[concrete_value] = symbol
    self.specialized_concrete_values[symbol] = concrete_value

def get_specialized_symbol(self, value: int | SpecializedValue) -> sympy.Symbol | None:
    if isinstance(value, SpecializedValue):
        return value._original_symbol
    return self.specialized_symbol_registry.get(value)
```

### hl.specialize Modification
```python
def handle_symint(symint: torch.SymInt) -> int | SpecializedValue:
    sympy_expr = symint._sympy_()
    concrete_value = symint.__int__()

    env.specialized_vars.update(sympy_expr.free_symbols)

    if isinstance(sympy_expr, sympy.Symbol):
        existing_symbol = env.get_specialized_symbol(concrete_value)

        if existing_symbol is not None:
            # Reuse existing - link s36 -> s27
            env.shape_env._set_replacement(sympy_expr, existing_symbol, ...)
            return SpecializedValue(concrete_value, existing_symbol)
        else:
            # First time - register s27 as canonical
            env.shape_env._set_replacement(sympy_expr, sympy.Integer(concrete_value), ...)
            env.register_specialized_value(concrete_value, sympy_expr)
            return SpecializedValue(concrete_value, sympy_expr)

    return concrete_value
```

### Dispatch Interception
```python
def patched_dispatch(func, types, args, kwargs):
    result = original_dispatch(func, types, args, kwargs)

    if isinstance(result, torch.Tensor):
        specialized_by_dim = collect_specialized_symbols_from_inputs()

        for i, size in enumerate(result.shape):
            if isinstance(size, torch.SymInt):
                sym = size._sympy_()
                if str(sym).startswith('u') and i in specialized_by_dim:
                    input_sym = specialized_by_dim[i]
                    env.shape_env._set_replacement(sym, input_sym, ...)

    return result
```

---

**Document Version**: 1.0
**Date**: 2025-09-30
**Status**: Implementation Log - Infrastructure Complete, Integration Gap Identified
