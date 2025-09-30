# Robust Solution Proposal: Symbol Reuse in Reductions

## Executive Summary

This document proposes a systematic fix for the symbolic mismatch issue in Helion's layer normalization backward pass (and similar operations). The core problem is that `hl.specialize()` converts SymInts to plain Python ints, losing provenance information that causes fresh unbacked symbols to be created during reductions, leading to shape mismatches like `5632 vs u4`.

## Problem Analysis

### Current Behavior

```python
n = x.size(1)              # n is SymInt(s27) with hint 5632
n = hl.specialize(n)       # n becomes int(5632) - LOSES SYMBOL!
buf = torch.zeros(n, ...)  # Creates fresh SymInt(u4) with hint 5632
result = torch.sum(buf)    # Creates SymInt(u5), sees mismatch: 5632 (int) vs u4 (SymInt)
```

### Root Cause

**Information Loss**: When `hl.specialize` converts a SymInt to a plain Python `int`, the symbolic provenance is lost. Later operations that need to create SymInts for that dimension have no way to know it was specialized, so they create fresh unbacked symbols.

### Why This Matters

- **Non-power-of-two feature sizes fail**: TritonBench accuracy checks fail for layer_norm_bwd with sizes like 5120, 5632, etc.
- **FakeTensor constraint errors**: Shape mismatches abort compilation before kernels can even run
- **Inconsistent symbolic information**: Some dimensions are concrete ints, others are unbacked SymInts with the same hint

## Proposed Solution: Multi-Layered Approach

The solution requires changes at multiple levels of Helion's compilation stack to systematically preserve and reuse specialized symbols.

### Layer 1: Specialized Value Type (Preserve Provenance)

Create a wrapper type that acts like an int but preserves the original symbol.

**File**: `helion/language/constexpr.py`

```python
class SpecializedValue(int):
    """
    A concrete integer value that remembers its symbolic origin.

    Behaves exactly like int for Python operations (indexing, arithmetic, etc.)
    but carries metadata for Helion's compilation system.

    Examples:
        >>> n = hl.specialize(x.size(1))  # Returns SpecializedValue(5632, s27)
        >>> n == 5632                      # True
        >>> n + 1                          # 5633 (plain int)
        >>> buf = torch.zeros(n, ...)      # Helion can detect this came from specialization
    """
    __slots__ = ('_concrete_value', '_original_symbol', '_specialized_vars')

    def __new__(cls, value: int, original_symbol: sympy.Symbol):
        instance = super().__new__(cls, value)
        return instance

    def __init__(self, value: int, original_symbol: sympy.Symbol):
        self._concrete_value = value
        self._original_symbol = original_symbol
        self._specialized_vars = {original_symbol}

    def __repr__(self):
        return f"SpecializedValue({self._concrete_value}, symbol={self._original_symbol})"

    def __reduce__(self):
        """Support for pickling - serialize as plain int."""
        return (int, (self._concrete_value,))

    # Arithmetic operations return plain ints for safety
    # (we can't guarantee provenance after arithmetic)
    def __add__(self, other):
        return int(self) + other

    def __mul__(self, other):
        return int(self) * other

    # Add other arithmetic operations as needed...
```

**Modify `hl.specialize` to return `SpecializedValue`:**

```python
@_decorators.type_propagation(specialize)
def _(value: TypeInfo, *, origin: Origin) -> TypeInfo:
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.type_propagation import TypeInfo
    import sympy

    if origin.is_device():
        raise exc.SpecializeOnDevice

    proxy = value.proxy()
    env = CompileEnvironment.current()

    def handle_symint(symint: torch.SymInt) -> int | SpecializedValue:
        sympy_expr = symint._sympy_()
        concrete_value = symint.__int__()

        # Record specialized variables
        env.specialized_vars.update(sympy_expr.free_symbols)

        # Set replacement in ShapeEnv
        if isinstance(sympy_expr, sympy.Symbol):
            env.shape_env._set_replacement(
                sympy_expr,
                sympy.Integer(concrete_value),
                "hl.specialize"
            )
            env.shape_env.var_to_val[sympy_expr] = sympy.Integer(concrete_value)

            # Register in the environment
            env.register_specialized_value(concrete_value, sympy_expr)

            # Return SpecializedValue instead of plain int
            return SpecializedValue(concrete_value, sympy_expr)

        return concrete_value

    specialized = _convert_specializable(proxy, on_symint=handle_symint)
    return TypeInfo.from_example(specialized, origin=origin)
```

### Layer 2: CompileEnvironment Integration (Symbol Registry)

Maintain a registry of specialized symbols that can be queried throughout compilation.

**File**: `helion/_compiler/compile_environment.py`

```python
class CompileEnvironment:
    def __init__(self, device: torch.device, settings: Settings):
        # ... existing init ...

        # Maps concrete values to their specialized symbols
        self.specialized_symbol_registry: dict[int, sympy.Symbol] = {}

        # Maps SymPy symbols to their concrete specialized values
        self.specialized_concrete_values: dict[sympy.Symbol, int] = {}

    def register_specialized_value(self, concrete_value: int, symbol: sympy.Symbol):
        """
        Register a specialized dimension for symbol reuse.

        This creates a bidirectional mapping between concrete values and their
        original specialized symbols, enabling later operations to reuse the
        correct symbol instead of creating fresh unbacked ones.

        Args:
            concrete_value: The concrete integer value (e.g., 5632)
            symbol: The original symbolic dimension (e.g., s27)
        """
        self.specialized_symbol_registry[concrete_value] = symbol
        self.specialized_concrete_values[symbol] = concrete_value

    def get_specialized_symbol(self, value: int | SpecializedValue) -> sympy.Symbol | None:
        """
        Get the specialized symbol for a value.

        Args:
            value: Either a SpecializedValue or a plain int that may match
                   a registered specialized value

        Returns:
            The original specialized symbol if this value was specialized,
            None otherwise.
        """
        if isinstance(value, SpecializedValue):
            return value._original_symbol
        elif value in self.specialized_symbol_registry:
            return self.specialized_symbol_registry[value]
        return None

    def create_symint_for_dimension(
        self,
        value: int | torch.SymInt | SpecializedValue,
        *,
        hint: int | None = None
    ) -> int | torch.SymInt:
        """
        Create appropriate SymInt for a dimension value.

        If the value is specialized, returns a SymInt constrained to the
        original symbol. Otherwise creates a fresh unbacked SymInt.

        This is the KEY method that enables symbol reuse.

        Args:
            value: The dimension value (may be specialized)
            hint: Optional hint for unbacked symbols

        Returns:
            Either a concrete int, a SymInt reusing a specialized symbol,
            or a fresh unbacked SymInt
        """
        if isinstance(value, torch.SymInt):
            return value

        if isinstance(value, SpecializedValue):
            # Reuse the original symbol
            symbol = value._original_symbol
            concrete = value._concrete_value

            # Create a SymInt for the original symbol
            # It already has a replacement set, so FakeTensor will treat it as concrete
            return self.shape_env.create_symintnode(
                symbol,
                hint=concrete,
                source=None
            )

        specialized_symbol = self.get_specialized_symbol(value)
        if specialized_symbol is not None:
            # This int came from a specialized dimension
            concrete = self.specialized_concrete_values[specialized_symbol]
            return self.shape_env.create_symintnode(
                specialized_symbol,
                hint=concrete,
                source=None
            )

        # Not specialized - create fresh unbacked symbol
        if hint is None:
            hint = value if isinstance(value, int) else 8192
        return self.create_unbacked_symint(hint=hint)
```

**Update `to_fake` to handle `SpecializedValue`:**

```python
def to_fake(self, obj: object, origin: Origin) -> object:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return self._to_fake_tensor(obj, origin.to_source())

    # IMPORTANT: Check for SpecializedValue BEFORE checking for int
    # (since SpecializedValue is a subclass of int)
    if isinstance(obj, SpecializedValue):
        # Return a SymInt that references the original specialized symbol
        return self.create_symint_for_dimension(obj)

    if isinstance(obj, (bool, int, float)):
        if isinstance(obj, bool):
            with self.shape_env.ignore_fresh_unbacked_symbols():
                return self.shape_env.create_unbacked_symbool()
        if isinstance(obj, int):
            # Check if this int is a specialized value (registered)
            specialized_sym = self.get_specialized_symbol(obj)
            if specialized_sym is not None:
                # Reuse the specialized symbol
                return self.create_symint_for_dimension(
                    SpecializedValue(obj, specialized_sym)
                )
            return self.create_unbacked_symint(hint=obj)
        if isinstance(obj, float):
            with self.shape_env.ignore_fresh_unbacked_symbols():
                return self.shape_env.create_unbacked_symfloat()

    # ... rest of to_fake unchanged ...
```

### Layer 3: Tensor Creation Interception (Critical Path)

Intercept tensor creation operations to preserve specialized symbols.

**File**: `helion/language/creation_ops.py`

```python
@_decorators.register_fake(full)
def _full_fake(
    shape: list[int | torch.SymInt | SpecializedValue],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")

    env = CompileEnvironment.current()

    # Convert SpecializedValue dimensions to proper SymInts that reuse symbols
    converted_shape = []
    for dim in shape:
        if isinstance(dim, SpecializedValue):
            # This will reuse the original specialized symbol
            converted_shape.append(env.create_symint_for_dimension(dim))
        elif isinstance(dim, int):
            # Check if this plain int matches a specialized value
            converted_shape.append(env.create_symint_for_dimension(dim))
        else:
            converted_shape.append(dim)

    env.add_kernel_tensor_size(converted_shape)
    return torch.empty(
        converted_shape,
        dtype=dtype,
        device=env.device if device is None else device,
    )
```

### Layer 4: Reduction Operations (Where the Bug Manifests)

Modify reduction fake implementations to preserve specialized symbols.

**File**: `helion/language/reduce_ops.py`

```python
def _fake_reduce_tensor(
    tensor: torch.Tensor, dim: int | None, keep_dims: bool
) -> torch.Tensor:
    """
    Helper to create a fake tensor with reduced dimensions.

    CRITICAL: Must preserve specialized symbols for non-reduced dimensions
    to avoid creating fresh unbacked symbols.
    """
    env = CompileEnvironment.current()

    if dim is None:
        # Reduce all dimensions
        if keep_dims:
            return torch.empty(
                [1] * tensor.ndim, dtype=tensor.dtype, device=tensor.device
            )
        return torch.empty([], dtype=tensor.dtype, device=tensor.device)

    # Reduce specific dimension
    # Normalize negative dimension
    if dim < 0:
        dim = tensor.ndim + dim

    new_shape = []
    for i, size in enumerate(tensor.shape):
        if i == dim:
            # This dimension is being reduced
            if keep_dims:
                new_shape.append(1)
            # else: dimension is removed
        else:
            # Preserve this dimension's symbol
            # This is critical: we must use the same SymInt, not create a new one
            if isinstance(size, torch.SymInt):
                sympy_expr = size._sympy_()

                # Check if this is a specialized dimension
                if sympy_expr in env.specialized_concrete_values:
                    # Reuse the exact same SymInt to maintain symbol identity
                    new_shape.append(size)
                else:
                    # Non-specialized SymInt - still preserve it
                    new_shape.append(size)
            else:
                # Plain int - check if it's specialized
                new_shape.append(env.create_symint_for_dimension(size))

    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)
```

### Layer 5: Type Propagation Integration

Ensure TensorType properly handles specialized dimensions.

**File**: `helion/_compiler/type_propagation.py`

```python
class TensorType(TypeInfo):
    def __init__(self, origin: Origin, fake_value: torch.Tensor) -> None:
        super().__init__(origin)
        self.fake_value = fake_value

        if origin.is_device():
            # Validate and potentially fix specialized dimensions
            self._ensure_specialized_symbols(fake_value.size())
            CompileEnvironment.current().add_kernel_tensor_size(fake_value.size())

    def _ensure_specialized_symbols(self, sizes: torch.Size):
        """
        Ensure specialized dimensions use their original symbols.

        This is a safety check: if we detect an unbacked symbol with a hint
        that matches a specialized value, we link them.
        """
        env = CompileEnvironment.current()

        for i, size in enumerate(sizes):
            if isinstance(size, torch.SymInt):
                sympy_expr = size._sympy_()

                # Check if this is an unbacked symbol
                if not isinstance(sympy_expr, sympy.Symbol):
                    continue

                symbol_name = str(sympy_expr)
                if not symbol_name.startswith('u'):
                    # Not an unbacked symbol
                    continue

                # This is an unbacked symbol - check if its hint matches a specialized value
                if sympy_expr in env.shape_env.var_to_val:
                    hint = env.shape_env.var_to_val[sympy_expr]
                    if isinstance(hint, sympy.Integer):
                        hint_val = int(hint)
                        specialized_sym = env.get_specialized_symbol(hint_val)

                        if specialized_sym is not None:
                            # This unbacked symbol should have reused the specialized symbol!
                            # Set replacement to link them
                            env.shape_env._set_replacement(
                                sympy_expr,
                                specialized_sym,  # Replace with specialized symbol
                                "reuse specialized dimension"
                            )
```

### Layer 6: Validation & Debugging Support

Add validation to catch issues early in development.

**File**: `helion/_compiler/compile_environment.py`

```python
class CompileEnvironment:
    def validate_specialized_symbols(self, *, verbose: bool = False) -> list[str]:
        """
        Validate that specialized symbols are being reused correctly.

        This can be called in debug mode or during testing to detect issues.

        Args:
            verbose: If True, print detailed information about each issue

        Returns:
            List of validation error messages (empty if no issues)
        """
        issues = []

        # Check for unbacked symbols with hints matching specialized values
        for sym, hint in self.shape_env.var_to_val.items():
            if not isinstance(sym, sympy.Symbol):
                continue

            symbol_name = str(sym)
            if not symbol_name.startswith('u'):
                # Not an unbacked symbol
                continue

            if isinstance(hint, sympy.Integer):
                hint_val = int(hint)
                if hint_val in self.specialized_symbol_registry:
                    expected_sym = self.specialized_symbol_registry[hint_val]

                    # Check if this symbol has a replacement set
                    replaced = self.shape_env.replace(sym)
                    if replaced == sym:
                        # No replacement set - this is a problem
                        msg = (
                            f"Unbacked symbol {sym} has hint {hint_val} "
                            f"matching specialized value, but is not replaced. "
                            f"Should reuse specialized symbol {expected_sym}"
                        )
                        issues.append(msg)
                        if verbose:
                            print(f"[VALIDATION WARNING] {msg}", file=sys.stderr)

        return issues

    def dump_symbol_info(self):
        """Debugging helper to print all symbol information."""
        print("=== Specialized Symbol Registry ===", file=sys.stderr)
        for value, symbol in self.specialized_symbol_registry.items():
            print(f"  {value} -> {symbol}", file=sys.stderr)

        print("\n=== ShapeEnv var_to_val (hints) ===", file=sys.stderr)
        for sym, hint in list(self.shape_env.var_to_val.items())[:20]:
            print(f"  {sym} -> {hint}", file=sys.stderr)

        print("\n=== Symbol Replacements ===", file=sys.stderr)
        for sym in list(self.shape_env.var_to_val.keys())[:20]:
            replaced = self.shape_env.replace(sym)
            if replaced != sym:
                print(f"  {sym} -> {replaced}", file=sys.stderr)
```

## Implementation Plan

### Phase 1: Foundation (Week 1)
**Goal**: Add infrastructure without breaking existing code

1. Add `SpecializedValue` class to `constexpr.py`
2. Add registry methods to `CompileEnvironment`
3. Write unit tests for `SpecializedValue`:
   - Test that it behaves like int
   - Test arithmetic operations
   - Test pickling
   - Test comparison operations

### Phase 2: Integration (Week 1-2)
**Goal**: Wire up the system end-to-end

1. Modify `hl.specialize` to return `SpecializedValue`
2. Update `to_fake` to handle `SpecializedValue`
3. Add `create_symint_for_dimension` method
4. Update `_full_fake` and other tensor creation operations
5. Add integration tests:
   - Test that specialized values create correct SymInts
   - Test registry is populated correctly

### Phase 3: Reduction Fixes (Week 2)
**Goal**: Fix the specific bug that triggered this work

1. Modify `_fake_reduce_tensor` to preserve specialized symbols
2. Update `TensorType.__init__` validation
3. Test with layer_norm_bwd reproduction script
4. Test with TritonBench non-power-of-two sizes

### Phase 4: Validation & Polish (Week 3)
**Goal**: Make the system robust and debuggable

1. Add `validate_specialized_symbols` for debugging
2. Add comprehensive tests for edge cases:
   - Multiple specialized values
   - Specialized values in nested structures
   - Specialized values in reductions
3. Update documentation
4. Performance testing

### Phase 5: Rollout (Week 3-4)
**Goal**: Deploy to production safely

1. Enable by default for new code
2. Gradual rollout with monitoring
3. Add telemetry to track symbol reuse success rate
4. Fix any issues discovered in production

## Advantages of This Approach

### 1. Explicit Provenance
`SpecializedValue` explicitly tracks where values came from, eliminating guesswork.

### 2. Backward Compatible
Since `SpecializedValue` is a subclass of `int`, existing code continues to work:
```python
n = hl.specialize(x.size(1))  # Returns SpecializedValue
indices = range(n)             # Works - SpecializedValue acts like int
if n > 100:                    # Works - comparison operators preserved
```

### 3. Systematic & Traceable
- Registry makes it easy to debug symbol reuse
- Validation can run in tests to catch regressions
- Clear interception points (to_fake, tensor creation, reductions)

### 4. Testable
Each layer can be tested independently:
- Unit tests for `SpecializedValue` behavior
- Integration tests for registry
- End-to-end tests for reductions

### 5. Performance
- No runtime overhead for non-specialized code paths
- Symbol reuse reduces ShapeEnv complexity
- Fewer guards needed when dimensions are known equal

## Potential Issues & Mitigations

### Issue 1: Type Checking Code

**Problem**: Code doing `type(n) == int` will fail.

**Mitigation**:
- `isinstance(n, int)` continues to work ✓
- Add type checking utilities if needed
- Document the behavior change

### Issue 2: Serialization

**Problem**: Pickling `SpecializedValue` needs special handling.

**Solution**: Implement `__reduce__` to serialize as plain int (already in proposal).

### Issue 3: Multiple Specialized Values with Same Concrete Value

**Problem**: If `batch_size=5632` and `hidden_dim=5632`, which symbol to use?

**Solution**:
- Registry uses first-registered symbol
- Alternative: Use tuple of (value, context) as key
- Document that multiple specializations of same value may conflict

### Issue 4: Arithmetic on Specialized Values

**Problem**: `n * 2` should return plain int or SpecializedValue?

**Solution**: Return plain int for safety (provenance lost after arithmetic).

## Testing Strategy

### Unit Tests
```python
def test_specialized_value_behaves_like_int():
    sv = SpecializedValue(5632, sympy.Symbol('s27'))
    assert sv == 5632
    assert sv > 5000
    assert sv + 1 == 5633
    assert isinstance(sv, int)

def test_specialized_value_pickling():
    sv = SpecializedValue(5632, sympy.Symbol('s27'))
    pickled = pickle.dumps(sv)
    unpickled = pickle.loads(pickled)
    assert unpickled == 5632
    assert type(unpickled) == int  # Serializes as plain int

def test_symbol_registry():
    env = CompileEnvironment(...)
    env.register_specialized_value(5632, sympy.Symbol('s27'))
    assert env.get_specialized_symbol(5632) == sympy.Symbol('s27')
```

### Integration Tests
```python
def test_specialized_value_in_tensor_creation():
    @helion.kernel
    def test_kernel(x: torch.Tensor):
        n = hl.specialize(x.size(1))
        buf = torch.zeros(n, device=x.device)
        # Verify that buf's shape uses the original symbol, not unbacked
        return buf

    x = torch.randn(16, 5632, device='cuda')
    result = test_kernel(x)
    # Validation: no unbacked symbols created

def test_reduction_preserves_specialized_symbols():
    @helion.kernel
    def test_kernel(x: torch.Tensor):
        n = hl.specialize(x.size(1))
        buf = torch.zeros(n, device=x.device)
        result = torch.sum(buf, dim=0)  # Should not create u4
        return result

    x = torch.randn(16, 5632, device='cuda')
    result = test_kernel(x)
```

### End-to-End Tests
```python
def test_layer_norm_bwd_non_power_of_two():
    """Test the original failure case."""
    from layer_norm_bwd_symbolic_mismatch import _layer_norm_bwd

    m, n = 4096, 5632  # Non-power-of-two
    grad_out = torch.randn(m, n, device='cuda', dtype=torch.float16)
    x = torch.randn(m, n, device='cuda', dtype=torch.float16)
    weight = torch.randn(n, device='cuda', dtype=torch.float16)
    mean = torch.randn(m, device='cuda', dtype=torch.float32)
    rstd = torch.randn(m, device='cuda', dtype=torch.float32)

    # Should not raise symbolic mismatch error
    grad_x, grad_weight, grad_bias = _layer_norm_bwd(
        grad_out, x, weight, mean, rstd, compute_bias_grad=True
    )

    assert grad_x.shape == (m, n)
    assert grad_weight.shape == (n,)
```

## Migration Path

### For Helion Users

**No changes required** - `SpecializedValue` behaves like `int` for all standard operations.

### For Helion Developers

**Code patterns that need updating:**
```python
# Before:
n = hl.specialize(x.size(1))  # Returns int
assert type(n) == int         # Passes

# After:
n = hl.specialize(x.size(1))  # Returns SpecializedValue
assert isinstance(n, int)     # Passes (recommended anyway)
assert type(n) == int         # Fails (bad pattern)
```

### For Testing

Add validation calls in test mode:
```python
if settings.debug:
    issues = env.validate_specialized_symbols(verbose=True)
    if issues:
        raise RuntimeError(f"Symbol validation failed: {issues}")
```

## Success Metrics

1. **TritonBench Accuracy**: layer_norm_bwd passes for all feature sizes
2. **Symbol Reuse Rate**: >90% of dimensions that should reuse symbols do so
3. **Guard Reduction**: Fewer shape guards generated when dimensions are specialized
4. **Test Coverage**: >95% coverage of new code paths
5. **Performance**: No regression in compilation time or runtime

## Future Enhancements

### 1. Multi-Value Specialization
Support specializing on multiple values with context:
```python
batch_size = hl.specialize(x.size(0), context="batch")
hidden_dim = hl.specialize(x.size(1), context="hidden")
```

### 2. Automatic Specialization
Auto-detect frequently-used dimensions and suggest specialization:
```python
if env.settings.auto_specialize:
    # Analyze usage patterns and specialize automatically
```

### 3. Symbol Provenance Tracking
Extend to track full provenance chain:
```python
# Track that n_half came from n through arithmetic
n = hl.specialize(x.size(1))  # SpecializedValue(5632, s27)
n_half = n // 2               # Could preserve provenance
```

## References

- Original issue: `layer_norm_bwd_debug.md`
- Reproduction script: `layer_norm_bwd_symbolic_mismatch.py`
- Related PyTorch docs: FakeTensor, ShapeEnv, SymInt
- TritonBench: layer_norm operator

## Appendix: Alternative Approaches Considered

### Alternative 1: Keep SymInts Throughout
**Idea**: Don't convert to int, keep as SymInt with replacement.

**Pros**: No new types needed

**Cons**:
- Breaks Python operations that expect concrete ints
- Can't use in range(), indexing, etc.

### Alternative 2: Hint-Based Matching (Current Implementation)
**Idea**: Match symbols by hint values retroactively.

**Pros**: Minimal code changes

**Cons**:
- Unreliable (multiple dims with same value)
- Timing-dependent
- Hard to debug

### Alternative 3: Compiler Pass After Tracing
**Idea**: Post-process traced graph to unify symbols.

**Pros**: Doesn't affect tracing

**Cons**:
- Too late - FakeTensor errors occur during tracing
- Can't fix the root cause

**Decision**: Alternative 1 with SpecializedValue wrapper is most robust.