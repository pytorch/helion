# ConfigSpace: User-Controlled Autotuner Search Space

## Context

Users currently have limited control over Helion's autotuner search space. They can either provide explicit `configs=[...]` to search over, or let the autotuner explore the full space. There's no way to partially constrain the space (e.g., pin `pid_type`, restrict `num_warps` to a subset, or derive block sizes from input shapes).

This feature adds a `config_space_fn` parameter to `@helion.kernel` that accepts a lambda receiving the kernel's real args and returning a `ConfigSpace` object. `ConfigSpace` accepts the same keyword arguments as `Config` but allows fragments (search ranges), pinned values, and derived lambdas.

## Target API

```python
# Derived functions can be defined externally — any callable with (args, config) -> value
def my_block_size(args, config):
    return args[0].shape[1] // config.num_reduction_ctas

@helion.kernel(
    config_space_fn=lambda args: helion.ConfigSpace(
        block_sizes=[
            BlockSizeFragment(1, 32),
            # inline lambda works:
            lambda args, config: (args[0].shape[1] // config.num_reduction_ctas),
            # external function works too:
            # my_block_size,
        ],
        pid_type='xyz',
        num_warps=NumWarpsFragment(4, 8),
        num_reduction_ctas=EnumFragment(choices=(2, 4, 8)),
    ),
)
def my_kernel(x, y): ...
```

## Design Decisions

- **ConfigSpace is a separate class** (not a subclass of Config), accepting `**kwargs` to avoid duplicating Config's parameter list. Validation of parameter names happens lazily in `apply_config_space()` when the actual `ConfigSpec` is available
- **Shorter lists are padded with None** (unconstrained) - error only if list is too long
- **Derived lambdas can only depend on independent (non-derived) params** - no chained deps. Detected at resolution time via None sentinels (see Step 4). Users who need chained derivations can compose plain Python helper functions instead (see `docs/design/config_space_qa.md` Q3–Q4).
- **Derived function signature is `(args, config) -> value`** — takes both proxy args and current config. This allows external functions (not just inline lambdas) since args don't need to be captured via closure
- **Fragment overrides must match the original fragment type** — `num_warps` must use `NumWarpsFragment` (not `EnumFragment`), `block_sizes` must use `BlockSizeFragment`, etc. This preserves ML encoding consistency (log2 vs one-hot) for surrogate models. See `docs/design/config_space_qa.md` Q2 for full rationale.
- **Smart derived-value approach (not filter)** — Derived values are computed in `unflatten()` after independent params are resolved, removing derived params from the search space entirely. This avoids wasted search dimensions, stalled PatternSearch, spurious surrogate model correlations, and wasted compilations. See `docs/design/config_space_qa.md` Q1 for full rationale.

---

## Implementation Steps

### Step 1: Create `helion/autotuner/config_space.py`

New file defining the `ConfigSpace` class:

```python
class ConfigSpace:
    def __init__(self, **kwargs: object):
        # Accepts the same kwargs as Config (block_sizes, num_warps, pid_type, etc.)
        # plus user-defined tunables. No duplication of Config's parameter list —
        # validation happens lazily in apply_config_space() when we have the actual
        # ConfigSpec and can check parameter names, fragment types, and ranges.
        self.entries: dict[str, object] = {}
        for key, value in kwargs.items():
            self.entries[key] = self._classify(value)
        # Classification per value:
        # - ConfigSpecFragment -> stored as-is (constrained search)
        # - Callable (not Fragment) -> DerivedValue wrapper
        # - Plain value -> pinned (no search)
        # For list-valued params: each element independently classified
```

Key helper types (private, used internally):
- `DerivedValue(fn)` - wraps a callable with signature `(args, config) -> value`
- Classification logic: `isinstance(v, ConfigSpecFragment)` -> fragment, `callable(v)` -> derived, else -> pinned

### Step 2: Add `DerivedFragment` to `helion/autotuner/config_fragment.py`

```python
@dataclasses.dataclass
class DerivedFragment(ConfigSpecFragment):
    """Fragment whose value is derived from args and other config params at search time.
    Not searchable - excluded from flat search space.
    fn signature: (args, config) -> value"""
    fn: Callable  # Callable[[tuple, Config], object]

    def default(self): raise RuntimeError("DerivedFragment has no default")
    def random(self): raise RuntimeError("DerivedFragment is not searchable")
    def pattern_neighbors(self, current): return []
    def dim(self): return 0
    def encode(self, value): return []
```

### Step 3: Add `apply_config_space()` to `ConfigSpec` in `helion/autotuner/config_spec.py`

New fields on `ConfigSpec`:
```python
derived_config_values: list[tuple[str, int, Callable]] = field(default_factory=list)
# Each entry: (param_name, list_index, fn) where fn has signature (args, config) -> value
# e.g., ("block_sizes", 1, my_block_size_fn)

config_space_proxy_args: tuple | None = None
# Proxy-wrapped args stored for passing to derived functions at resolution time
```

New method `apply_config_space(space: ConfigSpace, proxy_args: tuple)`:

- Stores `proxy_args` on self for later use by derived functions
- **Pinned scalar** (e.g., `pid_type='xyz'`): Set `self.allowed_pid_types = ('xyz',)` or for `num_warps`, constrain fragment to single value
- **Fragment scalar** (e.g., `num_warps=NumWarpsFragment(4, 8)`): Replace the fragment used in `flat_config` - store as override that `flat_config` checks. Must match the original fragment type (validated via `isinstance` check).
- **Pinned list element** (e.g., `block_sizes=[32, None]`): Modify `BlockSizeSpec.min_size = BlockSizeSpec.max_size = 32`
- **Fragment list element** (e.g., `block_sizes=[BlockSizeFragment(1,32), None]`): Modify `BlockSizeSpec.min_size/max_size` to intersect ranges

**Fragment type validation** (applied when processing fragment overrides):
1. Check `isinstance(override, type(default_fragment))` or compatible supertype
2. For `PowerOfTwoFragment`/`IntegerFragment`: intersect ranges — `max(user.low, default.low)`, `min(user.high, default.high)` — error if empty
3. For `EnumFragment`: intersect choices — `set(user.choices) & set(default.choices)` — error if empty
4. Store the result as the override — `flat_config()` uses it instead of the default
- **Derived list element** (e.g., `block_sizes=[None, my_fn]`): Store in `derived_config_values` and mark that BlockSizeSpec position as derived
- **User-defined kwargs** (e.g., `num_reduction_ctas=EnumFragment(...)`): Add to `self.user_defined_tunables`
- **Pad with None**: If `len(block_sizes) < len(self.block_sizes)`, treat missing entries as unconstrained. Error if too long.

New fields for fragment overrides:
```python
fragment_overrides: dict[str, ConfigSpecFragment] = field(default_factory=dict)
# e.g., {"num_warps": NumWarpsFragment(4, 8)}
pinned_overrides: dict[str, object] = field(default_factory=dict)
# e.g., {"pid_type": "xyz"}  -- used during flat_config to force a value
```

Modify `flat_config()` to check `fragment_overrides` and `pinned_overrides`:
- When emitting `num_warps`: if `"num_warps"` in `fragment_overrides`, use that fragment instead of the default `NumWarpsFragment`
- When emitting a value: if key is in `pinned_overrides`, use `EnumFragment(choices=(pinned_value,))` to make it a single-choice "search"

### Step 4: Modify `ConfigGeneration` in `helion/autotuner/config_generation.py`

In `__init__`:
- When `_collect_spec` encounters a `DerivedFragment`, do NOT add it to `self.flat_spec`
- Instead record it in `self.derived_entries: list[tuple[str, int, Callable]]` from `config_spec.derived_config_values`

In `unflatten()`:
- After building the config from flat values, resolve derived values by calling each fn with `(proxy_args, config)`:
```python
proxy_args = self.config_spec.config_space_proxy_args

# Set all derived positions to None before resolution.
# If a derived fn reads another derived position, it gets None.
# Arithmetic on None raises TypeError, which we catch and wrap.
for param_name, list_index, fn in self.derived_entries:
    config.config[param_name][list_index] = None

# Resolve derived values
for param_name, list_index, fn in self.derived_entries:
    try:
        value = fn(proxy_args, config)
    except TypeError as e:
        raise InvalidConfigSpace(
            f"Derived function for {param_name}[{list_index}] failed: {e}. "
            f"This usually means it reads another derived parameter. "
            f"Derived functions can only depend on independent parameters. "
            f"To compose derived values, use shared helper functions instead "
            f"of reading from config (see docs/design/config_space_qa.md Q4)."
        ) from e
    if isinstance(value, torch.SymInt):
        raise InvalidConfigSpace("Derived values must be concrete")
    config.config[param_name][list_index] = value
self.config_spec.normalize(config, _fix_invalid=True)
```

This means external functions work naturally:
```python
def my_block_size(args, config):
    return args[0].shape[1] // config.num_reduction_ctas
# my_block_size receives (proxy_args, config) — proxy validates shape access
```

Also skip `DerivedFragment` in `default_flat()` and `random_flat()`.

### Step 5: Wire `config_space_fn` into kernel machinery

**`helion/runtime/kernel.py`:**

1. Add `config_space_fn` param to `kernel()` (all 3 overloads) and `Kernel.__init__`
2. Store as `self.config_space_fn`
3. Pass through `functools.partial` in the decorator case
4. In `BoundKernel.__init__`, after `HostFunction` creation:
```python
if self.kernel.config_space_fn is not None:
    from ..autotuner.config_space import ConfigSpace
    proxy_args = self._make_config_space_proxy_args(args)
    config_space = self.kernel.config_space_fn(proxy_args)
    assert isinstance(config_space, ConfigSpace)
    self.env.config_spec.apply_config_space(config_space, proxy_args)
```

### Step 5b: Proxy args for static shape validation

Add `_make_config_space_proxy_args()` to `BoundKernel`:

- For each arg: if it's a tensor, wrap in `_TensorProxy`; otherwise pass through
- `_TensorProxy` delegates everything to the real tensor except `.shape` and `.size()`
- `.shape[dim]` / `.size(dim)` checks corresponding `fake_arg.size(dim)`:
  - If concrete int → allowed (static_shapes=True or that dim is inherently static)
  - If SymInt with symbols all in `env.specialized_vars` → allowed (hl.specialize / mark_static)
  - If SymInt with unspecialized symbols → raise `InvalidConfigSpace` with helpful message

```python
class _ShapeProxy:
    def __init__(self, real_tensor, fake_tensor, env, arg_name):
        ...

    def __getitem__(self, dim):
        fake_size = self._fake.size(dim)
        if isinstance(fake_size, torch.SymInt):
            symbols = fake_size._sympy_().free_symbols
            if not symbols.issubset(self._env.specialized_vars):
                raise InvalidConfigSpace(
                    f"Cannot use {self._arg_name}.shape[{dim}] in config_space_fn "
                    f"because dimension {dim} is dynamic. Use static_shapes=True, "
                    f"hl.specialize(), or torch._dynamo.mark_static() to make it static."
                )
        return self._real.size(dim)

    def __len__(self):
        return self._real.dim()

    def __iter__(self):
        return iter(self._real.shape)


class _TensorProxy:
    def __init__(self, real_tensor, fake_tensor, env, arg_name):
        ...

    @property
    def shape(self):
        return self._shape_proxy

    def size(self, dim=None):
        if dim is None:
            return self._real.size()
        return self._shape_proxy[dim]

    def __getattr__(self, name):
        return getattr(self._real, name)  # dtype, device, etc. pass through
```

Validation behavior by scenario:

| Scenario | `fake_arg.size(dim)` | In `specialized_vars`? | Result |
|----------|---------------------|----------------------|--------|
| `static_shapes=True` (default) | concrete `int` | N/A | Allowed |
| `static_shapes=False` + `hl.specialize()` | `SymInt` | Yes | Allowed |
| `static_shapes=False` + `mark_static()` | `SymInt` | Yes | Allowed |
| `static_shapes=False`, no specialization | `SymInt` | No | **Raises error** |

### Step 6: Public API exports

**`helion/__init__.py`:** Add `from .autotuner.config_space import ConfigSpace` and add to `__all__`

**`helion/autotuner/__init__.py`:** Add `from .config_space import ConfigSpace as ConfigSpace`

### Step 7: Error handling in `helion/exc.py`

Add `InvalidConfigSpace` exception for:
- Too many entries in a list param
- Derived value returns non-concrete (SymInt) value
- Derived value depends on another derived value
- Fragment range doesn't intersect with IR-derived range

---

## Files to Modify

| File | Change |
|------|--------|
| `helion/autotuner/config_space.py` | **NEW** - ConfigSpace class |
| `helion/autotuner/config_fragment.py` | Add `DerivedFragment` |
| `helion/autotuner/config_spec.py` | Add `apply_config_space()`, new fields, modify `flat_config()` |
| `helion/autotuner/config_generation.py` | Handle `DerivedFragment` in init/unflatten |
| `helion/runtime/kernel.py` | Wire `config_space_fn` through kernel/BoundKernel, add `_make_config_space_proxy_args` |
| `helion/autotuner/config_space.py` | Also contains `_TensorProxy` and `_ShapeProxy` for shape validation |
| `helion/__init__.py` | Export `ConfigSpace` |
| `helion/autotuner/__init__.py` | Export `ConfigSpace` |
| `helion/exc.py` | Add `InvalidConfigSpace` |
| `test/test_config_space.py` | **NEW** - Tests |

## Verification

1. **Unit tests** (`test/test_config_space.py`):
   - Pinned scalar value constrains search (e.g., all generated configs have `pid_type='xyz'`)
   - Fragment constraint limits search range (e.g., `num_warps` only in `(4, 8)`)
   - Derived value correctly computed from other config params
   - Partial list padding works (shorter list doesn't error)
   - Too-long list raises `InvalidConfigSpace`
   - Dynamic shape access in lambda raises error
   - User-defined tunables work in ConfigSpace
   - Integration: autotuning with ConfigSpace produces valid configs within constraints

2. **Existing tests**: Run `pytest test/` to ensure no regressions

3. **Manual smoke test**: Create a simple kernel with `config_space_fn`, verify autotuning only explores the constrained space by checking generated configs
