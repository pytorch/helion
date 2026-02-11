# ConfigSpace Design Q&A

Design discussion capturing key decisions on search strategy and fragment type matching.

---

## Q1: Smart derived-value approach vs. dumb filter approach?

**Decision: Use the smart derived-value approach.**

The plan does a simplified version of a DAG-based approach:

1. **Independent params** (fragments, pinned values, user-defined tunables) form the flat search space -- these are the "root nodes"
2. **Derived params** are computed in `unflatten()` from the independent params -- these are the "leaf nodes"
3. Because we forbid chaining (derived can't depend on derived), there's no full DAG -- just roots -> one level of derived leaves. No topological sort needed.

**Key distinction: derived params are not searched — they're deterministically computed.** The search algorithm only sees independent params. Derived params have no search dimension, no randomization, no encoding in the surrogate model. After the search algorithm proposes values for independent params, each derived param is computed exactly once via `fn(proxy_args, config)`. This is stronger than "randomize independent params, then constrain the range for derived params" — derived params don't have a range at all.

The flow is:
```
Search algo (PatternSearch/DE/etc.)
  -> varies flat_values (independent params only — derived params not included)
  -> unflatten() builds Config from flat_values
  -> resolve derived values: fn(proxy_args, config) for each derived param (deterministic, not searched)
  -> normalize(config, _fix_invalid=True)
  -> benchmark
```

### Why this is better than a filter approach

A filter approach (`(config) -> bool`) has real problems with how the search algorithms work:

1. **Wasted search dimensions.** If `block_sizes[1]` is derived from `num_reduction_ctas`, the filter approach still has `block_sizes[1]` as a searchable dimension. PatternSearch would generate neighbors varying it (e.g., doubling/halving), then filter them all out. The derived approach removes it from the search space entirely -- fewer dimensions = faster convergence.

2. **PatternSearch neighbors.** `_generate_neighbors()` in `pattern_search.py:211-246` generates neighbors by varying one fragment at a time. If many neighbors are filtered out, PatternSearch sees them as "all neighbors are worse" and stalls at a local optimum that isn't actually optimal -- it just ran out of valid neighbors.

3. **Differential evolution mutations.** `differential_mutation()` combines three parent configs fragment-by-fragment. If the derived param is in the search space, mutations to it are random noise that gets filtered out. The filter can't guide the mutation away from the derived param.

4. **Surrogate model encoding.** `encode_config()` produces a fixed-dimension vector from `flat_spec`. If derived values are in the encoding, the surrogate model learns spurious correlations with the derived dimension, since those values are determined by other params anyway.

5. **No wasted compilations.** Each benchmark requires a full Triton compile + execution. The filter approach compiles configs only to discover they're invalid. The derived approach never generates invalid configs.

---

## Q2: Should constrained fragments match the original fragment type?

**Decision: Yes, require same fragment type.**

### The encoding problem with EnumFragment

Looking at the actual encoding implementations in `config_fragment.py`:

| Fragment type | `encode()` | `dim()` |
|---|---|---|
| `PowerOfTwoFragment` (used by `NumWarpsFragment`, `BlockSizeFragment`) | `[log2(value)]` | 1 |
| `IntegerFragment` | `[float(value)]` | 1 |
| `EnumFragment` | one-hot vector | `len(choices)` |
| `BooleanFragment` | `[0.0]` or `[1.0]` | 1 |

If `num_warps` defaults to `NumWarpsFragment(1, 32)` (dim=1, log2 encoding) but the user overrides with `EnumFragment(choices=(4, 8))` (dim=2, one-hot encoding):

- The encoding dimension changes within a single autotuning run -- this is fine because `ConfigGeneration` is created *after* `apply_config_space()` modifies `ConfigSpec`, so the dimension is consistent across all configs in that run.
- **But the encoding semantics change.** Log2 encoding preserves ordering: the surrogate model learns that 4 < 8 < 16 because `[2.0] < [3.0] < [4.0]`. One-hot encoding treats them as unrelated categories: `[1,0]` vs `[0,1]`. For power-of-two params, ordinal encoding is strictly better for the surrogate.

### What same fragment type looks like in practice

Every param already has a natural fragment type. The user just narrows the range:

| Param | Default fragment | User constraint | Result |
|---|---|---|---|
| `num_warps` | `NumWarpsFragment(1, 32, 4)` | `NumWarpsFragment(4, 8)` | Search over {4, 8}, log2 encoding |
| `num_stages` | `IntegerFragment(1, 8, 1)` | `IntegerFragment(2, 4)` | Search over {2, 3, 4}, linear encoding |
| `pid_type` | `EnumFragment(('flat','xyz',...))` | `EnumFragment(('xyz',))` | Pinned to 'xyz', one-hot encoding |
| `maxnreg` | `EnumFragment((None,32,64,128,256))` | `EnumFragment((64, 128))` | Search over {64, 128}, one-hot encoding |
| `block_sizes[i]` | `BlockSizeFragment(min, max, default)` | `BlockSizeFragment(16, 64)` | Search over {16, 32, 64}, log2 encoding |

The encoding stays consistent with the default, just with a narrower range.

### Is contiguous-range-only a real limitation?

For power-of-two params, the user can only express contiguous power-of-two ranges (e.g., `{4, 8, 16}` but not `{4, 16}`). In practice:

- **num_warps**: You almost always want a contiguous range (e.g., "between 4 and 16 warps"). Skipping powers of two is very rare.
- **block_sizes**: Same reasoning. `BlockSizeFragment(32, 128)` covers `{32, 64, 128}`.
- **num_stages**: `IntegerFragment(2, 4)` covers `{2, 3, 4}`. Contiguous integers are natural.
- **pid_type / maxnreg**: These are already `EnumFragment`, so the user provides a subset of choices -- no contiguity constraint.

For the rare case where someone wants non-contiguous values (e.g., `num_warps` in `{4, 16}` only), they can use `configs=[Config(num_warps=4, ...), Config(num_warps=16, ...)]` explicitly.

### Validation approach

When `apply_config_space()` processes a fragment override:
1. Check `isinstance(override, type(default_fragment))` or compatible supertype
2. For `PowerOfTwoFragment`/`IntegerFragment`: intersect ranges (`max(user.low, default.low)`, `min(user.high, default.high)`), error if empty
3. For `EnumFragment`: intersect choices (`set(user.choices) & set(default.choices)`), error if empty
4. Store the result as the override -- `flat_config()` uses it instead of the default

This is simpler to implement too, since we don't need to handle mixed fragment types in the search algorithms.

---

## Q3: How are chained dependency violations detected?

**Decision: Use None sentinels — let Python's type system do the work.**

### The problem

Derived functions receive the full `config` object with signature `(args, config) -> value`. If derived A reads a config position that is itself derived (B), and B hasn't been resolved yet, we need to catch this.

### The mechanism

Before the resolution loop in `unflatten()`, set all derived positions to `None`:

```python
# Before resolution
for param_name, list_index, fn in self.derived_entries:
    config.config[param_name][list_index] = None

# Resolution
for param_name, list_index, fn in self.derived_entries:
    try:
        value = fn(proxy_args, config)
    except TypeError as e:
        raise InvalidConfigSpace(
            f"Derived function for {param_name}[{list_index}] failed: {e}. "
            f"This usually means it reads another derived parameter."
        ) from e
    config.config[param_name][list_index] = value
```

If a derived function reads another derived position, it gets `None`. The common case — arithmetic — naturally raises `TypeError`:

- `None // 4` → `TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'`
- `None + x` → `TypeError`
- `None * x` → `TypeError`

The error is caught and wrapped with a clear message pointing to the root cause.

### Why this is clean

- **Zero new classes.** No proxies, no wrappers, no sentinel objects with dunder methods.
- **Fine-grained.** In a mixed list like `block_sizes = [Fragment(...), derived_fn]`, only the derived position (`[1]`) is set to `None`. Independent positions (`[0]`) remain accessible. A derived function can read `config.block_sizes[0]` without issue.
- **~5 lines of implementation** in `unflatten()`.

### Coverage gap

If a derived function reads `None` without doing arithmetic (e.g., `x = config.derived_thing; return 42`), we don't catch it. This is pathological — why read a value you don't use? — and the documentation makes the contract clear.

---

## Q4: Can users achieve chained derivations without system support?

**Yes — via plain Python function composition.**

### The pattern

Instead of derived B reading derived A from `config`, derived B calls the same function that computes A:

```python
def compute_block_1(args, config):
    return args[0].shape[1] // config.num_reduction_ctas

def compute_block_2(args, config):
    # Don't read config.block_sizes[1] — call the function directly
    return compute_block_1(args, config) // 2

@helion.kernel(
    config_space_fn=lambda args: ConfigSpace(
        block_sizes=[
            BlockSizeFragment(1, 32),
            compute_block_1,       # derived from independent param
            compute_block_2,       # "chains" through plain Python call
        ],
        num_reduction_ctas=EnumFragment(choices=(2, 4, 8)),
    ),
)
def my_kernel(x, y): ...
```

Both functions only read independent params from `config`. The "chaining" is just Python function composition — invisible to the config system, which sees two independent derived values.

### Why this always works

Any chain `root → A → B → C` can be flattened: each derived function computes its result directly from the roots by calling shared helpers. The intermediate steps are just Python functions calling each other. Every derived value is ultimately a deterministic function of `(args, independent_params)`.

### Redundant computation

`compute_block_1` gets called twice (once for `block_sizes[1]`, once inside `compute_block_2`). This is at autotuning time, not kernel execution — the cost is negligible. For complex cases, `@functools.lru_cache` on a pure helper works.

### Implication

The one-level restriction (roots → leaves, no chaining) isn't a limitation — it's a simplification. Users get full expressiveness through the language itself. The system stays simple (flat search space + one resolution pass), and the None sentinel (Q3) nudges users toward the helper-function pattern when they accidentally try to chain via config reads.
