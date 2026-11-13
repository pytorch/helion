# Kernel

The `Kernel` class is the main entry point for executing Helion GPU kernels.

```{eval-rst}
.. currentmodule:: helion

.. autoclass:: Kernel
   :members:
   :show-inheritance:
```

## Overview

A `Kernel` object is typically created via the `@helion.kernel` decorator. It manages:

- **Compilation** of Python functions to GPU code
- **Autotuning** to find optimal configurations
- **Caching** of compiled kernels
- **Execution** with automatic argument binding

The kernel compilation process converts Python functions using `helion.language` constructs into optimized Triton GPU kernels.

### Host tensor aliases and storage proofs

Some CuTe reduction and Metal lane-loop lowerings require buffers to have
non-overlapping storage. Selecting a tensor in a host `if` statement does not
allocate storage: every possible branch, including views and nested selections,
must satisfy the disjointness requirement. A distinct `torch.empty_like(...)`
or `clone()` allocation can provide that proof; an `out=` operation writing an
existing argument does not. Out-of-place native tensor arithmetic creates fresh
storage, but augmented assignment such as `tensor += 1` does not establish that
allocation proof.

Replacing tensor storage with `set_` invalidates the affected storage proofs,
including when replacement occurs in a sibling branch or a loop that may run
zero times, or the original storage is later restored. Ordinary native writes
that keep the storage, such as `fill_`, preserve alias proofs.
Shape and stride changes through a host-selected tensor require each possible
owner's runtime metadata. Rank-changing mutations through such a selection are
rejected because one branch can retain the original rank.
Rank-changing host mutations are also rejected when execution is conditional,
including later `and`/`or` operands, later chained-comparison operands, and
comprehension filters or bodies. A single traced rank cannot describe both
executing and skipping the mutation. An unconditional first operand or outer
comprehension iterable can still change rank before subsequent device loops.

Opaque host helpers and overloaded Python operations discard existing storage
proofs because their storage changes may be transient or affect
captured tensors. This also removes allocation alignment and exact-layout facts;
Triton may need additional runtime stride arguments as a result.
An output allocated after such a call has independent fresh storage and can
still be proved separate from direct inputs. This does not restore input-input
span facts or prove a selected join independent. Outputs created before the
call remain conservative even when they were not passed to it, since Python
closures can reach captured tensors.
Known native operations and pure builtins with supported tensor/scalar arguments
retain the appropriate checks. Unproved Python effects can prevent a lowering
even when a particular execution happens to be safe.

Host f-strings support builtin scalars, tensors and proved builtin collections.
Formatting custom objects is rejected when their conversion protocols cannot be
proved safe: those runtime hooks can change tensor rank or metadata without being
traced during binding. This includes `!s`, `!r`, `!a` and nested format specs.
The same restriction applies to `str`, `repr`, `ascii`, `format` and percent
formatting, including `str.format`, `format_map` and bytes formatting.
Dataclasses, named tuples and container subclasses use structural
proxies during tracing; their custom truth, length, conversion, binary operator
and membership protocols are also rejected. Membership and sequence equality
that could call contained records' equality methods are rejected too. Identity
comparisons, empty native sequences, proved unequal sequence lengths, and
membership that only observes builtin dictionary keys retain their ordinary
behavior. Passing these proxies to opaque Python helpers is rejected because the
helper could invoke a runtime record method that tracing cannot see.
This can reject a pure helper too; pass its declared fields explicitly, for
example `helper(record.tensor)`, so the compiler can trace the actual operands.
Accessing declared fields, native named-tuple length/truth, and inspecting the
length or truth of a plain builtin collection containing such records remain
supported. `type(record)`, ordinary `isinstance(record, RecordType)`, and
`hasattr(record, "declared_field")` are also supported when the query cannot call
custom attribute or instance-check hooks. Undeclared attributes and custom
properties remain unsupported. Native tuple/list conversion of a named tuple,
including `tuple(torch.sort(x))`, preserves its positional values.
Custom formatting remains unsupported in assertion messages, including messages
on statically false assertions: those hooks can affect later invocations too.
Format a declared field instead, such as `f"shape: {record.tensor.shape}"`.
Plain list and tuple concatenation or repetition also remain supported;
these outer operations do not invoke the contained records' binary protocols.

When disjointness cannot be proved, optional optimizations are disabled. A
lowering that requires independent buffers instead raises
`helion.exc.BackendUnsupported` during binding or code generation. Unsafe or
unproved aliasing patterns accepted by older versions can therefore stop
compiling; this shared compiler behavior can affect backends beyond CuTe.

Host sizes can pass through symbolic float conversion and division before an
integer extent, including `int(x.size(0) / 2)` and assigning that quotient to a
local first. Additive and multiplicative floating-point size expressions remain
runtime values: simplifying them algebraically can change ordered IEEE rounding.
Their conversion to dynamic integer extents may therefore still be unsupported;
use equivalent integer shape arithmetic when its rounding semantics are intended.

### Host tensor indexing

Host tensor subscripts may contain at most one ellipsis. For example,
`view = x[..., :4]` selects up to four elements along the last dimension before
entering a device loop. A repeated ellipsis such as `x[..., ..., :]` raises
`helion.exc.TypeInferenceError` during binding, before invoking PyTorch indexing.

### Host collection predicates

A named builtin list, tuple, or dictionary created on the host may be tested
with `if values`, `if not values`, `bool(values)`, or `len(values)` inside a
device loop. When filtering makes its contents dynamic, Helion passes the actual
host length to the device; representative elements used during type inference
do not determine the predicate. The collection is evaluated once by the original
host code.

Collection values or lengths returned by opaque helpers are rejected when the
compiler cannot prove their runtime semantics. A filtered generator or a custom
collection with overridden truth/length methods cannot use this device
projection. Evaluate its scalar predicate explicitly in supported host code
instead. Dynamic device conditional expressions (`a if values else b`) remain
unsupported; use an `if` statement. These cases previously could compile using
representative contents and produce an incorrect result.

## Creation and Usage

### Basic Kernel Creation

```python
import torch
import helion
import helion.language as hl

@helion.kernel
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(a)
    for i in hl.grid(a.size(0)):
        result[i] = a[i] + b[i]
    return result

# Usage
a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = vector_add(a, b)  # Automatically compiles and executes
```

### With Custom Settings

```python
@helion.kernel(
    autotune_effort="none",    # Skip autotuning
    print_output_code=True      # Debug generated code
)
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```

### With Restricted Configurations

```python
@helion.kernel(configs=[
    helion.Config(block_sizes=[32], num_warps=4),
    helion.Config(block_sizes=[64], num_warps=8)
])
def optimized_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```


## BoundKernel

When you call `kernel.bind(args)`, you get a `BoundKernel` that's specialized for specific argument types and (optionally) shapes:

```python
# Bind once, execute many times
bound = my_kernel.bind((example_tensor,))
result1 = bound(tensor1)  # Compatible tensor (same dtype/device)
result2 = bound(tensor2)  # Compatible tensor (same dtype/device)

# With static_shapes=True, tensors must have exact same shapes/strides
@helion.kernel(static_shapes=True)
def shape_specialized_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass

bound_static = shape_specialized_kernel.bind((torch.randn(100, 50),))
result = bound_static(torch.randn(100, 50))  # Must be exactly [100, 50]
```

```{warning}
Helion shape-specializes kernels by default (`static_shapes=True`) for the best performance. Bound kernels and caches require tensors with the exact same shapes and strides as the examples you compile against. Set `static_shapes=False` if you need the same compiled kernel to serve many shapes.
With dynamic shapes (`static_shapes=False`), Helion also specializes on if a tensor shape is 0 or 1 and whether a tensor needs 64-bit indexing (more than ``2**31 - 1`` elements).
This 64-bit indexing specialization can be avoided by setting `index_dtype=torch.int64`.
```

### BoundKernel Methods

The returned BoundKernel has these methods:

- `__call__(*args)` - Execute with bound arguments
- `autotune(args, **kwargs)` - Autotune this specific binding
- `set_config(config)` - Set and compile specific configuration
- `to_code(config, options=...)` - Generate source code for the selected backend
- `to_triton_code(config)` - Backward-compatible alias for `to_code(config)`
- `compile_config(config)` - Compile for specific configuration

## Advanced Usage

### Manual Autotuning

```python
# Separate autotuning from execution
kernel = my_kernel

# Find best config
config = kernel.autotune(example_args, num_iterations=100)

# Later, use the optimized config
result = kernel(actual_args)  # Uses cached config
```

### Config Management

```python
bound = kernel.bind(args)

# set a specific configuration
bound.set_config(helion.Config(block_sizes=[64], num_warps=8))

# generate Triton code for the bound kernel
triton_code = bound.to_triton_code(config)
print(triton_code)
```

### Standalone Export

Export a kernel with its launcher and helpers so it can run without Helion:

```python
from pathlib import Path

bound = kernel.bind(args)
source = bound.to_code(
    config, options=helion.OutputCodeOptions(allow_helion_deps=False)
)
Path("exported_kernel.py").write_text(source)
```

The exported module contains native backend source (for example, Triton or CuTe)
and a host entrypoint with the original kernel's name. Import that entrypoint and
call it directly; no `BoundKernel` is needed:

```python
from exported_kernel import my_kernel

result = my_kernel(*args)
```

The module still requires PyTorch and the backend's runtime dependencies. Its
input types and specialization constraints remain those of the exported kernel;
export does not make a shape-specialized kernel accept arbitrary shapes. The
source and launcher can be edited directly, without regenerating a Helion config.
Unsupported backend export paths raise `NotImplementedError`.

Raw generated entrypoints preserve only a trailing run of portable immutable
defaults. Callable, mutable and config-fragment defaults must be supplied
explicitly, as must any defaults preceding them: Python does not allow required
positional parameters after optional ones. Additional required parameters
introduced during generation remove all defaults from the raw entrypoint.
Default expressions from the original source are not replayed. Calling a raw
entrypoint with a required argument omitted raises `TypeError`; the decorated
kernel and the result of `compile_config` still apply the original defaults.
Capture defaults once before binding and pass those same arguments to the
generated entrypoint:

```python
args = kernel.normalize_args(*args)
bound = kernel.bind(args)
# Generate and load bound.to_code(config), then call its entrypoint with *args.
```

CuTe export currently requires `static_shapes=True` and static tensor layouts at
each native launch. Dynamic
launch layouts and plans that require additional descriptor preparation are not
yet supported.

## Caching and Specialization

Kernels are automatically cached based on:

- **Argument types** (dtype, device)
- **Tensor shapes** (default: `static_shapes=True`)

When a compile-time `is` or `is not` comparison depends on a container returned
by an ordinary Python helper, Helion conservatively creates a fresh binding for
each invocation. The helper may choose an input, a captured container, or a new
container; equal argument values do not establish that choice. These bindings
do not use shared dispatch caches or persistent autotune caches. Ordinary
argument-container comparisons retain guarded reuse, and comparisons against
`None` or a different type do not need this restriction.

Opaque Python helpers that receive configuration fragments may inspect their
identity. Helion retains at most 16 such bindings per kernel, in insertion order,
including the explicit constexpr objects needed by their traces. Repeated calls
reuse a resident binding without executing the helper again. Once evicted, a
binding is traced again on its next use; compatible built-in-domain autotune
selections can be retained separately, but custom domains may require autotuning again.
An explicit `BoundKernel.set_config()` choice remains local to that bound object;
eviction does not publish it as an autotune result or replace a retained winner.
Explicitly retained `BoundKernel` objects remain usable. `Kernel.reset()` clears
both caches. This finite retention policy replaces unbounded owner retention;
compile-time Python side effects must not rely on a binding remaining cached.

The default autotuner and `Cache[Search]` factories retain search and best-of-K
selection without cache I/O. Direct construction of the exact built-in
`LocalAutotuneCache` and `StrictLocalAutotuneCache` classes also returns this
uncached wrapper, rather than an instance of the requested cache class.
Direct `RemoteAutotuneCache`, `StrictRemoteAutotuneCache` and custom cache
constructors without fresh-binding support reject these bindings before cache
initialization. The factory form substitutes Helion's uncached wrapper for
unsupported cache classes, preserving the supplied `Search` class.

`AOTAutotuneCache` supports fresh bindings through both direct construction and
`AOTAutotuneCache[Search]`. Explicit `evaluate` and `compile` modes select the
AOT heuristic or default without starting autotuning. `collect` runs the search
and records its artifacts, but does not reuse an automatic winner for a fresh
binding. Custom subclasses declaring fresh-binding support retain their own
constructor behavior.
`HELION_ASSERT_CACHE_HIT=1` rejects a search that cannot read a cache, unless
cache reads were explicitly bypassed by force or `HELION_SKIP_CACHE`.

For these fresh bindings, `Kernel.autotune`, `autotune_multi` and automatic
first-use tuning retain up to 16 selected configurations in memory. Each call
still traces and compiles anew. A selection is reused only after the complete
frontend program, built-in tunable domains, compiler policy and target match;
no inputs or compiled launchers are retained. Changed domains or branches tune
separately. `Kernel.reset()` clears selections, and force/skip requests bypass
their reuse. Custom fragments or opaque domain/config values cannot establish
this compatibility: Helion warns once per kernel and may tune each invocation.

Calling such a `BoundKernel` preserves a configuration selected by `set_config`
or `autotune`, validating and compiling it against the fresh trace. If the new
arguments select a shareable binding that already has a configuration, that
binding keeps its own selection. An unconfigured shareable binding receives
the explicit configuration from the fresh bound. A callable
returned directly by `compile_config` remains the compiled program for that
trace and configuration; it does not perform argument rebinding.

By default (`static_shapes=True`), Helion treats shapes and strides as compile-time constants, baking them into generated Triton code for the best performance. To reuse a single compiled kernel across size variations, set `static_shapes=False`, which instead buckets each dimension as `{0, 1, ≥2}` and allows more inputs to share the same cache entry.

```{note}
Dynamic buckets also track whether any tensor exceeds the ``torch.int32`` indexing limit so that cache entries diverge as soon as large inputs show up. Set ``index_dtype=torch.int64`` on the kernel to avoid this.
```

```python
# These create separate cache entries
tensor_float = torch.randn(100, dtype=torch.float32, device='cuda')
tensor_int = torch.randint(0, 10, (100,), dtype=torch.int32, device='cuda')

result1 = my_kernel(tensor_float)  # Compiles for float32
result2 = my_kernel(tensor_int)    # Compiles for int32 (separate cache)
```

## Settings vs Config in Kernel Creation

When creating kernels, you'll work with two distinct types of parameters:

### Settings: Compilation Control
Settings control **how the kernel is compiled** and the development environment:

```python
@helion.kernel(
    # Settings parameters
    autotune_effort="none",      # Skip autotuning for development
    autotune_effort="quick",     # Smaller autotuning budget when search is enabled
    print_output_code=True,       # Debug: show generated Triton code
    print_repro=True,             # Debug: show Helion kernel code, config, and caller code as a standalone repro script
    static_shapes=True,           # Compilation optimization strategy
    autotune_log_level=logging.DEBUG  # Verbose autotuning output
)
def debug_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```

### Config: Execution Control
Config parameters control **how the kernel executes** on GPU hardware:

```python
@helion.kernel(
    # Config parameters
    config=helion.Config(
        block_sizes=[64, 128],    # GPU tile sizes
        num_warps=8,              # Thread parallelism
        num_stages=4,             # Pipeline stages
        indexing='block_ptr'      # Memory access strategy
    )
)
def production_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```

### Combined Usage
You can specify both Settings and Config together:

```python
@helion.kernel(
    # Settings: control compilation
    print_output_code=False,      # No debug output
    static_shapes=True,           # Shape specialization
    # Config: control execution
    config=helion.Config(
        block_sizes=[32, 32],     # Execution parameters
        num_warps=4
    )
)
def optimized_kernel(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    pass
```

For more details, see {doc}`settings` (compilation control) and {doc}`config` (execution control).

## See Also

- {doc}`settings` - Compilation behavior and debugging options (controls **how kernels are compiled**)
- {doc}`config` - GPU execution parameters and optimization strategies (controls **how kernels execute**)
- {doc}`exceptions` - Exception handling and error diagnostics
- {doc}`language` - Helion language constructs for kernel authoring
- {doc}`autotuner` - Autotuning configuration and search strategies
