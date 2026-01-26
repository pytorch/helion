# Deployment and Autotuning

Helion’s autotuner explores a large search space which is a
time-consuming process, so production deployments should generate
autotuned configs **ahead of time**. Run the autotuner on a development
workstation or a dedicated tuning box that mirrors your target
GPU/accelerator. Check tuned configs into your repository alongside the kernel,
or package them as data files and load them with `helion.Config.load`
(see {doc}`api/config`). This keeps production kernel startup fast and
deterministic, while also giving explicit control over when autotuning
happens.

If you don't specify pre-tuned configs, Helion will autotune on the
first call for each specialization key. This is convenient for
experimentation, but not ideal for production since the first call
pays a large tuning cost.  Helion writes successful tuning results to
an on-disk cache (overridable with `HELION_CACHE_DIR`, skippable
with `HELION_SKIP_CACHE`, see {doc}`api/settings`) so repeated
runs on the same machine can reuse prior configs.  For more on
caching see {py:class}`~helion.autotuner.local_cache.LocalAutotuneCache`
and {py:class}`~helion.autotuner.local_cache.StrictLocalAutotuneCache`.

The rest of this document covers strategies for pre-tuning and deploying
tuned configs, which is the recommended approach for production workloads.

## Run Autotuning Jobs

The simplest way to launch autotuning straight through the kernel call:

```python
import torch, helion

@helion.kernel()
def my_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ...

example_inputs = (
    torch.randn(1048576, device="cuda"),
    torch.randn(1048576, device="cuda"),
)

# First call triggers autotuning, which is cached for future calls, and prints the best config found.
my_kernel(*example_inputs)
```
Set `HELION_FORCE_AUTOTUNE=1` to re-run tuning even when cached configs
exist (documented in {doc}`api/settings`).

Call `my_kernel.autotune(example_inputs)` explicitly to separate
tuning from execution (see {doc}`api/kernel`).
`autotune()` returns the best config found, which you can save for
later use.  Tune against multiple sizes by invoking `autotune` with
a list of representative shapes, for example:

```python
datasets = {
  "s": (
      torch.randn(2**16, device="cuda"),
      torch.randn(2**16, device="cuda"),
  ),
  "m": (
      torch.randn(2**20, device="cuda"),
      torch.randn(2**20, device="cuda"),
  ),
  "l": (
      torch.randn(2**24, device="cuda"),
      torch.randn(2**24, device="cuda"),
  ),
}

for tag, args in datasets.items():
  config = my_kernel.autotune(args)
  config.save(f"configs/my_kernel_{tag}.json")
```

### Direct Control Over Autotuners

When you need more control, construct autotuners
manually. {py:class}`~helion.autotuner.pattern_search.PatternSearch` is the default
autotuner:

```python
from helion.autotuner import PatternSearch

bound = my_kernel.bind(example_inputs)
tuner = PatternSearch(
    bound,
    example_inputs,
    # Double the defaults to explore more candidates:
    initial_population=200,  # Default is 100.
    copies=10,               # Default is 5.
    max_generations=40,      # Default is 20.
)
best_config = tuner.autotune()
best_config.save("configs/my_kernel.json")
```

- Adjust `initial_population`, `copies`, or `max_generations` to trade
tuning time versus coverage, or try different search algorithms.

- Use different input tuples to produce multiple saved configs
(`my_kernel_large.json`, `my_kernel_fp8.json`, etc.).

- Tuning runs can be seeded with `HELION_AUTOTUNE_RANDOM_SEED` if you
need more reproducibility; see {doc}`api/settings`.  Note this only
affects which configs are tried, not the timing results.

## Deploy a Single Config

If one configuration wins for every production call, bake it into the decorator:

```python
best = helion.Config.load("configs/my_kernel.json")

@helion.kernel(config=best)
def my_kernel(x, y):
    ...
```

The supplied `config` applies to **all** argument shapes, dtypes, and
devices that hit this kernel. This is ideal for workloads with a single
critical path or when you manage routing externally. `helion.Config.save`
/ `load` make it easy to copy configs between machines; details live
in {doc}`api/config`.  One can also copy and paste the config from the
autotuner output.


## Deploy Multiple Configs

When you expect variability, supply a small list of candidates:

```python
candidate_configs = [
    helion.Config.load("configs/my_kernel_small.json"),
    helion.Config.load("configs/my_kernel_large.json"),
]

@helion.kernel(configs=candidate_configs, static_shapes=True)
def my_kernel(x, y):
    ...
```

Helion performs a lightweight benchmark (similar to Triton’s autotune)
the first time each specialization key is seen, running each provided
config and selecting the fastest.

A key detail here is controlling the specialization key, which
determines when to re-benchmark. Options include:

- **Default (`static_shapes=True`):** Helion shape-specializes on the exact
  shape/stride signature, rerunning the selection whenever those shapes
  differ. This delivers the best per-shape performance but requires all calls
  to match the example shapes exactly.

- **`static_shapes=False`:** switch to bucketed dynamic shapes. Helion
  reuses results as long as tensor dtypes and device types stay constant.
  Shape changes only trigger a re-selection when a dimension size crosses
  the buckets `{0, 1, ≥2}`. Helion also tracks whether any tensor exceeds the
  `torch.int32` indexing limit (more than ``2**31 - 1`` elements) and will
  automatically regenerate code with 64-bit indexing in that case. Use this
  mode when you need one compiled kernel to handle many input sizes, and pin
  ``@helion.kernel(..., index_dtype=torch.int64)`` if large tensors are the norm
  so you avoid an extra specialization boundary.

- **Custom keys:** pass `key=` to group calls however you like.
This custom key is in addition to the above.

As an example, you could trigger re-tuning with power-of-two bucketing:

```python
@helion.kernel(
    configs=candidate_configs,
    key=lambda x, y: helion.next_power_of_2(x.numel()),
    static_shapes=False,
)
def my_kernel(x, y):
    ...
```

See {doc}`api/kernel` for the full decorator reference.

## Selective Shape Specialization

The `static_shapes` setting is all-or-nothing: either every dimension is
specialized (`static_shapes=True`) or dimensions are bucketed dynamically
(`static_shapes=False`). Sometimes you want finer control - specializing
only specific dimensions while keeping others dynamic.

Helion provides two APIs for selective shape specialization:

| API | Location | Effect |
|-----|----------|--------|
| `hl.specialize()` | Inside kernel | Dimension always specialized for all calls |
| `torch._dynamo.mark_static()` | Outside kernel | Dimension specialized only for marked tensors |

### `hl.specialize()` - Internal Specialization

Use {func}`~helion.language.specialize` inside the kernel to make specific
dimensions compile-time constants. This applies to **every call** to the kernel:

```python
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=False)
def rms_norm_fwd(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    m, n = x.size()
    hl.specialize(n)  # hidden dimension becomes a compile-time constant
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms = torch.rsqrt(mean_x_squared + eps)
        normalized = x_tile * inv_rms[:, None]
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
    return out

# Every call specializes on n - different hidden sizes = different cache entries
weight_4096 = torch.randn([4096], device="cuda")
weight_2048 = torch.randn([2048], device="cuda")
result1 = rms_norm_fwd(torch.randn([2048, 4096], device="cuda"), weight_4096)  # compiles for n=4096
result2 = rms_norm_fwd(torch.randn([1024, 4096], device="cuda"), weight_4096)  # reuses n=4096
result3 = rms_norm_fwd(torch.randn([2048, 2048], device="cuda"), weight_2048)  # compiles for n=2048
```

Use `hl.specialize()` when a dimension is performance-critical and you want
it specialized regardless of how the kernel is called.

### `torch._dynamo.mark_static()` - External Specialization

Use `torch._dynamo.mark_static()` **before** calling the kernel to specialize
dimensions on specific tensors. This is useful when you want the **same kernel**
to serve both dynamic and specialized code paths:

```python
@helion.kernel(static_shapes=False)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty([m, n], device=x.device, dtype=x.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out

# Dynamic call - all dimensions remain symbolic
x_dyn = torch.randn([m, k], device="cuda", dtype=torch.float16)
y_dyn = torch.randn([k, n], device="cuda", dtype=torch.float16)
result = matmul(x_dyn, y_dyn)

# Specialized call - mark specific dimensions as compile-time constants
x_opt = torch.randn([64, 128], device="cuda", dtype=torch.float16)
y_opt = torch.randn([128, 56], device="cuda", dtype=torch.float16)
torch._dynamo.mark_static(x_opt, [0, -1])  # specialize dims 0 and -1 (M and K)
torch._dynamo.mark_static(y_opt, 1)        # specialize dim 1 (N)
result = matmul(x_opt, y_opt)  # generates code with 64, 128, 56 as constants
```

This pattern enables a **single kernel definition** to serve both:
- Fully dynamic fallback paths (for rare edge-case shapes)
- Optimized hot paths (with shape constants baked into generated code)

### Combining Both APIs

The two APIs form a **union** - you can use `hl.specialize()` for dimensions
that should always be specialized, and `mark_static()` for additional
per-call specialization:

```python
@helion.kernel(static_shapes=False)
def fn(x: torch.Tensor) -> torch.Tensor:
    hl.specialize(x.size(0))  # dim 0 always specialized (internal)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * 2
    return out

# mark_static on dim 1 combines with hl.specialize on dim 0
x = torch.randn([320, 640], device="cuda")
torch._dynamo.mark_static(x, -1)  # specialize dim 1 (external)
result = fn(x)  # both 320 and 640 become constants
```

### Cache Behavior

Each unique combination of specialized dimension values creates a separate
cache entry:
- Unspecialized calls share one dynamic cache entry
- Calls with `mark_static()` create entries keyed by the specialized values
- Different specialized values (e.g., `[64, 128]` vs `[48, 96]`) create separate entries

## Advanced Manual Deployment

Some teams prefer to skip all runtime selection, using Helion only as
an ahead-of-time compiler.  For this use case we provide `Kernel.bind`
and `BoundKernel.compile_config`, enabling wrapper patterns that let
you implement bespoke routing logic.  For example, to route based on
input size:

```python
bound = my_kernel.bind(example_inputs)

small_cfg = helion.Config.load("configs/my_kernel_small.json")
large_cfg = helion.Config.load("configs/my_kernel_large.json")

small_run = bound.compile_config(small_cfg)  # Returns a callable
large_run = bound.compile_config(large_cfg)

def routed_my_kernel(x, y):
    runner = small_run if x.numel() <= 2**16 else large_run
    return runner(x, y)
```

`Kernel.bind` produces a `BoundKernel` tied to sample
input types. You can pre-compile as many configs as you need using
`BoundKernel.compile_config`.  **Warning:** `kernel.bind()` specializes,
and the result will only work with the same input types you passed.

- With `static_shapes=True` (default) the bound kernel only works for the
exact shape/stride signature of the example inputs.  The generated code
has shapes baked in, which often provides a performance boost.

- With `static_shapes=False` it will specialize on the input dtypes,
  device types, and whether each dynamic dimension falls into the 0, 1,
  or ≥2 bucket.  Python types are also specialized.  For dimensions that
  can vary across those buckets, supply representative inputs ≥2 to avoid
  excessive specialization.  Just like the autotuning flow above, Helion
  records whether any tensor crosses the int32 indexing limit when
  `static_shapes=False`; explicitly set `index_dtype=torch.int64` if your
  deployment commonly exceeds that threshold to avoid recompilation.

If you need to support multiple input types, bind multiple times with
representative inputs.

Alternately, you can export Triton source with
`bound.to_triton_code(small_cfg)` to drop Helion from your serving
environment altogether, embedding the generated kernel in a custom
runtime.  The Triton kernels could then be compiled down into PTX/cubins
to further remove Python from the critical path, but details on this
are beyond the scope of this document.
