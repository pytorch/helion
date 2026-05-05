# Ahead-of-Time (AOT) Autotuning

Helion's standard autotuning runs at the first call to a kernel and tunes
for the *exact* arguments seen.  That is great for development, but it
forces every cold-start invocation to pay a long tuning cost, and it does
not help when a deployed kernel needs to serve many different input shapes
without retuning each one.

The **AOT autotuning workflow** addresses both problems.  Offline, you
sweep a kernel over a representative shape set, tune each shape, then
distill the (shape → best-config) table into a small **decision-tree
heuristic**.  At runtime, the heuristic picks a tuned config for the
caller's shape in microseconds — no autotuner involved.  The tutorials
under [`tutorials/`](https://github.com/pytorch/helion/tree/main/tutorials)
ship pretuned heuristic files that demonstrate this end-to-end.

This doc covers the user-facing API, the offline workflow, and how the
runtime locates and applies a heuristic.  For the broader deployment story
(single-config and multi-config decoration patterns, `Kernel.bind`, etc.)
see {doc}`deployment_autotuning`.

## When to use AOT

- You ship Helion in a service or library that handles a wide range of
  input shapes (e.g., variable token counts in an LLM serving stack,
  variable matmul dimensions).
- You want zero-cost (or near-zero-cost) per-shape config selection at
  runtime — neither full autotuning nor a small benchmark sweep.
- You can identify a representative shape sweep at offline tuning time.
  AOT generalizes by *interpolating* the decision tree over input
  features (shape sizes, dtype, etc.); shapes outside the tuned regime
  may still pick a reasonable config but are not guaranteed optimal.

## Quick start: decorate a kernel for AOT

Use {py:func}`helion.experimental.aot_kernel` instead of
{py:func}`helion.kernel`.  The decorator wires the kernel into an
{py:class}`~helion.autotuner.aot_cache.AOTAutotuneCache`, which is what
loads the generated heuristic at runtime:

```python
import torch
import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out
```

Until you generate a heuristic, calling `vector_add` falls back to the
default Helion config and prints a one-time warning telling you to run the
AOT runner.

The decorator accepts a few extras:

- `batched=...` marks tensor dimensions whose size does not affect the
  optimal config (e.g., the batch dimension of an attention kernel), so
  the heuristic key ignores them and config selection generalizes
  across batch sizes.
- `key=fn` lets you supply a custom feature extractor when the default
  shape-feature extraction is not what you want (for example, deriving
  `(M, N, K)` from a custom argument layout).
- `collect_fn` / `measure_fn` define the input sweeps the offline
  workflow uses for each phase, so the workflow can run end-to-end from
  a single benchmark invocation.

See [`examples/aot_example.py`](https://github.com/pytorch/helion/blob/main/examples/aot_example.py)
for runnable demonstrations of each option, and
[`helion/experimental/aot_kernel.py`](https://github.com/pytorch/helion/blob/main/helion/experimental/aot_kernel.py)
for the full decorator reference.

## Offline workflow: collect → measure → evaluate

The AOT runner orchestrates a three-phase workflow over a benchmark
script that exercises the kernel across the shapes you care about:

```bash
python -m helion.experimental.aot_runner -- python my_benchmark.py
```

`my_benchmark.py` is *your* script — it imports the kernel and calls it
on every shape in the sweep.  The runner re-invokes the script three
times with different `HELION_AOT_MODE` settings:

1. **`collect`** — for each unique shape, autotune the kernel from
   scratch and record `(kernel, shape, config, timing)` triples in
   `tuned_configs_<hardware_id>.json`.
2. **`measure`** — replay the benchmark with every config discovered
   during `collect`, on every shape, and record the resulting timings
   in `measurements_<hardware_id>.csv`.  This step is what lets the
   heuristic generator pick a small config subset that performs well
   *across all shapes*, not just on the shape it was originally tuned
   for.
3. **`evaluate`** — fit a decision tree over the measurement matrix
   and emit the heuristic file (see *Generated artifacts* below for
   the layout).  Each tree leaf names a config index, and the
   companion list of configs is the output of subset selection from
   the measure phase.

Useful runner flags (run `python -m helion.experimental.aot_runner --help`
for the full list):

- `--kernel <name>` — restrict the workflow to specific kernels.
- `--output-dir <dir>` — where to drop `tuned_configs_*.json`,
  `measurements_*.csv`, and per-phase logs.
- `--threshold` / `--max-configs` — performance target for subset
  selection; trade off heuristic size vs. how close every shape gets to
  its individually-tuned best config (default: at most 10x slowdown
  vs. best, no more than 10 configs).

You can also run any phase manually if you only need one:

```bash
HELION_AOT_MODE=collect HELION_AOT_DATA_DIR=./aot_data python my_benchmark.py
HELION_AOT_MODE=measure HELION_AOT_DATA_DIR=./aot_data python my_benchmark.py
# ...then call the heuristic generator directly via aot_runner.
```

## Generated artifacts

The runner emits two kinds of output: per-run *data files* (collected
configs, raw measurements, logs) under the data directory, and the
*heuristic file* itself, which lands next to the kernel source so the
AOT cache can find it at runtime.

```
.helion_aot/                                 # default --output-dir
└── 20260505_120000_ab12cd/                  # one subdir per runner invocation
    ├── logs/
    │   ├── collect_<hardware_id>.log        # stdout/stderr of the collect phase
    │   └── measure_<hardware_id>.log        # stdout/stderr of the measure phase
    ├── tuned_configs_<hardware_id>.json     # collect output: best config per shape
    ├── measurements_<hardware_id>.csv       # measure output: timing matrix
    └── heuristic_summary_<hardware_id>.json # evaluate output: subset-selection report

<your kernel source dir>/
├── my_kernel.py                             # kernel source
└── _helion_aot_my_kernel_<device>_<cc>.py   # generated heuristic (commit this)
```

`<hardware_id>` is `<device_kind>_<sanitized_device_name>_<runtime_version>`
(e.g. `cuda_NVIDIA_B200_13.1`).  `<device>_<cc>` is the device kind and
compute capability used for runtime lookup (e.g. `cuda_sm100`,
`rocm_gfx950`).

Only the heuristic file is meant to be checked in — it is the entire
runtime artifact.  Everything under the data directory is disposable
per-run state used to build the heuristic.

## Heuristic file format

The generated heuristic file is plain Python — no JSON, no pickled
objects.  For a kernel `foo` it exposes two top-level functions:

```python
def key_foo(*args) -> int:
    """Decision-tree-derived config index for the given args."""
    ...

def autotune_foo(*args) -> dict:
    """Return the config dict for the given args."""
    _C = [
        {"block_sizes": [...], "num_warps": ...},  # config 0
        {"block_sizes": [...], "num_warps": ...},  # config 1
        # ...
    ]
    return _C[key_foo(*args)]
```

`key_foo` doubles as the runtime *cache key*: the
`AOTAutotuneCache` stores compiled artifacts keyed on its return value,
so once a config has been compiled for a given key the kernel call is
just `dict-lookup → cached compiled function`.

Because the file is plain Python, it is easy to inspect, hand-edit, or
regenerate.  Helion ignores `_helion_aot_*.py` for ruff and pyrefly so
they do not need to follow the project's lint rules.

## Runtime lookup and compute-capability fallback

At kernel-call time, `AOTAutotuneCache` looks for a heuristic file in
this order:

1. `$HELION_HEURISTIC_DIR` if set (useful for A/B-comparing heuristics
   generated under different conditions).
2. `_helion_aot_<kernel>_<device_kind>_<compute_capability>.py` next
   to the kernel's source file.
3. The same filename but with older compatible compute capabilities
   within the same device family — e.g. on `sm120`, Helion tries
   `sm120`, `sm100`, `sm90`, ... in order.  A single tuned heuristic
   can therefore serve multiple GPU generations with the same ISA.

If no file is found, the cache falls back to the kernel's default
config and prints a one-time warning naming the AOT runner.

## Pretuning a kernel for new hardware

A heuristic file is specific to one device kind + compute capability —
to add support for a new GPU, generate a fresh heuristic on that
hardware and commit it alongside any existing files.  The recipe works
for any GPU; the device-kind / compute-capability suffix is whatever
{py:class}`~helion.autotuner.aot_cache.HardwareInfo` reports for the
target.

### Step-by-step

1. **Get on the target GPU.**  The runner detects the device with
   `torch.cuda.get_device_capability()` (or the ROCm / XPU equivalent)
   and bakes that into the emitted filename, so this must run on the
   actual hardware you are targeting — not the laptop you happen to
   be editing on.

2. **Confirm the kernel uses `@helion.experimental.aot_kernel(...)`**
   (see *Quick start* above).

3. **Run the AOT workflow.**  Point the runner at any benchmark script
   that exercises the kernel across the shape sweep you want covered:

   ```bash
   # Tutorial example: tune layer_norm on the current GPU.
   python -m helion.experimental.aot_runner -- python tutorials/layer_norm/layer_norm.py

   # User-authored kernel: same pattern.
   python -m helion.experimental.aot_runner -- python my_benchmark.py
   ```

   The three phases run back-to-back.  Plan for a long wall-clock —
   `collect` runs full autotuning on every distinct shape, which is
   typically 5–15 minutes per shape on a recent GPU.

4. **Locate the emitted heuristic file.**  See *Generated artifacts* —
   the heuristic file lands next to the kernel source (e.g.
   `tutorials/layer_norm/_helion_aot_layer_norm_cuda_sm90.py`); the
   data dir holds disposable intermediates.

5. **Commit the heuristic file.**  Keep the existing pretuned files
   for other compute capabilities — the runtime picks the right one
   per GPU.

6. **Verify.**  Re-run the benchmark on the target GPU; the
   `[0s] Found cached config for <kernel>, skipping autotuning.` line
   in the output confirms the new heuristic is being used.

### Adding shapes to the sweep

If you add new shapes to the kernel's benchmark, regenerate the
heuristic the same way — Helion does not extrapolate cleanly outside
the shapes used during `collect`/`measure`.  Tuning on a sweep that
covers the small / medium / large regimes you expect at serving time
gives the decision tree the leverage it needs to generalize within
those bounds.

## Related references

- {doc}`deployment_autotuning` — the broader deployment story (single-
  and multi-config decoration, `Kernel.bind`, dynamic-shape buckets).
- {py:class}`~helion.autotuner.aot_cache.AOTAutotuneCache` — the
  runtime cache that consumes generated heuristics.
- [`examples/aot_example.py`](https://github.com/pytorch/helion/blob/main/examples/aot_example.py)
  — runnable end-to-end example.
- [`tutorials/`](https://github.com/pytorch/helion/tree/main/tutorials)
  — pretuned tutorial kernels you can use as templates.
