# Choosing a Backend

Helion compiles the same Python tile language through several code generation
backends. Choose a backend for your hardware before autotuning; Helion tunes
configurations within that backend, and does not search across backends.

**On NVIDIA B200 or B300, we recommend `HELION_BACKEND=cute` for better
performance.** Install the {ref}`CuTe dependencies <cute-backend-installation>`
and use PyTorch built with CUDA 13 or later. Triton remains the default on all
hardware; Helion emits a once-per-process recommendation when another backend
is used on these GPUs, without changing your selection.

The recommendation uses Python's `UserWarning`, like the experimental-backend
warning. If you turn warnings into errors (`-W error` or pytest's
`filterwarnings = ["error"]`), the first matching compilation raises an
exception. Python's warning filters can ignore this recommendation by message;
Helion's `ignore_warnings` setting applies to `helion.exc.BaseWarning` instead.

## Supported Backends and Status

```{include} ../README.md
:start-after: <!-- backend-table-start -->
:end-before: <!-- backend-table-end -->
```

These labels describe **Helion's integration**, not the maturity of the
underlying compiler. They match the backend states in this Helion version:

- **Stable:** the established compatibility path. Hardware, operation and
  dependency restrictions still apply.
- **Beta:** usable for evaluation and workloads covered by the backend, but
  backend-specific options do not have a backwards-compatibility guarantee.
- **Experimental:** under development, with incomplete coverage and evolving
  behavior. Helion warns once when an experimental backend is used.

## Selecting a Backend

Set the environment variable before importing modules that define kernels:

```bash
HELION_BACKEND=cute python my_kernel.py
```

Or choose per kernel:

```python
import helion
import torch

@helion.kernel(backend="cute")
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    ...
```

An explicit `backend=` argument takes precedence over `HELION_BACKEND`.
Without either, Helion uses `triton`. Environment defaults are read when a
kernel is defined, usually at import time. Selecting a backend does not
install its dependencies, move input tensors to a different device, or
automatically fall back to another backend when an operation is unsupported.
See {doc}`api/settings` for the full settings reference.

## Backend Details

### Triton

[Triton](https://github.com/triton-lang/triton) is the default and a good
starting point for NVIDIA and AMD GPUs. Follow {doc}`installation` for the
PyTorch and Triton requirements. Its search space includes tile sizes, warps,
pipeline stages and indexing strategies, subject to the installed compiler
and hardware. Triton remains useful when you need its operation coverage or
existing Triton-specific configurations.

### CuTe DSL

[CuTe DSL](https://github.com/NVIDIA/cutlass) exposes NVIDIA-specific layouts,
memory transfers and matrix instructions. Helion's beta backend uses them for
Blackwell optimizations, including TMA transfers, tensor-core matrix
multiplication and warp-specialized schedules. On **B200/B300**, start with
`HELION_BACKEND=cute` for better performance, then measure your workload.

Use the pinned `helion[cute]` dependencies for the Helion version you install;
for a source checkout, use `pip install -e '.[cute]'` or
`./scripts/install_cute.sh`. See the
{ref}`CuTe installation instructions <cute-backend-installation>`.
The backend requires a CUDA 13+ build of PyTorch, even if a newer system CUDA
toolkit is already installed. The B200/B300 recommendation does not imply
that every Blackwell GPU has the same instructions or performance profile.

### TileIR

TileIR feeds Triton code through NVIDIA's CUDA Tile IR toolchain. Use it to
evaluate that compiler on supported Blackwell hardware. Install
[Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR), then set
both `ENABLE_TILE=1` and `HELION_BACKEND=tileir`. Prefer a separate environment
when comparing it with upstream Triton, since they provide different Triton
distributions. The {doc}`tileir_backend` guide explains its `num_ctas` and
`occupancy` knobs and unsupported operations. For B200/B300 performance, our
starting recommendation is CuTe.

### Pallas

[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) targets Google TPUs.
Install `helion[pallas-tpu]` for JAX's TPU dependencies. Running with PyTorch
TPU tensors also requires a compatible `torch_tpu` installation; the extra
does not install that bridge. TPU setup depends on your PyTorch and TPU runtime.

For development without a TPU, install `helion[pallas-cpu]` and set
`HELION_BACKEND=pallas HELION_PALLAS_INTERPRET=1`. Interpret mode executes on
CPU for correctness and lowering checks; its timings do not predict TPU
performance. `pallas-interpret` is a test-runner alias, not a `HELION_BACKEND`
value. Pallas has its own dtype, operation and layout restrictions.

### Gluon

[Gluon](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)
is an experimental alternative that compiles Helion's Triton lowering through
Gluon, with explicit layout and Blackwell scheduling support. It currently
requires NVIDIA compute capability 10.x and Triton 3.8 or later. Use
`HELION_BACKEND=gluon` to evaluate it; do not assume that Triton configs or
features, such as block-pointer indexing, are supported unchanged.

### Metal

Metal generates Metal Shading Language for Apple GPUs. Use
`HELION_BACKEND=metal` with an MPS-capable PyTorch installation on macOS and
inputs on `device="mps"`. This experimental path has different dtype and
operation coverage from the Linux GPU backends; the CUDA examples need their
input-device selection adapted.

### FlyDSL

FlyDSL generates kernels for AMD GPUs. Install a compatible ROCm build of
PyTorch and the `flydsl` package, then set `HELION_BACKEND=flydsl`. ROCm
PyTorch uses `device="cuda"` for GPU tensors. This experimental backend has
a smaller configuration surface and incomplete operation coverage; Triton is
the stable starting point on AMD.

## Switching, Tuning and Deployment

Keep the algorithm in portable `helion.language` and PyTorch operations when
you intend to compare backends. Inline compiler code, assembly, supported
dtypes and some operations are backend-specific.

Retune for each backend and target device. A `helion.Config` saved for one
backend is not a portable performance preset: another backend may reject its
keys or interpret shared knobs differently. Remove incompatible explicit
`config=`/`configs=` values before tuning the new backend. Compare correctness
and performance on the same hardware and representative inputs, with
autotuning enabled; `autotune_effort="none"` is for iteration, not performance
comparisons.

Deploy the selected backend with its runtime dependencies and configurations
tuned for it. See {doc}`deployment_autotuning` and {doc}`api/config`.
To implement an external backend, see the experimental {doc}`api/backend_api`.
