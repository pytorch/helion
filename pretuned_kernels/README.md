# Helion Pretuned Kernels

This directory contains pretuned Helion kernels, benchmark shape sweeps, and
checked-in AOT heuristic files.  They are meant to be useful copy/paste starting
points for common kernel patterns while also being runnable examples for people
who want to quickly try Helion.

The checked-in `sm100` heuristics are pretuned on NVIDIA B200 so these kernels
can run immediately without online autotuning.  Treat the files as kernel
recipes: copy the kernel and its local `_helion_aot_*` heuristic into your code,
then retune when your target shapes or hardware differ materially from the
included sweep.

Each kernel module has a `main()` that benchmarks against PyTorch eager.

## File structure

```
pretuned_kernels/
├── README.md
├── vector_add/
│   ├── vector_add.py                          # the kernel + main()
│   └── _helion_aot_vector_add_cuda_sm100.py   # auto-loaded heuristic
├── softmax/
├── layer_norm/
├── rms_norm/
└── cross_entropy/
```

| Kernel | Shape sweep | PyTorch baseline |
|---|---|---|
| `vector_add` | `2**i for i in range(19, 29)` | `x + y` |
| `softmax` | Triton tutorial `M=4096, N=128*i for i in range(2, 100)` + realistic long-context shapes | `F.softmax` |
| `layer_norm` | Triton tutorial `M=4096, N=512*i for i in range(2, 32)` + realistic hidden-size shapes | `F.layer_norm` |
| `rms_norm` | TritonBench `(M=2048, H)` default + NPOT shapes + realistic LLM hidden-size and production-style shapes | `F.rms_norm` |
| `cross_entropy` | TritonBench/Liger token-vocab sweep + realistic LLM vocabulary shapes | `F.cross_entropy` |

## Scope

Use this directory as a collection of pretuned kernels and runnable examples.
For production code, copy the relevant kernel pattern into the application.  If
the shapes or target hardware differ from the included sweep, generate and
commit an AOT heuristic for the application's target shapes and hardware.

For AOT pretuned kernels, Helion's runtime looks for
`_helion_aot_<kernel>_cuda_sm<NN>.py` next to the kernel source file.  Helion
looks for AOT heuristic files for the current compute capability first, then
falls back to older compatible CUDA/ROCm capabilities.  For example, on
`sm120`, if no `sm120` heuristic exists, an `sm100` heuristic file can be used.

## Running benchmarks

Each kernel module has a `main()` that benchmarks the Helion kernel against
PyTorch eager across the included shape set:

```bash
cd pretuned_kernels/softmax
python softmax.py
```

## Adding a heuristic for new hardware

These kernels ship pretuned heuristics for `sm100` (B200).  To add another GPU,
run the AOT autotune workflow on that hardware against the kernel — the runner
emits a new
`_helion_aot_<kernel>_<device>_<cc>.py` next to the kernel source, which
you commit alongside the existing one(s).  Helion picks the right one at
runtime based on the running GPU's compute capability (with fallback to
older compatible capabilities, e.g. `sm120` → `sm100`).

See [`docs/aot_autotuning.md`](../docs/aot_autotuning.md) for the
end-to-end workflow, runner CLI, generated artifacts, and runtime
fallback rules — including a worked "Pretuning a kernel for new
hardware" walkthrough.
