# Investigation: IMA with `hl.grid(N)` and dynamic N

## Summary

`chunk_bwd_dh_diag_fused` and similar kernels using `hl.grid(N)` where N is
a dynamic (non-specialized) value produce an illegal memory access (IMA) when:

1. The kernel is first compiled for one value of C (e.g. C=32, giving N=T/C=4)
2. Then called with a different C (e.g. C=64, giving N=32)
3. The second call is exercised rapidly via `do_bench`

The crash does **not** reproduce:
- With `CUDA_LAUNCH_BLOCKING=1` (serialized kernel launches)
- When the kernel is called in isolation (even 100x rapidly)
- When only one value of C is used throughout the process
- In a minimal standalone repro of the same kernel pattern

The crash **does** reproduce consistently when the full backward pipeline
runs after a test phase that compiled with different C.

## Workaround

Specializing N with `hl.specialize()` fixes the IMA:

```python
# Before (crashes):
N = q_scaled.size(1)

# After (works):
N = hl.specialize(q_scaled.size(1))
```

This causes a recompilation per distinct N value, which is acceptable since
N = T/C typically takes a small number of values per model configuration.

## Generated code analysis

The generated Triton kernel receives N as a dynamic argument and uses it in:

```python
for offset_3 in tl.range(0, tl.cast(N, tl.int32), ...):
    sub_1 = -1 + N + -1 * offset_3   # i.e. N - 1 - i_t
    tl.store(dh_all + sub_1 * dh_all_stride_1 + ..., ...)
```

The loop bounds and indexing look correct. However, the kernel also has
`_RDIM_SIZE_2: tl.constexpr` parameters that are set to
`triton.next_power_of_2(dh_init.size(1))` — these control the size of
`tl.arange` used for the D dimension.

When the kernel is compiled for shape1 (D=32 → `_RDIM_SIZE_2=32`) and then
a new compilation happens for shape2 (D=128 → `_RDIM_SIZE_2=128`), both
compilations share the same source but differ in constexpr values. The
suspicion is that something in the Triton compilation or caching interacts
badly when `hl.grid(N)` has a dynamic range **and** the kernel has multiple
constexpr-specialized variants coexisting.

## What's NOT the cause

- **tensor_descriptor indexing**: crash reproduces with all-pointer indexing
- **persistent_blocked pid_type**: crash reproduces with flat pid_type
- **block_sizes**: crash reproduces with every tested block_size value
- **Any single config parameter**: every parameter variant crashes; only
  the presence/absence of a prior compilation with different C matters

## Affected kernels

All four kernels using `hl.grid(N)` with unspecialized N:

- `chunk_fwd_h_diag_fused` (line 336)
- `chunk_fwd_phase1_diag_fused` (line 409)
- `chunk_bwd_dh_diag_fused` (line 490)
- `chunk_bwd_dh_correction_diag_fused` (line 524)

## Likely root cause

The interaction between:
1. `hl.grid(N)` producing `tl.range(0, N)` with dynamic N
2. Multiple constexpr-specialized compilations of the same kernel source
3. Rapid repeated execution via `do_bench` without full synchronization

may cause Triton to reuse a compiled kernel variant with incorrect grid
bounds or rdim sizes. This needs investigation at the Helion/Triton level,
potentially with `compute-sanitizer --tool memcheck` to identify the exact
out-of-bounds access.
