# LayerNorm-BWD TritonBench Failures

## Current Status
- Helion's `examples/layer_norm.layer_norm_bwd` works for power-of-two widths but fails for other feature sizes during TritonBench accuracy checks.
- Failures happen _before_ gradients are compared: FakeTensor raises shape-mismatch errors such as `The size of tensor a (5632) must match the size of tensor b (u4)` or `Expected ndim=2, but got ndim=1` while compiling the Helion kernel.
- These errors originate from reductions that mix tensors whose last dimension is a Python int (after Helion specialization) with tensors whose dimension is tracked as a symbolic SymInt (`u4`). The specialization information is not propagated through the reduction, so FakeTensor still sees the symbolic size.

## Experiments (all unsuccessful)
- **Kernel-level rewrites**
  - Mimicked the RMSNorm backward pattern (per-block partial buffers, feature specialization with `hl.specialize`).
  - Replaced accumulation loops with block-level reductions and `torch.einsum`.
  - Attempted to allocate per-sample accumulators, remove partial buffers, and cast everything to FP32.
  - Each variant still surfaced symbolic sizes (often `u4` or `u3`) once reductions executed.
- **Forcing specialization**
  - Modified `hl.specialize` to record `_set_replacement` entries in the underlying `ShapeEnv` for every SymInt it encounters.
  - Added `torch._check` assertions and `narrow`/slicing operations to gate the shapes.
  - Despite writing replacements into `ShapeEnv`, reductions continued introducing new symbolic expressions that did not inherit the concrete value.
- **Runtime workarounds**
  - Reset kernels per input, forced autotune, ensured buffers were allocated with explicit device/shape. These changes did not influence the FakeTensor mismatch.

## Root Cause (Framework Level)
- When `hl.specialize` converts a SymInt into a Python int, the eventual arithmetic that produces reduction outputs materializes new symbols (`u4`, etc.).
- The specialization information recorded for the original symbol does not automatically carry over to expressions created during reductions, so FakeTensor treats those dimensions as independent symbolic variables.
- Without framework support for reusing or constraining these derived symbols, TritonBench encounters `int` vs `SymInt` mismatches and aborts the accuracy run.

## Suggested Path Forward
- Extend Helion's type propagation / `ShapeEnv` integration so that specialization constraints apply transitively to expressions created inside reductions. That likely means:
  - Tracking the relationship between the original SymInt and any SymPy expressions derived from it, and
  - Updating `_set_replacement` (or a new helper) to rewrite subsequent symbols produced during reductions to concrete integers when the original dimension was specialized.
- Once the framework maintains consistent symbolic information through reductions, the existing layer-norm backward kernel should compile for arbitrary hidden sizes and TritonBench accuracy checks should pass.

Until that happens, the benchmark will continue to fail for widths that are not powers of two, because each accuracy run rebuilds the kernel and re-encounters the same FakeTensor constraint errors.

### Refinement: Reusing Symbols in Reductions
- The core bug is that reductions mint a fresh SymInt (e.g., `u4`) for the reduced dimension even when the input dimension was already specialized.
- Potential framework fixes:
  - Teach the reduction lowering/type propagation to reuse the original symbol if the reduction output length is algebraically the same dimension.
  - Have `CompileEnvironment.add_kernel_tensor_size` (and related helpers) consult specialization data before creating a new symbol; if the size expression only depends on specialized symbols, emit the concrete integer directly.
  - Enhance `hl.specialize` + ShapeEnv so any future SymInts derived from the specialized dimension map back to the original symbol (e.g., via `_set_replacement`).
- With symbol reuse in place, reductions will no longer introduce `u4`, so FakeTensor sees consistent concrete sizes (`5120 == 5120`) and TritonBench accuracy runs should succeed across all feature widths.
