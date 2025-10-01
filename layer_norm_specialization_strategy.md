# Layer-Norm Specialization Progress

## Implemented Changes
- Added `SpecializedValue` wrapper so `hl.specialize` preserves the originating SymInt instead of collapsing to a plain integer (`helion/language/constexpr.py`).
- Extended `CompileEnvironment` with a provenance-aware symbol registry that hands back canonical SymInts and links newly minted symbols to their specialized counterparts (`helion/_compiler/compile_environment.py`).
- Taught persistent reductions to reuse specialized extents for masking while still emitting a Triton-friendly power-of-two `_RDIM_SIZE` when the extent is known at specialization time (`helion/_compiler/reduction_strategy.py`).
- Updated tensor creation and indexing paths to keep specialized provenance alive through fake tensor construction and slicing, ensuring device-side tensors reuse the original symbol (`helion/language/creation_ops.py`, `helion/_compiler/type_propagation.py`).
- Updated the tile dispatcher to resolve `SpecializedValue` integers (and literal ints) back to their reduction block ids so generated kernels pick up the correct RDIM symbol instead of a baked-in extent (`helion/_compiler/tile_dispatch.py`).
- Relaxed host/device assignment rules for already-device tensors and allowed host returns to read device values, enabling the accumulator pattern in the reproducer (`helion/_compiler/type_propagation.py`).
- Monkey-patched PyTorch fake tensor `aten.sum.dim_IntList` so reductions reuse the canonical SymInt, registered the fast-path hook, and relied on the provenance registry for shape reuse (`torch/_subclasses/fake_impls.py`).
- Adjusted device IR capture heuristics to treat device-resident tensors as proper loop arguments rather than scalars (`helion/_compiler/device_ir.py`).

## Current Status
- The original `s42` vs `u*` mismatch is resolved; reductions now reuse the specialized symbol and avoid guard violations.
- The `minimal_repro.py` kernel now lowers without triggering Triton's power-of-two guard; `_RDIM_SIZE_1` stays bound to the reduction symbol and the mask uses the true (non-power-of-two) extent.
- Coverage for the minimal reproducer now lives in `test/test_specialize.py:TestSpecialize.test_specialize_layer_norm_backward_sum`, which exercises the kernel and passes.

## Next Steps
1. Re-run the broader `layer_norm_bwd_symbolic_mismatch.py` scenario to confirm coverage beyond the minimal repro.
2. Add targeted tests for the `SpecializedValue` flow, covering tensor creation, slicing, and reduction cases to guard against regressions.
