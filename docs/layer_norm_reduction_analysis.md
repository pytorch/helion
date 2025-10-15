# Layer Norm Reduction Debugging Notes

_Date: October 15, 2025_

## Background

Running `python examples/layer_norm.py` attempts to exercise forward and backward layer-norm kernels generated via Helion. The backward weight/bias kernel performs a reduction across the batch dimension (`m`). The example sets up `m = 1_152_000` and `n = 16`, which stresses reduction code generation.

## Observed Failure

The Triton compiler aborts with

```
ValueError: numel (2097152) exceeds triton maximum tensor numel (1048576)
```

This comes from the generated kernel fragment

```python
rows = tl.arange(0, 2097152)
```

inside `_helion_layer_norm_bwd_dwdb`. The compiler hoists a `tl.arange` outside the reduction loop, producing a vector twice the Triton limit before masking occurs.

## Root Cause

* Helion’s reduction lowering hoists loads that depend on reduction indices (`rindex_*`) to the outer scope.
* When the reduction dimension (`m`) becomes very large, the hoisted `tl.arange` instantiates the full length at once, ignoring that the loop bodies iterate in manageable chunks.
* Triton enforces `TRITON_MAX_TENSOR_NUMEL = 1_048_576`. Any `tl.arange` (or block tensor) larger than that is rejected regardless of subsequent masking.

Thus, the generated kernel violates Triton’s per-tensor size limit before execution, long before masking can clip the iteration space.

## Generated Kernel Evidence

A fresh run of `python examples/layer_norm.py` (October 15, 2025) generated the Triton kernel below. The highlighted portion shows why the compilation aborts:

```python
rows = tl.arange(0, 2097152)
load = tl.load(x + (rows[:, None] * 16 + indices_0[None, :] * 1), (rows < 1152000)[:, None], other=0)
sum_1_acc = tl.full([2097152, _BLOCK_SIZE_0], 0, tl.float32)
for roffset_1 in tl.range(0, 2097152, _REDUCTION_BLOCK_1):
    v_4_copy = v_4
    sum_1_acc = sum_1_acc + v_4_copy
```

`rows` materializes the full reduction axis (2,097,152 elements) even though we later loop over 4,096-element tiles. The loop induction variable `roffset_1` never participates in the pointer math, so every iteration re-adds the same full tensor. Besides breaching Triton’s tensor-size cap, this defeats streaming reductions entirely.

## What Does Not Work

* Simply clamping `reduction_loop` block sizes to 1,048,576 is insufficient. The loop structure still asks for a `tl.arange` of size equal to the full reduction extent, not the per-iteration chunk.
* Replacing the `iota`/`arange` lowering with cached reduction indices is risky: expressions that reference those indices must remain inside the loop. Without restructuring the hoisting logic, this approach yields undefined symbols (`rindex_1` used before assignment) or duplicated work.

## Implementation Progress — October 15, 2025 (afternoon)

* `helion/_compiler/inductor_lowering.py::codegen_iota` now resolves reduction-spanning iotas to the active `LoopedReductionStrategy` index variable instead of emitting a global `tl.arange`. The lowering annotates FX nodes with `loop_index` metadata, including the reduction block id and mask.
* `SubscriptIndexing.create` consumes that metadata so pointer math and masks reuse loop-scoped `rindex_*`/mask variables. This prevents re-materializing large tensors once the index variable is in scope.
* `LoopedReductionStrategy.codegen_reduction` now keeps accumulators at the post-reduction shape and updates them per chunk (sum, argmin/argmax paths updated accordingly). Accumulated chunks are cast to the computation dtype before combining, and final casts occur after the loop.
* Guardrails were added:
  * `helion/runtime/settings.py` exposes `max_triton_tensor_numel` (default 1_048_576).
  * `CompileEnvironment.ensure_tensor_within_triton_limit()` plus a new `HelionReductionTooLarge` exception enforce the cap when block-level tensors would exceed Triton limits.
  * `GenerateAST.lift_symnode` now checks both `get_block_id` and `resolve_block_id`, reducing accidental hoisting of reduction-dependent symnodes.

### Current Failure Mode

Despite the changes above, `python examples/layer_norm.py` still fails—now with a different kernel:

```python
load = tl.load(x + (rindex_1[:, None] * 16 + indices_0[None, :] * 1), None)
...
for roffset_1 in tl.range(0, 2097152, _REDUCTION_BLOCK_1):
    rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
    v_4_copy = v_4
    ...
```

`rindex_1` is defined inside the loop, but earlier loads (`tl.load` for `x`/`grad_out`/`mean`/`rstd`) still live outside the loop body. As a result, the kernel references `rindex_1` before assignment, and Triton raises `NameError('rindex_1 is not defined')`. We need to re-schedule reduction-dependent loads/temporaries so they are emitted within the loop body (or lowered to expressions that incorporate the loop offset explicitly).

## Proposed Direction

### 1. Reuse loop-local reduction indices instead of global `arange`

* Detect `torch.ops.prims.iota` / `hl.arange` that span a reduction dimension and emit block-scoped expressions. In `helion/_compiler/inductor_lowering.py::codegen_iota`, resolve the extent via `CompileEnvironment.resolve_block_id(...)`; when we are inside `LoopedReductionStrategy` for that block, return the loop-local index (`ctx.cg.index_var(block_id)`) rather than allocating a fresh `tl.arange` of the full extent. Preserve dtype/step handling and reuse `ctx.cg.mask_var(block_id)` for masks instead of computing `(rows < ...)` tensors.
* Extend `SubscriptIndexing.create` in `helion/_compiler/indexing_strategy.py` to consume a lightweight metadata flag (e.g. `node.meta["loop_index_block_id"]`) emitted by the lowering above. When present, materialize pointer math as `({index_var}){expand}` and merge the loop mask, preventing any hoisted global tensors.
* Guard hoisting in `GenerateAST.lift_symnode`: if the sympy expression references a reduction block tracked by the active loop (`DeviceLoopState.block_id_to_info`), stop propagating the temporary past that loop. This keeps any residual reduction-dependent temporaries (e.g. affine transforms of `rindex`) within the correct scope.

### 2. Accumulate reduction chunks immediately

* Rework `helion/_compiler/reduction_strategy.py::LoopedReductionStrategy.codegen_reduction` so the accumulator matches the post-reduction shape, not the full pre-reduction tensor. For a sum-style reduction, allocate `tl.full([*output_shape], default, acc_dtype)` once, and within the loop:
  1. Evaluate the chunked `input_name` (now indexed by `rindex`).
  2. Reduce the chunk along `dim` locally (`chunk = self.call_reduction_function(...)`).
  3. Combine into the running accumulator via `combine_fn`.
  This removes the need for `2097152`-element temporaries and ensures every iteration only touches `_REDUCTION_BLOCK` elements.
* For argmin/argmax, maintain parallel accumulators for values and indices, combining chunk-level winners before the loop advances.
* Preserve dtype semantics: cast chunk reductions to accumulation dtype before combining, and defer the final cast until after the loop (mirroring today’s outer-suffix logic).

### 3. Enforce block-limited temporaries and fail fast when exceeded

* Wherever we still create `tl.full`/`tl.arange` inside reduction lowering, cap their shape using the chosen block size (`_REDUCTION_BLOCK_*`). Assert or guard that `block_size * remaining_dims <= TRITON_MAX_TENSOR_NUMEL`; if autotuning picks an unsafe value, raise an actionable `HelionReductionTooLarge` sourced from `helion/_compiler/compile_environment.py`.
* Consider surfacing a `max_reduction_chunk` knob in `helion/_compiler/compile_environment.py::CompileEnvironment.settings` so large-problem users can override Triton’s limit gracefully instead of hitting a ValueError in Triton internals.

### 4. Testing and instrumentation follow-ups

* Add an integration regression once the lowering changes land: synthesize a reduction with `m >= 1_500_000` in the examples/test harness to ensure generated kernels keep every tensor ≤ `TRITON_MAX_TENSOR_NUMEL`.
* Emit a debug counter (guarded by `HELION_DEBUG` or similar) when we rewrite an `iota` to a loop-local index, so future regressions in index scoping are easy to spot during kernel dumps.

## Next Steps

* Re-schedule reduction-dependent loads/code so they are emitted inside the loop where `rindex_*` exists. This likely requires evaluating the reduction body per chunk (inside `LoopedReductionStrategy.codegen_reduction`) instead of hoisting `output_name` entirely outside the loop.
* Ensure `tl.load` mask generation uses loop-local masks (`mask_var`) once the loads move into the loop; verify that guardrails still trigger if `_REDUCTION_BLOCK_*` is oversized.
* Remove temporary debug prints and re-run `python examples/layer_norm.py` plus high-dimension regression coverage to confirm no `tl.arange` exceeds the configured cap and that Triton kernels compile successfully.
* Once the kernel executes, add the planned large-dimension regression test and optional instrumentation (debug counter) so future regressions surface quickly.

## Open Questions

* Can we reuse Triton’s persistent reduction patterns to avoid generating large intermediate tensors altogether?
* Should Helion’s autotuner expose a user-facing knob for maximum reduction chunk size?

Once these structural changes land, the layer-norm example should compile and run successfully even for very large batch dimensions.

## Progress — October 15, 2025 (evening)

### What changed

* Refactored `ReductionLowering.codegen` so the reduction body is materialized via a callable (`materialize_reduction_input`) instead of eagerly generating AST in the pre-loop scope (`helion/_compiler/inductor_lowering.py`). The callable reuses the existing Inductor kernel handlers and gives the strategy a chance to schedule the body after the loop indices exist.
* Updated `LoopedReductionStrategy.codegen_reduction` to:
  * relocate any loop-index-dependent statements that accidentally landed in surrounding scopes by scanning both the enclosing statement list and the loop's `outer_prefix` and pushing them into `inner_statements` (`helion/_compiler/reduction_strategy.py`).
  * keep the previously generated reduction body statements (if any) inside the loop without re-materializing a large `tl.arange` tensor.

### Current outcome

* `python examples/layer_norm.py` still aborts with `NameError('rindex_1 is not defined')`. The generated Triton kernel continues to hoist the entire reduction body (loads of `x`, `grad_out`, `mean`, `rstd` and the ensuing pointwise ops) ahead of the loop, so the loop-local `rindex_1` definition never exists when those statements execute.
* Instrumentation shows that `materialize_reduction_input()` returns a cached `OpsValue` (e.g., `v_0`) whose defining statements were emitted before the reduction loop was constructed. The current relocation pass does not catch them because they reside in the parent statement list rather than the loop's `outer_prefix`.

### Next actions

1. Teach `LoopedReductionStrategy.codegen_reduction` (or a dedicated scheduler helper) to migrate already-emitted statements that reference `rindex_*` from the surrounding block into the loop body **before** the `for` node is attached. This likely requires capturing the indices of the newly appended AST nodes as soon as the reduction graph starts executing, so they can be spliced into `inner_statements` later.
2. Alternatively (or additionally), materialize the reduction body a second time inside the loop using a fresh Inductor virtualization context so the returned `OpsValue` produces new loop-scoped temporaries. That approach will need a cache flush (or a new `GraphInterpreter`) to prevent reuse of the original `v_*` names.
3. Once the body executes inside the loop, re-run `python examples/layer_norm.py` to confirm the `NameError` disappears and that Triton no longer sees tensors exceeding `settings.max_triton_tensor_numel`.
4. After the runtime fix lands, audit the relocation step to ensure it handles companion reductions (bias gradient) and leaves non-reduction statements untouched.

### Open questions

* Can we hook into the reduction graph entry point (e.g., `ReductionLoopGraphInfo.codegen`) to snapshot AST length and relocate everything emitted afterwards if it references loop indices?
* Would clearing Inductor's `OpsValue` cache (if exposed) before `materialize_reduction_input()` allow the loop-local re-materialization without additional AST surgery?
