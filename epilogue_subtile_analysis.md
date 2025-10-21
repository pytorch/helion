# Epilogue Subtile Bias Placement Analysis

## Current Flow
- `TensorDescriptorIndexingStrategy.codegen_store` forwards the full epilogue expression into `_codegen_epilogue_subtile_store` (`helion/_compiler/indexing_strategy.py:389-468`).
- `_codegen_epilogue_subtile_store` lifts that value, reshapes, and splits it, so pointwise ops like `acc + bias` execute on the full tile before reshaping.
- Result: generated Triton kernels perform bias add pre-subtile, as seen in `HELION_PRINT_OUTPUT_CODE` output where `acc + bias` appears before the reshape/split sequence.

## Target Behavior
- Apply pointwise epilogue ops after the subtile split so each subtile stores the finalized value.
- Maintain the current subtile store fallback when the pattern is unsupported.

## Proposed Architecture
- Create an “epilogue graph base” that captures the pure pointwise DAG feeding a device store.
  - Nodes hold originating AST, dependencies, and `TypeInfo` (from `ExtendedAST._type_info`).
  - Roots are non-pointwise producers (accumulator, tensor loads, reductions).
- Extend `Codegen.lift` to populate a def-use map (`temp_name -> defining ast.AST`) for later traversal.
- Add `PointwiseGraphCollector`:
  - Walk backward from `store_value` gathering pointwise nodes.
  - Validate against an allowlist (binary ops, casting, comparisons, simple elementwise calls).
  - Abort if any unsupported op, aliasing risk, or missing shape info is encountered.
- Introduce `EpilogueGraphBase.materialize(subtile_idx)`:
  1. Identify subtile geometry (split axis, factor, offsets) via `CompileEnvironment` block metadata.
  2. Emit roots: reuse values invariant to the split or create sliced views when needed.
  3. Replay the graph topologically to clone pointwise ops per subtile, substituting the subtile tensors.
  4. For the accumulator root, emit the `tl.reshape(...).permute(...).split()` once, cache the tuple, and index per subtile.
- Update `_codegen_epilogue_subtile_store` workflow:
  1. Attempt to build the graph. On failure, fall back to the existing reshape/split of the full value.
  2. Hoist shared computations (bias loads, scalar constants) ahead of subtile replay so both halves reuse them.
  3. Materialize each subtile expression and emit individual stores with the per-subtile tensors.

## Implementation Outline
1. **Infrastructure**
   - New helper module (e.g., `_compiler/epilogue_utils.py`) housing `PointwiseGraphCollector`, `EpilogueGraphBase`, and `SubtileContext`.
   - Extend `GenerateAST.add_statement` / `Codegen.lift` with a def-use index for temporary reuse and purity tagging.
2. **Collector & Graph**
   - Traverse AST using the def-use index, gathering only pure pointwise nodes.
   - Use `TypeInfo.fake_value` sizes and `CompileEnvironment.get_block_id` to understand broadcast semantics.
   - Track side-effect-free status to allow safe duplication across subtiles.
3. **Materialization**
   - `SubtileContext` provides expressions for `offset_1` and the shifted offset, along with block sizes (`block_m`, `block_n_half`).
   - Graph replay clones operations with the correct slicing; broadcasting logic inserts `[:, :half]` vs. `[:, half:]` or equivalent reshape operations.
4. **Integration**
   - `_codegen_epilogue_subtile_store` first tries the graph path; on success it emits two stores using the subtile-specific expressions, avoiding a full-tile reshape.
   - Keep fallback path to ensure unchanged behavior when patterns are unsupported.
5. **Testing**
   - Add a kernel exercising bias + cast + activation to confirm codegen emits pointwise ops per subtile post-split.
   - Include a negative test where a non-pointwise op forces fallback.

## Risks & Mitigations
- **Shape inference errors**: rely on `TensorType.fake_value`, symbolic shape metadata, and explicit guards; bail out instead of emitting incorrect code.
- **Increased temp usage**: hoist shared loads and let DCE clean unused temps; monitor shared memory usage post-change.
- **Unsupported ops**: start with conservative allowlist and expand as needed; ensure fallback path stays stable.
- **Broadcast semantics**: carefully handle tensors that are broadcast along the split axis by adjusting slicing logic per subtile.

## Next Steps
1. Prototype the collector and materializer, validating on `HELION_PRINT_OUTPUT_CODE` kernels.
2. Wire the new path into `_codegen_epilogue_subtile_store` behind a feature flag for staged rollout.
3. Re-run `examples/matmul.py` to confirm bias ops execute after the subtile split and resource usage remains within limits.

## What does "graph base" mean
They’re asking for a little “mini graph” (a base or root object) that represents the pointwise computation feeding the epilogue store. Instead of looking only at the final AST node when emitting the store, we would walk backward through the temporary values and build a DAG of every pure pointwise op (adds, casts, activations, etc.) that leads into the store. That graph base then becomes the reusable anchor: once you split the store into subtiles, you replay the graph for each subtile so all the pointwise ops run after the split. Essentially, “graph base” means a structured representation of those upstream pointwise nodes that the code generator can capture once and rematerialize per subtile, rather than relying on the already-flattened AST expression.