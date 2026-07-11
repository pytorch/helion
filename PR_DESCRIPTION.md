## Summary

This PR improves the existing `pallas_loop_type="compact_worklist"` path for jagged ordered reductions. The loop shape already existed: compact-worklist work items cover compact/output tiles, and each work item runs an inner ordered reduction over another range. This PR makes that existing path cheaper by caching the inner reduction inputs in VMEM and, when possible, caching prepared versions of those inputs.

The two goals are:

- Cache inner reduction data in VMEM: when several compact work items reuse the same ordered range, load the ordered operands into a resident VMEM window once for that range and read them locally instead of streaming them from HBM for every work item.
- Do expensive prep work once per cached range: when the resident data needs a repeated preparation step, compute that prepared view once and store it in scratch. This PR implements direct transpose hoisting when the ordered dimension moves; the same structure can later support prep such as dequantization, layout conversion, scale application, or bias preparation.

The lowering has two pieces:

- VMEM residency: compact-worklist detection marks leading-dimension ordered loads as resident candidates. The Pallas lowering gives each active operand a `pl.Element(C)` resident window keyed by the ordered range start, then rewrites ordered-loop reads to use local offsets inside that window.
- Transpose hoisting: a tiled-FX pass looks for a direct full-slice ordered load from a resident operand with a single `aten.permute.default` user. When admitted, codegen allocates a scratch prep cache, refills it once per `(range_start, range_len)`, and lowers the matched `permute` to a cache read.

For jagged attention-shaped kernels, the repeated work items are Q/output tiles for the same sequence. Those tiles all reduce over the same K/V range, so K/V can be resident in VMEM, and eligible `k[tile, ...].permute(...)` / `v[tile, ...].permute(...)` operations can be computed once for the range instead of once per Q tile.

```python
for seq in hl.grid(B):
    q_start, q_end = q_offsets[seq], q_offsets[seq + 1]
    kv_start, kv_end = kv_offsets[seq], kv_offsets[seq + 1]
    for tile_q in hl.tile(q_start, q_end):
        q_blk = q[tile_q, :, :].transpose(0, 1)
        acc = hl.zeros([H, tile_q, D], dtype=torch.float32)
        for tile_kv in hl.tile(kv_start, kv_end):
            k_blk = k[tile_kv, :, :].transpose(0, 1)
            v_blk = v[tile_kv, :, :].transpose(0, 1)
            scores = torch.bmm(q_blk, k_blk.transpose(-2, -1))
            acc = acc + torch.bmm(scores.to(v.dtype), v_blk)
        out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)
```

| Kernel | Grid | Cached work items | Cache owner | Prep work |
|---|---|---|---|---|
| Jagged attention | total Q/output tiles across sequences | K/V token windows | Q/output tiles in the same sequence/range | transpose that moves the ordered dimension |
| Grouped GEMM / MoE-style future direction | total M/output tiles across experts, usually per N tile; K is the ordered/reduction axis | RHS weights, scales, and bias over K | M tiles with the same expert and N tile | unpack/dequantize, scale application, bias prep |

## What Goes Into VMEM

The current resident-operand detection is automatic, but intentionally narrow. During compact-worklist detection, the compiler scans tensor indexing in the compact grid body:

- a load indexed by the ordered tile on its leading dimension is classified as an `ordered_reduction` candidate;
- tensors used only for offset bounds are excluded because they feed the host-side worklist builder, not the device kernel body;
- ordered-axis stores are rejected;
- non-leading ordered-axis indexing is not made resident in this PR;
- the resident path is only activated if the ordered range is proven to be packed-consecutive offsets, the compact range has a packed active-owner mask, and VMEM can hold at least one ordered block.

So the rule is not “everything in the ordered loop goes into VMEM.” It is “leading-dimension ordered loads that pass the range/window proof become resident operands.” If the proof or budget fails, the kernel keeps the existing streamed `emit_pipeline` lowering.

The prep cache is a separate optional layer on top of those resident operands. It only applies when the tiled FX graph has a direct full-slice ordered load from a resident operand followed by a supported `aten.permute.default`.

## Resident VMEM Window

For each active resident ordered operand, the runtime BlockSpec slices a per-range window:

```python
pl.BlockSpec(
    (
        pl.Element(C, padding=(0, C)),
        *[pl.Element(full_dim) for full_dim in trailing_shape],
    ),
    index_map=lambda wid, *refs: (
        range_start_ref[wid],
        0,
        ...,
    ),
)
```

`C` is a compile-time physical window chosen once in backend setup from resident operand shapes/dtypes, ordered block size, device VMEM budget, and optional prep-cache scratch cost.

The loop body reads the resident window using a local ordered offset:

```python
resident_operand[
    pl.ds(ordered_offset - range_start_ref[_wid], ordered_block),
    :,
    ...,
]
```

## Runtime Guard

For eager/torch launches, the launcher checks that active owners do not exceed the compiled resident window:

```python
max(diff(ordered_offsets)[diff(compact_offsets) > 0]) <= C
```

Sources with zero compact work are ignored because they never produce a worklist item and never read the resident window.

The JAX export path cannot run this host check because offsets are tracers, so it carries the current contract that caller-provided inputs must stay within the compiled window. A future max-extent bound can make this overflow-proof by construction.

## Cache Identities

This PR separates two identities that were easy to conflate:

- `resident_key_fields = ("range_start",)`: identifies the physical resident window slice loaded by the Pallas BlockSpec. Phase 1 still requires `range_start` because local reads compute `ordered_offset - range_start`.
- `prep_key_fields = ("range_start", "range_len")`: identifies the logical prepared contents. It includes `range_len` so tail masking/refill behavior is correct even when two ranges have the same start-like value but different live extents.

The generated compact-worklist metadata remains the single source of truth for these fields.

## Transpose Hoist / Prep Cache

The optional prep-cache path detects one narrow tiled-FX pattern:

```python
load = memory_ops.load(host_tensor, [ordered_tile, :, :, ...])
prep = torch.ops.aten.permute.default(load, perm)
```

The detector only admits this when `host_tensor` is an active resident operand, the load is from the ordered for-loop graph, the ordered tile indexes the leading loaded dimension, every non-leading dimension is a full slice, the load has exactly one user, that user is `aten.permute.default`, the permutation moves the ordered dimension away from the leading loaded dimension, and there is only one prep descriptor for the same `(graph_id, host_arg)`. Detection is semantic only: it records `(host_arg, load_node, prep_node, perm)`. Backend setup admits or drops it after the resident-window budget is known.

Without the prep cache, the resident window can still remove repeated HBM loads, but each compact work item still transposes inside the ordered loop:

```python
def ordered_body(ordered_tile_id, acc):
    local_ordered = pl.ds(
        ordered_offset - range_start[work_id],
        ORDERED_BLOCK,
    )
    ordered_tile = resident_window[local_ordered, :, :]
    prepared_tile = jnp.transpose(ordered_tile, [1, 0, 2])
    return update(acc, prepared_tile)

jax.lax.fori_loop(0, num_ordered_tiles, ordered_body, acc)
```

With the prep cache, codegen registers scratch with the permuted resident-window shape and refills it once when the prep key changes. For the current transpose prep, the key is `(range_start, range_len)`:

```python
previous_work = jnp.maximum(work_id - 1, 0)
key_changed = (
    (range_start[work_id] != range_start[previous_work])
    | (range_len[work_id] != range_len[previous_work])
)

@pl.when((work_id == 0) | key_changed)
def refill_prep_cache():
    def refill_ordered_block(ordered_tile_id, carry):
        local_ordered = block_slice(ordered_tile_id, ORDERED_BLOCK)
        ordered_tile = resident_window[local_ordered, :, :]
        prep_cache[:, local_ordered, :] = jnp.transpose(ordered_tile, [1, 0, 2])
        return carry

    jax.lax.fori_loop(0, num_ordered_tiles, refill_ordered_block, ())
    mask_or_zero_final_partial_block(prep_cache, range_len[work_id])
```

The normal Pallas `aten.permute.default` lowering then becomes a prep-cache read:

```python
def ordered_body(ordered_tile_id, acc):
    local_ordered = pl.ds(
        ordered_offset - range_start[work_id],
        ORDERED_BLOCK,
    )
    prepared_tile = prep_cache[:, local_ordered, :]
    return update(acc, prepared_tile)

jax.lax.fori_loop(0, num_ordered_tiles, ordered_body, acc)
```

Full refill tiles are copied/transposed without a mask. Only the final partial tile is masked along the ordered axis after permutation. If any prep validation fails, the compiler falls back to resident-only lowering; residency is correctness-bearing, while prep hoisting is optional.

## Notes / Limitations

- This is Pallas-only.
- The current resident window is for leading-dimension ordered loads.
- The current prep recognizer intentionally handles only direct `permute` of a full ordered-tile load where the ordered dimension moves.
- Resident windows currently require the Phase 1 key shape `("range_start",)`.
- If the ordered range is not a packed-consecutive offset pattern, the kernel falls back to the existing streamed path.
- JAX-exported kernels rely on the caller respecting the compiled resident window until max-extent sizing is added.
- Grouped GEMM / MoE residency is not implemented in this PR; it is a useful future application of the same resident-key / prep-key split, but it will need richer key fields and resident layouts.

## Test Plan

- `conda run -n helion ruff check ...`
- `conda run -n helion python -m pytest test/test_pallas_compact_worklist.py -q`

The test coverage includes resident-window activation/fallback, runtime window guards, prep descriptor detection/rejection, generated prep-cache reads, 3-D and 4-D tail-mask axis placement, ordered-axis store rejection, and the no-prep resident-only path.
