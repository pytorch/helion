# Pallas `fori_loop` tensor double buffering

## 1. Summary

The Pallas `fori_loop` lowering already routes suitable loop-local tensors
through an explicit HBM-to-VMEM DMA:

```python
copy = pltpu.make_async_copy(hbm_tile(i), scratch, semaphore)
copy.start()
copy.wait()
value = compute(scratch)
```

This design adds a per-input-tensor load buffer count that changes an existing
load route from depth one to depth two. It does not create a second eligibility
system. The existing tensor DMA classifier remains authoritative:

1. Classify tensors with the existing Pallas streaming logic.
2. For each admitted input load, read that input tensor's requested count.
3. Keep depth one when the count is `1`.
4. Use two VMEM stages and two semaphore slots when the count is `2`.
5. Fall back to count `1` when the tensor has no existing DMA load route.

The depth-two schedule is the usual prime, prefetch, wait, compute sequence:

```python
# Immediately before this fori_loop.
if num_iterations > 0:
    copy(hbm_tile(0), scratch.at[0], semaphore.at[0]).start()

for i in range(num_iterations):
    if i + 1 < num_iterations:
        copy(
            hbm_tile(i + 1),
            scratch.at[(i + 1) % 2],
            semaphore.at[(i + 1) % 2],
        ).start()

    copy(
        hbm_tile(i),
        scratch.at[i % 2],
        semaphore.at[i % 2],
    ).wait()
    value = compute(scratch.at[i % 2])
```

The implementation stays local to the existing lowering. It adds no per-load
identity, route planner, scheduling IR, AST rewrite pass, or `ForiDmaLoad`
abstraction.

## 2. Goals and invariants

The design has four primary invariants:

- **Existing eligibility is the only eligibility.** Double buffering never
  turns a direct, gather, resident, or otherwise non-DMA access into a DMA.
- **The option selects depth, not placement.** A count of `1` means depth one; a
  count of `2` means depth two only for an already admitted input-load route.
- **Unsupported count requests are no-ops.** They do not invalidate a config and
  do not disable buffering for other tensors.
- **Scheduling is constructed in final order.** The lowering appends prime,
  prefetch, wait, compute, and store statements to their final lexical scopes.
  It does not search or move previously emitted AST.

An all-one configuration must produce the same generated code as omitting the
field.

## 3. Configuration surface

The Pallas-only config field is:

```python
pallas_load_buffer_count: list[int]
```

There is one entry for each tensor leaf in the kernel's bound formal inputs, in
the deterministic pytree-flattened order already produced by `LiftTensorArgs`.
Non-tensor values and `Tile` objects do not consume entries. For example:

```python
def kernel(q, k, v, scale: float): ...

Config(
    pallas_loop_type="fori_loop",
    pallas_load_buffer_count=[1, 2, 2],
)
```

The list length is finalized from the same flattened formal arguments used to
bind tensor inputs. Values are currently restricted to `1` or `2`, represented
by an `IntegerFragment(1, 2, 1)` for each input. The default is an all-one list.
The field participates in autotuning only when the kernel has Pallas inner
loops; two configs may intentionally lower identically when a count of `2`
names an input without an eligible route.

For a loop type other than `fori_loop`, or for a kernel without Pallas inner
loops, the field is inactive and has no effect. Inactive values are discarded
before list shape and element validation because no lowering can consume them.

The count is tensor-wide. If one input tensor has an admitted DMA route in two
separate loops, its one count applies to both loop-local routes, each with
its own resources. Multiple loads of that tensor in one loop share the existing
tensor route and therefore share one staged scratch buffer. There are no
per-load combinations. This does not make unsupported combinations of
same-tensor routes valid; existing route ownership, addressing, and host-padding
constraints remain authoritative.

## 4. Existing DMA classification is authoritative

`_codegen_fori_loop` first runs the existing loop-tensor and streamed-tensor
classification. In particular, the existing machinery remains responsible for:

- deriving the per-iteration VMEM shape;
- checking TPU DMA alignment and supported contiguous row-slab cases;
- distinguishing read-only loads from load/store tensors;
- rejecting tensors whose placement conflicts with root-scope accesses;
- rejecting atomic storage from the manual load-DMA route;
- retaining compact-worklist and resident-cache placement decisions; and
- constructing the HBM slice, outer-grid offsets, scalar-selected dimensions,
  padding, masks, and jagged begin/end expressions.

Only after those decisions are complete does the requested count take effect. The
lowering iterates the already admitted tensor routes and chooses a resource
depth. There is no double-buffer-specific shape, indexing, nesting, or
load-pattern classifier.

This makes fallback precise and local. If `x` is admitted and `y` is not, a
configuration of `[2, 2]` double-buffers `x` and leaves `y` on its existing
path. It does not reject the config or force both tensors onto a common path.

## 5. Mapping a route to a formal input

Codegen may see a FakeTensor or a view rather than the exact object stored in
the bound arguments. Count lookup therefore uses existing tensor identity
without introducing source-load metadata:

1. Flatten the bound formal tensor inputs with `LiftTensorArgs`.
2. Record input slots by exact tensor identity.
3. Also record input slots by backing-storage identity.
4. For an admitted route, prefer an exact identity match.
5. If there is no exact match, use the backing-storage match only when it
   identifies exactly one formal tensor slot.
6. If no unique slot exists, keep the route at depth one.

The storage lookup lets ordinary views, `None`/newaxis indexing, and view-shaped
FakeTensors inherit the count of their source input. The uniqueness rule avoids
assigning an arbitrary count when several formal inputs alias the
same compile-time FakeTensor storage. Compiler-created tensors and output-only
tensors have no formal input slot and therefore remain unchanged.

This is not runtime alias analysis. Distinct formal arguments that happen to
alias only in a particular call remain distinct config slots under the existing
compiler aliasing contract. The storage fallback is for compiler-visible views
derived from one formal input, such as attention K/V reshapes.

No FX load index is assigned. Graph copying, rolling, and config-dependent load
counts cannot change the meaning of the list because its identity is the bound
tensor input, not a load occurrence.

## 6. Resources and loop-local state

For every tensor admitted by the existing DMA path, the lowering already
allocates a VMEM scratch buffer and DMA semaphore. The effective count only changes
their leading stage shape.

For an existing depth-one route with logical VMEM shape `S`:

```python
scratch.shape == S
semaphore.shape == ()
```

For a selected depth-two input load:

```python
scratch.shape == (2, *S)
semaphore.shape == (2,)
```

The runtime scratch-argument machinery and VMEM accounting continue to see
ordinary scratch shapes. `register_dma_semaphore` accepts the optional shape and
still delegates to the normal scratch registration path.

`ForiLoopState` retains the existing tensor-to-scratch and
tensor-to-semaphore maps. It additionally carries only a set of tensor names
whose load route is depth two:

```python
_double_buffered_tensors: set[str]
```

When ordinary load codegen resolves a name in that set, it selects:

```python
scratch.at[innermost_loop_index % 2]
```

and then applies the same logical indexing operations it already used. The
stage dimension is purely physical and never appears in the Helion program's
logical tensor rank.

Semaphore names, copy records, trip-count expressions, and next-iteration
indices remain local variables in `_codegen_fori_loop`. They do not require a
new persistent compiler object.

## 7. Lowering algorithm

For each `ForLoopGraphInfo` lowered through `jax.lax.fori_loop`, codegen performs
these steps in order:

1. Compute the existing `grid_parts`, block sizes, begins, iteration steps, and
   slice sizes.
2. Classify loop loads and stores and set up loop-carried scratch exactly as
   before.
3. Run `_classify_pipelined_tensors` and the existing compact/resident filters.
4. Build the mapping from bound formal inputs to exact tensor and backing
   storage identities.
5. For each admitted tensor route, compute:

   ```python
   load_buffer_count = (
       config.pallas_load_buffer_count[input_slot]
       if direction == "load" and route_has_one_formal_input_slot
       else 1
   )
   ```

6. Mark the tensor as HBM-backed through the same Pallas memory-space map.
7. Allocate depth-one or depth-two scratch and semaphore shapes.
8. Record selected loads for prime, prefetch, and current-stage waits.
9. Emit the loop body in the lexical order specified below.
10. Wrap that body with the existing inside-out `jax.lax.fori_loop` construction.

The test for `direction == "load"` is not a second DMA classifier. It states the
scope of this feature: asynchronous input loads. Existing store routes remain
depth one.

## 8. Exact statement ordering

Let `N` be the existing final grid extent of the staged, innermost emitted loop.
When at least one admitted load is depth two, codegen first binds:

```python
num_iterations = grid_parts[-1]
```

It reuses this exact value both as the `fori_loop` upper bound and as the guard
for DMA scheduling.

### 8.1 Prime

Immediately before entering the staged `fori_loop`, codegen emits one guarded
prime function:

```python
@pl.when(num_iterations > 0)
def prime():
    for tensor in depth_two_loads:
        make_copy(
            hbm_slice(tensor, inner_iteration=0),
            tensor.scratch.at[0],
            tensor.semaphore.at[0],
        ).start()
```

The prime starts every selected load and deliberately does not wait.

### 8.2 Body iteration `j`

The generated body has this order:

1. Compute the existing per-iteration offsets, indices, and masks.
2. If `j + 1 < num_iterations`, start every selected load for iteration
   `j + 1` into stage `(j + 1) % 2`.
3. For every admitted depth-one load, start and immediately wait for its current
   tile using its scalar semaphore.
4. Wait for every selected load's current tile in stage `j % 2`.
5. Execute the original graph body. Selected tensor reads resolve to
   `scratch.at[j % 2]`; all other reads use their existing route.
6. Write back loop-carried values, when present.
7. Start and wait for existing store DMAs.

Starting the next transfer before current waits maximizes the overlap window.
Waiting before graph execution prevents compute from reading an incomplete
stage. Synchronous stores remain after compute, preserving existing hazards and
write ordering.

For the final iteration, the `j + 1` guard suppresses prefetch. After its current
stage waits and compute finish, no transfer remains in flight.

## 9. Multidimensional and nested loops

A multidimensional `ForLoopGraphInfo` is emitted as nested `fori_loop` calls.
Double buffering stages the innermost emitted dimension only:

- `loop_vars[-1]` selects `stage = j % 2`;
- `grid_parts[-1]` is the staged trip count;
- outer loop indices remain fixed while the inner prefetch train runs; and
- the prime is inserted immediately before the inner `fori_loop`, inside the
  enclosing loop body.

Consequently, every outer iteration primes inner tile zero with that outer
iteration's HBM coordinates. A stage from one outer iteration is never consumed
as the first stage of the next outer iteration.

Each separately lowered source loop owns its own local resources and scheduling
statements. A scalar `hl.grid` prefix such as `x[b, :, :]` composes naturally:
the existing slice builder applies `b` as an outer offset, while the inner tile
dimension supplies the staged iteration.

Likewise, a blockwise access such as `x[tile_b.index, :, :]` buffers whenever
the ordinary tensor classifier already gives it a DMA route. It needs no
double-buffer-specific indexing case.

Depth selection does not redefine route ownership for nested loops. A nested
same-storage pattern must already be represented correctly by the ordinary DMA
lowering; this option only changes that route's buffer depth.

The design does not make loop-carried state an eligibility criterion. Carried
values continue to use their existing scratch and are read or written during
the compute portion after current input waits.

## 10. Dynamic ranges and `jagged_tile`

Dynamic and jagged loops do not have a separate buffering algorithm. The
schedule reuses all expressions produced by the ordinary `fori_loop` lowering:

- the runtime begin and end;
- the per-program iteration step and slice size;
- `grid_parts[-1]`, which is the actual per-program
  `ceil_div(end - begin, block_size)` trip count; and
- the existing masks, padding, and HBM Ref transforms.

The double-buffer lowering never computes a second maximum or an independent
jagged trip count. `hl.jagged_tile` may already use a maximum while constructing
its ordinary parent extent; staging simply reuses the resulting
`grid_parts[-1]`. For an empty segment, `N == 0`, the prime guard is false, and
the loop body does not execute. For a partial final tile, the existing DMA
padding and load mask remain responsible for safety and semantic zeroing. A
direct dense load indexed by `hl.jagged_tile` is eligible only when the same
existing classifier would already give it a manual DMA route.

For a multidimensional jagged loop, the prime is still placed at the staged
inner-loop boundary, so its begin, end, and outer indices are the values for the
current program and current enclosing iteration.

Direct rank-two `jagged_tile` coverage also exposed a shared Pallas mask-broadcast
gap: ordinary load masks and deferred nonlinear `_mask_to` masks were expanded
as one-dimensional tile masks instead of using the existing jagged mask shape.
The implementation uses `jagged_tile_expand_str` in those two shared mask paths,
matching the existing Triton behavior. This is a prerequisite Pallas parity fix,
not part of the prime/prefetch schedule, and its tests exercise both ordinary
and depth-two DMA routes.

## 11. Mutation, atomics, and stores

This feature double-buffers read-only input load routes only.

- If an input storage is both loaded and stored in the loop, a requested count of `2`
  is ignored and the existing depth-one load/store scratch is retained. The
  storage check also covers distinct host views of the same backing tensor.
- If a storage participates in an atomic operation, the existing DMA classifier
  keeps it off the manual input-DMA route. Its requested count is therefore a no-op.
- Output stores retain their existing start/wait sequence after compute.

This avoids stale reads, read-modify-write alias hazards, and premature stage
reuse. Asynchronous or double-buffered stores require a separate design because
their completion and reuse constraints differ from read-only prefetch.

The atomic rule also corrects the ordinary depth-one classifier: Pallas lowers
atomics through the raw tensor ref, so that storage cannot safely be remapped to
HBM by another inner loop even when double buffering is disabled. Regression
coverage executes both the feature-off and selected configurations.

## 12. Fallback semantics

A count of `2` is silently reduced to `1` when its tensor has no admitted input-load DMA
route. Examples include:

- an unaligned access rejected by existing streaming rules;
- indirect or gather indexing rejected by the existing DMA classifier;
- a tensor also accessed at a scope incompatible with HBM remapping;
- mutable or atomic storage;
- a compact/resident route selected by existing machinery;
- an output or compiler-created tensor with no formal input slot; and
- a view whose backing storage maps ambiguously to several formal inputs.

Fallback is per tensor and per loop route. It neither raises an invalid-config
error nor changes another tensor's eligibility.

## 13. Implementation shape

The implementation consists of small extensions to existing machinery:

- config plumbing for `pallas_load_buffer_count`;
- config length finalization from flattened formal tensor inputs;
- shaped DMA-semaphore scratch registration;
- a `_double_buffered_tensors` marker on `ForiLoopState`;
- a stage selection in existing scratch-name resolution; and
- local prime, prefetch, and wait statement lists in `_codegen_fori_loop`.

The supporting `None`/newaxis subscript normalization is shared with
`emit_pipeline` because both lowerings interpret the same tensor-dimension
metadata. It corrects dimension mapping without changing emit-pipeline
scheduling. Direct jagged mask expansion is likewise shared and independent of
the selected buffer depth.

It intentionally does not add:

- per-load IDs or FX metadata;
- a per-load tuning surface;
- a new eligibility or route hierarchy;
- `ForiDmaLoad` or another one-use load model;
- a generic pipeline event or iteration-point abstraction;
- an AST search, mutation, or statement-reordering pass; or
- changes to `pltpu.emit_pipeline` scheduling.

The tensor marker is sufficient because the existing manual DMA route is
already tensor-keyed. This also gives future work a straightforward foundation:
other depths or store scheduling can extend resource and ordering decisions
without replacing tensor placement or indexing analysis.

## 14. Worked tiled-matmul schedule

Consider a reduction loop with two admitted tensor loads:

```python
for tile_k in hl.tile(K):
    acc += x[tile_m, tile_k] @ y[tile_k, tile_n]
```

With both formal input counts set to `2`, the existing route shapes `SX` and
`SY` become:

```python
x_scratch: (2, *SX)
x_semaphore: (2,)
y_scratch: (2, *SY)
y_semaphore: (2,)
```

For `N = ceil_div(K, BLOCK_K)`, the generated behavior is:

```python
if N > 0:
    x_copy(0, stage=0).start()
    y_copy(0, stage=0).start()

for k in range(N):
    if k + 1 < N:
        x_copy(k + 1, stage=(k + 1) % 2).start()
        y_copy(k + 1, stage=(k + 1) % 2).start()

    x_copy(k, stage=k % 2).wait()
    y_copy(k, stage=k % 2).wait()
    acc += x_scratch.at[k % 2] @ y_scratch.at[k % 2]
```

For `N == 1`, only the prime and current waits execute. For `N == 2`, iteration
zero starts stage one before waiting on stage zero; iteration one does not
prefetch, waits on stage one, and completes compute. A non-divisible final K
tile uses the same padding and K-tail mask as the existing depth-one route.

If only `x` has count `2`, `x` follows this schedule while `y` retains its
ordinary start/wait sequence in each body iteration. If another formal input,
such as an attention query loaded outside the reduction loop, has count `2`
but no loop-local DMA route, that entry has no effect while eligible key/value
inputs still buffer.

## 15. Validation plan

Tests should cover configuration, generated structure, correctness, and
performance.

### 15.1 Configuration and identity

- validate exactly one integer count in `{1, 2}` per flattened formal tensor input;
- discard inactive values before validation and omit the field when there are no
  Pallas inner loops;
- verify defaults and serialization;
- verify that non-tensor arguments do not consume slots;
- verify that a view and newaxis load inherit their unique source input's count;
- execute the shared newaxis normalization under `emit_pipeline`; and
- verify that one tensor used in multiple loops applies its count to each
  admitted route.

### 15.2 Code generation and ordering

- assert that all-one code is byte-for-byte identical to baseline;
- check depth-two VMEM and semaphore shapes;
- check stage-zero prime before the loop;
- check next-stage start before current-stage waits;
- check every current wait before the first staged compute read;
- check staged reads use `j % 2`;
- check stores remain after compute and synchronous; and
- parse multidimensional output to confirm the prime is inside the enclosing
  loop and immediately before the staged inner loop.

### 15.3 Eligibility and fallback

- eligible and ineligible selected tensors in the same kernel;
- repeated loads of one tensor sharing one scratch route;
- unaligned loads;
- an admitted blockwise `tile.index` load;
- root-scope sibling accesses;
- mutable input storage;
- atomic storage; and
- an ineligible attention query with eligible key/value loads.

Each fallback test should compare generated code with the corresponding count of
`1` or an omitted field, not merely check that compilation succeeds.

### 15.4 Numerical coverage

- tiled matmul and batched matmul, including a partial reduction tail;
- attention over several sequence and head shapes;
- scalar grid prefixes such as `x[b, :, :]`;
- multidimensional device loops;
- zero-, one-, and two-iteration loops;
- runtime `hl.tile(start, end)` with empty and partial segments;
- direct `hl.jagged_tile` loads and downstream jagged masking; and
- the existing loop-carried-state kernels.

### 15.5 Performance

Benchmark representative matmul, attention, dense reduction, jagged softmax,
and another jagged kernel. Compare:

- `fori_loop` depth one;
- `fori_loop` depth two; and
- `emit_pipeline` for non-jagged kernels.

For every benchmark, first verify numerical correctness and inspect generated
code for the intended resource shapes and ordering. Treat a requested count of
`2` that has no route as a correctness/fallback case, not evidence of buffering.
