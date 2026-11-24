# Known Gaps

Defects that are on `main`, that someone found, and whose fix was deferred
because it would be too large a change (`AGENTS.md`, Fix What You Find). An
entry is a defect that can be reproduced from what it says: a wrong result,
a crash or an internal error on a valid program, memory read or written
outside a tensor, or a false fact in an analysis that later decisions read.
A limit is not a gap: a dtype or a form that a backend declines with its own
error, an optimization that does not cover a form, a search that spends a
candidate it could have skipped. The list is no wish list, and no reason to
leave a defect alone: a gap whose fix turns out to be small is fixed, and
its entry goes.

Who edits the list:

- Whoever defers a defect adds its entry and names it in the reply to the
  user; `AGENTS.md`, Fix What You Find, says in which commit. The reviewers
  check an entry with the commit that adds or edits it, and do not raise a
  listed gap against a change that leaves it as it is. A `Deferred:` review
  note cites the entry by its heading (`AGENTS.md`, Answering the findings).
- The commit that fixes a gap removes its entry. A commit that narrows a
  gap, or that learns more about one, updates its entry: an entry is true at
  the commit that holds it.

Entry format: a `## slug` heading (lowercase words joined by hyphens), then
these labelled lines.

- Area: the component, and the backends or configs that reach the defect.
- What is wrong: the exact form, with a reproducer of a few lines where one
  exists (the statements that matter, and the loops they stand in), and what
  was measured where.
- Since: the change that introduced it when that is known; otherwise that it
  was on `main` when the entry was added.
- Why deferred: what a fix would take.
- Pinned by: the tests that hold the current behavior, or none.

## ptxas-signed-extrema-chains

- Area: CuTe backend, elementwise integer extrema. The chain rule of
  `helion/_compiler/cute/handler_ops.py` (`_nan_extremum`,
  `_in_extremum_chain`, `_same_elements_operand`), integer `torch.fmax` /
  `torch.fmin` (`aten_lowering.NumericExtremumLowering`), and chains written
  as compares and selects.

- What is wrong: ptxas (CUDA 13.3, sm_100a) fuses two signed Int32 or Int16
  maxima, or two minima, of three values into one three-operand instruction
  and loses the negation of one operand: `max(max(-a, -b), -c)` returns wrong
  elements, silently. The backend avoids the fusion by comparing unsigned keys
  in the links of a chain that it recognizes: two `torch.maximum` /
  `torch.minimum` / `torch.clamp` extrema of one kind joined directly, through
  copies and views (a `repeat` of one along every dim and a flip of no dim
  among them), through a value identity that an operation and a literal name
  (`+ 0`, `* 1`, a rounding, `-(-x)`, to int64 and back, ...), through a
  variable that an extremum assigned, and the link that a loop carries. The
  forms below compile to the same chain and are not recognized. Their int32
  and int16 results are wrong; int8, unsigned and int64 are right. Keys on
  either of the two links are enough.

  Measured on a B200 against eager. `x` is a [16, 256] tensor of int32 or
  int16 values over the dtype's whole range; counts are wrong elements of
  4,096 for int32 and for int16 (in brackets: int16 in packets of four or
  eight elements) at each of six thread layouts, or wrong rows of 16 at each
  of three (1, 4 and 16 rows per block).

  Forms 1 and 2 put BETWEEN into

      out = torch.empty_like(x)
      for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
          v = x[tile_m, tile_n]
          inner = torch.maximum(-v, -(v * 2))
          out[tile_m, tile_n] = torch.maximum(BETWEEN, -(v + 1))

  and compare with
  `torch.maximum(torch.maximum(-x, -(x * 2)), -(x + 1))`.

  1. A stack of the inner maximum with an unbind of it:
     BETWEEN = `torch.stack([inner, inner], dim=-1).unbind(-1)[0]`.
     2,803 and 2,788 (1,388).
  2. A computation that leaves the value as it is by the values or the range
     of its operands, and not by its operation and a literal:
     `inner + (v - v)`; `inner + (v * 0)`; `torch.where(v == v, inner, v)`;
     `torch.clamp(inner, max=-(lowest + 1))`, an upper clamp at the dtype's
     highest value (`lowest`: a `hl.constexpr` argument, its lowest value);
     `inner.to(torch.float64).to(x.dtype)`.
     2,803 and 2,788 (1,388) each.
  3. Form 2 in a loop's body, with the later maximum handed on and no plain
     copy between the two:

         out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
         for tile_m in hl.tile(x.size(0)):
             last = hl.zeros([tile_m], dtype=x.dtype)
             for column in hl.grid(x.size(1)):
                 value = x[tile_m, column]
                 inner = torch.maximum(-value, -(value * 2))
                 moved = inner + (value - value)
                 last = torch.maximum(moved, -(value + 1))
             out[tile_m] = last

     against the last column of the expression above: 11 and 13 rows. (With
     `kept = moved` and `torch.maximum(kept, ...)` the later link is keyed and
     the rows are right.)
  4. A negation pair around a conversion pair, or the reverse, on the value
     that a loop's body hands back:

         out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
         for tile_m in hl.tile(x.size(0)):
             best = hl.full([tile_m], lowest, dtype=x.dtype)
             for column in hl.grid(x.size(1)):
                 top = torch.maximum(best, -x[tile_m, column])
                 best = torch.neg(torch.neg(top).to(torch.int64).to(x.dtype))
             out[tile_m] = best

     (`lowest`: a `hl.constexpr` argument, the dtype's lowest value), also as
     `torch.neg(torch.neg(top.to(torch.int64))).to(x.dtype)`, against
     `torch.amax(-x, dim=-1)`: 8 and 10 rows. The walk from a link towards
     its users follows one pair, not a pair inside a pair. On the value the
     body takes in, and between two maxima, the same forms are right (found
     from the later link).
  5. A widening in two steps and back, int16 only (for int32 the first step
     is no conversion): BETWEEN =
     `inner.to(torch.int32).to(torch.int64).to(x.dtype)` in the frame of
     forms 1 and 2: 2,788 (1,388). On the value a loop's body hands back
     (`best = torch.maximum(best, -x[tile_m, column]).to(torch.int32)
     .to(torch.int64).to(x.dtype)`): 10 rows. The rule follows a conversion
     to one wider integer and back.
  6. The chain written as compares and selects, with `v` as above:

         a, b, c = -v, -(v * 2), -(v + 1)
         first = torch.where(a > b, a, b)
         out[tile_m, tile_n] = torch.where(first > c, first, c)

     2,603 and 2,599 (1,350), on another input of the same kind. The
     generated code holds no extremum; the toolchain makes one of the compare
     and the select.
  7. Integer `torch.fmax` / `torch.fmin`:
     `torch.fmax(torch.fmax(-v, -(v * 2)), -(v + 1))`.
     2,603 and 2,599 (1,350), on the input of form 6. They keep their signed
     form on purpose (Why deferred).

  Over the probe set of 1,449 cases behind forms 1 to 3 (92 forms of two
  extrema with something between them, each at its layouts for int32, int16
  and int8), 78 are wrong: 12 of form 1, 60 of form 2, 6 of form 3. Forms 4
  and 5, with the same forms on the value a loop's body takes in (54 cases):
  21, of which 12 of form 4 and 9 of form 5. Forms 6 and 7: 24 of their 36
  cases.

  Not measured: a select on a dim of size one, or a narrow over the whole
  extent, on the value that a loop's body takes in or hands back, as in

      best = torch.maximum(best, -x[tile_m, column]).unsqueeze(0).select(0, 0)

  inside `for column in hl.grid(x.size(1))`. Between two maxima in
  straight-line code these two emit no statement, the later link is keyed,
  and the results are right. At a loop's boundary neither part of the rule
  sees them: that they copy shows in their operand's sizes, not in their
  literal arguments, and the statement in which a loop hands its value back
  is emitted after the body. A `repeat` of one and a flip of no dim in that
  place were wrong (8 and 10 rows of 16) before the chain rule followed them.

- Since: on origin/main, where the directly joined chain and the chains
  through views, copies and identities are wrong as well (764 of the 1,449
  cases; main keys the Int32 combine of a reduction loop only). Narrowed by
  "[cutedsl] Fix integer extrema in reductions and register regions" (the
  accumulates of reductions, register programs), "[cutedsl] Add
  cute_guarded_lanes, cute_fmnmx_extrema and N-D packet tile seeds" (directly
  joined elementwise links and the link a loop carries), "[cutedsl] Fold
  vector lanes in the reduction's own dtype" (the lane fold of an integer
  extremum) and "[cutedsl] Follow views and copies in integer extremum chains"
  (links behind views, copies, value identities and variables: 696 wrong
  cases of the 1,449 before it, 78 after).

- Why deferred: the complete workaround is unsigned keys on every Int32 and
  Int16 extremum, and it does not fit the backend as it is.
  - The grouped GEMM worklist schedule, the row-union MMA pipeline and the
    persistent tcgen05 plan read the plain text of their Int32 index extrema
    back. With keys on every elementwise extremum 277 tests of the CuTe CPU
    lane fail (the `test_cute_grouped_*` modules,
    `test_cute_shared_rhs_grouped.py`, `test_cute_flex_score_mod.py`,
    `test_cute_mma_dag_scalar_indices.py`). Following a chain through every
    elementwise node instead, the rule that needs no list of operations,
    fails 275 tests of that lane (267 in 15 grouped and shared-RHS matmul
    modules), keys extrema that are no chain (`inner + 1`) and changes one
    ported better-benchmark kernel.
  - A top-k's sorting networks are built from integer `torch.fmax` /
    `torch.fmin` over encoded keys; keying those slows a default bf16 top-k
    by 20% on a B200.
  - Form 6 has no extremum in the generated code to put keys on.
  - Forms 4 and 5 are identities by their operations alone and were found by
    the last review of "[cutedsl] Follow views and copies in integer extremum
    chains", which ships without them. Measured as an experiment on top of that
    commit: one pass over the graph in order for the walk towards a link's
    users, and a walk back through consecutive widening conversions, make all
    54 cases right, change no pinned test, no other probe table and no ported
    kernel, and pass the CuTe CPU lane. They need pins and a review.
  Closing forms 1 to 3 for good takes keys on every Int32 and Int16 extremum,
  and so the matmul lowerings changed to read keyed index text. A ptxas that
  keeps the negation closes all seven, and the keys can then go.

- Pinned by: `test/test_cute_integer_extrema.py::
  test_numeric_extrema_keep_the_plain_form` holds form 7's signed code (its
  docstring names the wrong results). No test holds the wrong results of
  forms 1 to 6. The rule next to them is pinned in
  `test/test_cute_fmnmx_extrema.py`: what is keyed by
  `test_the_links_of_an_int32_or_int16_chain_compare_unsigned_keys`,
  `test_lower_clamp_bounds_a_relu_and_a_carried_maximum_are_links`,
  `test_links_that_copies_and_views_separate_compare_unsigned_keys`,
  `test_links_that_value_identities_separate_compare_unsigned_keys` and
  `test_an_extremum_that_reads_the_variable_of_another_is_keyed` (the later
  link only); what stays plain, and has to while other lowerings read the
  text, by `test_extrema_outside_a_chain_keep_the_plain_form`,
  `test_a_conversion_that_narrows_the_value_is_no_identity` and the plain
  look-alikes of the identities test. On a GPU,
  `test/test_cute_guarded_lanes_cuda.py::
  test_chains_of_integer_extrema_of_negated_values_equal_eager` compares the
  covered chains with eager; a fix for a form above adds it there.

## index-tensor-out-of-bounds

- Area: compiled kernels, Triton and CuTe backends: a load or a store in a
  device loop through an index tensor.
- What is wrong: compiled code does not bound the elements of an index
  tensor. The Triton backend emits no mask for them
  (`tl.store(out + (indices_0 + 1) * 1, v_2, None)`), and the CuTe backend
  masks `index < size` and has no term for a negative element. An element
  out of range reads or writes memory outside the tensor.

  ```python
  for tile in hl.tile(out.size(0)):
      out[tile.index + 1] = (tile.index + 1).float()
  ```

  With `out` a view of four elements of a larger buffer, the Triton code,
  run through Triton's CPU interpreter, writes 4.0 into the buffer element
  behind the view; `out[tile] = x[tile.index + 1]` reads that element.
  Reference mode does not follow: it stores the elements in range only, and
  a load returns the edge element.
- Since: on `main` when this entry was added.
- Why deferred: a fix decides what an index out of range means in device
  code (eager PyTorch raises) and adds a mask or a check to every indexed
  load and store of every backend, whose generated code and timing change.
- Pinned by: reference mode's side only (no test holds the compiled store),
  in
  `test/test_ref_eager.py::TestRefEagerMisc::test_store_leaves_out_index_elements_out_of_range`.

## memory-op-facts-behind-new-axis

- Area: the device-IR analysis whose `memory_op_facts` the autotuner's
  seeds and heuristics read (`helion/_compiler/device_ir_analysis.py`),
  every backend: a load or a store whose subscript holds `None`.
- What is wrong: a fact takes its per-dim tuples (`subscript_block_ids`,
  `subscript_strides` and the tuples beside them) by position in the
  subscript, and a `None` takes a position and no dim of the tensor, so
  behind it the tuples are a dim off.

  ```python
  for tile_m, tile_n in hl.tile([m, n]):
      out[tile_m, tile_n] = x[tile_m, tile_n] + bias[None, tile_n]
  ```

  The load of `bias` records `subscript_block_ids == (None,)`, where
  `bias[tile_n][None, :]` records the column tile's block. A store
  `out[None, tile_m, tile_n] = ...` into an 8 x 64 `out` records
  `(None, row)` with strides `(64, 1)`: the row tile on the stride-1 dim.
  What reads the tuples misses the access or takes the wrong tile for a
  dim. No wrong result is known; the CuTe store policy does not read the
  store facts of a kernel that has such a store.
- Since: on `main` when this entry was added.
- Why deferred: the correction is local, but it changes what the heuristics
  see for kernels that work today, and with that the seeds, hints and
  policies those kernels get on every backend. It takes a census of the
  configs that change and timings of its own.
- Pinned by: the store's tuples, in
  `test/test_cute_packet_store_policy.py::test_a_store_under_a_new_axis_keeps_the_policy`.

## softmax-two-pass-inf-tiles

- Area: `examples/softmax.py`, `softmax_two_pass`: a property of its
  statements, at a config whose column block is smaller than the row.
- What is wrong: a row whose leading column tiles hold only `-inf` comes
  back NaN where `torch.softmax` returns zeros and a one. In such a tile
  `mi_next` is `-inf`, and both `torch.exp(mi - mi_next)` and
  `torch.exp(values - mi_next[:, None])` take the exponential of
  `-inf - -inf`. A row of 64 elements that is `-inf` up to its last element
  is NaN with a column block of 32 and right with one of 64 (the kernel's
  statements in reference mode).
- Since: on `main` when this entry was added.
- Why deferred: passes of the CuTe backend match this recurrence by its
  statements and are specified to keep its NaN
  (`helion/_compiler/cute/online_to_3pass.py`: "leading all--inf logical
  tiles must still poison the denominator"). A guard in the example changes
  what they match, and has to be timed on the softmax benchmarks.
- Pinned by: the NaN that the rewrite of that pass keeps after a leading
  tile of `-inf`, in
  `test/test_cute_online_to_3pass_guards.py::test_first_logical_tile_not_a_lane_subgroup_controls_poison`
  and
  `test/test_cute_online_to_3pass_guards.py::test_global_max_and_sum_cannot_replace_first_tile_statistic`.

## vector-lane-reduction-hoist

- Area: CuTe backend, `hoist_warp_reduce` and the lane loops it leaves
  alone, at configs with a `cute_vector_widths` entry above 1.
- What is wrong: the pass that runs a vector lane loop's cross-thread reduce
  once per tile does not check the loop's other statements, and a loop it
  leaves alone keeps a per-lane reduce whose result is read as the tile's.
  Four kernels at `block_sizes [4, 128]`, `num_threads [0, 32]`,
  `cute_vector_widths [1, 4]`. Each is `for tile_m in hl.tile(m):` around
  `for tile_n in hl.tile(n):` over an fp32 `x` of 16 x 126 (`y` has its
  shape); below is what follows `v = x[tile_m, tile_n]` in the column loop.
  `top`, `total` and `count` are per-row values (`hl.full` or `hl.zeros` of
  `[tile_m]`) that start ahead of the column loop and are stored after it.

  ```python
  # the generated code has no store to y
  top = torch.maximum(top, torch.amax(v, dim=-1))
  y[tile_m, tile_n] = v * 2
  total = total + v.sum(-1)
  # the write is placed ahead of the tile's loads
  x[tile_m, 0] = total
  total = total + v.sum(-1)
  # total, read after the loop, is one lane's sum
  x[tile_m, tile_n] = v + 1
  total = v.sum(-1)
  # counts the lanes with a positive element, not the tiles
  count = count + torch.any(v > 0, dim=-1)
  ```
- Since: on `main` when this entry was added.
- Why deferred: the passes decide from the text of generated code, by
  patterns that each know some spellings, and a fix of these forms leaves
  the next spelling open. It needs a redesign: one parser for every
  cross-thread reduce, hoisting only bodies whose every statement it
  recognizes, and a model of the lane loop to test it against.
- Pinned by: none.

## fused-product-drops-carried-binding

- Area: CuTe backend, `fuse_fma`, default configs included.
- What is wrong: a product that one addition reads is fused into that
  addition and its statement deleted, also when its name is read again
  after the loop.

  ```python
  other = hl.zeros([tile_m], dtype=torch.float32)
  last = hl.full([tile_m], 100.0, dtype=torch.float32)
  for tile_n in hl.tile(n):
      v = x[tile_m, tile_n]
      last = torch.amin(v, dim=-1) * 0.01
      other = last + 0.5
  out[tile_m] = other * 1000.0 + last
  ```

  At the default config the tile loop holds
  `other = cute.math.fma(amin, v_0, v_2)` and no statement that binds
  `last`, which keeps its initial 100. With the pass replaced by the
  identity the statement is there.
- Since: on `main` when this entry was added.
- Why deferred: the pass counts a product's reads by name, ahead of the pass
  that renames loop-carried values. A fix changes which products contract,
  and with that the floating-point results of kernels that compile today,
  so it needs a census of generated code and a GPU run of its own.
- Pinned by: none.

## carried-update-repeats-per-lane

- Area: CuTe backend, the placement of statements in lane loops, default
  configs included.
- What is wrong: a tile loop's carried update is placed inside the lane loop
  when the loop body also holds a nested tile loop with a reduction, so it
  runs once per lane. Inside `for tile_m in hl.tile(m):`, with `x` and `out`
  fp32 of m x n:

  ```python
  run = hl.zeros([tile_m], dtype=torch.float32)
  for tile_a in hl.tile(n):
      va = x[tile_m, tile_a]
      acc = hl.zeros([tile_m, tile_a], dtype=torch.float32)
      for tile_b in hl.tile(n):
          vb = x[tile_m, tile_b]
          acc = acc + torch.tanh(va[:, :, None] * vb[:, None, :]).sum(-1)
      run = run + 1.0
      out[tile_m, tile_a] = acc * run[:, None]
  ```

  At the default config on a 16 x 16 input, the generated code, executed on
  the CPU, stores 240 of 256 elements that differ from reference mode. The
  same loop without the nested one is right.
- Since: on `main` when this entry was added.
- Why deferred: refusing the placement would refuse the default config of
  such a kernel, so the fix is a lane split that keeps the carried update
  outside the lane loop: a change to how lane loops are built, with the
  CuTe GPU lanes behind it.
- Pinned by: none.

## rolled-graph-subscripts-unresolved

- Area: `DeviceIRAnalysis.memory_op_facts`
  (`helion/_compiler/device_ir_analysis.py`), the per-subscript fields
  (`subscript_affine_block_ids`, `subscript_block_ids`,
  `subscript_index_scales`) of the ops in the rolled alternative graph of a
  full slice's loop. The readers of those fields reach it: the coalescing
  axes of `device_ir.py` (the loop over `memory_op_facts` that picks
  `coalescing_sensitive` block ids), the CuTe heuristics and store policies
  (`helion/_compiler/cute/`).
- What is wrong: in the rolled graph a subscript computed outside the loop
  is a placeholder, and the fields read the placeholder instead of what the
  enclosing graph passes in. For

  ```
  for tile in hl.tile(out.size(0)):
      out[tile, :] = x[2 * tile.index, :]   # x: [8192, 96] bf16
  ```

  the load lists `subscript_affine_block_ids == (0, None)` in graph 0 and
  `(None, None)` in its rolled copy (graph 1), so the copy looks like a
  full-extent op of the whole tensor (786432 elements against a 96-element
  row). `pointwise_fact` takes `max_op_slab_numel` from the ops outside
  rolled graphs, so the seed's starvation cap does not read the copy.
  Reproduced, bind only, on upstream main at 111948cb.
- Since: on `main` when the entry was added.
- Why deferred: resolving the placeholder (as `_placeholder_resolver` does
  for `gather_numel` and `row_gather`) changes these fields for every kernel
  with a full slice whose subscripts are computed outside the slice loop,
  and with them the coalescing axes and the CuTe seeds and store policies;
  judging it takes a census and timings of its own on every backend.
- Pinned by:
  `test/test_autotuner_heuristics.py::TestPointwiseGatheredSlab::test_rolled_copy_of_an_op_reads_its_index_in_the_enclosing_graph`.

## pointwise-seed-grid-axes-uncounted

- Area: the Triton pointwise seed on sm100
  (`TritonPointwiseSeedHeuristic._warps_for_sm100`,
  `helion/_compiler/autotuner_heuristics/triton.py`), its residency term,
  for every kernel with an `hl.grid` axis or a masked last tile, row
  lookups (`PointwiseElementwiseFact.row_gather`) included.
- What is wrong: the residency term counts programs as
  `total_numel // tile_numel`. `total_numel` multiplies the tiled block dims
  only, so an `hl.grid` axis is left out and a nested `hl.tile` under it, a
  loop inside the program, is counted as programs; the floor also drops a
  masked last tile. `max_op_slab_numel` charges a grid axis as untiled
  fan-out, which lifts the starvation cap. Bind only, B200 model:

  ```
  for i in hl.grid(ids.size(0)):            # 4096 ids
      for tile_d in hl.tile(table.size(1)):  # table: [32768, 1024] bf16
          out[i, tile_d] = table[ids[i], tile_d]
  ```

  launches 4096 programs; the term counts 1024 // 4 = 256 and the seed is
  `([4], 16)` (upstream main at 111948cb: `([1], None)`, its tile smaller
  because main charges the table's rows). `table[ids[tile_t], tile_d]` with
  6144 ids of a `[12288, 513]` table launches 1536 programs and counts 769:
  `([4, 1024], 8)` (main: `([2, 1024], None)`). The same count on main
  gives `([4], 16)` for `x[i, tile_d] * 2` under `hl.grid` over a
  `[4096, 256]` x.
- Since: on `main` when the entry was added. The row lookups reach it since
  the sm100 warps rule caps only their lanes term.
- Why deferred: reading the launched grid
  (`TritonH100MatmulHeuristic._launch_grid`, which itself reads a symbolic
  grid extent at the generic size hint) changes the warps of every sm100
  pointwise kernel with an `hl.grid` axis or a masked tile, the cells the
  sm100 seed was tuned on among them; judging it takes timings of its own.
- Pinned by:
  `test/test_autotuner_heuristics.py::TestPointwiseGatheredSlab::test_rows_behind_a_loaded_index_get_no_lane_warps_on_b200`
  (`"grid index on the last dim"`, 768 programs counted for 4 launched).

## pointwise-seed-tunable-region-lanes

- Area: the Triton pointwise seed on sm100
  (`TritonPointwiseSeedHeuristic._warps_for_sm100`) for a gather whose
  region is sized by a registered tunable.
- What is wrong: `gather_numel` is `None` for such a region, so
  `pointwise_fact` charges the op its untiled extents, the whole table, as
  `max_op_slab_numel`, and the starvation cap does not bound the warps
  (the grid is 32 programs, so a row lookup gets the residency warps). For

  ```
  width = hl.register_tunable("width", PowerOfTwoFragment(1, 64, 8))
  for tile_t in hl.tile(ids.size(0)):        # 64 ids
      cols = hl.arange(width)
      out[tile_t.index[:, None], cols[None, :]] = table[
          ids[tile_t][:, None], cols[None, :]  # table: [8192, 64] bf16
      ]
  ```

  `max_op_slab_numel` is 524288 and the seed is `([2], 16)`: 16 warps for a
  program of 16 elements at the default width (128 at the widest). Upstream
  main at 111948cb gives the same seed. Bind only.
- Since: on `main` when the entry was added.
- Why deferred: counting the region at the tunable's default
  (`CompileEnvironment.tunable_default_hint` in `_numel_at_unit_tiles`) is
  not enough on its own: it also shrinks the slab, and the register-capped
  `BLOCK_FLOOR` then puts all 64 tokens in one `[64]` tile, one program for
  148 SMs (bind only). The fix waits on that floor for the row axis of
  whole-row tiles, a change of the pointwise seed's tiles on every target
  that needs timings of its own.
- Pinned by:
  `test/test_autotuner_heuristics.py::TestPointwiseGatheredSlab::test_rows_behind_a_loaded_index_get_no_lane_warps_on_b200`
  (`"rows of a tunable width"`).

## pointwise-slab-counts-rolled-copies

- Area: `DeviceIRAnalysis.pointwise_fact`
  (`helion/_compiler/device_ir_analysis.py`), `slab_numel` and `sfu_ops`,
  read by the Triton pointwise seed's byte and register budgets and its
  warps on every target.
- What is wrong: `memory_op_facts` lists the ops of a full slice's rolled
  alternative graph next to the originals (one indexing slot each, which is
  right for the config), and `slab_numel` sums both, though a config runs
  one of the two. For `out[tile, :] = x[2 * tile.index, :]` (x
  `[8192, 96]` bf16) the load and store add 288 and their copies 288 more:
  `slab_numel` is 576, also on upstream main at 111948cb. Every pointwise
  kernel with a full slice is charged about twice its slab (rope among
  them). `sfu_ops` walks the rolled graphs too: `torch.cos(torch.sin(
  table[ids[tile_t], :]))` counts 4 where a config runs 2, which crosses
  `SFU_W8` (3); 4096 ids of a `[4096, 512]` table get `([8], 8)` on B200,
  the same on upstream main.
- Since: on `main` when the entry was added.
- Why deferred: dropping the copies from the sum halves the slab of every
  pointwise kernel with a full slice, and with it their tiles on sm90,
  sm100 and sm120 (`[32]` to `[64]` for the kernel above on B200), and
  dropping them from `sfu_ops` moves the warps of every such kernel with a
  transcendental; judging it takes timings of its own.
- Pinned by:
  `test/test_autotuner_heuristics.py::TestPointwiseGatheredSlab::test_rolled_copy_of_an_op_reads_its_index_in_the_enclosing_graph`
  (`slab_numel == 576`).
