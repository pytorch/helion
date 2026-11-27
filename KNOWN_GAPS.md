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

## indirect-memory-effects-in-inline-assembly

- Area: compiler memory dependencies for `hl.inline_asm_elementwise`.
- What is wrong: an impure assembly snippet can store through addresses
  loaded from a tensor. The dependency checker treats the loaded addresses
  as ordinary computed values; it does not record the write. For example,
  use the store from `test_dead_code_side_effects.py` with operands
  `[addresses[t], x[t]]` in a 256-element tile, then read
  `scratch[(t.index + 32) % 256]` into an output. If the addresses point
  to the corresponding elements of `scratch`, the reader may run before
  another warp's write. With Triton CPU code generation the assembly was
  present, but there was no barrier. Replacing the assembly with
  `scratch[t] = x[t]` emitted a barrier.
- Since: reproduced on `main` at `a2f18c2d0`.
- Why deferred: the assembly interface declares purity but has no way to
  identify the memory read or written by arbitrary register operands. An
  unknown-memory effect must reach dependency checks, reduction dimension
  merging and schedule selection. Treating every impure snippet as reading
  and writing all memory can introduce barriers or rule out schedules for
  working kernels even when the snippet only inspects a register. This
  needs its own compatibility and performance evaluation across backends.
- Pinned by: none. The tests in `test_dead_code_side_effects.py` check that
  the assembly survives dead-code elimination, not its memory dependencies.

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
- What is wrong: compiled code does not consistently bound tensor indices.
  Ordinary Triton pointer accesses can omit the mask
  (`tl.store(out + (indices_0 + 1) * 1, v_2, None)`). Triton tensor
  descriptor loads check that the tile overlaps the tensor before narrowing
  its origin; lanes outside the tensor return zero, including on the masked
  pointer fallback for an overlapping tile with an invalid native origin.
  CuTe can check only
  `index < size` for a vector index or a scalar loaded or computed in device
  code, so a negative element can read or write outside the tensor.
  CuTe checks both bounds at the original integer width for a rank-zero
  tensor argument used directly as the index: an out-of-range load returns
  zero, and an out-of-range store or atomic does not write. This does not
  extend to an index loaded explicitly with `idx[()]`.

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

## triton-native-nested-range-width

- Area: Ordinary nested `hl.grid` and `hl.tile` on Triton, Gluon and TileIR.
  Cartesian roots, descriptors and CuTe fragment loops use separate lowering.
- What is wrong: Ordinary native loops can narrow bounds to the storage
  index type. With Int32 indexing, two Python arguments `lo=2**32` and
  `hi=4` make an empty `hl.grid(lo, hi)` execute four times. A vector
  `hl.tile(-64, hi, block_size=4)` with Python `hi=129` and Int8
  indexing can skip all 49 tiles. Loaded wide integers
  can narrow too: for an interval starting at `2**32 + 1`, an Int32
  counter can enable all four lanes of a partial tile. Int64 indexing
  gives zero and 49 iterations for these two Python examples. Widen an
  expression's operands before doing arithmetic that might overflow.
  Even with Int64 indexing, the increment after the last live tile must
  fit. With loaded bounds `[2**63 - 3, 2**63 - 1)` and width 4, that
  increment wraps below the stop and the loop can continue. A dynamic
  descending stride also cannot rely on the native normalization of a
  literal negative step. Iterate over a bounded zero-based span and
  add the base to each live coordinate instead.
- Since: The narrow-header cases reproduce on unmodified main at
  `0092ca313`; the dynamic-stride and loaded-bound cases also reproduce
  on the stack base `4b86f71b0`.
- Why deferred: These native counters are also used for grouped GEMM and
  jagged attention. A general replacement has to preserve their native
  fast paths and saved static-range configs while handling tail masks,
  signed and unsigned arguments and dynamic stride direction. Widening
  all potentially large counters changes compiled code for ordinary
  inputs. This root-grid change leaves the native lowering in place;
  selecting and measuring a replacement is a separate compiler change.
- Pinned by: The native nested-loop compatibility tests exercise working
  boundary and saved-config behavior, not the incorrect iteration counts.

## observed-tile-count-arithmetic

- Area: An observed `tile.count` inside an ordinary nested device
  loop. Root tiles and eager evaluation have separate count paths.
- What is wrong: On Triton, a nested tile over loaded Int8 bounds
  `[-64, 65)`, width 4 and `index_dtype=torch.int8` executes 33 tiles
  but can report -31. With loaded signed Int64 bounds
  `[-2**62, 2**62 - 1)`, width 4 and `index_dtype=torch.int64`,
  adding 3 for ceiling division can overflow even though the result
  fits. At width 1, `[-2**62, 2**62)` needs a count above INT64_MAX.
  Pallas likewise cannot represent a count above INT32_MAX in a
  signed scalar with JAX 64-bit arithmetic disabled.
- Since: These nested calculations reproduce on main and the
  originally submitted Cartesian-grid patch. The Triton Int8 loop's
  iteration count and stored value were checked in its generated code.
- Why deferred: A replacement must preserve the signedness of the native
  loop's endpoints through each backend's integer conversions, retain
  Python arithmetic for static bounds, and define how a body consumes a
  result larger than a signed device value. Direct conversion to Uint64
  can zero-extend negative CuTe values; casting both Pallas bounds to
  UInt32 can discard a full word of distance. If the bounds are known
  on the host, pass the calculated count as a representable scalar.
  Split larger intervals if the result does not fit.
- Pinned by: none of the incorrect counts are required by tests.

## cute-nested-loop-narrows-runtime-bounds

- Area: ordinary nested CuTe loops with a signed 32-bit counter, including
  those with 8- or 16-bit storage indices. A gather
  that reads its source tile again checks dynamic loop bounds before that
  read. Computed fragments use a separate lowering
  ([fragment loop widths](https://github.com/pytorch/helion/pull/4672)).
- What is wrong: a runtime exclusive stop of `2**31` becomes `-2**31`,
  so an ascending loop can silently skip every write:

  ```python
  import torch
  import helion
  import helion.language as hl

  @helion.kernel(backend="cute", config=helion.Config(block_sizes=[]))
  def fill(out, lo, hi):
      for row in hl.grid(2):
          for tile in hl.tile(lo, hi, block_size=4):
              out[row, tile.index - tile.begin] = 7
      return out

  out = torch.full((2, 4), -1, device="cuda", dtype=torch.int64)
  print(fill(out, 2**31 - 4, 2**31))
  ```

  Both rows should contain four sevens. CPU replay with signed 32-bit
  conversion left all eight cells at -1; the same kernel with `(0, 4)`
  wrote all eight. Offline SM100a compilation confirmed a signed 32-bit
  comparison after narrowing the 64-bit stop. For this example, pass
  `count = hi - lo` from the host and use `hl.tile(0, count, block_size=4)`
  and `out[row, tile] = 7`. This writes all eight cells. If the body needs
  the absolute coordinate, add its Int64 origin to `tile.index.to(torch.int64)`.
  Split the range on the host if its count or counter increments cannot fit.
  A runtime step must also be nonzero; ordinary nested loops without the
  gather check do not validate this precondition.
- Since: reproduced on `main` at `5a17ffd4f` and on the rebased parent
  `83addb13d`.
- Why deferred: inserting a trapping check before every nested loop
  prevented 202 existing CuTe tests from selecting or compiling their
  schedules, including collective matmuls with bounds read from GPU
  memory. A general fix must preserve those schedules while widening their
  counters or checking their operands before narrowing.
- Pinned by: none for the incorrect no-gather result.

## cute-gather-rereads-loaded-tiles

- Area: CuTe `torch.gather` after a write through an alias of the source.
- What is wrong: The gather reads the source buffer again. If another
  argument shares that storage, an intervening store can replace the value
  that the kernel already loaded, even with one thread.

  ```python
  import torch
  import helion
  import helion.language as hl

  @helion.kernel(
      backend="cute", static_shapes=True, config=helion.Config(block_sizes=[1, 1])
  )
  def gather_after_write(x, alias, ids):
      out = torch.empty_like(x)
      for m, n in hl.tile(ids.shape):
          saved = x[m, n]
          alias[m, n] = 0
          out[m, n] = torch.gather(saved, 0, ids[m, n])
      return out

  x = torch.ones((1, 1), device="cuda")
  ids = torch.zeros_like(x, dtype=torch.int64)
  print(gather_after_write(x, x.view_as(x), ids))
  ```

- Since: On main at `4ca0e2699`, this kernel compiled for SM100a. CPU
  replay of the generated code returned `[[0.0]]`; the saved tile held
  `[[1.0]]`, so the gather should return `[[1.0]]`.
- Why deferred: In a tile with multiple threads, gathering from the saved
  values may require moving them between threads. Keeping the memory reread
  instead requires proving that intervening stores cannot change its source,
  including through other kernel arguments.
  To preserve the original values, save `snapshot = x.clone()` before the
  write in eager PyTorch. In the one-tile example above, call
  `torch.gather(snapshot, 0, ids)` outside the compiled kernel. With
  tile-local indices, [gather each tile
  separately](docs/backends.md#cute-dsl); index zero in a later tile selects
  that tile's first row.
- Pinned by: none.

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
- Pinned by: `test/test_autotuner_heuristics.py::TestPointwiseGatheredSlab::test_rolled_copy_of_an_op_reads_its_index_in_the_enclosing_graph`.

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
- Pinned by: `test/test_autotuner_heuristics.py::TestPointwiseGatheredSlab::test_rows_behind_a_loaded_index_get_no_lane_warps_on_b200`
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
- Pinned by: `test/test_autotuner_heuristics.py::TestPointwiseGatheredSlab::test_rows_behind_a_loaded_index_get_no_lane_warps_on_b200`
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
- Pinned by: `test/test_autotuner_heuristics.py::TestPointwiseGatheredSlab::test_rolled_copy_of_an_op_reads_its_index_in_the_enclosing_graph`
  (`slab_numel == 576`).

## ref-eager-singleton-tile-load

- Area: Eager indexing and Triton and CuTe loads of a statically known
  size-one axis, including CuTe computed fragments.
- What is wrong: With `x = torch.ones(1)` and `hl.tile(4, block_size=4)`,
  the compiled bare-tile load broadcasts across all four live lanes and
  sums to 4. Eager mode slices the tensor and sums to 1. An explicit
  all-true lane mask gives the same discrepancy. With an explicit coordinate
  tensor such as `hl.arange(4)`, Triton and ordinary CuTe can still broadcast
  a singleton, whereas computed fragments and masked eager loads check
  physical bounds. Computed-fragment tile stores also check physical bounds.
- Since: Triton and eager already differed on main at `362b7b182`.
  Ordinary CuTe already broadcast these loads. Computed CuTe fragments
  now use the same rule when the input extent is provably one.
- Why deferred: Matching the documented compiled load requires changing
  eager tile slicing and explicit-mask evaluation together. It would
  change the ref-mode results of existing programs; the user asked to
  keep the documented compiled rule and track the eager change separately.
- Pinned by: `test_singleton_tile_broadcast_feeds_fragment_sum_and_scan`
  and `test_masked_singleton_tile_applies_each_lane_mask` in
  `test/test_cute_fragment_value_codegen.py` check the compiled rule.
  No test requires the eager discrepancy.

## cute-fragment-captures-across-reused-tile-blocks

- Area: CuTe computed fragments in nested loops using the same
  `hl.register_block_size`.
- What is wrong: In a four-lane outer tile, save `ids = outer.index` and
  `saved_end = outer.end`. In a nested two-lane tile with the same block
  ID and origin zero, compute `v = ids + inner.begin`, then
  `out[:] = torch.cumsum(v, dim=0)`, `metrics[0] = v.sum()` and
  `metrics[1] = saved_end`. CPU replay of the generated CuTe kernel
  writes `[0, 1, 0, 0]`, 1, and 2 instead of `[0, 1, 3, 6]`, 6, and 4.
  The fixed contiguous destinations require neither a singleton load nor
  an indirect store.
- Since: Present on main at both the stack base `a2f18c2d0` and the
  current main `29c1bc1f2`: CPU replay returns the same wrong values
  there, on the parent `ddfc9ec52`, and on the corrected fragment load.
- Why deferred: Reductions and scans resolve symbolic shape axes against
  the active block, and scalar bounds can outlive the loop where they
  were defined. Preserving distinct bindings through symbolic scalar
  expressions, fragment transformations and nested graph placeholders
  requires a change to loop provenance beyond fragment-load indexing.
  A workaround is to use distinct block IDs or to consume the captured
  value before entering the inner loop.
- Pinned by: none.

## triton-grid-product-stored-in-int64

- Area: Python grid-index arithmetic used as a stored value in Triton and
  in ordinary and computed-fragment CuTe kernels.
- What is wrong: For an Int64 output of shape (1, 2), nest
  `for i in hl.grid(2, 4)` inside `for r in hl.grid(1)` and store
  `out[r, i - 2] = i * 2**30`. Eager stores [2147483648, 3221225472].
  With default indexing, both backends multiply in Int32 and sign-extend
  the result. Their SM103 programs store [-2147483648, -1073741824].
  An explicit `index_dtype=torch.int64` still gives those negative values
  for this nested loop on both backends. For the top-level Triton grid,
  explicit Int64 indexing does give the eager values. The default Triton
  interpreter reproduces the top-level wrap but misses the nested one.
  Unchecked CuTe computed fragments also evaluate the product in Int32
  with default indexing. Unlike the ordinary nested lowering, a computed
  fragment widens the index before multiplication under explicit Int64 indexing.
- Since: Pristine main at `229afece6` for both backends; also reproduced
  on `6e81e9c09` for Triton.
- Why deferred: Ordinary lowering must distinguish Python index arithmetic
  used as a value from address arithmetic and explicitly typed tensor math.
  Correcting it calls for separate coverage of existing configurations and
  the cost of widening ordinary kernels. This entry concerns unchecked
  stores only.
- Pinned by: `test_grid_induction_product_uses_kernel_index_type` in
  `test/test_cute_fragment_loop_captures.py` checks computed-fragment
  index widths; it does not exercise the ordinary nested-store overflow.

## grouped-d-shift-store-below-output

- Area: CuTe backend, a grouped GEMM over a device layout (offsets or split
  sizes) at a config that names `tcgen05_grouped_d_shift=True`: a config
  passed to the kernel or read from the autotune cache. No search proposes
  the option (`helion/_compiler/cute/grouped_d_shift.py`, `d_shift_searched`).
- What is wrong: the bounded D shift stores every tile of a group through
  one host-encoded rank-3 TensorMap whose address is M rows below the output
  (`helion/runtime/cute/host_wrappers/tcgen05.py`,
  `append_epilogue_tma_wrapper`: `arg.iterator - shape0 * stride0`, with the
  extents (M, N, M + 1), or (N, M, M + 1) for an N,M worklist), so that the
  rows past a group's end fall outside the row extent and are not written.
  The last tile of every group whose rows do not fill it is then a store
  whose box leaves the tensor. `cuTensorMapEncodeTiled` gives a descriptor
  the large allocation class once its extents hold 128 KiB, and PTX ISA 9.5
  ("Out of Boundary Access") leaves such a store undefined unless the
  128 KiB from the descriptor's address are accessible: here that is memory
  below the output, which the kernel does not own.

  ```python
  out = torch.empty((rows, columns), device=a.device, dtype=a.dtype)
  for group_tile, row_tile, column_tile in hl.tile(
      [groups, rows, columns], block_size=[1, block_m, block_n]
  ):
      group = group_tile.index.sum()
      start = offsets[group]
      extent = offsets[group + 1] - offsets[group]
      row_indices = start + row_tile.index
      valid = row_tile.index < extent
      acc = hl.zeros([row_tile, column_tile], dtype=torch.float32)
      for inner in hl.tile(reduction, block_size=block_k):
          left = hl.load(a, [row_indices, inner], extra_mask=valid[:, None])
          acc = torch.addmm(acc, left, weights[group, column_tile, inner].T)
      hl.store(out, [row_indices, column_tile], acc, extra_mask=valid[:, None])
  ```

  With a bf16 `a` of 2048 x 2048, `weights` of 16 x 1536 x 2048, int32
  `offsets`, block sizes registered as (256, 128, 64) and the config
  `helion.Config(block_sizes=[256, 128, 64],
  pid_type="persistent_interleaved", tcgen05_cluster_m=2,
  tcgen05_grouped_mode="worklist_nm", tcgen05_grouped_full_allocation_a=True,
  tcgen05_grouped_d_shift=True)`, the wrapper plans of the generated code
  carry `'bounded_output_shift': True` and its store of the output names no
  per-group descriptor; without the last option the store names the group's
  own descriptor, which is based on the group's first row inside the output.
  Measured on a B200 (driver 580.82.07) only: a store through a descriptor
  of the shift's geometry writes the in-bounds bytes and touches no other
  page, while a load through it reads below the output; the shift never
  loads. No wrong result and no fault is known, and no other device or
  driver was measured.
- Since: on `main` when this entry was added, where the autotuner proposed
  the option as well (178 of the 214 compiler seeds of the binding above,
  and a coordinate of every search of it). "[cutedsl] Keep the bounded
  grouped D shift out of autotune seeds and search" narrowed it to configs
  that name the option.
- Why deferred: the descriptor's class is chosen where the CuTe DSL encodes
  the descriptor, inside its compiled host code, which Helion cannot reach.
  A small-class descriptor takes a CuTe DSL on PTX ISA 9.5 (`.alloc_small`
  on `tensormap.replace`), a descriptor that Helion encodes itself and
  passes to `cuTensorMapSetAllocationClass` (CUDA 13.5), or a copy that the
  kernel makes and clears on the device as CuTe DSL 4.8.1 does: its
  experimental `tensormap_replace`
  (`cutlass/experimental/primitives/nvvm_wrapper.py`) clears bit 21 of
  32-bit word 2 of a descriptor whose `global_dim` or `global_stride` it
  replaces (its "always-safe out-of-bounds fetch mode"), unless its caller
  passes `keep_oob_fetch_window`. The shift's descriptor is static and the
  kernel holds no copy of it: making and clearing one is other generated
  code for every config that names the option, judged on a device by the
  pages its stores touch and by what it costs. Storing the tail
  tiles through the group's own descriptor instead helps only where that
  descriptor is small-class or keeps 128 KiB of the output from its base. It
  also changes the generated code and the timing of every config that names
  the option, and gives up the fixed descriptors the option exists for (on
  that B200 the same config without the option is 0.1% to 6.5% slower), so
  it takes device timings of its own. Refusing the option would stop saved
  configs and cache entries from compiling.
- Pinned by: the descriptor's address and extents, in
  `test/test_cute_grouped_d_shift.py::test_fixed_bounded_d_view_and_live_coordinates_mask_exact_rows`;
  that a config which names the option still compiles to it, and that no
  search proposes it, in `test/test_cute_grouped_d_shift_gate.py`
  (`test_an_explicit_config_keeps_the_shift_and_its_code`,
  `test_a_search_gets_no_seed_that_stores_through_the_shift`).

## grouped-rewritten-tensormap-keeps-large-class

- Area: CuTe backend, the grouped tcgen05 kernels that rewrite a TensorMap
  for each group on the device. On the general grouped path (a grouped GEMM
  over a device layout: offsets or split sizes) that is the output's
  descriptor of a worklist config that does not name
  `tcgen05_grouped_d_shift=True`, the default config and every compiler seed
  of the binding below among them, and the A and B descriptors where the
  config leaves them mutable. The flat grouped kernel
  (`helion/_compiler/cute/tcgen05_flat_grouped_kernel.py`) rewrites the
  descriptors of its row-packed operand and of its output in the same way.
- What is wrong: the kernel copies a host-encoded descriptor into shared
  memory (`TensorMapManager.init_tensormap_from_atom`) and rewrites the copy
  for each group (`update_tensormap`): the output's to the address
  `out.iterator + global_m_start * out.layout.stride[0]` and the extents
  (problem_m, problem_n, 1), or (problem_n, problem_m, 1) for an N,M
  worklist. The rewrite replaces the address, extents and strides; the CuTe
  DSL's `update_tma_descriptor` documents every other field as unchanged,
  and the generated code does nothing else to the copy. The copy therefore
  keeps the allocation class that `cuTensorMapEncodeTiled` chose for the
  whole tensor, the large one once the tensor holds 128 KiB, and PTX ISA 9.5
  ("Out of Boundary Access") leaves a large-class operation whose box leaves
  the tensor undefined unless the 128 KiB from the descriptor's address are
  accessible. For a group whose first row is within 128 KiB of the tensor's
  end they reach past the tensor, into memory the kernel does not own, and
  the last tile of such a group is a box that leaves the group's extents
  whenever its rows do not fill the tile.

  With the kernel and the tensors of grouped-d-shift-store-below-output and
  its config without `tcgen05_grouped_d_shift`, the generated code holds
  `init_tensormap_from_atom(tcgen05_tma_store_atom, ...)`, one
  `update_tensormap` of that copy under the test for a new group, and a
  store of the output that names
  `tma_desc_ptr=tcgen05_grouped_d_tensormap_desc_ptr`. The output is
  2048 x 1536 in bf16, 3072 bytes a row: a group that starts in its last 42
  rows has less than 128 KiB of the output from its first row. The offsets
  are device data, so any launch can hold such a group, for example a last
  group over rows [2047, 2048). The default config of that binding, an N,M
  worklist, copies and rewrites its A and B descriptors (`tma_atom_a`,
  `tma_atom_b`) in the same way and loads through them.
  Measured on a B200 (driver 580.82.07) only: through a large-class
  descriptor with the shift's geometry (grouped-d-shift-store-below-output)
  a store writes the in-bounds bytes and touches no other page, while a load
  reads outside the tensor; the flat grouped kernel, loading short groups at
  the end of its row-packed fp32 operand, touches pages up to 128 KiB past
  the operand. No wrong result is known; a read past the operand is an
  illegal memory access where that memory is not mapped.
- Since: on `main` when this entry was added.
- Why deferred: nothing the kernel calls sets the class of its copy. The
  CuTe DSL's rewrite leaves it, and the driver declares a descriptor as
  opaque quadwords (`CUtensorMap`). A fix has to find the class inside the
  copy and clear it, for the device and driver at hand, in the generated
  code of every kernel that rewrites a descriptor, and it is judged on a
  device: by the pages a launch touches past a short last group, and by
  what the added instructions cost each kernel.
- Pinned by: none for the class. That a worklist without the shift stores
  through its group's descriptor, rewritten at the group's first row, is in
  `test/test_cute_grouped_d_shift_gate.py::test_an_mn_tile_without_the_shift_stores_through_its_groups_descriptor`.
