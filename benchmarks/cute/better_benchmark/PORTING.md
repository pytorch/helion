# Porting a better-benchmark repro to Helion

A port is `kernels/<repro_id>.py`: the Helion kernel(s) equivalent to the
Triton kernel(s) TorchInductor generates for the repro, plus a
`forward(inputs)` that takes the repro's `make_inputs()` list and returns what
`Repro.forward` returns. The comparison is only meaningful if the port does
the same work as Inductor's kernels, so follow these rules. Each port is
self-contained: it repeats the kernels it shares with other ports (the online
softmax statistics, the BatchNorm Welford partials), so that a port reads
alone and a change to one port leaves the others' generated code, and the
autotune records pinned to it, alone.

## Steps

1. Read the repro: `$BB/repros/canonical/<repro_id>/repro.py` (the FX graph
   and the input order), `shapes.json` (the shape points; `["S", [...]]`
   entries are shape-parameter lists passed to `forward`) and `meta.json`.
2. Dump Inductor's code at the shape point you port:
   `python benchmarks/cute/better_benchmark/inductor_code.py <repro_id> --shape <hash> --output /tmp/<repro_id>.py`.
   The header lists the kernel launches; the `call()` function shows the
   buffers each kernel reads and writes.
3. Write the port (template below).
4. Check it on both backends, then shake out the CuTe codegen with random
   configs:
   `python benchmarks/cute/better_benchmark/check_port.py <repro_id> --shape <hash>`
   `python benchmarks/cute/better_benchmark/check_port.py <repro_id> --shape <hash> --backends cute --random-configs 6`
   (`--print-code` shows the generated code.) A port is done when both
   commands print `PASS`: outputs match eager in count, shape, dtype and
   strides, integer outputs exactly, outputs that depend on random numbers
   match Inductor's, and every config passes the anchored gate (a loosened
   variant of better-benchmark's, see `common.AnchoredGate`). An integer or
   boolean output that Inductor's own compiled output does not reproduce
   must equal Inductor's exactly instead, and the check prints it (see
   Outputs below). Every checked call starts from poisoned memory, and a
   port or a config that leaves part of an output unwritten fails
   (`UnwrittenOutput`), whatever values the memory it was given held.
   Random configs the backend declines at compile time are "rejected", which
   is fine; "warn" marks a config the autotuner would skip with a warning (a
   PTXAS failure, an unexpected error inside the CuTe backend): not a
   failure, but possibly a codegen bug to report.

## Rules

- **Same kernels, same traffic.** One `@helion.kernel` per Triton kernel
  Inductor launches, fusing the same ops: read the same input tensors, write
  the same outputs and the same intermediate buffers (a split reduction's
  workspace, for example), and nothing else. Do not merge, split or skip
  kernels, and do not move work to the host. Host code may only allocate
  outputs and take views (`view`, `permute`, `expand`, `as_strided`).
  ATen work that Inductor's generated wrapper does on the host goes into a
  kernel of the port when a kernel can express it: the
  `workspace.sum(dim=0)` after a mix-order reduction
  (`sum_sum_sum_3cac3bbf4e8f`), the `copy_` that fills a complex output
  (`var_mean_d59520e0c4ae`). The one exception is an ATen call that the
  wrapper makes between its kernels and that a Helion kernel cannot express
  on both backends: it stays the same ATen call at the same place in
  `forward()`, and the port's docstring names the call and says why
  (`pointwise_59acdb823e92` sorts 4000 ids with `torch.sort`, as the
  wrapper's `aten.sort.stable` does: Helion's `torch.sort` compares every
  pair of elements, which the Triton backend cannot compile at that length).
  Both implementations then spend the same time in the library, and the
  comparison stays one of the kernels. One departure is prescribed: the
  split statistics of "Counts and statistics" pass four workspaces where
  Inductor passes three, three of them in fp64 (28 bytes per block and
  channel against 12); that traffic is part of the port's time.
- **Same math.** Load in the input dtype, compute in fp32 like Inductor
  (`.to(torch.float32)` at loads, cast to the output dtype at stores), keep
  the graph's op order and constants, and use the functions the graph names
  (`torch.exp`, `torch.rsqrt`, true division), not approximations of them;
  do not enable `fast_math`. Integer index math (iota, fmod, floor-divide)
  uses the same formulas as the graph. The split statistics of the next
  rule are the exception: they are taken in fp64.
  Keep the fusion of the captured shape point at every shape: which values
  stay in fp32 between the graph's bf16 casts and which pass through a bf16
  store that more math reads back. Where Inductor compiles another shape
  with another fusion, its result follows another rounding there and the
  port's differs from it by that rounding, which is not an approximation
  (`var_mean_mean_e790938418f4`'s pooled mean at 128x32x112x112, where
  Inductor averages the stored bf16 activations in kernels of their own).
  Where a port's algorithm differs from the one in Inductor's kernel, its
  results must not be less accurate on average than Inductor's, on the
  inputs a model would feed it and not only on the repro's centered `randn`
  inputs: random differences are fine (another summation order, FMA
  contraction); a bias or a systematically larger error is not. The port's
  docstring says what differs and why it is not less accurate; the next
  rule says how to check it.
- **Counts and statistics.** fp32 `/` is not an IEEE division, on either
  backend or in Inductor's Triton: it multiplies by the divisor's correctly
  rounded reciprocal (`div.full`), so a quotient by a count that is not a
  power of two carries the reciprocal's relative error, up to 2**-24 and the
  same for every dividend (-0.34 * 2**-24 for 392 or 6272). A Python number
  or `hl.constexpr` divisor is the same multiply written out (with static
  shapes, `torch.sum(x, -1) / hidden` at `hidden = 192` computes
  `sum * 0.005208333333333333`). Where Inductor's kernel divides a sum by a
  count (a persistent reduction's statistics), write the same division, so
  both sides round alike: `torch.mean` for a mean, `/` by the count for a
  variance, a literal divisor for a literal in the graph (`div(x, 512)`);
  keep other run-time counts runtime arguments.
  Where Inductor's kernel runs Welford (`welford_reduce`,
  `welford_combine`: every looped or split `var_mean`), its mean is never
  the quotient of a sum: the recurrences divide only differences, and a mean
  that is not small against the spread stays within about half an ulp for
  any count (on centered data, where it is small, Inductor's batch mean
  measured 2 to 11 ulp).
  Reductions over full slices do not express the recurrence, and the fp32
  statistics they do express (a block's mean, the sum of squared deviations
  from it, an n-way combine of the partials) are less accurate than it in
  ways that the repro's centered inputs hide:
  - the mean is a sum times a rounded reciprocal at every level, a bias
    that every block shares once a channel's mean is a few standard errors
    from zero (1.4 to 3.2 times Inductor's error on ReLU outputs at the
    study's four such points);
  - the squared deviations of bf16 values are exact numbers with biased low
    bits (a square is 0, 1 or 4 modulo 8), so an fp32 accumulator rounds
    them one way, the more the longer the chain a config gives it (rstd 9
    times Inductor's error, 5 ulp, at one tuned config with one accumulator
    per block, where the default config's 32 show nothing);
  - a block's statistics folded chunk by chunk in fp32, where a loop cuts
    the block, depend on the chunk size (rstd up to 56 times Inductor's
    error at chunks of 16).

  So take such statistics in fp64, in one pass, as moments about a pivot:
  with `pivot` the block's first row and `shifted = x.to(torch.float64) -
  pivot`, the block's sum is `pivot * n + torch.sum(shifted)` and its m2 is
  `torch.sum(shifted * shifted) - torch.sum(shifted) ** 2 / n`. For bf16
  values the deviations and their squares are exact in fp64 (up to an
  exponent spread of 17 bits) and their sums are while they fit 53 bits;
  beyond that, and for fp32 values, they round at 2**-53, far below an fp32
  ulp. A loop adds its chunks' sums without a fold or a division, both sums
  share one loop over the rows, and the pivot, one of the block's values,
  keeps the cancellation in m2 harmless for any offset.
  Pass the partials on in fp64 workspaces (the sum, the mean and m2; the
  count fits fp32): partials rounded to fp32 add their roundings to the
  result, several ulps of the mean and of rstd where blocks have few rows
  or the data is centered, at shape points whose Inductor kernel does not
  split. Combine them in fp64: add the block sums and divide once for the
  mean (exact sums give the mean of the data, zero included, rounded once),
  and take the second moment about the first block's mean from the stored
  block means' deviations (the n-way formula about a pivot; an fp64
  division per block is what the stored means save). Take the last level's
  variance and `rstd = torch.rsqrt(var + eps)` in fp64 and round each
  result once (in fp32, `torch.rsqrt` is the GPU's approximate reciprocal
  square root on both backends and in Inductor's kernel, off by more than
  half an ulp in one direction over ranges of its argument, which an exact
  variance no longer offsets; in fp64 it measured within an fp64 ulp on
  both backends, while `torch.sqrt` of an fp64 value does not compile on
  Triton, whose `tl.sqrt_rn` takes fp32 only). Store a NaN sum for a block
  that holds an infinity or a NaN, so that its channel's statistics are NaN
  as the recurrence returns them, and give a block without elements a zero
  weight and a guarded quotient
  (`kernels/var_mean_mean_5a1b8c788e4b.py`; with masked loads
  `kernels/var_mean_mean_e790938418f4.py`; with index math
  `kernels/var_mean_var_mean_5cc92f5d49c1.py`;
  `test/test_better_benchmark_statistics.py` holds each kernel to the
  statistics of what it read and each port's `forward()` to the statistics
  of the activation, rounded once).
  Two things go with fp64 statements. A Python scalar of the graph is an
  fp32 constant in the graph's fp32 ops (`add(var, 1e-05)` adds fp32's
  1e-05, in eager on CUDA and in Inductor's kernel alike), and in an fp64
  statement the literal is another number, up to half an fp32 ulp away
  (0.4 for 0.9): hand the kernel the fp32 value
  (`torch.tensor(0.9, dtype=torch.float32).item()`, as an `hl.constexpr`
  argument), or eps biases rstd where the variance is small and a momentum
  weight biases the running statistic. And take the momentum updates of the
  running statistics in fp64 as well, rounded once: the variance's from the
  fp64 variance, the mean's from the mean as the kernel stores it (the
  tensor the graph's update reads, and an output the update can be checked
  against). In fp32, `stat * 0.1 + running * 0.9` rounds two or three
  times, depending on which product a compiler contracts into the sum
  (Inductor's kernel contracts the running statistic's; 1.3 times its error
  where the other one is contracted or none).
  Check by measuring the mean ULP error and the bias of every output
  against an exact float64 evaluation of the graph, next to Inductor's, on
  inputs that are not centered as well as on the repro's (ReLU outputs,
  per-channel offsets of 0.01 to 100 standard deviations, a constant
  offset, offsets that differ between images, an infinity), at the tuned
  and at the default config, and at the repro's other shape points, where
  Inductor's kernels differ. What fp32 sums of squares lose shows on a
  coarse grid (`3 + 0.1 * randn`) and on sparse ReLU outputs
  (`relu(randn - 2)`) at a config with one accumulator per block; what
  fp32 partials lose shows on few-valued channels (a constant with one
  element a bf16 step off) at a shape point with few rows per block.
  Two ports are open exceptions to this rule.
  `kernels/var_mean_b19e82f2a47b.py` and `kernels/var_mean_3a2d56cdea00.py`
  (a LayerNorm over the hidden dim) are looped reductions at their ported
  shape point, where Inductor runs Welford, and they still take two-pass
  fp32 statistics over a full slice; their docstrings give the measured
  effect: Inductor's mean error at CuTe's default and campaign configs and
  on Triton, 1.1 to 1.3 times it on near-constant rows at 3 of 6 random
  CuTe configs. Converting them to the fp64 moments above is open.
- **Let the autotuner tile.** `@helion.kernel(static_shapes=True)` with no
  config. Tile parallel dimensions with `hl.tile`, and express reductions as
  torch reductions over full slices (`x[tile_m, :].sum(-1)`, `torch.amax`,
  `torch.mean`) so the autotuner chooses persistent vs looped reductions.
  Only add an explicit inner `hl.tile` loop when the algorithm needs one
  (e.g. an online softmax over a reduction dim that Inductor also loops).
  A kernel may tile a view with contiguous dims merged where Inductor's
  flat index runs across them (`[N, C, H * W]` of an `[N, C, H, W]`
  buffer, as `pointwise_457ec7dce9c1`'s concatenation does; taking the
  view is host code, and a load that needs the row and the column takes
  them from the tile's index with Inductor's division and remainder). It
  is still one kernel with the same loads and stores. Either form is a
  port: where the search vectorizes one and not the other, report that as
  a compiler gap rather than reshape more ports around it.
- **Stay shape-generic** where it is natural: derive sizes from the input
  tensors and shape-parameter lists rather than hardcoding the captured
  numbers; graph literals (iota lengths, scalar constants, eps) are fine. A
  split reduction whose blocks need not divide the rows evenly loops over a
  block's rows with masked loads (`var_mean_mean_e790938418f4`). A port
  that handles only some layouts of its repro checks the layout in
  `forward()` before its first launch, so it never writes into an input
  (running statistics, say) and then fails.
- **Random numbers.** Inductor's `inductor_random` draws one
  Philox4x32-10 evaluation per element, at the element's row-major flat
  index in the random tensor, from a seed in the repro's seed-table input
  (`inductor_lookup_seed`), keeping word 0 (`tl.rand`). Draw with
  `hl.rand([tile, ...], seed=seeds[i])` over the same logical shape and pin
  `@helion.kernel(static_shapes=True, cute_rng_stream="word0")`: word 0 is
  the Triton backend's only stream, while the CuTe default (`auto`) maps
  offset `i` to `Philox(seed, i // 4)[i % 4]`, which gives other numbers
  for a quarter of Inductor's RNG work. With word0 the masks are
  bit-identical to Inductor's on both backends, so the checks compare the
  outputs that depend on them with Inductor's compiled outputs (eager draws
  from its global generator).
- **Indexing.** Gathers and index math use `tile.index` arithmetic with
  tensor indexing (`x[idx, :]`) or `hl.load(x, [idx0, idx1])`. Strided or
  permuted inputs can be indexed directly; do not copy them to contiguous
  buffers.
- **Outputs.** Return the same number of outputs in the same order with the
  same shapes, dtypes and strides as eager (`check_port.py` reports
  mismatches). An output that is a view of an input or of another output is
  a view in the port too. If the repro writes into an input (`copy_`), the
  port writes into the same input. Where eager leaves an integer result to
  its implementation and Inductor's kernels settle it another way (`topk`
  returns tied values in an implementation-defined order, Inductor's
  decomposition in index order), follow Inductor's kernels: the checks
  compare an integer or boolean output with eager's first, and one that
  Inductor's own default compile does not reproduce either with that
  compile's, which the port must then equal element for element
  (`common.check_outputs`). `check_port.py` prints such an output ("exact
  output vs Inductor") with the number of elements in which Inductor's
  differs from eager's, and the autotune and bench records name it
  (`exact_vs_inductor`); say in the port's docstring what differs and why
  (`pointwise_59acdb823e92`).
- **Leave the backend alone.** Do not set `backend=` in the decorator
  (`HELION_BACKEND` selects it) and do not edit Helion's compiler while
  porting. If the Triton backend passes but CuTe fails (compile error, crash,
  wrong values, a random config that fails), keep the natural port, save a
  minimal standalone script that reproduces the failure, and report it. The
  one exception: when CuTe declines the natural form at its default config,
  a port may reorder its own statements if that changes no result, saying
  why in a comment (`mean_var_236af0aff0ee` reads both seeds before its first
  draw).
- **Let the harness follow the kernels.** `autotune.py` tunes each kernel the
  way the port runs it: the first one alone, every later one behind the
  kernel calls before it (README, How a port is tuned). It follows the calls
  that `forward` makes through the kernels' module-level names, in the
  order `forward` makes them, and the tensors by their storage. So call
  every kernel by its global name (not through an alias bound at import, a
  default argument or a closure), launch the kernels in Inductor's order,
  and hand a kernel the earlier kernel's output itself or a view of it
  (`view`, `permute`, a slice, `as_strided`), not a copy and not a view as
  another dtype: a kernel that reads an earlier output through a copy is
  tuned behind kernels that wrote another buffer, and one that reads it as
  another dtype is tuned alone, as is every kernel after it (the record's
  `regimes` says so). Host code
  between two kernels is not replayed: what it computes is a constant of
  the later kernels' searches, read flushed (where Inductor's wrapper calls
  ATen between two kernels the port must too, and that kernel is then
  tuned on a colder operand than the port hands it).

## Examples

- `kernels/var_mean_004f9a9ea198.py`: one persistent-reduction kernel (a
  gathered residual add feeding a LayerNorm, its statistics two
  `torch.mean`s) with index math from the graph.
- `kernels/var_mean_42fad1ece813.py`: Inductor's three-kernel BatchNorm
  training forward (split reduction writing each block's sum, mean and
  second moment to fp64 workspaces, a combine kernel that updates the
  running statistics in place, then a pointwise normalize), on
  a channels-last input viewed as `[N*H*W, C]`. `forward()` rejects other
  layouts before its first launch, so an unsupported shape point never
  leaves the running statistics updated.

## Template

```python
"""better-benchmark repro <repro_id> (<suite>/<mode>/<model>).

Inductor emits <N> kernel(s): <names>. <What they compute, in one paragraph.>
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def <descriptive_name>(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        ...
        out[tile_m, :] = result.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1, arg1_1 = inputs[:2]
    out = <descriptive_name>(arg0_1, arg1_1)
    return (out,)
```
