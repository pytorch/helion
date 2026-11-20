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
  comparison stays one of the kernels.
- **Same math.** Load in the input dtype, compute in fp32 like Inductor
  (`.to(torch.float32)` at loads, cast to the output dtype at stores), keep
  the graph's op order and constants, and use exact functions (`torch.exp`,
  `torch.rsqrt`, true division); do not enable `fast_math`. Integer index
  math (iota, fmod, floor-divide) uses the same formulas as the graph.
- **Divide by counts the way Inductor does.** A Python number or
  `hl.constexpr` divisor becomes a multiply by its reciprocal (with static
  shapes, `torch.sum(x, -1) / hidden` at `hidden = 192` computes
  `sum * 0.005208333333333333`), which rounds differently from a division
  unless the divisor is a power of two. Inductor divides a mean, a variance
  and a Welford combine by their element count, so write means as
  `torch.mean` (which divides by the count) and divide a combine by the
  summed weights, a tensor; keep other run-time counts runtime arguments.
  Only a divisor that is a literal in the graph (`div(x, 512)`) is a
  reciprocal multiply in Inductor too.
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
  training forward (split reduction writing Welford partials to fp32
  workspaces, a combine kernel that divides by the summed weights and
  updates the running statistics in place, then a pointwise normalize), on
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
