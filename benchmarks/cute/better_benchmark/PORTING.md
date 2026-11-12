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
   variant of better-benchmark's, see `common.AnchoredGate`).
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
  port writes into the same input.
- **Leave the backend alone.** Do not set `backend=` in the decorator
  (`HELION_BACKEND` selects it) and do not edit Helion's compiler while
  porting. If the Triton backend passes but CuTe fails (compile error, crash,
  wrong values, a random config that fails), keep the natural port, save a
  minimal standalone script that reproduces the failure, and report it. The
  one exception: when CuTe declines the natural form at its default config,
  a port may reorder its own statements if that changes no result, saying
  why in a comment (`mean_var_236af0aff0ee` reads both seeds before its first
  draw).

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
