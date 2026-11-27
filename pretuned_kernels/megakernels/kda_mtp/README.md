# Kimi-K3 KDA verify megakernel (Gluon)

One launch for the KDA layer's speculative verification step: the TP8 shard
(12 heads), one request, T = 8 draft tokens, ReplaySSM cache ring. The output
projection's TP all-reduce is left out. The roots and how they are scheduled
are described in the docstring of `kda_mtp_verify.py`.

```
python -m pretuned_kernels.megakernels.kda_mtp.kda_mtp_verify   # check, then time vs SGLang
SGL_NVIDIA_MEGA=/path/to/sgl-nvidia-mega \
    python -m pretuned_kernels.megakernels.kda_mtp.mega_compare  # + sgl-nvidia-mega
```

`correctness_check()` needs neither baseline: it compares the output and every
cache write with the FP32 PyTorch reference (`_reference.py`), for a request and
for a padding slot, and runs the step twice for the same bits. Without SGLang,
`main()` times the kernel alone.

## Numbers (1 x GB300, us)

Cold L2 by a 256 MB *read* flush before each call (no dirty lines to write
back). "Single" is one CUDA-graph replay, event timed, trimmed mean; "steady" is
20 x (flush, call) in one graph minus a graph of the flushes, per call (the
graph launch amortized, as in a whole-model graph). The three run interleaved in
one process (`mega_compare.py`); ranges over 4 processes of 3 rounds. The
Helion row was re-measured alone for the 2-split q/k/v layout (on that node
the 4-split layout measured 31.7-34.1 / 28.8-29.1).

| | single | steady |
|---|---|---|
| Helion (this kernel) | 31.7-32.9 | 27.7-28.5 |
| sgl-nvidia-mega (3 PDL kernels) | 31.9-33.6 | 28.3-28.6 |
| SGLang AUTO verify path | 37.0-37.9 | 30.9-31.2 |

The projection GEMMs run at HBM speed (about 20 us); the rest is the recurrence
chain after them. Interleave other kernels between calls when you time it (as
the benches do): back to back, the kernel's code stays in the SMs' instruction
caches and it looks faster than it is in a model.

## What matters

- Split-K projections store a partial per split and their readers sum them in
  a fixed order (no atomics: deterministic). The q/k/v projection uses 2 K
  splits of 3584 over 64-row tiles (144 tasks) with K steps of 128, so a ring
  slot still holds a 16 KB weight tile (K steps of 64 on 64-row tiles halve
  the bytes in flight per SM: +2-4%). 4 splits of 128-row tiles made every
  recurrence task read 4 partials from L2 after the q/k/v wait: 0.9-1.2 us
  slower in the steady state on GB300, about what FP32 atomic sums had saved.
  Each conv tap sums the 2 partials at its shifted rows as two [8, C] loads
  (the re-reads of shifted rows hit L1): no scratch round trip before the
  conv, and no cross-warp reduction (a [S, 8, C] load summed over S reduces
  across warps).
- The recurrence waits for f_a first and computes the decay while the q/k/v
  projection finishes; it reads the q/k/v sums only after. The compiler places
  each scoped wait at the first read of what it orders, in source order: the
  read order in the source is the wait order.
- The norm reads the gate (ready early) before the recurrence output.
- f_a runs as 8 K splits of 896 with beta computed in the same tasks by a
  register dot from a plain-load alias of the hidden states (`hidp`): a second
  TMA dot, or `hl.prefetch` calls in those tasks, made the f_a root 2x slower.
- No L2 prefetch at the end of the gate projection: a root's release waits for
  its trailing prefetch issue (it delayed the norm about 3 us); the output
  projection's producer loads `w_out` ahead of its wait anyway.
- Every global load before any layout conversion in the recurrence (a
  conversion is a shared-memory round trip with a CTA barrier, serializing the
  load latencies), per-channel vectors loaded as [8, C] (zero-stride index)
  rather than broadcast, and the conv weight passed contiguous.

## Tried, no gain

Per-head WY chunked delta rule; full-head (128-row) recurrence tasks; the norm
fused into the output projection; the decay as its own root; the gate
projection after the recurrence on its SMs; O-proj self-prefetch of its tail
weights; 6-way K split; K = 128 projection steps; norm by last arriver in the
recurrence. For the deterministic q/k/v sum (GB300): shifting the summed rows
in registers (a select-and-sum: +43%; a 0/1 shift matrix dot: +5%), 32- or
64-row recurrence tasks to share the sums (+7%, +39%), 2 K splits of 128-row
tiles (72 tasks: +10%), one split of 32-row tiles (shared memory overflow),
8-token projection tiles (shared memory overflow), and the conv per token in
the delta rule's layout (one scratch round trip fewer, 4.5x slower). f_a in 4
K splits (half the partials the recurrence sums) and the gate projection in 2
K splits of 64-row tiles: no gain.
