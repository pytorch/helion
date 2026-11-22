# Kimi-K3 KDA verify megakernel (Gluon, GB200)

One launch for the KDA layer's speculative verification step: the TP8 shard
(12 heads), one request, T = 8 draft tokens, ReplaySSM cache ring. Comms (the
output projection's TP all-reduce) are left out. Root by root design notes are
in the module docstring of `kda_mtp_verify.py`.

```
python -m pretuned_kernels.megakernels.kda_mtp.kda_mtp_verify   # check + bench (+ SGLang if installed)
python -m pretuned_kernels.megakernels.kda_mtp.mega_compare     # + sgl-nvidia-mega (SGL_NVIDIA_MEGA=path)
```

## Numbers (1 x GB200, us)

Cold L2 by a 256 MB *read* flush before each call (no dirty lines to write
back). "Single" is one CUDA-graph replay, event timed, trimmed mean; "steady" is
20 x (flush, call) in one graph minus a graph of the flushes, per call (the
graph launch amortized, as in a whole-model graph).

| | single | steady |
|---|---|---|
| Helion (this kernel) | 31.3-32.3 | 27.3-28.6 |
| sgl-nvidia-mega (3 PDL kernels) | 31.2-31.5 | 27.9-28.4 |
| SGLang AUTO verify path | 36.3-37.5 | 30.6-31.4 |

The projection GEMMs run at HBM speed (about 20 of the 31 us); the rest is the
recurrence chain after them. Interleave other kernels between calls when you
time it (as the benches do): back to back, the kernel's code stays in the SMs'
instruction caches and it looks faster than it is in a model.

## What matters

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
recurrence.
