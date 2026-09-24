# SM103 Mamba-2 SSD final results

`RESULTS.json` records the fresh cold-full CuTe result for each of the six
benchmark shapes against reference SHA
`5b0db79a7090e74e8e56b6b40432287b8b5710267c51f24c9eb9e1e5aea1be2a`.
Every top-level row is a strict latency win. The faster fixed-source manual
frontiers for shapes 1 and 2 are retained as nested evidence rather than used
for the official cold ratios.

`SM103_COLD_CONFIGS.json` preserves the exact configs selected by the final
ordinary cold searches for shapes 1 and 2. `SM103_CONFIGS.json` separately
preserves the faster manual-frontier configs and names their required general
compiler mechanisms.

`SM103_FINAL_COLD_CONFIGS.json` contains the official cold winners for all six
shapes. `configs/nvidia_sm103.json` contains the multi-chunk shape 0–2 entries,
and `configs/nvidia_sm103_single_chunk.json` contains the fused shape 3–5
entries. The exact four-stage kernel and runner are under `kernel/`; the fused
single-chunk implementation is under `single_chunk_kernel/`.

The reference JSON and original breadth harness are unchanged. All six cold
results pass the original harness. Shapes 1 and 2 additionally have exact
five-seed FP64, BF16-output, repeatability, input
immutability, fresh-output, two-graph/three-poison validation and zero-error
memcheck/racecheck evidence at the paths recorded in `RESULTS.json`.

The large-shape pipeline follows the pinned Mamba Triton storage contract:
chunk states, cumulative decay/prefix values, raw `dt`, and `C@B` are FP32;
state-passing output and the public output are BF16. The final public input and
output contract is unchanged.

`FINAL_VERIFICATION.json` records the exact-final default and CuTe suite
receipts, targeted regression checks, sanitizer results, lint classification,
and independent review. The suite failures are the pinned historical baseline
set only; no new failure remains.

`POST_REBASE_VERIFICATION.json` records a six-shape fixed-config replay after
rebasing onto `origin/main`, using the unchanged CUPTI/CUDA-graph harness and
the committed kernel/config payloads.
