# Explicit BT32 production integration plan

CPU-only source review, 2026-09-24. No production edits, GPU launches, or benchmark
results are introduced here. The current manually compiled port is a diagnostic;
it does not satisfy the all-40 cold-full goal or the requested per-shape ratio
of at least 1.01. This plan integrates a separately expressed numerical policy,
not a tile-size override on the existing BT16 computation.

## 1. Preserve the separate arithmetic proof

Start from the frozen `matcher/matcher.patch` and its 56 CPU tests. Its immutable
step/region facts distinguish `(16, native_bt16_bf16_rhs_v1)` from
`(32, centered_bt32_fp32_rhs_v2)` (`matcher/chunk_prefill.py:504–569`). The BT32
matcher independently proves centering, restoration, output-scale placement,
the FP16/BF16 inverse polynomial, FP32 projection/residual/beta before one BF16
RHS cast, masks, and FP32 authoritative state (`:1151–1509`). Only enclosing
chunk offsets/iota/extents derive from the proved chunk size. Preserve the
original BT16 arithmetic/inverse proofs and generated wrapper unchanged.

Add the explicit carrier as a separate function in
`benchmarks/cute/kda_prefill_fused_bt32.py`, keeping `carrier-v2.py` arithmetic.
Place the validated device closure in a new compiler package such as
`helion/_compiler/cute/chunk_prefill_bt32/` (device/common/factor/state/issuer/output).
Retain source attribution. Do not import artifact files or a FlashInfer runtime.
Only enable the new exact policy in `_has_chunk_prefill_engine` after its device
implementation is validated. The compiler selects by whole-DAG proof; it must
not dispatch by benchmark index, tensor names, or a hand-picked H/T threshold.

Shared admission must retain the v9 fix: hardware/fast-math/width/flat/i32/grid
checks in `_chunk_prefill_static_eligible`, exact live storage-source replay
before narrowing search, and the captured positive classifier at planning
(`helion/_compiler/cute/chunk_prefill.py:128–194`). Keep negative alias/alignment
bindings distinct and generic search intact. BT32 additionally requires a
target that can launch 1024 threads with its opt-in shared-memory footprint;
do not infer this only from a benchmark GPU name. Initial integration can retain
the proved int64-cu subset unless the int32 address proof and host are tested.

## 2. Use the existing wrapper family with a distinct ABI

Minimal change: retain wrapper kind `chunk_prefill_sm100`, give only BT32 plans
`device_abi=2`, `chunk_size=32`, the exact numerical-policy string, `threads=1024`,
and `schedule=single`. Add a separate BT32 plan emitter before the unchanged
BT16 emitter. This reuses the existing owned-buffer, capture-context, and launch
machinery without broadening every wrapper-kind filter.

`runtime/cute/chunk_prefill.py:22` currently requires ABI1/512 threads. Validate
the new tuple explicitly, with single schedule only; reject forged mixed tuples.
At `:132–191`, generate the BT32 host call with 1024 and a **Python literal 0**
for checkpoint stride: port `device.py:248` declares that argument Constexpr,
whereas the old wrapper emits `cutlass.Int32(0)`. Keep all checkpoint/index/
segment arguments absent or at their supported defaults. Select the BT32 host
in `runtime/cute/launcher.py:1974–1977` from the validated plan. The host owns its
1024-thread launch and `min_blocks_per_mp=1` (`port-v9/device.py:349–373`), not the
frontend seed's four-warps setting.

The external view ABI is already reusable: packed BF16 q/k/v/gate/beta and
output, FP32 a_log/bias, immutable FP32 initial and separate FP32 final state,
runtime FP32 scale/gate-scale, original sequence indices. Existing 16-byte
contiguity/disjoint-output guards remain mandatory. The raw port accepts
all-empty input through a state-copy path, but existing plan validation requires
positive total_tokens (`runtime/cute/chunk_prefill.py:46–49`); expose all-empty
production support only with an explicit BT32 validation branch and wrapper
test. Mixed empty sequences with positive total remain in the main contract.

## 3. Implement the three task orders before advertising them

* **identity:** current BT32 behavior. Preserve it as the default/reference seed.
* **longest_first:** thread optional `seq_order` and constexpr `TASK_ORDER` through
  the BT32 host into its kernel. Before loading cu bounds/state, map grid sequence
  slot to original sequence index by a stable descending-length/ascending-index
  selector over the full `cu.shape[0]-1` domain. Pass that mapped index to every
  state access and cu lookup; output addresses still derive from original cu.
  Do not call the existing BT16 helper unchanged: its candidate stride is 512
  (`chunk_prefill_tmem.py:4059`), which causes duplicate candidate ownership with
  1024 active threads for larger N. A BT32-local helper with stride 1024 and a
  full 1024-participant barrier before role divergence is the smallest safe port.
  Keep the BT16 helper source unchanged. Account for any extra shared slot in
  actual compile resources and mark the published sequence index warp-uniform.
* **longest_first_precompute:** reuse `sequence_order.py:12–63` and existing
  `PrefillResources` (4*N bytes, current-stream/capture-owned). Relax the new
  host's identity-only/no-seq-order checks and read the supplied permutation.
  The wrapper already passes TASK_ORDER=identity with a precomputed order
  (`runtime/cute/chunk_prefill.py:165–186`). Preserve the per-call prepass in
  `_launch_cute_entry` (`launcher.py:6128–6153`), including inside CUDA graphs.
  Never cache the permutation's contents across calls or sort/read cu on CPU.

Order changes must be bitwise equivalent within the same BT32 implementation,
including ties/empty sequences, N=1/31/32/33 and N>1024. No prefix-tail schedule
is implemented by this port; its host explicitly rejects segmentation.

## 4. Narrow search to real options, not nine unsupported combinations

`ConfigSpec.enable_cute_chunk_prefill_task_order_search` currently unconditionally
creates 3 orders x 3 schedules (`config_spec.py:2007–2020`). Add an explicit
legal-schedule argument whose old default remains unchanged. For the BT32
engine use schedule choices `(single,)` and the three implemented orders.
The existing heuristic already enumerates each fragment's Cartesian product
(`_compiler/autotuner_heuristics/cute.py:3031–3068`), so no shape-specific seed
logic is needed. Keep the frontend block-size coordinate pinned to 64 rather
than deleting it; tensor-numel constraints still refer to that coordinate.
Test the actual default population, not only config normalization: three BT32
effective candidates, nine unchanged BT16 candidates, no unsupported prefix
candidate, and full fallback freedom on rejected bindings. Invalid manually
pinned BT32 prefix configs must fail before compilation.

## 5. Cache and correctness gates

Wrapper plans participate in both compilation identity and disk keys
(`runtime/cute/launcher.py:2300–2360`). The existing `helion_key()` recursively
hashes the installed Helion source tree (`autotuner/base_cache.py:80–86`), so
putting all device modules under `helion/` includes their bodies. Test distinct
BT16/BT32 keys and invalidation when a BT32 helper changes; do not rely only on
the unchanged generated stub. Reuse capture ownership and fast-relaunch checks
for the order buffer, especially eager→capture and two independent streams.

Required tests: preserve all matcher negatives and renamed-DAG acceptance;
ABI/policy/thread mismatch refusal; real runtime annotation resolution;
alias/misalignment rebinding; unimplemented schedule refusal; int64-cu ragged
and mixed-empty cases around 31/32/33; nonzero FP32 initial state; full-state
writeback; poisoned 30 replays and input immutability; untimed independent
carrier/oracle checks plus unchanged breadth FP64 tolerances. Do not require
BT32 bitwise equality with BT16, whose rounding policy is different. Do require
BT32 ordering arms to be bitwise equal. Weak-decay numerical failures remain
failures, never a reason to loosen the shared audit. Run cute-verify before
integration is considered complete.

## 6. Honest full-40 provider comparison

Extend the benchmark with an explicit provider identity (`native_bt16` versus
`centered_bt32`) and provider-specific stage name/required plan marker. Record
chunk size and the actual cast policy; pinned config loading must reject a
different provider/policy, not just matching `schedule=fused` and FP32 state.
`make_stage_compiler` already performs real `bound.autotune(normalized, force=True)`
with full effort/no generation or time caps (`kda_prefill_hillclimb.py:273–354`).
Use it unchanged in substance, retaining normal accuracy checks and an identity
seed baseline, followed by independent FP64 validation before accepting timing.

Freeze a new full compiler/library/FlashInfer release. For each of all 40 exact
breadth shapes, cold-tune each eligible Helion provider in its own fresh process
and cache with a recorded fresh search seed. Keep every candidate log/config.
If choosing the better Helion provider, record that outer provider selection
explicitly and retain both cold records; changing arithmetic policy is not a
backend knob on the BT16 carrier. Independently pin the selected cold winner in
a new process, on the same GPU/allocation as its baselines, and measure balanced
full-call CUDA-graph events with cooldown and L2 flushing. Include sorting and
all state-copy work. Never form a final ratio from two different runs/GPUs.

Retain CAKE v0.7.0 commit4d75a33 and its unchanged full-state workspace route.
Also retain the existing direct native-CuTe baseline where claiming “best
available baseline”; select the fastest **validated same-run baseline**. The
current release runner hardcodes `cake,helion` (`local/runner/run_shape.py`), so
its provider list and audit expectations must be explicitly revised for such a
claim. Package metadata alone is not source provenance. Keep all original
before artifacts and report invalid before providers as invalid. Only accepted
cold winners plus independent pinned verification can update the all-40 goal;
the current port/probe measurements remain `goal_eligible=false`.
