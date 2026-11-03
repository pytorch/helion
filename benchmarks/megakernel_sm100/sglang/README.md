# SGLang baselines (B200, batch 1 decode, cold L2)

This directory times ONE SGLang decoder layer, MoE block or MLA attention call the way SGLang's
CUDA-graph decode runs it.

Every timing uses the shared cold-L2 timer `../coldl2.py`, with ONE `time_cold_many` session per process
(coldl2 allows one: from a process's third profiler session on, CUPTI slows multi-kernel CUDA graphs). The
timer captures the call in a CUDA graph, flushes L2 before every call (see `../README.md`; inputs the call
updates in place are restored just before the flush) and returns the median CUPTI span from the call's first
kernel start to its last kernel end over 200 calls, plus the profiler-free `loop_us`.

| `--workload` | round file (`run_all.sh`) | row (`workload` / `case`) |
|---|---|---|
| `qwen3_8b` | `sglang_qwen3_8b_g<GPU>_r<k>.json` | `qwen3_8b_layer` / `ctx1024` |
| `llama32_1b` | `sglang_llama32_1b_g<GPU>_r<k>.json` | `llama32_1b_layer` / `ctx1024` |
| `qwen3_30b_moe` | `sglang_qwen3_30b_moe_g<GPU>_r<k>.json` | `qwen3_30b_a3b_moe` / `b1` |
| `dsv3_mla` | `sglang_dsv3_mla_g<GPU>_r<k>.json` | `deepseek_v3_mla` / `ctx1024` |
| (TP4: `run_tp4.sh`, `sglang_dsv3_moe_tp4.py`) | `sglang_deepseek_v3_moe_tp4_r<k>.json` | `deepseek_v3_moe_tp4` / `spread`, `clustered`, `edge` |

JSON schema: `{"system": "sglang", "gpu": coldl2.gpu_info(), "versions": {...}, "rows": [...]}`. A row has
`workload`, `case`, `impl`, `median_us`, `p10_us`, `p90_us`, `reps`, `spans_us`, plus `processes` (per-process
median/p10/p90), `kernel_trace_check`, `correctness`, `diagnostics`, and for MLA `mla_call` (the exact
FlashInfer arguments). The row's headline is repeated at the top level with `backends`, `commit`, `timing`,
`notes`.

## Environment

- **SGLang.** Checkout `~/local/sglang` at `02b983603d34c6af09e554ed2b8bf8d85290d9ae` (main, 2026-10-23),
  unmodified, installed in the venv `~/local/hvm_envs/sglang`: torch 2.14.1+cu130, triton 3.8.0,
  flashinfer 0.7.0.post1.
- **Activation.** `source ../env.sh` and run through its `sglang_python` (puts the venv's bin first on
  PATH; `SGLANG_PY` overrides the interpreter). `env.sh` also provides `wait_idle`. The driver sets
  `CUDA_VISIBLE_DEVICES` for its child processes from `--gpu` / `$GPU`.
- **Model configs.** config.json only (dummy weights), in `$SGLANG_MODELS_DIR`, default `models/` here:
  `qwen3_8b_bf16`, `llama3_2_1b`, `qwen3_30b_a3b`, and `deepseek_v3_fp4` (with its `hf_quant_config.json`)
  for the TP4 block.
- **DeepSeek-V3.** The driver writes the public DeepSeek-V3 config to
  `$SGLANG_BENCH_SCRATCH/dsv3_bf16_tp8rank/config.json` with 16 attention heads (one TP8 rank), every layer
  dense, bf16.
- **Tokenizer.** Llama and the DeepSeek config dir need one: `$LLAMA_TOKENIZER` (default
  `~/local/hvm_envs/ae_e2e/llama-3.2-1b`); any Hugging Face tokenizer directory works, since the prompt is
  synthetic token ids (e.g. `huggingface-cli download Qwen/Qwen3-0.6B --include 'tokenizer*' --local-dir DIR`).
- **FlashInfer autotune.** SGLang tunes FlashInfer at startup and keeps the result in its cache; at TP4 its
  rank-consistency check drops caches whose ranks disagree, so every TP4 process tunes again, as a production
  restart would. The winning tactic of the shared expert's GEMM varies between processes (0.8-2.1 us); the
  kernel lists in each result record which one ran.
- **Other paths.** `SGLANG_DIR` (to report the commit), `COLDL2_PATH`, `SGLANG_BENCH_SCRATCH` (per-process
  JSONs and logs; `run_all.sh` and `run_tp4.sh` default it to `$RESULTS/sglang_scratch`, the scripts to `./sglang_scratch`).

## Commands

```bash
cd benchmarks/megakernel_sm100
# one round (one process per workload); repeat with ROUND=2,3,... -- summarize.py takes the best round:
GPU=5 ROUND=1 RESULTS=results bash sglang/run_all.sh
# one workload, N processes in one file (headline = median over processes):
source env.sh
sglang_python sglang/sglang_layer.py --workload qwen3_8b --processes 3 --gpu 5 --out /tmp/sglang_qwen3_8b.json
# MLA cross-check calling FlashInfer with SGLang's arguments without building a model (seconds):
CUDA_VISIBLE_DEVICES=5 sglang_python sglang/sglang_mla_standalone.py [--eager-path]
```

How `sglang_layer.py` runs: the parent starts `--processes` child processes (default 1), each after
`nvidia-smi --query-compute-apps` shows no process on the GPU. A child times in one `time_cold_many`
session and checks the GPU right before and after it; if another process appeared, the parent redoes that
process (up to 3 times, listed under `timing.redone_processes`). The headline `median_us` is the median over
processes of each process's median; `p10_us`/`p90_us` are over all spans.

## How the call is built (all through SGLang's own code)

### 1. Model

- The driver replaces `one_batch.latency_test` with its own work function, then calls
  `sglang.benchmark.one_batch.main`.
- The server args are therefore resolved exactly as for:

  ```bash
  python -m sglang.benchmark.one_batch --model-path <cfg> --load-format dummy --dtype bfloat16 \
    --skip-tokenizer-init --json-model-override-args '{"num_hidden_layers": 4}' --batch-size 1
  ```

  This includes a decode CUDA graph for batch 1.
- The ModelRunner is built by `one_batch.load_model`.
- No backend is overridden. SGLang picks these on B200:

| Model | Attention | Decode CUDA graph | MoE |
|---|---|---|---|
| Qwen3-8B, Llama-1B | `trtllm_mha` (`TRTLLMHAAttnBackend`, FlashInfer trtllm-gen paged decode), page size 64 | `full` | n/a |
| Qwen3-30B-A3B | `trtllm_mha` | `full` | runner `auto` → `FLASHINFER_TRTLLM` (FlashInfer `trtllm_bf16_moe`) |
| DeepSeek-V3 | `trtllm_mla` (`TRTLLMMLABackend`), page size 64, bf16 KV | `full` | n/a |

- FlashInfer autotunes the bf16 MoE at every startup, and the choice varies run to run: the 3 processes
  picked 3 different gate/up and down BMM tactics.
- For DeepSeek-V3, FlashInfer's MLA autotune picks the trtllm-gen kernel.

### 2. Weights

- SGLang's dummy weights are uniform ±1e-3, norm gammas included, which makes every activation about
  1e-11.
- For the dense and MoE workloads the driver re-draws them in place at realistic scales:
  - matrices: uniform with unit output variance;
  - norm gammas: uniform in [0.8, 1.2].
- Kernels, layouts and graphs stay SGLang's. `--sglang-dummy-weights` keeps the original dummy weights.
- For MLA, the timed call's query and KV rows are refilled with N(0, 1).

### 3. Decode batch

- A warm-up `one_batch.latency_test_run_once` runs first.
- Then `one_batch.extend` prefills CONTEXT-1 prompt tokens.
- The decode step is prepared as `one_batch.decode` does it: `ScheduleBatch.prepare_for_decode`, then
  `ForwardBatch.init_new`.
- So `seq_lens` = CONTEXT: the new token attends over CONTEXT-1 cached rows plus its own (position 1023
  for ctx 1024).

### 4. Per-step metadata (excluded from the timed call)

This setup runs once before capture, as SGLang's decode graph runner does it:

- `init_forward_metadata_out_graph` + `init_forward_metadata_in_graph`;
- for `trtllm_mha`: the `update_trtllm_mha_graph_metadata` kernel at the head of SGLang's decode graph.
  It fills the graph's static `cache_seqlens` (checked: [1024]) and page table;
- for `trtllm_mla`: the `create_flashmla_kv_indices_triton` block-table build.

These are not part of the timed call either:

- graph input copies;
- embedding;
- the other layers;
- final norm;
- LM head;
- sampling.

### 5. Timed call

It runs inside SGLang's `model_capture_mode()` and `forward_context(ForwardContext(attn_backend), forward_batch)`,
the same contexts the decode graph runner captures in.

- **layer** (`model.model.layers[2](positions, hidden_states, forward_batch)`, SGLang's signature)
  - Before each call, the residual stream (`forward_batch.residual_stream`) is rebuilt with
    `ResidualStream.from_handoff(hidden, residual, PlainAdd)`. That is the state layer 1 leaves, and the
    inputs are captured from an eager forward.
  - The layer therefore starts with the fused add+RMSNorm of the previous layer's MLP output into the
    residual.
  - Qwen3-8B runs 11 kernels:
    1. fused_add_rmsnorm
    2. QKV GEMM (TgvGemm)
    3. q/k norm
    4. RoPE
    5. KV store
    6. trtllm-gen decode attention
    7. o_proj (TgvGemm)
    8. fused_add_rmsnorm
    9. gate/up dual GEMM with fused SiLU*up
    10. down GEMM (cuBLAS split-K)
    11. split-K reduce
  - Llama-1B runs the same 10 kernels minus the q/k norm, and o_proj is a cuBLAS GEMM.
- **moe** (the rest of `Qwen3MoeDecoderLayer.forward` after `self_attn`)
  - Call sequence: `attn_boundary.finish` → `ffn_boundary.prepare` → `mlp(h, forward_batch)` →
    `ffn_boundary.finish`.
  - It runs 7 kernels:
    1. fused_add_rmsnorm (post-attention; `residual += attn_out` happens inside this kernel)
    2. router GEMM (cuBLAS split-K)
    3. split-K reduce
    4. routing (softmax, top-8, renormalized)
    5. gate/up BMM with SwiGLU
    6. down BMM
    7. finalize (weighted sum of the 8 experts)
  - There is no standalone residual add. The MoE output is added to the residual inside the next layer's
    input fused add+RMSNorm, which is not timed.
- **mla** (one `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(**kwargs)`)
  - The kwargs are recorded from SGLang's `TRTLLMMLABackend._run_decode_kernel` during an eager run of
    the decode graph's body (`model.forward` under the graph metadata and capture mode). The call
    therefore uses SGLang's own buffers and arguments:
    - query `[1, 1, 16, 576]`;
    - paged KV view `[pages, 1, 64, 576]`;
    - static block table `[1, 2560]`, padded to the 163840-token context as the graph path pads it;
    - `seq_lens`, `max_seq_len` = 163840;
    - `bmm1_scale` = 0.13523 (= yarn mscale² / √192);
    - the 150 MB workspace;
    - the persistent multi-CTA counter;
    - `enable_pdl=True` (`is_arch_support_pdl()`), as shipped;
    - `skip_softmax_threshold_scale_factor=None`.
  - It runs 1 kernel: `fmhaSm100fKernel_...HQk576HV512...MultiCtasKv...`, grid [74, 2, 1].
  - Not timed: SGLang's fused KV-write + q-concat kernel (`set_mla_kv_concat_q`) that runs before it.
  - Diagnostic under `diagnostics`, timed interleaved in the same session: the same call with the
    eager-path arguments (`max_seq_len` = seq len, narrow block table; another kernel variant, grid
    [2, 4, 1]). Not the headline, because SGLang decodes with CUDA graphs by default.

The fused add+RMSNorm overwrites its hidden/residual pair in place. Those inputs are restored before
every timed call (coldl2 `resets`). The restore runs ahead of the flush and outside the span, so every call sees identical inputs and identical MoE routing.

## Validation (every process; stored in the JSON)

Numerics run before the timing session; the kernel traces run after it (later profiler sessions are used
only for kernel names and launch shapes, never for times).

### `kernel_trace_check`

- The real decode step `ModelRunner.forward(forward_batch)` is profiled. SGLang replays its decode CUDA
  graph there; `can_run_graph` is asserted.
- The timed call's kernel sequence must appear contiguously in the real decode's kernel list. Kernels are
  matched on name + grid + block + shared memory.
- Expected count:
  - 3x for a full layer: layers 1-3. Layer 0 starts with a plain RMSNorm instead of a fused add.
  - 4x for the MoE block and for the MLA call.
- The JSON lists the timed call's kernels and the whole real decode, each tagged `[timed-call match]` or
  `[outside]`. The kernels outside the matches are:
  - clamp_position, input copy, metadata kernel, embedding;
  - layer 0's RMSNorm and attention (dense), or each layer's attention part (MoE, MLA);
  - final norm, LM head, output copy.
- All rows pass in every process.

### `correctness`

- **layer, moe.** The timed graph's output and updated residual are compared with the same layer's
  output and residual inside an eager full forward of the SGLang model (forward hooks), from identical
  inputs. They are bitwise equal.
- **mla.** The output is compared with FP32 PyTorch softmax attention over the same paged rows: rel L2
  2e-3 to 2.3e-3 at every context.

## Results

Smoke test of the round protocol (2026-10-29, GPU 5, one process per workload, 200 calls; trace check and
numerics pass): Qwen3-8B layer 88.86 us, Llama-1B layer 47.71 us, Qwen3-30B MoE block 28.19 us, DeepSeek-V3
MLA 13.98 us (eager-path diagnostic 14.05 us). The campaign numbers come from the interleaved all-system
runs of `../run_all.sh`.

MLA cross-check (`sglang_mla_standalone.py`): the same FlashInfer call on a 4096-page test pool gives about
13.2 us with the graph-path arguments and 14.1 us with the eager-path arguments (the model-built call reads
SGLang's 46 GB KV pool).

Earlier (superseded) protocol: pooling 5 `time_cold` runs per process read 2-5% higher for multi-kernel
calls (Qwen 91.9, Llama 49.4, MoE 29.1 us; MLA unaffected at 13.8 us), because the runs after a process's
second profiler session are slowed by CUPTI. Those files are in `results/old_protocol_sglang/`.

## DeepSeek-V3 NVFP4 MoE block at TP4 (`sglang_dsv3_moe_tp4.py`, `run_tp4.sh`)

```bash
cd benchmarks/megakernel_sm100
GPUS4=0,1,2,4 ROUND=1 RESULTS=results bash sglang/run_tp4.sh   # -> results/sglang_deepseek_v3_moe_tp4_r1.json
```

One run = one benchmark process that spawns the four TP rank processes; each rank times once
(`time_cold_many`). `run_tp4.sh` sources `../env.sh` and runs `wait_idle "$GPUS4"`; the driver also waits
until `nvidia-smi` shows no foreign process on the four GPUs (matched by UUID), the ranks check again right
before and after their timing session, and a run another process overlapped is redone (up to 3 times,
`timing.redone_runs`). Logs and the raw per-rank JSON go to `$SGLANG_BENCH_SCRATCH/tp4_r<k>/`.

**Block (same boundary as Helion's `deepseek_v3_moe_nvfp4_tp` and vLLM's in `../deepseek_tp4.py`).** One
decode token's MoE input (post-attention RMSNorm output, replicated) through the router GEMM, DeepSeek-V3
grouped top-8 of 256 + the shared expert on each rank's 512-wide intermediate shard, the all-reduce of the
four partial outputs, the residual add and the next RMSNorm. In SGLang that is the tail of
`DeepseekV2DecoderLayer.forward` plus the head of the next layer, called through SGLang's own objects:

```python
fb.residual_stream = ResidualStream(residual)          # as ffn_boundary.prepare leaves it
with layer.ffn_boundary.exit(fb) as ffn_exit, moe_output_buffer_ctx(buf):
    h = layer.mlp(h_in, fb, None)                       # DeepseekV2MoE
h = ffn_exit.finish(h)
out = next_layer.attn_boundary.prepare(h, fb, ...)      # fused all-reduce + residual add + RMSNorm
```

**Model.** `one_batch.main` + `one_batch.load_model` with `--tp-size 4`, the DeepSeek-V3 NVFP4 (modelopt)
checkpoint config `deepseek_v3_fp4` with `{"num_hidden_layers": 5, "first_k_dense_replace": 0}`, dummy
weights, no backend overridden. SGLang picks the FlashInfer TRT-LLM NVFP4 MoE (routing inside the kernel;
`--disable-shared-experts-fusion` is set automatically, so the NVFP4 shared expert runs on SGLang's alt
stream), and FlashInfer's all-reduce fusion for the all-reduce + residual + RMSNorm.

**Weights.** SGLang's dummy loader draws floats in ±1e-3 (input scales can come out negative) and leaves the
packed FP4 weights uninitialized. Each rank patches `model_loader.loader.initialize_dummy_weights` with
`realistic_init` (runs before SGLang's `process_weights_after_loading`, so SGLang re-lays the values out
itself): packed FP4 weights random bytes; FP8 block scales U[0.5, 1.5]; FP32 global weight scales
1 / (2.93 sqrt(1.083 K)) so each projection has unit output variance (K = full contraction length); activation
input scales 0.01 (no FP8 block-scale saturation for unit-variance activations); bf16 matrices unit output
variance; norm gammas U[0.8, 1.2]. Seeds are per parameter name; expert and shared-expert shards also add
the rank, so the four partial outputs differ while replicated tensors are identical on every rank.

**Routing cases** (Helion's `ROUTING_CASES`; case k on layer k+1, the next layer supplies the RMSNorm):
spread (3,17,35,50,77,95,113,127) on layer 1, clustered (0..7) on layer 2, edge
(128,159,160,191,192,223,224,255) on layer 3. Each router weight is set as Helion's `_router_weight` does (from
the token's actual MoE input, layer by layer), so the router logits are 8, 7, ..., 1 on the case's experts
and -12 elsewhere; correction bias ((7e mod 17) - 8) / 1000 as Helion's. `routing_check` recomputes the
grouped top-8 in PyTorch from the bf16 router logits. Inputs (MoE input, residual) are the ones the block
sees in SGLang's eager forward of the decode token.

**Timing.** As `../deepseek_tp4.py`: each case is a CUDA graph, captured inside SGLang's `graph_capture()`
and `model_capture_mode()` (as SGLang's decode graph runner captures; warm-up calls first); the three cases
are interleaved in one `time_cold_many(graph=False)` session per rank; before every call the residual is
restored, L2 is flushed and the ranks align with `torch.ops.symm_mem.one_shot_all_reduce` of 16 bf16 on
SGLang's TP group, all outside the span. Row `median_us` = the largest of the four ranks' medians;
`rank_medians_us` = all four; `p10_us`/`p90_us` of the slowest rank; `rank0` = rank 0's full stats.

**Validation (every rank).**

- (a) Each case's graph replay vs the next layer's attention input and residual inside SGLang's eager
  forward (identical inputs).
- (b) vs an independent FP32 reference: grouped top-8 routing, NVFP4-dequantized experts and shared expert
  (raw weights kept by `realistic_init` before SGLang re-lays them out; expert w13 rows are stored
  [up; gate] for the TRT-LLM layout, SGLang's `load_up_proj_weight_first`), emulated NVFP4 activation
  quantization (bit-exact with FlashInfer's `fp4_quantize` when the global scale is the FP32 reciprocal of
  the input scale, as SGLang computes it), routed weights x 2.5, sum over the four ranks, residual add,
  RMSNorm. Also each rank's pre-all-reduce partial and shared-expert output from SGLang's eager MoE vs the
  reference's. Control: the reference without the case's lowest-weight expert must be off by more than 5x
  the error.
- (c) Kernel trace: per CUDA stream, the timed call's kernel sequence (name, grid, block, smem) must appear
  in the real decode step (`ModelRunner.forward` replaying SGLang's decode graph) once for every MoE block
  followed by another layer (4 of 5).

**Smoke result** (2026-10-29, GPUs 0,1,2,4, round 1, 200 calls per case, all checks pass):

| case | median_us (max over ranks) | rank medians |
|---|---|---|
| spread | 39.62 | 39.62, 39.15, 39.49, 38.98 |
| clustered | 42.08 | 41.86, 41.38, 42.08, 41.58 |
| edge | 39.33 | 39.33, 38.91, 39.33, 38.83 |

Mean over cases 40.34 us. An earlier debug run on the same GPUs gave 42.69 / 40.93 / 40.61 us, so expect
about +-2 us between processes. For comparison, the in-graph MoE block in an old SGLang decode trace (PDL
overlap with the attention, warm launch) measured 34-36.5 us.

- Timed call kernels (identical for all cases): main stream `tiny_n_gemm_kernel` (router) ->
  `NVFP4QuantizeLinear` -> `routingDeepSeek::routingMainKernel` -> `routingIndicesClusterKernel` -> `bmm_E2m1`
  (gate/up + SwiGLU) -> `bmm_Bfloat16_E2m1` (down) -> `sglang::moeFinalizeKernel` ->
  `trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` (all-reduce + residual + RMSNorm); alt stream
  `NVFP4QuantizeSwizzled` -> `nvfp4_gemm_swiglu_nvfp4_quant` (shared gate/up) -> `dense_blockscaled_gemm`
  (shared down). In the real decode the main-stream sequence appears 4x (layers 0-3) and the alt-stream
  sequence 5x.
- Numerics, rel L2: vs SGLang's eager forward 1.1-1.6e-3 (not bitwise: the eager forward is not the
  CUDA-graph path); vs the FP32 reference 2.5-2.6e-3 (output), 1.8-2.0e-3 (residual); per-rank partial
  2.7-3.0e-3, shared expert 1.6-1.7e-3; control without the lowest-weight expert 3.1-5.6e-2.

## Fairness notes

- **One profiler session per process.** A process's third and later profiler sessions slow multi-kernel
  CUDA graphs (single-kernel calls are unaffected), so every system times once per process and repeats
  over processes/rounds.
- **Isolation vs the real decode.** One call is timed in isolation, after an L2 flush.
  - In SGLang's decode graph, consecutive layers run back to back with programmatic dependent launch.
    The TgvGemm `pdl1` kernels and the trtllm-gen attention can overlap the previous kernel's tail.
  - Earlier traces (old protocol): for dense layers and MoE, the in-graph warm spans were 1-5 us below the
    isolated cold spans; for MLA at 1K they agree (at 16K-64K, not measured any more, the MLA kernel took
    1.4-1.6x longer inside SGLang's graph than isolated).
- **Cold vs warm.**
  - Qwen3-8B streams about 385 MB of weights per layer, which is more than L2.
  - Llama-1B's weights per layer (about 122 MB) are about L2-sized.
  - The MLA call at 1K is about 8 us warm and about 13.8 us cold. Its KV, block table, workspace partials
    and multi-CTA counters all come from DRAM, through a chain of dependent misses.
- **DeepSeek-V3 MoE TP4.** The timed graph starts at the router GEMM. In SGLang's decode graph the router
  GEMM is launched early with PDL and overlaps the attention block's tail; isolated, it cannot. The block's
  values (routing aside) do not change its kernels or their launch shapes. The reported number is the
  slowest rank's median, as for Helion and vLLM in `../deepseek_tp4.py`.
- **DeepSeek-V3 MLA model.** It is built at TP1 with 16 heads instead of at TP8 with 128.
  - The per-rank attention call is the same: shapes, buffers, `max_seq_len`, backend, PDL.
  - The MLP is dense, because the attention call does not depend on it.
