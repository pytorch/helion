# B200 megakernel comparison (`megakernel_sm100`)

This package compares Helion's single-kernel decode regions in
`pretuned_kernels/megakernels` with vLLM, SGLang, Mirage and ThunderKittens on
one B200. All timings use one cold-L2 protocol. It lives on the unmerged
branch `eche/megakernel-sm100-bench`, one commit on top of the commit that adds
the megakernels; check that branch out to reproduce. One command reproduces
every number:

```bash
GPU=5 GPUS4=0,1,2,4 ROUNDS=3 benchmarks/megakernel_sm100/run_all.sh
```

The command writes `results/summary.md` (tables) and `results/summary.json`,
plus one JSON file per system, workload and round. `GPU` is one idle B200 and
`GPUS4` four others. `SYSTEMS` (e.g. `"helion vllm"`) and `PARTS` (`one_gpu`,
`tp4`, `whole_model`) select a subset. The published numbers ran the parts as
separate invocations at the same time, on disjoint GPUs, into one `RESULTS`
directory:

```bash
GPU=7 ROUNDS=3 SYSTEMS="helion vllm sglang" PARTS=one_gpu RESULTS=$R benchmarks/megakernel_sm100/run_all.sh &
GPU=7 GPUS4=0,1,2,4 ROUNDS=3 SYSTEMS="helion vllm sglang" PARTS=tp4 RESULTS=$R benchmarks/megakernel_sm100/run_all.sh &
GPU=6 SYSTEMS="mirage thunderkittens" PARTS=whole_model RESULTS=$R benchmarks/megakernel_sm100/run_all.sh
```

## Workloads

All workloads decode one token at batch 1, with random weights and caches.

| Workload | Helion kernel | Boundary (what one call computes) |
|---|---|---|
| `qwen3_8b_layer` | `qwen3_8b/qwen3_decode_layer_bf16` | Qwen3-8B BF16 decoder layer at context 1024: fused residual add + RMSNorm, QKV, QK-norm + RoPE, KV-cache write, paged attention, O projection, residual add + RMSNorm, gate/up, SiLU, down |
| `llama32_1b_layer` | `llama32_1b_decode_layer_bf16` | The same for Llama-3.2-1B (no QK-norm) |
| `qwen3_30b_a3b_moe` | `qwen3_30b_a3b_moe` | Qwen3-30B-A3B BF16 MoE block: RMSNorm, router, top-8 of 128 experts; each system's call includes one residual add or none (see the backends below); 3 routing cases |
| `deepseek_v3_mla` | `flash_mla` | DeepSeek-V3 MLA decode, 16 heads (TP8), 576-wide keys in 64-token pages, context 1024 |
| `deepseek_v3_moe_tp4` | `deepseek_v3_moe_nvfp4_tp` | DeepSeek-V3 NVFP4 MoE at TP4 (router, top-8 of 256 experts, shared expert), the all-reduce, residual add and RMSNorm; 3 routing cases, 4 GPUs |

## Systems and versions

| System | Version | Environment |
|---|---|---|
| Helion | this branch (`main` + `hl.prefetch` + the megakernels), default settings | torch 2.14.1+cu130, triton 3.8.0, flashinfer 0.7.0.post1 |
| vLLM | `main@6a95c89c37` (0.33.0rc1.dev375), its per-commit wheel | torch 2.14.0+cu130 (its pin), triton 3.8.0, flashinfer 0.7.0.post1 (it pins 0.7.2.post1; see below) |
| SGLang | `main@02b983603` (0.5.23.dev747) | torch 2.14.1+cu130, triton 3.8.0, flashinfer 0.7.0.post1 |
| Mirage | `mpk@2edbcf40` (the default branch; `main` is stale) | see `mirage/README.md` |
| ThunderKittens | Megakernels `main@7309cec` | see `thunderkittens/README.md` |

Every framework runs with its default backend selection.

- **vLLM:**
  - **Decoder layers:** vLLM's `Qwen3DecoderLayer` / `LlamaDecoderLayer` compiled with its `support_torch_compile` under a real model config; attention and linears on `auto`. The attention metadata's `max_seq_len` is the context, 1024 (the config's `max_model_len` is 8192), which favours vLLM: with `max_seq_len` at a deployment's `max_model_len` its layers take 0.7-0.9 us (Qwen3-8B, 8K-40K) and 0.8-1.7 us (Llama, 8K-128K) longer (reviewer's measurement).
  - **Qwen3-30B-A3B MoE:** vLLM's RMSNorm + `Qwen3MoeSparseMoeBlock`, compiled the same way, so the norm is one Inductor kernel as in vLLM's model. vLLM fuses the residual add into the next layer's add + RMSNorm, so its call leaves the add out; Helion's call includes it.
  - **MLA:** FlashInfer's TRT-LLM-gen MLA called as vLLM's decode CUDA graph calls it: `max_seq_len` = DeepSeek-V3's `max_model_len` (163840) and a block table that wide. FlashInfer's time depends on `max_seq_len`: at context 1024 its `loop_us` is 12.3 us with 1024, 10.0 us with 4096, 10.2 us with 32768 and 12.0 us with 163840 (reviewer's measurement; Helion 8.2 us).
  - **DeepSeek TP4:** vLLM's NVFP4 MoE path by hand: its `GateLinear` router, which for DeepSeek-V3 (BF16 weight, no FP32 router dtype) runs `F.linear` (vLLM removed its `dsv3_router_gemm` kernel in `8e92248f79`; its remaining router kernels need FP32 logits), NVFP4 activation quant, FlashInfer's TRT-LLM NVFP4 MoE with PDL on (FlashInfer's default, as vLLM calls it), the NVFP4 shared expert on a side stream, then FlashInfer's fused all-reduce + residual + RMSNorm (MNNVL backend, one-shot, PDL, FP32 accumulation, as vLLM's fusion pass sets them). vLLM autotunes FlashInfer's MoE tactic at startup, every token-count bucket up to its `max_num_batched_tokens`; the harness tunes the one-token bucket, the one a batch-1 decode uses (more buckets fail here, because the harness gives FlashInfer a one-token routing-replay buffer to fill, for the routing check). The tuned tactic gives bit-identical output and is 0.3 us faster than FlashInfer's default (the MoE call unsharded on one GPU, mean `loop_us`).
- **SGLang:** layers are built by its own model loader with dummy weights and timed as its decode CUDA graph runs them (`sglang/README.md`). SGLang autotunes FlashInfer at startup; at TP4 it re-tunes in every process (its rank-consistency check drops per-rank caches that disagree), and the winning shared-expert GEMM tactic varies between processes.
- **MLA:** both frameworks use FlashInfer's TRT-LLM-gen kernel on B200, with PDL on as they ship it.

`pins.py` records each environment's versions in `results/pins_*.json` and flags differences from the table.

## Timing protocol

`coldl2.py` is the single timer for every system:

1. **Capture:** the call is captured in a CUDA graph, so host launch gaps do not count, as in a CUDA-graph decode step. As frameworks do, the warmup calls and the capture run on one stream. Helion's launcher keeps its launch state per stream; capturing on a stream other than the warmup's baked a zero-fill of that state into the graph, which every replay repeated (2.7 us on the MLA, 0.4-1.4 us on the decoder layers and the MoE block; the TP4 kernel allocates its state before capture).
2. **Flush:** before every timed replay, L2 (126 MB) is flushed: a 512 MiB int32 buffer is copied into a 1 GiB int64 buffer, which is then summed. The flush ends in a read, so L2 holds only clean lines of the int64 buffer and the timed call pays no write-backs.
3. **Reset:** state the call updates in place is restored before the flush. That state is a residual and the KV-cache row it writes.
4. **Measure:** the time of a call is its GPU span, from the start of its first kernel to the end of its last kernel. The span is measured with CUPTI (`torch.profiler`), and the call's kernels are found by the correlation ids of the launches made inside the timed region, so the flush never counts.
5. **One profiling session per process:** from a process's third `torch.profiler` session on, CUPTI leaves multi-kernel CUDA graphs measurably slower. vLLM's Qwen3-8B layer goes from 98 to 102 us; Helion's single kernel is unaffected. So `time_cold_many` times every call of a process (e.g. the three routing cases) in one session, interleaved, and refuses a second session. `run_all.sh` runs one process per system, workload and round.
6. **Profiler-free time (`loop_us`, the primary column):** CUPTI's per-kernel cost biases spans both ways. Span minus loop is 0.4 us for Helion's Qwen3-8B kernel but 3.9 us for vLLM's 13-kernel layer, and 5.5 us for Helion's TP4 kernel. So before profiling, every call is also timed without CUPTI: cudaEvents around 200 back-to-back (reset, flush, call) iterations, minus the same loop with a one-kernel graph in place of the call, median of 7 trials. This is the call's increment over a trivial graph. After the capture fix in step 1, every Helion graph is a single kernel.
7. **Aggregate:**
   - Each process reports the median of 200 reps per call.
   - `run_all.sh` runs each system `ROUNDS` times, alternating the order of systems between rounds.
   - `summarize.py` credits each system with each case's best round (the minimum over rounds, per routing case), so a baseline that varies between processes gets its best process. SGLang's startup autotune and its PDL overlap vary that way, and so does vLLM's TP4 call, which tunes FlashInfer's MoE tactic in every process; Helion's rounds agree within 0.7 us.

**TP4 (`deepseek_tp4.py`, `sglang/run_tp4.sh`):**

- **Align:** after the flush, every rank runs a one-shot all-reduce of 32 bytes, so the ranks start together. The align step is outside the timed span.
- **Report:** the reported time is the largest of the four ranks' medians (likewise `loop_us`). Every rank's median is recorded.
- **Warm bytes:** the align all-reduce runs on the symmetric-memory group Helion's TP4 kernel also uses, so a few of its signal-pad lines can start the call in L2.

**Mirage and ThunderKittens** can only run whole models, so a single layer cannot be launched on its own. Their per-layer times are slopes over the number of layers, with distinct weights and KV caches per layer and a flush before each launch. Their READMEs give the details.

`run_all.sh` waits until its GPUs run no other process before every step, polls them every 2 s while the step runs, and runs the step again if another process appeared (`timed` in `env.sh`; the SGLang runners also check around their timing). Timings on a shared GPU are not comparable.

### What the flush does and does not do on B200

Measured while building the protocol (`loop_us`, B200):

- **What precedes a call changes it.** After a pure read flush (an FP32 sum of 512 MiB), SM-bound work runs slower for a few us: a 20k-cycle spin took 9.3-9.7 us instead of 8.2 us. After the copy + sum flush it does not.
- **L2 resists streaming eviction.** A line read once per call is evicted by the flush: a buffer re-read every call times the same as one last touched 64 calls earlier. A line read twice between flushes survives every streaming flush tried, a 2 GiB read included. Data a call reads more than once (activations, data shared by many thread blocks) can start L2-warm. This holds for every system.
- **One set of inputs.** Rotating through 8 deep copies of every input (distinct addresses, each last touched 8 calls earlier) changed Helion's kernels by -0.6 to -1.9 us (decoder layers -0.6 to -0.8, MoE -1.5, MLA -1.9) and FlashInfer's MLA (vLLM's environment) by +1.0 us. So the single input set is the conservative choice for Helion.
- **The MLA comparison depends on what precedes the call.** FlashInfer's MLA kernel, as vLLM's decode graph launches it (`max_seq_len` 163840: grid 74 x 2), is far more sensitive to the preceding memory traffic than Helion's. Second reviewer's measurements (`loop_us`, us):

  | Before the call | Helion | FlashInfer | Helion speedup |
  |---|---|---|---|
  | coldl2 flush (copy + sum, this harness) | 7.8-8.3 | 11.6-12.4 | 1.45-1.51x |
  | coldl2 flush + 50 us idle spin | 8.3 | 12.4 | 1.5x |
  | 512 MiB read | 8.5-8.7 | 10.4-11.1 | about 1.22x |
  | 256 MiB or 1.5 GiB zero-fill (Triton's `do_bench`, `pretuned_kernels/_bench.py`) | 8.2 | 8.2 | 1.00x |
  | four BF16 GEMVs of 32 MB each | 8.4 | 8.5 | 1.01x |
  | four BF16 GEMVs of 128 MB each | 9.2-10.8 | 8.5-9.4 | 0.87-0.92x |

  So the MLA ratio is a range, about 0.9-1.5x: a tie after a zero-fill or modest GEMVs, which is closest to a decode step. The other workloads barely move. vLLM / Helion after the coldl2, zero-fill and read flushes is 1.08 / 1.05 / 1.04 (MoE), 1.10 / 1.12 / 1.11 (Llama) and 0.99 / 1.06 / 0.97 (Qwen3-8B).
- **Isolated calls drop cross-boundary PDL.** In a decode step, a baseline's first kernel (FlashInfer's MLA, SGLang's fused add+RMSNorm) overlaps its prologue with the previous kernel, about 0.3-0.45 us per kernel boundary in SGLang's traces. Timing each call alone removes that overlap at the call's start. Helion's kernels do not use PDL.

## Correctness

Every call is checked before it is timed, and its outputs from the last timed replay are checked again afterwards:

- **vLLM run:**
  - **Reference outputs:** it checks each output and saves vLLM's outputs to `results/refs`.
  - **Input fingerprints:** it saves sums of every input along with the outputs.
- **Helion run:**
  - **Same inputs:** it rebuilds the inputs with the same seeded code and checks they match the saved fingerprints.
  - **Same rotary table:** the decoder layers use vLLM's rotary table.
- **Bounds:** the bounds sit a few times above a correct run's error and below that of a kernel that skips work:
  - **Decoder layers:** Helion's output, residual and written KV row against vLLM's. Max abs error <= 0.03, and relative L2 error <= 1e-2 (output, KV row) or 2e-3 (residual). A correct run gives 0.016 and 6e-3 / 9e-4. Attending over 896 of the 1024 KV rows gives 0.07 and 1.4e-2 on the residual.
  - **MLA:** both systems against an FP32 reference: max abs error <= 5e-3, relative L2 <= 5e-3 (correct: 1e-3, 2e-3; 1023 of 1024 rows: 2e-2, 2.6e-2).
  - **Qwen3-30B-A3B MoE:** both systems against an FP32 reference: relative L2 error of the MoE delta < 2e-2, plus exact top-8 routing for Helion.
  - **DeepSeek TP4:**
    - Every rank holds its own expert and shared-expert weights (seeded by rank) and the same token. The residual is about as large as the MoE output.
    - Helion's normalized output and residual sum are checked against an FP32 epilogue of vLLM's all-reduced MoE output (similarity error < 1e-4; correct: 2e-5), plus exact top-8 routing.
    - The vLLM run asserts the check catches a missing rank (error 2.5-3.8e-2) and a skipped residual add (0.5).

## Building the environments

Set `HELION_PY`, `VLLM_PY`, `SGLANG_PY`, `MIRAGE_DIR` and `TK_DIR` (see `env.sh`) to your copies:

```bash
# Helion
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install triton==3.8.0 flashinfer-python==0.7.0.post1 && pip install -e .
# vLLM main@6a95c89c37, from its per-commit wheel (in a separate environment)
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/6a95c89c371939997d9ce3b616360b529760f0bc/vllm-0.33.0rc1.dev375%2Bg6a95c89c3-cp38-abi3-manylinux_2_28_x86_64.whl
pip install --no-deps flashinfer-python==0.7.0.post1   # see below
pip install scikit-learn pytest   # only for running the pretuned modules' own checks there
# SGLang
git clone https://github.com/sgl-project/sglang && git -C sglang checkout 02b983603
pip install -e "sglang/python[all]"           # with torch 2.14.1, triton 3.8.0
```

The SGLang runners read their model configs from `sglang/models` and need any Hugging Face tokenizer directory in `LLAMA_TOKENIZER` (`sglang/README.md`).

`env.sh` defaults to the paths of the machine the numbers were taken on, and also sets a few fixes for that machine, each only where its path exists:

- `HOST_LIBS` supplies a newer `libstdc++` for FlashInfer's prebuilt libraries.
- `pyfix/sitecustomize.py` makes FlashInfer fall back to its JIT build when its AOT libraries cannot load.
- `FLASHINFER_CUBIN_DIR` points at a local cubin cache, and `VLLM_HAS_FLASHINFER_CUBIN=1` tells vLLM it exists. Without it, vLLM probes NVIDIA's download server, fails, and turns TRT-LLM attention off: its Qwen3-8B layer then takes 106 us instead of 96 us (`loop_us`).

The vLLM environment runs FlashInfer 0.7.0.post1, the Helion and SGLang environments' version, while vLLM main pins 0.7.2.post1. FlashInfer downloads its TRT-LLM cubins at run time, and on that machine NVIDIA's server and GitHub were unreachable. 0.7.2.post1's MoE cubins (`batched_gemm-fdba669-16ba8ed`) were not cached anywhere, so vLLM's MoE paths failed under it. Every other cubin set these paths load is the same in both versions. The older FlashInfer does not slow vLLM's DeepSeek MoE (measured unsharded on one GPU): run with the same router in the previous vLLM environment (`6cddad414`, FlashInfer 0.6.18), the TRT-LLM MoE's kernels are the same cubins with the same times (24.5 and 14.4 us) and the call takes 54.2 vs 53.8 us. With network access, install vLLM's pin instead.

## Results

B200 (148 SMs), 2026-10-30, measured as `e8bc9f27c` (the commit `pins_helion.json`
records; this branch differs from it in this README, in dropping the one-GPU
DeepSeek MoE workload from the runners, in the Mirage/ThunderKittens
build-directory defaults, in moving the TP4 kernel's inputs and vLLM call into
its own package unchanged, in dropping the MoE block's unused batch-2/4
kernel (with `run_layers.py`'s two calls into it adjusted), which leaves the
batch-1 kernel's generated code unchanged apart from source-line comments, and
in moving the kernels to `qwen3_8b/` and `qwen3_30b_a3b_moe/` (the MoE kernel
renamed `qwen3_moe` -> `qwen3_30b_a3b_moe`). Every environment ran
torch 2.14 (vLLM its pin, 2.14.0; the others 2.14.1), Triton 3.8.0 and FlashInfer
0.7.0.post1; `pins_*.json` showed no mismatch. Cold-L2 times in us, lower is
better: each system's best of 3 rounds per case, routing cases averaged; speedup
is the best baseline's time over Helion's. The one-GPU workloads ran on GPU 7
with Helion, vLLM and SGLang interleaved (order alternated); the TP4 block ran on
GPUs 0, 1, 2 and 4 at the same time. No step saw another process on its GPUs.
Mirage and ThunderKittens ran afterwards on GPU 6 into the same directory (the
third command, at this branch's runners except their build-directory defaults);
ThunderKittens was built and run with the vLLM environment's torch 2.14.0. Largest spread of a case over rounds:

- **Helion:** within 0.2 us; the TP4 kernel's `loop_us` within 0.7 us.
- **vLLM:** within 0.25 us, except at TP4 (2.9 us `loop_us`, 2.6 us span), where each process tunes FlashInfer's MoE tactic at startup.
- **SGLang:** within 0.4 us; TP4 within 0.6 us.

**Profiler-free (`loop_us`, primary):**

| Workload | Helion | vLLM | SGLang | Mirage | ThunderKittens | Helion speedup |
|---|---|---|---|---|---|---|
| Qwen3-8B decoder layer (BF16, ctx 1024) | 93.0 | 95.0 | 88.5 | 125.3 | - | 0.95x vs SGLang |
| Llama-3.2-1B decoder layer (BF16, ctx 1024) | 44.8 | 50.3 | 47.5 | 75.2 | 62.5 | 1.06x vs SGLang |
| Qwen3-30B-A3B MoE block (BF16) | 22.9 | 27.7 | 26.7 | 46.0 | - | 1.17x vs SGLang |
| DeepSeek-V3 MLA decode (16 heads, ctx 1024) | 8.2 | 12.4 | 12.4 | 18.6 | - | 1.51x vs SGLang (flush-dependent: 1.0-1.5x) |
| DeepSeek-V3 NVFP4 MoE + all-reduce + RMSNorm (TP4) | 28.8 | 40.7 | 36.6 | - | - | 1.27x vs SGLang |

**CUPTI span** (first kernel start to last kernel end):

| Workload | Helion | vLLM | SGLang | Mirage | ThunderKittens | Helion speedup |
|---|---|---|---|---|---|---|
| Qwen3-8B decoder layer (BF16, ctx 1024) | 95.0 | 98.3 | 91.3 | 124.5 | - | 0.96x vs SGLang |
| Llama-3.2-1B decoder layer (BF16, ctx 1024) | 46.3 | 53.3 | 50.4 | 74.7 | 62.5 | 1.09x vs SGLang |
| Qwen3-30B-A3B MoE block (BF16) | 24.9 | 30.1 | 29.0 | 46.0 | - | 1.17x vs SGLang |
| DeepSeek-V3 MLA decode (16 heads, ctx 1024) | 10.1 | 13.8 | 14.0 | 18.6 | - | 1.36x vs vLLM (flush-dependent) |
| DeepSeek-V3 NVFP4 MoE + all-reduce + RMSNorm (TP4) | 33.6 | 43.7 | 40.4 | - | - | 1.20x vs SGLang |

- **DeepSeek TP4:** vLLM main's call is slower than `6cddad414`'s was in the previous campaign (40.7 vs 39.4 us `loop_us`). vLLM removed its DeepSeek-V3 router kernel (`dsv3_router_gemm`, `8e92248f79`), and DeepSeek-V3's gate now runs `F.linear` (cuBLAS split-K): alone and cold, 4.2-4.4 us against 0.55 us.
- **vLLM's last case:** in vLLM's multi-case processes the case timed last can run slower (the Qwen3 MoE's `route_2` by about 1.4 us; one TP4 round's `edge` by 3-4 us), whichever case comes last; the cause is unknown (reviewer's finding). SGLang is the best baseline on both rows, so their speedups do not depend on it.
- **MLA:** the ratio depends on what precedes the call, and on the `max_seq_len` vLLM passes (see the vLLM backend notes above). It is a tie after a zero-fill flush or modest GEMVs (see "What the flush does and does not do"), so read it as 1.0-1.5x, not 1.5x.
- **Mirage and ThunderKittens:** per-layer slopes of whole-model launches (GPU 6, same flush and timer; `mirage/README.md`, `thunderkittens/README.md`).
  - The slope leaves out each launch's fixed cost: MPK's termination, 100-150 us per launch, and ThunderKittens' LM head. It also includes the overlap between consecutive layers that a single call cannot get.
  - Mirage's split-KV attention (its demo's `--split-kv-cache`, 113-114 us on Qwen3-8B) gives wrong output (15-18% relative L2 error at one layer), so the table uses its builder configuration.
  - Mirage's MLA runs without its KV gather.
- **TP4 span:** the span overstates Helion's TP4 kernel relative to the loop more than the baselines' (4.8 us vs 3.0-3.8 us). Helion's span is a single kernel, so the difference is CUPTI's cost on it.

**SGLang's decoder layers run in two modes across processes.** The kernels are
identical, but in some processes each of the 9 gaps between kernels overlaps
0.3-0.5 us more under PDL:

| Mode | Llama loop | Llama span | Qwen3-8B span |
|---|---|---|---|
| Slower (the tables above) | 47.5-48.4 | 50.4-50.7 | 91.0-91.3 |
| Faster | 45.6 | 47.7-47.8 | 88.8 |

- **Slower mode:** every process of this campaign, 8 more on GPU 5, and 12 of 14 on GPU 3.
- **Faster mode:** 2 of 14 processes on GPU 3 (Llama span 47.8 and 48.3 us), and every round of an earlier session with the same SGLang code and versions (an older coldl2 without `loop_us`).
- **Against the faster mode,** Helion's speedup is 1.02x loop / 1.03x span on Llama, and 0.94x span on Qwen3-8B.

Backends each framework picked by default:

- **Decoder layers:**
  - vLLM: FlashInfer attention, unquantized linears.
  - SGLang: TRT-LLM MHA attention (64-token pages).
- **Qwen3-30B-A3B MoE:** both frameworks run FlashInfer's TRT-LLM BF16 MoE.
  - SGLang's block is 7 kernels from its fused add+RMSNorm to the weighted sum.
  - vLLM's call is its compiled RMSNorm plus `Qwen3MoeSparseMoeBlock`.
  - Neither includes a standalone residual add: in both frameworks it is fused into the next layer's add+RMSNorm. Helion's call includes it.
  - SGLang's routing comes from its own model inputs (one case, `b1`).
- **MLA:** both frameworks use FlashInfer's TRT-LLM-gen MLA kernel.
- **DeepSeek TP4:** both frameworks run FlashInfer's TRT-LLM NVFP4 MoE with the shared expert on a second stream, then FlashInfer's fused all-reduce + residual + RMSNorm (MNNVL one-shot). vLLM's router GEMM is cuBLAS (`F.linear`); SGLang's is its own small-N GEMM (`tiny_n_gemm_kernel`).

`sglang/README.md` documents how each SGLang call is built from SGLang's own model code and checked against its real decode step (kernel trace and numerics). Qwen3-8B is the known gap: SGLang's layer is faster.
