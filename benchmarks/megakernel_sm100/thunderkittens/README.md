# ThunderKittens Megakernels baseline (B200, cold L2, per layer)

HazyResearch Megakernels (github.com/HazyResearch/Megakernels, branch main, commit
**7309cec801537b61fea3b50d7dfe454a6cde578e**, 2025-06-02; ThunderKittens submodule
664c108d16f12707a73d3072ab525f26fb2b4f62), low-latency Llama-3.2-1B megakernel
(`demos/low-latency-llama`). It exists only for Llama-3.2-1B.

## What a fresh machine needs (outside this repo)

1. **The Megakernels checkout** with its submodule, `TK_DIR` (default `$HOME/local/Megakernels`
   in `../env.sh`):
   ```bash
   git clone https://github.com/HazyResearch/Megakernels $TK_DIR
   cd $TK_DIR && git checkout 7309cec801537b61fea3b50d7dfe454a6cde578e && git submodule update --init
   ```
   It is used in place (`tk_llama_layers.py` puts `$TK_DIR` on `sys.path`); not pip-installed.
2. **CUDA 12.8**, `TK_CUDA_HOME` (default `/usr/local/cuda-12.8`): its nvcc builds the kernels and
   its `lib64/libcublas.so.12` is loaded at run time.
3. **A Python with torch 2.14.0+cu130 and transformers**, `TK_PY` (default `$VLLM_PY` of
   `../env.sh`, the vLLM environment, which has both; `../README.md` says how it is built). The
   megakernels modules it imports also need einops, tqdm, safetensors and huggingface_hub (all in
   the vLLM environment); `accelerate` is stubbed.

`run_all.sh` checks each of these and stops with a message naming the variable to set.

## Build

`build_tk.sh <L> <OUT>` copies `$TK_DIR/demos/low-latency-llama` to `OUT/L<L>/src`, sets
`LLAMA_1B_NUM_LAYERS` to L (the kernel bakes the layer count in) and compiles it with the
Makefile's B200 flags (`$TK_CUDA_HOME/bin/nvcc`, `-arch=sm_100a`) and the pybind11 headers shipped
with `TK_PY`'s torch, into `OUT/L<L>/mk_llama.cpython-312-x86_64-linux-gnu.so` (~30 s per L).
`run_all.sh` builds the missing layer counts into `TK_BUILD` (default `$RESULTS/tk_build`, so
every results directory builds its own megakernels with its `TK_PY`).

## Run

```bash
GPU=6 RESULTS=/path/to/results bash run_all.sh   # builds missing L, 2 rounds, then ../mirage/summarize.py
# one process (one layer count) by hand:
source ../env.sh && cd /tmp && TK_DIR=$TK_DIR CUDA_VISIBLE_DEVICES=6 \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.8/lib64 \
  $VLLM_PY <this dir>/tk_llama_layers.py --layers 4 --round 1 --build-dir /tmp/tk_build --out /tmp/tk_L4.json
```

Env: `GPU`, `RESULTS` (required), `RUNS` (per-process JSONs, default `$RESULTS/tk_runs`), `NPROC`
(rounds, default 2), `LAYERS` (default 1,2,4,8,16), `TK_DIR`, `TK_PY`, `TK_CUDA_HOME`, `TK_BUILD`,
`COLDL2_PY` (default `../coldl2.py`). The merged result is `$RESULTS/tk_llama32_1b_layer.json`.
GPU sharing: `run_all.sh` starts a process once no other process has been on the GPU for 30 s and
runs it under `../env.sh`'s `timed` (repeated if another process appears on the GPU while it
runs); in addition the driver polls the GPU every 0.5 s (NVML, by UUID) during its timing session
and exits without results if another process showed up, and `run_all.sh` reruns it (up to 5
attempts; failed attempts' logs are kept as `*.attempt<N>.log`).

## Method

A process can load only one build (two builds in one process gave wrong results or hung), so a
round runs one process per L = 1, 2, 4, 8, 16 (alternating the order between rounds). A process
builds 2 models of its L (own random weights and KV caches, real Llama-3.2-1B shapes), fills KV
rows 0..1022 with random values and times one megakernel launch at `pos_id = 1023` (1024 KV rows)
per model in one `coldl2.time_cold_many` session (CUDA graph; before every launch the barrier
reset the kernel needs, then the L2 flush). Per-layer = least-squares slope over every
(L, model median) point of a round; the result merges the rounds. The LM head inside the launch is
a fixed per-launch cost and cancels. The same fit on coldl2's profiler-free `loop_us` is reported
as `loop_us` next to `median_us`.
Checks: logits vs megakernels' own PyTorch path for the same step (relative L2, argmax), the output
changes when KV row 0 is perturbed and does not when row 1024 is, outputs still right after timing.
