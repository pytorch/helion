# Mirage MPK baseline (B200, cold L2, per layer)

Per-layer decode cost of Mirage's persistent megakernel (MPK) for the four workloads of this
benchmark, measured the same way as every other system: `../coldl2.py` (`time_cold_many`: an L2
flush before every call, CUPTI span of the call's kernels, CUDA-graph replay; plus its
profiler-free `loop_us` cross-check).

## Which Mirage

Mirage's default branch `mpk` (`refs/remotes/origin/HEAD` of github.com/mirage-project/mirage;
the branch named `main` is stale), commit **2edbcf4035d61b66fc16a1d36ffab41c4c401549**
(2026-10-29, "Fit linear_kernel pipeline depth to the shared memory budget (#813)"), plus the
benchmark-only hook `mirage_mpk_bench.patch` (it only chooses how many decode steps a launch runs
and from which position; see the patch header).

## What a fresh machine needs (outside this repo)

1. **CUDA 13.0** (`CUDA_HOME`, default `/usr/local/cuda-13.0` in `../env.sh`): MPK compiles every
   megakernel with its nvcc at run time.
2. **The Helion Python environment** (`HELION_PY` in `../env.sh`, torch 2.14.1+cu130; see
   `../README.md`) and on top of it a venv with Mirage's Python requirements, `MIRAGE_PY`
   (default `$HOME/local/mirage-bench/.venv/bin/python`):
   ```bash
   $HELION_PY -m venv --system-site-packages $HOME/local/mirage-bench/.venv
   $HOME/local/mirage-bench/.venv/bin/pip install -r $MIRAGE_DIR/requirements.txt  # transformers 4.57.1, accelerate 1.8.0, z3-solver, ...
   ```
3. **A Mirage checkout at mpk@2edbcf40 with the patch, built in place**, `MIRAGE_DIR` (default
   `$HOME/local/mirage-mpk` in `../env.sh`):
   ```bash
   git clone --recursive https://github.com/mirage-project/mirage $MIRAGE_DIR
   cd $MIRAGE_DIR && git checkout 2edbcf4035d61b66fc16a1d36ffab41c4c401549 && git submodule update --init --recursive
   git apply <this repo>/benchmarks/megakernel_sm100/mirage/mirage_mpk_bench.patch
   # with network: builds the Rust libraries (cargo), the runtime (cmake) and python/mirage/core*.so
   PATH=$CUDA_HOME/bin:$PATH $MIRAGE_PY -m pip install -e . -v
   ```
   Without network (how the numbers were taken): `deps/cutlass`, `deps/json` and the two Rust
   libraries `build/abstract_subexpr`, `build/formal_verifier` copied from an earlier Mirage build
   (their sources are identical), then
   ```bash
   Z3=$($MIRAGE_PY -c "import z3, os; print(os.path.dirname(z3.__file__))"); M=$MIRAGE_DIR
   mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DZ3_CXX_INCLUDE_DIRS=$Z3/include/ \
     -DZ3_LIBRARIES=$Z3/lib/libz3.so -DABSTRACT_SUBEXPR_LIB=$M/build/abstract_subexpr/release \
     -DABSTRACT_SUBEXPR_LIBRARIES=$M/build/abstract_subexpr/release/libabstract_subexpr.so \
     -DFORMAL_VERIFIER_LIB=$M/build/formal_verifier/release \
     -DFORMAL_VERIFIER_LIBRARIES=$M/build/formal_verifier/release/libformal_verifier.so \
     -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ && make -j64 && cd ..   # build/libmirage_runtime.a
   MIRAGE_SKIP_NATIVE_BUILD=1 $MIRAGE_PY setup.py build_ext --inplace           # python/mirage/core*.so
   ```

`run_all.sh` checks each of these and stops with a message naming the variable to set. The
checkout is used in place: `mirage_layers.py` puts `$MIRAGE_HOME/python` first on `sys.path`
(`run_all.sh` sets `MIRAGE_HOME=$MIRAGE_DIR`; MPK compiles each megakernel against
`$MIRAGE_HOME`'s headers, so a wrong one would silently compile another branch).

## Run

```bash
GPU=6 RESULTS=/path/to/results bash run_all.sh                  # everything -> $RESULTS/mirage_*.json
GPU=6 RESULTS=/path/to/results bash run_all.sh dsv3_mla         # one workload (then re-merges all runs)
# one process by hand:
source ../env.sh && cd /tmp && CUDA_VISIBLE_DEVICES=6 MIRAGE_HOME=$MIRAGE_DIR \
  $MIRAGE_PY <this dir>/mirage_layers.py --workload qwen3_8b --build-dir /tmp/mpk_build --out /tmp/x.json
```

Env: `GPU`, `RESULTS` (required), `RUNS` (per-process JSONs, default `$RESULTS/mirage_runs`),
`NPROC` (2), `MIRAGE_DIR`, `MIRAGE_PY`, `MIRAGE_BUILD` (megakernel builds, default
`$RESULTS/mirage_build`, so concurrent runs do not share builds), `COLDL2_PY` (default `../coldl2.py`). Driver args:
`--layers` (default 1,2,4,8), `--instances` (builds per L, default 3), `--placement-seed`,
`--variant`, `--reps` (100).
`run_all.sh` runs every main workload in `NPROC` processes (MLA: 2 x `NPROC`; process p > 1 passes
`--placement-seed p-1`, a random-sized GPU pad before each build so that its builds land at other
addresses), Qwen3 with split-KV attention (`--variant builder-splitkv`) in `NPROC` processes too,
and Llama with demo/llama3's graph (`--variant demo`) in one process. Llama with split-KV is
impossible on mpk (its split-KV attention always applies q/k norm). The merged results go to
`$RESULTS/mirage_<workload>.json` (`summarize.py`; per workload the headline config is the fastest
one run in >= 2 processes whose L=1 output is within 2.5% relative L2 of the PyTorch reference;
`notes.json` holds the per-workload notes copied into it). Qwen3 split-KV is ~12 us/layer faster
but its output is wrong in this harness (15-18% relative L2 at L=1, at page sizes 4096, 256 and
64), so the Qwen3 headline is the demo-default non-split attention (see `notes.json`).
Timing: one `coldl2.time_cold_many` session per process with every model interleaved.
GPU sharing: `run_all.sh` starts a process once no other process has been on the GPU for 30 s and
runs it under `../env.sh`'s `timed` (repeated if another process appears on the GPU while it
runs); in addition the driver polls the GPU every 0.5 s (NVML, by UUID) during its timing session
and exits without results if another process showed up, and `run_all.sh` reruns it (up to 5
attempts; failed attempts' logs are kept as `*.attempt<N>.log`).

## Method

* Mirage compiles a whole graph, so a layer cannot be launched alone. Each process builds every
  workload with L = 1, 2, 4, 8 layers (MLA: 1, 2, 4, 8, 16), every layer with its own weights and
  KV cache, and times one launch per model that runs exactly one decode step (patch:
  `MPK_BENCH_DECODE=1`, `mpk_bench_set_steps(1)`, request admitted at position 1023 with pages
  covering 1024 rows). Per-layer = least-squares slope of T(L) over L, so the fixed per-launch
  work (launch, MPK scheduler start, embedding, and MPK's termination: ~100-150 us, see below) is
  not charged per layer. The same fit on coldl2's profiler-free `loop_us` of every model is
  reported as `loop_us` next to `median_us`.
* Before every timed launch: `init_request_resources()` (Mirage's own request reset, host
  synchronous) and then the L2 flush; the launch (prepare + worker + scheduler kernels) is
  replayed from a CUDA graph.
* Graphs follow Mirage's own code: dense layers = `demo/qwen3/demo.py` / `models/qwen3/builder.py`
  on sm100 (rmsnorm, QKV linear, paged attention with q/k norm + RoPE, split-K o_proj and down,
  rmsnorm, gate/up linear, silu_mul; grids from `models/utils.py`; page size 4096 = demo default,
  no split-KV = demo default); Llama-3.2-1B is the same graph with `enable_qk_norm=False`
  (variant `builder`) or `demo/llama3/demo.py`'s graph (variant `demo`: residual linears, its
  own grids); MoE = `demo/qwen3/demo_30B_A3B.py`'s layer calls and grids; MLA =
  `models/deepseek_v3/builder.py`'s TP8 decode path (`mla_mtp_decode_tp8` + its reduce) with the
  KV already contiguous (no `mla_kv_gather`), max_seq = 1024 so the split count is 8.
  Intermediates are shared by all layers as in Mirage's demos.
* Checks per model: output vs a PyTorch reference of the whole L-layer stack (dense: also the new
  K row Mirage wrote; references over 1023 / 1025 rows must not match), the step counter ends at
  1024, the outputs are still right after all timed replays, and no model's launch wrote into
  another model's buffers.
* MPK's per-launch termination (one scheduler thread posting a termination event to each of the
  80 schedulers with release atomics) takes 100-150 us and varies by ~+-20 us between otherwise
  identical builds; hence several builds per L and the regression over all of them.
