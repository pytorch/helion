# Grouped-GEMM provider benchmark

`compare_grouped_gemm_defaults.py` compares Helion with selectable public
defaults from DeepGEMM, QuACK, cuDNN, cuBLASLt, and CUTLASS on the eight
reviewed BF16 grouped-GEMM shapes.

Run the campaign from a dedicated Python 3.12 environment with the validated
dependency set below; a normal development environment is not assumed to
satisfy these pins. The benchmark fails closed on the provider versions,
module origins, native ABI versions, and source commits it can verify, while
the complete measured environment is recorded here for reproducibility.

| Dependency | Validated version |
|---|---:|
| PyTorch | `2.13.0+cu130` |
| `nvidia-cutlass-dsl` and its three DSL library packages | `4.7.0` |
| `nvidia-cuda-runtime` | `13.3.29` |
| `nvidia-cuda-nvcc` | `13.3.73` |
| `nvidia-cublas` | `13.6.1.10` |
| `nvidia-cudnn-frontend` | `1.27.0` |
| `nvidia-cudnn-cu13` | `9.24.0.43` (backend `9.24.0`) |
| `quack-kernels` | `0.6.4`, editable checkout at `60d88082272a256fa9b3b2ab631c82cfa78337c6` |

The CUDA runtime and nvcc distributions must share the same installed
`nvidia/cu13` prefix so `bin/nvcc` is beside the runtime package's `lib`
directory. DeepGEMM must be a clean, built checkout at
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`; CUTLASS must be a clean checkout
at `dcf215af68a2d08d305076c152a06f201728cd53`. Their nested dependency and
module-origin checks are enforced by the benchmark.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python \
  benchmarks/cute/compare_grouped_gemm_defaults.py \
  --providers deepgemm,quack,cudnn,cublaslt,cutlass \
  --replicates 3 \
  --helion-selection final_reviewed_aot \
  --deepgemm-root /path/to/DeepGEMM \
  --cutlass-root /path/to/cutlass \
  --output-dir /tmp/grouped-gemm-results \
  >/tmp/grouped-gemm.log 2>&1 </dev/null &
```

`--providers` accepts any nonempty ordered subset. The two source-root flags
are required only when their provider is selected. Run the command detached
for long campaigns, especially `live_autotune`.

Helion modes are:

- `final_reviewed_aot`: use the checked reviewed configuration.
- `compiler_heuristic`: use `ConfigSpec.default_config()` after compiler
  heuristic promotion.
- `live_autotune`: run a fresh forced full compiler-config search for every
  row. Packing and B layout remain fixed to the reviewed profile.

Every provider/replicate gets a fresh process and fresh compiler caches. Each
row uses identical logical inputs and a shared FP32 oracle. Both implementations
are compiled, checked, and captured before timing. The worker then performs a
10-second thermal warmup and calls
`pretuned_kernels._bench.bench_pre_captured_cudagraphs(..., rep=102)`.
That timer clears L2 before every replay and rotates then reverses ordering, so
Helion and the provider each run first in exactly 51 samples.

Run from a clean Helion checkout and write `--output-dir` outside that checkout.
The controller records and rechecks the source commit/tree after every worker.

The campaign resolves the selected GPU to its UUID, requires that GPU to have
no compute applications immediately before and after every worker, and checks
both worker results and telemetry against that UUID. Worker processes receive a
clean set of Helion, provider, compiler, and CUDA loader controls; `CUDA_HOME`
and cuDNN's CUDA runtime are resolved from the installed CUDA runtime package.
SIGINT and SIGTERM terminate the active worker process group before the campaign
exits.

The output directory contains one `result.json`, `worker.log`, and
`telemetry.csv` per provider/replicate plus `summary.json`. Telemetry is sampled
with `nvidia-smi` every five seconds. The summary reports per-provider geometric
mean speedup, wins, worst row, Helion config distributions, and any active GPU
clock-event reason samples. A speedup above one favors Helion.

Provider and Helion selection, autotuning, compilation, graph capture, input
packing, and preprocessing are excluded from latency. Results therefore cover
prepacked BF16 CUDA-graph replay on the recorded GPU; they are not end-to-end
startup measurements or an unqualified SOTA claim.
