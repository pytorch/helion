# Grouped-GEMM provider-baseline benchmark

`compare_grouped_gemm_defaults.py` compares Helion with reproducible public
baselines from DeepGEMM, QuACK, cuDNN, cuBLASLt, and CUTLASS on the eight
reviewed BF16 grouped-GEMM shapes. QuACK is deliberately run with
its public default `tuned=True`; every fresh replicate tunes independently,
and its tuning and selected-config resolution run before graph capture and
timing. CUTLASS uses the first compatible public registry operator without a
timing search.

```bash
CUDA_VISIBLE_DEVICES=0 nohup /tmp/helion-grouped-gemm-venv/bin/python \
  benchmarks/cute/compare_grouped_gemm_defaults.py \
  --providers deepgemm,quack,cudnn,cublaslt,cutlass \
  --replicates 3 \
  --helion-selection compiler_heuristic \
  --deepgemm-root /path/to/DeepGEMM \
  --quack-root /path/to/quack \
  --cutlass-root /path/to/cutlass \
  --output-dir /tmp/grouped-gemm-results \
  >/tmp/grouped-gemm.log 2>&1 </dev/null &
```

The adapters enforce the provider versions used by the campaign: DeepGEMM
2.6.1 at `559d79fb`; the QuACK upstream-main snapshot `c8ec3170`, whose
distribution and module metadata still report 0.6.4 but which is **not** the
formal `v0.6.4` release; cuDNN frontend 1.27.0 with backend 9.24.0.43; cuBLAS
13.6.1.10; and CUTLASS 4.7.0 at its `v4.7.0` tag (`dcf215af`). The QuACK
snapshot includes the post-release CUDA DSL 4.7 migration and is labeled
`quack-main@c8ec3170 (post-v0.6.4, non-release)` in worker results. The formal
`v0.6.4` tag pins CUDA DSL 4.6.2 and therefore cannot share Helion's 4.7.0
worker environment. It is not the QuACK source measured by this paired
campaign. Binary files and imported provider modules are hashed in each worker
result.

The normal Helion lock follows PyTorch's CUDA package versions and is not the
publication environment for this benchmark. Starting from a working Helion
CuTe environment, create a disposable overlay rather than replacing packages
in the development environment:

```bash
python -m venv --system-site-packages /tmp/helion-grouped-gemm-venv
/tmp/helion-grouped-gemm-venv/bin/python -m pip install \
  nvidia-cuda-runtime==13.3.29 nvidia-cuda-nvcc==13.3.73 \
  nvidia-nvvm==13.3.73 nvidia-cuda-crt==13.3.73 \
  nvidia-cuda-cccl==13.3.3.4.1 \
  nvidia-cuda-nvrtc==13.3.33 nvidia-cublas==13.6.1.10 \
  nvidia-cudnn-cu13==9.24.0.43 nvidia-cudnn-frontend==1.27.0 \
  apache-tvm-ffi==0.1.13.post3 torch-c-dlpack-ext==0.1.5 einops==0.8.2
/tmp/helion-grouped-gemm-venv/bin/python -m pip install --no-deps \
  quack-kernels==0.6.4
```

Run the campaign with that environment's Python executable. The controller
rejects missing or mismatched packages before starting a worker.

`--providers` accepts any nonempty ordered subset. The DeepGEMM and CUTLASS
source-root flags are required when their provider is selected. QuACK accepts
either its pinned editable install or an explicit clean Python-source checkout
through `--quack-root`; both modes require `quack-kernels==0.6.4` distribution
metadata only as an installation/dependency contract. That version string does
not identify the measured source as the `v0.6.4` release. A source override is
rejected if that distribution contains native artifacts that cannot be tied to
the checkout. Run the command detached for long campaigns, especially
`live_autotune`.

Helion modes are:

- `final_reviewed_aot`: use the checked reviewed configuration on the canonical
  benchmark B layout; this mode remains B200-only.
- `compiler_heuristic`: use `ConfigSpec.default_config()` after compiler
  heuristic promotion. This is the default and the supported GB300/SM103 mode.
- `live_autotune`: run a fresh forced full compiler-config search for every
  row. This mode is exploratory, because every provider worker may discover a
  different Helion winner; its summaries are marked non-publishable.

Every provider/replicate gets a fresh process and fresh compiler caches. Fixed
provider-selection modes must resolve to the same config in every replicate.
Fresh-autotuned modes such as QuACK may select different configs; the summary
reports the complete per-row config-hash distribution instead of rejecting the
campaign. This does not weaken the fixed Helion-config check. Each row uses
identical seeded logical inputs, one contiguous K-major B layout for all
implementations, and a shared FP32 oracle. The reviewed manifest still selects
Helion's A packing. It selects the timed configuration only in
`final_reviewed_aot`; compiler and live modes merely record it as a reference.
Its preferred B layout is recorded separately when it differs from the
canonical benchmark layout.
Both implementations are compiled, checked with elementwise tolerances and a
global normalized-error bound, replayed twice for exact repeatability, and
captured before timing. The worker then performs a 10-second thermal warmup and
calls
`pretuned_kernels._bench.bench_pre_captured_cudagraphs(..., rep=102)`.
That timer clears L2 before every replay and rotates then reverses ordering, so
Helion and the provider each run first in exactly 51 samples.
Summaries are publication-eligible only for fixed Helion selection modes with
at least three fresh-process replicates. Variation from a declared
fresh-autotuned provider does not make the campaign ineligible; fixed provider
modes still require config invariance.

Run from a clean Helion checkout and write `--output-dir` outside that checkout.
The controller records and rechecks the source commit/tree after every worker.

The campaign resolves the selected GPU to its UUID, requires that GPU to have
no compute applications immediately before and after every worker, and checks
both worker results and telemetry against that UUID. Throughout every worker,
the controller independently records all compute applications on that GPU once
per second and rejects a PID whose process group differs from the isolated
worker process group. Worker children in that group are allowed. Worker
processes receive a clean set of Helion, provider, compiler, and CUDA loader
controls. The CUDA
stack must come from the exact installed distributions
`nvidia-cuda-runtime==13.3.29`, `nvidia-cuda-nvcc==13.3.73`,
`nvidia-nvvm==13.3.73`, and `nvidia-cuda-crt==13.3.73`. The campaign uses
`nvidia-cuda-cccl==13.3.3.4.1`, `nvidia-cuda-nvrtc==13.3.33`, and
`nvidia-cublas==13.6.1.10` from that same package root. The CCCL pin is required
for fresh provider JITs such as DeepGEMM, not only for Helion compilation. The
campaign also requires `nvcc` to report release 13.3, version 13.3.73. It
preloads the pinned CUDA runtime, NVRTC, cuBLAS, and cuBLASLt libraries, then
verifies their mapped paths and hashes after each worker; it never falls back
to an arbitrary toolkit or a library from the underlying development
environment. `TORCH_*` and `PYTORCH_*` controls are scrubbed along with the
other compiler/runtime controls. Each result records semantic CUDA package,
artifact, compiler, and driver identities. The summary rejects any semantic
stack drift across providers or replicates while deliberately ignoring
installation-path differences.
SIGINT and SIGTERM terminate the active worker process group before the campaign
exits. GPU-idle and software power-cap clock events are reported and allowed;
the latter is reproducible only when the reported power limit remains constant
within and across all workers. Hardware slowdown, thermal slowdown, power-brake,
sync-boost, application-clock, and unknown reason bits invalidate the run.

The reviewed AOT artifact remains B200-specific and is rejected on GB300. The
compiler recognizes the grouped matmul and segmented addressing from DeviceIR;
kernels do not annotate themselves as grouped GEMMs. For compact device split
sizes `[G]` or prefix offsets `[G + 1]`, source-M tile families are compiler
schedule choices. For the benchmark's prepacked `[G, 4]` segment table, the
compiler derives legal families from segment starts and capacities and
specializes when the compatible family set changes. Target-specific SM103
policies only rank otherwise-legal candidates, while SM100 retains the
established B200 ranking. The measured table contains source-256 K-major and
N-major workloads fitted in local sweeps; only its K-major entries can match
this canonical-layout campaign. Treat exact benchmark-row results as in-sample,
and use separate held-out distributions for generalization claims. The caller
supplies the canonical B layout and reviewed A packing before binding; the
compiler does not rewrite caller-owned inputs. Expected M/group remains only
benchmark input-generation data and is not passed to the kernel. Reviewed AOT
dispatch uses G/N/K, B major, packed M, and a source tile admitted by the
compiler's worklist validator. When multiple source tiles are compatible,
dispatch shares the compiler's preference order: 224, then 256, then the
compact-32 fallback. Ordinary worklist tensors cache that source tile by tensor
mutation version, avoiding a value read on repeated calls. An
inference-mode worklist has no version counter, so dispatch re-reads its contents
and keys the compatibility guard from those values, matching the launcher's
value-based metadata key.

The output directory contains one `result.json`, `worker.log`, `telemetry.csv`,
and `compute_applications.jsonl` per provider/replicate plus `summary.json`.
Telemetry is sampled with `nvidia-smi` every five seconds; GPU compute
applications are sampled independently every second. Every process snapshot
records the worker process group and the PID, process group, name, and memory
use of each visible compute application.
`--output-dir` is controller-only;
workers receive only their isolated `--run-dir`. The summary reports
per-provider geometric mean speedup, wins, worst row, provider selection
stability, Helion/provider config distributions, and any
active GPU clock-event reason samples. Each row also retains the Helion and
provider median latency from every replicate and their median across
replicates, so absolute timing regressions remain visible after refactors. The
shared timer currently returns only one median per implementation, so raw
paired samples are not retained. A speedup above one favors Helion. The printed
summary explicitly marks exploratory or under-replicated runs as not
publication-eligible.

Provider and Helion selection, autotuning, compilation, graph capture, input
packing, and preprocessing are excluded from latency. Results therefore cover
prepacked BF16 CUDA-graph replay under a canonical K-major deployment contract;
they are not end-to-end startup measurements, provider-best tuning claims, or
an unqualified SOTA claim.
