# Standalone causal attention on GB300

This folder benchmarks **Helion CuTe only**, forward and backward. Copy the
folder to any location before setup. It includes the Helion kernel definitions,
saved GB300 configurations, and the exact Helion compiler/runtime sources used
by the experiment. It can run independently of the attention-gym harness.

## Cases

BF16, contiguous `[B,H,S,D]`, `B=4`, `H=16`, `D=64`, with sequence
lengths **2048, 4096, 8192**. The Q/K/V generator uses seed 0; backward dO uses
a separate generator with seed 1.

## Setup

Requires Linux AArch64, a GB300 GPU with a CUDA 13 compatible driver,
a C compiler, make, OpenSSL development headers, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
bash setup_env.sh
```

The script builds CPython 3.12.14 and its pinned library dependencies from
verified source archives, then creates separate
`.envs/forward` and `.envs/backward` environments with the recorded package
pins. Set `CAUSAL_REPRO_PYTHON=/path/to/python3.12` to use an existing
Python **3.12.14** interpreter.

The source snapshots under `sources/` contain Helion's compiler/runtime
package and license. Their hashes and original revisions are in `pins.json`.
They are extracted into `.engines/`:

| Component | Forward | Backward |
|---|---|---|
| Helion source | a16ffc60 | 61c840f39 |
| Torch | 2.15.0.dev20260829+cu130 | 2.16.0.dev20261014+cu130 |
| Triton | 3.8.0+gitb252c7c4 | 3.9.0+gitfc8d31a5 |
| cuDNN | 9.24.0.43 | 9.26.0.51 |
| CuTe DSL | 4.7.0 | 4.7.0 |

Torch/Triton use the exact archived nightly wheel URLs and SHA256 hashes.
The full version locks include Helion's dependencies; only the included
Helion kernels are timed. No new Helion autotuning is performed.

## Benchmark

Choose idle physical GPUs and new output directories. GPUs 1 and 3 were used
for the original causal forward and backward measurements on this host.

```bash
python3 benchmark.py --mode forward --gpu 1 --out results/forward
python3 benchmark.py --mode backward --gpu 3 --out results/backward
```

Each command runs all three sequence lengths in separate worker processes.
To run one case, add `--sequence 4096`. `--python /path/to/python` can select
an existing environment with the exact pinned libraries; the included Helion
source snapshots are still used.

## Kernels and timing

[kernels.py](kernels.py) contains:

- `attn_stage`: output-only causal forward, with the included causal stage.
- `_flex_attention_bwd_preprocess`: delta and natural-log LSE to exp2-LSE.
- `helion_flex_attention_backward_kernel`: fused dQ/dK/dV backward.

The three original kernel bodies are preserved. The saved configs under
`configs/forward` and `configs/backward` are copied byte for byte from the
final GB300 campaign. Backward includes preprocess, main, and combined JSON
for each shape.

Correctness is checked against untimed FP32 math. Backward prepares reference
output and natural-log LSE outside timing, then times preprocess + main +
dQ conversion to BF16. Its original 128x128 BlockMask tables are also prepared
outside timing with PyTorch's metadata builder.

Each case uses three cache calls, a 5-second BF16 matmul warmup, and five
CUDA-event measurements with 500 ms warmup + 500 ms measurement. The final
latency is the median of the five run medians. Processes are separated by
20 seconds of idle time. Generated CuTe code is checked for the fused forward
or backward execution path.

Outputs include per-shape JSON/logs, generated kernel code, `results.jsonl`,
and a manifest. Every record includes the exact configs, source revision,
package versions, latency windows, correctness error, and useful TFLOP/s.
The original measurements are included in `pins.json` for reference.
