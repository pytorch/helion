# Attention-gym figure comparison on GB300

This suite covers the 29 rows in Google Doc Tab 9 with three representative
cases per row (87 cases). The original B200 figure's shape manifest was not
published. This selection uses attention-gym's B200 preset and scaled versions
of its specialized examples; it does not claim to recover the original 87 shapes.

Upstream source: [meta-pytorch/attention-gym at
4d3bdcf2ea511f335fcf789d7ca1ff88e325c395](https://github.com/meta-pytorch/attention-gym/tree/4d3bdcf2ea511f335fcf789d7ca1ff88e325c395).
The adapted generators retain the upstream BSD license in attention_gym.LICENSE.

## Shapes

All cases use BF16 and batch 4. Q has shape `[4, 16, M, Dqk]`;
K has `[4, Hkv, N, Dqk]`; V has `[4, Hkv, N, Dv]`.
Output has shape `[4, 16, M, Dv]`. Export the complete shape manifest and GPU
assignments with the campaign driver's `--dry-run` command below.

| Group | Variant | Three `(M, N)` cases | Hkv | Dqk / Dv |
|---|---|---|---:|---|
| Dense | noop | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Dense | causal | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Dense | gqa-causal | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 4 | 128 / 128 |
| Dense | mla-causal | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 1 | 192 / 128 |
| Static mask | sliding-window | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Static mask | prefix-lm | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Static mask | dilated-sw | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Static mask | block-diffusion | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Static mask | natten2d | `(1024, 1024)`, `(4096, 4096)`, `(16384, 16384)` | 16 | 64 / 64 |
| Static mask | sta2d | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Static mask | sta3d | `(4096, 4096)`, `(8192, 8192)`, `(16384, 16384)` | 16 | 64 / 64 |
| Static mask | batchify | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Static mask | svg-spatial | `(2176, 2176)`, `(4224, 4224)`, `(8320, 8320)` | 16 | 64 / 64 |
| Static mask | svg-temporal | `(2176, 2176)`, `(4224, 4224)`, `(8320, 8320)` | 16 | 64 / 64 |
| Gather mask | document | `(4096, 4096)`, `(8192, 8192)`, `(16384, 16384)` | 16 | 64 / 64 |
| Gather mask | shared-prefix | `(4096, 4096)`, `(8192, 8192)`, `(16384, 16384)` | 16 | 64 / 64 |
| Gather mask | global-sw | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Gather mask | flamingo | `(2048, 1024)`, `(4096, 2048)`, `(8192, 4096)` | 16 | 64 / 64 |
| Gather mask | vsa | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Speculative | jetspec-tree | `(64, 2112)`, `(64, 4160)`, `(64, 8256)` | 16 | 64 / 64 |
| Speculative | jetspec-training | `(128, 2176)`, `(128, 4224)`, `(128, 8320)` | 16 | 64 / 64 |
| Score mod | alibi | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Score mod | rel-bias | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Score mod | softcap | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Score mod | sandwich | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 64 / 64 |
| Score mod | sigmoid-act | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 16 | 128 / 128 |
| Score mod | graphormer-spatial | `(512, 512)`, `(1024, 1024)`, `(2048, 2048)` | 16 | 64 / 64 |
| Score mod | graphormer-edge | `(512, 512)`, `(1024, 1024)`, `(2048, 2048)` | 16 | 64 / 64 |
| Combined | gemma2 | `(2048, 2048)`, `(4096, 4096)`, `(8192, 8192)` | 4 | 128 / 128 |

## Layouts and parameters

- Ordinary attention selects sequence lengths 2048, 4096, and 8192 from the
  upstream B200 preset. GQA uses 16/4 heads and head dimension 128.
- MLA uses a 128-dimensional latent component and a 64-dimensional positional
  component, concatenated in Q/K. Their single 192-dimensional dot product
  equals the sum of the two component dot products under the common attention
  scale. V/output dimension is 128 and KV heads are shared (16/1 GQA).
  The common scale is `1/sqrt(192)`. This numerical definition adapts the
  upstream visualization, which scales its latent dot separately and uses
  an unscaled RoPE contribution by default.
- Score-only ALiBi, relative bias, sandwich, softcap, and sigmoid activation
  follow the upstream generators without an additional causal mask. Softcap
  is `30*tanh(score/30)`; sigmoid activation is `log(sigmoid(score)+1)`.
- Sliding window is causal with `q-kv <= 1024`; dilated sliding window is
  bidirectional with radius 512 and dilation 2; prefix-LM prefix length is 1024.
- NAT2D uses square canvases 32x32, 64x64, and 128x128 with a 13x13 neighborhood.
  STA2D uses canvases 32x64, 64x64, and 64x128, tile 8x8, and kernel 24x24.
  STA3D uses canvases 16x16x16, 16x16x32, and 16x32x32, tile 4x4x4,
  and kernel 12x12x12. STA inputs are interpreted in tile-major token order.
- SVG uses 128 prompt tokens plus 8, 16, or 32 frames of 256 tokens; width is
  2. Spatial SVG includes the first video frame as an attention sink and rounds
  the band width to 128; temporal SVG follows the upstream modulo-frame mask.
- Block diffusion splits each sequence into equally sized noised/clean halves
  with block size 128. Batchify is noncausal attention within 256-token groups.
- Document masks use 12 seeded variable-length documents summing to M, with
  causality inside each document. FA4 receives the identical offsets through
  its varlen interface. Shared-prefix uses eight equal-length documents in
  two groups, each containing one shared prefix and three continuations.
- Global sliding window uses radius 512, the first 64 global tokens, and a
  global token every 1024 positions. Flamingo has 16 images and attends to
  the most recent four image intervals; each case has M=2*N.
- VSA measures its fine sparse attention pass with provided top-k tile lists,
  not its coarse selector or gated coarse/fine combination. Each 128-token
  query tile selects four distinct KV tiles. Lists vary by batch and head and
  are generated with modifier seed 0, identically for every backend.
- JetSpec tree uses 64 nodes in a binary tree and prefix lengths 2048, 4096,
  or 8192. JetSpec training uses 128 query tokens in 16-token draft blocks
  and the same prefix lengths. The upper-left/tree mask is preserved for
  these rectangular shapes.
- Graphormer uses undirected ring graphs with 512, 1024, or 2048 nodes,
  maximum spatial distance 4, three edge-path positions, and eight edge types.
  Edge labels vary by batch; bias tables vary by head. Precomputed shortest
  paths use clockwise tie-breaking. Biases use modifier seed 0.
- Gemma2 combines 16/4 GQA, dimension 128, softcap 50, and causal window 1024.

## Measurement

The six displayed implementations are forced cuDNN SDPA, native FA4,
FlexAttention Triton, FlexAttention Triton with max autotuning,
FlexAttention CuTe/FLASH, and Helion CuTe. Unsupported native operations are
recorded as SKIP; compilation/runtime errors and numerical failures remain
separate outcomes. No SDPA backend fallback is permitted.
FlexAttention CuTe uses 256-by-128 BlockMask tiles, as required by its FA4
forward kernel; the other masked paths use 128-by-128 tiles.
cuDNN receives dense bias in the input dtype (BF16). Each row is centered by
its finite maximum before conversion; a row constant cancels in softmax and
centering preserves the influential differences in large positional biases.
Fully masked rows retain negative infinity. Checks still compare against the
FP32 modifier definition; any remaining bias quantization failures are
excluded from the comparison.

The output contract is forward attention only. Helion and FA4 omit LSE writes;
the installed FlexAttention implementation may still calculate its internal
LSE. Helion's `noop` and `causal` cases reuse the regular output-only kernels
from `examples/attention.py`; the other variants use the FlexAttention kernel.
Q/K and P/V use FP32 accumulation with BF16 dot operands. Reference softmax
runs in FP32; checks use atol=0.05 and rtol=0.02, including finiteness, shape,
and dtype. Fully masked rows produce zero. All implementations receive the
same input seed and fixed mask/bias parameters.

Input creation, required layout preparation, BlockMask construction, and static
bias/mask tables are outside timing. Each implementation runs in a fresh
subprocess. Defaults are a 10-second thermal warmup followed by five CUDA-event
runs, with a 1-second warmup and 500-ms measurement per run. The score is the
median of run medians. Cold Helion autotuning and winning configs are recorded
by the campaign driver. Use idle GPUs; keep all implementations of a case on
the same physical GPU. Different cases may be sharded across GPUs.

FLOPs count only visible query/key pairs, multiplied by `2*(Dqk+Dv)` and the
actual batch/head mask coverage. The report shows mean TFLOP/s for the three
cases in each row. Speedups first select the fastest correct displayed baseline
for each case, then take a geometric mean of `baseline_ms/helion_ms`. The overall
geometric mean weights all 87 cases equally. Partial results are visibly marked;
failed correctness results never contribute throughput or speedup.

## Commands

Export shapes and GPU assignments without running kernels:

```bash
python benchmarks/cute/attngym_campaign.py --suite figure \
  --out artifacts/attngym-gb300/results --gpus 2,3 --dry-run \
  --manifest-output manifest.json
```

Benchmark one row (three shapes) or one exact case:

```bash
CUDA_VISIBLE_DEVICES=2 HELION_FA4_ROOT=/data/users/shangdiy/flash-attention \
  python benchmarks/cute/compare_attngym_backends.py --suite figure \
  --variant alibi --impl all --json results.jsonl
CUDA_VISIBLE_DEVICES=2 HELION_BACKEND=cute \
  python benchmarks/cute/compare_attngym_backends.py --suite figure \
  --shape-id jetspec-tree.p4096 --impl helion-cute --json results.jsonl
```

Run a full cold-autotune campaign with one GPU per case:

```bash
HELION_FA4_ROOT=/data/users/shangdiy/flash-attention \
  python benchmarks/cute/attngym_campaign.py --suite figure \
  --out artifacts/attngym-gb300/results --gpus 2,3
```

Generate the comparison from one campaign's JSONL files:

```bash
python benchmarks/cute/attngym_report.py \
  --input artifacts/attngym-gb300/results_gpu2.jsonl \
          artifacts/attngym-gb300/results_gpu3.jsonl \
  --manifest artifacts/attngym-gb300/results_manifest.json \
  --expected-shapes 87 --plot-output comparison.png \
  --csv-output comparison.csv --summary-output comparison.json
```

Legacy 16-variant behavior remains available with `--suite legacy` (the default).
Its correctness checks retain the original absolute tolerance: 0.03 for BF16
and 0.02 for FP16.
