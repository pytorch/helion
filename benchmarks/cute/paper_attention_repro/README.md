# GB300 forward and backward attention experiments

`run.py` reproduces the completed paper-matched GB300 campaign: 29 variants,
three BF16 shapes each, and six implementations: Helion CuTe, SDPA, FA4,
FlexAttention Triton, FlexAttention Triton max autotune, and FlexAttention CuTe.

`reference_cases.json` records the original shapes, parameters, useful FLOPs,
and generator hashes. Workers call the archived paper harness's generators
and `compare.run_single`. The prepared runtime contains the source trees
and isolated environments used by the completed runs. `runtime.json`
locates it relative to this folder; see [ENVIRONMENT.md](ENVIRONMENT.md).

## Repeat timings with the saved GB300 winners

From the repository root, use idle GPUs and new output directories:

```bash
python3 benchmarks/cute/paper_attention_repro/run.py \
  --mode forward --gpus 0,1 \
  --config-dir artifacts/attngym-gb300-paper-cold-20261021T235025Z/configs/forward \
  --out artifacts/attention-forward-replay

python3 benchmarks/cute/paper_attention_repro/run.py \
  --mode backward --gpus 2,3 \
  --config-dir artifacts/attngym-gb300-paper-cold-20261021T235025Z/configs/backward \
  --out artifacts/attention-backward-replay
```

These commands use the final GB300 configs: 87 forward winners and 84
backward preprocess/main pairs. The three unavailable backward MLA cases
produce explicit SKIP records.

All shapes and implementations of a variant stay on one physical GPU.
`--gpus` accepts any available list, including `0,1,2,3`. The original
per-variant assignments are in the completed run's `completion_report.json`;
use `--variants` to replay groups on their original GPUs when comparing
absolute timings.

For a fresh search, replace `--config-dir ...` with `--retune`. The
1800-second full-autotune budget applies per forward kernel and separately
to each backward preprocess/main kernel. Both fresh and config-based
backward runs use measured source `61c840f39`.

## Inspect a run without launching kernels

```bash
python3 benchmarks/cute/paper_attention_repro/run.py \
  --mode forward --gpus 0,1 --dry-run --out /tmp/attention-forward-plan

python3 benchmarks/cute/paper_attention_repro/run.py \
  --mode backward --gpus 2,3 --dry-run --out /tmp/attention-backward-plan
```

`--variants sliding-window`, `--shapes 0`, and
`--impls sdpa,flex-cute,helion-cute` select subsets. `--runtime` selects
another prepared runtime manifest or pointer file.

## Measurement contract

The timer is fixed to the paper protocol: three cache calls, a 5-second
BF16 matmul warmup, and five CUDA-event windows with 500 ms warmup and
500 ms measurement. Latency is the median of the five window medians.
SDPA uses a 2-second thermal warmup per candidate. Worker processes on
one GPU are separated by 20 seconds of idle time.

Forward Helion returns output only for the eleven fused variants and
output plus natural-log LSE for the eighteen sparse variants. Correctness
and the fused CuTe flash path are checked. Input tensors, FA4 layout copies,
BlockMasks, and closure tables are prepared outside timing.

SDPA selects the fastest correct cuDNN, flash, or efficient candidate.
Its mask/bias creation is outside timing for fused forward variants, inside
timing for sparse forward variants, and inside timing for masked backward
variants, following the original mixed preparation rule.

Backward times dQ/dK/dV, including Helion delta/LSE preprocessing and final
dQ conversion. Forward state is prepared outside timing. The archived SDPA
backward timer also rebuilds and discards a dense mask/bias; autograd uses
the mask saved by its forward. This extra conversion matches the paper's
timing boundary.

Original VSA uses 128-token query tiles and is unsupported by FlexAttention
CuTe's 256-row FLASH blocks. These SKIP records are preserved.

## Outputs and completed results

Each run saves manifests, source snapshots, hardware captures, cell JSON/logs,
`results.jsonl`, `summary.json`, and winning configs. Speedup uses the fastest
passing displayed baseline per shape, aggregated with a geometric mean.

The completed campaign has 522 records per mode:

- Forward: 87 Helion PASS; 1.54245999x over 87 matched shapes. Eleven earlier
  GB300 configs were reused after checks; the other 76 were tuned on GB300.
- Backward: 84 Helion PASS and three MLA SKIPs; 1.71326093x over 84 shapes.

Forward uses archived compiler cleanup `a16ffc60`; the original measured
paper compiler revisions remain unavailable. The latest backward MLA pass-7
source/configs are also unavailable. These limits are recorded in each result.

The CSV/JSON/PDF reproduction bundle is under
`artifacts/attngym-gb300-results-20261023T193337Z/`; its README describes
regenerating the two PDFs from the exported data.

## Paper harness source

`attention-gym.bundle` is the archived attention-gym harness repository
(branch `flex-template`), the same history as the prepared runtime's
`repos/attention-gym.git`. It holds the forward fused harness `b530d94` (11
variants) and sparse harness `f7844a4` (18 variants); their
`benchmarks/helion_attn/variants.py` hashes match `sources` in
`reference_cases.json`. Read a variant's definition without the prepared
runtime:

```bash
git clone -b flex-template benchmarks/cute/paper_attention_repro/attention-gym.bundle /tmp/attention-gym
git -C /tmp/attention-gym checkout b530d94   # or f7844a4 for sparse variants
```

In `benchmarks/helion_attn/variants.py`, `VARIANTS["<name>"]` gives shapes,
parameters and the Helion template, `_mods_<name>` the FlexAttention
`score_mod`/`mask_mod`, `STAGE_<NAME>` the Helion stage body, and
`variant_flops` the useful FLOPs. `kernels.py` maps templates to kernels
(`TEMPLATES`) and formats stages (`stage_for`).

## CPU checks

```bash
python3 -m pytest test/test_paper_attention_repro.py -q
```
