# Helion TPU attention results

This directory is a reproducible snapshot of BF16 attention performance on one TPU v7 device. It is intentionally isolated from Helion's production package and is carried by a **not-for-commit** branch commit.

The comparison covers:

- dense and causal attention, both with shape `[8, 32, S, D]`;
- `S` from 1K through 64K;
- head dimensions 64, 128, and 256;
- the Helion Pallas backend and Tokamax SplashAttention.

The recorded runs use Helion compiler revision `1496cb1b9`, JAX/JAXlib 0.11.2,
libtpu 0.0.48, TorchTPU 0.1.1, PyTorch 2.15.0.dev20260921+cpu, and Tokamax
revision `980b70443409f0aa5cad0995a065f5289f5d0f8c`. The kernel sources and
fixed configurations are included here; no configuration search is repeated
when running the benchmark.

## Files

- `attention.py`: dense online-softmax Helion kernel.
- `causal_attention.py`: causal Helion kernel with a triangular panel worklist and diagonal-subtile skipping.
- `tokamax_attention.py`: Tokamax SplashAttention adapter used by both comparisons.
- `configs.py`: fixed measured-best Helion and Tokamax configurations. The benchmark never autotunes.
- `benchmark.py`: correctness checks, isolated-process timing, JSON/Markdown output, and plots.
- `results.json`: machine-readable measurements and exact configurations.
- `results.md`: generated result tables.
- `dense_attention.png` and `causal_attention.png`: generated throughput plots.

## Run

Install Helion, TorchTPU, JAX/libtpu, Tokamax, and Matplotlib, then run from the repository root:

```bash
export HELION_BACKEND=pallas
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
python -m tpu_attention_results.benchmark
```

The driver adds these required runtime flags while preserving unrelated existing flags:

```text
--xla_tpu_dvfs_p_state=7
--xla_tpu_scoped_vmem_limit_kib=65536
```

TorchTPU and native JAX cannot own PJRT in the same process. The parent therefore launches each implementation and shape in a fresh child process and checkpoints `results.json` after every child. Resume an interrupted sweep with:

```bash
python -m tpu_attention_results.benchmark --resume
```

For a quick smoke test:

```bash
python -m tpu_attention_results.benchmark \
  --modes dense causal \
  --head-dims 64 \
  --sequence-lengths 1024 \
  --samples 3 \
  --output-dir /tmp/tpu-attention-smoke
```

## Measurement conventions

Both modes use B=8/H=32. The dense measurements and their execution path are
retained unchanged. Every worker checks an all-ones result (`atol=rtol=0.02`)
and exact repeated-call determinism before timing. The causal workers also
check random BF16 inputs against a sampled FP32 reference (`atol=rtol=0.01`,
relative L2 at most 0.02). That check runs the full kernel and compares six
query rows in three batch/head pairs; it is not a full quadratic reference.

Causal all-ones and repeated-call checks compare every output element
head-by-head on CPU. This avoids full-size comparison temporaries in HBM
without changing either tolerance. Validation remains outside timing.

Reported latency is the median of 20 synchronized samples after three warmups.
Every sample contains one call, for both modes and both providers. Compilation,
tuning, and static mask/worklist preparation are outside timing.

Both Helion modes call their kernels directly with TorchTPU tensors through
Helion's TorchTPU-aware Pallas launcher. The causal timing includes its public
wrapper's cached worklist lookup, tensor views, and kernel call. Tokamax uses
JAX. Each worker runs in a fresh process so these runtime interfaces never
initialize PJRT in the same process.

The causal comparison computes `softmax(Q K^T / sqrt(D)) V` with a causal mask.
Its Tokamax adapter supplies the required caller-side Q scaling inside the
timed JAX function. For causal attention, the benchmark measures both legal
batch layouts (batch-vmap and flattened batch/head dimensions) and reports
the faster Tokamax median. Both candidate timings remain in `results.json`.
The dense adapter remains the original batch-vmap call.

Effective throughput counts both attention matrix multiplications. Dense attention counts `4 * B * H * S² * D` operations. Causal attention counts only valid lower-triangular query/key pairs.
