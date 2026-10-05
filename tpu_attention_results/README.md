# Helion TPU attention results

This directory is a reproducible snapshot of BF16 attention performance on one TPU v7 device. It is intentionally isolated from Helion's production package and is carried by a **not-for-commit** branch commit.

The comparison covers:

- dense attention with shape `[8, 32, S, D]`;
- causal attention with shape `[1, 8, S, D]`;
- `S` from 1K through 64K;
- head dimensions 64, 128, and 256;
- the Helion Pallas backend and Tokamax SplashAttention.

The recorded run uses JAX 0.11.2, a PyTorch 2.15 nightly, and Tokamax commit
`980b70443409f0aa5cad0995a065f5289f5d0f8c`. The Helion sources are the parent
of this result-only commit.

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
  --causal-calls 10 \
  --output-dir /tmp/tpu-attention-smoke
```

## Measurement conventions

Each worker checks an all-ones result and exact repeated-call determinism before timing. Reported latency is the median of 20 samples after three warmups. Dense samples contain one call; causal samples contain 50 calls to make launch-scale measurements stable.

Dense Helion calls use the TorchTPU launcher. Causal Helion calls use the same
generated Pallas program through `kernel.jax_fn`, wrapped in `jax.jit`; Tokamax
also uses JAX. Each worker runs in a fresh process so these runtime interfaces
never initialize PJRT in the same process.

Effective throughput counts both attention matrix multiplications. Dense attention counts `4 * B * H * S² * D` operations. Causal attention counts only valid lower-triangular query/key pairs.
