# Linear Attention Examples

Helion implementations of chunked linear attention covering the major linear
attention / state-space model variants. Every example includes correctness
tests (forward, backward, and recurrent step) against both a pure-PyTorch
reference **and** [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(FLA) / [mamba](https://github.com/state-spaces/mamba), plus benchmarks
comparing Helion kernel performance against those libraries.

This work builds on ideas from the
Attention Engine project.

## Dependencies

The examples require the following optional packages for reference comparisons
and benchmarks:

```bash
# flash-linear-attention (FLA) — reference for most variants
pip install fla

# mamba — reference for Mamba-2 SSD only
pip install mamba_ssm
```

If these are not installed, the examples will still run their core correctness
tests against the pure-PyTorch reference, but FLA/mamba comparisons and
benchmarks will be skipped with a warning.

## Variants

| Example | Decay | Correction | FLA baseline |
|---------|-------|------------|--------------|
| `example_simple_gla.py` | scalar | none | `chunk_simple_gla` |
| `example_full_gla.py` | diagonal | none | `chunk_gla` |
| `example_delta_rule.py` | none | rank-1 | `chunk_delta_rule` |
| `example_gated_delta_rule.py` | scalar | rank-1 | `chunk_gated_delta_rule` |
| `example_vanilla_linear_attn.py` | none | none | `chunk_linear_attn` |
| `example_retention.py` | scalar (fixed) | none | `chunk_retention` |
| `example_mamba2_ssd.py` | scalar | none | `mamba_chunk_scan_combined` |
| `example_kda.py` | diagonal | rank-1 | `chunk_kda` |

## Architecture

**`linear_attention_engine.py`** -- All Helion kernels (15 `@helion.kernel()`
functions including fused recurrent step), `ChunkedLinearAttnFn` autograd wrapper,
`LinearAttentionEngine` class, and the `chunked_linear_attn()` / `recurrent_step()`
public entry points.

**`linear_attention_utils.py`** -- Pure-PyTorch chunked reference implementation,
naive recurrent reference, WY decomposition helpers, input generators.

## Running

```bash
# Single example (test + benchmark):
python -m examples.linear.example_simple_gla

# Run all tests via pytest:
pytest test/test_examples.py -k "test_linear" -v

# Run monkey-patch tests (plugs our engine into FLA layers):
pytest test/test_examples.py -k "monkeypatch" -v
```

## Acknowledgements

- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
  (FLA) by Songlin Yang, Yu Zhang *et al.* -- the reference Triton
  implementations of GLA, DeltaNet, retention, and other linear attention
  variants that our examples test against.

- [mamba](https://github.com/state-spaces/mamba) by Albert Gu and
  Tri Dao -- the Mamba-2 SSD Triton kernel used as baseline for the
  Mamba-2 example.

- Attention Engine -- a DSL-based approach to generating chunked linear attention
  kernels that inspired the generalized engine design.

## Raw-input decode programs

`recurrent_decode.py` provides serving-oriented GDN, KDA, Mamba selective state
update, and Retention programs. These accept raw gates and preserve indexed or
separate state buffers; they complement the prepared-input recurrent engine.
The GDN MTP programs are explicit alternatives: `gdn_mtp_fp32` keeps FP32
normalized q/k, while `gdn_mtp_fp16` rounds normalized q/k through FP16 and keeps
all state/checkpoint accumulation in FP32. Neither selects a configuration by
batch size. The benchmark cases choose the FP32 B1/T2 program and FP16 B16/T4
program explicitly.

List the ten raw benchmark cases and their optional installed providers:

```bash
python -m benchmarks.cute.compare_recurrent_decode_backends --list
python -m benchmarks.cute.compare_recurrent_decode_backends \
  --case gdn-decode-b64-h16-hv32-d128-fp32 --implementation helion --effort full
```

Use `--config FILE` only to verify an explicit configuration; normal discovery
uses the ordinary defaults and compiler seeds. The benchmark objective includes
raw gate/normalization work, layout adaptations, output, all required final and
intermediate state writes, and readonly/pool guards. Reset and cache clearing
remain outside the timed call. Optional FlashInfer, FLA, Mamba, vLLM and SGLang
imports occur only for the requested provider; unavailable providers raise their
original import/API error. These measurements use the installed provider version
and do not inherit any historical benchmark qualification.

The Retention adapters retain four distinct boundaries: chunk/recurrent with
fixed-gamma arithmetic timed, and chunk/recurrent with input-independent gamma
prepared once. Chunk state-layout conversions always remain timed. All decode
oracles use the original FP64 recurrence and `atol=0.02`, `rtol=0.01`; the
Retention head-decay coefficient is formed in FP32 first. The two portable
`test_recurrent_decode_*` modules exercise small CPU cases without code emission.

## Fixed-decay Retention prefill

`retention_prefill.py` exposes `retention_prefill_fixed_decay`, a single-call
T128/D128 RetNet program with C64 prefix storage. Its input prefix must follow
the documented fixed per-head FP32 decay recipe; arbitrary learned or
per-token gates are not supported. It reads token-major BF16 q/k/v and a
readonly K-major FP32 initial state, returns a fresh BF16 output, and writes the
supplied FP32 final-state destination. The algorithm uses query tiles of 32;
compiler configurations remain ordinary search choices, without shape dispatch.

```bash
python -m benchmarks.cute.compare_retention_prefill_backends --list
python -m benchmarks.cute.compare_retention_prefill_backends \
  --case retention-prefill-b1-t128-h8-d128-fp32 --effort full \
  --source-workers 8 --native-workers 32 --compile-timeout 120 \
  --output /tmp/retention-prefill-full
```

The benchmark retains the original FP64 output/state oracle (`atol=0.02`,
`rtol=0.01`), readonly storage checks, poison/reset guards, and complete-call
30-sample search / 120-sample final objective. All default/natural seeds remain
available. Worker counts are optional scheduling settings; omitted counts use
serial compilation. `--config FILE --effort none` verifies an explicit config.

Four optional installed FLA routes preserve the original comparison boundaries:
`fla-prepared`, `fla-recurrent-prepared`, `fla-chunk-prepared-gamma`, and
`fla-recurrent-prepared-gamma`. The first two time FP32 fixed-gamma arithmetic;
the latter two prepare it once. All use the original K-major state directly.
Their module hashes are recorded, and these public measurements form a new
epoch; historical paired qualifications are not copied into the results.
