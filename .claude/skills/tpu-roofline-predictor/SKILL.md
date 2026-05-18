---
name: tpu-roofline-predictor
description: Predict TPU v7x kernel runtime from an LLO dump. Reads `*-final_bundles.txt` + `*-final_hlo-static-per-bundle-utilization.txt`, runs a calibrated roofline model, and reports the binding lane / bottleneck. Pair with `hlo_sidecar.py` for kernels with data-dependent trip counts (RPA, splash SWA, emit_pipeline). Use when the user asks to predict, calibrate, or diagnose Pallas kernel performance, or when investigating why a kernel is slow.
---

# TPU Roofline Predictor

## What it does

`scripts/tpu_roofline.py` predicts the per-call runtime of a TPU v7x kernel (Helion-on-Pallas or pure msl-tpu-kernel Pallas) given a static LLO bundle dump and the kernel's HBM byte count. ±6% accuracy on most kernels; explicit fail-loud refusal on the rest.

Calibration constants live at the top of `scripts/tpu_roofline.py` (CLOCK_GHZ=2.2, BF16_EFFECTIVE_TFLOPS=912, HBM_EFFECTIVE_GBPS=3160, per-lane realization factors). Bound to **TPU v7x + the current LLO dump format**. Re-calibration needed for v5p/v6e/etc.

## The end-to-end workflow

Every prediction follows one of two paths depending on whether the LLO encodes the runtime trip count:

```
                    ┌─ trip count in LLO (most kernels)
                    ↓
  produce LLO ──→ predict ──→ done
       │
       │   ┌─ predictor refuses (deep dynamic loops, RPA / splash SWA / emit_pipeline)
       │   ↓
       └→ measure once ──→ hlo_sidecar back-solve --persist
                                    │
                                    ↓
                          predict (auto-loads K from meta.yaml) ──→ done
```

Step-by-step:

### 1–3. Produce, identify, and archive the LLO dump

Steps 1, 2, and 3 are already documented end-to-end in the `tpu-test` skill, which is the canonical reference for anything that runs on the TPU pod (sync, build, env vars, dump dirs, kubectl recipes). See `.claude/skills/tpu-test/SKILL.md`:

- **§ Environment** — Envs 1–4 (root paths, venvs, `TPU_VISIBLE_CHIPS` mapping, activation rule)
- **§ LLO Profiling → Step 1: Dump LLO on the pod** — full `kubectl exec` recipe with `LIBTPU_INIT_ARGS=--xla_jf_dump_to=...`, `HELION_BACKEND=pallas`, etc.
- **§ LLO Profiling → Step 2: Identify the right kernel dump** — naming patterns (Helion `custom_kernel.<N>` vs msl-tpu-kernel descriptive suffixes), how to filter out XLA reference dumps and `schedule-analysis` files
- **§ LLO Profiling → Step 3: Copy dumps back to the Mac**

Two predictor-specific add-ons not in tpu-test:

**Runner templates for msl-tpu-kernel kernels.** Use `scripts/llo_runner_gmm.py` and `scripts/llo_runner_rpa.py` as templates — single-config runners that produce one clean named dump and print `Avg latency: <X> us`. msl-tpu-kernel lives at `/mnt/hyperdisk/Code3/msl-tpu-kernel/` and is symlinked into the other env roots (`/mnt/hyperdisk/msl-tpu-kernel`, `/mnt/hyperdisk/Code{2,4}/msl-tpu-kernel`). In any env, add it to `PYTHONPATH` in the kubectl exec command:
```bash
PYTHONPATH=<env_root>/msl-tpu-kernel:$PYTHONPATH python scripts/llo_runner_<X>.py ...
```

**Pair the dump with Pallas source** when archiving (this is what makes the entry directly useful for kernel-writing, not just historical performance data):

```bash
ENTRY=~/Code/helion/llo/<descriptive_name>_<shape>_bf16
mkdir -p $ENTRY/source

# Helion-generated Pallas (directly runnable with JAX, no Helion at runtime)
python scripts/extract_helion_pallas.py \
    --example examples/<X>.py --kernel <X> \
    --shapes '8x32x8192x256;8x32x8192x256;8x32x8192x256' --dtype bf16 \
    --config '{"block_sizes": [8, 512, 512], "pallas_loop_type": "emit_pipeline", "pallas_pre_broadcast": true}' \
    --out $ENTRY/source/generated_pallas.py

# OR for msl-tpu-kernel kernels (already pure Pallas — copy the kernel file)
cp ~/Code/msl-tpu-kernel/msl_tpu_kernel/kernels/<family>/<kernel>.py $ENTRY/source/
cp <your-runner>.py $ENTRY/source/
```

Optionally write a `meta.yaml` recording the measured timing, config, and notes — see existing entries under `~/Code/helion/llo/` for the schema. `meta.yaml` is also where `hlo_sidecar.py back-solve --persist` writes `inner_loop_iters: K` (picked up automatically by the predictor on the next run).

### 4. Predict against the archived LLO entry

```bash
cd ~/Code/helion
python scripts/tpu_roofline.py $ENTRY \
    --inputs "bf16:1024x4096,bf16:8x4096x4096" \
    --outputs "bf16:1024x4096" \
    --measured-us 102.08
```

`--inputs` / `--outputs` syntax: `dtype:HxWxD,dtype:HxW,...`. Supported dtypes: bf16, f16, f32, fp8/f8, i8, i32, i64, u8.

Three things can happen:
- **Clean prediction**: Error is reported and you're done.
- **`⚠ Loaded inner_loop_iters=K from meta.yaml`**: a previous back-solve persisted K; predictor auto-uses it.
- **REFUSING TO PREDICT banner (exit 2)**: the LLO has deep dynamic loops without static trip annotations. Go to step 5.

### 5. (When refused) Measure → back-solve → persist

The kernel has data-dependent trip counts (RPA's kv_lens, splash SWA's mask iters, emit_pipeline kernels with shape-dependent inner loops). Trip count is NOT in the LLO and can't be derived statically.

```bash
# 5a. You already have a measurement from step 1 (the runner prints it)
# 5b. Back-solve K such that the predictor reproduces the measurement,
#     and persist it to meta.yaml:
python scripts/hlo_sidecar.py back-solve \
    --llo-dir $ENTRY \
    --inputs "..." --outputs "..." \
    --measured-us <X> --persist

# 5c. Now re-run the predictor — it auto-loads K from meta.yaml:
python scripts/tpu_roofline.py $ENTRY \
    --inputs "..." --outputs "..." \
    --measured-us <X>
# → prints: "ℹ Loaded inner_loop_iters=K from meta.yaml"
```

Verified on three prior outliers: rpa_decode/prefill/mixed and gmm_v2_M=4096 all land within ±1.5% after back-solve.

### Optional flags

- `--advise` — counterfactual each calibrated lane at 100% realization, estimate potential savings, emit pattern-based suggestions.
- `--what-if-realization 'XLU=1.0'` — manual counterfactual on one lane.
- `--lower-bound` — bypass the refuse-guard and report the prediction as a strict lower bound (true runtime ≥ this).
- `--inner-loop-iters K` — explicit override (wins over meta.yaml).
- `--dynamic-bundles N` — completely override the parser's bundle count.

## Measurement protocol (calibration assumes this)

All `MIN_TIME_FLOOR_US`, `ADDITIVE_OVERHEAD_US`, and per-lane realization constants in `scripts/tpu_roofline.py` were fit against measurements taken this way:

```python
for _ in range(iters):
    out = fn(*inputs)          # async — futures, no sync
out.block_until_ready()        # single device sync at the very end
elapsed_per_call_us = (perf_counter() - t0) / iters * 1e6
```

This captures **steady-state per-call cost in an async pipeline** — dispatch and DMA setup overlap with previous iterations' execution. It does NOT measure cold per-call latency; per-call sync adds ~10–30 µs of Python+dispatch overhead between iters and would need separate calibration.

Runners in `scripts/llo_runner_*.py` follow this convention. When writing a new runner, match it — otherwise measured vs predicted will drift in a way that looks like a model bug but is actually a measurement-protocol mismatch.

Practical: `iters >= 30` for kernels well above the 30 µs floor; `iters >= 100` for kernels close to the floor (variance dominates).

## Common gotchas

- **Helion kernel naming**: `custom_kernel.<N>` is your kernel; the *largest* dump in the directory is usually the XLA baseline reference. Grep for `vmatmul`/`vexp` content to confirm.
- **emit_pipeline `--inner-loop-iters`**: even with static shapes, `pallas_loop_type=emit_pipeline` hides the inner K-loop inside the body. For attention with `block_n < S`, K = `S/block_n` (e.g. 16 for S=8192, block_n=512).
- **bytes accounting for GMM at large M**: `--inputs` undercounts when RHS is re-read across N-tiles. Open TODO (see predictor docstring).
- **bmm-style phi stalls**: ~10-15% of compute time goes to loop-carried-dep stalls the model doesn't capture. Open TODO; not worth fixing unless on critical path.
- **Raw `pl.pallas_call` runners MUST `jax.jit` the entry point** — `pl.pallas_call(...)` itself doesn't jit; calling the returned function from Python re-traces on every invocation, paying full JAX dispatch overhead (100×–1000× slower for µs-scale kernels). Helion handles this internally; raw msl-tpu-kernel `example-collections/` runners (e.g. maxtext `ragged_mqa`) do NOT. If a measured-vs-predicted gap is implausibly large (~99% off), check whether the runner is jit-wrapped before blaming the predictor. Concrete: `ragged_mqa` B=32 S=4096 D=128 measured 153 ms un-jitted, 268 µs jitted (570× difference, same LLO).

## Tooling

| Script | Purpose |
|---|---|
| `scripts/tpu_roofline.py` | The predictor itself |
| `scripts/llo_parse.py` | LLO dump parser (fork of msl-tpu-kernel's `llo_tool.py`) |
| `scripts/hlo_sidecar.py` | Trace + back-solve K for dynamic kernels |
| `scripts/extract_helion_pallas.py` | Emit the post-Helion Pallas source for a Helion kernel + config |
| `scripts/llo_runner_gmm.py` | Single-config GMM v2 runner |
| `scripts/llo_runner_rpa.py` | Single-config RPA runner (decode + prefill modes) |

All five scripts have detailed top-of-file docstrings with the exact `LIBTPU_INIT_ARGS=--xla_jf_dump_to=...` invocation and the dump-identification recipe.

## Database

`~/Code/helion/llo/` (gitignored). 20 entries as of 2026-05-15: 14 within ±6% calibration band, 6 outliers all documented in `INDEX.md` with root cause and fix path. Each entry should carry `source/` (the Pallas kernel) so it's a "source ↔ compiled-output ↔ measured-perf" triangle, not just historical data.

See `INDEX.md` for the canonical entry list, calibration anchors, and outlier explanations.
