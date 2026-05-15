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

### 1. Produce the LLO dump (on the TPU pod)

LLO (Low-Level Optimizer) dumps are emitted by libtpu when `--xla_jf_dump_to=<dir>` is set in `LIBTPU_INIT_ARGS`. They contain the final VLIW bundles (`*-final_bundles.txt`) and per-bundle lane utilization (`*-final_hlo-static-per-bundle-utilization.txt`) — the two files this predictor consumes.

**Pod setup** (canonical envs per `.claude/skills/tpu-test/SKILL.md`):

| Env | Root | Venv | TPU chip | Required env vars |
|---|---|---|---|---|
| 1 | `/mnt/hyperdisk/` | `/mnt/hyperdisk/env/.venv` | 0 | `ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_VISIBLE_CHIPS=0` |
| 2 | `/mnt/hyperdisk/Code2/` | `/mnt/hyperdisk/Code2/env/.venv` | 2 | `... TPU_VISIBLE_CHIPS=2` |
| 3 | `/mnt/hyperdisk/Code3/` | `/mnt/hyperdisk/Code3/env/.venv` | 3 | `... TPU_VISIBLE_CHIPS=3` |
| 4 | `/mnt/hyperdisk/Code4/` | `/mnt/hyperdisk/Code4/env/.venv` | 1 | `... TPU_VISIBLE_CHIPS=1` |

Pick the env that matches the current session (deterministic from session name) — or ask the user.

**Sync your code to the pod first:**
```bash
kubectl cp scripts/<your-runner>.py yifeixu-torchtpu:/mnt/hyperdisk/Code3/helion/scripts/<your-runner>.py
```

**Helion-on-Pallas kernel:**
```bash
kubectl exec yifeixu-torchtpu -- bash -c "
  source /mnt/hyperdisk/Code3/env/.venv/bin/activate &&
  rm -rf /tmp/llo_dump && mkdir -p /tmp/llo_dump &&
  cd /mnt/hyperdisk/Code3/helion &&
  ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_VISIBLE_CHIPS=3 \
  HELION_BACKEND=pallas \
  LIBTPU_INIT_ARGS='--xla_jf_dump_to=/tmp/llo_dump' \
  python <your-runner>.py
"
```

**msl-tpu-kernel kernel** (uses `pl.pallas_call` directly):
```bash
# msl-tpu-kernel is rsync'd to /mnt/hyperdisk/Code3/msl-tpu-kernel and added via PYTHONPATH
kubectl exec yifeixu-torchtpu -- bash -c "
  source /mnt/hyperdisk/Code3/env/.venv/bin/activate &&
  rm -rf /tmp/llo_dump && mkdir -p /tmp/llo_dump &&
  cd /mnt/hyperdisk/Code3/helion &&
  PYTHONPATH=/mnt/hyperdisk/Code3/msl-tpu-kernel:\$PYTHONPATH \
  ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_VISIBLE_CHIPS=3 \
  LIBTPU_INIT_ARGS='--xla_jf_dump_to=/tmp/llo_dump' \
  python scripts/llo_runner_gmm.py --M 1024 --K 4096 --N 4096 --G 8
"
```

The runner should:
1. Build inputs of the exact shape/dtype you want to study.
2. Call the kernel once outside any timing loop to trigger compilation (LLO is emitted during compile).
3. Call it again, timed, for the measurement. The runner should print `Avg latency: <X> us`.

Use `scripts/llo_runner_gmm.py` / `scripts/llo_runner_rpa.py` as templates — single-config runners with full LIBTPU recipe in their docstrings.

**Heads-up**: a single JIT'd function call emits **many** dumps (per-fusion, plus XLA reference dumps, plus utility ops). Step 2 below shows how to pick out *your* kernel from the noise.

### 2. Identify the right dump

```bash
kubectl exec yifeixu-torchtpu -- ls -laS /tmp/llo_dump/ | grep final_bundles.txt | head
```

- **Helion-on-Pallas**: filename pattern is `custom_kernel.<N>-final_bundles.txt`. The **largest** dump is usually the XLA reference (jnp.matmul / jnp.exp baselines), NOT your kernel. Verify the kernel match by grepping for `vmatmul` (matmul-shaped) or `vexp` (softmax-shaped) content.
- **msl-tpu-kernel**: names are descriptive (`gmm_v2-g_8-m_1024-k_4096-n_4096-tm_128-tk_4096-tn_2048.1-71-final_bundles.txt`, `RPA-bq_32-bkvp_32-p_64-1.1-71-final_bundles.txt`). The numerical suffix encodes block sizes, useful for cross-checking.
- Always also grab the matching `*-final_hlo-static-per-bundle-utilization.txt` — it has the per-bundle lane breakdown the predictor needs.

The `*-schedule-analysis_final_bundles.txt` is a separate analysis output; ignore it (you want the plain `*-final_bundles.txt`).

### 3. Archive to the LLO database

```bash
ENTRY=~/Code/helion/llo/<descriptive_name>_<shape>_bf16
mkdir -p $ENTRY/source

# Copy both LLO files from the pod (use the exact filenames from step 2)
kubectl cp yifeixu-torchtpu:/tmp/llo_dump/<kernel>-...final_bundles.txt $ENTRY/final_bundles.txt
util=$(kubectl exec yifeixu-torchtpu -- bash -c "ls /tmp/llo_dump/<kernel>-...utilization.txt | head -1")
kubectl cp yifeixu-torchtpu:$util $ENTRY/utilization.txt

# Pair with Pallas source so the entry is a "source ↔ LLO ↔ measured-perf" triangle
cp <kernel-source>.py $ENTRY/source/
cp <your-runner>.py   $ENTRY/source/
```

For Helion kernels, get the generated Pallas (post-Helion, pure JAX/Pallas, directly runnable) via:
```bash
python scripts/extract_helion_pallas.py \
    --example examples/<X>.py --kernel <X> \
    --shapes '8x32x8192x256;8x32x8192x256;8x32x8192x256' --dtype bf16 \
    --config '{"block_sizes": [8, 512, 512], "pallas_loop_type": "emit_pipeline", "pallas_pre_broadcast": true}' \
    --out $ENTRY/source/generated_pallas.py
```

For msl-tpu-kernel kernels, the source IS already Pallas — copy the kernel file directly (e.g. `~/Code/msl-tpu-kernel/msl_tpu_kernel/kernels/megablox/gmm_v2.py`).

Optionally write a `meta.yaml` to record the measured timing, config, and any notes — see existing entries under `~/Code/helion/llo/` for the schema.

### 4. Predict

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

## Common gotchas

- **Helion kernel naming**: `custom_kernel.<N>` is your kernel; the *largest* dump in the directory is usually the XLA baseline reference. Grep for `vmatmul`/`vexp` content to confirm.
- **emit_pipeline `--inner-loop-iters`**: even with static shapes, `pallas_loop_type=emit_pipeline` hides the inner K-loop inside the body. For attention with `block_n < S`, K = `S/block_n` (e.g. 16 for S=8192, block_n=512).
- **bytes accounting for GMM at large M**: `--inputs` undercounts when RHS is re-read across N-tiles. Open TODO (see predictor docstring).
- **bmm-style phi stalls**: ~10-15% of compute time goes to loop-carried-dep stalls the model doesn't capture. Open TODO; not worth fixing unless on critical path.

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
