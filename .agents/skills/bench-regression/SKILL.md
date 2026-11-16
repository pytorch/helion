---
name: bench-regression
description: Investigate a Helion benchmark dashboard regression (helionlang.com/dashboard) — find the cause and classify it. Auto-activate when the user reports a perf drop/spike on a dashboard platform (e.g. b200 cute) around a given date.
---

Goal: explain a dashboard move and sort it into one of four causes — **real regression**, **benchmark method change**, **hardware issue**, or **noise**. Don't blame a commit until the data points at one.

**1. Pull the data, isolate the platform.** Numbers are per-nightly, not live — fetch the JSON, don't WebFetch the page (too big to page through):

```bash
curl -s https://helionlang.com/dashboard/dashboard-data.json -o /tmp/dash.json
```

Each `summary[]` entry has `platform_short` (e.g. `b200_cute`) and a `history[]` of nightlies with `helion_speedup_geomean`, `triton_speedup_geomean`, `torch_compile_speedup_geomean`, `helion_latency_avg_ms`, `sha`. Filter to the platform+kernel and print the series around the date.

For the **Pretuned** tab, use the sibling URL — same structure, different `summary[]` fields (`geomean`, `best_speedup`, `helion_wins`/`total`, `baselines`, `cudagraph`):

```bash
curl -s https://helionlang.com/dashboard/pretuned-dashboard-data.json -o /tmp/pretuned.json
```

**2. Classify** — read the whole series, not one point, and sort the move into one of four causes:
- **Real regression** → Helion genuinely got slower. A code change made the kernel (or the config the autotuner picks) worse. Find the commit (step 3).
- **Benchmark method change** → the kernel is unchanged; how it's *measured* changed. A change to the harness, timer, launcher, autotune settings, input shapes, or baseline flips the reported number without touching real performance. The pre- or post-change value is an artifact — decide which is the honest one.
- **Hardware issue** → the environment changed, not Helion. Runner swap, GPU/driver change, thermal throttling — affects everything running on that machine, not just Helion.
- **Noise** → nothing changed; it's measurement scatter. The move isn't a persistent step, just normal run-to-run variance.

Useful cross-checks when deciding: is the move Helion-only or shared with `triton_*`/`torch_compile_*` (shared → environment/baseline, not Helion)? Does it persist or revert (revert → noise)? Do independent metrics agree — e.g. `helion_latency_avg_ms` and speedup come from *different timers* (speedup uses `--cudagraph`; latency uses `--latency-measure-mode triton_do_bench`, see `benchmark.yml`), so if one moves and the other doesn't, suspect a measurement change, not real perf. Don't reconstruct `baseline = speedup * latency` — they're different modes; the product is meaningless.

**3. Find the commit** (real regression only). Window = between last-good and first-bad nightly `sha`:

```bash
git log --first-parent --pretty="%h %cI %s" <good_sha>..<bad_sha>
```

Match the suspect to the affected slice: a `b200_cute`-only drop comes from a `[cute]` commit, not Pallas/Triton/TPU. Check which kernels the platform runs (`kernels_cute` default in `.github/workflows/benchmark_dispatch.yml`) and whether they keep `--cudagraph` (`remove_flags` in `benchmarks/run.py`) before trusting the cudagraph-timed speedup. Confirm by reading the diff for a plausible mechanism.

**Report:** the cause (one of the four), the first-bad sha + culprit commit if real, and the true speedup.
