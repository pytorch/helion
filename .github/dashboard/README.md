# Helion Benchmark Dashboard

Interactive dashboard for visualizing Helion benchmark results across GPU platforms (H100, B200, MI325X, MI350X). Tracks kernel latency, speedup vs eager, compile time, and accuracy over time.

## Generating dashboard data

```bash
python .github/dashboard/process_for_dashboard.py \
    --cache-dir <path-to-benchmark-cache> \
    --runs-meta <path-to-runs-meta.json> \
    --output .github/dashboard/dashboard-data.json
```

- `--cache-dir`: directory containing per-run subdirectories with `helionbench.json` files
- `--runs-meta`: JSON file with run metadata (run_id, sha, full_sha, date, branch)
- `--existing-data`: (optional) previous `dashboard-data.json` for incremental updates

## Viewing locally

```bash
cd .github/dashboard
python -m http.server 8899
# Open http://localhost:8899/index.html
```
