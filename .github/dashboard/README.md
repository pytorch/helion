# Helion Benchmark Dashboard

Interactive dashboard for visualizing Helion benchmark results across GPU platforms (H100, B200, B200 CuTe, MI325X, MI350X). Tracks kernel latency, speedup vs eager, compile time, and accuracy over time.

Deployed at https://helionlang.com/dashboard/ alongside the main docs. The docs-deploy workflow runs `build_dashboard_data.py` after each docs push or Benchmark Dispatch completion, generating `dashboard-data.json` that `index.html` fetches at runtime.

The **Pretuned** tab is fed by a separate pipeline: the nightly **Benchmark Pretuned Dispatch** workflow runs every kernel under `pretuned_kernels/` (via `pretuned_kernels/run.py`) against its checked-in heuristic and uploads aggregate `SUMMARY` metrics. `build_pretuned_dashboard_data.py` aggregates those into `pretuned-dashboard-data.json`, which `index.html` lazily fetches when the Pretuned tab is opened. It tracks geomean speedup vs PyTorch eager, win counts, and best speedup per kernel/platform over time.

## Local development

```bash
# Generate data (requires gh CLI authenticated with pytorch/helion access)
python .github/dashboard/build_dashboard_data.py \
    --repo pytorch/helion \
    --existing-url https://helionlang.com/dashboard/dashboard-data.json \
    --output .github/dashboard/dashboard-data.json

# Pretuned-kernel data (Pretuned tab)
python .github/dashboard/build_pretuned_dashboard_data.py \
    --repo pytorch/helion \
    --existing-url https://helionlang.com/dashboard/pretuned-dashboard-data.json \
    --output .github/dashboard/pretuned-dashboard-data.json

# Serve
cd .github/dashboard
python -m http.server 8899
# Open http://localhost:8899/index.html
```
