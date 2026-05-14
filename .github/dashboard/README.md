# Helion Benchmark Dashboard

Interactive dashboard for visualizing Helion benchmark results across GPU platforms (H100, B200, MI325X, MI350X). Tracks kernel latency, speedup vs eager, compile time, and accuracy over time.

Deployed at https://helionlang.com/dashboard/ alongside the main docs. The docs-deploy workflow runs `build_dashboard_data.py` after each docs push or Benchmark Dispatch completion, generating `dashboard-data.json` that `index.html` fetches at runtime.

## Local development

```bash
# Generate data (requires gh CLI authenticated with pytorch/helion access)
python .github/dashboard/build_dashboard_data.py \
    --repo pytorch/helion \
    --existing-url https://helionlang.com/dashboard/dashboard-data.json \
    --output .github/dashboard/dashboard-data.json

# Serve
cd .github/dashboard
python -m http.server 8899
# Open http://localhost:8899/index.html
```
