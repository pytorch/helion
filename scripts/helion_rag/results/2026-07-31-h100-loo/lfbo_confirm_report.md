# Leave-one-workload-out results

## Diagnostics

- Held-out shape leakage (per-cell proxy): 0
- Mean same-kernel neighbour rate: 1.0
- Tier-1 coverage: 1.0
- Margins: {'perf_lfbo': 1.02, 'time': 1.05, 'cv_med': 0.0083, 'n_workloads': 61}

## rag_lfbo vs lfbo

- Verdict: **non_inferior_only**
- perf ratio: 0.9923 [0.9635, 1.0131] (n=44) (delta_perf=1.02)
- autotune-time ratio: 0.8948 [0.8094, 1.0009] (n=44)
- end-to-end ratio: 0.9066 [0.8208, 1.0105] (n=44)
- completion: 97.8% (McNemar p=0.750, passes=True)
- completion table: {'both_complete': 43, 'baseline_complete_candidate_failed': 1, 'baseline_failed_candidate_complete': 1, 'both_failed': 0}
- perf tipping points: {'1.25': 0.9973963034010809, '1.5': 1.0015377607765843, '2.0': 1.0081074977462317}
