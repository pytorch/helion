# MXFP8 latency by shape

`repro_mxfp8_latency_by_shape.py` launches the packed MXFP8 GEMM, checks it
against the FP64 reference, times it, and writes the latency plot. Helion is
reinstalled at each commit before that commit's kernels run:

- `8781adf88` parent Helion-Triton
- `71a07049f` packed descriptor lowering
- `9192f5909` split-K and swapped operands, and Helion-CuTe

```bash
python benchmarks/cute/repro_mxfp8_latency_by_shape.py            # seed configs
python benchmarks/cute/repro_mxfp8_latency_by_shape.py --tune quick  # + autotuned Triton
```

Triton runs each commit's `autotune_effort="none"` seed. `--tune quick` (or
`full`) adds a `Tuned Helion-Triton` series that runs Helion's autotuner at the
final commit; the seed is in the search space. CuTe uses the saved SM103
configs. Latencies are CUPTI kernel times from a CUDA graph with a cold L2
cache: 10 s of thermal warmup, then the median of five runs.

Outputs in `artifacts/mxfp8_dense_gemm_repro/`:

- `mxfp8_latency_by_shape.csv` (`series,m,n,k,us`)
- `mxfp8_latency_by_shape.png`, with a geomean of every shape beside the per-shape bars

`plot_mxfp8_latency_by_shape.py --csv <file>` redraws the PNG from any such CSV.
It splits shapes into memory-bound and compute-bound by ideal arithmetic
intensity against the B300 ridge point. No latency is hardcoded.

Requirements: an SM100/SM103 GPU, Helion's runtime dependencies,
`matplotlib`, and `flashinfer` with `cupti-python`.
