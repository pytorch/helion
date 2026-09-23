## Benchmarking

Performance comparison between Helion, torch.compile, Triton, and PyTorch eager is done by leveraging [TritonBench](https://github.com/meta-pytorch/tritonbench).

Currently supported kernels for performance comparison are listed in `KERNEL_MAPPINGS` in `benchmarks/run.py`.

To run the benchmark:

`$ python benchmarks/run.py --metrics speedup,accuracy --kernel <kernel_name>`

e.g. for `vector_add` kernel:

`$ python benchmarks/run.py --metrics speedup,accuracy --kernel vector_add`

To run against another Helion backend, pass `--helion-backend` or set `HELION_BACKEND`.
For example, to benchmark the CuTe backend:

`$ python benchmarks/run.py --helion-backend cute --metrics speedup,accuracy --kernel gemm`

### CUDA Graph Benchmarking

For more accurate kernel performance measurements, especially during autotuning, you can enable CUDA graph benchmarking:

```bash
export HELION_BENCHMARK_CUDAGRAPH=1
python benchmarks/run.py --metrics speedup --kernel <kernel_name>
```

CUDA graph benchmarking reduces kernel launch overhead and provides timing that better represents deployment scenarios where cuda graph is used.

### Comparing the example catalogue

`benchmarks/cute/compare_example_backends.py` compares the non-distributed
examples at three workload sizes. It uses the existing example callables and
references, Helion's Triton and CuTe backends, ATen, `torch.compile`, and optional
Quack or handwritten baselines where an adapter exists. Shared linear attention
engine kernels are exercised through their public example wrappers.

```bash
python benchmarks/cute/compare_example_backends.py list
python benchmarks/cute/compare_example_backends.py run \
  --gpu 0 --modules int4_gemm,matmul_split_k --output /tmp/example-comparison
python benchmarks/cute/compare_example_backends.py report \
  --input /tmp/example-comparison/results.json --output /tmp/example-report
```

Omit `--modules` for the complete catalogue. `--shapes 0,1,2` is the default;
each module has small, medium and large presets appropriate to its operation.
The run refuses an existing output directory and checks for other GPU processes
before each worker. Use a GPU you have reserved; the prelaunch check is not a
scheduler or protection against another user starting work during a measurement.

Every implementation runs in its own subprocess and private compiler caches,
with the same seeded inputs and GPU. Every run draws a fresh autotuning seed and
assigns distinct recorded worker seeds; use `--seed N` to reproduce a run. The
result records GPU identity/power limit/driver and relevant software versions. Helion uses ordinary FULL autotuning without
a time or generation cap. Optional baseline dependencies are never installed by
the tool: unavailable imports, compilation failures and numerical failures are
saved alongside successful results. Normal calls retain each provider's output
allocation behavior; explicitly named direct/manual adapters use prepared output
buffers. Backward cases prepare the forward graph once and time its gradient
call with identical upstream gradients. Dropout checks distribution and scaling
because exact seeded masks can differ between implementations.

Output structure, tensor shape and dtype must match the reference before numeric
comparison. The linear attention cases explicitly declare which implementations
return input dtype while their recurrent reference returns FP32. GRPO promotes
only reference logits for FP32 log-softmax; both implementations still return
FP32 losses and BF16 input gradients. These exceptions, required output contracts,
and actual/reference output metadata are saved with each result. Numeric error
metrics and tolerances remain the same as in the example comparisons.

Both CUDA-event and CUDA-graph timings use five samples by default, each a
`do_bench(warmup=10, rep=50, return_mode="median")` median. The report uses only
successful CUDA-graph measurements with correctness checks before and after
timing. It selects the fastest valid implementation per backend and the fastest
available external baseline. Speedups are **Triton/CuTe** and
**min(baseline, Triton)/CuTe**; higher is better. The matplotlib PNG/PDF bar chart
shows geometric means over the requested shapes; missing shapes make the
corresponding variant N/A rather than silently improving its mean. Modules whose
case inventory fails also remain visible as N/A, with per-module/shape coverage
in `inventory.csv`. CSV files
retain the shape latencies and selected implementation names.

These are exploratory comparisons. The tool does not reproduce a historical
hillclimb's independent post-search replay, retained historical configurations,
interleaved fixed validation or performance acceptance decisions. Keep generated
results and compiler artifacts outside the submitted source tree.
