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
