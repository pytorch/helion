## Benchmarking

Performance comparison between Helion, torch.compile, Triton, and PyTorch eager is done by leveraging [TritonBench](https://github.com/pytorch-labs/tritonbench).

Currently supported kernels for performance comparison are in `benchmark/`.

To run the benchmark:

`$ python benchmark/run.py --metrics speedup,accuracy --kernel <kernel_name>`

e.g. for `vector_add` kernel:

`$ python benchmark/run.py --metrics speedup,accuracy --kernel vector_add`
