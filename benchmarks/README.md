## Benchmarking

Performance comparison between Helion, torch.compile, Triton, and PyTorch eager is done by leveraging [TritonBench](https://github.com/meta-pytorch/tritonbench).

Currently supported kernels for performance comparison are listed in `KERNEL_MAPPINGS` in `benchmarks/run.py`.

To run the benchmark:

`$ python benchmarks/run.py --metrics speedup,accuracy --kernel <kernel_name>`

e.g. for `vector_add` kernel:

`$ python benchmarks/run.py --metrics speedup,accuracy --kernel vector_add`


## Benchmarking for Distributed Kernels

Performance comparison between Helion, torch.dist and Kraken (Triton for Symmetric Memory Kernel) script is available at `benchmarks/run_distributed.py`.

Currently supported kernels for performance comparison are listed in `OP_BENCH` in `benchmarks/run_distributed.py`.

To run benchmark on 1 node and 8 GPUs:

`$ torchrun --nnodes 1 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint localhost:0 --no_python python3 benchmarks/run_distributed.py <op_name> <benchmark_args>`

e.g. for `allreduce` kernel with `bfloat16` dtype:
`$ torchrun --nnodes 1 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint localhost:0 --no_python python3 benchmarks/run_distributed.py allreduce -dtype bfloat16`
