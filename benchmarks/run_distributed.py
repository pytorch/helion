from __future__ import annotations

import sys

from benchmarks.distributed import AGMatmulBench as AGMatmulBench
from benchmarks.distributed import AllReduceBench as AllReduceBench
import torch.distributed as dist

OP_BENCH = {
    "allreduce": AllReduceBench,
    "ag_matmul": AGMatmulBench,
}


def main() -> None:
    try:
        dist.init_process_group("nccl")
    except ValueError:
        print("""
Failed to initialize process group. Are you running with torchrun?
run distributed benchmark with:
torchrun \
--nnodes 1 --nproc-per-node 8 \
--rdzv-backend c10d --rdzv-endpoint localhost:0 \
--no_python python3 \
benchmarks/run_distributed.py <op>
""")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python3 benchmarks/run_distributed.py <op>")
        print(f"Available ops: {OP_BENCH.keys()}")
        sys.exit(1)

    op = sys.argv[1]

    if op not in OP_BENCH:
        print(f"Unknown op: {op}")
        print(f"value ops: {OP_BENCH.keys()}")
        sys.exit(1)

    op_bench = OP_BENCH[op]()
    op_bench.run()
    op_bench.print_results(metric="time_us")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
