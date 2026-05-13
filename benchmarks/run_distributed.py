from __future__ import annotations

from benchmarks.distributed import AllReduceBench as AllReduceBenchmark
import torch.distributed as dist


def main() -> None:
    bench = AllReduceBenchmark()
    bench.run()
    bench.print_results(metric="time_us")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
