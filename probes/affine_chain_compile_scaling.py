from __future__ import annotations

import argparse
import time

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, autotune_effort="none")
def affine_chain(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer in hl.tile(n):
        tmp[producer] = x[producer] + 1.0

    for consumer in hl.tile(n):
        out[consumer] = tmp[consumer] * 2.0

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=65_536)
    parser.add_argument("--producer-block", type=int, default=16)
    parser.add_argument("--consumer-block", type=int, default=32)
    parser.add_argument("--worker-multiplier", type=int, default=1)
    parser.add_argument("--dump-triton", action="store_true")
    args = parser.parse_args()

    x = torch.randn(args.size, device="cuda")
    bound = affine_chain.bind((x,))
    config = helion.Config(
        block_sizes=[args.producer_block, args.consumer_block],
        pid_type="persistent_blocked",
        cross_loop_schedule="static_pipeline",
        num_sm_multiplier=args.worker_multiplier,
        num_warps=1,
    )
    bound.config_spec.normalize(config.config)
    begin = time.perf_counter()
    source = bound.to_triton_code(config)
    elapsed = time.perf_counter() - begin
    print(
        "COMPILE",
        {
            "seconds": elapsed,
            "bytes": len(source),
            "lines": len(source.splitlines()),
            "tl_where": source.count("tl.where"),
        },
        flush=True,
    )
    if args.dump_triton:
        print(source)


if __name__ == "__main__":
    main()
