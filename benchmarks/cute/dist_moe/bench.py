"""Compare a separate capacity check with one fused into the Dist-MoE route layout.

From the checkout root, run `PYTHONPATH=. python benchmarks/cute/dist_moe/bench.py`
on one CUDA GPU. Each graph contains one layout call and its check. Timings include
any gaps between graph launches issued from Python.
"""

from __future__ import annotations

import argparse
from itertools import accumulate
from statistics import median
from typing import NamedTuple

from layout_kernel import STATUS_MESSAGE  # pyrefly: ignore [missing-import]
from layout_kernel import layout_chunks  # pyrefly: ignore [missing-import]
import torch

WORLD = 2
CAPACITY = 2432
EXPERT_ROWS = 2048
ATOM = 128


def inputs(fixture: str) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.full((32, 8), 16, dtype=torch.int32)
    if fixture == "concentrated":
        counts[:, ::2] = 0
        counts[:, 1::2] = 32
    elif fixture == "changed":
        # Move routes between ranks so their padded receive totals differ.
        delta = torch.arange(32, dtype=torch.int32) % 5 + 1
        counts[:, :2] += delta[:, None]
        counts[:, 7] -= 2 * delta
    elif fixture == "empty":
        counts.zero_()
    elif fixture != "balanced":
        raise ValueError(f"Unknown fixture: {fixture}")
    return counts, torch.ones(WORLD, dtype=torch.bool)


def reference(
    counts: torch.Tensor, valid: torch.Tensor, rank: int
) -> tuple[torch.Tensor, ...]:
    """Build the expected CPU layout, enumerating each physical atom in order."""
    chunks = counts.reshape(len(counts), WORLD, -1)
    local = chunks[:, rank]
    prior = torch.zeros_like(local)
    for i in range(1, len(local)):
        prior[i] = prior[i - 1] + local[i - 1]

    totals = chunks.sum(0, dtype=torch.int32)
    padded = ((totals + ATOM - 1) // ATOM) * ATOM
    sizes = padded[rank]
    edges = torch.tensor([0, *accumulate(sizes.tolist())], dtype=torch.int64)
    recv = padded.sum(1, dtype=torch.int32)
    ok = bool(
        valid.all() and recv[rank] <= CAPACITY and totals[rank].max() <= EXPERT_ROWS
    )
    atoms = torch.full((CAPACITY // ATOM,), -1, dtype=torch.int32)
    if ok:
        rows = [
            expert * EXPERT_ROWS + offset
            for expert, size in enumerate(sizes.tolist())
            for offset in range(0, size, ATOM)
        ]
        atoms[: len(rows)] = torch.tensor(rows, dtype=atoms.dtype)
    return prior, sizes, edges, recv, torch.tensor([ok]), atoms


class Program(NamedTuple):
    graph: torch.cuda.CUDAGraph
    outputs: tuple[torch.Tensor, ...]


def capture(
    counts: torch.Tensor, valid: torch.Tensor, rank: int, *, fused: bool
) -> Program:
    atoms = counts.new_empty(CAPACITY // ATOM)

    def forward() -> tuple[torch.Tensor, ...]:
        args = (counts, valid, rank, CAPACITY, EXPERT_ROWS, atoms)
        outputs = layout_chunks(*args, True) if fused else layout_chunks(*args)
        if not fused:
            torch._assert_async(outputs[-1], STATUS_MESSAGE)
        return (*outputs, atoms)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = forward()
    return Program(graph, outputs)


def time_pair(
    separate: Program, fused: Program, windows: int, replays: int
) -> tuple[float, float, float]:
    samples: tuple[list[float], list[float]] = ([], [])
    programs = (separate, fused)
    for program in programs:
        for _ in range(20):
            program.graph.replay()
    torch.cuda.synchronize()
    for window in range(windows):
        for arm in (0, 1) if window % 2 == 0 else (1, 0):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                programs[arm].graph.replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000 / replays)
    separate_us, fused_us = samples
    delta = median(b - a for a, b in zip(separate_us, fused_us, strict=True))
    return median(separate_us), median(fused_us), delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="Verify without timing"
    )
    parser.add_argument("--windows", type=int, default=24, help="Paired timing windows")
    parser.add_argument(
        "--replays", type=int, default=100, help="Graph replays per window and arm"
    )
    args = parser.parse_args()
    if args.windows < 2 or args.windows % 2 or args.replays < 1:
        parser.error("Use an even number of windows >= 2 and at least one replay.")

    counts, valid = (t.cuda() for t in inputs("balanced"))
    ranks = [
        (
            capture(counts, valid, rank, fused=False),
            capture(counts, valid, rank, fused=True),
        )
        for rank in range(WORLD)
    ]
    for fixture in ("balanced", "changed", "concentrated", "empty"):
        cpu_counts, cpu_valid = inputs(fixture)
        counts.copy_(cpu_counts)
        valid.copy_(cpu_valid)
        for rank, programs in enumerate(ranks):
            expected = reference(cpu_counts, cpu_valid, rank)
            for program in programs:
                for out in program.outputs:
                    out.fill_(False if out.dtype == torch.bool else -2)
                program.graph.replay()
                for actual, want in zip(program.outputs, expected, strict=True):
                    torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
    print("Both arms matched the CPU reference on both ranks and all four fixtures")
    if args.check_only:
        return

    for rank, (separate, fused) in enumerate(ranks):
        for fixture in ("balanced", "concentrated"):
            cpu_counts, cpu_valid = inputs(fixture)
            counts.copy_(cpu_counts)
            valid.copy_(cpu_valid)
            a, b, delta = time_pair(separate, fused, args.windows, args.replays)
            print(
                f"rank={rank} {fixture}: separate={a:.3f} us, fused={b:.3f} us; "
                f"median paired fused - separate={delta:+.3f} us (negative favors fused)"
            )


if __name__ == "__main__":
    main()
