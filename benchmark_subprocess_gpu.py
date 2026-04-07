"""Benchmark subprocess spawn overhead and GPU tensor creation in child process."""
from __future__ import annotations

import multiprocessing as mp
import time


def child_create_tensor(pipe_conn, size: int) -> None:
    """Child process: time GPU tensor creation and send timings back."""
    t0 = time.perf_counter()
    import torch

    t1 = time.perf_counter()
    x = torch.zeros(size, device="cuda")
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    # A second tensor creation to see amortized cost
    y = torch.zeros(size, device="cuda")
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    pipe_conn.send(
        {
            "import_torch": t1 - t0,
            "first_tensor": t2 - t1,
            "second_tensor": t3 - t2,
        }
    )
    pipe_conn.close()
    del x, y


def child_noop(pipe_conn) -> None:
    """Child process: do nothing, just send back a timestamp."""
    pipe_conn.send({"arrived": time.perf_counter()})
    pipe_conn.close()


def bench_spawn_overhead(n_trials: int = 5) -> None:
    ctx = mp.get_context("spawn")
    times = []
    for i in range(n_trials):
        parent_conn, child_conn = ctx.Pipe()
        t0 = time.perf_counter()
        p = ctx.Process(target=child_noop, args=(child_conn,))
        p.start()
        msg = parent_conn.recv()
        p.join()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"  Trial {i}: spawn + join (noop child) = {elapsed*1000:.1f} ms")
    avg = sum(times) / len(times)
    print(f"  Average: {avg*1000:.1f} ms\n")


def bench_gpu_tensor_in_child(n_trials: int = 3, size: int = 1024 * 1024) -> None:
    ctx = mp.get_context("spawn")
    for i in range(n_trials):
        parent_conn, child_conn = ctx.Pipe()
        t0 = time.perf_counter()
        p = ctx.Process(target=child_create_tensor, args=(child_conn, size))
        p.start()
        child_timings = parent_conn.recv()
        p.join()
        t1 = time.perf_counter()
        total = t1 - t0
        print(f"  Trial {i}: total = {total*1000:.1f} ms")
        print(f"    import torch       = {child_timings['import_torch']*1000:.1f} ms")
        print(f"    1st tensor (cuda)  = {child_timings['first_tensor']*1000:.1f} ms")
        print(f"    2nd tensor (cuda)  = {child_timings['second_tensor']*1000:.1f} ms")
    print()


def bench_gpu_tensor_in_parent() -> None:
    """Baseline: tensor creation cost in the current (parent) process."""
    import torch

    # Warm up CUDA context
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    sizes = [1024, 1024 * 1024, 16 * 1024 * 1024]
    for size in sizes:
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            x = torch.zeros(size, device="cuda")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            del x
        avg = sum(times) / len(times)
        print(f"  Parent process tensor({size:>12,}) = {avg*1e6:.1f} us")
    print()


def main():
    print("=" * 60)
    print("1. Baseline: GPU tensor creation in parent process")
    print("=" * 60)
    bench_gpu_tensor_in_parent()

    print("=" * 60)
    print("2. Spawn overhead (noop child, no GPU)")
    print("=" * 60)
    bench_spawn_overhead()

    print("=" * 60)
    print("3. Spawn + GPU tensor creation in child")
    print("=" * 60)
    bench_gpu_tensor_in_child()


if __name__ == "__main__":
    main()
