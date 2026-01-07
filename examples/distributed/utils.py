"""Utilities for Helion distributed kernels including Triton helpers and benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from multiprocessing import get_context
import os
import pickle
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import TypeVar

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
from triton import runtime
import triton.language as tl

import helion
from helion import CustomBenchmarkResult

if TYPE_CHECKING:
    from types import ModuleType

_T = TypeVar("_T")


def _load_module_from_path(name: str, path: str) -> ModuleType:
    """Load a module from a file path and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _get_tid():  # noqa: ANN202
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_ntid():  # noqa: ANN202
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_flat_tid():  # noqa: ANN202
    tid_x, tid_y, tid_z = _get_tid()
    ntid_x, ntid_y, _ = _get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def _get_flat_bid():  # noqa: ANN202
    return (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )


@triton.jit
def _send_signal(addrs, sem: tl.constexpr) -> None:  # noqa: ANN001
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.{sem}.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=addrs.dtype,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal(addrs, sem: tl.constexpr) -> None:  # noqa: ANN001
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.{sem}.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def symm_mem_sync(
    signal_pad_ptrs,  # noqa: ANN001
    block_id,  # noqa: ANN001
    rank: tl.constexpr,
    world_size: tl.constexpr,
    hasPreviousMemAccess: tl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    hasSubsequentMemAccess: tl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> None:
    """
    Synchronizes blocks with matching block_id across participating devices.

    Note: the function itself is not a system level barrier/fence. It is a
    building block for expressing different synchronization patterns.

    Pattern 0: Ensures that all writes to symm_mem buffers from previous
    kernels across all devices are visible to the current kernel:

        symm_mem_sync(..., hasPreviousMemAccess=False, hasSubsequentMemAccess=True)

    Pattern 1: Ensures that all writes to symm_mem buffers from the current
    block are visible to all remote blocks with matching blockIdx:

        symm_mem_sync(..., hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

    Pattern 2: Ensures that symm_mem buffers read by the current kernel are safe
    for writing by subsequent kernels across all devices.

        symm_mem_sync(..., hasPreviousMemAccess=True, hasSubsequentMemAccess=False)

    CUDA graph friendliness:

        This barrier operates through atomic operations on a zero-filled signal
        pad, which resets to a zero-filled state after each successful
        synchronization. This design eliminates the need for incrementing a
        flag from host.
    """
    if block_id is None:
        block_id = _get_flat_bid()
    flat_tid = _get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if hasPreviousMemAccess:
        tl.debug_barrier()

    if flat_tid < world_size:
        _send_signal(send_addrs, "release" if hasPreviousMemAccess else "relaxed")
        _wait_signal(wait_addrs, "acquire" if hasSubsequentMemAccess else "relaxed")

    if hasSubsequentMemAccess:
        tl.debug_barrier()


# =============================================================================
# Distributed Benchmarking Utilities
# =============================================================================


def do_bench_fixed_iters(
    fn: Callable[[], object],
    *,
    warmup: int = 3,
    rep: int = 10,
    return_mode: Literal["min", "max", "mean", "median"] = "median",
) -> float:
    """
    Benchmark a function with fixed iteration counts.

    Unlike triton's do_bench which uses time-based warmup/rep (in milliseconds),
    this version uses fixed iteration counts. This is useful when you need
    deterministic iteration counts, such as for distributed kernels where all
    ranks must execute the same number of iterations to stay synchronized.

    Args:
        fn: The function to benchmark (should take no arguments).
        warmup: Number of warmup iterations (default: 3).
        rep: Number of benchmark iterations (default: 10).
        return_mode: How to aggregate timing results - "min", "max", "mean", or "median".

    Returns:
        The benchmark timing in milliseconds.
    """
    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]

    # Warmup
    for _ in range(warmup):
        fn()
    di.synchronize()

    # Benchmark with individual timing per iteration
    timings: list[float] = []
    for _ in range(rep):
        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        di.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    agg_fn = {
        "min": min,
        "max": max,
        "mean": statistics.mean,
        "median": statistics.median,
    }[return_mode]
    return agg_fn(timings)


@dataclass
class DistributedBenchmarkConfig:
    """Serializable config passed to worker subprocess."""

    module_path: str
    kernel_name: str
    config_dict: dict[str, Any]
    rank: int
    world_size: int
    tmpdir: str
    seed: int
    inputs_fn_name: str
    inputs_fn_module_path: str
    config_index: int = 0  # Used to create per-config FileStore paths
    repeat: int = 50


# Global Gloo process group for coordination (lazily initialized)
_gloo_coord_group: dist.ProcessGroup | None = None


def _get_or_create_gloo_group() -> dist.ProcessGroup:
    """Get or create a Gloo process group for CPU-side coordination.

    This is used to coordinate autotuning across ranks.
    Gloo is preferred for CPU-side coordination because it doesn't require GPU synchronization.
    """
    global _gloo_coord_group
    rank = dist.get_rank()
    if _gloo_coord_group is None:
        print(f"[DEBUG _get_or_create_gloo_group] rank={rank} creating Gloo group...")
        # Create a new Gloo group with all ranks
        _gloo_coord_group = dist.new_group(backend="gloo")
        print(f"[DEBUG _get_or_create_gloo_group] rank={rank} Gloo group created")
    else:
        print(f"[DEBUG _get_or_create_gloo_group] rank={rank} reusing existing Gloo group")
    return _gloo_coord_group


def _broadcast_config_list(
    config_dicts: list[dict[str, Any]],
    gloo_group: dist.ProcessGroup,
    rank: int,
) -> list[dict[str, Any]]:
    """Broadcast the list of config dicts from rank 0 to all other ranks.

    Uses Gloo backend for CPU-side coordination. This ensures all ranks
    test the same configs in the same order during autotuning.
    """
    print(f"[DEBUG _broadcast_config_list] rank={rank} starting, num_configs={len(config_dicts)}")

    # Serialize config list on rank 0
    if rank == 0:
        config_bytes = pickle.dumps(config_dicts)
        size_tensor = torch.tensor([len(config_bytes)], dtype=torch.int64)
        print(f"[DEBUG _broadcast_config_list] rank={rank} serialized {len(config_bytes)} bytes")
    else:
        size_tensor = torch.tensor([0], dtype=torch.int64)

    # Broadcast size first
    print(f"[DEBUG _broadcast_config_list] rank={rank} broadcasting size...")
    dist.broadcast(size_tensor, src=0, group=gloo_group)
    size = int(size_tensor.item())
    print(f"[DEBUG _broadcast_config_list] rank={rank} size={size}")

    # Broadcast the actual config data
    if rank == 0:
        data_tensor = torch.frombuffer(bytearray(config_bytes), dtype=torch.uint8).clone()
    else:
        data_tensor = torch.zeros(size, dtype=torch.uint8)

    print(f"[DEBUG _broadcast_config_list] rank={rank} broadcasting config data...")
    dist.broadcast(data_tensor, src=0, group=gloo_group)
    print(f"[DEBUG _broadcast_config_list] rank={rank} broadcast done")

    # Deserialize on non-rank-0
    if rank != 0:
        config_dicts = pickle.loads(data_tensor.numpy().tobytes())
        print(f"[DEBUG _broadcast_config_list] rank={rank} deserialized {len(config_dicts)} configs")

    print(f"[DEBUG _broadcast_config_list] rank={rank} returning {len(config_dicts)} configs")
    return config_dicts


def _distributed_benchmark_worker(config: DistributedBenchmarkConfig) -> None:
    """Worker that runs in subprocess to benchmark a kernel config."""
    status = 1
    timing = float("inf")
    output_cpu = None
    # Per-config store path prevents stale FileStore data from killed workers
    file_store_path = os.path.join(config.tmpdir, f"store_{config.config_index}")

    try:
        torch.cuda.set_device(config.rank)
        store = dist.FileStore(file_store_path, config.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=config.world_size,
            rank=config.rank,
            store=store,
        )
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        assert group is not None
        symm_mem.enable_symm_mem_for_group(group.group_name)

        module = _load_module_from_path("kernel_module", config.module_path)
        torch.manual_seed(config.seed + config.rank)
        inputs_module = _load_module_from_path(
            "inputs_module", config.inputs_fn_module_path
        )
        inputs_fn = getattr(inputs_module, config.inputs_fn_name)
        args = inputs_fn()

        kernel = getattr(module, config.kernel_name)
        helion_config = helion.Config(**config.config_dict)
        bound_kernel = kernel.bind(args)
        compiled_fn = bound_kernel.compile_config(helion_config)

        # Run kernel once to capture output for accuracy checking.
        dist.barrier()
        output = compiled_fn(*args)
        torch.cuda.synchronize()
        # Move output to CPU for pickling (GPU tensors can't be pickled with spawn)
        output_cpu = _move_tensors_to_device(output, torch.device("cpu"))
        dist.barrier()

        # Use do_bench_fixed_iters with fixed iteration counts.
        # This is essential for distributed kernels where all ranks must execute
        # the same number of iterations to stay synchronized during collectives.
        # Uses 1 warmup + repeat benchmarks to match interleaved_bench behavior.
        dist.barrier()
        timing = do_bench_fixed_iters(
            lambda: compiled_fn(*args),
            warmup=1,
            rep=config.repeat,
            return_mode="median",
        )
        dist.barrier()

        dist.destroy_process_group()
        status = 0
    except Exception:
        traceback.print_exc()
    finally:
        # Each worker writes its own result to a rank-specific file
        # This is needed for operations like reduce-scatter where each rank produces different output
        rank_result_path = os.path.join(config.tmpdir, f"result_rank{config.rank}.pkl")
        with open(rank_result_path, "wb") as f:
            pickle.dump({"status": status, "timing": timing, "output": output_cpu}, f)
        os._exit(status)


def _move_tensors_to_device(obj: _T, device: torch.device) -> _T:
    """Recursively move all tensors in a pytree structure to the specified device.

    This is needed because benchmark outputs are serialized/deserialized during
    broadcast, which puts them on CPU. We need to move them back to the correct
    GPU device for accuracy checking against the baseline.
    """
    from torch.utils._pytree import tree_map_only

    return tree_map_only(torch.Tensor, lambda t: t.to(device), obj)


def _broadcast_results(
    results: list[CustomBenchmarkResult],
    gloo_group: dist.ProcessGroup,
    rank: int,
    device: torch.device,
) -> list[CustomBenchmarkResult]:
    """Broadcast benchmark results from rank 0 to all other ranks.

    Args:
        results: List of results (only valid on rank 0)
        gloo_group: Gloo process group for coordination
        rank: Current rank
        device: Device to move output tensors to

    Returns:
        List of CustomBenchmarkResult on all ranks
    """
    print(f"[DEBUG _broadcast_results] rank={rank} starting, num_results={len(results)}")

    # Serialize results on rank 0 (only timings, outputs stay on each rank's device)
    if rank == 0:
        timings = [r.timing for r in results]
        timings_bytes = pickle.dumps(timings)
        size_tensor = torch.tensor([len(timings_bytes)], dtype=torch.int64)
        print(f"[DEBUG _broadcast_results] rank={rank} timings={timings}, serialized {len(timings_bytes)} bytes")
    else:
        size_tensor = torch.tensor([0], dtype=torch.int64)

    print(f"[DEBUG _broadcast_results] rank={rank} broadcasting size...")
    dist.broadcast(size_tensor, src=0, group=gloo_group)
    size = int(size_tensor.item())
    print(f"[DEBUG _broadcast_results] rank={rank} size={size}")

    if rank == 0:
        data_tensor = torch.frombuffer(bytearray(timings_bytes), dtype=torch.uint8).clone()
    else:
        data_tensor = torch.zeros(size, dtype=torch.uint8)

    print(f"[DEBUG _broadcast_results] rank={rank} broadcasting data...")
    dist.broadcast(data_tensor, src=0, group=gloo_group)
    print(f"[DEBUG _broadcast_results] rank={rank} broadcast done")

    if rank != 0:
        timings = pickle.loads(data_tensor.numpy().tobytes())
        print(f"[DEBUG _broadcast_results] rank={rank} received timings={timings}")
        # Non-rank-0 processes don't have outputs (workers weren't spawned here)
        # Return results with None outputs - accuracy checking should only happen on rank 0
        results = [
            CustomBenchmarkResult(timing=t, output=None)
            for t in timings
        ]

    print(f"[DEBUG _broadcast_results] rank={rank} returning {len(results)} results")
    return results


def distributed_benchmark(
    fns: list[Any],
    *,
    repeat: int = 1,
    desc: str | None = None,
    timeout: float = 30.0,
    inputs_fn: Callable[[], tuple[Any, ...]],
) -> list[float | CustomBenchmarkResult]:
    """Benchmark function for distributed autotuning with rank-0 coordination.

    Rank 0 acts as the coordinator and spawns ALL worker processes (one per GPU/rank).
    Non-rank-0 processes simply wait for results to be broadcast from rank 0.
    This ensures all workers test the same configs and their collectives align.

    Workers coordinate via FileStore + NCCL for GPU collectives.
    Coordination between parent processes uses Gloo.

    Usage:
        from functools import partial

        @helion.kernel(
            autotune_benchmark_fn=partial(distributed_benchmark, inputs_fn=create_inputs),
            autotune_baseline_fn=reference_impl,
        )
        def my_kernel(...): ...

        def create_inputs() -> tuple:
            '''Zero-arg function that returns kernel args.'''
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            device = torch.device(f"cuda:{rank}")
            # Create inputs...
            return (a, b, buffer, rank, world_size, group_name)

    Args:
        fns: List of BenchmarkCallable objects from the autotuner.
        repeat: Number of benchmark iterations (1 warmup + repeat benchmarks).
        desc: Description for progress display (unused).
        timeout: Maximum time in seconds to wait for each config.
        inputs_fn: Zero-arg function that returns kernel args tuple. Called in worker
            subprocess after dist.init_process_group().

    Returns:
        List of CustomBenchmarkResult with timing and output for accuracy checking.
        Returns inf timing for failed/timed-out configs.
    """
    print(f"[DEBUG distributed_benchmark] called with {len(fns)} fns, repeat={repeat}, desc={desc}")

    if not fns:
        print("[DEBUG distributed_benchmark] no fns, returning []")
        return []
    if not dist.is_initialized():
        print("[DEBUG distributed_benchmark] dist not initialized, fallback to non-distributed")
        # Fallback for non-distributed case
        # pyrefly: ignore [bad-return]
        return [fn() if callable(fn) else float("inf") for fn in fns]

    bound_kernel = fns[0].kernel
    module_path = bound_kernel.kernel.fn.__code__.co_filename
    kernel_name = bound_kernel.kernel.fn.__name__

    inputs_fn_name = inputs_fn.__name__
    inputs_fn_module_path = inputs_fn.__code__.co_filename

    seed = torch.initial_seed() % (2**31)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[DEBUG distributed_benchmark] rank={rank}/{world_size} kernel={kernel_name} seed={seed}")

    # Get Gloo group for CPU-side coordination
    gloo_group = _get_or_create_gloo_group()

    # Rank 0 broadcasts the config list to all ranks (so all ranks know how many results to expect)
    config_dicts = [dict(fn.config) for fn in fns]
    print(f"[DEBUG distributed_benchmark] rank={rank} before broadcast, num_configs={len(config_dicts)}")
    config_dicts = _broadcast_config_list(config_dicts, gloo_group, rank)
    print(f"[DEBUG distributed_benchmark] rank={rank} after broadcast, num_configs={len(config_dicts)}")

    if rank == 0:
        # === RANK 0: Coordinator - spawns ALL workers ===
        print(f"[DEBUG distributed_benchmark] rank={rank} COORDINATOR MODE - spawning workers for all ranks")
        tmpdir = tempfile.mkdtemp()
        print(f"[DEBUG distributed_benchmark] rank={rank} tmpdir={tmpdir}")
        results: list[CustomBenchmarkResult] = []
        ctx = get_context("spawn")

        try:
            for config_index, config_dict in enumerate(config_dicts):
                print(f"[DEBUG distributed_benchmark] rank={rank} config {config_index}/{len(config_dicts)}: {config_dict}")
                # Spawn workers for ALL ranks (not just rank 0)
                processes: list[Any] = []
                for worker_rank in range(world_size):
                    bench_config = DistributedBenchmarkConfig(
                        module_path=module_path,
                        kernel_name=kernel_name,
                        config_dict=config_dict,
                        rank=worker_rank,
                        world_size=world_size,
                        tmpdir=tmpdir,
                        seed=seed,
                        inputs_fn_name=inputs_fn_name,
                        inputs_fn_module_path=inputs_fn_module_path,
                        config_index=config_index,
                        repeat=repeat,
                    )

                    print(f"[DEBUG distributed_benchmark] rank={rank} spawning worker for worker_rank={worker_rank}")
                    p = ctx.Process(target=_distributed_benchmark_worker, args=(bench_config,))
                    p.daemon = True
                    p.start()
                    processes.append(p)
                    print(f"[DEBUG distributed_benchmark] rank={rank} worker {worker_rank} spawned, pid={p.pid}")

                # Wait for all workers with timeout
                print(f"[DEBUG distributed_benchmark] rank={rank} waiting for {len(processes)} workers (timeout={timeout}s)")
                start_time = time.time()
                while any(p.is_alive() for p in processes) and time.time() - start_time < timeout:
                    time.sleep(0.1)

                elapsed = time.time() - start_time
                alive_count = sum(1 for p in processes if p.is_alive())
                print(f"[DEBUG distributed_benchmark] rank={rank} wait done after {elapsed:.1f}s, {alive_count} workers still alive")

                # Kill any workers that are still alive
                for i, p in enumerate(processes):
                    if p.is_alive():
                        print(f"[DEBUG distributed_benchmark] rank={rank} killing worker {i} (pid={p.pid})")
                        p.kill()
                        p.join(timeout=5)

                # Read rank 0's result (use rank 0's timing as representative)
                timing = float("inf")
                output: Any = None
                result_path = os.path.join(tmpdir, "result_rank0.pkl")

                if os.path.exists(result_path):
                    try:
                        with open(result_path, "rb") as f:
                            result = pickle.load(f)
                            timing = result.get("timing", float("inf"))
                            output = result.get("output", None)
                        print(f"[DEBUG distributed_benchmark] rank={rank} config {config_index} result: timing={timing}")
                    except Exception as e:
                        print(f"[DEBUG distributed_benchmark] rank={rank} config {config_index} failed to read result: {e}")
                else:
                    print(f"[DEBUG distributed_benchmark] rank={rank} config {config_index} no result file found")

                # Clean up all result files for next config
                for worker_rank in range(world_size):
                    result_file = os.path.join(tmpdir, f"result_rank{worker_rank}.pkl")
                    if os.path.exists(result_file):
                        os.remove(result_file)

                # Ensure timing is inf when output is None (worker failed)
                if output is None:
                    timing = float("inf")

                # Move output to correct device
                output = _move_tensors_to_device(output, device)

                results.append(CustomBenchmarkResult(timing=timing, output=output))

            print(f"[DEBUG distributed_benchmark] rank={rank} all configs done, {len(results)} results")

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # Broadcast results to all other ranks
        print(f"[DEBUG distributed_benchmark] rank={rank} broadcasting {len(results)} results to other ranks")
        results = _broadcast_results(results, gloo_group, rank, device)

    else:
        # === NON-RANK-0: Just wait for results from rank 0 ===
        # We don't spawn any workers - rank 0 spawns workers for all ranks
        print(f"[DEBUG distributed_benchmark] rank={rank} WAITING MODE - waiting for results from rank 0")
        results = _broadcast_results([], gloo_group, rank, device)
        print(f"[DEBUG distributed_benchmark] rank={rank} received {len(results)} results from rank 0")

    # pyrefly: ignore [bad-return]
    return results
