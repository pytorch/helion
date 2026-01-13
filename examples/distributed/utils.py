"""Utilities for Helion distributed kernels including Triton helpers and benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from multiprocessing import get_context
from multiprocessing import shared_memory
from multiprocessing.queues import Queue as QueueType
import os
import pickle
import shutil
import statistics
import struct
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


# Module-level persistent pool for reuse across benchmark calls
@dataclass
class _PoolConfig:
    """Configuration for a persistent worker pool."""
    module_path: str
    kernel_name: str
    world_size: int
    seed: int
    inputs_fn_name: str
    inputs_fn_module_path: str


_persistent_pool: PersistentWorkerPool | None = None
_pool_config: _PoolConfig | None = None
# Counter for unique pool generations (ensures unique FileStore/shm names after recovery)
_pool_generation: int = 0


def _get_or_create_pool(
    *,
    module_path: str,
    kernel_name: str,
    world_size: int,
    seed: int,
    inputs_fn_name: str,
    inputs_fn_module_path: str,
    tmpdir: str,
) -> PersistentWorkerPool:
    """Get existing pool if compatible, or create a new one."""
    global _persistent_pool, _pool_config

    new_config = _PoolConfig(
        module_path=module_path,
        kernel_name=kernel_name,
        world_size=world_size,
        seed=seed,
        inputs_fn_name=inputs_fn_name,
        inputs_fn_module_path=inputs_fn_module_path,
    )

    # Check if we can reuse the existing pool
    if _persistent_pool is not None and _pool_config == new_config:
        if _persistent_pool.is_alive():
            return _persistent_pool
        else:
            _persistent_pool.shutdown()

    # Need to create a new pool - shutdown old one first if exists
    if _persistent_pool is not None:
        _persistent_pool.shutdown()

    _persistent_pool = PersistentWorkerPool(
        module_path=module_path,
        kernel_name=kernel_name,
        world_size=world_size,
        seed=seed,
        inputs_fn_name=inputs_fn_name,
        inputs_fn_module_path=inputs_fn_module_path,
        tmpdir=tmpdir,
        generation=_pool_generation,
    )
    _pool_config = new_config
    return _persistent_pool


def shutdown_persistent_pool() -> None:
    """Explicitly shutdown the persistent worker pool.

    After shutdown, the next call to _get_or_create_pool will create a fresh pool
    with a new generation number, ensuring unique FileStore/shm names.
    """
    global _persistent_pool, _pool_config, _pool_generation
    if _persistent_pool is not None:
        _persistent_pool.shutdown()
        _persistent_pool = None
        _pool_config = None
        # Increment generation so next pool gets unique resource names
        _pool_generation += 1


def reset_persistent_worker_state() -> None:
    """Reset all persistent worker state.

    This should be called at the start of tests to ensure clean state.
    """
    global _persistent_pool, _pool_config, _pool_generation
    if _persistent_pool is not None:
        _persistent_pool.shutdown()
        _persistent_pool = None
        _pool_config = None
    _pool_generation = 0


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


@dataclass
class PersistentWorkerInit:
    """Initial configuration for persistent worker."""

    module_path: str
    kernel_name: str
    rank: int
    world_size: int
    seed: int
    inputs_fn_name: str
    inputs_fn_module_path: str
    file_store_path: str
    shm_name: str  # Name of shared memory segment for results


@dataclass
class BenchmarkTask:
    """A single benchmark task sent to persistent worker."""

    config_dict: dict[str, Any]
    repeat: int
    task_id: int


# Sentinel value to signal worker shutdown
_WORKER_SHUTDOWN = None

# Special task_id used to signal worker initialization is complete
_WORKER_INIT_COMPLETE = -1

# Result struct format: status (int), timing (double), output_size (int)
_RESULT_HEADER_FORMAT = "=idI"
_RESULT_HEADER_SIZE = struct.calcsize(_RESULT_HEADER_FORMAT)


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
        # Create a new Gloo group with all ranks
        _gloo_coord_group = dist.new_group(backend="gloo")
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

    # Serialize config list on rank 0
    if rank == 0:
        config_bytes = pickle.dumps(config_dicts)
        size_tensor = torch.tensor([len(config_bytes)], dtype=torch.int64)
    else:
        size_tensor = torch.tensor([0], dtype=torch.int64)

    # Broadcast size first
    dist.broadcast(size_tensor, src=0, group=gloo_group)
    size = int(size_tensor.item())

    # Broadcast the actual config data
    if rank == 0:
        data_tensor = torch.frombuffer(bytearray(config_bytes), dtype=torch.uint8).clone()
    else:
        data_tensor = torch.zeros(size, dtype=torch.uint8)

    dist.broadcast(data_tensor, src=0, group=gloo_group)

    # Deserialize on non-rank-0
    if rank != 0:
        config_dicts = pickle.loads(data_tensor.numpy().tobytes())

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


# =============================================================================
# Persistent Worker Implementation (Fast Path)
# =============================================================================

# Shared memory size for results (128MB to handle larger matmul outputs)
# The all_gather_matmul kernel produces ~54MB outputs (4096 x 6656 x bf16)
_SHM_SIZE = 128 * 1024 * 1024


def _write_result_to_shm(
    shm: shared_memory.SharedMemory,
    status: int,
    timing: float,
    output: Any,
) -> None:
    """Write benchmark result to shared memory."""
    output_bytes = pickle.dumps(output) if output is not None else b""
    header = struct.pack(_RESULT_HEADER_FORMAT, status, timing, len(output_bytes))

    # Write header + output to shared memory
    total_size = _RESULT_HEADER_SIZE + len(output_bytes)
    if total_size > len(shm.buf):
        raise ValueError(
            f"Result too large: {total_size} > {len(shm.buf)}. "
            f"Increase _SHM_SIZE in utils.py"
        )

    shm.buf[:_RESULT_HEADER_SIZE] = header
    if output_bytes:
        shm.buf[_RESULT_HEADER_SIZE : _RESULT_HEADER_SIZE + len(output_bytes)] = (
            output_bytes
        )


def _read_result_from_shm(
    shm: shared_memory.SharedMemory,
) -> tuple[int, float, Any]:
    """Read benchmark result from shared memory."""
    header = bytes(shm.buf[:_RESULT_HEADER_SIZE])
    status, timing, output_size = struct.unpack(_RESULT_HEADER_FORMAT, header)

    if output_size > 0:
        output_bytes = bytes(
            shm.buf[_RESULT_HEADER_SIZE : _RESULT_HEADER_SIZE + output_size]
        )
        output = pickle.loads(output_bytes)
    else:
        output = None

    return status, timing, output


def _persistent_worker_loop(
    init: PersistentWorkerInit,
    task_queue: QueueType[BenchmarkTask | None],
    done_queue: QueueType[int],
) -> None:
    """Long-running worker that processes multiple benchmark tasks with pipelining.

    Key optimizations:
    1. NCCL initialized once at startup (not per-config)
    2. CUDA context stays warm across benchmarks
    3. Results written to shared memory (no pickle file I/O)
    4. Module and inputs loaded once
    5. Pipelined compilation: uses background thread to compile next config
       while GPU benchmarks current one
    """
    import queue
    import threading

    rank = init.rank

    try:
        # One-time initialization
        torch.cuda.set_device(rank)

        # Warm up CUDA context
        _ = torch.zeros(1, device=f"cuda:{rank}")
        torch.cuda.synchronize()

        # Initialize NCCL once for all benchmarks
        store = dist.FileStore(init.file_store_path, init.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=init.world_size,
            rank=rank,
            store=store,
        )
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        assert group is not None
        symm_mem.enable_symm_mem_for_group(group.group_name)

        # Load module and inputs once
        module = _load_module_from_path("kernel_module", init.module_path)
        torch.manual_seed(init.seed + rank)
        inputs_module = _load_module_from_path(
            "inputs_module", init.inputs_fn_module_path
        )
        inputs_fn = getattr(inputs_module, init.inputs_fn_name)
        args = inputs_fn()

        kernel = getattr(module, init.kernel_name)

        # Open shared memory for writing results
        shm = shared_memory.SharedMemory(name=init.shm_name)

        # Signal that initialization is complete
        # This allows the pool to wait for all workers to be ready before returning
        # which prevents race conditions with tmpdir cleanup
        done_queue.put(_WORKER_INIT_COMPLETE)

        # Background compilation thread
        compile_request_queue: queue.Queue[BenchmarkTask | None] = queue.Queue()
        compile_result_queue: queue.Queue[tuple[int, Any]] = queue.Queue()
        compile_thread_stop = threading.Event()

        def compile_thread_fn() -> None:
            """Background thread that compiles configs."""
            while not compile_thread_stop.is_set():
                try:
                    task = compile_request_queue.get(timeout=0.1)
                    if task is None:
                        break
                    try:
                        cfg = helion.Config(**task.config_dict)
                        bound = kernel.bind(args)
                        compiled = bound.compile_config(cfg)
                        compile_result_queue.put((task.task_id, compiled))
                    except Exception:
                        traceback.print_exc()
                        compile_result_queue.put((task.task_id, None))
                except queue.Empty:
                    continue

        compile_thread = threading.Thread(target=compile_thread_fn, daemon=True)
        compile_thread.start()

        # Cache for pre-compiled functions and pending tasks
        compiled_cache: dict[int, Any] = {}
        pending_tasks: list[BenchmarkTask] = []

        # Process tasks until shutdown
        while True:
            # Get next task from pending queue or main queue
            if pending_tasks:
                task = pending_tasks.pop(0)
            else:
                task = task_queue.get()
            if task is _WORKER_SHUTDOWN:
                break


            # Check if we have a pre-compiled function for this task
            if task.task_id in compiled_cache:
                compiled_fn = compiled_cache.pop(task.task_id)
            else:
                # Compile synchronously (no pre-compilation available)
                try:
                    helion_config = helion.Config(**task.config_dict)
                    bound_kernel = kernel.bind(args)
                    compiled_fn = bound_kernel.compile_config(helion_config)
                except Exception:
                    traceback.print_exc()
                    compiled_fn = None

            # Check for next task(s) and start background compilation
            # Drain all available tasks to maximize pipelining
            while True:
                try:
                    peek_task = task_queue.get_nowait()
                    if peek_task is _WORKER_SHUTDOWN:
                        pending_tasks.append(peek_task)  # type: ignore[arg-type]
                        break
                    else:
                        # Queue for background compilation
                        compile_request_queue.put(peek_task)
                        pending_tasks.append(peek_task)
                except Exception:
                    break  # Queue empty

            status = 1
            timing = float("inf")
            output_cpu = None

            if compiled_fn is not None:
                try:
                    # Run kernel once to capture output
                    dist.barrier()
                    output = compiled_fn(*args)
                    torch.cuda.synchronize()
                    output_cpu = _move_tensors_to_device(output, torch.device("cpu"))
                    dist.barrier()

                    # Benchmark - while this runs on GPU, background thread compiles next
                    dist.barrier()
                    timing = do_bench_fixed_iters(
                        lambda: compiled_fn(*args),
                        warmup=1,
                        rep=task.repeat,
                        return_mode="median",
                    )
                    dist.barrier()
                    status = 0

                except Exception:
                    traceback.print_exc()

            # Check for completed background compilations
            while True:
                try:
                    task_id, compiled = compile_result_queue.get_nowait()
                    compiled_cache[task_id] = compiled
                except queue.Empty:
                    break

            # Write result to shared memory
            _write_result_to_shm(shm, status, timing, output_cpu)

            # Signal completion
            done_queue.put(task.task_id)

        # Stop compilation thread
        compile_thread_stop.set()
        compile_request_queue.put(None)
        compile_thread.join(timeout=2)

        # Clean shutdown
        shm.close()
        dist.destroy_process_group()

    except Exception:
        traceback.print_exc()

    os._exit(0)


class PersistentWorkerPool:
    """Manages a pool of persistent workers for fast distributed benchmarking.

    Workers are spawned once and reused across all benchmark configs.
    Communication uses multiprocessing queues (fast) and shared memory (for results).
    NCCL is initialized once per worker, not per-config.
    """

    def __init__(
        self,
        *,
        module_path: str,
        kernel_name: str,
        world_size: int,
        seed: int,
        inputs_fn_name: str,
        inputs_fn_module_path: str,
        tmpdir: str,
        generation: int = 0,
        init_timeout: float = 60.0,
    ) -> None:
        self.world_size = world_size
        self.tmpdir = tmpdir
        self._processes: list[Any] = []
        self._task_queues: list[QueueType[BenchmarkTask | None]] = []
        self._done_queues: list[QueueType[int]] = []
        self._shm_segments: list[shared_memory.SharedMemory] = []

        ctx = get_context("spawn")
        # Use generation in FileStore path to ensure fresh NCCL init after recovery
        file_store_path = os.path.join(tmpdir, f"persistent_store_gen{generation}")

        for rank in range(world_size):
            # Create shared memory for this worker's results
            # Include generation to avoid conflicts with old pool's shm
            shm_name = f"helion_bench_{os.getpid()}_gen{generation}_{rank}"
            try:
                # Try to unlink any existing shm with same name
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=_SHM_SIZE)
            self._shm_segments.append(shm)

            # Create queues for this worker
            task_queue: QueueType[BenchmarkTask | None] = ctx.Queue()
            done_queue: QueueType[int] = ctx.Queue()
            self._task_queues.append(task_queue)
            self._done_queues.append(done_queue)

            # Create worker init config
            init = PersistentWorkerInit(
                module_path=module_path,
                kernel_name=kernel_name,
                rank=rank,
                world_size=world_size,
                seed=seed,
                inputs_fn_name=inputs_fn_name,
                inputs_fn_module_path=inputs_fn_module_path,
                file_store_path=file_store_path,
                shm_name=shm_name,
            )

            # Spawn worker
            p = ctx.Process(
                target=_persistent_worker_loop,
                args=(init, task_queue, done_queue),
            )
            p.daemon = True
            p.start()
            self._processes.append(p)

        # Wait for all workers to complete initialization (NCCL + symm_mem)
        # This is critical to avoid race conditions where tmpdir is cleaned up
        # before workers finish using the FileStore for NCCL init
        init_complete = [False] * world_size
        start_time = time.time()
        while not all(init_complete) and time.time() - start_time < init_timeout:
            for rank, done_q in enumerate(self._done_queues):
                if not init_complete[rank]:
                    try:
                        signal = done_q.get(timeout=0.1)
                        if signal == _WORKER_INIT_COMPLETE:
                            init_complete[rank] = True
                    except Exception:
                        pass

        if all(init_complete):
            pass  # All workers initialized successfully
        else:
            failed = [i for i, c in enumerate(init_complete) if not c]
            # Log warning about workers that failed to initialize (for debugging)
            _ = failed  # Suppress unused variable warning

    def send_task(
        self,
        config_dict: dict[str, Any],
        repeat: int,
        task_id: int,
    ) -> None:
        """Send a benchmark task to all workers (non-blocking).

        Use collect_result() to get the result later.
        """
        task = BenchmarkTask(config_dict=config_dict, repeat=repeat, task_id=task_id)
        for q in self._task_queues:
            q.put(task)

    def collect_result(
        self,
        task_id: int,
        timeout: float = 30.0,
    ) -> tuple[float, Any]:
        """Wait for and collect result for a specific task_id.

        Returns (timing, output) from rank 0 worker.
        """
        # Wait for all workers to complete this task
        completed = [False] * self.world_size
        start_time = time.time()

        # Track which task IDs we've seen from each worker
        # (in case results arrive out of order)
        received_ids: list[set[int]] = [set() for _ in range(self.world_size)]

        while not all(completed) and time.time() - start_time < timeout:
            for rank, done_q in enumerate(self._done_queues):
                if not completed[rank]:
                    try:
                        result_id = done_q.get(timeout=0.01)
                        received_ids[rank].add(result_id)
                        if task_id in received_ids[rank]:
                            completed[rank] = True
                    except Exception:
                        pass  # Queue empty or timeout

        if not all(completed):
            # Some workers timed out
            return float("inf"), None

        # Read result from rank 0's shared memory
        status, timing, output = _read_result_from_shm(self._shm_segments[0])

        if status != 0:
            return float("inf"), None

        return timing, output

    def benchmark(
        self,
        config_dict: dict[str, Any],
        repeat: int,
        task_id: int,
        timeout: float = 30.0,
    ) -> tuple[float, Any]:
        """Benchmark a single config across all workers (blocking).

        Convenience method that combines send_task and collect_result.
        Returns (timing, output) from rank 0 worker.
        """
        self.send_task(config_dict=config_dict, repeat=repeat, task_id=task_id)
        return self.collect_result(task_id=task_id, timeout=timeout)

    def is_alive(self) -> bool:
        """Check if all worker processes are still alive."""
        return all(p.is_alive() for p in self._processes)

    def shutdown(self) -> None:
        """Shutdown all workers and cleanup resources."""
        # Send shutdown signal to all workers
        for q in self._task_queues:
            try:
                q.put(_WORKER_SHUTDOWN)
            except Exception:
                pass

        # Wait for workers to exit
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)

        # Cleanup shared memory
        for shm in self._shm_segments:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


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

    # Serialize results on rank 0 (only timings, outputs stay on each rank's device)
    if rank == 0:
        timings = [r.timing for r in results]
        timings_bytes = pickle.dumps(timings)
        size_tensor = torch.tensor([len(timings_bytes)], dtype=torch.int64)
    else:
        size_tensor = torch.tensor([0], dtype=torch.int64)

    dist.broadcast(size_tensor, src=0, group=gloo_group)
    size = int(size_tensor.item())

    if rank == 0:
        data_tensor = torch.frombuffer(bytearray(timings_bytes), dtype=torch.uint8).clone()
    else:
        data_tensor = torch.zeros(size, dtype=torch.uint8)

    dist.broadcast(data_tensor, src=0, group=gloo_group)

    if rank != 0:
        timings = pickle.loads(data_tensor.numpy().tobytes())
        # Non-rank-0 processes don't have outputs (workers weren't spawned here)
        # Return results with None outputs - accuracy checking should only happen on rank 0
        results = [
            CustomBenchmarkResult(timing=t, output=None)
            for t in timings
        ]

    return results


def distributed_benchmark(
    fns: list[Any],
    *,
    repeat: int = 1,
    desc: str | None = None,
    timeout: float = 30.0,
    inputs_fn: Callable[[], tuple[Any, ...]],
    use_persistent_workers: bool = True,
) -> list[float | CustomBenchmarkResult]:
    """Benchmark function for distributed autotuning with rank-0 coordination.

    Rank 0 acts as the coordinator and spawns worker processes (one per GPU/rank).
    Non-rank-0 processes simply wait for results to be broadcast from rank 0.
    This ensures all workers test the same configs and their collectives align.

    Workers coordinate via FileStore + NCCL for GPU collectives.
    Coordination between parent processes uses Gloo.

    Performance optimization: When use_persistent_workers=True (default), workers
    are spawned once and reused across all configs. This avoids the overhead of
    process spawning, CUDA context initialization, and NCCL setup per config.

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
        use_persistent_workers: If True (default), use persistent workers for faster
            benchmarking. Workers are spawned once and reused across all configs.

    Returns:
        List of CustomBenchmarkResult with timing and output for accuracy checking.
        Returns inf timing for failed/timed-out configs.
    """

    if not fns:
        return []

    if not dist.is_initialized():
        raise RuntimeError(
            "distributed_benchmark requires torch.distributed to be initialized. "
            "Call dist.init_process_group() before running the kernel."
        )

    # Require explicit seed for distributed autotuning to ensure all ranks use the same seed
    if os.environ.get("HELION_AUTOTUNE_RANDOM_SEED") is None:
        raise RuntimeError(
            "Distributed autotuning requires an explicit random seed. "
            "Set HELION_AUTOTUNE_RANDOM_SEED environment variable to ensure "
            "all ranks generate the same configs. Example: HELION_AUTOTUNE_RANDOM_SEED=42"
        )

    bound_kernel = fns[0].kernel
    module_path = bound_kernel.kernel.fn.__code__.co_filename
    kernel_name = bound_kernel.kernel.fn.__name__

    inputs_fn_name = inputs_fn.__name__
    inputs_fn_module_path = inputs_fn.__code__.co_filename

    seed = torch.initial_seed() % (2**31)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")


    # Get Gloo group for CPU-side coordination
    gloo_group = _get_or_create_gloo_group()

    # Rank 0 broadcasts the config list to all ranks (so all ranks know how many results to expect)
    config_dicts = [dict(fn.config) for fn in fns]
    config_dicts = _broadcast_config_list(config_dicts, gloo_group, rank)

    if rank == 0:
        # === RANK 0: Coordinator ===
        tmpdir = tempfile.mkdtemp()
        results: list[CustomBenchmarkResult] = []

        try:
            if use_persistent_workers:
                # Fast path: use persistent workers
                results = _benchmark_with_persistent_workers(
                    config_dicts=config_dicts,
                    module_path=module_path,
                    kernel_name=kernel_name,
                    world_size=world_size,
                    seed=seed,
                    inputs_fn_name=inputs_fn_name,
                    inputs_fn_module_path=inputs_fn_module_path,
                    tmpdir=tmpdir,
                    repeat=repeat,
                    timeout=timeout,
                    device=device,
                )
            else:
                # Slow path: spawn new workers for each config
                results = _benchmark_with_new_workers(
                    config_dicts=config_dicts,
                    module_path=module_path,
                    kernel_name=kernel_name,
                    world_size=world_size,
                    seed=seed,
                    inputs_fn_name=inputs_fn_name,
                    inputs_fn_module_path=inputs_fn_module_path,
                    tmpdir=tmpdir,
                    repeat=repeat,
                    timeout=timeout,
                    device=device,
                )


        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # Broadcast results to all other ranks
        results = _broadcast_results(results, gloo_group, rank, device)

    else:
        # === NON-RANK-0: Just wait for results from rank 0 ===
        # We don't spawn any workers - rank 0 spawns workers for all ranks
        results = _broadcast_results([], gloo_group, rank, device)

    # pyrefly: ignore [bad-return]
    return results


def _benchmark_with_persistent_workers(
    *,
    config_dicts: list[dict[str, Any]],
    module_path: str,
    kernel_name: str,
    world_size: int,
    seed: int,
    inputs_fn_name: str,
    inputs_fn_module_path: str,
    tmpdir: str,
    repeat: int,
    timeout: float,
    device: torch.device,
    pipeline_depth: int = 2,
) -> list[CustomBenchmarkResult]:
    """Benchmark configs using persistent workers (fast path).

    Workers are spawned once, NCCL is initialized once, and all configs
    are processed by the same workers. Results are passed via shared memory.
    The worker pool is reused across calls for efficiency.

    Pipelining: sends multiple tasks ahead to allow workers to overlap
    compilation of task N+1 with benchmarking of task N.
    """
    pool = _get_or_create_pool(
        module_path=module_path,
        kernel_name=kernel_name,
        world_size=world_size,
        seed=seed,
        inputs_fn_name=inputs_fn_name,
        inputs_fn_module_path=inputs_fn_module_path,
        tmpdir=tmpdir,
    )

    results: list[CustomBenchmarkResult] = []
    next_to_send = 0
    next_to_collect = 0

    while next_to_collect < len(config_dicts):
        # Send tasks up to pipeline_depth ahead
        while next_to_send < len(config_dicts) and next_to_send < next_to_collect + pipeline_depth:
            config_dict = config_dicts[next_to_send]
            pool.send_task(
                config_dict=config_dict,
                repeat=repeat,
                task_id=next_to_send,
            )
            next_to_send += 1

        # Collect next result in order
        timing, output = pool.collect_result(
            task_id=next_to_collect,
            timeout=timeout,
        )

        # Only shutdown pool if workers actually died, not on normal config errors
        # (e.g., OutOfResources). Workers can handle errors and continue processing.
        # The new pool will have a new generation number for unique resource names.
        if timing == float("inf") and not pool.is_alive():
            shutdown_persistent_pool()
            # Create a new pool for remaining configs. The pool's FileStore is only
            # needed during NCCL init, so reusing the pool later with a different
            # tmpdir is fine - workers communicate via NCCL and shared memory after init.
            pool = _get_or_create_pool(
                module_path=module_path,
                kernel_name=kernel_name,
                world_size=world_size,
                seed=seed,
                inputs_fn_name=inputs_fn_name,
                inputs_fn_module_path=inputs_fn_module_path,
                tmpdir=tmpdir,
            )
            # Re-send any tasks that were pipelined but not yet collected
            # (they were lost when the old pool died)
            for task_idx in range(next_to_collect + 1, next_to_send):
                pool.send_task(
                    config_dict=config_dicts[task_idx],
                    repeat=repeat,
                    task_id=task_idx,
                )

        # Move output to correct device
        output = _move_tensors_to_device(output, device)
        results.append(CustomBenchmarkResult(timing=timing, output=output))
        next_to_collect += 1

    return results


def _benchmark_with_new_workers(
    *,
    config_dicts: list[dict[str, Any]],
    module_path: str,
    kernel_name: str,
    world_size: int,
    seed: int,
    inputs_fn_name: str,
    inputs_fn_module_path: str,
    tmpdir: str,
    repeat: int,
    timeout: float,
    device: torch.device,
) -> list[CustomBenchmarkResult]:
    """Benchmark configs by spawning new workers for each config (slow path).

    This is the original implementation - kept as fallback.
    """
    results: list[CustomBenchmarkResult] = []
    ctx = get_context("spawn")

    for config_index, config_dict in enumerate(config_dicts):
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

            p = ctx.Process(target=_distributed_benchmark_worker, args=(bench_config,))
            p.daemon = True
            p.start()
            processes.append(p)

        # Wait for all workers with timeout
        start_time = time.time()
        while any(p.is_alive() for p in processes) and time.time() - start_time < timeout:
            time.sleep(0.1)

        elapsed = time.time() - start_time
        alive_count = sum(1 for p in processes if p.is_alive())

        # Kill any workers that are still alive
        for i, p in enumerate(processes):
            if p.is_alive():
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
            except Exception:
                pass  # Failed to read result
        else:
            pass  # No result file found

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

    return results
