from __future__ import annotations

import contextlib
from datetime import timedelta
import multiprocessing as mp
import os
import pickle
import tempfile
import time
from typing import ClassVar
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import run_tests

from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import skipIfRocm

import helion
from examples.distributed.one_shot_allreduce_bias_rmsnorm import (
    one_shot_allreduce_bias_rmsnorm_kernel,
    reference_one_shot_allreduce_bias_rmsnorm,
)


def _distributed_benchmark_worker(
    kernel_source_path: str,
    config_dict: dict,
    rank: int,
    world_size: int,
    file_store_path: str,
    result_path: str,
    input_data_path: str,
) -> None:
    """
    Worker function that runs in a subprocess to benchmark a kernel config.
    """
    import sys
    import traceback

    status = 1
    timing = float("inf")
    try:
        # Set up CUDA device
        torch.cuda.set_device(rank)

        # Initialize distributed
        store = dist.FileStore(file_store_path, world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
        )

        # Set up symmetric memory
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        # Load input data
        with open(input_data_path, "rb") as f:
            input_data = pickle.load(f)

        # Create kernel inputs
        N, D = input_data["N"], input_data["D"]
        dtype = input_data["dtype"]
        eps = input_data["eps"]

        torch.manual_seed(42 + rank)
        x = torch.randn(N, D, dtype=dtype, device=f"cuda:{rank}")

        torch.manual_seed(42)
        bias = torch.randn(D, dtype=dtype, device=f"cuda:{rank}")
        weight = torch.randn(D, dtype=dtype, device=f"cuda:{rank}")

        symm_mem_buffer = symm_mem.empty(N, D, dtype=dtype, device=f"cuda:{rank}")
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

        # Load and compile kernel with specific config
        import importlib.util

        spec = importlib.util.spec_from_file_location("kernel_module", kernel_source_path)
        module = importlib.util.module_from_spec(spec)  # pyre-ignore
        sys.modules["kernel_module"] = module
        spec.loader.exec_module(module)  # pyre-ignore

        kernel = module._test_allreduce_bias_rmsnorm_kernel
        config = helion.Config(**config_dict)

        # Build args tuple for binding and calling
        args = (
            x,
            symm_mem_buffer,
            bias,
            weight,
            symm_mem_hdl.signal_pad_ptrs_dev,
            eps,
            symm_mem_hdl.rank,
            symm_mem_hdl.world_size,
            group.group_name,
        )

        # Bind kernel with args to get BoundKernel, then compile with config
        bound_kernel = kernel.bind(args)
        compiled_fn = bound_kernel.compile_config(config)

        # Warmup
        torch.cuda.synchronize()
        dist.barrier()

        _ = compiled_fn(*args)
        torch.cuda.synchronize()

        # Benchmark
        dist.barrier()
        start = time.perf_counter()
        _ = compiled_fn(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()
        timing = (end - start) * 1000  # Convert to ms

        dist.barrier()
        dist.destroy_process_group()
        status = 0
    except Exception:
        traceback.print_exc()
        timing = float("inf")
    finally:
        # Write result (only rank 0)
        if rank == 0:
            with open(result_path, "wb") as f:
                pickle.dump({"status": status, "timing": timing}, f)
        os._exit(status)


# Global to store test file path for the benchmark function
_TEST_FILE_PATH: str | None = None
_TEST_INPUT_DATA: dict | None = None


def _coordinated_distributed_benchmark_fn(
    fns: list,
    *,
    repeat: int = 1,
    desc: str | None = None,
) -> list[float]:
    """
    Coordinated distributed benchmark function.

    Only one process (determined by gloo rank 0) runs the actual benchmarks.
    Other processes wait and receive results via broadcast.
    """
    global _TEST_FILE_PATH, _TEST_INPUT_DATA

    # Get current rank and world_size from the existing distributed context
    if not dist.is_initialized():
        # Not in distributed context, just run benchmarks directly
        return [fn() if callable(fn) else float("inf") for fn in fns]

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create a gloo process group for coordination if not exists
    gloo_pg = dist.new_group(backend="gloo")

    timings: list[float] = []

    # Only rank 0 runs the actual benchmarks
    if rank == 0:
        for fn in fns:
            config = fn.config
            config_dict = dict(config)  # Config is a Mapping, so dict() works

            # Create temp files for coordination
            with tempfile.TemporaryDirectory() as tmpdir:
                file_store_path = os.path.join(tmpdir, "store")
                result_path = os.path.join(tmpdir, "result.pkl")
                input_data_path = os.path.join(tmpdir, "input.pkl")

                # Save input data
                with open(input_data_path, "wb") as f:
                    pickle.dump(_TEST_INPUT_DATA, f)

                # Spawn worker processes
                ctx = mp.get_context("spawn")
                processes = []
                for worker_rank in range(world_size):
                    p = ctx.Process(
                        target=_distributed_benchmark_worker,
                        args=(
                            _TEST_FILE_PATH,
                            config_dict,
                            worker_rank,
                            world_size,
                            file_store_path,
                            result_path,
                            input_data_path,
                        ),
                    )
                    p.daemon = True
                    p.start()
                    processes.append(p)

                # Wait for all processes with timeout (30 seconds)
                timeout = 30.0
                start_time = time.time()
                all_done = False
                while time.time() - start_time < timeout:
                    if all(not p.is_alive() for p in processes):
                        all_done = True
                        break
                    time.sleep(0.1)

                # Kill any remaining processes
                for p in processes:
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=5)

                # Read result
                if all_done and os.path.exists(result_path):
                    try:
                        with open(result_path, "rb") as f:
                            result = pickle.load(f)
                        timing = result.get("timing", float("inf"))
                    except Exception:
                        timing = float("inf")
                else:
                    timing = float("inf")

            timings.append(timing)
    else:
        # Other ranks just wait
        timings = [float("inf")] * len(fns)

    # Broadcast timings from rank 0 to all ranks
    timings_tensor = torch.tensor(timings, dtype=torch.float64)
    dist.broadcast(timings_tensor, src=0, group=gloo_pg)
    timings = timings_tensor.tolist()

    dist.destroy_process_group(gloo_pg)
    return timings


# Re-decorate the kernel from examples with test-specific settings:
# - Multiple configs including one that hangs (block_sizes=[1])
# - Custom benchmark function for distributed subprocess spawning with timeout
_test_allreduce_bias_rmsnorm_kernel = helion.jit(
    configs=[
        helion.Config(block_sizes=[8], num_warps=8),
        helion.Config(block_sizes=[1], num_warps=8),  # This one hangs!
    ],
    autotune_baseline_fn=lambda x, symm_mem_buffer, bias, weight, signal_pad_ptrs, eps, rank, world_size, group_name: reference_one_shot_allreduce_bias_rmsnorm(
        x, bias, weight, eps
    ),
    autotune_precompile=None,
    autotune_benchmark_fn=_coordinated_distributed_benchmark_fn,
    static_shapes=True,
)(one_shot_allreduce_bias_rmsnorm_kernel.fn)


@instantiate_parametrized_tests
class TestExamplesDist(TestCase, MultiProcessTestCase):
    _nvshmem_env: ClassVar[dict[str, str]] = {
        # Configure NVSHMEM to use smaller heap and work without NVSwitch
        # Default heap is 128GB which fails cuMemMap on AWS H100 instances
        "NVSHMEM_SYMMETRIC_SIZE": "4G",
        # Disable NVLink Switch features (not available on AWS H100 instances)
        "NVSHMEM_DISABLE_NVLS": "1",
    }

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._class_stack = contextlib.ExitStack()
        cls._class_stack.enter_context(patch.dict(os.environ, cls._nvshmem_env))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_stack.close()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        super().tearDown()

    @property
    def world_size(self) -> int:
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=60), dist.group.WORLD
        )
        torch.manual_seed(42 + self.rank)

    def _cleanup_process(self):
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_all_gather_matmul(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "all_gather_matmul.py")

        M, N, K = 4096, 6656, 16384

        a_shared = symm_mem.empty(
            M // self.world_size, K, dtype=torch.bfloat16, device=self.device
        ).normal_()

        b = (
            torch.randn((K, N), device=self.device, dtype=torch.bfloat16)
            .T.contiguous()
            .T
        )

        symm_mem_group = dist.group.WORLD
        if symm_mem_group is None:
            raise RuntimeError("No symmetric memory group available")
        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=symm_mem_group)
        a_shape = list(a_shared.shape)
        a_shape[0] *= symm_mem_hdl.world_size
        a_out = torch.empty(a_shape, dtype=a_shared.dtype, device=a_shared.device)
        progress = torch.zeros(
            symm_mem_hdl.world_size,
            dtype=torch.uint32,
            device=a_shared.device,
        )
        backend_stream = mod.copy_engine_all_gather_w_progress(
            a_out, a_shared, progress, 1
        )

        code, result = code_and_output(
            mod.helion_matmul_w_progress,
            (a_out, a_shared, b, progress, 1, symm_mem_hdl.rank),
        )

        if self.rank == 0:
            if not hasattr(self.__class__, "_expected_journal"):
                from helion._testing import AssertExpectedJournal

                self.__class__._expected_journal = AssertExpectedJournal(self.__class__)
            self.assertExpectedJournal(code)

        # Synchronize CUDA before running reference
        torch.cuda.synchronize()

        golden_a = a_shared.clone()
        ag_golden, mm_golden = torch.ops.symm_mem.fused_all_gather_matmul(
            golden_a, [b], gather_dim=0, group_name=symm_mem_group.group_name
        )

        torch.testing.assert_close(result, mm_golden[0], rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(a_out, ag_golden)

        torch.cuda.current_stream().wait_stream(backend_stream)
        self._cleanup_process()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_all_reduce(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "all_reduce.py")

        # Only NVSHMEM backend implements `get_remote_tensor` for now.
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        N = 16384
        dtype = torch.bfloat16

        a_shared = symm_mem.empty(
            N // self.world_size, dtype=dtype, device=self.device
        ).normal_()

        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=group)
        local_signal_pad = symm_mem_hdl.get_signal_pad(
            symm_mem_hdl.rank, dtype=torch.int32
        ).view(-1, symm_mem_hdl.world_size)
        signal_pad_addrs = mod.dev_array_to_tensor_short(
            symm_mem_hdl.signal_pad_ptrs_dev,
            (symm_mem_hdl.world_size,),
            dtype=torch.uint64,
            device=a_shared.device,
        )

        code, result = code_and_output(
            mod.one_shot_all_reduce_kernel,
            (
                signal_pad_addrs,
                local_signal_pad,
                a_shared,
                symm_mem_hdl.rank,
                group.group_name,
            ),
        )

        if self.rank == 0:
            if not hasattr(self.__class__, "_expected_journal"):
                from helion._testing import AssertExpectedJournal

                self.__class__._expected_journal = AssertExpectedJournal(self.__class__)
            self.assertExpectedJournal(code)

        # Synchronize CUDA before running reference
        torch.cuda.synchronize()

        a_shared_ref = symm_mem.empty(
            N // self.world_size, dtype=dtype, device=self.device
        )
        a_shared_ref.copy_(a_shared)
        expected = mod.reference_one_shot_all_reduce(a_shared_ref)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

        self._cleanup_process()

    @pytest.mark.timeout(120)
    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_one_shot_allreduce_bias_rmsnorm_autotune_timeout(self):
        """
        Test that distributed autotuning properly handles hanging configs.

        With spawn mode, hanging kernels run in subprocess that get killed on timeout.
        After autotuning, GPUs should be in a clean state (not hung).
        """
        global _TEST_FILE_PATH, _TEST_INPUT_DATA

        self._init_process()

        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        N, D = 128, 4096
        dtype = torch.float32
        eps = 1e-5

        # Set up global variables for the benchmark function
        _TEST_FILE_PATH = __file__
        _TEST_INPUT_DATA = {"N": N, "D": D, "dtype": dtype, "eps": eps}

        x = torch.randn(N, D, dtype=dtype, device=self.device)

        torch.manual_seed(42)
        bias = torch.randn(D, dtype=dtype, device=self.device)
        weight = torch.randn(D, dtype=dtype, device=self.device)

        x_ref = x.clone()

        symm_mem_buffer = symm_mem.empty(N, D, dtype=dtype, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

        # block_sizes=[1] would hang; autotuner should timeout, kill subprocess, and skip it
        result = _test_allreduce_bias_rmsnorm_kernel(
            x,
            symm_mem_buffer,
            bias,
            weight,
            symm_mem_hdl.signal_pad_ptrs_dev,
            eps,
            symm_mem_hdl.rank,
            symm_mem_hdl.world_size,
            group.group_name,
        )

        # GPU should NOT be hung - synchronize should complete normally
        torch.cuda.synchronize()

        # Verify correctness
        expected = reference_one_shot_allreduce_bias_rmsnorm(x_ref, bias, weight, eps)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        self._cleanup_process()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_one_shot_allreduce_bias_rmsnorm(self):
        self._init_process()

        mod = import_path(
            EXAMPLES_DIR / "distributed" / "one_shot_allreduce_bias_rmsnorm.py"
        )

        # Only NVSHMEM backend implements `get_remote_tensor` for now.
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        N, D = 128, 4096
        dtype = torch.float32
        eps = 1e-5

        x = torch.randn(N, D, dtype=dtype, device=self.device)

        torch.manual_seed(42)
        bias = torch.randn(D, dtype=dtype, device=self.device)
        weight = torch.randn(D, dtype=dtype, device=self.device)

        x_ref = x.clone()

        symm_mem_buffer = symm_mem.empty(N, D, dtype=dtype, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

        code, result = code_and_output(
            mod.one_shot_allreduce_bias_rmsnorm_kernel,
            (
                x,
                symm_mem_buffer,
                bias,
                weight,
                symm_mem_hdl.signal_pad_ptrs_dev,
                eps,  # EPS constexpr
                symm_mem_hdl.rank,  # RANK constexpr
                symm_mem_hdl.world_size,  # WORLD_SIZE constexpr
                group.group_name,  # GROUP_NAME constexpr
            ),
        )

        if self.rank == 0:
            if not hasattr(self.__class__, "_expected_journal"):
                from helion._testing import AssertExpectedJournal

                self.__class__._expected_journal = AssertExpectedJournal(self.__class__)
            self.assertExpectedJournal(code)

        torch.cuda.synchronize()

        expected = mod.reference_one_shot_allreduce_bias_rmsnorm(
            x_ref, bias, weight, eps
        )

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        self._cleanup_process()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skip_if_lt_x_gpu(4)
    def test_matmul_reduce_scatter(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "matmul_reduce_scatter.py")

        # Only NVSHMEM backend implements `get_remote_tensor` for now.
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        symm_mem.enable_symm_mem_for_group(group.group_name)

        M, N, K = 512, 768, 1024
        dtype = torch.float32

        # Each rank has the same random seed for reproducibility
        torch.manual_seed(42 + self.rank)
        a = torch.randn(M, K, dtype=dtype, device=self.device)

        # Weight matrix is the same across all ranks
        torch.manual_seed(42)
        b = torch.randn(K, N, dtype=dtype, device=self.device)

        # Clone for reference computation
        a_ref = a.clone()
        b_ref = b.clone()

        # Setup symmetric memory like the wrapper does
        symm_mem_buffer = symm_mem.empty(M, N, dtype=dtype, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

        code, result = code_and_output(
            mod.matmul_reduce_scatter_kernel,
            (
                a,
                b,
                symm_mem_buffer,
                symm_mem_hdl.signal_pad_ptrs_dev,
                symm_mem_hdl.rank,  # RANK constexpr
                symm_mem_hdl.world_size,  # WORLD_SIZE constexpr
                group.group_name,  # GROUP_NAME constexpr
            ),
        )

        if self.rank == 0:
            if not hasattr(self.__class__, "_expected_journal"):
                from helion._testing import AssertExpectedJournal

                self.__class__._expected_journal = AssertExpectedJournal(self.__class__)
            self.assertExpectedJournal(code)

        torch.cuda.synchronize()

        expected = mod.reference_matmul_reduce_scatter(a_ref, b_ref)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

        self._cleanup_process()


if __name__ == "__main__":
    run_tests()
