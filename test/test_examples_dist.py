from __future__ import annotations

import contextlib
from datetime import timedelta
import os
from typing import ClassVar
from unittest.mock import patch

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
