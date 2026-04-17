from __future__ import annotations

import contextlib
from datetime import timedelta
import json
import os
import time
from typing import ClassVar
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import run_tests

import helion
from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfXPU

_AGENT_DEBUG_LOG_PATH = "/root/.cursor/debug-logs/debug-412933.log"
_AGENT_DEBUG_SESSION_ID = "412933"


def _agent_debug_log(
    run_id: str, hypothesis_id: str, location: str, message: str, data: dict
) -> None:
    payload = {
        "sessionId": _AGENT_DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        log_dir = os.path.dirname(_AGENT_DEBUG_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def _set_preferred_symm_mem_backend(
    run_id: str, rank: int, device: torch.device
) -> str:
    preferred = "NVSHMEM"
    try:
        symm_mem.set_backend(preferred)
        selected = preferred
    except RuntimeError as exc:
        selected = str(symm_mem.get_backend(device) or "unknown")
        # region agent log
        _agent_debug_log(
            run_id=run_id,
            hypothesis_id="H12",
            location="test/test_examples_dist.py:_set_preferred_symm_mem_backend",
            message="Preferred symmetric memory backend unavailable; using fallback",
            data={
                "rank": rank,
                "preferred_backend": preferred,
                "selected_backend": selected,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        # endregion
    # region agent log
    _agent_debug_log(
        run_id=run_id,
        hypothesis_id="H12",
        location="test/test_examples_dist.py:_set_preferred_symm_mem_backend",
        message="Selected symmetric memory backend",
        data={"rank": rank, "selected_backend": selected},
    )
    # endregion
    return selected


@onlyBackends(["triton"])
@instantiate_parametrized_tests
class TestExamplesDist(TestCase, MultiProcessTestCase):
    _nvshmem_env: ClassVar[dict[str, str]] = {
        # Configure NVSHMEM to use smaller heap and work without NVSwitch
        # Default heap is 128GB which fails cuMemMap on AWS H100 instances
        "NVSHMEM_SYMMETRIC_SIZE": "4G",
        # Disable NVLink Switch features (not available on AWS H100 instances)
        "NVSHMEM_DISABLE_NVLS": "1",
        # Disable NCCL's NVLS (NVLink SHARP) multicast which requires NVSwitch/Fabric Manager
        "NCCL_NVLS_ENABLE": "0",
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
        # region agent log
        _agent_debug_log(
            run_id=f"examples_dist_init_pid{os.getpid()}",
            hypothesis_id="H6",
            location="test/test_examples_dist.py:_init_process",
            message="Initialized distributed examples process",
            data={
                "test_name": getattr(self, "_testMethodName", "unknown"),
                "rank": self.rank,
                "world_size": self.world_size,
                "device": str(self.device),
                "hip": torch.version.hip,
            },
        )
        # endregion

    def _cleanup_process(self):
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_all_gather_matmul(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "all_gather_matmul.py")

        if torch.version.hip is not None:
            M, N, K = 512, 1024, 1024
        else:
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

        # region agent log
        _agent_debug_log(
            run_id=f"examples_dist_ag_matmul_pid{os.getpid()}",
            hypothesis_id="H7",
            location="test/test_examples_dist.py:test_all_gather_matmul",
            message="Calling all_gather_matmul kernel",
            data={
                "rank": self.rank,
                "symm_rank": symm_mem_hdl.rank,
                "group_name": symm_mem_group.group_name,
            },
        )
        # endregion
        kernel = mod.helion_matmul_w_progress
        if torch.version.hip is not None:
            # region agent log
            _agent_debug_log(
                run_id=f"examples_dist_ag_matmul_pid{os.getpid()}",
                hypothesis_id="H7",
                location="test/test_examples_dist.py:test_all_gather_matmul",
                message="Using ROCm-safe matmul config",
                data={
                    "rank": self.rank,
                    "config": {
                        "block_sizes": [64, 64, 32],
                        "num_warps": 4,
                        "num_stages": 2,
                        "indexing": "pointer",
                    },
                },
            )
            # endregion
            kernel = helion.kernel(
                config=helion.Config(
                    block_sizes=[64, 64, 32],
                    num_warps=4,
                    num_stages=2,
                    indexing="pointer",
                ),
                static_shapes=True,
            )(mod.helion_matmul_w_progress.fn)
        try:
            _, result = code_and_output(
                kernel,
                (a_out, a_shared, b, progress, 1, symm_mem_hdl.rank),
            )
        except Exception as exc:
            # region agent log
            _agent_debug_log(
                run_id=f"examples_dist_ag_matmul_pid{os.getpid()}",
                hypothesis_id="H7",
                location="test/test_examples_dist.py:test_all_gather_matmul",
                message="All_gather_matmul kernel failed",
                data={
                    "rank": self.rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            # endregion
            raise

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

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_all_reduce(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "all_reduce.py")

        selected_backend = _set_preferred_symm_mem_backend(
            run_id=f"examples_dist_allreduce_pid{os.getpid()}",
            rank=self.rank,
            device=self.device,
        )
        group = dist.group.WORLD
        if selected_backend == "NVSHMEM":
            symm_mem.enable_symm_mem_for_group(group.group_name)

        N = 16384
        dtype = torch.bfloat16

        a_shared = symm_mem.empty(
            N // self.world_size, dtype=dtype, device=self.device
        ).normal_()

        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=group)

        # region agent log
        _agent_debug_log(
            run_id=f"examples_dist_allreduce_pid{os.getpid()}",
            hypothesis_id="H8",
            location="test/test_examples_dist.py:test_all_reduce",
            message="Calling examples all_reduce kernel",
            data={
                "rank": self.rank,
                "symm_rank": symm_mem_hdl.rank,
                "group_name": group.group_name,
            },
        )
        # endregion
        kernel = mod.one_shot_all_reduce_kernel
        if torch.version.hip is not None:
            kernel = helion.kernel(
                config=helion.Config(block_sizes=[8192], num_warps=16),
                static_shapes=True,
            )(mod.one_shot_all_reduce_kernel.fn)
        try:
            _, result = code_and_output(
                kernel,
                (
                    symm_mem_hdl.signal_pad_ptrs_dev,
                    a_shared,
                    symm_mem_hdl.rank,
                    group.group_name,
                    symm_mem_hdl.world_size,
                ),
            )
        except Exception as exc:
            # region agent log
            _agent_debug_log(
                run_id=f"examples_dist_allreduce_pid{os.getpid()}",
                hypothesis_id="H8",
                location="test/test_examples_dist.py:test_all_reduce",
                message="Examples all_reduce kernel failed",
                data={
                    "rank": self.rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            # endregion
            raise

        # Synchronize CUDA before running reference
        torch.cuda.synchronize()

        a_shared_ref = symm_mem.empty(
            N // self.world_size, dtype=dtype, device=self.device
        )
        a_shared_ref.copy_(a_shared)
        expected = mod.reference_one_shot_all_reduce(a_shared_ref)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

        self._cleanup_process()

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "kernel_name",
        (
            "one_shot_allreduce_bias_rmsnorm_kernel",
            "two_shot_allreduce_bias_rmsnorm_kernel",
        ),
    )
    def test_allreduce_bias_rmsnorm(self, kernel_name):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "allreduce_bias_rmsnorm.py")

        selected_backend = _set_preferred_symm_mem_backend(
            run_id=f"examples_dist_rmsnorm_pid{os.getpid()}",
            rank=self.rank,
            device=self.device,
        )
        group = dist.group.WORLD
        if selected_backend == "NVSHMEM":
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

        # region agent log
        _agent_debug_log(
            run_id=f"examples_dist_rmsnorm_pid{os.getpid()}",
            hypothesis_id="H9",
            location="test/test_examples_dist.py:test_allreduce_bias_rmsnorm",
            message="Calling examples allreduce_bias_rmsnorm kernel",
            data={
                "rank": self.rank,
                "kernel_name": kernel_name,
                "symm_rank": symm_mem_hdl.rank,
                "group_name": group.group_name,
            },
        )
        # endregion
        kernel = getattr(mod, kernel_name)
        if (
            torch.version.hip is not None
            and kernel_name == "two_shot_allreduce_bias_rmsnorm_kernel"
        ):
            kernel = helion.jit(config=helion.Config(block_sizes=[4], num_warps=16))(
                kernel.fn
            )
        try:
            _, result = code_and_output(
                kernel,
                (
                    symm_mem_buffer,
                    x,
                    bias,
                    weight,
                    symm_mem_hdl.signal_pad_ptrs_dev,
                    eps,  # EPS constexpr
                    symm_mem_hdl.rank,  # RANK constexpr
                    symm_mem_hdl.world_size,  # WORLD_SIZE constexpr
                    group.group_name,  # GROUP_NAME constexpr
                ),
            )
        except Exception as exc:
            # region agent log
            _agent_debug_log(
                run_id=f"examples_dist_rmsnorm_pid{os.getpid()}",
                hypothesis_id="H9",
                location="test/test_examples_dist.py:test_allreduce_bias_rmsnorm",
                message="Examples allreduce_bias_rmsnorm kernel failed",
                data={
                    "rank": self.rank,
                    "kernel_name": kernel_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            # endregion
            raise

        torch.cuda.synchronize()

        expected = mod.reference_allreduce_bias_rmsnorm(x_ref, bias, weight, eps)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        self._cleanup_process()

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_matmul_reduce_scatter_kernel(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "matmul_reduce_scatter.py")

        selected_backend = _set_preferred_symm_mem_backend(
            run_id=f"examples_dist_matmul_rs_kernel_pid{os.getpid()}",
            rank=self.rank,
            device=self.device,
        )
        group = dist.group.WORLD
        if selected_backend == "NVSHMEM":
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

        # region agent log
        _agent_debug_log(
            run_id=f"examples_dist_matmul_rs_kernel_pid{os.getpid()}",
            hypothesis_id="H10",
            location="test/test_examples_dist.py:test_matmul_reduce_scatter_kernel",
            message="Calling examples matmul_reduce_scatter kernel",
            data={
                "rank": self.rank,
                "symm_rank": symm_mem_hdl.rank,
                "group_name": group.group_name,
            },
        )
        # endregion
        try:
            _, result = code_and_output(
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
        except Exception as exc:
            # region agent log
            _agent_debug_log(
                run_id=f"examples_dist_matmul_rs_kernel_pid{os.getpid()}",
                hypothesis_id="H10",
                location="test/test_examples_dist.py:test_matmul_reduce_scatter_kernel",
                message="Examples matmul_reduce_scatter kernel failed",
                data={
                    "rank": self.rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            # endregion
            raise

        torch.cuda.synchronize()

        expected = mod.reference_matmul_reduce_scatter(a_ref, b_ref)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

        self._cleanup_process()

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_matmul_reduce_scatter_wrapper(self):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "matmul_reduce_scatter.py")

        selected_backend = _set_preferred_symm_mem_backend(
            run_id=f"examples_dist_matmul_rs_wrapper_pid{os.getpid()}",
            rank=self.rank,
            device=self.device,
        )
        group = dist.group.WORLD
        if selected_backend == "NVSHMEM":
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
        symm_mem.rendezvous(symm_mem_buffer, group.group_name)

        # region agent log
        _agent_debug_log(
            run_id=f"examples_dist_matmul_rs_wrapper_pid{os.getpid()}",
            hypothesis_id="H11",
            location="test/test_examples_dist.py:test_matmul_reduce_scatter_wrapper",
            message="Calling examples matmul_reduce_scatter wrapper",
            data={"rank": self.rank, "group_name": group.group_name},
        )
        # endregion
        try:
            result = mod.helion_matmul_reduce_scatter(
                symm_mem_buffer,
                a,
                b,
            )
        except Exception as exc:
            # region agent log
            _agent_debug_log(
                run_id=f"examples_dist_matmul_rs_wrapper_pid{os.getpid()}",
                hypothesis_id="H11",
                location="test/test_examples_dist.py:test_matmul_reduce_scatter_wrapper",
                message="Examples matmul_reduce_scatter wrapper failed",
                data={
                    "rank": self.rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            # endregion
            raise

        torch.cuda.synchronize()

        expected = mod.reference_matmul_reduce_scatter(a_ref, b_ref)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

        self._cleanup_process()


if __name__ == "__main__":
    run_tests()
