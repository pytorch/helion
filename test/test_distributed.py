from __future__ import annotations

import contextlib
from datetime import timedelta
import os
import unittest

import torch
from torch import Tensor
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import run_tests

import helion
from helion._dist_utils import all_gather_object
from helion._dist_utils import sync_object
from helion._dist_utils import sync_seed
from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU
from helion.autotuner import search_algorithms
from helion.autotuner.effort_profile import _PROFILES
from helion.autotuner.effort_profile import AutotuneEffortProfile
from helion.autotuner.effort_profile import DifferentialEvolutionConfig
from helion.autotuner.effort_profile import PatternSearchConfig
from helion.autotuner.effort_profile import RandomSearchConfig
import helion.language as hl

autotuner_names = ["fixed", *search_algorithms]


def custom_get_timeout(test_id: str) -> int:
    """
    Use a larger timeout setting to autotune distributed kernels.
    """
    return int(os.getenv("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", "600"))


def one_shot_allreduce_kernel(
    a_shared: torch.Tensor,
    my_rank: hl.constexpr,
    group_name: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
) -> torch.Tensor:
    out = torch.empty_like(a_shared)
    N = out.size(0)
    a_shared_tuple = torch.ops.symm_mem.get_remote_tensors(a_shared, group_name)

    for tile_n in hl.tile(N):
        acc = hl.zeros([tile_n], dtype=a_shared.dtype, device=a_shared.device)

        for a in a_shared_tuple:
            acc += a[tile_n]

        out[tile_n] = acc
    return out


# make it easy to use a 'smaller' profile than 'quick' in unit test
pattern_search_config = PatternSearchConfig(
    initial_population=6,
    copies=2,
    max_generations=3,
)

differential_evolution_config = DifferentialEvolutionConfig(
    population_size=10,
    max_generations=5,
)

random_search_config = RandomSearchConfig(
    count=20,
)

profile = AutotuneEffortProfile(
    pattern_search=pattern_search_config,
    lfbo_pattern_search=pattern_search_config,
    differential_evolution=differential_evolution_config,
    random_search=random_search_config,
)


@onlyBackends(["triton"])
@instantiate_parametrized_tests
class TestDistributed(TestCase, MultiProcessTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._class_stack = contextlib.ExitStack()
        cls._class_stack.enter_context(
            unittest.mock.patch.dict(
                os.environ,
                {
                    "HELION_DIST_CHECK_CONFIG_CONSISTANCY": "1",
                    "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "50",
                    "HELION_CAP_REBENCHMARK_REPEAT": "50",
                },
            )
        )
        cls._class_stack.enter_context(
            unittest.mock.patch.object(
                torch.testing._internal.common_distributed,
                "get_timeout",
                custom_get_timeout,
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_stack.close()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        super().tearDown()

    @property
    def world_size(self) -> int:
        return int(os.getenv("TEST_WORLD_SIZE", "4"))

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(
            unittest.mock.patch.dict(
                _PROFILES,
                {"full": profile},
            )
        )
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=self.rank,
        )
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=60), dist.group.WORLD
        )
        torch.manual_seed(42 + self.rank)

    def _cleanup_process(self):
        self._exit_stack.close()
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_sync_seed(self):
        def _all_eq(xlist: list[Tensor]) -> bool:
            assert len(xlist) > 1
            lhs = xlist[0]
            return all(torch.allclose(lhs.cpu(), rhs.cpu()) for rhs in xlist[1:])

        self._init_process()
        torch.manual_seed(42 + self.rank)
        pg_name = dist.group.WORLD.group_name

        x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x, pg_name)

        self.assertFalse(_all_eq(xlist))

        with sync_seed(process_group_name=pg_name):
            x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x, pg_name)
        self.assertTrue(_all_eq(xlist))

        x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x, pg_name)
        self.assertFalse(_all_eq(xlist))

        self._cleanup_process()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize("autotuner", autotuner_names)
    def test_allreduce(self, autotuner):
        self._init_process()
        if autotuner == "fixed":
            kernel = helion.kernel(
                config=helion.Config(
                    block_sizes=[8192],
                    num_warps=32,
                ),
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(one_shot_allreduce_kernel)
            context = contextlib.nullcontext()
        elif autotuner == "FiniteSearch":
            kernel = helion.kernel(
                configs=[
                    helion.Config(block_sizes=[8192], num_warps=16),
                    helion.Config(block_sizes=[4096], num_warps=16),
                ],
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(one_shot_allreduce_kernel)
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )
        else:
            kernel = helion.kernel(
                one_shot_allreduce_kernel,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )

        with context:
            self.do_test_allreduce(kernel)

        self._cleanup_process()

    def do_test_allreduce(self, kernel):
        group = dist.group.WORLD

        N = 16384
        dtype = torch.bfloat16

        a_shared = symm_mem.empty(N, dtype=dtype, device=self.device).normal_()

        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=group)

        result = kernel(
            a_shared,
            symm_mem_hdl.rank,
            group.group_name,
            symm_mem_hdl.world_size,
        )

        torch.cuda.synchronize()

        expected = torch.empty_like(result).copy_(a_shared)
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "kernel_name",
        (
            "one_shot_allreduce_bias_rmsnorm_kernel",
            "two_shot_allreduce_bias_rmsnorm_kernel",
        ),
    )
    @parametrize("autotuner", autotuner_names)
    def test_allreduce_bias_rmsnorm(self, kernel_name, autotuner):
        """
        There is a similar test in test/test_examples_dist.py.
        The current test focus more on autotuning functionality.
        """
        self._init_process()
        mod = import_path(EXAMPLES_DIR / "distributed" / "allreduce_bias_rmsnorm.py")

        kernel = getattr(mod, kernel_name).fn
        if autotuner == "fixed":
            fixed_config = helion.Config(
                block_sizes=[8], num_warps=8, reduction_loops=[1024]
            )

            kernel = helion.kernel(
                config=fixed_config,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(kernel)
            context = contextlib.nullcontext()
        elif autotuner == "FiniteSearch":
            kernel = helion.kernel(
                configs=[
                    helion.Config(block_sizes=[8], num_warps=8),
                    helion.Config(block_sizes=[8], num_warps=4),
                ],
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(kernel)
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )
        else:
            kernel = helion.kernel(
                kernel, ignore_warnings=[helion.exc.TensorOperationInWrapper]
            )
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )

        with context:
            self.do_test_allreduce_bias_rmsnorm(
                kernel, mod.reference_allreduce_bias_rmsnorm
            )

        self._cleanup_process()

    def do_test_allreduce_bias_rmsnorm(self, kernel, ref_kernel):
        N, D = 128, 4096
        eps = 1e-5
        x = torch.randn(N, D, device=self.device)
        symm_mem_buffer = symm_mem.empty(N, D, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD.group_name)
        torch.manual_seed(42)
        bias = torch.randn(D, device=self.device)
        weight = torch.randn(D, device=self.device)

        result = kernel(
            symm_mem_buffer,
            x,
            bias,
            weight,
            symm_mem_hdl.signal_pad_ptrs_dev,
            eps,
            symm_mem_hdl.rank,
            symm_mem_hdl.world_size,
            dist.group.WORLD.group_name,
        )

        expected = ref_kernel(x, bias, weight)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize("autotuner", autotuner_names)
    def test_matmul_reduce_scatter(self, autotuner):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "matmul_reduce_scatter.py")

        kernel = mod.matmul_reduce_scatter_kernel.fn
        _SymmetricMemory.signal_pad_size = 1024 * 1024 * 16
        if autotuner == "fixed":
            # small block on purpose to test large grid
            fixed_config = helion.Config(
                block_sizes=[2, 2, 32],
                num_warps=8,
                num_stages=3,
                indexing="block_ptr",
            )

            kernel = helion.kernel(
                config=fixed_config,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(kernel)
            context = contextlib.nullcontext()
        elif autotuner == "FiniteSearch":
            kernel = helion.kernel(
                configs=[
                    helion.Config(
                        block_sizes=[64, 64, 32],
                        num_warps=8,
                        num_stages=3,
                        indexing="block_ptr",
                    ),
                    helion.Config(
                        block_sizes=[64, 64, 32],
                        num_warps=4,
                        num_stages=3,
                        indexing="block_ptr",
                    ),
                ],
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(kernel)
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )
        else:
            kernel = helion.kernel(
                kernel, ignore_warnings=[helion.exc.TensorOperationInWrapper]
            )
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )

        with context:
            self.do_test_matmul_reduce_scatter(
                kernel, mod.reference_matmul_reduce_scatter
            )
        self._cleanup_process()

    def do_test_matmul_reduce_scatter(self, kernel, ref_kernel):
        M, N, K = 512, 768, 1024

        torch.manual_seed(42 + self.rank)
        a = torch.randn(M, K, device=self.device)

        # Weight matrix is the same across all ranks
        torch.manual_seed(42)
        b = torch.randn(K, N, device=self.device)

        symm_mem_buffer = symm_mem.empty(M, N, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD.group_name)

        result = kernel(
            a,
            b,
            symm_mem_buffer,
            symm_mem_hdl.signal_pad_ptrs_dev,
            symm_mem_hdl.rank,  # RANK constexpr
            symm_mem_hdl.world_size,  # WORLD_SIZE constexpr
            dist.group.WORLD.group_name,  # GROUP_NAME constexpr
        )

        expected = ref_kernel(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_fp8_matmul_reduce_scatter(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FP8 requires CUDA compute capability >= 9.0")
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "fp8_matmul_reduce_scatter.py")

        _SymmetricMemory.signal_pad_size = 1024 * 1024 * 16
        M, N, K = 512, 768, 1024

        torch.manual_seed(42 + self.rank)
        a = torch.randn(M, K, device=self.device).to(torch.float8_e4m3fn)

        torch.manual_seed(42)
        b = (
            torch.randn(K, N, device=self.device)
            .to(torch.float8_e4m3fn)
            .t()
            .contiguous()
            .t()
        )

        scale_a = torch.rand(M, 1, device=self.device)
        scale_b = torch.rand(1, N, device=self.device)

        symm_mem_buffer = symm_mem.empty(M, N, dtype=torch.bfloat16, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD.group_name)

        result = mod.fp8_matmul_reduce_scatter_kernel(
            a,
            b,
            scale_a,
            scale_b,
            symm_mem_buffer,
            symm_mem_hdl.signal_pad_ptrs_dev,
            RANK=symm_mem_hdl.rank,
            WORLD_SIZE=symm_mem_hdl.world_size,
            GROUP_NAME=dist.group.WORLD.group_name,
        )

        expected = mod.reference_fp8_matmul_reduce_scatter(a, b, scale_a, scale_b)

        torch.testing.assert_close(result, expected, rtol=8e-1, atol=8e-1)
        self._cleanup_process()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_two_dim_parallel_matmul(self):
        self._init_process()
        mod = import_path(EXAMPLES_DIR / "distributed" / "two_dim_parallel_matmul.py")
        _SymmetricMemory.signal_pad_size = 1024 * 1024 * 16

        tp_size = 2
        sp_size = self.world_size // tp_size
        mesh = init_device_mesh(
            "cuda",
            (sp_size, tp_size),
            mesh_dim_names=("sp", "tp"),
        )
        tp_group = mesh.get_group("tp")
        sp_rank = self.rank // tp_size
        tp_rank = self.rank % tp_size

        kernel = mod.two_dim_parallel_matmul_kernel.fn
        kernel = helion.kernel(
            kernel, ignore_warnings=[helion.exc.TensorOperationInWrapper]
        )
        context = unittest.mock.patch.dict(
            os.environ, {"HELION_AUTOTUNER": "LFBOTreeSearch"}
        )

        with context:
            self.do_test_two_dim_parallel_matmul(
                kernel,
                mod.reference_two_dim_parallel_matmul,
                tp_group,
                sp_rank,
                tp_rank,
                tp_size,
                sp_size,
            )
        self._cleanup_process()

    def do_test_two_dim_parallel_matmul(
        self,
        helion_fn,
        ref_fn,
        tp_group,
        sp_rank,
        tp_rank,
        tp_size,
        sp_size,
    ):
        M, K, N = 512, 256, 512
        dtype = torch.float32

        M_local = M // sp_size
        K_local = K // tp_size

        torch.manual_seed(42 + sp_rank * tp_size + tp_rank)
        a_local = torch.randn(M_local, K_local, dtype=dtype, device=self.device)

        torch.manual_seed(42 + tp_rank)
        b_local = torch.randn(K_local, N, dtype=dtype, device=self.device)

        tp_group_name = tp_group.group_name  # type: ignore[missing-attribute]
        self.assertIsNotNone(tp_group_name)
        self.assertEqual(dist.get_world_size(tp_group), 2)
        self.assertEqual(dist.get_world_size(), 4)

        symm_mem_buf = symm_mem.empty(M_local, N, dtype=dtype, device=self.device)
        hdl = symm_mem.rendezvous(symm_mem_buf, tp_group_name)

        import unittest.mock

        with (
            unittest.mock.patch(
                "helion._dist_utils.sync_seed", wraps=sync_seed
            ) as mock_sync_seed,
            unittest.mock.patch(
                "helion.autotuner.base_search.all_gather_object",
                wraps=all_gather_object,
            ) as mock_all_gather_object,
            unittest.mock.patch(
                "helion.autotuner.base_search.sync_object", wraps=sync_object
            ) as mock_sync_object,
        ):
            result = helion_fn(
                a_local,
                b_local,
                symm_mem_buf,
                hdl.signal_pad_ptrs_dev,
                TP_RANK=hdl.rank,
                TP_SIZE=hdl.world_size,
                GROUP_NAME=tp_group_name,
            )

        def _assert_pgn_group_size_2(mock: unittest.mock.MagicMock) -> None:
            mock.assert_called()
            pg_names = dist.distributed_c10d._world.pg_names  # type: ignore[attr-defined]
            for call in mock.call_args_list:
                pgn = call.kwargs.get("process_group_name")
                self.assertIsNotNone(pgn)
                group = next(g for g, name in pg_names.items() if name == pgn)
                self.assertEqual(dist.get_world_size(group), 2)

        _assert_pgn_group_size_2(mock_sync_seed)
        _assert_pgn_group_size_2(mock_all_gather_object)
        _assert_pgn_group_size_2(mock_sync_object)

        expected = ref_fn(a_local, b_local, tp_group)
        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    run_tests()
