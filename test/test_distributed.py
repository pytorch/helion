from __future__ import annotations

from datetime import timedelta

import torch
from torch import Tensor
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU
from helion._testing import import_path
from helion._testing import EXAMPLES_DIR
from helion._utils import all_gather_object
from helion._utils import sync_seed


import torch
import torch.distributed as dist
import contextlib
import unittest
import os
from torch.testing._internal.common_distributed import MultiProcessTestCase
from datetime import timedelta
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import run_tests
from helion._testing import onlyBackends
from torch.utils.cpp_extension import load_inline
import helion.language as hl
import helion
from helion._utils import sync_seed
from helion._testing import TestCase
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU
from helion._testing import code_and_output
import torch.distributed._symmetric_memory as symm_mem
from helion.autotuner import search_algorithms

# ["fixed", "PatternSearch", "LFBOPatternSearch", "LFBOTreeSearch", "DifferentialEvolutionSearch", "DESurrogateHybrid", "FiniteSearch", "RandomSearch"]
autotuner_names = ["fixed", *search_algorithms]

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


@onlyBackends(["triton"])
@instantiate_parametrized_tests
class TestDistributed(TestCase, MultiProcessTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._class_stack = contextlib.ExitStack()
        cls._class_stack.enter_context(unittest.mock.patch.dict(os.environ, {
            "CHECK_CONFIG_CONSISTANCY": "1",
        }))

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
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_sync_seed(self):
        def _all_eq(xlist: list[Tensor]) -> bool:
            assert len(xlist) > 1
            lhs = xlist[0]
            return all(torch.allclose(lhs.cpu(), rhs.cpu()) for rhs in xlist[1:])

        self._init_process()
        torch.manual_seed(42 + self.rank)

        x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x)

        self.assertFalse(_all_eq(xlist))

        with sync_seed():
            x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x)
        self.assertTrue(_all_eq(xlist))

        x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x)
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
            )(one_shot_allreduce_kernel)
            context = contextlib.nullcontext()
        elif autotuner == "FiniteSearch":
            kernel = helion.kernel(
                configs=[
                    helion.Config(
                        block_sizes=[8192],
                        num_warps=16
                    ),
                    helion.Config(
                        block_sizes=[4096],
                        num_warps=16
                    ),
                ]
            )(one_shot_allreduce_kernel)
            context = unittest.mock.patch.dict(os.environ, {"HELION_AUTOTUNER": autotuner})
        else:
            kernel = helion.kernel(one_shot_allreduce_kernel)
            context = unittest.mock.patch.dict(os.environ, {"HELION_AUTOTUNER": autotuner})

        with context:
            self.do_test_allreduce(kernel)

        self._cleanup_process()

    def do_test_allreduce(self, kernel):
        group = dist.group.WORLD

        N = 16384
        dtype = torch.bfloat16

        a_shared = symm_mem.empty(
            N, dtype=dtype, device=self.device
        ).normal_()

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
            # "one_shot_allreduce_bias_rmsnorm_kernel",
            "two_shot_allreduce_bias_rmsnorm_kernel",
        ),
    )
    # @parametrize("autotuner", ["DifferentialEvolutionSearch", "DESurrogateHybrid", "RandomSearch"])
    # TODO enable all autotuners
    @parametrize("autotuner", ["LFBOTreeSearch"])
    @unittest.mock.patch.dict(os.environ, {"CHECK_INPUT_ACCURACY": "0"}) # TODO get rid of this
    def test_allreduce_bias_rmsnorm(self, kernel_name, autotuner):
        """
        There is a similar test in test/test_examples_dist.py.
        The current test focus more on autotuning funcionality.
        """
        self._init_process()
        mod = import_path(EXAMPLES_DIR / "distributed" / "allreduce_bias_rmsnorm.py")

        kernel = getattr(mod, kernel_name).fn
        if autotuner == "fixed":
            kernel = helion.kernel(
                config=helion.Config(
                    block_sizes=[8],
                    num_warps=8,
                ),
            )(kernel)
            context = contextlib.nullcontext()
        elif autotuner == "FiniteSearch":
            kernel = helion.kernel(
                configs=[
                    helion.Config(
                        block_sizes=[8],
                        num_warps=8
                    ),
                    helion.Config(
                        block_sizes=[8],
                        num_warps=4
                    ),
                ]
            )(kernel)
            context = unittest.mock.patch.dict(os.environ, {"HELION_AUTOTUNER": autotuner})
        else:
            kernel = helion.kernel(kernel)
            context = unittest.mock.patch.dict(os.environ, {"HELION_AUTOTUNER": autotuner})

        with context:
            self.do_test_allreduce_bias_rmsnorm(kernel, mod.reference_allreduce_bias_rmsnorm)

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
    # @parametrize("autotuner", ["DifferentialEvolutionSearch", "DESurrogateHybrid", "RandomSearch"])
    # TODO enable all autotuners
    @parametrize("autotuner", ["LFBOTreeSearch"])
    @unittest.mock.patch.dict(os.environ, {"CHECK_INPUT_ACCURACY": "0"}) # TODO get rid of this
    def test_matmul_reduce_scatter(self, autotuner):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "matmul_reduce_scatter.py")

        kernel = mod.matmul_reduce_scatter_kernel.fn
        if autotuner == "fixed":
            kernel = helion.kernel(
                config=helion.Config(
                    block_sizes=[64, 64, 32],
                    num_warps=8,
                    num_stages=3,
                    indexing="block_ptr",
                ),
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
                ]
            )(kernel)
            context = unittest.mock.patch.dict(os.environ, {"HELION_AUTOTUNER": autotuner})
        else:
            kernel = helion.kernel(kernel)
            context = unittest.mock.patch.dict(os.environ, {"HELION_AUTOTUNER": autotuner})

        with context:
            self.do_test_matmul_reduce_scatter(kernel, mod.reference_matmul_reduce_scatter)
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
            symm_mem_hdl.signal_pad_ptrs_dev,
            symm_mem_buffer,
            symm_mem_hdl.rank,  # RANK constexpr
            symm_mem_hdl.world_size,  # WORLD_SIZE constexpr
            dist.group.WORLD.group_name,  # GROUP_NAME constexpr
        )

        expected = ref_kernel(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    run_tests()
