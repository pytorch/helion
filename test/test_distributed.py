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
from helion._utils import all_gather_object
from helion._utils import sync_seed


@onlyBackends(["triton"])
class TestDistributed(TestCase, MultiProcessTestCase):
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


if __name__ == "__main__":
    run_tests()
