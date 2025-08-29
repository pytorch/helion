from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import run_tests

import helion
from helion._testing import code_and_output
import helion.language as hl


@helion.jit
def symm_mem_sync_kernel(
    remote_signal_pad_ptrs: torch.Tensor,
    local_signal_pad: torch.Tensor,
    rank: hl.constexpr,
) -> None:
    N, world_size = local_signal_pad.size()
    world_size = hl.specialize(world_size)
    for n in hl.grid(N):
        ptr_tile = remote_signal_pad_ptrs[:]
        multicast_signalpad = hl.multicast_like(local_signal_pad, ptr_tile)
        hl.signal(multicast_signalpad, [n, rank], signal=1, wait_for=0, scope="sys")
        for world in hl.tile(world_size, block_size=world_size):
            hl.wait(local_signal_pad, [n, world], signal=1, update=0, scope="sys")


@instantiate_parametrized_tests
class SymmMemBarrier(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        # world_size > 2 is needed to verify accumulation order
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
        torch.manual_seed(42 + self.rank)

    @skip_if_lt_x_gpu(4)
    def test_symm_mem_barrier(self):
        self._init_process()
        t = symm_mem.empty(4096, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
        local_signal_pad_t = symm_mem_hdl.get_signal_pad(
            symm_mem_hdl.rank, (32, symm_mem_hdl.world_size), dtype=torch.int32
        )
        signal_pad_pointers_t = torch.as_tensor(
            symm_mem_hdl.signal_pad_ptrs, dtype=torch.uint64
        ).to(self.device)

        code, result = code_and_output(
            symm_mem_sync_kernel,
            (
                signal_pad_pointers_t,
                local_signal_pad_t,
                symm_mem_hdl.rank,
            ),
        )

        signal_pad = symm_mem_hdl.get_signal_pad(symm_mem_hdl.rank)
        assert signal_pad.eq(0).all().item()

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
