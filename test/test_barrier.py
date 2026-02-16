from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.exc as exc
import helion.language as hl


@helion.kernel()
def barrier_dep_single(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for t in hl.tile(x.size(0)):
        tmp[t] = x[t] * 2

    hl.barrier()

    for t in hl.tile(x.size(0)):
        out[t] = tmp[t] + 1

    return out


@helion.kernel()
def barrier_multiple(x: torch.Tensor) -> torch.Tensor:
    buf1 = torch.empty_like(x)
    buf2 = torch.empty_like(x)
    out = torch.empty_like(x)

    for t in hl.tile(x.size(0)):
        buf1[t] = x[t] + 3

    hl.barrier()

    for t in hl.tile(x.size(0)):
        buf2[t] = buf1[t] * 2

    hl.barrier()

    for t in hl.tile(x.size(0)):
        out[t] = buf2[t] - 5

    return out


@helion.kernel()
def barrier_groups(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    buf = torch.empty_like(x)
    buf2 = torch.empty_like(x)
    out = torch.empty_like(x)

    # group 1: independent loops
    for t in hl.tile(x.size(0)):
        buf[t] = x[t] + 1
    for t in hl.tile(x.size(0)):
        buf2[t] = y[t] + 5

    hl.barrier()

    # group 2: consumes both buffers
    for t in hl.tile(x.size(0)):
        out[t] = (buf[t] + buf2[t]) * 2

    hl.barrier()

    for t in hl.tile(x.size(0)):
        out[t] = out[t] + 7

    return out


@onlyBackends(["triton"])
class TestBarrier(RefEagerTestBase, TestCase):
    @skipIfTileIR("TileIR does not support barrier operations")
    def test_dep_across_barrier(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            barrier_dep_single,
            (x,),
            block_sizes=[8, 8],
            pid_type="persistent_blocked",
        )
        expected = x * 2 + 1
        torch.testing.assert_close(out, expected)
        self.assertExpectedJournal(code)

    @skipIfTileIR("TileIR does not support barrier operations")
    def test_multiple_barriers(self) -> None:
        x = torch.arange(6, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            barrier_multiple,
            (x,),
            block_sizes=[8, 8, 8],
            pid_type="persistent_blocked",
        )
        expected = (x + 3) * 2 - 5
        torch.testing.assert_close(out, expected)
        self.assertExpectedJournal(code)

    @skipIfTileIR("TileIR does not support barrier operations")
    def test_multiple_loops_between_barriers(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        y = torch.arange(8, device=DEVICE, dtype=torch.float32) * 3
        code, out = code_and_output(
            barrier_groups,
            (x, y),
            block_sizes=[8, 8, 8, 8],
            pid_type="persistent_blocked",
        )
        expected = ((x + 1) + (y + 5)) * 2 + 7
        torch.testing.assert_close(out, expected)
        self.assertExpectedJournal(code)

    @skipIfRefEager("pid_type validation is only enforced in compiled mode")
    def test_non_persistent_pid_type_errors(self) -> None:
        x = torch.arange(4, device=DEVICE, dtype=torch.float32)
        with self.assertRaisesRegex(exc.BarrierRequiresPersistent, "requires pid_type"):
            code_and_output(
                barrier_dep_single,
                (x,),
                block_sizes=[4, 4],
                pid_type="flat",
            )

    @skipIfTileIR("TileIR does not support barrier operations")
    def test_default_config_is_persistent(self) -> None:
        x = torch.arange(4, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            barrier_dep_single,
            (x,),
            block_sizes=[4, 4],
            pid_type="persistent_blocked",
        )
        expected = x * 2 + 1
        torch.testing.assert_close(out, expected)
        # Can't see pid_type in ref-mode code; rely on normalization to succeed.
