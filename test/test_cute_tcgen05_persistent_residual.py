"""Statements beside a persistent tcgen05 matmul's warp roles.

With a persistent pid_type the TMA, MMA and epilogue roles each walk the tiles
in a role-local loop; any other observable statement of the tile body (even
one constant store) would run in a second, shared work-tile loop after them,
and that launch hangs.  Codegen refuses it (``Tcgen05PersistentProgramIDs``);
the refusal is checked on generated code only, so nothing here can hang, and
the flat pid_type still runs the same kernel exactly.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_and_side_store(
    a: torch.Tensor, b: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        side[tile_m.id, tile_n.id] = 1.0
    return out


def _args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    b = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    return a, b, torch.zeros(8, 8, device=DEVICE)


def _config(bound: helion.runtime.kernel.BoundKernel, pid_type: str) -> helion.Config:
    with bound.env:
        values = dict(bound.config_spec.default_config().config)
    values.update(block_sizes=[128, 64, 128], pid_type=pid_type)
    if pid_type != "flat":
        values.update(tcgen05_persistence_model="static_persistent")
    return helion.Config(**values)


@onlyBackends(["cute"])
@skipIfCudaCapabilityLessThan((10, 0), reason="tcgen05 needs sm100")
class TestCuteTcgen05PersistentResidual(TestCase):
    def test_statement_beside_persistent_roles_is_refused(self) -> None:
        bound = _matmul_and_side_store.bind(_args())
        for pid_type in ("persistent_blocked", "persistent_interleaved"):
            with (
                self.subTest(pid_type=pid_type),
                self.assertRaisesRegex(
                    helion.exc.BackendUnsupported,
                    "beside the warp roles of a persistent tcgen05 matmul",
                ),
            ):
                bound.to_triton_code(_config(bound, pid_type))

    def test_flat_pid_runs_the_statement_exactly(self) -> None:
        args = _args()
        a, b, side = args
        bound = _matmul_and_side_store.bind(args)
        config = _config(bound, "flat")
        self.assertIn("tcgen05", bound.to_triton_code(config))
        out = bound.compile_config(config)(*args)
        torch.testing.assert_close(out, a.float() @ b.float(), atol=5e-2, rtol=1e-2)
        torch.testing.assert_close(side[:2, :4], torch.ones(2, 4, device=DEVICE))
        self.assertEqual(int(side.count_nonzero()), 8)
