"""Atomics in the body of a warp-specialized tcgen05 matmul.

The warp roles walk the tiles in loops of their own and the body's other
statements run outside them, once per CTA (``cute/atomic_ops.py``,
``reject_atomics_beside_warp_roles``).  A release / acquire atomic would need a
block-wide barrier between roles that wait on each other (a deadlock), and a
relaxed one counts once per CTA, so both are refused unless every tile has
exactly one CTA (not several tiles per persistent CTA, nor a two-CTA pair per
tile).  The refusals are checked on generated code only: nothing here launches
a kernel that could hang.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
import helion.language as hl

_PERSISTENT = {"block_sizes": [128, 128, 64], "pid_type": "persistent_blocked"}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_then_count(
    a: torch.Tensor, b: torch.Tensor, counter: torch.Tensor, sem: hl.constexpr
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty([m, n], device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        hl.atomic_add(counter, [0], 1, sem=sem)
    return out


def _args(size: int, sem: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    a = torch.randn(size, size, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=DEVICE, dtype=torch.bfloat16)
    return a, b, torch.zeros(1, device=DEVICE, dtype=torch.int32), sem


@onlyBackends(["cute"])
@skipIfCudaCapabilityLessThan((10, 0), reason="tcgen05 needs sm100")
class TestCuteTcgen05Atomics(TestCase):
    def test_ordered_atomic_beside_warp_roles_is_refused(self) -> None:
        for sem in ("release", "acquire", "acq_rel"):
            for config in (_PERSISTENT, {"block_sizes": [128, 128, 64]}):
                with (
                    self.subTest(sem=sem, config=str(config)),
                    self.assertRaisesRegex(
                        helion.exc.BackendUnsupported, "separate the warp roles"
                    ),
                ):
                    _matmul_then_count.bind(_args(512, sem)).to_triton_code(
                        helion.Config(**config)
                    )

    def test_atomic_in_multi_tile_persistent_body_is_refused(self) -> None:
        # 256 tiles on fewer SMs: each CTA walks several tiles.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "once per CTA, not once per tile"
        ):
            _matmul_then_count.bind(_args(2048, "relaxed")).to_triton_code(
                helion.Config(**_PERSISTENT)
            )

    def test_atomic_in_two_cta_body_is_refused(self) -> None:
        # A two-CTA pair shares each 256-row tile: one shot, but two CTAs per
        # tile, so a per-CTA count would come out twice the tile count.
        bound = _matmul_then_count.bind(_args(512, "relaxed"))
        config = helion.Config(
            block_sizes=[256, 128, 64],
            tcgen05_cluster_m=2,
            pid_type="persistent_blocked",
        )
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "two-CTA pairs share each tile: it would run once per CTA",
        ):
            bound.to_triton_code(config)

    def test_relaxed_atomic_counts_each_tile_when_ctas_own_one(self) -> None:
        for size, config in (
            (2048, {"block_sizes": [128, 128, 64]}),  # one tile per CTA
            (512, _PERSISTENT),  # one-shot persistent: 16 tiles
        ):
            with self.subTest(size=size, config=str(config)):
                args = _args(size, "relaxed")
                a, b, counter, _ = args
                bound = _matmul_then_count.bind(args)
                cfg = helion.Config(**config)
                self.assertIn("tcgen05", bound.to_triton_code(cfg))
                out = bound.compile_config(cfg)(*args)
                torch.testing.assert_close(out, a @ b, atol=1e-1, rtol=1e-2)
                self.assertEqual(counter.item(), (size // 128) ** 2)
