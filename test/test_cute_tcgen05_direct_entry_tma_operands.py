from __future__ import annotations

import unittest

from examples.broadcast_matmul import broadcast_matmul
import torch

import helion
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _ordinary_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@onlyBackends(["cute"])
class TestCuteTcgen05DirectEntryTmaOperands(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")
        torch.manual_seed(0)

    def test_unaligned_stride_lhs_default_config_runs(self) -> None:
        """An unaligned input must not select the TMA-only direct-entry seed.

        The 8192x768x1024 GEMM is shape-eligible for the flat-role / TVM-FFI
        seed, but a 772-element bf16 row stride (1544 bytes) fails the
        TensorMap alignment proof, so codegen keeps the scalar SMEM producers
        and rejects the seed config. The default config has to fall back to a
        compiling cluster_m=1 config and produce correct results.
        """
        a = torch.randn([8192, 772], device=DEVICE, dtype=torch.bfloat16)[:, :768]
        b = torch.randn([768, 1024], device=DEVICE, dtype=torch.bfloat16)
        bound = _ordinary_gemm.bind((a, b))
        spec = bound.config_spec
        self.assertFalse(spec.cute_tcgen05_matmul_operands_tma_provable)
        self.assertFalse(spec._tcgen05_full_tile_direct_entry_seed_eligible())
        config = spec.default_config()
        self.assertIsNot(config.config.get("tcgen05_tvm_ffi_launch"), True)
        self.assertIsNot(config.config.get("tcgen05_flat_role_coordinates"), True)
        self.assertEqual(config.config["tcgen05_cluster_m"], 1)
        bound.set_config(config)
        out = bound(a, b)
        torch.testing.assert_close(out, torch.matmul(a, b), rtol=1e-2, atol=1e-1)

    def test_reshape_view_lhs_keeps_direct_entry_seed(self) -> None:
        """A pointer-preserving host view keeps the direct-entry seed.

        ``broadcast_matmul`` flattens ``x`` with ``x.reshape([b * m, k])``;
        codegen proves the view's TMA descriptor through the input's recorded
        base alignment, and the bind-time proof must agree so the validated
        FFI default is kept and compiles.
        """
        x = torch.randn([16, 512, 768], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([768, 1024], device=DEVICE, dtype=torch.bfloat16)
        bound = broadcast_matmul.bind((x, w))
        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_matmul_operands_tma_provable)
        self.assertTrue(spec._tcgen05_full_tile_direct_entry_seed_eligible())
        config = spec.default_config()
        self.assertIs(config.config.get("tcgen05_tvm_ffi_launch"), True)
        self.assertIs(config.config.get("tcgen05_flat_role_coordinates"), True)
        bound.set_config(config)
        out = bound(x, w)
        torch.testing.assert_close(out, torch.matmul(x, w), rtol=1e-2, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
