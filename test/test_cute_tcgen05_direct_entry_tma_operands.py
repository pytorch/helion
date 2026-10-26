from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

from examples.broadcast_matmul import broadcast_matmul
from examples.grouped_gemm import grouped_gemm_jagged
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_shared_rhs_grouped import _bind as _bind_shared_rhs_grouped
from test.test_cute_shared_rhs_grouped import _inputs as _shared_rhs_grouped_inputs

import helion
from helion._compiler.autotuner_heuristics.cute import CuteTcgen05ClusterM2FfiHeuristic
from helion._compiler.cute.cute_mma import _CuteMmaNode
from helion._compiler.cute.cute_mma import _tcgen05_grouped_rhs_keeps_store_protocol
from helion._compiler.cute.cute_mma import _tcgen05_mma_output_tma_store_provable
from helion._compiler.cute.cute_mma import analyze_cute_mma_node
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._hardware import HardwareInfo
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


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


def _make_ordinary_gemm_into() -> helion.Kernel:
    """A fresh kernel per bind: ``Kernel.bind`` memoizes by argument key."""

    @helion.kernel(backend="cute", static_shapes=True)
    def _ordinary_gemm_into(
        a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        m, k = a.shape
        _, n = b.shape
        for tile_m, tile_n in hl.tile((m, n)):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(out.dtype)
        return out

    return _ordinary_gemm_into


def _direct_entry_seeds(
    bound: helion.runtime.kernel.BoundKernel,
) -> list[helion.Config]:
    return [
        config
        for config in bound.config_spec.compiler_seed_configs
        if config.config.get("tcgen05_tvm_ffi_launch")
        or config.config.get("tcgen05_flat_role_coordinates")
    ]


def _output_destinations(
    m: int, n: int, dtype: torch.dtype
) -> dict[str, tuple[torch.Tensor, bool]]:
    """Output tensors for an ``m x n`` GEMM paired with their TMA-store legality.

    Only the plain row-major allocation satisfies the TensorMap proof the
    direct-entry TMA store epilogue needs: an outer stride that is not a
    16-byte multiple, a base offset of 8 bytes, or an N-major view all take
    the SIMT store body, which rejects the flat-role / TVM-FFI seed.
    """
    return {
        "aligned": (torch.empty([m, n], device=DEVICE, dtype=dtype), True),
        "under_aligned_stride": (
            torch.empty([m, n + 4], device=DEVICE, dtype=dtype)[:, :n],
            False,
        ),
        "n_major": (torch.empty([n, m], device=DEVICE, dtype=dtype).t(), False),
        "offset_base": (
            torch.empty([m * n + 4], device=DEVICE, dtype=dtype)[4:].view(m, n),
            False,
        ),
    }


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

    def _assert_output_destination_default_runs(
        self, out: torch.Tensor, *, tma_store_legal: bool
    ) -> None:
        a = torch.randn([8192, 768], device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn([768, 1024], device=DEVICE, dtype=torch.bfloat16)
        bound = _make_ordinary_gemm_into().bind((a, b, out))
        spec = bound.config_spec
        self.assertEqual(
            spec.cute_tcgen05_matmul_operands_tma_provable, tma_store_legal
        )
        self.assertEqual(
            spec._tcgen05_full_tile_direct_entry_seed_eligible(), tma_store_legal
        )
        config = spec.default_config()
        self.assertEqual(
            config.config.get("tcgen05_tvm_ffi_launch") is True, tma_store_legal
        )
        self.assertEqual(
            config.config.get("tcgen05_flat_role_coordinates") is True,
            tma_store_legal,
        )
        bound.set_config(config)
        result = bound(a, b, out)
        torch.testing.assert_close(result, torch.matmul(a, b), rtol=1e-2, atol=1e-1)

    def test_unaligned_output_stride_default_config_runs(self) -> None:
        """An under-aligned output must not select the TMA-store direct-entry seed.

        The 8192x768x1024 GEMM is shape-eligible for the flat-role / TVM-FFI
        seed and its A/B operands pass the TensorMap proof, but a 1028-element
        bf16 output row stride (2056 bytes) fails it for the destination, so
        codegen takes the SIMT store body and rejects the seed config. The
        default config has to fall back to a compiling config and produce
        correct results.
        """
        out = torch.empty([8192, 1028], device=DEVICE, dtype=torch.bfloat16)[:, :1024]
        self._assert_output_destination_default_runs(out, tma_store_legal=False)

    def test_n_major_output_default_config_runs(self) -> None:
        """An N-major output must not select the TMA-store direct-entry seed.

        The plain matmul stages its D tile row-major, so a column-major
        destination fails the TMA-store proof even though its base and outer
        stride are aligned.
        """
        out = torch.empty([1024, 8192], device=DEVICE, dtype=torch.bfloat16).t()
        self._assert_output_destination_default_runs(out, tma_store_legal=False)


@contextlib.contextmanager
def _blackwell_bind_patches() -> Iterator[None]:
    with (
        patch_cute_mma_support(),
        patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


@onlyBackends(["cute"])
class TestCuteTcgen05DirectEntryTmaOutputProof(unittest.TestCase):
    """Bind-time output-destination proof on a patched Blackwell target."""

    def test_output_destination_gates_direct_entry_seed(self) -> None:
        a = torch.empty([8192, 768], device=DEVICE, dtype=torch.bfloat16)
        b = torch.empty([768, 1024], device=DEVICE, dtype=torch.bfloat16)
        for name, (out, tma_store_legal) in _output_destinations(
            8192, 1024, torch.bfloat16
        ).items():
            with self.subTest(output=name), _blackwell_bind_patches():
                bound = _make_ordinary_gemm_into().bind((a, b, out))
                spec = bound.config_spec
                self.assertEqual(
                    spec.cute_tcgen05_matmul_operands_tma_provable, tma_store_legal
                )
                self.assertEqual(
                    spec._tcgen05_full_tile_direct_entry_seed_eligible(),
                    tma_store_legal,
                )
                self.assertEqual(
                    CuteTcgen05ClusterM2FfiHeuristic.name in spec.autotuner_heuristics,
                    tma_store_legal,
                )
                self.assertEqual(len(_direct_entry_seeds(bound)), int(tma_store_legal))
                default = spec.default_config().config
                self.assertEqual(
                    default.get("tcgen05_tvm_ffi_launch") is True, tma_store_legal
                )

    def _single_mma_candidate(
        self, bound: helion.runtime.kernel.BoundKernel
    ) -> tuple[torch.fx.Node, _CuteMmaNode]:
        device_ir = bound.host_function.device_ir
        with bound.env, bound.host_function:
            candidates = [
                (node, candidate)
                for graph_info in device_ir.graphs
                for node in graph_info.graph.nodes
                if (candidate := analyze_cute_mma_node(node, device_ir=device_ir))
                is not None
            ]
        self.assertEqual(len(candidates), 1)
        return candidates[0]

    def _assert_output_proof_consulted(
        self, bound: helion.runtime.kernel.BoundKernel, *, consulted: bool
    ) -> None:
        """Whether the bind-time output proof evaluates the store destination.

        The kernels allocate their output, so the destination is aligned and
        the proof holds; forcing the destination check to fail shows whether
        the grouped form is exempt from it.
        """
        node, candidate = self._single_mma_candidate(bound)
        self.assertEqual(
            _tcgen05_grouped_rhs_keeps_store_protocol(candidate.operands.rhs),
            not consulted,
        )
        device_ir = bound.host_function.device_ir
        with bound.env, bound.host_function:
            self.assertTrue(
                _tcgen05_mma_output_tma_store_provable(
                    bound.env, node, candidate, device_ir.graphs
                )
            )
            with patch(
                "helion._compiler.cute.cute_mma._tcgen05_tma_destination_is_legal",
                return_value=False,
            ):
                self.assertEqual(
                    _tcgen05_mma_output_tma_store_provable(
                        bound.env, node, candidate, device_ir.graphs
                    ),
                    not consulted,
                )

    def test_shared_rhs_packed_group_keeps_output_proof(self) -> None:
        """The shared rank-2 RHS stores through the plain epilogue.

        Its RHS is grouped (a packed group over the shared weight) but not the
        rank-3 N,K-major form, so codegen proves its destination like a plain
        GEMM's; the bind-time proof must not exempt it.
        """
        with _mock_cuda_unavailable():
            bound = _bind_shared_rhs_grouped(
                grouped_gemm_jagged.fn, _shared_rhs_grouped_inputs()
            )
        self.assertTrue(bound.config_spec.cute_tcgen05_search_enabled)
        _node, candidate = self._single_mma_candidate(bound)
        rhs = candidate.operands.rhs
        self.assertTrue(rhs.rhs_is_grouped)
        self.assertFalse(rhs.rhs_rank3_grouped_nt)
        self.assertIsNotNone(rhs.rhs_packed_group)
        self.assertIsNone(rhs.rhs_segment_group)
        self.assertTrue(bound.config_spec.cute_tcgen05_matmul_operands_tma_provable)
        self._assert_output_proof_consulted(bound, consulted=True)

    def test_rank3_nk_major_grouped_rhs_is_exempt_from_output_proof(self) -> None:
        """The rank-3 N,K-major grouped RHS keeps its own store protocol."""

        @helion.kernel(backend="cute", static_shapes=True)
        def grouped_worklist(
            a_packed: torch.Tensor, b_grouped: torch.Tensor, worklist: torch.Tensor
        ) -> torch.Tensor:
            m_total, k = a_packed.shape
            _groups, n, k2 = b_grouped.shape
            assert k == k2
            block_m = hl.register_block_size(256)
            block_n = hl.register_block_size(128)
            block_k = hl.register_block_size(64, 128)
            out = torch.empty(
                [m_total, n], dtype=a_packed.dtype, device=a_packed.device
            )
            for work_tile, tile_m, tile_n in hl.tile(
                [worklist.size(0), 256, n],
                block_size=[1, block_m, block_n],
            ):
                work_id = work_tile.begin
                group_id = worklist[work_id, 0]
                start = worklist[work_id, 1]
                valid_m = worklist[work_id, 2]
                store_m = worklist[work_id, 3]
                local_m = tile_m.index
                row = start + local_m
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k, block_size=block_k):
                    a_block = hl.load(
                        a_packed,
                        [row, tile_k],
                        extra_mask=(local_m < valid_m)[:, None],  # pyrefly: ignore[bad-index]
                    )
                    acc = torch.addmm(
                        acc, a_block, b_grouped[group_id, tile_n, tile_k].T
                    )
                hl.store(
                    out,
                    [row, tile_n],
                    acc.to(out.dtype),
                    extra_mask=(local_m < store_m)[:, None],  # pyrefly: ignore[bad-index]
                )
            return out

        groups, tiles_per_group = 4, 2
        a = torch.empty(
            (groups * tiles_per_group * 256, 128), device=DEVICE, dtype=torch.bfloat16
        )
        b = torch.empty((groups, 256, 128), device=DEVICE, dtype=torch.bfloat16)
        rows = [
            [group, (group * tiles_per_group + tile) * 256, 256, 256]
            for group in range(groups)
            for tile in range(tiles_per_group)
        ]
        worklist = torch.tensor(rows, device=DEVICE, dtype=torch.int32)
        with (
            patch_cute_mma_support(),
            patch(
                "helion._compiler.autotuner_heuristics.cute."
                "tcgen05_runtime_n_ptx_compatible",
                return_value=True,
            ),
            patch(
                "helion._compiler.cute.cutedsl_compat.check_cute_backend_requirements"
            ),
            patch(
                "helion._hardware.get_hardware_info",
                return_value=HardwareInfo(
                    device_kind="cuda",
                    hardware_name="NVIDIA B200",
                    runtime_version="12.8",
                    compute_capability="sm100",
                ),
            ),
        ):
            bound = grouped_worklist.bind((a, b, worklist))
        _node, candidate = self._single_mma_candidate(bound)
        self.assertTrue(candidate.operands.rhs.rhs_rank3_grouped_nt)
        self._assert_output_proof_consulted(bound, consulted=False)


if __name__ == "__main__":
    unittest.main()
