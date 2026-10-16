from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

import torch

import helion
from helion._compiler.autotuner_heuristics.cute import CuteTcgen05ClusterM2FfiHeuristic
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

# The flags ``BoundKernel.__init__`` derives from the traced host function.
# Compiler seeds (``compiler_seed_configs``) consult them, so they must be
# final before the seeds are computed.
_HOST_FUNCTION_FACT_FLAGS = (
    "cute_tcgen05_aux_kernel_detected",
    "cute_tcgen05_exact_shape_aux_kernel_detected",
    "cute_tcgen05_matmul_has_non_tcgen05_operand",
    "cute_tcgen05_rowvec_aux_facts",
)


def _make_plain_gemm() -> helion.Kernel:
    """A fresh kernel per bind: ``Kernel.bind`` memoizes by argument key."""

    @helion.kernel(backend="cute", static_shapes=True)
    def _plain_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        _, n = y.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(x.dtype)
        return out

    return _plain_gemm


@helion.kernel(backend="cute", static_shapes=True)
def _residual_gemm(
    x: torch.Tensor, y: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + residual[tile_m, tile_n]).to(x.dtype)
    return out


@helion.kernel(
    backend="cute",
    static_shapes=True,
    cute_materialize_transformed_operands=False,
    cute_region_fission=False,
)
def _bf16xint16_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = w.shape
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            w_tile = w[tile_k, tile_n].to(torch.bfloat16)
            acc = hl.dot(x[tile_m, tile_k], w_tile, acc=acc)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


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


def _direct_entry_seeds(
    bound: helion.runtime.kernel.BoundKernel,
) -> list[helion.Config]:
    return [
        config
        for config in bound.config_spec.compiler_seed_configs
        if config.config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY)
        or config.config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY)
    ]


@onlyBackends(["cute"])
class TestCuteTcgen05SeedFlagOrdering(TestCase):
    def _bind_recording_seed_time_flags(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
    ) -> tuple[helion.runtime.kernel.BoundKernel, dict[str, object]]:
        """Bind ``kernel`` and snapshot the host-function flags as the seeds see them."""
        kernel_module = sys.modules["helion.runtime.kernel"]
        real_compiler_seed_configs: Callable[..., list[helion.Config]] = (
            kernel_module.compiler_seed_configs
        )
        seed_time_flags: dict[str, object] = {}

        def recording_compiler_seed_configs(
            env: object, device_ir: object
        ) -> list[helion.Config]:
            assert not seed_time_flags
            spec = env.config_spec  # pyrefly: ignore[missing-attribute]
            for flag in _HOST_FUNCTION_FACT_FLAGS:
                seed_time_flags[flag] = getattr(spec, flag)
            return real_compiler_seed_configs(env, device_ir)

        with (
            _blackwell_bind_patches(),
            patch.object(
                kernel_module,
                "compiler_seed_configs",
                recording_compiler_seed_configs,
            ),
        ):
            bound = kernel.bind(args)
        self.assertEqual(set(seed_time_flags), set(_HOST_FUNCTION_FACT_FLAGS))
        return bound, seed_time_flags

    def _assert_flags_final_at_seed_time(
        self, kernel: helion.Kernel, args: tuple[object, ...]
    ) -> helion.runtime.kernel.BoundKernel:
        bound, seed_time_flags = self._bind_recording_seed_time_flags(kernel, args)
        final_flags = {
            flag: getattr(bound.config_spec, flag) for flag in _HOST_FUNCTION_FACT_FLAGS
        }
        self.assertEqual(seed_time_flags, final_flags)
        return bound

    def test_aux_kernel_flags_final_before_compiler_seeds(self) -> None:
        """The aux-store detectors run before the compiler seeds read them.

        The exact-shape residual epilogue sets all three aux facts; a seed
        computed while they still held their defaults would be built for a
        plain matmul.
        """
        args = (
            torch.empty([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([1024, 4096], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([1024, 4096], device=DEVICE, dtype=torch.bfloat16),
        )
        bound = self._assert_flags_final_at_seed_time(_residual_gemm, args)
        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_aux_kernel_detected)
        self.assertTrue(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertIsNotNone(spec.cute_tcgen05_rowvec_aux_facts)

    def test_non_tcgen05_operand_flag_final_before_compiler_seeds(self) -> None:
        """The bf16 x int16 GEMM's cast operand is known when seeds are computed.

        Operand materialization is disabled so the int16 load feeds the dot
        directly, which is the shape the non-tcgen05 operand detector exists
        for.
        """
        args = (
            torch.empty([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([1024, 4096], device=DEVICE, dtype=torch.int16),
        )
        bound = self._assert_flags_final_at_seed_time(_bf16xint16_gemm, args)
        self.assertTrue(bound.config_spec.cute_tcgen05_matmul_has_non_tcgen05_operand)
        self.assertEqual(_direct_entry_seeds(bound), [])

    def test_non_tcgen05_operand_keeps_direct_entry_seed_out_of_compiler_seeds(
        self,
    ) -> None:
        """A non-tcgen05 matmul operand gates the FFI seed at seed time.

        ``full_tile_direct_entry_seed_eligible`` rejects such kernels, and the
        FFI heuristic evaluates it inside ``compiler_seed_configs``. The flag
        has to be set by then: on a shape that is otherwise eligible for the
        flat-role / TVM-FFI direct-entry seed, no compiler seed may carry the
        direct-entry knobs and the heuristic must not fire.
        """
        args = (
            torch.empty([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([1024, 4096], device=DEVICE, dtype=torch.bfloat16),
        )
        with _blackwell_bind_patches():
            eligible = _make_plain_gemm().bind(args)
        self.assertTrue(
            eligible.config_spec._tcgen05_full_tile_direct_entry_seed_eligible()
        )
        self.assertIn(
            CuteTcgen05ClusterM2FfiHeuristic.name,
            eligible.config_spec.autotuner_heuristics,
        )
        self.assertEqual(len(_direct_entry_seeds(eligible)), 1)

        with (
            _blackwell_bind_patches(),
            patch(
                "helion.runtime.kernel.host_function_matmul_has_non_tcgen05_operand",
                return_value=True,
            ),
        ):
            bound = _make_plain_gemm().bind(args)
        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_matmul_has_non_tcgen05_operand)
        self.assertFalse(spec._tcgen05_full_tile_direct_entry_seed_eligible())
        self.assertNotIn(
            CuteTcgen05ClusterM2FfiHeuristic.name, spec.autotuner_heuristics
        )
        self.assertEqual(_direct_entry_seeds(bound), [])
        default = spec.default_config().config
        self.assertIsNot(default.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY), True)
        self.assertIsNot(default.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY), True)


if __name__ == "__main__":
    unittest.main()
