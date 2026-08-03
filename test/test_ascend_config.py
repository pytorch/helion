from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Callable
from typing import Iterator
from unittest.mock import patch

import torch

from helion._compiler.ascend.config import _npu_cap_reduction_loops
from helion._compiler.ascend.config import _npu_default_reduction_loop
from helion._compiler.ascend.config import _npu_max_tensor_numel
from helion._compiler.ascend.config import _npu_ub_budget_elements
from helion._compiler.backend import TritonBackend
from helion._testing import TestCase
from helion._testing import skipIfFn
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import get_valid_eviction_policies


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def skipUnlessNPU(reason: str) -> Callable[[Callable], Callable]:
    """Skip a test unless an Ascend NPU is available (deferred to run time)."""
    return skipIfFn(lambda: not _npu_available(), reason)


@contextmanager
def _force_npu_absent() -> Iterator[None]:
    """Treat NPU as absent for the duration of the block.

    On hosts with torch_npu installed, mock ``torch.npu.is_available`` to False;
    on hosts without torch_npu, NPU is already absent (``hasattr(torch, "npu")``
    is False) so no mocking is needed. Keeps the non-NPU regression tests
    portable across CUDA-only and NPU-capable hosts.
    """
    if hasattr(torch, "npu"):
        with patch("torch.npu.is_available", return_value=False):
            yield
    else:
        yield


class TestNPUConfigHelpers(TestCase):
    """The env-tunable NPU caps in ascend/config.py parse defaults, overrides, and bad input.

    These are pure env parsing -- no NPU hardware required -- so they run everywhere
    and guard the config-layer tolerance added by the Ascend port.
    """

    def test_ub_budget_default_and_override(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_NPU_UB_BUDGET_ELEMENTS", None)
            self.assertEqual(_npu_ub_budget_elements(), 2048)
            with patch.dict(os.environ, {"HELION_NPU_UB_BUDGET_ELEMENTS": "512"}):
                self.assertEqual(_npu_ub_budget_elements(), 512)

    def test_ub_budget_invalid_falls_back(self) -> None:
        with patch.dict(os.environ, {"HELION_NPU_UB_BUDGET_ELEMENTS": "notanint"}):
            self.assertEqual(_npu_ub_budget_elements(), 2048)

    def test_max_tensor_numel_default_and_override(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_NPU_MAX_TENSOR_NUMEL", None)
            self.assertEqual(_npu_max_tensor_numel(), 8192)
            with patch.dict(os.environ, {"HELION_NPU_MAX_TENSOR_NUMEL": "1024"}):
                self.assertEqual(_npu_max_tensor_numel(), 1024)

    def test_default_reduction_loop_default_and_override(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_NPU_DEFAULT_REDUCTION_LOOP", None)
            self.assertEqual(_npu_default_reduction_loop(), 16)
            with patch.dict(os.environ, {"HELION_NPU_DEFAULT_REDUCTION_LOOP": "8"}):
                self.assertEqual(_npu_default_reduction_loop(), 8)


@skipUnlessNPU("requires Ascend NPU")
class TestNPUConfigSpecGuards(TestCase):
    """ConfigSpec NPU guards restrict pid types / indexing / eviction policies on NPU.

    These guards are backend-agnostic: they fire on NPU detection regardless of the
    selected backend, so PR1's config tolerance is exercised with the default triton
    backend (no AscendBackend required).
    """

    def test_eviction_policies_restricted_on_npu(self) -> None:
        # NPU only allows the empty eviction policy (no first/last).
        self.assertEqual(get_valid_eviction_policies("triton"), ("",))

    def test_allowed_pid_types_drop_xyz_on_npu(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
        # NPU keeps flat + persistent variants for the coreDim<=65535 limit, drops xyz.
        self.assertEqual(
            spec.allowed_pid_types,
            ("flat", "persistent_blocked", "persistent_interleaved"),
        )
        self.assertNotIn("xyz", spec.allowed_pid_types)

    def test_valid_indexing_types_pointer_only_on_npu(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
        # Conservative indexing: NPU restricts to pointer only.
        self.assertEqual(spec.valid_indexing_types(), ("pointer",))

    def test_downgrade_unsupported_indexing_rewrites_block_ptr(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
        cfg = {"indexing": "block_ptr"}
        spec.downgrade_unsupported_indexing(cfg)
        self.assertEqual(cfg["indexing"], "pointer")


@skipUnlessNPU("requires Ascend NPU")
class TestNPUNormalizeReductionLoops(TestCase):
    """normalize() delegates NPU reduction_loop capping to _npu_cap_reduction_loops.

    The cap logic itself is a pure function (env-driven budget/default), so it is
    unit-tested directly below; this class verifies the normalize call site wires
    the helper in (and writes its return value back) on NPU.
    """

    def test_normalize_invokes_cap_on_npu(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
        cfg: dict = {"block_sizes": [8, 8], "reduction_loops": [None]}
        with patch(
            "helion.autotuner.config_spec._npu_cap_reduction_loops",
            return_value=[16],
        ) as mock_cap:
            spec.normalize(cfg, _fix_invalid=True)
        # The NPU cap hook must fire when reduction_loops is a list on NPU ...
        mock_cap.assert_called_once()
        # ... and its return value is wired back into the config.
        self.assertEqual(cfg["reduction_loops"], [16])


class TestNPUCapReductionLoops(TestCase):
    """_npu_cap_reduction_loops materializes None and floors over-large loops to the UB budget.

    Pure logic (env-driven budget/default), so it runs on any host -- no NPU required.
    """

    @staticmethod
    def _cap(reduction_loops: list, block_sizes: list) -> list | None:
        # Pin the env-driven caps so assertions are deterministic on any host.
        with patch.dict(
            os.environ,
            {
                "HELION_NPU_UB_BUDGET_ELEMENTS": "2048",
                "HELION_NPU_DEFAULT_REDUCTION_LOOP": "16",
            },
        ):
            return _npu_cap_reduction_loops(reduction_loops, block_sizes)

    def test_none_materialized_to_default(self) -> None:
        # tile_product=64; budget=2048; cap=32; default 16 fits.
        self.assertEqual(self._cap([None], [8, 8]), [16])

    def test_overlarge_capped_to_budget(self) -> None:
        # 128 > cap(32) -> floored to 32.
        self.assertEqual(self._cap([128], [8, 8]), [32])

    def test_no_change_returns_none(self) -> None:
        # 16 <= cap(32) -> no change -> None (caller skips assignment).
        self.assertIsNone(self._cap([16], [8, 8]))

    def test_multiple_loops_each_capped(self) -> None:
        self.assertEqual(self._cap([None, 128], [8, 8]), [16, 32])

    def test_tight_budget_floors_default(self) -> None:
        # tile_product=4096 > budget=2048 -> cap=1, so even the default 16 floors to 1.
        self.assertEqual(self._cap([None], [64, 64]), [1])


class TestNonNPURegression(TestCase):
    """With NPU detection forced absent, the guards revert to stock triton behavior.

    This is the regression guard: the Ascend config tolerance must not restrict
    non-NPU (CUDA) configs. Uses ``_force_npu_absent`` so the tests are portable
    to CUDA-only hosts that lack torch_npu entirely.
    """

    @patch(
        "helion.autotuner.config_spec.supports_amd_cdna_tunables", return_value=False
    )
    def test_eviction_policies_unrestricted_without_npu(self, _amd) -> None:
        # triton + no AMD + no NPU -> the full ("", "first", "last") policy set.
        with _force_npu_absent():
            self.assertEqual(
                get_valid_eviction_policies("triton"), ("", "first", "last")
            )

    def test_xyz_pid_type_restored_without_npu(self) -> None:
        with _force_npu_absent():
            spec = ConfigSpec(backend=TritonBackend())
            self.assertIn("xyz", spec.allowed_pid_types)

    def test_downgrade_is_noop_without_npu(self) -> None:
        with _force_npu_absent():
            spec = ConfigSpec(backend=TritonBackend())
            cfg = {"indexing": "block_ptr"}
            spec.downgrade_unsupported_indexing(cfg)
            # Without NPU the downgrade is a no-op (block_ptr is valid on stock triton).
            self.assertEqual(cfg["indexing"], "block_ptr")
