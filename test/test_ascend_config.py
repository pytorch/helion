from __future__ import annotations

from contextlib import contextmanager
import os
import types
from typing import Callable
from typing import Iterator
from unittest.mock import patch

import torch

from helion._compiler.ascend.config import _npu_cap_reduction_loops
from helion._compiler.ascend.config import _npu_default_reduction_loop
from helion._compiler.ascend.config import _npu_max_tensor_numel
from helion._compiler.ascend.config import _npu_ub_budget_elements
from helion._compiler.ascend.config import is_npu
from helion._compiler.ascend.config import reset_is_npu
from helion._compiler.backend import TritonBackend
from helion._testing import TestCase
from helion._testing import skipIfFn
from helion.autotuner.benchmarking import _coerce_triton_timing
from helion.autotuner.benchmarking import _npu_summarize_times
from helion.autotuner.block_id_sequence import BlockIdSequence
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import get_valid_eviction_policies
from helion.autotuner.local_cache import _device_hardware_and_runtime
from helion.exc import InvalidConfig


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
    portable across CUDA-only and NPU-capable hosts.  ``reset_is_npu`` clears
    the latched ``is_npu()`` result on entry and exit so the mock is honored.
    """
    reset_is_npu()
    if hasattr(torch, "npu"):
        with patch("torch.npu.is_available", return_value=False):
            try:
                yield
            finally:
                reset_is_npu()
    else:
        try:
            yield
        finally:
            reset_is_npu()


@contextmanager
def _force_npu_present() -> Iterator[None]:
    """Treat NPU as present for the duration of the block.

    Mirror of ``_force_npu_absent``: on hosts without torch_npu, inject a
    minimal fake ``torch.npu`` attribute so ``is_npu()`` turns True; on hosts
    with torch_npu, mock ``is_available`` to True. ``reset_is_npu`` clears the
    latched result on entry and exit so the mock is honored.
    """
    reset_is_npu()
    if hasattr(torch, "npu"):
        with patch("torch.npu.is_available", return_value=True):
            try:
                yield
            finally:
                reset_is_npu()
    else:
        torch.npu = types.SimpleNamespace(is_available=lambda: True)
        try:
            yield
        finally:
            del torch.npu
            reset_is_npu()


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


class TestIsNpuLatch(TestCase):
    """``is_npu`` latches ``True`` for the process but keeps re-checking ``False``.

    Runs on any host: on NPU hosts the latch path is exercised, and
    ``_force_npu_absent`` verifies the mock is honored after a reset.
    """

    def test_false_is_not_latched(self) -> None:
        with _force_npu_absent():
            # On an NPU host the mock forces False; re-checking means the mock
            # is honored on every call (a latched True would ignore it).
            self.assertFalse(is_npu())
            self.assertFalse(is_npu())

    def test_reset_clears_latch(self) -> None:
        # Latch a value (whatever the host reports), then confirm the reset
        # makes the next call re-query the (mocked) availability.
        first = is_npu()
        reset_is_npu()
        with _force_npu_absent():
            self.assertFalse(is_npu())
        reset_is_npu()
        self.assertEqual(is_npu(), first)


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


class TestCoerceTritonTiming(TestCase):
    """``_coerce_triton_timing`` locks down the NPU do_bench Tensor path."""

    def test_scalars(self) -> None:
        self.assertEqual(_coerce_triton_timing(1.5), 1.5)
        self.assertEqual(_coerce_triton_timing(2), 2.0)

    def test_zero_dim_tensor(self) -> None:
        self.assertEqual(_coerce_triton_timing(torch.tensor(3.5)), 3.5)

    def test_tuples(self) -> None:
        self.assertEqual(_coerce_triton_timing((1.0, 2.0)), (1.0, 2.0))
        self.assertEqual(
            _coerce_triton_timing((torch.tensor(1.0), 2.0)),
            (1.0, 2.0),
        )


class TestNPUSummarizeTimes(TestCase):
    """``_npu_summarize_times`` handles Tensor-returning triton builds."""

    def test_tensor_output_coerced_to_float(self) -> None:
        def fake_summarize(times, quantiles, return_mode):
            assert isinstance(times, torch.Tensor)
            return torch.tensor(2.5)

        with patch("triton.testing._summarize_statistics", fake_summarize, create=True):
            self.assertEqual(_npu_summarize_times([1.0, 2.0], None, "mean"), 2.5)

    def test_tuple_of_tensors_coerced(self) -> None:
        def fake_summarize(times, quantiles, return_mode):
            return (torch.tensor(1.0), 2.0)

        with patch("triton.testing._summarize_statistics", fake_summarize, create=True):
            self.assertEqual(
                _npu_summarize_times([1.0, 2.0], [0.5, 0.9], "all"), (1.0, 2.0)
            )


class TestDeviceHardwareAndRuntime(TestCase):
    """``_device_hardware_and_runtime`` covers the NPU cache-key branch host-agnostically."""

    def test_npu_device(self) -> None:
        hardware, runtime_name = _device_hardware_and_runtime(
            torch.device("npu"), "triton"
        )
        self.assertEqual(hardware, "npu")
        # Without torch_npu the ImportError fallback yields "unknown"; with
        # torch_npu installed it is the real device name.
        self.assertIsInstance(runtime_name, str)

    def test_cpu_pallas_interpret(self) -> None:
        hardware, runtime_name = _device_hardware_and_runtime(
            torch.device("cpu"), "pallas"
        )
        self.assertEqual(hardware, "pallas_interpret")
        self.assertEqual(runtime_name, "interpret")


class TestBlockIdSequenceZeroSlots(TestCase):
    """Zero-slot tunables drop user values only on NPU (upstream error elsewhere)."""

    def test_values_dropped_on_npu(self) -> None:
        seq = BlockIdSequence()
        with _force_npu_present():
            self.assertEqual(seq._normalize("range_warp_specialize", [1, 2, 3]), [])

    def test_values_rejected_without_npu(self) -> None:
        seq = BlockIdSequence()
        with _force_npu_absent(), self.assertRaises(InvalidConfig):
            seq._normalize("range_warp_specialize", [1, 2, 3])


class TestCoerceNpuTlRangeTunables(TestCase):
    """``coerce_npu_tl_range_tunables`` follows ``is_npu()`` (single source of truth)."""

    def test_coerced_on_npu(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
        cfg = {"range_unroll_factors": [2, 4]}
        with _force_npu_present():
            spec.coerce_npu_tl_range_tunables(cfg)
        self.assertEqual(cfg["range_unroll_factors"], [0, 0])

    def test_no_coercion_without_npu(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
        cfg = {"range_unroll_factors": [2, 4]}
        with _force_npu_absent():
            spec.coerce_npu_tl_range_tunables(cfg)
        self.assertEqual(cfg["range_unroll_factors"], [2, 4])
