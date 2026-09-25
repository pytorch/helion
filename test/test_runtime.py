from __future__ import annotations

from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

from helion._testing import DEVICE
import helion.runtime
from helion.runtime.triton.launcher import _distributed_launch_fingerprint
from helion.runtime.triton.launcher import _get_distributed_readiness_signal
from helion.runtime.triton.launcher import _get_persistent_state
from helion.runtime.triton.launcher import default_launcher as triton_default_launcher


def _tpu_device() -> torch.device:
    try:
        return torch.device("tpu")
    except RuntimeError:
        return cast("torch.device", SimpleNamespace(type="tpu", index=None))


class TestRuntimeGetNumSm(unittest.TestCase):
    def test_pallas_interpret_cpu_returns_one(self) -> None:
        with patch("helion.runtime._module_is_pallas_interpret", return_value=True):
            self.assertEqual(helion.runtime.get_num_sm(torch.device("cpu")), 1)
            self.assertEqual(
                helion.runtime.get_num_sm(torch.device("cpu"), reserved_sms=8),
                1,
            )

    def test_normal_cpu_still_unsupported(self) -> None:
        with (
            patch("helion.runtime._module_is_pallas_interpret", return_value=False),
            self.assertRaisesRegex(
                AssertionError,
                "TODO: implement for other devices",
            ),
        ):
            helion.runtime.get_num_sm(torch.device("cpu"))

    def test_tpu_returns_one(self) -> None:
        device = _tpu_device()

        self.assertEqual(helion.runtime.get_num_sm(device), 1)
        self.assertEqual(helion.runtime.get_num_sm(device, reserved_sms=8), 1)


class TestTritonLauncher(unittest.TestCase):
    def test_distributed_fingerprint_tracks_specialization_not_dynamic_shape(
        self,
    ) -> None:
        kernel = SimpleNamespace(
            src="kernel source",
            params=(
                SimpleNamespace(is_constexpr=False),
                SimpleNamespace(is_constexpr=False),
                SimpleNamespace(is_constexpr=True),
            ),
        )

        def fingerprint(tensor: torch.Tensor, dynamic_size: int, block: int) -> str:
            return _distributed_launch_fingerprint(
                kernel,
                (8,),
                (tensor, dynamic_size, block),
                process_group_name="group-a",
                num_warps=4,
                num_stages=2,
                ptx_options=None,
                launch_cooperative_grid=False,
                launch_options={},
                state_schema=((64, "torch.uint32"),),
                readiness_slots=32,
            )

        first = fingerprint(torch.empty(7), 17, 64)
        # Ordinary runtime extents may vary without changing the compiled
        # schedule specialization.
        self.assertEqual(first, fingerprint(torch.empty(11), 19, 64))
        # A constexpr schedule parameter must select a different fingerprint.
        self.assertNotEqual(first, fingerprint(torch.empty(11), 19, 128))
        self.assertNotEqual(
            first,
            _distributed_launch_fingerprint(
                kernel,
                (8,),
                (torch.empty(7), 17, 64),
                process_group_name="group-b",
                num_warps=4,
                num_stages=2,
                ptx_options=None,
                launch_cooperative_grid=False,
                launch_options={},
                state_schema=((64, "torch.uint32"),),
                readiness_slots=32,
            ),
        )

    def test_distributed_readiness_uses_independent_signal_state(self) -> None:
        class FakeJITFunction:
            def run(self, *args: object, **kwargs: object) -> object:
                return "launched"

        payload = torch.empty(8)
        with (
            patch(
                "helion.runtime.triton.launcher._get_remote_copy_signal",
                return_value=object(),
            ) as remote_copy_signal,
            patch(
                "helion.runtime.triton.launcher._get_distributed_readiness_signal",
                return_value=(object(), 1234, 16),
            ) as readiness_signal,
        ):
            result = triton_default_launcher(
                FakeJITFunction(),
                (3,),
                num_warps=2,
                num_stages=1,
                _remote_copy_signal_dst=payload,
                _remote_copy_signal_slots_per_program=2,
                _remote_copy_process_group_name="group",
                _distributed_readiness_device_anchor=payload,
                _distributed_readiness_signal_slots=8,
                _distributed_readiness_process_group_name="group",
            )

        self.assertEqual(result, "launched")
        remote_copy_signal.assert_called_once_with(ANY, payload, "group", 6)
        readiness_signal.assert_called_once_with(
            ANY,
            payload,
            "group",
            8,
            launch_fingerprint=ANY,
        )

    def test_residency_check_uses_exact_compiled_specialization(self) -> None:
        compiled_kernel = object()
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class FakeJITFunction:
            def run(self, *args: object, **kwargs: object) -> object:
                calls.append((args, kwargs))
                if kwargs["warmup"]:
                    return compiled_kernel
                return "launched"

        argument = object()
        with patch(
            "helion.runtime.triton.launcher._validate_resident_program_capacity"
        ) as validate:
            result = triton_default_launcher(
                FakeJITFunction(),
                (8,),
                argument,
                num_warps=2,
                num_stages=3,
                _minimum_resident_programs=7,
                ptx_options="--opt",
            )

        self.assertEqual(result, "launched")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], (argument,))
        self.assertEqual(
            calls[0][1],
            {
                "grid": (8,),
                "warmup": True,
                "num_warps": 2,
                "num_stages": 3,
                "launch_cooperative_grid": False,
                "ptx_options": "--opt",
            },
        )
        self.assertEqual(calls[1][0], (argument,))
        self.assertEqual(calls[1][1], {**calls[0][1], "warmup": False})
        validate.assert_called_once_with(
            compiled_kernel,
            (argument,),
            num_warps=2,
            required_programs=7,
        )

    def test_cross_loop_occupancy_failure_is_an_invalid_config(self) -> None:
        with (
            patch(
                "helion.runtime._triton_default_launcher",
                side_effect=RuntimeError(
                    "Cross-loop scheduling requires 16 concurrently resident "
                    "programs, but this kernel/device can residently execute only 12."
                ),
            ),
            self.assertRaises(helion.exc.InvalidConfig),
        ):
            helion.runtime.default_launcher(
                object(),
                (16,),
                num_warps=1,
                num_stages=1,
            )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPersistentTritonState(unittest.TestCase):
    def test_distributed_readiness_capacity_failure_is_collective(self) -> None:
        class FakeHandle:
            rank = 0
            signal_pad_ptrs_dev = 1234

            def get_signal_pad(self, rank: int, *, dtype: torch.dtype) -> torch.Tensor:
                assert rank == self.rank
                assert dtype is torch.uint64
                return torch.zeros(4, dtype=torch.uint64)

        kernel = SimpleNamespace()
        base = torch.empty(8, device=DEVICE)
        stream = SimpleNamespace(cuda_stream=1, synchronize=MagicMock())

        def gather_statuses(output, value, *, group):
            self.assertEqual(group, "resolved-group")
            stream.synchronize.assert_called_once_with()
            output[:] = [value, (value[0], 64)]

        with (
            patch(
                "torch.distributed._symmetric_memory.empty",
                return_value=torch.empty(1, dtype=torch.uint8, device=DEVICE),
            ),
            patch(
                "torch.distributed._symmetric_memory.rendezvous",
                return_value=FakeHandle(),
            ),
            patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value="resolved-group",
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=gather_statuses,
            ) as all_gather,
            patch("torch.cuda.current_stream", return_value=stream),
            self.assertRaisesRegex(RuntimeError, "signal pad capacities"),
        ):
            _get_distributed_readiness_signal(
                kernel,
                base,
                "group",
                8,
                launch_fingerprint="fingerprint",
            )
        all_gather.assert_called_once()

    def test_distributed_readiness_rejects_rank_fingerprint_mismatch(self) -> None:
        class FakeHandle:
            rank = 0
            signal_pad_ptrs_dev = 1234

            def get_signal_pad(self, rank: int, *, dtype: torch.dtype) -> torch.Tensor:
                assert rank == self.rank
                assert dtype is torch.uint64
                return torch.zeros(64, dtype=torch.uint64)

        kernel = SimpleNamespace()
        base = torch.empty(8, device=DEVICE)
        stream = SimpleNamespace(cuda_stream=1, synchronize=MagicMock())

        def gather_fingerprints(output, value, *, group):
            self.assertEqual(group, "resolved-group")
            output[:] = [value, ("different-rank-fingerprint", value[1])]

        with (
            patch(
                "torch.distributed._symmetric_memory.empty",
                return_value=torch.empty(1, dtype=torch.uint8, device=DEVICE),
            ),
            patch(
                "torch.distributed._symmetric_memory.rendezvous",
                return_value=FakeHandle(),
            ),
            patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value="resolved-group",
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=gather_fingerprints,
            ),
            patch("torch.cuda.current_stream", return_value=stream),
            self.assertRaisesRegex(
                RuntimeError,
                "identical kernel schedules and launch geometry",
            ),
        ):
            _get_distributed_readiness_signal(
                kernel,
                base,
                "group",
                8,
                launch_fingerprint="this-rank-fingerprint",
            )

    def test_distributed_readiness_state_is_stream_local_and_dedicated(self) -> None:
        class FakeHandle:
            rank = 0

            def __init__(self, pointer: int) -> None:
                self.signal_pad_ptrs_dev = pointer
                self.signal_pad = torch.ones(64, dtype=torch.uint64)

            def get_signal_pad(self, rank: int, *, dtype: torch.dtype) -> torch.Tensor:
                assert rank == self.rank
                assert dtype is torch.uint64
                return self.signal_pad

        kernel = SimpleNamespace()
        other_kernel = SimpleNamespace()
        base = torch.empty(8, device=DEVICE)
        first_stream = SimpleNamespace(cuda_stream=1, synchronize=MagicMock())
        second_stream = SimpleNamespace(cuda_stream=2, synchronize=MagicMock())
        first_handle = FakeHandle(1234)
        second_handle = FakeHandle(5678)
        other_kernel_handle = FakeHandle(9012)
        variant_handle = FakeHandle(3456)
        workspaces = (
            torch.empty(1, dtype=torch.uint8, device=DEVICE),
            torch.empty(1, dtype=torch.uint8, device=DEVICE),
            torch.empty(1, dtype=torch.uint8, device=DEVICE),
            torch.empty(1, dtype=torch.uint8, device=DEVICE),
        )
        gathered = 0

        def gather_fingerprints(output, value, *, group):
            nonlocal gathered
            self.assertEqual(group, "resolved-group")
            handle = (
                first_handle,
                second_handle,
                other_kernel_handle,
                variant_handle,
            )[gathered]
            self.assertTrue(torch.all(handle.signal_pad[-8:] == 0))
            self.assertGreaterEqual(
                first_stream.synchronize.call_count
                + second_stream.synchronize.call_count,
                1,
            )
            gathered += 1
            output[:] = [value, value]

        with (
            patch(
                "torch.distributed._symmetric_memory.rendezvous",
                side_effect=(
                    first_handle,
                    second_handle,
                    other_kernel_handle,
                    variant_handle,
                ),
            ) as rendezvous,
            patch(
                "torch.distributed._symmetric_memory.empty",
                side_effect=workspaces,
            ) as allocate,
            patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value="resolved-group",
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=gather_fingerprints,
            ) as all_gather,
            patch(
                "torch.cuda.current_stream",
                side_effect=(
                    first_stream,
                    first_stream,
                    second_stream,
                    first_stream,
                    first_stream,
                ),
            ),
        ):
            first = _get_distributed_readiness_signal(kernel, base, "group", 8)
            first_handle.signal_pad[-8:].fill_(7)
            retained = _get_distributed_readiness_signal(kernel, base, "group", 8)
            second = _get_distributed_readiness_signal(kernel, base, "group", 8)
            independent = _get_distributed_readiness_signal(
                other_kernel, base, "group", 8
            )
            variant = _get_distributed_readiness_signal(
                kernel,
                base,
                "group",
                8,
                launch_fingerprint="different-config",
            )

        self.assertEqual(first[0].data_ptr(), retained[0].data_ptr())
        self.assertEqual(first[1:], (1234, 56))
        self.assertEqual(second[1:], (5678, 56))
        self.assertEqual(independent[1:], (9012, 56))
        self.assertEqual(variant[1:], (3456, 56))
        self.assertNotEqual(first[0].data_ptr(), second[0].data_ptr())
        self.assertNotEqual(first[0].data_ptr(), independent[0].data_ptr())
        self.assertEqual(rendezvous.call_count, 4)
        self.assertEqual(allocate.call_count, 4)
        self.assertEqual(all_gather.call_count, 4)
        self.assertTrue(torch.all(first_handle.signal_pad[-8:] == 7))
        self.assertTrue(torch.all(second_handle.signal_pad[-8:] == 0))

    def test_is_retained_and_namespaced_by_launch_configuration(self) -> None:
        kernel = SimpleNamespace()
        like = torch.empty(1, device=DEVICE)
        namespace = ((1,), 1, 2, None, False, (), ((8, torch.uint32),))
        state = _get_persistent_state(kernel, like, namespace, 0, 8, torch.uint32)
        state.fill_(7)

        retained = _get_persistent_state(kernel, like, namespace, 0, 8, torch.uint32)
        independent = _get_persistent_state(
            kernel,
            like,
            (*namespace[:-1], ((16, torch.uint32),)),
            0,
            16,
            torch.uint32,
        )

        self.assertEqual(retained.data_ptr(), state.data_ptr())
        self.assertEqual(retained[0].item(), 7)
        self.assertNotEqual(independent.data_ptr(), state.data_ptr())
        self.assertEqual(torch.count_nonzero(independent).item(), 0)
