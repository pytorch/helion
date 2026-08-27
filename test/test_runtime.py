from __future__ import annotations

import inspect
import threading
import time
import unittest
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

import helion.runtime
from helion.runtime.triton.launcher import _get_persistent_state
from helion.runtime.triton.launcher import _limit_resident_programs_per_sm
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
    def test_public_launcher_documents_persistent_state_serialization(self) -> None:
        documentation = inspect.getdoc(helion.runtime.default_launcher)

        self.assertIsNotNone(documentation)
        assert documentation is not None
        self.assertIn("must be serialized", documentation)
        self.assertIn("Independent streams receive", documentation)
        self.assertIn("independent state", documentation)

    def test_cross_loop_dispatch_diagnostic_is_exposed(self) -> None:
        class FakeJITFunction:
            def run(self, *args: object, **kwargs: object) -> object:
                return "launched"

        kernel = FakeJITFunction()
        result = triton_default_launcher(
            kernel,
            (1,),
            num_warps=1,
            num_stages=1,
            _cross_loop_dispatch_kind="static",
            _cross_loop_fallback_reason="command_plan_unavailable: test",
        )

        self.assertEqual(result, "launched")
        self.assertEqual(
            kernel._helion_cross_loop_dispatch,
            ("static", "command_plan_unavailable: test"),
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

    def test_persistent_state_creation_is_locked(self) -> None:
        kernel = SimpleNamespace()
        like = SimpleNamespace(device=torch.device("cuda"))
        stream = SimpleNamespace(cuda_stream=17)
        allocations = []

        class FakeState:
            def numel(self) -> int:
                return 8

        def allocate(*args, **kwargs):
            state = FakeState()
            allocations.append(state)
            time.sleep(0.05)
            return state

        with (
            patch("torch.cuda.current_stream", return_value=stream),
            patch(
                "helion.runtime.triton.launcher."
                "_reject_allocation_during_cuda_graph_capture"
            ),
            patch("torch.zeros", side_effect=allocate),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = tuple(
                executor.map(
                    lambda _: _get_persistent_state(
                        kernel,
                        like,
                        ("concurrent",),
                        0,
                        8,
                        torch.uint32,
                    ),
                    range(2),
                )
            )

        self.assertEqual(len(allocations), 1)
        self.assertIs(results[0], results[1])

    def test_residency_mutation_and_launch_are_locked_together(self) -> None:
        metadata_type = namedtuple("Metadata", ("shared",))
        compiled = SimpleNamespace(metadata=metadata_type(shared=0))
        expected_target = threading.local()
        first_mutation_started = threading.Event()
        launches = []

        class FakeJITFunction:
            def run(self, *args: object, **kwargs: object) -> object:
                if kwargs["warmup"]:
                    return compiled
                launches.append((expected_target.value, compiled.metadata.shared))
                return "launched"

        def mutate(_compiled, _args, *, num_warps, target_programs):
            _compiled.metadata = metadata_type(shared=target_programs)
            if target_programs == 7:
                first_mutation_started.set()
                time.sleep(0.05)

        def launch(target: int) -> object:
            expected_target.value = target
            if target == 8:
                first_mutation_started.wait()
            return triton_default_launcher(
                FakeJITFunction(),
                (8,),
                object(),
                num_warps=2,
                num_stages=3,
                _target_resident_programs_per_sm=target,
            )

        with (
            patch(
                "helion.runtime.triton.launcher._limit_resident_programs_per_sm",
                side_effect=mutate,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = tuple(executor.map(launch, (7, 8)))

        self.assertEqual(results, ("launched", "launched"))
        self.assertEqual(sorted(launches), [(7, 7), (8, 8)])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPersistentTritonState(unittest.TestCase):
    def test_is_retained_and_namespaced_by_launch_configuration(self) -> None:
        kernel = SimpleNamespace()
        like = torch.empty(1, device="cuda")
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

    def test_is_independent_across_streams(self) -> None:
        kernel = SimpleNamespace()
        like = torch.empty(1, device="cuda")
        namespace = ((1,), 1, 2, None, False, (), ((8, torch.uint32),))
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        states = []
        for stream in streams:
            with torch.cuda.stream(stream):
                states.append(
                    _get_persistent_state(
                        kernel,
                        like,
                        namespace,
                        0,
                        8,
                        torch.uint32,
                    )
                )

        self.assertNotEqual(states[0].data_ptr(), states[1].data_ptr())

    def test_cached_residency_target_restores_launch_metadata(self) -> None:
        metadata_type = namedtuple("Metadata", ("shared", "target"))
        compiled = SimpleNamespace(
            run=object(),
            function=1,
            metadata=metadata_type(shared=1024, target="cuda"),
            packed_metadata=None,
        )
        like = torch.empty(1, device="cuda")
        success = 0
        active_device = []
        device_entries = []

        def occupancy(*_args):
            self.assertEqual(active_device, [like.device])
            return success, 8 if _args[-1] < 16 * 1024 else 4

        def get_attribute(*_args):
            self.assertEqual(active_device, [like.device])
            return success, 0

        def set_attribute(*_args):
            self.assertEqual(active_device, [like.device])
            return (success,)

        driver = SimpleNamespace(
            CUfunction=int,
            CUresult=SimpleNamespace(CUDA_SUCCESS=success),
            CUfunction_attribute=SimpleNamespace(
                CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES=1,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=2,
            ),
            cuOccupancyMaxActiveBlocksPerMultiprocessor=occupancy,
            cuFuncGetAttribute=get_attribute,
            cuFuncSetAttribute=set_attribute,
        )
        backend = SimpleNamespace(
            pack_metadata=lambda metadata: ("packed", metadata.shared)
        )

        class DeviceContext:
            def __enter__(self):
                active_device.append(like.device)
                device_entries.append(like.device)

            def __exit__(self, *_args):
                active_device.pop()

        with (
            patch("torch.cuda.device", return_value=DeviceContext()),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(shared_memory_per_block_optin=64 * 1024),
            ),
            patch("triton.compiler.compiler.make_backend", return_value=backend),
            patch("importlib.import_module", return_value=driver),
        ):
            _limit_resident_programs_per_sm(
                compiled,
                (like,),
                num_warps=2,
                target_programs=4,
            )
            limited_shared = compiled.metadata.shared
            self.assertEqual(limited_shared, 16 * 1024)
            self.assertFalse(active_device)

            _limit_resident_programs_per_sm(
                compiled,
                (like,),
                num_warps=2,
                target_programs=8,
            )
            self.assertEqual(compiled.metadata.shared, 1024)

            _limit_resident_programs_per_sm(
                compiled,
                (like,),
                num_warps=2,
                target_programs=4,
            )
            self.assertEqual(compiled.metadata.shared, limited_shared)
            self.assertEqual(compiled.packed_metadata, ("packed", limited_shared))
            self.assertEqual(device_entries.count(like.device), 4)

            with patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                return_value=True,
            ):
                _limit_resident_programs_per_sm(
                    compiled,
                    (like,),
                    num_warps=2,
                    target_programs=8,
                )
            self.assertEqual(compiled.metadata.shared, 1024)
            self.assertEqual(compiled.packed_metadata, ("packed", 1024))
            self.assertEqual(device_entries.count(like.device), 4)

    def test_first_residency_configuration_is_not_capture_safe(self) -> None:
        metadata_type = namedtuple("Metadata", ("shared", "target"))
        compiled = SimpleNamespace(
            run=object(),
            function=1,
            metadata=metadata_type(shared=1024, target="cuda"),
            packed_metadata=None,
        )
        like = torch.empty(1, device="cuda")

        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(RuntimeError, "warm up.*capture stream"),
        ):
            _limit_resident_programs_per_sm(
                compiled,
                (like,),
                num_warps=2,
                target_programs=4,
            )

    def test_residency_metadata_update_is_atomic_on_pack_failure(self) -> None:
        metadata_type = namedtuple("Metadata", ("shared", "target"))
        original_metadata = metadata_type(shared=1024, target="cuda")
        compiled = SimpleNamespace(
            run=object(),
            function=1,
            metadata=original_metadata,
            packed_metadata=("original", 1024),
        )
        like = torch.empty(1, device="cuda")

        def fail_pack(_metadata):
            raise RuntimeError("packing failed")

        backend = SimpleNamespace(pack_metadata=fail_pack)

        with (
            patch(
                "helion.runtime.triton.launcher."
                "_compute_clc_resident_shared_bytes",
                return_value=16 * 1024,
            ),
            patch("triton.compiler.compiler.make_backend", return_value=backend),
            self.assertRaisesRegex(RuntimeError, "packing failed"),
        ):
            _limit_resident_programs_per_sm(
                compiled,
                (like,),
                num_warps=2,
                target_programs=4,
            )

        self.assertEqual(compiled.metadata, original_metadata)
        self.assertEqual(compiled.packed_metadata, ("original", 1024))
        self.assertEqual(compiled._helion_clc_residency_limits, {})

    def test_persistent_state_rejects_first_allocation_during_capture(self) -> None:
        kernel = SimpleNamespace()
        like = torch.empty(1, device="cuda")
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(RuntimeError, "warm up.*capture stream"),
        ):
            _get_persistent_state(
                kernel,
                like,
                ("capture",),
                0,
                1,
                torch.uint64,
            )

    def test_preallocated_compiler_state_is_capture_safe(self) -> None:
        kernel = SimpleNamespace()
        like = torch.empty(1, device="cuda")
        state = _get_persistent_state(
            kernel,
            like,
            ("capture",),
            0,
            1,
            torch.uint64,
        )

        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            retained_state = _get_persistent_state(
                kernel,
                like,
                ("capture",),
                0,
                1,
                torch.uint64,
            )

        self.assertEqual(state.data_ptr(), retained_state.data_ptr())
