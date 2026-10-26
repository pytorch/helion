from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest.mock import patch

import torch

from helion._testing import DEVICE
import helion.runtime
from helion.runtime.triton.launcher import _get_persistent_state
from helion.runtime.triton.launcher import _validate_resident_program_capacity
from helion.runtime.triton.launcher import default_launcher as triton_default_launcher

if TYPE_CHECKING:
    from collections.abc import Iterator


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

    def test_residency_counts_the_warps_the_compiled_kernel_launches(self) -> None:
        # A warp-specialized kernel launches more warps than its config's
        # num_warps (Triton records the total in the compiled metadata); the
        # residency check must count them, or it overestimates resident CTAs.
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            tensor = torch.empty(1, device="cuda")
        occupancy_threads: list[int] = []
        attributes = {
            "CU_FUNC_ATTRIBUTE_NUM_REGS": 255,
            "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES": 0,
            "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR": 65536,
            "CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK": 1024,
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR": 233472,
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR": 2048,
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR": 32,
        }

        def occupancy(function: object, threads: int, shared: int) -> object:
            occupancy_threads.append(threads)
            # The API answers one block per SM for any tcgen05 kernel.
            return driver.CUresult.CUDA_SUCCESS, 1

        driver = SimpleNamespace(
            CUresult=SimpleNamespace(CUDA_SUCCESS=0),
            CUfunction=lambda handle: handle,
            CUfunction_attribute=SimpleNamespace(**{k: k for k in attributes}),
            CUdevice_attribute=SimpleNamespace(**{k: k for k in attributes}),
            cuOccupancyMaxActiveBlocksPerMultiprocessor=occupancy,
            cuFuncGetAttribute=lambda attr, fn: (0, attributes[attr]),
            cuDeviceGet=lambda index: (0, index),
            cuDeviceGetAttribute=lambda attr, device: (0, attributes[attr]),
        )

        def compiled(launched_warps: int) -> SimpleNamespace:
            return SimpleNamespace(
                run=None,
                function=1,
                metadata=SimpleNamespace(
                    shared=4096, num_warps=launched_warps, tmem_size=128
                ),
            )

        with (
            patch.dict(sys.modules, {"cuda.bindings.driver": driver}),
            patch("torch.cuda.device", lambda device: contextlib.nullcontext()),
            patch(
                "torch.cuda.get_device_properties",
                lambda device: SimpleNamespace(multi_processor_count=148),
            ),
        ):
            # 255 registers: two 4-warp CTAs fit an SM's registers, one 8-warp CTA.
            _validate_resident_program_capacity(
                compiled(4), (tensor,), num_warps=4, required_programs=296
            )
            self.assertEqual(occupancy_threads, [128])
            with self.assertRaisesRegex(RuntimeError, "residently execute only 148"):
                _validate_resident_program_capacity(
                    compiled(8), (tensor,), num_warps=4, required_programs=296
                )
            self.assertEqual(occupancy_threads, [128, 256])

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

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA graphs")
    def test_state_first_requested_in_a_capture_is_zeroed_outside_the_graph(
        self,
    ) -> None:
        # The state is zero when the graph first runs, and a replay does not
        # re-zero it: no allocation or fill is recorded into the graph.
        like = torch.empty(1, device=DEVICE)
        namespace = ((1,), 1, 2, None, False, (), ((8, torch.int32),))
        for mode in ("global", "thread_local", "relaxed"):
            with self.subTest(capture_error_mode=mode):
                kernel = SimpleNamespace()
                seen = torch.full((8,), -1, dtype=torch.int32, device=DEVICE)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode=mode):  # pyrefly: ignore[bad-argument-type]
                    state = _get_persistent_state(
                        kernel, like, namespace, 0, 8, torch.int32
                    )
                    seen.copy_(state)
                    state.add_(1)
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(seen.tolist(), [0] * 8)
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(seen.tolist(), [1] * 8)
                self.assertEqual(state.tolist(), [2] * 8)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA graphs")
    def test_state_allocated_in_a_capture_outlives_its_cache_entry(self) -> None:
        # The graph addresses the state; once the launcher's cache (on the
        # kernel object) is gone, the memory must still not be handed out
        # again, or replays would race with whatever reuses it.
        from helion.runtime.triton import launcher

        like = torch.empty(1, device=DEVICE)
        namespace = ((1,), 1, 2, None, False, (), ((64, torch.int32),))
        kernel = SimpleNamespace()
        seen = torch.full((64,), -1, dtype=torch.int32, device=DEVICE)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state = _get_persistent_state(kernel, like, namespace, 0, 64, torch.int32)
            seen.copy_(state)
            state.add_(1)
        address = state.data_ptr()
        self.assertIn(address, [t.data_ptr() for t in launcher._CAPTURED_STATES])
        del kernel, state
        others = [torch.zeros(64, dtype=torch.int32, device=DEVICE) for _ in range(64)]
        self.assertNotIn(address, [t.data_ptr() for t in others])
        for k in range(3):
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(seen.tolist(), [k] * 64)
        self.assertTrue(all(torch.count_nonzero(t).item() == 0 for t in others))


class TestPersistentStateCaptureCheck(unittest.TestCase):
    def test_capture_is_checked_on_the_state_device(self) -> None:
        # A launch on device 1 while device 0 is current: whether the state is
        # requested in a capture is a question about device 1's stream.
        from helion.runtime.triton import launcher

        current = [0]

        @contextlib.contextmanager
        def device(d: torch.device) -> Iterator[None]:
            saved, current[0] = current[0], torch.device(d).index
            try:
                yield
            finally:
                current[0] = saved

        with (
            patch.object(torch.cuda, "device", device),
            patch.object(
                torch.cuda, "is_current_stream_capturing", lambda: current[0] == 1
            ),
        ):
            self.assertTrue(launcher._is_capturing(torch.device("cuda", 1)))
            self.assertFalse(launcher._is_capturing(torch.device("cuda", 0)))
            self.assertEqual(current[0], 0)
