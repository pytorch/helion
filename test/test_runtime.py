from __future__ import annotations

from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch

import torch

from helion._testing import DEVICE
import helion.runtime
from helion.runtime.triton import launcher as triton_launcher
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
        # The check reads the warp count off the compiled kernel (a
        # warp-specialized kernel's includes its partitions').
        validate.assert_called_once_with(
            compiled_kernel, (argument,), required_programs=7
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

    @unittest.skipIf(triton_launcher.triton is None, "requires Triton")
    def test_scratch_allocator_queries_each_active_driver_once(self) -> None:
        queried: list[object] = []

        class FakeDriver:
            def __init__(self, backend: str) -> None:
                self.backend = backend

            def get_current_target(self) -> object:
                queried.append(self)
                return SimpleNamespace(backend=self.backend)

        cpu, meta = FakeDriver("cpu"), FakeDriver("meta")
        drivers = SimpleNamespace(active=cpu)
        fake_triton = SimpleNamespace(runtime=SimpleNamespace(driver=drivers))
        buffers = []
        with (
            patch("helion.runtime.triton.launcher.triton", fake_triton),
            patch("helion.runtime.triton.launcher._alloc_device", None),
        ):
            for active in (cpu, cpu, meta, cpu):
                drivers.active = active
                buffers.append(triton_launcher._alloc_fn(16, 128, None))

        self.assertEqual(queried, [cpu, meta, cpu])
        self.assertEqual(
            [buffer.device.type for buffer in buffers], ["cpu", "cpu", "meta", "cpu"]
        )
        self.assertEqual((buffers[0].numel(), buffers[0].dtype), (16, torch.int8))


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
