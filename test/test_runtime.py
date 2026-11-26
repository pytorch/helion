from __future__ import annotations

import contextlib
import functools
import sys
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import onlyOn
from helion._testing import skipIfFn
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.triton_codegen import cpu_codegen as cpu_codegen
import helion.language as hl
import helion.runtime
from helion.runtime.triton import launcher as triton_launcher
from helion.runtime.triton.launcher import _get_persistent_state
from helion.runtime.triton.launcher import default_launcher as triton_default_launcher

# The launcher's checks run against a faked driver; the persistent state and
# padding tests run on the GPU (``cuda``).
pytestmark = pytest.mark.cpu_only

if triton_launcher.triton is not None:
    import triton
    import triton.language as tl

    @triton.jit
    def _clock(dummy):
        return tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            "=l,r",
            [dummy],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _timed_dot(
        x,
        w,
        y,
        record,
        K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        IEEE: tl.constexpr,
    ):
        """Each program's product of ``x`` and ``w`` into ``y``, and its SM and
        when it ran into ``record``."""
        program = tl.program_id(0)
        start = _clock(program)
        rows, cols = tl.arange(0, BM), tl.arange(0, BN)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k in tl.range(0, K, BK):
            ks = k + tl.arange(0, BK)
            a = tl.load(x + rows[:, None] * K + ks[None, :])
            b = tl.load(w + ks[:, None] * BN + cols[None, :])
            if IEEE:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
        tl.store(y + program * BM * BN + rows[:, None] * BN + cols[None, :], acc)
        sm = tl.inline_asm_elementwise(
            "mov.u32 $0, %smid;",
            "=r,r",
            [program],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.store(record + program * 3, sm.to(tl.int64))
        tl.store(record + program * 3 + 1, start)
        tl.store(record + program * 3 + 2, _clock(program))


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

    def test_cluster_size_takes_the_resident_clusters(self) -> None:
        """With a cluster size the count is at most the SMs whole clusters
        keep resident less the reserved ones, in whole clusters, and at least
        one cluster; clusters need an NVIDIA CUDA device of sm_90 and up."""
        device = torch.device("cuda", 0)
        capacities = {2: 148, 4: 132, 8: 120}
        properties = SimpleNamespace(multi_processor_count=148, major=10, minor=0)
        with (
            patch("torch.cuda.get_device_properties", return_value=properties),
            patch("torch.version.hip", None),
            patch.object(
                triton_launcher,
                "get_cluster_capacity",
                side_effect=lambda device, size: capacities[size],
            ),
        ):
            for cluster_size, reserved, expected in (
                (1, 0, 148),
                (2, 0, 148),
                (4, 0, 132),
                (8, 0, 120),
                (4, 16, 116),
                (2, 8, 140),
                (4, 130, 4),
            ):
                with self.subTest(cluster_size=cluster_size, reserved=reserved):
                    self.assertEqual(
                        helion.runtime.get_num_sm(
                            device, reserved_sms=reserved, cluster_size=cluster_size
                        ),
                        expected,
                    )
        for other in ("mps", "cpu"):
            with self.assertRaisesRegex(ValueError, "NVIDIA CUDA"):
                helion.runtime.get_num_sm(torch.device(other), cluster_size=2)
        with (
            patch("torch.version.hip", "6.4"),
            self.assertRaisesRegex(ValueError, "NVIDIA CUDA"),
        ):
            helion.runtime.get_num_sm(device, cluster_size=2)
        with (
            patch("torch.version.hip", None),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            self.assertRaisesRegex(ValueError, "sm_90"),
        ):
            helion.runtime.get_num_sm(device, cluster_size=2)

    def test_cluster_capacity_of_the_current_device(self) -> None:
        """A device without an index is the current one, for the capacity's
        cache too; clusters past the portable 8 have no capacity."""
        with (
            patch.object(
                triton_launcher,
                "_cluster_capacity",
                side_effect=lambda index, size: (index, size),
            ),
            patch("torch.cuda.current_device", return_value=1),
        ):
            for device, expected in (
                (torch.device("cuda"), (1, 4)),
                (torch.device("cuda", 0), (0, 4)),
            ):
                self.assertEqual(
                    triton_launcher.get_cluster_capacity(device, 4), expected
                )
            with self.assertRaises(ValueError):
                triton_launcher.get_cluster_capacity(torch.device("cuda", 0), 16)


class TestTritonLauncher(unittest.TestCase):
    def test_residency_check_uses_exact_compiled_specialization(self) -> None:
        # The launcher restores the packed metadata (a padded cohort's) after
        # the launch.
        compiled_kernel = SimpleNamespace(
            _init_handles=lambda: None, packed_metadata=(2, 1, 0)
        )
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class FakeJITFunction:
            def run(self, *args: object, **kwargs: object) -> object:
                calls.append((args, kwargs))
                if kwargs["warmup"]:
                    return compiled_kernel
                return "launched"

        argument = object()
        with (
            patch(
                "helion.runtime.triton.launcher._validate_resident_program_capacity"
            ) as validate,
            patch("helion.runtime.triton.launcher._fit_tensor_memory") as fit,
        ):
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
            compiled_kernel, (argument,), required_programs=7, cluster_size=1
        )
        fit.assert_called_once_with(compiled_kernel)

    def test_launch_pads_shared_memory_for_tensor_memory(self) -> None:
        """A launch whose CTAs wait on each other (a cohort's, a cluster's)
        takes the least dynamic shared memory that lets an SM hold no more of
        its CTAs than their 512 tensor memory columns hold (one past them
        blocks in its allocation, so peers waiting on its work wait forever),
        whatever SMs the block scheduler picks.  It is found on an empty
        kernel's occupancy: the kernel's own, as CUDA reports a tcgen05
        kernel's, says one CTA an SM.  A cohort's capacity counts the padded
        CTAs, in whole clusters a GPC."""
        success = object()
        gpcs = (6, 6)
        per_sm, optin, static = 228 * 1024, 227 * 1024, 2048

        def fit_by_shared(total: int) -> int:
            # The SM reserves per_sm - optin a CTA.
            return per_sm // (total + per_sm - optin)

        def blocks(function: object, threads: int, shared: int) -> int:
            if function == "probe":
                # No static shared memory, few registers.
                return min(32, fit_by_shared(shared))
            # The kernel: 3 CTAs an SM by registers, fewer by shared memory.
            return min(3, fit_by_shared(static + shared))

        def clusters(function: object, config: SimpleNamespace) -> tuple:
            # Clusters stay in their GPC.
            ctas = blocks(function, 128, config.sharedMemBytes)
            return success, sum(ctas * sms // config.gridDimX for sms in gpcs)

        opted_in: list[object] = []
        driver = SimpleNamespace(
            CUresult=SimpleNamespace(CUDA_SUCCESS=success),
            CUfunction=lambda handle: handle,
            CUfunction_attribute=SimpleNamespace(
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=0,
                CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES=1,
            ),
            CUlaunchConfig=SimpleNamespace,
            CUlaunchAttribute=lambda: SimpleNamespace(
                value=SimpleNamespace(clusterDim=SimpleNamespace())
            ),
            CUlaunchAttributeID=SimpleNamespace(
                CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION=0
            ),
            cuModuleLoadData=lambda ptx: (success, "module"),
            cuModuleGetFunction=lambda module, name: (success, "probe"),
            cuModuleUnload=lambda module: (success,),
            cuFuncGetAttribute=lambda attribute, function: (success, static),
            cuFuncSetAttribute=lambda function, attribute, value: (
                opted_in.append((function, value)) or (success,)
            ),
            cuOccupancyMaxActiveBlocksPerMultiprocessor=lambda function, threads, shared: (
                success,
                blocks(function, threads, shared),
            ),
            cuOccupancyMaxActiveClusters=clusters,
        )
        properties = SimpleNamespace(
            multi_processor_count=sum(gpcs),
            shared_memory_per_multiprocessor=per_sm,
            shared_memory_per_block_optin=optin,
        )
        argument = MagicMock(spec=torch.Tensor)
        argument.device = torch.device("cuda", 0)
        arguments = [(argument,)]

        def launch(columns: int, cluster_size: int) -> tuple[int, int, int]:
            """The CTAs an SM holds by shared memory and runs of the launch,
            and its cohort's capacity."""
            # Packed as fbtriton packs it: a preferred cluster shape follows,
            # which the padding passes through.
            compiled = SimpleNamespace(
                run=None,
                function=1,
                packed_metadata=(4, cluster_size, 0, 2, 1, 1),
                metadata=SimpleNamespace(
                    tmem_size=columns, target=SimpleNamespace(arch=100)
                ),
            )
            triton_launcher._fit_tensor_memory(compiled)
            shared = compiled.packed_metadata[2]
            self.assertEqual(compiled.packed_metadata[3:], (2, 1, 1))
            held = fit_by_shared(static + shared)
            if shared:
                # The least padding: a byte less holds another CTA; the kernel
                # may take it.
                self.assertEqual(fit_by_shared(static + shared - 1), held + 1)
                self.assertIn((1, shared), opted_in)
            for required in range(1, 40):
                try:
                    triton_launcher._validate_resident_program_capacity(
                        compiled,
                        arguments[0],
                        required_programs=required,
                        cluster_size=cluster_size,
                    )
                except RuntimeError as error:
                    self.assertIn(f"execute only {required - 1}.", str(error))
                    return min(held, 32), blocks(1, 128, shared), required - 1
            raise AssertionError("no capacity below 40")

        with (
            patch.dict(sys.modules, {"cuda.bindings.driver": driver}),
            patch.object(torch.cuda, "device", lambda _: contextlib.nullcontext()),
            patch.object(torch.cuda, "current_stream", lambda _: None),
            patch.object(torch.cuda, "current_device", lambda: 0),
            patch.object(torch.cuda, "get_device_properties", lambda _: properties),
        ):
            for columns, cluster_size, expected in (
                (0, 1, (32, 3, 36)),
                # 3 CTAs an SM by registers, but the block scheduler is told by
                # shared memory alone: padded to 4.
                (128, 1, (4, 3, 36)),
                (256, 1, (2, 2, 24)),
                (512, 1, (1, 1, 12)),
                (0, 4, (32, 3, 32)),
                (256, 4, (2, 2, 24)),
                # One CTA an SM: one whole cluster of 4 a GPC of 6, not 12 // 4.
                (512, 4, (1, 1, 8)),
            ):
                with self.subTest(columns=columns, cluster_size=cluster_size):
                    self.assertEqual(launch(columns, cluster_size), expected)
            # Only host tensor descriptors: the current device's capacity.
            arguments[0] = (object(),)
            self.assertEqual(launch(512, 4), (1, 1, 8))
            # No whole cluster of 8 fits a GPC of 6 SMs: a cluster refusal.
            compiled = SimpleNamespace(
                function=1,
                packed_metadata=(4, 8, 0),
                metadata=SimpleNamespace(
                    tmem_size=512, target=SimpleNamespace(arch=100)
                ),
            )
            triton_launcher._fit_tensor_memory(compiled)
            with self.assertRaisesRegex(
                RuntimeError, f"^{triton_launcher.CLUSTER_LAUNCH_REFUSED}"
            ):
                triton_launcher._validate_resident_program_capacity(
                    compiled, (argument,), required_programs=8, cluster_size=8
                )

    def test_launch_holds_the_metadata_lock(self) -> None:
        """A launch in clusters, of a resident cohort or of a cooperative grid
        compiles its specialization, patches the compiled kernel's packed metadata (its
        cluster size, its shared memory padded for tensor memory), launches
        and restores it under its kernel's lock, so a concurrent launch of the
        same kernel never sees or undoes another's patch, nor replaces the
        patched kernel by compiling it again; another kernel's launches do not
        wait."""
        other = SimpleNamespace()

        def cluster_grid(
            kernel: SimpleNamespace, grid: tuple[int, ...], size: int
        ) -> tuple[int, ...]:
            kernel.packed_metadata = (4, size, 0)
            return (grid[0] // size, *grid[1:])

        def fit_tensor_memory(kernel: SimpleNamespace) -> None:
            warps, ctas, _ = kernel.packed_metadata
            kernel.packed_metadata = (warps, ctas, 1024)

        class FakeJITFunction:
            def __init__(self) -> None:
                self.compiled_kernel = SimpleNamespace(
                    _init_handles=lambda: None, packed_metadata=(4, 1, 0)
                )
                self.held: list[tuple[bool, bool, object]] = []

            def run(self, *args: object, **kwargs: object) -> object:
                locks = (
                    triton_launcher._patched_launch_lock(self).locked(),
                    triton_launcher._patched_launch_lock(other).locked(),
                )
                if kwargs.get("warmup"):
                    # Compiled under the lock: no other launch's compile can
                    # replace the kernel this launch patches.
                    self.held.append((*locks, "compile"))
                    return self.compiled_kernel
                self.held.append((*locks, self.compiled_kernel.packed_metadata))
                return "launched"

        for options, launched, required in (
            ({"_cluster_size": 2}, (4, 2, 1024), 2),
            ({"_minimum_resident_programs": 8}, (4, 1, 1024), 8),
            ({"launch_cooperative_grid": True}, (4, 1, 1024), None),
        ):
            kernel = FakeJITFunction()
            with (
                self.subTest(options=options),
                patch.object(triton_launcher, "_cluster_grid", cluster_grid),
                patch.object(triton_launcher, "_fit_tensor_memory", fit_tensor_memory),
                patch.object(
                    triton_launcher, "_validate_resident_program_capacity"
                ) as validate,
                patch.object(triton_launcher, "_has_tensor_memory", lambda: True),
            ):
                result = triton_default_launcher(
                    kernel, (8,), object(), num_warps=4, num_stages=1, **options
                )
                self.assertEqual(result, "launched")
                self.assertEqual(
                    kernel.held, [(True, False, "compile"), (True, False, launched)]
                )
                self.assertEqual(kernel.compiled_kernel.packed_metadata, (4, 1, 0))
                # A cohort, or one whole cluster, must be resident.
                self.assertEqual(
                    [call.kwargs["required_programs"] for call in validate.mock_calls],
                    [] if required is None else [required],
                )
                self.assertFalse(triton_launcher._patched_launch_lock(kernel).locked())
        # Without tensor memory a cooperative grid patches nothing: one plain
        # launch, no lock.
        kernel = FakeJITFunction()
        with patch.object(triton_launcher, "_has_tensor_memory", lambda: False):
            triton_default_launcher(
                kernel,
                (8,),
                object(),
                num_warps=4,
                num_stages=1,
                launch_cooperative_grid=True,
            )
        self.assertEqual(kernel.held, [(False, False, (4, 1, 0))])
        # Nor where codegen found the kernel takes none, on any device.
        kernel = FakeJITFunction()
        with patch.object(
            triton_launcher, "_has_tensor_memory", side_effect=AssertionError
        ):
            triton_default_launcher(
                kernel,
                (8,),
                object(),
                num_warps=4,
                num_stages=1,
                launch_cooperative_grid=True,
                _tensor_memory=False,
            )
        self.assertEqual(kernel.held, [(False, False, (4, 1, 0))])

    def test_patched_launch_of_an_unloadable_kernel_raises_its_error(self) -> None:
        """A patched launch loads the compiled kernel before its driver
        queries, so a kernel that cannot load raises its own error (Triton
        keeps such a kernel cached), and a missing cuda-bindings says so."""
        error = RuntimeError("out of resource: shared memory")
        compiled_kernel = SimpleNamespace(
            _init_handles=MagicMock(side_effect=error), packed_metadata=(4, 1, 0)
        )
        kernel = MagicMock()
        kernel.run.return_value = compiled_kernel
        with self.assertRaisesRegex(RuntimeError, "out of resource"):
            triton_default_launcher(
                kernel,
                (8,),
                object(),
                num_warps=4,
                num_stages=1,
                _minimum_resident_programs=8,
            )
        with (
            patch.dict(sys.modules, {"cuda.bindings.driver": None}),
            self.assertRaisesRegex(RuntimeError, "cuda-bindings"),
        ):
            triton_launcher._cuda_driver()

    def test_tensor_memory_targets_and_budget(self) -> None:
        """Tensor memory on tcgen05 targets only (not sm_120), and Triton's
        column budget: 512 columns, 576 on sm_107."""
        for major, expected in ((9, False), (10, True), (11, True), (12, False)):
            triton_launcher._device_has_tensor_memory.cache_clear()
            with (
                self.subTest(major=major),
                patch.object(
                    torch.cuda, "get_device_capability", return_value=(major, 0)
                ),
            ):
                self.assertEqual(triton_launcher._device_has_tensor_memory(0), expected)
        triton_launcher._device_has_tensor_memory.cache_clear()
        for columns, arch, expected in (
            (0, 100, None),
            (64, 100, 8),
            (512, 100, 1),
            (64, 107, 9),
            (512, 107, 1),
        ):
            metadata = SimpleNamespace(
                tmem_size=columns, target=SimpleNamespace(arch=arch)
            )
            with self.subTest(columns=columns, arch=arch):
                self.assertEqual(
                    triton_launcher._tensor_memory_ctas(metadata), expected
                )

    def test_shared_memory_opt_in_only_rises(self) -> None:
        """The dynamic shared memory limit is one attribute a function: a
        launch asking for less than an earlier one's (a cluster launch's
        unpadded size after a padded launch's) keeps the larger limit."""
        success = object()
        opted_in: list[int] = []
        driver = SimpleNamespace(
            CUresult=SimpleNamespace(CUDA_SUCCESS=success),
            CUfunction=lambda handle: handle,
            CUfunction_attribute=SimpleNamespace(
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=0
            ),
            cuFuncSetAttribute=lambda function, attribute, value: (
                opted_in.append(value) or (success,)
            ),
        )
        compiled = SimpleNamespace(function=1)
        with patch.dict(sys.modules, {"cuda.bindings.driver": driver}):
            for shared in (74753, 1024, 74753, 80000):
                self.assertIsNone(triton_launcher._opt_in_shared(compiled, shared))
        self.assertEqual(opted_in, [74753, 80000])

    def test_cluster_launch_refusals_carry_the_prefix(self) -> None:
        """Every refusal of a launch in clusters starts with
        CLUSTER_LAUNCH_REFUSED, which the default launcher raises as
        InvalidConfig: clusters past the portable 8, a grid that does not
        divide into clusters, a kernel compiled for several CTAs (its program
        id is the cluster's), with scratch memory Triton would size by the
        grid of clusters, or whose dynamic shared memory does not fit beside
        the static slab."""
        success = object()
        driver = SimpleNamespace(
            CUresult=SimpleNamespace(CUDA_SUCCESS=success),
            CUfunction=lambda handle: handle,
            CUfunction_attribute=SimpleNamespace(
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=0
            ),
            cuFuncSetAttribute=lambda *args: ("CUDA_ERROR_INVALID_VALUE",),
        )
        plain = {"global_scratch_size": 0, "profile_scratch_size": 0}
        for grid, packed, metadata, size in (
            ((32,), (4, 1, 0), plain, 16),
            ((7,), (4, 1, 0), plain, 2),
            ((8,), (4, 4, 0), plain, 2),
            ((8,), (4, 2, 0), plain, 2),
            ((8,), (4, 1, 0), {**plain, "global_scratch_size": 64}, 2),
            ((8,), (4, 1, 0), {**plain, "profile_scratch_size": 64}, 2),
            ((8,), (4, 1, 200 * 1024), plain, 2),
        ):
            kernel = SimpleNamespace(
                packed_metadata=packed,
                metadata=SimpleNamespace(**metadata),
                run=None,
                function=1,
            )
            with (
                self.subTest(grid=grid, packed=packed, metadata=metadata),
                patch.dict(sys.modules, {"cuda.bindings.driver": driver}),
                self.assertRaisesRegex(
                    RuntimeError, f"^{triton_launcher.CLUSTER_LAUNCH_REFUSED}"
                ),
            ):
                triton_launcher._cluster_grid(kernel, grid, size)

    def test_cross_loop_occupancy_failure_is_an_invalid_config(self) -> None:
        for message in (
            (
                "Cross-loop scheduling requires 16 concurrently resident "
                "programs, but this kernel/device can residently execute only 12."
            ),
            (
                f"{triton_launcher.CLUSTER_LAUNCH_REFUSED}a grid of (7,) does not "
                "divide into clusters of 2"
            ),
            (
                f"{triton_launcher.TENSOR_MEMORY_REFUSED}232448 bytes of dynamic "
                "shared memory still put more than 1 CTAs on an SM"
            ),
        ):
            with (
                self.subTest(message=message),
                patch(
                    "helion.runtime._triton_default_launcher",
                    side_effect=RuntimeError(message),
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


@pytest.mark.cuda
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


@pytest.mark.cuda
@onlyOn(["cuda", "rocm", "xpu"])
@unittest.skipIf(triton_launcher.triton is None, "requires Triton")
@skipIfRocm("tensor memory and the CUDA driver queries are NVIDIA's")
@skipIfFn(
    lambda: not triton_launcher._has_tensor_memory(),
    "tensor memory needs a tcgen05 GPU (sm_100, sm_103, sm_110)",
)
@onlyBackends(["triton"])
@skipIfTileIR("the timed kernel reads %smid and %globaltimer by inline PTX")
class TestTensorMemoryPadding(unittest.TestCase):
    def test_padding_holds_the_block_scheduler(self) -> None:
        """On the GPU, a launch padded for a tensor memory footprint never has
        more CTAs on one SM than the footprint allows (their SMs and run
        times recorded), and computes correctly: a tcgen05 kernel, whose own
        occupancy CUDA reports as one CTA an SM while the block scheduler
        places several, and one past 48 KB of shared memory."""
        sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        grid = (sms * 8,)
        for dtype, shape, k, stages, ieee in (
            (torch.bfloat16, (128, 16, 16), 8192, 1, False),
            (torch.float32, (64, 64, 64), 2048, 3, True),
        ):
            bm, bn, bk = shape
            x = torch.randn([bm, k], device=DEVICE, dtype=dtype)
            w = torch.randn([k, bn], device=DEVICE, dtype=dtype)
            y = torch.empty([grid[0], bm, bn], device=DEVICE)
            record = torch.zeros([grid[0], 3], device=DEVICE, dtype=torch.int64)

            launch = functools.partial(
                _timed_dot.run,
                *(x, w, y, record, k, bm, bn, bk, ieee),
                grid=grid,
                num_warps=4,
                num_stages=stages,
            )
            compiled = launch(warmup=True)
            compiled._init_handles()
            metadata, packed = compiled.metadata, compiled.packed_metadata
            self.assertEqual(packed[2] > 48 * 1024, ieee)
            for columns in (256, 512):
                fit = 512 // columns
                with self.subTest(ieee=ieee, columns=columns):
                    compiled.metadata = metadata._replace(tmem_size=columns)
                    try:
                        triton_launcher._fit_tensor_memory(compiled)
                        y.zero_()
                        launch(warmup=False)
                        torch.cuda.synchronize()
                    finally:
                        # The padding is cached a kernel (its footprint is
                        # fixed).
                        vars(compiled).pop("_helion_tensor_memory_shared", None)
                        compiled.metadata, compiled.packed_metadata = metadata, packed
                    self.assertLessEqual(_most_ctas_on_an_sm(record), fit)
                    expected = (x.float() @ w.float()).expand_as(y)
                    torch.testing.assert_close(y, expected, rtol=2e-2, atol=2e-1)


def _most_ctas_on_an_sm(record: torch.Tensor) -> int:
    """The most programs that ran on one SM at once, by their recorded SMs,
    starts and ends (an end before a start at the same time)."""
    events: dict[int, list[tuple[int, int]]] = {}
    for sm, start, end in record.tolist():
        events.setdefault(sm, []).extend([(start, 1), (end, -1)])
    most = 0
    for sm_events in events.values():
        running = 0
        for _, change in sorted(sm_events):
            running += change
            most = max(most, running)
    return most


@helion.kernel(backend="triton")
def _two_phases(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] * 2
    hl.barrier()
    for tile in hl.tile(x.size(0)):
        out[tile] = y[tile] + 1
    return out


@helion.kernel(backend="triton")
def _two_phases_after_dot(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    y = torch.empty([x.size(0), w.size(1)], dtype=torch.float32, device=x.device)
    out = torch.empty_like(y)
    for tm, tn in hl.tile([x.size(0), w.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk], w[tk, tn], acc=acc)
        y[tm, tn] = acc
    hl.barrier()
    for tm, tn in hl.tile([x.size(0), w.size(1)]):
        out[tm, tn] = y[tm, tn] + 1
    return out


@skipUnlessBackends(["triton"])
@skipIfRefEager("renders generated code")
def test_cooperative_launch_without_tensor_memory(cpu_codegen: None) -> None:
    """Codegen tells the launcher when a cooperative grid's kernel takes no
    tensor memory (no matmul), sparing its launch the patched path; a kernel
    with a dot keeps it."""
    persistent = {"pid_type": "persistent_blocked"}
    plain = _two_phases._bind_isolated((torch.empty(64),)).to_code(
        helion.Config(block_sizes=[32, 32], **persistent)
    )
    a = torch.empty([64, 64], dtype=torch.bfloat16)
    dot = _two_phases_after_dot._bind_isolated((a, a)).to_code(
        helion.Config(block_sizes=[32] * 5, **persistent)
    )
    for code in (plain, dot):
        assert "launch_cooperative_grid=True" in code
    assert "_tensor_memory=False" in plain
    assert "_tensor_memory" not in dot
