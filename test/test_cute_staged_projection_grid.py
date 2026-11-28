from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from helion import exc
from helion._compiler.cute.autotune.config_keys import PHASE_PARTICIPATION_KEY
from helion._compiler.cute.autotune.config_keys import STAGED_GRID_KEY
from helion._compiler.cute.autotune.staged_projection_grid import normalize_staged_grid
from helion._compiler.cute.autotune.staged_projection_grid import staged_task_bound
from helion.runtime.cute import compile_cache
from helion.runtime.cute import cooperative_residency as residency

pytestmark = pytest.mark.cpu_only


def test_largest_domain_counts_independent_instances_and_excludes_row_tails():
    spec = Mock()
    spec.has_explicit_pdl = False
    spec.num_sm = 7
    spec.cute_staged_row_domains = ((0, 5, 32), (1, 19, 30))
    spec.block_sizes.valid_block_ids.return_value = []
    spec.cute_tile_fixed_block_sizes = {0: 8, 1: 4}
    config = {
        "cute_staged_indexed_rows": True,
        "pid_type": "persistent_interleaved",
        "cross_loop_pipeline": "barrier",
        "cute_defer_tile_reduce": True,
        "block_sizes": [],
    }
    assert staged_task_bound(spec, config) == 20
    spec.has_explicit_pdl = True
    assert staged_task_bound(spec, config | {"cute_pdl": False}) == 0
    spec.has_explicit_pdl = False
    for change in (
        {"cute_pdl": True},
        {"cute_cluster_n": 2},
        {"pid_type": "persistent_blocked"},
        {"num_sm_multiplier": 2},
        {"cute_staged_indexed_rows": False},
    ):
        assert staged_task_bound(spec, config | change) == 0

    for changes in ({STAGED_GRID_KEY: 1}, {STAGED_GRID_KEY: 2, "cute_pdl": True}):
        invalid = config | {PHASE_PARTICIPATION_KEY: True} | changes
        with pytest.raises(exc.InvalidConfig, match="active phase joins require"):
            normalize_staged_grid(spec, invalid.copy(), fix_invalid=False)
        normalize_staged_grid(spec, invalid, fix_invalid=True)
        assert PHASE_PARTICIPATION_KEY not in invalid
        assert STAGED_GRID_KEY not in invalid


def test_compiled_stride_is_specialized_to_the_actual_grid_and_context():
    schema = (("tensor",), ("scalar_constexpr", "int", 17, 17))
    assert residency.stride_from_schema(schema, 1) == 17
    for index in (None, 0, 2, -1, True):
        with pytest.raises(exc.InvalidConfig, match="integer phase-stride"):
            residency.stride_from_schema(schema, index)
    proof = residency.ProvenCooperativeGrids(17)
    compiled = Mock()
    args = ("operand", 17, 1, 1, "stream")
    with (
        patch.object(
            residency,
            "stream_resources",
            side_effect=[(20, 9), (20, 9), (21, 9), (21, 9)],
        ),
        patch.object(residency, "check_cooperative_residency") as check,
    ):
        with pytest.raises(exc.InvalidConfig, match="physical grid"):
            proof.check(compiled, ("operand", 16, 1, 1, "stream"))
        proof.check(compiled, args)
        proof.check(compiled, args)
        proof.check(compiled, args)
        # A distinct loaded executable in the *same* context needs its own
        # proof, even when another first-compile thread raced with it.
        second = Mock()
        proof.check(second, args)
        assert [c.args[0] for c in check.call_args_list] == [compiled, compiled, second]


def test_driver_admission_uses_actual_function_block_and_dynamic_shared_bytes():
    driver = Mock(wraps=residency.driver)
    ok = residency.driver.CUresult.CUDA_SUCCESS
    attrs = residency.driver.CUdevice_attribute
    driver.CUresult = residency.driver.CUresult
    driver.CUstreamCaptureStatus = residency.driver.CUstreamCaptureStatus
    driver.CUdevice_attribute = attrs
    driver.cuCtxGetDevice.return_value = (ok, 3)
    driver.cuDeviceGetAttribute.side_effect = lambda mode, device: (
        ok,
        5 if mode == attrs.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT else 1,
    )
    driver.cuStreamIsCapturing.return_value = (
        ok,
        residency.driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE,
    )
    driver.cuStreamCreate.return_value = (ok, 7)
    driver.cuStreamBeginCapture.return_value = (ok,)
    driver.cuStreamEndCapture.return_value = (ok, 8)
    driver.cuGraphGetNodes.side_effect = lambda graph, count: (ok, [9], 1)
    params = SimpleNamespace(
        gridDimX=1,
        gridDimY=1,
        gridDimZ=1,
        blockDimX=4,
        blockDimY=32,
        blockDimZ=1,
        sharedMemBytes=11776,
        func=51,
    )
    driver.cuGraphKernelNodeGetParams.return_value = (ok, params)
    driver.cuGraphKernelNodeGetAttribute.return_value = (
        ok,
        SimpleNamespace(cooperative=True),
    )
    driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.return_value = (ok, 2)
    driver.cuGraphDestroy.return_value = (ok,)
    driver.cuStreamDestroy.return_value = (ok,)
    compiled = Mock()
    with patch.object(residency, "driver", driver):
        residency.check_cooperative_residency(compiled, (99, 10, 1, 1, 6), 5)
        with pytest.raises(exc.InvalidConfig, match="exceeds actual residency"):
            residency.check_cooperative_residency(compiled, (99, 11, 1, 1, 6), 5)
    driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.assert_called_with(
        51, 128, 11776
    )
    assert driver.cuGraphDestroy.call_count == driver.cuStreamDestroy.call_count == 2
    assert all(call.args == (99, 1, 1, 1, 7) for call in compiled.call_args_list)
    driver.cuGraphLaunch.assert_not_called()


def test_green_or_sm_limited_stream_cannot_reuse_an_ordinary_context_proof():
    driver = Mock(wraps=residency.driver)
    ok = residency.driver.CUresult.CUDA_SUCCESS
    driver.CUresult = residency.driver.CUresult
    driver.CUdevice_attribute = residency.driver.CUdevice_attribute
    driver.CUdevResourceType = residency.driver.CUdevResourceType
    driver.cuCtxGetCurrent.return_value = (ok, 11)
    driver.cuCtxGetDevice.return_value = (ok, 0)
    driver.cuCtxGetId.return_value = (ok, 19)
    driver.cuDeviceGetAttribute.return_value = (ok, 10)
    driver.cuStreamGetCtx_v2.return_value = (ok, 11, 0)
    driver.cuCtxGetDevResource.return_value = (
        ok,
        SimpleNamespace(sm=SimpleNamespace(smCount=10)),
    )
    compiled = Mock()
    proof = residency.ProvenCooperativeGrids(15)
    args = (15, 1, 1, 5)
    with (
        patch.object(residency, "driver", driver),
        patch.object(residency, "check_cooperative_residency") as probe,
    ):
        proof.check(compiled, args)
        driver.cuStreamGetCtx_v2.return_value = (ok, 11, 31)
        with pytest.raises(exc.InvalidConfig, match="regular stream"):
            proof.check(compiled, (*args[:-1], 6))
        driver.cuStreamGetCtx_v2.return_value = (ok, 12, None)
        with pytest.raises(exc.InvalidConfig, match="regular stream"):
            proof.check(compiled, (*args[:-1], 7))
        driver.cuStreamGetCtx_v2.return_value = (ok, 11, None)
        driver.cuCtxGetDevResource.return_value = (
            ok,
            SimpleNamespace(sm=SimpleNamespace(smCount=3)),
        )
        with pytest.raises(exc.InvalidConfig, match="full SM allocation"):
            proof.check(compiled, args)
        assert probe.call_count == 1


def test_racing_first_compiles_each_probe_the_callable_that_runs():
    rendezvous = Barrier(2)
    proven = set()
    args = (11, 1, 1, 3)

    def compile_image(*_args):
        def launch(*launch_args):
            assert launch in proven and launch_args == args
            return launch

        # Both threads reach compilation before either caches its host image.
        rendezvous.wait(timeout=5)
        return launch

    def check(compiled, args, sms):
        proven.add(compiled)

    wrapper = compile_cache.CompiledCuteLauncher(object(), None, expected_grid_x=11)
    with (
        patch.object(compile_cache, "compile_with_options", side_effect=compile_image),
        patch.object(residency, "stream_resources", return_value=(19, 8)),
        patch.object(residency, "check_cooperative_residency", side_effect=check),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        left = pool.submit(wrapper, *args)
        right = pool.submit(wrapper, *args)
        assert {left.result(), right.result()} == proven
    assert len(proven) == 2
