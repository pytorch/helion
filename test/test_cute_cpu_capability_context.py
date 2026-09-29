from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import _compat
from helion._testing import skipUnlessBackends
from helion.autotuner import config_spec
import helion.language as hl
from helion.language import loops

pytest.importorskip("triton")


def _cached_capabilities():
    return (
        _compat._supports_maxnreg,
        _compat._supports_tensor_descriptor,
        _compat._is_hip,
        loops.use_tileir_tunables,
        config_spec.num_compute_units,
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_cpu_capabilities_preserve_process_caches(nested: bool, raises: bool) -> None:
    original = _cached_capabilities()
    before = [function.cache_info() for function in original]
    expected = (
        pytest.raises(RuntimeError, match="context exit") if raises else nullcontext()
    )
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        expected,
        _mock_cuda_unavailable(),
    ):
        outer = _cached_capabilities()
        with _mock_cuda_unavailable() if nested else nullcontext():
            assert torch.cuda.is_available() is False
            assert [function() for function in _cached_capabilities()] == [
                False,
                False,
                False,
                False,
                128,
            ]
        assert all(a is b for a, b in zip(outer, _cached_capabilities(), strict=True))
        if raises:
            raise RuntimeError("context exit")
    assert all(a is b for a, b in zip(original, _cached_capabilities(), strict=True))
    assert [function.cache_info() for function in original] == before


def _pointwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        output[tile] = x[tile] + y[tile]
    return output


def _check_codegen_cache_isolation(warm: bool) -> None:
    original = _cached_capabilities()
    assert original[0].cache_info().currsize == 0
    assert not torch.cuda.is_initialized()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.version.hip", None),
        patch("torch.version.xpu", None),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
        patch(
            "helion._compat.DeviceProperties.create",
            return_value=SimpleNamespace(type="cuda", warp_size=32),
        ),
        patch(
            "torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")
        ) as lazy,
    ):
        if warm:
            assert [function() for function in original] == [
                True,
                True,
                False,
                False,
                148,
            ]
        before = [function.cache_info() for function in original]
        args = (torch.empty(64), torch.empty(64))
        with _mock_cuda_unavailable():
            kernel = helion.kernel(_pointwise, backend="cute", autotune_effort="none")
            code = _cpu_bind(kernel, args).to_code(helion.Config(block_sizes=[32]))
            assert "@cute.kernel" in code
        assert all(
            a is b for a, b in zip(original, _cached_capabilities(), strict=True)
        )
        assert [function.cache_info() for function in original] == before
        assert _compat.supports_maxnreg() is True
        # Model only the uncached CUDA register/warp facts used by normalization.
        with (
            patch("helion.autotuner.config_spec._regs_per_block", return_value=65536),
            patch(
                "helion.autotuner.config_spec.warps_to_threads",
                side_effect=lambda n: 32 * n,
            ),
            patch("helion.runtime.get_num_sm", return_value=148),
        ):
            kernel = helion.kernel(_pointwise, backend="triton", autotune_effort="none")
            code = _cpu_bind(kernel, args).to_code(
                helion.Config(
                    block_sizes=[32],
                    num_warps=4,
                    pid_type="persistent_blocked",
                    num_sm_multiplier=3,
                    maxnreg=100,
                )
            )
        assert "maxnreg=100" in code
        assert "_NUM_SM * 3" in code
        lazy.assert_not_called()
        assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("warm", [False, True])
@skipUnlessBackends(["cute"])
def test_cpu_codegen_keeps_later_triton_maxnreg(warm: bool) -> None:
    # Each child starts with a real cold cache; no process-wide cache is cleared.
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root))
    environment.pop("HELION_BACKEND", None)
    environment.pop("HELION_AUTOTUNE_EFFORT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "from test.test_cute_cpu_capability_context import _check_codegen_cache_isolation; "
                f"_check_codegen_cache_isolation({warm!r})"
            ),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
