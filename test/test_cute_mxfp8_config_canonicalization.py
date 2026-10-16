"""Proof-scoped native MXFP8 source deduplication and fallback independence."""

from __future__ import annotations

from itertools import product

import pytest
import torch

from test.test_cute_mxfp8_mma import _config
from test.test_cute_mxfp8_mma import _cpu_inputs
from test.test_cute_mxfp8_mma import _cpu_target
from test.test_cute_mxfp8_mma import _mxfp8
from test.test_cute_mxfp8_mma import _wrapper_plan

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner import EnumFragment

pytestmark = skipUnlessBackends(["cute"])

_SOURCE_ONLY_CHOICES: tuple[dict[str, object], ...] = (
    {"block_sizes": [16, 64]},
    {"cute_vector_widths": [2, 4, 2, 4], "cute_lane_layouts": ["strided"] * 4},
    {"cute_pdl": True, "cute_collective_mma": True, "cute_collective_stages": 3},
    {"loop_orders": [[1, 0]], "cute_collective_copy": "tma"},
)


@pytest.mark.parametrize("changes", _SOURCE_ONLY_CHOICES)
def test_cpu_source_only_choices_share_native_source(
    changes: dict[str, object],
) -> None:
    initialized = torch.cuda.is_initialized()
    with _cpu_target():
        bound = _mxfp8._bind_isolated(_cpu_inputs((137, 273, 288)))
        base = _config(bound)
        varied = _config(bound, **changes)
        normalized = bound.config_spec.normalized_config(base)
        assert bound.config_spec.normalized_config(varied) == normalized
        original_source, varied_source = bound.to_code(base), bound.to_code(varied)
        assert _wrapper_plan(original_source)["format"] == "mxfp8"
        assert original_source == varied_source
        with bound.env:
            generation = bound.config_spec.create_config_generation()
            assert generation.canonicalize_flat(
                generation.flatten(base)
            ) == generation.canonicalize_flat(generation.flatten(varied))
            assert generation.unflatten(generation.flatten(base)) == normalized
    assert torch.cuda.is_initialized() is initialized


def test_cpu_native_geometry_is_independent_and_complete() -> None:
    fields = (
        "cute_scaled_tile_n",
        "cute_scaled_tile_k",
        "cute_scaled_stages",
        "cute_scaled_cluster_m",
        "cute_scaled_persistent",
    )
    geometries = tuple(
        product((64, 128, 256), (128, 256), (2, 3, 4, 5), (1, 2), (False, True))
    )
    with _cpu_target():
        bound = _mxfp8._bind_isolated(_cpu_inputs((137, 273, 288)))
        normalized_geometries = set()
        for geometry in geometries:
            config = _config(bound, **dict(zip(fields, geometry, strict=True)))
            normalized = bound.config_spec.normalized_config(config)
            result = tuple(normalized[key] for key in fields)
            assert result == geometry
            normalized_geometries.add(result)
        assert len(normalized_geometries) == 96


def test_cpu_scalar_choices_are_not_canonicalized() -> None:
    with _cpu_target():
        bound = _mxfp8._bind_isolated(_cpu_inputs((137, 273, 288)))
        baseline = _config(bound, cute_scaled_mma=False)
        changed = _config(
            bound, cute_scaled_mma=False, block_sizes=[16, 64], cute_pdl=True
        )
        first = bound.config_spec.normalized_config(baseline)
        second = bound.config_spec.normalized_config(changed)
        assert first != second
        assert second["block_sizes"] == [16, 64]
        assert second["cute_pdl"] is True


def test_cpu_native_root_preserves_user_tunables() -> None:
    with _cpu_target():
        bound = _mxfp8._bind_isolated(_cpu_inputs((137, 273, 288)))
        bound.config_spec.user_defined_tunables["application_choice"] = EnumFragment(
            choices=(0, 1)
        )
        baseline = bound.config_spec.normalized_config(
            _config(bound, application_choice=0)
        )
        changed = bound.config_spec.normalized_config(
            _config(bound, application_choice=1)
        )
        assert baseline["application_choice"] == 0
        assert changed["application_choice"] == 1
        assert baseline != changed


def test_cpu_nonflat_native_config_still_declines() -> None:
    with _cpu_target():
        bound = _mxfp8._bind_isolated(_cpu_inputs((137, 273, 288)))
        config = _config(bound, pid_type="persistent_interleaved", block_sizes=[16, 64])
        with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
            bound.to_code(config)


def test_cuda_source_only_choices_preserve_native_results() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native MXFP8 required")
    torch.manual_seed(0)
    m, n, k = 73, 273, 288
    values = (
        (torch.randn((m, k), device=DEVICE) / 3).to(torch.float8_e4m3fn),
        (torch.randn((n, k), device=DEVICE) / 3).to(torch.float8_e4m3fn),
        torch.randint(125, 129, (m, k // 32), dtype=torch.uint8, device=DEVICE),
        torch.randint(125, 129, (n, k // 32), dtype=torch.uint8, device=DEVICE),
        torch.empty((m, n), dtype=torch.bfloat16, device=DEVICE),
    )
    bound = _mxfp8._bind_isolated(values)
    baseline = _config(bound)
    expected = bound.compile_config(baseline)(*values).clone()
    for changes in _SOURCE_ONLY_CHOICES:
        actual = bound.compile_config(_config(bound, **changes))(*values)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)
