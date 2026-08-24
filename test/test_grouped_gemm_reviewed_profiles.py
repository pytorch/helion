from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

from pretuned_kernels.grouped_gemm_deepgemm import (
    grouped_gemm_deepgemm as pretuned_deepgemm,
)
from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles
import pytest
import torch

from helion._compat import requires_cuda_version
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])


def _require_runtime_cuda13_sm100() -> None:
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    if DEVICE.type != "cuda":
        pytest.skip("tcgen05 selected-path codegen needs CUDA fake inputs")
    if not requires_cuda_version("13"):
        pytest.skip("tcgen05 selected-path runtime needs CUDA >= 13")
    from helion._compiler.cute.mma_support import get_cute_mma_support

    with torch.cuda.device(DEVICE):
        major, _minor = torch.cuda.get_device_capability(DEVICE)
    if major < 10:
        pytest.skip("tcgen05 requires SM100+")
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")


def _packed_args(
    groups: int,
    n: int,
    k: int,
    *,
    source_m_tile: int,
    b_major: str,
    active_groups: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    starts = torch.arange(active_groups, device=DEVICE) * source_m_tile
    a_packed = torch.zeros(
        (active_groups * source_m_tile, k),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    a_packed[starts, :16] = 1
    physical_shape = (groups, k, n) if b_major == "n" else (groups, n, k)
    b_grouped = torch.zeros(
        physical_shape,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    if b_major == "n":
        b_grouped = b_grouped.transpose(1, 2)
    b_grouped[:active_groups, :, :16] = 1
    worklist = torch.tensor(
        [
            (
                group,
                group * source_m_tile if group < active_groups else 0,
                1 if group < active_groups else 0,
                source_m_tile if group < active_groups else 0,
            )
            for group in range(groups)
        ],
        device=DEVICE,
        dtype=torch.int32,
    )
    return (a_packed, b_grouped, worklist), starts


def _reviewed_args(
    shape: tuple[int, int, int, int],
    *,
    b_major: str,
    all_groups_active: bool,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    groups, _expected_m, n, k = shape
    return _packed_args(
        groups,
        n,
        k,
        source_m_tile=(
            reviewed_profiles.exact_reviewed_worklist_profile(*shape).source_m_tile
        ),
        b_major=b_major,
        active_groups=groups if all_groups_active else 1,
    )


def _assert_active_rows(
    output: torch.Tensor,
    starts: torch.Tensor,
) -> None:
    expected = torch.full(
        (starts.numel(), output.size(1)),
        16,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(output[starts], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output[starts + 1],
        torch.zeros_like(expected),
        rtol=0,
        atol=0,
    )


@pytest.fixture
def evaluated_kernel() -> Iterator[Any]:
    with patch.dict(
        os.environ,
        {"HELION_AOT_MODE": "evaluate", "HELION_CUTE_MMA_IMPL": "tcgen05"},
        clear=False,
    ):
        kernel = pretuned_deepgemm.grouped_gemm_deepgemm
        kernel.reset()
        try:
            yield kernel
        finally:
            kernel.reset()


def _run_reviewed_shapes(
    kernel: Any,
    shapes: tuple[tuple[int, int, int, int], ...],
    *,
    all_groups_active: bool,
) -> tuple[list[Any], list[str | None]]:
    bounds = []
    cache_keys = []
    for shape in shapes:
        profile = reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES[shape]
        args, starts = _reviewed_args(
            shape,
            b_major=profile.b_major,
            all_groups_active=all_groups_active,
        )
        output = kernel(*args)
        torch.cuda.synchronize()
        _assert_active_rows(output, starts)
        bound = kernel.bind(args)
        bounds.append(bound)
        cache_keys.append(bound.backend_cache_key())
        del output, args
        torch.cuda.empty_cache()
    return bounds, cache_keys


@pytest.mark.parametrize(
    "order",
    (
        pytest.param(("k", "n"), id="k-then-n"),
        pytest.param(("n", "k"), id="n-then-k"),
    ),
)
def test_pretuned_aot_shared_config_specializes_b_major(
    order: tuple[str, str],
    evaluated_kernel: Any,
) -> None:
    _require_runtime_cuda13_sm100()

    profiles = {
        "k": (32, 192, 6144, 7168),
        "n": (32, 192, 7168, 3072),
    }
    profile_values = [
        reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES[shape]
        for shape in profiles.values()
    ]
    assert profile_values[0].config_name == profile_values[1].config_name
    bounds, cache_keys = _run_reviewed_shapes(
        evaluated_kernel,
        tuple(profiles[layout] for layout in order),
        all_groups_active=True,
    )

    assert bounds[0] is not bounds[1]
    assert None not in cache_keys
    assert len(set(cache_keys)) == 2


@pytest.mark.parametrize(
    ("prior_shape", "next_shape"),
    (
        pytest.param(
            (32, 192, 4096, 2048),
            (6, 1024, 4096, 2048),
            id="group-count-transition",
        ),
        pytest.param(
            (32, 20, 4096, 4096),
            (32, 20, 4096, 2048),
            id="k-transition",
        ),
    ),
)
def test_pretuned_aot_shared_config_preserves_profile_specialization(
    prior_shape: tuple[int, int, int, int],
    next_shape: tuple[int, int, int, int],
    evaluated_kernel: Any,
) -> None:
    _require_runtime_cuda13_sm100()
    prior_profile = reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES[prior_shape]
    next_profile = reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES[next_shape]
    assert prior_profile.config_name == next_profile.config_name
    assert prior_profile.b_major == next_profile.b_major == "n"

    bounds, backend_cache_keys = _run_reviewed_shapes(
        evaluated_kernel,
        (prior_shape, next_shape),
        all_groups_active=False,
    )

    assert bounds[0] is not bounds[1]
    assert None not in backend_cache_keys
    assert len(set(backend_cache_keys)) == 2


def test_pretuned_aot_canonical_three_argument_call(evaluated_kernel: Any) -> None:
    _require_runtime_cuda13_sm100()

    groups = 2
    n = 128
    k = 128
    source_m_tile = reviewed_profiles.LEGACY_M_ALIGNMENT
    args, starts = _packed_args(
        groups,
        n,
        k,
        source_m_tile=source_m_tile,
        b_major="k",
        active_groups=groups,
    )

    output = evaluated_kernel(*args)
    torch.cuda.synchronize()
    _assert_active_rows(output, starts)


def test_independent_kernel_factory_preserves_packing_bindings() -> None:
    _require_runtime_cuda13_sm100()

    groups, n, k = 2, 128, 128

    source_tiles = (
        reviewed_profiles.SMALL_M_ALIGNMENT,
        reviewed_profiles.LEGACY_M_ALIGNMENT,
        reviewed_profiles.PROFILED_M_ALIGNMENT,
    )
    kernels = []
    bound_cases = []
    with patch.dict(
        os.environ,
        {"HELION_AOT_MODE": "evaluate", "HELION_CUTE_MMA_IMPL": "tcgen05"},
        clear=False,
    ):
        kernels = [
            pretuned_deepgemm.create_grouped_gemm_deepgemm_kernel()
            for _ in source_tiles
        ]
        for kernel, source_m_tile in zip(kernels, source_tiles, strict=True):
            args, starts = _packed_args(
                groups,
                n,
                k,
                source_m_tile=source_m_tile,
                b_major="k",
                active_groups=groups,
            )
            bound = kernel.bind(args)
            seed = next(
                config
                for config in bound.config_spec.compiler_seed_configs
                if config.config.get("tcgen05_grouped_worklist_source_m_tile")
                == source_m_tile
            )
            effective = bound.config_spec.normalized_config(seed)
            assert seed.config != effective.config
            assert seed.config["tcgen05_grouped_worklist_source_m_tile"] == (
                source_m_tile
            )
            assert effective.config["tcgen05_grouped_worklist_source_m_tile"] == (
                source_m_tile
            )
            bound.set_config(effective)
            bound_cases.append((bound, args, effective, starts))

        assert len({id(kernel) for kernel in kernels}) == len(source_tiles)
        assert len({id(case[0]) for case in bound_cases}) == len(source_tiles)
        for kernel, (bound, args, _effective, starts) in zip(
            kernels,
            bound_cases,
            strict=True,
        ):
            assert bound.kernel is kernel
            output = bound(*args)
            _assert_active_rows(output, starts)
        torch.cuda.synchronize()
