from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_source64 import _bound
from test.test_cute_grouped_source64 import _wide
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_shared_rhs_grouped import _target

from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES,
)
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.runtime.cute.launcher import _tcgen05_grouped_device_source_m_tile_supported
from helion.runtime.cute.launcher import _tcgen05_grouped_device_split_total_clusters
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_device_split_sizes

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@pytest.fixture(scope="module")
def generated_plans() -> list[dict[str, object]]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        _target(),
    ):
        bound, args = _bound()
        assert args[2].dtype == torch.int32
        return _plans(bound.to_code(_wide(bound)))


def test_compiler_profile_reaches_both_device_offset_gates(
    generated_plans: list[dict[str, object]],
) -> None:
    grouped, ab, d = generated_plans
    profile = cast("dict[str, object]", grouped["source64_profile"])
    assert profile["ab_stage_count"] == ab["ab_stage_count"] == 4
    assert profile["c_stage_count"] == d["c_stage_count"] == 2
    assert profile["input_dtype"] == ab["input_dtype"] == d["output_dtype"]
    assert profile["acc_dtype"] == ab["acc_dtype"] == "cutlass.Float32"
    assert profile["offset_dtype"] == "torch.int32"
    assert profile["acc_stage_count"] == 2 and profile["consumer_regs"] == 256
    assert profile["full_coverage_mode"] == "fixed_tma_dense_local"
    assert profile["consumer_local"] is True and profile["use_2cta_instrs"] is False
    assert _tcgen05_grouped_device_source_m_tile_supported(grouped)
    _validate_tcgen05_grouped_device_split_sizes(
        grouped, torch.tensor([0, 64, 256, 448, 640], dtype=torch.int32)
    )
    assert _tcgen05_grouped_device_split_total_clusters(grouped) == 40


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("full_coverage_mode", "fixed_tma_dense"),
        ("full_coverage_mode", "off"),
        ("consumer_local", False),
        ("consumer_local", 1),
        ("use_2cta_instrs", True),
        ("use_2cta_instrs", 0),
        ("ab_stage_count", 2),
        ("ab_stage_count", 4.0),
        ("acc_stage_count", 1),
        ("c_stage_count", 1),
        ("consumer_regs", 240),
        ("input_dtype", "cutlass.Float16"),
        ("acc_dtype", "cutlass.BFloat16"),
        ("offset_dtype", "torch.int64"),
        ("unproved_extra_field", True),
    ],
)
def test_profile_neighbors_fail_both_host_gates(
    generated_plans: list[dict[str, object]], key: str, value: object
) -> None:
    plan = deepcopy(generated_plans[0])
    cast("dict[str, object]", plan["source64_profile"])[key] = value
    _reject_both(plan)


@pytest.mark.parametrize("profile", [None, {}, [], "source64", True])
def test_unproved_source64_profile_is_rejected(
    generated_plans: list[dict[str, object]], profile: object
) -> None:
    plan = deepcopy(generated_plans[0])
    plan["source64_profile"] = profile
    _reject_both(plan)
    plan.pop("source64_profile")
    _reject_both(plan)


@pytest.mark.parametrize(
    "missing",
    [
        "full_coverage_mode",
        "consumer_local",
        "use_2cta_instrs",
        "ab_stage_count",
        "acc_stage_count",
        "c_stage_count",
        "consumer_regs",
        "input_dtype",
        "acc_dtype",
        "offset_dtype",
    ],
)
def test_each_profile_fact_is_required(
    generated_plans: list[dict[str, object]], missing: str
) -> None:
    plan = deepcopy(generated_plans[0])
    cast("dict[str, object]", plan["source64_profile"]).pop(missing)
    _reject_both(plan)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kind", "tcgen05_ab_tma"),
        ("scheduler_mode", "static"),
        ("bm", 256),
        ("bn", 32),
        ("bk", 64),
        ("cluster_m", 2),
        ("cluster_m", True),
        ("cluster_n", 2),
        ("orientation", "mn"),
        ("shared_rhs", False),
        ("worklist_metadata", False),
        ("device_split_sizes", False),
        ("device_layout_kind", "split_sizes"),
        ("dynamic_ab_tensormaps", False),
        ("dynamic_ab_tensormap_rank", 3),
        ("dynamic_d_tensormap", False),
        ("fixed_tensormaps", True),
        ("direct_pointer_metadata", True),
        ("external_direct_pointer_metadata", True),
        ("use_2cta_instrs", True),
        ("m_size", 672),
        ("n_size", 160),
        ("k_total_size", 192),
        ("group_count", True),
        ("group_count", 0),
        ("group_count", 1 << 30),
        ("m_size", 1 << 30),
        ("n_size", 1 << 30),
    ],
)
def test_profile_does_not_admit_wrong_geometry_or_descriptor_path(
    generated_plans: list[dict[str, object]], key: str, value: object
) -> None:
    plan = deepcopy(generated_plans[0])
    plan[key] = value
    _reject_both(plan)


def _reject_both(plan: dict[str, object]) -> None:
    assert not _tcgen05_grouped_device_source_m_tile_supported(plan)
    with pytest.raises(BackendUnsupported):
        _validate_tcgen05_grouped_device_split_sizes(
            plan, torch.empty(5, dtype=torch.int32)
        )
    with pytest.raises(BackendUnsupported):
        _tcgen05_grouped_device_split_total_clusters(plan)


def test_float_width_is_rejected_by_the_existing_integer_parser(
    generated_plans: list[dict[str, object]],
) -> None:
    plan = deepcopy(generated_plans[0])
    plan["source_m_tile"] = 64.0
    assert not _tcgen05_grouped_device_source_m_tile_supported(plan)
    with pytest.raises(AssertionError):
        _validate_tcgen05_grouped_device_split_sizes(
            plan, torch.empty(5, dtype=torch.int32)
        )
    with pytest.raises(AssertionError):
        _tcgen05_grouped_device_split_total_clusters(plan)


def test_source64_requires_actual_int32_offsets_without_reading_values(
    generated_plans: list[dict[str, object]],
) -> None:
    plan = generated_plans[0]
    # False certificates are clipped and scheduled on the device; host admission
    # must not read values or require the dense-union certificate to be true.
    offsets = torch.tensor([-10, 513, 1, 5000, -20], dtype=torch.int32)
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("value read")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("value read")),
    ):
        _validate_tcgen05_grouped_device_split_sizes(plan, offsets)
    with pytest.raises(BackendUnsupported, match="Int32 layout dtype"):
        _validate_tcgen05_grouped_device_split_sizes(plan, offsets.to(torch.int64))


@pytest.mark.parametrize("source_m_tile", [32, 224, 256])
@pytest.mark.parametrize("block_k", [64, 128])
def test_legacy_source_profiles_do_not_acquire_the_source64_restrictions(
    generated_plans: list[dict[str, object]], source_m_tile: int, block_k: int
) -> None:
    assert TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES == (32, 224, 256)
    plan = deepcopy(generated_plans[0])
    plan.pop("source64_profile")
    plan.update(source_m_tile=source_m_tile, bn=source_m_tile, bm=256, cluster_m=2)
    plan["bk"] = block_k
    _validate_tcgen05_grouped_device_split_sizes(
        plan, torch.tensor([0, 64, 256, 448, 640], dtype=torch.int64)
    )
    expected = 4 * ((640 + source_m_tile - 1) // source_m_tile)
    assert _tcgen05_grouped_device_split_total_clusters(plan) == expected
