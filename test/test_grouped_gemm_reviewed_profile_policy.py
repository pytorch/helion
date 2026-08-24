from __future__ import annotations

from hashlib import sha256
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

from pretuned_kernels.grouped_gemm_deepgemm import (
    _helion_aot_grouped_gemm_deepgemm_cuda_sm100 as reviewed_heuristic,
)
from pretuned_kernels.grouped_gemm_deepgemm import (
    grouped_gemm_deepgemm as pretuned_deepgemm,
)
from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

from helion._compiler.autotuner_heuristics import cute as cute_heuristics
from helion._compiler.backend import CuteBackend
from helion._compiler.cute.grouped_worklist_policy import (
    get_grouped_worklist_target_policy,
)
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import L2GroupingSpec
from helion.autotuner.config_spec import LoopOrderSpec

# Intentionally ungated: this tests pure schema/config construction without
# CuTe runtime or CUDA; do not add skipUnlessBackends.


@pytest.fixture(scope="module")
def reviewed_config_spec() -> ConfigSpec:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 0),
        device=torch.device("cpu"),
        num_sm=148,
    )
    spec.cute_tcgen05_search_enabled = True
    for block_id, size_hint in enumerate((4096, 4096, 4096)):
        spec.block_sizes.append(
            BlockSizeSpec(
                block_id=block_id,
                size_hint=size_hint,
                max_size=256 if block_id < 2 else 128,
            )
        )
    spec.loop_orders.append(LoopOrderSpec([0, 1, 2]))
    spec.l2_groupings.append(L2GroupingSpec([0, 1, 2]))
    return spec


def _make_profile_key_inputs(
    groups: int,
    n: int,
    k: int,
    profile: reviewed_profiles.ReviewedWorklistProfile,
    actual_ms: tuple[int, ...],
    *,
    stored_ms: tuple[int, ...] | None = None,
    b_padding: int = 0,
    a_dtype: torch.dtype = torch.bfloat16,
    b_dtype: torch.dtype = torch.bfloat16,
    worklist_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if stored_ms is None:
        stored_ms = tuple(
            (actual_m + profile.source_m_tile - 1)
            // profile.source_m_tile
            * profile.source_m_tile
            for actual_m in actual_ms
        )
    packed_m = sum(stored_ms)
    a = torch.empty((packed_m, k), device="meta", dtype=a_dtype)
    if profile.b_major == "k":
        b = torch.empty(
            (groups, n, k + b_padding),
            device="meta",
            dtype=b_dtype,
        )[:, :, :k]
    else:
        b = torch.empty(
            (groups, k, n + b_padding),
            device="meta",
            dtype=b_dtype,
        )[:, :, :n].transpose(1, 2)
    rows = []
    start = 0
    for group, (actual_m, stored_m) in enumerate(
        zip(actual_ms, stored_ms, strict=True)
    ):
        rows.append((group, start, actual_m, stored_m))
        start += stored_m
    return a, b, torch.tensor(rows, dtype=worklist_dtype)


def test_reviewed_profile_selection_and_config_values() -> None:
    assert reviewed_profiles.SOURCE_M_TILES == (32, 224, 256)
    assert (
        tuple(
            reviewed_profiles.validate_source_m_tile(tile)
            for tile in reviewed_profiles.SOURCE_M_TILES
        )
        == reviewed_profiles.SOURCE_M_TILES
    )
    profile = reviewed_profiles.exact_reviewed_worklist_profile(6, 20, 4096, 4096)
    first = reviewed_profiles.reviewed_config_values(profile.config_name)
    second = reviewed_profiles.reviewed_config_values(profile.config_name)

    assert first == second
    assert first is not second
    assert first["block_sizes"] is not second["block_sizes"]
    assert first["loop_orders"] is not second["loop_orders"]
    assert first["block_sizes"] == [256, 128, 128]
    assert first["tcgen05_grouped_worklist_source_m_tile"] == 32
    assert first["tcgen05_grouped_runtime_direct"] is True
    profile = reviewed_profiles.exact_reviewed_worklist_profile(6, 20, 7168, 3072)
    config = reviewed_profiles.reviewed_config_values(profile.config_name)

    assert profile.config_name == "_SOURCE32_BK128_AB5_R256_RSV32_DIRECT_CONFIG"
    assert config["tcgen05_grouped_static_reserved_sms"] == 32


def test_pretuned_kernel_api_has_no_expected_m_hint() -> None:
    assert tuple(
        inspect.signature(pretuned_deepgemm._GROUPED_GEMM_DEEPGEMM_BODY).parameters
    ) == ("a_packed", "b_grouped", "worklist")
    assert tuple(
        inspect.signature(reviewed_heuristic.autotune_grouped_gemm_deepgemm).parameters
    ) == (
        "groups",
        "n",
        "k",
        "b_major",
        "a_layout",
        "b_layout",
        "worklist_layout",
        "source_m_tile",
        "packed_m",
        "normalized_worklist",
    )


def test_aot_key_distinguishes_b_layout_and_packed_extent() -> None:
    a = torch.empty(256, 64, dtype=torch.bfloat16)
    a_larger = torch.empty(384, 64, dtype=torch.bfloat16)
    b_k_major = torch.empty(2, 128, 64, dtype=torch.bfloat16)
    b_n_major = b_k_major.transpose(1, 2).contiguous().transpose(1, 2)
    b_strided = torch.empty(2, 128, 128, dtype=torch.bfloat16)[:, :, ::2]
    worklist = torch.tensor(
        ((0, 0, 100, 128), (1, 128, 100, 128)),
        dtype=torch.int32,
    )
    larger_worklist = torch.tensor(
        ((0, 0, 161, 192), (1, 192, 161, 192)),
        dtype=torch.int32,
    )

    keys = {
        pretuned_deepgemm._selected_key(a, b_k_major, worklist),
        pretuned_deepgemm._selected_key(a, b_n_major, worklist),
        pretuned_deepgemm._selected_key(a_larger, b_k_major, larger_worklist),
    }

    assert len(keys) == 3
    assert all(key[7] == reviewed_profiles.SMALL_M_ALIGNMENT for key in keys)
    with pytest.raises(ValueError, match="contiguous K-major or N-major"):
        pretuned_deepgemm._selected_key(a, b_strided, worklist)


def test_aot_key_selects_exact_official_profiles_without_expected_m() -> None:
    assert len(reviewed_profiles.REVIEWED_DISPATCH_PROFILES) == 8
    for shape, actual_ms in zip(
        reviewed_profiles.OFFICIAL_SHAPES,
        reviewed_profiles.official_actual_ms(seed=0),
        strict=True,
    ):
        profile = reviewed_profiles.exact_reviewed_worklist_profile(
            shape.groups,
            shape.expected_m_per_group,
            shape.n,
            shape.k,
        )
        a, b, worklist = _make_profile_key_inputs(
            shape.groups,
            shape.n,
            shape.k,
            profile,
            actual_ms,
        )

        key = pretuned_deepgemm._selected_key(a, b, worklist)

        assert key == (
            shape.groups,
            shape.n,
            shape.k,
            profile.b_major,
            reviewed_profiles.tensor_layout_signature(
                tuple(int(value) for value in a.shape),
                tuple(int(value) for value in a.stride()),
                int(a.storage_offset()),
                str(a.dtype),
            ),
            reviewed_profiles.tensor_layout_signature(
                tuple(int(value) for value in b.shape),
                tuple(int(value) for value in b.stride()),
                int(b.storage_offset()),
                str(b.dtype),
            ),
            reviewed_profiles.tensor_layout_signature(
                tuple(int(value) for value in worklist.shape),
                tuple(int(value) for value in worklist.stride()),
                int(worklist.storage_offset()),
                str(worklist.dtype),
            ),
            profile.source_m_tile,
            int(a.size(0)),
            reviewed_profiles.worklist_signature(
                tuple(tuple(int(value) for value in row) for row in worklist.tolist())
            ),
        )
        assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*key) == (
            reviewed_profiles.reviewed_config_values(profile.config_name)
        )


def test_aot_key_abstains_for_profiles_without_authoritative_inputs() -> None:
    official_shapes = {
        (shape.groups, shape.expected_m_per_group, shape.n, shape.k)
        for shape in reviewed_profiles.OFFICIAL_SHAPES
    }
    unmeasured_profiles = {
        shape: profile
        for shape, profile in reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES.items()
        if shape not in official_shapes
    }
    assert len(unmeasured_profiles) == 16
    for (groups, expected_m_per_group, n, k), profile in unmeasured_profiles.items():
        a, b, worklist = _make_profile_key_inputs(
            groups,
            n,
            k,
            profile,
            (expected_m_per_group,) * groups,
        )
        key = pretuned_deepgemm._selected_key(a, b, worklist)

        assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*key) is None


def test_aot_key_rejects_skewed_groups_with_same_packed_volume() -> None:
    shape = reviewed_profiles.OFFICIAL_SHAPES[0]
    actual_ms = reviewed_profiles.official_actual_ms(seed=0)[0]
    skewed_ms = (actual_ms[0] + 1, actual_ms[1] - 1, *actual_ms[2:])
    profile = reviewed_profiles.exact_reviewed_worklist_profile(
        shape.groups,
        shape.expected_m_per_group,
        shape.n,
        shape.k,
    )
    a, b, worklist = _make_profile_key_inputs(
        shape.groups,
        shape.n,
        shape.k,
        profile,
        actual_ms,
    )
    skewed_a, skewed_b, skewed_worklist = _make_profile_key_inputs(
        shape.groups,
        shape.n,
        shape.k,
        profile,
        skewed_ms,
    )

    key = pretuned_deepgemm._selected_key(a, b, worklist)
    skewed_key = pretuned_deepgemm._selected_key(
        skewed_a,
        skewed_b,
        skewed_worklist,
    )

    assert sum(actual_ms) == sum(skewed_ms)
    assert key[:-1] == skewed_key[:-1]
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*key) is not None
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*skewed_key) is None


def test_aot_key_rejects_nonminimal_worklist_padding() -> None:
    shape = reviewed_profiles.OFFICIAL_SHAPES[0]
    actual_ms = reviewed_profiles.official_actual_ms(seed=0)[0]
    profile = reviewed_profiles.exact_reviewed_worklist_profile(
        shape.groups,
        shape.expected_m_per_group,
        shape.n,
        shape.k,
    )
    stored_ms = tuple(
        (actual_m + profile.source_m_tile - 1)
        // profile.source_m_tile
        * profile.source_m_tile
        for actual_m in actual_ms
    )
    padded_stores = (stored_ms[0] + profile.source_m_tile, *stored_ms[1:])
    a, b, worklist = _make_profile_key_inputs(
        shape.groups,
        shape.n,
        shape.k,
        profile,
        actual_ms,
    )
    padded_a, padded_b, padded_worklist = _make_profile_key_inputs(
        shape.groups,
        shape.n,
        shape.k,
        profile,
        actual_ms,
        stored_ms=padded_stores,
    )

    key = pretuned_deepgemm._selected_key(a, b, worklist)
    padded_key = pretuned_deepgemm._selected_key(
        padded_a,
        padded_b,
        padded_worklist,
    )

    assert key[:4] == padded_key[:4]
    assert key[4] != padded_key[4]
    assert key[5:8] == padded_key[5:8]
    assert padded_key[8] == key[8] + profile.source_m_tile
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*key) is not None
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*padded_key) is None


@pytest.mark.parametrize("row_index", (0, 1))
@pytest.mark.parametrize("view_kind", ("padded", "offset"))
def test_aot_key_rejects_noncanonical_grouped_b_layout(
    row_index: int,
    view_kind: str,
) -> None:
    shape = reviewed_profiles.OFFICIAL_SHAPES[row_index]
    actual_ms = reviewed_profiles.official_actual_ms(seed=0)[row_index]
    profile = reviewed_profiles.exact_reviewed_worklist_profile(
        shape.groups,
        shape.expected_m_per_group,
        shape.n,
        shape.k,
    )
    a, b, worklist = _make_profile_key_inputs(
        shape.groups,
        shape.n,
        shape.k,
        profile,
        actual_ms,
    )
    if view_kind == "padded":
        candidate_a, candidate_b, candidate_worklist = _make_profile_key_inputs(
            shape.groups,
            shape.n,
            shape.k,
            profile,
            actual_ms,
            b_padding=1,
        )
    elif profile.b_major == "k":
        candidate_a = a
        candidate_b = torch.empty(
            (shape.groups + 1, shape.n, shape.k),
            device="meta",
            dtype=b.dtype,
        )[1:]
        candidate_worklist = worklist
    else:
        candidate_a = a
        candidate_b = torch.empty(
            (shape.groups + 1, shape.k, shape.n),
            device="meta",
            dtype=b.dtype,
        )[1:].transpose(1, 2)
        candidate_worklist = worklist

    key = pretuned_deepgemm._selected_key(a, b, worklist)
    candidate_key = pretuned_deepgemm._selected_key(
        candidate_a,
        candidate_b,
        candidate_worklist,
    )

    assert key[:4] == candidate_key[:4]
    assert key[4:7] != candidate_key[4:7]
    assert key[7:] == candidate_key[7:]
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*key) is not None
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*candidate_key) is None


@pytest.mark.parametrize(
    "view_kind",
    ("a_padded", "a_offset", "worklist_padded", "worklist_offset"),
)
def test_aot_key_rejects_noncanonical_a_and_worklist_layout(view_kind: str) -> None:
    shape = reviewed_profiles.OFFICIAL_SHAPES[0]
    actual_ms = reviewed_profiles.official_actual_ms(seed=0)[0]
    profile = reviewed_profiles.exact_reviewed_worklist_profile(
        shape.groups,
        shape.expected_m_per_group,
        shape.n,
        shape.k,
    )
    a, b, worklist = _make_profile_key_inputs(
        shape.groups,
        shape.n,
        shape.k,
        profile,
        actual_ms,
    )
    candidate_a = a
    candidate_worklist = worklist
    if view_kind == "a_padded":
        candidate_a = torch.empty(
            (int(a.size(0)), shape.k + 1),
            device="meta",
            dtype=a.dtype,
        )[:, : shape.k]
    elif view_kind == "a_offset":
        candidate_a = torch.empty(
            (int(a.size(0)) + 1, shape.k),
            device="meta",
            dtype=a.dtype,
        )[1:]
    elif view_kind == "worklist_padded":
        storage = torch.empty((shape.groups, 5), dtype=torch.int32)
        candidate_worklist = storage[:, :4]
        candidate_worklist.copy_(worklist)
    else:
        storage = torch.empty((shape.groups + 1, 4), dtype=torch.int32)
        candidate_worklist = storage[1:]
        candidate_worklist.copy_(worklist)

    key = pretuned_deepgemm._selected_key(a, b, worklist)
    candidate_key = pretuned_deepgemm._selected_key(
        candidate_a,
        b,
        candidate_worklist,
    )

    assert key[:4] == candidate_key[:4]
    assert key[4:7] != candidate_key[4:7]
    assert key[7:] == candidate_key[7:]
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*key) is not None
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*candidate_key) is None


@pytest.mark.parametrize("dtype_case", ("a_float16", "b_float32", "worklist_int64"))
def test_aot_key_rejects_unreviewed_input_dtypes(dtype_case: str) -> None:
    shape = reviewed_profiles.OFFICIAL_SHAPES[0]
    actual_ms = reviewed_profiles.official_actual_ms(seed=0)[0]
    profile = reviewed_profiles.exact_reviewed_worklist_profile(
        shape.groups,
        shape.expected_m_per_group,
        shape.n,
        shape.k,
    )
    a, b, worklist = _make_profile_key_inputs(
        shape.groups,
        shape.n,
        shape.k,
        profile,
        actual_ms,
    )
    if dtype_case == "a_float16":
        candidate = _make_profile_key_inputs(
            shape.groups,
            shape.n,
            shape.k,
            profile,
            actual_ms,
            a_dtype=torch.float16,
        )
    elif dtype_case == "b_float32":
        candidate = _make_profile_key_inputs(
            shape.groups,
            shape.n,
            shape.k,
            profile,
            actual_ms,
            b_dtype=torch.float32,
        )
    else:
        candidate = _make_profile_key_inputs(
            shape.groups,
            shape.n,
            shape.k,
            profile,
            actual_ms,
            worklist_dtype=torch.int64,
        )

    key = pretuned_deepgemm._selected_key(a, b, worklist)
    candidate_key = pretuned_deepgemm._selected_key(*candidate)

    assert key[:4] == candidate_key[:4]
    assert key[4:7] != candidate_key[4:7]
    assert key[7:] == candidate_key[7:]
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*key) is not None
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*candidate_key) is None


def test_aot_key_caches_worklist_analysis_until_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = torch.empty(448, 64, dtype=torch.bfloat16)
    b = torch.empty(2, 128, 64, dtype=torch.bfloat16)
    worklist = torch.tensor(
        ((0, 0, 1, 224), (1, 224, 1, 224)),
        dtype=torch.int32,
    )
    reads = 0
    read_worklist_rows = pretuned_deepgemm._worklist_rows

    def count_reads(
        tensor: torch.Tensor,
        mutation_key: tuple[object, ...],
    ) -> tuple[tuple[int, ...], ...]:
        nonlocal reads
        reads += 1
        return read_worklist_rows(tensor, mutation_key)

    pretuned_deepgemm._WORKLIST_FACTS_CACHE.clear()
    monkeypatch.setattr(pretuned_deepgemm, "_worklist_rows", count_reads)
    first = pretuned_deepgemm._selected_key(a, b, worklist)
    repeated = pretuned_deepgemm._selected_key(a, b, worklist)

    assert first == repeated
    assert first[7] == reviewed_profiles.LEGACY_M_ALIGNMENT
    assert reads == 1
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*first) is None

    worklist.copy_(
        torch.tensor(
            ((0, 0, 1, 32), (1, 32, 1, 416)),
            dtype=torch.int32,
        )
    )
    changed = pretuned_deepgemm._selected_key(a, b, worklist)

    assert changed[7] == reviewed_profiles.SMALL_M_ALIGNMENT
    assert changed != first
    assert reads == 2
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*changed) is None


def test_aot_key_tracks_inference_worklist_values() -> None:
    a = torch.empty(448, 64, dtype=torch.bfloat16)
    b = torch.empty(2, 128, 64, dtype=torch.bfloat16)
    with torch.inference_mode():
        worklist = torch.tensor(
            ((0, 0, 1, 224), (1, 224, 1, 224)),
            dtype=torch.int32,
        )
        first = pretuned_deepgemm._selected_key(a, b, worklist)
        worklist.copy_(
            torch.tensor(
                ((0, 0, 1, 32), (1, 32, 1, 416)),
                dtype=torch.int32,
            )
        )
        changed = pretuned_deepgemm._selected_key(a, b, worklist)

    assert first[7] == reviewed_profiles.LEGACY_M_ALIGNMENT
    assert changed[7] == reviewed_profiles.SMALL_M_ALIGNMENT
    assert changed != first


def test_aot_key_accepts_constant_backed_fake_worklist_values() -> None:
    concrete_worklist = torch.tensor(((0, 0, 1, 224),), dtype=torch.int32)
    mode = FakeTensorMode()
    a = mode.from_tensor(torch.empty(224, 64, dtype=torch.bfloat16))
    b = mode.from_tensor(torch.empty(1, 128, 64, dtype=torch.bfloat16))
    worklist = FakeTensor(
        mode,
        torch.empty_like(concrete_worklist, device="meta"),
        concrete_worklist.device,
        constant=concrete_worklist,
    )

    with mode:
        key = pretuned_deepgemm._selected_key(a, b, worklist)

    assert key == (
        1,
        128,
        64,
        "k",
        "torch.bfloat16|224,64|64,1|0",
        "torch.bfloat16|1,128,64|8192,64,1|0",
        "torch.int32|1,4|4,1|0",
        reviewed_profiles.LEGACY_M_ALIGNMENT,
        224,
        "0,0,1,224",
    )


def test_aot_key_uses_compiler_preference_for_ambiguous_worklist() -> None:
    a = torch.empty(1792, 64, dtype=torch.bfloat16)
    b = torch.empty(1, 128, 64, dtype=torch.bfloat16)
    worklist = torch.tensor(((0, 0, 1700, 1792),), dtype=torch.int32)

    key = pretuned_deepgemm._selected_key(a, b, worklist)

    assert key[7] == reviewed_profiles.LEGACY_M_ALIGNMENT


def test_aot_key_rejects_unknown_fake_worklist_values() -> None:
    with FakeTensorMode():
        a = torch.empty(224, 64, dtype=torch.bfloat16)
        b = torch.empty(1, 128, 64, dtype=torch.bfloat16)
        worklist = torch.empty(1, 4, dtype=torch.int32)

    with pytest.raises(ValueError, match="requires concrete packed-worklist values"):
        pretuned_deepgemm._selected_key(a, b, worklist)


def test_official_benchmark_manifest_is_exact_and_fixed() -> None:
    official_rows = []
    for shape, actual_ms in zip(
        reviewed_profiles.OFFICIAL_SHAPES,
        reviewed_profiles.official_actual_ms(seed=0),
        strict=True,
    ):
        profile = reviewed_profiles.exact_reviewed_worklist_profile(
            shape.groups,
            shape.expected_m_per_group,
            shape.n,
            shape.k,
        )
        official_rows.append(
            {
                "shape": tuple(shape),
                "actual_ms": actual_ms,
                "config_name": profile.config_name,
                "b_major": profile.b_major,
                "source_m_tile": profile.source_m_tile,
            }
        )
    manifest = {
        "official_rows": official_rows,
        "reviewed_configs": {
            name: reviewed_profiles.reviewed_config_values(name)
            for name in reviewed_profiles.REVIEWED_CONFIG_NAMES
        },
        "public_shape_profiles": [
            {
                "shape": shape,
                "config_name": profile.config_name,
                "b_major": profile.b_major,
                "source_m_tile": profile.source_m_tile,
            }
            for shape, profile in sorted(
                reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES.items()
            )
        ],
    }
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    expected_digest = "82d3cf3b33b5e755c3dacbfe2f4d53163cb6ffaa5e72354b8ebafebf6f5c10ae"
    assert expected_digest == reviewed_profiles.REVIEWED_PROFILE_MANIFEST_SHA256
    assert sha256(serialized.encode()).hexdigest() == expected_digest
    with pytest.raises(ValueError, match="no exact reviewed"):
        reviewed_profiles.exact_reviewed_worklist_profile(1, 1, 1, 1)


@pytest.mark.parametrize(
    ("source_m_tile", "expected_cluster_m"),
    (
        (reviewed_profiles.SMALL_M_ALIGNMENT, 1),
        (reviewed_profiles.LEGACY_M_ALIGNMENT, 2),
    ),
)
def test_worklist_config_spec_derives_cluster_m_and_runtime_direct_default(
    source_m_tile: int,
    expected_cluster_m: int,
) -> None:
    direct = reviewed_profiles.WorklistConfigSpec(
        7,
        source_m_tile=source_m_tile,
    )
    values = reviewed_profiles.worklist_config_values(direct)

    assert direct.cluster_m == expected_cluster_m
    assert direct.runtime_direct is True
    assert values["tcgen05_cluster_m"] == expected_cluster_m
    assert values["tcgen05_grouped_runtime_direct"] is True

    nondirect = reviewed_profiles.WorklistConfigSpec(
        7,
        source_m_tile=source_m_tile,
        runtime_direct=False,
    )
    assert "tcgen05_grouped_runtime_direct" not in (
        reviewed_profiles.worklist_config_values(nondirect)
    )


@pytest.mark.parametrize("config_name", reviewed_profiles.REVIEWED_CONFIG_NAMES)
def test_every_reviewed_config_normalizes(
    reviewed_config_spec: ConfigSpec,
    config_name: str,
) -> None:
    requested = reviewed_profiles.reviewed_config_values(config_name)
    normalized = reviewed_config_spec.normalized_config(requested)
    spec = reviewed_profiles.reviewed_config_spec(config_name)
    reviewed_from_spec = reviewed_profiles.worklist_config_values(spec)
    compiler_values = cute_heuristics.tcgen05_grouped_worklist_config_values(
        spec.source_m_tile,
        spec.block_k,
        spec.ab_stages,
        spec.consumer_regs,
        runtime_direct=spec.runtime_direct,
        l2_swizzle_size=spec.l2_swizzle_size,
        reserved_sms=spec.reserved_sms,
        clc=spec.clc,
    )

    assert requested == reviewed_profiles.reviewed_config_values(config_name)
    assert requested == reviewed_from_spec == compiler_values
    assert normalized.config["tcgen05_grouped_mode"] == "worklist_nm"
    source_m_tile = normalized.config["tcgen05_grouped_worklist_source_m_tile"]
    assert source_m_tile in (
        reviewed_profiles.SMALL_M_ALIGNMENT,
        reviewed_profiles.LEGACY_M_ALIGNMENT,
        reviewed_profiles.PROFILED_M_ALIGNMENT,
    )
    assert normalized.config["tcgen05_cluster_m"] == (
        1 if source_m_tile == reviewed_profiles.SMALL_M_ALIGNMENT else 2
    )


def test_reviewed_config_keys_match_compiler_seed_schema() -> None:
    reviewed_keys = frozenset(
        key
        for config_name in reviewed_profiles.REVIEWED_CONFIG_NAMES
        for key in reviewed_profiles.reviewed_config_values(config_name)
    )

    assert reviewed_keys == reviewed_profiles.WORKLIST_CONFIG_KEYS
    assert cute_heuristics._TCGEN05_GROUPED_WORKLIST_CONFIG_KEYS == (
        reviewed_profiles.WORKLIST_CONFIG_KEYS
    )


def test_worklist_config_schema_contracts_survive_optimized_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewed_spec = reviewed_profiles.WorklistConfigSpec(7)
    compiler_schema = cute_heuristics._TCGEN05_GROUPED_WORKLIST_CONFIG_KEYS
    reviewed_schema = reviewed_profiles.WORKLIST_CONFIG_KEYS
    monkeypatch.setattr(
        cute_heuristics,
        "_TCGEN05_GROUPED_WORKLIST_CONFIG_KEYS",
        compiler_schema - {"block_sizes"},
    )
    with pytest.raises(
        RuntimeError, match=r"missing from its schema: \['block_sizes'\]"
    ):
        cute_heuristics.tcgen05_grouped_worklist_config_values(224, 64, 7, 240)

    monkeypatch.setattr(
        reviewed_profiles,
        "WORKLIST_CONFIG_KEYS",
        reviewed_schema - {"block_sizes"},
    )
    with pytest.raises(
        RuntimeError, match=r"missing from its schema: \['block_sizes'\]"
    ):
        reviewed_profiles.worklist_config_values(reviewed_spec)


def test_reviewed_heuristic_contract_and_invalid_inputs() -> None:
    assert {
        profile.b_major
        for profile in reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES.values()
    } <= {"k", "n"}
    assert not hasattr(reviewed_heuristic, "key_grouped_gemm_deepgemm")
    assert reviewed_heuristic.SUPPORTED_HARDWARE_NAMES == ("NVIDIA B200",)
    dispatch_key, dispatch = next(
        iter(reviewed_profiles.REVIEWED_DISPATCH_PROFILES.items())
    )
    args = (*dispatch_key, dispatch.packed_m, dispatch.worklist_signature)
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*args) == (
        reviewed_profiles.reviewed_config_values(dispatch.config_name)
    )

    assert (
        reviewed_heuristic.autotune_grouped_gemm_deepgemm(
            1,
            1,
            1,
            b_major="k",
            a_layout="torch.bfloat16|224,1|1,1|0",
            b_layout="torch.bfloat16|1,1,1|1,1,1|0",
            worklist_layout="torch.int32|1,4|4,1|0",
            source_m_tile=reviewed_profiles.LEGACY_M_ALIGNMENT,
            packed_m=reviewed_profiles.LEGACY_M_ALIGNMENT,
            normalized_worklist="0,0,1,224",
        )
        is None
    )

    with pytest.raises(ValueError, match="must be 'k' or 'n'"):
        reviewed_heuristic.autotune_grouped_gemm_deepgemm(
            *dispatch_key[:3],
            "strided",
            *dispatch_key[4:],
            dispatch.packed_m,
            dispatch.worklist_signature,
        )
    with pytest.raises(ValueError, match="packed M must be a positive integer"):
        reviewed_heuristic.autotune_grouped_gemm_deepgemm(
            *dispatch_key,
            0,
            dispatch.worklist_signature,
        )
    with pytest.raises(ValueError, match=reviewed_profiles.SOURCE_M_TILE_ERROR):
        reviewed_heuristic.autotune_grouped_gemm_deepgemm(
            *dispatch_key[:7],
            64,
            dispatch.packed_m,
            dispatch.worklist_signature,
        )
    assert (
        reviewed_heuristic.autotune_grouped_gemm_deepgemm(
            *dispatch_key,
            dispatch.packed_m + 1,
            dispatch.worklist_signature,
        )
        is None
    )
    assert (
        reviewed_heuristic.autotune_grouped_gemm_deepgemm(
            *dispatch_key,
            dispatch.packed_m,
            f"{dispatch.worklist_signature};extra",
        )
        is None
    )


def test_b200_official_reviewed_profiles_remain_rank_zero() -> None:
    b200_identity = ("cuda", "NVIDIA B200", "sm100")
    for shape, actual_ms in zip(
        reviewed_profiles.OFFICIAL_SHAPES,
        reviewed_profiles.official_actual_ms(seed=0),
        strict=True,
    ):
        profile = reviewed_profiles.exact_reviewed_worklist_profile(
            shape.groups,
            shape.expected_m_per_group,
            shape.n,
            shape.k,
        )
        source_m_tile = profile.source_m_tile
        packed_m = sum(
            (actual_m + source_m_tile - 1) // source_m_tile * source_m_tile
            for actual_m in actual_ms
        )
        selected = cute_heuristics.tcgen05_grouped_worklist_seed_configs(
            groups=shape.groups,
            packed_m=packed_m,
            n=shape.n,
            k=shape.k,
            b_major=profile.b_major,
            source_m_tile=source_m_tile,
            num_sm=148,
            target_hardware_identity=b200_identity,
        )[0]

        assert selected.config == reviewed_profiles.reviewed_config_values(
            profile.config_name
        )


def test_gb300_policy_signatures_match_official_seed_zero_worklists() -> None:
    expected = set()
    for shape, actual_ms in zip(
        reviewed_profiles.OFFICIAL_SHAPES,
        reviewed_profiles.official_actual_ms(seed=0),
        strict=True,
    ):
        profile = reviewed_profiles.exact_reviewed_worklist_profile(
            shape.groups,
            shape.expected_m_per_group,
            shape.n,
            shape.k,
        )
        assert profile.source_m_tile == 256
        rows = []
        start = 0
        for group, actual_m in enumerate(actual_ms):
            stored_m = (actual_m + 255) // 256 * 256
            rows.append((group, start, actual_m, stored_m))
            start += stored_m
        expected.add(tuple(rows))

    policy = get_grouped_worklist_target_policy(("cuda", "NVIDIA GB300", "sm103"))
    assert policy.reviewed_worklist_rows() == frozenset(expected)


def test_reviewed_heuristic_loads_with_only_its_sibling_profile(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / Path(reviewed_heuristic.__file__).name
    profile = tmp_path / "reviewed_profiles.py"
    shutil.copy2(reviewed_heuristic.__file__, artifact)
    shutil.copy2(reviewed_profiles.__file__, profile)
    script = """
import importlib.util
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

path = pathlib.Path(sys.argv[1])
def load(name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

with ThreadPoolExecutor(max_workers=2) as pool:
    first, second = pool.map(load, (
        "isolated_grouped_gemm_aot_first",
        "isolated_grouped_gemm_aot_second",
    ))
dispatch_key, dispatch = next(iter(first._REVIEWED.REVIEWED_DISPATCH_PROFILES.items()))
print(json.dumps({
    "config": first.autotune_grouped_gemm_deepgemm(
        *dispatch_key, dispatch.packed_m, dispatch.worklist_signature
    ),
    "shared_profile_module": first._REVIEWED is second._REVIEWED,
    "shared_state_module": (
        first._PROFILE_STATE_MODULE_NAME == second._PROFILE_STATE_MODULE_NAME
        and first._PROFILE_STATE_MODULE_NAME in sys.modules
    ),
    "uses_sys_global_lock_registry": (
        "_helion_grouped_gemm_profile_locks" in vars(sys)
    ),
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["shared_profile_module"] is True
    assert payload["shared_state_module"] is True
    assert payload["uses_sys_global_lock_registry"] is False
    dispatch_key, dispatch = next(
        iter(reviewed_profiles.REVIEWED_DISPATCH_PROFILES.items())
    )
    assert payload["config"] == reviewed_heuristic.autotune_grouped_gemm_deepgemm(
        *dispatch_key,
        dispatch.packed_m,
        dispatch.worklist_signature,
    )


def _load_reviewed_heuristic(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, reviewed_heuristic.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aot_loader_reuses_canonical_and_rejects_foreign_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = _load_reviewed_heuristic("heuristic")
    assert canonical._REVIEWED is reviewed_profiles

    foreign = ModuleType("foreign_reviewed_profiles")
    foreign.__file__ = str(tmp_path / "reviewed_profiles.py")
    monkeypatch.setitem(
        sys.modules,
        "pretuned_kernels.grouped_gemm_deepgemm.reviewed_profiles",
        foreign,
    )
    module = _load_reviewed_heuristic(
        "pretuned_kernels.grouped_gemm_deepgemm.collision_heuristic",
    )
    assert module._REVIEWED is not foreign
    assert (
        Path(module._REVIEWED.__file__).resolve()
        == Path(reviewed_profiles.__file__).resolve()
    )


def test_reviewed_heuristic_reports_missing_sibling_profile(tmp_path: Path) -> None:
    artifact = tmp_path / Path(reviewed_heuristic.__file__).name
    shutil.copy2(reviewed_heuristic.__file__, artifact)
    result = subprocess.run(
        [sys.executable, "-I", str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires sibling reviewed_profiles.py" in result.stderr
