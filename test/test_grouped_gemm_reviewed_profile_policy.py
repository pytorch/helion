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


def test_reviewed_profile_selection_and_config_values() -> None:
    assert reviewed_profiles.SOURCE_M_TILES == (32, 224, 256)
    assert (
        tuple(
            reviewed_profiles.validate_source_m_tile(tile)
            for tile in reviewed_profiles.SOURCE_M_TILES
        )
        == reviewed_profiles.SOURCE_M_TILES
    )
    assert reviewed_profiles.reviewed_config_name(32, 1, 1, 32) == ("_SOURCE32_CONFIG")
    assert (
        reviewed_profiles.reviewed_config_name(32, 1, 1, 224)
        == (reviewed_profiles.AOT_CONFIG_NAMES[-1])
    )
    assert reviewed_profiles.reviewed_config_name(32, 1, 1, 256) == (
        "_SOURCE256_BK64_AB6_R240_DIRECT_CONFIG"
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
    assert reviewed_profiles.reviewed_config_name(6, 4096, 4096, 32) == (
        profile.config_name
    )
    assert (
        reviewed_profiles.reviewed_config_name(6, 4096, 4096, 224)
        == reviewed_profiles.AOT_CONFIG_NAMES[-1]
    )
    with pytest.raises(ValueError, match=reviewed_profiles.SOURCE_M_TILE_ERROR):
        reviewed_profiles.reviewed_config_name(6, 4096, 4096, 64)
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
    ) == ("groups", "n", "k", "b_major", "source_m_tile", "packed_m")


def test_aot_key_distinguishes_b_layout_and_packed_extent() -> None:
    a = torch.empty(256, 64)
    a_larger = torch.empty(384, 64)
    b_k_major = torch.empty(2, 128, 64)
    b_n_major = b_k_major.transpose(1, 2).contiguous().transpose(1, 2)
    b_strided = torch.empty(2, 128, 128)[:, :, ::2]
    worklist = torch.tensor(((0, 0, 100, 128), (1, 128, 100, 128)))
    larger_worklist = torch.tensor(((0, 0, 161, 192), (1, 192, 161, 192)))

    keys = {
        pretuned_deepgemm._selected_key(a, b_k_major, worklist),
        pretuned_deepgemm._selected_key(a, b_n_major, worklist),
        pretuned_deepgemm._selected_key(a_larger, b_k_major, larger_worklist),
    }

    assert len(keys) == 3
    assert all(key[-2] == reviewed_profiles.SMALL_M_ALIGNMENT for key in keys)
    with pytest.raises(ValueError, match="contiguous K-major or N-major"):
        pretuned_deepgemm._selected_key(a, b_strided, worklist)


def test_aot_key_selects_all_reviewed_profiles_without_expected_m() -> None:
    for shape, profile in reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES.items():
        groups, expected_m_per_group, n, k = shape
        actual_ms = (expected_m_per_group,) * groups
        source_m_tile = profile.source_m_tile
        packed_m = sum(
            (actual_m + source_m_tile - 1) // source_m_tile * source_m_tile
            for actual_m in actual_ms
        )
        a = torch.empty((packed_m, k), device="meta")
        if profile.b_major == "k":
            b = torch.empty((groups, n, k), device="meta")
        else:
            b = torch.empty(
                (groups, k, n),
                device="meta",
            ).transpose(1, 2)
        worklist_rows = []
        start = 0
        for group, actual_m in enumerate(actual_ms):
            stored_m = (actual_m + source_m_tile - 1) // source_m_tile * source_m_tile
            worklist_rows.append((group, start, actual_m, stored_m))
            start += stored_m
        worklist = torch.tensor(worklist_rows, dtype=torch.int32)

        key = pretuned_deepgemm._selected_key(a, b, worklist)

        assert key == (
            groups,
            n,
            k,
            profile.b_major,
            source_m_tile,
            packed_m,
        )
        assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*key) == (
            reviewed_profiles.reviewed_config_values(profile.config_name)
        )


def test_aot_key_caches_worklist_analysis_until_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = torch.empty(448, 64)
    b = torch.empty(2, 128, 64)
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

    pretuned_deepgemm._SOURCE_M_TILE_CACHE.clear()
    monkeypatch.setattr(pretuned_deepgemm, "_worklist_rows", count_reads)
    first = pretuned_deepgemm._selected_key(a, b, worklist)
    repeated = pretuned_deepgemm._selected_key(a, b, worklist)

    assert first == repeated
    assert first[-2] == reviewed_profiles.LEGACY_M_ALIGNMENT
    assert reads == 1
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*first) == (
        reviewed_profiles.reviewed_config_values(reviewed_profiles.AOT_CONFIG_NAMES[-1])
    )

    worklist.copy_(
        torch.tensor(
            ((0, 0, 1, 32), (1, 32, 1, 416)),
            dtype=torch.int32,
        )
    )
    changed = pretuned_deepgemm._selected_key(a, b, worklist)

    assert changed[-2] == reviewed_profiles.SMALL_M_ALIGNMENT
    assert changed != first
    assert reads == 2
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(*changed) == (
        reviewed_profiles.reviewed_config_values("_SOURCE32_CONFIG")
    )


def test_aot_key_tracks_inference_worklist_values() -> None:
    a = torch.empty(448, 64)
    b = torch.empty(2, 128, 64)
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

    assert first[-2] == reviewed_profiles.LEGACY_M_ALIGNMENT
    assert changed[-2] == reviewed_profiles.SMALL_M_ALIGNMENT
    assert changed != first


def test_aot_key_accepts_constant_backed_fake_worklist_values() -> None:
    concrete_worklist = torch.tensor(((0, 0, 1, 224),), dtype=torch.int32)
    mode = FakeTensorMode()
    a = mode.from_tensor(torch.empty(224, 64))
    b = mode.from_tensor(torch.empty(1, 128, 64))
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
        reviewed_profiles.LEGACY_M_ALIGNMENT,
        224,
    )


def test_aot_key_uses_compiler_preference_for_ambiguous_worklist() -> None:
    a = torch.empty(1792, 64)
    b = torch.empty(1, 128, 64)
    worklist = torch.tensor(((0, 0, 1700, 1792),), dtype=torch.int32)

    key = pretuned_deepgemm._selected_key(a, b, worklist)

    assert key[-2] == reviewed_profiles.LEGACY_M_ALIGNMENT


def test_aot_key_rejects_unknown_fake_worklist_values() -> None:
    with FakeTensorMode():
        a = torch.empty(224, 64)
        b = torch.empty(1, 128, 64)
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
    assert cute_heuristics.tcgen05_grouped_worklist_config_keys() == (
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


def test_reviewed_heuristic_selects_every_profile_and_fallback() -> None:
    assert {
        profile.b_major
        for profile in reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES.values()
    } <= {"k", "n"}
    assert len(reviewed_profiles.REVIEWED_DISPATCH_PROFILES) == 24
    for shape, profile in reviewed_profiles.REVIEWED_PUBLIC_SHAPE_PROFILES.items():
        groups, _expected_m_per_group, n, k = shape
        for b_major in ("k", "n"):
            assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(
                groups,
                n,
                k,
                b_major=b_major,
                source_m_tile=profile.source_m_tile,
                packed_m=groups * profile.source_m_tile,
            ) == (reviewed_profiles.reviewed_config_values(profile.config_name))

    assert not hasattr(reviewed_heuristic, "key_grouped_gemm_deepgemm")
    assert reviewed_heuristic.SUPPORTED_HARDWARE_NAMES == ("NVIDIA B200",)
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(
        6,
        4096,
        4096,
        b_major="k",
        source_m_tile=256,
        packed_m=1536,
    ) == reviewed_heuristic.autotune_grouped_gemm_deepgemm(
        6,
        4096,
        4096,
        b_major="n",
        source_m_tile=256,
        packed_m=3072,
    )
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(
        6, 4096, 4096, source_m_tile=32
    ) != reviewed_heuristic.autotune_grouped_gemm_deepgemm(
        6, 4096, 4096, source_m_tile=256
    )
    assert reviewed_heuristic.autotune_grouped_gemm_deepgemm(1, 1, 1) == (
        reviewed_profiles.reviewed_config_values(reviewed_profiles.AOT_CONFIG_NAMES[-1])
    )
    with pytest.raises(ValueError, match="must be 'k' or 'n'"):
        reviewed_heuristic.autotune_grouped_gemm_deepgemm(
            6,
            4096,
            4096,
            b_major="strided",
        )
    with pytest.raises(ValueError, match="packed M must be a positive integer"):
        reviewed_heuristic.autotune_grouped_gemm_deepgemm(6, 4096, 4096, packed_m=0)


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
print(json.dumps({
    "config": first.autotune_grouped_gemm_deepgemm(
        6, 4096, 4096, source_m_tile=32
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
    assert payload["config"] == reviewed_heuristic.autotune_grouped_gemm_deepgemm(
        6, 4096, 4096, source_m_tile=32
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
