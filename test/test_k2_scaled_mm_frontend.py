"""CPU checks for K2 benchmark body selection and archived-config conversion."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _source_module(relative_path: str) -> ModuleType:
    # The baseline Python also has HKL's benchmarks package installed. Load the
    # requested worktree files directly so that package cannot shadow this one.
    path = Path(__file__).resolve().parents[1] / relative_path
    name = f"_k2_frontend_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _source_module("benchmarks/cute/k2_scaled_mm_hillclimb.py")
heuristic = _source_module(
    "pretuned_kernels/scaled_mm_cute/_helion_aot_scaled_mm_cute_cuda_sm103.py"
)


@pytest.mark.parametrize("shape", worker.SHAPES)
def test_cold_body_selection(shape: tuple[int, int, int]) -> None:
    implementation, config, migration = worker._kernel_config_for_run(shape, "cold")
    assert implementation == (
        "frontend_transposed" if shape == (256, 4096, 2048) else "hkl_mn"
    )
    assert config is None and migration is None


@pytest.mark.parametrize("index", range(4))
def test_current_physical_configs_select_frontend(index: int) -> None:
    key = heuristic._KEYS_scaled_mm_transposed_cute[index]
    original = heuristic._CONFIGS_scaled_mm_transposed_cute[index]
    before = copy.deepcopy(original)
    implementation, config, migration = worker._kernel_config_for_run(
        key[:3], "fixed", original
    )
    assert implementation == "frontend_transposed"
    assert config == original == before
    assert migration == {"original_config": original, "changes": {}}


@pytest.mark.parametrize("index", range(4))
def test_archived_nm_conversion_preserves_source_and_scale_mode(index: int) -> None:
    key = heuristic._KEYS_scaled_mm_transposed_cute[index]
    physical = heuristic._CONFIGS_scaled_mm_transposed_cute[index]
    archived = {
        **copy.deepcopy(physical),
        "block_sizes": [64, 128, 256],
        "tcgen05_orientation": "nm",
        "tcgen05_cta_group": "auto",
        "indexing": ["pointer", "tensor_descriptor", "pointer"],
    }
    payload = {"config": archived, "status": "PASS", "shape": key[:3]}
    before = json.dumps(payload, sort_keys=True)
    implementation, config, migration = worker._kernel_config_for_run(
        key[:3], "fixed", payload
    )
    assert implementation == "frontend_transposed"
    assert config == physical
    assert json.dumps(payload, sort_keys=True) == before
    assert migration is not None
    assert migration["original_config"] == archived
    assert set(migration["changes"]) == {
        "block_sizes",
        "tcgen05_cta_group",
        "tcgen05_orientation",
        "indexing",
    }
    # Later normalization of the returned config cannot mutate the source or
    # the original-config provenance retained for the new result file.
    assert config is not None
    config["block_sizes"][0] = 256
    assert json.dumps(payload, sort_keys=True) == before
    assert migration["original_config"]["block_sizes"] == [64, 128, 256]


@pytest.mark.parametrize("shape", [worker.SHAPES[i] for i in (0, 2, 3)])
def test_archived_nm_wrong_shape_is_rejected(shape: tuple[int, int, int]) -> None:
    archived = {"block_sizes": [64, 128, 256], "tcgen05_orientation": "nm"}
    with pytest.raises(ValueError, match="Archived NM configs require"):
        worker._kernel_config_for_run(shape, "fixed", archived)
    assert archived == {"block_sizes": [64, 128, 256], "tcgen05_orientation": "nm"}


@pytest.mark.parametrize("orientation", [None, "mn"])
def test_legacy_m256_mn_configs_retain_vendored_body(orientation: str | None) -> None:
    archived = {"block_sizes": [64, 32, 256], "tcgen05_cta_group": "auto"}
    if orientation is not None:
        archived["tcgen05_orientation"] = orientation
    implementation, config, _migration = worker._kernel_config_for_run(
        (256, 4096, 2048), "fixed", archived
    )
    assert implementation == "hkl_mn"
    assert config == {"block_sizes": [64, 32, 256], "tcgen05_cta_group": "auto"}


def test_explicit_mn_takes_priority_over_physical_tile_pattern() -> None:
    archived = {
        **copy.deepcopy(heuristic._CONFIGS_scaled_mm_transposed_cute[0]),
        "tcgen05_orientation": "mn",
    }
    implementation, config, _migration = worker._kernel_config_for_run(
        (256, 4096, 2048), "fixed", archived
    )
    assert implementation == "hkl_mn"
    assert config == heuristic._CONFIGS_scaled_mm_transposed_cute[0]


def test_result_metadata_identifies_future_frontend_tile() -> None:
    physical = {
        **copy.deepcopy(heuristic._CONFIGS_scaled_mm_transposed_cute[0]),
        "block_sizes": [128, 128, 128],
    }
    implementation, config, _migration = worker._kernel_config_for_run(
        (256, 4096, 2048),
        "fixed",
        {"config": physical, "kernel_implementation": "frontend_transposed"},
    )
    assert implementation == "frontend_transposed"
    assert config == physical


def test_explicit_legacy_metadata_preserves_m256_body() -> None:
    physical_pattern = copy.deepcopy(heuristic._CONFIGS_scaled_mm_transposed_cute[0])
    implementation, config, _migration = worker._kernel_config_for_run(
        (256, 4096, 2048),
        "fixed",
        {"config": physical_pattern, "kernel_implementation": "hkl_mn"},
    )
    assert implementation == "hkl_mn"
    assert config == physical_pattern
