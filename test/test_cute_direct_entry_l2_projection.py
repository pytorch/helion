from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion.autotuner.block_id_sequence import BlockIdSequence
from helion.autotuner.config_spec import L2GroupingSpec

if TYPE_CHECKING:
    from helion.autotuner.config_spec import ConfigSpec


def _metadata(
    monkeypatch: pytest.MonkeyPatch, root_ids: tuple[int, ...], matmul_root: int
) -> CuteTcgen05Config:
    sequence: BlockIdSequence[L2GroupingSpec] = BlockIdSequence()
    for root in root_ids:
        sequence.append(L2GroupingSpec([root, root + 1]))
    metadata = CuteTcgen05Config.__new__(CuteTcgen05Config)
    metadata.config_spec = cast(
        "ConfigSpec",
        SimpleNamespace(l2_groupings=sequence, indexing=SimpleNamespace(length=6)),
    )
    metadata.matmul_block_ids = (matmul_root, matmul_root + 1, matmul_root + 2)
    monkeypatch.setattr(metadata, "full_tile_direct_entry_seed_eligible", lambda: True)
    monkeypatch.setattr(metadata, "full_tile_direct_entry_seed_bk", lambda: 128)
    monkeypatch.setattr(
        metadata, "_matmul_seed_block_sizes", lambda **kwargs: [256, 256, 128]
    )
    return metadata


@pytest.mark.parametrize(
    "roots,matmul_root,values,expected",
    [
        ((11,), 11, [8], [2]),
        ((3, 11), 11, [8, 16], [8, 2]),
        ((3, 11, 29), 11, [8, 16, 32], [8, 2, 32]),
        ((3, 11, 29), 29, [8, 16, 32], [8, 16, 2]),
    ],
)
def test_direct_entry_projection_owns_only_its_mapped_root(
    monkeypatch: pytest.MonkeyPatch,
    roots: tuple[int, ...],
    matmul_root: int,
    values: list[object],
    expected: list[int],
) -> None:
    metadata = _metadata(monkeypatch, roots, matmul_root)
    original = list(values)
    seed = metadata.full_tile_direct_entry_seed_config(l2_groupings=values)
    assert seed is not None and seed.l2_groupings == expected
    assert values == original and seed.l2_groupings is not values
    config: dict[str, object] = {
        "l2_groupings": values,
        "tcgen05_tvm_ffi_launch": True,
        "unrelated_coordinate": 7,
    }
    metadata._fix_target1_tvm_ffi_search_config(config)
    assert config["l2_groupings"] == expected
    assert config["unrelated_coordinate"] == 7
    assert values == original
    first = deepcopy(config)
    metadata._fix_target1_tvm_ffi_search_config(config)
    assert config == first
    default = metadata.full_tile_direct_entry_seed_config()
    assert default is not None
    expected_default = [1] * len(roots)
    expected_default[roots.index(matmul_root)] = 2
    assert default.l2_groupings == expected_default


def test_one_root_seed_retains_complete_original_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(monkeypatch, (11,), 11)
    seed = metadata.full_tile_direct_entry_seed_config()
    assert seed is not None
    assert seed.config == {
        "block_sizes": [256, 256, 128],
        "l2_groupings": [2],
        "num_warps": 8,
        "num_stages": 4,
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 3,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_l2_swizzle_size": 1,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_persistence_model": "static_persistent",
        "tcgen05_layout_strategy": "explicit_epi_tile",
        "tcgen05_layout_overrides_epi_tile_m": 128,
        "tcgen05_layout_overrides_epi_tile_n": 32,
        "tcgen05_layout_overrides_d_store_box_n": 32,
        "tcgen05_flat_role_coordinates": True,
        "tcgen05_tvm_ffi_launch": True,
    }
    monkeypatch.setattr(metadata, "full_tile_direct_entry_seed_eligible", lambda: False)
    values: list[object] = [8]
    assert metadata.full_tile_direct_entry_seed_config(l2_groupings=values) is None
    assert values == [8]
