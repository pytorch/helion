from __future__ import annotations

import ast
import hashlib
from itertools import product
import json

import pytest

from helion._compiler.cute import chained_tcgen_stage as stages
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan

_CASES = tuple(
    product(
        (False, True),
        ("independent", "arena", "preallocated"),
        ("row_major", "xor"),
        (128, 512),
    )
)


def _allocation(grouped: bool, storage: str, mode: str, threads: int) -> list[str]:
    geometries = (
        stages.StageGeometry((16, 16, 16), False),
        stages.StageGeometry((16, 32, 16), False),
        stages.StageGeometry((32, 128, 64), True),
    )
    groups = (
        (
            ContractionGroup((0, 1), geometries[:2]),
            ContractionGroup((2,), geometries[2:]),
        )
        if grouped
        else None
    )
    workspace = None
    if storage != "independent":
        workspace = SharedMemoryLayoutPlan(
            (
                SharedBufferRegion("chain_0_c", 0, 1024, 0, 1, 128),
                SharedBufferRegion("chain_1_c", 1024, 2048, 0, 1, 128),
                SharedBufferRegion("chain_2_c", 3072, 16384, 0, 1, 128),
            ),
            19456,
        )
    return stages.allocate_stages(
        geometries,
        groups,
        workspace,
        ScratchLayouts(mode),
        threads,
        workspace_allocated=storage == "preallocated",
    )


def _allocation_digest() -> str:
    return hashlib.sha256(
        json.dumps([(case, _allocation(*case)) for case in _CASES]).encode()
    ).hexdigest()


def test_all_legacy_allocation_variants_match_frozen_emitted_bytes() -> None:
    # Captured before extracting allocate_tmem_resources, over all 24 combinations
    # of grouped/single stages, independent/shared/preallocated C, layouts and CTAs.
    assert (
        _allocation_digest()
        == "bedf63cff6da22dfe0c53aed010d7463dcaf6b2568b73cdb41913421247a57b8"
    )


@pytest.mark.parametrize("grouped,storage,mode,threads", _CASES)
def test_stage_allocation_retains_completion_ids_and_workspace_ownership(
    grouped: bool, storage: str, mode: str, threads: int
) -> None:
    source = "\n".join(_allocation(grouped, storage, mode, threads))
    assert f"chain_allocator.allocate({64 if grouped else 32})" in source
    assert "chain_bars = cute.arch.alloc_smem(cutlass.Int64, 3, alignment=16)" in source
    assert source.count("cute.arch.mbarrier_init(chain_bars +") == 3
    assert "cute.arch.mbarrier_init(chain_bars + 2, 1)" in source
    assert f"NamedBarrier(barrier_id=1, num_threads={threads})" in source
    assert source.count("chain_c_workspace =") == (storage == "arena")
    assert source.count("cute.make_tensor(") == 3
    if storage != "independent":
        assert "chain_c_workspace + 0" in source
        assert "chain_c_workspace + 256" in source
        assert "chain_c_workspace + 768" in source
    assert source.endswith("cute.arch.sync_threads()")


@pytest.mark.parametrize(
    "columns,allocation",
    [
        (1, 32),
        (16, 32),
        (32, 32),
        (33, 64),
        (64, 64),
        (65, 128),
        (128, 128),
        (160, 256),
        (256, 256),
        (257, 512),
        (512, 512),
    ],
)
def test_resource_columns_round_to_legal_allocator_sizes(
    columns: int, allocation: int
) -> None:
    lines = stages.allocate_tmem_resources(columns, 1, 128)
    assert f"chain_allocator.allocate({allocation})" in lines
    assert lines[-1] == "chain_tptr = chain_allocator.retrieve_ptr(cutlass.Float32)"
    assert not any("workspace" in line for line in lines)


@pytest.mark.parametrize("threads", [128, 160, 384, 512, 1024])
def test_caller_allocations_precede_cta_wide_protocol_without_role_ownership(
    threads: int,
) -> None:
    shared = (
        "recurrence_a = cute.arch.alloc_smem(cutlass.BFloat16, 8192, alignment=128)",
        "recurrence_b = cute.arch.alloc_smem(cutlass.BFloat16, 2048, alignment=128)",
        "frames = cute.arch.alloc_smem(cutlass.Uint8, 99840, alignment=128)",
        "carry_and_c = cute.arch.alloc_smem(cutlass.Float32, 32768, alignment=128)",
    )
    lines = stages.allocate_tmem_resources(160, 15, threads, shared)
    assert lines[:5] == [
        "from cutlass.cute.nvgpu import tcgen05",
        "from cutlass.utils import blackwell_helpers as chain_sm100",
        "from cutlass.utils import TmemAllocator",
        "import cutlass.pipeline as chain_pipeline",
        "chain_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
    ]
    assert tuple(lines[5:9]) == shared
    assert (
        lines[9] == "chain_bars = cute.arch.alloc_smem(cutlass.Int64, 15, alignment=16)"
    )
    source = "\n".join(lines)
    assert source.count("mbarrier_init(chain_bars +") == 15
    assert source.index("mbarrier_init(chain_bars + 14, 1)") < source.index(
        "mbarrier_init_fence()"
    )
    assert f"NamedBarrier(barrier_id=1, num_threads={threads})" in source
    ordered = (
        "chain_allocator.allocate(256)",
        "chain_allocator.relinquish_alloc_permit()",
        "chain_allocator.wait_for_alloc()",
        "chain_tptr = chain_allocator.retrieve_ptr(cutlass.Float32)",
    )
    assert [lines.index(line) for line in ordered] == sorted(
        lines.index(line) for line in ordered
    )
    tree = ast.parse(source)
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
    assert len(branches) == 1
    assert ast.unparse(branches[0].test) == "chain_thread == 0"
    assert all("mbarrier_init(" in ast.unparse(node) for node in branches[0].body)
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    assert stages.free_stages() == [
        "cute.arch.sync_threads()",
        "chain_allocator.free(chain_tptr)",
    ]


@pytest.mark.parametrize("columns", [0, -1, True, False, 16.0, 513, 1024])
def test_invalid_column_requests_fail_closed(columns) -> None:
    with pytest.raises(ValueError, match="columns"):
        stages.allocate_tmem_resources(columns, 1, 128)


@pytest.mark.parametrize("count", [0, -1, True, False, 1.0])
def test_invalid_completion_barrier_count_fails_closed(count) -> None:
    with pytest.raises(ValueError, match="barrier count"):
        stages.allocate_tmem_resources(32, count, 128)


@pytest.mark.parametrize("threads", [0, 32, 96, 127, 129, 1025, 2048, True, 128.0])
def test_invalid_cta_participant_count_fails_closed(threads) -> None:
    with pytest.raises(ValueError, match="CTA participants"):
        stages.allocate_tmem_resources(32, 1, threads)
