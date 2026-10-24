"""The ``warp_mma`` matmul family on the grouped row union (CPU codegen).

``examples/grouped_gemm.py``'s jagged kernel (``out[rows of group g] =
A_packed[rows] @ B``, ``static_shapes=False``) is one dense GEMM over the
union of the group rows.  The grouped heuristic admits the register-MMA family
on the first binding's extents; codegen re-proves the row union, specializes
the extents under guards and renders one CTA per tile with membership-
predicated stores.  Runtime numerics live in ``test_cute_warp_mma_gemm_gpu.py``.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_coverage_search import _bound
from test.test_cute_shared_rhs_grouped import _target
from test.test_cute_warp_mma_gemm import _repaired

import helion
from helion._compiler.cute import cute_warp_mma_gemm as family
from helion._compiler.cute.grouped_row_union import CONFIG_KEY as ROW_UNION_KEY
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])

FAMILY = family.WARP_MMA_FAMILY_KEY
WARPS = family.WARP_MMA_WARPS_KEY
RT = "_helion_warp_mma"


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        _target(),
        # The admission needs the device's SM count (the fake target has none).
        patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=148),
    ):
        yield
    torch.set_num_threads(old_threads)


def _config(block_sizes: list[int], warps: int, **overrides: object) -> helion.Config:
    values: dict[str, object] = {
        "block_sizes": block_sizes,
        FAMILY: family.MATMUL_FAMILY_WARP_MMA,
        WARPS: warps,
    }
    values.update(overrides)
    return helion.Config(**values)


def _body(bound: BoundKernel, config: helion.Config) -> str:
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    return source[
        source.index("@cute.kernel") : source.index("_helion_cute_wrapper_plans")
    ]


@pytest.fixture(scope="module")
def s0() -> BoundKernel:
    # 640 x 128 x 128 over four groups of 160 rows: the catalogue's shape 0.
    bound, _args = _bound(0)
    return bound


def test_grouped_row_union_admits_and_seeds_the_family(s0: BoundKernel) -> None:
    with s0.env:
        config = s0.config_spec._cute_tcgen05_config
        assert config.grouped_row_union_supported
        assert config.warp_mma_admitted
        assert config.warp_mma_grouped_extents == (640, 128, 128)
        seeds = config._warp_mma_seed_configs()
        assert {
            (tuple(seed.config["block_sizes"]), seed.config[WARPS]) for seed in seeds
        } >= {((32, 32, 128), 4), ((64, 16, 128), 4), ((16, 32, 128), 2)}
        for seed in seeds:
            assert seed.config[FAMILY] == family.MATMUL_FAMILY_WARP_MMA
        # The floors reach the family's atoms while the default (non-autotuned)
        # block keeps the tcgen05 tile the kernel always ran.
        m_spec, n_spec = s0.config_spec.block_sizes[0], s0.config_spec.block_sizes[1]
        assert (m_spec.min_size, n_spec.min_size) == (16, 8)
        assert (m_spec.default_size, n_spec.default_size) == (128, 16)
        flat = s0.config_spec.flat_config(lambda x: x.default()).config
        assert flat["block_sizes"][:2] == [128, 16]
        assert flat[FAMILY] == family.MATMUL_FAMILY_TCGEN05
        # A tcgen05-family request below the tcgen05 floors is clamped back.
        clamped = s0.config_spec.normalized_config(
            helion.Config(block_sizes=[16, 8, 64])
        )
        assert clamped.config["block_sizes"][:2] == [128, 8]


def test_grouped_row_union_renders_predicated_register_mma_tiles(
    s0: BoundKernel,
) -> None:
    with s0.env:
        normalized = s0.config_spec.normalized_config(
            _config([32, 32, 64], 4, **{ROW_UNION_KEY: True})
        )
    # The family lowers the union itself: the row-union carrier's coverage
    # coordinate is inert and dropped from the identity.
    assert ROW_UNION_KEY not in normalized.config
    assert normalized.config["pid_type"] == "flat"
    source = s0.to_code(normalized)
    body = source[
        source.index("@cute.kernel") : source.index("_helion_cute_wrapper_plans")
    ]
    plan = re.search(r"_helion_cute_wrapper_plans = (\[.*?\])\n", source).group(1)
    assert "'kind': 'helion_warp_mma_gemm'" in plan
    assert "'total_tiles': 80" in plan and "'threads': 128" in plan
    assert "'offsets_name': 'group_offsets'" in plan
    # Five Int32 offsets (plain loads, after any dependent-launch wait), four
    # clamped intervals, a membership flag for each of the lane's two output
    # rows; A row-major, B N-contiguous (.trans).
    assert body.count("group_offsets.iterator + cutlass.Int64(") == 5
    assert "ldg_s32" not in body and "wm_offsets" not in body
    assert body.count("wm_start_") >= 4 and "cutlass.Int32(640)" in body
    _assert_membership_chains_are_binary(source, groups=4)
    assert body.count(f"{RT}.mma_bf16(") == 16
    assert body.count("if wm_sel_0_0:") == 2 and body.count("if wm_sel_0_1:") == 2
    assert body.count(f"{RT}.stg_b32(") == 4
    assert "ldmatrix_x4_trans(" in body and "tcgen05" not in body
    # One CTA per tile of the union on the flat grid (the launcher takes the
    # plan's ``total_tiles`` over Helion's (group, m, n) grid tuple); no group
    # loop, no persistent scheduler.
    assert "StaticPersistentTileScheduler" not in source
    assert "_launcher(_helion_grouped_gemm_jagged, (" in source
    assert "A_packed, B, out, group_offsets, block=(128, 1, 1))" in source


def _assert_membership_chains_are_binary(source: str, *, groups: int) -> None:
    """Every ``or`` / ``and`` of the rendered module is binary: the DSL
    preprocessor copies the accumulated left operand of a flat chain four
    times per term (4^G nodes for a G-term chain), so the membership test of
    each lane row is emitted right-nested."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BoolOp):
            assert len(node.values) == 2, ast.unparse(node)
    (selected,) = [
        line for line in source.splitlines() if line.strip().startswith("wm_sel_0_0 =")
    ]
    assert selected.count(" or (") == groups - 1
    assert selected.count("wm_start_") == groups


def _bound_groups(groups: int, rows_per_group: int = 32) -> BoundKernel:
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        cute_segmented_matmul_tiling=True,
    )
    return kernel.bind(
        (
            torch.empty((rows_per_group * groups, 128), dtype=torch.bfloat16),
            torch.empty((128, 128), dtype=torch.bfloat16),
            torch.arange(groups + 1, dtype=torch.int32) * rows_per_group,
        )
    )


def test_group_count_is_capped_at_one_warp_of_groups() -> None:
    # 32 groups (1024 rows): admitted; the body reads 33 offsets and tests
    # each lane row against 32 right-nested intervals.  33 groups: the
    # heuristic withholds the family and a pinned family config fails closed
    # (the body's prologue grows with the group count; the flat chain used to
    # take the DSL preprocessor 46 s at ten groups and never finished at
    # twelve).
    capped = _bound_groups(family.WARP_MMA_GROUPED_MAX_GROUPS)
    with capped.env:
        assert capped.config_spec._cute_tcgen05_config.warp_mma_admitted
        normalized = capped.config_spec.normalized_config(_config([32, 32, 64], 4))
    source = capped.to_code(normalized)
    assert source.count("group_offsets.iterator + cutlass.Int64(") == 33
    _assert_membership_chains_are_binary(source, groups=32)

    over = _bound_groups(family.WARP_MMA_GROUPED_MAX_GROUPS + 1)
    with over.env:
        state = over.config_spec._cute_tcgen05_config
        # The row-union carrier's own domain stops one short of the cap too;
        # ``detect_warp_mma_grouped_gemm`` replaces that domain with the
        # family's, which keeps the cap.
        assert not state.grouped_row_union_supported
        assert not state.warp_mma_admitted and state._warp_mma_seed_configs() == []
        with pytest.raises(InvalidConfig, match="not admitted"):
            over.config_spec.normalized_config(_config([32, 32, 64], 4))
    assert _repaired(over, _config([32, 32, 64], 4))[FAMILY] == "tcgen05"


def test_grouped_tiles_must_divide_the_union(s0: BoundKernel) -> None:
    # 640 rows: 128-row tiles are outside the family; repair halves them.
    with s0.env, pytest.raises(InvalidConfig, match="block_m 128"):
        s0.config_spec.normalized_config(_config([128, 32, 64], 4))
    repaired = _repaired(s0, _config([128, 32, 64], 4))
    assert repaired["block_sizes"][0] == 64
    assert repaired[FAMILY] == family.MATMUL_FAMILY_WARP_MMA
