from __future__ import annotations

import ast
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from test.test_cute_persistent_load_subpackets import fixture
from test.test_cute_persistent_load_subpackets import transform

from helion._compiler.cute import memory_ops as m
from helion._compiler.reduction_strategy import LoopedReductionStrategy
from helion._compiler.reduction_strategy import PersistentReductionStrategy


def context(
    mask,
    *,
    extra="",
    shape=(16, 128),
    stride=(128, 1),
    aligned=True,
    gathered=False,
    launch_coordinates=False,
    dtype=None,
    rolled=False,
    subpackets=True,
    alignment_requests=None,
):
    outer = []
    grid = NS(
        lane_setup_statements=ast.parse(
            "indices_0 = row_base\nindices_1 = base + cutlass.Int32(synthetic_lane_1)"
        ).body,
        lane_loops=[("synthetic_lane_1", 8)],
        vec_lane_wrappers={},
        hoist_parent_statements=outer,
    )
    body = ast.parse(
        "token = indices_0 % 2048\nvalid = operator.lt(token, 2048)\nrow = token * 32 + indices_0 // 2048 % 32\n"
        + extra
    ).body
    if launch_coordinates:
        grid.lane_setup_statements = grid.lane_setup_statements[1:]
        body = (
            ast.parse(
                "pid_flat = cutlass.Int32(cute.arch.block_idx()[0])\n"
                "offsets_0 = pid_flat * _BLOCK_SIZE_0 + cutlass.Int32(cute.arch.thread_idx()[1])\n"
                "indices_0 = offsets_0\n"
            ).body
            + body
        )
    state = NS(
        codegen=NS(
            current_grid_state=grid,
            statements_stack=[outer, body],
            active_device_loops={},
        ),
        device_function=NS(
            cute_state=NS(emitting_tile_loop=False),
            block_size_var_cache={(0,): "_BLOCK_SIZE_0"},
        ),
        fx_node=NS(users={}, args=(None, [None, slice(None)])),
    )
    strategy = object.__new__(
        LoopedReductionStrategy if rolled else PersistentReductionStrategy
    )
    strategy.block_ids = [1]
    strategy.index_var = lambda block: "indices_1"
    strategy._synthetic_cute_lane_var = "synthetic_lane_1"
    strategy._cute_lane_base_index_var = "base"
    strategy._cute_lane_body = []
    strategy._cute_reduction_vec_width = 8
    strategy._cute_reduction_vec_mode = "unroll"
    strategy._cute_reduction_lane_extent = 8
    strategy._mask_var = None
    tensor = torch.empty_strided(
        shape, stride, dtype=torch.bfloat16 if dtype is None else dtype
    )
    env = NS(
        backend=NS(name="cute"),
        block_sizes={1: NS(reduction=True)},
        runtime_value_for_tensor=lambda t: t,
        resolve_block_id=lambda n: 0 if n == 16 else 1,
    )
    with (
        patch.object(m.CompileEnvironment, "current", return_value=env),
        patch.object(
            m,
            "_cute_lane_strategy",
            side_effect=lambda state, bid: strategy if bid == 1 else NS(),
        ),
        patch.object(m, "_matching_block_ids", return_value=[1]),
        patch.object(m, "is_cute_direct_iota_index", return_value=False),
        patch.object(
            m,
            "runtime_tensor_has_specialized_alignment",
            side_effect=lambda env, tensor, alignment: (
                (
                    alignment_requests.append(alignment)
                    if alignment_requests is not None
                    else None
                )
                or aligned
            ),
        ),
        patch.object(m, "cute_reduction_vector_layout_aligned", return_value=True),
    ):
        return m._cute_vector_load_ctx(
            state,
            tensor,
            [torch.arange(16) if gathered else 0, slice(None)],
            ["row" if gathered else "indices_0", "indices_1"],
            ast.parse(mask, mode="eval").body if mask is not None else None,
            allow_gathered_index=True,
            allow_persistent_subpackets=subpackets,
        )


@pytest.mark.parametrize(
    "mask", ["valid", "valid and token >= 0", "True", "False", None]
)
def test_uniform_masks_keep_fragment_admission(mask):
    assert context(mask) == (8, 1, "unroll")


@pytest.mark.parametrize(
    "mask,extra",
    [
        ("indices_1 < 127", ""),
        ("alias < 127", "alias = indices_1\n"),
        ("valid", "indices_0 = indices_1\n"),
        ("indices_1 % 2 == 0", ""),
        ("unknown", ""),
        ("flag", "flag = (flags.iterator + token).load()\n"),
        ("flag", "flag = mystery(token)\n"),
        ("alias", "alias = other\nother = alias\n"),
    ],
)
def test_lane_dependencies_and_unknown_masks_remain_scalar(mask, extra):
    assert context(mask, extra=extra) is None


@pytest.mark.parametrize(
    "extra,expected",
    [
        ("", True),
        ("row = cutlass.Int32(cute.arch.block_idx()[2])\n", True),
        ("row = cutlass.Int32(cute.arch.thread_idx()[0])\n", True),
        ("_BLOCK_SIZE_0 = 16\n", True),
        ("_BLOCK_SIZE_0 = 16\n_BLOCK_SIZE_0 = 16\n", True),
        ("row = cute.arch.block_idx()[3]\n", False),
        ("row = cute.arch.block_idx()[-1]\n", False),
        ("row = cute.arch.block_idx()[indices_1]\n", False),
        ("row = cute.arch.block_idx(1)[0]\n", False),
        ("row = cute.arch.clock()[0]\n", False),
        ("cute = other\n", False),
        ("_BLOCK_SIZE_0 = indices_1\n", False),
        ("row = _BLOCK_SIZE_UNKNOWN\n", False),
        ("_BLOCK_SIZE_0 = 16\n_BLOCK_SIZE_0 = 32\n", False),
        ("_BLOCK_SIZE_0 += 16\n", False),
        ("if indices_1: \n    _BLOCK_SIZE_0 = 16\n", False),
        ("_BLOCK_SIZE_0 = indices_1\n_BLOCK_SIZE_0 = 16\n", False),
        ("row = (index.iterator + indices_0).load()\n", False),
        ("row = indices_1\n", False),
    ],
)
def test_gathered_rows_use_only_established_grid_coordinates(extra, expected):
    value = context("valid", gathered=True, launch_coordinates=True, extra=extra)
    assert (value == (8, 1, "unroll")) is expected


@pytest.mark.parametrize(
    "options",
    [
        {"shape": (16, 126), "stride": (126, 1)},
        {"stride": (129, 1)},
        {"aligned": False},
        {"rolled": True},
    ],
)
def test_grid_mask_keeps_extent_alignment_and_rolled_gates(options):
    assert context("valid", **options) is None


def test_fp32_fragment_checks_physical_packet_alignment():
    requests = []
    assert context("valid", dtype=torch.float32, alignment_requests=requests) == (
        8,
        1,
        "unroll",
    )
    assert requests and set(requests) == {16}
    assert context("valid", dtype=torch.float32, subpackets=False) is None


@pytest.mark.parametrize(
    "expression,vectorized",
    [
        ("cute.arch.thread_idx()", True),
        ("cute.arch.block_idx()", True),
        ("cute.arch.lane_idx()", True),
        ("cute.arch.thread_idx(1)", False),
        ("cute.arch.thread_idx(dim=1)", False),
        ("cute.arch.sync_threads()", False),
        ("cute.arch.membar_cta()", False),
        ("unknown()", False),
    ],
)
def test_subpacket_motion_accepts_only_read_only_coordinate_queries(
    expression, vectorized
):
    code = ast.unparse(
        ast.Module(
            body=transform(fixture(suffix="coordinate = " + expression)),
            type_ignores=[],
        )
    )
    assert (code.count("cute.arch.load(") == 2) is vectorized
    assert expression in code
