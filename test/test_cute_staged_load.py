"""CPU tests: ``cute_staged_load`` reads a CTA's loads of one tensor from a
shared-memory image its threads fill with packets, declining wherever the
generated kernel is not proved to read a box of rows of that tensor; the
autotuner offers the knob on a transposing kernel's structure."""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics.cute_transposed_store_seeds import (
    CuteTransposedStoreSeedsHeuristic,
)
from helion._compiler.cute.staged_loads import _thread_index_subscripted_only
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def copy3d(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile(out.size()):
        out[tile_z, tile_y, tile_x] = x[tile_z, tile_y, tile_x]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def scaled_copy3d(x: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile(out.size()):
        out[tile_z, tile_y, tile_x] = x[tile_z, tile_y, tile_x] * 2.0 + b[tile_x]


# A store-side config of a [z, 64, 64] tile: x vectors of 8 bf16 per thread
# (16-byte stores), 4 y values per thread, 16 x 8 threads.
_STORE_SIDE = {
    "block_sizes": [1, 64, 64],
    "num_threads": [1, 16, 8],
    "cute_vector_widths": [1, 1, 8],
}


def _transposed(
    z: int = 2, y: int = 128, x: int = 192, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` of stride 1 along y (a transposed view), ``out`` contiguous."""
    return (
        torch.empty(z, x, y, dtype=dtype).transpose(1, 2),
        torch.empty(z, y, x, dtype=dtype),
    )


def _bound(kernel: Any, args: tuple[object, ...]) -> BoundKernel[Any]:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        return cpu_bind(kernel, args)


def _code(kernel: Any, args: tuple[object, ...], config: dict[str, object]) -> str:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, args)
        return bound.to_code(
            bound.config_spec.normalized_config(helion.Config.from_dict(config))
        )


def test_a_transposing_copy_reads_its_footprint_through_a_swizzled_image() -> None:
    code = _code(copy3d, _transposed(), {**_STORE_SIDE, "cute_staged_load": True})
    # 64 rows of x by 64 contiguous y: the image of a CTA's footprint, filled
    # by 4 rounds of 16-byte packets of its 128 threads, read after one
    # barrier.
    assert "cute.arch.alloc_smem(cutlass.BFloat16, 4096, alignment=16)" in code
    assert code.count("cute.arch.sync_threads()") == 1
    assert code.count("cute.arch.load(x.iterator + (") == 4
    # The load reads the image, its 16-byte chunk swizzled by the row's bits
    # 3-5 (the shift under which the warp's reads take the fewest wavefronts,
    # the larger of two that tie; the fill's 4 rounds swizzle the same way);
    # no load reads x directly.
    loads = re.findall(r"load = (.*)\.load\(\)", code)
    assert len(loads) == 1
    assert loads[0].startswith("(_cta_image + (")
    # The swizzle of a thread's loads is its x coordinate's low bits (rows
    # 8 apart: the row's bits 3-5), computed once per thread.
    assert "_cta_image_swizzle = _cta_thread_1 >> 0 & 7" in code
    assert "^ _cta_image_swizzle" in loads[0]
    assert code.count(">> 3 & 7") == 4
    assert code.count("x.iterator") == 4
    # The threads relabeled so that each warp's 16-byte stores run along x
    # (8 threads on a row), and the CTAs ordered x-block first (3 of them).
    assert "_cta_thread_1 = _cta_image_thread // 1 % 8" in code
    assert "_cta_order_digit_2 = _cta_order_block % 3" in code


def test_the_staged_load_keeps_the_rest_of_the_body() -> None:
    # Another load (b, contiguous along x) stays direct; only x is staged.
    b = torch.empty(192, dtype=torch.bfloat16)
    x, out = _transposed()
    code = _code(scaled_copy3d, (x, b, out), {**_STORE_SIDE, "cute_staged_load": True})
    assert "cute.arch.alloc_smem(cutlass.BFloat16, 4096, alignment=16)" in code
    assert "(b.iterator + " in code
    assert code.count("cute.arch.load(x.iterator + (") == 4


def test_fp32_stages_32_wide_rows() -> None:
    x, out = _transposed(dtype=torch.float32)
    config = {
        "block_sizes": [1, 32, 32],
        "num_threads": [1, 8, 8],
        "cute_vector_widths": [1, 1, 4],
        "cute_staged_load": True,
    }
    code = _code(copy3d, (x, out), config)
    # 32 rows of 32 fp32: 4 KB of image in 16-byte chunks of 4 elements.
    assert "cute.arch.alloc_smem(cutlass.Float32, 1024, alignment=16)" in code
    assert code.count("cute.arch.sync_threads()") == 1


@pytest.mark.parametrize(
    "case",
    # A partial tile (masked loads); the knob off.
    ["partial", "off"],
)
def test_declines_leave_the_loads_direct(case: str) -> None:
    if case == "partial":
        args, config = _transposed(y=100), {**_STORE_SIDE, "cute_staged_load": True}
    else:
        args, config = _transposed(), dict(_STORE_SIDE)
    code = _code(copy3d, args, config)
    assert "_cta_image" not in code
    assert "cute.arch.sync_threads()" not in code


def test_the_knob_is_offered_on_transposing_loads_only() -> None:
    assert _bound(copy3d, _transposed()).config_spec.cute_transposed_load_sites
    contiguous = (
        torch.empty(2, 128, 192, dtype=torch.bfloat16),
        torch.empty(2, 128, 192, dtype=torch.bfloat16),
    )
    assert not _bound(copy3d, contiguous).config_spec.cute_transposed_load_sites


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def copy2d(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile(out.size()):
        out[tile_y, tile_x] = x[tile_y, tile_x]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def copy4d(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile_n, tile_h, tile_w, tile_c in hl.tile(out.size()):
        out[tile_n, tile_h, tile_w, tile_c] = x[tile_n, tile_h, tile_w, tile_c]


@pytest.mark.parametrize(
    "case",
    # A rank-2 transpose in int32; NCHW to channels-last (rank 4, a box of
    # 64 channels by 64 positions of W per CTA; its four grid digits keep
    # the CTA order).
    ["rank2_int32", "nchw_to_nhwc"],
)
def test_other_ranks_and_dtypes_stage(case: str) -> None:
    if case == "rank2_int32":
        kernel = copy2d
        args = (
            torch.empty(256, 256, dtype=torch.int32).t(),
            torch.empty(256, 256, dtype=torch.int32),
        )
        config = {
            "block_sizes": [32, 32],
            "num_threads": [8, 8],
            "cute_vector_widths": [1, 4],
        }
    else:
        kernel = copy4d
        args = (
            torch.empty(4, 128, 16, 64, dtype=torch.bfloat16).permute(0, 2, 3, 1),
            torch.empty(4, 16, 64, 128, dtype=torch.bfloat16),
        )
        config = {
            "block_sizes": [1, 1, 64, 64],
            "num_threads": [1, 1, 16, 8],
            "cute_vector_widths": [1, 1, 1, 8],
        }
    code = _code(kernel, args, {**config, "cute_staged_load": True})
    assert code.count("cute.arch.sync_threads()") == 1
    assert "_cta_image_swizzle" in code
    assert ("_cta_order_index" in code) == (case == "rank2_int32")


def test_one_element_per_thread_reads_relabeled_coordinates_it_defines() -> None:
    # A thread per element on both tiles: the grid strategy names each
    # thread's indices at the top of the body, ahead of the load loop; the
    # relabeled coordinates open the body, before those statements.
    config = {
        "block_sizes": [1, 32, 32],
        "num_threads": [1, 32, 32],
        "cute_staged_load": True,
    }
    code = _code(copy3d, _transposed(), config)
    assert "_cta_image" in code
    body = code[code.index("def _helion_copy3d") :]
    first_use = body.index("_cta_thread_", body.index("indices_"))
    for definition in re.findall(r"(_cta_thread_\d+) = ", body):
        assert body.index(f"{definition} = ") < first_use


def test_the_transposed_store_seeds_plant_a_staged_load() -> None:
    x, out = _transposed(z=4, y=512, x=768)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(copy3d, (x, out))
        with bound.env:
            seeds = CuteTransposedStoreSeedsHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        staged = [seed for seed in seeds if seed.config.get("cute_staged_load")]
        assert [dict(seed.config) for seed in staged] == [
            {
                "block_sizes": [1, 64, 64],
                "num_threads": [1, 16, 8],
                "cute_vector_widths": [1, 1, 8],
                "cute_staged_load": True,
            }
        ]
        code = bound.to_code(bound.config_spec.normalized_config(staged[0]))
    assert "_cta_image" in code


@pytest.mark.parametrize(
    ("body", "relabels"),
    [
        ("i = cutlass.Int32(cute.arch.thread_idx()[1]) * 8", True),
        # A name as the axis, a lane index, the index unpacked whole: reads
        # of the thread's identity the relabeling would not rewrite.
        ("axis = 0\nsaved = cute.arch.thread_idx()[axis]", False),
        ("lane = cute.arch.lane_idx()", False),
        ("tx, ty, tz = cute.arch.thread_idx()", False),
    ],
)
def test_threads_relabel_only_where_every_read_is_rewritten(
    body: str, relabels: bool
) -> None:
    assert _thread_index_subscripted_only(ast.parse(body).body) is relabels
