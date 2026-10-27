"""Host ``torch.zeros`` outputs the kernel overwrites completely become ``torch.empty``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import skipUnlessBackends
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

pytestmark = skipUnlessBackends("cute")


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable():
        yield


def _host_code(
    fn: Callable[..., object],
    *args: object,
    backend: str = "cute",
    overrides: dict[str, object] | None = None,
) -> str:
    kernel = helion.kernel(
        fn,
        backend=backend,
        static_shapes=True,
        autotune_effort="full",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    bound = _cpu_bind(kernel, args)
    config = bound.config_spec.default_config()
    if overrides:
        config = helion.Config(**{**config.config, **overrides})
    code = bound.to_code(config)
    # Drop the ``# src[...]`` provenance comments, which quote the original
    # ``torch.zeros`` source line.
    return "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith("#")
    )


def _row_sums(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    return out


def _two_outputs_one_conditional(
    x: torch.Tensor, flag: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full = torch.zeros([x.size(0), x.size(1)], dtype=torch.float32, device=x.device)
    partial = torch.zeros([x.size(0), x.size(1)], dtype=torch.float32, device=x.device)
    scratch = x.new_zeros([x.size(0), x.size(1)], dtype=torch.float32)
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_m, tile_n].to(torch.float32)
        full[tile_m, tile_n] = value * 2
        scratch[tile_m, tile_n] = value
        if flag != 0.0:
            partial[tile_m, tile_n] = value
    return full, partial, scratch


def _accumulating_store(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] += x[tile, :].to(torch.float32).sum(-1)
    return out


def _host_read_before_launch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    other = torch.empty_like(out)
    alias = out.view(-1)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
        other[tile] = x[tile, :].to(torch.float32).amax(-1)
    return alias, other


def _metadata_read_before_launch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    other = torch.empty_like(out)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
        other[tile] = x[tile, :].to(torch.float32).amax(-1)
    return out, other


def _partial_dimension(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0) + 1], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    return out


def _prefix_rows_with_host_fill(x: torch.Tensor) -> torch.Tensor:
    # grpo_loss backward: the loop covers all but the last row, which the
    # host zeroes after the kernel.
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0) - 1):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    out[-1, :] = 0
    return out


def _prefix_rows_without_host_fill(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0) - 1):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


def _prefix_rows_read_before_fill(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0) - 1):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    total = out.sum()
    out[-1, :] = 0
    return out, total


def _inner_loop_full_range(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


def _inner_loop_partial_range(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(1, x.size(1)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


def _inner_loop_conditional(x: torch.Tensor, flag: float) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0)):
        if flag != 0.0:
            for tile_n in hl.tile(x.size(1)):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


def _masked_store(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        value = x[tile, :].to(torch.float32).sum(-1)
        hl.store(out, [tile], value, extra_mask=value > 0)
    return out


def _split_k_workspace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """examples/split_k_barrier: ``tmp``'s last dim is indexed by the K tile's id."""
    m, k = a.shape
    _, n = b.shape
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(16, 512, 64))
    block_k = helion.next_power_of_2(helion.cdiv(k, split_k))
    tmp = torch.zeros((m, n, split_k), device=a.device, dtype=a.dtype)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n, tile_k_outer in hl.tile(
        [m, n, k], block_size=[None, None, block_k]
    ):
        acc = hl.zeros([tile_m, tile_n], device=a.device, dtype=a.dtype)
        for tile_k_inner in hl.tile(tile_k_outer.begin, tile_k_outer.end):
            acc = torch.addmm(acc, a[tile_m, tile_k_inner], b[tile_k_inner, tile_n])
        tmp[tile_m, tile_n, tile_k_outer.id] = acc
    hl.barrier()
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.sum(tmp[tile_m, tile_n, :], dim=-1)
    return out


def _store_then_load_in_one_root(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    other = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
        other[tile] = out[tile] * 2.0
    return out, other


def _split_k_workspace_read_first(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(16, 512, 64))
    block_k = helion.next_power_of_2(helion.cdiv(k, split_k))
    tmp = torch.zeros((m, n, split_k), device=a.device, dtype=a.dtype)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.sum(tmp[tile_m, tile_n, :], dim=-1)
    hl.barrier()
    for tile_m, tile_n, tile_k_outer in hl.tile(
        [m, n, k], block_size=[None, None, block_k]
    ):
        acc = hl.zeros([tile_m, tile_n], device=a.device, dtype=a.dtype)
        for tile_k_inner in hl.tile(tile_k_outer.begin, tile_k_outer.end):
            acc = torch.addmm(acc, a[tile_m, tile_k_inner], b[tile_k_inner, tile_n])
        tmp[tile_m, tile_n, tile_k_outer.id] = acc
    return out


def _scalar_tile_end_store(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=torch.int32)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m.end, tile_n.end] = 1
    return out


def _scalar_tile_begin_store(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=torch.int32)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m.begin, tile_n.begin] = 1
    return out


def _root_begin_one(x: torch.Tensor) -> torch.Tensor:
    # The root tile indexes [1, n); the trailing host fill matches the
    # remainder pattern, so without the begin check the fill would be dropped
    # and out[0] left uninitialized.
    out = torch.zeros([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(1, x.size(0)):
        out[tile] = x[tile] * 2
    out[-1] = 0.0
    return out


def _split_k_workspace_begin(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Tile ids of a root whose K range starts at 128 begin at 128 // block_k.
    m, k = a.shape
    _, n = b.shape
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(16, 512, 64))
    block_k = helion.next_power_of_2(helion.cdiv(k, split_k))
    tmp = torch.zeros((m, n, split_k), device=a.device, dtype=a.dtype)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n, tile_k_outer in hl.tile(
        [0, 0, 128], [m, n, k], block_size=[None, None, block_k]
    ):
        acc = hl.zeros([tile_m, tile_n], device=a.device, dtype=a.dtype)
        for tile_k_inner in hl.tile(tile_k_outer.begin, tile_k_outer.end):
            acc = torch.addmm(acc, a[tile_m, tile_k_inner], b[tile_k_inner, tile_n])
        tmp[tile_m, tile_n, tile_k_outer.id] = acc
    hl.barrier()
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.sum(tmp[tile_m, tile_n, :], dim=-1)
    return out


def _shared_block_two_extents(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # One registered block size drives two root loops of different extents;
    # the block then has no static size.
    block = hl.register_block_size(x.size(0))
    rows = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    cols = torch.zeros([x.size(1)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0), block_size=block):
        rows[tile] = x[tile, :].to(torch.float32).sum(-1)
    for tile in hl.tile(x.size(1), block_size=block):
        cols[tile] = x[:, tile].to(torch.float32).sum(0)
    return rows, cols


def _zeros_into_out(x: torch.Tensor) -> torch.Tensor:
    buffer = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device, out=buffer)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    return buffer


def test_full_root_store_drops_the_fill() -> None:
    code = _host_code(_row_sums, torch.empty((64, 32), dtype=torch.bfloat16))
    assert "out = torch.empty([x.size(0)]" in code or "out = torch.empty([64]" in code
    assert "torch.zeros(" not in code


def test_conditional_store_and_new_zeros_pair() -> None:
    code = _host_code(
        _two_outputs_one_conditional, torch.empty((64, 32), dtype=torch.bfloat16), 1.0
    )
    assert "full = torch.empty(" in code
    assert "partial = torch.zeros(" in code
    assert "scratch = x.new_empty(" in code


@pytest.mark.parametrize(
    "fn",
    [
        _accumulating_store,
        _host_read_before_launch,
        _partial_dimension,
        _masked_store,
    ],
)
def test_unproven_fills_are_kept(fn: Callable[..., object]) -> None:
    code = _host_code(fn, torch.empty((64, 32), dtype=torch.bfloat16))
    assert "out = torch.zeros(" in code
    assert "out = torch.empty(" not in code


def test_full_range_inner_loop_drops_the_fill() -> None:
    code = _host_code(
        _inner_loop_full_range, torch.empty((64, 32), dtype=torch.bfloat16)
    )
    assert "out = torch.empty_like(x)" in code


def test_prefix_coverage_completed_by_a_host_fill() -> None:
    code = _host_code(
        _prefix_rows_with_host_fill, torch.empty((65, 32), dtype=torch.bfloat16)
    )
    assert "out = torch.empty_like(x)" in code
    # The remainder fill itself moves into the kernel (host_remainder_fills):
    # the host statement is gone and the kernel writes the last row.
    assert "out[-1, :] = 0" not in code
    assert (
        "_cute_remainder_fill(out, dim=0, remainder=1, value=0, is_float=True)" in code
    )


@pytest.mark.parametrize(
    "fn",
    [
        _prefix_rows_without_host_fill,
        _prefix_rows_read_before_fill,
        _inner_loop_partial_range,
    ],
)
def test_incomplete_prefix_or_partial_loop_keeps_the_fill(
    fn: Callable[..., object],
) -> None:
    code = _host_code(fn, torch.empty((65, 32), dtype=torch.bfloat16))
    assert "out = torch.zeros_like(x)" in code


def test_conditional_inner_loop_keeps_the_fill() -> None:
    code = _host_code(
        _inner_loop_conditional, torch.empty((64, 32), dtype=torch.bfloat16), 1.0
    )
    assert "out = torch.zeros_like(x)" in code


def test_metadata_reads_do_not_keep_the_fill() -> None:
    code = _host_code(
        _metadata_read_before_launch, torch.empty((64, 32), dtype=torch.bfloat16)
    )
    assert "out = torch.empty(" in code
    assert "other = torch.empty_like(out)" in code


def test_triton_host_code_is_unchanged() -> None:
    code = _host_code(
        _row_sums, torch.empty((64, 32), dtype=torch.bfloat16), backend="triton"
    )
    assert "out = torch.zeros(" in code


def test_tile_id_workspace_behind_a_barrier_drops_the_fill() -> None:
    # k = 1024 with the default split_k = 64: block_k = 16, 64 K tiles cover
    # every slot of the workspace before the barrier-ordered reads.
    a = torch.empty((128, 1024), dtype=torch.float32)
    b = torch.empty((1024, 256), dtype=torch.float32)
    code = _host_code(_split_k_workspace, a, b)
    assert "tmp = torch.empty((" in code
    assert "tmp = torch.zeros(" not in code


def test_tile_id_workspace_with_unwritten_slots_keeps_the_fill() -> None:
    # k = 100: block_k = next_power_of_2(cdiv(100, 64)) = 2 gives 50 K tiles,
    # so slots [50, 64) stay zero and the second phase sums them.
    a = torch.empty((128, 100), dtype=torch.float32)
    b = torch.empty((100, 256), dtype=torch.float32)
    code = _host_code(_split_k_workspace, a, b)
    assert "tmp = torch.zeros((" in code


def test_tile_id_workspace_read_before_the_stores_keeps_the_fill() -> None:
    a = torch.empty((128, 1024), dtype=torch.float32)
    b = torch.empty((1024, 256), dtype=torch.float32)
    code = _host_code(_split_k_workspace_read_first, a, b)
    assert "tmp = torch.zeros((" in code


def test_load_in_the_storing_phase_keeps_the_fill() -> None:
    # Only a grid-wide barrier orders every store before a load; a load in the
    # same phase (here the same root) keeps the fill.
    code = _host_code(
        _store_then_load_in_one_root, torch.empty((64, 32), dtype=torch.bfloat16)
    )
    assert "out = torch.zeros(" in code


def test_tile_id_workspace_triton_host_code_is_unchanged() -> None:
    a = torch.empty((128, 1024), dtype=torch.float32)
    b = torch.empty((1024, 256), dtype=torch.float32)
    code = _host_code(_split_k_workspace, a, b, backend="triton")
    assert "tmp = torch.zeros((" in code


def test_tile_id_workspace_in_ordered_phase_launches_drops_the_fill() -> None:
    # ``pid_type="flat"`` launches the two barrier phases as separate stages
    # (materialized fission); each stage sees one root, so the wrapper's shared
    # allocation is proven against the whole kernel.
    a = torch.empty((128, 1024), dtype=torch.float32)
    b = torch.empty((1024, 256), dtype=torch.float32)
    code = _host_code(_split_k_workspace, a, b, overrides={"pid_type": "flat"})
    assert "_helion_cute_region_0_module" in code
    assert "tmp = torch.empty((" in code
    assert "tmp = torch.zeros(" not in code


@pytest.mark.parametrize("fn", [_scalar_tile_end_store, _scalar_tile_begin_store])
def test_scalar_tile_begin_end_stores_keep_the_fill(fn: Callable[..., object]) -> None:
    # ``tile.end`` / ``tile.begin`` resolve to the tile's block id through
    # get_block_id but write one element each (test_misc's test_config_flatten_issue).
    code = _host_code(fn, torch.empty((64, 64), dtype=torch.float32))
    assert "out = torch.zeros_like(x, dtype=torch.int32)" in code
    assert "empty_like" not in code


def test_root_tile_with_a_nonzero_begin_keeps_the_fill() -> None:
    code = _host_code(_root_begin_one, torch.empty((64,), dtype=torch.float32))
    assert "out = torch.zeros([x.size(0)]" in code or "out = torch.zeros([64]" in code
    assert "out = torch.empty(" not in code


def test_tile_id_workspace_with_a_nonzero_begin_keeps_the_fill() -> None:
    a = torch.empty((128, 1024), dtype=torch.float32)
    b = torch.empty((1024, 256), dtype=torch.float32)
    code = _host_code(_split_k_workspace_begin, a, b)
    assert "tmp = torch.zeros((" in code


def test_block_without_a_static_size_keeps_the_fill() -> None:
    code = _host_code(
        _shared_block_two_extents, torch.empty((64, 32), dtype=torch.bfloat16)
    )
    assert "rows = torch.zeros(" in code
    assert "cols = torch.zeros(" in code
    assert "torch.empty([" not in code.replace("torch.empty([x.size(0)]", "")


def test_zeros_with_an_out_keyword_keeps_the_fill() -> None:
    code = _host_code(_zeros_into_out, torch.empty((64, 32), dtype=torch.bfloat16))
    assert "out = torch.zeros([x.size(0)]" in code or "out = torch.zeros([64]" in code
    assert "out=buffer" in code
    assert "out = torch.empty(" not in code
