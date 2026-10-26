"""An atomic indexed by ``hl.arange(k)`` adds each of its k entries once per tile.

The index is the same k values on every thread of the tile, so the atomic is
part of the tile program: one thread issues the k adds.  It was lowered
through the iota matcher's per-thread coordinate instead: ``hl.arange(4)``
under a 32-row tile became ``indices_0 // 8`` on every row thread, so each
entry received 8 adds per full tile (8.0 instead of 2.0 over two row tiles,
the literal ``block_size=8`` twin included) and the partial second tile wrote
past the 4-entry buffer; ``hl.arange(3)`` rendered a ``cute.make_identity_tensor``
that failed at launch.  The loop form ``for _tensor_index_i in range(k)``
existed for host-tensor values only and rejected a scalar value and a list
index; every bare arange index takes it now, on the leader thread of every
launch axis, the exited and the argument-sized ones included (the ghost
clauses).  An index computed from an arange (arithmetic on it, a second
component beside it) declines instead of taking the per-thread form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])


def _count_entries_after(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """Static rows, argument-sized columns; the four entries counted once per row tile after the column loop."""
    out = torch.empty_like(x)
    count = torch.zeros([4], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [hl.arange(4)], 1.0)
    return count


def _count_entries_after_literal(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    count = torch.zeros([4], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=8):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [hl.arange(4)], 1.0)
    return count


def _count_entries_inside(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """The four entries counted once per (row tile, column tile)."""
    out = torch.empty_like(x)
    count = torch.zeros([4], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
            hl.atomic_add(count, [hl.arange(4)], 1.0)
    return count


def _count_three_entries_after(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """An extent no block is a multiple of."""
    out = torch.empty_like(x)
    count = torch.zeros([3], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [hl.arange(3)], 1.0)
    return count


def _count_entries_and_use_the_result(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """The old values the atomic returns would be one per thread; the leader loop has none."""
    out = torch.empty_like(x)
    count = torch.zeros([4], dtype=x.dtype, device=x.device)
    total = torch.zeros([1], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        old = hl.atomic_add(count, [hl.arange(4)], 1.0)
        hl.atomic_add(total, [0], old.sum())
    return total


def _count_a_symbolic_number_of_entries(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """``hl.arange(bsz)``: the entry count is only known at launch."""
    out = torch.empty_like(x)
    count = torch.zeros([64], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [hl.arange(bsz)], 1.0)
    return count


def _count_even_entries_after(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """``hl.arange(0, 8, 2)``: the even entries of an 8-entry buffer."""
    out = torch.empty_like(x)
    count = torch.zeros([8], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [hl.arange(0, 8, 2)], 1.0)
    return count


def _add_weights_after(x: torch.Tensor, w: torch.Tensor, bsz: int) -> torch.Tensor:
    """A host tensor value, read per entry."""
    out = torch.empty_like(x)
    total = torch.zeros([4], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(total, [hl.arange(4)], w)
    return total


def _add_one_weight_after(x: torch.Tensor, w: torch.Tensor, bsz: int) -> torch.Tensor:
    """A one-entry host tensor value broadcasts to every entry, as torch does."""
    out = torch.empty_like(x)
    total = torch.zeros([4], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(total, [hl.arange(4)], w)
    return total


def _add_the_block_size_after(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """A symbolic scalar value: the int argument itself."""
    out = torch.empty_like(x)
    total = torch.zeros([4], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(total, [hl.arange(4)], bsz)
    return total


def _count_shifted_entries_after(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """An index computed from the arange: declined."""
    out = torch.empty_like(x)
    count = torch.zeros([8], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [hl.arange(4) + 2], 1.0)
    return count


def _count_a_column_after(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """A second index component beside the arange: declined."""
    out = torch.empty_like(x)
    count = torch.zeros([4, 3], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1), block_size=bsz):
            out[tile_b, tile_m] = x[tile_b, tile_m] * 2
        hl.atomic_add(count, [hl.arange(4), 1], 1.0)
    return count


def _count_entries_in_a_tile_of_one(x: torch.Tensor) -> torch.Tensor:
    """No thread axis at all: the loop runs unguarded, once."""
    count = torch.zeros([4], dtype=x.dtype, device=x.device)
    for _tile in hl.tile(1):
        hl.atomic_add(count, [hl.arange(4)], 1.0)
    return count


def _add_row_sums_after_literal(x: torch.Tensor) -> torch.Tensor:
    """A per-row register value with a tile-uniform index: declined, each thread holds its own row (an extent equal to the row block included)."""
    total = torch.zeros([32], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        acc = hl.zeros([tile_b], dtype=x.dtype)
        for tile_m in hl.tile(x.size(1), block_size=8):
            acc += x[tile_b, tile_m].sum(dim=1)
        hl.atomic_add(total, [hl.arange(32)], acc)
    return total


_LANE_CONFIGS = [
    pytest.param({}, id="default"),
    pytest.param({"num_threads": [1]}, id="one_thread"),
    pytest.param({"num_threads": [8]}, id="lane_loop"),
]


def _kernel(fn: Callable[..., torch.Tensor]) -> helion.Kernel:
    return helion.kernel(
        fn, backend="cute", static_shapes=False, autotune_effort="none"
    )


def _config(bound: BoundKernel, **overrides: object) -> helion.Config:
    config = dict(bound.config_spec.default_config().config)
    config.update(overrides)
    return helion.Config.from_dict(config)


def _render(fn: Callable[..., torch.Tensor], *args: object, **overrides: object) -> str:
    with _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(fn), args)
        return bound.to_code(_config(bound, **overrides))


def _loop_and_guard(code: str) -> tuple[str, str | None]:
    """The entry loop header guarding the kernel's atomic and the ``if`` test inside it, if any."""
    lines = [line.rstrip() for line in code.splitlines()]
    (position,) = [
        index for index, line in enumerate(lines) if "cute.arch.atomic_add(" in line
    ]
    above = lines[position - 1].strip()
    if above.startswith("if "):
        guard: str | None = above[3:-1]
        loop = lines[position - 2].strip()
    else:
        guard = None
        loop = above
    assert loop.startswith("for _tensor_index_i in range("), code
    return loop, guard


def _thread_axes_launched(code: str) -> list[int]:
    """The launch axes with more than one thread, static or argument-sized."""
    (line,) = [line for line in code.splitlines() if "_launcher(" in line]
    block = line[line.index("block=") : -1]
    inner = block[block.index("(") + 1 :].rstrip(")")
    inner = inner.lstrip("(")
    return [
        axis for axis, entry in enumerate(inner.split(", ")) if entry.strip() != "1"
    ]


@pytest.mark.parametrize("overrides", _LANE_CONFIGS)
@pytest.mark.parametrize(
    ("fn", "args", "extent"),
    [
        pytest.param(_count_entries_after, (8,), 4, id="after"),
        pytest.param(_count_three_entries_after, (8,), 3, id="three_after"),
        pytest.param(_count_even_entries_after, (8,), 4, id="even_after"),
        pytest.param(_count_entries_after_literal, (), 4, id="after_literal"),
    ],
)
def test_an_arange_indexed_atomic_loops_over_its_entries_on_the_tile_leader(
    fn: Callable[..., torch.Tensor],
    args: tuple[object, ...],
    extent: int,
    overrides: dict[str, object],
) -> None:
    """Every launch axis with threads, the exited column loop's included, guards the loop with its leader; the folded per-thread coordinate is not the index."""
    if fn is _count_entries_after_literal and overrides:
        pytest.skip("the literal twin pins the default config only")
    code = _render(fn, torch.randn(40, 100), *args, **overrides)
    loop, guard = _loop_and_guard(code)
    assert loop == f"for _tensor_index_i in range({extent}):", code
    assert guard is not None, code
    for axis in _thread_axes_launched(code):
        assert f"cute.arch.thread_idx()[{axis}] == 0" in guard, code
    assert "cutlass.Int32(iota)" not in code, code


def test_an_arange_indexed_atomic_without_a_thread_axis_loops_unguarded() -> None:
    code = _render(_count_entries_in_a_tile_of_one, torch.randn(40, 100))
    loop, guard = _loop_and_guard(code)
    assert loop == "for _tensor_index_i in range(4):", code
    assert guard is None, code


def test_a_one_entry_host_value_is_read_once_for_every_entry() -> None:
    """The loop read ``w[_tensor_index_i]`` past the end of a one-entry value; under dynamic shapes the length is a size symbol, so the entry is selected at runtime."""
    code = _render(_add_one_weight_after, torch.randn(40, 100), torch.tensor([3.0]), 8)
    assert "w[_tensor_index_i if w_size_0 > 1 else 0]" in code, code
    static = helion.kernel(
        _add_one_weight_after,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with _mock_cuda_unavailable():
        bound = _cpu_bind(static, (torch.randn(40, 100), torch.tensor([3.0]), 8))
        code = bound.to_code(bound.config_spec.default_config())
    assert "w[0]" in code, code


def test_a_symbolic_scalar_value_is_the_one_value_of_every_entry() -> None:
    code = _render(_add_the_block_size_after, torch.randn(40, 100), 8)
    loop, _guard = _loop_and_guard(code)
    assert loop == "for _tensor_index_i in range(4):", code
    assert "val=cutlass.Float32(bsz)" in code, code


@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_count_shifted_entries_after, id="shifted"),
        pytest.param(_count_a_column_after, id="second_component"),
    ],
)
def test_an_index_computed_from_an_arange_is_declined(
    fn: Callable[..., torch.Tensor],
) -> None:
    """The resolver sees the bare arange chain only; the per-thread form would repeat the atomic across the tile (24 and 16 instead of 4 and 4 for ``hl.arange(4) % 2``)."""
    with pytest.raises(exc.BackendUnsupported, match="computed from hl.arange"):
        _render(fn, torch.randn(40, 100), 8)


def test_a_host_value_of_another_static_length_is_declined() -> None:
    static = helion.kernel(
        _add_one_weight_after,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with (
        _mock_cuda_unavailable(),
        pytest.raises(exc.BackendUnsupported, match="value of length 3"),
    ):
        bound = _cpu_bind(
            static, (torch.randn(40, 100), torch.tensor([1.0, 2.0, 3.0]), 8)
        )
        bound.to_code(bound.config_spec.default_config())


def test_an_arange_indexed_atomic_whose_result_is_used_is_declined() -> None:
    """Neither the leader loop nor the per-thread form has a result to hand back: declining beats the per-thread form's repeated adds."""
    with pytest.raises(exc.BackendUnsupported, match="leader.thread"):
        _render(_count_entries_and_use_the_result, torch.randn(40, 100), 8)


def test_an_arange_of_symbolic_length_is_declined() -> None:
    with pytest.raises(exc.BackendUnsupported, match="symbolic length"):
        _render(_count_a_symbolic_number_of_entries, torch.randn(40, 100), 8)


def test_an_arange_indexed_atomic_with_a_per_thread_value_is_declined() -> None:
    """One column thread keeps the row sums serial, so the CPU stub reaches the atomic (a warp reduction would ask the device for its shared-memory budget)."""
    with pytest.raises(exc.BackendUnsupported, match="device tensor value"):
        _render(_add_row_sums_after_literal, torch.randn(40, 100), num_threads=[1])


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
@pytest.mark.parametrize("bsz", [8, 32, 40])
@pytest.mark.parametrize("overrides", _LANE_CONFIGS)
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_count_entries_after, id="after"),
        pytest.param(_count_entries_inside, id="inside"),
        pytest.param(_count_three_entries_after, id="three_after"),
        pytest.param(_count_even_entries_after, id="even_after"),
        pytest.param(_add_weights_after, id="weights_after"),
        pytest.param(_add_one_weight_after, id="one_weight_after"),
        pytest.param(_add_the_block_size_after, id="block_size_after"),
    ],
)
def test_arange_indexed_atomics_add_once_per_tile(
    fn: Callable[..., torch.Tensor], overrides: dict[str, object], bsz: int
) -> None:
    """Two row tiles of 32 over 40 rows: every entry counts 2 (per column tile as well inside the loop) at 8, 32 and 40 columns per block, under 32, 1 and 8 column threads; a launch over 1024 threads raises at launch, and the atomic inside a lane loop it does not vary along is declined."""
    torch.manual_seed(bsz)
    x = torch.randn(40, 100, device=DEVICE)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
    one_weight = torch.tensor([3.0], device=DEVICE)
    row_tiles = 2.0
    column_tiles = float(-(-100 // bsz))
    expected = {
        _count_entries_after: torch.full([4], row_tiles, device=DEVICE),
        _count_entries_inside: torch.full([4], row_tiles * column_tiles, device=DEVICE),
        _count_three_entries_after: torch.full([3], row_tiles, device=DEVICE),
        _count_even_entries_after: torch.tensor(
            [2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0], device=DEVICE
        ),
        _add_weights_after: row_tiles * w,
        _add_one_weight_after: torch.full([4], row_tiles * 3.0, device=DEVICE),
        _add_the_block_size_after: torch.full([4], row_tiles * bsz, device=DEVICE),
    }[fn]
    weights = {_add_weights_after: w, _add_one_weight_after: one_weight}
    args = (x, weights[fn], 8) if fn in weights else (x, 8)
    bound = _kernel(fn).bind(args)
    try:
        compiled = bound.compile_config(_config(bound, **overrides))
    except exc.BackendUnsupported as error:
        assert fn is _count_entries_inside and overrides, error
        assert "would repeat once per lane" in str(error), error
        return
    call_args = (x, weights[fn], bsz) if fn in weights else (x, bsz)
    num_threads = overrides.get("num_threads")
    static_threads = 32 if num_threads is None else num_threads[0]  # pyrefly: ignore [index-error]
    if bsz * static_threads > 1024:
        with pytest.raises(exc.BackendUnsupported, match="too large"):
            compiled(*call_args)
        return
    torch.testing.assert_close(compiled(*call_args), expected)


@requires_cuda
def test_the_literal_twin_and_the_tile_of_one_add_once_per_tile() -> None:
    x = torch.randn(40, 100, device=DEVICE)
    literal = _kernel(_count_entries_after_literal).bind((x,))
    torch.testing.assert_close(literal(x), torch.full([4], 2.0, device=DEVICE))
    once = _kernel(_count_entries_in_a_tile_of_one).bind((x,))
    torch.testing.assert_close(once(x), torch.ones([4], device=DEVICE))
