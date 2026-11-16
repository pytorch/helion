"""Epilogue subtiling splits a tile the config fixes, by the extent it gives it.

``_find_split_candidate`` takes a tile dim only when ``epilogue_subtile``
divides the extent the config gives the tile.  A block size over another
block size (``block_size=[bs * 2]`` with ``bs = hl.register_block_size(...)``)
or a host helper over arithmetic on a tunable (``next_power_of_2(foo * 2)``)
is fixed by the config as well, written inline in ``block_size`` or not:
the store splits where the factor divides that extent and stays whole where
the pieces would be zero lanes wide.  A store that reads the split tile's
index off a loop input's size stays whole too, and a kernel offers
``epilogue_subtile`` only where some config splits a store: along a tile, or
along the full slice of a store through a tensor descriptor.
Triton renders on CPU (CuTe refuses a store split over a symbolic tile).
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing.triton_codegen import cpu_codegen as cpu_codegen
from helion._utils import triton_is_available
from helion.autotuner.config_fragment import PowerOfTwoFragment
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    skipUnlessBackends(["triton"]),
    pytest.mark.skipif(
        not triton_is_available(), reason="Triton is required for source generation"
    ),
]


@helion.kernel(backend="triton", autotune_effort="none")
def scale_into_doubled_block(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    block_size = hl.register_block_size(1, 64)
    for tile_n in hl.tile([n], block_size=[block_size * 2]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(backend="triton", autotune_effort="none")
def scale_into_helper_block(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    foo = hl.register_tunable("foo", PowerOfTwoFragment(1, 64))
    block_size = helion.next_power_of_2(foo * 2)
    for tile_n in hl.tile([n], block_size=[block_size]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(backend="triton", autotune_effort="none")
def scale_into_inline_helper_block(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    foo = hl.register_tunable("foo", PowerOfTwoFragment(1, 64))
    for tile_n in hl.tile([n], block_size=[helion.next_power_of_2(foo * 2)]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(backend="triton", autotune_effort="none")
def subtract_row_total(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            total = total + x[tile_m, tile_n].sum(-1)
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] - total[:, None]
    return out


@helion.kernel(backend="triton", autotune_effort="none")
def subtract_from_row_total(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            total = total + x[tile_m, tile_n].sum(-1)
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = total[:, None] - x[tile_m, tile_n]
    return out


@helion.kernel(backend="triton", autotune_effort="none")
def scale_rows_first(
    x: torch.Tensor, s: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = s[tile_m][:, None] * x[tile_m, tile_n]
    return out


@helion.kernel(backend="triton", autotune_effort="none")
def scale_into_unit_columns(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 1]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(backend="triton", autotune_effort="none")
def scale_unit_rows(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0), block_size=1):
        out[tile_m, :] = x[tile_m, :] * 2.0
    return out


@helion.kernel(backend="triton", autotune_effort="none", static_shapes=False)
def scale_unit_rows_dynamic(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0), block_size=1):
        out[tile_m, :] = x[tile_m, :] * 2.0
    return out


@helion.kernel(backend="triton", autotune_effort="none")
def scale_2d(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(backend="triton", autotune_effort="none")
def add_row_slice(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    k = a.size(2)
    out = torch.empty_like(a)
    for tile0, tile1 in hl.tile([a.size(0), a.size(1)]):
        out[tile0, tile1, 0:k] = a[tile0, tile1, 0:k] + b[tile0, 0:k]
    return out


@helion.kernel(backend="triton", autotune_effort="none", static_shapes=False)
def subtract_row_total_by_width(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=w.size(0)):
            total = total + x[tile_m, tile_n].sum(-1)
        for tile_n in hl.tile(n, block_size=w.size(0)):
            out[tile_m, tile_n] = x[tile_m, tile_n] - total[:, None]
    return out


@helion.kernel(backend="triton", autotune_effort="none", static_shapes=False)
def scale_by_width(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    for tile_n in hl.tile(n, block_size=w.size(0)):
        out[tile_n] = x[tile_n] * 2.0
    return out


@pytest.mark.parametrize(
    ("kernel", "values", "epilogue_subtile", "stores"),
    [
        (scale_into_doubled_block, {"block_sizes": [16]}, 2, 2),
        # A 2-wide tile under a factor of 4 stays whole (no zero-width pieces).
        (scale_into_doubled_block, {"block_sizes": [1]}, 4, 1),
        (scale_into_helper_block, {"foo": 16}, 2, 2),
        (scale_into_helper_block, {"foo": 1}, 4, 1),
        # The same helper written inline: no host variable names its result.
        (scale_into_inline_helper_block, {"foo": 16}, 2, 2),
        (scale_into_inline_helper_block, {"foo": 1}, 4, 1),
    ],
)
@skipIfRefEager("renders generated code")
def test_config_fixed_tile_splits_by_its_configured_extent(
    cpu_codegen: None,
    kernel: helion.Kernel,
    values: dict[str, object],
    epilogue_subtile: int,
    stores: int,
) -> None:
    x = torch.empty(1024)
    bound = kernel._bind_isolated((x, torch.empty_like(x)))
    config = helion.Config(
        **(bound.config_spec.default_config().config | values),
        epilogue_subtile=epilogue_subtile,
    )
    code = bound.to_code(config)
    assert code.count("tl.store(") == stores


@pytest.mark.parametrize(
    ("kernel", "block_sizes", "stores", "offered"),
    [
        # A 1-wide inner tile does not split, and neither does the outer tile
        # instead (codegen failed).
        (subtract_row_total, [8, 1, 1], 1, True),
        # The row total first: the split starts from ``total[:, None]``, whose
        # only tile is the row tile, and the store reads that tile's index off
        # the size of the total carried into the loop, which no piece can
        # offset (codegen failed, however wide the inner tile).
        (subtract_from_row_total, [8, 1, 1], 1, False),
        (subtract_from_row_total, [8, 16, 16], 1, False),
    ],
)
@skipIfRefEager("renders generated code")
def test_row_total_stores_stay_whole_in_either_operand_order(
    cpu_codegen: None,
    kernel: helion.Kernel,
    block_sizes: list[int],
    stores: int,
    offered: bool,
) -> None:
    x = torch.empty(64, 256)
    bound = kernel._bind_isolated((x, torch.empty_like(x)))
    assert bound.config_spec.epilogue_subtile_candidate_enabled is offered
    config = helion.Config(
        **(bound.config_spec.default_config().config | {"block_sizes": block_sizes}),
        epilogue_subtile=2,
    )
    assert bound.to_code(config).count("tl.store(") == stores


@skipIfRefEager("renders generated code")
def test_row_tile_indexed_in_the_stores_loop_splits(cpu_codegen: None) -> None:
    # The row scale first starts the split from a value whose only tile is the
    # row tile, and the store, in the loop over both tiles, indexes that tile
    # directly: its halves store.
    x = torch.empty(64, 256)
    bound = scale_rows_first._bind_isolated((x, torch.empty(64), torch.empty_like(x)))
    config = helion.Config(
        **(bound.config_spec.default_config().config | {"block_sizes": [8, 16]}),
        epilogue_subtile=2,
    )
    assert bound.to_code(config).count("tl.store(") == 2


@skipIfRefEager("renders generated code")
def test_dynamic_inner_tile_keeps_the_store_whole_whatever_its_hint(
    cpu_codegen: None,
) -> None:
    # ``block_size=w.size(0)`` stays dynamic under every config.  Its hint of 3
    # is odd, and the store stays whole rather than splitting the row tile
    # instead, whose halves would slice the row total (codegen failed).
    x = torch.empty(64, 256)
    bound = subtract_row_total_by_width._bind_isolated(
        (x, torch.empty(3), torch.empty_like(x))
    )
    config = helion.Config(
        **bound.config_spec.default_config().config, epilogue_subtile=2
    )
    assert bound.to_code(config).count("tl.store(") == 1


@pytest.mark.parametrize(
    ("kernel", "args", "enabled"),
    [
        # A block size no config fixes: no config splits its store.
        (scale_by_width, (torch.empty(1024), torch.empty(8), torch.empty(1024)), False),
        (scale_into_doubled_block, (torch.empty(1024), torch.empty(1024)), True),
        # An inner tile every config leaves 1 wide, a literal block size or a
        # dim of size 1: the split never moves on to the row tile.
        (scale_into_unit_columns, (torch.empty(64, 256), torch.empty(64, 256)), False),
        (scale_2d, (torch.empty(64, 1), torch.empty(64, 1)), False),
        # Rows 1 wide, but the descriptor store a split takes splits its full
        # slice.
        (scale_unit_rows, (torch.empty(64, 256), torch.empty(64, 256)), True),
    ],
)
@skipIfRefEager("ref eager mode builds no device IR to offer epilogue_subtile")
def test_only_tiles_a_config_can_split_offer_the_split(
    cpu_codegen: None, kernel: helion.Kernel, args: tuple[object, ...], enabled: bool
) -> None:
    bound = kernel._bind_isolated(args)
    assert bound.config_spec.epilogue_subtile_candidate_enabled is enabled


@skipIfRefEager("ref eager mode builds no device IR to offer epilogue_subtile")
def test_a_dynamic_full_slice_offers_the_split_by_its_hint(cpu_codegen: None) -> None:
    # A guard that the factor divides the input's size would hold for this
    # bind only, and the kernel serves other sizes.
    x = torch.empty(64, 256)
    bound = scale_unit_rows_dynamic._bind_isolated((x, torch.empty_like(x)))
    assert bound.config_spec.epilogue_subtile_candidate_enabled
    assert not any("Mod" in str(guard.expr) for guard in bound.env.shape_env.guards)


@pytest.mark.parametrize(("rows", "block_sizes"), [(16, [8, 16]), (1, None)])
@skipIfRefEager("renders generated code")
def test_a_slice_operand_broadcasts_by_tile(
    cpu_codegen: None, rows: int, block_sizes: list[int] | None
) -> None:
    # ``b[tile0, 0:k]`` lines up with ``a``'s tiles by static evaluations the
    # bind reads after the candidate check's guards, which must not be the
    # ones the ShapeEnv cached before them
    # (``CompileEnvironment.maybe_evaluate_static``).  An inner dim of size 1,
    # whose extent the check takes without testing its hint, raised
    # ShapeMismatch when they were.
    bound = add_row_slice._bind_isolated((torch.empty(8, rows, 2), torch.empty(8, 4)))
    config = bound.config_spec.default_config()
    if block_sizes is not None:
        config = helion.Config(**(config.config | {"block_sizes": block_sizes}))
    assert "tl.store(" in bound.to_code(config)
