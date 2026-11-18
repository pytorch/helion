"""CuTe component packets: one vector store for the K trailing elements of a
tile lane's elements.

A tile lane on a dim of stride K whose K trailing elements are contiguous
(``out[tile, c]`` of an ``[n, K]`` tensor: the real and imaginary parts of a
complex64 buffer) has no stride-1 packet: each of its sibling stores writes
one 4-byte element in every K.  The sibling stores of one lane body instead
share one list and one flush of ``V * K`` values at component 0 of the lane
base (``CuteComponentGroup``).  The lowering proves the packet
(``_cute_component_store_ctx``: contiguity, extent and start multiples of V,
one store no wider than the device takes, base and stride alignment) and,
once the lane body is complete, the group (``component_group_refusal``: every
component once, in order, nothing with effects between the siblings, and no
other access of the tensor or of one that may alias it in the lane loop).
Whatever it cannot prove keeps its scalar stores.

A store may also name every component at once, through a tile that covers
the trailing dim whole on the lane's own thread (``out[tile_e, tile_c]``):
that tile's lane loop then becomes a trace-time component loop inside the
lane's V-loop (its index the literal lane, so ``tile_c.index == c`` folds
per lane) and the one store appends once per component.
"""

from __future__ import annotations

import ast
import logging
import re
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest
import torch

import helion
from helion._compiler.cute import target_facts
from helion._compiler.cute.emitters.lane_packets import CuteComponentGroup
from helion._compiler.cute.emitters.lane_packets import CuteComponentStore
from helion._compiler.cute.emitters.lane_packets import CuteTileVecStoreSite
from helion._compiler.cute.memory_ops import component_group_refusal
from helion._compiler.cute.memory_ops import demote_reordered_tile_vec_stores
from helion._compiler.cute.packet_store_policy import lower_packet_store_policy
from helion._compiler.lane_loop_context import LaneLoopContext
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.unfold_kernels import sliding_windows
from helion._testing.unfold_kernels import sliding_windows_into
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

_MEMORY_OPS_LOG = "helion._compiler.cute.memory_ops"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def pairs(x: torch.Tensor) -> torch.Tensor:
    # The (real, imaginary) pairs of a complex64 tensor of x's values.
    n = x.size(0)
    out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        real = x[tile]
        out[tile, 0] = real
        out[tile, 1] = torch.zeros_like(real)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def caller_pairs(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # The same stores into a tensor the caller provides.
    n = x.size(0)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile, 0] = value
        out[tile, 1] = value * 2
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def dynamic_pairs(x: torch.Tensor) -> torch.Tensor:
    # ``pairs`` over an extent known only at run time.
    n = x.size(0)
    out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile, 0] = value
        out[tile, 1] = value * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def triples(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # Three sibling stores: all of an [n, 3] tensor's components, or three of
    # the four of an [n, 4] tensor.
    n = x.size(0)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile, 0] = value
        out[tile, 1] = value * 2
        out[tile, 2] = value * 3
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def quads(x: torch.Tensor) -> torch.Tensor:
    # Components over two trailing dims: [n, 2, 2] holds four per element.
    n = x.size(0)
    out = torch.empty([n, 2, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile, 0, 0] = value
        out[tile, 0, 1] = value + 1
        out[tile, 1, 0] = value + 2
        out[tile, 1, 1] = value + 3
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def caller_quads(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # ``quads`` into a tensor the caller provides.
    n = x.size(0)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile, 0, 0] = value
        out[tile, 0, 1] = value + 1
        out[tile, 1, 0] = value + 2
        out[tile, 1, 1] = value + 3
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def shifted_pairs(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # The lane coordinate is the tile's index plus four.
    n = x.size(0)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile + 4, 0] = value
        out[tile + 4, 1] = value * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def repeated_component(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile, 0] = value
        out[tile, 1] = value * 2
        out[tile, 0] = value + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def reversed_components(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile, 1] = value * 2
        out[tile, 0] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def read_between(x: torch.Tensor) -> torch.Tensor:
    # The second sibling stores what a load reads back from the first.
    n = x.size(0)
    out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        out[tile, 0] = x[tile]
        back = out[tile, 0]
        out[tile, 1] = back * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def read_before(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # Each lane reads its element before the siblings overwrite it.
    n = x.size(0)
    for tile in hl.tile(n):
        old = out[tile, 1]
        out[tile, 0] = x[tile] + old
        out[tile, 1] = old * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def store_between(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Another tensor's store runs between the siblings of each group.
    n = x.size(0)
    a = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    b = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        value = x[tile]
        a[tile, 0] = value
        b[tile, 0] = value * 2
        a[tile, 1] = value + 1
        b[tile, 1] = value * 3
    return a, b


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def two_groups(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.size(0)
    a = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    b = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        value = x[tile]
        a[tile, 0] = value
        a[tile, 1] = value + 1
        b[tile, 0] = value * 2
        b[tile, 1] = value * 3
    return a, b


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def caller_two_groups(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # ``two_groups`` into tensors the caller provides, which may be one.
    n = x.size(0)
    for tile in hl.tile(n):
        value = x[tile]
        a[tile, 0] = value
        a[tile, 1] = value + 1
        b[tile, 0] = value * 2
        b[tile, 1] = value * 3
    return a, b


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_pairs(x: torch.Tensor) -> torch.Tensor:
    # The lane tile is a device loop inside the rows' grid.
    m, n = x.shape
    out = torch.empty([m, n, 2], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n):
            value = x[tile_m, tile_n]
            out[tile_m, tile_n, 0] = value
            out[tile_m, tile_n, 1] = -value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_pairs_from_two(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # ``row_pairs`` whose lane tiles start at column 2.
    m, n = x.shape
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(2, n):
            value = x[tile_m, tile_n]
            out[tile_m, tile_n, 0] = value
            out[tile_m, tile_n, 1] = -value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_pairs_then_first(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # After the lane loop every thread reads the row's first pair.
    m, n = x.shape
    out = torch.empty([m, n, 2], dtype=x.dtype, device=x.device)
    first = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n):
            value = x[tile_m, tile_n]
            out[tile_m, tile_n, 0] = value
            out[tile_m, tile_n, 1] = -value
        first[tile_m] = out[tile_m, 0, 1]
    return out, first


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_read_between(x: torch.Tensor) -> torch.Tensor:
    # ``read_between`` with the lane tile as a device loop.
    m, n = x.shape
    out = torch.empty([m, n, 2], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n, 0] = x[tile_m, tile_n]
            back = out[tile_m, tile_n, 0]
            out[tile_m, tile_n, 1] = back * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def grid_pairs(x: torch.Tensor) -> torch.Tensor:
    # Both axes on the grid; the lanes walk the columns.
    m, n = x.shape
    out = torch.empty([m, n, 2], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        value = x[tile_m, tile_n]
        out[tile_m, tile_n, 0] = value
        out[tile_m, tile_n, 1] = -value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def caller_grid_pairs(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # ``grid_pairs`` into a tensor the caller provides.
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        value = x[tile_m, tile_n]
        out[tile_m, tile_n, 0] = value
        out[tile_m, tile_n, 1] = -value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def column_pairs(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # The lanes walk the columns on the grid and the rows are a device loop
    # inside their vector loop: a store's row is defined below the place a
    # packet of the lane would flush at.
    m, n = x.shape
    for tile_n in hl.tile(n):
        for tile_m in hl.tile(m):
            value = x[tile_m, tile_n]
            out[tile_m, tile_n, 0] = value
            out[tile_m, tile_n, 1] = -value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def pairs_of_transposed(xt: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # ``xt`` is contiguous along the grid's first axis, whose vector loop
    # ``cute_vloop_sink`` moves inside the lanes of the second for such a
    # load; the siblings' lanes are the second axis.
    n, m = xt.shape
    for tile_m, tile_n in hl.tile([m, n]):
        value = xt[tile_n, tile_m].transpose(0, 1)
        out[tile_m, tile_n, 0] = value
        out[tile_m, tile_n, 1] = -value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def column_sum_pairs(x: torch.Tensor) -> torch.Tensor:
    # A device loop reduces the rows before the siblings (the shape
    # ``cute_vloop_sink`` distributes the V-loop over).
    m, n = x.shape
    out = torch.empty([n, 2], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            acc += torch.sum(x[tile_m, tile_n], dim=0)
        out[tile_n, 0] = acc
        out[tile_n, 1] = acc * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def pairs_then_first_row(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # A store that depends on no lane follows the siblings: the lane-loop
    # distribution places it after the loop, behind the flush.
    n = x.size(0)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile, 0] = value
        out[tile, 1] = value * 2
        out[tile.begin, 1] = 0.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def scatter_component(
    pair: torch.Tensor, grad: torch.Tensor, component: hl.constexpr
) -> torch.Tensor:
    # select_scatter(pair, grad, 1, component): one tile covers the components.
    elements, components = pair.shape
    out = torch.empty([elements, components], dtype=pair.dtype, device=pair.device)
    for tile_e, tile_c in hl.tile([elements, components]):
        chosen = (tile_c.index == component)[None, :]
        out[tile_e, tile_c] = torch.where(
            chosen, grad[tile_e][:, None], pair[tile_e, tile_c]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def broadcast_components(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # The stored value reads no component coordinate.
    elements, components = out.shape
    for tile_e, tile_c in hl.tile([elements, components]):
        out[tile_e, tile_c] = x[tile_e][:, None] + hl.zeros(
            [tile_e, tile_c], dtype=x.dtype
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def components_and_first(pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # A store of one element per lane beside the store of every component.
    elements, components = pair.shape
    out = torch.empty([elements, components], dtype=pair.dtype, device=pair.device)
    first = torch.empty([elements], dtype=pair.dtype, device=pair.device)
    for tile_e, tile_c in hl.tile([elements, components]):
        out[tile_e, tile_c] = pair[tile_e, tile_c] * 2
        first[tile_e] = pair[tile_e, 0] + 1
    return out, first


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def planes_to_pairs(planes: torch.Tensor) -> torch.Tensor:
    # planes: [components, elements], each component contiguous along the
    # elements; the load's row is the component coordinate.
    components, elements = planes.shape
    out = torch.empty([elements, components], dtype=planes.dtype, device=planes.device)
    for tile_e, tile_c in hl.tile([elements, components]):
        out[tile_e, tile_c] = planes[tile_c, tile_e].transpose(0, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def pairs_and_planes(
    planes: torch.Tensor, pair: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Two stores of every component; the planes are loaded after the first.
    components, elements = planes.shape
    doubled = torch.empty_like(pair)
    out = torch.empty_like(pair)
    for tile_e, tile_c in hl.tile([elements, components]):
        doubled[tile_e, tile_c] = pair[tile_e, tile_c] * 2
        out[tile_e, tile_c] = planes[tile_c, tile_e].transpose(0, 1)
    return doubled, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def cube_components(x: torch.Tensor) -> torch.Tensor:
    m, n, k = x.shape
    out = torch.empty([m, n, k], dtype=x.dtype, device=x.device)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        out[tile_m, tile_n, tile_k] = x[tile_m, tile_n, tile_k] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def weigh_components_in_place(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # The lane body reads the tensor it stores, and an operand per element.
    for tile_e, tile_c in hl.tile(x.size()):
        x[tile_e, tile_c] = x[tile_e, tile_c] * weight[tile_e][:, None]
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def normalize_components(x: torch.Tensor) -> torch.Tensor:
    # A reduction over the components of each element.
    out = torch.empty_like(x)
    for tile_e, tile_c in hl.tile(x.size()):
        value = x[tile_e, tile_c]
        out[tile_e, tile_c] = value / value.sum(-1, keepdim=True)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def cumsum_components(x: torch.Tensor) -> torch.Tensor:
    # A scan over the components of each element.
    out = torch.empty_like(x)
    for tile_e, tile_c in hl.tile(x.size()):
        out[tile_e, tile_c] = hl.cumsum(x[tile_e, tile_c], dim=1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def reverse_cumsum_components(x: torch.Tensor) -> torch.Tensor:
    # The scan runs from the last component down.
    out = torch.empty_like(x)
    for tile_e, tile_c in hl.tile(x.size()):
        out[tile_e, tile_c] = hl.cumsum(x[tile_e, tile_c], dim=1, reverse=True)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def caller_components(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # One tile covers the components of a tensor the caller provides.
    elements, components = x.shape
    for tile_e, tile_c in hl.tile([elements, components]):
        out[tile_e, tile_c] = x[tile_e, tile_c] * 2
    return out


def _config(
    block: int = 256, threads: int = 64, width: int = 4, **knobs: object
) -> helion.Config:
    return helion.Config(
        block_sizes=[block], num_threads=[threads], cute_vector_widths=[width], **knobs
    )


def _lane_config(
    block: tuple[int, int] = (256, 2),
    threads: tuple[int, int] = (64, 1),
    widths: tuple[int, int] = (4, 1),
    order: tuple[int, int] = (0, 1),
    **knobs: object,
) -> helion.Config:
    """An ``[elements, components]`` tile: 64 threads of 4 elements each
    along the elements, every thread walking all the components."""
    return helion.Config(
        block_sizes=list(block),
        num_threads=list(threads),
        cute_vector_widths=list(widths),
        loop_orders=[list(order)],
        **knobs,
    )


_ROW_CONFIG = helion.Config(
    block_sizes=[256], num_threads=[64], cute_vector_widths=[1, 4]
)
# ``_ROW_CONFIG`` with chunks of 2 columns.
_ROW_PAIR_CONFIG = helion.Config(
    block_sizes=[256], num_threads=[64], cute_vector_widths=[1, 2]
)
_GRID_CONFIG = helion.Config(
    block_sizes=[2, 256], num_threads=[2, 64], cute_vector_widths=[1, 4]
)
_COLUMN_SUM_CONFIG = helion.Config(
    block_sizes=[16, 64], num_threads=[4, 1], cute_vector_widths=[4, 1]
)
# ``column_pairs``: 64 threads of 4 columns each, 16 rows per device-loop tile.
_COLUMN_CONFIG = helion.Config(
    block_sizes=[256, 16], num_threads=[64, 1], cute_vector_widths=[4, 1]
)
# One flattened tile over rows and columns: the lanes run along both.
_FLAT_GRID_CONFIG = helion.Config(
    block_sizes=[1, 512],
    num_threads=[1, 128],
    cute_vector_widths=[1, 4],
    flatten_loops=[True],
)
# ``pairs_of_transposed`` with vector loops on both grid axes; with
# ``cute_vloop_sink`` the first axis's loop would sink into the second's lanes.
_BOTH_AXES_CONFIGS: list[dict[str, list[int]]] = [
    {"block_sizes": [64, 64], "num_threads": [4, 4], "cute_vector_widths": [4, 2]},
    {"block_sizes": [64, 64], "num_threads": [4, 4], "cute_vector_widths": [4, 4]},
    {"block_sizes": [16, 256], "num_threads": [1, 64], "cute_vector_widths": [4, 4]},
]

_FLUSH = re.compile(r"_cute_store_u(?:16|32)(?:_vec|x\d+_streaming)\(")


def _code(
    kernel: object,
    args: tuple[object, ...],
    config: helion.Config,
    capability: tuple[int, int] = (10, 0),
) -> str:
    return cpu_bind(kernel, args, capability=capability).to_code(config)  # type: ignore[arg-type]


def _device_function(code: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )


def _stores(code: str) -> tuple[int, int]:
    """``(packet flushes, scalar stores)`` of the generated device function."""
    source = ast.unparse(_device_function(code))
    return len(_FLUSH.findall(source)), source.count(".store(")


def _appends_per_lane(code: str, collector: str) -> list[str]:
    """The values the V-loop body appends to ``collector``, in order."""
    function = _device_function(code)
    (vloop,) = (
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and ast.unparse(node.iter).startswith("cutlass.range_constexpr")
        and any(
            ast.unparse(statement).startswith(f"{collector}.append(")
            for statement in node.body
        )
    )
    return [
        ast.unparse(statement.value.args[0])  # type: ignore[attr-defined]
        for statement in vloop.body
        if ast.unparse(statement).startswith(f"{collector}.append(")
    ]


@pytest.mark.parametrize(
    ("dtype", "block", "width", "layout", "packet_bytes"),
    [
        (torch.float32, 256, 2, "blocked", 16),
        (torch.float32, 256, 4, "blocked", 32),
        (torch.float32, 256, 4, "strided", 32),
        (torch.bfloat16, 512, 8, "blocked", 32),
    ],
    ids=["fp32-16B", "fp32-32B", "fp32-32B-strided", "bf16-32B"],
)
def test_sibling_stores_share_one_packet(
    dtype: torch.dtype, block: int, width: int, layout: str, packet_bytes: int
) -> None:
    code = _code(
        pairs,
        (torch.empty(4096, dtype=dtype),),
        _config(block, threads=block // (2 * width), width=width)
        if layout == "blocked"
        else _config(block, block // (2 * width), width, cute_lane_layouts=[layout]),
    )
    assert _stores(code) == (1, 0)
    # One flush at component 0 of the lane base: the trailing dim is left out
    # of the pointer, which the lane-loop passes read along the lane dim.
    helper = "_cute_store_u32_vec" if dtype.itemsize == 4 else "_cute_store_u16_vec"
    assert (
        f"{helper}(out.iterator + cutlass.Int32(lane_base_0) * "
        "cutlass.Int32(out.layout.stride[0]), _tile_store_vals_0_0)"
    ) in code
    # Each lane appends its components in memory order.
    values = _appends_per_lane(code, "_tile_store_vals_0_0")
    assert len(values) == 2
    assert "real" in values[0] and "full" in values[1]
    assert width * 2 * dtype.itemsize == packet_bytes


def test_components_over_two_trailing_dims() -> None:
    code = _code(quads, (torch.empty(4096),), _config(width=2))
    assert _stores(code) == (1, 0)
    values = _appends_per_lane(code, "_tile_store_vals_0_0")
    assert [value.split("(")[1].split(")")[0] for value in values] == [
        "value",
        "v_1",
        "v_3",
        "v_5",
    ]


def test_masked_tail_tile_predicates_the_flush() -> None:
    # 1000 elements in tiles of 256: the last tile is partial, and since
    # 1000 is a multiple of 4 its mask is uniform across each chunk.
    code = _code(pairs, (torch.empty(1000),), _config())
    assert _stores(code) == (1, 0)
    function = _device_function(code)
    (guard,) = (
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and "_cute_store_u32_vec" in ast.unparse(node.body)
    )
    assert ast.unparse(guard.test) == "mask_0"


# One case per proof obligation of ``_cute_component_store_ctx``: each changes
# the one fact its name says and must keep every sibling store scalar.
@pytest.mark.parametrize(
    ("kernel", "args", "config", "capability", "scalar_stores"),
    [
        # 1002 elements: a chunk of 4 could straddle the extent.
        (pairs, (torch.empty(1002),), _config(), (10, 0), 2),
        # 8 lanes of 2 fp32 components are 64 bytes, wider than any store.
        (pairs, (torch.empty(4096),), _config(512, 64, 8), (10, 0), 2),
        # Before sm_100 the widest store is 16 bytes.
        (pairs, (torch.empty(4096),), _config(), (9, 0), 2),
        # 2 lanes of 3 fp32 components are 24 bytes: no vector store.
        (
            triples,
            (torch.empty(4096), torch.empty(4096, 3)),
            _config(width=2),
            (10, 0),
            3,
        ),
        # The output starts 8 bytes into its storage: a 32-byte packet at a
        # multiple of 4 elements is misaligned.
        (
            caller_pairs,
            (torch.empty(4096), torch.empty(4097, 2)[1:]),
            _config(),
            (10, 0),
            2,
        ),
        # Its components are 2 elements apart (every other column).
        (
            caller_pairs,
            (torch.empty(4096), torch.empty(4096, 4)[:, ::2]),
            _config(),
            (10, 0),
            2,
        ),
        # The first two columns of rows of four: the pairs are contiguous
        # but 4 elements apart, so a packet would overwrite the columns
        # between them.
        (
            caller_pairs,
            (torch.empty(4096), torch.empty(4096, 4)[:, :2]),
            _config(),
            (10, 0),
            2,
        ),
        # Every other pair of a buffer: the lane stride is 4 elements again.
        (
            caller_pairs,
            (torch.empty(4096), torch.empty(8192, 2)[::2]),
            _config(),
            (10, 0),
            2,
        ),
        # Overlapping pairs: an element's second component is the next
        # element's first.
        (
            caller_pairs,
            (torch.empty(4096), torch.empty(8194).as_strided((4096, 2), (2, 2))),
            _config(),
            (10, 0),
            2,
        ),
        # The trailing dims transposed: the components are contiguous, but
        # not in the order of their indices.
        (
            caller_quads,
            (torch.empty(4096), torch.empty(4096, 2, 2).transpose(1, 2)),
            _config(width=2),
            (10, 0),
            4,
        ),
        # Tiles from column 2: every chunk of 4 starts 2 pairs off a
        # multiple of 4, where the base alignment proves nothing.  (Rows of
        # 1028 pairs are a multiple of the packet apart.)
        (
            row_pairs_from_two,
            (torch.empty(8, 1026), torch.empty(8, 1028, 2)[:, :1026]),
            _ROW_CONFIG,
            (10, 0),
            2,
        ),
        # The lane coordinate is not the tile's own index.
        (
            shifted_pairs,
            (torch.empty(4096), torch.empty(4100, 2)),
            _config(),
            (10, 0),
            2,
        ),
        # Rows of 1025 pairs viewed as 1024: a row is 2050 elements after
        # the one before, no multiple of the packet's 8.
        (
            caller_grid_pairs,
            (torch.empty(8, 1024), torch.empty(8, 1025, 2)[:, :1024]),
            _GRID_CONFIG,
            (10, 0),
            2,
        ),
        # The row coordinate is a device loop's, defined inside the lanes'
        # vector loop: a flush after that loop could not address the rows.
        (
            column_pairs,
            (torch.empty(64, 1024), torch.empty(64, 1024, 2)),
            _COLUMN_CONFIG,
            (10, 0),
            2,
        ),
        # One flattened tile: its lane base is an offset into the flattened
        # rows and columns, not a column.
        (
            caller_grid_pairs,
            (torch.empty(8, 1024), torch.empty(8, 1024, 2)),
            _FLAT_GRID_CONFIG,
            (10, 0),
            2,
        ),
    ],
    ids=[
        "extent-not-multiple",
        "wider-than-a-store",
        "sm90-32-bytes",
        "not-a-power-of-two",
        "misaligned-base",
        "strided-components",
        "wide-rows",
        "every-other-row",
        "overlapping-elements",
        "transposed-trailing-dims",
        "tile-start-not-multiple",
        "shifted-lane",
        "misaligned-row-stride",
        "row-defined-in-the-lanes",
        "flattened-grid",
    ],
)
def test_unproven_packet_keeps_scalar_stores(
    kernel: object,
    args: tuple[torch.Tensor, ...],
    config: helion.Config,
    capability: tuple[int, int],
    scalar_stores: int,
) -> None:
    assert _stores(_code(kernel, args, config, capability)) == (0, scalar_stores)


def test_transposed_buffer_keeps_its_stride_one_packets() -> None:
    # A transposed [2, n] buffer: the lane dim is the fast one, so each
    # sibling is a stride-1 packet of its own row, as before.
    out = torch.empty(2, 4096).t()
    code = _code(caller_pairs, (torch.empty(4096), out), _config())
    assert _stores(code) == (2, 0)
    assert "cutlass.Int32(1) * cutlass.Int32(out.layout.stride[1])" in code


def test_dynamic_extent_keeps_scalar_stores() -> None:
    # Nothing proves a run-time extent a multiple of the vector width (the
    # stride-1 tile packets decline it the same way).
    code = _code(dynamic_pairs, (torch.empty(4096),), _config())
    assert _stores(code) == (0, 2)
    assert "cute.arch.load(x.iterator" not in code


def test_aligned_view_of_caller_storage_packs() -> None:
    x = torch.empty(4096)
    assert _stores(_code(caller_pairs, (x, torch.empty(4096, 2)), _config())) == (1, 0)
    # 16 bytes into the storage: a 16-byte packet stays aligned, a 32-byte
    # one does not.
    view = torch.empty(4098, 2)[2:]
    assert _stores(_code(caller_pairs, (x, view), _config(width=2))) == (1, 0)
    assert _stores(_code(caller_pairs, (x, view), _config(width=4))) == (0, 2)
    # The layouts ``test_unproven_packet_keeps_scalar_stores`` declines, with
    # the one fact proven: row-major trailing dims, tiles from column 2 in
    # chunks of 2, rows a multiple of the packet apart.
    quads_out = torch.empty(4096, 2, 2)
    assert _stores(_code(caller_quads, (x, quads_out), _config(width=2))) == (1, 0)
    from_two = (torch.empty(8, 1026), torch.empty(8, 1028, 2)[:, :1026])
    assert _stores(_code(row_pairs_from_two, from_two, _ROW_PAIR_CONFIG)) == (1, 0)
    rows = (torch.empty(8, 1024), torch.empty(8, 1024, 2))
    assert _stores(_code(caller_grid_pairs, rows, _GRID_CONFIG)) == (1, 0)


def test_sm90_packs_sixteen_bytes() -> None:
    code = _code(pairs, (torch.empty(4096),), _config(width=2), capability=(9, 0))
    assert _stores(code) == (1, 0)


# One case per condition of ``component_group_refusal``, with the reason the
# lowering logs when it restores the scalar stores.
@pytest.mark.parametrize(
    ("kernel", "args", "scalar_stores", "reason"),
    [
        (
            repeated_component,
            (torch.empty(4096),),
            3,
            "did not join in component order",
        ),
        (
            reversed_components,
            (torch.empty(4096),),
            2,
            "components 1 to 1 are not stored",
        ),
        (
            triples,
            (torch.empty(4096), torch.empty(4096, 4)),
            3,
            "components 3 to 3 are not stored",
        ),
        (
            read_before,
            (torch.empty(4096), torch.empty(4096, 2)),
            2,
            "out accessed in its lane loop",
        ),
        (
            store_between,
            (torch.empty(4096),),
            4,
            "a statement with effects runs between its sibling stores",
        ),
        (
            column_sum_pairs,
            (torch.empty(64, 1024),),
            2,
            "a statement of unknown memory effect runs in its lane loop",
        ),
    ],
    ids=[
        "repeated",
        "reversed",
        "incomplete",
        "read-before",
        "store-between",
        "reduction",
    ],
)
def test_unproven_group_restores_scalar_stores(
    kernel: object,
    args: tuple[torch.Tensor, ...],
    scalar_stores: int,
    reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = (
        _COLUMN_SUM_CONFIG
        if kernel is column_sum_pairs
        else _config(width=2 if kernel is triples else 4)
    )
    with caplog.at_level(logging.DEBUG, logger=_MEMORY_OPS_LOG):
        code = _code(kernel, args, config)
    assert _stores(code) == (0, scalar_stores)
    assert reason in caplog.text


def _refusal(
    body: str,
    *,
    members: tuple[int, ...] = (1, 3),
    components: int = 2,
    broken: bool = False,
    scope: tuple[int, ...] | None = None,
    other_store: int | None = None,
    packets: tuple[str, ...] = (),
) -> str | None:
    """``component_group_refusal`` for a group of ``out`` whose siblings are
    the statements ``members`` of ``body``, in a kernel of the tensors
    ``out``, ``x`` (proven disjoint from ``out``) and ``y`` (not).

    ``scope`` selects and orders the statements of the lane loop (all of
    ``body`` by default), ``other_store`` names a statement that is another
    collected store's append (of ``y``) and ``packets`` the tensors with a
    packet loaded above the V-loop."""
    statements = ast.parse(body).body
    placeholder = ast.parse("pass").body[0]
    group = CuteComponentGroup(
        components,
        "pointer",
        None,
        [
            CuteComponentStore(first, 1, statements[index], placeholder)
            for first, index in enumerate(members)
        ],
        broken,
    )
    site = CuteTileVecStoreSite(
        "vals", "out", statements[members[0]], placeholder, None, placeholder, group
    )
    context = LaneLoopContext(block_id=0)
    context.vec_stores.append(site)
    if other_store is not None:
        context.vec_stores.append(
            CuteTileVecStoreSite(
                "others", "y", statements[other_store], None, None, placeholder
            )
        )
    for name in packets:
        context.vec_loads[name, "pointer", None] = ("packet", torch.float32)
    return component_group_refusal(
        site,
        context,
        statements if scope is None else [statements[index] for index in scope],
        {"out", "x", "y"},
        {frozenset(("out", "x"))},
    )


_WHOLE_GROUP = """
a = (x.iterator + i).load()
vals.append(a)
b = a * 2
vals.append(b)
c = b + 1
"""


def test_group_refusal_accepts_a_whole_group() -> None:
    # Register work and loads of a disjoint tensor around and between the
    # siblings, and a packet of that tensor above the V-loop.
    assert _refusal(_WHOLE_GROUP) is None
    assert _refusal(_WHOLE_GROUP, packets=("x",)) is None


# One mutation of the whole group per condition of the proof.
@pytest.mark.parametrize(
    ("body", "options", "reason"),
    [
        (_WHOLE_GROUP, {"broken": True}, "did not join in component order"),
        (_WHOLE_GROUP, {"components": 3}, "components 2 to 2 are not stored"),
        # A sibling the lane-loop distribution placed outside the loop.
        (_WHOLE_GROUP, {"scope": (0, 1, 2, 4)}, "not a statement of the lane loop"),
        # The siblings in the other order.
        (_WHOLE_GROUP, {"scope": (0, 3, 2, 1, 4)}, "not a statement of the lane loop"),
        (
            _WHOLE_GROUP.replace("b = a * 2", "(x.iterator + i).store(a)"),
            {},
            "a statement with effects runs between",
        ),
        (
            _WHOLE_GROUP.replace("b = a * 2", "cute.arch.sync_threads()"),
            {},
            "a statement with effects runs between",
        ),
        (
            _WHOLE_GROUP.replace("b = a * 2", "others.append(a)"),
            {"other_store": 2},
            "a statement with effects runs between",
        ),
        # Reads of the tensor, or of one that may alias it: before, between
        # and after the siblings.
        (
            _WHOLE_GROUP.replace("x.iterator", "out.iterator"),
            {},
            "out accessed in its lane loop",
        ),
        (
            _WHOLE_GROUP.replace("b = a * 2", "b = (out.iterator + i).load()"),
            {},
            "out accessed in its lane loop",
        ),
        (
            _WHOLE_GROUP.replace("c = b + 1", "c = (y.iterator + i).load()"),
            {},
            "y accessed in its lane loop",
        ),
        # Another collected store of a tensor that may alias it.
        (
            _WHOLE_GROUP.replace("c = b + 1", "others.append(b)"),
            {"other_store": 4},
            "y accessed in its lane loop",
        ),
        # Accesses the names do not show.
        (
            _WHOLE_GROUP.replace("c = b + 1", "c = (pointer + i).load()"),
            {},
            "a load in its lane loop names no tensor",
        ),
        (
            _WHOLE_GROUP.replace("c = b + 1", "c = helper(pointer)"),
            {},
            "a statement of unknown memory effect",
        ),
        (
            _WHOLE_GROUP.replace("c = b + 1", "cute.arch.sync_threads()"),
            {},
            "a statement of unknown memory effect",
        ),
        (_WHOLE_GROUP, {"packets": ("y",)}, "a packet of y is loaded above"),
        (_WHOLE_GROUP, {"packets": ("out",)}, "a packet of out is loaded above"),
    ],
)
def test_group_refusal_names_the_unproven_condition(
    body: str, options: dict[str, object], reason: str
) -> None:
    refusal = _refusal(body, **options)  # type: ignore[arg-type]
    assert refusal is not None and reason in refusal


def test_input_sharing_the_output_storage_restores_scalar_stores(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # x is the second component of every pair of out: not proven disjoint.
    out = torch.empty(4096, 2)
    x = out.view(-1)[1::2]
    with caplog.at_level(logging.DEBUG, logger=_MEMORY_OPS_LOG):
        code = _code(caller_pairs, (x, out), _config())
    assert _stores(code) == (0, 2)
    assert "accessed in its lane loop" in caplog.text


def test_outputs_sharing_storage_restore_scalar_stores(
    caplog: pytest.LogCaptureFixture,
) -> None:
    x = torch.empty(4096)
    distinct = (x, torch.empty(4096, 2), torch.empty(4096, 2))
    assert _stores(_code(caller_two_groups, distinct, _config())) == (2, 0)
    # One tensor passed for both outputs: the second group's stores must
    # land after the first's, element by element.
    out = torch.empty(4096, 2)
    with caplog.at_level(logging.DEBUG, logger=_MEMORY_OPS_LOG):
        code = _code(caller_two_groups, (x, out, out), _config())
    assert _stores(code) == (0, 4)
    assert "accessed in its lane loop" in caplog.text


def test_vloop_sink_keeps_the_scalar_stores_of_a_reduced_body() -> None:
    # The sink distributes the V-loop over the device loop; the siblings,
    # which stay scalar beside a reduction, remain one lane's stores.
    config = helion.Config.from_dict(
        {**_COLUMN_SUM_CONFIG.config, "cute_vloop_sink": True}
    )
    code = _code(column_sum_pairs, (torch.empty(64, 1024),), config)
    assert "_vsink_vec" in code
    assert _stores(code) == (0, 2)


def test_vloop_sink_keeps_the_sibling_packet_of_an_inner_axis() -> None:
    # The siblings' list is created, filled and flushed once per lane of the
    # first axis's vector loop, which ``cute_vloop_sink`` would move inside
    # the second axis's lanes for the load of ``xt``: the list created once,
    # every lane's values appended to it and the flush repeated per lane.
    # The sink pass refuses a body that creates a trace-time list, so the
    # knob-off code runs.
    args = (torch.empty(256, 64), torch.empty(64, 256, 2))
    for layout in _BOTH_AXES_CONFIGS:
        on, off = (
            _code(
                pairs_of_transposed, args, helion.Config(**layout, cute_vloop_sink=sink)
            )
            for sink in (True, False)
        )
        assert on == off
        assert "_vsink" not in on
        assert _stores(on) == (1, 0)
        assert len(_appends_per_lane(on, "_tile_store_vals_1_0")) == 2


def test_two_groups_and_nested_lane_loops_pack() -> None:
    assert _stores(_code(two_groups, (torch.empty(4096),), _config())) == (2, 0)
    for kernel, config in ((row_pairs, _ROW_CONFIG), (grid_pairs, _GRID_CONFIG)):
        code = _code(kernel, (torch.empty(8, 1024),), config)
        assert _stores(code) == (1, 0)
        # The row coordinate stays in the pointer; the lane base replaces
        # the column, and the component dim is left out.
        assert re.search(
            r"_cute_store_u32_vec\(out\.iterator \+ cutlass\.Int32\(indices_0\) \* "
            r"cutlass\.Int32\(out\.layout\.stride\[0\]\) \+ "
            r"cutlass\.Int32\(lane_base_1\) \* cutlass\.Int32\(out\.layout\.stride\[1\]\), ",
            code,
        )


def test_lane_unroll_loads_first_keeps_the_packet() -> None:
    code = _code(pairs, (torch.empty(4096),), _config(256, 16, 4, cute_lane_unroll=2))
    assert "_lane_unroll_loads_0" in code
    assert _stores(code) == (1, 0)


def test_lane_invariant_store_follows_the_flush() -> None:
    # Every thread zeroes the tile's first row after its own flush, the
    # row's owner included, so the pair needs no barrier.
    code = _code(
        pairs_then_first_row, (torch.empty(4096), torch.empty(4096, 2)), _config()
    )
    assert _stores(code) == (1, 1)
    body = ast.unparse(_device_function(code))
    assert body.index("_cute_store_u32_vec(") < body.index(".store(")


def test_read_after_the_lane_loop_waits_for_the_flush() -> None:
    # Every thread reads the row's first pair, which one thread's flush
    # stores: the barrier pass sees the flush as a store of ``out``.
    code = _code(row_pairs_then_first, (torch.empty(8, 1024),), _ROW_CONFIG)
    body = ast.unparse(_device_function(code))
    flush = body.index("_cute_store_u32_vec(out.iterator")
    assert flush < body.index("cute.arch.sync_threads()") < body.index(".load()")


@pytest.mark.parametrize(
    ("kernel", "args", "config"),
    [
        (read_between, (torch.empty(4096),), _config()),
        (row_read_between, (torch.empty(8, 1024),), _ROW_CONFIG),
    ],
    ids=["grid", "device-loop"],
)
def test_read_between_siblings_keeps_scalar_stores(
    kernel: object, args: tuple[torch.Tensor, ...], config: helion.Config
) -> None:
    # The read-after-write barrier before the load keeps the grid's body
    # scalar and stands between the siblings of the device loop's group.
    code = _code(kernel, args, config)
    assert _stores(code) == (0, 2)
    body = ast.unparse(_device_function(code))
    first, second = (match.start() for match in re.finditer(r"\.store\(", body))
    reload = body.index("back = (out.iterator")
    assert first < body.index("cute.arch.sync_threads()") < reload < second


@pytest.mark.parametrize(
    ("width", "flushes"),
    [(2, ["_cute_store_u32x4_streaming"]), (4, ["_cute_store_u32x4_streaming"] * 2)],
    ids=["16B", "32B-split"],
)
def test_streaming_policy_takes_the_component_packet(
    width: int, flushes: list[str]
) -> None:
    code = _code(
        pairs, (torch.empty(4096),), _config(width=width, cute_store_policy="streaming")
    )
    calls = [
        node.func.id
        for node in ast.walk(_device_function(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_cute_store_")
    ]
    assert calls == flushes
    if width == 4:
        # Two 16-byte stores: lanes 0-3 at the base, 4-7 four elements on.
        assert "_tile_store_vals_0_0[0:4])" in code
        assert "+ 4, _tile_store_vals_0_0[4:8])" in code


def _policy(source: str) -> str:
    body = ast.parse(source).body
    lower_packet_store_policy(
        body,
        policy="streaming",
        global_tensor_names={"out"},
        target_device_capability=(10, 0),
        cuda_version=None,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_store_policy_counts_every_append_of_a_collector() -> None:
    # Two appends per iteration of a 2-lane loop, directly or in a nested
    # constexpr loop over the components: 4 fp32 lanes, one 16-byte flush.
    siblings = """
vals = []
for lane in cutlass.range_constexpr(2):
    vals.append(a)
    vals.append(b)
_cute_store_u32_vec(out.iterator + base, vals)
"""
    nested = """
vals = []
for lane in cutlass.range_constexpr(2):
    for component in cutlass.range_constexpr(2):
        vals.append(a)
_cute_store_u32_vec(out.iterator + base, vals)
"""
    for source in (siblings, nested):
        assert "_cute_store_u32x4_streaming(out.iterator + base, vals)" in _policy(
            source
        )
    # A list the loop also reads has no known width: the store stays plain.
    read = siblings.replace("vals.append(b)", "vals.append(vals[0])")
    assert "_cute_store_u32_vec(out.iterator + base, vals)" in _policy(read)


def _scatter_args(
    elements: int = 4096, components: int = 2, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, int]:
    return (
        torch.empty(elements, components, dtype=dtype),
        torch.empty(elements, dtype=dtype),
        0,
    )


def _component_loops(tree: ast.AST | str) -> list[ast.For]:
    """The trace-time component loops of the device function: a tile's lane
    loop whose index is the literal lane."""
    if isinstance(tree, str):
        tree = _device_function(tree)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and ast.unparse(node.iter).startswith("cutlass.range_constexpr")
        and re.fullmatch(
            rf"indices_\d+ = cutlass\.Int32\({ast.unparse(node.target)}\)",
            ast.unparse(node.body[0]),
        )
    ]


def test_covering_tile_becomes_a_component_loop() -> None:
    code = _code(scatter_component, _scatter_args(), _lane_config())
    assert _stores(code) == (1, 0)
    assert (
        "_cute_store_u32_vec(out.iterator + cutlass.Int32(lane_base_0) * "
        "cutlass.Int32(out.layout.stride[0]), _tile_store_vals_0_0)"
    ) in code
    # The component tile's loop runs at trace time inside the V-loop, its
    # index the literal lane: the tile has one tile, at zero.
    (loop,) = _component_loops(code)
    assert ast.unparse(loop.iter) == "cutlass.range_constexpr(2)"
    body = [ast.unparse(statement) for statement in loop.body]
    assert body[0] == "indices_1 = cutlass.Int32(lane_1)"
    assert "tile_offset_1" not in code
    # One append per component, in the loop; the element's operand is one
    # packet above the V-loop and the component's a scalar load per lane.
    assert sum(line.startswith("_tile_store_vals_0_0.append(") for line in body) == 1
    assert "cute.arch.load(grad.iterator + cutlass.Int32(lane_base_0)" in code
    assert sum(".load()" in line for line in body) == 1


@pytest.mark.parametrize(
    ("args", "config", "capability", "packet_lanes"),
    [
        (_scatter_args(), _lane_config(widths=(2, 1)), (10, 0), 4),
        (_scatter_args(), _lane_config(widths=(2, 1)), (9, 0), 4),
        (
            _scatter_args(),
            _lane_config(cute_lane_layouts=["strided", "blocked"]),
            (10, 0),
            8,
        ),
        (
            _scatter_args(components=4),
            _lane_config((256, 4), widths=(2, 1)),
            (10, 0),
            8,
        ),
        (
            _scatter_args(dtype=torch.bfloat16),
            _lane_config((512, 2), widths=(8, 1)),
            (10, 0),
            16,
        ),
        # 1000 elements in tiles of 256: the flush is under the tile mask.
        (_scatter_args(1000), _lane_config(), (10, 0), 8),
    ],
    ids=["16B", "sm90-16B", "strided", "four-components", "bf16", "masked-tail"],
)
def test_component_loop_packets(
    args: tuple[torch.Tensor, torch.Tensor, int],
    config: helion.Config,
    capability: tuple[int, int],
    packet_lanes: int,
) -> None:
    code = _code(scatter_component, args, config, capability)
    assert _stores(code) == (1, 0)
    function = _device_function(code)
    (loop,) = _component_loops(function)
    vloop = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For) and any(loop is child for child in node.body)
    )
    lanes = ast.literal_eval(vloop.iter.args[0]) * ast.literal_eval(loop.iter.args[0])  # type: ignore[attr-defined]
    assert lanes == packet_lanes
    if args[0].shape[0] == 1000:
        assert "if mask_0:\n            _cute_store_u32_vec(" in code


# The component loop needs the trailing tile whole on the lane's own thread,
# innermost, without a vector loop of its own, and a packet the device
# stores at once; anything else keeps today's code.
@pytest.mark.parametrize(
    ("args", "config", "stores"),
    [
        # The components on thread axes.
        (_scatter_args(), _lane_config(threads=(64, 0)), (0, 1)),
        (_scatter_args(), _lane_config(threads=(64, 2)), (0, 1)),
        # The component loop outside the element lanes.
        (_scatter_args(), _lane_config(order=(1, 0)), (0, 1)),
        # No vector loop over the elements.
        (_scatter_args(), _lane_config(widths=(1, 1)), (0, 1)),
        # The components as a vector loop of their own: a packet per element.
        (_scatter_args(), _lane_config(widths=(4, 2)), (1, 0)),
        # A tile of one component: two tiles along the dim.
        (_scatter_args(), _lane_config(block=(256, 1)), (0, 1)),
        # 1002 elements: a chunk of 4 could straddle the extent.
        (_scatter_args(1002), _lane_config(), (0, 1)),
        # 4 lanes of 4 fp32 components are 64 bytes.
        (_scatter_args(components=4), _lane_config((256, 4)), (0, 1)),
    ],
    ids=[
        "component-threads",
        "two-component-threads",
        "component-loop-outside",
        "scalar-elements",
        "component-vector-loop",
        "partial-component-tile",
        "extent-not-multiple",
        "wider-than-a-store",
    ],
)
def test_other_layouts_keep_their_code(
    args: tuple[torch.Tensor, torch.Tensor, int],
    config: helion.Config,
    stores: tuple[int, int],
) -> None:
    code = _code(scatter_component, args, config)
    assert _stores(code) == stores
    assert _component_loops(code) == []
    assert "indices_1 = cutlass.Int32(lane_1)" not in code


def test_covering_tile_keeps_scalar_stores_on_unproven_layouts() -> None:
    # The layout proof of the sibling stores holds for a covering tile too.
    x = torch.empty(4096, 2)
    code = _code(caller_components, (x, torch.empty(4096, 2)), _lane_config())
    assert _stores(code) == (1, 0)
    assert len(_component_loops(code)) == 1
    # Pairs 4 elements apart (the first columns of rows of four, every other
    # pair of a buffer) and a transposed buffer, whose components are 4096
    # elements apart.
    for out in (
        torch.empty(4096, 4)[:, :2],
        torch.empty(8192, 2)[::2],
        torch.empty(2, 4096).t(),
    ):
        code = _code(caller_components, (x, out), _lane_config())
        assert _stores(code) == (0, 1)
        assert _component_loops(code) == []


def test_value_without_a_component_coordinate_restores_the_scalar_store(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Nothing the store appends depends on the component: the lane-loop
    # distribution would place the append outside the component loop, once
    # per element.
    args = (torch.empty(4096), torch.empty(4096, 2))
    with caplog.at_level(logging.DEBUG, logger=_MEMORY_OPS_LOG):
        code = _code(broadcast_components, args, _lane_config())
    assert _stores(code) == (0, 1)
    assert (
        "a store of every component does not run once per component, in order"
        in caplog.text
    )


@pytest.mark.parametrize(
    ("kernel", "args", "loops"),
    [
        (weigh_components_in_place, (torch.empty(1000, 2), torch.empty(1000)), 1),
        (normalize_components, (torch.empty(1000, 2),), 2),
    ],
    ids=["in-place", "reduction"],
)
def test_refused_group_keeps_the_component_loop(
    kernel: object, args: tuple[torch.Tensor, ...], loops: int
) -> None:
    # The tile becomes a component loop when its store is lowered, with the
    # packet proved; the group is refused once the lane body is complete
    # (the body reads the tensor, or reduces).  The store returns to its
    # scalar form and the loop stays a trace-time loop, its index the lane:
    # the tile has one tile, at zero, so that is the code of the lane loop.
    code = _code(kernel, args, _lane_config())
    assert _stores(code) == (0, 1)
    component_loops = _component_loops(code)
    assert len(component_loops) == loops
    assert ".store(" in ast.unparse(component_loops[-1])
    assert "tile_offset_1" not in code and "for lane_1 in range(" not in code
    # The lane's vector loop still ends at its own body: a per-element
    # operand loads as one packet above it.
    assert ("cute.arch.load(weight.iterator" in code) == (len(args) == 2)


def test_reverse_scan_over_the_components_keeps_scalar_stores(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A scan over the component tile carries its value through the
    # component loop.  Forward, the loop walks the components upwards and
    # the store appends them in memory order: one packet.
    args = (torch.empty(1000, 2),)
    code = _code(cumsum_components, args, _lane_config())
    assert _stores(code) == (1, 0)
    (loop,) = _component_loops(code)
    assert ast.unparse(loop.iter) == "cutlass.range_constexpr(2)"
    # A reverse scan runs the loop from the last component down, so its
    # appends would fill the list against the order the flush stores it
    # (every pair swapped).  The group is refused and the scalar stores run
    # in the reversed loop.
    with caplog.at_level(logging.DEBUG, logger=_MEMORY_OPS_LOG):
        code = _code(reverse_cumsum_components, args, _lane_config())
    assert _stores(code) == (0, 1)
    (loop,) = _component_loops(code)
    assert ast.unparse(loop.iter) == "cutlass.range_constexpr(1, -1, -1)"
    assert ".store(" in ast.unparse(loop)
    assert (
        "a store of every component does not run once per component, in order"
        in caplog.text
    )


def test_store_of_one_element_stays_outside_the_component_loop() -> None:
    code = _code(components_and_first, (torch.empty(4096, 2),), _lane_config())
    assert _stores(code) == (2, 0)
    (loop,) = _component_loops(code)
    inside = ast.unparse(loop)
    assert "_tile_store_vals_0_0.append(" in inside
    assert "_tile_store_vals_0_1.append(" not in inside
    assert "_cute_store_u32_vec(first.iterator" in code


def test_lane_packet_reading_a_component_coordinate_stays_scalar() -> None:
    # ``planes[tile_c, tile_e]`` is contiguous along the element lanes, but
    # its row is the component coordinate, defined inside the V-loop: a
    # packet above the V-loop could not address it, whether the load was
    # lowered before the tile became a component loop or after.
    code = _code(planes_to_pairs, (torch.empty(2, 4096),), _lane_config())
    assert _stores(code) == (1, 0)
    assert "cute.arch.load(planes.iterator" not in code
    (loop,) = _component_loops(code)
    assert "planes.iterator" in ast.unparse(loop)
    args = (torch.empty(2, 4096), torch.empty(4096, 2))
    code = _code(pairs_and_planes, args, _lane_config())
    # Both stores keep their packets: the second reuses the component loop.
    assert _stores(code) == (2, 0)
    assert "cute.arch.load(planes.iterator" not in code
    assert len(_component_loops(code)) == 1


def test_vloop_sink_keeps_the_component_loop_inside_the_vector_loop() -> None:
    # ``planes[tile_c, tile_e]`` is contiguous along the lanes, the load
    # ``cute_vloop_sink`` moves a vector loop inside a loop for: here the
    # component loop, where the appends would run component by component
    # under a flush that stores the list element by element.  The sink pass
    # refuses an append inside a loop it would move the vector loop into, so
    # the knob-off code runs.
    args = (torch.empty(2, 4096),)
    for widths in ((2, 1), (4, 1)):
        on, off = (
            _code(
                planes_to_pairs,
                args,
                _lane_config(widths=widths, cute_vloop_sink=sink),
            )
            for sink in (True, False)
        )
        assert on == off
        assert "_vsink" not in on
        assert _stores(on) == (1, 0)
        function = _device_function(on)
        (loop,) = _component_loops(function)
        assert any(
            ast.unparse(statement).startswith("_tile_store_vals_0_0.append(")
            for statement in loop.body
        )
        (vloop,) = (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For) and ast.unparse(node.target) == "vec_lane_0"
        )
        assert any(loop is child for child in vloop.body)


def test_a_gather_respelled_for_the_sink_keeps_its_slot_tile() -> None:
    # A two-slot window, ``out[n, k] = flat[n + channels * (k - pad)]``: the
    # slots are the two components of an ``[n, 2]`` output, and the gather is
    # contiguous along the lanes.  Two nests fit, and ``cute_vloop_sink``
    # picks.  Without it the slot tile is the component loop inside the
    # vector loop: one packet store, an element load per lane and slot.
    args = (torch.empty(2, 32, 16), 2, 1)
    packet = _code(sliding_windows, args, _lane_config())
    assert _stores(packet) == (1, 0)
    assert len(_component_loops(packet)) == 1
    assert "_cute_load_v16b_if(" not in packet
    assert packet.count(".load()") == 1
    # With it the knob respells the gather for the sink, and the slot tile
    # stays the lane loop the vector loop sinks into: one predicated packet
    # load per slot, and the scalar stores ``cute_staged_store`` flushes.
    sunk = _code(
        sliding_windows,
        args,
        _lane_config(cute_vloop_sink=True, cute_lane_unroll=16),
    )
    assert sunk.count("_cute_load_v16b_if(flat.iterator") == 2
    assert ".load()" not in sunk
    assert not _component_loops(sunk)
    assert _stores(sunk) == (0, 2)


def test_a_gather_the_sink_cannot_widen_joins_the_component_packet() -> None:
    # The gathered view starts one element into its storage, so no packet
    # follows it: the knob respells nothing, and the slots join the
    # component packet as without the knob.
    buffer = torch.empty(1040)
    args = (buffer[1:1025], torch.empty(1024, 2), 32, 16, 1)
    on, off = (
        _code(sliding_windows_into, args, _lane_config(cute_vloop_sink=sink))
        for sink in (True, False)
    )
    assert on == off
    assert _stores(on) == (1, 0)
    assert len(_component_loops(on)) == 1


def test_component_loops_of_a_three_dim_tile() -> None:
    # Rows on a thread axis, lanes along the columns, components innermost.
    rows = helion.Config(
        block_sizes=[2, 256, 2],
        num_threads=[2, 64, 1],
        cute_vector_widths=[1, 4, 1],
        loop_orders=[[0, 1, 2]],
    )
    code = _code(cube_components, (torch.empty(8, 1024, 2),), rows)
    assert _stores(code) == (1, 0)
    assert len(_component_loops(code)) == 1
    # Two component dims: four components per element, row-major.
    both = helion.Config(
        block_sizes=[256, 2, 2],
        num_threads=[64, 1, 1],
        cute_vector_widths=[2, 1, 1],
        loop_orders=[[0, 1, 2]],
    )
    code = _code(cube_components, (torch.empty(1024, 2, 2),), both)
    assert _stores(code) == (1, 0)
    outer, inner = _component_loops(code)
    assert ast.unparse(outer.target) == "lane_1"
    assert any(inner is child for child in outer.body)
    assert ast.unparse(inner.target) == "lane_2"


def test_group_refusal_checks_the_component_loops() -> None:
    # A store of every component must run inside every component loop.
    spanning = ast.parse("a = 1\nvals.append(a)").body
    placeholder = ast.parse("pass").body[0]

    def refusal(scope: tuple[list[ast.stmt], list[ast.stmt]]) -> str | None:
        group = CuteComponentGroup(
            2, "pointer", None, [CuteComponentStore(0, 2, spanning[1], placeholder)]
        )
        site = CuteTileVecStoreSite(
            "vals", "out", spanning[1], placeholder, None, placeholder, group
        )
        context = LaneLoopContext(block_id=0)
        context.vec_stores.append(site)
        return component_group_refusal(site, context, spanning, {"out"}, set(), scope)

    assert refusal((spanning, spanning)) is None
    for scope in (([], []), ([], spanning)):
        reason = refusal(scope)
        assert reason is not None and "once per component, in order" in reason
    # A store of one component must run inside none.
    statements = ast.parse(_WHOLE_GROUP).body
    group = CuteComponentGroup(
        2,
        "pointer",
        None,
        [
            CuteComponentStore(0, 1, statements[1], placeholder),
            CuteComponentStore(1, 1, statements[3], placeholder),
        ],
    )
    site = CuteTileVecStoreSite(
        "vals", "out", statements[1], placeholder, None, placeholder, group
    )
    context = LaneLoopContext(block_id=0)
    context.vec_stores.append(site)
    names, disjoint = {"out", "x"}, {frozenset(("out", "x"))}
    assert component_group_refusal(site, context, statements, names, disjoint) is None
    reason = component_group_refusal(
        site, context, statements, names, disjoint, ([], [statements[3]])
    )
    assert reason is not None and "repeats in a component loop" in reason


def test_plain_store_in_a_component_loop_returns_to_its_scalar_form() -> None:
    # A store collected once per lane whose append ended up inside a
    # component loop would append once per component: it gets its scalar
    # store back, and its list and its flush leave the lane body.
    def demote(in_component_loop: bool) -> tuple[list[ast.stmt], list[ast.stmt]]:
        body = ast.parse("vals.append(a)\nb = a + 1").body
        scalar = ast.parse("(first.iterator + i).store(a)").body[0]
        created = ast.parse("vals = []").body[0]
        flush = ast.parse("flush(first.iterator + base, vals)").body[0]
        site = CuteTileVecStoreSite("vals", "first", body[0], scalar, created, flush)
        context = LaneLoopContext(block_id=0)
        context.lane_body = [created, flush]
        context.vec_stores.append(site)
        function = SimpleNamespace(arguments=[], proven_disjoint_tensor_pairs=set)
        demoted = demote_reordered_tile_vec_stores(
            context,
            cast("Any", function),
            body,
            component_scope=([], [body[0]] if in_component_loop else []),
        )
        assert demoted == ([site] if in_component_loop else [])
        assert context.vec_stores == ([] if in_component_loop else [site])
        return body, cast("list[ast.stmt]", context.lane_body)

    body, lane_body = demote(True)
    assert ast.unparse(body[0]) == "(first.iterator + i).store(a)"
    assert lane_body == []
    body, lane_body = demote(False)
    assert ast.unparse(body[0]) == "vals.append(a)"
    assert [ast.unparse(statement) for statement in lane_body] == [
        "vals = []",
        "flush(first.iterator + base, vals)",
    ]


def _pairs_of(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.stack([first, second], dim=-1)


def _device_stores(packet_bytes: int) -> bool:
    """Whether the GPU under test stores a component packet of
    ``packet_bytes`` at once (0: no packet forms on any device).

    The lowering packs up to the device's widest store, 32 bytes from sm_100
    and 16 before, so the 32-byte cases of the CUDA tests keep their scalar
    stores on an older GPU (``sm90-32-bytes`` above) and are still compared
    with eager there.
    """
    capability = torch.cuda.get_device_capability(DEVICE)
    return 0 < packet_bytes <= target_facts.widest_packet_bytes(capability)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pairs_match_eager() -> None:
    # Packets (16 and 32 bytes, a strided layout, a masked tail tile, a
    # loads-first lane unroll, streaming stores) and the scalar fallbacks (an
    # extent no multiple of the width, a group wider than a store), each
    # with its packet's bytes.
    cases: list[tuple[int, helion.Config, int]] = [
        (4096, _config(width=2), 16),
        (4096, _config(), 32),
        (4096, _config(cute_lane_layouts=["strided"]), 32),
        (1000, _config(), 32),
        (4096, _config(256, 16, 4, cute_lane_unroll=2), 32),
        (4096, _config(cute_store_policy="streaming"), 32),
        (1002, _config(), 0),
        (4096, _config(512, 64, 8), 0),
    ]
    for n, config, packet_bytes in cases:
        x = torch.randn(n, device=DEVICE)
        bound = pairs.bind((x,))
        packed = _device_stores(packet_bytes)
        assert (_stores(bound.to_code(config))[0] > 0) == packed
        out = bound.compile_config(config)(x)
        torch.testing.assert_close(
            out, _pairs_of(x, torch.zeros_like(x)), rtol=0, atol=0
        )
    x = torch.randn(4096, device=DEVICE).to(torch.bfloat16)
    out = pairs.bind((x,)).compile_config(_config(512, 32, 8))(x)
    torch.testing.assert_close(out, _pairs_of(x, torch.zeros_like(x)), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_caller_views_match_eager() -> None:
    x = torch.randn(4096, device=DEVICE)
    expected = _pairs_of(x, x * 2)
    # An aligned output (a 32-byte packet), views 8 and 16 bytes into their
    # storage (the second takes 16-byte packets), components two elements
    # apart and a transposed buffer (a stride-1 packet per sibling); NaN
    # marks a missed store.
    outputs = [
        (torch.empty(4096, 2, device=DEVICE), 4, int(_device_stores(32))),
        (torch.empty(4097, 2, device=DEVICE)[1:], 4, 0),
        (torch.empty(4098, 2, device=DEVICE)[2:], 2, 1),
        (torch.empty(4096, 4, device=DEVICE)[:, ::2], 4, 0),
        (torch.empty(2, 4096, device=DEVICE).t(), 4, 2),
    ]
    for out, width, flushes in outputs:
        out.fill_(float("nan"))
        bound = caller_pairs.bind((x, out))
        config = _config(width=width)
        assert _stores(bound.to_code(config))[0] == flushes
        result = bound.compile_config(config)(x, out)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    # The other views ``test_unproven_packet_keeps_scalar_stores`` declines:
    # pairs 4 elements apart (the first columns of rows of four, every other
    # pair of a buffer), whose neighbours must stay untouched.
    wide = torch.zeros(4096, 4, device=DEVICE)
    skipping = torch.zeros(8192, 2, device=DEVICE)
    for view, between in (
        (wide[:, :2], wide[:, 2:]),
        (skipping[::2], skipping[1::2]),
    ):
        view.fill_(float("nan"))
        bound = caller_pairs.bind((x, view))
        assert _stores(bound.to_code(_config())) == (0, 2)
        result = bound.compile_config(_config())(x, view)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert not bool(between.any())
    # Transposed trailing dims, tiles from column 2 (in chunks of 4 and of
    # 2), a shifted lane coordinate, rows 2050 elements apart, the rows as a
    # device loop inside the lanes and one flattened tile.
    quads_out = torch.full((4096, 2, 2), float("nan"), device=DEVICE).transpose(1, 2)
    result = caller_quads.bind((x, quads_out)).compile_config(_config(width=2))(
        x, quads_out
    )
    torch.testing.assert_close(
        result,
        torch.stack([x, x + 1, x + 2, x + 3], -1).view(-1, 2, 2),
        rtol=0,
        atol=0,
    )
    wider = torch.randn(8, 1026, device=DEVICE)
    for config, flushes in ((_ROW_CONFIG, 0), (_ROW_PAIR_CONFIG, 1)):
        out = torch.full((8, 1028, 2), float("nan"), device=DEVICE)[:, :1026]
        bound = row_pairs_from_two.bind((wider, out))
        assert _stores(bound.to_code(config))[0] == flushes
        result = bound.compile_config(config)(wider, out)
        torch.testing.assert_close(
            result[:, 2:], _pairs_of(wider, -wider)[:, 2:], rtol=0, atol=0
        )
        assert bool(torch.isnan(result[:, :2]).all())
    out = torch.full((4100, 2), float("nan"), device=DEVICE)
    result = shifted_pairs.bind((x, out)).compile_config(_config())(x, out)
    torch.testing.assert_close(result[4:], expected, rtol=0, atol=0)
    assert bool(torch.isnan(result[:4]).all())
    rows = torch.randn(8, 1024, device=DEVICE)
    for out, config in (
        (torch.full((8, 1025, 2), float("nan"), device=DEVICE)[:, :1024], _GRID_CONFIG),
        (torch.full((8, 1024, 2), float("nan"), device=DEVICE), _FLAT_GRID_CONFIG),
    ):
        bound = caller_grid_pairs.bind((rows, out))
        assert _stores(bound.to_code(config)) == (0, 2)
        result = bound.compile_config(config)(rows, out)
        torch.testing.assert_close(result, _pairs_of(rows, -rows), rtol=0, atol=0)
    tall = torch.randn(64, 1024, device=DEVICE)
    out = torch.full((64, 1024, 2), float("nan"), device=DEVICE)
    result = column_pairs.bind((tall, out)).compile_config(_COLUMN_CONFIG)(tall, out)
    torch.testing.assert_close(result, _pairs_of(tall, -tall), rtol=0, atol=0)
    # A strided input: scalar loads beside the packet store.
    strided = torch.randn(8192, device=DEVICE)[::2]
    out = torch.full((4096, 2), float("nan"), device=DEVICE)
    bound = caller_pairs.bind((strided, out))
    assert _stores(bound.to_code(_config())) == (
        (1, 0) if _device_stores(32) else (0, 2)
    )
    torch.testing.assert_close(
        bound.compile_config(_config())(strided, out),
        _pairs_of(strided, strided * 2),
        rtol=0,
        atol=0,
    )
    # An extent known only at run time: scalar stores, a partial last tile.
    for n in (4096, 1001):
        dynamic = torch.randn(n, device=DEVICE)
        result = dynamic_pairs.bind((dynamic,)).compile_config(_config())(dynamic)
        torch.testing.assert_close(
            result, _pairs_of(dynamic, dynamic * 2), rtol=0, atol=0
        )
    # One tensor passed for both outputs of two groups: the later stores win.
    out = torch.full((4096, 2), float("nan"), device=DEVICE)
    first, second = caller_two_groups.bind((x, out, out)).compile_config(_config())(
        x, out, out
    )
    assert first is second or first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(second, _pairs_of(x * 2, x * 3), rtol=0, atol=0)
    # The input is the second component of every output pair.
    out = torch.randn(4096, 2, device=DEVICE)
    alias = out.view(-1)[1::2]
    expected = _pairs_of(alias.clone(), alias * 2)
    bound = caller_pairs.bind((alias, out))
    assert _stores(bound.to_code(_config())) == (0, 2)
    torch.testing.assert_close(
        bound.compile_config(_config())(alias, out), expected, rtol=0, atol=0
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_groups_match_eager() -> None:
    x = torch.randn(4096, device=DEVICE)

    def run(kernel: object, args: tuple[torch.Tensor, ...], config: helion.Config):
        return kernel.bind(args).compile_config(config)(*args)  # type: ignore[attr-defined]

    def same(actual: object, expected: object) -> None:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    same(
        run(quads, (x,), _config(width=2)),
        torch.stack([x, x + 1, x + 2, x + 3], -1).view(-1, 2, 2),
    )
    same(run(repeated_component, (x,), _config()), _pairs_of(x + 1, x * 2))
    same(run(reversed_components, (x,), _config()), _pairs_of(x, x * 2))
    same(run(read_between, (x,), _config()), _pairs_of(x, x * 2))
    old = torch.randn(4096, 2, device=DEVICE)
    expected = _pairs_of(x + old[:, 1], old[:, 1] * 2)
    same(run(read_before, (x, old), _config()), expected)
    partial = torch.full((4096, 4), -1.0, device=DEVICE)
    same(
        run(triples, (x, partial), _config(width=2)),
        torch.stack([x, x * 2, x * 3, torch.full_like(x, -1.0)], -1),
    )
    for kernel in (store_between, two_groups):
        a, b = run(kernel, (x,), _config())
        same(a, _pairs_of(x, x + 1))
        same(b, _pairs_of(x * 2, x * 3))
    rows = torch.randn(8, 1024, device=DEVICE)
    same(run(row_pairs, (rows,), _ROW_CONFIG), _pairs_of(rows, -rows))
    same(run(grid_pairs, (rows,), _GRID_CONFIG), _pairs_of(rows, -rows))
    out = torch.full((4096, 2), float("nan"), device=DEVICE)
    expected = _pairs_of(x, x * 2)
    expected[::256, 1] = 0.0
    same(run(pairs_then_first_row, (x, out), _config()), expected)
    # Vector loops on both grid axes with ``cute_vloop_sink``: the knob-off
    # code runs (the sunk form repeated the flush with every lane's values).
    xt = torch.randn(256, 64, device=DEVICE)
    for layout in _BOTH_AXES_CONFIGS:
        out = torch.full((64, 256, 2), float("nan"), device=DEVICE)
        config = helion.Config(**layout, cute_vloop_sink=True)
        same(run(pairs_of_transposed, (xt, out), config), _pairs_of(xt.t(), -xt.t()))
    columns = torch.randn(64, 1024, device=DEVICE)
    total = columns.sum(0)
    for sink in (False, True):
        config = helion.Config.from_dict(
            {**_COLUMN_SUM_CONFIG.config, "cute_vloop_sink": sink}
        )
        torch.testing.assert_close(
            run(column_sum_pairs, (columns,), config),
            _pairs_of(total, total * 2),
            rtol=1e-5,
            atol=1e-4,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_component_loops_match_eager() -> None:
    def scatter(
        elements: int,
        components: int,
        component: int,
        config: helion.Config,
        packet_bytes: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        pair = torch.randn(elements, components, device=DEVICE).to(dtype)
        grad = torch.randn(elements, device=DEVICE).to(dtype)
        bound = scatter_component.bind((pair, grad, component))
        packed = _device_stores(packet_bytes)
        assert (_stores(bound.to_code(config))[0] > 0) == packed
        out = bound.compile_config(config)(pair, grad, component)
        expected = torch.select_scatter(pair, grad, 1, component)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    # Either component of a pair in 32- and 16-byte packets, blocked and
    # strided lanes, a masked tail tile, lanes unrolled loads first,
    # streaming stores, four components, bf16; then the layouts that keep
    # today's code.  The last number is the packet's bytes.
    scatter(4096, 2, 0, _lane_config(), 32)
    scatter(4096, 2, 1, _lane_config(widths=(2, 1)), 16)
    scatter(4096, 2, 1, _lane_config(cute_lane_layouts=["strided", "blocked"]), 32)
    scatter(1000, 2, 0, _lane_config(), 32)
    scatter(4096, 2, 0, _lane_config(threads=(16, 1), cute_lane_unroll=2), 32)
    scatter(4096, 2, 1, _lane_config(cute_store_policy="streaming"), 32)
    scatter(4096, 4, 2, _lane_config((256, 4), widths=(2, 1)), 32)
    scatter(4096, 2, 0, _lane_config((512, 2), widths=(8, 1)), 32, torch.bfloat16)
    scatter(1002, 2, 0, _lane_config(), 0)
    scatter(4096, 2, 1, _lane_config(order=(1, 0)), 0)
    scatter(4096, 2, 0, _lane_config(threads=(64, 2)), 0)
    scatter(4096, 4, 3, _lane_config((256, 4)), 0)

    x = torch.randn(4096, device=DEVICE)
    out = torch.full((4096, 2), float("nan"), device=DEVICE)
    bound = broadcast_components.bind((x, out))
    result = bound.compile_config(_lane_config())(x, out)
    torch.testing.assert_close(result, x[:, None].expand(-1, 2), rtol=0, atol=0)

    # A refused group on a component loop: an in-place update and a
    # reduction over the components, with a partial last tile.
    pair = torch.rand(1000, 2, device=DEVICE) + 0.5
    weight = torch.randn(1000, device=DEVICE)
    weighed = pair.clone()
    result = weigh_components_in_place.bind((weighed, weight)).compile_config(
        _lane_config()
    )(weighed, weight)
    torch.testing.assert_close(result, pair * weight[:, None], rtol=0, atol=0)
    result = normalize_components.bind((pair,)).compile_config(_lane_config())(pair)
    torch.testing.assert_close(
        result, pair / pair.sum(-1, keepdim=True), rtol=1e-6, atol=0
    )

    # A scan over the components: forward in the packet, reverse with the
    # scalar stores of its reversed component loop.
    for kernel, expected in (
        (cumsum_components, pair.cumsum(1)),
        (reverse_cumsum_components, pair.flip(1).cumsum(1).flip(1)),
    ):
        result = kernel.bind((pair,)).compile_config(_lane_config())(pair)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    pair = torch.randn(4096, 2, device=DEVICE)
    doubled, first = components_and_first.bind((pair,)).compile_config(_lane_config())(
        pair
    )
    torch.testing.assert_close(doubled, pair * 2, rtol=0, atol=0)
    torch.testing.assert_close(first, pair[:, 0] + 1, rtol=0, atol=0)

    planes = torch.randn(2, 4096, device=DEVICE)
    result = planes_to_pairs.bind((planes,)).compile_config(_lane_config())(planes)
    torch.testing.assert_close(result, planes.t(), rtol=0, atol=0)
    doubled, result = pairs_and_planes.bind((planes, pair)).compile_config(
        _lane_config()
    )(planes, pair)
    torch.testing.assert_close(doubled, pair * 2, rtol=0, atol=0)
    torch.testing.assert_close(result, planes.t(), rtol=0, atol=0)

    for shape, config in (
        (
            (8, 1024, 2),
            helion.Config(
                block_sizes=[2, 256, 2],
                num_threads=[2, 64, 1],
                cute_vector_widths=[1, 4, 1],
                loop_orders=[[0, 1, 2]],
            ),
        ),
        (
            (1024, 2, 2),
            helion.Config(
                block_sizes=[256, 2, 2],
                num_threads=[64, 1, 1],
                cute_vector_widths=[2, 1, 1],
                loop_orders=[[0, 1, 2]],
            ),
        ),
    ):
        cube = torch.randn(*shape, device=DEVICE)
        result = cube_components.bind((cube,)).compile_config(config)(cube)
        torch.testing.assert_close(result, cube + 1, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_component_loops_on_views_and_under_the_sink_match_eager() -> None:
    # ``cute_vloop_sink`` with a load contiguous along the lanes: the
    # knob-off packet runs (the sunk form stored each chunk's values
    # component by component).
    planes = torch.randn(2, 4096, device=DEVICE)
    bound = planes_to_pairs.bind((planes,))
    for widths in ((2, 1), (4, 1)):
        config = _lane_config(widths=widths, cute_vloop_sink=True)
        packed = _device_stores(widths[0] * 2 * planes.element_size())
        assert (_stores(bound.to_code(config)) == (1, 0)) == packed
        result = bound.compile_config(config)(planes)
        torch.testing.assert_close(result, planes.t(), rtol=0, atol=0)
    # A tensor of the caller's: aligned (a packet), then the views the
    # packet declines, whose neighbouring elements must stay untouched.  NaN
    # marks a missed store.
    x = torch.randn(4096, 2, device=DEVICE)
    wide = torch.zeros(4096, 4, device=DEVICE)
    skipping = torch.zeros(8192, 2, device=DEVICE)
    for out, between, flushes in (
        (torch.empty(4096, 2, device=DEVICE), None, int(_device_stores(32))),
        (wide[:, :2], wide[:, 2:], 0),
        (skipping[::2], skipping[1::2], 0),
        (torch.empty(2, 4096, device=DEVICE).t(), None, 0),
    ):
        out.fill_(float("nan"))
        bound = caller_components.bind((x, out))
        assert _stores(bound.to_code(_lane_config()))[0] == flushes
        result = bound.compile_config(_lane_config())(x, out)
        torch.testing.assert_close(result, x * 2, rtol=0, atol=0)
        assert between is None or not bool(between.any())
