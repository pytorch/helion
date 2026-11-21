"""Which elements of each ``matmul2d`` cooperative tensor a lane holds.

MPP's register operands (``mpp_registers``) run a matmul on
``mpp::tensor_ops::matmul2d`` at ``execution_simdgroup`` scope with each
computed operand a cooperative tensor every lane fills in registers,
element by element.  Which elements a lane holds is implementation defined
("the layout of cooperative_tensor depends on operation, data type, number
of threads in opscope", MPPTensorOpsMatMul2d.h): an M4 lays each operand
out as the 8x8 blocks of ``simdgroup_matrix``, two adjacent elements of
each per lane, while the M5's neural accelerators use another layout
(MLX's M5 GEMM fragments, rows ``{fm, fm + 8}`` by columns ``fn .. fn + 3``
per lane).

Helion writes a kernel's MSL on the GPU that runs it, so the first kernel
that needs a layout reads it there (:func:`probe`): a tiny kernel writes
``get_multidimensional_index`` of every element of every lane, once per
process for each op (descriptor, element types and which inputs are
device tensors).  Where no GPU runs the kernel (codegen bound on CPU
tensors), the M4's layout stands in (:func:`blocked_layout`), so the
generated code is deterministic there.

Each operand's coordinates are split into a per-lane base and a
per-element offset where that is exact (:func:`factor`): an element's
position is then its lane's base, a name the fold (``affine_fold``) knows
the range and the multiple of, plus a constant, and its index arithmetic
folds as a fragment element's does over ``_fm`` and ``_fn``.  A base that
is a sum of the lane index's bits is computed from them; another is read
from a table.  An operand that doesn't factor reads each element's
coordinates from ``get_multidimensional_index`` at run time: correct, but
its index arithmetic can't fold.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from ._constants import SIMD_WIDTH
from .affine_fold import Fact

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The three cooperative tensors of an op, in ``get_*_cooperative_tensor``
#: order.  Each element's position is ``(row, col)`` in the op's
#: coordinates: the left input's ``(m, k)``, the right input's ``(k, n)`` and
#: the destination's ``(m, n)``.
OPERANDS = ("left", "right", "dest")


@dataclasses.dataclass(frozen=True)
class Atom:
    """One ``matmul2d`` op at SIMD-group scope: its M x N x K, the MSL
    element types of its left and right inputs and its destination, and,
    for each input, whether it is a device tensor (a view the op reads)
    rather than a cooperative tensor, and the descriptor's transpose flag.
    A cooperative input is filled element by element, so its flag decides
    only the order of ``get_multidimensional_index``'s coordinates
    (innermost dimension first) and the layout the implementation picks:
    the left input is M x K and the right N x K (transposed), as the
    hand-written quantized matmuls give them; a device input's flag is its
    view's."""

    m: int
    n: int
    k: int
    left: str
    right: str
    dest: str
    left_device: bool = False
    right_device: bool = False
    transpose_left: bool = False
    transpose_right: bool = True

    def descriptor(self) -> str:
        """The op's ``matmul2d_descriptor``: multiply-accumulate, without
        relaxed precision."""
        flags = ", ".join(
            str(f).lower() for f in (self.transpose_left, self.transpose_right, False)
        )
        return (
            f"matmul2d_descriptor({self.m}, {self.n}, {self.k}, {flags}, "
            "matmul2d_descriptor::mode::multiply_accumulate)"
        )

    def op_type(self) -> str:
        return f"matmul2d<{self.descriptor()}, execution_simdgroup>"

    def device_tensor(self, operand: str) -> str:
        """The MSL type of a device input, as the destination's type names
        it: its element type over static extents, the innermost first."""
        if operand == "left":
            dtype = self.left
            extents = (self.m, self.k) if self.transpose_left else (self.k, self.m)
        else:
            dtype = self.right
            extents = (self.k, self.n) if self.transpose_right else (self.n, self.k)
        return (
            f"tensor<device {dtype}, extents<int32_t, {extents[0]}, {extents[1]}>, "
            "tensor_inline>"
        )

    def tensor(self, op: str, operand: str) -> str:
        """The expression that makes ``operand``'s cooperative tensor of the
        op object ``op``."""
        types = f"{self.left}, {self.right}, {self.dest}"
        if operand != "dest":
            return f"{op}.get_{operand}_input_cooperative_tensor<{types}>()"
        inputs = [
            self.device_tensor(side)
            if device
            else (
                f"metal::remove_addrspace_t<decltype({op}.get_{side}_input_"
                f"cooperative_tensor<{types}>())>"
            )
            for side, device in (
                ("left", self.left_device),
                ("right", self.right_device),
            )
        ]
        return (
            f"{op}.get_destination_cooperative_tensor<{inputs[0]}, {inputs[1]}, "
            f"{self.dest}>()"
        )


#: An operand's elements: ``coords[lane][element] = (row, col)``.
Coords = tuple[tuple[tuple[int, int], ...], ...]


@dataclasses.dataclass(frozen=True)
class Layout:
    """The elements each lane holds of an atom's three cooperative tensors,
    by operand (:data:`OPERANDS`)."""

    left: Coords
    right: Coords
    dest: Coords

    def of(self, operand: str) -> Coords:
        return getattr(self, operand)


def blocked_layout(atom: Atom) -> Layout:
    """The layout of an M4 (Apple GPU family 9), as probed: every operand,
    ``rows x cols``, is a grid of 8x8 ``simdgroup_matrix`` blocks in
    row-major order, and lane ``l`` holds elements ``(fm, fn)`` and ``(fm,
    fn + 1)`` of each, ``fm = ((l / 4) & 4) + ((l / 2) % 4)`` and ``fn = ((l
    / 4) & 2) * 2 + (l % 2) * 2``."""

    def operand(rows: int, cols: int) -> Coords:
        lanes = []
        for lane in range(SIMD_WIDTH):
            fm = ((lane // 4) & 4) + ((lane // 2) % 4)
            fn = ((lane // 4) & 2) * 2 + (lane % 2) * 2
            lanes.append(
                tuple(
                    (8 * r + fm, 8 * c + fn + j)
                    for r in range(rows // 8)
                    for c in range(cols // 8)
                    for j in range(2)
                )
            )
        return tuple(lanes)

    def device(present: bool, coords: Coords) -> Coords:
        return coords if present else tuple(() for _ in range(SIMD_WIDTH))

    return Layout(
        device(not atom.left_device, operand(atom.m, atom.k)),
        device(not atom.right_device, operand(atom.k, atom.n)),
        operand(atom.m, atom.n),
    )


_PROBE = r"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
template <typename T>
void _helion_coop_write(thread T& t, device int* o, ushort slots) {{
    o[0] = t.get_capacity();
    for (ushort i = 0; i < slots; i++) {{
        int a0 = -1, a1 = -1;
        if (i < t.get_capacity() && t.is_valid_element(i)) {{
            auto x = t.get_multidimensional_index(i); a0 = x[0]; a1 = x[1];
        }}
        o[1 + 2 * i] = a0; o[2 + 2 * i] = a1;
    }}
}}
kernel void _helion_coop_layout(
    device int* out [[buffer(0)]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    constexpr auto desc = {descriptor};
    matmul2d<desc, execution_simdgroup> op;
    device int* o = out + lane * {stride};
{writes}
}}
"""


def op_position(atom: Atom, operand: str, index: tuple[int, int]) -> tuple[int, int]:
    """An element's ``(row, col)`` from ``get_multidimensional_index``, which
    lists the operand's innermost dimension first: an untransposed left
    input's K (M x K), a transposed right input's K (N x K) and the
    destination's N."""
    i0, i1 = index
    if operand == "left":
        return (i0, i1) if atom.transpose_left else (i1, i0)
    if operand == "right":
        return (i0, i1) if atom.transpose_right else (i1, i0)
    return (i1, i0)


#: The coordinates of an element that isn't valid on its lane.
INVALID = (-1, -1)


@functools.cache
def probe(atom: Atom) -> Layout:
    """The layout of ``atom`` on this process's Metal GPU, read by running
    a kernel that writes it; an element that isn't valid on its lane is at
    :data:`INVALID` (its operand then doesn't factor).  A device input has
    no elements: its coordinates are empty."""
    slots = max(atom.m * atom.k, atom.k * atom.n, atom.m * atom.n) // SIMD_WIDTH * 2
    block = 1 + 2 * slots
    stride = 3 * block
    op = "op"
    writes = []
    for o, operand in enumerate(OPERANDS):
        if (atom.left_device, atom.right_device, False)[o]:
            continue
        writes += [
            f"    auto t{o} = {atom.tensor(op, operand)};",
            f"    _helion_coop_write(t{o}, o + {o * block}, {slots});",
        ]
    source = _PROBE.format(
        descriptor=atom.descriptor(), stride=stride, writes="\n".join(writes)
    )
    # Codegen may ask under its fake tensor mode; the probe runs for real.
    with unset_fake_temporarily():
        lib = torch.mps.compile_shader(source)  # type: ignore[attr-defined]
        out = torch.full((SIMD_WIDTH * stride,), -1, dtype=torch.int32, device="mps")
        lib._helion_coop_layout(out, threads=SIMD_WIDTH, group_size=SIMD_WIDTH)
        rows = out.cpu().view(SIMD_WIDTH, stride).tolist()
    operands: list[Coords] = []
    for o, operand in enumerate(OPERANDS):
        lanes = []
        for row in rows:
            base = o * block
            capacity = max(row[base], 0)
            # Every lane holds as many elements (capacity is the tensor's
            # storage, one per lane).
            assert capacity <= slots and capacity == max(rows[0][base], 0), (
                atom,
                operand,
            )
            index = [
                (row[base + 1 + 2 * i], row[base + 2 + 2 * i]) for i in range(capacity)
            ]
            lanes.append(
                tuple(
                    INVALID if p[0] < 0 else op_position(atom, operand, p)
                    for p in index
                )
            )
        operands.append(tuple(lanes))
    return Layout(*operands)


def layout(atom: Atom, on_gpu: bool) -> Layout:
    """``atom``'s layout where the kernel runs: probed on the GPU when
    ``on_gpu``, else the M4's (:func:`blocked_layout`)."""
    return probe(atom) if on_gpu else blocked_layout(atom)


def _runs(values: Sequence[int]) -> int:
    """How many runs of consecutive integers ``values`` form."""
    ordered = sorted(set(values))
    return 1 + sum(b != a + 1 for a, b in itertools.pairwise(ordered))


def k_order(layout: Layout, k: int) -> tuple[int, ...]:
    """A relabelling of the op's K positions, ``order[kappa]``, that the left
    and right inputs both use, chosen so each lane's elements of an operand
    row lie in as few runs of adjacent positions as it can: the permutation
    of the K index's bits with the fewest runs per lane and row of either
    input, the identity on a tie.  Any relabelling both inputs share gives
    the same product.  Over a register run's steps, the tile's K position of
    op position ``kappa`` at step ``s`` is ``steps * order[kappa] + s``, so a
    lane's run of ``r`` positions is ``r * steps`` adjacent elements of a
    weight row: on an M4, whose lanes hold K ``fm`` and ``fm + 8`` of the
    right input, the order that interleaves the halves reads one run of
    words per row instead of two, which lets a 6-bit code's 48 bits (8
    positions) fill three words exactly."""
    bits = k.bit_length() - 1
    assert 1 << bits == k, k
    if any(
        element == INVALID
        for coords in (layout.left, layout.right)
        for lane in coords
        for element in lane
    ):
        return tuple(range(k))
    groups: list[list[int]] = []
    for coords, row, col in ((layout.left, 0, 1), (layout.right, 1, 0)):
        for lane in coords:
            by_row: dict[int, list[int]] = {}
            for element in lane:
                by_row.setdefault(element[row], []).append(element[col])
            groups += by_row.values()

    def score(order: Sequence[int]) -> int:
        return sum(_runs([order[kappa] for kappa in group]) for group in groups)

    best = tuple(range(k))
    best_score = score(best)
    for perm in itertools.permutations(range(bits)):
        order = tuple(
            sum(((kappa >> i) & 1) << perm[i] for i in range(bits))
            for kappa in range(k)
        )
        if (value := score(order)) < best_score:
            best, best_score = order, value
    return best


@dataclasses.dataclass(frozen=True)
class Factored:
    """One coordinate of an operand's elements, factored: element ``e`` of
    lane ``l`` lies at ``base[l] + offset[e]``, ``min(base) == 0``."""

    base: tuple[int, ...]
    offset: tuple[int, ...]

    def constant(self) -> bool:
        """Whether every lane's base is 0: the coordinate is the offset."""
        return not any(self.base)


def factor(coords: Coords, d: int) -> Factored | None:
    """Coordinate ``d`` (0: row, 1: col) of an operand's elements as a
    per-lane base plus a per-element offset, or None when it isn't one."""
    if any(element == INVALID for lane in coords for element in lane):
        return None
    base = [lane[0][d] for lane in coords]
    least = min(base)
    base = [b - least for b in base]
    offset = [coords[0][e][d] - base[0] for e in range(len(coords[0]))]
    for b, lane in zip(base, coords, strict=True):
        if any(lane[e][d] != b + o for e, o in enumerate(offset)):
            return None
    return Factored(tuple(base), tuple(offset))


def lane_bits(base: Sequence[int]) -> tuple[int, ...] | None:
    """The weight of each bit of the lane index when ``base[l]`` is the sum
    of the weights of the bits set in ``l``, else None."""
    weights = tuple(base[1 << i] - base[0] for i in range(SIMD_WIDTH.bit_length() - 1))
    if base[0] != 0:
        return None
    for lane, b in enumerate(base):
        if b != sum(w for i, w in enumerate(weights) if lane >> i & 1):
            return None
    return weights


def lane_expr(lane: str, weights: Sequence[int]) -> str:
    """MSL that sums ``weights[i]`` over the bits ``i`` set in ``lane``."""
    terms = []
    for i, w in enumerate(weights):
        if not w:
            continue
        bit = f"(({lane} >> {i}) & 1)" if i else f"({lane} & 1)"
        terms.append(bit if w == 1 else f"{bit} * {w}")
    return " + ".join(terms) or "0"


def base_fact(base: Sequence[int]) -> Fact:
    """What the fold knows of a lane base: the gcd of its values, its least
    and its greatest."""
    return Fact(math.gcd(*base) or 1, min(base), max(base))
