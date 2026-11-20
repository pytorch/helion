from __future__ import annotations

import collections
import contextlib
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
import torch
from torch._inductor.codegen.common import CSEProxy
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.common import DeferredLine
from torch._inductor.codegen.common import TensorArg
from torch._inductor.codegen.simd import SIMDKernel
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import MutationLayoutSHOULDREMOVE
from torch._inductor.ir import Reduction
from torch._inductor.ir import ReinterpretView
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.shape_propagation import ShapePropagationOpsHandler
from torch._inductor.utils import IndentedBuffer
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.value_ranges import bound_sympy

from .. import config as helion_config
from .overrides import HelionOverrides
from .overrides import HelionPrinter
from .overrides import upcast_compute_type

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from typing_extensions import Self

    from torch._inductor.ops_handler import ReductionType


# A tile name per dim, or 1 for a broadcast dim.
TileShape = tuple[str | int, ...]
TILE_NAMES = {"z0_": "tile_z", "y": "tile_y", "x": "tile_x", "r0_": "tile_r"}
TILE_AXES = {name: axis for axis, name in TILE_NAMES.items()}
ARG_REDUCTIONS = OrderedSet(["argmax", "argmin"])
# The torch function that reduces a block for each plain reduction type.
REDUCTION_FUNCTIONS = {
    "sum": "sum",
    "prod": "prod",
    "max": "amax",
    "min": "amin",
    "any": "any",
}
REDUCTIONS = OrderedSet(
    [
        "sum",
        "prod",
        "max",
        "min",
        "any",
        "welford_reduce",
        "welford_combine",
        "online_softmax_reduce",
        *ARG_REDUCTIONS,
    ]
)


def names_row_dim(rnumel: sympy.Expr) -> bool:
    """Whether HelionKernel.row_dim() can name the dim of a row of `rnumel`
    elements, so that its reduction can be persistent. Not for a row of fewer
    than 16 that is not a power of two: Helion sizes a factory's dim of under
    16 by its value, so hl.zeros([tile_x, 12]).size(1) is 12 rather than the
    reduction dim of in0[tile_x, :], and hl.arange(12) beside the slice fails
    with "Attempting to broadcast a dimension of length 12". Such a row keeps
    a loop over tile_r, of one block."""
    if not rnumel.is_Integer:
        return False
    size = int(rnumel)
    return size >= 16 or size == next_power_of_2(size)


def accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp16 and bf16 accumulate in fp32, as in Triton's reductions, also with
    config.triton.codegen_upcast_to_fp32 off. In bf16, a sum of 65536 ones in
    blocks of 128 stops at 32768, where 32768 + 128 rounds to 32768."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def tile_axes(shape: Iterable[str | int] | None) -> list[str]:
    """The axes of the tiles in `shape`, without its size-1 dims."""
    return [TILE_AXES[str(dim)] for dim in shape or () if dim != 1]


def scope_shape(axes: Iterable[str], scope: Sequence[str]) -> TileShape:
    """Shape of a value over `axes` that broadcasts against the tiles of `scope`,
    e.g. (tile_y, 1) for {y} in (y, x)."""
    present = [axis for axis in scope if axis in axes]
    if not present:
        return ()
    start = scope.index(present[0])
    return tuple(TILE_NAMES[axis] if axis in present else 1 for axis in scope[start:])


def feeds_only_dots(node: torch.fx.Node | None) -> bool:
    """Whether the value of loop-body `node` only reaches dots, through casts.
    Such a value keeps its dtype, since a dot reads bf16 operands as bf16; any
    other value is computed in the compute type, as in Triton kernels."""
    return node is not None and all(
        user.target == "dot" or (user.target == "to_dtype" and feeds_only_dots(user))
        for user in node.users
    )


@dataclass(frozen=True)
class HelionTensorArg:
    # Placeholder for the parameter, named by codegen_kernel().
    ref: str
    # Buffer or ReinterpretView of a buffer.
    operand: IRNode
    output: bool


class HelionCSEProxy(CSEProxy):
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:  # noqa: ANN401
        args = cast("HelionKernel", self.kernel).align_args(args)
        result = super()._default(name, args, kwargs)
        # CSEProxy propagates dtypes and shapes only for Inductor's own backends,
        # not for cuda_backend "helion". Without shapes, softmax's sum
        # broadcasts exp(x - max), a [tile_x, tile_r] value, as one without
        # tile_r, and Helion fails on
        #     tmp5 = tmp4.expand(tile_r)
        # with "expand: the requested shape has too few dimensions!". Without
        # dtypes, (x * 2 - 1).to(torch.uint8) skips the int16 step that wraps a
        # negative value as eager does (HelionOverrides.to_dtype), and every
        # store casts: out0[tile_x] = tmp4.to(torch.float32).
        if isinstance(result, CSEVariable) and name != "masked":
            if result.dtype is None:
                dtype = getattr(DtypePropagationOpsHandler(), name)(*args, **kwargs)
                # Its rules upcast as Inductor's upcast_compute_type() does, only
                # for cuda_backend "triton".
                if name != "to_dtype" or kwargs.get("use_compute_types", True):
                    dtype = upcast_compute_type(dtype)
                result.dtype = dtype
            if result.shape is None:
                result.shape = getattr(ShapePropagationOpsHandler(), name)(
                    *args, **kwargs
                )
        return result

    def indirect_indexing(
        self,
        var: CSEVariable,
        size: sympy.Expr | int,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        # Indices are int64, as in Triton, where an int8 or int32 i of x[i] meets
        # its wrap's tl.full([1], size, tl.int64) and promotes. A 0-d int64 does
        # not promote: max_pool2d's backward, which reads int8 offsets, would get
        #     tmp2 = tmp0 + hl.full([], 4, dtype=torch.int64)
        #     tmp4 = torch.where(tmp3, tmp2, tmp0)
        #     hl.atomic_add(out0, [... + 957*(tmp4 // 2) + ...], tmp5)
        # with 957*(tmp4 // 2) computed in int8, wrapping, and an illegal memory
        # access. var's recorded dtype does not say: Inductor records tmp2 as
        # int64, so the cast is unconditional (and free for an int64 var).
        kernel = cast("HelionKernel", self.kernel)
        var = kernel.cse.generate(
            kernel.compute,
            f"{var}.to(torch.int64)",
            bounds=var.bounds,
            dtype=torch.int64,
            shape=var.shape,
        )
        symbol = super().indirect_indexing(var, size, check, wrap_neg)
        # Named after var or its wrap, both CSE variables.
        kernel.indirect[symbol] = kernel.cse.varname_map[symbol.name]
        return symbol

    def dot(self, x: CSEVariable, y: CSEVariable) -> CSEVariable:
        return cast("HelionKernel", self.kernel).dot(x, y)

    def to_dtype(
        self,
        x: CSEVariable,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> CSEVariable:
        kernel = cast("HelionKernel", self.kernel)
        kwargs = {"src_dtype": src_dtype, "use_compute_types": use_compute_types}
        node = V.interpreter.current_node
        if use_compute_types and x.dtype == dtype and feeds_only_dots(node):
            # A bf16 or fp16 value that only reaches dots keeps its dtype. Upcast,
            # as in
            #     tmp1 = tmp0.to(torch.float32)
            #     acc = torch.addmm(acc, tmp1, tmp3)
            # it makes the dot an fp32 one (dot_precision="ieee"). The result is
            # the same: bf16 and fp16 values are exact in fp32 and tf32, and the
            # dot accumulates in fp32 either way.
            return x
        if not (kernel.rolled and x in kernel.dot_results):
            return self._default("to_dtype", (x, dtype), kwargs)
        # A bf16 matmul output saved for backward is rounded between the dot and
        # its reduction. Emitted in the K loop, the rounding reads the running sum
        # and the store after the loop cannot use it:
        #     for tile_r in hl.tile(64):
        #         acc = torch.addmm(acc, tmp1, tmp3)
        #         tmp4 = acc.to(torch.bfloat16)
        #         tmp5 = tmp4.to(torch.float32)
        #     out0[tile_y, tile_x] = tmp5.to(torch.bfloat16)
        # Helion: "Device value 'tmp5' cannot be carried between top-level loops".
        # After the loop it rounds the final sum once, as eager does. Only in the
        # K loop: an epilogue's cast, as in (a @ b).to(torch.int32) + 1, is already
        # after the loop, and in the stores it would follow the + 1 that reads it.
        # So would the epilogue of a K in one block (a persistent reduction),
        # which has no loop: silu(a @ b) would read tmp3 before
        #     tmp3 = tmp2.to(torch.float32)
        with kernel.swap_buffers(kernel.stores, kernel.stores, kernel.stores):
            value = self._default("to_dtype", (x, dtype), kwargs)
        # So that tmp5 = tmp4.to(torch.float32) follows tmp4 out of the loop.
        kernel.dot_results.add(value)
        # The epilogue then reads value, not out0[tile_y, tile_x] from memory.
        kernel.outer_vars.add(value)
        return value


# The longest row a kernel that names its row (rdim, codegen_rdim()) names by a
# factory's dim, which Helion cannot roll, so that it stays one block whatever
# the autotuner tries, as Triton's rule reads an inner reduction's: a row of up
# to 1024 elements fits one block, and leaving its loop to the autotuner
# (Config.reduction_loops) widens the search, where a quick search can end on a
# loop over chunks of the row: for LayerNorm over Swin's shifted windows, rows
# of 256, it picked reduction_loops [16], slower than the row as one block. A
# kernel that does not name its row, as a plain sum's in0[tile_x, :], keeps the
# row's reduction_loops slot, at any length.
UNROLLED_ROW_MAX = 1024


class HelionKernel(SIMDKernel[CSEVariable]):
    """A Helion kernel: pointwise range trees become one hl.tile loop and the
    reduction range tree an inner hl.tile loop, or, in a persistent reduction
    (should_use_persistent_reduction()), one block of the whole row. Loads and
    stores subscript tensors by tile, e.g. in0[tile_y, tile_r], or in0[tile_y, :]
    for the whole row; Helion picks block sizes and masks partial tiles."""

    overrides = HelionOverrides  # type: ignore[assignment]

    @staticmethod
    def supports_tiling(tiling: dict[str, sympy.Expr]) -> bool:
        return OrderedSet(tiling) <= OrderedSet(TILE_NAMES) and all(
            isinstance(numel, (int, sympy.Integer)) for numel in tiling.values()
        )

    def __init__(self, tiling: dict[str, sympy.Expr], **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(tiling, **kwargs)
        self.outer_vars: OrderedSet[CSEVariable] = OrderedSet()
        # A matmul orders r0_ before x so dot operands read [y, r0_] and [r0_, x].
        matmul = self.is_native_matmul
        order = ("z0_", "y", "r0_", "x") if matmul else ("z0_", "y", "x", "r0_")
        self.axes = tuple(axis for axis in order if axis in self.numels)
        self.outer_axes = tuple(axis for axis in self.axes if axis != "r0_")
        roots = {tree.prefix: tree for tree in self.range_trees}
        self.symbols = {axis: roots[axis].index_sym() for axis in self.axes}
        self.tile_ranges: dict[sympy.Expr, sympy.Expr] = {
            self.symbols[axis]: self.numels[axis] for axis in self.axes
        }
        self.arguments: dict[tuple[Any, ...], HelionTensorArg] = {}
        # One buffer, so that a masked load follows the computation of its mask.
        self.loads = self.compute
        self.load_masks: list[CSEVariable] = []
        self.accumulator_count = 0
        self.has_fp32_dot = False
        # Whether the kernel draws random numbers; see rand().
        self.uses_rand = False
        # Whether a persistent reduction names the row's dim rdim, and the
        # parameter and subscript of a full row it defines rdim by; see
        # row_dim() and codegen_rdim().
        self.uses_rdim = False
        self.row_anchor: tuple[str, list[str]] | None = None
        # Dot accumulators and the casts of them that run after the K loop.
        self.dot_results: OrderedSet[CSEVariable] = OrderedSet()
        # Values stored outside the current reduction loop; see disable_reduction().
        self.outer_stores: dict[str, CSEVariable] = {}
        # Whether an indirect index needs a bounds check; see check_bounds().
        self.needs_bounds_check = False
        # The values indices read, by the symbols that stand for them.
        self.indirect: dict[sympy.Symbol, CSEVariable] = {}

    def __enter__(self) -> Self:
        super().__enter__()
        self.exit_stack.enter_context(
            V.set_ops_handler(HelionCSEProxy(self, HelionOverrides()))
        )
        return self

    @property
    def in_reduction(self) -> bool:
        """Whether codegen is in the reduction, not in a DisableReduction part of
        the node schedule."""
        return self.inside_reduction and self.features.is_reduction()

    @property
    def rolled(self) -> bool:
        """Whether codegen is in a loop over tile_r: in the reduction, and not
        a persistent one."""
        return self.in_reduction and not self.persistent_reduction

    def should_use_persistent_reduction(self) -> bool:
        """Whether the reduction reads each row as one block (a persistent
        reduction) instead of in a loop over tile_r. e / e.sum(-1, keepdim=True)
        with e = torch.exp(x - x.amax(-1, keepdim=True)) * 2 and x [64, 300]:
            for tile_x in hl.tile(64):
                tmp0 = in0[tile_x, :]
                tmp1 = torch.amax(tmp0, -1)
                ...
                tmp4 = torch.exp(tmp3)
                ...
                tmp7 = torch.sum(tmp6, -1)
                ...
                out0[tile_x, :] = tmp9
        reads x once and computes the exp once, where three loops over tile_r
        read x in each and compute the exp in the last two. Helion can still roll
        the row into a loop (Config.reduction_loops)."""
        # A row that row_dim() cannot name keeps its loop, whatever the
        # InductorChoices class in use decides.
        return (
            self.inside_reduction
            and names_row_dim(self.features.reduction_numel)
            and V.choices.should_use_persistent_reduction(
                self.features.with_tiling_scores(self.tiling_scores),
                self.cooperative_reduction,
            )
        )

    def dim_code(self, dim: object) -> str:
        """A dim of a value's shape, as code: in a persistent reduction tile_r
        is no loop variable but the whole row (row_dim())."""
        if self.persistent_reduction and str(dim) == "tile_r":
            return self.row_dim()
        return str(dim)

    def row_dim(self) -> str:
        """The dim of a whole row, as code, in a persistent reduction: rdim,
        the dim of an operand's full row (codegen_rdim()), as with R = 3000:
            rdim = in0[tile_x, :].size(1)
            tmp1 = tmp0.reshape([tile_x, 1]).expand(tile_x, rdim)
            tmp5 = (hl.arange(rdim)).to(torch.int64)
        The literal 3000 is not the dim of in0[tile_x, :], which Helion pads to
        4096 lanes: .expand(tile_x, 3000) fails with "The size of tensor a (u1)
        must match the size of tensor b (3000)", and so does hl.arange(3000)
        beside the slice. A literal power of two, or the dim of a factory,
        hl.zeros([tile_x, 262144]).size(1), is a dim the row shares, but one
        Helion cannot roll into a loop: with hl.arange(262144) in the kernel,
        its Config has no reduction_loops, and a row of 262144 elements is read
        as one block whatever the autotuner tries."""
        self.uses_rdim = True
        return "rdim"

    def dims_code(self, shape: Iterable[object]) -> str:
        return ", ".join(map(self.dim_code, shape))

    def coordinate(self, tile: str) -> str:
        """A subscript's coordinate for `tile`: `:`, the whole row, for tile_r in
        a persistent reduction, e.g. in0[tile_x, :]."""
        return ":" if self.persistent_reduction and tile == "tile_r" else tile

    def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable:  # noqa: ANN401
        var = super().create_cse_var(*args, **kwargs)
        if not self.in_reduction:
            self.outer_vars.add(var)
        return var

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return str(dtype)

    def indirect_vars(self, index: sympy.Expr) -> dict[sympy.Symbol, CSEVariable]:
        """The values that `index` reads: x.gather(1, i) loads x at 100*x1 + tmp4,
        tmp4 = i[x1, x0] (see HelionCSEProxy.indirect_indexing())."""
        return {
            symbol: self.indirect[symbol]
            for symbol in index.free_symbols
            if symbol in self.indirect
        }

    def index_axes(self, index: sympy.Expr) -> list[str]:
        """The axes of `index`: those of its tile indices and of the values it
        reads (indirect_vars())."""
        axes = OrderedSet(
            axis
            for var in self.indirect_vars(index).values()
            for axis in tile_axes(var.shape)
        )
        return [
            axis
            for axis in self.axes
            if self.symbols[axis] in index.free_symbols or axis in axes
        ]

    def value_shape(self, axes: Iterable[str]) -> TileShape:
        """Shape of a value over `axes`: their tiles in the kernel's axis order,
        (z, y, r0_, x) with a dot and (z, y, x, r0_) without. A value holds only
        its own tiles, since an r0_-only value in a matmul's K loop is [tile_r]
        next to [tile_y, tile_r] but [tile_r, 1] next to [tile_r, tile_x];
        align_args() reshapes values as ops combine them."""
        return tuple(TILE_NAMES[axis] for axis in self.axes if axis in axes)

    def index_tensor(self, axis: str, scope: Sequence[str]) -> str:
        name = f"{TILE_NAMES[axis]}.index"
        if axis == "r0_" and self.persistent_reduction:
            # The whole row's indices, e.g. for a flat index:
            #     in1[1024*tile_x.index.reshape([tile_x, 1]) + hl.arange(1024)]
            name = f"hl.arange({self.row_dim()})"
        axes = scope[scope.index(axis) :]
        if len(axes) == 1:
            return name
        # reshape() rather than slicing: Helion's tcgen05 store path proves the
        # layout of a store subscript only through reshape, +, *, // and %.
        shape = ", ".join(
            self.dim_code(TILE_NAMES[a]) if a == axis else "1" for a in axes
        )
        return f"{name}.reshape([{shape}])"

    def align(self, value: CSEVariable, target: TileShape) -> str:
        shape = tuple(value.shape or ())
        present = tile_axes(shape)
        target_axes = tile_axes(target)
        if any(axis not in target_axes for axis in present) or present != sorted(
            present, key=target_axes.index
        ):
            raise NotImplementedError(f"Cannot broadcast Helion {shape} to {target}")
        if not present:
            return str(value)
        aligned = scope_shape(present, target_axes)
        if aligned == shape:
            return str(value)
        return f"{value}.reshape([{self.dims_code(aligned)}])"

    def align_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """The values among an op's `args`, reshaped to broadcast against each
        other. F.pad(a, (0, 8, 0, 4)) @ b masks a with (tile_y < 36) & (tile_r < 64):
        [tile_y] & [tile_r] fails with "expected str(d1) == str(d2), got tile_y !=
        tile_r", while [tile_y, 1] & [tile_r] gives the [tile_y, tile_r] mask."""
        values = [arg for arg in args if isinstance(arg, CSEVariable)]
        axes = OrderedSet(axis for value in values for axis in tile_axes(value.shape))
        scope = [axis for axis in self.axes if axis in axes]

        def align(arg: object) -> object:
            if not isinstance(arg, CSEVariable):
                return arg
            axes = tile_axes(arg.shape)
            shape = scope_shape(axes, scope)
            if shape == tuple(arg.shape or ()):
                return arg
            line = f"{arg}.reshape([{self.dims_code(shape)}])"
            return self.cse.generate(self.compute, line, dtype=arg.dtype, shape=shape)

        return tuple(map(align, args))

    def tile_index(self, index: sympy.Expr) -> sympy.Expr:
        """`index` over the range tree roots, i.e. the tile indices."""
        index = sympy.sympify(index)
        entries: dict[sympy.Expr, sympy.Expr] = {
            symbol: self.range_tree_nodes[symbol].expr
            for symbol in index.free_symbols
            if symbol in self.range_tree_nodes
        }
        index = sympy.expand(sympy_subs(index, entries))
        return V.graph.sizevars.simplify_with_ranges(index, self.tile_ranges)

    def tensor_arg(self, operand: IRNode, *, output: bool = False) -> HelionTensorArg:
        layout = operand.get_layout()
        key = (
            operand.get_name(),
            tuple(layout.size),
            tuple(layout.stride),
            layout.offset,
        )
        if key not in self.arguments:
            ref = f"__arg{len(self.arguments)}__"
            self.arguments[key] = HelionTensorArg(ref, operand, output)
            if output:
                self.args.output(operand.get_name())
            else:
                self.args.input(operand.get_name())
        return self.arguments[key]

    def tiled_view(
        self, buffer: IRNode, index: sympy.Expr, axes: Sequence[str]
    ) -> IRNode | None:
        """A view of `buffer` with one dim per axis in `axes`, if `index` is affine
        in those tiles and the view stays inside the storage."""
        layout = buffer.get_layout()
        expanded = sympy.expand(index)
        sizes = [self.numels[axis] for axis in axes]
        strides = [expanded.coeff(self.symbols[axis]) for axis in axes]
        offset = expanded.xreplace({self.symbols[axis]: 0 for axis in self.axes})
        if not all(
            isinstance(value, (int, sympy.Integer))
            for value in (*sizes, *strides, offset)
        ):
            return None
        expected = offset + sum(
            self.symbols[a] * s for a, s in zip(axes, strides, strict=True)
        )
        end = offset + sum(
            (size - 1) * stride for size, stride in zip(sizes, strides, strict=True)
        )
        # The storage size can be symbolic, e.g. x[0] @ w with x [s, 40, 72] and a
        # dynamic s: end = 2879 < 2880 * s holds since s >= 2.
        if (
            offset < 0
            or any(stride < 0 for stride in strides)
            or not V.graph.sizevars.statically_known_lt(end, layout.storage_size())
            or not V.graph.sizevars.statically_known_equals(index, expected)
        ):
            return None
        return self.view(buffer, sizes, strides, offset)

    @staticmethod
    def view(
        buffer: IRNode,
        sizes: Sequence[sympy.Expr | int],
        strides: Sequence[sympy.Expr],
        offset: sympy.Expr,
    ) -> IRNode:
        """`buffer`, or a view of its storage with these sizes, strides and offset."""
        layout = buffer.get_layout()
        if (list(sizes), list(strides), offset) == (
            list(layout.size),
            list(layout.stride),
            layout.offset,
        ):
            return buffer
        view = FixedLayout(
            layout.device,
            layout.dtype,
            list(sizes),
            list(strides),
            offset,
            layout.is_pinned,
        )
        return ReinterpretView(data=buffer, layout=view)

    def dim_assignments(
        self, sizes: tuple[sympy.Expr | int, ...], axes: tuple[str, ...]
    ) -> Iterator[tuple[str | None, ...]]:
        """Each way to give the dims of `sizes` the tile `axes`, in order, and
        None to the size-1 dims left out. Choosing per dim among the axes of its
        size and None, then keeping the choices that hold `axes` in order, would
        try 2**21 choices for the 21 size-1 dims of x.sum(-1, keepdim=True)'s
        output with x: [1] * 20 + [128], where tile_x is 1."""
        if not sizes:
            if not axes:
                yield ()
            return
        size, rest = sizes[0], sizes[1:]
        if axes and V.graph.sizevars.statically_known_equals(
            size, self.numels[axes[0]]
        ):
            for dims in self.dim_assignments(rest, axes[1:]):
                yield (axes[0], *dims)
        if size == 1:
            for dims in self.dim_assignments(rest, axes):
                yield (None, *dims)

    def subscript(
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        *,
        output: bool = False,
    ) -> tuple[HelionTensorArg, list[str], TileShape]:
        """The parameter and subscript that read `buffer` at `index`, and its shape.

        Tries the buffer's own dims (dropping size-1 dims the index skips), then
        a tiled_view(), then a flat_subscript().
        """
        layout = buffer.get_layout()
        for dims in self.dim_assignments(tuple(layout.size), tuple(axes)):
            expected = layout.offset + sum(  # pyrefly: ignore [unsupported-operation]
                self.symbols[dim] * stride  # pyrefly: ignore [unsupported-operation]
                for dim, stride in zip(dims, layout.stride, strict=True)
                if dim is not None
            )
            if V.graph.sizevars.statically_known_equals(index, expected):
                kept = [i for i, dim in enumerate(dims) if dim is not None]
                operand = self.view(
                    buffer,
                    [layout.size[i] for i in kept],
                    [layout.stride[i] for i in kept],
                    layout.offset,
                )
                tiles = tuple(TILE_NAMES[dim] for dim in dims if dim is not None)
                arg = self.tensor_arg(operand, output=output)
                coordinates = [*map(self.coordinate, tiles)] or ["()"]
                self.note_row(arg, coordinates)
                return arg, coordinates, tiles
        if (view := self.tiled_view(buffer, index, axes)) is not None:
            tiles = tuple(TILE_NAMES[axis] for axis in axes)
            coordinates = [*map(self.coordinate, tiles)] or ["()"]
            arg = self.tensor_arg(view, output=output)
            self.note_row(arg, coordinates)
            return arg, coordinates, tiles
        return self.flat_subscript(buffer, index, axes, output=output)

    def note_row(self, arg: HelionTensorArg, coordinates: list[str]) -> None:
        """Keep the first full row of an input (in0[tile_x, :]) as the row
        codegen_rdim() defines rdim by."""
        if self.row_anchor is None and ":" in coordinates and not arg.output:
            self.row_anchor = (arg.ref, coordinates)

    def flat_subscript(
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        *,
        output: bool = False,
    ) -> tuple[HelionTensorArg, list[str], TileShape]:
        """A flat view of the storage read at `index`, like a Triton pointer offset."""
        size = buffer.get_layout().storage_size()
        operand = self.view(buffer, [size], [sympy.S.One], sympy.S.Zero)
        arg = self.tensor_arg(operand, output=output)
        return arg, [self.index_tensor_expr(index, axes)], self.value_shape(axes)

    def disable_reduction(self) -> contextlib.AbstractContextManager[None]:
        outside = super().disable_reduction()

        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            with outside:
                yield
            if self.persistent_reduction:
                # No loop: the CSE store cache keeps the values stored outside
                # the reduction.
                return
            # Back in a reduction loop: values stored outside it are read
            # from outer_stores instead of the CSE store cache.
            for name, value in [*self.cse.store_cache.items()]:
                if value in self.outer_vars:
                    self.outer_stores[name] = value
                    del self.cse.store_cache[name]

        return ctx()

    def codegen_body(self) -> None:
        if not (self.compute or self.stores):
            return
        # A persistent reduction has no loop: one body, so that a value stored
        # and read again in the kernel comes from the CSE store cache, and a
        # buffer only the kernel reads is not written (remove_kernel_local_buffers()).
        if self.rolled:
            self.body.writeline(f"for tile_r in hl.tile({self.numels['r0_']}):")
            with self.body.indent():
                self.body.splice(self.compute)
            self.body.splice(self.stores)
            # Values from the loop cannot be carried to the next top-level loop.
            self.cse.invalidate(self.outer_vars)
        else:
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.compute.clear()
        self.stores.clear()

    def codegen_kernel(self) -> str:
        # No buffer or kernel names, so identical schedules share one module.
        self.codegen_body()
        code = IndentedBuffer()
        code.writelines(
            [
                "import torch",
                "import helion",
                "import helion.language as hl",
                "",
            ]
        )
        settings = (
            f'backend="{helion_config.backend}", static_shapes=True, '
            f'autotune_effort="{helion_config.autotune_effort}"'
        )
        if self.has_fp32_dot:
            tf32 = torch.backends.cuda.matmul.fp32_precision == "tf32"
            settings += f', dot_precision="{"tf32" if tf32 else "ieee"}"'
        if helion_config.fast_math:
            settings += ", fast_math=True"
        # CuTe's hl.rand draws offset i from Philox(seed, i // 4)[i % 4] by
        # default (cute_rng_stream "auto", four numbers per Philox evaluation),
        # where Inductor and Helion's Triton backend draw Philox(seed, i)[0]:
        # word0 keeps a model's dropout masks the same on every backend.
        if self.uses_rand and helion_config.backend == "cute":
            settings += ', cute_rng_stream="word0"'
        code.writeline(f"@helion.kernel({settings})")
        names = self.kernel_arg_names()
        signature = ", ".join(f"{name}: torch.Tensor" for name in names.values())
        code.writeline(f"def kernel({signature}) -> None:")
        with code.indent():
            tiles = ", ".join(TILE_NAMES[axis] for axis in self.outer_axes)
            sizes = ", ".join(str(self.numels[axis]) for axis in self.outer_axes)
            sizes = sizes if len(self.outer_axes) == 1 else f"[{sizes}]"
            code.writeline(f"for {tiles} in hl.tile({sizes}):")
            with code.indent():
                self.codegen_rdim(code, TILE_NAMES[self.outer_axes[-1]])
                code.splice(self.body)
        return re.sub(r"__arg\d+__", lambda ref: names[ref[0]], code.getvalue())

    def codegen_rdim(self, code: IndentedBuffer, tile: str) -> None:
        """Define rdim (row_dim()): for a row of more than UNROLLED_ROW_MAX
        elements, the dim of the first full row the kernel reads, as in rdim =
        in0[tile_x, :].size(1), whose load Helion drops as only its size is
        used; else the dim of a factory beside `tile`, hl.zeros([tile_x,
        300]).size(1), which the row shares but Helion cannot roll."""
        if not self.uses_rdim:
            return
        rolled = self.numels["r0_"] > UNROLLED_ROW_MAX
        if self.row_anchor is not None and rolled:
            ref, coordinates = self.row_anchor
            dim = coordinates.index(":")
            code.writeline(f"rdim = {ref}[{', '.join(coordinates)}].size({dim})")
        else:
            size = self.numels["r0_"]
            code.writeline(f"rdim = hl.zeros([{tile}, {size}]).size(1)")

    def kernel_args(self) -> list[HelionTensorArg]:
        """Inputs, then outputs whose stores were not removed."""
        removed = (
            self.removed_buffers
            | self.inplaced_to_remove
            | V.graph.removed_buffers
            | V.graph.inplaced_to_remove
        )
        args = [
            arg
            for arg in self.arguments.values()
            if not (arg.output and arg.operand.get_name() in removed)
        ]
        return sorted(args, key=lambda arg: arg.output)

    def kernel_arg_names(self) -> dict[str, str]:
        counts: collections.Counter[str] = collections.Counter()
        names = {}
        for arg in self.kernel_args():
            prefix = "out" if arg.output else "in"
            names[arg.ref] = f"{prefix}{counts[prefix]}"
            counts[prefix] += 1
        return names

    def call_args(self) -> list[str]:
        # An input whose storage an output took over is passed as that output.
        # Reuse can chain inside one kernel: with buf1 reusing buf0 and buf2
        # reusing buf1, the wrapper runs `buf1 = buf0; del buf0` and
        # `buf2 = buf1; del buf1`, so buf0 is passed as buf2.
        reused = {inp: out for out, inp in self.inplace_update_buffers.items()}
        args = []
        for arg in self.kernel_args():
            operand = arg.operand
            name = operand.get_name()
            while name in reused:
                name = reused[name]
            if name != operand.get_name():
                target = V.graph.get_buffer(name)
                operand = ReinterpretView(data=target, layout=operand.get_layout())
            args.append(
                operand.codegen_reference()
                if isinstance(operand, ReinterpretView)
                else operand.get_name()
            )
        return args

    def codegen_nan_check(self) -> None:
        # Check each buffer, as Triton does, not call_args(): a flat operand is
        # passed as a view of its whole storage span, which can include memory
        # outside the tensor.
        wrapper = V.graph.wrapper_code
        _, call_args, signatures, _ = self.args.python_argdefs()
        for arg, signature in zip(call_args, signatures, strict=True):
            if isinstance(signature, TensorArg):
                wrapper.writeline(f"assert not {arg}.isnan().any().item()")
                wrapper.writeline(f"assert not {arg}.isinf().any().item()")

    def call_kernel(
        self, name: str, node: IRNode | None = None, deallocate_ws: bool = True
    ) -> None:
        wrapper = V.graph.wrapper_code
        wrapper.writeline(wrapper.wrap_kernel_call(name, self.call_args()))

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        if name in self.outer_stores:
            return self.outer_stores[name]
        buffer = V.graph.get_buffer(name)
        index = self.tile_index(index)
        axes = self.index_axes(index)
        mask, mask_axes = self.load_mask(
            axes,
            bool(self.indirect_vars(index))
            or (bool(self.load_masks) and not self.in_range(buffer, index)),
        )
        if not mask_axes <= OrderedSet(axes):
            axes = [axis for axis in self.axes if axis in axes or axis in mask_axes]
            arg, coordinates, shape = self.flat_subscript(buffer, index, axes)
        else:
            arg, coordinates, shape = self.subscript(buffer, index, axes)
        # A bf16 a in (a + 1).float() @ b is read as fp32, so that a + 1 is fp32 as
        # its CSE dtype says. Read as bf16, bf16 + hl.full([], 1.0, dtype=float32)
        # stays bf16 and the dot fails: "lhs and rhs bitwidth must be the same".
        dtype = buffer.get_dtype()
        if not feeds_only_dots(V.interpreter.current_node):
            dtype = upcast_compute_type(dtype)
        cast_suffix = "" if dtype == buffer.get_dtype() else f".to({dtype})"
        subscript = ", ".join(coordinates)
        if mask is None:
            line = f"{arg.ref}[{subscript}]{cast_suffix}"
        else:
            # Helion masks only lanes past the tile range; the body of
            # ops.masked() must not read where its mask is false.
            mask = self.align(mask, shape)
            indices = "" if coordinates == ["()"] else self.index_list(coordinates)
            line = f"hl.load({arg.ref}, [{indices}], extra_mask={mask}){cast_suffix}"
        return self.cse.generate(self.loads, line, dtype=dtype, shape=shape)

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: str | None = None,
    ) -> None:
        if mode not in (None, "atomic_add"):
            raise NotImplementedError(f"Helion kernels do not support {mode} stores")
        buffer = V.graph.get_buffer(name)
        layout = buffer.get_layout()
        if isinstance(layout, MutationLayoutSHOULDREMOVE):
            buffer = layout.get_buffer()
        index = self.tile_index(index)
        # The subscript has the tiles of the index and of the value, so the
        # index broadcasts to the value as Triton's pointer does. index_put(out,
        # (arange(n) * 0,), x, True) adds each x[i] at out[0], index 0:
        #     hl.atomic_add(out0, [tile_x], tmp0)  # out0: a stride-0 view of out
        # Subscripted out0[()], the [tile_x] value fails: "Cannot broadcast Helion
        # ('tile_x',) to ()".
        store_axes = OrderedSet([*self.index_axes(index), *tile_axes(value.shape)])
        axes = [axis for axis in self.axes if axis in store_axes]
        arg, coordinates, shape = self.subscript(buffer, index, axes, output=True)
        dtype = buffer.get_dtype()
        cast_suffix = "" if value.dtype == dtype else f".to({dtype})"
        buffer_code = self.compute if self.in_reduction else self.stores
        value_code = f"{self.align(value, shape)}{cast_suffix}"
        if mode == "atomic_add":
            # index_put(out, (i,), x, accumulate=True) with i = arange(4) // 2
            # adds x[0] and x[1] to out[0]:
            #     hl.atomic_add(out0, [(tile_x.index // 2)], tmp0)
            index_list = "" if coordinates == ["()"] else self.index_list(coordinates)
            line = f"hl.atomic_add({arg.ref}, [{index_list}], {value_code})"
        else:
            line = f"{arg.ref}[{', '.join(coordinates)}] = {value_code}"
        # DeferredLine, so that remove_kernel_local_buffers() can drop it.
        buffer_code.writeline(DeferredLine(name, line))

    @staticmethod
    def index_list(coordinates: Sequence[str]) -> str:
        """`coordinates` as the index list of hl.load or hl.atomic_add, where the
        whole row is slice(None): a masked load through its buffer's own layout
        in a persistent reduction is
            tmp0 = hl.load(in0, [tile_x, slice(None)], extra_mask=tmp3)
        where hl.load(in0, [tile_x, :], extra_mask=tmp3) is a SyntaxError."""
        return ", ".join("slice(None)" if c == ":" else c for c in coordinates)

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        index = self.tile_index(expr)
        dtype = upcast_compute_type(dtype)
        if not index.free_symbols:
            value = int(index) if index.is_Integer else float(index)
            line = HelionOverrides.constant(value, dtype)
            return self.cse.generate(self.compute, line, dtype=dtype, shape=())
        axes = self.index_axes(index)
        return self.cse.generate(
            self.compute,
            f"({self.index_tensor_expr(index, axes)}).to({dtype})",
            dtype=dtype,
            shape=self.value_shape(axes),
        )

    def index_tensor_expr(self, index: sympy.Expr, axes: Sequence[str]) -> str:
        indirect = self.indirect_vars(index)
        values = OrderedSet(indirect)
        symbols = OrderedSet(self.symbols[axis] for axis in axes)
        if index.free_symbols - symbols - values:
            raise NotImplementedError(f"Unsupported Helion index: {index}")
        scope = [axis for axis in self.axes if axis in axes]
        # tile.index is int32; widen it if any subexpression could overflow. One
        # that reads a value with tiles is int64 already
        # (HelionCSEProxy.indirect_indexing()): x.gather(1, i) reads x at
        # tmp1 + 100*(tile_x.index // 10). A 0-d value does not promote an int32
        # tensor, so an index that reads one always widens: in
        # ((tile_x.index + 4*tmp0) % 100) with tmp0 = 2**29, the sum would wrap.
        ranges = {
            self.symbols[axis]: ValueRanges(0, self.numels[axis] - 1) for axis in axes
        }
        # The terms without a value are summed in int32 before the value's term
        # joins: bound their sum too.
        rest = index.xreplace(dict.fromkeys(values, sympy.Integer(0)))
        bounds = [
            bound_sympy(term, ranges)
            for term in (rest, *sympy.preorder_traversal(index))
            if not term.free_symbols & values
        ]
        widen = any(not (b.lower >= -(2**31) and b.upper < 2**31) for b in bounds)
        widen |= any(not tile_axes(var.shape) for var in indirect.values())
        suffix = ".to(torch.int64)" if widen else ""
        replacements = {
            self.symbols[axis]: sympy.Symbol(self.index_tensor(axis, scope) + suffix)
            for axis in axes
        }
        # A value read as an index broadcasts like a tile index:
        # softmax(embedding(i, w), -1) reads w at
        #     in1[64*tmp1.reshape([tile_x, 1]) + tile_r.index]
        # where tmp1 = in0[tile_x].to(torch.int64) alone would be [tile_x] against
        # [tile_r].
        shape = self.value_shape(scope)
        for symbol, var in indirect.items():
            replacements[symbol] = sympy.Symbol(self.align(var, shape))
        expr = HelionPrinter().doprint(index.xreplace(replacements), simplify=False)
        # Broadcast to the tiles of `axes` the index skips, e.g. those of a store's
        # value or a load's mask: (tile_x.index // 2).expand(tile_y, tile_x), or
        # hl.full([tile_x], 0, dtype=torch.int32) for the index 0.
        if OrderedSet(self.index_axes(index)) >= OrderedSet(scope):
            return expr
        tiles = self.dims_code(TILE_NAMES[axis] for axis in scope)
        if not index.free_symbols:
            dtype = torch.int64 if widen else torch.int32
            return f"hl.full([{tiles}], {expr}, dtype={dtype})"
        return f"({expr}).expand({tiles})"

    def in_range(self, buffer: IRNode, index: sympy.Expr) -> bool:
        """Whether a direct `index` is within `buffer`'s storage on every lane."""
        ranges = {
            symbol: ValueRanges(0, self.numels[axis] - 1)
            for axis, symbol in self.symbols.items()
        }
        bound = bound_sympy(index, ranges)
        sizevars = V.graph.sizevars
        return sizevars.statically_known_geq(
            bound.lower, 0
        ) and sizevars.statically_known_lt(
            bound.upper, buffer.get_layout().storage_size()
        )

    def load_mask(
        self, axes: Sequence[str], unproven: bool
    ) -> tuple[CSEVariable | None, OrderedSet[str]]:
        """The extra_mask of a load over `axes`, and its axes: the masks that are
        0-d or over any of those axes, or all of them for a load whose address
        is not proven in range. A mask over other axes only cannot move a direct
        load's address and is left to masked(), but an indirect one's can be out
        of range on the lanes a mask turns off, even at one address for all of
        them, as for _unsafe_masked_index(x, mask, [i.expand(64)], 0.0), and so
        can a direct one: [torch.full((64,), 2**40)] reads x at 2**40."""
        load_axes = OrderedSet(axes)
        guards = []
        guard_axes: OrderedSet[str] = OrderedSet()
        for mask in self.load_masks:
            mask_axes = OrderedSet(tile_axes(mask.shape))
            if unproven or not mask_axes or mask_axes & load_axes:
                guards.append(mask)
                guard_axes |= mask_axes
        if not guards:
            return None, guard_axes
        result = guards[0]
        for guard in guards[1:]:
            result = V.ops.logical_and(result, guard)
        return result, guard_axes

    def masked(
        self,
        mask: CSEVariable,
        body: Callable[[], CSEVariable],
        other: bool | float,
    ) -> CSEVariable:
        self.load_masks.append(mask)
        try:
            result = body()
        finally:
            self.load_masks.pop()
        if result.dtype is None:
            raise AssertionError(f"expected a dtype for masked value {result}")
        return V.ops.where(mask, result, V.ops.constant(other, result.dtype))

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        # Triton checks x.gather(1, i) with an i of unknown range by
        #     tl.device_assert((0 <= tmp4) & (tmp4 < 100), "index out of bounds: ...")
        # Helion kernels have no device assert, so such a kernel fails with
        # "Unsupported: check_bounds" (HelionScheduling). Raising here, inside
        # the node's codegen, would log "Error in codegen" before the error.
        self.needs_bounds_check |= lower or upper

    def rand(self, seed: CSEVariable, offset: CSEVariable) -> str:
        """Inductor's ops.rand: Triton's tl.rand(seed, offset.to(tl.uint32)), a
        uniform float32 from the first word of Philox(seed) at the counter
        (offset, 0, 0, 0), where offset is the element's flat index in int32.
        hl.rand with explicit offsets draws the same numbers:
            tmp0 = in0[()]                          # seeds[i] (ops.load_seed)
            tmp1 = (tile_x.index).to(torch.int32)   # offset
            tmp2 = hl.rand([], seed=tmp0, offsets=tmp1.to(torch.int64) & 0xFFFFFFFF)
        Its counter is the offset's low and high 32 bits. Past 2**31 elements the
        int32 offset wraps negative, whose high word would be 0xFFFFFFFF in
        hl.rand(..., offsets=tmp1.to(torch.int64)) but is 0 in Triton's uint32:
        the elements from 2**31 on would get other numbers than Inductor's."""
        self.uses_rand = True
        return (
            f"hl.rand([], seed={seed}, offsets={offset}.to(torch.int64) & 0xFFFFFFFF)"
        )

    def reduction_accumulator(
        self, reduction_type: str, src_dtype: torch.dtype, dtype: torch.dtype
    ) -> CSEVariable:
        """An accumulator of `dtype` that starts at `reduction_type`'s initial
        value for `src_dtype`, e.g. -inf for max."""
        init = Reduction.default_accumulator(
            reduction_type, accumulator_dtype(src_dtype)
        )
        if not isinstance(init, (bool, float, int)):
            raise AssertionError(f"expected one initial value, got {init}")
        return self.accumulator(init, dtype)

    def accumulator(self, init: bool | float, dtype: torch.dtype) -> CSEVariable:
        tiles = tuple(TILE_NAMES[axis] for axis in self.outer_axes)
        name = f"acc{self.accumulator_count or ''}"
        self.accumulator_count += 1
        acc = CSEVariable(name, ValueRanges.unknown(), dtype, tiles)
        self.outer_vars.add(acc)
        shape = ", ".join(tiles)
        if init == 0 and dtype != torch.bool:
            self.body.writeline(f"{acc} = hl.zeros([{shape}], dtype={dtype})")
        else:
            value = constant_repr(init)
            self.body.writeline(f"{acc} = hl.full([{shape}], {value}, dtype={dtype})")
        return acc

    def dot(self, lhs: CSEVariable, rhs: CSEVariable) -> CSEVariable:
        batch = ["z0_"] if "z0_" in self.numels else []
        lhs = self.dot_operand(lhs, [*batch, "y", "r0_"])
        rhs = self.dot_operand(rhs, [*batch, "r0_", "x"])
        # Each dot has its own accumulator, so horizontally fused dots stay apart.
        self.has_fp32_dot |= torch.float32 in (lhs.dtype, rhs.dtype)
        acc = self.accumulator(0, torch.float32)
        op = "baddbmm" if "z0_" in self.numels else "addmm"
        self.compute.writeline(f"{acc} = torch.{op}({acc}, {lhs}, {rhs})")
        self.dot_results.add(acc)
        return acc

    def dot_operand(self, value: CSEVariable, axes: Sequence[str]) -> CSEVariable:
        """`value` expanded to the matrix tile over `axes`. x @ torch.ones(72, 136)
        multiplies by a constant, and with the 0-d tmp1 = hl.full([], 1.0, ...),
        torch.addmm(acc, tmp0, tmp1) fails with "b must be 2D"."""
        tiles = tuple(TILE_NAMES[axis] for axis in axes)
        if tuple(value.shape or ()) == tiles:
            return value
        line = f"{self.align(value, tiles)}.expand({self.dims_code(tiles)})"
        return self.cse.generate(self.compute, line, dtype=value.dtype, shape=tiles)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        if reduction_type == "dot":
            return value
        if reduction_type not in REDUCTIONS:
            raise AssertionError(f"Helion kernels do not support {reduction_type}")
        if isinstance(value, tuple) and reduction_type not in (
            "welford_combine",
            "online_softmax_reduce",
        ):
            raise AssertionError(f"expected one value to reduce, got {value}")

        def upcast(value: CSEVariable) -> CSEVariable:
            if value.dtype is None or accumulator_dtype(value.dtype) == value.dtype:
                return value
            line = f"{value}.to(torch.float32)"
            return self.cse.generate(
                self.compute, line, dtype=torch.float32, shape=value.shape
            )

        # As in Triton, a reduction of a value already reduced reuses its result:
        # the mean and m2 of torch.var_mean(x, -1) are two buffers, whose nodes
        # each reduce x with welford_reduce. Without the reuse, the second node
        # writes three more accumulators, acc3 to acc5, which recompute the
        # mean, m2 and weight.
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]
        values = [
            self.over_tile_r(upcast(v))
            for v in (value if isinstance(value, tuple) else (value,))
        ]
        acc_dtype = accumulator_dtype(dtype)
        if reduction_type in ARG_REDUCTIONS:
            result = self.arg_reduce(
                reduction_type, values[0], accumulator_dtype(src_dtype)
            )
        elif reduction_type in ("welford_reduce", "welford_combine"):
            result = self.welford(reduction_type, values, acc_dtype)
        elif reduction_type == "online_softmax_reduce":
            result = self.online_softmax_reduce(values, acc_dtype)
        elif self.persistent_reduction:
            # The row is one block, whose reduction is the result,
            #     tmp1 = torch.sum(tmp0, -1)
            # rather than acc = hl.zeros([tile_x], ...) and acc = acc + tmp1.
            fn = REDUCTION_FUNCTIONS[reduction_type]
            result = self.reduce_block_value(fn, values[0], acc_dtype)
        else:
            acc = self.reduction_accumulator(reduction_type, src_dtype, acc_dtype)
            fn = REDUCTION_FUNCTIONS[reduction_type]
            part, _ = self.reduce_block(fn, values[0], acc_dtype)
            block = {
                "sum": f"{acc} + {part}",
                "prod": f"{acc} * {part}",
                "max": f"torch.maximum({acc}, {part})",
                "min": f"torch.minimum({acc}, {part})",
                "any": f"{acc} | {part}",
            }[reduction_type]
            self.compute.writeline(f"{acc} = {block}")
            result = acc
        self.cse.reduction_cache[cache_key] = cast("Any", result)
        return result

    def over_tile_r(self, value: CSEVariable) -> CSEVariable:
        """`value`, broadcast over tile_r if it has no tile_r, so that each block
        reduces one copy per lane: tmp1 = tmp0.reshape([tile_x, 1]).expand(tile_x,
        tile_r). Adding hl.zeros([tile_x, tile_r]) instead would turn -0.0 into
        0.0, the max of z.expand(64, 300) with z all -0.0."""
        if "tile_r" in map(str, value.shape or ()):
            return value
        axes = tile_axes(value.shape)
        shape = self.value_shape([*axes, "r0_"])
        line = f"{self.align(value, shape)}.expand({self.dims_code(shape)})"
        return self.cse.generate(self.compute, line, dtype=value.dtype, shape=shape)

    def reduce_block(
        self, fn: str, value: CSEVariable, dtype: torch.dtype
    ) -> tuple[str, TileShape]:
        """torch.<fn> of a block of `value` over tile_r, e.g. torch.sum(tmp0, -1),
        for an accumulator of `dtype`, and its shape."""
        # Reduce tile_r wherever the axis order puts it: last without a dot, before
        # tile_x in a matmul's K loop, as in (a @ b, (a[:, :, None] * b[None]).sum(1)):
        #     acc = torch.addmm(acc, tmp0, tmp1)
        #     tmp3 = tmp2 * tmp1                     # [tile_y, tile_r, tile_x]
        #     acc1 = acc1 + torch.sum(tmp3, -2)
        # A sum over -1 there reduces tile_x: "Shape mismatch between [u0, u1] and
        # [u0, u2]". The result is aligned to the accumulator, e.g. (tile_y, 1).
        dims = [str(dim) for dim in value.shape or ()]
        kept = [TILE_AXES[dim] for dim in dims if dim not in ("tile_r", "1")]
        # torch.sum and torch.prod return int64 for an integer tile. For
        # x.sum(-1, dtype=torch.int32), acc = acc + torch.sum(tmp0, -1) would make
        # the int32 acc int64: "Tensor mismatch in control flow for variable
        # 'None': dtype torch.int32 != torch.int64".
        args = [str(value), str(dims.index("tile_r") - len(dims))]
        integer = value.dtype is not None and not value.dtype.is_floating_point
        if fn in ("sum", "prod") and integer and dtype != torch.int64:
            args.append(f"dtype={dtype}")
        part = f"torch.{fn}({', '.join(args)})"
        aligned = scope_shape(kept, self.outer_axes)
        if aligned != self.value_shape(kept):
            part = f"{part}.reshape([{self.dims_code(aligned)}])"
        return part, aligned

    def reduce_block_value(
        self, fn: str, value: CSEVariable, dtype: torch.dtype
    ) -> CSEVariable:
        """reduce_block() as a value, for a combine written with V.ops."""
        line, shape = self.reduce_block(fn, value, dtype)
        return self.cse.generate(self.compute, line, dtype=dtype, shape=shape)

    def arg_reduce(
        self, reduction_type: str, value: CSEVariable, dtype: torch.dtype
    ) -> CSEVariable:
        """The first index of the max (argmax) or min (argmin) of x over tile_r,
        as torch.argmax gives it: a NaN is both the max and the min. Each block
        is reduced to its max and the smallest index that holds it, and folded
        into the accumulators acc and acc_index as Triton's
        max_with_first_index and maximum_with_index do, with V.ops.
        torch.argmax(x, -1), with the locals each line comes from:
            tmp1 = (tile_r.index).to(torch.int64)          # index
            tmp2 = torch.amax(tmp0, -1)                    # block
            ...
            tmp6 = torch.logical_or(tmp4, tmp5)            # hit: x == block or isnan(x)
            tmp7 = hl.full([], 9223372036854775807, dtype=torch.int64)
            tmp8 = torch.where(tmp6, tmp1, tmp7)
            tmp9 = torch.amin(tmp8, -1)                    # first
            tmp10 = tmp2 > acc                             # larger
            ...
            tmp21 = torch.logical_or(tmp16, tmp20)         # take
            tmp22 = torch.where(tmp21, tmp9, acc1)
            tmp23 = torch.where(tmp21, tmp2, acc)
            acc1 = tmp22
            acc = tmp23
        take is "larger, or equal at a smaller index". The tie makes the first
        block of a row of -inf win: argmax is 0 there, not the initial index,
        int64 max."""
        ops = V.ops
        index = ops.index_expr(self.symbols["r0_"], torch.int64)
        largest = reduction_type.startswith("argmax")
        index_max = torch.iinfo(torch.int64).max
        block = self.reduce_block_value("amax" if largest else "amin", value, dtype)
        hit = ops.eq(value, block)
        floating = dtype.is_floating_point
        if floating:
            hit = ops.logical_or(hit, ops.isnan(value))
        sentinel = ops.constant(index_max, torch.int64)
        first = self.reduce_block_value(
            "amin", ops.where(hit, index, sentinel), torch.int64
        )
        if self.persistent_reduction:
            # The row is one block: its first index of the max is the argmax,
            # with no accumulators acc and acc1 to fold it into.
            return first
        acc = self.reduction_accumulator(reduction_type, dtype, dtype)
        acc_index = self.accumulator(index_max, torch.int64)
        larger = ops.gt(block, acc) if largest else ops.lt(block, acc)
        equal = ops.eq(block, acc)
        if floating:
            block_nan, acc_nan = ops.isnan(block), ops.isnan(acc)
            larger = ops.logical_or(
                larger, ops.logical_and(block_nan, ops.logical_not(acc_nan))
            )
            equal = ops.logical_or(equal, ops.logical_and(block_nan, acc_nan))
        take = ops.logical_or(larger, ops.logical_and(equal, ops.lt(first, acc_index)))
        new_index = ops.where(take, first, acc_index)
        new_value = ops.where(take, block, acc)
        self.compute.writeline(f"{acc_index} = {new_index}")
        self.compute.writeline(f"{acc} = {new_value}")
        return acc_index

    def online_softmax_reduce(
        self, values: Sequence[CSEVariable], dtype: torch.dtype
    ) -> tuple[CSEVariable, CSEVariable]:
        """The max m of x over tile_r and the sum of exp(x - m), which softmax
        and log_softmax lower to (prepare_softmax_online). `values` is x, or x
        and a weight w of each exp: a split reduction gives its first kernel
        (x, 1), and (-inf, 0) past the end of R, and its second the first's
        (max, sum) pairs. Each block of tile_r is reduced to its max and its sum
        of exps, and folded into the accumulators acc_max and acc_sum with
        V.ops. torch.softmax(x, -1), with the locals each line comes from:
            tmp3 = torch.amax(tmp0, -1)               # block_max
            ...
            tmp6 = torch.exp(tmp5)
            tmp8 = torch.where(tmp7, tmp2, tmp6)      # term: x == -inf ? 0 : tmp6
            tmp9 = torch.sum(tmp8, -1)                # block_sum
            tmp10 = torch.maximum(acc, tmp3)          # new_max
            ...
            tmp14 = torch.exp(tmp13)                  # scale: exp(-|acc - tmp3|)
            ...
            tmp20 = torch.where(tmp16, tmp2, tmp19)   # acc_part: acc == -inf ? 0 : ...
            ...
            tmp27 = torch.where(tmp24, tmp25, tmp26)  # new_sum: tmp10 == -inf ? nan : ...
            acc1 = tmp27
            acc = tmp10
        Of the two rescale factors exp(acc - new_max) and exp(block_max -
        new_max), one is 1 and the other scale, so one exp does. A -inf adds
        nothing: its term is 0, also in a block of only -inf (masked out, the
        padding past the end of R, or the -inf half of a split reduction), where
        x - block_max would make it exp(-inf - -inf) = nan; an accumulator whose
        max is -inf adds nothing either. The sum of a row of only -inf is nan,
        as in eager (exp(-inf - -inf)): so is the partial sum a split's first
        kernel writes for such a part of a row, which its second, whose term is
        0 for a max of -inf, ignores."""
        ops = V.ops
        value, *weight = values
        neg_inf = ops.constant(float("-inf"), dtype)
        zero = ops.constant(0.0, dtype)
        block_max = self.reduce_block_value("amax", value, dtype)
        exp = ops.exp(ops.sub(value, block_max))
        term = ops.where(
            ops.eq(value, neg_inf), zero, ops.mul(weight[0], exp) if weight else exp
        )
        block_sum = self.reduce_block_value("sum", term, dtype)
        if self.persistent_reduction:
            # The row is one block: its max and sum are the result, with the
            # sum nan for a row of only -inf, as below.
            nan = ops.constant(float("nan"), dtype)
            return block_max, ops.where(ops.eq(block_max, neg_inf), nan, block_sum)
        acc_max = self.accumulator(float("-inf"), dtype)
        acc_sum = self.accumulator(0, dtype)
        new_max = ops.maximum(acc_max, block_max)
        scale = ops.exp(ops.neg(ops.abs(ops.sub(acc_max, block_max))))
        one = ops.constant(1.0, dtype)
        acc_part = ops.where(
            ops.eq(acc_max, neg_inf),
            zero,
            ops.mul(acc_sum, ops.where(ops.eq(acc_max, new_max), one, scale)),
        )
        block_part = ops.mul(
            block_sum, ops.where(ops.eq(block_max, new_max), one, scale)
        )
        new_sum = ops.where(
            ops.eq(new_max, neg_inf),
            ops.constant(float("nan"), dtype),
            ops.add(acc_part, block_part),
        )
        self.compute.writeline(f"{acc_sum} = {new_sum}")
        self.compute.writeline(f"{acc_max} = {new_max}")
        return acc_max, acc_sum

    def welford(
        self, reduction_type: str, values: Sequence[CSEVariable], dtype: torch.dtype
    ) -> tuple[CSEVariable, CSEVariable, CSEVariable]:
        """The mean, m2 (the sum of squared deviations from the mean) and weight
        (the count) of x over tile_r, which var_mean and layer_norm lower to.
        welford_reduce reduces x; welford_combine reduces the (mean, m2, weight)
        partials of a split reduction's first kernel. Each block of tile_r is
        reduced to its own weight, mean and m2, in sums over the block in
        registers (the mean, its correction, then m2), and folded into the
        accumulators acc_mean, acc_m2 and acc_weight as Triton's welford_combine
        does (Chan et al.), with V.ops.
        torch.var_mean(x, -1), with the locals each line comes from:
            tmp3 = tmp2.expand(tile_r)                  # one
            tmp4 = torch.sum(tmp3, -1)                  # weight
            tmp5 = tmp2 / tmp4                          # fraction
            tmp6 = tmp0 * tmp5
            tmp7 = torch.sum(tmp6, -1)
            ...
            tmp11 = torch.where(tmp10, tmp1, tmp7)      # m0
            ...
            tmp16 = tmp11 + tmp15                       # mean
            tmp17 = tmp16.reshape([tile_x, 1])
            tmp18 = tmp0 - tmp17                        # dev
            ...
            tmp22 = tmp18 - tmp21                       # dev less its mean
            tmp23 = tmp22 * tmp22                       # spread(dev)
            tmp24 = torch.sum(tmp23, -1)                # m2
            tmp25 = acc2 + tmp4                         # new_weight
            ...
            tmp28 = torch.where(tmp26, tmp1, tmp27)     # delta
            ...
            tmp31 = torch.where(tmp29, tmp1, tmp30)     # share
            tmp32 = tmp28 * tmp31
            tmp33 = acc + tmp32                         # new_mean
            ...
            acc = tmp33
            acc1 = tmp38                                # new_m2
            acc2 = tmp25
        A row of inf gets the mean inf, as from Triton's persistent reductions
        (eager and Triton's looped ones give nan): the delta of two equal means
        is 0, where inf - inf would make it nan. A row of 2e19 gets the
        variance 0: see the order of the products below. As in any float32
        Welford, the merge takes each block's rounded mean, so on a row whose
        spread is below the float32 spacing of its mean the variance depends on
        the block size: [1e8, 1e8 + 8] * 32 + [1e8 + 8, 1e8 + 16] * 32, of
        variance 32.25, gives 32.25 in one block of 128, 80.6 in blocks of 64
        and 124.7 in blocks of 16."""
        ops = V.ops
        zero = ops.constant(0.0, dtype)

        def finite_or_zero(value: CSEVariable) -> CSEVariable:
            isinf, isnan = ops.isinf(value), ops.isnan(value)
            return ops.where(ops.logical_or(isinf, isnan), zero, value)

        # The block's weight, each value's share of it (1 / weight for x, w[r] /
        # weight for a partial r, as w[r] times a mean of 1e37 overflows), and
        # each value's term of m2 given its deviation from the block's mean.
        if reduction_type == "welford_reduce":
            (value,) = values
            if self.persistent_reduction:
                # The row is one block of R values: hl.full([], 300.0, ...),
                # rather than the sum of hl.full([], 1.0, ...).expand(rdim),
                # which would name the row's dim (row_dim()).
                weight = ops.constant(float(self.numels["r0_"]), dtype)
            else:
                # Helion masks the lanes past the end of R, so this counts the
                # others.
                one = self.over_tile_r(ops.constant(1.0, dtype))
                weight = self.reduce_block_value("sum", one, dtype)
            fraction = ops.truediv(ops.constant(1.0, dtype), weight)
            empty = None

            def spread(dev: CSEVariable) -> CSEVariable:
                return ops.mul(dev, dev)

        else:
            value, part_m2, part_weight = values
            weight = self.reduce_block_value("sum", part_weight, dtype)
            fraction = ops.truediv(part_weight, weight)
            # A block of weight 0, the (0, 0, 0) partials past the end of R in a
            # masked split, gets the mean and m2 0 rather than 0 / 0.
            empty = ops.eq(weight, zero)

            def spread(dev: CSEVariable) -> CSEVariable:
                # m2[r] + w[r] * dev^2, the weight first: a partial of weight 0
                # adds 0, where dev * dev could overflow to inf, and inf * 0 is nan.
                return ops.add(part_m2, ops.mul(ops.mul(part_weight, dev), dev))

        def weighted_mean(term: CSEVariable) -> CSEVariable:
            total = self.reduce_block_value("sum", ops.mul(term, fraction), dtype)
            return total if empty is None else ops.where(empty, zero, total)

        # The mean is m0 + weighted_mean(x - m0), with m0 = weighted_mean(x) (0
        # if inf or nan): weighted before they are summed, neither sum overflows
        # where sum(x) / weight would (a row of 1e37 sums to inf over 4096
        # lanes). The second corrects the rounding of the first, so that equal
        # values give their value exactly: (c + c + c) / 3 of c = 100553424 is
        # 100553432, which would make m2 192 rather than 0.
        m0 = finite_or_zero(weighted_mean(value))
        mean = ops.add(m0, weighted_mean(ops.sub(value, m0)))
        # The deviations from the mean, less their own mean, which is not 0 when
        # the mean rounds: of 100000000 and 100000008 in turn, the mean
        # 100000004 rounds to 100000000, and the deviations 0 and 8 would give
        # the variance 32 rather than 16.
        dev = ops.sub(value, mean)
        dev = ops.sub(dev, weighted_mean(dev))
        m2 = self.reduce_block_value("sum", spread(dev), dtype)
        if self.persistent_reduction:
            # The row is one block, so its mean, m2 and weight are the result, a
            # two-pass variance as in Triton's persistent reductions, with no
            # accumulators to combine them into.
            return mean, m2, weight
        acc_mean = self.accumulator(0, dtype)
        acc_m2 = self.accumulator(0, dtype)
        acc_weight = self.accumulator(0, dtype)
        # Triton's welford_combine of the accumulators and the block, with 0 for
        # the difference of two equal means, as inf - inf is nan, and for the
        # share of a new_weight of 0: a split's chunk past the end of R, as in
        # R = 9000 in 1024 chunks of 9 from a custom reduction_split_factor(),
        # holds only blocks of weight 0. The weights multiply delta before delta
        # does: on the first block acc_weight is 0 and delta the block's mean,
        # whose square overflows float32 for a mean of 2e19, and inf * 0 would
        # make m2 nan.
        new_weight = ops.add(acc_weight, weight)
        delta = ops.where(ops.eq(acc_mean, mean), zero, ops.sub(mean, acc_mean))
        share = ops.where(
            ops.eq(new_weight, zero), zero, ops.truediv(weight, new_weight)
        )
        new_mean = ops.add(acc_mean, ops.mul(delta, share))
        new_m2 = ops.add(
            ops.add(acc_m2, m2),
            ops.mul(ops.mul(delta, ops.mul(acc_weight, share)), delta),
        )
        self.compute.writeline(f"{acc_mean} = {new_mean}")
        self.compute.writeline(f"{acc_m2} = {new_m2}")
        self.compute.writeline(f"{acc_weight} = {new_weight}")
        return acc_mean, acc_m2, acc_weight
