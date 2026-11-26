from __future__ import annotations

import collections
import contextlib
from dataclasses import dataclass
import itertools
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._inductor.codegen.common import CSEProxy
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.common import DeferredLine
from torch._inductor.codegen.common import TensorArg
from torch._inductor.codegen.simd import PartialAccumulate
from torch._inductor.codegen.simd import SIMDKernel
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import MutationLayoutSHOULDREMOVE
from torch._inductor.ir import Reduction
from torch._inductor.ir import ReinterpretView
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.shape_propagation import ShapePropagationOpsHandler
from torch._inductor.utils import IndentedBuffer
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import Identity
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.value_ranges import bound_sympy

from .. import config as helion_config
from ..support import names_rdim
from .overrides import HelionOverrides
from .overrides import HelionPrinter
from .overrides import number_constant
from .overrides import upcast_compute_type
from .tiling import UNROLLED_RDIM_MAX
from .tiling import contiguous_reads
from .tiling import static_int

_T = TypeVar("_T")

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


def rounded_dot_dtype(node: torch.fx.Node | None) -> torch.dtype | None:
    """The dtype loop-body dot `node` rounds its product to before anything else
    reads it: its reduction's dtype, reached through casts to that dtype only, as
    in ops.reduction(bf16, bf16, "dot", dot) for a bf16 a @ b, or with
    to_dtype(dot, bf16) between them when the product is saved for backward.
    None for a product read in fp32, as torch.mm(a, b, out_dtype=torch.float32)
    reads it."""
    dtypes: list[torch.dtype] = []
    while node is not None and len(node.users) == 1:
        (node,) = node.users
        if node.target == "to_dtype":
            dtypes.append(cast("torch.dtype", node.args[2]))
        elif node.target == "reduction":
            dtypes.append(cast("torch.dtype", node.args[1]))
            return dtypes[0] if len(set(dtypes)) == 1 else None
        else:
            return None
    return None


@dataclass(frozen=True)
class HelionTensorArg:
    # Placeholder for the parameter, named by codegen_kernel().
    ref: str
    # Buffer or ReinterpretView of a buffer.
    operand: IRNode
    output: bool


@dataclass(frozen=True)
class HelionParam:
    """A parameter of the kernel (HelionKernel.kernel_params())."""

    name: str
    # What the wrapper passes for it.
    reference: str
    # The arguments it holds: one, or an input and an output an in-place
    # buffer joins.
    args: tuple[HelionTensorArg, ...]


# The ops Inductor writes as Python operators (BasicMathOpsMixin), which take
# a Python number for an operand: ops.add(x, 0.5) is x + 0.5.
OPERATOR_OPS = OrderedSet(
    ["add", "sub", "mul", "truediv", "pow", "eq", "ne", "lt", "le", "gt", "ge", "neg"]
)


class HelionCSEProxy(CSEProxy):
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:  # noqa: ANN401
        kernel = cast("HelionKernel", self.kernel)
        # A Python number (HelionOverrides.constant()) is an operand of an
        # operator beside a tensor, which rounds it to the tensor's dtype as
        # eager does. Any other op takes a 0-d tensor of it, as
        # torch.maximum(x, hl.full([], 0.0, dtype=torch.float32)) does, and so
        # does an operator of numbers alone, which Python would compute in
        # doubles (16777217.0 == 16777216.0 is false, in fp32 true), and at
        # compile time (1.0 / 0.0 raises).
        if name == "where":
            condition, a, b = args
            if (chosen := kernel.proven_choice(condition, a, b)) is not None:
                return chosen
            args = (
                kernel.as_tensor(condition),
                a,
                kernel.as_tensor(b) if kernel.is_number(a) else b,
            )
        elif name not in OPERATOR_OPS:
            args = tuple(map(kernel.as_tensor, args))
        elif name == "truediv" and kernel.is_number(args[0]):
            # A number over a tensor: Python would evaluate c / x as
            # x.__rtruediv__(c), which Helion traces as a reciprocal times c,
            # inf for 1e-39 / 1e-39 (the reciprocal of 1e-39 overflows
            # float32), where hl.full([], c, ...) / x divides.
            args = (kernel.as_tensor(args[0]), *args[1:])
        elif all(kernel.is_number(arg) for arg in args if isinstance(arg, CSEVariable)):
            first = next(
                i for i, arg in enumerate(args) if isinstance(arg, CSEVariable)
            )
            args = (*args[:first], kernel.as_tensor(args[first]), *args[first + 1 :])
        args = kernel.align_args(args)
        result = super()._default(name, args, kwargs)
        if (
            isinstance(result, CSEVariable)
            and name == "constant"
            and number_constant(*args)
        ):
            kernel.numbers.add(result.name)
        # CSEProxy propagates dtypes and shapes only for Inductor's own backends,
        # not for cuda_backend "helion". Without shapes, softmax's sum
        # broadcasts exp(x - max), a [tile_x, tile_r] value, as one without
        # tile_r, and Helion fails on
        #     tmp5 = tmp4.expand(tile_r)
        # with "expand: the requested shape has too few dimensions!". Without
        # dtypes, (x * 2 - 1).to(torch.uint8) skips the int16 step that wraps a
        # negative value as eager does (HelionOverrides.to_dtype), and every
        # store casts: out0[tile_x] = tmp4.to(torch.float32).
        if isinstance(result, CSEVariable):
            cast("HelionKernel", self.kernel).note_predicate(name, args, result)
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
        var = kernel.as_tensor(var)
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
        kernel.indirect_sizes[symbol] = sympy.sympify(size)
        return symbol

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        value = super().load(name, index)
        if value.dtype is None or feeds_only_dots(V.interpreter.current_node):
            return value
        dtype = upcast_compute_type(value.dtype)
        if dtype == value.dtype:
            return value
        # A value stored and read again in the kernel comes from the CSE store
        # cache as it is: in the compute type, but for the bf16 product of a dot
        # whose K is one block (HelionKernel.dot()). (a @ b + 1000) - 1000 would
        # compute in bf16:
        #     tmp2 = torch.mm(tmp0, tmp1)
        #     tmp4 = tmp2 + tmp3  # tmp3 = hl.full([], 1000.0, dtype=torch.float32)
        # bf16, since a 0-d fp32 does not promote it, and bf16 spacing near 1000
        # is 4: off by up to 2 from Inductor's Triton kernel. Read in fp32, as
        # the load of out0 from memory would be:
        #     tmp3 = tmp2.to(torch.float32)
        kernel = cast("HelionKernel", self.kernel)
        line = f"{value}.to({dtype})"
        return kernel.cse.generate(kernel.compute, line, dtype=dtype, shape=value.shape)

    def dot(self, x: CSEVariable, y: CSEVariable) -> CSEVariable:
        kernel = cast("HelionKernel", self.kernel)
        rounded = rounded_dot_dtype(V.interpreter.current_node)
        return kernel.dot(kernel.as_tensor(x), kernel.as_tensor(y), rounded)

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
            return kernel.as_tensor(x)
        source = kernel.cast_sources.get(x.name)
        if source is not None and source.dtype == dtype:
            # A cast back to a value's dtype, through one that holds each of
            # its values exactly, is the value: any(x.to(torch.int64).to(
            # torch.bool)) reads the bool x, where each lane would widen to
            # int64 and back.
            return source
        if not (kernel.rolled and x in kernel.dot_results):
            value = self._default("to_dtype", (x, dtype), kwargs)
            if exact_cast(x.dtype, value.dtype):
                kernel.cast_sources[value.name] = x
            return value
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


def exact_cast(src: torch.dtype | None, dst: torch.dtype | None) -> bool:
    """Whether a cast from the bool or integer dtype `src` to `dst` and back is
    the identity: bool to any dtype but float8_e8m0fnu (whose values are the
    powers of two: False casts to 2**-127, and back to True), and an integer to
    a wider integer or to a float whose significand holds it (int8 to
    bfloat16, int32 to float64). A float `src` is left out: a cast back to
    bfloat16 or float16 computes in float32 (use_compute_types), which the
    narrow value would not."""
    if src is None or dst is None or src == dst or dst.is_complex:
        return False
    if src == torch.bool:
        return dst != torch.float8_e8m0fnu
    if src.is_floating_point or src.is_complex or dst == torch.bool:
        return False
    info = torch.iinfo(src)
    if dst.is_floating_point:
        # Integers up to 2 / eps are exact: 256 in bfloat16, 2**24 in float32.
        return 2 / torch.finfo(dst).eps >= max(-info.min, info.max)
    target = torch.iinfo(dst)
    return target.min <= info.min and info.max <= target.max


# The segments a persistent rdim is emitted over (HelionKernel.rdim_segment()):
# at most this many, each a power of two of at least this many elements; a
# pointwise axis's segments (pointwise_segment()) are as many and at least as
# long. Shorter ones turn a dense tile into strided runs: emitted per segment,
# torch.stack([a, b, c, d], -1) would tile [1024, 1] and store every fourth
# element four times over, where its [1024, 4] tile stores them in one run.
RDIM_SEGMENTS_MAX = 4
RDIM_SEGMENT_MIN = 16


def as_less(relation: sympy.Rel) -> tuple[sympy.Basic, sympy.Basic, bool]:
    """`relation` as lhs < rhs or lhs <= rhs, and whether it is strict: a > b
    as b < a (HelionKernel.proven(), masked_extents())."""
    lhs, rhs = relation.lhs, relation.rhs
    if isinstance(relation, (sympy.Gt, sympy.Ge)):
        lhs, rhs = rhs, lhs
    return lhs, rhs, isinstance(relation, (sympy.Lt, sympy.Gt))


class HelionKernel(SIMDKernel[CSEVariable]):
    """A Helion kernel: pointwise range trees become one hl.tile loop and the
    reduction range tree an inner hl.tile loop, or, in a persistent reduction
    (should_use_persistent_reduction()), one block of the whole row. Loads and
    stores subscript tensors by tile, e.g. in0[tile_y, tile_r], or in0[tile_y, :]
    for the whole row; Helion picks block sizes and masks partial tiles."""

    # SIMDScheduling's mix-order codegen reads a Triton kernel's fixed config
    # (codegen_mix_order_reduction()); Helion's autotuner picks the kernel's.
    fixed_config = None

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
        # So does an outer reduction, whose x is contiguous in memory and r0_
        # strided, so that a tile's last dim is the contiguous one, as in the
        # first kernel of a split var_mean over the first dim of a [401408, 144]:
        #     tmp0 = in0[tile_y, tile_r, tile_x]          # x of stride 1
        #     tmp7 = torch.sum(tmp6, -2)
        # where in0[tile_y, tile_x, tile_r] puts the dim of stride 144 last,
        # and the autotuner must find lanes and vectors along a dim that is not.
        outer = self.inside_reduction and self.reads_strided_rdim()
        order = (
            ("z0_", "y", "r0_", "x")
            if self.is_native_matmul or outer
            else ("z0_", "y", "x", "r0_")
        )
        self.axes = tuple(axis for axis in order if axis in self.numels)
        self.outer_axes = tuple(axis for axis in self.axes if axis != "r0_")
        roots = {tree.prefix: tree for tree in self.range_trees}
        self.symbols = {axis: roots[axis].index_sym() for axis in self.axes}
        # Each tile's extent: the axis's numel, but for a split x
        # (split_x()).
        self.extents = {axis: self.numels[axis] for axis in self.axes}
        self.tile_ranges: dict[sympy.Expr, sympy.Expr] = {
            self.symbols[axis]: self.extents[axis] for axis in self.axes
        }
        self.arguments: dict[tuple[Any, ...], HelionTensorArg] = {}
        # The CSE variables that hold a Python number (HelionOverrides.constant()),
        # not a tensor; see as_tensor().
        self.numbers: OrderedSet[str] = OrderedSet()
        self.load_masks: list[CSEVariable] = []
        # The index or integer each CSE variable of index_expr() or constant()
        # holds, and the bounds on tiles each mask of them proves; see
        # note_predicate().
        self.symbolic: dict[str, sympy.Expr] = {}
        self.predicates: dict[str, list[sympy.Rel]] = {}
        # The value each cast that keeps every value exactly was cast from
        # (HelionCSEProxy.to_dtype()).
        self.cast_sources: dict[str, CSEVariable] = {}
        self.accumulator_count = 0
        self.has_fp32_dot = False
        # Whether the kernel draws random numbers; see rand().
        self.uses_rand = False
        # Whether a persistent reduction names its rdim (rdim = ...), and the
        # parameter and subscript of a full rdim it defines rdim by; see
        # rdim_code() and codegen_rdim().
        self.uses_rdim = False
        self.rdim_anchor: tuple[str, list[str]] | None = None
        # Dot accumulators and the casts of them that run after the K loop.
        self.dot_results: OrderedSet[CSEVariable] = OrderedSet()
        # Values stored outside the current reduction loop; see disable_reduction().
        self.outer_stores: dict[str, CSEVariable] = {}
        # Whether an indirect index needs a bounds check; see check_bounds().
        self.needs_bounds_check = False
        # The values indices read, by the symbols that stand for them, and the
        # size of the dim each indexes.
        self.indirect: dict[sympy.Symbol, CSEVariable] = {}
        self.indirect_sizes: dict[sympy.Symbol, sympy.Expr] = {}
        # A persistent rdim's constant cuts that each node's comparisons of the
        # r index make (a cat's or pad's r < 128); the nodes emitted over
        # segments of the rdim, and the segments, each in turn (rdim_segment());
        # see HelionScheduling.create_kernel_choices().
        self.rdim_cuts: dict[str, OrderedSet[int]] = {}
        self.segment_nodes: list[str] = []
        self.segments: list[tuple[int, int]] = []
        self.segment: tuple[int, int] | None = None
        self.rdim_extent = 0
        # The constant cuts that comparisons of a pointwise axis's index make
        # (a cat's c < 60 along channels), the axis whose equal segments the
        # kernel emits its whole body over, and the segment's start while it
        # does; see pointwise_segments() and pointwise_segment().
        self.axis_cuts: dict[str, OrderedSet[int]] = {}
        self.axis_segments: tuple[str, list[tuple[int, int]]] | None = None
        self.axis_segment: tuple[str, int] | None = None

    def __enter__(self) -> Self:
        if self.mix_order_reduction and "y" not in self.symbols:
            self.split_x()
        super().__enter__()
        self.exit_stack.enter_context(
            V.set_ops_handler(HelionCSEProxy(self, HelionOverrides()))
        )
        return self

    def split_x(self) -> None:
        """Tile x of a mix-order reduction (HelionScheduling.
        codegen_mix_order_reduction()) as splits of rsplit_size: tile_y over
        the splits, on the grid, and tile_x over the x of a split, in a loop.
        Index x is x % split of split x // split, so an index reads
        x = split * tile_y + tile_x (tile_index()): the rdim at x of an [x, r]
        tensor is in0[tile_y, tile_x, :], in its [splits, split, r] view."""
        xnumel, split = self.numels["x"], self.rsplit_size
        if xnumel % split:
            raise AssertionError(f"a split of {split} does not divide x of {xnumel}")
        self.axes = ("y", "x", "r0_")
        self.outer_axes = ("y", "x")
        self.symbols["y"] = sympy.Symbol("ysplit", integer=True, nonnegative=True)
        self.extents = {
            "y": xnumel // split,
            "x": sympy.Integer(split),
            "r0_": self.numels["r0_"],
        }
        self.tile_ranges = {self.symbols[a]: self.extents[a] for a in self.axes}

    def partial_accumulate(
        self,
        name: str,
        reduction_type: ReductionType,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        """A term of a mix-order pair's reduction over x, which codegen_kernel()
        reduces over the x of each split."""
        if reduction_type not in ("sum", "prod"):
            raise AssertionError(f"a reduction over x of {reduction_type}")
        value = self.as_tensor(value)
        self.saved_partial_accumulate.append(
            PartialAccumulate(name, reduction_type, value)
        )

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
        """Whether the reduction reads each rdim as one block (a persistent
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
        the rdim into a loop (Config.reduction_loops), unless a factory's dim
        names it (rdim_code(): an rdim of up to UNROLLED_RDIM_MAX the kernel
        names)."""
        # An rdim that rdim_code() cannot name keeps its loop, whatever the
        # InductorChoices class in use decides.
        return (
            self.inside_reduction
            and names_rdim(self.features.reduction_numel)
            and V.choices.should_use_persistent_reduction(
                self.features.with_tiling_scores(self.tiling_scores),
                self.cooperative_reduction,
            )
        )

    def reads_strided_rdim(self) -> bool:
        """Whether the reduction is an outer one, whose rdims are strided and
        whose pointwise dims are contiguous: Inductor's hint says so (OUTER,
        OUTER_TINY), or, for the hint DEFAULT that Inductor gives a reduction
        with an x too large to split, its loads read more bytes with a unit
        stride along a pointwise dim than along r0_ (contiguous_reads()), as
        Inductor's num_splits() decides from their strides: the sum over y of a
        [16384, 30522] gradient (a bias's), but not F.embedding(i, w).sum(-1),
        whose gathered rdims of w are contiguous along r0_."""
        hint = self.features.get_reduction_hint()
        if hint in (ReductionHint.OUTER, ReductionHint.OUTER_TINY):
            return True
        if hint != ReductionHint.DEFAULT:
            return False
        numel = static_int(self.features.numel)
        rnumel = static_int(self.features.reduction_numel)
        if numel is None or rnumel is None:
            return False
        found = contiguous_reads(self.features.node_schedule, numel, rnumel)
        if found is None:
            return False
        along_r, along_x = found
        return along_r < along_x

    def dim_code(self, dim: object) -> str:
        """A dim of a value's shape, as code: in a persistent reduction tile_r
        is no loop variable but the whole rdim (rdim_code()), or the segment of it
        emitted (rdim_segment())."""
        if self.persistent_reduction and str(dim) == "tile_r":
            return self.rdim_code()
        return str(dim)

    def rdim_code(self) -> str:
        """The dim of a whole rdim, as code, in a persistent reduction: rdim,
        the dim of an operand's full rdim (codegen_rdim()), as with R = 3000:
            rdim = in0[tile_x, :].size(1)
            tmp1 = tmp0.reshape([tile_x, 1]).expand(tile_x, rdim)
            tmp5 = (hl.arange(rdim)).to(torch.int64)
        The literal 3000 is not the dim of in0[tile_x, :], which Helion pads to
        4096 lanes: .expand(tile_x, 3000) fails with "The size of tensor a (u1)
        must match the size of tensor b (3000)", and so does hl.arange(3000)
        beside the slice. A literal power of two, or the dim of a factory,
        hl.zeros([tile_x, 262144]).size(1), is a dim the rdim shares, but one
        Helion cannot roll into a loop: with hl.arange(262144) in the kernel,
        its Config has no reduction_loops, and an rdim of 262144 elements is read
        as one block whatever the autotuner tries."""
        if self.segment is not None:
            # A power of two (rdim_cut_segments()), which Helion takes as the dim of
            # the segment's slices, in0[tile_y, tile_x, 128:256]
            # (segment_subscript()).
            lo, hi = self.segment
            return str(hi - lo)
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
        (z, y, r0_, x) with a dot or an outer reduction (reads_strided_rdim())
        and (z, y, x, r0_) without. A value holds only
        its own tiles, since an r0_-only value in a matmul's K loop is [tile_r]
        next to [tile_y, tile_r] but [tile_r, 1] next to [tile_r, tile_x];
        align_args() reshapes values as ops combine them."""
        return tuple(TILE_NAMES[axis] for axis in self.axes if axis in axes)

    def index_tensor(self, axis: str, scope: Sequence[str]) -> str:
        name = f"{TILE_NAMES[axis]}.index"
        if axis == "r0_" and self.persistent_reduction:
            # The whole row's indices, e.g. for a flat index:
            #     in1[1024*tile_x.index.reshape([tile_x, 1]) + hl.arange(rdim)]
            name = f"hl.arange({self.rdim_code()})"
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
            aligned = self.cse.generate(
                self.compute, line, dtype=arg.dtype, shape=shape
            )
            # A reshape keeps what the tiles' ranges prove of a condition
            # (proven()): the masks of torch.cat([x, y], 0) over x's part of y
            # only reach torch.where as [tile_y, 1].
            if arg.name in self.predicates:
                self.predicates.setdefault(aligned.name, self.predicates[arg.name])
            return aligned

        return tuple(map(align, args))

    def tile_index(self, index: sympy.Expr) -> sympy.Expr:
        """`index` over the range tree roots, i.e. the tile indices. Without
        Identity, which Inductor wraps around a term to keep it from being
        simplified, as cat's lowering does: x[:, :50] of cat([x, y], -1) reads
        x at 100*y0 + Identity(x1), which no view of x matches, and the load
        would be flat:
            tmp5 = hl.load(in0, [100*tile_y.index.reshape([tile_y, 1])
                + (tile_x.index)], extra_mask=tmp4)
        rather than tmp5 = in0[tile_y, tile_x]."""
        index = sympy.sympify(index).replace(Identity, lambda arg: arg)
        entries: dict[sympy.Expr, sympy.Expr] = {
            symbol: self.range_tree_nodes[symbol].expr
            for symbol in index.free_symbols
            if symbol in self.range_tree_nodes
        }
        index = sympy.expand(sympy_subs(index, entries))
        if "y" in self.symbols and self.mix_order_reduction:
            # Split x (split_x()): index x is tile_x of split tile_y.
            x = self.symbols["x"]
            index = sympy.expand(
                # pyrefly: ignore [unsupported-operation]
                index.xreplace({x: self.extents["x"] * self.symbols["y"] + x})
            )
        if self.segment is not None:
            # Within a segment of the rdim (rdim_segment()), its tile counts from
            # the segment's start.
            r = self.symbols["r0_"]
            lo = sympy.Integer(self.segment[0])
            index = sympy.expand(index.xreplace({r: sympy.Add(r, lo)}))
        if self.axis_segment is not None:
            # Within a segment of a pointwise axis (pointwise_segment()), its
            # tile counts from the segment's start too.
            axis, start = self.axis_segment
            symbol = self.symbols[axis]
            index = sympy.expand(
                index.xreplace({symbol: sympy.Add(symbol, sympy.Integer(start))})
            )
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
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        extents: dict[str, sympy.Expr],
    ) -> IRNode | None:
        """A view of `buffer` with one dim per axis in `axes`, of `extents`, if
        `index` is affine in those tiles and the view stays inside the storage
        (affine_view())."""
        return self.affine_view(
            buffer,
            index,
            [self.symbols[axis] for axis in axes],
            [extents[axis] for axis in axes],
        )

    def affine_view(
        self,
        buffer: IRNode,
        index: sympy.Expr,
        symbols: Sequence[sympy.Expr],
        sizes: Sequence[sympy.Expr],
    ) -> IRNode | None:
        """A view of `buffer` with a dim of each of `sizes` along each of
        `symbols`, if `index` is affine in them and the view stays inside the
        storage: its strides are the symbols' coefficients and its offset the
        rest of the index."""
        expanded = sympy.expand(index)
        strides = [expanded.coeff(symbol) for symbol in symbols]
        offset = expanded.xreplace(dict.fromkeys(symbols, sympy.S.Zero))
        if not all(
            isinstance(value, (int, sympy.Integer))
            for value in (*sizes, *strides, offset)
        ):
            return None
        expected = offset + sum(
            symbol * stride for symbol, stride in zip(symbols, strides, strict=True)
        )
        end = offset + sum(
            # pyrefly: ignore [unsupported-operation]
            (size - 1) * stride
            for size, stride in zip(sizes, strides, strict=True)
        )
        # The storage size can be symbolic, e.g. x[0] @ w with x [s, 40, 72] and a
        # dynamic s: end = 2879 < 2880 * s holds since s >= 2.
        if (
            offset < 0
            or any(stride < 0 for stride in strides)
            or not V.graph.sizevars.statically_known_lt(
                end, buffer.get_layout().storage_size()
            )
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
        self,
        sizes: tuple[sympy.Expr | int, ...],
        axes: tuple[str, ...],
        extents: dict[str, sympy.Expr],
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
        if axes and V.graph.sizevars.statically_known_equals(size, extents[axes[0]]):
            for dims in self.dim_assignments(rest, axes[1:], extents):
                yield (axes[0], *dims)
        if size == 1:
            for dims in self.dim_assignments(rest, axes, extents):
                yield (None, *dims)

    def subscript(
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        *,
        output: bool = False,
        bounds: dict[str, sympy.Expr] | None = None,
    ) -> tuple[HelionTensorArg, list[str], TileShape]:
        """The parameter and subscript that read `buffer` at `index`, and its shape.

        Tries an rdim segment's slice of the whole rdim (segment_subscript()), the
        buffer's own dims (dropping size-1 dims the index skips) or a
        tiled_view() (tiled_subscript()), for a load a gather_subscript() or a
        quotient_subscript(), then a flat_subscript(). `bounds` are the extents a
        masked load's mask bounds its tiles to (masked_extents()): the view
        need only cover those, as the pad of x from 30522 to 30528 reads
            tmp5 = hl.load(in0, [tile_y, tile_x], extra_mask=tmp4)
        of in0 [16384, 30522], where without them the view of an x of 30528
        would run past the storage and the load be flat:
            tmp5 = hl.load(in0, [tile_x.index + 30522*tile_y.index.reshape(
                [tile_y, 1])], extra_mask=tmp4)
        whose lanes the backend cannot tell are consecutive along x.
        """
        if (sliced := self.segment_subscript(buffer, index, axes, output)) is not None:
            return sliced
        extents = {**self.extents, **(bounds or {})}
        if (
            tiled := self.tiled_subscript(buffer, index, axes, extents, output)
        ) is not None:
            return tiled
        if not output and (gather := self.gather_subscript(buffer, index, axes)):
            return gather
        if not output and (quotient := self.quotient_subscript(buffer, index, axes)):
            return quotient
        return self.flat_subscript(buffer, index, axes, output=output)

    def tiled_subscript(
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        extents: dict[str, sympy.Expr],
        output: bool,
    ) -> tuple[HelionTensorArg, list[str], TileShape] | None:
        """The parameter and subscript that read `buffer` at `index` through
        its own dims (dropping size-1 dims the index skips) or a tiled_view(),
        a coordinate per tile; None when neither does, with no parameter or
        rdim taken on the way (segment_subscript() tries this alone, where
        subscript()'s flat fallback would name the rdim)."""
        layout = buffer.get_layout()
        for dims in self.dim_assignments(tuple(layout.size), tuple(axes), extents):
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
                self.note_rdim(arg, coordinates)
                return arg, coordinates, tiles
        if (view := self.tiled_view(buffer, index, axes, extents)) is not None:
            tiles = tuple(TILE_NAMES[axis] for axis in axes)
            coordinates = [*map(self.coordinate, tiles)] or ["()"]
            arg = self.tensor_arg(view, output=output)
            self.note_rdim(arg, coordinates)
            return arg, coordinates, tiles
        return None

    def segment_subscript(
        self, buffer: IRNode, index: sympy.Expr, axes: Sequence[str], output: bool
    ) -> tuple[HelionTensorArg, list[str], TileShape] | None:
        """Within a segment of a persistent rdim (rdim_segment()), the subscript
        of the whole rdim, its dim sliced to the elements the segment reads:
        the second half of rdims of 256 as in0[tile_y, tile_x, 128:256], of the
        parameter that reads whole rdims, in0[tile_y, tile_x, :]. A view of each
        part would be a parameter of its own, and the CuTe backend packs the
        stores of an output only when no other parameter shares its storage
        (cute_tensor_base_is_aligned()): out0 and out1, the halves of one
        output, were stored a lane at a time."""
        if self.segment is None or "r0_" not in axes:
            return None
        r = self.symbols["r0_"]
        length = self.segment[1] - self.segment[0]
        stride = index.coeff(r)
        # sympify: of the index r alone, xreplace() returns the int itself.
        start = sympy.sympify(index.xreplace(dict.fromkeys(self.symbols.values(), 0)))
        if not (
            isinstance(stride, sympy.Integer)
            and stride > 0
            and isinstance(start, sympy.Integer)
        ):
            return None
        # Where the segment starts within its rdim of the parameter.
        first = (
            int(start - buffer.get_layout().offset) // int(stride) % self.rdim_extent
        )
        if first + length > self.rdim_extent:
            return None
        shifted = sympy.expand(index.xreplace({r: sympy.Add(r, sympy.Integer(-first))}))
        with self.rdim_state(self.rdim_extent, None):
            # The parameter's own dims or a tiled view, a coordinate per tile:
            # subscript() would fall back to a flat read, which takes a
            # parameter and names the rdim even where this declines.
            tiled = self.tiled_subscript(
                buffer, shifted, axes, dict(self.extents), output
            )
        if tiled is None:
            return None
        # tile_r's coordinate is the whole rdim, ":", in a persistent rdim.
        arg, coordinates, tiles = tiled
        coordinates = list(coordinates)
        coordinates[tiles.index("tile_r")] = f"{first}:{first + length}"
        return arg, coordinates, tiles

    def quotient_subscript(
        self, buffer: IRNode, index: sympy.Expr, axes: Sequence[str]
    ) -> tuple[HelionTensorArg, list[str], TileShape] | None:
        """The parameter, subscript and shape that read `buffer` at an index
        that takes the quotient of the first tile's index by a constant, and
        is affine in it and in the other tiles: a view of the storage whose
        first dim the quotient indexes, as gather_subscript() does for a value
        read from memory. repeat_kv's expand of k, stored token-major [S, Hk,
        D] with Hk = 4 and D = 256, to Hk * n heads, x + 256*(z // 2) +
        1024*y, would read flat,
            in0[tile_x.index + 256*(tile_z.index.reshape([tile_z, 1, 1]) // 2)
                + 1024*tile_y.index.reshape([tile_y, 1])]
        one element per lane, where the [Hk, S, D] view subscripted by the
        quotient and the tiles,
            in0[tile_z.index // 2, tile_y, tile_x]
        reads consecutive elements along x, which the backend vectorizes."""
        quotients = index.atoms(FloorDiv)
        if len(quotients) != 1:
            return None
        ((quotient),) = quotients
        numerator, divisor = quotient.args
        axis_of = {symbol: axis for axis, symbol in self.symbols.items()}
        if (
            numerator not in axis_of
            or not isinstance(divisor, sympy.Integer)
            or divisor <= 1
        ):
            return None
        axis = axis_of[numerator]
        if axis == "r0_":
            # A reduced dim's quotient (w.repeat_interleave(2) read at r // 2)
            # is a one-dim gather either way: the flat read keeps it.
            return None
        head = sympy.Symbol("quotient", integer=True, nonnegative=True)
        affine = index.xreplace({quotient: head})
        if numerator in affine.free_symbols:
            return None
        extent = self.extents[axis]
        if not isinstance(extent, (int, sympy.Integer)):
            return None
        size = sympy.Integer(-(-int(extent) // int(divisor)))
        code = f"{TILE_NAMES[axis]}.index // {int(divisor)}"
        return self.leading_subscript(buffer, affine, axes, (head, size, code), [axis])

    def leading_subscript(
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        lead: tuple[sympy.Symbol, sympy.Expr, str],
        lead_axes: Sequence[str],
    ) -> tuple[HelionTensorArg, list[str], TileShape] | None:
        """The parameter, subscript and shape that read `buffer` at `index`
        through a view whose first dim a leading index indexes, `lead`: its
        symbol in `index`, the dim's size and its code, a value read from
        memory (gather_subscript()) or a tile's quotient (quotient_subscript()).
        The view's other dims are the tiles of `axes` the leading index's
        tiles `lead_axes` leave, which must come after those in the kernel's
        axis order: Helion puts the leading index's tiles first in the
        result."""
        symbol, size, code = lead
        other = [axis for axis in axes if axis not in lead_axes]
        if [axis for axis in self.axes if axis in axes] != [*lead_axes, *other]:
            return None
        view = self.affine_view(
            buffer,
            index,
            [symbol, *(self.symbols[axis] for axis in other)],
            [size, *(self.extents[axis] for axis in other)],
        )
        if view is None:
            return None
        tiles = [TILE_NAMES[axis] for axis in other]
        coordinates = [code, *map(self.coordinate, tiles)]
        return self.tensor_arg(view), coordinates, self.value_shape(axes)

    def gather_subscript(
        self, buffer: IRNode, index: sympy.Expr, axes: Sequence[str]
    ) -> tuple[HelionTensorArg, list[str], TileShape] | None:
        """The parameter, subscript and shape that read `buffer` at an index of
        one value i, as in w[i] of an embedding table w [V, E]: a view of the
        storage whose first dim i indexes and whose other dims are tiles the
        value does not have. Read at 1024*tmp1 + r, with tmp1 = i[tile_y,
        tile_x], the flat subscript
            in1[hl.arange(1024) + 1024*tmp1.reshape([tile_y, tile_x, 1])]
        becomes the [V, 1024] view of w subscripted by the value and the rdim:
            in1[tmp1, :]
        whose loads of an rdim's consecutive elements the backend can vectorize.
        Helion puts the value's tiles first in the result, then the other
        tiles, so those must come after the value's in the kernel's axis
        order; and an index that also reads one of the value's tiles, as in
        x.gather(1, i), is a gather along a tile, which stays flat."""
        indirect = self.indirect_vars(index)
        if len(indirect) != 1:
            return None
        ((symbol, value),) = indirect.items()
        size = self.indirect_sizes[symbol]
        # Helion keeps a size-1 dim of the value: tmp1 of [tile_y, 1] would
        # read [tile_y, 1, tile_x] beside the kernel's [tile_y, tile_x].
        if any(str(dim) == "1" for dim in value.shape or ()):
            return None
        value_axes = tile_axes(value.shape)
        if not value_axes or any(
            self.symbols[axis] in index.free_symbols for axis in value_axes
        ):
            return None
        return self.leading_subscript(
            buffer, index, axes, (symbol, size, str(value)), value_axes
        )

    def note_rdim(self, arg: HelionTensorArg, coordinates: list[str]) -> None:
        """Keep the first full rdim of an input (in0[tile_x, :]) as the rdim
        codegen_rdim() defines rdim by."""
        if self.rdim_anchor is None and ":" in coordinates and not arg.output:
            self.rdim_anchor = (arg.ref, coordinates)

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
        if not (self.loads or self.compute or self.stores):
            return
        # A persistent reduction has no loop: one body, so that a value stored
        # and read again in the kernel comes from the CSE store cache, and a
        # buffer only the kernel reads is not written (remove_kernel_local_buffers()).
        if self.rolled:
            self.body.writeline(f"for tile_r in hl.tile({self.numels['r0_']}):")
            with self.body.indent():
                self.body.splice(self.loads)
                self.body.splice(self.compute)
            self.body.splice(self.stores)
            # Values from the loop cannot be carried to the next top-level loop.
            self.cse.invalidate(self.outer_vars)
        else:
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.loads.clear()
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
        kernel_params = self.kernel_params()
        # The parameter name of each argument's placeholder.
        names = {arg.ref: param.name for param in kernel_params for arg in param.args}
        params = [param.name for param in kernel_params]
        if self.mix_order_reduction:
            # The [partials, splits, r] workspace of the partials of the
            # reduction over x (codegen_split_x()).
            params.append("ws")
        signature = ", ".join(f"{name}: torch.Tensor" for name in params)
        code.writeline(f"def kernel({signature}) -> None:")
        with code.indent():
            if self.mix_order_reduction:
                self.codegen_split_x(code)
            else:
                tiles = ", ".join(TILE_NAMES[axis] for axis in self.outer_axes)
                sizes = ", ".join(str(self.extents[axis]) for axis in self.outer_axes)
                sizes = sizes if len(self.outer_axes) == 1 else f"[{sizes}]"
                code.writeline(f"for {tiles} in hl.tile({sizes}):")
                with code.indent():
                    self.codegen_rdim(code, TILE_NAMES[self.outer_axes[-1]])
                    code.splice(self.body)
        return re.sub(r"__arg\d+__", lambda ref: names[ref[0]], code.getvalue())

    def codegen_rdim(self, code: IndentedBuffer, tile: str) -> None:
        """Define rdim (rdim_code()): for an rdim of more than UNROLLED_RDIM_MAX
        elements, the dim of the first full rdim the kernel reads, as in rdim =
        in0[tile_x, :].size(1), whose load Helion drops as only its size is
        used; else the dim of a factory beside `tile`, hl.zeros([tile_x,
        300]).size(1), which the rdim shares but Helion cannot roll. A mix-order
        kernel defines rdim outside its loop over tile_x, where an rdim of an
        operand has no tile_x to read it at."""
        if not self.uses_rdim:
            return
        rolled = self.numels["r0_"] > UNROLLED_RDIM_MAX
        if self.rdim_anchor is not None and not self.mix_order_reduction and rolled:
            ref, coordinates = self.rdim_anchor
            dim = coordinates.index(":")
            code.writeline(f"rdim = {ref}[{', '.join(coordinates)}].size({dim})")
        else:
            size = self.numels["r0_"]
            code.writeline(f"rdim = hl.zeros([{tile}, {size}]).size(1)")

    def codegen_split_x(self, code: IndentedBuffer) -> None:
        """The loops of a mix-order reduction (split_x()): the body over the x
        of each split, and the terms of the reduction over x (one partial per
        partial_accumulate()) summed over them, as for RMSNorm's backward:
            for tile_y in hl.tile(9000):
                rdim = hl.zeros([tile_y, 512]).size(1)
                part0 = hl.zeros([tile_y, rdim], dtype=torch.float32)
                for tile_x in hl.tile(128):
                    ...
                    part0 = part0 + torch.sum(tmp25, 1)
                ws[0, tile_y, :] = part0
        Helion reduces only the x a partial tile_x holds: torch.sum over a
        tile's dim masks the lanes past its end."""
        rdim = self.rdim_code()  # before codegen_rdim(), as it can name rdim
        shape = self.value_shape(["y", "x", "r0_"])
        code.writeline(f"for tile_y in hl.tile({self.extents['y']}):")
        with code.indent():
            self.codegen_rdim(code, "tile_y")
            for index, partial in enumerate(self.saved_partial_accumulate):
                if partial.reduction_type == "sum":
                    line = f"hl.zeros([tile_y, {rdim}], dtype=torch.float32)"
                else:
                    line = f"hl.full([tile_y, {rdim}], 1.0, dtype=torch.float32)"
                code.writeline(f"part{index} = {line}")
            code.writeline(f"for tile_x in hl.tile({self.extents['x']}):")
            with code.indent():
                code.splice(self.body)
                for index, partial in enumerate(self.saved_partial_accumulate):
                    value = partial.value
                    term = self.align(value, shape)
                    if tuple(value.shape or ()) != shape:
                        term = f"{term}.expand({self.dims_code(shape)})"
                    if value.dtype != torch.float32:
                        term = f"{term}.to(torch.float32)"
                    fn = REDUCTION_FUNCTIONS[partial.reduction_type]
                    combine = "+" if partial.reduction_type == "sum" else "*"
                    code.writeline(
                        f"part{index} = part{index} {combine} torch.{fn}({term}, 1)"
                    )
            for index in range(len(self.saved_partial_accumulate)):
                code.writeline(f"ws[{index}, tile_y, :] = part{index}")

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

    def kernel_params(self) -> list[HelionParam]:
        """The kernel's parameters, inputs first.

        An input whose storage an output took over (an in-place buffer) is
        passed as that output. Arguments passed the same storage at the same
        view are one parameter, named in_out0 for an input and an output, as
        Triton's in_out_ptr0, rather than one each, as for the RMSNorm after a
        matmul whose output it adds to (buf1 takes over the matmul's buf0):
            def kernel(in0, in1, in2, out0, out1):
                tmp1 = in1[tile_x, :]
                ...
                out0[tile_x, :] = tmp2.to(torch.bfloat16)
            kernel(arg2_1, buf1, arg3_1, buf1, buf3)
        is
            def kernel(in0, in_out0, in1, out0):
                tmp1 = in_out0[tile_x, :]
                ...
                in_out0[tile_x, :] = tmp2.to(torch.bfloat16)
            kernel(arg2_1, buf1, arg3_1, buf3)
        CuTe must assume two parameters alias at any offset, and rejects each
        config that reduces an rdim over fewer threads than its elements
        ("synthetic-lane reduction would reorder a potentially aliasing
        write"): with in1 and out0, only num_threads=[1, 512] (a thread per
        element of the rdim of 512) compiles; with in_out0, also
        num_threads=[2, 32] (a warp per rdim), as Triton's kernel reduces it.
        """
        # Reuse can chain inside one kernel: with buf1 reusing buf0 and buf2
        # reusing buf1, the wrapper runs `buf1 = buf0; del buf0` and
        # `buf2 = buf1; del buf1`, so buf0 is passed as buf2.
        reused = {inp: out for out, inp in self.inplace_update_buffers.items()}
        groups: dict[tuple[Any, ...], tuple[str, list[HelionTensorArg]]] = {}
        for arg in self.kernel_args():
            operand = arg.operand
            name = operand.get_name()
            while name in reused:
                name = reused[name]
            if name != operand.get_name():
                target = V.graph.get_buffer(name)
                operand = ReinterpretView(data=target, layout=operand.get_layout())
            reference = (
                operand.codegen_reference()
                if isinstance(operand, ReinterpretView)
                else operand.get_name()
            )
            layout = operand.get_layout()
            view = (
                name,
                layout.dtype,
                tuple(layout.size),
                tuple(layout.stride),
                layout.offset,
            )
            groups.setdefault(view, (reference, []))[1].append(arg)
        counts: collections.Counter[str] = collections.Counter()
        params: list[HelionParam] = []
        for reference, args in groups.values():
            kinds = {arg.output for arg in args}
            prefix = "in_out" if len(kinds) == 2 else "out" if True in kinds else "in"
            params.append(
                HelionParam(f"{prefix}{counts[prefix]}", reference, tuple(args))
            )
            counts[prefix] += 1
        return params

    def call_args(self) -> list[str]:
        return [param.reference for param in self.kernel_params()]

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
        args = self.call_args()
        for workspace in self.args.workspace_args:
            wrapper.generate_workspace_allocation(workspace)
        if self.mix_order_reduction:
            # The workspace as [partials, splits, r] (codegen_split_x()).
            (workspace,) = self.args.workspace_args
            dims = (
                len(self.saved_partial_accumulate),
                self.extents["y"],
                self.numels["r0_"],
            )
            args.append(f"{workspace.outer_name}.view{dims}")
        wrapper.writeline(wrapper.wrap_kernel_call(name, args))
        if deallocate_ws:
            self.deallocate_workspaces()

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
            bounds = None if mask is None else self.masked_extents(mask)
            arg, coordinates, shape = self.subscript(buffer, index, axes, bounds=bounds)
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
        # As Inductor's Triton kernels do (TritonKernel.get_load_buffer()), the
        # loads that need no computed value come first, then the computation:
        # a dropout's seeds, tmp0 = in0[()] and tmp5 = in1[()], are read before
        # the first draw, where CuTe takes a persistent rdim only when no 0-d
        # load sits between the lanes' statements of a sweep. A masked load
        # (its mask a computed value) or an indirect one (its index read from
        # memory) follows that computation.
        buffer = (
            self.compute
            if mask is not None or self.indirect_vars(index)
            else self.loads
        )
        return self.cse.generate(buffer, line, dtype=dtype, shape=shape)

    def is_number(self, value: object) -> bool:
        """Whether `value` is a CSE variable that holds a Python number."""
        return isinstance(value, CSEVariable) and value.name in self.numbers

    def as_tensor(self, value: _T) -> _T:
        """`value`, or for a variable that holds a Python number, a 0-d tensor
        of it, for an op that takes no Python number:
            tmp2 = 0.0
            tmp3 = hl.full([], tmp2, dtype=torch.float32)
            tmp4 = torch.maximum(tmp1, tmp3)"""
        if not self.is_number(value):
            return value
        assert isinstance(value, CSEVariable) and value.dtype is not None
        line = f"hl.full([], {value}, dtype={value.dtype})"
        return cast(
            "_T", self.cse.generate(self.compute, line, dtype=value.dtype, shape=())
        )

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: str | None = None,
    ) -> None:
        if mode not in (None, "atomic_add"):
            raise NotImplementedError(f"Helion kernels do not support {mode} stores")
        value = self.as_tensor(value)
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
        if mode == "atomic_add":
            # Each element the node iterates over adds once, so an atomic
            # store has every tile of the node's, also one that neither the
            # index nor the value has. index_put(out, (arange(n) * 0,),
            # ones(n), True) adds 1.0 at out[0] n times,
            #     hl.atomic_add(out0, [tile_x], tmp1)  # tmp1: 1.0, 0-d
            # where hl.atomic_add(out0, [], tmp1) added it once per tile:
            # out[0] was 32, not 1024, in blocks of 32.
            store_axes |= OrderedSet(
                axis for axis in self.axes if axis != "r0_" or self.in_reduction
            )
        axes = [axis for axis in self.axes if axis in store_axes]
        arg, coordinates, shape = self.subscript(buffer, index, axes, output=True)
        dtype = buffer.get_dtype()
        cast_suffix = "" if value.dtype == dtype else f".to({dtype})"
        buffer_code = self.compute if self.in_reduction else self.stores
        value_code = self.align(value, shape)
        value_axes = tile_axes(value.shape)
        aligned = scope_shape(value_axes, tile_axes(shape))
        if value_axes and len(aligned) < len(shape):
            # A value without the store's leading tiles, which align() leaves
            # out, expanded over them. x.expand(64, 7, 31) * 2 reads x's
            # stride-0 dim with no tile:
            #     tmp0 = in0[tile_x]
            #     out0[tile_y, tile_x] = tmp2
            # fails with "Expected ndim=2, but got ndim=1", where
            #     out0[tile_y, tile_x] = tmp2.expand(tile_y, tile_x)
            # stores tmp2 at every y. A value align() gives the store's rank
            # stores as it is, as out0[tile_x, :] = tmp3.reshape([tile_x, 1])
            # does, and so does a 0-d value (out0[tile_x] = tmp1 of tmp1 =
            # hl.full([], 0.0, ...)): Helion broadcasts both.
            value_code = f"{value_code}.expand({self.dims_code(shape)})"
        value_code += cast_suffix
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
        whole rdim is slice(None) and a segment of it (segment_subscript())
        slice(lo, hi): a masked load through its buffer's own layout in a
        persistent reduction is
            tmp0 = hl.load(in0, [tile_x, slice(None)], extra_mask=tmp3)
        where hl.load(in0, [tile_x, :], extra_mask=tmp3) is a SyntaxError."""

        def entry(coordinate: str) -> str:
            if coordinate == ":":
                return "slice(None)"
            if bounds := re.fullmatch(r"(\d+):(\d+)", coordinate):
                return f"slice({bounds[1]}, {bounds[2]})"
            return coordinate

        return ", ".join(map(entry, coordinates))

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        index = self.tile_index(expr)
        dtype = upcast_compute_type(dtype)
        if not index.free_symbols:
            value = int(index) if index.is_Integer else float(index)
            line = HelionOverrides.constant(value, dtype)
            var = self.cse.generate(self.compute, line, dtype=dtype, shape=())
            if number_constant(value, dtype):
                self.numbers.add(var.name)
        else:
            axes = self.index_axes(index)
            var = self.cse.generate(
                self.compute,
                f"({self.index_tensor_expr(index, axes)}).to({dtype})",
                dtype=dtype,
                shape=self.value_shape(axes),
            )
        if dtype != torch.bool and not dtype.is_floating_point:
            # Only where the cast keeps every value: int8(tile_x.index) < 100
            # holds at 128..255 too, where the cast wraps negative.
            bound = bound_sympy(index, self.value_ranges(self.symbols))
            info = torch.iinfo(dtype)
            sizevars = V.graph.sizevars
            if sizevars.statically_known_geq(
                bound.lower, info.min
            ) and sizevars.statically_known_leq(bound.upper, info.max):
                self.symbolic[var.name] = index
        return var

    def note_predicate(
        self, name: str, args: tuple[Any, ...], result: CSEVariable
    ) -> None:
        """Keep what `result` of op `name` says of the tiles: the integer of a
        constant, and the comparisons of indices and integers (and their
        conjunctions) a mask is, as the mask tile_x.index < 30522 of a pad of
        x from 30522 to 30528 (masked_extents())."""

        def value(arg: object) -> sympy.Expr | None:
            if isinstance(arg, CSEVariable):
                return self.symbolic.get(arg.name)
            if isinstance(arg, int) and not isinstance(arg, bool):
                return sympy.Integer(arg)
            return None

        if name == "constant" and isinstance(args[0], bool):
            # A bool constant is a comparison that always or never holds.
            one, zero = sympy.Integer(1), sympy.Integer(0)
            self.predicates[result.name] = [
                sympy.Lt(zero, one, evaluate=False)
                if args[0]
                else sympy.Lt(one, zero, evaluate=False)
            ]
        elif name == "constant":
            number, dtype = args[0], args[1]
            if (
                isinstance(number, int)
                and not isinstance(number, bool)
                and dtype != torch.bool
                and not dtype.is_floating_point
                and torch.iinfo(dtype).min <= number <= torch.iinfo(dtype).max
            ):
                self.symbolic[result.name] = sympy.Integer(number)
        elif name in ("lt", "le", "gt", "ge") and len(args) == 2:
            lhs, rhs = value(args[0]), value(args[1])
            if lhs is not None and rhs is not None:
                relation = {
                    "lt": sympy.Lt,
                    "le": sympy.Le,
                    "gt": sympy.Gt,
                    "ge": sympy.Ge,
                }
                self.predicates[result.name] = [
                    relation[name](lhs, rhs, evaluate=False)
                ]
                self.note_rdim_cut(self.predicates[result.name][0])
                self.note_axis_cut(self.predicates[result.name][0])
        elif name in ("and_", "logical_and") and len(args) == 2:
            parts = [
                self.predicates.get(arg.name) if isinstance(arg, CSEVariable) else None
                for arg in args
            ]
            if parts[0] is not None and parts[1] is not None:
                self.predicates[result.name] = [*parts[0], *parts[1]]

    def note_rdim_cut(self, relation: sympy.Rel) -> None:
        """Keep where `relation`, a comparison of a persistent rdim's index with
        an integer, cuts the rdim, by the node it is of: r < 128 at 128, the
        cut of rotate_half's cat([-y[..., 128:], y[..., :128]], -1)."""
        if not self.persistent_reduction or self.segment is not None:
            return
        node = self.current_node
        if node is None or "r0_" not in self.symbols:
            return
        lhs, rhs, strict = as_less(relation)
        r = self.symbols["r0_"]
        if lhs == r and isinstance(rhs, sympy.Integer):
            cut = int(rhs) + (0 if strict else 1)
        elif rhs == r and isinstance(lhs, sympy.Integer):
            cut = int(lhs) + (1 if strict else 0)
        else:
            return
        extent = self.extents["r0_"]
        if isinstance(extent, (int, sympy.Integer)) and 0 < cut < int(extent):
            self.rdim_cuts.setdefault(node.get_name(), OrderedSet()).add(cut)

    def note_axis_cut(self, relation: sympy.Rel) -> None:
        """Keep where `relation`, a comparison of a pointwise axis's index with
        an integer, cuts the axis: c < 60 at 60, the cut of a cat of two
        [N, 60, H, W] parts along channels (pointwise_segments())."""
        if self.axis_segment is not None or self.mix_order_reduction:
            return
        lhs, rhs, strict = as_less(relation)
        axis_of = {self.symbols[axis]: axis for axis in self.outer_axes}
        if lhs in axis_of and isinstance(rhs, sympy.Integer):
            axis, cut = axis_of[lhs], int(rhs) + (0 if strict else 1)
        elif rhs in axis_of and isinstance(lhs, sympy.Integer):
            axis, cut = axis_of[rhs], int(lhs) + (1 if strict else 0)
        else:
            return
        extent = self.extents[axis]
        if isinstance(extent, (int, sympy.Integer)) and 0 < cut < int(extent):
            self.axis_cuts.setdefault(axis, OrderedSet()).add(cut)

    def pointwise_segments(self) -> tuple[str, list[tuple[int, int]]] | None:
        """The pointwise axis and the equal segments of it to emit the whole
        body over (pointwise_segment()), when the cuts of one axis
        (note_axis_cut()) make 2 to RDIM_SEGMENTS_MAX segments of one length,
        at least RDIM_SEGMENT_MIN:
        the parts of cat([x, relu(bn(y))], 1) of two [512, 60, 28, 28]
        channels-last parts, cut at 60 of 120 channels. None for cuts on
        several axes, unequal segments (a pad's 355 and 23567 of 23923), and
        for a kernel that loops over tile_r, whose loop the body's emissions
        would share."""
        if self.rolled_reduction() or len(self.axis_cuts) != 1:
            return None
        ((axis, cuts),) = self.axis_cuts.items()
        extent = self.extents[axis]
        if not isinstance(extent, (int, sympy.Integer)):
            return None
        points = [0, *sorted(cuts), int(extent)]
        segments = list(itertools.pairwise(points))
        lengths = {hi - lo for lo, hi in segments}
        if (
            not 2 <= len(segments) <= RDIM_SEGMENTS_MAX
            or len(lengths) != 1
            or min(lengths) < RDIM_SEGMENT_MIN
        ):
            return None
        return axis, segments

    def rolled_reduction(self) -> bool:
        """Whether the kernel reduces in a loop over tile_r (not persistent)."""
        return self.features.is_reduction() and not self.persistent_reduction

    def segment_axis(self, axis: str, segments: list[tuple[int, int]]) -> None:
        """Emit the kernel over the equal `segments` of pointwise `axis`
        (pointwise_segments()): its tile spans one segment, and the body is
        emitted once for each (pointwise_segment())."""
        length = segments[0][1] - segments[0][0]
        self.axis_segments = (axis, segments)
        self.extents[axis] = sympy.Integer(length)
        self.tile_ranges[self.symbols[axis]] = sympy.Integer(length)

    @contextlib.contextmanager
    def pointwise_segment(self, start: int) -> Iterator[None]:
        """Emit within the segment of the segmented axis (segment_axis()) that
        starts at `start`: its tile counts from `start` (tile_index()), so that
        the comparisons of a cat's mask are proven (proven()) and its parts are
        read and written as views of their parameters. For the GhostNet cat
        of x and relu(bn(y)) along channels and its mean over H and W, the
        whole channel axis of 120 reads y flat, behind a mask:
            for tile_y, tile_x in hl.tile([512, 120]):
                tmp5 = hl.load(in0, [tile_y, slice(None), tile_x], extra_mask=tmp4)
                tmp11 = hl.load(in1, [(-60) + tile_x.index + 60*hl.arange(rdim)
                    .reshape([rdim, 1]) + 47040*tile_y.index...], extra_mask=tmp8)
        and its two segments of 60 read both parts as tiles, the second
        stored through a view of the output that starts at channel 60:
            for tile_y, tile_x in hl.tile([512, 60]):
                tmp0 = in0[tile_y, :, tile_x]
                out0[tile_y, :, tile_x] = tmp0
                ...
                tmp11 = in1[tile_y, :, tile_x].to(torch.float32)
                ...
                out2[tile_y, :, tile_x] = tmp21.to(torch.bfloat16)"""
        assert self.axis_segments is not None
        axis = self.axis_segments[0]
        saved = self.axis_segment
        self.axis_segment = (axis, start)
        try:
            yield
        finally:
            self.axis_segment = saved

    def rdim_cut_segments(self, names: Iterable[str]) -> list[tuple[int, int]] | None:
        """The segments the cuts of the nodes `names` (note_rdim_cut()) make of
        the rdim, when each is a power of two of at least RDIM_SEGMENT_MIN
        elements (whose literal dim Helion takes, rdim_code()); else None."""
        cuts = OrderedSet(cut for name in names for cut in self.rdim_cuts.get(name, ()))
        extent = self.extents.get("r0_")
        if not cuts or not isinstance(extent, (int, sympy.Integer)):
            return None
        points = [0, *sorted(cuts), int(extent)]
        segments = list(itertools.pairwise(points))
        if len(segments) > RDIM_SEGMENTS_MAX or any(
            hi - lo < RDIM_SEGMENT_MIN or (hi - lo) & (hi - lo - 1)
            for lo, hi in segments
        ):
            return None
        return segments

    @contextlib.contextmanager
    def rdim_segment(self, lo: int, hi: int) -> Iterator[None]:
        """Emit within the segment [lo, hi) of a persistent rdim: its tile
        counts from lo (tile_index()) over hi - lo elements, so that the
        comparisons of a cat's or pad's mask are proven (proven()) and its
        parts are read and written as slices of the whole rdim's parameters
        (segment_subscript()), as rotate_half after an RMSNorm (Gemma's q/k
        norm and RoPE), where the whole rdim would read the halves as flat
        masked loads:
            tmp8 = hl.load(in1, [128 + hl.arange(rdim) + 256*tile_x.index...],
                           extra_mask=tmp7)              # r < 128
        and the segments read and write them as the hand-written kernel does,
            tmp8 = in0[tile_y, tile_x, 128:256]
            out0[tile_y, tile_x, 0:128] = ...
        HelionScheduling.create_kernel_choices() picks the segments."""
        self.rdim_extent = int(self.extents["r0_"])
        with self.rdim_state(hi - lo, (lo, hi)):
            yield

    @contextlib.contextmanager
    def rdim_state(
        self, extent: int, segment: tuple[int, int] | None
    ) -> Iterator[None]:
        """Within the block, tile_r spans `extent` elements of the rdim and the
        kernel emits within `segment` (None: the whole rdim); rdim_segment() and
        segment_subscript() each set and restore these together."""
        r = self.symbols["r0_"]
        saved = (self.extents["r0_"], self.tile_ranges[r], self.segment)
        self.extents["r0_"] = self.tile_ranges[r] = sympy.Integer(extent)
        self.segment = segment
        try:
            yield
        finally:
            self.extents["r0_"], self.tile_ranges[r], self.segment = saved

    def masked_extents(self, mask: CSEVariable) -> dict[str, sympy.Expr]:
        """The tile extents `mask` bounds below the kernel's: the axes it
        proves below a number, as tile_x.index < 30522 bounds tile_x of a pad
        to 30528 at 30522."""
        extents: dict[str, sympy.Expr] = {}
        axis_of = {symbol: axis for axis, symbol in self.symbols.items()}
        for relation in self.predicates.get(mask.name, ()):
            lhs, rhs, strict = as_less(relation)
            # lhs < rhs, or lhs <= rhs
            if lhs in axis_of and isinstance(rhs, sympy.Integer):
                bound = sympy.Integer(int(rhs) + (0 if strict else 1))
                axis = axis_of[lhs]
                # Not a persistent rdim, whose whole rdim in0[tile_x, :] would
                # be the view's, shorter than the kernel's.
                if self.coordinate(TILE_NAMES[axis]) == ":":
                    continue
                current = extents.get(axis, self.extents[axis])
                if bound > 0 and V.graph.sizevars.statically_known_lt(bound, current):
                    extents[axis] = bound
        return extents

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
        ranges = self.value_ranges(axes)
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
        # A value read as an index broadcasts like a tile index: in
        # F.embedding(i, w).t(), over [tile_y, tile_x], tmp1 of tmp0 =
        # in0[tile_x] indexes w at
        #     in1[tile_y.index.reshape([tile_y, 1]) + 64*tmp1]
        # where the value is [tile_x] beside [tile_y, 1]. (A gather whose
        # value's tiles come first is a view instead, in1[tmp1, tile_r] or
        # in1[tmp1, :], gather_subscript().)
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

    def value_ranges(self, axes: Iterable[str]) -> dict[sympy.Symbol, ValueRanges]:
        """The range of each tile of `axes`' index, 0 to its extent - 1, as
        bound_sympy() takes it."""
        return {
            cast("sympy.Symbol", self.symbols[axis]): ValueRanges(
                0, self.extents[axis] - 1
            )
            for axis in axes
        }

    def proven_choice(self, condition: object, a: object, b: object) -> object | None:
        """The operand torch.where(condition, a, b) takes on every lane when the
        tiles' ranges prove `condition` (proven()), if it is a value of the
        where's dtype and shape; else None. Within a cat's segment
        (pointwise_segment()) the select of the parts,
            tmp25 = torch.where(tmp4, tmp7, tmp24)
        with tmp4 = tile_x.index + 60 < 60, false on every lane of the second
        segment, is the second part's value tmp24, with no select."""
        if not isinstance(condition, CSEVariable):
            return None
        truth = self.proven(condition)
        if truth is None:
            return None
        chosen, other = (a, b) if truth else (b, a)
        if not isinstance(chosen, CSEVariable) or self.is_number(chosen):
            return None
        shapes = {
            tuple(value.shape or ())
            for value in (condition, chosen, other)
            if isinstance(value, CSEVariable) and not self.is_number(value)
        }
        widest = max(shapes, key=len)
        if tuple(chosen.shape or ()) != widest or not all(
            set(tile_axes(shape)) <= set(tile_axes(widest)) for shape in shapes
        ):
            return None
        if isinstance(other, CSEVariable) and other.dtype != chosen.dtype:
            return None
        return chosen

    def proven(self, condition: CSEVariable) -> bool | None:
        """Whether the tiles' ranges make `condition` true on every lane (True)
        or false (False), from the comparisons note_predicate() kept of it;
        None if they do neither or it kept none. Inductor's value ranges of the
        variable would not do: they ignore an integer cast's wrap, so the
        condition int8(tile_x.index) < 0 over tiles of 128 to 255 has the
        ranges [False, False], where every lane is true."""
        relations = self.predicates.get(condition.name)
        if not relations:
            return None
        ranges = self.value_ranges(self.symbols)
        known = V.graph.sizevars
        truths = []
        for relation in relations:
            lhs, rhs, strict = as_less(relation)
            # lhs - rhs, against 0
            bound = bound_sympy(sympy.Add(lhs, sympy.Mul(-1, rhs)), ranges)
            low, high = bound.lower, bound.upper
            if strict:
                truth = (
                    known.statically_known_lt(high, 0),
                    known.statically_known_geq(low, 0),
                )
            else:
                truth = (
                    known.statically_known_leq(high, 0),
                    known.statically_known_gt(low, 0),
                )
            truths.append(True if truth[0] else False if truth[1] else None)
        if False in truths:
            return False
        return True if all(truths) else None

    def in_range(self, buffer: IRNode, index: sympy.Expr) -> bool:
        """Whether a direct `index` is within `buffer`'s storage on every lane."""
        bound = bound_sympy(index, self.value_ranges(self.symbols))
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
        """body() where `mask` is true, else `other`; body() alone when the
        tiles' ranges prove `mask` true (proven()), as for the tile_x.index < 100
        of (torch.cat([x, y], -1) * 2)[:, :50] over tiles of x < 50 only,
        where the masked load
            tmp4 = tmp0 < tmp3
            tmp5 = hl.load(in0, [100*tile_y.index.reshape([tile_y, 1])
                + (tile_x.index)], extra_mask=tmp4)
            tmp6 = 0.0
            tmp7 = torch.where(tmp4, tmp5, tmp6)
        becomes tmp5 = in0[tile_y, tile_x], with no where."""
        truth = self.proven(mask)
        if truth is True:
            return body()
        self.load_masks.append(mask)
        try:
            result = body()
        finally:
            self.load_masks.pop()
        if result.dtype is None:
            raise AssertionError(f"expected a dtype for masked value {result}")
        if truth is False:
            # Proven false, as cat's mask for the sink over those tiles: the
            # value is `other`, and Helion drops the body's unused loads.
            return V.ops.constant(other, result.dtype)
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

    def dot(
        self, lhs: CSEVariable, rhs: CSEVariable, rounded: torch.dtype | None
    ) -> CSEVariable:
        """`lhs @ rhs`, whose product is rounded to `rounded` (rounded_dot_dtype())
        before anything else reads it."""
        batch = ["z0_"] if "z0_" in self.numels else []
        lhs = self.dot_operand(lhs, [*batch, "y", "r0_"])
        rhs = self.dot_operand(rhs, [*batch, "r0_", "x"])
        self.has_fp32_dot |= torch.float32 in (lhs.dtype, rhs.dtype)
        if self.persistent_reduction and lhs.dtype == rhs.dtype == rounded:
            # K in one block: one matmul of the whole rows, which accumulates in
            # fp32 and rounds once, as the store of an accumulator would:
            #     tmp2 = torch.bmm(tmp0, tmp1)
            #     out0[tile_z, tile_y, tile_x] = tmp2
            # rather than
            #     acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
            #     acc = torch.baddbmm(acc, tmp0, tmp1)
            #     out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)
            # Both round the fp32 sum once, so the first gives the same result
            # and leaves the backend a plain matmul, which its matmul path can
            # take as it stands. A product read in fp32 (rounded is None), as
            # by torch.mm(a, b, out_dtype=torch.float32), keeps the
            # accumulator: torch.mm of bf16 tiles would round it to bf16.
            op = "bmm" if batch else "mm"
            tiles = tuple(TILE_NAMES[axis] for axis in [*batch, "y", "x"])
            line = f"torch.{op}({lhs}, {rhs})"
            return self.cse.generate(self.compute, line, dtype=rounded, shape=tiles)
        # Each dot has its own accumulator, so horizontally fused dots stay apart.
        acc = self.accumulator(0, torch.float32)
        op = "baddbmm" if batch else "addmm"
        self.compute.writeline(f"{acc} = torch.{op}({acc}, {lhs}, {rhs})")
        self.dot_results.add(acc)
        return acc

    def dot_operand(self, value: CSEVariable, axes: Sequence[str]) -> CSEVariable:
        """`value` expanded to the matrix tile over `axes`. x @ torch.ones(72, 136)
        multiplies by a constant, and with its 0-d tensor (HelionCSEProxy.dot()),
        tmp2 = hl.full([], tmp1, ...) of tmp1 = 1.0, torch.addmm(acc, tmp0, tmp2)
        fails with "b must be 2D"."""
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
        value = (
            tuple(self.as_tensor(v) for v in value)
            if isinstance(value, tuple)
            else self.as_tensor(value)
        )
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
        # Reduce tile_r wherever the axis order puts it: last without a dot or an
        # outer reduction, before tile_x in an outer reduction (torch.sum(tmp0,
        # -2) of in0[tile_y, tile_r, tile_x]) and in a matmul's K loop, as in
        # (a @ b, (a[:, :, None] * b[None]).sum(1)):
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
            tmp7 = tmp3 == tmp1                       # block_max == -inf, per x
            tmp8 = tmp7.reshape([tile_x, 1])
            tmp9 = torch.where(tmp8, tmp2, tmp6)      # term
            tmp10 = torch.sum(tmp9, -1)               # block_sum
            tmp11 = torch.maximum(acc, tmp3)          # new_max
            ...
            tmp15 = torch.exp(tmp14)                  # scale: exp(-|acc - tmp3|)
            ...
            tmp21 = torch.where(tmp17, tmp2, tmp20)   # acc_part: acc == -inf ? 0 : ...
            ...
            tmp28 = torch.where(tmp25, tmp26, tmp27)  # new_sum: tmp11 == -inf ? nan : ...
            acc1 = tmp28
            acc = tmp11
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
        if weight:
            # A partial (max, sum) pair of a split: one of max -inf adds
            # nothing, also where its sum is nan (a part of an rdim of only -inf).
            term = ops.where(ops.eq(value, neg_inf), zero, ops.mul(weight[0], exp))
        else:
            # Only a block of only -inf needs the 0: elsewhere exp(-inf - m) is
            # 0 already. One comparison per x of the block, where x == -inf
            # would compare each element:
            #     tmp7 = tmp3 == tmp1                 # tmp3: block_max, [tile_x]
            #     tmp8 = tmp7.reshape([tile_x, 1])
            #     tmp9 = torch.where(tmp8, tmp2, tmp6)
            term = ops.where(ops.eq(block_max, neg_inf), zero, exp)
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
        reduced to its own weight, mean and m2, welford_reduce's in two sums
        over the block in registers (the mean, then the deviations' spread),
        welford_combine's from its partials (their weights' sum, their weighted
        mean corrected by their deviations from it, their m2 plus the spread of
        their means), and folded into the accumulators acc_mean, acc_m2 and
        acc_weight as Triton's welford_combine does (Chan et al.), with V.ops.
        torch.var_mean(x, -1), with the locals each line comes from:
            tmp3 = tmp2 * (tile_r.end - tile_r.begin)   # weight
            tmp4 = tmp2 / tile_r.block_size             # scale
            tmp5 = tmp0 * tmp4
            tmp6 = torch.sum(tmp5, -1)
            tmp7 = hl.full([], tmp3, dtype=torch.float32)
            tmp8 = tmp7 * tmp4
            tmp9 = tmp6 / tmp8                          # mean
            tmp10 = tmp9.reshape([tile_x, 1])
            tmp11 = tmp0 - tmp10                        # dev
            tmp12 = tmp11 * tmp11
            tmp13 = torch.sum(tmp12, -1)                # m2
            tmp14 = acc2 + tmp3                         # new_weight
            ...
            tmp17 = torch.where(tmp15, tmp1, tmp16)     # delta
            ...
            tmp20 = torch.where(tmp18, tmp1, tmp19)     # share
            tmp21 = tmp17 * tmp20
            tmp22 = acc + tmp21                         # new_mean
            ...
            acc = tmp22
            acc1 = tmp27                                # new_m2
            acc2 = tmp14
        A row of inf gets the mean inf, as from Triton's persistent reductions
        (eager and Triton's looped ones give nan): the delta of two equal means
        is 0, where inf - inf would make it nan. An rdim of 2e19 in blocks of 64
        gets the variance 0: see the order of the products below. As in any
        float32 Welford, the merge takes each block's rounded mean, so on an rdim
        whose spread is below the float32 spacing of its mean the variance
        depends on the block size, and a block whose own mean rounds keeps the
        deviations' mean in its m2, as Inductor's persistent reductions do
        (welford_reduce_fallback())."""
        ops = V.ops
        zero = ops.constant(0.0, dtype)

        def block_sum(
            term: CSEVariable, empty: CSEVariable | None = None
        ) -> CSEVariable:
            total = self.reduce_block_value("sum", term, dtype)
            return total if empty is None else ops.where(empty, zero, total)

        # Two sums over each block, as Inductor's welford_reduce_fallback()
        # reduces a persistent rdim: the mean, then m2, the deviations' spread.
        # As there, a mean that rounds leaves the deviations a mean of their
        # own: of 100000000 and 100000008 in turn, the mean 100000004 rounds to
        # 100000000, and the deviations 0 and 8 give the variance 32, where
        # eager gives 16.
        # The accumulators of a loop: mean, m2 and weight.
        accumulators = (
            None
            if self.persistent_reduction
            else tuple(self.accumulator(0, dtype) for _ in range(3))
        )
        if reduction_type == "welford_reduce":
            (value,) = values
            # A loop's block takes the mean sum(x * s) / (weight * s), s = 1 / B
            # for B the block size: both products are exact (where x * s stays
            # normal, as x * (1 / weight) has to), so the mean is
            # sum(x) / weight, and the scaled sum does not overflow where
            # sum(x) would (a block of 1e37 sums to inf over 4096 lanes).
            # sum(x * (1 / weight)) would round each term, and an rdim of 768
            # values of 100000 get the mean 100000.0078. A persistent rdim takes
            # Inductor's own sum(x) / weight, as Triton's persistent reductions
            # and hand-written kernels (torch.mean(x, -1)) do: the scale is one
            # more multiply per element, 6% of better-benchmark's
            # var_mean_sum_53c82 (a LayerNorm's statistics, rdims of 512) on
            # B200 in the CuTe backend, and an rdim whose sum overflows (4096
            # values of 1e37) gets the mean inf and the variance inf, as from
            # those, where eager's Welford keeps them finite.
            if self.persistent_reduction:
                # The rdim is one block of R values: the number 300.0,
                # rather than the sum of hl.full([], 1.0, ...).expand(rdim),
                # which would name the rdim (rdim_code()).
                weight = ops.constant(float(int(self.numels["r0_"])), dtype)
                scale = None
            else:
                # The block's lanes within R (Helion masks the others): a
                # scalar, where a sum of hl.full([], 1.0, ...).expand(tile_r)
                # would be one more block sum; B is the block size.
                one = ops.constant(1.0, dtype)
                line = f"{one} * (tile_r.end - tile_r.begin)"
                weight = self.cse.generate(self.compute, line, dtype=dtype, shape=())
                line = f"{one} / tile_r.block_size"
                scale = self.cse.generate(self.compute, line, dtype=dtype, shape=())
                # Where one is a Python number (number_constant(): float32),
                # both are, which HelionCSEProxy._default divides by a tensor
                # as 0-d tensors: the merge's share, weight / (acc_weight +
                # weight), written tmp3 / tmp13, would be Tensor.__rtruediv__,
                # a reciprocal times the weight, which rounds (1 / 41 * 41 is
                # 0.99999994): an rdim of 41 values of 3.0 in a block of 64 got
                # the mean 2.9999998. A float64 one is hl.full([], 1.0, ...),
                # a 0-d tensor, and so are they.
                if self.is_number(one):
                    self.numbers.update((weight.name, scale.name))
            if scale is None:
                # A block sum of x that the rdim took before, as of
                # torch.mean(x, -1) beside torch.var(x, -1), is the Welford
                # block's too: the CSE hands block_sum(value) that sum's
                # variable, one sum of x for both, where there were two (and a
                # sum after it shares the Welford block's). A loop's sums
                # accumulate in acc lines, which the CSE does not hold: each
                # keeps its own.
                mean = ops.truediv(block_sum(value), weight)
            else:
                total = block_sum(ops.mul(value, scale))
                mean = ops.truediv(total, ops.mul(weight, scale))
            dev = ops.sub(value, mean)
            m2 = block_sum(ops.mul(dev, dev))
        else:
            value, part_m2, part_weight = values
            weight = self.reduce_block_value("sum", part_weight, dtype)
            # A block of weight 0, the (0, 0, 0) partials past the end of R in a
            # masked split, gets m2 0 rather than 0 / 0, and the mean 0 or, in a
            # loop, acc_mean, which the merge below then keeps: its 0 against
            # an acc_mean of inf would make delta -inf, and -inf * 0 nan.
            empty = ops.eq(weight, zero)
            # The mean is the partials' means weighted by w[r] / weight, then
            # corrected by the weighted deviations from it, as the sum of each
            # mean times its share rounds. The deviations are of the halves,
            # m / 2 - first / 2, within float32 where m - first can overflow
            # (-3e38 from a first pass of 3e38), and their weighted sum s is
            # added twice, first + s + s. Where the first pass is not finite (a
            # rdim with an inf) it is the mean. Equal means, as of a constant
            # rdim's chunks, give that mean itself (their max equals their min),
            # as Triton's welford_combine gives them (delta 0), where the
            # weighted sum would round. (An anchor at the largest mean would
            # make every deviation as large as the spread of the means, and
            # their rounding with it: one value of 1e5 among 1025 of randn()
            # moved the mean by 7e-3.)
            valid = ops.gt(part_weight, zero)
            neg_inf = ops.constant(float("-inf"), dtype)
            pos_inf = ops.constant(float("inf"), dtype)
            top = self.reduce_block_value(
                "amax", ops.where(valid, value, neg_inf), dtype
            )
            bottom = self.reduce_block_value(
                "amin", ops.where(valid, value, pos_inf), dtype
            )
            fraction = ops.truediv(part_weight, weight)
            first = block_sum(ops.where(valid, ops.mul(value, fraction), zero))
            half = ops.constant(0.5, dtype)
            shift = ops.sub(ops.mul(value, half), ops.mul(first, half))
            shifts = block_sum(ops.where(valid, ops.mul(shift, fraction), zero))
            corrected = ops.where(
                ops.logical_or(ops.isinf(first), ops.isnan(first)),
                first,
                ops.add(ops.add(first, shifts), shifts),
            )
            combined = ops.where(ops.eq(top, bottom), top, corrected)
            empty_mean = zero if accumulators is None else accumulators[0]
            mean = ops.where(empty, empty_mean, combined)
            # m2[r] + w[r] * dev^2, the weight first: a partial of weight 0 adds
            # 0, where dev * dev could overflow to inf, and inf * 0 is nan.
            dev = ops.sub(value, mean)
            spread = ops.add(part_m2, ops.mul(ops.mul(part_weight, dev), dev))
            m2 = block_sum(spread, empty)
        if accumulators is None:
            # The row is one block, so its mean, m2 and weight are the result, a
            # two-pass variance as in Triton's persistent reductions, with no
            # accumulators to combine them into.
            return mean, m2, weight
        acc_mean, acc_m2, acc_weight = accumulators
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
