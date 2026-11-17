from __future__ import annotations

import collections
import contextlib
from dataclasses import dataclass
import itertools
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
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import MutationLayoutSHOULDREMOVE
from torch._inductor.ir import Reduction
from torch._inductor.ir import ReinterpretView
from torch._inductor.utils import IndentedBuffer
from torch._inductor.utils import sympy_subs
from torch._inductor.utils import upcast_compute_type
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.value_ranges import bound_sympy

from . import config as helion_config
from .overrides import HelionOverrides
from .overrides import HelionPrinter

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
REDUCTIONS = OrderedSet(["sum", "prod", "max", "min", "online_softmax_reduce"])


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
        return super()._default(name, args, kwargs)

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
        if not (kernel.in_loop and x in kernel.dot_results):
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
        with kernel.swap_buffers(kernel.stores, kernel.stores, kernel.stores):
            value = self._default("to_dtype", (x, dtype), kwargs)
        # So that tmp5 = tmp4.to(torch.float32) follows tmp4 out of the loop.
        kernel.dot_results.add(value)
        # The epilogue then reads value, not out0[tile_y, tile_x] from memory.
        kernel.outer_vars.add(value)
        return value


class HelionKernel(SIMDKernel[CSEVariable]):
    """A Helion kernel: pointwise range trees become one hl.tile loop and the
    reduction range tree an inner hl.tile loop. Loads and stores subscript
    tensors by tile, e.g. in0[tile_y, tile_r]; Helion picks block sizes and
    masks partial tiles."""

    overrides = HelionOverrides  # type: ignore[assignment]

    @staticmethod
    def supports_tiling(tiling: dict[str, sympy.Expr]) -> bool:
        return OrderedSet(tiling) <= OrderedSet(TILE_NAMES) and all(
            isinstance(numel, (int, sympy.Integer)) for numel in tiling.values()
        )

    def __init__(self, tiling: dict[str, sympy.Expr], **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(tiling, **kwargs)
        self.outer_vars: OrderedSet[CSEVariable] = OrderedSet()
        if not self.supports_tiling(tiling):
            raise AssertionError(f"Helion kernels do not tile {tiling}")
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
        # Dot accumulators and the casts of them that run after the K loop.
        self.dot_results: OrderedSet[CSEVariable] = OrderedSet()
        # Values stored outside the current reduction loop; see disable_reduction().
        self.outer_stores: dict[str, CSEVariable] = {}

    def __enter__(self) -> Self:
        super().__enter__()
        self.exit_stack.enter_context(
            V.set_ops_handler(HelionCSEProxy(self, HelionOverrides()))
        )
        return self

    @property
    def in_loop(self) -> bool:
        return self.inside_reduction and self.features.is_reduction()

    def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable:  # noqa: ANN401
        var = super().create_cse_var(*args, **kwargs)
        if not self.in_loop:
            self.outer_vars.add(var)
        return var

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return str(dtype)

    def index_axes(self, index: sympy.Expr) -> list[str]:
        return [axis for axis in self.axes if self.symbols[axis] in index.free_symbols]

    def value_shape(self, axes: Iterable[str]) -> TileShape:
        """Shape of a value over `axes`: their tiles in the kernel's axis order,
        (z, y, r0_, x) with a dot and (z, y, x, r0_) without. A value holds only
        its own tiles, since an r0_-only value in a matmul's K loop is [tile_r]
        next to [tile_y, tile_r] but [tile_r, 1] next to [tile_r, tile_x];
        align_args() reshapes values as ops combine them."""
        return tuple(TILE_NAMES[axis] for axis in self.axes if axis in axes)

    def index_tensor(self, axis: str, scope: Sequence[str]) -> str:
        name = f"{TILE_NAMES[axis]}.index"
        axes = scope[scope.index(axis) :]
        if len(axes) == 1:
            return name
        # reshape() rather than slicing: Helion's tcgen05 store path proves the
        # layout of a store subscript only through reshape, +, *, // and %.
        shape = ", ".join(TILE_NAMES[a] if a == axis else "1" for a in axes)
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
        return f"{value}.reshape([{', '.join(map(str, aligned))}])"

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
            line = f"{arg}.reshape([{', '.join(map(str, shape))}])"
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
        candidates = [
            [
                axis
                for axis in axes
                if V.graph.sizevars.statically_known_equals(size, self.numels[axis])
            ]
            + ([None] if size == 1 else [])
            for size in layout.size
        ]
        for dims in itertools.product(*candidates):
            if [dim for dim in dims if dim is not None] != list(axes):
                continue
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
                return arg, list(tiles) or ["()"], tiles
        if (view := self.tiled_view(buffer, index, axes)) is not None:
            tiles = tuple(TILE_NAMES[axis] for axis in axes)
            return self.tensor_arg(view, output=output), list(tiles) or ["()"], tiles
        return self.flat_subscript(buffer, index, axes, output=output)

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
        if self.in_loop:
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
                code.splice(self.body)
        return re.sub(r"__arg\d+__", lambda ref: names[ref[0]], code.getvalue())

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
        mask, mask_axes = self.load_mask(axes)
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
            indices = "" if coordinates == ["()"] else subscript
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
        value_axes = tile_axes(value.shape)
        axes = [
            axis
            for axis in self.axes
            if self.symbols[axis] in index.free_symbols or axis in value_axes
        ]
        arg, coordinates, shape = self.subscript(buffer, index, axes, output=True)
        dtype = buffer.get_dtype()
        cast_suffix = "" if value.dtype == dtype else f".to({dtype})"
        buffer_code = self.compute if self.in_loop else self.stores
        value_code = f"{self.align(value, shape)}{cast_suffix}"
        if mode == "atomic_add":
            # index_put(out, (i,), x, accumulate=True) with i = arange(4) // 2
            # adds x[0] and x[1] to out[0]:
            #     hl.atomic_add(out0, [(tile_x.index // 2)], tmp0)
            index_list = "" if coordinates == ["()"] else ", ".join(coordinates)
            line = f"hl.atomic_add({arg.ref}, [{index_list}], {value_code})"
        else:
            line = f"{arg.ref}[{', '.join(coordinates)}] = {value_code}"
        # DeferredLine, so that remove_kernel_local_buffers() can drop it.
        buffer_code.writeline(DeferredLine(name, line))

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
        if index.free_symbols - OrderedSet(self.symbols[axis] for axis in axes):
            raise NotImplementedError(f"Unsupported Helion index: {index}")
        scope = [axis for axis in self.axes if axis in axes]
        # tile.index is int32; widen it if any subexpression could overflow.
        ranges = {
            self.symbols[axis]: ValueRanges(0, self.numels[axis] - 1) for axis in axes
        }
        bounds = [bound_sympy(term, ranges) for term in sympy.preorder_traversal(index)]
        widen = any(not (b.lower >= -(2**31) and b.upper < 2**31) for b in bounds)
        suffix = ".to(torch.int64)" if widen else ""
        replacements = {
            self.symbols[axis]: sympy.Symbol(self.index_tensor(axis, scope) + suffix)
            for axis in axes
        }
        expr = HelionPrinter().doprint(index.xreplace(replacements), simplify=False)
        # Broadcast to the tiles of `axes` the index skips, e.g. those of a store's
        # value or a load's mask: (tile_x.index // 2).expand(tile_y, tile_x), or
        # hl.full([tile_x], 0, dtype=torch.int32) for the index 0.
        if all(self.symbols[axis] in index.free_symbols for axis in scope):
            return expr
        tiles = ", ".join(TILE_NAMES[axis] for axis in scope)
        if not index.free_symbols:
            dtype = torch.int64 if widen else torch.int32
            return f"hl.full([{tiles}], {expr}, dtype={dtype})"
        return f"({expr}).expand({tiles})"

    def load_mask(
        self, axes: Sequence[str]
    ) -> tuple[CSEVariable | None, OrderedSet[str]]:
        """The extra_mask of a load over `axes`, and its axes: the masks that are
        0-d or over any of those axes. A mask over other axes only cannot move the
        load's address and is left to masked()."""
        load_axes = OrderedSet(axes)
        guards = []
        guard_axes: OrderedSet[str] = OrderedSet()
        for mask in self.load_masks:
            mask_axes = OrderedSet(tile_axes(mask.shape))
            if not mask_axes or mask_axes & load_axes:
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
        line = f"{self.align(value, tiles)}.expand({', '.join(tiles)})"
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
        if isinstance(value, tuple) and reduction_type != "online_softmax_reduce":
            raise AssertionError(f"expected one value to reduce, got {value}")
        # As in Triton, a reduction of a value already reduced reuses its result:
        # softmax's max and sum are two buffers, whose nodes each reduce x with
        # online_softmax_reduce. Without the reuse, the second node writes another
        # pair of accumulators, acc2 and acc3, which recompute the max and sum.
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]
        values = [
            self.over_tile_r(v)
            for v in (value if isinstance(value, tuple) else (value,))
        ]
        acc_dtype = upcast_compute_type(dtype)
        if reduction_type == "online_softmax_reduce":
            result = self.online_softmax_reduce(values, acc_dtype)
        else:
            init = Reduction.default_accumulator(
                reduction_type, upcast_compute_type(src_dtype)
            )
            if not isinstance(init, (bool, float, int)):
                raise AssertionError(f"expected one initial value, got {init}")
            acc = self.accumulator(init, acc_dtype)
            fn = {"sum": "sum", "prod": "prod", "max": "amax", "min": "amin"}
            part, _ = self.reduce_block(fn[reduction_type], values[0], acc_dtype)
            block = {
                "sum": f"{acc} + {part}",
                "prod": f"{acc} * {part}",
                "max": f"torch.maximum({acc}, {part})",
                "min": f"torch.minimum({acc}, {part})",
            }[reduction_type]
            self.compute.writeline(f"{acc} = {block}")
            result = acc
        self.cse.reduction_cache[cache_key] = cast("Any", result)
        return result

    def over_tile_r(self, value: CSEVariable) -> CSEVariable:
        """`value`, broadcast over tile_r if it has no tile_r, so that each block
        reduces one copy per lane."""
        if "tile_r" in map(str, value.shape or ()):
            return value
        axes = tile_axes(value.shape)
        shape = self.value_shape([*axes, "r0_"])
        zeros = f"hl.zeros([{', '.join(map(str, shape))}], dtype={value.dtype})"
        line = f"{self.align(value, shape)} + {zeros}"
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
            part = f"{part}.reshape([{', '.join(map(str, aligned))}])"
        return part, aligned

    def online_softmax_reduce(
        self, values: Sequence[CSEVariable], dtype: torch.dtype
    ) -> tuple[CSEVariable, CSEVariable]:
        """The max m of x and the sum of exp(x - m), which softmax lowers to
        (prepare_softmax_online). `values` is x, or x and a weight w of each exp:
        a split reduction gives its first kernel (x, 1), and (-inf, 0) past the
        end of R, and its second the first's (max, sum) pairs. Like the other
        reductions, each block of tile_r is reduced, here to its max and its sum
        of exps, and folded into the accumulators acc_max and acc_sum with
        V.ops, which reshape the [tile_x] block max next to the [tile_x, tile_r]
        block. torch.softmax(x, -1), with the locals each line comes from:
            tmp1 = torch.amax(tmp0, -1)             # block_max
            tmp5 = torch.where(tmp3, tmp4, tmp1)    # shift: tmp1 == -inf ? 0 : tmp1
            tmp6 = tmp5.reshape([tile_x, 1])
            tmp7 = tmp0 - tmp6
            tmp8 = torch.exp(tmp7)                  # exp
            tmp9 = torch.sum(tmp8, -1)              # block_sum
            tmp10 = torch.maximum(acc, tmp1)        # new_max
            ...
            tmp14 = torch.exp(tmp13)                # scale: exp(-|acc - tmp1|)
            ...
            acc1 = tmp22    # new_sum: acc1 * (1 if acc == tmp10 else tmp14) + tmp9 * (1 if tmp1 == tmp10 else tmp14)
            acc = tmp10
        Of the two rescale factors exp(acc - new_max) and exp(block_max -
        new_max), one is 1 and the other scale, so one exp does. A block of -inf
        (masked out, padding past the end of R, or the -inf half of a split
        reduction) is shifted by 0 rather than by its max: its exps are then
        exp(-inf) = 0, where x - block_max would give exp(-inf - -inf) = nan,
        and nan * 0 = nan for the padding too, which reaches the sum and every
        output."""
        ops = V.ops
        value, *weight = values

        def block(fn: str, arg: CSEVariable) -> CSEVariable:
            line, shape = self.reduce_block(fn, arg, dtype)
            return self.cse.generate(self.compute, line, dtype=dtype, shape=shape)

        acc_max = self.accumulator(float("-inf"), dtype)
        acc_sum = self.accumulator(0, dtype)
        block_max = block("amax", value)
        neg_inf = ops.constant(float("-inf"), dtype)
        shift = ops.where(
            ops.eq(block_max, neg_inf), ops.constant(0.0, dtype), block_max
        )
        exp = ops.exp(ops.sub(value, shift))
        block_sum = block("sum", ops.mul(weight[0], exp) if weight else exp)
        new_max = ops.maximum(acc_max, block_max)
        scale = ops.exp(ops.neg(ops.abs(ops.sub(acc_max, block_max))))
        one = ops.constant(1.0, dtype)
        new_sum = ops.add(
            ops.mul(acc_sum, ops.where(ops.eq(acc_max, new_max), one, scale)),
            ops.mul(block_sum, ops.where(ops.eq(block_max, new_max), one, scale)),
        )
        self.compute.writeline(f"{acc_sum} = {new_sum}")
        self.compute.writeline(f"{acc_max} = {new_max}")
        return acc_max, acc_sum
