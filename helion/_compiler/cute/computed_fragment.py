"""CTA-local SIMT lowering for multidimensional static fragments.

The scalar lowering gives each block ID one coordinate. Static dimensions,
including repeated dimensions of a matrix, need independent coordinates. This
path owns a complete root: its pure expressions retain logical index maps and
its loads, contractions, and loop carries are materialized in shared memory.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch._inductor.ir import Reduction
from torch._inductor.virtualized import V
from torch.fx import Node
from torch.fx.node import map_arg

from ... import exc
from ..._compat import shape_env_size_hint
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import inline_asm_ops
from ...language import matmul_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_ops
from ...language import view_ops
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..device_ir import HelperFunctionGraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import control_flow_parent_entries
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing
from ..inductor_lowering import GenerateASTFromInductor
from ..inductor_lowering import PointwiseLowering
from ..inductor_lowering import ReductionLowering
from ..inductor_lowering import SympyExprLowering
from ..inductor_lowering import install_inductor_kernel_handlers
from ..matmul_utils import _compute_out_dtype
from ..matmul_utils import _needs_f32_accumulator
from ..variable_origin import BlockSizeOrigin
from ..variable_origin import GridOrigin
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileEndOrigin
from ..variable_origin import TileIdOrigin
from .fragment_expression import FragmentExpression
from .independent_reduction import independent_reduction_coordinates
from .tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    from ..autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


def _resolve(value: object, values: dict[Node, object]) -> object:
    if isinstance(value, Node):
        return values[value]
    if isinstance(value, list):
        return [_resolve(item, values) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve(item, values) for item in value)
    return value


def _loop_carry_slots(node: Node) -> dict[int, int]:
    captured = cast("list[Node]", node.args[3])
    slots: dict[int, int] = {}
    for getitem in node.users:
        assert getitem.target is operator.getitem
        output_index = cast("int", getitem.args[1])
        for user in getitem.users:
            if user.target is _tracing_ops._phi and user.args[1] is getitem:
                slots[output_index] = captured.index(cast("Node", user.args[0]))
    return slots


@dataclass(frozen=True)
class Fragment:
    shape: tuple[int, ...]
    dtype: torch.dtype
    element: Callable[[tuple[str, ...]], str]
    resident: bool = False
    dependencies: tuple[Fragment, ...] = ()
    storage: str | None = None

    def read(self, indices: tuple[str, ...]) -> str:
        assert len(indices) == len(self.shape)
        return self.element(indices)

    def broadcast(self, indices: tuple[str, ...]) -> str:
        indices = indices[len(indices) - len(self.shape) :] if self.shape else ()
        return self.read(
            tuple(
                "0" if size == 1 else index
                for size, index in zip(self.shape, indices, strict=True)
            )
        )


@dataclass(frozen=True)
class HostTensor:
    name: str
    value: torch.Tensor


class FragmentOps(GenerateASTFromInductor):
    """Use Inductor's index expressions instead of dropping them as SIMT does."""

    def __init__(
        self, compiler: FragmentCompiler, inputs: dict[str, tuple[Node, Fragment]]
    ) -> None:
        super().__init__(compiler.cg, {})
        self.compiler = compiler
        self.inputs = inputs

    def load(self, name: str, index: sympy.Expr) -> str:
        node, value = self.inputs[name]
        fake = node.meta["val"]
        if not isinstance(fake, torch.Tensor):
            return self._lift(expr_from_string(value.read(())))
        assert isinstance(fake, torch.Tensor)
        # Inductor indexes the fake input's physical layout, including its
        # transposes and broadcasts. Recover coordinates in that same layout.
        strides = self.compiler.shape(fake.stride())
        indices = ["0"] * len(value.shape)
        remaining = self.compiler.sym(index)
        for dim in sorted(
            range(len(strides)), key=lambda dim: strides[dim], reverse=True
        ):
            if value.shape[dim] == 1 or strides[dim] == 0:
                continue
            indices[dim] = f"(({remaining}) // {strides[dim]})"
            remaining = f"(({remaining}) % {strides[dim]})"
        coordinates = self.compiler.coordinate_locals(tuple(indices))
        return self._lift(expr_from_string(value.read(coordinates)))

    def to_dtype(
        self,
        x: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        # Fragment loads and shared buffers contain logical typed scalars.
        # The parent handler's FP8 byte decoder belongs to ordinary raw-byte
        # global loads; applying it to a typed Float8 register corrupts its bits.
        return self._lift(self._create_cast_expr(x, dtype))

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> str:
        return self._lift(
            self._cast_scalar_ast(expr_from_string(self.compiler.sym(expr)), dtype)
        )


class FragmentCompiler:
    def __init__(self, cg: GenerateAST, graphs: list[GraphInfo] | None = None) -> None:
        self.cg = cg
        self.df = cg.device_function
        self.env = CompileEnvironment.current()
        self.offsets: dict[int, str] = {}
        self.bounds: dict[int, str] = {}
        self.allocator = ""
        self.smem_bytes = 0
        self.buffers: list[tuple[str, torch.dtype, int]] = []
        self.scopes: list[dict[Node, object]] = []
        self.held: list[object] = []
        self.threads = 128
        self.thread = ""
        self.sym_indices: dict[sympy.Symbol, str] = {}
        self.expression: FragmentExpression | None = None
        self.graphs = cg.codegen_graphs if graphs is None else graphs

    def begin(self) -> None:
        self.allocator = self.df.new_var("fragment_smem")
        self.thread = self.df.new_var("fragment_thread")

    def allocations(self) -> list[ast.AST]:
        return [
            statement_from_string(f"{self.allocator} = cutlass.utils.SmemAllocator()"),
            *[
                statement_from_string(
                    f"{name} = {self.allocator}.allocate_tensor({self.dtype(dtype)}, cute.make_layout(({count},)), byte_alignment=16)"
                )
                for name, dtype, count in self.buffers
            ],
        ]

    def metadata_guarded(self, expr: sympy.Expr) -> bool:
        if "input_tensor_metadata" not in self.env.compiler_fact_specialization_facts:
            return False
        # The exact metadata guard covers sizes and strides, not runtime scalar
        # arguments, device values, or storage offsets.
        return all(
            isinstance(symbol, sympy.Symbol)
            and any(
                isinstance(source, TensorPropertySource)
                and source.prop in (TensorProperty.SIZE, TensorProperty.STRIDE)
                for source in self.env.shape_env.var_to_sources.get(symbol, ())
            )
            for symbol in expr.free_symbols
        )

    def configured_expr(self, value: sympy.Basic) -> sympy.Basic:
        """Resolve logical dimensions using their owning block, not aliases.

        Reduction blocks can reuse a tile symbol or have a derived full-axis
        extent. Fragment roots own their iteration/reduction geometry, so a
        reduction's logical numel is required rather than a padded tracing hint.
        Runtime symbols with no block-size origin remain symbolic.
        """
        origins = HostFunction.current().expr_to_origin

        def resolve(expr: sympy.Basic, visiting: frozenset[sympy.Basic]) -> sympy.Basic:
            substitutions = {}
            for symbol in expr.free_symbols:
                info = origins.get(symbol)
                if info is None or not isinstance(info.origin, BlockSizeOrigin):
                    continue
                if symbol in visiting:
                    raise exc.InvalidConfig(
                        f"cyclic computed fragment block-size extent: {symbol}"
                    )
                block = self.env.block_sizes[info.origin.block_id]
                replacement = (
                    block.numel
                    if block.reduction
                    else self.df.resolved_block_size(block.block_id)
                )
                if replacement is None:
                    continue
                if isinstance(replacement, torch.SymInt):
                    replacement = replacement._sympy_()
                resolved = resolve(sympy.sympify(replacement), visiting | {symbol})
                if block.reduction:
                    logical = self.env.specialize_expr(sympy.sympify(resolved))
                    if logical.free_symbols and self.metadata_guarded(logical):
                        # Storage has a static capacity; masks and scans still
                        # use block.numel's runtime logical bound. Exact tensor
                        # metadata in the binding key makes this hint a proof,
                        # including when a direct BoundKernel call is replayed.
                        resolved = sympy.Integer(
                            self.env.backend.static_rdim_size(
                                max(1, shape_env_size_hint(self.env.shape_env, logical))
                            )
                        )
                substitutions[symbol] = resolved
            return expr.xreplace(substitutions)

        return resolve(value, frozenset())

    def extent(self, value: object) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, torch.SymInt):
            value = value._sympy_()
        assert isinstance(value, sympy.Expr)
        value = self.configured_expr(value)
        value = self.env.specialize_expr(sympy.sympify(value))
        if not value.is_number:
            raise exc.InvalidConfig(
                f"computed fragments require static local extents: {value}"
            )
        return int(value)

    def shape(self, shape: object) -> tuple[int, ...]:
        return tuple(self.extent(x) for x in cast("tuple[object, ...]", shape))

    def sym(self, value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            value = value._sympy_()
        if isinstance(value, sympy.Basic):
            value = sympy.sympify(self.configured_expr(value))

            def symbol_expression(symbol: sympy.Symbol) -> str:
                if symbol in self.sym_indices:
                    return self.sym_indices[symbol]
                origin_info = HostFunction.current().expr_to_origin.get(symbol)
                origin = origin_info.origin if origin_info is not None else None
                if isinstance(origin, GridOrigin) and origin.block_id in self.offsets:
                    offset = self.offsets[origin.block_id]
                    size = self.df.resolved_block_size(origin.block_id)
                    if isinstance(origin, TileIdOrigin):
                        return f"({offset}) // {size}"
                    if isinstance(origin, TileEndOrigin):
                        return (
                            f"min(({offset}) + {size}, {self.bounds[origin.block_id]})"
                        )
                    if type(origin) in (GridOrigin, TileBeginOrigin):
                        return offset
                return self.df.literal_expr(symbol)

            substitutions = {
                symbol: sympy.Symbol(f"({symbol_expression(symbol)})")
                for symbol in value.free_symbols
            }
            return self.env.backend.sympy_printer_expr(value.xreplace(substitutions))
        return self.df.literal_expr(value)

    def scalar(self, value: object) -> str:
        if isinstance(value, Fragment):
            assert math.prod(value.shape) == 1
            return value.read(tuple("0" for _ in value.shape))
        return self.sym(value)

    def emit(self, source: str) -> None:
        self.cg.add_statement(statement_from_string(source))

    def dtype(self, dtype: torch.dtype) -> str:
        return self.env.backend.dtype_str(dtype)

    def cast(self, expression: str, dtype: torch.dtype) -> str:
        return f"{self.dtype(dtype)}({expression})"

    def coordinate_locals(self, coordinates: tuple[str, ...]) -> tuple[str, ...]:
        """Evaluate composed scalar coordinates in their current lexical body.

        Recursively expanding a reshape or physical-stride inverse duplicates
        its incoming coordinates in every dimension. Name those pure integer
        expressions before visiting the next fragment. Do not cast them or
        reuse locals across reads, loops, branches, or shared-memory writes.
        """
        if (
            self.expression is not None
            and self.expression.body is self.cg.statements_stack[-1]
        ):
            return tuple(self.expression.coordinate(value) for value in coordinates)
        result = []
        for coordinate in coordinates:
            expression = expr_from_string(coordinate)
            if isinstance(expression, (ast.Name, ast.Constant)):
                result.append(coordinate)
                continue
            result.append(self.cg.lift(expression, prefix="fragment_coordinate").id)
        return tuple(result)

    def pointwise_read(
        self,
        producer: object,
        coordinates: tuple[str, ...],
        evaluate: Callable[[], str],
    ) -> str:
        previous = self.expression
        if previous is None or previous.body is not self.cg.statements_stack[-1]:
            self.expression = FragmentExpression(self.cg)
        try:
            assert self.expression is not None
            return self.expression.read(producer, coordinates, evaluate)
        finally:
            self.expression = previous

    @staticmethod
    def referenced_buffers(values: Iterable[object]) -> set[str]:
        live: set[str] = set()
        seen: set[int] = set()

        def visit(value: object) -> None:
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, Fragment):
                if value.storage is not None:
                    live.add(value.storage)
                for child in value.dependencies:
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        for value in values:
            visit(value)
        return live

    def live_buffers(self) -> set[str]:
        return self.referenced_buffers(
            [*[value for scope in self.scopes for value in scope.values()], *self.held]
        )

    def allocate(self, value: Fragment) -> Fragment:
        count = math.prod(value.shape)
        live = self.live_buffers()
        available = sorted(
            (capacity, name, index)
            for index, (name, dtype, capacity) in enumerate(self.buffers)
            if name not in live and dtype == value.dtype
        )
        selected = next((item for item in available if item[0] >= count), None)
        if selected is None and available:
            selected = available[-1]
        if selected is None:
            name = self.df.new_var("fragment_buffer")
            self.buffers.append((name, value.dtype, count))
            self.smem_bytes += (count * value.dtype.itemsize + 15) // 16 * 16
        else:
            capacity, name, index = selected
            if capacity >= count:
                count = capacity
            else:
                # All allocations are declared at the root. A dead buffer can
                # grow for its next lifetime without retaining a separate size
                # class in shared memory; earlier flat indices stay valid.
                self.buffers[index] = (name, value.dtype, count)
                self.smem_bytes += (count * value.dtype.itemsize + 15) // 16 * 16 - (
                    capacity * value.dtype.itemsize + 15
                ) // 16 * 16
        return Fragment(
            value.shape,
            value.dtype,
            lambda indices: f"{name}[{self.flatten(indices, value.shape)}]",
            True,
            (),
            name,
        )

    @staticmethod
    def flatten(indices: tuple[str, ...], shape: tuple[int, ...]) -> str:
        return (
            " + ".join(
                f"({index}) * {math.prod(shape[i + 1 :])}"
                for i, index in enumerate(indices)
            )
            or "0"
        )

    @staticmethod
    def coordinates(index: str, shape: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(
            "0" if size == 1 else f"(({index}) // {math.prod(shape[i + 1 :])} % {size})"
            for i, size in enumerate(shape)
        )

    def elements(
        self,
        shape: tuple[int, ...],
        body: Callable[[tuple[str, ...]], None],
        *,
        threads_per_element: int = 1,
    ) -> None:
        assert threads_per_element in (1, 32)
        assert self.threads % threads_per_element == 0
        index = self.df.new_var("fragment_index")
        start = (
            self.thread
            if threads_per_element == 1
            else f"{self.thread} // {threads_per_element}"
        )
        loop = statement_from_string(
            f"for {index} in range({start}, {math.prod(shape)}, {self.threads // threads_per_element}):\n    pass"
        )
        assert isinstance(loop, ast.For)
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            body(self.coordinates(index, shape))
        self.cg.add_statement(loop)
        self.emit("cute.arch.sync_threads()")

    def copy(
        self, source: Fragment, target: Fragment, *, threads_per_element: int = 1
    ) -> None:
        assert source.shape == target.shape

        def write(indices: tuple[str, ...]) -> None:
            # Cooperative producers execute in every participating lane; only
            # their leader writes the resulting shared element.
            value = self.cast(source.read(indices), target.dtype)
            store = f"{target.read(indices)} = {value}"
            if threads_per_element != 1:
                store = f"if {self.thread} % {threads_per_element} == 0:\n    {store}"
            self.emit(store)

        self.elements(
            source.shape,
            write,
            threads_per_element=threads_per_element,
        )

    def materialize(
        self, value: Fragment, *, copy: bool = False, threads_per_element: int = 1
    ) -> Fragment:
        if value.resident and not copy:
            return value
        self.held.append(value)
        result = self.allocate(value)
        self.copy(value, result, threads_per_element=threads_per_element)
        self.held.pop()
        return result

    def memory(self, node: Node, values: dict[Node, object], store: bool) -> object:
        tensor = values[cast("Node", node.args[0])]
        indices = cast("list[object]", node.args[1])
        output = (
            cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])
            if store
            else cast("torch.Tensor", node.meta["val"])
        )
        # A scalar or broadcast value does not describe the indexed write
        # domain. Use the same destination shape rule as fake tensor loads.
        shape = self.shape(
            SubscriptIndexing.compute_shape(
                output,
                [
                    index.meta["val"] if isinstance(index, Node) else index
                    for index in indices
                ],
            )
            if store
            else output.shape
        )
        extra_mask = node.args[3] if store else node.args[2]
        mask_value = values[extra_mask] if isinstance(extra_mask, Node) else extra_mask
        if isinstance(tensor, Fragment):
            if (
                store
                or extra_mask is not None
                or node.args[3] is not None
                or any(index is not None and index != slice(None) for index in indices)
            ):
                raise exc.BackendUnsupported("cute", "computed fragment indexed memory")

            def view(coords: tuple[str, ...]) -> str:
                selected = tuple(
                    coord
                    for coord, index in zip(coords, indices, strict=False)
                    if index is not None
                )
                return tensor.read(
                    self.coordinate_locals((*selected, *coords[len(indices) :]))
                )

            return Fragment(
                shape, output.dtype, view, tensor.resident, dependencies=(tensor,)
            )
        assert isinstance(tensor, HostTensor)

        def address(coords: tuple[str, ...]) -> tuple[str, str]:
            position = 0
            tensor_indices: list[str] = []
            masks: list[str] = []
            dim = 0
            for index in indices:
                if index is None:
                    position += 1
                    continue
                proxy = index.meta["val"] if isinstance(index, Node) else index
                block = (
                    self.env.resolve_block_id(proxy)
                    if isinstance(proxy, torch.SymInt)
                    else None
                )
                if (
                    block is not None
                    and cast("torch.SymInt", proxy)._sympy_()
                    != self.env.block_sizes[block].var._sympy_()
                ):
                    block = None
                if isinstance(index, slice):
                    assert index == slice(None)
                    expression = coords[position]
                    position += 1
                elif block is not None:
                    expression = f"({self.offsets[block]}) + ({coords[position]})"
                    masks.append(f"(({expression}) < ({self.bounds[block]}))")
                    position += 1
                elif isinstance(index, Node):
                    value = values[index]
                    if isinstance(value, Fragment) and value.shape:
                        expression = value.read(
                            coords[position : position + len(value.shape)]
                        )
                        position += len(value.shape)
                    else:
                        expression = self.scalar(value)
                else:
                    expression = self.scalar(index)
                tensor_indices.append(expression)
                size = self.sym(tensor.value.shape[dim])
                masks.append(f"(({expression}) >= 0 and ({expression}) < ({size}))")
                dim += 1
            if isinstance(mask_value, Fragment):
                masks.append(mask_value.broadcast(coords))
            elif mask_value is not None:
                masks.append(self.scalar(mask_value))
            stride = tensor.value.stride()
            offset = (
                " + ".join(
                    f"cutlass.Int64({index}) * cutlass.Int64({self.sym(stride[dim])})"
                    for dim, index in enumerate(tensor_indices)
                )
                or "0"
            )
            return f"({tensor.name}.iterator + ({offset}))", " and ".join(
                masks
            ) or "True"

        if store:
            value = _resolve(node.args[2], values)

            def write(coords: tuple[str, ...]) -> None:
                pointer, mask = address(coords)
                stored = (
                    value.broadcast(coords)
                    if isinstance(value, Fragment)
                    else self.scalar(value)
                )
                # Round/truncate to the logical tensor dtype before CuTe
                # converts to the pointer's storage type. In particular, an
                # unsigned byte must retain 128..255 when its i8 pointer is signed.
                self.emit(
                    f"if {mask}:\n    {pointer}.store({self.cast(stored, tensor.value.dtype)})"
                )

            self.elements(shape, write)
            return None

        def load(coords: tuple[str, ...]) -> str:
            pointer, mask = address(coords)
            name = self.df.new_var("fragment_load")
            self.emit(f"{name} = {self.cast('0', output.dtype)}")
            # Pointer arithmetic can erase unsigned signedness (Uint8 -> Int8),
            # and bool pointers use byte storage. Restore the logical type before
            # a masked branch joins its typed zero or the value enters arithmetic.
            loaded = self.cast(f"{pointer}.load()", output.dtype)
            self.emit(f"if {mask}:\n    {name} = {loaded}")
            return name

        return self.materialize(Fragment(shape, output.dtype, load))

    def logical_axis_extent(self, fake: torch.Tensor, dim: int, capacity: int) -> str:
        """Use logical bounds for arithmetic over a padded local allocation."""
        bid = self.env.resolve_block_id(fake.shape[dim])
        extent = str(capacity)
        if bid is not None:
            if bid in self.offsets:
                extent = f"min({extent}, ({self.bounds[bid]}) - ({self.offsets[bid]}))"
            elif self.env.block_sizes[bid].reduction:
                extent = f"min({extent}, {self.sym(self.env.block_sizes[bid].numel)})"
        return extent

    def dot(self, node: Node, values: dict[Node, object]) -> Fragment:
        lhs, rhs = (values[cast("Node", x)] for x in node.args[:2])
        assert isinstance(lhs, Fragment) and isinstance(rhs, Fragment)
        # Operand recipes must finish before the contraction begins: the same
        # input can have different coordinates in its two operand roles.
        lhs = self.materialize(lhs)
        self.held.append(lhs)
        rhs = self.materialize(rhs)
        self.held.append(rhs)
        acc = values[node.args[2]] if isinstance(node.args[2], Node) else None
        assert acc is None or isinstance(acc, Fragment)
        out = cast("torch.Tensor", node.meta["val"])
        shape = self.shape(out.shape)
        out_dtype = cast("torch.dtype | None", node.args[3]) or _compute_out_dtype(
            lhs.dtype, rhs.dtype, acc.dtype if acc is not None else None
        )
        compute_dtype = (
            torch.float32 if _needs_f32_accumulator(lhs.dtype, rhs.dtype) else out_dtype
        )
        result = self.allocate(Fragment(shape, out_dtype, lambda _: "0"))

        left_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])
        contraction = self.logical_axis_extent(left_fake, -1, lhs.shape[-1])

        def contract(coords: tuple[str, ...]) -> None:
            total = self.df.new_var("fragment_dot")
            k = self.df.new_var("fragment_k")
            initial = acc.broadcast(coords) if acc is not None else "0"
            self.emit(f"{total} = {self.cast(initial, compute_dtype)}")
            # All operands are in shared memory, so this serial per-output
            # contraction has ordinary IEEE FP32 multiplication and addition.
            left = lhs.broadcast((*coords[:-2], coords[-2], k))
            right = rhs.broadcast((*coords[:-2], k, coords[-1]))
            self.emit(
                f"for {k} in range({contraction}):\n    {total} = {total} + {self.cast(left, compute_dtype)} * {self.cast(right, compute_dtype)}"
            )
            self.emit(f"{result.read(coords)} = {self.cast(total, out_dtype)}")

        self.elements(shape, contract)
        self.held.pop()
        self.held.pop()
        return result

    def pointwise(self, node: Node, values: dict[Node, object]) -> Fragment:
        lowering = cast("PointwiseLowering | ReductionLowering", node.meta["lowering"])
        inputs: list[Node] = []
        map_arg(
            (node.args, {**node.kwargs, "_extra_deps": None}),
            lambda x: inputs.append(x),
        )
        lookup: dict[str, tuple[Node, Fragment]] = {}
        for name, source in zip(lowering.input_names, inputs, strict=True):
            value = values[source]
            if not isinstance(value, Fragment):
                scalar = self.scalar(value)
                value = Fragment((), torch.float32, lambda _, scalar=scalar: scalar)
            lookup[name] = (source, value)
        fake = cast("torch.Tensor", node.meta["val"])
        ranges = self.shape(lowering.buffer.data.ranges)
        result_shape = self.shape(fake.shape)

        def element(
            coords: tuple[str, ...], reduction_coords: tuple[str, ...] = ()
        ) -> str:
            if ranges != result_shape:
                coords = self.coordinate_locals(
                    self.coordinates(self.flatten(coords, result_shape), ranges)
                )
            prior = self.sym_indices
            self.sym_indices = {
                sympy.Symbol(f"i{i}"): coord
                for i, coord in enumerate((*coords, *reduction_coords))
            }
            try:
                with (
                    node.meta["location"],
                    V.set_current_node(node),
                    install_inductor_kernel_handlers(self.cg, {}),
                    V.set_ops_handler(FragmentOps(self, lookup)),
                ):
                    indices = [
                        sympy.Symbol(f"i{i}")
                        for i in range(len(lowering.buffer.data.ranges))
                    ]
                    if isinstance(lowering, ReductionLowering):
                        reduction_indices = [
                            sympy.Symbol(f"i{i + len(indices)}")
                            for i in range(len(reduction_coords))
                        ]
                        return str(
                            lowering.buffer.data.inner_fn(indices, reduction_indices)
                        )
                    return str(lowering.buffer.data.inner_fn(indices))
            finally:
                self.sym_indices = prior

        dependencies = tuple(value for _, value in lookup.values())
        if isinstance(lowering, ReductionLowering):
            from ..autotuner_heuristics.cute_fragment_reduction import (
                fragment_warp_reduction_supported,
            )

            assert isinstance(lowering.buffer.data, Reduction)
            reduction_shape = self.shape(lowering.buffer.data.reduction_ranges)
            indexed = lowering.reduction_type in ("argmin", "argmax")
            value_dtype = lowering.buffer.data.src_dtype if indexed else fake.dtype
            compute_dtype = (
                torch.float32
                if value_dtype in (torch.float16, torch.bfloat16)
                else value_dtype
            )
            warp = self.df.config.get(
                "cute_fragment_reduction", "serial"
            ) == "warp" and fragment_warp_reduction_supported(node)

            def combine(left: str, right: str) -> str:
                expression = self.env.backend.reduction_combine_expr(
                    lowering.reduction_type, left, right, compute_dtype
                )
                if (
                    lowering.reduction_type in ("min", "max")
                    and compute_dtype.is_floating_point
                ):
                    # A parallel fold must preserve NaNs from either operand,
                    # including a NaN encountered in a preceding local tile.
                    expression = f"{left} if {left} != {left} else ({expression})"
                return expression

            def reduce(coords: tuple[str, ...]) -> str:
                reduction_type = lowering.reduction_type
                total = self.df.new_var(f"fragment_{reduction_type}")
                index = self.df.new_var("fragment_reduce_index")
                identity = (
                    0
                    if indexed
                    else Reduction.default_accumulator(reduction_type, compute_dtype)
                )
                self.emit(
                    f"{total} = {self.cast(self.scalar(identity), compute_dtype)}"
                )
                selected = self.df.new_var("fragment_selected_index") if indexed else ""
                if indexed:
                    self.emit(f"{selected} = {self.cast('0', fake.dtype)}")
                reduction_range = str(math.prod(reduction_shape))
                if warp:
                    reduction_range = f"{self.thread} % 32, {reduction_range}, 32"
                loop = statement_from_string(
                    f"for {index} in range({reduction_range}):\n    pass"
                )
                # The fragment owner retains full producer coordinates and
                # carries this accumulator across configured reduction tiles.
                # Generic graph rolling would instead capture a full fragment
                # inside a tile-local graph, losing both its coordinate window
                # and its implicit reduction carry.
                tile = self.df.config.reduction_loops
                block = lowering.block_index
                tile = self.env.config_spec.reduction_loops.config_get(
                    tile, block, None
                )
                tiled_loop: ast.For | None = None
                if tile is not None:
                    tile = self.extent(tile)
                    tile_begin = self.df.new_var("fragment_reduce_tile")
                    tiled_loop = cast(
                        "ast.For",
                        statement_from_string(
                            f"for {tile_begin} in range(0, {math.prod(reduction_shape)}, {tile}):\n    pass"
                        ),
                    )
                    assert isinstance(tiled_loop, ast.For)
                    start = (
                        f"{tile_begin} + ({self.thread} % 32)" if warp else tile_begin
                    )
                    stop = f"min({tile_begin} + {tile}, {math.prod(reduction_shape)})"
                    reduction_range = f"{start}, {stop}" + (", 32" if warp else "")
                    loop = statement_from_string(
                        f"for {index} in range({reduction_range}):\n    pass"
                    )
                assert isinstance(loop, ast.For)
                loop.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                    value = element(coords, self.coordinates(index, reduction_shape))
                    if indexed:
                        value = self.cast(value, compute_dtype)
                        compare = ">" if reduction_type == "argmax" else "<"
                        better = f"({value}) {compare} {total}"
                        if compute_dtype.is_floating_point:
                            better = f"({better}) or (({value}) != ({value}) and {total} == {total})"
                        # The increasing local reduction coordinate and strict
                        # comparison preserve the first tie, including NaNs.
                        self.emit(
                            f"if {index} == 0 or ({better}):\n"
                            f"    {total} = {value}\n"
                            f"    {selected} = {self.cast(index, fake.dtype)}"
                        )
                    else:
                        self.emit(f"{total} = {combine(total, value)}")
                if tiled_loop is not None:
                    tiled_loop.body = [loop]
                    self.cg.add_statement(tiled_loop)
                else:
                    self.cg.add_statement(loop)
                if warp:
                    # Output coordinates are warp-uniform, including a short
                    # final output tile. Lanes with no input use the identity;
                    # the complete warp reconverges before each shuffle.
                    peer = self.df.new_var("fragment_reduce_peer")
                    for distance in (16, 8, 4, 2, 1):
                        self.emit(
                            f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                        )
                        self.emit(f"{total} = {combine(total, peer)}")
                return selected if indexed else self.cast(total, fake.dtype)

            return self.materialize(
                Fragment(
                    self.shape(fake.shape),
                    fake.dtype,
                    reduce,
                    dependencies=dependencies,
                ),
                threads_per_element=32 if warp else 1,
            )
        producer = object()
        return Fragment(
            self.shape(fake.shape),
            fake.dtype,
            lambda coords: self.pointwise_read(
                producer, coords, lambda: self.cast(element(coords), fake.dtype)
            ),
            dependencies=dependencies,
        )

    def loop(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        graph_id = node.args[0]
        assert isinstance(graph_id, int)
        graph = self.graphs[graph_id]
        assert isinstance(graph, ForLoopGraphInfo)
        args = [values[x] for x in cast("list[Node]", node.args[3])]
        placeholders = list(graph.graph.find_nodes(op="placeholder"))
        output_node = graph.graph.find_nodes(op="output")[0]
        outputs = cast("list[Node]", output_node.args[0])
        carry_slots = _loop_carry_slots(node)
        # Match each output to its captured value through the actual phi edge.
        # Read-only captures can appear before, between, or after mutable ones.
        carries: list[Fragment] = []
        self.held.append(carries)
        inner_args = list(args)
        for index, output in enumerate(outputs):
            if index in carry_slots:
                slot = carry_slots[index]
                carry = self.materialize(cast("Fragment", args[slot]), copy=True)
                inner_args[slot] = carry
            else:
                fake = cast("torch.Tensor", output.meta["val"])
                carry = self.allocate(
                    Fragment(self.shape(fake.shape), fake.dtype, lambda _: "0")
                )
            carries.append(carry)
        body: list[ast.AST] = []
        outer_offsets, outer_bounds = dict(self.offsets), dict(self.bounds)
        with self.cg.set_statements(body):
            for bid, _begin, end in zip(
                graph.block_ids,
                cast("list[object]", node.args[1]),
                cast("list[object]", node.args[2]),
                strict=True,
            ):
                self.offsets[bid] = self.df.new_var("fragment_tile")
                self.bounds[bid] = self.scalar(
                    values[end] if isinstance(end, Node) else end
                )
            results = self.graph(
                graph.graph, dict(zip(placeholders, inner_args, strict=True))
            )
            assert isinstance(results, list)
            # Snapshot every value that reads a carry before writing any carry.
            # Disjoint recipes can write directly: none of their inputs changes
            # during the parallel assignment. Keep both sets live throughout.
            snapshots = []
            self.held.extend([results, snapshots])
            destinations = self.referenced_buffers(carries)
            for result in results:
                value = cast("Fragment", result)
                snapshots.append(
                    self.materialize(value, copy=True)
                    if self.referenced_buffers([value]) & destinations
                    else value
                )
            for target, source in zip(carries, snapshots, strict=True):
                self.copy(source, target)
            self.held.pop()
            self.held.pop()
        for index in reversed(range(len(graph.block_ids))):
            bid = graph.block_ids[index]
            begin = cast("list[object]", node.args[1])[index]
            start = self.scalar(values[begin] if isinstance(begin, Node) else begin)
            step = self.df.resolved_block_size(bid)
            if node.target is _tracing_ops._for_loop_step:
                explicit = cast("list[object]", node.args[4])[index]
                if explicit is not None:
                    step = self.scalar(
                        values[explicit] if isinstance(explicit, Node) else explicit
                    )
            loop = statement_from_string(
                f"for {self.offsets[bid]} in range(cutlass.Int32({start}), cutlass.Int32({self.bounds[bid]}), {step}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body = cast("list[ast.stmt]", body)
            body = [loop]
        self.offsets, self.bounds = outer_offsets, outer_bounds
        for stmt in body:
            self.cg.add_statement(stmt)
        self.held.pop()
        return carries

    def serial_scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        source = self.materialize(source)
        self.held.append(source)
        result = self.allocate(source)
        dim = cast("int", node.args[2]) % len(source.shape)
        reverse = bool(node.args[3])
        fake = cast("torch.Tensor", node.meta["val"])
        extent = self.logical_axis_extent(fake, dim, source.shape[dim])

        def line(coords: tuple[str, ...]) -> None:
            total = self.df.new_var("fragment_scan")
            index = self.df.new_var("fragment_scan_index")
            initialized = self.df.new_var("fragment_scan_initialized")
            self.emit(f"{total} = {self.cast('0', source.dtype)}")
            self.emit(f"{initialized} = False")
            position = f"{source.shape[dim] - 1} - {index}" if reverse else index
            location = (*coords[:dim], position, *coords[dim:])
            value = source.read(location)
            self.emit(
                f"for {index} in range({source.shape[dim]}):\n"
                f"    if ({position}) < ({extent}):\n"
                f"        if {initialized}:\n"
                f"            {total} = {self.cast(f'{total} + {value}', source.dtype)}\n"
                f"        else:\n"
                f"            {total} = {value}\n"
                f"        {initialized} = True\n"
                f"        {result.read(location)} = {total}\n"
                f"    else:\n"
                f"        {result.read(location)} = {self.cast('0', source.dtype)}"
            )

        self.elements((*source.shape[:dim], *source.shape[dim + 1 :]), line)
        self.held.pop()
        return result

    def scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        mode = self.df.config.get("cute_fragment_scan", "serial")
        if mode == "serial":
            return self.serial_scan(node, values)
        if mode == "cooperative":
            return self.cooperative_scan(node, values)
        raise exc.InvalidConfig(f"unsupported computed fragment scan: {mode!r}")

    def cooperative_scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        self.held.append(source)
        current = self.allocate(source)
        self.held.append(current)
        other = self.allocate(source)
        self.held.append(other)
        dim = cast("int", node.args[2]) % len(source.shape)
        reverse = bool(node.args[3])
        fake = cast("torch.Tensor", node.meta["val"])
        extent = self.logical_axis_extent(fake, dim, source.shape[dim])

        def initialize(coords: tuple[str, ...]) -> None:
            self.emit(
                f"if ({coords[dim]}) < ({extent}):\n"
                f"    {current.read(coords)} = {self.cast(source.read(coords), source.dtype)}\n"
                f"else:\n"
                f"    {current.read(coords)} = {self.cast('0', source.dtype)}"
            )

        self.elements(source.shape, initialize)
        # Every stage reads only the preceding stage. A CTA barrier follows
        # each complete tile write, including when a thread owns several
        # elements. Never overwrite the input: it may have other consumers.
        for stage in range((source.shape[dim] - 1).bit_length()):
            distance = 1 << stage

            def combine(
                coords: tuple[str, ...],
                distance: int = distance,
                current: Fragment = current,
                other: Fragment = other,
            ) -> None:
                position = coords[dim]
                neighbor = (
                    f"({position}) + {distance}"
                    if reverse
                    else f"({position}) - {distance}"
                )
                in_range = (
                    f"({neighbor}) < ({extent})" if reverse else f"({neighbor}) >= 0"
                )
                location = (*coords[:dim], neighbor, *coords[dim + 1 :])
                left = current.read(location)
                right = current.read(coords)
                self.emit(
                    f"if ({position}) < ({extent}) and ({in_range}):\n"
                    f"    {other.read(coords)} = {self.cast(f'{left} + {right}', source.dtype)}\n"
                    f"else:\n"
                    f"    {other.read(coords)} = {right}"
                )

            self.elements(source.shape, combine)
            current, other = other, current
        self.held.pop()
        self.held.pop()
        self.held.pop()
        return current

    def conditional(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        info = self.graphs[cast("int", node.args[1])]
        assert isinstance(info, IfGraphInfo)
        assert info.branches_outputs is not None
        merged: list[Fragment] = []
        self.held.append(merged)
        fake_outputs = cast("list[torch.Tensor]", node.meta["val"])
        for fake in fake_outputs[: len(info.branches_outputs)]:
            merged.append(
                self.allocate(
                    Fragment(self.shape(fake.shape), fake.dtype, lambda _: "0")
                )
            )
        branches: list[list[ast.AST]] = []
        for side in range(2):
            branch = self.graphs[cast("int", node.args[1 + side])]
            captures = [
                values[source] for source in cast("list[Node]", node.args[3 + side])
            ]
            names = info.if_arg_names if side == 0 else info.else_arg_names
            assert names is not None
            captured = dict(zip(names, captures, strict=True))
            body: list[ast.AST] = []
            with self.cg.set_statements(body):
                outputs = self.graph(
                    branch.graph,
                    dict(
                        zip(
                            branch.graph.find_nodes(op="placeholder"),
                            captures,
                            strict=True,
                        )
                    ),
                )
                assert isinstance(outputs, list)
                self.held.append(outputs)
                for target, slots in zip(merged, info.branches_outputs, strict=True):
                    slot = slots[side]
                    source = outputs[slot] if isinstance(slot, int) else captured[slot]
                    assert isinstance(source, Fragment)
                    self.copy(source, target)
                self.held.pop()
            branches.append(body or [ast.Pass()])
        predicate = node.args[0]
        proxy = predicate.meta["val"] if isinstance(predicate, Node) else predicate
        constant = self.df.evaluate_constexpr_condition(proxy)
        test = (
            repr(constant)
            if constant is not None
            else self.scalar(
                values[predicate] if isinstance(predicate, Node) else predicate
            )
        )
        statement = statement_from_string(f"if {test}:\n    pass")
        assert isinstance(statement, ast.If)
        statement.body = cast("list[ast.stmt]", branches[0])
        statement.orelse = cast("list[ast.stmt]", branches[1])
        self.cg.add_statement(statement)
        self.held.pop()
        return [*merged, *merged]

    def graph(self, graph: torch.fx.Graph, values: dict[Node, object]) -> object:
        uses = {node: len(node.users) for node in graph.nodes}
        self.scopes.append(values)
        try:
            for node in graph.nodes:
                if node.op == "placeholder":
                    continue
                if node.op == "output":
                    return _resolve(node.args[0], values)
                with node.meta["location"], V.set_current_node(node):
                    values[node] = self.node(node, values)
                for source in node.all_input_nodes:
                    uses[source] -= 1
                    if uses[source] == 0:
                        values.pop(source, None)
                if uses[node] == 0:
                    values.pop(node, None)
        finally:
            self.scopes.pop()
        raise AssertionError("graph has no output")

    def node(self, node: Node, values: dict[Node, object]) -> object:
        target = node.target
        args = cast("tuple[object, ...]", _resolve(node.args, values))
        fake = node.meta.get("val")
        if target is _tracing_ops._host_tensor:
            assert isinstance(fake, torch.Tensor)
            return HostTensor(
                self.df.tensor_arg(fake, prefer_name=cast("str", node.args[0])).name,
                fake,
            )
        if target in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
            return fake
        if target is _tracing_ops._new_var:
            return args[0]
        if target is _tracing_ops._phi:
            return args[1]
        if target is operator.getitem:
            sequence, index = args
            assert isinstance(sequence, (list, tuple)) and isinstance(index, int)
            return sequence[index]
        if target is _tracing_ops._if:
            return self.conditional(node, values)
        if _tracing_ops.is_for_loop_target(target):
            return self.loop(node, values)
        if target in (memory_ops.load, memory_ops.store):
            return self.memory(node, values, target is memory_ops.store)
        if target is matmul_ops.dot:
            return self.dot(node, values)
        if target is scan_ops._associative_scan:
            return self.scan(node, values)
        if target is inline_asm_ops.inline_asm_elementwise:
            assert isinstance(fake, torch.Tensor)
            asm, constraints, operands, dtype, _is_pure, _pack = args
            assert isinstance(operands, (tuple, list))
            fragments = tuple(cast("Fragment", operand) for operand in operands)

            def inline_asm(coords: tuple[str, ...]) -> str:
                inputs = ", ".join(
                    self.cast(operand.broadcast(coords), operand.dtype)
                    for operand in fragments
                )
                arguments = f"({inputs},)" if inputs else "()"
                return (
                    f"_cute_inline_asm_elementwise({arguments}, asm={asm!r}, "
                    f"constraints={constraints!r}, dtype={self.dtype(fake.dtype)}, "
                    "is_pure=True)"
                )

            # Opaque scalar programs may be expensive and can feed several
            # consumers. Evaluate each logical element once before sharing it.
            return self.materialize(
                Fragment(
                    self.shape(fake.shape),
                    fake.dtype,
                    inline_asm,
                    dependencies=fragments,
                )
            )
        if target in (tile_ops.tile_begin, tile_ops.tile_end, tile_ops.tile_id):
            return fake
        if target is tile_ops.tile_index:
            assert isinstance(fake, torch.Tensor)
            block_id = self.env.resolve_block_id(args[0])
            assert block_id in self.offsets
            offset = self.offsets[block_id]
            return Fragment(
                self.shape(fake.shape),
                fake.dtype,
                lambda coords: self.cast(f"({offset}) + ({coords[0]})", fake.dtype),
            )
        if isinstance(node.meta["lowering"], (PointwiseLowering, ReductionLowering)):
            return self.pointwise(node, values)
        if not isinstance(fake, torch.Tensor):
            return fake
        shape = self.shape(fake.shape)
        if target in (
            creation_ops.full,
            torch.ops.aten.full.default,
            torch.ops.aten.scalar_tensor.default,
            _tracing_ops._constant_tensor,
        ):
            value = (
                args[1]
                if target in (creation_ops.full, torch.ops.aten.full.default)
                else args[0]
            )
            return Fragment(
                shape,
                fake.dtype,
                lambda _: self.cast(self.scalar(value), fake.dtype),
                dependencies=(value,) if isinstance(value, Fragment) else (),
            )
        if target is torch.ops.prims.iota.default:
            start, step = node.kwargs.get("start", 0), node.kwargs.get("step", 1)
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: self.cast(
                    f"{self.scalar(start)} + ({coords[0]}) * {self.scalar(step)}",
                    fake.dtype,
                ),
            )
        source = args[0]
        assert isinstance(source, Fragment)
        if target is torch.ops.aten.alias.default:
            return source
        if target is torch.ops.aten.view.dtype:
            # Equal-width reinterpretation retains element ownership. Round
            # any widened arithmetic back to its declared dtype first.
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: (
                    f"{self.cast(source.read(coords), source.dtype)}"
                    f".bitcast({self.dtype(fake.dtype)})"
                ),
                dependencies=(source,),
            )
        if target is _tracing_ops._mask_to:
            source_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])

            def masked(coords: tuple[str, ...]) -> str:
                masks = []
                for coord, size in zip(coords, source_fake.shape, strict=True):
                    bid = self.env.resolve_block_id(size)
                    if bid is None:
                        continue
                    if bid in self.offsets:
                        masks.append(
                            f"(({self.offsets[bid]}) + ({coord}) < ({self.bounds[bid]}))"
                        )
                    elif self.env.block_sizes[bid].reduction:
                        masks.append(
                            f"(({coord}) < ({self.sym(self.env.block_sizes[bid].numel)}))"
                        )
                mask = " and ".join(masks) or "True"
                return f"({self.cast(source.read(coords), fake.dtype)} if {mask} else {self.cast(self.scalar(args[1]), fake.dtype)})"

            return Fragment(shape, fake.dtype, masked, dependencies=(source,))
        if target is view_ops.subscript:
            slices = cast("list[object]", args[1])

            def subscript(coords: tuple[str, ...]) -> str:
                result: list[str] = []
                dim = 0
                source_dim = 0
                for item in slices:
                    if item is None:
                        dim += 1
                    elif isinstance(item, slice):
                        begin = item.start or 0
                        step = item.step or 1
                        result.append(
                            f"({self.scalar(begin)}) + ({coords[dim]}) * ({self.scalar(step)})"
                        )
                        dim += 1
                        source_dim += 1
                    else:
                        position = self.scalar(item)
                        result.append(f"({position}) % {source.shape[source_dim]}")
                        source_dim += 1
                result.extend(coords[dim:])
                return source.read(self.coordinate_locals(tuple(result)))

            return Fragment(shape, fake.dtype, subscript, source.resident, (source,))
        if target is torch.ops.aten.permute.default:
            order = cast("list[int]", args[1])
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: source.read(
                    tuple(coords[order.index(i)] for i in range(len(order)))
                ),
                source.resident,
                (source,),
            )
        if target in (torch.ops.aten.expand.default, torch.ops.aten.clone.default):
            return Fragment(
                shape, fake.dtype, source.broadcast, source.resident, (source,)
            )
        if target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
        ):
            # Singleton views preserve logical element order. Read through the
            # source recipe so permuted layouts and resident aliases retain
            # their own physical addressing and shared-buffer lifetimes.
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: source.read(
                    self.coordinate_locals(
                        self.coordinates(self.flatten(coords, shape), source.shape)
                    )
                ),
                source.resident,
                (source,),
            )
        if target is torch.ops.aten.where.self:
            condition, lhs, rhs = cast("tuple[Fragment, Fragment, Fragment]", args)
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: (
                    f"({self.cast(lhs.broadcast(coords), fake.dtype)} if {condition.broadcast(coords)} else {self.cast(rhs.broadcast(coords), fake.dtype)})"
                ),
                dependencies=(condition, lhs, rhs),
            )
        raise exc.BackendUnsupported("cute", f"computed fragment operation {target}")


def computed_fragment_supported(
    env: CompileEnvironment, graphs: list[GraphInfo]
) -> bool:
    """Structural root ownership shared by search discovery and code generation.

    Exact local sizes, strides, scalar predicates and shared capacity remain
    configuration-dependent checks in the emitter.
    """
    graph_by_id = {info.graph_id: info for info in graphs}
    independent_reductions = independent_reduction_coordinates(env, graphs)

    def configured_axis(size: int | torch.SymInt) -> bool:
        if isinstance(size, int):
            return size > 0
        expr = env.specialize_expr(cast("sympy.Expr", size._sympy_()))
        if expr.is_number:
            return int(expr) > 0
        return all(env.get_block_id(symbol) is not None for symbol in expr.free_symbols)

    carried: set[Node] = set()
    captures: dict[Node, Node] = {}
    parents = control_flow_parent_entries(graphs)
    for info in graphs:
        if info.graph_id not in parents:
            continue
        node, argument_slot = parents[info.graph_id]
        placeholders = list(info.graph.find_nodes(op="placeholder"))
        captures.update(
            zip(
                placeholders,
                cast("list[Node]", node.args[argument_slot]),
                strict=True,
            )
        )
        if _tracing_ops.is_for_loop_target(node.target):
            for slot in _loop_carry_slots(node).values():
                value = placeholders[slot].meta.get("val")
                if isinstance(value, torch.Tensor) and value.ndim > 1:
                    carried.add(placeholders[slot])

    def reads_carry(node: Node) -> bool:
        pending = [node]
        seen: set[Node] = set()
        while pending:
            source = pending.pop()
            if source in seen:
                continue
            seen.add(source)
            if source in carried:
                return True
            if source.target in (memory_ops.load, _tracing_ops._host_tensor):
                continue
            if source in captures:
                pending.append(captures[source])
            else:
                pending.extend(source.all_input_nodes)
        return False

    # Only roots needing independent static coordinates or resident computed
    # contractions/scans use this fallback. Explicitly tiled contractions and
    # native collective configurations keep their existing code generation.
    def computed(node: Node) -> bool:
        if node.target in (
            torch.ops.aten.permute.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
            torch.ops.prims.convert_element_type.default,
            view_ops.subscript,
        ):
            return isinstance(node.args[0], Node) and computed(node.args[0])
        return node.target not in (memory_ops.load, _tracing_ops._host_tensor)

    def repeated_axes(value: torch.Tensor) -> bool:
        sizes = [
            size._sympy_() if isinstance(size, torch.SymInt) else size
            for size in value.shape
        ]
        sizes = [size for size in sizes if size != 1]
        return len(sizes) != len(set(sizes))

    def needs_coordinates(node: Node) -> bool:
        if node in independent_reductions:
            return True
        if node.target is _tracing_ops._host_tensor:
            return False
        if node.target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
        ):
            source = node.args[0]
            assert isinstance(source, Node)
            before = cast("torch.Tensor", source.meta["val"])
            after = cast("torch.Tensor", node.meta["val"])
            # Factoring a full static axis requires independent coordinates for
            # every factor. A scalar recipe for its producer cannot represent
            # both the original flat consumer and the factored consumer.
            static_before = [
                size for size in before.shape if isinstance(size, int) and size != 1
            ]
            static_after = [
                size for size in after.shape if isinstance(size, int) and size != 1
            ]
            symbolic_before = [
                size._sympy_()
                for size in before.shape
                if isinstance(size, torch.SymInt)
            ]
            symbolic_after = [
                size._sympy_() for size in after.shape if isinstance(size, torch.SymInt)
            ]
            if (
                static_before != static_after
                and math.prod(static_before) == math.prod(static_after)
                and symbolic_before == symbolic_after
            ):
                return True
        if node.target is scan_ops._associative_scan:
            source = node.args[1]
            return isinstance(source, Node) and computed(source)
        if isinstance(node.meta.get("lowering"), ReductionLowering):
            if any(reads_carry(source) for source in node.all_input_nodes):
                return True
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and node.target in (
            torch.ops.aten.gt.Tensor,
            torch.ops.aten.ge.Tensor,
            torch.ops.aten.lt.Tensor,
            torch.ops.aten.le.Tensor,
            torch.ops.aten.eq.Tensor,
            torch.ops.aten.ne.Tensor,
        ):
            if repeated_axes(value):
                return True
        if node.target is not matmul_ops.dot:
            return False
        lhs, rhs = node.args[:2]
        assert isinstance(lhs, Node) and isinstance(rhs, Node)
        left, right = lhs.meta["val"], rhs.meta["val"]
        bid = env.resolve_block_id(left.shape[-1])
        static_k = isinstance(left.shape[-1], int) and left.shape[-1] > 1
        if bid is not None:
            static_k = env.block_sizes[bid].reduction
        return (
            left.dtype == right.dtype
            and left.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and static_k
            and (
                # A single scalar block coordinate cannot represent two axes
                # within the same operand. This also applies to captured full
                # matrix loads, not just matrices created by comparisons.
                repeated_axes(left)
                or repeated_axes(right)
                # Resident state cannot be reconstructed from global loads for
                # each contraction coordinate, including after a dtype cast.
                or reads_carry(lhs)
                or reads_carry(rhs)
                # Existing scalar/native half-precision recipes retain their
                # paths when their operands have independent coordinates.
                or (
                    left.dtype == torch.float32
                    and (
                        computed(lhs)
                        or computed(rhs)
                        or left.ndim > 2
                        or right.ndim > 2
                    )
                )
            )
        )

    if not any(needs_coordinates(node) for info in graphs for node in info.graph.nodes):
        return False
    supported = {
        _tracing_ops._if,
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        _tracing_ops._phi,
        _tracing_ops._constant_tensor,
        _tracing_ops._mask_to,
        _tracing_ops._for_loop,
        _tracing_ops._for_loop_step,
        memory_ops.load,
        memory_ops.store,
        matmul_ops.dot,
        scan_ops._associative_scan,
        inline_asm_ops.inline_asm_elementwise,
        creation_ops.full,
        torch.ops.aten.full.default,
        view_ops.subscript,
        tile_ops.tile_begin,
        tile_ops.tile_end,
        tile_ops.tile_id,
        tile_ops.tile_index,
        operator.getitem,
        torch.ops.aten.sym_size.int,
        torch.ops.prims.iota.default,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.where.self,
        torch.ops.aten.view.dtype,
        torch.ops.aten.alias.default,
    }
    if any(
        node.op == "call_function"
        and node.target not in supported
        and not isinstance(
            node.meta.get("lowering"),
            (PointwiseLowering, SympyExprLowering, ReductionLowering),
        )
        for info in graphs
        for node in info.graph.nodes
    ):
        return False
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is torch.ops.aten.view.dtype:
                source = cast("Node", node.args[0]).meta["val"]
                target = node.meta["val"]
                if (
                    not isinstance(source, torch.Tensor)
                    or not isinstance(target, torch.Tensor)
                    or source.dtype == torch.bool
                    or target.dtype == torch.bool
                    or source.dtype.itemsize != target.dtype.itemsize
                ):
                    return False
            if node.target is inline_asm_ops.inline_asm_elementwise:
                if (
                    node.args[4] is not True
                    or node.args[5] != 1
                    or not isinstance(node.args[3], torch.dtype)
                    or not isinstance(node.meta.get("val"), torch.Tensor)
                ):
                    return False
            if node.target is scan_ops._associative_scan:
                if node.args[4]:
                    return False
                value = node.meta["val"]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype
                    not in (
                        torch.int8,
                        torch.uint8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.float64,
                    )
                    or value.ndim == 0
                ):
                    return False
                if not -value.ndim <= cast("int", node.args[2]) < value.ndim:
                    return False
                # Every local axis must be constant or have a configured block
                # owner. Config-dependent values and shared capacity are checked
                # again by the emitter, never replaced by a shape-specific mode.
                if not all(configured_axis(size) for size in value.shape):
                    return False
                helper = graph_by_id[cast("int", node.args[0])]
                assert isinstance(helper, HelperFunctionGraphInfo)
                nodes = list(helper.graph.nodes)
                if len(nodes) != 4:
                    return False
                lhs, rhs, add, output = nodes
                if (
                    lhs.op != "placeholder"
                    or rhs.op != "placeholder"
                    or add.target is not torch.ops.aten.add.Tensor
                    or add.args not in ((lhs, rhs), (rhs, lhs))
                    or add.kwargs.get("alpha", 1) != 1
                    or output.op != "output"
                    or output.args != (add,)
                ):
                    return False
            lowering = node.meta.get("lowering")
            if isinstance(
                lowering, ReductionLowering
            ) and lowering.reduction_type not in (
                "sum",
                "max",
                "min",
                "prod",
                "argmin",
                "argmax",
            ):
                return False
            if _tracing_ops.is_for_loop_target(node.target) and any(
                not isinstance(value, torch.Tensor) for value in node.meta["val"]
            ):
                return False
    return True


def computed_fragment_specialization_facts(
    env: CompileEnvironment, ir: DeviceIR
) -> frozenset[CompilerHeuristicSpecializationFact]:
    """Guard shape hints used as capacities before bound-cache publication."""
    if env.settings.static_shapes or ir.host_function is None:
        return frozenset()
    with ir.host_function:
        if computed_fragment_supported(
            env, ir.graphs
        ) or independent_reduction_coordinates(
            env, ir.graphs, include_rolled_outputs=True
        ):
            return frozenset({"input_tensor_metadata"})
    return frozenset()


def codegen_computed_fragment_root(cg: GenerateAST) -> bool:
    """Own static axes and complete contractions that scalar indexing cannot express."""
    from ..autotuner_heuristics.cute_fragment_reduction import (
        fragment_warp_reduction_supported,
    )

    if (
        CompileEnvironment.current().backend_name != "cute"
        or len(cg.host_function.device_ir.root_ids) != 1
        or cg.device_function.config.get("cute_collective_mma", False)
        or cg.device_function.config.get("cute_register_chain", False)
    ):
        return False
    root = cg.current_root_graph_info
    assert root is not None
    graphs = cg.codegen_graphs

    scan_required = (
        cg.device_function.config.get("cute_fragment_scan", "serial") == "cooperative"
        and root.graph_id
        in CompileEnvironment.current().config_spec.cute_fragment_scan_root_ids
    )
    reduction_required = (
        cg.device_function.config.get("cute_fragment_reduction", "serial") == "warp"
        and root.graph_id
        in CompileEnvironment.current().config_spec.cute_fragment_reduction_root_ids
    )

    def decline() -> bool:
        if scan_required or reduction_required:
            raise exc.InvalidConfig(
                "cooperative scan requires a computed fragment root"
                if scan_required
                else "warp reduction requires a computed fragment root"
            )
        return False

    if scan_required and not any(
        node.target is scan_ops._associative_scan
        for info in graphs
        for node in info.graph.nodes
    ):
        return decline()
    if reduction_required and not any(
        fragment_warp_reduction_supported(node)
        for info in graphs
        for node in info.graph.nodes
    ):
        return decline()
    if not computed_fragment_supported(CompileEnvironment.current(), graphs):
        return decline()
    # This owner implements configured reduction tiling directly. Keep the
    # logical producer graph, including every other codegen transformation,
    # rather than applying scalar graph rolling before fragment ownership.
    graphs = cg.host_function.device_ir.build_codegen_graphs(
        cg.device_function.config, roll_reductions=False
    )
    if not computed_fragment_supported(CompileEnvironment.current(), graphs):
        return decline()
    root = graphs[root.graph_id]
    compiler = FragmentCompiler(cg, graphs)
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is _tracing_ops._if:
                predicate = node.args[0]
                value = (
                    predicate.meta["val"] if isinstance(predicate, Node) else predicate
                )
                if (
                    isinstance(value, torch.Tensor)
                    and math.prod(compiler.shape(value.shape)) != 1
                ):
                    return decline()
    for info in graphs:
        for node in info.graph.nodes:
            if not isinstance(
                node.meta.get("lowering"), (PointwiseLowering, ReductionLowering)
            ):
                continue
            for source in node.all_input_nodes:
                value = source.meta.get("val")
                if not isinstance(value, torch.Tensor):
                    if isinstance(
                        value,
                        (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool),
                    ):
                        continue
                    return decline()
                span = 1
                for stride, size in sorted(
                    zip(
                        compiler.shape(value.stride()),
                        compiler.shape(value.shape),
                        strict=True,
                    )
                ):
                    if size <= 1 or stride == 0:
                        continue
                    if stride < span:
                        return decline()
                    span += (size - 1) * stride
    grid = cg.current_grid_state
    assert grid is not None
    compiler.begin()
    for bid, info in grid.block_id_to_info.items():
        compiler.offsets[bid] = grid.strategy.grid_origin_var(bid)
        compiler.bounds[bid] = cast("str", info.end_var_name)
    body: list[ast.AST] = []
    with cg.set_statements(body):
        compiler.emit(f"{compiler.thread} = cutlass.Int32(cute.arch.thread_idx()[0])")
        compiler.graph(root.graph, {})
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(compiler.env.device)
    if capacity and compiler.smem_bytes > capacity:
        raise exc.InvalidConfig(
            f"computed fragments need {compiler.smem_bytes} shared bytes, exceeding {capacity}"
        )
    cg.device_function.cute_state.owned_root_block_dims = (compiler.threads, 1, 1)
    for statement in (*compiler.allocations(), *body):
        cg.add_statement(statement)
    return True
