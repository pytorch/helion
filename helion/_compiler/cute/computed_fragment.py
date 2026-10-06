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
from torch._inductor.ir import Reduction
from torch._inductor.virtualized import V
from torch.fx import Node
from torch.fx.node import map_arg

from ... import exc
from ...language import _tracing_ops
from ...language import atomic_ops
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
from ..variable_origin import GridOrigin
from ..variable_origin import HostOrigin
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileEndOrigin
from ..variable_origin import TileIdOrigin
from .captured_reduction import captured_reduction_coordinates
from .captured_reduction import physical_capture_axes
from .direct_affine_plan import DIRECT_AFFINE_ORDINARY_SCHEDULE
from .fragment_expression import FragmentExpression
from .fragment_indexing import memory_index_coordinates
from .fragment_storage import aligned_shared_bytes
from .fragment_storage import configured_fragment_expr
from .fragment_storage import metadata_guarded
from .free_iota_reduction import free_iota_reductions
from .free_iota_reduction import owned_iota_reduction_axes
from .independent_reduction import independent_reduction_coordinates
from .local_atomic import local_atomic_allocations
from .local_atomic import prove_local_atomics
from .local_atomic import terminal_finalizer_inputs
from .local_atomic import terminal_loop_symbols
from .private_scalar_loops import private_scalar_loop_nodes
from .private_scalar_loops import privatize_scalar_loop
from .register_loads import host_load_is_readonly
from .register_loads import lane_private_load
from .tcgen05_config import CuteTcgen05Config
from .warp_results import warp_result_chain

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
    logical_domain: Callable[[tuple[str, ...]], tuple[str, ...]] | None = None

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

    def domain(self, indices: tuple[str, ...]) -> tuple[str, ...]:
        """Coordinate ownership, independent of a value or memory-load mask."""
        return tuple(
            dict.fromkeys(
                (
                    *(
                        f"0 <= ({index}) and ({index}) < ({size})"
                        for index, size in zip(indices, self.shape, strict=True)
                    ),
                    *(self.logical_domain(indices) if self.logical_domain else ()),
                )
            )
        )

    def broadcast_domain(self, indices: tuple[str, ...]) -> tuple[str, ...]:
        indices = indices[len(indices) - len(self.shape) :] if self.shape else ()
        return self.domain(
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
        self.threads = cast("int", self.df.config.get("cute_fragment_threads", 128))
        self.thread = ""
        self.sym_indices: dict[sympy.Symbol, str] = {}
        self.expression: FragmentExpression | None = None
        self.local_allocations: frozenset[Node] = frozenset()
        self.local_logical_sizes: dict[int, int] = {}
        self.local_storage: list[Fragment] = []
        self.pending_local_atomics: set[str] = set()
        self.scalar_ordered_tickets: dict[Node, Fragment] = {}
        self.graphs = cg.codegen_graphs if graphs is None else graphs
        self.private_scalar_loops = (
            private_scalar_loop_nodes(self.graphs)
            if self.df.config.get("cute_fragment_private_scalar_loops", False)
            else frozenset()
        )
        self.warp_result_nodes: dict[Node, int] = {}
        if self.df.config.get("cute_fragment_warp_results", False):
            for graph in self.graphs:
                for node in graph.graph.nodes:
                    self.warp_result_nodes.update(
                        warp_result_chain(node, self.env, self.threads)
                    )

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
        return metadata_guarded(self.env, expr)

    def configured_expr(self, value: sympy.Basic) -> sympy.Basic:
        return configured_fragment_expr(self.env, value, self.df.resolved_block_size)

    def extent(self, value: object) -> int:
        resolved = self.static_extent(value)
        if resolved is None:
            raise exc.InvalidConfig(
                f"computed fragments require static local extents: {value}"
            )
        return resolved

    def static_extent(self, value: object) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, torch.SymInt):
            value = value._sympy_()
        assert isinstance(value, sympy.Expr)
        value = self.configured_expr(value)
        value = self.env.specialize_expr(sympy.sympify(value))
        return int(value) if value.is_number else None

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

    def host_stride(self, tensor: HostTensor, dim: int) -> str:
        # Fresh wrapper allocations have configured symbolic strides, which
        # need the fragment block-size resolver. Other host layouts must use
        # the ordinary metadata argument/specialization proof: fake stride
        # factors alone need not have a recoverable host symbol origin.
        if self.env.tensor_layout_is_symbolically_exact(tensor.value) and all(
            tensor.value.untyped_storage() != source.untyped_storage()
            for source in self.env.input_sources
        ):
            return self.sym(tensor.value.stride(dim))
        return self.df.tensor_stride(tensor.value, dim).name

    def emit(self, source: str) -> None:
        self.cg.add_statement(statement_from_string(source))

    def synchronize(self) -> None:
        self.emit("cute.arch.sync_threads()")
        self.pending_local_atomics.clear()

    def synchronize_local_atomics(self) -> None:
        if self.pending_local_atomics:
            self.synchronize()

    def predicate(self, masks: list[str] | tuple[str, ...]) -> str:
        """Bound SDK short-circuit AST expansion without changing mask order.

        The CuTe preprocessor repeats the accumulated left operand in each
        short-circuit check. A flat N-way conjunction therefore expands
        exponentially even when its DAG is small. Name successive prefixes in
        this exact lexical body; each binary conjunction keeps Python/DSL
        short-circuit behavior, and no value is cached across a later mutation.
        """
        source = " and ".join(masks) or "True"
        expression = expr_from_string(source)
        if not isinstance(expression, ast.BoolOp):
            return source
        operands: list[ast.expr] = []

        def flatten(value: ast.expr) -> None:
            if isinstance(value, ast.BoolOp) and type(value.op) is type(expression.op):
                for child in value.values:
                    flatten(child)
            else:
                operands.append(value)

        flatten(expression)
        if len(operands) <= 3:
            return source
        operation = "and" if isinstance(expression.op, ast.And) else "or"
        current = self.df.new_var("fragment_predicate")
        self.emit(f"{current} = {ast.unparse(operands[0])}")
        for operand in operands[1:]:
            previous = current
            current = self.df.new_var("fragment_predicate")
            self.emit(f"{current} = {previous} {operation} ({ast.unparse(operand)})")
        return current

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
            self.smem_bytes += aligned_shared_bytes(count, value.dtype)
        else:
            capacity, name, index = selected
            # A dead allocation can be recycled even when its last user was
            # an unused atomic. Finish that lifetime before ordinary writes.
            if name in self.pending_local_atomics:
                self.synchronize_local_atomics()
            if capacity >= count:
                count = capacity
            else:
                # All allocations are declared at the root. A dead buffer can
                # grow for its next lifetime without retaining a separate size
                # class in shared memory; earlier flat indices stay valid.
                self.buffers[index] = (name, value.dtype, count)
                self.smem_bytes += aligned_shared_bytes(
                    count, value.dtype
                ) - aligned_shared_bytes(capacity, value.dtype)
        return Fragment(
            value.shape,
            value.dtype,
            lambda indices: f"{name}[{self.flatten(indices, value.shape)}]",
            True,
            (),
            name,
            value.logical_domain,
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
        warp_owned: bool = False,
        synchronize: bool = True,
    ) -> None:
        if warp_owned:
            assert threads_per_element == 1
            threads_per_element = 32
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
            if warp_owned:
                branch = cast(
                    "ast.If",
                    statement_from_string(f"if {self.thread} % 32 == 0:\n    pass"),
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    body(self.coordinates(index, shape))
                self.cg.add_statement(branch)
            else:
                body(self.coordinates(index, shape))
        self.cg.add_statement(loop)
        if synchronize:
            self.synchronize()

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

    def aggregate_local_updates(
        self,
        shape: tuple[int, ...],
        storage: str,
        variables: tuple[str, ...],
        prepare: Callable[[tuple[str, ...]], None],
    ) -> None:
        """Aggregate one private relaxed Int32 update epoch, without new barriers.

        All CTA threads enter every round. Only lanes in the ballot execute
        match/redux, using the same member mask for each equal wrapped address.
        No observer can read the private target until its existing epoch barrier.
        """
        assert self.threads % 32 == 0
        size = math.prod(shape)
        iteration = self.df.new_var("fragment_atomic_round")
        index = self.df.new_var("fragment_atomic_element")
        address, value, active = variables
        loop = cast(
            "ast.For",
            statement_from_string(
                f"for {iteration} in range({(size + self.threads - 1) // self.threads}):\n    pass"
            ),
        )
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            self.emit(f"{index} = {iteration} * {self.threads} + {self.thread}")
            # Dominating definitions are required by CuTe's staged SSA lowering.
            self.emit(f"{address} = cutlass.Int64(0)")
            self.emit(f"{value} = cutlass.Int32(0)")
            self.emit(f"{active} = False")
            valid = cast(
                "ast.If", statement_from_string(f"if {index} < {size}:\n    pass")
            )
            valid.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", valid.body)):
                prepare(self.coordinates(index, shape))
            self.cg.add_statement(valid)
            members = self.df.new_var("fragment_atomic_members")
            peers = self.df.new_var("fragment_atomic_peers")
            total = self.df.new_var("fragment_atomic_total")
            self.emit(f"{members} = cute.arch.vote_ballot_sync({active})")
            self.emit(
                f"if {active}:\n"
                f"    {peers} = cute.arch.match_sync({members}, {address})\n"
                f"    {total} = cute.arch.warp_redux_sync({value}, 'add', {peers})\n"
                f"    if ({peers} & cute.arch.lanemask_lt()) == 0:\n"
                f"        cute.arch.atomic_add(({storage}.iterator + {address}).llvm_ptr, "
                f"{total}, sem='relaxed', scope='cta')"
            )
        self.cg.add_statement(loop)

    def atomic_add(self, node: Node, values: dict[Node, object]) -> Fragment | None:
        """Each contribution is owned once; barriers are outside lane loops."""
        target, indices, value, sem = cast(
            "tuple[object, object, object, object]", _resolve(node.args, values)
        )
        assert isinstance(target, (Fragment, HostTensor))
        assert isinstance(indices, (tuple, list))
        fake = cast("torch.Tensor", node.meta["val"])
        shape = self.shape(fake.shape)
        target_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])
        if len(indices) != target_fake.ndim or any(
            isinstance(i, slice) or i is None for i in indices
        ):
            raise exc.InvalidConfig(
                "fragment atomics require scalar/tensor indices for every axis"
            )
        local = isinstance(target, Fragment)
        if sem != "relaxed" and (local or target_fake.dtype != torch.int32):
            raise exc.InvalidConfig(
                "ordered fragment atomics require a global int32 target"
            )
        if node.users and (local or target_fake.dtype != torch.int32):
            raise exc.InvalidConfig(
                "fragment atomic results require a global int32 target"
            )
        if isinstance(target, Fragment) and (
            target.storage is None or len(target.shape) != 1
        ):
            raise exc.InvalidConfig("CTA-local atomics require direct resident storage")
        if target_fake.dtype not in (torch.float32, torch.int32):
            raise exc.InvalidConfig("fragment atomic add supports float32/int32")

        if sem in ("release", "acq_rel"):
            # Each worker publishes its preceding global writes before any
            # elected scalar/vector owner performs the release operation.
            # A fence in only the owner would not publish other workers' writes.
            # Both operations are collective, outside contribution masks/loops.
            self.emit("cute.arch.fence_acq_rel_gpu()")
            self.synchronize()

        def logical_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            masks = [
                f"({coord}) < ({self.logical_axis_extent(fake, dim, shape[dim])})"
                for dim, coord in enumerate(coords)
            ]
            # Indexing can pad the result shape after an index tensor has
            # retained its concrete logical extent. Destination bounds do not
            # exclude those extra contributions when an index wraps or clamps.
            # Preserve each index operand's broadcast domain independently.
            index_nodes = cast("list[Node | int]", node.args[1])
            for source, index in zip(index_nodes, indices, strict=True):
                if not isinstance(index, Fragment):
                    continue
                index_fake = cast("torch.Tensor", cast("Node", source).meta["val"])
                offset = len(coords) - len(index.shape)
                for dim, capacity in enumerate(index.shape):
                    if capacity == 1:
                        continue  # A singleton broadcasts over the output axis.
                    extent = self.logical_axis_extent(index_fake, dim, capacity)
                    mask = f"({coords[offset + dim]}) < ({extent})"
                    if mask not in masks:
                        masks.append(mask)
                if index.logical_domain is not None:
                    for mask in index.broadcast_domain(coords):
                        if mask not in masks:
                            masks.append(mask)
            return tuple(masks)

        # The side effect is emitted once below, never embedded in a lazy
        # element recipe. Repeated consumers read this immutable snapshot.
        warp_groups = self.warp_result_nodes.get(node)
        warp_owned = warp_groups is not None and math.prod(shape) == warp_groups
        register = (
            self.df.new_var("fragment_warp_atomic")
            if node.users and warp_owned
            else None
        )
        if register is not None:
            self.emit(f"{register} = {self.cast('0', target_fake.dtype)}")
        result = (
            Fragment(
                shape,
                target_fake.dtype,
                lambda _coords: cast("str", register),
                True,
                logical_domain=logical_domain,
            )
            if register is not None
            else self.allocate(
                Fragment(
                    shape,
                    target_fake.dtype,
                    lambda coords: "0",
                    logical_domain=logical_domain,
                )
            )
            if node.users
            else None
        )

        aggregate = (
            local
            and target_fake.dtype == torch.int32
            and sem == "relaxed"
            and not node.users
            and not warp_owned
            and self.df.config.get("cute_fragment_atomic_aggregation", False) is True
        )
        if aggregate:
            root = self.cg.current_root_graph_info
            assert root is not None
            aggregate = (
                root.graph_id
                in self.env.config_spec.cute_fragment_atomic_aggregation_root_ids
            )
        aggregate_vars = (
            tuple(
                self.df.new_var(name)
                for name in (
                    "fragment_atomic_address",
                    "fragment_atomic_value",
                    "fragment_atomic_active",
                )
            )
            if aggregate
            else None
        )

        def update(coords: tuple[str, ...]) -> None:
            masks = list(logical_domain(coords))
            positions = []
            for dim, index in enumerate(indices):
                position = (
                    index.broadcast(coords)
                    if isinstance(index, Fragment)
                    else self.scalar(index)
                )
                position = self.cg.lift(
                    expr_from_string(position), prefix="fragment_atomic_index"
                ).id
                # The logical allocation can be smaller than its padded capacity.
                size = self.sym(target_fake.shape[dim])
                if local:
                    assert isinstance(target, Fragment) and target.storage is not None
                    size = str(self.local_logical_sizes[id(target)])
                logical_extent = (
                    self.local_logical_sizes[id(target)]
                    if isinstance(target, Fragment)
                    else target_fake.shape[dim]
                )
                if (
                    isinstance(index, int)
                    and isinstance(logical_extent, int)
                    and not -logical_extent <= index < logical_extent
                ):
                    raise exc.InvalidConfig(
                        "fragment atomic static index is outside the logical target"
                    )
                # Python/PyTorch indexing wraps one valid negative extent.
                # Widen before adding a possibly 64-bit host extent; this is
                # address arithmetic, not a conversion of the update value.
                wrapped = self.df.new_var("fragment_atomic_wrapped")
                self.emit(f"{wrapped} = cutlass.Int64({position})")
                self.emit(
                    f"if {wrapped} < 0:\n    {wrapped} = {wrapped} + cutlass.Int64({size})"
                )
                masks.append(f"0 <= ({wrapped}) and ({wrapped}) < ({size})")
                positions.append(wrapped)
            if isinstance(target, Fragment):
                pointer = f"({target.storage}.iterator + ({positions[0]})).llvm_ptr"
                scope = "cta"
            else:
                offset = " + ".join(
                    f"cutlass.Int64({position}) * cutlass.Int64({self.host_stride(target, dim)})"
                    for dim, position in enumerate(positions)
                )
                pointer = f"({target.name}.iterator + ({offset})).llvm_ptr"
                scope = "gpu"
            contribution = (
                value.broadcast(coords)
                if isinstance(value, Fragment)
                else self.scalar(value)
            )
            if aggregate_vars is not None:
                address, update_value, active = aggregate_vars
                self.emit(f"{address} = cutlass.Int64({positions[0]})")
                self.emit(
                    f"if {self.predicate(masks)}:\n"
                    f"    {update_value} = {self.cast(contribution, torch.int32)}\n"
                    f"    {active} = {update_value} != 0"
                )
                return
            atomic = (
                f"cute.arch.atomic_add({pointer}, "
                f"{self.cast(contribution, target_fake.dtype)}, "
                f"sem={sem!r}, scope={scope!r})"
            )
            if result is None:
                self.emit(f"if {self.predicate(masks)}:\n    {atomic}")
            else:
                previous = self.df.new_var("fragment_atomic_previous")
                self.emit(f"{previous} = {self.cast('0', target_fake.dtype)}")
                self.emit(f"if {self.predicate(masks)}:\n    {previous} = {atomic}")
                self.emit(f"{result.read(coords)} = {previous}")

        if aggregate_vars is None:
            self.elements(shape, update, warp_owned=warp_owned, synchronize=not local)
        else:
            assert isinstance(target, Fragment) and target.storage is not None
            self.aggregate_local_updates(shape, target.storage, aggregate_vars, update)
        if local:
            assert isinstance(target, Fragment) and target.storage is not None
            self.pending_local_atomics.add(target.storage)
        if register is not None:
            # Preserve the existing atomic ordering barrier, including unrelated
            # alias loads. Only the return-value exchange becomes warp-local.
            self.emit(
                f"if {self.thread} // 32 < {warp_groups}:\n"
                f"    {register} = cute.arch.shuffle_sync({register}, 0, mask=0xffffffff, mask_and_clamp=31)"
            )
        if result is not None and not shape and sem in ("acquire", "acq_rel"):
            self.scalar_ordered_tickets[node] = result
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

            def view_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
                selected = tuple(
                    coord
                    for coord, index in zip(coords, indices, strict=False)
                    if index is not None
                )
                return (*selected, *coords[len(indices) :])

            def view(coords: tuple[str, ...]) -> str:
                return tensor.read(self.coordinate_locals(view_coordinates(coords)))

            return Fragment(
                shape,
                output.dtype,
                view,
                tensor.resident,
                dependencies=(tensor,),
                logical_domain=lambda coords: tensor.domain(view_coordinates(coords)),
            )
        assert isinstance(tensor, HostTensor)
        index_coordinates = memory_index_coordinates(
            self.env,
            [
                index.meta["val"] if isinstance(index, Node) else index
                for index in indices
            ],
            {
                i: value.shape
                for i, index in enumerate(indices)
                if isinstance(index, Node)
                and isinstance(value := values[index], Fragment)
            },
            shape,
        )

        def selected_coordinates(
            ordinal: int, coords: tuple[str, ...]
        ) -> tuple[str, ...]:
            return tuple(
                "0" if axis is None else coords[axis]
                for axis in index_coordinates[ordinal]
            )

        def address(coords: tuple[str, ...]) -> tuple[str, str]:
            tensor_indices: list[str] = []
            masks: list[str] = []
            dim = 0
            for ordinal, index in enumerate(indices):
                if index is None:
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
                    expression = selected_coordinates(ordinal, coords)[0]
                elif block is not None:
                    expression = f"({self.offsets[block]}) + ({selected_coordinates(ordinal, coords)[0]})"
                    masks.append(f"(({expression}) < ({self.bounds[block]}))")
                elif isinstance(index, Node):
                    value = values[index]
                    if isinstance(value, Fragment) and value.shape:
                        selected = selected_coordinates(ordinal, coords)
                        domain = value.domain(selected)
                        masks.extend(domain)
                        # Index fragments may have exact static capacity while
                        # the indexed output is padded. Read only coordinates
                        # owned by the index, before dereferencing host memory.
                        expression = self.df.new_var("fragment_tensor_index")
                        self.emit(f"{expression} = {self.cast('0', value.dtype)}")
                        branch = statement_from_string(
                            f"if {self.predicate(domain)}:\n    pass"
                        )
                        assert isinstance(branch, ast.If)
                        branch.body.clear()
                        with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                            self.emit(f"{expression} = {value.read(selected)}")
                        self.cg.add_statement(branch)
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
            offset = (
                " + ".join(
                    f"cutlass.Int64({index}) * cutlass.Int64({self.host_stride(tensor, dim)})"
                    for dim, index in enumerate(tensor_indices)
                )
                or "0"
            )
            return f"({tensor.name}.iterator + ({offset}))", self.predicate(masks)

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

            self.elements(
                shape,
                write,
                warp_owned=self.warp_result_nodes.get(node) == math.prod(shape),
            )
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

        index_domains: list[tuple[int, Fragment]] = []
        domain_bounds: list[tuple[int, str]] = []
        dim = 0
        for ordinal, index in enumerate(indices):
            if index is None:
                continue
            if isinstance(index, slice):
                domain_bounds.append((ordinal, self.sym(tensor.value.shape[dim])))
            elif isinstance(index, Node) and isinstance(values[index], Fragment):
                index_domains.append((ordinal, cast("Fragment", values[index])))
            elif isinstance(index, Node):
                proxy = index.meta["val"]
                if isinstance(proxy, torch.SymInt):
                    block = self.env.resolve_block_id(proxy)
                    if (
                        block in self.offsets
                        and proxy._sympy_() == self.env.block_sizes[block].var._sympy_()
                    ):
                        domain_bounds.append(
                            (
                                ordinal,
                                f"({self.bounds[block]}) - ({self.offsets[block]})",
                            )
                        )
            dim += 1

        def logical_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            return (
                *(
                    f"({selected_coordinates(ordinal, coords)[0]}) < ({bound})"
                    for ordinal, bound in domain_bounds
                ),
                *(
                    mask
                    for ordinal, index_value in index_domains
                    for mask in index_value.domain(
                        selected_coordinates(ordinal, coords)
                    )
                ),
            )

        loaded_fragment = Fragment(
            shape,
            output.dtype,
            load,
            logical_domain=logical_domain if index_domains or domain_bounds else None,
        )
        if (
            self.df.config.get("cute_fragment_register_loads", False)
            and math.prod(shape) <= self.threads
            and lane_private_load(node, self.env)
            and host_load_is_readonly(node, self.env, self.graphs)
        ):
            # One typed scalar per active lane, in the current lexical body.
            # The graph proof forbids remapping, carries and branch escapes.
            # Keep the ordinary atomic/store element loops and their barriers.
            name = self.df.new_var("fragment_register_load")
            self.emit(f"{name} = {self.cast('0', output.dtype)}")
            branch = statement_from_string(
                f"if {self.thread} < {math.prod(shape)}:\n    pass"
            )
            assert isinstance(branch, ast.If)
            branch.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                loaded = load(self.coordinates(self.thread, shape))
                self.emit(f"{name} = {self.cast(loaded, output.dtype)}")
            self.cg.add_statement(branch)
            return Fragment(
                shape,
                output.dtype,
                lambda _coords: name,
                True,
                logical_domain=loaded_fragment.logical_domain,
            )
        return self.materialize(loaded_fragment)

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

            if node in self.warp_result_nodes:
                groups = self.warp_result_nodes[node]
                name = self.df.new_var("fragment_warp_reduction")
                self.emit(f"{name} = {self.cast('0', fake.dtype)}")
                branch = cast(
                    "ast.If",
                    statement_from_string(
                        f"if {self.thread} // 32 < {groups}:\n    pass"
                    ),
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    coords = self.coordinates(
                        f"{self.thread} // 32", self.shape(fake.shape)
                    )
                    if warp:
                        self.emit(f"{name} = {reduce(coords)}")
                    else:
                        leader = cast(
                            "ast.If",
                            statement_from_string(
                                f"if {self.thread} % 32 == 0:\n    pass"
                            ),
                        )
                        leader.body.clear()
                        with self.cg.set_statements(cast("list[ast.AST]", leader.body)):
                            self.emit(f"{name} = {reduce(coords)}")
                        self.cg.add_statement(leader)
                        self.emit(
                            f"{name} = cute.arch.shuffle_sync({name}, 0, mask=0xffffffff, mask_and_clamp=31)"
                        )
                self.cg.add_statement(branch)
                # Keep the existing lifetime/mutation ordering boundary.
                self.synchronize()
                return Fragment(
                    self.shape(fake.shape), fake.dtype, lambda _coords: name, True
                )
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
        logical_dependencies = tuple(
            value for value in dependencies if value.logical_domain is not None
        )
        # These elementwise operations preserve broadcast coordinates. Other
        # lowerings (for example gather) define a different output domain.
        pointwise_domain = isinstance(node.target, torch._ops.OpOverload) and (
            torch.Tag.pointwise in node.target.tags
            or node.target is torch.ops.aten._to_copy.default
        )
        return Fragment(
            self.shape(fake.shape),
            fake.dtype,
            lambda coords: self.pointwise_read(
                producer, coords, lambda: self.cast(element(coords), fake.dtype)
            ),
            dependencies=dependencies,
            logical_domain=(
                lambda coords: tuple(
                    mask
                    for value in logical_dependencies
                    for mask in value.broadcast_domain(coords)
                )
            )
            if pointwise_domain and logical_dependencies
            else None,
        )

    def loop(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        graph_id = node.args[0]
        assert isinstance(graph_id, int)
        graph = self.graphs[graph_id]
        assert isinstance(graph, ForLoopGraphInfo)
        if self.local_allocations and any(
            self.df.resolved_block_size(bid) != 1 for bid in graph.block_ids
        ):
            raise exc.InvalidConfig("CTA-local atomics require uniform scalar loops")
        loop_index_type = "Int32"
        if self.local_allocations:
            parent, _slot = control_flow_parent_entries(self.graphs)[graph_id]
            owner = next(info for info in self.graphs if info.graph is parent.graph)
            if isinstance(owner, IfGraphInfo):
                self.uniform_finalizer_symbols(
                    terminal_loop_symbols(node, self.graphs), set(graph.block_ids)
                )
                # A proved scalar finalizer loop accepts runtime host integers.
                # Keep their signed 64-bit range through its induction variable;
                # narrowing to tile-index Int32 changes bounds and body indices.
                loop_index_type = "Int64"
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
                incoming = cast("Fragment", args[slot])
                carry = (
                    incoming
                    if id(incoming) in self.local_logical_sizes
                    else self.materialize(incoming, copy=True)
                )
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
                    and id(value) not in self.local_logical_sizes
                    else value
                )
            for target, source in zip(carries, snapshots, strict=True):
                if id(target) in self.local_logical_sizes:
                    if source is not target:
                        raise exc.InvalidConfig(
                            "CTA-local atomic loop changed its allocation"
                        )
                else:
                    self.copy(source, target)
            self.held.pop()
            self.held.pop()
            # The backedge can revisit the same atomic target. Finish this
            # iteration's epoch even when no loop carry was materialized.
            self.synchronize_local_atomics()
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
                f"for {self.offsets[bid]} in range(cutlass.{loop_index_type}({start}), cutlass.{loop_index_type}({self.bounds[bid]}), {step}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body = cast("list[ast.stmt]", body)
            body = [loop]
        self.offsets, self.bounds = outer_offsets, outer_bounds
        if node in self.private_scalar_loops:
            if len(body) != 1 or not isinstance(body[0], ast.For):
                raise exc.InvalidConfig("private scalar loops require one scalar axis")
            outgoing = {carry.storage for carry in carries}
            if None in outgoing or any(carry.shape for carry in carries):
                raise exc.InvalidConfig(
                    "private scalar loops require shared scalar carries"
                )
            body = list(
                privatize_scalar_loop(self, body[0], cast("set[str]", outgoing))
            )
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
        from ..autotuner_heuristics.cute_fragment_warp_scan import warp_scan_supported

        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        dim = cast("int", node.args[2]) % len(source.shape)
        root = self.cg.current_root_graph_info
        assert root is not None
        if (
            self.df.config.get("cute_fragment_warp_scan", False)
            and root.graph_id in self.env.config_spec.cute_fragment_warp_scan_root_ids
            and warp_scan_supported(source.dtype, source.shape[dim])
        ):
            return self.warp_scan(node, values)
        mode = self.df.config.get("cute_fragment_scan", "serial")
        if mode == "serial":
            return self.serial_scan(node, values)
        if mode == "cooperative":
            return self.cooperative_scan(node, values)
        raise exc.InvalidConfig(f"unsupported computed fragment scan: {mode!r}")

    def warp_scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        """Two CTA phases, each with full-warp, typed prefix exchanges.

        A warp owns one row chunk. The second phase scans at most 32 chunk
        totals, then adds the exclusive chunk carry to its own partials.
        No input is overwritten and no collective is guarded by a lane test.
        """
        from ..autotuner_heuristics.cute_fragment_warp_scan import warp_scan_supported

        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        dim = cast("int", node.args[2]) % len(source.shape)
        capacity = source.shape[dim]
        if not warp_scan_supported(source.dtype, capacity):
            raise exc.InvalidConfig(
                "warp-prefix scan requires Float32/64 or Int32/64 and axis capacity <= 1024"
            )
        assert self.threads % 32 == 0
        reverse = bool(node.args[3])
        fake = cast("torch.Tensor", node.meta["val"])
        extent = self.logical_axis_extent(fake, dim, capacity)
        row_shape = (*source.shape[:dim], *source.shape[dim + 1 :])
        chunks = (capacity + 31) // 32
        work_shape = (*row_shape, chunks)
        zero = self.cast("0", source.dtype)
        self.held.append(source)
        result = self.allocate(source)
        self.held.append(result)
        totals = self.allocate(Fragment(work_shape, source.dtype, lambda _: zero))
        self.held.append(totals)

        def prefix(value: str, lane: str) -> None:
            for distance in (1, 2, 4, 8, 16):
                peer = self.df.new_var("fragment_scan_peer")
                self.emit(
                    f"{peer} = cute.arch.shuffle_sync_up({value}, offset={distance}, "
                    "mask=0xffffffff, mask_and_clamp=0)"
                )
                self.emit(
                    f"if {lane} >= {distance}:\n"
                    f"    {value} = {self.cast(f'{peer} + {value}', source.dtype)}"
                )

        def positions(coords: tuple[str, ...]) -> tuple[str, str, tuple[str, ...]]:
            lane = self.df.new_var("fragment_scan_lane")
            rank = self.df.new_var("fragment_scan_rank")
            self.emit(f"{lane} = {self.thread} % 32")
            self.emit(f"{rank} = ({coords[-1]}) * 32 + {lane}")
            position = f"({extent}) - 1 - {rank}" if reverse else rank
            row = coords[:-1]
            return lane, rank, (*row[:dim], position, *row[dim:])

        def partials(coords: tuple[str, ...]) -> None:
            lane, rank, location = positions(coords)
            value = self.df.new_var("fragment_scan_value")
            self.emit(f"{value} = {zero}")
            # A lazy source read can emit statements. Keep all such statements
            # under the logical-bound guard, including reverse-tail addresses.
            branch = cast(
                "ast.If", statement_from_string(f"if {rank} < ({extent}):\n    pass")
            )
            branch.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                self.emit(f"{value} = {self.cast(source.read(location), source.dtype)}")
            branch.orelse = [statement_from_string(f"{value} = {zero}")]
            self.cg.add_statement(branch)
            prefix(value, lane)
            row = coords[:-1]
            padding = (*row[:dim], rank, *row[dim:])
            self.emit(
                f"if {rank} < ({extent}):\n"
                f"    {result.read(location)} = {value}\n"
                f"elif {rank} < {capacity}:\n"
                f"    {result.read(padding)} = {zero}"
            )
            last = self.df.new_var("fragment_scan_last")
            total = self.df.new_var("fragment_scan_total")
            self.emit(f"{last} = min(31, max(0, ({extent}) - ({coords[-1]}) * 32 - 1))")
            self.emit(
                f"{total} = cute.arch.shuffle_sync({value}, {last}, "
                "mask=0xffffffff, mask_and_clamp=31)"
            )
            self.emit(
                f"if {lane} == 0:\n"
                f"    {totals.read(coords)} = {total} if ({coords[-1]}) * 32 < ({extent}) else {zero}"
            )

        self.elements(work_shape, partials, threads_per_element=32)

        def carries(coords: tuple[str, ...]) -> None:
            lane, rank, location = positions(coords)
            value = self.df.new_var("fragment_scan_chunks")
            self.emit(
                f"{value} = {totals.read((*coords[:-1], lane))} if {lane} < {chunks} else {zero}"
            )
            prefix(value, lane)
            carry = self.df.new_var("fragment_scan_carry")
            self.emit(
                f"{carry} = cute.arch.shuffle_sync({value}, max(0, ({coords[-1]}) - 1), "
                "mask=0xffffffff, mask_and_clamp=31)"
            )
            # Chunk zero retains its partial bit-for-bit: adding an artificial
            # zero would change negative zero and is unnecessary.
            partial = result.read(location)
            self.emit(
                f"if {rank} < ({extent}) and ({coords[-1]}) > 0:\n"
                f"    {result.read(location)} = {self.cast(f'{carry} + {partial}', source.dtype)}"
            )

        self.elements(work_shape, carries, threads_per_element=32)
        self.held.pop()
        self.held.pop()
        self.held.pop()
        return result

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

    def uniform_finalizer_symbols(
        self, nodes: frozenset[Node], loop_blocks: set[int] | None = None
    ) -> None:
        for node in nodes:
            value = node.meta["val"]
            expression = value._sympy_() if isinstance(value, torch.SymInt) else value
            for symbol in sympy.sympify(expression).free_symbols:
                entry = HostFunction.current().expr_to_origin.get(symbol)
                origin = entry.origin if entry is not None else None
                if not isinstance(origin, HostOrigin) and not (
                    type(origin) is GridOrigin
                    and (
                        origin.block_id in self.offsets
                        or loop_blocks is not None
                        and origin.block_id in loop_blocks
                    )
                    and self.df.resolved_block_size(origin.block_id) == 1
                ):
                    raise exc.InvalidConfig(
                        "local finalizer requires proved uniform scalar origins"
                    )

    def conditional(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        if self.local_allocations:
            tickets, symbols = terminal_finalizer_inputs(node, self.graphs)
            predicate_value = values[cast("Node", node.args[0])]
            predicate_buffers = self.referenced_buffers([predicate_value])
            for ticket in tickets:
                result = self.scalar_ordered_tickets.get(ticket)
                if (
                    result is None
                    or result.shape != ()
                    or not result.resident
                    or result.storage is None
                    or result.storage not in predicate_buffers
                    or ticket in self.warp_result_nodes
                ):
                    raise exc.InvalidConfig(
                        "local finalizer requires one shared scalar ticket and CTA broadcast"
                    )
            self.uniform_finalizer_symbols(symbols)
        info = self.graphs[cast("int", node.args[1])]
        assert isinstance(info, IfGraphInfo)
        assert info.branches_outputs is not None
        # An unchanged branch output can be captured only by the other branch.
        # Both lists bind values in this outer graph, not branch-local results.
        outer_captures: dict[str, Node] = {}
        for side, names in enumerate((info.if_arg_names, info.else_arg_names)):
            assert names is not None
            for name, source in zip(
                names, cast("list[Node]", node.args[3 + side]), strict=True
            ):
                if name in outer_captures and outer_captures[name] is not source:
                    raise exc.InvalidConfig(
                        "computed fragment conditional has conflicting outer captures"
                    )
                outer_captures[name] = source
        for slots in info.branches_outputs:
            if any(
                isinstance(slot, str) and slot not in outer_captures for slot in slots
            ):
                raise exc.InvalidConfig(
                    "computed fragment conditional requires a captured unchanged output"
                )
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
                    source = (
                        outputs[slot]
                        if isinstance(slot, int)
                        else values[outer_captures[slot]]
                    )
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
        from .producer_cache import producer_cache_candidates

        cache_nodes = (
            producer_cache_candidates(graph, self.shape)
            if self.df.config.get("cute_fragment_producer_cache", False)
            and not self.scopes
            else frozenset()
        )
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
                    if node in cache_nodes:
                        value = values[node]
                        assert isinstance(value, Fragment)
                        values[node] = self.materialize(value)
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
        # CTA-local targets have a proved initialize/update/final-read lifetime.
        # Coalesce only updates to disjoint targets. Repeated updates to one
        # target retain their collective epoch boundary, including floating
        # association across threads. Finish pending writes before a read,
        # storage reuse, or conservative effect boundary.
        # Alias-only nodes above do not access memory. Check physical storage,
        # including lazy dependencies, so consumers cannot hide an alias.
        local_update = (
            target is atomic_ops.atomic_add
            and isinstance(args[0], Fragment)
            and id(args[0]) in self.local_logical_sizes
            and not node.users
        )
        read_args = args[1:] if local_update else args
        if (
            target is _tracing_ops._if
            or _tracing_ops.is_for_loop_target(target)
            or (target is atomic_ops.atomic_add and not local_update)
            or (
                local_update
                and cast("Fragment", args[0]).storage in self.pending_local_atomics
            )
            or self.pending_local_atomics
            & self.referenced_buffers(
                [
                    *read_args,
                    *(_resolve(value, values) for value in node.kwargs.values()),
                ]
            )
        ):
            # A loop can execute zero times. Never let a barrier generated only
            # in its body discharge a write from before that loop.
            self.synchronize_local_atomics()
        if target is _tracing_ops._if:
            return self.conditional(node, values)
        if _tracing_ops.is_for_loop_target(target):
            return self.loop(node, values)
        if target is atomic_ops.atomic_add:
            return self.atomic_add(node, values)
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
            declared_shape = (
                self.shape(args[0])
                if target in (creation_ops.full, torch.ops.aten.full.default)
                else shape
            )
            result = Fragment(
                shape,
                fake.dtype,
                lambda _: self.cast(self.scalar(value), fake.dtype),
                dependencies=(value,) if isinstance(value, Fragment) else (),
                logical_domain=(
                    lambda coords: tuple(
                        f"({coord}) < ({size})"
                        for coord, size in zip(coords, declared_shape, strict=True)
                    )
                )
                if declared_shape != shape
                else None,
            )
            if node in self.local_allocations:
                result = self.materialize(result)
                assert result.storage is not None
                self.local_storage.append(result)
                self.local_logical_sizes[id(result)] = cast("list[int]", node.args[0])[
                    0
                ]
            return result
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
                logical_domain=source.logical_domain,
            )
        if target is _tracing_ops._mask_to:
            source_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])

            def masked(coords: tuple[str, ...]) -> str:
                # A local allocation can have padded physical capacity while
                # retaining its declared logical domain. Its initialized pad
                # values are not necessarily neutral for this reduction.
                masks = list(source.domain(coords)) if source.logical_domain else []
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
                mask = self.predicate(masks)
                return f"({self.cast(source.read(coords), fake.dtype)} if {mask} else {self.cast(self.scalar(args[1]), fake.dtype)})"

            return Fragment(
                shape,
                fake.dtype,
                masked,
                dependencies=(source,),
                logical_domain=source.logical_domain,
            )
        if target is view_ops.subscript:
            slices = cast("list[object]", args[1])

            def subscript_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
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
                return tuple(result)

            def subscript(coords: tuple[str, ...]) -> str:
                return source.read(
                    self.coordinate_locals(subscript_coordinates(coords))
                )

            return Fragment(
                shape,
                fake.dtype,
                subscript,
                source.resident,
                (source,),
                logical_domain=lambda coords: source.domain(
                    subscript_coordinates(coords)
                ),
            )
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
                logical_domain=lambda coords: source.domain(
                    tuple(coords[order.index(i)] for i in range(len(order)))
                ),
            )
        if target in (torch.ops.aten.expand.default, torch.ops.aten.clone.default):
            return Fragment(
                shape,
                fake.dtype,
                source.broadcast,
                source.resident,
                (source,),
                logical_domain=source.broadcast_domain,
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
            def view_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
                return self.coordinates(self.flatten(coords, shape), source.shape)

            def view_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
                # Indexing may pad a singleton view without preserving numel.
                # Bound the flat coordinate before unflattening can wrap it.
                return (
                    f"({self.flatten(coords, shape)}) < ({math.prod(source.shape)})",
                    *source.domain(view_coordinates(coords)),
                )

            return Fragment(
                shape,
                fake.dtype,
                lambda coords: source.read(
                    self.coordinate_locals(view_coordinates(coords))
                ),
                source.resident,
                (source,),
                logical_domain=view_domain,
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
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    physical_axes: frozenset[int] = frozenset(),
    owned_iota_axes: frozenset[int] = frozenset(),
) -> bool:
    """Structural root ownership shared by search discovery and code generation.

    Exact local sizes, strides, scalar predicates and shared capacity remain
    configuration-dependent checks in the emitter.
    """
    # Jagged loops carry per-parent bounds in their first captured tensor.
    # This owner currently tracks only scalar loop ends; preserve the ordinary
    # jagged lowering instead of treating the maximum end as every row's end.
    if any(
        isinstance(info, ForLoopGraphInfo)
        and any(env.is_jagged_tile(bid) for bid in info.block_ids)
        for info in graphs
    ):
        return False
    graph_by_id = {info.graph_id: info for info in graphs}
    independent_reductions = independent_reduction_coordinates(env, graphs)
    free_reductions = {
        node
        for node in free_iota_reductions(env, graphs)
        if node.meta["lowering"].block_index not in owned_iota_axes
    }
    captured_reductions = captured_reduction_coordinates(
        env, graphs, physical_axes=physical_axes
    )
    local_allocations = local_atomic_allocations(graphs)
    if not local_allocations and any(
        node.target is atomic_ops.atomic_add
        for info in graphs
        for node in info.graph.nodes
    ):
        return False

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
        if (
            node in local_allocations
            or node in independent_reductions
            or node in captured_reductions
            or node in free_reductions
        ):
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
        atomic_ops.atomic_add,
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

    env = CompileEnvironment.current()
    if env.backend_name != "cute":
        return False
    root = cg.current_root_graph_info
    assert root is not None
    threads_required = (
        cg.device_function.config.get("cute_fragment_threads", 128) != 128
        and root.graph_id in env.config_spec.cute_fragment_thread_root_ids
    )
    producer_cache_required = (
        cg.device_function.config.get("cute_fragment_producer_cache", False)
        and root.graph_id in env.config_spec.cute_fragment_producer_cache_root_ids
    )
    warp_scan_required = (
        cg.device_function.config.get("cute_fragment_warp_scan", False)
        and root.graph_id in env.config_spec.cute_fragment_warp_scan_root_ids
    )
    register_loads_required = (
        cg.device_function.config.get("cute_fragment_register_loads", False)
        and root.graph_id in env.config_spec.cute_fragment_register_load_root_ids
    )
    warp_results_required = (
        cg.device_function.config.get("cute_fragment_warp_results", False)
        and root.graph_id in env.config_spec.cute_fragment_warp_result_root_ids
    )
    local_required = bool(local_atomic_allocations(cg.host_function.device_ir.graphs))
    owned_iota_axes = (
        frozenset()
        if threads_required
        or register_loads_required
        or producer_cache_required
        or warp_scan_required
        or cg.device_function.config.get("cute_fragment_reduction", "serial")
        != "serial"
        else owned_iota_reduction_axes(cg)
    )
    free_required = any(
        node.graph is cg.host_function.device_ir.graphs[root.graph_id].graph
        and node.meta["lowering"].block_index not in owned_iota_axes
        for node in free_iota_reductions(env, cg.host_function.device_ir.graphs)
    )
    physical_axes = (
        frozenset()
        if threads_required
        or register_loads_required
        or producer_cache_required
        or warp_scan_required
        or warp_results_required
        or cg.device_function.config.get("cute_fragment_scan", "serial") != "serial"
        or cg.device_function.config.get("cute_fragment_reduction", "serial")
        != "serial"
        else physical_capture_axes(cg)
    )
    captured_required = bool(
        captured_reduction_coordinates(
            env,
            cg.host_function.device_ir.graphs,
            root_graph_id=root.graph_id,
            physical_axes=physical_axes,
        )
    )
    fragment_schedule_required = (
        (
            cg.device_function.config.get("cute_fragment_private_scalar_loops", False)
            and root.graph_id
            in env.config_spec.cute_fragment_private_scalar_loop_root_ids
        )
        or threads_required
        or register_loads_required
        or producer_cache_required
        or warp_scan_required
        or warp_results_required
        or (
            cg.device_function.config.get("cute_fragment_scan", "serial")
            == "cooperative"
            and root.graph_id in env.config_spec.cute_fragment_scan_root_ids
        )
        or (
            cg.device_function.config.get("cute_fragment_reduction", "serial") == "warp"
            and root.graph_id in env.config_spec.cute_fragment_reduction_root_ids
        )
    )
    state = cg.device_function.cute_state
    if any(
        plan is not None
        for plan in (
            state.block_scaled_plan,
            state.chunk_prepare_plan,
            state.chunk_recurrence_plan,
            state.single_token_rank1_plan,
            state.split_single_token_rank1_plan,
            state.fixed_token_rank1_plan,
            state.attention_flash_block_ids,
            state.attention_flash_bwd_block_ids,
        )
    ):
        # Pre-codegen selected a complete root owner. It must execute its
        # late layout/alias checks; an implicit coordinate requirement must
        # not bypass those checks by replacing its producer with fragments.
        if local_required or fragment_schedule_required:
            raise exc.InvalidConfig(
                "a planned root cannot share a computed fragment schedule"
            )
        return False
    if free_required:
        # A new complete owner changes the placement of global stores relative
        # to vector reductions. Separate destinations need a disjointness
        # proof; declining into the unowned scalar iota path would lose the
        # complete reduction. Repeated stores through one tensor retain their
        # existing ordered semantics.
        destinations = {
            cg.device_function.tensor_arg(target.meta["val"]).name
            for info in cg.host_function.device_ir.graphs
            for node in info.graph.nodes
            if node.target is memory_ops.store
            and isinstance(target := node.args[0], Node)
            and isinstance(target.meta.get("val"), torch.Tensor)
            and target.meta["val"] in HostFunction.current().tensor_to_origin
        }
        disjoint = cg.device_function.proven_disjoint_tensor_pairs()
        if any(
            frozenset((left, right)) not in disjoint
            for left in destinations
            for right in destinations
            if left < right
        ):
            if local_required or fragment_schedule_required:
                raise exc.InvalidConfig(
                    "fragment store destinations require disjoint storage"
                )
            raise exc.BackendUnsupported(
                "cute", "fragment store destinations require disjoint storage"
            )
    if (
        cg.device_function.config.get(
            "cute_affine_scan_schedule", DIRECT_AFFINE_ORDINARY_SCHEDULE
        )
        != DIRECT_AFFINE_ORDINARY_SCHEDULE
    ):
        # The direct affine schedule must first emit its ordinary producer,
        # then prove and replace that entire region. Implicit free-iota and
        # capture ownership must not erase the producer before that proof.
        if (
            local_required
            or fragment_schedule_required
            or cg.device_function.config.get("cute_reduction_schedule", "scalar")
            != "scalar"
        ):
            raise exc.InvalidConfig(
                "direct affine schedules cannot share a computed fragment root"
            )
        return False
    if cg.device_function.config.get("cute_reduction_schedule", "scalar") != "scalar":
        # An explicit resident schedule owns its rectangular carries and
        # performs its own complete proof during materialization. Replacing
        # its scalar producer here erases the structure that proof consumes.
        if local_required or free_required or fragment_schedule_required:
            raise exc.InvalidConfig(
                "resident reduction schedules cannot share a computed fragment root"
            )
        return False
    if (
        len(cg.host_function.device_ir.root_ids) != 1
        or cg.device_function.config.get("cute_collective_mma", False)
        or cg.device_function.config.get("cute_register_chain", False)
    ):
        if local_required:
            raise exc.InvalidConfig(
                "CTA-local atomics require a complete fragment root"
            )
        if free_required:
            raise exc.InvalidConfig(
                "free iota reductions require a complete fragment root"
            )
        if (
            captured_required
            or threads_required
            or register_loads_required
            or producer_cache_required
            or warp_scan_required
            or warp_results_required
        ):
            raise exc.InvalidConfig(
                "captured full reductions require a computed fragment root"
                if captured_required
                else "CTA threads/register loads require a computed fragment root"
            )
        return False
    graphs = (
        cg.host_function.device_ir.build_codegen_graphs(
            cg.device_function.config, roll_reductions=False
        )
        if captured_required or local_required or free_required
        else cg.codegen_graphs
    )

    scan_required = warp_scan_required or (
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
        if free_required:
            raise exc.InvalidConfig(
                "free iota reductions require a supported complete fragment root"
            )
        if local_required:
            raise exc.InvalidConfig(
                "CTA-local atomics require a supported complete fragment root"
            )
        if (
            captured_required
            or scan_required
            or reduction_required
            or threads_required
            or register_loads_required
            or producer_cache_required
            or warp_scan_required
            or warp_results_required
        ):
            raise exc.InvalidConfig(
                "captured full reductions require a computed fragment root"
                if captured_required
                else "cooperative scan requires a computed fragment root"
                if scan_required
                else "warp reduction requires a computed fragment root"
                if reduction_required
                else "CTA threads/register loads require a computed fragment root"
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
    if not computed_fragment_supported(
        CompileEnvironment.current(),
        graphs,
        physical_axes=physical_axes,
        owned_iota_axes=owned_iota_axes,
    ):
        return decline()
    # This owner implements configured reduction tiling directly. Keep the
    # logical producer graph, including every other codegen transformation,
    # rather than applying scalar graph rolling before fragment ownership.
    if not captured_required and not local_required and not free_required:
        graphs = cg.host_function.device_ir.build_codegen_graphs(
            cg.device_function.config, roll_reductions=False
        )
    if not computed_fragment_supported(
        CompileEnvironment.current(),
        graphs,
        physical_axes=physical_axes,
        owned_iota_axes=owned_iota_axes,
    ):
        return decline()
    root = graphs[root.graph_id]
    compiler = FragmentCompiler(cg, graphs)
    compiler.local_allocations = prove_local_atomics(graphs)
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
            lowering = node.meta.get("lowering")
            if not isinstance(lowering, (PointwiseLowering, ReductionLowering)):
                continue
            ranges = lowering.buffer.data.ranges
            if isinstance(lowering, ReductionLowering):
                assert isinstance(lowering.buffer.data, Reduction)
                ranges = [*ranges, *lowering.buffer.data.reduction_ranges]
            if any(compiler.static_extent(dim) is None for dim in ranges):
                return decline()
            for source in node.all_input_nodes:
                value = source.meta.get("val")
                if not isinstance(value, torch.Tensor):
                    if isinstance(
                        value,
                        (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool),
                    ):
                        continue
                    return decline()
                shape = tuple(compiler.static_extent(dim) for dim in value.shape)
                strides = tuple(compiler.static_extent(dim) for dim in value.stride())
                if any(dim is None for dim in (*shape, *strides)):
                    # Ordinary lowering can retain host constexpr dimensions.
                    # This owner requires proved capacities and coordinate strides.
                    return decline()
                span = 1
                for stride, size in sorted(
                    zip(
                        cast("tuple[int, ...]", strides),
                        cast("tuple[int, ...]", shape),
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
    if local_required:
        if any(
            cg.device_function.resolved_block_size(bid) != 1
            for bid in grid.block_id_to_info
        ):
            raise exc.InvalidConfig("CTA-local atomics require scalar grid owners")
    compiler.begin()
    for bid, info in grid.block_id_to_info.items():
        compiler.offsets[bid] = grid.strategy.grid_origin_var(bid)
        compiler.bounds[bid] = cast("str", info.end_var_name)
    body: list[ast.AST] = []
    with cg.set_statements(body):
        compiler.emit(f"{compiler.thread} = cutlass.Int32(cute.arch.thread_idx()[0])")
        compiler.graph(root.graph, {})
        # Root lifetime can be nested in an outer generated grid loop.
        compiler.synchronize_local_atomics()
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(compiler.env.device)
    if local_required and not capacity:
        raise exc.InvalidConfig(
            "CTA-local atomics require a known shared memory capacity"
        )
    if producer_cache_required and not capacity:
        raise exc.InvalidConfig(
            "shared producer caches require a known shared memory capacity"
        )
    if capacity and compiler.smem_bytes > capacity:
        raise exc.InvalidConfig(
            f"computed fragments need {compiler.smem_bytes} shared bytes, exceeding {capacity}"
        )
    cg.device_function.cute_state.owned_root_block_dims = (compiler.threads, 1, 1)
    for statement in (*compiler.allocations(), *body):
        cg.add_statement(statement)
    return True
