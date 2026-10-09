"""Scalarize a tensor program over statically owned subgroup registers.

Static views, gathers and predicates describe logical coordinates. Walking only
the demanded output coordinates removes unused scalar operations before lowering
each register read to a local reference or a subgroup shuffle. The program does
not attach any meaning to its values, permutations, or selected output extent.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
import math
import operator
import struct
import textwrap
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codegen.simd import constant_repr
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx import Graph
from torch.fx import GraphModule
from torch.fx import Node
from torch.fx.node import map_arg

from ... import exc
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from .register_tensor import emit_register_tensor
from .register_tensor import plan_register_tensor
from .row_fragment import RowFragment
from .row_fragment import RowFragmentEmitter
from .row_fragment import RowFragmentLayout
from .row_fragment import row_fragment_tensor_inputs

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Sequence

    from torch.fx.node import Argument

    from ..generate_ast import GenerateAST


_FLAT_VIEWS = frozenset(
    {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.alias.default,
    }
)
_INDEX_OPS = frozenset(
    {
        *_FLAT_VIEWS,
        torch.ops.aten.arange.default,
        torch.ops.aten.arange.start,
        torch.ops.aten.arange.start_step,
        torch.ops.aten.expand.default,
        torch.ops.aten.gather.default,
        torch.ops.aten.index_select.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
        torch.ops.aten.cat.default,
        torch.ops.aten.stack.default,
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.full.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.prims.iota.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.view.dtype,
    }
)
_REDUCTIONS: dict[object, str] = {
    torch.ops.aten.sum.dim_IntList: "sum",
    torch.ops.aten.sum.default: "sum",
    torch.ops.aten.any.dim: "any",
    torch.ops.aten.any.dims: "any",
    torch.ops.aten.any.default: "any",
    torch.ops.aten.all.dim: "all",
    torch.ops.aten.all.dims: "all",
    torch.ops.aten.all.default: "all",
}
_EXTRA_POINTWISE = frozenset(
    {
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.view.dtype,
        torch.ops.aten.floor_divide.default,
        torch.ops.aten.floor_divide.Scalar,
    }
)
_SCALAR_OPERATORS = frozenset(
    {
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
        operator.lshift,
        operator.rshift,
        operator.and_,
        operator.or_,
        operator.xor,
        operator.neg,
        operator.pos,
        operator.invert,
        operator.not_,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    }
)


@dataclass(frozen=True)
class _Element:
    node: Node
    index: int


@dataclass(frozen=True)
class _Literal:
    value: bool | int | float = field(compare=False)
    dtype: torch.dtype
    _key: bool | int | bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Numeric equality would merge signed zeros (and lose NaN payloads).
        key = (
            struct.pack("!d", self.value)
            if isinstance(self.value, float)
            else self.value
        )
        object.__setattr__(self, "_key", key)


_Value = _Element | _Literal
_Lanes = tuple[_Value | None, ...]


def _shape(node: Node) -> tuple[int, ...]:
    tensor = node.meta["val"]
    if not isinstance(tensor, torch.Tensor) or any(
        type(size) is not int for size in tensor.shape
    ):
        raise exc.BackendUnsupported("cute", "register program needs static tensors")
    return tuple(tensor.shape)


def _coordinates(index: int, shape: tuple[int, ...]) -> list[int]:
    coords = []
    for size in reversed(shape):
        coords.append(index % size)
        index //= size
    assert index == 0
    return list(reversed(coords))


def _flat(coords: Sequence[int], shape: tuple[int, ...]) -> int:
    index = 0
    for coordinate, size in zip(coords, shape, strict=True):
        assert 0 <= coordinate < size
        index = index * size + coordinate
    return index


def _broadcast(index: int, shape: tuple[int, ...], source: tuple[int, ...]) -> int:
    if shape == source:
        return index
    coords = _coordinates(index, shape)
    if not source:
        return 0
    coords = coords[-len(source) :]
    return _flat(
        [0 if size == 1 else i for i, size in zip(coords, source, strict=True)], source
    )


def _specialize_constant_captures(
    module: GraphModule, operands: Sequence[object]
) -> GraphModule:
    """Bind literal conditional captures without modifying a cached branch.

    Dynamo can retain scalar placeholders after a shape-specialized make_fx
    trace, even when they are unused. Live scalar captures and arithmetic on
    them also become ordinary FX constants before register ownership is planned.
    """
    if all(isinstance(operand, Node) for operand in operands):
        return module
    graph = Graph()
    copied: dict[Node, Argument] = {}
    for placeholder, operand in zip(
        module.graph.find_nodes(op="placeholder"), operands, strict=True
    ):
        if isinstance(operand, Node):
            copied[placeholder] = graph.node_copy(placeholder, copied.__getitem__)
        else:
            copied[placeholder] = cast("Argument", operand)
    for node in module.graph.nodes:
        if node.op == "placeholder":
            continue
        if (
            node.op == "call_function"
            and node.target in _SCALAR_OPERATORS
            and all(not isinstance(copied[arg], Node) for arg in node.all_input_nodes)
        ):
            args = map_arg(node.args, copied.__getitem__)
            kwargs = map_arg(node.kwargs, copied.__getitem__)
            copied[node] = cast("Callable[..., Argument]", node.target)(*args, **kwargs)
        else:
            copied[node] = graph.node_copy(node, copied.__getitem__)
    return GraphModule(module, graph)


def register_program_supported(
    module: GraphModule, inputs: Collection[Node], *, lanes: int
) -> bool:
    """Check structural admission without emitting or executing tensor data."""
    if not 0 < lanes <= 32 or lanes & (lanes - 1):
        return False
    for node in module.graph.nodes:
        if node in inputs or node.op in ("get_attr", "output"):
            continue
        if node.op != "call_function":
            return False
        if node.target is torch.ops.higher_order.cond:
            predicate, true_graph, false_graph, arguments = node.args
            if not isinstance(predicate, Node) or lanes != 32:
                return False
            value = predicate.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                return False
            for branch in (true_graph, false_graph):
                if not isinstance(branch, Node) or branch.op != "get_attr":
                    return False
                child = operator.attrgetter(str(branch.target))(module)
                if not isinstance(child, GraphModule):
                    return False
                child = _specialize_constant_captures(child, arguments)
                if not register_program_supported(
                    child, tuple(child.graph.find_nodes(op="placeholder")), lanes=lanes
                ):
                    return False
            continue
        if node.target is operator.getitem:
            if not (
                isinstance(node.args[0], Node)
                and node.args[0].target is torch.ops.higher_order.cond
            ):
                return False
        elif not (
            isinstance(node.target, torch._ops.OpOverload)
            and not node.target._schema.is_mutable
            and torch.Tag.nondeterministic_seeded not in node.target.tags
            and (
                node.target in _INDEX_OPS
                or node.target in _EXTRA_POINTWISE
                or node.target in _REDUCTIONS
                or torch.Tag.pointwise in node.target.tags
            )
        ):
            return False
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor) or any(
            type(size) is not int or size <= 0 for size in value.shape
        ):
            return False
        if node.target in _REDUCTIONS:
            source = cast("Node", node.args[0]).meta["val"]
            dimensions = (
                node.args[1] if len(node.args) > 1 else list(range(source.ndim))
            )
            if isinstance(dimensions, int):
                dimensions = [dimensions]
            if not isinstance(dimensions, (list, tuple)) or any(
                type(dim) is not int for dim in dimensions
            ):
                return False
            dims = sorted(dim % source.ndim for dim in dimensions) or list(
                range(source.ndim)
            )
            if not dims or dims != list(range(dims[0], source.ndim)):
                return False
            outputs = math.prod(source.shape[: dims[0]])
            extent = math.prod(source.shape[dims[0] :])
            if outputs >= lanes:
                if outputs % lanes:
                    return False
            elif lanes % outputs or (outputs != 1 and extent % (lanes // outputs)):
                return False
    return True


def _lane_bit_terms(
    values: dict[int, int], lanes: int, output_bits: int
) -> tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Factor static lane values into affine bits and remaining lookup masks."""
    width = lanes.bit_length() - 1
    active = sum(1 << lane for lane in values)
    basis = [
        (1 << lanes) - 1,
        *[
            sum(1 << lane for lane in range(lanes) if lane & (1 << bit))
            for bit in range(width)
        ],
    ]
    affine = []
    for coefficients in sorted(range(1 << len(basis)), key=int.bit_count):
        truth = 0
        for bit, mask in enumerate(basis):
            if coefficients & (1 << bit):
                truth ^= mask
        affine.append((coefficients, truth))
    constant = 0
    shifted: dict[int, int] = defaultdict(int)
    lookups = []
    for output_bit in range(output_bits):
        truth = sum(
            1 << lane for lane, value in values.items() if value & (1 << output_bit)
        )
        for coefficients, candidate in affine:
            if (candidate ^ truth) & active == 0:
                if coefficients & 1:
                    constant |= 1 << output_bit
                for input_bit in range(width):
                    if coefficients & (1 << (input_bit + 1)):
                        shifted[output_bit - input_bit] |= 1 << input_bit
                break
        else:
            lookups.append((truth, output_bit))
    return constant, tuple(sorted(shifted.items())), tuple(lookups)


class RegisterProgramEmitter:
    """Demanded-coordinate code generation, independent of tensor semantics.

    The scalar callback is the ordinary backend pointwise lowering. Separating
    it from register routing also makes the routing program executable in CPU
    tests without CUDA initialization or a native compiler.
    """

    def __init__(
        self,
        module: GraphModule,
        inputs: dict[Node, RowFragment],
        *,
        lanes: int,
        lane_expr: str,
        new_var: Callable[[str], str],
        emit: Callable[[str], None],
        scalar: Callable[[Node, list[str], Callable[[str], None]], str],
        dtype_str: Callable[[torch.dtype], str],
    ) -> None:
        assert 0 < lanes <= 32 and lanes & (lanes - 1) == 0
        self.module = module
        self.inputs = inputs
        self.lanes = lanes
        self.lane = lane_expr
        self.new_var = new_var
        self.emit = emit
        self.scalar = scalar
        self.dtype_str = dtype_str
        self.constants: dict[Node, object] = {}
        self.constant_elements: dict[Node, list[bool | int | float]] = {}
        self.canonical: dict[_Element, _Value] = {}
        self.cache: dict[_Lanes, str] = {}
        self.register_values: dict[tuple[Node, int], str] = {}
        self.resident_selects: set[tuple[Node, int]] = set()
        self.reductions: dict[tuple[Node, int], str] = {}
        self.conditionals: dict[Node, tuple[RowFragment, ...]] = {}
        with unset_fake_temporarily():
            self._evaluate_constants()
        self._plan_resident_selects()

    def _evaluate_constants(self) -> None:
        def resolve(node: Node) -> Argument:
            return cast("Argument", self.constants[node])

        for node in self.module.graph.nodes:
            if node in self.inputs:
                continue
            if node.op == "get_attr":
                value = operator.attrgetter(str(node.target))(self.module)
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                self.constants[node] = value
            elif (
                node.op == "call_function"
                and isinstance(node.target, torch._ops.OpOverload)
                and not node.target._schema.is_mutable
                and (
                    node.target in _INDEX_OPS
                    or node.target in _EXTRA_POINTWISE
                    or torch.Tag.pointwise in node.target.tags
                )
                and all(argument in self.constants for argument in node.all_input_nodes)
            ):
                args = map_arg(node.args, resolve)
                kwargs = map_arg(node.kwargs, resolve)
                if "device" in kwargs:
                    kwargs = {**kwargs, "device": torch.device("cpu")}
                self.constants[node] = node.target(*args, **kwargs)

    def _constant(self, node: Node, index: int) -> _Literal:
        value = self.constants[node]
        assert isinstance(value, torch.Tensor)
        if node not in self.constant_elements:
            with unset_fake_temporarily():
                self.constant_elements[node] = value.reshape(-1).tolist()
        scalar = self.constant_elements[node][index]
        assert isinstance(scalar, (bool, int, float))
        return _Literal(scalar, value.dtype)

    def _argument(
        self, argument: object, index: int, shape: tuple[int, ...]
    ) -> _Element:
        if not isinstance(argument, Node):
            raise exc.BackendUnsupported("cute", "register tensor operand expected")
        return _Element(argument, _broadcast(index, shape, _shape(argument)))

    def _selected_argument(self, node: Node, index: int) -> _Element:
        shape = _shape(node)
        condition = cast("Node", node.args[0])
        predicate = self._constant(
            condition, _broadcast(index, shape, _shape(condition))
        )
        return self._argument(node.args[1 if predicate.value else 2], index, shape)

    def _plan_resident_selects(self) -> None:
        """Keep a lane-varying merge of distinct producers in one register.

        Forwarding such a select through its next permutation would separately
        shuffle every producer. Register-invariant predicates still fold away
        dead branches, and selects that only remap one producer remain views.
        Graph order ensures a previously selected register is visible while
        planning later selects, without recursion through the whole program.
        """
        for node in self.module.graph.nodes:
            if not (
                node.target is torch.ops.aten.where.self
                and node not in self.constants
                and node.args[0] in self.constants
            ):
                continue
            shape = _shape(node)
            extent = math.prod(shape)
            registers = (extent + self.lanes - 1) // self.lanes
            condition_shape = _shape(cast("Node", node.args[0]))
            if len(shape) == 2 and shape[0] == self.lanes:
                padded = (1,) * (2 - len(condition_shape)) + condition_shape
                if padded[0] == 1:
                    continue
            for register in range(registers):
                producers = set()
                for lane in range(self.lanes):
                    index = lane * registers + register
                    if index >= extent:
                        continue
                    value = self._resolve(self._selected_argument(node, index))
                    producers.add(value.node if isinstance(value, _Element) else value)
                    if len(producers) > 1:
                        self.resident_selects.add((node, register))
                        break

    def _resolve(self, value: _Value) -> _Value:
        path = []
        while isinstance(value, _Element):
            cached = self.canonical.get(value)
            if cached is not None:
                value = cached
                break
            path.append(value)
            result = self._resolve_one(value)
            if result == value:
                break
            value = result
        for original in path:
            self.canonical[original] = value
        return value

    def _resolve_one(self, value: _Element) -> _Value:
        node, index = value.node, value.index
        target = node.target
        shape = _shape(node)
        result: _Value = value
        if node in self.constants:
            result = self._constant(node, index)
        elif node in self.inputs:
            pass
        elif target in _FLAT_VIEWS:
            result = _Element(cast("Node", node.args[0]), index)
        elif target is torch.ops.aten.expand.default:
            result = self._argument(node.args[0], index, shape)
        elif target in (
            torch.ops.aten.gather.default,
            torch.ops.aten.index_select.default,
        ):
            source, dim, indices = node.args[:3]
            assert isinstance(source, Node) and isinstance(indices, Node)
            assert isinstance(dim, int)
            if indices not in self.constants:
                return value
            coords = _coordinates(index, shape)
            position = index if target is torch.ops.aten.gather.default else coords[dim]
            coords[dim] = int(self._constant(indices, position).value)
            result = _Element(source, _flat(coords, _shape(source)))
        elif target in (torch.ops.aten.slice.Tensor, torch.ops.aten.select.int):
            source = cast("Node", node.args[0])
            source_shape = _shape(source)
            dim = cast("int", node.args[1]) % len(source_shape)
            coords = _coordinates(index, shape)
            if target is torch.ops.aten.select.int:
                coords.insert(dim, cast("int", node.args[2]) % source_shape[dim])
            else:
                start = node.args[2] if len(node.args) > 2 else None
                end = node.args[3] if len(node.args) > 3 else None
                step = node.args[4] if len(node.args) > 4 else 1
                begin, _, stride = slice(start, end, step).indices(source_shape[dim])
                coords[dim] = begin + coords[dim] * stride
            result = _Element(source, _flat(coords, source_shape))
        elif target in (torch.ops.aten.transpose.int, torch.ops.aten.permute.default):
            source = cast("Node", node.args[0])
            coords = _coordinates(index, shape)
            if target is torch.ops.aten.transpose.int:
                first, second = cast("tuple[int, int]", node.args[1:3])
                coords[first], coords[second] = coords[second], coords[first]
            else:
                permutation = [
                    dim % len(shape) for dim in cast("Sequence[int]", node.args[1])
                ]
                coords = [coords[permutation.index(dim)] for dim in range(len(shape))]
            result = _Element(source, _flat(coords, _shape(source)))
        elif target in (torch.ops.aten.cat.default, torch.ops.aten.stack.default):
            tensors = cast("Sequence[Node]", node.args[0])
            dim = cast("int", node.args[1] if len(node.args) > 1 else 0) % len(shape)
            coords = _coordinates(index, shape)
            if target is torch.ops.aten.stack.default:
                source = tensors[coords.pop(dim)]
            else:
                source = tensors[0]
                for source in tensors:
                    width = _shape(source)[dim]
                    if coords[dim] < width:
                        break
                    coords[dim] -= width
            result = _Element(source, _flat(coords, _shape(source)))
        elif target is torch.ops.aten.where.self and node.args[0] in self.constants:
            registers = (math.prod(shape) + self.lanes - 1) // self.lanes
            if (node, index % registers) not in self.resident_selects:
                result = self._selected_argument(node, index)
        return result

    def _predicate(self, lanes: set[int]) -> str:
        """Represent common lane partitions compactly, arbitrary ones exactly."""
        for bit in range(self.lanes.bit_length() - 1):
            mask = 1 << bit
            for polarity in (False, True):
                if lanes == {
                    lane for lane in range(self.lanes) if bool(lane & mask) == polarity
                }:
                    return f"(({self.lane} & {mask}) {'!=' if polarity else '=='} 0)"
        if len(lanes) == 1:
            return f"({self.lane} == {next(iter(lanes))})"
        mask = sum(1 << lane for lane in lanes)
        return f"(((cutlass.Uint32({mask}) >> {self.lane}) & cutlass.Uint32(1)) != 0)"

    def _select(self, expressions: dict[int, str]) -> str:
        groups: dict[str, set[int]] = defaultdict(set)
        for lane, expression in expressions.items():
            groups[expression].add(lane)
        choices = list(groups.items())

        def balanced(items: list[tuple[str, set[int]]]) -> str:
            if len(items) == 1:
                return items[0][0]
            middle = len(items) // 2
            lanes = set().union(*(lanes for _, lanes in items[:middle]))
            return f"({balanced(items[:middle])} if {self._predicate(lanes)} else {balanced(items[middle:])})"

        # CuTe recursively stages both sides of an expression. A linear chain
        # of lane selects can make that work exponential in the warp width.
        return balanced(choices)

    def _lane_index(self, owners: dict[int, int]) -> str:
        """Lower a static lane map to bit operations, including partial maps.

        Permutations commonly transform lane bits with XOR and shifts. Solve
        each output bit as an affine function over GF(2), treating inactive
        destinations as don't-cares. An arbitrary remaining bit uses a 32-bit
        lookup mask, so even an irregular permutation has bounded expression
        depth and needs no dynamically staged conditional expressions.
        """
        constant, shifted, masks = _lane_bit_terms(
            owners, self.lanes, self.lanes.bit_length() - 1
        )
        lookups = []
        for truth, output_bit in masks:
            lookup = f"((cutlass.Uint32({truth}) >> {self.lane}) & cutlass.Uint32(1))"
            lookups.append(f"({lookup} << {output_bit})" if output_bit else lookup)
        terms = []
        for shift, mask in shifted:
            value = self.lane if mask == self.lanes - 1 else f"({self.lane} & {mask})"
            if shift:
                value = f"({value} {'<<' if shift > 0 else '>>'} {abs(shift)})"
            terms.append(value)
        terms.extend(lookups)
        if constant:
            terms.append(str(constant))
        return f"({' ^ '.join(terms)})" if terms else "0"

    def _integer_literals(self, values: _Lanes) -> str | None:
        """Spell compile-time integer lane tables as bit expressions, not branches."""
        literals = {
            lane: value for lane, value in enumerate(values) if value is not None
        }
        if not literals or not all(
            isinstance(value, _Literal) for value in literals.values()
        ):
            return None
        literals = cast("dict[int, _Literal]", literals)
        dtypes = {value.dtype for value in literals.values()}
        if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.int32, torch.int64):
            return None
        dtype = next(iter(dtypes))
        bits = 32 if dtype == torch.int32 else 64
        unsigned = "cutlass.Uint32" if bits == 32 else "cutlass.Uint64"
        mask = (1 << bits) - 1
        integers = {lane: int(value.value) & mask for lane, value in literals.items()}
        if len(set(integers.values())) == 1:
            return f"{self.dtype_str(dtype)}({constant_repr(next(iter(literals.values())).value)})"
        constant, shifted, masks = _lane_bit_terms(integers, self.lanes, bits)
        lookups = []
        for truth, output_bit in masks:
            lookup = f"{unsigned}((cutlass.Uint32({truth}) >> {self.lane}) & cutlass.Uint32(1))"
            lookups.append(f"({lookup} << {output_bit})" if output_bit else lookup)
        terms = []
        for shift, input_mask in shifted:
            value = f"({unsigned}({self.lane}) & {unsigned}({input_mask}))"
            if shift:
                value = f"({value} {'<<' if shift > 0 else '>>'} {abs(shift)})"
            terms.append(value)
        terms.extend(lookups)
        if constant:
            terms.append(f"{unsigned}({constant})")
        expression = " ^ ".join(terms) if terms else f"{unsigned}(0)"
        return f"{self.dtype_str(dtype)}({expression})"

    def _assign(self, expression: str) -> str:
        name = self.new_var("register_value")
        self.emit(f"{name} = {expression}")
        return name

    def _read_registers(
        self,
        routes: dict[int, tuple[int, int]],
        read_register: Callable[[int], str],
    ) -> str:
        if all(source == lane for lane, (source, _) in routes.items()):
            return self._select(
                {
                    lane: read_register(register)
                    for lane, (_, register) in routes.items()
                }
            )

        # A shuffle reads one register per source lane. Partition only when
        # separate destinations ask the same source lane for different registers.
        groups: list[dict[int, tuple[int, int]]] = []
        owners: list[dict[int, int]] = []
        for lane, (owner, register) in routes.items():
            for group_index, owner_registers in enumerate(owners):
                if owner not in owner_registers or owner_registers[owner] == register:
                    group = groups[group_index]
                    break
            else:
                group, owner_registers = {}, {}
                groups.append(group)
                owners.append(owner_registers)
            group[lane] = owner, register
            owner_registers[owner] = register
        results = {}
        for group, owner_registers in zip(groups, owners, strict=True):
            source = self._select(
                {
                    lane: read_register(register)
                    for lane, register in owner_registers.items()
                }
            )
            offsets = {lane ^ owner for lane, (owner, _) in group.items()}
            if len(offsets) == 1:
                offset = next(iter(offsets))
                expression = (
                    source
                    if offset == 0
                    else f"cute.arch.shuffle_sync_bfly({source}, offset={offset})"
                )
            else:
                owner = self._lane_index(
                    {lane: owner for lane, (owner, _) in group.items()}
                )
                expression = f"cute.arch.shuffle_sync({source}, offset={owner}, mask_and_clamp={((32 - self.lanes) << 8) | 31})"
            name = self._assign(expression)
            results.update((lane, name) for lane in group)
        return self._select(results)

    def _canonical_register(self, node: Node, register: int) -> str:
        key = (node, register)
        if key in self.register_values:
            return self.register_values[key]
        extent = math.prod(_shape(node))
        registers = (extent + self.lanes - 1) // self.lanes
        value = self._emit_lanes(
            tuple(
                _Element(node, lane * registers + register)
                if lane * registers + register < extent
                else None
                for lane in range(self.lanes)
            )
        )
        self.register_values[key] = value
        return value

    def _read_producers(self, values: _Lanes) -> str | None:
        """Select resident values from distinct producers before a shuffle.

        Static views can make each source lane supply a different producer's
        register. Grouping those demands by producer first would shuffle each
        producer separately and select afterward. The existing route partition
        still keeps separate shuffles when one source lane must supply different
        values to multiple destinations. Mixed dtypes retain ordinary scalar
        selection so routing never changes their promotion semantics.
        """
        active = [value for value in values if value is not None]
        elements = [value for value in active if isinstance(value, _Element)]
        if (
            len(elements) != len(active)
            or len(
                {
                    cast("torch.Tensor", value.node.meta["val"]).dtype
                    for value in elements
                }
            )
            != 1
        ):
            return None
        slots: dict[tuple[Node, int], int] = {}
        routes = {}
        for lane, value in enumerate(values):
            if value is None:
                continue
            assert isinstance(value, _Element)
            if value.node in self.inputs:
                owner, register = self.inputs[value.node].layout.owner(value.index)
            else:
                extent = math.prod(_shape(value.node))
                owned = (extent + self.lanes - 1) // self.lanes
                owner, register = divmod(value.index, owned)
            slot = slots.setdefault((value.node, register), len(slots))
            routes[lane] = owner, slot
        ordered = list(slots)

        def read(slot: int) -> str:
            node, register = ordered[slot]
            if node in self.inputs:
                return f"{self.inputs[node].name}[{register}]"
            return self._canonical_register(node, register)

        return self._read_registers(routes, read)

    def _emit_dynamic_gather(self, node: Node, values: _Lanes) -> str:
        source, dim, indices = node.args[:3]
        assert isinstance(source, Node) and isinstance(indices, Node)
        source_shape = _shape(source)
        dim = cast("int", dim) % len(source_shape)
        stride = math.prod(source_shape[dim + 1 :])
        shape = _shape(node)
        index = self._emit_lanes(
            tuple(
                _Element(indices, value.index) if isinstance(value, _Element) else None
                for value in values
            )
        )
        bases = {}
        for lane, value in enumerate(values):
            if isinstance(value, _Element):
                coords = _coordinates(value.index, shape)
                coords[dim] = 0
                bases[lane] = str(_flat(coords, source_shape))
        flat = self._assign(f"({index}) * {stride} + ({self._select(bases)})")
        registers = (math.prod(source_shape) + self.lanes - 1) // self.lanes
        owner = self._assign(f"cutlass.Int32({flat} // {registers})")
        register = self._assign(f"cutlass.Int32({flat} % {registers})")
        result = self.new_var("register_gather")
        dtype = self.dtype_str(cast("torch.Tensor", node.meta["val"]).dtype)
        self.emit(f"{result} = {dtype}(0)")
        for slot in range(registers):
            resident = self._canonical_register(source, slot)
            peer = self._assign(
                f"cute.arch.shuffle_sync({resident}, offset={owner}, mask_and_clamp={((32 - self.lanes) << 8) | 31})"
            )
            self.emit(f"{result} = {peer} if {register} == {slot} else {result}")
        return result

    def _reduction_inputs(self, node: Node) -> tuple[Node, int, int]:
        source = cast("Node", node.args[0])
        shape = _shape(source)
        dimensions = node.args[1] if len(node.args) > 1 else list(range(len(shape)))
        if isinstance(dimensions, int):
            dimensions = [dimensions]
        assert isinstance(dimensions, (list, tuple))
        dims = sorted(cast("int", dim) % len(shape) for dim in dimensions)
        if not dims:
            dims = list(range(len(shape)))
        first = dims[0]
        if dims != list(range(first, len(shape))):
            raise exc.BackendUnsupported(
                "cute", "register reduction needs contiguous trailing axes"
            )
        return source, math.prod(shape[:first]), math.prod(shape[first:])

    def _reduce_values(self, node: Node, values: list[str], *, groups: int = 1) -> str:
        kind = _REDUCTIONS[node.target]
        output_dtype = cast("torch.Tensor", node.meta["val"]).dtype
        accumulator_dtype = (
            torch.float32
            if output_dtype in (torch.float16, torch.bfloat16)
            else output_dtype
        )
        dtype = self.dtype_str(accumulator_dtype if kind == "sum" else torch.int32)
        identity = "1" if kind == "all" else "0"
        operation = {"sum": "+", "any": "|", "all": "&"}[kind]
        accumulator = self._assign(f"{dtype}({identity})")
        for value in values:
            if kind != "sum":
                value = f"({value} != 0)"
            elif accumulator_dtype != output_dtype:
                # An explicit reduction dtype converts inputs before summing;
                # reduced-precision sums still accumulate those values in FP32.
                value = f"{self.dtype_str(output_dtype)}({value})"
            accumulator = self._assign(
                f"{dtype}({accumulator} {operation} {dtype}({value}))"
            )
        for stage in range(groups.bit_length() - 1):
            peer = self._assign(
                f"cute.arch.shuffle_sync_bfly({accumulator}, offset={1 << stage})"
            )
            accumulator = self._assign(f"{dtype}({accumulator} {operation} {peer})")
        if kind != "sum":
            result = self._assign(f"({accumulator} != 0)")
            if output_dtype != torch.bool:
                result = self._assign(f"{self.dtype_str(output_dtype)}({result})")
            return result
        if accumulator_dtype != output_dtype:
            return self._assign(f"{self.dtype_str(output_dtype)}({accumulator})")
        return accumulator

    def _emit_reduction(self, node: Node, values: _Lanes) -> str:
        source, outputs, extent = self._reduction_inputs(node)
        if outputs >= self.lanes and outputs % self.lanes == 0:
            registers = outputs // self.lanes

            def reduced_register(register: int) -> str:
                key = (node, register)
                if key not in self.reductions:
                    inputs = [
                        self._emit_lanes(
                            tuple(
                                _Element(
                                    source,
                                    (lane * registers + register) * extent + item,
                                )
                                for lane in range(self.lanes)
                            )
                        )
                        for item in range(extent)
                    ]
                    self.reductions[key] = self._reduce_values(node, inputs)
                return self.reductions[key]

            return self._read_registers(
                {
                    lane: divmod(value.index, registers)
                    for lane, value in enumerate(values)
                    if isinstance(value, _Element)
                },
                reduced_register,
            )
        if outputs > self.lanes or self.lanes % outputs != 0:
            raise exc.BackendUnsupported(
                "cute", "register reduction does not align with subgroup ownership"
            )
        groups = self.lanes // outputs
        if outputs != 1 and extent % groups != 0:
            raise exc.BackendUnsupported(
                "cute", "register reduction splits a physical lane"
            )
        key = (node, -1)
        if key not in self.reductions:
            total = outputs * extent
            registers = (total + self.lanes - 1) // self.lanes
            inputs = []
            for register in range(registers):
                active = {
                    lane
                    for lane in range(self.lanes)
                    if lane * registers + register < total
                }
                value = self._emit_lanes(
                    tuple(
                        _Element(source, lane * registers + register)
                        if lane in active
                        else None
                        for lane in range(self.lanes)
                    )
                )
                if len(active) != self.lanes:
                    identity_value = 1 if _REDUCTIONS[node.target] == "all" else 0
                    source_dtype = cast("torch.Tensor", source.meta["val"]).dtype
                    identity = f"{self.dtype_str(source_dtype)}({identity_value})"
                    value = f"({value} if {self._predicate(active)} else {identity})"
                inputs.append(value)
            self.reductions[key] = self._reduce_values(node, inputs, groups=groups)
        result = self.reductions[key]
        if outputs == 1:
            return result
        return self._read_registers(
            {
                lane: (value.index * groups, 0)
                for lane, value in enumerate(values)
                if isinstance(value, _Element)
            },
            lambda register: result,
        )

    def _materialize(self, node: Node) -> RowFragment:
        if node in self.inputs:
            return self.inputs[node]
        extent = math.prod(_shape(node))
        registers = (extent + self.lanes - 1) // self.lanes
        dtype = cast("torch.Tensor", node.meta["val"]).dtype
        name = self.new_var("register_argument")
        self.emit(
            f"{name} = cute.make_rmem_tensor({registers}, {self.dtype_str(dtype)})"
        )
        for register in range(registers):
            expression = self._canonical_register(node, register)
            self.emit(f"{name}[{register}] = {expression}")
        return RowFragment(
            name, dtype, extent, RowFragmentLayout(self.lanes, registers, self.lane)
        )

    def _emit_cond(self, node: Node) -> tuple[RowFragment, ...]:
        if node in self.conditionals:
            return self.conditionals[node]
        predicate, true_graph, false_graph, operands = node.args
        assert isinstance(predicate, Node)
        if self.lanes != 32 or math.prod(_shape(predicate)) != 1:
            raise exc.BackendUnsupported(
                "cute",
                "register conditional requires one scalar predicate over a complete warp",
            )
        condition = self._emit_lanes(
            tuple(_Element(predicate, 0) for lane in range(self.lanes))
        )
        assert isinstance(operands, (list, tuple))
        arguments = [
            self._materialize(operand)
            for operand in operands
            if isinstance(operand, Node)
        ]
        outputs = []
        result_types = node.meta["val"]
        assert isinstance(result_types, (tuple, list))
        for tensor in result_types:
            assert isinstance(tensor, torch.Tensor)
            extent = tensor.numel()
            registers = (extent + self.lanes - 1) // self.lanes
            name = self.new_var("register_conditional")
            self.emit(
                f"{name} = cute.make_rmem_tensor({registers}, {self.dtype_str(tensor.dtype)})"
            )
            outputs.append(
                RowFragment(
                    name,
                    tensor.dtype,
                    extent,
                    RowFragmentLayout(self.lanes, registers, self.lane),
                )
            )
        branches = []
        for graph_node in (true_graph, false_graph):
            assert isinstance(graph_node, Node)
            module = self.constants[graph_node]
            assert isinstance(module, GraphModule)
            module = _specialize_constant_captures(module, operands)
            statements = []
            child = RegisterProgramEmitter(
                module,
                dict(
                    zip(
                        module.graph.find_nodes(op="placeholder"),
                        arguments,
                        strict=True,
                    )
                ),
                lanes=self.lanes,
                lane_expr=self.lane,
                new_var=self.new_var,
                emit=statements.append,
                scalar=self.scalar,
                dtype_str=self.dtype_str,
            )
            returned = child.run()
            for output, value in zip(outputs, returned, strict=True):
                assert output.extent == value.extent and output.dtype == value.dtype
                for register in range(output.num_registers):
                    statements.append(
                        f"{output.name}[{register}] = {value.name}[{register}]"
                    )
            branches.append("\n".join(statements))
        self.emit(
            f"if {condition}:\n{textwrap.indent(branches[0], '    ')}\nelse:\n{textwrap.indent(branches[1], '    ')}"
        )
        result = tuple(outputs)
        self.conditionals[node] = result
        return result

    def _emit_lanes(self, values: _Lanes) -> str:
        values = tuple(
            self._resolve(value) if value is not None else None for value in values
        )
        if values in self.cache:
            return self.cache[values]
        active = [value for value in values if value is not None]
        assert active
        groups: dict[Node | None, set[int]] = defaultdict(set)
        for lane, value in enumerate(values):
            if value is not None:
                groups[value.node if isinstance(value, _Element) else None].add(lane)
        if len(groups) > 1:
            routed = self._read_producers(values)
            if routed is not None:
                result = routed
            else:
                choices = {}
                for lanes in groups.values():
                    partial = tuple(
                        value if lane in lanes else None
                        for lane, value in enumerate(values)
                    )
                    expression = self._emit_lanes(partial)
                    choices.update((lane, expression) for lane in lanes)
                result = self._select(choices)
        elif next(iter(groups)) is None:
            result = self._integer_literals(values)
            if result is None:
                result = self._select(
                    {
                        lane: f"{self.dtype_str(value.dtype)}({constant_repr(value.value)})"
                        for lane, value in enumerate(values)
                        if isinstance(value, _Literal)
                    }
                )
        else:
            node = next(iter(groups))
            assert isinstance(node, Node)
            if (
                node.target is operator.getitem
                and isinstance(node.args[0], Node)
                and node.args[0].target is torch.ops.higher_order.cond
            ):
                self.inputs[node] = self._emit_cond(node.args[0])[
                    cast("int", node.args[1])
                ]
            if node in self.inputs:
                fragment = self.inputs[node]
                result = self._read_registers(
                    {
                        lane: fragment.layout.owner(value.index)
                        for lane, value in enumerate(values)
                        if isinstance(value, _Element)
                    },
                    lambda register: f"{fragment.name}[{register}]",
                )
            else:
                shape = _shape(node)
                extent = math.prod(shape)
                owned = (extent + self.lanes - 1) // self.lanes
                routes = {
                    lane: divmod(value.index, owned)
                    for lane, value in enumerate(values)
                    if isinstance(value, _Element)
                }
                registers = {register for _, register in routes.values()}
                canonical = (
                    len(registers) == 1
                    and all(lane == owner for lane, (owner, _) in routes.items())
                    and len(routes)
                    == sum(
                        lane * owned + next(iter(registers)) < extent
                        for lane in range(self.lanes)
                    )
                )
                if node.target in _REDUCTIONS:
                    result = self._emit_reduction(node, values)
                    self.cache[values] = result
                    return result
                if not canonical:
                    # Preserve computed values in their ordinary ownership.
                    # Rematerializing a producer at a permuted coordinate can
                    # duplicate every arithmetic operation and input shuffle.
                    result = self._read_registers(
                        routes,
                        lambda register: self._canonical_register(node, register),
                    )
                    self.cache[values] = result
                    return result
                if node.target is torch.ops.aten.gather.default:
                    result = self._emit_dynamic_gather(node, values)
                    self.cache[values] = result
                    return result
                if (
                    node.target is torch.ops.aten.where.self
                    and node.args[0] in self.constants
                ):
                    result = self._assign(
                        self._emit_lanes(
                            tuple(
                                self._selected_argument(node, value.index)
                                if isinstance(value, _Element)
                                else None
                                for value in values
                            )
                        )
                    )
                    self.cache[values] = result
                    return result
                if not (
                    isinstance(node.target, torch._ops.OpOverload)
                    and (
                        torch.Tag.pointwise in node.target.tags
                        or node.target in _EXTRA_POINTWISE
                    )
                ):
                    raise exc.BackendUnsupported(
                        "cute", f"register scalar operation {node.target}"
                    )
                inputs = row_fragment_tensor_inputs(node)
                operands = []
                for argument in inputs:
                    refs = tuple(
                        self._argument(argument, value.index, _shape(node))
                        if isinstance(value, _Element)
                        else None
                        for value in values
                    )
                    operands.append(self._emit_lanes(refs))
                result = self._assign(self.scalar(node, operands, self.emit))
        self.cache[values] = result
        return result

    def _prepare_registers(self, outputs: Sequence[Node]) -> None:
        """Plan demands iteratively, then emit producers in graph order.

        A long chain of functional updates should require linear compiler
        work and no Python recursion proportional to its length. Static views
        still forward each demand to its surviving scalar operation, so dead
        branches and discarded comparator outputs never enter the schedule.
        """
        pending = [
            _Element(node, index)
            for node in outputs
            for index in range(math.prod(_shape(node)))
        ]
        demanded: dict[Node, set[int]] = defaultdict(set)
        while pending:
            value = self._resolve(pending.pop())
            if isinstance(value, _Literal) or value.node in self.inputs:
                continue
            node = value.node
            extent = math.prod(_shape(node))
            registers = (extent + self.lanes - 1) // self.lanes
            register = value.index % registers
            if register in demanded[node]:
                continue
            demanded[node].add(register)
            if (
                node.target in _REDUCTIONS
                or node.target is torch.ops.aten.gather.default
            ):
                source = cast("Node", node.args[0])
                if len(demanded[node]) == 1:
                    pending.extend(
                        _Element(source, index)
                        for index in range(math.prod(_shape(source)))
                    )
                if node.target is torch.ops.aten.gather.default:
                    indices = cast("Node", node.args[2])
                    pending.extend(
                        _Element(indices, lane * registers + register)
                        for lane in range(self.lanes)
                        if lane * registers + register < extent
                    )
                continue
            if (
                node.target is operator.getitem
                and isinstance(node.args[0], Node)
                and node.args[0].target is torch.ops.higher_order.cond
            ):
                conditional = node.args[0]
                predicate = cast("Node", conditional.args[0])
                pending.append(_Element(predicate, 0))
                for argument in cast("Sequence[Argument]", conditional.args[3]):
                    if not isinstance(argument, Node):
                        continue
                    pending.extend(
                        _Element(argument, index)
                        for index in range(math.prod(_shape(argument)))
                    )
                continue
            if (
                node.target is torch.ops.aten.where.self
                and node.args[0] in self.constants
            ):
                pending.extend(
                    self._selected_argument(node, lane * registers + register)
                    for lane in range(self.lanes)
                    if lane * registers + register < extent
                )
                continue
            for argument in row_fragment_tensor_inputs(node):
                pending.extend(
                    cast(
                        "_Element",
                        self._argument(
                            argument, lane * registers + register, _shape(node)
                        ),
                    )
                    for lane in range(self.lanes)
                    if lane * registers + register < extent
                )
        for node in list(self.module.graph.nodes):
            for register in sorted(demanded.get(node, ())):
                self._canonical_register(node, register)

    def run(self) -> tuple[RowFragment, ...]:
        output = next(reversed(self.module.graph.nodes))
        assert output.op == "output"
        results = output.args[0]
        outputs = (results,) if isinstance(results, Node) else results
        assert isinstance(outputs, (tuple, list))
        assert all(isinstance(node, Node) for node in outputs)
        self._prepare_registers(cast("Sequence[Node]", outputs))
        fragments = []
        for node in outputs:
            assert isinstance(node, Node)
            shape = _shape(node)
            if len(shape) != 2 or shape[0] != self.lanes or shape[1] <= 0:
                raise exc.BackendUnsupported(
                    "cute", "register output must have shape [lanes, registers]"
                )
            registers = shape[1]
            dtype = cast("torch.Tensor", node.meta["val"]).dtype
            name = self.new_var("register_output")
            fragment = RowFragment(
                name,
                dtype,
                math.prod(shape),
                RowFragmentLayout(self.lanes, registers, self.lane),
            )
            self.emit(
                f"{name} = cute.make_rmem_tensor({registers}, {self.dtype_str(dtype)})"
            )
            for register in range(registers):
                expression = self._emit_lanes(
                    tuple(
                        _Element(node, lane * registers + register)
                        for lane in range(self.lanes)
                    )
                )
                self.emit(f"{name}[{register}] = {expression}")
            fragments.append(fragment)
        return tuple(fragments)


def emit_register_program(
    cg: GenerateAST,
    module: GraphModule,
    inputs: Sequence[RowFragment],
    *,
    lanes: int,
    lane_expr: str,
) -> tuple[RowFragment, ...]:
    """Lower a tensor program compactly when supported, otherwise scalarize."""
    placeholders = list(module.graph.find_nodes(op="placeholder"))
    if len(placeholders) != len(inputs):
        raise ValueError("register program input count does not match its graph")
    if (plan := plan_register_tensor(module, inputs, lanes=lanes)) is not None:
        return emit_register_tensor(cg, plan, inputs, lane_expr=lane_expr)
    bound = dict(zip(placeholders, inputs, strict=True))
    backend = CompileEnvironment.current().backend

    def emit(source: str) -> None:
        for statement in ast.parse(source).body:
            cg.add_statement(statement)

    def load(node: Node) -> RowFragment:
        return bound[node]

    pointwise = RowFragmentEmitter(
        cg,
        load,
        layout=RowFragmentLayout(lanes, 1, lane_expr),
        reuse_scalar_lowering=True,
    )

    def scalar(node: Node, arguments: list[str], add: Callable[[str], None]) -> str:
        if node.target in (torch.ops.aten._to_copy.default, torch.ops.aten.view.dtype):
            dtype = cast("torch.Tensor", node.meta["val"]).dtype
            if node.target is torch.ops.aten.view.dtype:
                source = cast("Node", node.args[0]).meta["val"]
                if source.dtype.itemsize != dtype.itemsize:
                    raise exc.BackendUnsupported(
                        "cute", "register bitcast changes element size"
                    )
                return f"({arguments[0]}).bitcast({backend.dtype_str(dtype)})"
            return f"{backend.dtype_str(dtype)}({arguments[0]})"
        statements: list[ast.AST] = []
        with cg.set_statements(statements):
            result = pointwise.emit_pointwise_scalar(
                node, [expr_from_string(argument) for argument in arguments]
            )
        for statement in statements:
            add(ast.unparse(statement))
        return ast.unparse(result)

    return RegisterProgramEmitter(
        module,
        bound,
        lanes=lanes,
        lane_expr=lane_expr,
        new_var=cg.device_function.new_var,
        emit=emit,
        scalar=scalar,
        dtype_str=backend.dtype_str,
    ).run()
