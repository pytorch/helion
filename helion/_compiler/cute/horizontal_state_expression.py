"""Original scalar recipes for a proved horizontal state-update region.

This layer does not authorize shared reads or reorder memory effects. It lowers
the original pointwise operators and exposes each original scalar access to the
transport planner. A substituted old-state value needs a separate publication
and lifetime receipt before this program can be committed.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_ops
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_store_expression import _node_facts
from .horizontal_state_update import InapplicableStateUpdate
from .horizontal_state_update import StateTaskDomain
from .horizontal_state_update import _proven_int

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST


@dataclasses.dataclass(frozen=True)
class ScalarAccess:
    node: Node = dataclasses.field(repr=False)
    coordinates: tuple[str, ...]
    tensor: str
    indices: tuple[str, ...]
    predicate: str | None
    dtype: torch.dtype
    pointer: str

    def fields(self) -> tuple[object, ...]:
        return (
            self.node,
            self.coordinates,
            self.tensor,
            self.indices,
            self.predicate,
            self.dtype,
            self.pointer,
        )


class StateExpression(chain._Expression):
    """Use existing typed pointwise lowering with explicit original accesses."""

    def __init__(
        self,
        cg: GenerateAST,
        domain: StateTaskDomain,
        outer_coordinates: tuple[str, ...],
    ) -> None:
        super().__init__(cg, None, {})
        domain.check()
        if len(outer_coordinates) != len(domain.grid_extents) - 1:
            raise InapplicableStateUpdate("wrong original outer task coordinates")
        self.domain = domain
        self.origins = dict(
            zip(domain.grid_block_ids, (*outer_coordinates, "0"), strict=True)
        )
        self.original_accesses: list[ScalarAccess] = []
        self.tensor_names: dict[Node, str] = {}
        self.feature_invariance = None

    def tensor_name(self, node: Node) -> str:
        name = self.cg.device_function.tensor_arg(
            node.meta["val"], prefer_name=cast("str", node.args[0])
        ).name
        previous = self.tensor_names.setdefault(node, name)
        if previous != name:
            raise InapplicableStateUpdate("original tensor argument was rebound")
        return name

    def scalar_range(self, arg: object) -> tuple[int, int] | None:
        if isinstance(arg, Node) and arg.target is tile_ops.tile_id:
            axis = self.block_id(cast("Node", arg.args[0]))
            if axis in self.domain.grid_block_ids[:-1]:
                return (
                    0,
                    self.domain.grid_extents[self.domain.grid_block_ids.index(axis)]
                    - 1,
                )
        return super().scalar_range(arg)

    def access(self, node: Node, coordinates: tuple[str, ...]) -> ScalarAccess:
        """Canonical full axes plus original narrowed tensor-index predicates.

        Logical row/feature coordinates must already be in their original
        domains. Their original tile masks are therefore true. Dynamic tensor
        indices retain the original upper-bound-only comparison in index_dtype;
        no nonnegative or widened-address predicate is introduced here.
        """
        from ...language.memory_ops import _cute_scalar_pointer_expr

        env = CompileEnvironment.current()
        source = cast("Node", node.args[0])
        tensor = source.meta["val"]
        selectors = self.selectors(node)
        if len(selectors) != tensor.ndim or any(x is None for x in selectors):
            raise InapplicableStateUpdate("noncanonical scalar memory rank")
        shape = tuple(_proven_int(x) for x in tensor.shape)
        if any(x is None or x <= 0 for x in shape):
            raise InapplicableStateUpdate("unproved scalar memory extent")
        indices = self.indices(node, coordinates)
        predicates: list[str] = []
        extra = node.args[2] if node.target is memory_ops.load else node.args[3]
        if extra is not None:
            if isinstance(extra, Node):
                extra_shape = chain._shape(extra)
                offset = len(coordinates) - len(extra_shape)
                if offset < 0:
                    raise InapplicableStateUpdate("memory mask rank")
                mask_coords = tuple(
                    "0" if size == 1 else coordinates[i + offset]
                    for i, size in enumerate(extra_shape)
                )
                predicates.append(self.value(extra, mask_coords))
            elif type(extra) is bool:
                predicates.append(repr(extra))
            else:
                raise InapplicableStateUpdate("nonboolean memory mask")
        for dim, selector in enumerate(selectors):
            if isinstance(selector, Node) and isinstance(
                selector.meta.get("val"), torch.Tensor
            ):
                if selector.meta["val"].dtype not in (torch.int32, torch.int64):
                    raise InapplicableStateUpdate("noninteger memory selector")
                predicates.append(f"{env.index_type()}({indices[dim]}) < {shape[dim]}")
            elif isinstance(selector, Node):
                if selector.target is _tracing_ops._get_symnode:
                    if selector is not self.domain.region.row_selector:
                        raise InapplicableStateUpdate("noncanonical row selector")
                    if cast("int", shape[dim]) < self.domain.grid_extents[-1]:
                        raise InapplicableStateUpdate(
                            "row access has a narrower original bound"
                        )
                else:
                    bounds = self.scalar_range(selector)
                    if bounds is None or not 0 <= bounds[0] <= bounds[1] < cast(
                        "int", shape[dim]
                    ):
                        raise InapplicableStateUpdate("unproved scalar memory selector")
            elif type(selector) is not int or not 0 <= selector < cast(
                "int", shape[dim]
            ):
                raise InapplicableStateUpdate("unproved literal memory selector")
        name = self.tensor_name(source)
        predicate = " and ".join(f"({part})" for part in predicates) or None
        return ScalarAccess(
            node,
            coordinates,
            name,
            tuple(indices),
            predicate,
            tensor.dtype,
            _cute_scalar_pointer_expr(name, indices),
        )

    def value(self, node: Node, coordinates: tuple[str, ...]) -> str:
        if self.feature_invariance is not None:
            coordinates = self.feature_invariance.coordinates(node, coordinates)
        if node.target is not memory_ops.load or node in self.fragments:
            return super().value(node, coordinates)
        key = (node, coordinates)
        if key in self.memo:
            return self.memo[key]
        if node not in self.domain.region.loads or node.args[3] is not None:
            raise InapplicableStateUpdate("foreign or cache-qualified raw load")
        from ...language.memory_ops import _cute_scalar_load_expr

        access = self.access(node, coordinates)
        value = _cute_scalar_load_expr(
            access.tensor, list(access.indices), access.dtype
        )
        dtype = CompileEnvironment.current().backend.dtype_str(access.dtype)
        if access.dtype is torch.bool:
            value = f"cutlass.Boolean({value})"
        if access.predicate is not None:
            value = f"({value} if {access.predicate} else {dtype}(0))"
        result = self.bind_read(node, value)
        self.original_accesses.append(access)
        self.memo[key] = result
        return result


@dataclasses.dataclass(frozen=True)
class StateScalarProgram:
    """A typed original recipe, before any replacement acquires read authority."""

    domain: StateTaskDomain = dataclasses.field(repr=False)
    coordinates: tuple[str, str]
    old_state: ScalarAccess
    old_value: str
    lines: tuple[str, ...]
    update: str
    reduction_input: str
    storage_value: str
    reads: tuple[ScalarAccess, ...]
    read_bindings: tuple[tuple[Node, tuple[str, ...], str], ...]
    value_bindings: tuple[tuple[Node, tuple[str, ...], str], ...]
    facts: tuple[object, ...] = dataclasses.field(repr=False)
    config: object = dataclasses.field(repr=False)
    selection: tuple[object, ...] = dataclasses.field(default=(), repr=False)

    def fields(self) -> tuple[object, ...]:
        from .horizontal_state_update import _freeze

        values = tuple(
            getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("selection", "old_state", "reads")
        )
        return (
            _freeze(values),
            _freeze(self.old_state.fields()),
            _freeze(tuple(access.fields() for access in self.reads)),
        )

    def check(self, cg: GenerateAST) -> None:
        from .horizontal_state_update import _freeze

        self.domain.check()
        if self.selection != self.fields():
            raise InapplicableStateUpdate("scalar program receipt changed")
        if self.facts != _node_facts(tuple(self.domain.region.root.graph.nodes)):
            raise InapplicableStateUpdate("original scalar lowering changed")
        if self.config != _freeze(cg.device_function.config.config):
            raise InapplicableStateUpdate("original scalar configuration changed")


def lower_state_scalar_program(
    cg: GenerateAST,
    domain: StateTaskDomain,
    *,
    outer_coordinates: tuple[str, ...],
    coordinates: tuple[str, str],
) -> StateScalarProgram:
    """Lower one original logical point; no shared substitution is authorized."""
    from .horizontal_state_update import _freeze

    expression = StateExpression(cg, domain, outer_coordinates)
    region = domain.region
    old_value = expression.value(region.state_load, coordinates)
    old_state = next(
        a for a in expression.original_accesses if a.node is region.state_load
    )
    update = expression.value(region.update, coordinates)
    reduction_input = expression.value(
        cast("Node", region.reduction.args[0]), coordinates
    )
    storage_value = expression.value(
        cast("Node", region.state_store.args[2]), coordinates
    )
    program = StateScalarProgram(
        domain,
        coordinates,
        old_state,
        old_value,
        tuple(expression.lines),
        update,
        reduction_input,
        storage_value,
        tuple(expression.original_accesses),
        tuple(
            (node, coordinates, value)
            for (node, coordinates), value in expression.memo.items()
            if node.target is memory_ops.load
        ),
        tuple(
            (node, coordinates, value)
            for (node, coordinates), value in expression.memo.items()
        ),
        _node_facts(tuple(region.root.graph.nodes)),
        _freeze(cg.device_function.config.config),
    )
    program = dataclasses.replace(program, selection=program.fields())
    program.check(cg)
    return program


@dataclasses.dataclass(frozen=True)
class ScheduledScalarProgram:
    """An original scalar program split only at proven feature invariants."""

    original: StateScalarProgram
    setup: tuple[str, ...]
    invariant: tuple[str, ...]
    element: tuple[str, ...]
    output: tuple[str, ...]
    output_access: ScalarAccess
    state_access: ScalarAccess
    output_value: str
    selection: object = dataclasses.field(default=None, repr=False)

    def fields(self) -> object:
        from .horizontal_state_update import _freeze

        return (
            tuple(
                _freeze(getattr(self, field.name))
                for field in dataclasses.fields(self)
                if field.name not in ("selection", "output_access", "state_access")
            ),
            _freeze(self.output_access.fields()),
            _freeze(self.state_access.fields()),
        )

    def check(self, cg: GenerateAST) -> None:
        self.original.check(cg)
        if self.selection != self.fields():
            raise InapplicableStateUpdate("scheduled scalar program changed")


def lower_scheduled_scalar_program(
    cg: GenerateAST,
    domain: StateTaskDomain,
    *,
    outer_coordinates: tuple[str, ...],
    coordinates: tuple[str, str],
    reduced_value: str,
) -> ScheduledScalarProgram:
    from .horizontal_state_invariance import feature_invariance
    from .horizontal_state_update import _ancestors
    from .horizontal_state_update import _freeze

    expression = StateExpression(cg, domain, outer_coordinates)
    region = domain.region
    invariance = feature_invariance(region)
    invariance.check()
    expression.feature_invariance = invariance
    old_access = expression.access(region.state_load, coordinates)
    setup_cut = len(expression.lines)
    live = _ancestors(region.state_store) | _ancestors(region.output_store)
    for node, axes in invariance.axes:
        if node not in invariance.invariant or node not in live:
            continue
        logical = tuple(coordinates[0] if axis == "row" else "0" for axis in axes)
        expression.value(node, logical)
    invariant_cut = len(expression.lines)
    old_value = expression.value(region.state_load, coordinates)
    update = expression.value(region.update, coordinates)
    reduction_input = expression.value(
        cast("Node", region.reduction.args[0]), coordinates
    )
    storage_value = expression.value(
        cast("Node", region.state_store.args[2]), coordinates
    )
    element_cut = len(expression.lines)
    state_access = expression.access(region.state_store, coordinates)
    expression.fragments[region.reduction] = ((coordinates[0],), reduced_value)
    output_value = expression.value(
        cast("Node", region.output_store.args[2]), (coordinates[0],)
    )
    output_access = expression.access(region.output_store, (coordinates[0],))
    invariance.check()
    original = StateScalarProgram(
        domain,
        coordinates,
        old_access,
        old_value,
        tuple(expression.lines[:element_cut]),
        update,
        reduction_input,
        storage_value,
        tuple(expression.original_accesses),
        tuple(
            (node, coordinates, value)
            for (node, coordinates), value in expression.memo.items()
            if node.target is memory_ops.load
        ),
        tuple(
            (node, coordinates, value)
            for (node, coordinates), value in expression.memo.items()
        ),
        _node_facts(tuple(region.root.graph.nodes)),
        _freeze(cg.device_function.config.config),
    )
    original = dataclasses.replace(original, selection=original.fields())
    original.check(cg)
    scheduled = ScheduledScalarProgram(
        original,
        tuple(expression.lines[:setup_cut]),
        tuple(expression.lines[setup_cut:invariant_cut]),
        tuple(expression.lines[invariant_cut:element_cut]),
        tuple(expression.lines[element_cut:]),
        output_access,
        state_access,
        output_value,
    )
    scheduled = dataclasses.replace(scheduled, selection=scheduled.fields())
    scheduled.check(cg)
    return scheduled
