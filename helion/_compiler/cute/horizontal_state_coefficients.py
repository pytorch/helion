"""Complete readonly feature images inside one horizontal consumer cohort.

Each image is the original rank-one load of a contiguous full feature axis.
Only proven scalar task prefixes are admitted; no index-array, row-dependent,
explicit-mask or cache-qualified read is moved. Publication preserves the
original storage bits and every original feature has a consumer.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from .horizontal_state_expression import ScalarAccess
from .horizontal_state_expression import StateExpression
from .horizontal_state_update import InapplicableStateUpdate
from .horizontal_state_update import _ancestors
from .horizontal_state_update import _freeze
from .horizontal_state_update import _proven_int
from .memory_ops import cute_reduction_vector_layout_aligned

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .horizontal_state_expression import ScheduledScalarProgram


@dataclasses.dataclass(frozen=True)
class FeaturePublication:
    node: Node
    access: ScalarAccess
    binding: str
    shared: str
    words: int
    vector_width: int
    selection: object = dataclasses.field(default=None, repr=False)

    def facts(self) -> object:
        return _freeze(
            (
                self.node,
                self.access.fields(),
                self.binding,
                self.shared,
                self.words,
                self.vector_width,
            )
        )

    def check(self, cg: GenerateAST, program: ScheduledScalarProgram) -> None:
        program.check(cg)
        if self.selection != self.facts():
            raise InapplicableStateUpdate("readonly publication changed")
        if not _candidate(program, self.node):
            raise InapplicableStateUpdate("readonly feature image no longer applies")
        matches = [
            (node, coords, name)
            for node, coords, name in program.original.read_bindings
            if node is self.node
        ]
        if len(matches) != 1 or matches[0][2] != self.binding:
            raise InapplicableStateUpdate("readonly feature load binding changed")


def _candidate(program: ScheduledScalarProgram, node: Node) -> bool:
    region = program.original.domain.region
    if node is region.state_load or node not in region.loads:
        return False
    source = cast("Node", node.args[0]).meta["val"]
    selectors = cast("Sequence[object]", node.args[1])
    if (
        node.args[2] is not None
        or node.args[3] is not None
        or node.meta["val"].ndim != 1
        or len(selectors) != source.ndim
        or selectors[-1] is not region.feature_selector
        or _proven_int(source.shape[-1]) != region.feature_selector.args[0]
        or _proven_int(source.stride(-1)) != 1
        or source.dtype not in (torch.float16, torch.bfloat16)
    ):
        return False
    for selector in selectors[:-1]:
        if isinstance(selector, Node):
            if isinstance(selector.meta.get("val"), torch.Tensor):
                return False
            if {region.row_selector, region.feature_selector} & _ancestors(selector):
                return False
        elif type(selector) is not int:
            return False
    return True


def plan_feature_publications(
    cg: GenerateAST,
    program: ScheduledScalarProgram,
    outer_coordinates: tuple[str, ...],
) -> tuple[FeaturePublication, ...]:
    from ..compile_environment import CompileEnvironment

    program.check(cg)
    domain = program.original.domain
    result = []
    for node in domain.region.loads:
        if not _candidate(program, node):
            continue
        matches = [
            (coords, name)
            for read, coords, name in program.original.read_bindings
            if read is node
        ]
        if len(matches) != 1:
            continue
        expression = StateExpression(cg, domain, outer_coordinates)
        access = expression.access(node, ("0",))
        # Pure scalar task arithmetic is emitted inline. A generated prefix
        # load would require its own movement/effect proof and is left alone.
        if expression.lines:
            continue
        source = cast("Node", node.args[0]).meta["val"]
        vector_width = (
            8
            if cute_reduction_vector_layout_aligned(
                CompileEnvironment.current(), source, source.ndim - 1, 8
            )
            else 1
        )
        publication = FeaturePublication(
            node,
            access,
            matches[0][1],
            cg.device_function.new_var("horizontal_coefficient"),
            domain.state_shape[-1],
            vector_width,
        )
        publication = dataclasses.replace(publication, selection=publication.facts())
        publication.check(cg, program)
        result.append(publication)
    return tuple(result)
