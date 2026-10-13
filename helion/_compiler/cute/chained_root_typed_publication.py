"""An original typed descendant published from a completed native fragment.

The native FP32 completion remains the authority. A separate typed image is
additive until all graph users and emitted reads permit retiring that FP32
storage. No target is recognized by name, formula, or matrix size.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from .chained_root_vector_cache import NativeFragmentScope
from .chained_root_vector_cache import PublishedRootVector
from .chained_store_expression import _node_facts

if TYPE_CHECKING:
    from torch.fx import Node

    from .chained_fragment_epilogue import FragmentExpression
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput
    from .chained_root_native import RootNativeStage
    from .chained_root_native_emission import RootNativeCompletion
    from .chained_root_native_emission import RootNativeEmission


def _shape(node: Node) -> tuple[int, ...] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or any(
        type(size) is not int or size <= 0 for size in value.shape
    ):
        return None
    return tuple(value.shape)


def _dominates_uses(source: Node, target: Node, plan: ChainedMatmulPlan) -> bool:
    from .chained_matmul import _ancestors
    from .chained_matmul import _supported

    if source is target or source.graph is not target.graph or plan.region is None:
        return False
    ancestors = _ancestors(target)
    if source not in ancestors:
        return False
    barriers = {*plan.dots, *plan.region.scans, *plan.region.reductions}
    pending, seen = list(source.users), set()
    while pending:
        node = pending.pop()
        if node is target or node in seen:
            continue
        seen.add(node)
        if (
            node not in ancestors
            or node not in plan.region.nodes
            or node in barriers
            or node.op != "call_function"
            or node.is_impure()
            or not node.users
            or not _supported(node)
        ):
            return False
        pending.extend(node.users)
    return bool(source.users)


def typed_target(plan: ChainedMatmulPlan, source: Node) -> Node | None:
    """First same-shape half-precision value crossed by every source use."""
    from .chained_matmul import _supported

    shape = _shape(source)
    if plan.region is None or shape is None or len(shape) != 2:
        return None
    for node in plan.region.nodes:
        value = node.meta.get("val")
        if (
            node is not source
            and isinstance(value, torch.Tensor)
            and value.dtype in (torch.float16, torch.bfloat16)
            and _shape(node) == shape
            and node.op == "call_function"
            and not node.is_impure()
            and _supported(node)
            and _dominates_uses(source, node, plan)
        ):
            return node
    return None


def typed_publication_bytes(plan: ChainedMatmulPlan) -> int:
    """Conservative additive capacity; emitted retirement grants no credit."""
    if plan.root_native is None:
        return 0
    total = 0
    for stage in plan.root_native.stages:
        target = typed_target(plan, stage.node)
        if target is not None:
            rows, columns = stage.geometry.logical[:2]
            total += (rows * columns * 2 + 127) // 128 * 128
    return total


@dataclass(frozen=True)
class TypedPublicationDraft:
    scope: NativeFragmentScope
    target: Node
    dtype: torch.dtype
    shape: tuple[int, int]
    owner: str
    expression: FragmentExpression
    dependencies: tuple[tuple[Node, str], ...]
    graph_facts: tuple[object, ...]
    allocation: tuple[str, ...]
    publication: tuple[str, ...]
    selection: object = None

    def facts(self) -> object:
        return (
            id(self.scope),
            self.target,
            self.dtype,
            self.shape,
            self.owner,
            id(self.expression),
            tuple((node, id(owner)) for node, owner in self.dependencies),
            self.graph_facts,
            self.allocation,
            self.publication,
        )

    def check(self) -> None:
        from .chained_matmul import _UnsupportedChain
        from .chained_root_vector_cache import check_read_scope

        body, source = self.scope.body, self.scope.stage.node
        self.scope.check(body.plan)
        self.expression.check()
        check_read_scope(self.expression.expression)
        if (
            self.selection != self.facts()
            or typed_target(body.plan, source) is not self.target
            or _shape(self.target) != self.shape
            or self.target.meta["val"].dtype is not self.dtype
            or _node_facts(tuple(source.graph.nodes)) != self.graph_facts
            or self.expression.source is not source
            or self.expression.target is not self.target
            or self.expression.coordinates != self.scope.coordinates
            or self.expression.fragment != self.scope.fragment
            or any(
                self.expression.expression.boundaries.get(node) is not owner
                for node, owner in self.dependencies
            )
        ):
            raise _UnsupportedChain("native typed publication draft changed")
        for _, owner in self.dependencies:
            if isinstance(owner, PublishedRootVector):
                owner.publication.check()


def prepare_typed_publication(
    body: RootNativeEmission,
    stage: RootNativeStage,
    boundaries: dict[Node, str],
    scans: list[_ScanInput],
    lines: list[str],
) -> TypedPublicationDraft | None:
    from ..compile_environment import CompileEnvironment
    from .chained_fragment_epilogue import bind_fragment_expression
    from .chained_matmul import _UnsupportedChain
    from .chained_tcgen05 import _load_result

    if (
        body.cg.device_function.config.config.get("cute_chained_completed_image")
        is not True
    ):
        return None
    target = typed_target(body.plan, stage.node)
    if target is None or target in boundaries:
        return None
    prefix = f"chain_{stage.index}"
    shape = stage.geometry.logical[:2]
    coordinates = (f"{prefix}_typed_row", f"{prefix}_typed_column")
    index = f"{prefix}_typed_index"
    scope = NativeFragmentScope(
        body,
        stage,
        lines,
        tuple(lines),
        coordinates,
        index,
        f"{prefix}_values[{index}]",
        tuple(_load_result(prefix, shape, execution=body.execution)),
    )
    scope.check(body.plan)
    dtype = target.meta["val"].dtype
    dtype_str = CompileEnvironment.current().backend.dtype_str(dtype)
    vector_reads = [
        (owner.publication, tuple(owner.publication.reads))
        for owner in boundaries.values()
        if isinstance(owner, PublishedRootVector)
    ]
    try:
        expression = bind_fragment_expression(
            body.cg,
            body.plan,
            boundaries,
            scans,
            stage.node,
            target,
            coordinates,
            coordinates,
            scope.fragment,
            dtype_str,
            root_read_scope=scope,
        )
    except _UnsupportedChain:
        # A speculative expression can read the vector before declining later.
        # Only an emitted typed publication may retain those read receipts.
        for publication, reads in vector_reads:
            publication.reads[:] = reads
        return None
    owner = f"{prefix}_typed_c"
    rows, columns = shape
    allocation = (
        f"{owner}_ptr = cute.arch.alloc_smem({dtype_str}, {rows * columns}, alignment=128)",
        f"{owner} = cute.make_tensor({owner}_ptr, cute.make_layout({shape!r}, stride=({columns}, 1)))",
    )
    publication = tuple(
        scope.wrap(
            [
                *expression.expression.lines,
                f"{owner}[{coordinates[0]}, {coordinates[1]}] = {expression.masked_value()}",
            ]
        )
    )
    result = TypedPublicationDraft(
        scope,
        target,
        dtype,
        shape,
        owner,
        expression,
        tuple(
            (node, value)
            for node, value in boundaries.items()
            if any(key[0] is node for key in expression.expression.memo)
        ),
        _node_facts(tuple(stage.node.graph.nodes)),
        allocation,
        publication,
    )
    result = replace(result, selection=result.facts())
    result.check()
    return result


@dataclass(frozen=True)
class RootTypedPublication:
    draft: TypedPublicationDraft
    completion: RootNativeCompletion
    phase: tuple[str, ...]
    selection: object = None

    @property
    def target(self) -> Node:
        return self.draft.target

    @property
    def source(self) -> Node:
        return self.completion.stage.node

    @property
    def owner(self) -> str:
        return self.draft.owner

    def facts(self) -> object:
        return (
            id(self.draft),
            id(self.completion),
            self.phase,
            self.draft.allocation,
            self.draft.publication,
            self.draft.owner,
        )

    def check(self) -> None:
        from .chained_matmul import _UnsupportedChain

        self.draft.check()
        completion = self.completion
        completion.check(
            completion.body.plan,
            completion.stage.node,
            completion.owner,
            self.phase,
            completion.publication,
        )
        suffix = (
            *self.draft.allocation,
            *self.draft.publication,
            *completion.publication,
            completion.body.execution.sync,
        )
        if (
            self.selection != self.facts()
            or completion.body is not self.draft.scope.body
            or completion.stage is not self.draft.scope.stage
            or self.phase != completion.phase
            or not self.phase
            or self.phase[-1] != completion.body.execution.sync
            or self.phase[-len(suffix) :] != suffix
            or completion.body.typed_publications.get(self.target) is not self
        ):
            raise _UnsupportedChain("native typed publication completion changed")


def complete_typed_publication(
    draft: TypedPublicationDraft,
    completion: RootNativeCompletion,
    boundaries: dict[Node, str],
) -> RootTypedPublication:
    from .chained_matmul import _UnsupportedChain

    if draft.target in boundaries:
        raise _UnsupportedChain("typed publication would replace an existing owner")
    receipt = RootTypedPublication(draft, completion, completion.phase)
    receipt = replace(receipt, selection=receipt.facts())
    completion.body.typed_publications[draft.target] = receipt
    receipt.check()
    boundaries[draft.target] = draft.owner
    return receipt


@dataclass(frozen=True)
class TypedPublicationRetirement:
    """Exact old/new body for the caller's existing metadata-aware rewrite."""

    receipt: RootTypedPublication
    before: tuple[str, ...]
    after: tuple[str, ...]

    def check(self, lines: list[str]) -> None:
        from .chained_matmul import _UnsupportedChain

        self.receipt.check()
        if (
            tuple(lines) != self.before
            or _retired_body(self.receipt, lines) != self.after
        ):
            raise _UnsupportedChain("typed publication read cut changed")


def _retired_body(
    receipt: RootTypedPublication, lines: list[str]
) -> tuple[str, ...] | None:
    """Keep every issue/wait/join; erase only exact unused FP32 storage writes."""
    completion = receipt.completion
    remaining = list(lines)
    for block in (completion.allocation, completion.publication):
        matches = [
            i
            for i in range(len(remaining) - len(block) + 1)
            if tuple(remaining[i : i + len(block)]) == block
        ]
        if len(matches) != 1:
            return None
        i = matches[0]
        remaining[i : i + len(block)] = []
    names = {completion.owner, f"{completion.owner}_ptr"}
    tree = ast.parse("\n".join(remaining))
    if any(isinstance(node, ast.Name) and node.id in names for node in ast.walk(tree)):
        return None
    return tuple(remaining)


def retire_typed_publication(
    receipt: RootTypedPublication, lines: list[str]
) -> TypedPublicationRetirement | None:
    receipt.check()
    after = _retired_body(receipt, lines)
    if after is None:
        return None
    return TypedPublicationRetirement(receipt, tuple(lines), after)
