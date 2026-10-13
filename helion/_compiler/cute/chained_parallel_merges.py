"""Batch independent completed-image contractions after original admission.

This is a local schedule rewrite, not a second expression lowerer. Original
operand, native-copy, publication and exclusive-use checks run first. Only
contiguous, fully recorded warp stages can move. Their exact operand fills and
publications survive, with disjoint A/B slices and a joined ready frontier.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import TYPE_CHECKING

from .chained_execution import ChainedExecution
from .chained_warp_mma import complete_warp_columns
from .chained_warp_mma import emit_warp_mma
from .chained_warp_pair import WarpPairTeam
from .chained_warp_stage import _execution
from .chained_warp_stage import rebind_warp_execution

if TYPE_CHECKING:
    from torch.fx import Node

    from .chained_completed_image_emission import CompletedImageEmission
    from .chained_root_warp_stage import WarpOperandEmission
    from .chained_warp_pair import WarpPairAttempt
    from .chained_warp_pair import WarpPairStage


JOIN = "cute.arch.sync_threads()"


def _local_preparation(lines: tuple[str, ...], index: int) -> bool:
    """No collective, hidden store, allocation or opaque call may cross stages."""
    tree = ast.parse("\n".join(lines))
    calls = {
        "cutlass.range",
        "cutlass.range_constexpr",
        "cutlass.Int32",
        "cutlass.Int64",
        "cutlass.Float16",
        "cutlass.BFloat16",
        "cutlass.Float32",
        "cutlass.Float64",
        "cutlass.Boolean",
        "cute.make_tensor",
        "cute.make_layout",
        "cute.recast_ptr",
        *(f"operator.{name}" for name in ("lt", "le", "eq", "ne", "ge", "gt")),
        *(
            f"cute.math.{name}"
            for name in (
                "exp",
                "exp2",
                "log",
                "log2",
                "sqrt",
                "rsqrt",
                "sin",
                "cos",
                "tanh",
                "abs",
                "minimum",
                "maximum",
                "floor",
                "ceil",
                "trunc",
            )
        ),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(
            node, (ast.Assign, ast.For, ast.If)
        ):
            return False
        if isinstance(node, ast.Call) and ast.unparse(node.func) not in calls:
            return False
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            if not (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in (f"chain_{index}_a", f"chain_{index}_b")
            ):
                return False
        if isinstance(node, (ast.NamedExpr, ast.Lambda)):
            return False
    return True


def _flat(lines: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(line for block in lines for line in block.splitlines())


def _indented(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("    " + line for line in lines)


def ready_waves(
    stages: tuple[int, ...],
    dependencies: dict[int, frozenset[int]],
    capacity: int,
    *,
    weights: dict[int, int] | None = None,
    workspace_sizes: dict[int, tuple[int, int]] | None = None,
    workspace_capacities: tuple[int, int] | None = None,
) -> tuple[tuple[int, ...], ...] | None:
    """Stable independent frontiers packed by whole-warps and live workspaces."""
    if capacity < 1 or len(set(stages)) != len(stages):
        return None
    weights = dict.fromkeys(stages, 1) if weights is None else weights
    if set(weights) != set(stages) or any(
        type(weight) is not int or not 0 < weight <= capacity
        for weight in weights.values()
    ):
        return None
    if (workspace_sizes is None) != (workspace_capacities is None):
        return None
    if workspace_sizes is not None and set(workspace_sizes) != set(stages):
        return None
    remaining = set(stages)
    waves = []
    while remaining:
        ready: list[int] = []
        used = 0
        for stage in stages:
            if stage not in remaining or dependencies[stage] & remaining:
                continue
            if used + weights[stage] > capacity:
                continue
            if workspace_sizes is not None:
                assert workspace_capacities is not None
                if any(
                    workspace_offsets(
                        tuple(workspace_sizes[item][role] for item in (*ready, stage)),
                        limit,
                    )
                    is None
                    for role, limit in enumerate(workspace_capacities)
                ):
                    continue
            ready.append(stage)
            used += weights[stage]
        if not ready:
            return None
        waves.append(tuple(ready))
        remaining.difference_update(ready)
    return tuple(waves)


def workspace_offsets(
    extents: tuple[int, ...], capacity: int, alignment: int = 64
) -> tuple[int, ...] | None:
    """Half-element spans, preserving the original 128-byte base alignment."""
    offsets, end = [], 0
    for extent in extents:
        if extent < 0:
            return None
        begin = (end + alignment - 1) // alignment * alignment
        offsets.append(begin)
        end = begin + extent
    return tuple(offsets) if end <= capacity else None


@dataclass(frozen=True)
class _Stage:
    index: int
    node: Node
    source: tuple[str, ...]
    preparation: tuple[str, ...]
    declaration: tuple[str, ...]
    publication: tuple[str, ...]
    operands: WarpOperandEmission
    dtype: str
    shape: tuple[int, int, int]
    threads: int
    dependencies: frozenset[int]
    # Only actual completed-image cells, excluding duplicate native padding.
    reads: frozenset[tuple[int, int]]
    writes: frozenset[tuple[int, int]]
    workspace_sizes: tuple[int, int]
    compute: tuple[str, ...] = ()
    register_declaration: tuple[str, ...] = ()
    register_a: bool = False


@dataclass(frozen=True)
class PairWaveReceipt:
    """A new team-bound body; the original consumed pair is never rebound."""

    pair: WarpPairAttempt
    team: WarpPairTeam
    original: tuple[str, ...]
    source: tuple[str, ...]
    _selection: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_selection", self.facts())
        self.check()

    def facts(self) -> object:
        return (
            self.pair.selection(),
            self.team,
            _execution(self.team.execution),
            self.original,
            self.source,
        )

    def check(self) -> None:
        from .chained_matmul import _UnsupportedChain

        self.pair.check()
        if (
            self.pair.phase != "consumed"
            or self.team.layout != self.pair.layout
            or self.team.cta_threads != self.pair.plan.threads
            or self._selection != self.facts()
        ):
            raise _UnsupportedChain("completed pair wave owner changed")


@dataclass(frozen=True)
class ParallelMergeLowering:
    original: tuple[str, ...]
    result: tuple[str, ...]
    nodes: tuple[Node, ...]
    source: tuple[str, ...]
    waves: tuple[tuple[int, ...], ...]
    retired_pairs: tuple[WarpPairAttempt, ...] = ()
    teams: tuple[PairWaveReceipt, ...] = ()


def _stage(
    emission: CompletedImageEmission,
    index: int,
    source: tuple[str, ...],
    *,
    pair: WarpPairStage | None = None,
) -> _Stage | None:
    from ..compile_environment import CompileEnvironment
    from .chained_matmul import _ancestors

    plan = emission.plan
    node = plan.dots[index]
    operands = emission.warp_operands.get(node)
    if (
        operands is None
        or not operands.matches()
        or operands.asynchronous
        or operands.copies
        or operands.native_destinations
        or node.args[2] is not None
        or len(operands.layouts) != 2
    ):
        return None
    shape = plan.shapes[index]
    rows, columns, reduction = shape
    if rows < 16 or rows % 16 or reduction <= 0 or reduction % 8:
        return None
    if not complete_warp_columns(columns, 32):
        return None
    dtype = CompileEnvironment.current().backend.dtype_str(plan.operand_dtype(index))
    if dtype not in ("cutlass.Float16", "cutlass.BFloat16"):
        return None
    threads = 32 * min(plan.threads // 32, 2 ** (columns.bit_length() - 4))
    prefix = f"chain_{index}"
    register_declaration: tuple[str, ...] = ()
    if pair is None:
        compute = tuple(
            emit_warp_mma(
                prefix,
                dtype,
                shape,
                threads,
                dict(operands.axes),
                [f"{prefix}_acc.fill(0.0)"],
            )
        )
        publication = (
            f"{prefix}_sc = {prefix}_thr.partition_C({prefix}_c)",
            f"cute.autovec_copy({prefix}_acc, {prefix}_sc)",
        )
    else:
        pair.owner.check()
        if (
            pair.owner.phase != "consumed"
            or pair.owner.team is not None
            or pair.stage != index
            or pair._issue is None
            or not pair._publication
            or pair._inner_axes != operands.axes
        ):
            return None
        threads = pair.owner.layout.threads
        compute = (*pair._issue, *(pair.owner._fill if pair.producer else ()))
        publication = pair._publication
        if pair.producer:
            register_declaration = pair.owner._declaration
    declaration = (
        f"{prefix}_c_ptr = cute.arch.alloc_smem(cutlass.Float32, {rows * (columns + 4)}, alignment=128)",
        f"{prefix}_c = cute.make_tensor({prefix}_c_ptr, cute.make_layout({shape[:2]!r}, stride=({columns + 4}, 1)))",
    )
    assert emission.lowering is not None
    declaration = emission.lowering.segment(declaration)
    publication = emission.lowering.segment(publication)
    preparation = _flat(operands.lines)
    if not _local_preparation(preparation, index):
        return None
    body = _flat((*compute, *publication))
    expected = (
        *preparation,
        *_flat(register_declaration),
        JOIN,
        *declaration,
        *(
            (f"if chain_thread < {threads}:", *_indented(body))
            if threads < plan.threads
            else body
        ),
        JOIN,
    )
    # This excludes every unrecorded effect, alternate output, seed or barrier.
    if source != expected:
        return None
    dependencies: set[int] = set()
    reads: set[tuple[int, int]] = set()
    workspace_sizes = []
    for operand_index, layout in enumerate(operands.layouts):
        if (
            layout.node is not node.args[operand_index]
            or layout.role != ("a", "b")[operand_index]
            or layout.dtype != plan.operand_dtype(index)
            or layout.shape != ((rows, reduction), (columns, reduction))[operand_index]
            or layout.inner not in (0, 1)
            or min(layout.stride) != 1
            or max(layout.stride) % 8 != 0
        ):
            return None
        if pair is not None and not pair.producer and operand_index == 0:
            # The canonical A view remains solely for partition shape metadata.
            # The checked issue consumes the exact same-lane private A fragment.
            dependencies.add(pair.owner.source)
            workspace_sizes.append(0)
            continue
        views = [
            (image_index, image, view)
            for image_index, image in enumerate(emission.images)
            for view in image.operands
            if (view.stage, view.operand_index) == (index, operand_index)
        ]
        if views:
            if len(views) != 1 or (index, operand_index) not in emission.used:
                return None
            image_index, image, view = views[0]
            if view.operand is not layout.node:
                return None
            reads.update((image_index, address) for address in view.addresses)
            predecessors = {image.cells[address].source for address in view.addresses}
            dependencies.update(
                plan.dots.index(source_node)
                for source_node in predecessors
                if source_node in plan.dots
            )
            workspace_sizes.append(0)
        else:
            dependencies.update(
                plan.dots.index(ancestor)
                for ancestor in _ancestors(layout.node)
                if ancestor in plan.dots
            )
            workspace_sizes.append(
                1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(layout.shape, layout.stride, strict=True)
                )
            )
    if any(prior >= index for prior in dependencies):
        return None
    owner = emission.direct.owner(node)
    writes = (
        frozenset(
            (owner.image, destination)
            for destination, _ in owner.publication.locations(
                emission.images[owner.image].shape[1]
            )
        )
        if owner is not None
        else frozenset()
    )
    if owner is None and node in emission.sources:
        # Additive publication is not inside this native-stage receipt.
        return None
    return _Stage(
        index,
        node,
        source,
        preparation,
        declaration,
        publication,
        operands,
        dtype,
        shape,
        threads,
        frozenset(dependencies),
        frozenset(reads),
        writes,
        (workspace_sizes[0], workspace_sizes[1]),
        _flat(compute),
        _flat(register_declaration),
        pair is not None and not pair.producer,
    )


def _rebase(stage: _Stage, offsets: tuple[int, int]) -> tuple[str, ...] | None:
    """Rebase only the two original workspace pointer declarations."""
    result = list(stage.preparation)
    for role, size, offset in zip(
        ("a", "b"), stage.workspace_sizes, offsets, strict=True
    ):
        workspace = f"chain_{role}_workspace"
        # Parse the complete suite: individual loop-header lines are not suites.
        tree = ast.parse("\n".join(result))
        uses = [
            n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == workspace
        ]
        if not size and not (role == "a" and stage.register_a):
            if uses:
                return None
            continue
        target = f"chain_{stage.index}_{role}_ptr"
        forms = (
            f"{target} = {workspace}",
            f"{target} = cute.recast_ptr({workspace}, dtype={stage.dtype})",
        )
        matches = [i for i, line in enumerate(result) if line in forms]
        if len(uses) != 1 or len(matches) != 1:
            return None
        i = matches[0]
        result[i] = result[i].replace(workspace, f"({workspace} + {offset})")
    return tuple(result)


def _render_wave(
    stages: tuple[_Stage, ...], capacities: tuple[int, int]
) -> tuple[str, ...] | None:
    if len(stages) == 1:
        return stages[0].source
    allocations = tuple(
        workspace_offsets(
            tuple(stage.workspace_sizes[role] for stage in stages), capacity
        )
        for role, capacity in enumerate(capacities)
    )
    if any(offsets is None for offsets in allocations):
        return None
    a_offsets, b_offsets = allocations
    assert a_offsets is not None and b_offsets is not None
    output = []
    for stage, a_offset, b_offset in zip(stages, a_offsets, b_offsets, strict=True):
        preparation = _rebase(stage, (a_offset, b_offset))
        if preparation is None:
            return None
        output.extend((*preparation, *stage.declaration))
    output.append(JOIN)
    for team, stage in enumerate(stages):
        prefix = f"chain_{stage.index}"
        thread = f"{prefix}_team_thread"
        compute = emit_warp_mma(
            prefix,
            stage.dtype,
            stage.shape,
            32,
            dict(stage.operands.axes),
            [f"{prefix}_acc.fill(0.0)"],
            execution=ChainedExecution(32, thread=thread),
        )
        output.extend(
            (
                f"if {team * 32} <= chain_thread < {(team + 1) * 32}:",
                f"    {thread} = chain_thread - {team * 32}",
                *_indented(_flat((*compute, *stage.publication))),
            )
        )
    output.append(JOIN)
    return tuple(output)


def parallelize_completed_merges(
    emission: CompletedImageEmission,
    *,
    warp_pairs: tuple[WarpPairAttempt, ...] = (),
) -> ParallelMergeLowering | None:
    """One bounded contiguous warp region; all uncertain cases remain serial."""
    from .chained_completed_image_emission import native_copy_layout
    from .chained_direct_image import _native_c_coordinates

    plan = emission.plan
    lowering = emission.lowering
    if (
        lowering is None
        or plan.loop is not None
        or plan.root_native is None
        or plan.threads < 64
        or plan.threads % 32
    ):
        return None
    if warp_pairs:
        return _parallelize_pairs(emission, warp_pairs)
    candidates = [
        _stage(emission, index, lowering.segment(source))
        if node not in emission.native_completions
        else None
        for index, (node, source) in enumerate(emission.native_stages)
    ]
    # Require one contiguous region, with no row/scan/publication event between
    # recorded stages. Such events retain their original schedule unchanged.
    indexes = tuple(i for i, stage in enumerate(candidates) if stage is not None)
    if not indexes or indexes != tuple(range(indexes[0], indexes[-1] + 1)):
        return None
    stages = tuple(candidates[i] for i in indexes)
    assert all(stage is not None for stage in stages)
    stages = tuple(stage for stage in stages if stage is not None)
    original = tuple(line for stage in stages for line in stage.source)
    positions = [
        i
        for i, line in enumerate(lowering.result)
        if line == original[0] and lowering.result[i : i + len(original)] == original
    ]
    if len(positions) != 1:
        return None
    names = {}
    trees = {}
    for stage in stages:
        tree = ast.parse("\n".join(stage.source))
        trees[stage.index] = tree
        assigned = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        }
        if assigned & names.keys() or f"chain_{stage.index}_team_thread" in ast.unparse(
            tree
        ):
            return None
        names.update((name, stage.index) for name in assigned)
    dependencies = {}
    for stage in stages:
        cross_names = {
            names[node.id]
            for node in ast.walk(trees[stage.index])
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in names
            and names[node.id] != stage.index
        }
        if any(other >= stage.index for other in cross_names):
            return None
        dependencies[stage.index] = stage.dependencies | cross_names
    waves = ready_waves(indexes, dependencies, plan.threads // 32)
    if waves is None or all(len(wave) == 1 for wave in waves):
        return None
    by_index = {stage.index: stage for stage in stages}
    # Direct-image publications must be unique, including across all waves.
    writes: set[tuple[int, int]] = set()
    for stage in stages:
        if writes & stage.writes:
            return None
        writes.update(stage.writes)
    capacities = plan.root_native.original_workspace_elements
    result = []
    checked_c = set()
    for wave in waves:
        members = tuple(by_index[index] for index in wave)
        if len(wave) > 1:
            for stage in members:
                if any(stage.reads & other.writes for other in members):
                    return None
                key = (stage.shape, plan.operand_dtype(stage.index))
                if key not in checked_c:
                    if not _native_c_coordinates(
                        stage.shape, plan.operand_dtype(stage.index), 32
                    ):
                        return None
                    checked_c.add(key)
                for image in emission.images:
                    for view in image.operands:
                        if view.stage == stage.index and not native_copy_layout(
                            image,
                            view,
                            stage.operands.layouts[view.operand_index],
                            stage.shape,
                            32,
                        ):
                            return None
        rendered = _render_wave(members, capacities)
        if rendered is None:
            return None
        result.extend(rendered)
    start = positions[0]
    replacement = tuple(result)
    return ParallelMergeLowering(
        lowering.result,
        (
            *lowering.result[:start],
            *replacement,
            *lowering.result[start + len(original) :],
        ),
        tuple(stage.node for stage in stages),
        replacement,
        waves,
    )


@dataclass(frozen=True)
class _PairGroup:
    pair: WarpPairAttempt
    members: tuple[_Stage, _Stage]
    # Two operands per stage. Simultaneously prepared operands have disjoint spans.
    member_offsets: tuple[tuple[int, int], tuple[int, int]]
    workspace_sizes: tuple[int, int]

    @property
    def index(self) -> int:
        return self.pair.source

    @property
    def source(self) -> tuple[str, ...]:
        return tuple(line for member in self.members for line in member.source)

    @property
    def reads(self) -> frozenset[tuple[int, int]]:
        return self.members[0].reads | self.members[1].reads

    @property
    def writes(self) -> frozenset[tuple[int, int]]:
        return self.members[0].writes | self.members[1].writes


def _pair_group(
    emission: CompletedImageEmission, pair: WarpPairAttempt
) -> _PairGroup | None:
    pair.check()
    plan, lowering = emission.plan, emission.lowering
    assert lowering is not None and plan.root_native is not None
    if (
        pair.codegen is not emission.cg
        or pair.plan is not plan
        or pair.phase != "consumed"
        or pair.team is not None
        or pair.source + 1 >= len(emission.native_stages)
    ):
        return None
    members = []
    for token in (pair.producer, pair.consumer):
        node, original = emission.native_stages[token.stage]
        if node is not plan.dots[token.stage] or node in emission.native_completions:
            return None
        stage = _stage(emission, token.stage, lowering.segment(original), pair=token)
        if stage is None:
            return None
        members.append(stage)
    first, second = members
    if first.writes:
        # An externally published first C cannot have an exclusive register use.
        return None
    allocations = tuple(
        workspace_offsets(
            tuple(stage.workspace_sizes[role] for stage in members), capacity
        )
        for role, capacity in enumerate(plan.root_native.original_workspace_elements)
    )
    if any(offsets is None for offsets in allocations):
        return None
    a_offsets, b_offsets = allocations
    assert a_offsets is not None and b_offsets is not None
    sizes = tuple(
        offsets[-1] + second.workspace_sizes[role]
        for role, offsets in enumerate((a_offsets, b_offsets))
    )
    return _PairGroup(
        pair,
        (first, second),
        ((a_offsets[0], b_offsets[0]), (a_offsets[1], b_offsets[1])),
        (sizes[0], sizes[1]),
    )


def _pair_dependencies(
    groups: tuple[_PairGroup, ...],
) -> dict[int, frozenset[int]] | None:
    """Contract only the private pair edge; retain every other emitted dependency."""
    owner = {member.index: group.index for group in groups for member in group.members}
    names: dict[str, int] = {}
    trees = {}
    for group in groups:
        tree = ast.parse("\n".join(group.source))
        trees[group.index] = tree
        assigned = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        }
        if assigned & names.keys():
            return None
        names.update((name, group.index) for name in assigned)
    dependencies = {}
    for group in groups:
        predecessors = {
            owner.get(prior, prior)
            for member in group.members
            for prior in member.dependencies
        }
        predecessors.update(
            names[node.id]
            for node in ast.walk(trees[group.index])
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in names
        )
        predecessors.discard(group.index)
        if any(prior >= group.index for prior in predecessors):
            return None
        dependencies[group.index] = frozenset(predecessors)
    return dependencies


def _available_barriers(source: tuple[str, ...]) -> tuple[int, ...]:
    """Reserve distinct hardware named barriers, excluding the CTA barrier zero."""
    used = {0}
    for node in ast.walk(ast.parse("\n".join(source))):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.endswith(".NamedBarrier") or name == "cute.arch.barrier":
            ids = [item.value for item in node.keywords if item.arg == "barrier_id"]
            if len(ids) != 1 or not isinstance(ids[0], ast.Constant):
                return ()
            if type(ids[0].value) is not int:
                return ()
            used.add(ids[0].value)
    return tuple(index for index in range(1, 16) if index not in used)


def _render_pair_wave(
    groups: tuple[_PairGroup, ...],
    capacities: tuple[int, int],
    cta_threads: int,
    barrier_ids: tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[PairWaveReceipt, ...]] | None:
    allocations = tuple(
        workspace_offsets(tuple(group.workspace_sizes[role] for group in groups), limit)
        for role, limit in enumerate(capacities)
    )
    if any(offsets is None for offsets in allocations) or len(groups) > len(
        barrier_ids
    ):
        return None
    a_offsets, b_offsets = allocations
    assert a_offsets is not None and b_offsets is not None
    output: list[str] = []
    for group, a_base, b_base in zip(groups, a_offsets, b_offsets, strict=True):
        for member, (a_local, b_local) in zip(
            group.members, group.member_offsets, strict=True
        ):
            preparation = _rebase(member, (a_base + a_local, b_base + b_local))
            if preparation is None:
                return None
            output.extend((*preparation, *member.declaration))
    output.append(JOIN)
    first_thread = 0
    receipts = []
    for group, barrier_id in zip(groups, barrier_ids, strict=False):
        pair = group.pair
        prefix = f"chain_pair_{pair.source}_team"
        execution = ChainedExecution(
            pair.layout.threads,
            thread=f"{prefix}_thread",
            warp=f"{prefix}_warp",
            barriers=f"{prefix}_barrier",
            sync=f"{prefix}_barrier.arrive_and_wait()",
        )
        team = WarpPairTeam(
            pair.layout, execution, first_thread, cta_threads, barrier_id
        )
        first, second = group.members
        original = (
            *first.register_declaration,
            *first.compute,
            *first.publication,
            *second.compute,
        )
        # The only removed intermediate join is the proved same-lane C -> A edge.
        # All reads, including shared B reads, finish before in-place publication.
        body = (
            *rebind_warp_execution(original, pair.execution, execution),
            execution.sync,
            *rebind_warp_execution(second.publication, pair.execution, execution),
        )
        receipt = PairWaveReceipt(
            pair,
            team,
            original,
            (*team.bindings, *body),
        )
        receipt.check()
        output.extend(
            (
                team.declaration,
                f"if {team.predicate}:",
                *_indented((*team.bindings, *body)),
            )
        )
        receipts.append(receipt)
        first_thread += pair.layout.threads
    output.append(JOIN)
    return tuple(output), tuple(receipts)


def _parallelize_pairs(
    emission: CompletedImageEmission, pairs: tuple[WarpPairAttempt, ...]
) -> ParallelMergeLowering | None:
    from .chained_completed_image_emission import native_copy_layout
    from .chained_completed_register import pair_result_cut
    from .chained_completed_register import retire_result_cut
    from .chained_direct_image import _native_c_coordinates

    plan, lowering = emission.plan, emission.lowering
    assert lowering is not None and plan.root_native is not None
    selected = tuple(_pair_group(emission, pair) for pair in pairs)
    if any(group is None for group in selected):
        return None
    groups = tuple(group for group in selected if group is not None)
    members = tuple(member.index for group in groups for member in group.members)
    if not members or members != tuple(range(members[0], members[-1] + 1)):
        return None
    original = tuple(line for group in groups for line in group.source)
    positions = [
        i
        for i, line in enumerate(lowering.result)
        if line == original[0] and lowering.result[i : i + len(original)] == original
    ]
    if len(positions) != 1:
        return None
    dependencies = _pair_dependencies(groups)
    if dependencies is None:
        return None
    indexes = tuple(group.index for group in groups)
    capacities = plan.root_native.original_workspace_elements
    waves = ready_waves(
        indexes,
        dependencies,
        plan.threads // 32,
        weights={group.index: group.pair.layout.threads // 32 for group in groups},
        workspace_sizes={group.index: group.workspace_sizes for group in groups},
        workspace_capacities=capacities,
    )
    if waves is None:
        return None
    barriers = _available_barriers(lowering.result)
    if max(map(len, waves)) > len(barriers):
        return None
    existing_names = {
        node.id
        for node in ast.walk(ast.parse("\n".join(lowering.result)))
        if isinstance(node, ast.Name)
    }
    proposed_names = {
        f"chain_pair_{group.index}_team_{suffix}"
        for group in groups
        for suffix in ("thread", "warp", "barrier")
    } | {"chain_pair_pipeline"}
    if existing_names & proposed_names:
        return None
    written: set[tuple[int, int]] = set()
    for group in groups:
        if written & group.writes:
            return None
        written.update(group.writes)
    by_index = {group.index: group for group in groups}
    result = ["from cutlass import pipeline as chain_pair_pipeline"]
    receipts = []
    checked_c = set()
    for wave in waves:
        ready = tuple(by_index[index] for index in wave)
        for group in ready:
            if any(group.reads & other.writes for other in ready if other is not group):
                return None
            for member in group.members:
                layout = group.pair.layout
                key = (member.shape, plan.operand_dtype(member.index), layout.topology)
                if key not in checked_c:
                    if not _native_c_coordinates(
                        member.shape,
                        plan.operand_dtype(member.index),
                        layout.threads,
                        topology=layout.topology,
                    ):
                        return None
                    checked_c.add(key)
                for image in emission.images:
                    for view in image.operands:
                        if view.stage != member.index or (
                            member.register_a and view.operand_index == 0
                        ):
                            continue
                        if not native_copy_layout(
                            image,
                            view,
                            member.operands.layouts[view.operand_index],
                            member.shape,
                            layout.threads,
                            topology=layout.topology,
                        ):
                            return None
        rendered = _render_pair_wave(ready, capacities, plan.threads, barriers)
        if rendered is None:
            return None
        lines, teams = rendered
        result.extend(lines)
        receipts.extend(teams)
    start = positions[0]
    replacement = tuple(result)
    full = (
        *lowering.result[:start],
        *replacement,
        *lowering.result[start + len(original) :],
    )
    for group in groups:
        cut = pair_result_cut(group.pair, group.members[0].source)
        full = retire_result_cut(full, cut)
    # Bind receipts to the final bodies, after the one complete first-C read cut.
    tree = ast.parse("\n".join(full))
    team_scopes: dict[str, list[ast.If]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and node.body
            and isinstance(node.body[0], ast.Assign)
            and isinstance(node.body[0].targets[0], ast.Name)
        ):
            team_scopes.setdefault(node.body[0].targets[0].id, []).append(node)
    final_receipts = []
    for receipt in receipts:
        scopes = [
            node
            for node in team_scopes.get(receipt.team.execution.thread, ())
            if ast.unparse(node.test) == receipt.team.predicate
        ]
        if len(scopes) != 1 or not scopes[0].body:
            return None
        body = scopes[0].body
        assert body[-1].end_lineno is not None
        source = tuple(
            line[4:] for line in full[body[0].lineno - 1 : body[-1].end_lineno]
        )
        final_receipts.append(replace(receipt, source=source))
    # Retirement removes only recorded declarations/publications inside the region.
    suffix_size = len(lowering.result) - start - len(original)
    end = len(full) - suffix_size
    return ParallelMergeLowering(
        lowering.result,
        full,
        tuple(member.node for group in groups for member in group.members),
        full[start:end],
        waves,
        pairs,
        tuple(final_receipts),
    )
