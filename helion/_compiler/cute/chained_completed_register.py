"""Same-lane C-to-A pairs inside the original completed-image schedule."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .chained_matmul import _UnsupportedChain
from .chained_root_native import STRATEGY
from .chained_root_native import _ordinary_storage
from .chained_warp_pair import _collect_warp_pairs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_completed_image_emission import CompletedImageEmission
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput
    from .chained_warp_pair import WarpPairAttempt


def _legal_team(pair: WarpPairAttempt, threads: int) -> bool:
    """Use the pair's proved C/A topology for both members, within one CTA."""
    pair.check()
    return (
        32 <= pair.layout.threads <= threads
        and threads % 32 == 0
        and pair.layout.threads % 32 == 0
        and pair.layout.topology == (pair.layout.threads // 32, 1, 1)
    )


def plan_completed_pairs(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[_ScanInput],
    schedule: str,
) -> tuple[WarpPairAttempt, ...]:
    config = cg.device_function.config.config
    if (
        config.get("cute_chained_warp_register_a") is not True
        or config.get("cute_chained_completed_image") is not True
        or config.get("cute_chained_root_native") is not True
        or plan.strategy != STRATEGY
        or not plan.completed_image_bytes
        or plan.root_native is None
        or not _ordinary_storage(plan)
        or schedule != "cp_async"
    ):
        raise _UnsupportedChain(
            "completed register pairs require original native roots"
        )
    plan.root_native.check(plan)
    native = {item.index for item in plan.root_native.stages}
    pairs = tuple(
        pair
        for pair in _collect_warp_pairs(cg, plan, boundaries, scans)
        if pair.source not in native
        and pair.source + 1 not in native
        and _legal_team(pair, plan.threads)
    )
    if not pairs:
        raise _UnsupportedChain("completed register pairs lack a proved common owner")
    return pairs


def _flat(lines: Sequence[str]) -> tuple[str, ...]:
    return tuple(line for block in lines for line in block.splitlines())


@dataclass(frozen=True)
class PairResultCut:
    """The first result's actual declarations/publication, never a shape recipe."""

    pair: WarpPairAttempt
    owner: str
    declaration: tuple[str, ...]
    publication: tuple[str, ...]


def pair_result_cut(pair: WarpPairAttempt, source: Sequence[str]) -> PairResultCut:
    """Read the original first-C owner after both pair receipts completed."""
    pair.check()
    if pair.phase != "consumed" or not pair.producer._publication:
        raise _UnsupportedChain("register retirement precedes the pair completion")
    source = _flat(source)
    prefix = f"chain_{pair.source}"
    tree = ast.parse("\n".join(source))
    allocations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == prefix + "_c_ptr"
    ]
    if len(allocations) != 1:
        raise _UnsupportedChain("register retirement lacks unique private C owner")
    value = allocations[0].value
    if not (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) == "cute.arch.alloc_smem"
        and len(value.args) == 2
        and ast.unparse(value.args[0]) == "cutlass.Float32"
        and isinstance(value.args[1], ast.Constant)
        and type(value.args[1].value) is int
    ):
        raise _UnsupportedChain("register retirement C is not an original allocation")
    views = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == prefix + "_c"
    ]
    if (
        len(views) != 1
        or not isinstance(views[0].value, ast.Call)
        or ast.unparse(views[0].value.func) != "cute.make_tensor"
        or not views[0].value.args
        or ast.unparse(views[0].value.args[0]) != prefix + "_c_ptr"
    ):
        raise _UnsupportedChain("register retirement lacks its original result view")
    declarations = tuple(ast.unparse(node) for node in (*allocations, *views))
    return PairResultCut(pair, prefix + "_c", declarations, pair.producer._publication)


def _same_statement(left: ast.stmt, right: ast.stmt) -> bool:
    """Compare parsed fields without dumping unrelated, deeply nested bodies."""
    pending: list[tuple[object, object]] = [(left, right)]
    while pending:
        first, second = pending.pop()
        if type(first) is not type(second):
            return False
        if isinstance(first, ast.AST) and isinstance(second, ast.AST):
            first_fields = tuple(ast.iter_fields(first))
            second_fields = tuple(ast.iter_fields(second))
            if tuple(name for name, _ in first_fields) != tuple(
                name for name, _ in second_fields
            ):
                return False
            pending.extend(
                (a, b)
                for (_, a), (_, b) in reversed(
                    tuple(zip(first_fields, second_fields, strict=True))
                )
            )
        elif isinstance(first, list) and isinstance(second, list):
            if len(first) != len(second):
                return False
            pending.extend(reversed(tuple(zip(first, second, strict=True))))
        elif first != second:
            return False
    return True


def retire_result_cut(source: Sequence[str], cut: PairResultCut) -> tuple[str, ...]:
    """Retire once, only when the complete emitted body has no remaining read."""
    cut.pair.check()
    if (
        cut.pair.phase != "consumed"
        or cut.owner != f"chain_{cut.pair.source}_c"
        or cut.publication != cut.pair.producer._publication
    ):
        raise _UnsupportedChain("register retirement receipt changed")
    source = _flat(source)
    tree = ast.parse("\n".join(source))
    expected = ast.parse("\n".join((*cut.declaration, *cut.publication))).body
    statements = tuple(node for node in ast.walk(tree) if isinstance(node, ast.stmt))
    removed: set[int] = set()
    for wanted in expected:
        matches = [node for node in statements if _same_statement(node, wanted)]
        if len(matches) != 1:
            raise _UnsupportedChain(
                "register retirement original C publication changed"
            )
        node = matches[0]
        assert node.end_lineno is not None
        removed.update(range(node.lineno - 1, node.end_lineno))
    result = tuple(
        line for line_index, line in enumerate(source) if line_index not in removed
    )
    names = {
        node.id
        for statement in expected
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    if cut.owner not in names or any(
        not name.startswith(f"chain_{cut.pair.source}_") for name in names
    ):
        raise _UnsupportedChain("register retirement includes an unrelated owner")
    if any(
        isinstance(node, ast.Name) and node.id in names
        for node in ast.walk(ast.parse("\n".join(result)))
    ):
        raise _UnsupportedChain("completed register C still has an emitted reader")
    return result


def retire_pair_results(
    emission: CompletedImageEmission,
    lines: Sequence[str],
    sources: tuple[tuple[Node, tuple[str, ...]], ...],
    pairs: tuple[WarpPairAttempt, ...],
    *,
    cuts: tuple[PairResultCut, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[tuple[Node, tuple[str, ...]], ...]]:
    """Retire after all original pair/image receipts; record the actual new body."""
    result = _flat(lines)
    records = dict(sources)
    occupied: set[int] = set()
    selected = (
        tuple(
            pair_result_cut(pair, records[emission.plan.dots[pair.source]])
            for pair in pairs
        )
        if cuts is None
        else cuts
    )
    if tuple(cut.pair for cut in selected) != pairs:
        raise _UnsupportedChain("completed register retirement group coverage changed")
    for pair, cut in zip(pairs, selected, strict=True):
        pair.check()
        index = pair.source
        node = emission.plan.dots[index]
        if (
            pair.codegen is not emission.cg
            or pair.plan is not emission.plan
            or pair.phase != "consumed"
            or node in emission.sources
            or index in occupied
            or index + 1 in occupied
            or not _legal_team(pair, emission.plan.threads)
        ):
            raise _UnsupportedChain("completed register retirement owner changed")
        occupied.update((index, index + 1))
        old = _flat(records[node])
        result = retire_result_cut(result, cut)
        records[node] = retire_result_cut(old, cut)
    return result, tuple((node, records[node]) for node, _ in sources)
