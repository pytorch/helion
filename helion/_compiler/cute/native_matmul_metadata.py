"""Opt-in evidence for a committed, completely native matmul body.

Receipts are emitted by concrete native implementations, not planning strategies.
They are useful only inside the body attempt that actually installs their source.
An unknown legacy matmul signal is monotonic denial for the device function.
"""

from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
import textwrap
from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan

KEY = "cute_native_matmul_metadata"


def _snapshot(value: object) -> object:
    if isinstance(value, dict):
        return (dict, tuple((_snapshot(k), _snapshot(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_snapshot(v) for v in value))
    return (type(value), value)


def _revision(plan: ChainedMatmulPlan) -> object:
    from .chained_matmul import _register_bridge_revision

    return _snapshot(
        (
            _register_bridge_revision(plan),
            plan.axes,
            plan.dtype,
            tuple(plan.operand_dtype(i) for i in range(len(plan.dots))),
        )
    )


def _lines(lines: Sequence[str]) -> tuple[str, ...]:
    return tuple(line for block in lines for line in block.splitlines())


def _contains(body: tuple[str, ...], segment: tuple[str, ...]) -> bool:
    # Only a uniform enclosing-role indent is permitted. Relative indentation
    # (including a publication's control scope) is part of the stage receipt.
    expected = textwrap.dedent("\n".join(segment))
    return bool(segment) and any(
        textwrap.dedent("\n".join(body[i : i + len(segment)])) == expected
        for i in range(len(body) - len(segment) + 1)
    )


@dataclass(frozen=True)
class _NativeStage:
    attempt: object
    nodes: tuple[Node, ...]
    implementation: str
    revision: object
    source: tuple[str, ...]


@dataclass
class _BodyAttempt:
    codegen: GenerateAST
    device: object
    config: object
    stages: tuple[_NativeStage, ...] = ()
    receipts: tuple[object, ...] = ()
    installed: bool = False


@dataclass(frozen=True)
class _CommittedBody:
    attempt: _BodyAttempt
    plan: ChainedMatmulPlan
    revision: object
    stages: tuple[_NativeStage, ...]
    source: tuple[str, ...]
    body: tuple[str, ...]
    receipts: tuple[object, ...]


@dataclass
class MatmulLayouts:
    unknown: bool = False
    active: _BodyAttempt | None = None
    committed: list[_CommittedBody] = field(default_factory=list)


def _ledger(cg: GenerateAST) -> MatmulLayouts | None:
    # Once created, the obligation outlives changes to the mutable config.
    # In particular, an unknown use cannot disappear during a temporary opt-out.
    if cg.cute_matmul_layouts is not None:
        return cg.cute_matmul_layouts
    if cg.device_function.config.config.get(KEY, False) is not True:
        return None
    cg.cute_matmul_layouts = MatmulLayouts()
    return cg.cute_matmul_layouts


def record_unknown(cg: GenerateAST) -> None:
    cg.cute_uses_matmul = True
    ledger = _ledger(cg)
    if ledger is not None:
        ledger.unknown = True


@contextmanager
def body_attempt(cg: GenerateAST) -> Iterator[None]:
    ledger = _ledger(cg)
    if ledger is None:
        yield
        return
    previous = ledger.active
    attempt = _BodyAttempt(
        cg, cg.device_function, _snapshot(cg.device_function.config.config)
    )
    ledger.active = attempt
    try:
        yield
    finally:
        # Never import stage receipts from an abandoned previous invocation.
        ledger.active = previous


def record_native_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    nodes: tuple[Node, ...],
    implementation: str,
    lines: Sequence[str],
) -> None:
    ledger = _ledger(cg)
    if ledger is None:
        return
    attempt = ledger.active
    if (
        attempt is None
        or attempt.installed
        or attempt.device is not cg.device_function
        or attempt.config != _snapshot(cg.device_function.config.config)
        or implementation not in ("warp", "tcgen05", "register_island")
        or not nodes
        or len(set(nodes)) != len(nodes)
        or any(node not in plan.dots for node in nodes)
    ):
        raise exc.BackendUnsupported("cute", "invalid native matmul body attempt")
    stage = _NativeStage(attempt, nodes, implementation, _revision(plan), _lines(lines))
    attempt.stages += (stage,)
    attempt.receipts += (_stage_facts(stage),)


def _stage_facts(stage: _NativeStage) -> object:
    return (
        id(stage),
        id(stage.attempt),
        stage.nodes,
        stage.implementation,
        stage.revision,
        stage.source,
    )


def commit_body(cg: GenerateAST, plan: ChainedMatmulPlan, lines: Sequence[str]) -> None:
    cg.cute_uses_matmul = True
    ledger = _ledger(cg)
    if ledger is None:
        return
    attempt = ledger.active
    revision, source = _revision(plan), _lines(lines)
    if (
        ledger.unknown
        or attempt is None
        or attempt.installed
        or attempt.codegen is not cg
        or attempt.device is not cg.device_function
        or attempt.config != _snapshot(cg.device_function.config.config)
        or attempt.receipts != tuple(map(_stage_facts, attempt.stages))
        or Counter(node for stage in attempt.stages for node in stage.nodes)
        != Counter(plan.dots)
        or any(
            stage.attempt is not attempt
            or stage.revision != revision
            or not _contains(source, stage.source)
            for stage in attempt.stages
        )
    ):
        raise exc.BackendUnsupported(
            "cute", "incomplete committed native matmul coverage"
        )
    attempt.installed = True
    ledger.committed.append(
        _CommittedBody(
            attempt,
            plan,
            revision,
            attempt.stages,
            source,
            tuple(map(ast.dump, cg.device_function.body)),
            attempt.receipts,
        )
    )


def validate_native_metadata(cg: GenerateAST) -> bool:
    ledger = _ledger(cg)
    if ledger is None:
        return False
    from ...runtime.cute.launcher import _cute_wrapper_plan_bakes_tensor_shapes

    if ledger.unknown or not ledger.committed or ledger.active is not None:
        raise exc.BackendUnsupported(
            "cute", "native metadata requires committed native matmul uses"
        )
    body = tuple(map(ast.dump, cg.device_function.body))
    if len(ledger.committed) != 1:
        raise exc.BackendUnsupported("cute", "multiple native body replacements")
    committed = ledger.committed[0]
    attempt = committed.attempt
    if (
        attempt.codegen is not cg
        or attempt.device is not cg.device_function
        or not attempt.installed
        or attempt.config != _snapshot(cg.device_function.config.config)
        or committed.revision != _revision(committed.plan)
        or committed.stages != attempt.stages
        or committed.receipts != attempt.receipts
        or committed.receipts != tuple(map(_stage_facts, attempt.stages))
        or committed.body != body
        or any(
            not _cute_wrapper_plan_bakes_tensor_shapes(plan, native_metadata=True)
            for plan in cg.cute_wrapper_plans
        )
    ):
        raise exc.BackendUnsupported("cute", "committed native matmul body changed")
    return True
