from __future__ import annotations

import ast
from copy import deepcopy
import functools
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_full_coverage import FLAG
from test.test_cute_grouped_full_coverage import KEY
from test.test_cute_grouped_full_coverage import _dump
from test.test_cute_grouped_full_coverage import _run_dense_producer
from test.test_cute_grouped_k128_full_coverage import _bound
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_shared_rhs_grouped import _target

from helion._compiler.autotuner_heuristics.cute import _tcgen05_grouped_worklist_config
from helion._compiler.cute.grouped_full_coverage import Tcgen05GroupedFullCoveragePlan
from helion._compiler.cute.grouped_full_coverage import full_coverage_index_domain
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import CodeType


MATRIX = (
    (640, 128, 128, 4, 64),
    (640, 128, 128, 4, 128),
    (2560, 512, 512, 8, 128),
    (10240, 1024, 1024, 16, 128),
    (672, 256, 384, 4, 128),
    (672, 256, 640, 4, 128),
    (32, 512, 128, 4, 128),
    (128, 256, 128, 4, 128),
    (640, 128, 192, 4, 64),
)


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield
    torch.set_num_threads(previous)


@functools.lru_cache
def _sources(m: int, n: int, k: int, groups: int, block_k: int) -> tuple[str, str]:
    with _target():
        bound, _args = _bound(m, n, k, groups)
        config = _tcgen05_grouped_worklist_config(
            32,
            block_k,
            2 if block_k == 64 else 4,
            240 if block_k == 64 else 256,
            runtime_direct=False,
            l2_swizzle_size=1,
        )
        config.config[KEY] = "fixed_tma_dense"
        dense = bound.to_code(config)
        config.config[KEY] = "fixed_tma_dense_local"
        return dense, bound.to_code(config)


def _statements(nodes: list[ast.stmt]) -> str:
    return _dump(ast.Module(body=nodes, type_ignores=[]))


def _roles(tree: ast.Module) -> list[ast.If]:
    result = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.While)
            and ast.unparse(child.test)
            in {f"tcgen05_role_local_{i}_valid" for i in range(3)}
            for child in node.body
        )
    ]
    assert len(result) == 3
    return result


def _scheduler(tree: ast.Module) -> ast.If:
    (result,) = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(6)"
    )
    return result


def _slot(node: ast.Subscript) -> int:
    assert ast.unparse(node.value) == "tcgen05_work_tile_smem"
    assert isinstance(node.slice, ast.Call)
    assert ast.unparse(node.slice.func) == "cutlass.Int32"
    (value,) = node.slice.args
    assert isinstance(value, ast.Constant) and type(value.value) is int
    return value.value


def _inverse_local(dense: str, local: str) -> tuple[str, dict[str, object]]:
    """Reverse exactly 13 scheduling regions; all remaining source bytes survive.

    The replacement fragments come from the complete emitted dense parent. Each
    false branch, data-body boundary, typed default and dense field expression is
    checked before any replacement; whole-file equality then covers everything
    outside these scheduling regions, including host code and pipeline drains.
    """
    before, after = ast.parse(dense), ast.parse(local)
    parent_scheduler = _scheduler(before)
    publisher = parent_scheduler.body[0]
    assert isinstance(publisher, ast.If) and ast.unparse(publisher.test) == FLAG
    assert len(publisher.body) == 3 and isinstance(publisher.body[2], ast.While)
    publisher_loop = publisher.body[2]
    publisher_fields = {
        _slot(node.targets[0]): node.value
        for node in ast.walk(publisher_loop)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
    }
    assert set(publisher_fields) == {0, 1, 2, 3, 5, 6, 8}
    assert ast.unparse(publisher_fields[2]) == "cutlass.Int32(1)"
    edits: list[tuple[int, int, str]] = []
    initializers: list[dict[str, object]] = []
    old_lines, new_lines = (
        dense.splitlines(keepends=True),
        local.splitlines(keepends=True),
    )

    def replace(new: list[ast.stmt], old: list[ast.stmt]) -> None:
        assert new and old and new[-1].end_lineno and old[-1].end_lineno
        edits.append(
            (
                new[0].lineno - 1,
                new[-1].end_lineno,
                "".join(old_lines[old[0].lineno - 1 : old[-1].end_lineno]),
            )
        )

    for index, (parent, candidate) in enumerate(
        zip(_roles(before), _roles(after), strict=True)
    ):
        assert _dump(parent.test) == _dump(candidate.test)
        prefix = f"tcgen05_role_local_{index}"
        valid, counter, step = (
            f"{prefix}_valid",
            f"{prefix}_dense_index",
            f"{prefix}_dense_step",
        )
        old_pos = next(
            i for i, node in enumerate(parent.body) if isinstance(node, ast.While)
        )
        new_pos = next(
            i for i, node in enumerate(candidate.body) if isinstance(node, ast.While)
        )
        old_loop, loop = parent.body[old_pos], candidate.body[new_pos]
        assert isinstance(old_loop, ast.While) and isinstance(loop, ast.While)
        assert ast.unparse(loop.test) == ast.unparse(old_loop.test) == valid
        initial = candidate.body[new_pos - 1]
        metadata, advance = loop.body[0], loop.body[-1]
        assert all(
            isinstance(node, ast.If) and ast.unparse(node.test) == FLAG
            for node in (initial, metadata, advance)
        )
        assert (
            isinstance(initial, ast.If)
            and isinstance(metadata, ast.If)
            and isinstance(advance, ast.If)
        )
        assert _statements(initial.orelse) == _statements(
            parent.body[old_pos - 4 : old_pos]
        )
        assert _statements(advance.orelse) == _statements(old_loop.body[-4:])
        assert _statements(candidate.body[: old_pos - 4]) == _statements(
            parent.body[: old_pos - 4]
        )
        defaults = candidate.body[old_pos - 4 : new_pos - 1]
        names = [
            ast.unparse(node.targets[0])
            for node in metadata.body
            if isinstance(node, ast.Assign)
        ]
        expected_defaults = ast.parse(
            f"{counter} = cutlass.Int32(0)\n{step} = cutlass.Int32(0)\n"
            f"{valid} = cutlass.Boolean(False)\n"
            + "\n".join(f"{name} = cutlass.Int32(0)" for name in names)
        ).body
        assert _statements(defaults) == _statements(expected_defaults)
        initializers.extend(
            {"line": node.lineno, "statement": ast.unparse(node)} for node in defaults
        )
        total = ast.unparse(publisher_loop.test.comparators[0])
        assert _statements(initial.body) == _statements(
            ast.parse(
                f"{counter} = cutlass.Int32(cute.arch.block_idx()[2])\n"
                f"{step} = cutlass.Int32(cute.arch.grid_dim()[2])\n"
                f"{valid} = {counter} < {total}"
            ).body
        )
        assert _statements(advance.body) == _statements(
            ast.parse(
                f"{counter} = {counter} + {step}\n{valid} = {counter} < {total}"
            ).body
        )
        for fixed, dynamic in zip(metadata.body, metadata.orelse, strict=True):
            assert isinstance(fixed, ast.Assign) and isinstance(dynamic, ast.Assign)
            assert _dump(fixed.targets[0]) == _dump(dynamic.targets[0])
            assert isinstance(dynamic.value, ast.Subscript)
            expression = ast.unparse(publisher_fields[_slot(dynamic.value)])
            expression = expression.replace("tcgen05_grouped_dense_index", counter)
            assert _dump(fixed.value) == _dump(ast.parse(expression, mode="eval").body)
        release_pos = next(
            i
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.If) and ast.unparse(node.test) == f"not {FLAG}"
        )
        release = loop.body[release_pos]
        assert isinstance(release, ast.If) and not release.orelse
        old_prefix = [*metadata.orelse, *loop.body[1:release_pos], *release.body]
        assert _statements(old_prefix) == _statements(old_loop.body[: len(old_prefix)])
        assert _statements(loop.body[release_pos + 1 : -1]) == _statements(
            old_loop.body[len(old_prefix) : -4]
        )
        (terminal,) = candidate.body[new_pos + 1 :]
        assert (
            isinstance(terminal, ast.If) and ast.unparse(terminal.test) == f"not {FLAG}"
        )
        assert not terminal.orelse and _statements(terminal.body) == _statements(
            parent.body[old_pos + 1 :]
        )
        replace(
            candidate.body[old_pos - 4 : new_pos], parent.body[old_pos - 4 : old_pos]
        )
        replace(loop.body[: release_pos + 1], old_loop.body[: len(old_prefix)])
        replace([advance], old_loop.body[-4:])
        replace([terminal], parent.body[old_pos + 1 :])

    scheduler = _scheduler(after)
    assert _dump(scheduler.test) == _dump(parent_scheduler.test)
    (fallback,) = scheduler.body
    assert isinstance(fallback, ast.If) and ast.unparse(fallback.test) == f"not {FLAG}"
    assert not fallback.orelse
    assert _statements(fallback.body) == _statements(
        [*publisher.orelse, *parent_scheduler.body[1:]]
    )
    replace(scheduler.body, parent_scheduler.body)
    assert len(edits) == 13
    ordered = sorted(edits)
    assert all(left[1] <= right[0] for left, right in itertools.pairwise(ordered))
    for start, end, replacement in reversed(ordered):
        new_lines[start:end] = [replacement]
    restored = "".join(new_lines)
    assert restored == dense
    return restored, {"regions": len(edits), "initializers": initializers}


def _projection(source: str, value: bool, erase_lines: set[int]) -> ast.Module:
    class Project(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
            return None if node.lineno in erase_lines else self.generic_visit(node)

        def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
            text = ast.unparse(node.test)
            if text in (FLAG, f"not {FLAG}"):
                selected = value if text == FLAG else not value
                result = []
                for child in node.body if selected else node.orelse:
                    mapped = self.visit(child)
                    if isinstance(mapped, list):
                        result.extend(mapped)
                    elif mapped is not None:
                        result.append(mapped)
                return result
            return self.generic_visit(node)

    result = Project().visit(ast.parse(source))
    assert isinstance(result, ast.Module)
    return result


@pytest.mark.parametrize("shape", MATRIX)
@skipUnlessBackends(["cute"])
def test_complete_source_inverse_fallback_and_host_contract(
    shape: tuple[int, ...],
) -> None:
    dense, local = _sources(*shape)
    _restored, proof = _inverse_local(dense, local)
    lines = {item["line"] for item in proof["initializers"]}
    assert _dump(_projection(local, False, lines)) == _dump(
        _projection(dense, False, set())
    )
    assert _plans(dense) == _plans(local)
    assert len(proof["initializers"]) == 25  # Six counters, three valid, 16 fields.
    true_tree = _projection(local, True, set())
    assert not any(
        isinstance(node, ast.Subscript)
        and ast.unparse(node.value) == "tcgen05_work_tile_smem"
        for node in ast.walk(true_tree)
    )
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(true_tree)
        if isinstance(node, ast.Call)
    ]
    assert not any(name.startswith("tcgen05_sched_pipeline.") for name in calls)
    assert calls.count("cute.arch.sync_threads") == 3
    # Allocation publication is CTA-wide even when worklist metadata stays
    # role-local. All TMEM consumers then enter one common named-barrier wait.
    assert calls.count("tcgen05_tmem_allocator.wait_for_alloc") == 1
    ((publication, wait),) = (
        (node.body[index - 1], statement)
        for node in ast.walk(true_tree)
        if isinstance(node, ast.FunctionDef)
        for index, statement in enumerate(node.body)
        if isinstance(statement, ast.If)
        and any(
            isinstance(child, ast.Call)
            and ast.unparse(child.func) == "tcgen05_tmem_allocator.wait_for_alloc"
            for child in ast.walk(statement)
        )
    )
    assert ast.unparse(publication) == "cute.arch.sync_threads()"
    assert isinstance(wait.test, ast.BoolOp) and isinstance(wait.test.op, ast.Or)
    for function in (
        "tcgen05_c_pipeline.producer_tail",
        "tcgen05_ab_pipeline.producer_tail",
        "tcgen05_tmem_allocator.free",
        "tcgen05_tmem_allocator.relinquish_alloc_permit",
    ):
        assert function in calls


def _consumer_code(role: ast.If) -> tuple[CodeType, dict[str, int], str]:
    (loop,) = (node for node in role.body if isinstance(node, ast.While))
    initial = role.body[role.body.index(loop) - 1]
    metadata, advance = loop.body[0], loop.body[-1]
    assert (
        isinstance(initial, ast.If)
        and isinstance(metadata, ast.If)
        and isinstance(advance, ast.If)
    )
    fields = {
        ast.unparse(node.targets[0]): _slot(node.value) for node in metadata.orelse
    }
    record = ast.parse(
        "packets.append({"
        + ",".join(f"{name!r}: int({name})" for name in fields)
        + "})"
    ).body
    counter = ast.unparse(initial.body[0].targets[0])
    statements = [
        *initial.body,
        ast.While(
            test=deepcopy(loop.test),
            body=[*metadata.body, *record, *advance.body],
            orelse=[],
        ),
    ]
    tree = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    return compile(tree, "<emitted-local-schedule>", "exec"), fields, counter


@pytest.mark.parametrize("shape", (MATRIX[0], MATRIX[2], MATRIX[6], MATRIX[7]))
@skipUnlessBackends(["cute"])
def test_all_consumers_match_actual_dense_mailbox_sequence(
    shape: tuple[int, ...],
) -> None:
    m, n, k, groups, block_k = shape
    dense, local = _sources(*shape)
    scheduler = _scheduler(ast.parse(dense))
    branch = scheduler.body[0]
    assert isinstance(branch, ast.If)
    old_code = compile(
        ast.Module(body=[*branch.body, *scheduler.body[1:]], type_ignores=[]),
        "<emitted-dense-publisher>",
        "exec",
    )
    role_codes = [_consumer_code(role) for role in _roles(ast.parse(local))]
    total = (m // 32) * (n // 128)
    assert full_coverage_index_domain(groups, m, n, k, 32, 128, block_k)
    for grid in sorted({1, 3, 5, min(116, groups * total), total + 1}):
        assert grid <= groups * total
        for cta in range(grid):
            packets, events = _run_dense_producer(old_code, cta, grid)
            assert packets[-1][2] == 0
            assert events == ["acquire", "commit", "advance", "sync"] * len(packets)
            for code, fields, counter in role_codes:
                namespace = {
                    "cutlass": SimpleNamespace(Int32=int),
                    "cute": SimpleNamespace(
                        arch=SimpleNamespace(
                            block_idx=lambda cta=cta: (0, 0, cta),
                            grid_dim=lambda grid=grid: (1, 1, grid),
                        )
                    ),
                    "packets": [],
                }
                exec(code, namespace)
                assert namespace["packets"] == [
                    {name: packet[slot] for name, slot in fields.items()}
                    for packet in packets[:-1]
                ]
                expected_last = cta + len(packets[:-1]) * grid
                assert namespace[counter] == expected_last <= (1 << 31) - 1


@pytest.mark.parametrize("consumer_local", (False, True))
def test_plan_retains_signed_terminal_bound(consumer_local: bool) -> None:
    limit = (1 << 31) - 1
    # T=2, maximum admitted grid bound is 2*((MAX-1)/2). The last
    # valid index is one and the final addition is exactly MAX.
    plan = Tcgen05GroupedFullCoveragePlan(
        "covered", (limit - 1) // 2, 2, 1, 1, 1, 1, 1, consumer_local
    )
    assert plan.tiles - 1 + plan.groups * plan.tiles == limit
    with pytest.raises(AssertionError):
        Tcgen05GroupedFullCoveragePlan(
            "covered", plan.groups + 1, 2, 1, 1, 1, 1, 1, consumer_local
        )


@pytest.mark.parametrize("value", (0, 1, "local"))
def test_plan_requires_exact_boolean_role_policy(value: object) -> None:
    with pytest.raises(AssertionError):
        Tcgen05GroupedFullCoveragePlan("covered", 4, 640, 128, 128, 32, 128, 64, value)
