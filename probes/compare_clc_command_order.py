# ruff: noqa: ANN001, ANN201, ANN202
"""Prove that symbolic CLC command ranges reproduce a saved command table.

The CLC dispatcher used to load each command's physical task from an immutable
GPU buffer.  It now evaluates a closed-form relation over the command index.
This probe reads both lowerings of the same kernel and checks that they emit
the identical ``(root, physical task)`` sequence for every command, so a
scheduling change cannot hide behind a re-lowered dispatcher.

Run from the repository root::

    python probes/compare_clc_command_order.py \
        --table-lowered probes/sm_gantt_clc_current \
        --range-lowered probes/sm_gantt_clc_symbolic
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

TASK_DISPATCH_FUNCTION = "tile_dependency_clc_task"
COMMAND_INDEX_VAR = "tile_dependency_clc_command_index"
FLAT_TASK_VAR = "tile_dependency_clc_flat_physical_task"
ROOT_TASK_PREFIX = "tile_dependency_clc_root_"
ROOT_TASK_SUFFIX = "_task"
OUTLINED_ROOT_PREFIX = "tile_dependency_root_"
DEFAULT_WORKLOADS = (
    "qwen3_ffn",
    "qwen3_decode_layer",
    "deepseek_v3_moe",
    "nemotron3_nano_moe",
)


@dataclass(frozen=True)
class DispatchRange:
    """One ``begin <= selector < end`` arm of the lowered dispatch chain."""

    root: int
    begin: int
    end: int
    task_expr: ast.expr
    body: tuple[ast.stmt, ...]


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _dispatch_function(module: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == TASK_DISPATCH_FUNCTION:
            return node
    raise LookupError(f"no {TASK_DISPATCH_FUNCTION!r} in lowered source")


def _selector_bounds(
    test: ast.expr,
    selector: str,
    implied_begin: int,
) -> tuple[int, int]:
    """Read a dispatch guard, including an implied lower bound from ``else``."""
    values = (
        test.values
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
        else (test,)
    )
    bounds: dict[type[ast.cmpop], int] = {}
    for value in values:
        if (
            not isinstance(value, ast.Compare)
            or len(value.ops) != 1
            or not isinstance(value.left, ast.Name)
            or value.left.id != selector
        ):
            raise ValueError(f"unexpected dispatch guard: {ast.unparse(test)}")
        operator = type(value.ops[0])
        if operator not in (ast.GtE, ast.Lt) or operator in bounds:
            raise ValueError(f"unexpected dispatch guard: {ast.unparse(test)}")
        bounds[operator] = ast.literal_eval(value.comparators[0])
    if ast.Lt not in bounds:
        raise ValueError(f"dispatch guard has no upper bound: {ast.unparse(test)}")
    begin = bounds.get(ast.GtE, implied_begin)
    end = bounds[ast.Lt]
    if begin >= end:
        raise ValueError(f"empty or reversed dispatch range: {begin}..{end}")
    return begin, end


def _root_assignment(statement: ast.stmt) -> tuple[int, ast.expr]:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        raise ValueError(f"unexpected dispatch body: {ast.unparse(statement)}")
    target = statement.targets[0]
    if (
        not isinstance(target, ast.Name)
        or not target.id.startswith(ROOT_TASK_PREFIX)
        or not target.id.endswith(ROOT_TASK_SUFFIX)
    ):
        raise ValueError(f"unexpected dispatch target: {ast.unparse(target)}")
    root = int(target.id[len(ROOT_TASK_PREFIX) : -len(ROOT_TASK_SUFFIX)])
    return root, statement.value


def _dispatch_ranges(module: ast.Module, selector: str) -> list[DispatchRange]:
    """Flatten the lowered ``if``/``elif`` dispatch chain into ordered ranges."""
    ranges: list[DispatchRange] = []
    body: list[ast.stmt] = _dispatch_function(module).body
    implied_begin = 0
    while body:
        statement = body[0]
        if not isinstance(statement, ast.If):
            break
        begin, end = _selector_bounds(statement.test, selector, implied_begin)
        root, task_expr = _root_assignment(statement.body[0])
        ranges.append(
            DispatchRange(root, begin, end, task_expr, tuple(statement.body[1:]))
        )
        implied_begin = end
        if statement.orelse and (
            len(statement.orelse) != 1
            or not isinstance(statement.orelse[0], ast.If)
        ):
            raise ValueError("dispatch chain has unexpected trailing statements")
        body = statement.orelse
    if not ranges:
        raise LookupError("lowered source has no CLC dispatch ranges")
    if body:
        raise ValueError("dispatch function has unexpected trailing statements")
    return ranges


def _command_table(module: ast.Module) -> tuple[int, ...]:
    """Read the immutable command table out of ``_constant_buffer_specs``."""
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "_constant_buffer_specs":
                continue
            specs = keyword.value
            if not isinstance(specs, ast.Tuple) or len(specs.elts) != 1:
                raise ValueError("expected exactly one CLC constant buffer")
            spec = specs.elts[0]
            if not isinstance(spec, ast.Tuple):
                raise ValueError("malformed CLC constant buffer spec")
            return tuple(ast.literal_eval(spec.elts[1]))
    raise LookupError("lowered source has no CLC command table")


def _module_constants(module: ast.Module) -> dict[str, str]:
    """Collect generated module-level constexprs independent of source order."""
    result: dict[str, str] = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        value = statement.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Call)
            and ast.unparse(value.func) == "tl.constexpr"
        ):
            result[target.id] = ast.dump(value, include_attributes=False)
    return result


def _launcher_signature(module: ast.Module) -> tuple[tuple[str, ...], tuple]:
    """Capture launch geometry and metadata, excluding the removed table."""
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_launcher"
    ]
    if len(calls) != 1:
        raise ValueError(f"expected one _launcher call, found {len(calls)}")
    call = calls[0]
    ignored_keywords = {"_constant_buffer_specs", "_cross_loop_dispatch_kind"}
    return (
        tuple(ast.dump(argument, include_attributes=False) for argument in call.args),
        tuple(
            sorted(
                (
                    keyword.arg,
                    ast.dump(keyword.value, include_attributes=False),
                )
                for keyword in call.keywords
                if keyword.arg not in ignored_keywords
            )
        ),
    )


def _artifact_metadata(path: Path) -> dict[str, object]:
    """Read stable provenance and config metadata adjacent to one lowering."""
    suffix = path.parent.name
    if not suffix.endswith("_lowered"):
        raise ValueError(f"unexpected lowering directory name: {suffix!r}")
    metadata_path = path.parent.parent / f"{suffix.removesuffix('_lowered')}.json"
    if not metadata_path.exists():
        raise ValueError(f"missing artifact metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    fields = (
        "workload",
        "device",
        "cache_state",
        "l2_flush_bytes",
        "megakernel_config",
    )
    missing = [field for field in fields if field not in metadata]
    if missing:
        raise ValueError(f"missing {missing!r} in {metadata_path}")
    return {field: metadata[field] for field in fields}


def _evaluate(expr: ast.expr, command_index: int) -> int:
    """Evaluate a lowered task expression over non-negative integers.

    The final task ordinal is non-negative, but intermediate affine expressions
    may not be. Division and remainder therefore model Triton's signed
    truncation semantics rather than Python's floor semantics.
    """

    def integer(value: object) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, Fraction) and value.denominator == 1:
            return value.numerator
        raise ValueError(f"expected an integer, got {value!r}")

    def trunc_div(left: object, right: object) -> int:
        numerator = integer(left)
        denominator = integer(right)
        if denominator == 0:
            raise ZeroDivisionError("division by zero in task expression")
        quotient = abs(numerator) // abs(denominator)
        return -quotient if (numerator < 0) != (denominator < 0) else quotient

    def visit(node: ast.expr) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id != COMMAND_INDEX_VAR:
                raise ValueError(f"unexpected name in task expression: {node.id}")
            return command_index
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)  # type: ignore[operator]
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right  # type: ignore[operator]
            if isinstance(node.op, ast.Sub):
                return left - right  # type: ignore[operator]
            if isinstance(node.op, ast.Mult):
                return left * right  # type: ignore[operator]
            if isinstance(node.op, ast.FloorDiv):
                return trunc_div(left, right)
            if isinstance(node.op, ast.Mod):
                return integer(left) - trunc_div(left, right) * integer(right)
            if isinstance(node.op, ast.Div):
                return Fraction(integer(left), integer(right))
            raise ValueError(f"unsupported operator: {ast.unparse(node)}")
        if isinstance(node, ast.BoolOp):
            values = [visit(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            return any(values)
        if isinstance(node, ast.Compare):
            result = True
            left = visit(node.left)
            for operator, comparator in zip(node.ops, node.comparators, strict=True):
                right = visit(comparator)
                if isinstance(operator, ast.Lt):
                    result = result and left < right  # type: ignore[operator]
                elif isinstance(operator, ast.LtE):
                    result = result and left <= right  # type: ignore[operator]
                elif isinstance(operator, ast.Gt):
                    result = result and left > right  # type: ignore[operator]
                elif isinstance(operator, ast.GtE):
                    result = result and left >= right  # type: ignore[operator]
                elif isinstance(operator, ast.Eq):
                    result = result and left == right
                elif isinstance(operator, ast.NotEq):
                    result = result and left != right
                else:
                    raise ValueError(f"unsupported comparison: {ast.unparse(node)}")
                left = right
            return result
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name == "tl.cast":
                return int(visit(node.args[0]))  # type: ignore[arg-type]
            if name == "tl.where":
                condition, if_true, if_false = (visit(arg) for arg in node.args)
                return if_true if condition else if_false
            if name in ("tl.minimum", "tl.maximum"):
                left, right = (visit(arg) for arg in node.args)
                return (min if name == "tl.minimum" else max)(left, right)
            raise ValueError(f"unsupported call in task expression: {name}")
        raise ValueError(f"unsupported expression: {ast.unparse(node)}")

    value = visit(expr)
    if isinstance(value, Fraction):
        if value.denominator != 1:
            raise ValueError(
                f"task expression is not integral: {ast.unparse(expr)} = {value}"
            )
        return value.numerator
    if not isinstance(value, int):
        raise ValueError(f"task expression is not an integer: {ast.unparse(expr)}")
    return value


def table_command_order(path: Path) -> list[tuple[int, int]]:
    """``(root, physical task)`` per command, from the constant-buffer lowering."""
    module = _module(path)
    table = _command_table(module)
    ranges = _dispatch_ranges(module, FLAT_TASK_VAR)
    order: list[tuple[int, int]] = []
    for command_index, flat_task in enumerate(table):
        matches = [
            entry for entry in ranges if entry.begin <= flat_task < entry.end
        ]
        if len(matches) != 1:
            raise ValueError(
                f"command {command_index} flat task {flat_task} matched "
                f"{len(matches)} dispatch ranges"
            )
        entry = matches[0]
        # The table lowering always assigned ``flat_task - begin``.
        order.append((entry.root, flat_task - entry.begin))
    return order


def range_command_order(path: Path) -> list[tuple[int, int]]:
    """``(root, physical task)`` per command, from the symbolic-range lowering."""
    module = _module(path)
    ranges = _dispatch_ranges(module, COMMAND_INDEX_VAR)
    ordered = sorted(ranges, key=lambda entry: entry.begin)
    if ordered[0].begin != 0:
        raise ValueError(f"command ranges start at {ordered[0].begin}, not 0")
    for previous, following in zip(ordered, ordered[1:], strict=False):
        if previous.end != following.begin:
            raise ValueError(
                f"command ranges leave a gap between {previous.end} and "
                f"{following.begin}"
            )
    order: list[tuple[int, int]] = []
    for entry in ordered:
        for command_index in range(entry.begin, entry.end):
            order.append((entry.root, _evaluate(entry.task_expr, command_index)))
    return order


class _AlphaNormalizer(ast.NodeTransformer):
    """Normalize generated locals and inline helper identities structurally."""

    def __init__(self, functions, function_fingerprint, argument_names=()):
        self.functions = functions
        self.function_fingerprint = function_fingerprint
        self.names = {
            name: f"argument_{index}" for index, name in enumerate(argument_names)
        }
        self.next_local = 0

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in self.functions
            and node.func.id.startswith(OUTLINED_ROOT_PREFIX)
        ):
            fingerprint = self.function_fingerprint(node.func.id)
            digest = hashlib.sha256(fingerprint.encode()).hexdigest()
            node.func = ast.Name(id=f"helper_{digest}", ctx=ast.Load())
            node.args = [self.visit(argument) for argument in node.args]
            node.keywords = [self.visit(keyword) for keyword in node.keywords]
            return node
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if isinstance(node.ctx, ast.Store) and node.id not in self.names:
            self.names[node.id] = f"local_{self.next_local}"
            self.next_local += 1
        if node.id in self.names:
            node.id = self.names[node.id]
        return node


def _task_bodies(
    module: ast.Module,
    ranges: list[DispatchRange],
) -> dict[int, str]:
    """Fingerprint every dispatched root and its reachable outlined helpers."""
    functions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
    }
    function_cache: dict[str, str] = {}
    visiting: set[str] = set()

    def function_fingerprint(name: str) -> str:
        cached = function_cache.get(name)
        if cached is not None:
            return cached
        if name in visiting:
            raise ValueError(f"recursive lowered helper {name!r}")
        visiting.add(name)
        function = copy.deepcopy(functions[name])
        arguments = [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if function.args.vararg is not None:
            arguments.append(function.args.vararg)
        if function.args.kwarg is not None:
            arguments.append(function.args.kwarg)
        argument_names = tuple(argument.arg for argument in arguments)
        function.name = "lowered_helper"
        normalizer = _AlphaNormalizer(
            functions,
            function_fingerprint,
            argument_names,
        )
        for argument in arguments:
            argument.arg = normalizer.names[argument.arg]
        normalized = normalizer.visit(function)
        fingerprint = ast.dump(normalized, include_attributes=False)
        visiting.remove(name)
        function_cache[name] = fingerprint
        return fingerprint

    result: dict[int, str] = {}
    for entry in ranges:
        body = ast.Module(body=list(copy.deepcopy(entry.body)), type_ignores=[])
        normalized = _AlphaNormalizer(functions, function_fingerprint).visit(body)
        fingerprint = ast.dump(normalized, include_attributes=False)
        previous = result.setdefault(entry.root, fingerprint)
        if previous != fingerprint:
            raise ValueError(
                f"root {entry.root} has different bodies in separate dispatch ranges"
            )
    return result


def _sync_signature(module: ast.Module) -> Counter[tuple[str, str]]:
    """Count synchronization opcodes and their non-address metadata."""
    result: Counter[tuple[str, str]] = Counter()
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.startswith("tl.atomic_"):
            metadata = tuple(
                sorted(
                    (
                        keyword.arg,
                        ast.dump(keyword.value, include_attributes=False),
                    )
                    for keyword in node.keywords
                    if keyword.arg in ("sem", "scope")
                )
            )
            result[(name, repr(metadata))] += 1
            continue
        if name != "tl.inline_asm_elementwise":
            continue
        assembly = next(
            (
                ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg == "asm"
            ),
            "",
        )
        if any(
            token in assembly
            for token in (
                "bar.",
                "clusterlaunchcontrol",
                "griddepcontrol",
                "ld.acquire",
                "mbarrier",
                "nanosleep",
            )
        ):
            metadata = tuple(
                sorted(
                    (
                        keyword.arg,
                        ast.dump(keyword.value, include_attributes=False),
                    )
                    for keyword in node.keywords
                    if keyword.arg != "args"
                )
            )
            result[(name, repr(metadata))] += 1
    return result


def compare(
    table_path: Path,
    range_path: Path,
    *,
    check_task_bodies: bool = True,
) -> tuple[bool, str]:
    table_module = _module(table_path)
    range_module = _module(range_path)
    if _artifact_metadata(table_path) != _artifact_metadata(range_path):
        return False, "artifact provenance or megakernel config changed"
    if _module_constants(table_module) != _module_constants(range_module):
        return False, "module constexprs changed"
    if _launcher_signature(table_module) != _launcher_signature(range_module):
        return False, "launch geometry or metadata changed"
    table_ranges = _dispatch_ranges(table_module, FLAT_TASK_VAR)
    range_ranges = _dispatch_ranges(range_module, COMMAND_INDEX_VAR)
    table_order = table_command_order(table_path)
    range_order = range_command_order(range_path)
    if len(table_order) != len(range_order):
        return False, (
            f"command count changed: {len(table_order)} -> {len(range_order)}"
        )
    mismatches = [
        (index, expected, actual)
        for index, (expected, actual) in enumerate(
            zip(table_order, range_order, strict=True)
        )
        if expected != actual
    ]
    distinct = len(set(range_order))
    detail = f"{len(range_order)} commands, {distinct} distinct (root, task) pairs"
    if mismatches:
        head = ", ".join(
            f"command {index}: {expected} != {actual}"
            for index, expected, actual in mismatches[:5]
        )
        return False, f"{len(mismatches)} mismatches out of {detail}; {head}"
    if distinct != len(range_order):
        return False, f"{detail}; a physical task is dispatched more than once"
    table_bodies: dict[int, str] = {}
    if check_task_bodies:
        table_bodies = _task_bodies(table_module, table_ranges)
        range_bodies = _task_bodies(range_module, range_ranges)
        if table_bodies != range_bodies:
            changed_roots = sorted(
                root
                for root in table_bodies.keys() | range_bodies.keys()
                if table_bodies.get(root) != range_bodies.get(root)
            )
            return False, (
                f"{detail}; scheduled bodies changed for roots {changed_roots}"
            )
    table_sync = _sync_signature(table_module)
    range_sync = _sync_signature(range_module)
    if table_sync != range_sync:
        return False, f"{detail}; synchronization sites changed"
    try:
        _command_table(range_module)
    except LookupError:
        pass
    else:
        return False, f"{detail}; symbolic lowering still has a command table"
    body_detail = (
        f"{len(table_bodies)} transitive root bodies unchanged"
        if check_task_bodies
        else "root-body comparison skipped"
    )
    return True, (
        f"{detail}, {body_detail}, and "
        f"{sum(table_sync.values())} synchronization opcode/metadata entries unchanged"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-lowered",
        type=Path,
        default=Path("probes/sm_gantt_clc_current"),
        help="Directory of saved constant-buffer (command-table) lowerings.",
    )
    parser.add_argument(
        "--range-lowered",
        type=Path,
        default=Path("probes/sm_gantt_clc_symbolic"),
        help="Directory of symbolic command-range lowerings.",
    )
    parser.add_argument("--workload", action="append", default=None)
    parser.add_argument(
        "--command-order-only",
        action="store_true",
        help="Allow intentional generated task-body simplifications.",
    )
    args = parser.parse_args()

    workloads = args.workload or DEFAULT_WORKLOADS
    failures = 0
    for workload in workloads:
        suffix = f"{workload}_separate_vs_megakernel_sm_gantt_lowered"
        table_path = args.table_lowered / suffix / "megakernel_untraced.py"
        range_path = args.range_lowered / suffix / "megakernel_untraced.py"
        if not table_path.exists() or not range_path.exists():
            print(f"{workload:24s} DIFFER  missing lowering")
            failures += 1
            continue
        try:
            matched, detail = compare(
                table_path,
                range_path,
                check_task_bodies=not args.command_order_only,
            )
        except (LookupError, ValueError) as error:
            matched, detail = False, str(error)
        print(f"{workload:24s} {'MATCH' if matched else 'DIFFER'}  {detail}")
        failures += not matched
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
