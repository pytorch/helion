"""Complete FP32 register rows for a proved successive max/exp/sum pair."""

from __future__ import annotations

import ast
from itertools import pairwise
import math
from typing import TYPE_CHECKING
from typing import cast

from . import cluster_online_pair as pair
from .hoist_warp_reduce import _static_trip_count
from .rolled_cluster_online_pair import _prove
from .rolled_cluster_online_pair import _wrap_sites

if TYPE_CHECKING:
    from .register_row_tensor_copy import TensorCopyContext


def _trip_count(loop: ast.For, constants: dict[str, int] | None) -> int | None:
    from .fuse_two_pass_loads import _range_bounds
    from .fuse_two_pass_loads import _trip_count_for

    result = _static_trip_count(loop, constants)
    if result is not None:
        return result
    bounds = _range_bounds(loop.iter)
    return (
        _trip_count_for(*bounds, constants or {}, allow_dynamic_base=True)
        if bounds is not None
        else None
    )


def lower_register_row_reduction(
    body: list[ast.stmt],
    constants: dict[str, int] | None,
    *,
    tensor_copy_context: TensorCopyContext | None = None,
) -> list[ast.stmt]:
    from .fuse_two_pass_loads import _range_bounds

    working, owners = _wrap_sites(body)
    sites = pair._find_sites(working)
    if sites is None:
        return body
    for site_a, site_b in pairwise(sites):
        if not (
            id(site_a.for_node) in owners
            and id(site_b.for_node) in owners
            and site_a.op == "max"
            and site_b.op == "sum"
            and site_a.group_span == site_b.group_span
            and site_a.group_span is not None
            and site_a.group_span >= 64
            and site_a.cluster_n == site_b.cluster_n
            and site_a.cluster_n is not None
            and 1 < site_a.cluster_n <= 16
        ):
            continue
        proof = _prove(working, site_a, site_b)
        if proof is None:
            continue
        maximum, scale, _subtractions = proof
        caches = {
            pair._assigned_name(statement)
            for statement in working
            if pair._is_name_assign(statement)
            and isinstance(statement.value, ast.Call)
            and ast.unparse(statement.value.func) == "cute.make_rmem_tensor"
        }
        if any(
            name in caches
            for statement in working
            if statement is not site_a.for_node
            and statement is not site_b.for_node
            and not (
                pair._is_name_assign(statement)
                and pair._assigned_name(statement) in caches
            )
            for name in pair._loads(statement)
        ):
            continue
        reduced = pair._reduced_values(working, site_a, lambda name: name, caches)
        if reduced is None:
            continue
        _elements, chain, _values = reduced
        loops = [loop for loop, _holder in chain if loop is not site_a.for_node]
        trips = [_trip_count(loop, constants) for loop in loops]
        if not trips or any(trip is None or trip <= 0 for trip in trips):
            continue
        valid_trips = [trip for trip in trips if trip is not None]
        elements = math.prod(valid_trips)
        if not 2 <= elements <= 128:
            continue
        acc = site_a.call.args[0]
        if not isinstance(acc, ast.Name):
            continue
        folds = [
            node
            for statement in site_a.for_node.body[: site_a.stmt_idx]
            for node in ast.walk(statement)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and pair._assigned_name(node) == acc.id
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func)
            in ("cute.arch.fmax", "cute.math.max", "_cute_nan_max")
        ]
        if len(folds) != 1:
            continue
        fold = folds[0]
        first, second = cast("ast.Call", fold.value).args
        value = second if isinstance(first, ast.Name) and first.id == acc.id else first
        row = pair._unique_name(working, lambda name: name, "_register_row")
        values = pair._unique_name(working, lambda name: name, "_register_values")
        local_max = pair._unique_name(working, lambda name: name, "_register_max")
        frame = pair._unique_name(working, lambda name: name, "_register_frame")
        exponents = pair._unique_name(working, lambda name: name, "_register_exponents")
        global_max = pair._unique_name(
            working, lambda name: name, "_register_global_max"
        )
        path: list[tuple[str, int]] = []

        def load_statements(
            statements: list[ast.stmt],
            *,
            _path: list[tuple[str, int]] = path,
            _fold: ast.Assign = fold,
            _row: str = row,
            _value: ast.expr = value,
            _caches: set[str] = caches,
            _acc: ast.Name = acc,
            _working: list[ast.stmt] = working,
        ) -> list[ast.stmt]:
            result: list[ast.stmt] = []
            for statement in statements:
                if isinstance(statement, ast.For):
                    bounds = _range_bounds(statement.iter)
                    trip = _trip_count(statement, constants)
                    if (
                        bounds is None
                        or trip is None
                        or not isinstance(statement.target, ast.Name)
                    ):
                        raise AssertionError(
                            "proved register-row loop lost its static bound"
                        )
                    index = pair._unique_name(
                        _working, str, f"_register_index_{len(_path)}"
                    )
                    _path.append((index, trip))
                    iterator = pair._stmt(
                        f"for {index} in cutlass.range_constexpr({trip}):\n    pass"
                    )
                    assert isinstance(iterator, ast.For)
                    begin, _end, step = bounds
                    iterator.body = [
                        pair._stmt(
                            f"{statement.target.id} = {ast.unparse(begin)} + "
                            f"cutlass.Int32({index}) * {ast.unparse(step)}"
                        ),
                        *load_statements(statement.body),
                    ]
                    _path.pop()
                    result.append(iterator)
                elif statement is _fold:
                    position = "0"
                    for index, trip in _path:
                        position = f"({position}) * {trip} + {index}"
                    result.append(
                        pair._stmt(f"{_row}[{position}] = {ast.unparse(_value)}")
                    )
                elif (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Subscript)
                    and isinstance(statement.targets[0].value, ast.Name)
                    and statement.targets[0].value.id in _caches
                ) or (
                    pair._is_name_assign(statement)
                    and pair._assigned_name(statement) == _acc.id
                ):
                    continue
                else:
                    result.append(statement)
            return result

        copy_plan = None
        if tensor_copy_context is not None:
            from .register_row_tensor_copy import prove_tensor_copy

            copy_plan = prove_tensor_copy(
                working,
                site_a,
                site_b,
                fold,
                caches,
                chain,
                _values,
                valid_trips,
                constants or {},
                tensor_copy_context,
            )
        load_body = (
            [
                copy_plan.assignment(values),
                *(
                    statement
                    for statement in site_a.for_node.body[: site_a.stmt_idx]
                    if pair._is_name_assign(statement) and statement is not fold
                ),
            ]
            if copy_plan is not None
            else [
                pair._stmt(
                    f"{row} = cute.make_rmem_tensor({elements}, cutlass.Float32)"
                ),
                *load_statements(site_a.for_node.body[: site_a.stmt_idx]),
                pair._stmt(f"{values} = {row}.load()"),
            ]
        )
        site_a.for_node.body = [
            *load_body,
            pair._stmt(
                f"{local_max} = cutlass.Float32({values}.reduce(cute.ReductionOp.MAX, "
                "init_val=-cutlass.Float32.inf, reduction_profile=0))"
            ),
            pair._stmt(
                f"{maximum} = _cute_grouped_reduce_warp({local_max}, 'max', "
                f"cutlass.Float32(float('-inf')), {ast.unparse(site_a.call.args[3])}, "
                "pre=1, group_span=32)"
            ),
            pair._stmt(
                f"{frame} = cutlass.Float32(0) if {maximum} == "
                f"cutlass.Float32(float('-inf')) else {maximum}"
            ),
        ]
        sum_acc = site_b.call.args[0]
        if not isinstance(sum_acc, ast.Name):
            continue
        lane_assignments = [
            statement
            for statement in site_b.for_node.body[: site_b.stmt_idx]
            if pair._is_name_assign(statement)
            and pair._assigned_name(statement) == ast.unparse(site_b.call.args[3])
        ]
        sum_tail = site_b.for_node.body[site_b.stmt_idx + 1 :]
        site_b.for_node.body = [
            *lane_assignments,
            pair._stmt(
                f"{exponents} = cute.math.exp2(({values} - {frame}) * {scale!r}, fastmath=True)"
            ),
            pair._stmt(
                f"{sum_acc.id} = cutlass.Float32({exponents}.reduce(cute.ReductionOp.ADD, "
                "init_val=cutlass.Float32(0), reduction_profile=0))"
            ),
            pair._stmt(
                f"{global_max}, {site_b.target} = _cute_grouped_reduce_cluster_warp_online_pair("
                f"{sum_acc.id}, {maximum}, {ast.unparse(site_b.call.args[3])}, "
                f"{site_b.buf_name}, {site_b.mbar_name}, group_span={site_b.group_span}, "
                f"cluster_n={site_b.cluster_n}, scale={scale!r}, fastmath=False)"
            ),
            pair._stmt(f"{maximum} = {global_max}"),
            *sum_tail,
        ]
        pair._rebuffer_pair(working, site_a, site_b)
        for statement in working:
            if (
                pair._is_name_assign(statement)
                and pair._assigned_name(statement) == site_b.buf_name
            ):
                statement.value = cast(
                    "ast.Assign",
                    pair._stmt(
                        f"_x = cute.arch.alloc_smem(cutlass.Int64, {cast('int', site_b.group_span) // 32 * cast('int', site_b.cluster_n)})"
                    ),
                ).value
        working = [
            statement
            for statement in working
            if not (
                pair._is_name_assign(statement)
                and pair._assigned_name(statement) in caches
            )
        ]
        break
    result: list[ast.stmt] = []
    for statement in working:
        result.extend(
            cast("ast.For", statement).body if id(statement) in owners else [statement]
        )
    return result
