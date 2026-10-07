from __future__ import annotations

import ast
import dataclasses
import itertools
from typing import cast

import numpy as np
import pytest

from test.test_cute_resident_sequence import _execute
from test.test_cute_resident_sequence import _indent
from test.test_cute_resident_sequence import _reduce

from helion import exc
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.resident_sequence import SequenceRegion
from helion._compiler.cute.resident_sequence import materialize_resident_sequences


def _source(threads: int = 32, slots: int = 2) -> str:
    prefix = """row = cutlass.Int32(cute.arch.thread_idx()[1])
state_max = cutlass.Float32(initial_max)
state_sum = cutlass.Float32(initial_sum)
"""
    loops = []
    for ordinal in range(2):
        column = f"column{ordinal}"
        value = f"value{ordinal}"
        calculation = (
            f"""masked0 = value0 if column0 < width else cutlass.Float32(float('-inf'))
maximum0 = {_reduce("masked0", "max")}
new_max = cute.math.max(state_max, maximum0, propagate_nan=True)
"""
            if ordinal == 0
            else f"""exponential1 = cute.math.exp(value1 - state_max) if column1 < width else cutlass.Float32(0)
summed1 = {_reduce("exponential1")}
new_sum = state_sum + summed1
"""
        )
        scalar = (
            f"{column} = offset{ordinal} + cutlass.Int32(lane{ordinal}) * {threads} + cutlass.Int32(cute.arch.thread_idx()[0])\n"
            f"{value} = cutlass.Float32((x.iterator + row * width + {column}).load()) if {column} < width and row < rows else cutlass.Float32(0)\n"
            + calculation
        )
        loops.append(
            f"for offset{ordinal} in range(cutlass.Int32(0), cutlass.Int32(width), cutlass.Int32({threads * slots})):\n"
            f"    for lane{ordinal} in range({slots}):\n" + _indent(scalar, 8) + "\n"
        )
    return (
        prefix
        + "".join(loops)
        + "if row < rows and cute.arch.thread_idx()[0] == 0:\n"
        + "    (out.iterator + row * 2).store(state_max)\n"
        + "    (out.iterator + row * 2 + 1).store(state_sum)\n"
    )


def _rewrite(
    source: str,
    *,
    threads: int = 32,
    groups: int = 2,
    slots: int = 2,
    mutation: str = "",
) -> str:
    body = ast.parse(source).body
    loops = {
        node.target.id: node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"offset0", "offset1"}
    }
    regions = {}
    for ordinal in range(2):
        loop = loops[f"offset{ordinal}"]
        setattr(loop.body[0], HELION_LANE_LOOP_VAR_ATTR, f"lane{ordinal}")
        region = SequenceRegion(
            loop,
            f"lane{ordinal}",
            slots,
            None,
            1,
            0,
            threads,
            threads * groups,
            f"cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(cute.arch.thread_idx()[1]) * {threads}",
        )
        if ordinal == 1:
            if mutation == "reduction_axis":
                region = dataclasses.replace(region, reduction_axis=1)
            elif mutation == "group_threads":
                region = dataclasses.replace(region, threads=threads * 2)
            elif mutation == "thread_count":
                region = dataclasses.replace(region, thread_count=threads * groups * 2)
            elif mutation == "linear_thread":
                region = dataclasses.replace(region, linear_thread="different_group")
        if ordinal == 1 or mutation != "unregistered":
            regions[id(loop)] = region
    if mutation == "conditional_producer":
        index = body.index(loops["offset0"])
        branch = ast.parse("if width > 0:\n    pass").body[0]
        assert isinstance(branch, ast.If)
        branch.body = [loops["offset0"]]
        body[index] = branch
    elif mutation == "carry_branch":
        index = body.index(loops["offset1"])
        branch = ast.parse("if state_max > cutlass.Float32(0):\n    pass").body[0]
        assert isinstance(branch, ast.If)
        branch.body = [loops["offset1"]]
        body[index] = branch
    counter = itertools.count()
    renames = {"new_max": "state_max", "new_sum": "state_sum"}
    if mutation == "alias_write":
        renames["overwritten"] = "state_max"
    rewritten = materialize_resident_sequences(
        list(body),
        regions=regions,
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        rename_groups=renames,
        disjoint_pairs={frozenset(("x", "out"))},
        boundary_names={"x", "out", "width", "rows", "initial_max", "initial_sum"},
        resident=True,
        new_var=lambda hint: f"{hint}_{next(counter)}",
    )

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            node.id = renames.get(node.id, node.id)
            return node

    module = ast.Module(body=cast("list[ast.stmt]", rewritten), type_ignores=[])
    Rename().visit(module)
    return ast.unparse(module)


@pytest.mark.parametrize("threads,groups", [(32, 2), (64, 2), (128, 1)])
@pytest.mark.parametrize("columns", [0, 1, 63, 128, 129, 257])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_initialized_carries_between_resident_regions(
    threads: int, groups: int, columns: int, nonfinite: bool
) -> None:
    lowered = _rewrite(_source(threads), threads=threads, groups=groups)
    rows = max(groups - 1, 1)
    values = np.linspace(-4, 2, rows * columns, dtype=np.float32).reshape(rows, columns)
    if nonfinite and columns:
        values[:, : min(64, columns)] = -np.inf
        if columns > 129:
            values[:, -1] = np.nan
    initial = (-np.inf, 0.0)
    actual, tensor = _execute(
        lowered,
        values,
        rows=rows,
        threads=threads,
        groups=groups,
        initial=initial,
        softmax=True,
    )
    with np.errstate(all="ignore"):
        maximum = np.maximum.reduce(values, axis=1, initial=np.float32(-np.inf))
        summed = np.exp(values - maximum[:, None]).sum(axis=1)
    np.testing.assert_allclose(
        actual,
        np.stack((maximum, summed), axis=1),
        rtol=3e-6,
        atol=3e-5,
        equal_nan=True,
    )
    assert np.all(tensor.reads == 2)
    # These independent sweeps have no per-phase replay to cache.
    assert "cute.make_rmem_tensor" not in lowered


@pytest.mark.parametrize(
    "mutation",
    [
        "uninitialized",
        "varying_initializer",
        "varying_output",
        "intervening_loop",
        "conditional_write",
        "alias_write",
        "unregistered",
        "conditional_producer",
        "reduction_axis",
        "group_threads",
        "thread_count",
        "linear_thread",
        "carry_iterator",
        "carry_branch",
    ],
)
def test_prior_regions_do_not_establish_unproved_carries_or_cta_guards(
    mutation: str,
) -> None:
    source = _source()
    if mutation in {"uninitialized", "varying_initializer"}:
        source = source.replace(
            "state_max = cutlass.Float32(initial_max)\n",
            ""
            if mutation == "uninitialized"
            else "state_max = cutlass.Float32(cute.arch.thread_idx()[0])\n",
        ).replace(
            "new_max = cute.math.max(state_max, maximum0, propagate_nan=True)",
            "new_max = maximum0",
        )
    elif mutation == "varying_output":
        source = source.replace(
            "new_max = cute.math.max", "new_max = value0 + cute.math.max"
        )
    elif mutation in {"intervening_loop", "conditional_write", "alias_write"}:
        insertion = {
            "intervening_loop": "for warmup in range(1):\n    state_max = cutlass.Float32(cute.arch.thread_idx()[0])\n",
            "conditional_write": "if width > 0:\n    state_max = cutlass.Float32(cute.arch.thread_idx()[0])\n",
            "alias_write": "overwritten = cutlass.Float32(cute.arch.thread_idx()[0])\n",
        }[mutation]
        source = source.replace("for offset1 in", insertion + "for offset1 in")
    elif mutation == "carry_iterator":
        source = source.replace(
            "for offset1 in range(cutlass.Int32(0), cutlass.Int32(width)",
            "for offset1 in range(cutlass.Int32(0), cutlass.Int32(state_max)",
        )
    with pytest.raises(exc.BackendUnsupported, match="sequence proof declined"):
        _rewrite(source, mutation=mutation)
